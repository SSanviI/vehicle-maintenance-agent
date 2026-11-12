"""
Token Manager — Encrypted OAuth token storage and lifecycle.

Handles:
- Fernet-based encryption of access/refresh tokens at rest
- Token storage and retrieval from database
- Automatic token refresh when expired
- Token revocation on logout
- NEVER logs or exposes plaintext tokens
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict
from cryptography.fernet import Fernet
import hashlib
import base64

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from backend.app.database.models import OAuthToken
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.auth.oauth import refresh_access_token, revoke_token

logger = get_logger(__name__)

# Singleton Fernet instance
_fernet: Optional[Fernet] = None


def _get_fernet() -> Fernet:
    """Get or create Fernet encryption instance from APP_SECRET_KEY."""
    global _fernet
    if _fernet is None:
        settings = get_settings()
        # Derive a 32-byte key from APP_SECRET_KEY via SHA256
        key_bytes = hashlib.sha256(settings.app_secret_key.encode("utf-8")).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        _fernet = Fernet(fernet_key)
    return _fernet


def encrypt_token(plaintext: str) -> str:
    """Encrypt a token string using Fernet."""
    f = _get_fernet()
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_token(ciphertext: str) -> str:
    """Decrypt a token string using Fernet."""
    f = _get_fernet()
    return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")


async def save_tokens(
    db: AsyncSession,
    google_user_id: str,
    access_token: str,
    refresh_token: Optional[str],
    expires_in: Optional[int] = None,
    scopes: Optional[str] = None,
):
    """
    Encrypt and store OAuth tokens in the database.
    Creates or updates the token record for this Google user.
    """
    # Calculate expiry
    token_expiry = None
    if expires_in:
        token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)

    # Encrypt tokens
    access_encrypted = encrypt_token(access_token)
    refresh_encrypted = encrypt_token(refresh_token) if refresh_token else None

    try:
        # Check if token record exists
        stmt = select(OAuthToken).where(OAuthToken.google_user_id == google_user_id)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.access_token_encrypted = access_encrypted
            if refresh_encrypted:
                existing.refresh_token_encrypted = refresh_encrypted
            existing.token_expiry = token_expiry
            if scopes:
                existing.scopes = scopes
            existing.updated_at = datetime.utcnow()
        else:
            token_record = OAuthToken(
                id=uuid.uuid4(),
                google_user_id=google_user_id,
                access_token_encrypted=access_encrypted,
                refresh_token_encrypted=refresh_encrypted,
                token_expiry=token_expiry,
                scopes=scopes or "",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(token_record)

        await db.commit()
        logger.info(f"OAuth tokens saved for user {google_user_id[:8]}...")

    except Exception as e:
        logger.error(f"Failed to save OAuth tokens: {e}")
        await db.rollback()
        raise


async def get_valid_credentials(db: AsyncSession, google_user_id: str) -> Optional[Dict[str, str]]:
    """
    Load encrypted token, decrypt, check expiration, refresh if necessary.
    
    Returns:
        Dict with 'access_token' and optionally 'refresh_token', or None if unavailable.
    
    Raises:
        ValueError if credentials cannot be obtained (requires reauthorization).
    """
    try:
        stmt = select(OAuthToken).where(OAuthToken.google_user_id == google_user_id)
        result = await db.execute(stmt)
        token_record = result.scalar_one_or_none()

        if not token_record:
            logger.warning(f"No OAuth tokens found for user {google_user_id[:8]}...")
            return None

        # Decrypt access token
        access_token = decrypt_token(token_record.access_token_encrypted)

        # Check if expired (with 5-minute buffer)
        is_expired = False
        if token_record.token_expiry:
            is_expired = datetime.utcnow() >= (token_record.token_expiry - timedelta(minutes=5))

        if is_expired:
            # Attempt refresh
            if not token_record.refresh_token_encrypted:
                logger.warning("Access token expired and no refresh token available")
                raise ValueError("OAuth credentials expired — reauthorization required")

            refresh_token = decrypt_token(token_record.refresh_token_encrypted)
            try:
                new_token_data = await refresh_access_token(refresh_token)
                new_access = new_token_data["access_token"]
                new_expires_in = new_token_data.get("expires_in")
                new_refresh = new_token_data.get("refresh_token")

                # Save refreshed tokens
                await save_tokens(
                    db=db,
                    google_user_id=google_user_id,
                    access_token=new_access,
                    refresh_token=new_refresh or refresh_token,  # Keep old refresh if not rotated
                    expires_in=new_expires_in,
                    scopes=token_record.scopes,
                )

                access_token = new_access
                logger.info(f"Tokens refreshed for user {google_user_id[:8]}...")

            except ValueError:
                logger.error("Token refresh failed — reauthorization required")
                raise ValueError("OAuth credentials expired and refresh failed — reauthorization required")

        return {"access_token": access_token}

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving credentials: {e}")
        raise ValueError("Failed to retrieve OAuth credentials")


async def revoke_tokens(db: AsyncSession, google_user_id: str) -> bool:
    """
    Revoke Google credentials and remove from database.
    """
    try:
        stmt = select(OAuthToken).where(OAuthToken.google_user_id == google_user_id)
        result = await db.execute(stmt)
        token_record = result.scalar_one_or_none()

        if token_record:
            # Try to revoke the access token with Google
            try:
                access_token = decrypt_token(token_record.access_token_encrypted)
                await revoke_token(access_token)
            except Exception as e:
                logger.warning(f"Token revocation with Google failed (continuing cleanup): {e}")

            # Delete from database regardless
            await db.execute(
                delete(OAuthToken).where(OAuthToken.google_user_id == google_user_id)
            )
            await db.commit()
            logger.info(f"OAuth tokens revoked and deleted for user {google_user_id[:8]}...")
            return True

        return False

    except Exception as e:
        logger.error(f"Error revoking tokens: {e}")
        await db.rollback()
        return False

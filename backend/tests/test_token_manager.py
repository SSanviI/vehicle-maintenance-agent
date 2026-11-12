"""
Backend tests for OAuth token manager: encryption, decryption, expiry, and refresh.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from backend.app.auth.token_manager import (
    encrypt_token,
    decrypt_token,
    save_tokens,
    get_valid_credentials,
)
from backend.app.database.models import OAuthToken


def test_fernet_token_encryption_decryption():
    """Verify OAuth access and refresh tokens are encrypted at rest and decrypt correctly."""
    original_token = "ya29.a0ARW5m7_example_access_token_123456789"
    encrypted = encrypt_token(original_token)

    # Encrypted token MUST NOT contain the plaintext token
    assert original_token not in encrypted
    assert encrypted != original_token

    # Decrypting returns exact original plaintext
    decrypted = decrypt_token(encrypted)
    assert decrypted == original_token


@pytest.mark.asyncio
async def test_save_tokens(async_session):
    """Test encrypting and saving OAuth tokens to database."""
    user_id = "test-google-user-001"
    access_tok = "access_token_abc"
    refresh_tok = "refresh_token_xyz"

    await save_tokens(
        db=async_session,
        google_user_id=user_id,
        access_token=access_tok,
        refresh_token=refresh_tok,
        expires_in=3600,
        scopes="openid profile email https://www.googleapis.com/auth/gmail.send",
    )

    # Fetch from DB
    from sqlalchemy import select
    stmt = select(OAuthToken).where(OAuthToken.google_user_id == user_id)
    res = await async_session.execute(stmt)
    token_rec = res.scalar_one_or_none()

    assert token_rec is not None
    assert token_rec.access_token_encrypted != access_tok
    assert decrypt_token(token_rec.access_token_encrypted) == access_tok
    assert decrypt_token(token_rec.refresh_token_encrypted) == refresh_tok


@pytest.mark.asyncio
async def test_get_valid_credentials_unexpired(async_session):
    """Test retrieving unexpired OAuth credentials."""
    user_id = "test-google-user-002"
    access_tok = "valid_access_token_123"

    await save_tokens(
        db=async_session,
        google_user_id=user_id,
        access_token=access_tok,
        refresh_token="refresh_123",
        expires_in=3600,
    )

    creds = await get_valid_credentials(async_session, user_id)
    assert creds is not None
    assert creds["access_token"] == access_tok


@pytest.mark.asyncio
async def test_get_valid_credentials_expired_refresh_success(async_session):
    """Test automatic token refresh when access token is expired."""
    user_id = "test-google-user-003"
    old_access = "expired_access_token"
    refresh_tok = "valid_refresh_token"

    # Save token that expired 10 minutes ago
    await save_tokens(
        db=async_session,
        google_user_id=user_id,
        access_token=old_access,
        refresh_token=refresh_tok,
        expires_in=-600,  # Expired
    )

    new_access = "new_refreshed_access_token_456"

    # Mock refresh_access_token call
    with patch("backend.app.auth.token_manager.refresh_access_token", new_callable=AsyncMock) as mock_refresh:
        mock_refresh.return_value = {
            "access_token": new_access,
            "expires_in": 3600,
        }

        creds = await get_valid_credentials(async_session, user_id)
        assert creds["access_token"] == new_access
        mock_refresh.assert_called_once_with(refresh_tok)

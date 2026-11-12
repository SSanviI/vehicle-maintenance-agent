"""
OAuth 2.0 Flow — Google OAuth with PKCE.

Handles:
- Authorization URL generation with state + PKCE challenge
- Callback validation (state, code exchange with PKCE verifier)
- User info fetching from Google
- Server-side state store with TTL for security
"""

import hashlib
import base64
import secrets
import time
from typing import Optional, Dict, Tuple
from urllib.parse import urlencode

import httpx
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Required scopes — minimum necessary for identity + Gmail send
OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.send",
]

# Server-side state store: {state_token: {verifier, created_at, used}}
# TTL = 10 minutes for OAuth transactions
_oauth_state_store: Dict[str, Dict] = {}
STATE_TTL_SECONDS = 600  # 10 minutes


def _cleanup_expired_states():
    """Remove expired OAuth state entries."""
    now = time.time()
    expired = [k for k, v in _oauth_state_store.items() if now - v["created_at"] > STATE_TTL_SECONDS]
    for k in expired:
        del _oauth_state_store[k]


def _generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge."""
    # code_verifier: 43-128 characters, URL-safe
    verifier = secrets.token_urlsafe(64)
    # code_challenge: SHA256 hash of verifier, base64url-encoded
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    return verifier, challenge


def generate_oauth_url() -> Tuple[str, str]:
    """
    Generate Google OAuth authorization URL with state and PKCE.
    
    Returns:
        (authorization_url, state_token)
    """
    settings = get_settings()
    _cleanup_expired_states()

    # Generate cryptographic state token
    state = secrets.token_urlsafe(32)

    # Generate PKCE pair
    verifier, challenge = _generate_pkce_pair()

    # Store state + verifier server-side
    _oauth_state_store[state] = {
        "verifier": verifier,
        "created_at": time.time(),
        "used": False,
    }

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }

    auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    logger.info("OAuth authorization URL generated")
    return auth_url, state


def validate_state(state: Optional[str]) -> Optional[str]:
    """
    Validate OAuth state token and return PKCE verifier if valid.
    
    Returns:
        PKCE code_verifier if valid, None if invalid.
    """
    if not state:
        logger.warning("OAuth callback: missing state parameter")
        return None

    _cleanup_expired_states()

    entry = _oauth_state_store.get(state)
    if not entry:
        logger.warning("OAuth callback: invalid or expired state token")
        return None

    if entry["used"]:
        logger.warning("OAuth callback: reused state token (replay attack?)")
        del _oauth_state_store[state]
        return None

    # Check TTL
    if time.time() - entry["created_at"] > STATE_TTL_SECONDS:
        logger.warning("OAuth callback: expired state token")
        del _oauth_state_store[state]
        return None

    # Mark as used (one-time use)
    entry["used"] = True
    verifier = entry["verifier"]

    # Remove from store
    del _oauth_state_store[state]

    return verifier


async def exchange_code_for_tokens(code: str, verifier: str) -> Dict:
    """
    Exchange authorization code for tokens using PKCE verifier.
    
    Returns:
        Token response dict with access_token, refresh_token, etc.
    
    Raises:
        ValueError on failed exchange.
    """
    settings = get_settings()

    payload = {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "code": code,
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": settings.google_redirect_uri,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GOOGLE_TOKEN_URL, data=payload)

    if response.status_code != 200:
        logger.error(f"Token exchange failed: {response.status_code} — {response.text[:200]}")
        raise ValueError("Failed to exchange authorization code for tokens")

    token_data = response.json()
    logger.info("OAuth token exchange successful")
    return token_data


async def fetch_user_info(access_token: str) -> Dict:
    """
    Fetch Google user profile information.
    
    Returns:
        Dict with sub, name, email, picture, etc.
    
    Raises:
        ValueError on failure.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if response.status_code != 200:
        logger.error(f"User info fetch failed: {response.status_code}")
        raise ValueError("Failed to fetch Google user information")

    user_info = response.json()
    logger.info(f"Fetched Google user info for: {user_info.get('email', 'unknown')}")
    return user_info


async def refresh_access_token(refresh_token: str) -> Dict:
    """
    Refresh an expired access token.
    
    Returns:
        New token data dict.
    
    Raises:
        ValueError on refresh failure.
    """
    settings = get_settings()

    payload = {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GOOGLE_TOKEN_URL, data=payload)

    if response.status_code != 200:
        logger.error(f"Token refresh failed: {response.status_code}")
        raise ValueError("Failed to refresh access token — reauthorization required")

    token_data = response.json()
    logger.info("OAuth access token refreshed successfully")
    return token_data


async def revoke_token(token: str) -> bool:
    """
    Revoke an OAuth token (access or refresh).
    
    Returns:
        True if revoked successfully.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_REVOKE_URL,
                params={"token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if response.status_code == 200:
            logger.info("OAuth token revoked successfully")
            return True
        else:
            logger.warning(f"Token revocation returned {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Token revocation failed: {e}")
        return False

"""
Backend tests for Google OAuth 2.0 PKCE flow, state validation, session cookies, and logout.
"""

import pytest
import time
from backend.app.auth.oauth import (
    generate_oauth_url,
    validate_state,
    _oauth_state_store,
    _generate_pkce_pair,
)
from backend.app.auth.session import (
    create_session_token,
    validate_session_token,
)
from backend.app.auth.token_manager import (
    encrypt_token,
    decrypt_token,
)


def test_pkce_generation():
    """Verify PKCE code_verifier and code_challenge generation."""
    verifier, challenge = _generate_pkce_pair()
    assert len(verifier) >= 43
    assert len(challenge) > 0
    assert verifier != challenge


def test_oauth_url_generation():
    """Verify generate_oauth_url creates valid authorization URL with state and PKCE."""
    url, state = generate_oauth_url()
    assert "accounts.google.com" in url
    assert f"state={state}" in url
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert "scope=" in url
    assert "gmail.send" in url
    # Verify state store has verifier entry
    assert state in _oauth_state_store
    assert _oauth_state_store[state]["used"] is False


def test_state_validation_success():
    """Verify valid state token returns PKCE verifier and is consumed (one-time use)."""
    _, state = generate_oauth_url()
    verifier = validate_state(state)
    assert verifier is not None
    # Second attempt must fail (one-time use / replay prevention)
    assert validate_state(state) is None


def test_state_validation_invalid_and_missing():
    """Verify state validation rejects missing or fake state tokens."""
    assert validate_state(None) is None
    assert validate_state("") is None
    assert validate_state("fake-invalid-state-token") is None


def test_state_validation_expired():
    """Verify expired state tokens are rejected."""
    _, state = generate_oauth_url()
    # Artificially set state creation time to 15 minutes ago
    _oauth_state_store[state]["created_at"] = time.time() - 900
    assert validate_state(state) is None


def test_session_cookie_token_creation_and_validation():
    """Verify signed session token creation and decoding."""
    google_user_id = "google-user-123456789"
    session_token = create_session_token(google_user_id)
    assert session_token is not None

    decoded_gid = validate_session_token(session_token)
    assert decoded_gid == google_user_id


def test_invalid_session_cookie():
    """Verify invalid or tampered session cookies are rejected."""
    assert validate_session_token("invalid.tampered.cookie") is None
    assert validate_session_token("") is None


@pytest.mark.asyncio
async def test_resolve_user_from_session_cookie(async_session):
    """Verify resolving GoogleUser from database using valid signed session cookie."""
    import uuid
    from datetime import datetime
    from unittest.mock import MagicMock
    from backend.app.database.models import GoogleUser
    from backend.app.auth.session import _resolve_user

    google_user_id = f"test-google-{uuid.uuid4()}"
    user = GoogleUser(
        id=uuid.uuid4(),
        google_user_id=google_user_id,
        name="John Test",
        email="john@example.com",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    async_session.add(user)
    await async_session.commit()

    # Create signed session token
    session_token = create_session_token(google_user_id)

    # Mock Request with cookie
    mock_request = MagicMock()
    mock_request.cookies = {"vm_session": session_token}
    mock_request.headers = {}

    resolved = await _resolve_user(mock_request, async_session)
    assert resolved is not None
    assert resolved.google_user_id == google_user_id
    assert resolved.name == "John Test"
    assert resolved.email == "john@example.com"


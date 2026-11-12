"""
Auth API Routes — Google OAuth 2.0 login, callback, logout, and user info.

Routes:
    GET  /auth/login     — Redirect to Google OAuth consent screen
    GET  /auth/callback  — Handle Google OAuth callback
    POST /auth/logout    — Invalidate session and revoke tokens
    GET  /auth/me        — Return current authenticated user info
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.database.models import GoogleUser
from backend.app.database.session import get_db
from backend.app.auth.oauth import (
    generate_oauth_url,
    validate_state,
    exchange_code_for_tokens,
    fetch_user_info,
)
from backend.app.auth.token_manager import save_tokens, revoke_tokens
from backend.app.auth.session import (
    create_session_token,
    get_session_cookie_params,
    get_current_user,
    get_optional_user,
    SESSION_COOKIE_NAME,
)
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/login")
async def auth_login():
    """
    Initiate Google OAuth 2.0 login flow.
    Generates authorization URL with state + PKCE and redirects.
    """
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )

    auth_url, state = generate_oauth_url()
    return RedirectResponse(url=auth_url, status_code=302)


@router.post("/google")
async def auth_google_post():
    """
    Generate Google OAuth authorization URL and return JSON response.
    Returns: {"authorization_url": "https://accounts.google.com/..."}
    """
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )

    auth_url, state = generate_oauth_url()
    return {"authorization_url": auth_url, "state": state}


@router.get("/callback")
async def auth_callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Handle Google OAuth callback.
    Validates state, exchanges code with PKCE verifier, creates/updates user,
    stores encrypted tokens, sets session cookie, redirects to frontend.
    """
    settings = get_settings()
    
    # Extract query parameters
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        logger.warning(f"OAuth callback error from Google: {error}")
        return RedirectResponse(
            url=f"{settings.frontend_url}?auth_error={error}",
            status_code=302,
        )

    if not code:
        logger.warning("OAuth callback: missing authorization code")
        return RedirectResponse(
            url=f"{settings.frontend_url}?auth_error=missing_code",
            status_code=302,
        )

    # Validate state and get PKCE verifier
    verifier = validate_state(state)
    if not verifier:
        logger.warning("OAuth callback: state validation failed")
        return RedirectResponse(
            url=f"{settings.frontend_url}?auth_error=invalid_state",
            status_code=302,
        )

    try:
        logger.info("[AUTH] OAuth callback received")

        # Exchange code for tokens using PKCE verifier
        token_data = await exchange_code_for_tokens(code, verifier)
        access_token = token_data["access_token"]
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in")
        scopes = token_data.get("scope", "")

        # Fetch Google user identity
        user_info = await fetch_user_info(access_token)
        google_user_id = user_info["sub"]
        user_name = user_info.get("name", "")
        user_email = user_info.get("email", "")

        logger.info(f"[AUTH] Google user: {user_email}")

        # Create or update local user
        stmt = select(GoogleUser).where(GoogleUser.google_user_id == google_user_id)
        result = await db.execute(stmt)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            existing_user.name = user_name
            existing_user.email = user_email
            existing_user.updated_at = datetime.utcnow()
            logger.info(f"[AUTH] Local user found: {user_email}")
        else:
            new_user = GoogleUser(
                id=uuid.uuid4(),
                google_user_id=google_user_id,
                name=user_name,
                email=user_email,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(new_user)
            logger.info(f"[AUTH] Local user created: {user_email}")

        await db.commit()

        # Encrypt and store OAuth tokens
        await save_tokens(
            db=db,
            google_user_id=google_user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            scopes=scopes,
        )

        # Create signed session cookie
        session_token = create_session_token(google_user_id)
        logger.info(f"[AUTH] Session created for user {google_user_id[:8]}...")

        # Redirect ONLY to frontend_url with NO query params
        redirect_url = settings.frontend_url
        response = RedirectResponse(
            url=redirect_url,
            status_code=302,
        )
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_token,
            max_age=86400 * 7,
            path="/",
            domain=None,
            secure=settings.secure_cookies,
            httponly=True,
            samesite="lax",
        )
        logger.info(f"[AUTH] Session cookie set. Redirecting to frontend: {redirect_url}")

        return response

    except ValueError as e:
        logger.error(f"OAuth callback processing failed: {e}")
        err_msg = str(e).replace(" ", "_")
        return RedirectResponse(
            url=f"{settings.frontend_url}?auth_error={err_msg}",
            status_code=302,
        )
    except Exception as e:
        logger.error(f"Unexpected OAuth callback error: {e}", exc_info=True)
        err_msg = str(e).replace(" ", "_")[:100]
        return RedirectResponse(
            url=f"{settings.frontend_url}?auth_error={err_msg}",
            status_code=302,
        )


@router.post("/logout")
async def auth_logout(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Logout: invalidate session, clear cookie, revoke Google credentials.
    """
    from backend.app.auth.session import _resolve_user
    user = await _resolve_user(request, db)

    if user:
        try:
            await revoke_tokens(db, user.google_user_id)
        except Exception as e:
            logger.warning(f"Token revocation during logout failed: {e}")

    response = JSONResponse(content={"status": "logged_out"})
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
    )

    logger.info("[AUTH] User logged out successfully")
    return response


@router.get("/me")
async def auth_me(
    request: Request,
    user: GoogleUser = Depends(get_optional_user),
):
    """
    Return current authenticated user info.
    Returns null user data if not authenticated.
    """
    has_cookie = SESSION_COOKIE_NAME in request.cookies
    auth_header = request.headers.get("Authorization")
    has_auth_header = bool(auth_header and auth_header.startswith("Bearer "))

    logger.info(f"[AUTH] /auth/me call - Cookie present: {has_cookie}, Auth Header: {has_auth_header}, User resolved: {user is not None}")

    if user is None:
        return {
            "authenticated": False,
            "user": None,
        }

    return {
        "authenticated": True,
        "user": {
            "id": str(user.id),
            "name": user.name,
            "email": user.email,
        },
    }


@router.get("/status")
async def auth_status(
    request: Request,
    user: GoogleUser = Depends(get_optional_user),
):
    """
    Return current authentication status (compatible with /auth/status endpoint).
    Unauthenticated: {"authenticated": false}
    Authenticated: {"authenticated": true, "google_user_id": "...", "name": "...", "email": "..."}
    """
    has_cookie = SESSION_COOKIE_NAME in request.cookies
    auth_header = request.headers.get("Authorization")
    has_auth_header = bool(auth_header and auth_header.startswith("Bearer "))

    logger.info(f"[AUTH] /auth/status call - Cookie present: {has_cookie}, Auth Header: {has_auth_header}, User resolved: {user is not None}")

    if user is None:
        return {
            "authenticated": False,
        }

    return {
        "authenticated": True,
        "google_user_id": user.google_user_id,
        "name": user.name or user.email,
        "email": user.email,
    }

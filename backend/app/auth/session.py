"""
Session Management — Secure HttpOnly cookie-based sessions.

Handles:
- Signed session cookie creation (itsdangerous URLSafeTimedSerializer)
- Session validation and user resolution
- Cookie security: HttpOnly, SameSite=Lax, Secure in production
- FastAPI dependencies for authenticated/optional-auth routes
- NEVER exposes OAuth tokens to the browser
"""

from typing import Optional
from fastapi import Request, HTTPException, Depends
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.database.models import GoogleUser
from backend.app.database.session import get_db
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

SESSION_COOKIE_NAME = "vm_session"
SESSION_MAX_AGE = 86400 * 7  # 7 days

_serializer: Optional[URLSafeTimedSerializer] = None


def _get_serializer() -> URLSafeTimedSerializer:
    """Get or create session serializer."""
    global _serializer
    if _serializer is None:
        settings = get_settings()
        _serializer = URLSafeTimedSerializer(settings.app_secret_key, salt="vm-session")
    return _serializer


def create_session_token(google_user_id: str) -> str:
    """Create a signed session token containing the google_user_id."""
    s = _get_serializer()
    return s.dumps({"gid": google_user_id})


def validate_session_token(token: str) -> Optional[str]:
    """
    Validate a session token and return the google_user_id.
    Returns None if invalid or expired.
    """
    s = _get_serializer()
    try:
        data = s.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("gid")
    except SignatureExpired:
        logger.warning("Session cookie expired")
        return None
    except BadSignature:
        logger.warning("Invalid session cookie signature")
        return None
    except Exception as e:
        logger.warning(f"Session validation error: {e}")
        return None


def get_session_cookie_params() -> dict:
    """Get cookie parameters based on environment."""
    settings = get_settings()
    return {
        "key": SESSION_COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "secure": settings.secure_cookies,
        "max_age": SESSION_MAX_AGE,
        "path": "/",
    }


async def _resolve_user(request: Request, db: AsyncSession) -> Optional[GoogleUser]:
    """Internal: resolve GoogleUser strictly from HttpOnly session cookie."""
    token = request.cookies.get(SESSION_COOKIE_NAME)

    if not token:
        return None

    google_user_id = validate_session_token(token)
    if not google_user_id:
        return None

    try:
        stmt = select(GoogleUser).where(GoogleUser.google_user_id == google_user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        return user
    except Exception as e:
        logger.error(f"Error resolving user from session: {e}")
        return None


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> GoogleUser:
    """
    FastAPI dependency: require authenticated user.
    Raises 401 if not authenticated.
    """
    user = await _resolve_user(request, db)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in with Google.",
        )
    return user


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[GoogleUser]:
    """
    FastAPI dependency: optionally resolve authenticated user.
    Returns None for unauthenticated requests (allows normal ML flow).
    """
    return await _resolve_user(request, db)

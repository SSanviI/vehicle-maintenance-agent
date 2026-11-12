"""
Security utilities: input validation, sanitization.
"""

import re
from typing import Any


def sanitize_user_input(text: str) -> str:
    """Basic sanitization of user input text."""
    if not text:
        return ""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Limit length
    text = text[:5000]
    return text.strip()


def validate_conversation_id(conv_id: str) -> bool:
    """Validate conversation ID format (UUID)."""
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(conv_id))


def mask_sensitive_data(data: dict) -> dict:
    """Mask sensitive fields in a dict for logging."""
    sensitive_keys = {"api_key", "password", "secret", "token", "authorization"}
    masked = {}
    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys):
            masked[k] = "***MASKED***"
        elif isinstance(v, dict):
            masked[k] = mask_sensitive_data(v)
        else:
            masked[k] = v
    return masked

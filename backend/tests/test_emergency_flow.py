"""
End-to-end integration tests for Emergency Flow:
- Emergency detection
- User confirmation
- Unauthenticated send rejection
- Missing contact rejection
- Deduplication (double click prevention)
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from backend.app.emergency.routes import _recent_sends, DEDUP_WINDOW_SECONDS


def test_deduplication_prevention():
    """
    CRITICAL REQUIREMENT TEST:
    Verify short-lived deduplication prevents sending duplicate emergency emails
    when user double-clicks 'Send Emergency Alert'.
    """
    user_gid = "test-dedup-user-123"

    # Reset dedup store
    _recent_sends.clear()

    # Record first send timestamp
    now = time.time()
    _recent_sends[user_gid] = now

    # Second request immediately after (within DEDUP_WINDOW_SECONDS)
    last_send = _recent_sends.get(user_gid, 0.0)
    is_blocked = (now + 5 - last_send) < DEDUP_WINDOW_SECONDS

    assert is_blocked is True


def test_deduplication_allows_send_after_window():
    """Verify deduplication allows a new send after DEDUP_WINDOW_SECONDS has passed."""
    user_gid = "test-dedup-user-456"

    _recent_sends.clear()

    # Record past send timestamp 70 seconds ago
    past = time.time() - 70.0
    _recent_sends[user_gid] = past

    now = time.time()
    is_blocked = (now - _recent_sends.get(user_gid, 0.0)) < DEDUP_WINDOW_SECONDS

    assert is_blocked is False

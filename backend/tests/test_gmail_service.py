"""
Unit tests for GmailService: MIME construction, mocked API sending, error handling (auth, rate limits, 5xx).
"""

import pytest
from unittest.mock import patch, MagicMock
from googleapiclient.errors import HttpError
from httplib2 import Response

from backend.app.services.gmail_service import (
    send_email,
    build_emergency_email,
    GmailAuthError,
    GmailSendError,
)


def test_build_emergency_email():
    """Verify MIME content generation for plain text and HTML."""
    subject, body_text, body_html = build_emergency_email(
        user_name="John Driver",
        contact_name="Alice Smith",
        emergency_severity="CRITICAL",
        conditions=["Worn brakes combined with weak battery"],
        symptoms=["Brakes are worn out", "Battery is weak"],
        vehicle_info={"Vehicle_Age": 5, "Brake_Condition": "Worn Out"},
        prediction_summary="maintenance_required",
        model_score=0.96,
    )

    assert "Emergency Vehicle Alert" in subject
    assert "John Driver" in body_text
    assert "Alice Smith" in body_text
    assert "CRITICAL" in body_text
    assert "96.0%" in body_text
    assert "Worn brakes" in body_text

    assert "John Driver" in body_html
    assert "CRITICAL" in body_html
    assert "96.0%" in body_html


@patch("backend.app.services.gmail_service.build")
def test_send_email_success(mock_build):
    """Test successful email dispatch via mocked Gmail API."""
    mock_service = MagicMock()
    mock_messages = MagicMock()
    mock_send = MagicMock()

    mock_send.execute.return_value = {"id": "gmail_msg_id_12345"}
    mock_messages.send.return_value = mock_send
    mock_service.users.return_value.messages.return_value = mock_messages
    mock_build.return_value = mock_service

    msg_id = send_email(
        access_token="mock_access_token",
        to="recipient@example.com",
        subject="Test Subject",
        body_text="Test Body",
        body_html="<p>Test Body</p>",
    )

    assert msg_id == "gmail_msg_id_12345"
    mock_messages.send.assert_called_once()


@patch("backend.app.services.gmail_service.build")
def test_send_email_auth_error_401(mock_build):
    """Test 401 Unauthorized raises GmailAuthError."""
    mock_service = MagicMock()
    mock_messages = MagicMock()
    mock_send = MagicMock()

    resp = Response({"status": "401"})
    mock_send.execute.side_effect = HttpError(resp=resp, content=b"Unauthorized")
    mock_messages.send.return_value = mock_send
    mock_service.users.return_value.messages.return_value = mock_messages
    mock_build.return_value = mock_service

    with pytest.raises(GmailAuthError) as exc_info:
        send_email(
            access_token="expired_token",
            to="recipient@example.com",
            subject="Test",
            body_text="Test",
            body_html="<p>Test</p>",
        )

    assert "authentication/permission" in str(exc_info.value)


@patch("backend.app.services.gmail_service.build")
def test_send_email_rate_limit_429(mock_build):
    """Test 429 Rate Limit raises GmailSendError."""
    mock_service = MagicMock()
    mock_messages = MagicMock()
    mock_send = MagicMock()

    resp = Response({"status": "429"})
    mock_send.execute.side_effect = HttpError(resp=resp, content=b"Rate limit")
    mock_messages.send.return_value = mock_send
    mock_service.users.return_value.messages.return_value = mock_messages
    mock_build.return_value = mock_service

    with pytest.raises(GmailSendError) as exc_info:
        send_email(
            access_token="token",
            to="recipient@example.com",
            subject="Test",
            body_text="Test",
            body_html="<p>Test</p>",
        )

    assert "rate limit" in str(exc_info.value).lower()

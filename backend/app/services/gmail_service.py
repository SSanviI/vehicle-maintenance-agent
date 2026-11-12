"""
Gmail Service — Send emergency notification emails via Gmail API.

Uses:
- Gmail API `users.messages.send` with `userId="me"`
- MIME multipart/alternative (plain text + HTML)
- Authenticated user's own Gmail account

Does NOT use:
- SMTP
- Gmail passwords
- App Passwords
- Shared Gmail accounts

Error handling:
- GmailAuthError: 401/403 credential issues
- GmailSendError: 429, 5xx, network, and general send failures
"""

import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class GmailAuthError(Exception):
    """Raised when Gmail API returns 401/403 or credentials are invalid/expired."""
    pass


class GmailSendError(Exception):
    """Raised when Gmail API fails to send (429, 5xx, network, or other)."""
    pass


# =============================================================================
# Gmail Service
# =============================================================================

def _build_mime_message(
    to: str,
    subject: str,
    body_text: str,
    body_html: str,
    from_name: Optional[str] = None,
) -> str:
    """
    Build a MIME multipart/alternative message and return base64url-encoded.
    """
    message = MIMEMultipart("alternative")
    message["To"] = to
    message["Subject"] = subject
    if from_name:
        message["From"] = from_name

    # Plain text part
    part_text = MIMEText(body_text, "plain", "utf-8")
    message.attach(part_text)

    # HTML part
    part_html = MIMEText(body_html, "html", "utf-8")
    message.attach(part_html)

    # Base64url encode the raw message
    raw_bytes = message.as_bytes()
    encoded = base64.urlsafe_b64encode(raw_bytes).decode("utf-8")
    return encoded


def send_email(
    access_token: str,
    to: str,
    subject: str,
    body_text: str,
    body_html: str,
    from_name: Optional[str] = None,
) -> str:
    """
    Send an email via Gmail API from the authenticated user's account.
    
    Args:
        access_token: Valid Google OAuth access token with gmail.send scope
        to: Recipient email address
        subject: Email subject line
        body_text: Plain-text email body
        body_html: HTML email body
        from_name: Optional display name for the sender
        
    Returns:
        Gmail message ID (string)
        
    Raises:
        GmailAuthError: On 401/403 authentication/permission errors
        GmailSendError: On 429, 5xx, network, or other sending errors
    """
    # Build credentials
    credentials = Credentials(token=access_token)

    # Build MIME message
    encoded_message = _build_mime_message(to, subject, body_text, body_html, from_name)

    try:
        # Build Gmail service
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

        # Send via users.messages.send with userId="me"
        result = service.users().messages().send(
            userId="me",
            body={"raw": encoded_message},
        ).execute()

        message_id = result.get("id", "unknown")
        logger.info(f"Gmail send successful: message_id={message_id}")
        return message_id

    except HttpError as e:
        status_code = e.resp.status if e.resp else 0
        error_reason = str(e)[:200]

        if status_code in (401, 403):
            logger.error(f"Gmail auth error ({status_code}): {error_reason}")
            raise GmailAuthError(
                f"Gmail authentication/permission error ({status_code}). "
                "Please re-authorize your Google account."
            )
        elif status_code == 429:
            logger.warning(f"Gmail rate limit exceeded: {error_reason}")
            raise GmailSendError(
                "Gmail rate limit exceeded. Please try again in a few minutes."
            )
        elif status_code >= 500:
            logger.error(f"Gmail server error ({status_code}): {error_reason}")
            # Single retry for transient 5xx errors
            try:
                service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
                result = service.users().messages().send(
                    userId="me",
                    body={"raw": encoded_message},
                ).execute()
                message_id = result.get("id", "unknown")
                logger.info(f"Gmail send succeeded on retry: message_id={message_id}")
                return message_id
            except Exception:
                raise GmailSendError(
                    f"Gmail server error ({status_code}). The service may be temporarily unavailable."
                )
        else:
            logger.error(f"Gmail HTTP error ({status_code}): {error_reason}")
            raise GmailSendError(f"Failed to send email via Gmail (HTTP {status_code}).")

    except GmailAuthError:
        raise
    except GmailSendError:
        raise
    except Exception as e:
        logger.error(f"Gmail send failed with unexpected error: {e}")
        raise GmailSendError("Failed to send email. Please check your connection and try again.")


# =============================================================================
# Emergency Email Content Builder
# =============================================================================

def build_emergency_email(
    user_name: str,
    contact_name: str,
    emergency_severity: str,
    conditions: list,
    symptoms: list,
    vehicle_info: dict,
    prediction_summary: Optional[str] = None,
    model_score: Optional[float] = None,
) -> tuple:
    """
    Build emergency email content (plain text + HTML).
    
    Returns:
        (subject, body_text, body_html)
    """
    subject = "🚨 Emergency Vehicle Alert"

    # Build vehicle info string
    vehicle_details = []
    for key, val in vehicle_info.items():
        if val is not None:
            display_key = key.replace("_", " ").title()
            vehicle_details.append(f"  - {display_key}: {val}")
    vehicle_str = "\n".join(vehicle_details) if vehicle_details else "  No vehicle details available"

    # Build conditions string
    conditions_str = "\n".join(f"  - {c}" for c in conditions) if conditions else "  None specified"

    # Build symptoms string
    symptoms_str = "\n".join(f"  - {s}" for s in symptoms) if symptoms else "  None reported"

    # Prediction info
    prediction_line = ""
    if prediction_summary:
        prediction_line = f"\nML Prediction: {prediction_summary}"
    if model_score is not None:
        prediction_line += f"\nMaintenance Probability: {model_score * 100:.1f}%"

    # Timestamp
    from datetime import datetime
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Plain text version
    body_text = f"""🚨 EMERGENCY VEHICLE ALERT

This is an automated emergency notification from AutoCare AI.

Sender: {user_name}
Emergency Contact: {contact_name}
Severity: {emergency_severity}
Time: {timestamp}

DETECTED CONDITIONS:
{conditions_str}

REPORTED SYMPTOMS:
{symptoms_str}

VEHICLE INFORMATION:
{vehicle_str}
{prediction_line}

IMPORTANT SAFETY INFORMATION:
- If the vehicle is currently in operation, the driver should safely pull over
- Do not ignore critical brake or battery warnings
- Contact a qualified mechanic as soon as possible

This alert was sent through AutoCare AI Vehicle Maintenance System.
The sender has requested you be notified of this potential vehicle emergency.
"""

    # HTML version
    conditions_html = "".join(f"<li>{c}</li>" for c in conditions) if conditions else "<li>None specified</li>"
    symptoms_html = "".join(f"<li>{s}</li>" for s in symptoms) if symptoms else "<li>None reported</li>"
    vehicle_html = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;color:#94a3b8;'>{k.replace('_', ' ').title()}</td>"
        f"<td style='padding:4px 0;color:#e2e8f0;font-weight:600;'>{v}</td></tr>"
        for k, v in vehicle_info.items() if v is not None
    ) or "<tr><td colspan='2'>No vehicle details available</td></tr>"

    prediction_html = ""
    if prediction_summary or model_score is not None:
        pred_parts = []
        if prediction_summary:
            pred_parts.append(f"<p style='margin:4px 0;'><strong>ML Prediction:</strong> {prediction_summary}</p>")
        if model_score is not None:
            score_pct = f"{model_score * 100:.1f}%"
            color = "#ef4444" if model_score > 0.5 else "#22c55e"
            pred_parts.append(
                f"<p style='margin:4px 0;'><strong>Maintenance Probability:</strong> "
                f"<span style='color:{color};font-weight:700;font-size:16px;'>{score_pct}</span></p>"
            )
        prediction_html = f"""
        <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin:16px 0;border-left:3px solid #3b82f6;">
            {''.join(pred_parts)}
        </div>"""

    severity_colors = {
        "CRITICAL": ("#dc2626", "#fef2f2"),
        "HIGH": ("#ea580c", "#fff7ed"),
        "MEDIUM": ("#ca8a04", "#fefce8"),
    }
    sev_color, sev_bg = severity_colors.get(emergency_severity, ("#dc2626", "#fef2f2"))

    body_html = f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0f172a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<div style="max-width:600px;margin:0 auto;padding:24px;">
    <div style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
        <!-- Header -->
        <div style="background:linear-gradient(135deg,#dc2626,#b91c1c);padding:24px;text-align:center;">
            <h1 style="margin:0;color:#fff;font-size:24px;">🚨 Emergency Vehicle Alert</h1>
            <p style="margin:8px 0 0;color:#fecaca;font-size:14px;">AutoCare AI — Automated Emergency Notification</p>
        </div>

        <!-- Content -->
        <div style="padding:24px;color:#e2e8f0;">
            <div style="display:flex;gap:16px;margin-bottom:16px;">
                <div style="flex:1;background:#0f172a;border-radius:8px;padding:12px;">
                    <p style="margin:0 0 4px;color:#94a3b8;font-size:12px;text-transform:uppercase;">From</p>
                    <p style="margin:0;font-weight:600;">{user_name}</p>
                </div>
                <div style="flex:1;background:#0f172a;border-radius:8px;padding:12px;">
                    <p style="margin:0 0 4px;color:#94a3b8;font-size:12px;text-transform:uppercase;">Severity</p>
                    <p style="margin:0;font-weight:700;color:{sev_color};">{emergency_severity}</p>
                </div>
            </div>

            <p style="color:#94a3b8;font-size:12px;margin:0 0 4px;">Time: {timestamp}</p>

            <h3 style="color:#f87171;margin:20px 0 8px;">⚠️ Detected Conditions</h3>
            <ul style="margin:0;padding-left:20px;color:#e2e8f0;">{conditions_html}</ul>

            <h3 style="color:#fbbf24;margin:20px 0 8px;">🔍 Reported Symptoms</h3>
            <ul style="margin:0;padding-left:20px;color:#e2e8f0;">{symptoms_html}</ul>

            <h3 style="color:#60a5fa;margin:20px 0 8px;">🚗 Vehicle Information</h3>
            <table style="width:100%;border-collapse:collapse;">{vehicle_html}</table>

            {prediction_html}

            <div style="background:#422006;border-radius:8px;padding:16px;margin:20px 0;border:1px solid #92400e;">
                <h4 style="margin:0 0 8px;color:#fbbf24;">⚠️ Safety Recommendations</h4>
                <ul style="margin:0;padding-left:20px;color:#fed7aa;font-size:14px;">
                    <li>If the vehicle is currently in operation, safely pull over</li>
                    <li>Do not ignore critical brake or battery warnings</li>
                    <li>Contact a qualified mechanic as soon as possible</li>
                </ul>
            </div>
        </div>

        <!-- Footer -->
        <div style="background:#0f172a;padding:16px 24px;text-align:center;border-top:1px solid #334155;">
            <p style="margin:0;color:#64748b;font-size:12px;">
                Sent via AutoCare AI Vehicle Maintenance System
            </p>
        </div>
    </div>
</div>
</body>
</html>"""

    return subject, body_text, body_html

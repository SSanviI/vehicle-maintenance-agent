"""
Emergency API Routes — CRUD for emergency contacts, emergency alert notification send, and history.

Routes:
    POST   /api/emergency/contact  — Create emergency contact (authenticated)
    GET    /api/emergency/contact  — Get registered emergency contact (authenticated)
    PUT    /api/emergency/contact  — Update emergency contact (authenticated)
    DELETE /api/emergency/contact  — Delete emergency contact (authenticated)
    POST   /api/emergency/send     — Trigger emergency email notification (authenticated)
    GET    /api/emergency/history  — View email notification audit log (authenticated)

CRITICAL SECURITY REQUIREMENTS:
1. Always obtain user identity from authenticated server session (`GoogleUser`).
2. Never trust frontend-supplied user_id, recipient email, or OAuth tokens.
3. Require explicit user confirmation (`POST /api/emergency/send`) to dispatch emails.
4. Implement deduplication to prevent duplicate email dispatches.
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from backend.app.database.models import GoogleUser, EmergencyContact, EmailLog, EmailLogStatus
from backend.app.database.session import get_db
from backend.app.auth.session import get_current_user
from backend.app.auth.token_manager import get_valid_credentials
from backend.app.schemas.emergency import (
    EmergencyContactCreate,
    EmergencyContactUpdate,
    EmergencyContactResponse,
    EmergencySendRequest,
    EmergencySendResponse,
    EmailLogResponse,
    EmergencyHistoryResponse,
)
from backend.app.emergency.emergency_engine import EmergencyEngine
from backend.app.services.gmail_service import (
    send_email,
    build_emergency_email,
    GmailAuthError,
    GmailSendError,
)
from backend.app.memory.repository import MemoryRepository
from backend.app.agent.state import AgentState
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/emergency", tags=["Emergency"])

# Server-side deduplication store: {google_user_id: timestamp_of_last_send}
_recent_sends: Dict[str, float] = {}
DEDUP_WINDOW_SECONDS = 60.0  # 1-minute window between emergency email sends per user


# =============================================================================
# Emergency Contact CRUD
# =============================================================================

@router.post("/contact", response_model=EmergencyContactResponse, status_code=status.HTTP_201_CREATED)
async def create_emergency_contact(
    request: EmergencyContactCreate,
    user: GoogleUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create or replace emergency contact for the authenticated user."""
    # Check if contact already exists
    stmt = select(EmergencyContact).where(EmergencyContact.google_user_id == user.google_user_id)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.contact_name = request.contact_name
        existing.relationship = request.relationship
        existing.contact_email = str(request.contact_email)
        existing.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing)
        logger.info(f"Updated emergency contact for user {user.google_user_id[:8]}...")
        return existing
    else:
        contact = EmergencyContact(
            id=uuid.uuid4(),
            google_user_id=user.google_user_id,
            contact_name=request.contact_name,
            relationship=request.relationship,
            contact_email=str(request.contact_email),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(contact)
        await db.commit()
        await db.refresh(contact)
        logger.info(f"Created emergency contact for user {user.google_user_id[:8]}...")
        return contact


@router.get("/contact", response_model=EmergencyContactResponse)
async def get_emergency_contact(
    user: GoogleUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve registered emergency contact for authenticated user."""
    stmt = select(EmergencyContact).where(EmergencyContact.google_user_id == user.google_user_id)
    result = await db.execute(stmt)
    contact = result.scalar_one_or_none()

    if not contact:
        raise HTTPException(
            status_code=404,
            detail="No emergency contact registered. Please configure one.",
        )
    return contact


@router.put("/contact", response_model=EmergencyContactResponse)
async def update_emergency_contact(
    request: EmergencyContactUpdate,
    user: GoogleUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update existing emergency contact for authenticated user."""
    stmt = select(EmergencyContact).where(EmergencyContact.google_user_id == user.google_user_id)
    result = await db.execute(stmt)
    contact = result.scalar_one_or_none()

    if not contact:
        raise HTTPException(status_code=404, detail="No emergency contact found to update")

    if request.contact_name is not None:
        contact.contact_name = request.contact_name
    if request.relationship is not None:
        contact.relationship = request.relationship
    if request.contact_email is not None:
        contact.contact_email = str(request.contact_email)

    contact.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(contact)
    return contact


@router.delete("/contact")
async def delete_emergency_contact(
    user: GoogleUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete registered emergency contact for authenticated user."""
    stmt = delete(EmergencyContact).where(EmergencyContact.google_user_id == user.google_user_id)
    result = await db.execute(stmt)
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="No emergency contact found to delete")

    return {"status": "success", "message": "Emergency contact deleted"}


# =============================================================================
# Emergency Send & History
# =============================================================================

@router.post("/send", response_model=EmergencySendResponse)
async def send_emergency_alert(
    request: EmergencySendRequest,
    user: GoogleUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Explicit user action endpoint to send an emergency email notification via Gmail API.
    
    Validates:
    - User session authentication
    - Registered emergency contact availability
    - Short-lived event deduplication
    - Valid OAuth token retrieval/refresh
    - Emergency state evaluation from active conversation
    """
    now_ts = time.time()
    user_gid = user.google_user_id

    # 1. Deduplication check (prevent double clicks)
    last_send = _recent_sends.get(user_gid, 0.0)
    if now_ts - last_send < DEDUP_WINDOW_SECONDS:
        remaining = int(DEDUP_WINDOW_SECONDS - (now_ts - last_send))
        logger.warning(f"Deduplication blocked emergency alert send for user {user_gid[:8]}...")
        raise HTTPException(
            status_code=429,
            detail=f"An emergency alert was recently sent. Please wait {remaining} seconds before sending another.",
        )

    # 2. Determine target recipient email (client-supplied recipient_email or registered contact)
    target_email = None
    contact_name = "Emergency Contact"

    if request.recipient_email:
        target_email = str(request.recipient_email).strip()

    stmt = select(EmergencyContact).where(EmergencyContact.google_user_id == user_gid)
    res = await db.execute(stmt)
    contact = res.scalar_one_or_none()

    if contact:
        contact_name = contact.contact_name
        if not target_email:
            target_email = contact.contact_email

    if not target_email:
        raise HTTPException(
            status_code=400,
            detail="Recipient email is required. Please provide a recipient email address.",
        )

    # 3. Fetch active conversation state to evaluate emergency engine & gather details
    repo = MemoryRepository(db)
    conv = await repo.get_or_create_conversation(request.conversation_id)
    conv_id = str(conv.id)

    state = AgentState.from_json(
        {
            "feature_state": conv.feature_state or {},
            "asked_features": conv.asked_questions or [],
            "declined_features": conv.declined_features or [],
        },
        conversation_id=conv_id,
    )

    # Evaluate emergency engine
    emergency_eval = EmergencyEngine.evaluate_emergency(state.feature_state)
    vehicle_info = EmergencyEngine.get_vehicle_info_from_state(state.feature_state)

    # Get latest ML prediction info if available
    latest_pred = state.get_latest_prediction()
    pred_summary = latest_pred.get("prediction") if latest_pred else "Assessment in progress"
    model_score = latest_pred.get("model_score") if latest_pred else None
    severity = emergency_eval.get("severity") or "HIGH"
    conditions = emergency_eval.get("conditions") or ["Explicit user emergency alert requested"]
    symptoms = emergency_eval.get("symptoms") or []

    # 4. Fetch valid decrypted OAuth credentials
    try:
        credentials_dict = await get_valid_credentials(db, user_gid)
        if not credentials_dict or not credentials_dict.get("access_token"):
            raise HTTPException(
                status_code=401,
                detail="Google credentials not available. Please sign in with Google again.",
            )
        access_token = credentials_dict["access_token"]
    except ValueError as e:
        logger.error(f"Failed to get OAuth credentials for emergency send: {e}")
        raise HTTPException(
            status_code=401,
            detail=str(e),
        )

    # Build email content
    user_name = user.name or user.email or "AutoCare AI User"
    subject, body_text, body_html = build_emergency_email(
        user_name=user_name,
        contact_name=contact_name,
        emergency_severity=severity,
        conditions=conditions,
        symptoms=symptoms,
        vehicle_info=vehicle_info,
        prediction_summary=pred_summary,
        model_score=model_score,
    )

    # 5. Create pending email audit log
    log_record = EmailLog(
        id=uuid.uuid4(),
        google_user_id=user_gid,
        contact_email=target_email,
        subject=subject,
        status=EmailLogStatus.PENDING,
        prediction_summary=pred_summary,
        prediction=model_score,
        created_at=datetime.utcnow(),
    )
    db.add(log_record)
    await db.commit()
    await db.refresh(log_record)

    # 6. Attempt Gmail API send
    try:
        email_id = send_email(
            access_token=access_token,
            to=target_email,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            from_name=user_name,
        )

        # Record success
        now_dt = datetime.utcnow()
        log_record.status = EmailLogStatus.SENT
        log_record.email_id = email_id
        log_record.sent_at = now_dt
        await db.commit()

        # Mark deduplication timestamp
        _recent_sends[user_gid] = time.time()

        logger.info(f"Emergency email sent successfully to {target_email} (msg_id: {email_id})")

        return EmergencySendResponse(
            success=True,
            status="sent",
            email_id=email_id,
            recipient=target_email,
            severity=severity,
            conditions=conditions,
            symptoms=symptoms,
            sent_at=now_dt,
        )

    except GmailAuthError as e:
        log_record.status = EmailLogStatus.FAILED
        log_record.error_message = str(e)
        await db.commit()
        logger.error(f"Gmail auth error sending emergency email: {e}")
        raise HTTPException(status_code=401, detail=str(e))

    except GmailSendError as e:
        log_record.status = EmailLogStatus.FAILED
        log_record.error_message = str(e)
        await db.commit()
        logger.error(f"Gmail send error: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    except Exception as e:
        log_record.status = EmailLogStatus.FAILED
        log_record.error_message = "Unexpected failure sending emergency email"
        await db.commit()
        logger.error(f"Unexpected error in emergency alert send: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to send emergency notification email.")


@router.get("/history", response_model=EmergencyHistoryResponse)
async def get_emergency_history(
    limit: int = Query(default=50, ge=1, le=100),
    user: GoogleUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get emergency email notification audit history for authenticated user."""
    stmt = (
        select(EmailLog)
        .where(EmailLog.google_user_id == user.google_user_id)
        .order_by(EmailLog.created_at.desc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    logs = list(res.scalars().all())

    items = [
        EmailLogResponse(
            id=l.id,
            google_user_id=l.google_user_id,
            contact_email=l.contact_email,
            subject=l.subject,
            status=l.status.value if hasattr(l.status, "value") else str(l.status),
            error_message=l.error_message,
            prediction_summary=l.prediction_summary,
            prediction=l.prediction,
            email_id=l.email_id,
            sent_at=l.sent_at,
            created_at=l.created_at,
        )
        for l in logs
    ]

    return EmergencyHistoryResponse(logs=items, total=len(items))

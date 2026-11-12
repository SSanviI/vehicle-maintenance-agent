"""
Pydantic schemas for emergency contact management and emergency alert notifications.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class EmergencyContactBase(BaseModel):
    contact_name: str = Field(..., min_length=1, max_length=255)
    relationship: Optional[str] = Field(None, max_length=100)
    contact_email: EmailStr


class EmergencyContactCreate(EmergencyContactBase):
    pass


class EmergencyContactUpdate(BaseModel):
    contact_name: Optional[str] = Field(None, min_length=1, max_length=255)
    relationship: Optional[str] = Field(None, max_length=100)
    contact_email: Optional[EmailStr] = None


class EmergencyContactResponse(EmergencyContactBase):
    id: UUID
    google_user_id: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EmergencySendRequest(BaseModel):
    """
    Request to trigger emergency email notification.
    Accepts optional recipient_email and optional conversation_id.
    Note: Client DOES NOT specify user_id, tokens, or emergency details.
    Backend authoritatively determines user identity and emergency details.
    """
    recipient_email: Optional[EmailStr] = None
    conversation_id: Optional[str] = None


class EmergencySendResponse(BaseModel):
    success: bool
    status: str  # "sent" or "failed"
    email_id: Optional[str] = None
    recipient: str
    severity: str
    conditions: List[str]
    symptoms: List[str]
    error_message: Optional[str] = None
    sent_at: Optional[datetime] = None


class EmailLogResponse(BaseModel):
    id: UUID
    google_user_id: str
    contact_email: str
    subject: str
    status: str
    error_message: Optional[str] = None
    prediction_summary: Optional[str] = None
    prediction: Optional[float] = None
    email_id: Optional[str] = None
    sent_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class EmergencyHistoryResponse(BaseModel):
    logs: List[EmailLogResponse]
    total: int

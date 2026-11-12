"""
Pydantic schemas for chat and conversation endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime


class ChatRequest(BaseModel):
    """Incoming chat message from user."""
    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response to a chat message."""
    conversation_id: str
    message: str
    prediction: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    feature_state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class StreamChunk(BaseModel):
    """SSE stream chunk."""
    type: str  # "text", "prediction", "state_update", "done", "error"
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class ConversationResponse(BaseModel):
    id: str
    title: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0
    last_message_preview: Optional[str] = None

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    is_active: bool
    messages: List[MessageResponse]
    feature_state: Optional[Dict[str, Any]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: datetime

"""
Conversations API Endpoints — CRUD for conversation management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid

from backend.app.schemas.chat import (
    ConversationResponse, ConversationListResponse, ConversationDetailResponse, MessageResponse
)
from backend.app.memory.repository import MemoryRepository
from backend.app.database.session import get_db
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation session."""
    repo = MemoryRepository(db)
    conv = await repo.get_or_create_conversation()
    return ConversationResponse(
        id=str(conv.id),
        title=conv.title,
        is_active=conv.is_active,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=0,
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List active conversations ordered by last update time."""
    repo = MemoryRepository(db)
    convs = await repo.list_conversations(limit=limit)
    items = [
        ConversationResponse(
            id=c["id"],
            title=c["title"],
            is_active=c["is_active"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
            message_count=c["message_count"],
        )
        for c in convs
    ]
    return ConversationListResponse(conversations=items, total=len(items))


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation_detail(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full conversation details including messages, state, and predictions."""
    repo = MemoryRepository(db)
    conv = await repo.get_or_create_conversation(conversation_id)

    db_messages = await repo.get_messages(conversation_id)
    msg_responses = [
        MessageResponse(
            id=str(m.id),
            role=m.role.value if hasattr(m.role, "value") else str(m.role),
            content=m.content,
            metadata=m.metadata_,
            created_at=m.created_at,
        )
        for m in db_messages
    ]

    preds = [
        {
            "prediction_class": p.prediction_class,
            "model_score": p.model_score,
            "features_used": p.features_used,
            "missing_features": p.missing_features,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in (conv.predictions or [])
    ]

    return ConversationDetailResponse(
        id=str(conv.id),
        title=conv.title,
        is_active=conv.is_active,
        messages=msg_responses,
        feature_state=conv.feature_state,
        predictions=preds,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete (deactivate) a conversation."""
    repo = MemoryRepository(db)
    success = await repo.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or could not be deleted")
    return {"status": "success", "message": "Conversation deleted"}

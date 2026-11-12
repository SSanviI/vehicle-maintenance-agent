"""
Chat API Endpoint — Handles conversational interaction with streaming SSE support.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json
import asyncio

from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.conversation_service import ConversationService
from backend.app.database.session import get_db
from backend.app.core.security import sanitize_user_input
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Main conversational endpoint.
    Sends user message through Agent Controller loop.
    Returns natural language response + updated feature state & prediction.
    """
    clean_msg = sanitize_user_input(request.message)
    if not clean_msg:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    service = ConversationService(db)
    result = await service.handle_message(
        user_message=clean_msg,
        conversation_id=request.conversation_id,
    )

    return ChatResponse(
        conversation_id=result["conversation_id"],
        message=result["message"],
        prediction=result.get("prediction"),
        explanation=result.get("explanation"),
        feature_state=result.get("feature_state"),
    )


@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Server-Sent Events (SSE) streaming chat endpoint.
    Streams back text chunks, state updates, and prediction events in real time.
    """
    clean_msg = sanitize_user_input(request.message)
    if not clean_msg:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    service = ConversationService(db)

    async def event_generator():
        async for chunk in service.handle_message_stream(
            user_message=clean_msg,
            conversation_id=request.conversation_id,
        ):
            # Format as SSE
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

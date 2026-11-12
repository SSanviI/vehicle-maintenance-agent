"""
Persistent Memory Repository — Interacts with PostgreSQL database with in-memory fallback.
Manages conversations, messages, vehicle feature states, prediction logs, and summaries.
"""

from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from backend.app.database.models import (
    Conversation, Message, Prediction, ConversationSummary, 
    VehicleFeature, User, Vehicle, MessageRole, FeatureStatus, FeatureSource
)
from backend.app.agent.state import AgentState
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Global in-memory fallback store when PostgreSQL is unavailable
_inmemory_conversations: Dict[str, Dict[str, Any]] = {}
_inmemory_messages: Dict[str, List[Dict[str, Any]]] = {}
_inmemory_predictions: Dict[str, List[Dict[str, Any]]] = {}


class MemoryRepository:
    """PostgreSQL-backed persistent memory repository with in-memory fallback."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session

    async def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Get an existing conversation or create a new one."""
        if conversation_id and self.session:
            try:
                conv_uuid = uuid.UUID(conversation_id)
                stmt = select(Conversation).where(Conversation.id == conv_uuid)
                result = await self.session.execute(stmt)
                conv = result.scalar_one_or_none()
                if conv:
                    return conv
            except Exception as e:
                logger.warning(f"DB error fetching conversation {conversation_id}, using in-memory: {e}")

        # In-memory lookup fallback
        if conversation_id and conversation_id in _inmemory_conversations:
            data = _inmemory_conversations[conversation_id]
            conv = Conversation(
                id=uuid.UUID(conversation_id),
                title=data.get("title", "New Maintenance Assessment"),
                feature_state=data.get("feature_state", {}),
                asked_questions=data.get("asked_questions", []),
                declined_features=data.get("declined_features", []),
                is_active=data.get("is_active", True),
                created_at=data.get("created_at", datetime.utcnow()),
                updated_at=data.get("updated_at", datetime.utcnow()),
            )
            return conv

        # Create new conversation
        new_id = str(uuid.UUID(conversation_id)) if conversation_id else str(uuid.uuid4())
        conv = Conversation(
            id=uuid.UUID(new_id),
            title="New Maintenance Assessment",
            feature_state={},
            asked_questions=[],
            declined_features=[],
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        _inmemory_conversations[new_id] = {
            "id": new_id,
            "title": conv.title,
            "feature_state": {},
            "asked_questions": [],
            "declined_features": [],
            "is_active": True,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
        }
        _inmemory_messages[new_id] = []
        _inmemory_predictions[new_id] = []

        if self.session:
            try:
                self.session.add(conv)
                await self.session.commit()
                await self.session.refresh(conv)
            except Exception as e:
                logger.warning(f"DB error saving new conversation, falling back to in-memory: {e}")
                await self.session.rollback()

        return conv

    async def save_agent_state(self, conversation_id: str, state: AgentState):
        """Save AgentState to conversation record."""
        from sqlalchemy.orm.attributes import flag_modified
        from copy import deepcopy

        # A single JSON document is the persistence contract for AgentState.
        # Keep legacy columns mirrored only for older readers.
        state_dict = deepcopy(state.to_json())
        asked_list = list(state.asked_features)
        declined_list = list(state.declined_features)

        # Always update in-memory cache
        if conversation_id in _inmemory_conversations:
            _inmemory_conversations[conversation_id]["feature_state"] = state_dict
            _inmemory_conversations[conversation_id]["asked_questions"] = asked_list
            _inmemory_conversations[conversation_id]["declined_features"] = declined_list
            _inmemory_conversations[conversation_id]["updated_at"] = datetime.utcnow()

        if self.session:
            try:
                conv_uuid = uuid.UUID(conversation_id)
                stmt = select(Conversation).where(Conversation.id == conv_uuid)
                res = await self.session.execute(stmt)
                conv = res.scalar_one_or_none()

                if conv:
                    conv.feature_state = state_dict
                    conv.asked_questions = asked_list
                    conv.declined_features = declined_list
                    conv.updated_at = datetime.utcnow()
                    
                    flag_modified(conv, "feature_state")
                    flag_modified(conv, "asked_questions")
                    flag_modified(conv, "declined_features")
                    
                    await self.session.commit()
            except Exception as e:
                logger.warning(f"Failed to save agent state to DB: {e}")
                await self.session.rollback()

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Save a message to DB and in-memory fallback."""
        conv_uuid = uuid.UUID(conversation_id)
        role_enum = MessageRole.USER if role.lower() == "user" else (
            MessageRole.ASSISTANT if role.lower() == "assistant" else MessageRole.SYSTEM
        )

        msg_id = uuid.uuid4()
        now = datetime.utcnow()

        # In-memory store
        if conversation_id not in _inmemory_messages:
            _inmemory_messages[conversation_id] = []

        msg_dict = {
            "id": str(msg_id),
            "role": role.lower(),
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }
        _inmemory_messages[conversation_id].append(msg_dict)

        # Update in-memory title snippet if first user message
        if conversation_id in _inmemory_conversations:
            mem_conv = _inmemory_conversations[conversation_id]
            if mem_conv["title"] == "New Maintenance Assessment" and role.lower() == "user":
                snippet = content[:30].strip()
                mem_conv["title"] = f"Vehicle: {snippet}..." if snippet else "Maintenance Chat"

        msg = Message(
            id=msg_id,
            conversation_id=conv_uuid,
            role=role_enum,
            content=content,
            metadata_=metadata or {},
            created_at=now,
        )

        if self.session:
            try:
                self.session.add(msg)
                stmt = select(Conversation).where(Conversation.id == conv_uuid)
                res = await self.session.execute(stmt)
                conv = res.scalar_one_or_none()
                if conv and conv.title == "New Maintenance Assessment" and role.lower() == "user":
                    snippet = content[:30].strip()
                    conv.title = f"Vehicle: {snippet}..." if snippet else "Maintenance Chat"
                await self.session.commit()
            except Exception as e:
                logger.warning(f"DB error saving message, using in-memory: {e}")
                await self.session.rollback()

        return msg

    async def get_messages(self, conversation_id: str) -> List[Message]:
        """Fetch all messages for a conversation."""
        if self.session:
            try:
                conv_uuid = uuid.UUID(conversation_id)
                stmt = select(Message).where(Message.conversation_id == conv_uuid).order_by(Message.created_at.asc())
                res = await self.session.execute(stmt)
                msgs = list(res.scalars().all())
                if msgs:
                    return msgs
            except Exception as e:
                logger.warning(f"DB error fetching messages, using in-memory: {e}")

        # In-memory fallback
        raw_msgs = _inmemory_messages.get(conversation_id, [])
        result = []
        for m in raw_msgs:
            r_enum = MessageRole.USER if m["role"] == "user" else (
                MessageRole.ASSISTANT if m["role"] == "assistant" else MessageRole.SYSTEM
            )
            msg = Message(
                id=uuid.UUID(m["id"]),
                conversation_id=uuid.UUID(conversation_id),
                role=r_enum,
                content=m["content"],
                metadata_=m["metadata"],
                created_at=m["created_at"],
            )
            result.append(msg)
        return result

    async def save_prediction(
        self,
        conversation_id: str,
        prediction_data: Dict[str, Any],
        model_name: str = "XGBoost",
    ) -> Prediction:
        """Save a prediction result to DB and in-memory."""
        conv_uuid = uuid.UUID(conversation_id)
        now = datetime.utcnow()

        pred_dict = {
            "id": str(uuid.uuid4()),
            "prediction_class": prediction_data.get("prediction_class", 0),
            "model_score": prediction_data.get("model_score", 0.0),
            "features_used": prediction_data.get("features_used", []),
            "missing_features": prediction_data.get("missing_features", []),
            "feature_state_snapshot": prediction_data.get("feature_state", {}),
            "model_name": model_name,
            "created_at": now,
        }

        if conversation_id not in _inmemory_predictions:
            _inmemory_predictions[conversation_id] = []
        _inmemory_predictions[conversation_id].append(pred_dict)

        pred = Prediction(
            id=uuid.UUID(pred_dict["id"]),
            conversation_id=conv_uuid,
            prediction_class=pred_dict["prediction_class"],
            model_score=pred_dict["model_score"],
            features_used=pred_dict["features_used"],
            missing_features=pred_dict["missing_features"],
            feature_state_snapshot=pred_dict["feature_state_snapshot"],
            model_name=model_name,
            created_at=now,
        )

        if self.session:
            try:
                self.session.add(pred)
                await self.session.commit()
            except Exception as e:
                logger.warning(f"DB error saving prediction, using in-memory: {e}")
                await self.session.rollback()

        return pred

    async def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List active conversations ordered by last update."""
        if self.session:
            try:
                stmt = select(Conversation).where(Conversation.is_active == True).order_by(Conversation.updated_at.desc()).limit(limit)
                res = await self.session.execute(stmt)
                convs = res.scalars().all()
                if convs:
                    results = []
                    for c in convs:
                        count_stmt = select(func.count(Message.id)).where(Message.conversation_id == c.id)
                        c_res = await self.session.execute(count_stmt)
                        msg_count = c_res.scalar() or 0
                        results.append({
                            "id": str(c.id),
                            "title": c.title,
                            "is_active": c.is_active,
                            "created_at": c.created_at,
                            "updated_at": c.updated_at,
                            "message_count": msg_count,
                        })
                    return results
            except Exception as e:
                logger.warning(f"DB error listing conversations, using in-memory: {e}")

        # In-memory fallback
        results = []
        for cid, data in _inmemory_conversations.items():
            if data.get("is_active", True):
                msgs = _inmemory_messages.get(cid, [])
                results.append({
                    "id": cid,
                    "title": data.get("title", "New Maintenance Assessment"),
                    "is_active": True,
                    "created_at": data.get("created_at", datetime.utcnow()),
                    "updated_at": data.get("updated_at", datetime.utcnow()),
                    "message_count": len(msgs),
                })
        return results[:limit]

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Soft delete / deactivate a conversation."""
        if conversation_id in _inmemory_conversations:
            _inmemory_conversations[conversation_id]["is_active"] = False

        if self.session:
            try:
                conv_uuid = uuid.UUID(conversation_id)
                stmt = update(Conversation).where(Conversation.id == conv_uuid).values(is_active=False)
                await self.session.execute(stmt)
                await self.session.commit()
                return True
            except Exception as e:
                logger.warning(f"Failed to delete conversation {conversation_id} in DB: {e}")
                await self.session.rollback()

        return True

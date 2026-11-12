"""
Conversation Service — Orchestrates conversations through the ADK multi-agent system,
with database persistence and context compaction.

This is the ONLY execution path. There is no legacy fallback.
If the ADK runner fails, errors surface explicitly and safely.
"""

from typing import Dict, Any, Optional, List, AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.agent.adk_agents.runner import get_adk_runner
from backend.app.agent.state import AgentState
from backend.app.memory.repository import MemoryRepository
from backend.app.services.prediction_service import get_predictor
from backend.app.context.manager import ContextManager
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class ConversationService:
    """High-level service orchestrating conversation flow using Google ADK multi-agent system."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = MemoryRepository(db)
        self.predictor = get_predictor()
        self.adk_runner = get_adk_runner()
        self.context_mgr = ContextManager()

    async def handle_message(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message and return complete response.

        Uses the Google ADK multi-agent system exclusively.
        Errors surface explicitly — no silent fallbacks.
        """
        logger.info(f"[REQUEST_RECEIVED] conv_id={conversation_id} msg={user_message!r}")

        # 1. Get or create conversation from DB
        conv = await self.repo.get_or_create_conversation(conversation_id)
        conv_id = str(conv.id)

        # 2. Restore AgentState from DB
        persisted = conv.feature_state or {}
        # New records store the complete authoritative AgentState in this JSON
        # column.  The fallback preserves compatibility with older rows.
        if "feature_state" not in persisted:
            persisted = {
                "feature_state": persisted,
                "asked_features": conv.asked_questions or [],
                "declined_features": conv.declined_features or [],
            }
        state = AgentState.from_json(persisted, conversation_id=conv_id)
        logger.info(f"[STATE_LOADED] conv_id={conv_id} known_features={list(state.get_known_features().keys())}")

        # 3. Save incoming user message
        await self.repo.add_message(conv_id, "user", user_message)

        # 4. Fetch history for agent context
        db_messages = await self.repo.get_messages(conv_id)
        history = [
            {"role": m.role.value if hasattr(m.role, "value") else str(m.role), "content": m.content}
            for m in db_messages[:-1]  # exclude current user message (controller adds it)
        ]

        # 5. Execute ADK Multi-Agent System turn
        # (ControllerAgent -> InformationAgent/PredictionAgent/ResponseAgent -> JudgeAgent)
        result = await self.adk_runner.run_turn(
            user_message=user_message,
            conversation_id=conv_id,
            agent_state=state,
            conversation_history=history,
        )

        # 6. Save assistant response message
        msg_metadata = {}
        if result.get("prediction"):
            msg_metadata["prediction"] = result["prediction"]
        if result.get("explanation"):
            msg_metadata["explanation"] = result["explanation"]
        if result.get("judge_result"):
            msg_metadata["judge_result"] = result["judge_result"]

        await self.repo.add_message(conv_id, "assistant", result["response"], metadata=msg_metadata)

        # 7. Save prediction if generated
        if result.get("prediction"):
            await self.repo.save_prediction(conv_id, result["prediction"])

        # 8. Persist updated AgentState back to DB
        await self.repo.save_agent_state(conv_id, state)

        logger.info(f"[RESPONSE_SENT] conv_id={conv_id} len={len(result.get('response', ''))}")

        return {
            "conversation_id": conv_id,
            "message": result["response"],
            "prediction": result.get("prediction"),
            "explanation": result.get("explanation"),
            "feature_state": state.to_context_dict(),
            "judge_result": result.get("judge_result"),
        }

    async def handle_message_stream(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream processing of user message via ADK multi-agent runner.
        Uses ADK runner exclusively — no legacy fallback.
        """
        logger.info(f"[REQUEST_RECEIVED] [STREAM] conv_id={conversation_id} msg={user_message!r}")
        try:
            conv = await self.repo.get_or_create_conversation(conversation_id)
            conv_id = str(conv.id)

            persisted = conv.feature_state or {}
            if "feature_state" not in persisted:
                persisted = {
                    "feature_state": persisted,
                    "asked_features": conv.asked_questions or [],
                    "declined_features": conv.declined_features or [],
                }
            state = AgentState.from_json(persisted, conversation_id=conv_id)
            logger.info(f"[STATE_LOADED] [STREAM] conv_id={conv_id} known_features={list(state.get_known_features().keys())}")

            await self.repo.add_message(conv_id, "user", user_message)

            db_messages = await self.repo.get_messages(conv_id)
            history = [
                {"role": m.role.value if hasattr(m.role, "value") else str(m.role), "content": m.content}
                for m in db_messages[:-1]
            ]

            # Send conversation ID first
            yield {"type": "conversation_id", "content": conv_id}

            full_text = []
            latest_pred = None
            latest_explanation = None

            async for chunk in self.adk_runner.run_turn_stream(
                user_message=user_message,
                conversation_id=conv_id,
                agent_state=state,
                conversation_history=history,
            ):
                if chunk.get("type") == "text" and chunk.get("content"):
                    full_text.append(chunk["content"])
                elif chunk.get("type") == "prediction":
                    latest_pred = chunk.get("data")
                elif chunk.get("type") == "explanation":
                    latest_explanation = chunk.get("data")
                yield chunk

            # Save assistant message and state after stream finishes
            assistant_text = "".join(full_text)
            if assistant_text:
                msg_meta = {}
                if latest_pred:
                    msg_meta["prediction"] = latest_pred
                if latest_explanation:
                    msg_meta["explanation"] = latest_explanation
                await self.repo.add_message(conv_id, "assistant", assistant_text, metadata=msg_meta)

            if latest_pred:
                await self.repo.save_prediction(conv_id, latest_pred)

            await self.repo.save_agent_state(conv_id, state)
            logger.info(f"[RESPONSE_SENT] [STREAM] conv_id={conv_id} len={len(assistant_text)}")

        except Exception as e:
            logger.exception(f"[STREAM_ERROR] Error in handle_message_stream: {e}")
            fallback_text = (
                "I ran into an unexpected processing issue while evaluating your vehicle details. "
                "Please try sending your message again."
            )
            yield {"type": "text", "content": fallback_text}
            yield {"type": "done"}

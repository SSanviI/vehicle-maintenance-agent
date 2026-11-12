"""
Context Manager — Manages prompt context for the LLM agent.
Constructs token-budget-aware prompts combining system prompt, summary, state, and recent messages.
"""

from typing import List, Dict, Any, Optional, Tuple
from backend.app.agent.state import AgentState
from backend.app.context.compactor import ContextCompactor
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class ContextManager:
    """
    Assembles prompt context within specified token budgets.
    
    Structure sent to LLM:
    1. System Prompt (agent identity & instructions)
    2. Compacted Conversation Summary (old history)
    3. Vehicle & Prediction State
    4. Recent N Messages (uncompacted raw chat)
    5. Current User Message
    """

    def __init__(self, compactor: Optional[ContextCompactor] = None):
        self.settings = get_settings()
        self.compactor = compactor or ContextCompactor()
        self.recent_messages_limit = self.settings.max_recent_messages
        self.compaction_threshold = self.settings.compaction_threshold

    def build_history_for_agent(
        self,
        messages: List[Dict[str, Any]],
        summary: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Prepare context message history for the agent.
        Combines summary + recent uncompacted messages.
        """
        formatted_messages = []

        # If summary exists, include it as assistant/system context
        if summary:
            formatted_messages.append({
                "role": "assistant",
                "content": f"[CONVERSATION SUMMARY]\n{summary}"
            })

        # Include recent raw messages
        recent = messages[-self.recent_messages_limit:]
        for m in recent:
            formatted_messages.append({
                "role": m.get("role", "user"),
                "content": m.get("content", "")
            })

        return formatted_messages

    def should_compact(self, message_count: int) -> bool:
        """Check if message count exceeds threshold requiring compaction."""
        return message_count >= self.compaction_threshold

    async def maybe_compact(
        self,
        messages: List[Dict[str, Any]],
        current_summary: Optional[str] = None,
        feature_state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Compact messages if threshold exceeded.
        Returns (new_summary, remaining_uncompacted_messages).
        """
        if not self.should_compact(len(messages)):
            return current_summary, messages

        # Messages to compact (all except recent 10)
        split_idx = len(messages) - 10
        to_compact = messages[:split_idx]
        remaining = messages[split_idx:]

        new_summary = await self.compactor.compact_messages(
            messages=to_compact,
            existing_summary=current_summary,
            feature_state_snapshot=feature_state_snapshot,
        )

        logger.info(f"Compacted {len(to_compact)} messages into summary. {len(remaining)} messages remain in buffer.")
        return new_summary, remaining

"""
Context Compactor — Compacts old conversation messages into a structured summary.
Preserves vehicle identity, known/unknown/declined features, predictions, and decisions.
Discards small talk and redundant chatter.
"""

from typing import List, Dict, Any, Optional
from backend.app.llm.base import LLMMessage, LLMRole
from backend.app.llm.router import LLMRouter, get_llm_router
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

COMPACTOR_SYSTEM_PROMPT = """You are a conversation compaction specialist for a vehicle maintenance AI agent.

Your job is to condense a long chat history into a concise, high-density structured summary.

PRESERVE:
- Vehicle age, model, fuel type, transmission (if mentioned)
- Known feature values (Brake_Condition, Battery_Status, Reported_Issues, etc.)
- Features the user explicitly declined to answer ("don't know")
- Predictions made (model score, risk level)
- Key user decisions or preferences
- Specific complaints or symptoms described

DISCARD:
- Pleasantries, greetings, farewells
- Repetitive questions or answers
- Off-topic conversation

Format the output cleanly:
Vehicle Identity: <details or Unknown>
Known Features: <list of key=value>
Unknown/Declined: <list of features user didn't know>
Predictions & Risk: <latest score and risk level>
Key Highlights: <bullet points of important conversation points>
"""


class ContextCompactor:
    """Compacts old messages into a summary to keep token usage within budget."""

    def __init__(self, router: Optional[LLMRouter] = None):
        self.router = router or get_llm_router()

    async def compact_messages(
        self,
        messages: List[Dict[str, str]],
        existing_summary: Optional[str] = None,
        feature_state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Compact a list of message dicts ({role, content}) into a single summary string.
        """
        if not messages:
            return existing_summary or "No previous conversation history."

        conversation_text = ""
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            conversation_text += f"{role}: {content}\n"

        prompt = ""
        if existing_summary:
            prompt += f"Existing Summary:\n{existing_summary}\n\n"
        if feature_state_snapshot:
            prompt += f"Structured State Snapshot:\n{feature_state_snapshot}\n\n"

        prompt += f"New Messages to Compact:\n{conversation_text}\n\nCreate an updated summary:"

        llm_messages = [
            LLMMessage(role="system", content=COMPACTOR_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await self.router.generate(
                messages=llm_messages,
                role=LLMRole.FAST,
                temperature=0.2,
                max_tokens=600,
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to compact context: {e}")
            # Fallback simple string concatenation
            return (existing_summary + "\n" if existing_summary else "") + f"Recent turns ({len(messages)} messages) discussed vehicle details."

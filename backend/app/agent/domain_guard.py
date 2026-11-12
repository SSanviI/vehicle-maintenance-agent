"""
Domain Guard — Enforces closed-domain boundaries for Vehicle Maintenance AI Agent.
Prevents agent from answering off-topic queries (jokes, weather, purchasing, trivia)
and redirects user back to vehicle maintenance prediction.
"""

import re
from typing import Dict, Any, Tuple
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

OFF_TOPIC_PATTERNS = [
    r"\b(tell|say|know)\s+(me\s+)?a?\s*joke\b",
    r"\bjoke\b",
    r"\b(what'?s|how'?s)\s+(the\s+)?weather\b",
    r"\b(which|what)\s+(laptop|phone|computer|tv|house)\s+(should|can|to)\s+i?\s*buy\b",
    r"\b(which|what)\s+car\s+(should|can|to)\s+i?\s*buy\b",
    r"\bwho\s+won\s+the\s+(game|match|superbowl|world cup)\b",
    r"\b(recipe|how to cook|bake|dinner)\b",
    r"\b(politics|president|election)\b",
    r"\bwrite\s+(me\s+)?a\s+(poem|song|story|code|script)\b",
    r"\bpoem\b",
]

class DomainGuard:
    """Enforces closed domain for vehicle maintenance prediction."""

    @staticmethod
    def is_in_domain(user_message: str) -> Tuple[bool, str]:
        """
        Check if message is within the vehicle maintenance domain.
        Returns (is_allowed, redirection_message).
        """
        cleaned = user_message.strip().lower()

        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, cleaned):
                logger.info(f"DomainGuard intercepted off-topic query: '{user_message[:50]}...'")
                redirect_msg = (
                    "I am your AutoCare AI vehicle maintenance assistant. My sole purpose is to help assess "
                    "your vehicle's maintenance needs and provide model-based predictions. Let's return to "
                    "discussing your vehicle's condition."
                )
                return False, redirect_msg

        return True, ""

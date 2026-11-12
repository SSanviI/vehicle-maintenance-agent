"""
Question Guard — Hard feature boundary & question candidate validator.
Guarantees the agent ONLY asks about features in FINAL_MODEL_FEATURES
and rejects unsupported automotive features (Suspension, Mileage, Engine Oil, etc.).
"""

from typing import List, Optional, Dict, Any
from backend.app.ml.feature_schema import is_valid_feature, MODEL_FEATURE_SCHEMA
from backend.app.agent.state import AgentState
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class QuestionGuard:
    """Hard boundary validator for agent questioning."""

    def __init__(self, allowed_features: List[str] = None):
        # Default to the central model feature schema
        self.allowed_features = allowed_features or MODEL_FEATURE_SCHEMA

    def is_legal_model_feature(self, feature_name: str) -> bool:
        """Check if feature is strictly part of FINAL_MODEL_FEATURES."""
        if not feature_name:
            return False
        cleaned = str(feature_name).strip().replace(" ", "_").lower()
        for allowed in self.allowed_features:
            if allowed.lower() == cleaned or allowed.lower().replace("_", "") == cleaned.replace("_", ""):
                return True
        return False

    def get_legal_question_candidates(self, state: AgentState) -> List[str]:
        """
        Pre-calculate legal question candidates from state BEFORE question generation.
        Candidate must be:
        1. In FINAL_MODEL_FEATURES
        2. Status != 'known'
        3. Feature not in state.declined_features
        """
        askable = []
        for feat in self.allowed_features:
            if feat in state.declined_features:
                continue
            feat_info = state.feature_state.get(feat, {})
            if feat_info.get("status") != "known":
                askable.append(feat)
        return askable

    def validate_question(self, target_feature: str, state: AgentState) -> bool:
        """
        Hard validation for any question attempt.
        Returns True ONLY if target_feature is legal and askable.
        """
        if not self.is_legal_model_feature(target_feature):
            logger.warning(
                f"QuestionGuard REJECTED illegal question attempt: '{target_feature}' "
                f"is NOT in FINAL_MODEL_FEATURES!"
            )
            return False

        # Match exact case
        matched_feature = None
        for allowed in self.allowed_features:
            if allowed.lower() == target_feature.lower().replace(" ", "_"):
                matched_feature = allowed
                break

        if not matched_feature:
            return False

        if matched_feature in state.declined_features:
            logger.warning(
                f"QuestionGuard REJECTED question attempt: '{matched_feature}' "
                f"was already declined by user!"
            )
            return False

        if state.feature_state.get(matched_feature, {}).get("status") == "known":
            logger.warning(
                f"QuestionGuard REJECTED question attempt: '{matched_feature}' "
                f"is already known!"
            )
            return False

        return True

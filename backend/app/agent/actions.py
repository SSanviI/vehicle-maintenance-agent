"""
Agent Actions / Tools — Internal tools the agent can invoke.
All tool arguments are validated. No arbitrary code execution.
"""

from typing import Dict, Any, Optional
from backend.app.agent.state import AgentState
from backend.app.ml.predict import MaintenancePredictor
from backend.app.ml.preprocessing import FEATURE_DISPLAY_NAMES, SELECTED_FEATURES
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class AgentTools:
    """
    Internal tools available to the agent.
    The LLM selects tools through structured outputs; this class executes them.
    """

    def __init__(self, predictor: MaintenancePredictor):
        self.predictor = predictor

    def get_current_state(self, state: AgentState) -> Dict[str, Any]:
        """Get the current vehicle assessment state."""
        return state.to_context_dict()

    def predict_maintenance(self, state: AgentState) -> Dict[str, Any]:
        """
        Run ML prediction on current feature state.
        
        This is the ONLY path to a maintenance prediction.
        MUST be called EXACTLY ONCE per user turn.
        The returned prediction object is the SINGLE SOURCE OF TRUTH.
        
        All consumers (LLM explanation, prediction card, DB, frontend)
        MUST use this same returned object. No re-inference allowed.
        """
        if not self.predictor.is_loaded:
            return {
                "error": "ML model not available",
                "success": False,
            }

        result = self.predictor.predict(state.feature_state)

        if result.get("prediction") is not None:
            # Store previous score for UI comparison (but NOT passed to LLM)
            prev = state.get_latest_prediction()
            if prev:
                result["previous_score"] = prev.get("model_score")

            state.add_prediction(result)

            logger.info(
                f"Prediction stored [id={result.get('prediction_id', 'N/A')[:8]}]: "
                f"score={result.get('model_score')}"
            )

        return result

    def update_feature(self, state: AgentState, feature_name: str, 
                       value: Any, status: str = "known", 
                       source: str = "user") -> Dict[str, Any]:
        """Update a feature in the agent state."""
        if feature_name not in state.feature_state:
            return {
                "success": False,
                "error": f"Unknown feature: {feature_name}",
            }

        # Validate categorical values
        from backend.app.ml.preprocessing import CATEGORICAL_FEATURES, CATEGORICAL_VALUES
        if feature_name in CATEGORICAL_FEATURES and status == "known":
            valid_values = [v for v in CATEGORICAL_VALUES[feature_name] if v != "UNKNOWN"]
            if value not in valid_values:
                return {
                    "success": False,
                    "error": f"Invalid value '{value}' for {feature_name}. Valid: {valid_values}",
                }

        # Validate numerical values
        from backend.app.ml.preprocessing import NUMERICAL_FEATURES
        if feature_name in NUMERICAL_FEATURES and status == "known":
            try:
                value = int(value) if isinstance(value, (int, float, str)) else value
                if not isinstance(value, (int, float)):
                    return {"success": False, "error": f"Expected number for {feature_name}"}
                if value < 0:
                    return {"success": False, "error": f"{feature_name} cannot be negative"}
            except (ValueError, TypeError):
                return {"success": False, "error": f"Expected number for {feature_name}"}

        changed = state.update_feature(feature_name, value, status, source)
        return {
            "success": True,
            "changed": changed,
            "feature": feature_name,
            "value": value,
            "status": status,
        }

    def mark_feature_declined(self, state: AgentState, feature_name: str) -> Dict[str, Any]:
        """Mark a feature as declined by user. Will NOT ask again."""
        if feature_name not in state.feature_state:
            return {"success": False, "error": f"Unknown feature: {feature_name}"}

        state.mark_declined(feature_name)
        return {
            "success": True,
            "feature": feature_name,
            "status": "unavailable",
            "note": f"Will not ask about {feature_name} again",
        }

    def get_next_question(self, state: AgentState) -> Dict[str, Any]:
        """Determine the next useful question to ask."""
        next_feat = state.get_next_question_feature()

        if next_feat is None:
            return {
                "has_question": False,
                "note": "All features are either known or declined",
            }

        display_name = FEATURE_DISPLAY_NAMES.get(next_feat, next_feat)
        state.mark_asked(next_feat)

        return {
            "has_question": True,
            "feature": next_feat,
            "display_name": display_name,
        }

    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get information about a specific feature."""
        if feature_name not in SELECTED_FEATURES:
            return {"error": f"Unknown feature: {feature_name}"}

        from backend.app.ml.preprocessing import CATEGORICAL_FEATURES, CATEGORICAL_VALUES, NUMERICAL_FEATURES
        
        info = {
            "name": feature_name,
            "display_name": FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
            "required_by_model": True,
        }

        if feature_name in CATEGORICAL_FEATURES:
            info["type"] = "categorical"
            info["valid_values"] = [v for v in CATEGORICAL_VALUES[feature_name] if v != "UNKNOWN"]
        elif feature_name in NUMERICAL_FEATURES:
            info["type"] = "numerical"
            info["description"] = f"A number representing {FEATURE_DISPLAY_NAMES.get(feature_name, '')}"

        return info

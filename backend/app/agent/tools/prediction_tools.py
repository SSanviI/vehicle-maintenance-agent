"""
Prediction Tools — Hard programmatic gate and ML prediction execution.
Guarantees:
1. Random Forest prediction ONLY runs when known_feature_count >= 2.
2. The ML model is the sole authority for risk scores and classifications.
3. Every prediction generates a unique prediction_id (UUID).
4. No LLM can invent, estimate, or modify prediction results.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from backend.app.agent.state.state import AgentState, MIN_FEATURES_FOR_PREDICTION
from backend.app.services.prediction_service import get_predictor
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def execute_prediction_gate(
    state: AgentState,
    force_recalculate: bool = False,
) -> Dict[str, Any]:
    """
    Execute the deterministic prediction gate and run Random Forest inference if allowed.

    Args:
        state: Authoritative AgentState
        force_recalculate: If True, re-runs ML inference even if state didn't change

    Returns:
        Dict with:
            - allowed: bool (True only if known_count >= 2)
            - prediction: Dict or None (the ML result)
            - reason: str explanation
            - known_count: int
    """
    known_count = state.known_feature_count()

    # Hard programmatic gate — NEVER allow prediction with < 2 features
    if known_count < MIN_FEATURES_FOR_PREDICTION:
        logger.info(
            f"[PREDICTION_GATE] Blocked: {known_count} known feature(s). "
            f"Requires at least {MIN_FEATURES_FOR_PREDICTION}."
        )
        return {
            "allowed": False,
            "prediction": None,
            "reason": f"Need at least {MIN_FEATURES_FOR_PREDICTION} valid features before calculating risk (currently have {known_count}).",
            "known_count": known_count,
            "required_count": MIN_FEATURES_FOR_PREDICTION,
        }

    predictor = get_predictor()
    if not predictor.is_loaded:
        logger.error("[PREDICTION_GATE] ML model predictor not loaded")
        return {
            "allowed": False,
            "prediction": None,
            "reason": "ML model currently unavailable.",
            "known_count": known_count,
            "required_count": MIN_FEATURES_FOR_PREDICTION,
        }

    try:
        # Run genuine Random Forest prediction from partial feature state
        result = predictor.predict(state.feature_state)
        if result.get("error"):
            logger.error(f"[PREDICTION_GATE] Model error: {result['error']}")
            return {
                "allowed": False,
                "prediction": None,
                "reason": result["error"],
                "known_count": known_count,
            }

        # Store the current-turn result in prediction history.
        result["turn"] = state.turn_count
        state.add_prediction(result)
        logger.info(
            f"[PREDICTION_GATE] ML SUCCESS [{result.get('prediction_id', '')[:8]}]: "
            f"class={result.get('prediction_class')}, prob={result.get('maintenance_probability')}, "
            f"risk={result.get('risk_level')}, score={result.get('model_score')}"
        )

        return {
            "allowed": True,
            "prediction": result,
            "reason": "Prediction calculated successfully by Random Forest model.",
            "known_count": known_count,
        }

    except Exception as e:
        logger.error(f"[PREDICTION_GATE] Exception running ML prediction: {e}", exc_info=True)
        return {
            "allowed": False,
            "prediction": None,
            "reason": f"Prediction execution failed: {str(e)}",
            "known_count": known_count,
        }

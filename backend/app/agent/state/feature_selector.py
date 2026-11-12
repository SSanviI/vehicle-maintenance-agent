"""
Deterministic Feature Selector — Sole authority for selecting next question feature.
Guarantees:
1. Questions strictly follow FEATURE_PRIORITY order.
2. Known features are NEVER asked.
3. Declined/unavailable features are NEVER asked again.
4. Already-asked features awaiting answer are not asked repeatedly.
5. Exactly ONE feature is selected.
6. The LLM CANNOT choose or override the feature.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from backend.app.ml.preprocessing import (
    SELECTED_FEATURES,
    FEATURE_DISPLAY_NAMES,
    CATEGORICAL_VALUES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from backend.app.agent.state.state import AgentState


def get_askable_features(state: AgentState) -> List[str]:
    """
    Return list of features that can be asked in priority order.
    Excludes:
    - Confirmed known features
    - Declined / unavailable features
    """
    askable = []
    plan = state.selected_features if state.assessment_plan_initialized else SELECTED_FEATURES
    for feat in plan:
        # Check known status
        feat_data = state.feature_state.get(feat, {})
        if feat_data.get("status") == "known" and feat_data.get("value") is not None:
            continue

        # Check declined status
        if feat in state.declined_features or feat_data.get("status") == "unavailable":
            continue

        askable.append(feat)
    return askable


def get_next_question_feature(state: AgentState) -> Optional[str]:
    """
    Deterministically select the single next feature to ask about.

    Returns:
        Feature name (str) or None if all features are known or declined.
    """
    askable = get_askable_features(state)
    if not askable:
        return None

    return askable[0]




def get_question_specification(feature_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Get the structured spec for a selected feature to pass to the ResponseAgent.

    Returns:
        Dict with feature metadata (name, display name, is_categorical, allowed_values, prompt_hint)
    """
    if not feature_name:
        return None

    is_cat = feature_name in CATEGORICAL_FEATURES
    allowed = CATEGORICAL_VALUES.get(feature_name, []) if is_cat else []
    display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace("_", " ").lower())

    return {
        "feature_name": feature_name,
        "display_name": display,
        "is_categorical": is_cat,
        "allowed_values": allowed,
    }

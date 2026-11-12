"""
ADK Tool Functions — All FunctionTool-compatible functions for the multi-agent system.

These are plain Python functions (async or sync) that ADK wraps as FunctionTools.
Each function has clear docstrings because ADK uses them to auto-generate tool schemas.

Tool Categories:
  - Schema Tools: get_model_features, validate_feature
  - State Tools: update_feature_state, mark_feature_unavailable, get_next_feature_to_ask
  - Prediction Tools: run_ml_prediction
  - Domain Tools: check_domain
  - Context Tools: compact_conversation_context
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

from backend.app.core.logging import get_logger
from backend.app.agent.domain_guard import DomainGuard
from backend.app.agent.question_guard import QuestionGuard
from backend.app.ml.preprocessing import (
    SELECTED_FEATURES, FEATURE_DISPLAY_NAMES, CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES, NUMERICAL_FEATURES, FEATURE_PRIORITY
)
from backend.app.agent.state import MIN_FEATURES_FOR_PREDICTION

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons (to avoid circular imports at module load time)
# ---------------------------------------------------------------------------
_predictor = None
_question_guard = QuestionGuard(SELECTED_FEATURES)

# Path to feature schema JSON
_SCHEMA_PATH = Path(__file__).parent.parent.parent.parent.parent / "models" / "feature_schema.json"


def _get_predictor():
    global _predictor
    if _predictor is None:
        from backend.app.services.prediction_service import get_predictor
        _predictor = get_predictor()
    return _predictor


# ---------------------------------------------------------------------------
# Schema / Validation Tools
# ---------------------------------------------------------------------------

def get_model_features() -> str:
    """
    Returns the complete vehicle maintenance model feature schema as JSON.
    
    This is the AUTHORITATIVE source of valid features. The agent must ONLY
    ask about features present in this schema. Never invent features.
    
    Returns:
        JSON string with feature schema including names, types, and valid values.
    """
    try:
        if _SCHEMA_PATH.exists():
            with open(_SCHEMA_PATH, "r") as f:
                schema = json.load(f)
        else:
            # Build from preprocessing module as fallback
            schema = {
                "selected_features": SELECTED_FEATURES,
                "numerical_features": NUMERICAL_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "categorical_values": {k: [v for v in vals if v != "UNKNOWN"] for k, vals in CATEGORICAL_VALUES.items()},
                "feature_display_names": FEATURE_DISPLAY_NAMES,
                "feature_priority": FEATURE_PRIORITY,
            }
        logger.info("[TOOL] get_model_features called")
        return json.dumps(schema, indent=2)
    except Exception as e:
        logger.error(f"get_model_features error: {e}")
        return json.dumps({"error": str(e), "selected_features": SELECTED_FEATURES})


def validate_feature(feature_name: str) -> str:
    """
    Validates whether a feature name is part of the actual ML model schema.
    
    CRITICAL: The agent MUST call this before using any feature name.
    Any feature NOT in the model schema must be REJECTED — never add it to state.
    
    Args:
        feature_name: The feature name to validate (e.g. "Vehicle_Age", "Battery_Status")
    
    Returns:
        JSON with {"valid": bool, "canonical_name": str or null, "reason": str}
    """
    is_valid = _question_guard.is_legal_model_feature(feature_name)
    
    # Find canonical name
    canonical = None
    if is_valid:
        cleaned = feature_name.strip().replace(" ", "_").lower()
        for feat in SELECTED_FEATURES:
            if feat.lower() == cleaned or feat.lower().replace("_", "") == cleaned.replace("_", ""):
                canonical = feat
                break

    result = {
        "valid": is_valid,
        "canonical_name": canonical,
        "reason": "Feature is in the ML model schema" if is_valid else f"REJECTED: '{feature_name}' is NOT in the model schema. Valid features: {SELECTED_FEATURES}",
    }
    if not is_valid:
        logger.warning(f"[TOOL] validate_feature REJECTED: '{feature_name}'")
    else:
        logger.info(f"[TOOL] validate_feature OK: '{feature_name}' -> '{canonical}'")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# State Manipulation Tools
# ---------------------------------------------------------------------------

def update_feature_state(
    state_json: str,
    feature_name: str,
    value: str,
    status: str = "known",
    source: str = "user",
) -> str:
    """
    Updates a vehicle feature in the agent state.
    
    IMPORTANT: Only call this for validated model features.
    For categorical features, value must match one of the valid categorical values.
    For numerical features, value must be a number (as string, e.g. "6").
    
    Args:
        state_json: Current AgentState serialized as JSON string
        feature_name: The exact canonical feature name (e.g. "Vehicle_Age")
        value: The value to set (e.g. "6" for age, "Good" for condition)
        status: Feature status — "known", "unknown", or "unavailable"
        source: Who provided this — "user" or "inferred"
    
    Returns:
        JSON with {"success": bool, "state_json": updated_state, "changed": bool, "error": str or null}
    """
    try:
        from backend.app.agent.state import AgentState
        from backend.app.agent.actions import AgentTools

        # Validate feature first
        validation = json.loads(validate_feature(feature_name))
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "state_json": state_json,
                "changed": False,
                "error": validation["reason"],
            })

        canonical = validation["canonical_name"]
        state = AgentState.from_json(json.loads(state_json) if state_json else {})
        tools = AgentTools(_get_predictor())

        # Parse value for numeric features
        parsed_value = value
        if canonical in NUMERICAL_FEATURES and status == "known":
            try:
                parsed_value = int(float(str(value)))
            except (ValueError, TypeError):
                return json.dumps({
                    "success": False,
                    "state_json": state_json,
                    "changed": False,
                    "error": f"Expected a number for {canonical}, got: {value}",
                })

        result = tools.update_feature(state, canonical, parsed_value, status, source)

        if result["success"]:
            updated_json = json.dumps(state.to_json())
            logger.info(f"[TOOL] update_feature_state: {canonical}={parsed_value} (status={status}) changed={result.get('changed')}")
            return json.dumps({
                "success": True,
                "state_json": updated_json,
                "changed": result.get("changed", False),
                "feature": canonical,
                "value": parsed_value,
                "status": status,
                "error": None,
            })
        else:
            return json.dumps({
                "success": False,
                "state_json": state_json,
                "changed": False,
                "error": result.get("error", "Unknown error"),
            })

    except Exception as e:
        logger.error(f"update_feature_state error: {e}")
        return json.dumps({"success": False, "state_json": state_json, "changed": False, "error": str(e)})


def mark_feature_unavailable(state_json: str, feature_name: str) -> str:
    """
    Marks a feature as unavailable because the user said they don't know.
    
    This feature will NEVER be asked again in this conversation.
    The state stores it as status="unavailable", value=null.
    
    Use this when the user says variants of:
    "I don't know", "not sure", "no idea", "can't remember", "unavailable"
    
    Args:
        state_json: Current AgentState serialized as JSON string
        feature_name: The feature the user declined to answer
    
    Returns:
        JSON with {"success": bool, "state_json": updated_state, "error": str or null}
    """
    try:
        from backend.app.agent.state import AgentState
        from backend.app.agent.actions import AgentTools

        validation = json.loads(validate_feature(feature_name))
        if not validation["valid"]:
            return json.dumps({"success": False, "state_json": state_json, "error": validation["reason"]})

        canonical = validation["canonical_name"]
        state = AgentState.from_json(json.loads(state_json) if state_json else {})
        tools = AgentTools(_get_predictor())

        result = tools.mark_feature_declined(state, canonical)
        updated_json = json.dumps(state.to_json())
        logger.info(f"[TOOL] mark_feature_unavailable: {canonical} -> status=unavailable (never ask again)")
        return json.dumps({
            "success": result["success"],
            "state_json": updated_json,
            "feature": canonical,
            "status": "unavailable",
            "note": f"Will NOT ask about {canonical} again in this conversation",
            "error": result.get("error"),
        })

    except Exception as e:
        logger.error(f"mark_feature_unavailable error: {e}")
        return json.dumps({"success": False, "state_json": state_json, "error": str(e)})


def get_next_feature_to_ask(state_json: str) -> str:
    """
    Determines the next most useful model feature to ask the user about.
    
    Only returns features that are:
    1. In the actual ML model schema
    2. Not already known
    3. Not marked as unavailable (user said "I don't know")
    4. Not previously asked without answer
    
    Returns null if no more features need to be asked.
    
    Args:
        state_json: Current AgentState serialized as JSON string
    
    Returns:
        JSON with {"has_question": bool, "feature": str or null, "display_name": str or null,
                   "type": "categorical"|"numerical", "valid_values": list or null}
    """
    try:
        from backend.app.agent.state import AgentState

        state = AgentState.from_json(json.loads(state_json) if state_json else {})
        next_feat = state.get_next_question_feature()

        if next_feat is None:
            logger.info("[TOOL] get_next_feature_to_ask: no more features to ask")
            return json.dumps({
                "has_question": False,
                "feature": None,
                "display_name": None,
                "type": None,
                "valid_values": None,
                "note": "All model features are either known or unavailable",
            })

        feat_type = "categorical" if next_feat in CATEGORICAL_FEATURES else "numerical"
        valid_values = None
        if feat_type == "categorical":
            valid_values = [v for v in CATEGORICAL_VALUES.get(next_feat, []) if v != "UNKNOWN"]

        logger.info(f"[TOOL] get_next_feature_to_ask -> {next_feat}")
        return json.dumps({
            "has_question": True,
            "feature": next_feat,
            "display_name": FEATURE_DISPLAY_NAMES.get(next_feat, next_feat),
            "type": feat_type,
            "valid_values": valid_values,
            "note": f"Ask about {next_feat} — it is the next most useful model feature",
        })

    except Exception as e:
        logger.error(f"get_next_feature_to_ask error: {e}")
        return json.dumps({"has_question": False, "feature": None, "error": str(e)})


def get_current_state_summary(state_json: str) -> str:
    """
    Returns a structured summary of the current vehicle assessment state.
    
    Shows which features are known, unknown, and unavailable,
    plus the latest prediction result if available.
    
    Args:
        state_json: Current AgentState serialized as JSON string
    
    Returns:
        JSON summary of known/unknown/unavailable features and latest prediction
    """
    try:
        from backend.app.agent.state import AgentState

        state = AgentState.from_json(json.loads(state_json) if state_json else {})
        summary = state.to_context_dict()
        logger.info(f"[TOOL] get_current_state_summary: {len(summary.get('known_features', {}))} known features")
        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error(f"get_current_state_summary error: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# ML Prediction Tool
# ---------------------------------------------------------------------------

def run_ml_prediction(state_json: str) -> str:
    """
    Runs the ML maintenance prediction engine on current feature state.

    CRITICAL RULES:
    - This is the ONLY authorized way to make a maintenance prediction.
    - The LLM must NEVER calculate or estimate the prediction itself.
    - Requires at least MIN_FEATURES_FOR_PREDICTION (2) known features.
    - Only call this when the Controller decides prediction is warranted.
    - The returned result is the SINGLE SOURCE OF TRUTH for this prediction.

    Args:
        state_json: Current AgentState serialized as JSON string

    Returns:
        JSON prediction result with prediction class, probability, risk level,
        features used, missing features, and confidence note.
        Returns error JSON if fewer than 2 features are known.
    """
    try:
        from backend.app.agent.state import AgentState
        from backend.app.agent.actions import AgentTools

        state = AgentState.from_json(json.loads(state_json) if state_json else {})

        # HARD GATE: Require minimum 2 known features before running ML
        known_count = state.known_feature_count()
        if known_count < MIN_FEATURES_FOR_PREDICTION:
            msg = (
                f"Insufficient features for prediction: {known_count} known, "
                f"need at least {MIN_FEATURES_FOR_PREDICTION}. "
                "Ask the user for more information before predicting."
            )
            logger.warning(f"[TOOL] run_ml_prediction BLOCKED: {msg}")
            return json.dumps({
                "error": msg,
                "prediction": None,
                "known_count": known_count,
                "required_count": MIN_FEATURES_FOR_PREDICTION,
            })

        predictor = _get_predictor()
        if not predictor.is_loaded:
            return json.dumps({"error": "ML model not available", "prediction": None})

        tools = AgentTools(predictor)
        result = tools.predict_maintenance(state)

        if result.get("error"):
            logger.warning(f"[TOOL] run_ml_prediction error: {result['error']}")
            return json.dumps(result)

        # Include updated state_json so caller can persist the prediction
        result["state_json"] = json.dumps(state.to_json())
        logger.info(
            f"[TOOL] run_ml_prediction -> {result.get('prediction')} "
            f"(risk={result.get('risk_level')}, score={result.get('model_score')}, "
            f"features_used={known_count}/{len(SELECTED_FEATURES)})"
        )
        return json.dumps(result)

    except Exception as e:
        logger.error(f"run_ml_prediction error: {e}")
        return json.dumps({"error": str(e), "prediction": None})


# ---------------------------------------------------------------------------
# Domain Check Tool
# ---------------------------------------------------------------------------

def check_domain(user_message: str) -> str:
    """
    Checks whether the user's message is within the vehicle maintenance domain.
    
    Returns a redirect message if the query is off-topic (jokes, weather, etc.)
    The agent should return this redirect message directly without further processing.
    
    Args:
        user_message: The raw user message to check
    
    Returns:
        JSON with {"in_domain": bool, "redirect_message": str or null}
    """
    is_in_domain, redirect_msg = DomainGuard.is_in_domain(user_message)
    result = {
        "in_domain": is_in_domain,
        "redirect_message": redirect_msg if not is_in_domain else None,
    }
    if not is_in_domain:
        logger.info(f"[TOOL] check_domain: OFF-TOPIC query intercepted")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Context Compaction Tool
# ---------------------------------------------------------------------------

async def compact_conversation_context(
    messages_json: str,
    current_summary: str,
    state_json: str,
) -> str:
    """
    Compacts old conversation messages into a concise summary to manage context window.
    
    Call this when the conversation history exceeds the compaction threshold.
    The compacted summary replaces old raw messages, keeping only recent exchanges.
    
    IMPORTANT: The compactor preserves known features, unavailable features,
    predictions, and key user statements. It never invents new information.
    
    Args:
        messages_json: JSON array of old message dicts [{"role": ..., "content": ...}]
        current_summary: Existing summary string (or empty string)
        state_json: Current AgentState JSON for structured context
    
    Returns:
        JSON with {"summary": str, "success": bool}
    """
    try:
        from backend.app.context.compactor import ContextCompactor
        from backend.app.agent.state import AgentState

        messages = json.loads(messages_json) if messages_json else []
        state = AgentState.from_json(json.loads(state_json) if state_json else {})
        feature_snapshot = state.to_context_dict()

        compactor = ContextCompactor()
        summary = await compactor.compact_messages(
            messages=messages,
            existing_summary=current_summary or None,
            feature_state_snapshot=feature_snapshot,
        )

        logger.info(f"[TOOL] compact_conversation_context: compressed {len(messages)} messages into summary")
        return json.dumps({"summary": summary, "success": True})

    except Exception as e:
        logger.error(f"compact_conversation_context error: {e}")
        return json.dumps({"summary": current_summary, "success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# State Summary Tool (used by PredictionAgent and ResponseAgent)
# ---------------------------------------------------------------------------

def get_current_state_summary(state_json: str) -> str:
    """
    Returns a structured summary of the current AgentState.

    This tool allows specialist agents (PredictionAgent, ResponseAgent) to
    inspect the current feature state, known counts, and readiness for prediction
    WITHOUT having to parse raw state JSON themselves.

    IMPORTANT: PredictionAgent MUST call this before calling run_ml_prediction
    to verify that known_count >= 2.

    Args:
        state_json: Current AgentState serialized as JSON string

    Returns:
        JSON with:
        - known_features: dict of known feature name -> value
        - declined_features: list of declined feature names
        - unknown_features: list of still-unknown features
        - known_count: int count of known features
        - ready_for_prediction: bool (True if known_count >= MIN_FEATURES_FOR_PREDICTION)
        - min_features_required: int (always 2)
        - next_feature: name of next feature to ask or null
        - latest_prediction: most recent prediction dict or null
    """
    try:
        from backend.app.agent.state import AgentState

        state = AgentState.from_json(json.loads(state_json) if state_json else {})
        known = state.get_known_features()
        next_feat = state.get_next_question_feature()

        summary = {
            "known_features": known,
            "declined_features": list(state.declined_features),
            "unknown_features": state.get_unknown_features(),
            "known_count": len(known),
            "ready_for_prediction": state.has_enough_for_prediction(),
            "min_features_required": MIN_FEATURES_FOR_PREDICTION,
            "next_feature": next_feat,
            "next_feature_display": FEATURE_DISPLAY_NAMES.get(next_feat, next_feat) if next_feat else None,
            "latest_prediction": state.get_latest_prediction(),
            "turn_count": state.turn_count,
        }

        logger.debug(
            f"[TOOL] get_current_state_summary: known={len(known)}, "
            f"ready_for_prediction={summary['ready_for_prediction']}"
        )
        return json.dumps(summary)

    except Exception as e:
        logger.error(f"get_current_state_summary error: {e}")
        return json.dumps({
            "known_count": 0,
            "ready_for_prediction": False,
            "min_features_required": MIN_FEATURES_FOR_PREDICTION,
            "error": str(e),
        })


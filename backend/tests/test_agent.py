"""
Automated tests for Agent State, adaptive questioning, and 'I don't know' memory.
"""


from backend.app.agent.state import AgentState
from backend.app.agent.actions import AgentTools
from backend.app.services.prediction_service import get_predictor


def test_agent_state_initialization():
    """Verify fresh agent state setup."""
    from backend.app.ml.preprocessing import SELECTED_FEATURES
    state = AgentState(conversation_id="test-123")
    assert state.get_known_features() == {}
    assert len(state.get_unknown_features()) == len(SELECTED_FEATURES)
    assert len(state.declined_features) == 0



def test_dont_know_memory_and_no_reasking():
    """
    CRITICAL REQUIREMENT TEST:
    When user says "I don't know my brake condition",
    Brake_Condition must be marked 'unavailable' and NEVER asked again.
    """
    state = AgentState(conversation_id="test-456")
    tools = AgentTools(get_predictor())

    # User declines brake condition
    res = tools.mark_feature_declined(state, "Brake_Condition")
    assert res["success"] is True
    assert "Brake_Condition" in state.declined_features
    assert state.feature_state["Brake_Condition"]["status"] == "unavailable"

    # Verify Brake_Condition is NOT in askable features
    askable = state.get_askable_features()
    assert "Brake_Condition" not in askable

    # Verify next question skips Brake_Condition
    next_q = tools.get_next_question(state)
    assert next_q["feature"] != "Brake_Condition"


def test_incremental_prediction_updates():
    """
    Verify incremental predictions update state dynamically.
    Prediction requires at least 2 known features (MIN_FEATURES_FOR_PREDICTION=2).
    """
    from backend.app.agent.state import MIN_FEATURES_FOR_PREDICTION
    state = AgentState(conversation_id="test-789")
    tools = AgentTools(get_predictor())

    # Add first feature — no prediction yet (only 1 feature)
    tools.update_feature(state, "Vehicle_Age", 5)
    assert state.known_feature_count() == 1
    assert not state.has_enough_for_prediction()

    # Add second feature — now prediction is warranted (2 features)
    tools.update_feature(state, "Reported_Issues", 3)
    assert state.known_feature_count() == 2
    assert state.has_enough_for_prediction()
    pred1 = tools.predict_maintenance(state)
    assert pred1.get("model_score") is not None
    assert pred1.get("prediction") in ["maintenance_required", "no_maintenance_needed"]

    # Add third feature — prediction recalculates with more info
    tools.update_feature(state, "Brake_Condition", "Worn Out")
    pred2 = tools.predict_maintenance(state)
    assert pred2.get("model_score") is not None
    assert len(pred2.get("features_used", [])) >= len(pred1.get("features_used", []))
    # Prediction IDs must differ each call
    assert pred2["prediction_id"] != pred1["prediction_id"]

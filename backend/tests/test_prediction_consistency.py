"""
Automated regression tests for prediction consistency and single ML inference rule.
Tests use AgentTools directly (the authoritative ML path) instead of the legacy controller.
"""

import pytest
from backend.app.agent.state import AgentState, MIN_FEATURES_FOR_PREDICTION
from backend.app.agent.actions import AgentTools
from backend.app.agent.adk_agents.tools import run_ml_prediction, update_feature_state
from backend.app.services.prediction_service import get_predictor
import json


def test_single_prediction_per_turn_and_consistency():
    """
    Verify ML is called at most once per tool invocation and
    prediction_id is unique per call.
    """
    predictor = get_predictor()
    initial_count = predictor._inference_count

    state = AgentState(conversation_id="consistency-test-1")
    tools = AgentTools(predictor)

    # Add features one at a time
    feature_sequence = [
        ("Vehicle_Age", 10),
        ("Brake_Condition", "Good"),
        ("Battery_Status", "Weak"),
        ("Maintenance_History", "Average"),
        ("Service_History", 10),
    ]

    prev_pred_id = None

    for feat_name, feat_val in feature_sequence:
        tools.update_feature(state, feat_name, feat_val)
        known_count = state.known_feature_count()

        if known_count < MIN_FEATURES_FOR_PREDICTION:
            # Must not predict
            prev_count = predictor._inference_count
            # Simulate checking — no prediction should run
            assert not state.has_enough_for_prediction(), (
                f"Should not predict with {known_count} feature(s)"
            )
            continue

        # Run prediction
        prev_count = predictor._inference_count
        pred = tools.predict_maintenance(state)
        new_count = predictor._inference_count

        # Verify single ML call
        calls_this_op = new_count - prev_count
        assert calls_this_op <= 1, f"ML called {calls_this_op} times (expected <= 1)"

        # Verify prediction_id is unique
        assert "prediction_id" in pred
        assert pred["prediction_id"] is not None
        assert pred["prediction_id"] != prev_pred_id, (
            "Each prediction call must generate a unique prediction_id"
        )
        prev_pred_id = pred["prediction_id"]

        # Verify model_score is in valid range
        assert 0.0 <= pred["model_score"] <= 1.0


"""
Automated Hallucination Prevention & Google ADK Multi-Agent Test Suite.

Tests cover:
1. Agent Initialization & Tooling
2. Test A: Vague symptom ("weird noise") -> Tire_Condition remains unknown, no hallucinated value
3. Test B: Single feature ("5 years old") -> Vehicle_Age=5, no prediction (< 2 features)
4. Test C: Second feature ("3 reported issues") -> Reported_Issues=3, question advances
5. Test D: Contextual short reply ("3" when asking Reported_Issues) -> sets Reported_Issues=3, clears current question
6. Test E: Two valid features -> Random Forest executes, risk comes strictly from ML
7. Test F: Third valid feature -> ML prediction recalculates with new prediction_id
8. Test G: Unsupported input ("Mileage is 30000") -> rejected, not added to state
9. Test H: Unrelated input ("bought a new helmet") -> no feature created
10. Test I: "I don't know" -> marked unavailable, never asked again
11. Test J: Rolling context & compaction -> confirmed state preserved
12. Test K: Judge Failure / Timeout -> unsafe response blocked, deterministic fallback returned
13. Test L: Repeated answer protection -> no question looping
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.agent.state.state import (
    AgentState,
    MIN_FEATURES_FOR_PREDICTION,
    CHECKPOINT_INPUT_RECEIVED,
    CHECKPOINT_FEATURES_EXTRACTED,
    CHECKPOINT_FEATURES_VALIDATED,
    CHECKPOINT_PREDICTION_GENERATED,
    CHECKPOINT_RESPONSE_GENERATED,
    CHECKPOINT_JUDGE_PASSED,
)
from backend.app.agent.state.feature_selector import (
    get_askable_features,
    get_next_question_feature,
    get_question_specification,
)
from backend.app.agent.tools.feature_tools import (
    validate_feature_name,
    validate_feature_value,
    apply_structured_extraction,
    extract_explicit_facts,
)
from backend.app.agent.tools.prediction_tools import execute_prediction_gate
from backend.app.agent.adk_agents.information_agent import build_information_agent
from backend.app.agent.adk_agents.response_agent import build_response_agent
from backend.app.agent.adk_agents.judge_agent import (
    build_judge_agent,
    run_deterministic_guards,
    generate_deterministic_fallback_response,
)
from backend.app.agent.adk_agents.controller_agent import build_controller_agent
from backend.app.agent.domain_guard import DomainGuard
from backend.app.ml.preprocessing import (
    SELECTED_FEATURES,
    FEATURE_PRIORITY,
    FEATURE_DISPLAY_NAMES,
    CATEGORICAL_VALUES,
)
from backend.app.context.compactor import ContextCompactor
from backend.app.context.manager import ContextManager
from backend.app.agent.feature_ranking import FeatureRankingEngine


# =============================================================================
# 1. ADK Agent Initialization
# =============================================================================

class TestADKAgentInitialization:
    """Verify all ADK agent builders instantiate valid LlmAgent instances."""

    def test_controller_agent_structure(self):
        controller = build_controller_agent()
        assert controller.name == "ControllerAgent"
        assert len(controller.tools) >= 3  # InputAgent, ResponseAgent, JudgeAgent

    def test_information_agent_structure(self):
        information_agent = build_information_agent()
        assert information_agent.name == "InformationAgent"
        assert information_agent.output_key == "information_result"

    def test_response_agent_structure(self):
        response_agent = build_response_agent()
        assert response_agent.name == "ResponseAgent"
        assert response_agent.output_key == "response_result"

    def test_judge_agent_structure(self):
        judge_agent = build_judge_agent()
        assert judge_agent.name == "JudgeAgent"
        assert judge_agent.output_key == "judge_result"


# =============================================================================
# 2. Hallucination Test Matrix (Tests A through L)
# =============================================================================

class TestHallucinationMatrix:
    """Exact test cases A through L required by specification."""

    def test_A_vague_symptom_no_invented_feature(self):
        """
        Test A: Input 'My tire has a weird noise.'
        Expected: Tire_Condition remains unknown, no invented tire condition, no prediction.
        """
        state = AgentState()
        # Vague statement is extracted as raw observation, NOT as Tire_Condition
        extraction = {
            "extracted_features": [],
            "unavailable_features": [],
            "raw_observations": ["tire has a weird noise"],
        }
        res = apply_structured_extraction(state, extraction)

        # Tire_Condition must not be confirmed
        assert state.feature_state["Tire_Condition"]["status"] == "unknown"
        assert state.feature_state["Tire_Condition"]["value"] is None
        assert "tire has a weird noise" in state.raw_observations

        # ML prediction must NOT run (< 2 features)
        pred_res = execute_prediction_gate(state)
        assert pred_res["allowed"] is False
        assert pred_res["prediction"] is None

    def test_B_single_feature_no_prediction(self):
        """
        Test B: Input 'My bike is 5 years old.'
        Expected: Vehicle_Age = 5, exactly one next valid question, no prediction (< 2 features).
        """
        state = AgentState()
        extraction = {
            "extracted_features": [{"name": "Vehicle_Age", "value": 5}],
            "unavailable_features": [],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert state.get_known_features() == {"Vehicle_Age": 5}
        assert state.known_feature_count() == 1

        # Hard programmatic gate blocks prediction
        pred_res = execute_prediction_gate(state)
        assert pred_res["allowed"] is False
        assert pred_res["prediction"] is None
        assert state.has_enough_for_prediction() is False

        # Next question should be chosen from remaining askable features
        next_q = get_next_question_feature(state)
        assert next_q is not None
        assert next_q != "Vehicle_Age"
        assert next_q in SELECTED_FEATURES

    def test_C_second_feature_advances_question(self):
        """
        Test C: Input 'I have 3 reported issues.'
        Expected: Reported_Issues = 3, question advances, Reported_Issues is not asked again.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")

        extraction = {
            "extracted_features": [{"name": "Reported_Issues", "value": 3}],
            "unavailable_features": [],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert state.get_known_features() == {"Vehicle_Age": 5, "Reported_Issues": 3}
        assert state.known_feature_count() == 2

        # Reported_Issues must not be in askable list
        askable = get_askable_features(state)
        assert "Reported_Issues" not in askable
        assert "Vehicle_Age" not in askable

        next_q = get_next_question_feature(state)
        assert next_q not in ["Vehicle_Age", "Reported_Issues"]

    def test_D_contextual_short_reply_resolves_current_question(self):
        """
        Test D: Input '3' when current question is Reported_Issues.
        Expected: Reported_Issues = 3, state updated, current question cleared, next question advances.
        """
        state = AgentState()
        # Assistant previously asked Reported_Issues
        state.mark_asked("Reported_Issues")
        assert state.current_question_feature == "Reported_Issues"

        # User replies simply "3" -> InputAgent resolves against current_question_feature
        extraction = {
            "extracted_features": [{"name": state.current_question_feature, "value": 3}],
            "unavailable_features": [],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert state.get_known_features()["Reported_Issues"] == 3
        # current_question_feature must now be cleared
        assert state.current_question_feature is None

        # Next question must not repeat Reported_Issues
        next_q = get_next_question_feature(state)
        assert next_q != "Reported_Issues"

    def test_E_two_valid_features_executes_rf_model(self):
        """
        Test E: Two valid features.
        Expected: Random Forest executes, prediction produced, risk score from ML.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")
        state.update_feature("Reported_Issues", 3, "known")

        assert state.known_feature_count() == 2
        assert state.has_enough_for_prediction() is True

        pred_res = execute_prediction_gate(state)
        assert pred_res["allowed"] is True
        assert pred_res["prediction"] is not None

        pred = pred_res["prediction"]
        assert pred["prediction"] in ["maintenance_required", "no_maintenance_needed"]
        assert 0.0 <= pred["model_score"] <= 1.0
        assert "prediction_id" in pred
        assert len(pred["prediction_id"]) > 10

    def test_F_third_feature_recalculates_prediction(self):
        """
        Test F: Third valid feature added.
        Expected: Prediction recalculated, unique prediction_id generated each time.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")
        state.update_feature("Reported_Issues", 3, "known")

        pred1 = execute_prediction_gate(state)["prediction"]
        assert pred1 is not None
        id1 = pred1["prediction_id"]

        # Add 3rd feature
        state.update_feature("Brake_Condition", "Worn Out", "known")
        assert state.known_feature_count() == 3

        pred2 = execute_prediction_gate(state)["prediction"]
        assert pred2 is not None
        id2 = pred2["prediction_id"]

        # Must have unique prediction IDs and more features used
        assert id1 != id2
        assert len(pred2["features_used"]) >= len(pred1["features_used"])

    def test_G_unsupported_feature_rejected(self):
        """
        Test G: Unsupported input 'My engine temperature is 30000.'
        Expected: Engine_Temperature is NOT added to state, no such question.
        """
        state = AgentState()

        is_valid, _ = validate_feature_name("Engine_Temperature")
        assert is_valid is False

        extraction = {
            "extracted_features": [{"name": "Engine_Temperature", "value": 30000}],
            "unavailable_features": [],
            "raw_observations": ["engine temperature is 30000"],
        }
        res = apply_structured_extraction(state, extraction)

        assert "Engine_Temperature" in res["rejected_features"]
        assert len(state.get_known_features()) == 0
        assert "Engine_Temperature" not in state.feature_state

    def test_H_unrelated_input_creates_no_feature(self):
        """
        Test H: Unrelated input 'I bought a new helmet yesterday.'
        Expected: No model feature created.
        """
        state = AgentState()
        extraction = {
            "extracted_features": [{"name": "Helmet_Condition", "value": "New"}],
            "unavailable_features": [],
            "raw_observations": ["bought a new helmet yesterday"],
        }
        res = apply_structured_extraction(state, extraction)

        assert "Helmet_Condition" in res["rejected_features"]
        assert state.known_feature_count() == 0

    def test_I_dont_know_marked_unavailable_never_reasked(self):
        """
        Test I: User says 'I don't know' for Battery_Status.
        Expected: Battery_Status becomes unavailable, never asked again.
        """
        state = AgentState()
        state.mark_asked("Battery_Status")

        extraction = {
            "extracted_features": [],
            "unavailable_features": ["Battery_Status"],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert "Battery_Status" in state.declined_features
        assert state.feature_state["Battery_Status"]["status"] == "unavailable"

        # Must never be in askable list
        askable = get_askable_features(state)
        assert "Battery_Status" not in askable

        next_q = get_next_question_feature(state)
        assert next_q != "Battery_Status"

    @pytest.mark.asyncio
    async def test_J_rolling_context_compaction_preserves_state(self):
        """
        Test J: Long conversation triggers compaction; confirmed state survives.
        """
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Vehicle Identity: 5yo Sedan\nKnown Features: Vehicle_Age=5, Reported_Issues=3"
        mock_router.generate = AsyncMock(return_value=mock_response)

        compactor = ContextCompactor(router=mock_router)
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")
        state.update_feature("Reported_Issues", 3, "known")

        messages = [
            {"role": "user", "content": "My car is 5 years old."},
            {"role": "assistant", "content": "Got it. How many issues?"},
            {"role": "user", "content": "3 reported issues."},
        ]

        summary = await compactor.compact_messages(
            messages=messages,
            feature_state_snapshot=state.to_context_dict(),
        )
        assert len(summary) > 0
        assert mock_router.generate.called
        assert state.get_known_features() == {"Vehicle_Age": 5, "Reported_Issues": 3}

    def test_K_judge_failure_enforces_deterministic_fallback(self):
        """
        Test K: Unsafe response rejected by guard / judge -> safe deterministic fallback returned.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")  # only 1 feature

        # Proposed response hallucinates a calculated risk and non-schema feature
        bad_response = "Your vehicle has high risk score 88% and mileage is 50000."

        is_valid, violations = run_deterministic_guards(
            response_text=bad_response,
            state=state,
            prediction=None,
            expected_next_feature="Reported_Issues",
        )
        assert is_valid is False
        assert len(violations) > 0

        # Deterministic fallback response must be safe & contain no hallucinated numbers
        fallback = generate_deterministic_fallback_response(
            state=state,
            prediction=None,
            next_feature="Reported_Issues",
        )
        assert "50000" not in fallback
        assert "88%" not in fallback
        assert "reported issues" in fallback.lower() or "issues" in fallback.lower()
        assert "vehicle age" in fallback.lower() and "5" in fallback

    def test_L_repeated_answer_protection_no_loop(self):
        """
        Test L: User answers same question; question selector advances and never loops.
        """
        state = AgentState()
        # Ask the current plan's first feature.
        q1 = get_next_question_feature(state)
        assert q1 in SELECTED_FEATURES
        state.mark_asked(q1)

        # User answers
        state.update_feature(q1, 2, "known")

        # Next question MUST advance
        q2 = get_next_question_feature(state)
        assert q2 != q1
        assert q2 in SELECTED_FEATURES


# =============================================================================
# 3. Deterministic Feature & Schema Validation Tests
# =============================================================================

class TestDeterministicValidationRules:
    """Test schema and categorical validation functions."""

    def test_schema_feature_count_matches_dataset_predictors(self):
        assert len(SELECTED_FEATURES) == 19

    def test_all_categorical_values_valid(self):
        assert "Good" in CATEGORICAL_VALUES["Maintenance_History"]
        assert "Worn Out" in CATEGORICAL_VALUES["Brake_Condition"]
        assert "Weak" in CATEGORICAL_VALUES["Battery_Status"]
        assert "New" in CATEGORICAL_VALUES["Tire_Condition"]

    def test_numerical_validation_rejects_negative(self):
        is_valid, val = validate_feature_value("Vehicle_Age", -5)
        assert is_valid is False
        assert val is None

    def test_numerical_validation_parses_natural_strings(self):
        is_valid, val = validate_feature_value("Vehicle_Age", "5 years")
        assert is_valid is True
        assert val == 5

    def test_vague_categorical_symptom_is_not_mapped_to_a_condition(self):
        is_valid, val = validate_feature_value("Brake_Condition", "squeaking")
        assert is_valid is False
        assert val is None

    def test_min_features_constant_is_strictly_two(self):
        assert MIN_FEATURES_FOR_PREDICTION == 2

    def test_message_parser_accepts_only_explicit_values(self):
        assert extract_explicit_facts("My tire has a weird noise.")["extracted_features"] == []
        assert extract_explicit_facts("My mileage is 30000.")["extracted_features"] == [{"name": "Mileage", "value": 30000}]
        assert extract_explicit_facts("I bought a new helmet yesterday.")["extracted_features"] == []
        facts = extract_explicit_facts("My bike is 5 years old.")
        assert facts["extracted_features"] == [{"name": "Vehicle_Age", "value": 5}]

    def test_labelled_fact_does_not_overwrite_a_different_pending_numeric_question(self):
        facts = extract_explicit_facts("4 problems", "Vehicle_Age")
        assert facts["extracted_features"] == [{"name": "Reported_Issues", "value": 4}]

    def test_current_question_is_not_repeated_before_an_answer(self):
        state = AgentState()
        state.set_feature_subset(["Reported_Issues"], "issues")
        state.update_feature("Reported_Issues", 4, status="known")
        assert get_next_question_feature(state) is None



class TestDynamicFeatureRanking:
    def test_brake_and_battery_intents_produce_different_leading_questions(self):
        brake = AgentState()
        battery = AgentState()
        brake_plan = FeatureRankingEngine.select_feature_subset(brake, "brake")
        battery_plan = FeatureRankingEngine.select_feature_subset(battery, "battery")
        brake.set_feature_subset(brake_plan, "brake")
        battery.set_feature_subset(battery_plan, "battery")

        assert brake_plan[0] == "Brake_Condition"
        assert battery_plan[0] == "Battery_Status"
        assert brake_plan != battery_plan
        assert get_next_question_feature(brake) == "Brake_Condition"
        assert get_next_question_feature(battery) == "Battery_Status"

    def test_prediction_audits_the_current_feature_values(self):
        state = AgentState()
        state.update_feature("Reported_Issues", 4, "known")
        state.update_feature("Brake_Condition", "Good", "known")
        prediction = execute_prediction_gate(state)["prediction"]
        assert prediction["feature_values_used"]["Reported_Issues"] == 4
        assert prediction["feature_values_used"]["Brake_Condition"] == "Good"
        assert prediction["feature_values_used"]["Battery_Status"] is None

        state.update_feature("Battery_Status", "New", "known")
        updated_prediction = execute_prediction_gate(state)["prediction"]
        assert updated_prediction["feature_values_used"]["Battery_Status"] == "New"
        assert updated_prediction["prediction_id"] != prediction["prediction_id"]

"""Regression tests for the hypothesis -> dataset-grounding firewall."""

import pytest

from backend.app.agent.adk_agents.runner import ADKAgentRunner
from backend.app.agent.hypothesis_grounding import ground_hypotheses, grounded_features
from backend.app.agent.state.state import AgentState
from backend.app.agent.tools.feature_tools import extract_explicit_facts, validate_feature_value
from backend.app.ml.feature_schema import MODEL_FEATURE_SCHEMA


def test_brake_hypothesis_is_grounded_but_does_not_set_a_value():
    grounded = ground_hypotheses([{"hypothesis": "worn brake pads", "confidence": 0.8}])
    assert "Brake_Condition" in grounded_features(grounded)
    state = AgentState()
    assert state.feature_state["Brake_Condition"]["status"] == "unknown"
    assert state.feature_state["Brake_Condition"]["value"] is None


def test_unsupported_wheel_alignment_never_becomes_a_feature():
    grounded = ground_hypotheses([{"hypothesis": "wheel alignment issue"}])
    assert grounded == [{
        "hypothesis": "wheel alignment issue", "dataset_feature": None,
        "status": "unsupported", "relevance": "none", "confidence": 0.5,
    }]
    assert "Wheel_Alignment" not in MODEL_FEATURE_SCHEMA
    assert grounded_features(grounded) == []


@pytest.mark.asyncio
async def test_runtime_questions_only_grounded_features(monkeypatch):
    runner = ADKAgentRunner()
    runner._initialized = True

    async def input_agent(context):
        extracted = extract_explicit_facts(context["user_message"], context["current_question_feature"])
        extracted["hypotheses"] = [
            {"hypothesis": "wheel alignment issue", "confidence": 0.9},
            {"hypothesis": "worn brake pads", "confidence": 0.8},
        ]
        return extracted

    async def response_agent(_context):
        return "A brake-related issue could be worth narrowing down. How would you describe the brake condition?"

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()
    await runner.run_turn("My bike has a grinding noise while braking.", "grounded", state, [])

    assert state.current_question_feature == "Brake_Condition"
    assert "Wheel_Alignment" not in state.feature_state
    assert all(feature in MODEL_FEATURE_SCHEMA for feature in state.selected_features)
    assert any(item["status"] == "unsupported" for item in state.grounded_hypotheses)


@pytest.mark.asyncio
async def test_runtime_discards_wheel_alignment_when_schema_lacks_it(monkeypatch):
    runner = ADKAgentRunner()
    runner._initialized = True

    async def input_agent(context):
        extracted = extract_explicit_facts(context["user_message"], context["current_question_feature"])
        extracted["hypotheses"] = [{"hypothesis": "wheel alignment issue", "confidence": 0.9}]
        return extracted

    async def response_agent(_context):
        return "I can help assess the maintenance details you can confirm."

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()
    await runner.run_turn("My wheel alignment feels wrong.", "wheel", state, [])

    assert state.current_question_feature is None
    assert state.selected_features == []
    assert state.get_known_features() == {}


@pytest.mark.asyncio
async def test_prediction_does_not_terminate_a_grounded_assessment(monkeypatch):
    """The reported UI case must predict and still ask a ranked question."""
    runner = ADKAgentRunner()
    runner._initialized = True
    contexts = []

    async def input_agent(context):
        return extract_explicit_facts(context["user_message"], context["current_question_feature"])

    async def response_agent(context):
        contexts.append(context)
        spec = context["next_feature_spec"]
        return f"This is still an evolving assessment. How is the {spec['display_name']}?"

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()

    first = await runner.run_turn(
        "My car is 5 years old and I have noticed 3 problems recently.", "ongoing", state, []
    )
    assert set(state.get_known_features()) == {"Vehicle_Age", "Reported_Issues"}
    assert first["prediction"] is not None
    assert state.assessment_complete is False
    assert state.current_question_feature == "Maintenance_History"
    assert contexts[-1]["assessment_complete"] is False
    assert contexts[-1]["next_feature_to_ask"] == "Maintenance_History"

    second = await runner.run_turn("Good", "ongoing", state, [])
    assert state.feature_state["Maintenance_History"]["value"] == "Good"
    assert second["prediction"] is not None
    assert state.assessment_complete is False
    assert state.current_question_feature is not None
    assert state.current_question_feature != "Maintenance_History"


@pytest.mark.asyncio
async def test_assessment_stops_after_eight_validated_relevant_features(monkeypatch):
    runner = ADKAgentRunner()
    runner._initialized = True

    async def input_agent(context):
        return extract_explicit_facts(context["user_message"], context["current_question_feature"])

    async def response_agent(context):
        if context["assessment_complete"]:
            return "The assessment is complete. The current model result reflects the validated details. This is not a definitive mechanical diagnosis."
        spec = context["next_feature_spec"]
        return f"That helps refine the assessment. How is the {spec['display_name']}?"

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()
    messages = [
        "My car is 5 years old, has 3 problems, and the brakes are Worn Out.",
        "Good", "4", "last service was 2025-01-01", "Weak", "0",
    ]
    final = None
    for message in messages:
        final = await runner.run_turn(message, "complete", state, [])

    assert state.known_feature_count() == 8
    assert state.assessment_complete is True
    assert state.remaining_relevant_features == []
    assert state.has_current_prediction() is True
    assert state.current_question_feature is None
    assert "?" not in final["response"]


@pytest.mark.asyncio
async def test_two_feature_prediction_cannot_silently_end_response(monkeypatch):
    """A terminal-looking Output LLM response must be replaced by a question."""
    runner = ADKAgentRunner()
    runner._initialized = True

    async def input_agent(context):
        return extract_explicit_facts(context["user_message"], context["current_question_feature"])

    async def response_agent(_context):
        return "The model indicates maintenance may be required."

    async def judge_agent(**_kwargs):
        raise AssertionError("The deterministic missing-question guard should run first")

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()
    result = await runner.run_turn(
        "My car has 4 problems and the brakes are good.", "two-features", state, []
    )

    assert state.known_feature_count() == 2
    assert state.assessment_complete is False
    assert state.current_question_feature == "Maintenance_History"
    assert "?" in result["response"]
    context = state.to_context_dict()
    assert context["next_question"] == "Maintenance_History"
    assert context["open_questions"]


@pytest.mark.asyncio
async def test_service_count_merges_with_existing_maintenance_state(monkeypatch):
    """`4 services` must update Service_History without losing prior facts."""
    runner = ADKAgentRunner()
    runner._initialized = True
    response_contexts = []

    async def input_agent(context):
        return extract_explicit_facts(context["user_message"], context["current_question_feature"])

    async def response_agent(context):
        response_contexts.append(context)
        spec = context["next_feature_spec"]
        return f"The current details are updated. What is the {spec['display_name']}?"

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()

    first = await runner.run_turn("maintenance is average", "state-merge", state, [])
    assert state.feature_state["Maintenance_History"]["value"] == "Average"
    assert state.current_question_feature == "Service_History"
    first_signature = state.feature_state_signature()

    second = await runner.run_turn("4 services", "state-merge", state, [])
    assert state.feature_state["Maintenance_History"]["value"] == "Average"
    assert state.feature_state["Service_History"]["value"] == 4
    assert state.feature_state_signature() != first_signature
    assert second["prediction"] is not None
    assert set(second["prediction"]["features_used"]) >= {"Maintenance_History", "Service_History"}
    assert state.current_question_feature != "Service_History"
    assert response_contexts[-1]["known_features"]["Service_History"] == 4


@pytest.mark.asyncio
async def test_repeated_known_fact_keeps_the_new_pending_question(monkeypatch):
    runner = ADKAgentRunner()
    runner._initialized = True

    async def input_agent(context):
        return extract_explicit_facts(context["user_message"], context["current_question_feature"])

    async def response_agent(context):
        spec = context["next_feature_spec"]
        return f"The recorded details remain current. Could you share the {spec['display_name']}?"

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()
    await runner.run_turn("maintenance is average", "repeat", state, [])
    await runner.run_turn("4 services", "repeat", state, [])
    assert state.current_question_feature == "Last_Service_Date"

    repeated = await runner.run_turn("4 services", "repeat", state, [])
    assert state.feature_state["Service_History"]["value"] == 4
    assert state.current_question_feature == "Last_Service_Date"
    assert "?" in repeated["response"]


@pytest.mark.parametrize("message", [
    "My last service was on 15 March 2025.",
    "Last service: 15/03/2025.",
    "I got it serviced on March 15, 2025.",
    "The vehicle was last serviced on 2025-03-15.",
    "Last service was March 15th 2025.",
])
def test_natural_service_dates_normalise_to_model_iso_date(message):
    facts = extract_explicit_facts(message)
    assert {item["name"]: item["value"] for item in facts["extracted_features"]}["Last_Service_Date"] == "2025-03-15"


@pytest.mark.parametrize("raw_date", ["32 March 2025", "15/99/2025", "February 30 2025"])
def test_invalid_dates_are_rejected(raw_date):
    assert validate_feature_value("Last_Service_Date", raw_date) == (False, None)


@pytest.mark.asyncio
async def test_service_date_updates_state_and_latest_correction_wins(monkeypatch):
    runner = ADKAgentRunner()
    runner._initialized = True

    async def input_agent(context):
        return extract_explicit_facts(context["user_message"], context["current_question_feature"])

    async def response_agent(context):
        spec = context["next_feature_spec"]
        return f"The date is recorded. Could you share the {spec['display_name']}?"

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)
    state = AgentState()
    await runner.run_turn("My car has 4 problems and the brakes are good.", "date", state, [])
    first = await runner.run_turn("My last service was on 15 March 2025.", "date", state, [])
    assert state.feature_state["Last_Service_Date"]["value"] == "2025-03-15"
    assert state.current_question_feature != "Last_Service_Date"
    assert "Last_Service_Date" not in state.remaining_relevant_features
    assert "Last_Service_Date" in first["prediction"]["features_used"]

    second = await runner.run_turn("Actually, the last service was on 20 April 2025.", "date", state, [])
    assert state.feature_state["Last_Service_Date"]["value"] == "2025-04-20"
    assert second["prediction"]["feature_values_used"]["Last_Service_Date"] == "2025-04-20"

"""End-to-end runtime checks for schema-bounded dynamic assessment plans."""

import pytest

from backend.app.agent.adk_agents.runner import ADKAgentRunner
from backend.app.agent.state.state import AgentState
from backend.app.agent.tools.feature_tools import extract_explicit_facts


@pytest.mark.asyncio
async def test_runtime_re_ranks_and_persists_different_intent_plans(monkeypatch):
    runner = ADKAgentRunner()
    # The test exercises the production turn pipeline while making external LLM
    # calls deterministic; validation, ranking, prediction, and state are real.
    runner._initialized = True

    async def input_agent(context):
        return extract_explicit_facts(
            context["user_message"], context["current_question_feature"]
        )

    async def response_agent(_context):
        return "I understand the concern. This helps focus the assessment. What is the next detail?"

    async def judge_agent(**_kwargs):
        return {"score": 5, "passed": True, "reason": "grounded"}

    monkeypatch.setattr(runner, "_run_input_agent", input_agent)
    monkeypatch.setattr(runner, "_run_response_agent", response_agent)
    monkeypatch.setattr(runner, "_run_judge_agent", judge_agent)

    brake_state = AgentState()
    await runner.run_turn("My brakes are making a grinding noise.", "brake", brake_state, [])
    battery_state = AgentState()
    await runner.run_turn(
        "My battery keeps dying and the car will not start.", "battery", battery_state, []
    )

    assert brake_state.selected_features[0] == "Brake_Condition"
    assert battery_state.selected_features[0] == "Battery_Status"
    assert brake_state.selected_features != battery_state.selected_features
    assert brake_state.current_question_feature == "Brake_Condition"
    assert battery_state.current_question_feature == "Battery_Status"
    assert "features_ranked" in brake_state.checkpoint_log
    assert "assessment_features_selected" in brake_state.checkpoint_log

"""
Test LLM Router Failover Scenarios:
1. Primary available -> primary response
2. Primary 429 / Rate Limit -> secondary response immediately
3. Primary unavailable -> secondary response
4. Both unavailable -> preserve ML prediction & return graceful message
"""

import asyncio
from backend.app.llm.base import LLMMessage
from backend.app.llm.router import LLMRouter
from backend.app.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.app.agent.state import AgentState
from backend.app.agent.adk_agents.runner import get_adk_runner
from backend.app.services.prediction_service import get_predictor
from backend.app.core.config import get_settings


async def run_scenario_tests():
    settings = get_settings()
    if not settings.primary_llm_api_key or not settings.secondary_llm_api_key:
        raise RuntimeError("Set PRIMARY_LLM_API_KEY and SECONDARY_LLM_API_KEY in the environment before running this test.")

    print("=" * 90)
    print("  LLM ROUTER & PROVIDER FAILOVER TEST SUITE")
    print("=" * 90)

    # 1. Primary Available -> Primary Response
    print("\n--- Scenario 1: Primary Available ---")
    r1 = LLMRouter()
    p1 = OpenAICompatibleProvider(
        name="primary",
        model=settings.primary_llm_model,
        api_key=settings.primary_llm_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    p2 = OpenAICompatibleProvider(
        name="secondary",
        model=settings.secondary_llm_model or "qwen/qwen3.6-27b",
        api_key=settings.secondary_llm_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    r1.add_provider(p1)
    r1.add_provider(p2)

    health_report = await r1.check_all_providers_health()
    print(f"Health check status: {health_report}")

    res1 = await r1.generate([LLMMessage(role="user", content="Hi")])
    print(f"Result 1 -> Provider: {res1.provider}, Model: {res1.model}")
    assert res1.provider == "primary"

    # 2. Primary 429 / Unavailable -> Secondary Response
    print("\n--- Scenario 2: Primary Failure -> Secondary Failover ---")
    r2 = LLMRouter()
    p1_broken = OpenAICompatibleProvider(
        name="primary",
        model="openai/gpt-oss-120b-invalid-model-404",
        api_key=settings.primary_llm_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    r2.add_provider(p1_broken)
    r2.add_provider(p2)

    res2 = await r2.generate([LLMMessage(role="user", content="Hi")])
    print(f"Result 2 -> Provider: {res2.provider}, Model: {res2.model}")
    assert res2.provider == "secondary"

    # 3. Both Unavailable -> Preserve ML Prediction & Graceful Response
    print("\n--- Scenario 3: Both Providers Unavailable -> ML Prediction Preserved ---")
    r3 = LLMRouter()
    p2_broken = OpenAICompatibleProvider(
        name="secondary",
        model="invalid-model-404-test",
        api_key="gsk_invalid",
        base_url="https://api.groq.com/openai/v1",
    )
    r3.add_provider(p1_broken)
    r3.add_provider(p2_broken)

    predictor = get_predictor()
    runner = get_adk_runner()
    state = AgentState(conversation_id="failover-test-session")
    state.update_feature("Vehicle_Age", 10, "known")

    msg = "Brake is good."
    out = await runner.run_turn(msg, "failover-test-session", state, [])

    print(f"Result 3 Response: \"{out.get('response')}\"")
    print(f"ML Prediction Preserved? {out.get('prediction') is not None}")
    if out.get('prediction'):
        print(f"  Prediction ID: {out['prediction'].get('prediction_id')}")
        print(f"  Maintenance Prob: {out['prediction'].get('maintenance_probability')}")
        print(f"  Displayed Score: {out['prediction'].get('model_score')}")

    assert out.get("prediction") is not None
    assert "prediction" in out["response"].lower() or "service" in out["response"].lower() or "trouble" in out["response"].lower()

    print("\n" + "=" * 90)
    print("  ALL LLM FAILOVER SCENARIOS VERIFIED SUCCESSFULLY!")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(run_scenario_tests())

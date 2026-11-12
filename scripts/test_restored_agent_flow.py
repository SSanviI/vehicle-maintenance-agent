"""
Verification script for restored agent questioning flow and score consistency.
"""

import asyncio
import json
import joblib

from backend.app.agent.state import AgentState
from backend.app.agent.adk_agents.runner import get_adk_runner
from backend.app.services.prediction_service import get_predictor


async def run_restored_agent_test():
    predictor = get_predictor()
    runner = get_adk_runner()
    state = AgentState(conversation_id="restored-agent-session")

    print("=" * 100)
    print("  RESTORED AGENT QUESTIONING & SCORE CONSISTENCY TRACE")
    print("=" * 100)

    # Turn 1
    t1_msg = "My car is 10 years old. Tire is making weird sound."
    res1 = await runner.run_turn(t1_msg, "restored-agent-session", state, [])
    p1 = res1.get("prediction") or {}

    print(f"\n[Turn 1] User: \"{t1_msg}\"")
    print(f"  Extracted / Known State: {state.get_known_features()}")
    print(f"  Agent Question / Response: \"{res1.get('response')}\"")
    print(f"  ML Calls: 1 | Prediction ID: {p1.get('prediction_id')} | Maint Prob: {p1.get('maintenance_probability')} | Class: {p1.get('prediction_class')}")

    await asyncio.sleep(1.5)

    # Turn 2: User answers next feature asked (or Brake condition if asked)
    next_feat = state.current_question_feature or "Brake_Condition"
    t2_msg = "Brake is in good condition."
    res2 = await runner.run_turn(t2_msg, "restored-agent-session", state, [{"role": "user", "content": t1_msg}, {"role": "assistant", "content": res1.get("response", "")}])
    p2 = res2.get("prediction") or {}

    print(f"\n[Turn 2] User: \"{t2_msg}\"")
    print(f"  Extracted / Known State: {state.get_known_features()}")
    print(f"  Agent Question / Response: \"{res2.get('response')}\"")
    print(f"  ML Calls: 1 | Prediction ID: {p2.get('prediction_id')} | Maint Prob: {p2.get('maintenance_probability')} | Class: {p2.get('prediction_class')}")

    await asyncio.sleep(1.5)

    # Turn 3
    t3_msg = "Battery is new."
    res3 = await runner.run_turn(t3_msg, "restored-agent-session", state, [
        {"role": "user", "content": t1_msg}, {"role": "assistant", "content": res1.get("response", "")},
        {"role": "user", "content": t2_msg}, {"role": "assistant", "content": res2.get("response", "")}
    ])
    p3 = res3.get("prediction") or {}

    print(f"\n[Turn 3] User: \"{t3_msg}\"")
    print(f"  Extracted / Known State: {state.get_known_features()}")
    print(f"  Agent Question / Response: \"{res3.get('response')}\"")
    print(f"  ML Calls: 1 | Prediction ID: {p3.get('prediction_id')} | Maint Prob: {p3.get('maintenance_probability')} | Class: {p3.get('prediction_class')}")

    print("\n" + "=" * 100)
    print("  SCORE CONSISTENCY CHECK across Turns:")
    for turn_num, p in enumerate([p1, p2, p3], 1):
        if p:
            print(f"  Turn {turn_num}: ID={p.get('prediction_id')[:8]}, Class={p.get('prediction_class')}, Maint_Prob={p.get('maintenance_probability')}, Score={p.get('model_score')}")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(run_restored_agent_test())

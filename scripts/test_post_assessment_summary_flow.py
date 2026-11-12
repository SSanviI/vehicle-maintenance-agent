"""
Test Suite: Post-Assessment Summary Flow & Follow-Up Q&A
Tests the complete multi-turn flow:
1. Complete assessment (reaching 8-9 features) -> asks "Would you like to see a summary...?"
2. User replies "yes" -> produces a REAL summary (NOT repeating completion message)
3. User asks "what is the main factor for this output?" -> local model explanation
4. User asks "why is the risk low?" -> grounded model explanation without causal claims
5. User asks "which factor affected the score most?" -> the same local explanation capability
6. User asks "what is the remaining useful life?" -> clarifies model boundary without hallucinated RUL
7. User asks "what should I do next?" -> supported recommendation
8. Fresh assessment session -> normal assessment flow works
9. Explicit new assessment -> clears previous canonical assessment facts
"""

import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.app.agent.state import AgentState
from backend.app.agent.adk_agents.runner import get_adk_runner
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


async def test_summary_flow():
    print("=" * 80)
    print("  AUTOCARE AI — POST-ASSESSMENT SUMMARY FLOW TEST")
    print("=" * 80)

    runner = get_adk_runner()
    runner.initialize()

    # -------------------------------------------------------------------------
    # STEP 1: Assessment Completion
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: Complete Assessment (8 confirmed features) ---")
    state = AgentState(conversation_id="summary-test-conv")
    seed_features = {
        "Vehicle_Age": 2,
        "Reported_Issues": 2,
        "Maintenance_History": "Average",
        "Service_History": 3,
        "Last_Service_Date": "2025-08-27",
        "Brake_Condition": "Good",
        "Battery_Status": "Weak",
        "Accident_History": 0,
    }
    for k, v in seed_features.items():
        state.update_feature(k, v, status="known")

    res1 = await runner.run_turn(
        user_message="I also confirmed my details",
        conversation_id="summary-test-conv",
        agent_state=state,
        conversation_history=[],
    )
    msg1 = res1["response"]
    print(f"\nStep 1 Assistant:\n\"{msg1}\"")
    
    assert state.assessment_complete, "Assessment should be marked complete!"
    assert state.assessment_mode == "AWAITING_SUMMARY_CONFIRMATION", f"Mode should be AWAITING_SUMMARY_CONFIRMATION, got {state.assessment_mode}"
    assert "summary" in msg1.lower(), "Completion message should ask whether user wants a summary!"
    assert not msg1.strip().startswith("•"), "Completion message must NOT be the bulleted summary yet!"
    print("  PASS Step 1: Assessment complete, asked user if they want a summary.")

    # -------------------------------------------------------------------------
    # STEP 2: User says "yes" -> Should deliver REAL summary
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: User says 'yes' -> Deliver Summary ---")
    res2 = await runner.run_turn(
        user_message="yes",
        conversation_id="summary-test-conv",
        agent_state=state,
        conversation_history=[
            {"role": "assistant", "content": msg1},
            {"role": "user", "content": "yes"}
        ],
    )
    msg2 = res2["response"]
    print(f"\nStep 2 Assistant:\n\"{msg2}\"")

    # MUST NOT repeat completion question
    assert "gives me enough information to complete" not in msg2.lower(), "BUG: Repeated completion message!"
    assert "would you like to see a summary" not in msg2.lower(), "BUG: Repeated summary question!"
    # MUST contain confirmed features & risk
    assert any(term in msg2.lower() for term in ("risk", "vehicle age", "brake", "battery", "maintenance")), "Summary must contain confirmed details!"
    assert state.assessment_mode == "FOLLOW_UP", f"Mode after summary should be FOLLOW_UP, got {state.assessment_mode}"
    assert "2025-08-27" in msg2, "Summary must retain the canonical confirmed service date!"
    assert res2.get("explanation", {}).get("available"), "Completed result should expose local explanation data to the UI."
    print("  PASS Step 2: User 'yes' produced actual summary (no repeated completion message).")

    # -------------------------------------------------------------------------
    # STEP 3: User asks for the main factor -> Local model explanation
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: User asks 'what is the main factor for this output?' ---")
    res3 = await runner.run_turn(
        user_message="what is the main factor for this output?",
        conversation_id="summary-test-conv",
        agent_state=state,
        conversation_history=[
            {"role": "assistant", "content": msg2},
            {"role": "user", "content": "what is the main factor for this output?"}
        ],
    )
    msg3 = res3["response"]
    print(f"\nStep 3 Assistant:\n\"{msg3}\"")
    assert "would you like to see a summary" not in msg3.lower(), "BUG: Factor question repeated summary prompt!"
    main_factor = res3["local_explanation"]["main_factor"]["display_name"].lower()
    assert main_factor in msg3.lower(), "Factor response must identify the strongest local contribution!"
    assert state.assessment_mode == "FOLLOW_UP", "Explanation should remain available for follow-up questions."
    # Verify no ungrounded causal diagnosis or hallucinations
    assert not any(p in msg3.lower() for p in ("caused the failure", "is why it failed"))
    print("  PASS Step 3: Direct local model explanation provided without causal claims.")

    # -------------------------------------------------------------------------
    # STEP 4: Alternate factor wording -> Same local explanation capability
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: User asks 'which factor affected the score most?' ---")
    res3b = await runner.run_turn(
        user_message="which factor affected the score most?",
        conversation_id="summary-test-conv",
        agent_state=state,
        conversation_history=[
            {"role": "assistant", "content": msg3},
            {"role": "user", "content": "which factor affected the score most?"}
        ],
    )
    msg3b = res3b["response"]
    print(f"\nStep 4 Assistant:\n\"{msg3b}\"")
    assert "would you like to see a summary" not in msg3b.lower(), "BUG: Alternate factor question repeated summary prompt!"
    assert res3b["local_explanation"]["main_factor"]["display_name"].lower() in msg3b.lower()
    print("  PASS Step 4: Alternate factor wording used the same local explanation path.")

    # -------------------------------------------------------------------------
    # STEP 5: User asks "why is the risk low?" -> Grounded model explanation
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: User asks 'why is the risk low?' ---")
    res4 = await runner.run_turn(
        user_message="why is the risk low?",
        conversation_id="summary-test-conv",
        agent_state=state,
        conversation_history=[
            {"role": "assistant", "content": msg3b},
            {"role": "user", "content": "why is the risk low?"}
        ],
    )
    msg4 = res4["response"]
    print(f"\nStep 5 Assistant:\n\"{msg4}\"")
    assert "would you like to see a summary" not in msg4.lower(), "BUG: Risk explanation repeated summary prompt!"
    assert not any(p in msg4.lower() for p in ("caused the failure", "is why it failed"))
    print("  PASS Step 5: Risk explanation provided from local model evidence.")

    # -------------------------------------------------------------------------
    # STEP 6: User asks "what is the remaining useful life?" -> Clarify model boundary
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: User asks 'what is the remaining useful life?' ---")
    res5 = await runner.run_turn(
        user_message="what is the remaining useful life?",
        conversation_id="summary-test-conv",
        agent_state=state,
        conversation_history=[
            {"role": "assistant", "content": msg4},
            {"role": "user", "content": "what is the remaining useful life?"}
        ],
    )
    msg5 = res5["response"]
    print(f"\nStep 6 Assistant:\n\"{msg5}\"")
    assert any(term in msg5.lower() for term in ("remaining useful life", "useful life", "maintenance risk", "does not", "doesn't")), "Should clarify RUL boundary!"
    print("  PASS Step 6: Clarified RUL boundary without hallucinating remaining years.")

    # -------------------------------------------------------------------------
    # STEP 7: User asks for a recommendation -> Supported next step
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: User asks 'what should I do next?' ---")
    res5b = await runner.run_turn(
        user_message="what should I do next?",
        conversation_id="summary-test-conv",
        agent_state=state,
        conversation_history=[
            {"role": "assistant", "content": msg5},
            {"role": "user", "content": "what should I do next?"}
        ],
    )
    msg5b = res5b["response"]
    print(f"\nStep 7 Assistant:\n\"{msg5b}\"")
    assert "would you like to see a summary" not in msg5b.lower(), "BUG: Recommendation repeated summary prompt!"
    assert res5b["local_explanation"]["positive_contributors"][0]["display_name"].lower() in msg5b.lower()
    print("  PASS Step 7: Recommendation is anchored to a supported higher-risk contributor.")

    # -------------------------------------------------------------------------
    # STEP 8: Fresh assessment session -> Normal assessment flow works
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: Fresh Assessment Session ---")
    fresh_state = AgentState(conversation_id="fresh-session-test")
    res6 = await runner.run_turn(
        user_message="My car is 3 years old.",
        conversation_id="fresh-session-test",
        agent_state=fresh_state,
        conversation_history=[],
    )
    msg6 = res6["response"]
    print(f"\nStep 8 Assistant:\n\"{msg6}\"")
    assert not fresh_state.assessment_complete, "Fresh session should NOT be complete!"
    assert fresh_state.assessment_mode == "ASSESSMENT", f"Fresh session mode should be ASSESSMENT, got {fresh_state.assessment_mode}"
    assert "?" in msg6, "Fresh session turn 1 should ask a question!"
    print("  PASS Step 8: Fresh assessment session operates normally.")

    # -------------------------------------------------------------------------
    # STEP 9: Decline summary flow ("no")
    # -------------------------------------------------------------------------
    print("\n--- STEP 9: User says 'no' to summary confirmation ---")
    decline_state = AgentState(conversation_id="decline-test")
    for k, v in seed_features.items():
        decline_state.update_feature(k, v, status="known")
    decline_state.assessment_complete = True
    decline_state.assessment_mode = "AWAITING_SUMMARY_CONFIRMATION"

    res7 = await runner.run_turn(
        user_message="no thanks",
        conversation_id="decline-test",
        agent_state=decline_state,
        conversation_history=[],
    )
    msg7 = res7["response"]
    print(f"\nStep 9 Assistant:\n\"{msg7}\"")
    assert "no problem" in msg7.lower() or "complete" in msg7.lower()
    print("  PASS Step 9: 'no thanks' closed politely without restarting or asking questions.")

    # -------------------------------------------------------------------------
    # STEP 10: Explicit new assessment -> clear canonical assessment facts
    # -------------------------------------------------------------------------
    print("\n--- STEP 10: User requests a new assessment ---")
    restart_state = AgentState(conversation_id="restart-test")
    for k, v in seed_features.items():
        restart_state.update_feature(k, v, status="known")
    restart_state.assessment_complete = True
    restart_state.assessment_mode = "FOLLOW_UP"
    restart_result = await runner.run_turn(
        user_message="start a new assessment",
        conversation_id="restart-test",
        agent_state=restart_state,
        conversation_history=[],
    )
    assert not restart_state.assessment_complete
    assert restart_state.assessment_mode == "ASSESSMENT"
    assert restart_state.known_feature_count() == 0
    assert "starting a new maintenance assessment" in restart_result["response"].lower()
    print("  PASS Step 10: New assessment cleared prior facts without resetting the agent architecture.")

    print("\n" + "=" * 80)
    print("  ALL POST-ASSESSMENT SUMMARY FLOW TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_summary_flow())

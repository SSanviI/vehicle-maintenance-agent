"""
Comprehensive Regression Test Suite — Validating All 15 Test Scenarios (TEST A to TEST O)
Running through the actual multi-agent system and API layer.
"""

import sys
import asyncio
import re
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.app.agent.state import AgentState
from backend.app.agent.adk_agents.runner import get_adk_runner
from backend.app.agent.tools.feature_tools import (
    extract_explicit_facts,
    validate_feature_name,
    validate_feature_value,
    normalise_date_value,
    apply_structured_extraction,
)
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


async def run_all_tests():
    print("=" * 80)
    print("  AUTOCARE AI — COMPREHENSIVE REGRESSION SUITE (TESTS A - O)")
    print("=" * 80)

    runner = get_adk_runner()
    runner.initialize()

    # -------------------------------------------------------------------------
    # TEST A: SHORT AGE ("2" when asked age)
    # -------------------------------------------------------------------------
    print("\n[TEST A] Short numeric age: '2' when Vehicle_Age is active question")
    facts_a = extract_explicit_facts("2", current_question_feature="Vehicle_Age")
    extracted_a = {f["name"]: f["value"] for f in facts_a["extracted_features"]}
    assert extracted_a.get("Vehicle_Age") == 2, f"Failed: {extracted_a}"
    print(f"  PASS: Vehicle_Age = {extracted_a['Vehicle_Age']}")

    # -------------------------------------------------------------------------
    # TEST B: NATURAL AGE ("age is 2 years")
    # -------------------------------------------------------------------------
    print("\n[TEST B] Natural age: 'age is 2 years'")
    facts_b = extract_explicit_facts("age is 2 years")
    extracted_b = {f["name"]: f["value"] for f in facts_b["extracted_features"]}
    assert extracted_b.get("Vehicle_Age") == 2, f"Failed: {extracted_b}"
    print(f"  PASS: Vehicle_Age = {extracted_b['Vehicle_Age']}")

    # -------------------------------------------------------------------------
    # TEST C: BRAKE ("good brakes")
    # -------------------------------------------------------------------------
    print("\n[TEST C] Brake condition: 'good brakes'")
    facts_c = extract_explicit_facts("good brakes", current_question_feature="Brake_Condition")
    extracted_c = {f["name"]: f["value"] for f in facts_c["extracted_features"]}
    assert extracted_c.get("Brake_Condition") == "Good", f"Failed: {extracted_c}"
    print(f"  PASS: Brake_Condition = {extracted_c['Brake_Condition']}")

    # -------------------------------------------------------------------------
    # TEST D: BATTERY WITH TYPO ("batteru is weak")
    # -------------------------------------------------------------------------
    print("\n[TEST D] Battery typo: 'batteru is weak'")
    facts_d = extract_explicit_facts("batteru is weak", current_question_feature="Battery_Status")
    extracted_d = {f["name"]: f["value"] for f in facts_d["extracted_features"]}
    assert extracted_d.get("Battery_Status") == "Weak", f"Failed: {extracted_d}"
    print(f"  PASS: Battery_Status = {extracted_d['Battery_Status']}")

    # -------------------------------------------------------------------------
    # TEST E: BATTERY SHORT ("good" when Battery_Status is active)
    # -------------------------------------------------------------------------
    print("\n[TEST E] Battery short: 'good' when Battery_Status is active question")
    facts_e = extract_explicit_facts("good", current_question_feature="Battery_Status")
    extracted_e = {f["name"]: f["value"] for f in facts_e["extracted_features"]}
    assert extracted_e.get("Battery_Status") == "Good", f"Failed: {extracted_e}"
    print(f"  PASS: Battery_Status = {extracted_e['Battery_Status']}")

    # -------------------------------------------------------------------------
    # TEST F: SERVICE ("4 services")
    # -------------------------------------------------------------------------
    print("\n[TEST F] Service visits: '4 services'")
    facts_f = extract_explicit_facts("4 services", current_question_feature="Service_History")
    extracted_f = {f["name"]: f["value"] for f in facts_f["extracted_features"]}
    assert extracted_f.get("Service_History") == 4, f"Failed: {extracted_f}"
    print(f"  PASS: Service_History = {extracted_f['Service_History']}")

    # -------------------------------------------------------------------------
    # TEST G: DATE ("27/08/2025")
    # -------------------------------------------------------------------------
    print("\n[TEST G] Date short: '27/08/2025' when Last_Service_Date is active")
    facts_g = extract_explicit_facts("27/08/2025", current_question_feature="Last_Service_Date")
    extracted_g = {f["name"]: f["value"] for f in facts_g["extracted_features"]}
    assert extracted_g.get("Last_Service_Date") == "2025-08-27", f"Failed: {extracted_g}"
    print(f"  PASS: Last_Service_Date = {extracted_g['Last_Service_Date']}")

    # -------------------------------------------------------------------------
    # TEST H: DATE NATURAL ("last serviced on 27 august 2025")
    # -------------------------------------------------------------------------
    print("\n[TEST H] Date natural: 'last serviced on 27 august 2025'")
    facts_h = extract_explicit_facts("last serviced on 27 august 2025")
    extracted_h = {f["name"]: f["value"] for f in facts_h["extracted_features"]}
    assert extracted_h.get("Last_Service_Date") == "2025-08-27", f"Failed: {extracted_h}"
    print(f"  PASS: Last_Service_Date = {extracted_h['Last_Service_Date']}")

    # -------------------------------------------------------------------------
    # TEST I: SPELLING ERROR ("maintainance is avrage")
    # -------------------------------------------------------------------------
    print("\n[TEST I] Spelling error: 'maintainance is avrage'")
    facts_i = extract_explicit_facts("maintainance is avrage")
    extracted_i = {f["name"]: f["value"] for f in facts_i["extracted_features"]}
    assert extracted_i.get("Maintenance_History") == "Average", f"Failed: {extracted_i}"
    print(f"  PASS: Maintenance_History = {extracted_i['Maintenance_History']}")

    # -------------------------------------------------------------------------
    # TEST J: MULTIPLE FACTS ("My car is 5 years old, has 3 problems and has had 4 services.")
    # -------------------------------------------------------------------------
    print("\n[TEST J] Multiple facts extraction")
    msg_j = "My car is 5 years old, has 3 problems and has had 4 services."
    facts_j = extract_explicit_facts(msg_j)
    extracted_j = {f["name"]: f["value"] for f in facts_j["extracted_features"]}
    assert extracted_j.get("Vehicle_Age") == 5, f"Age failed: {extracted_j}"
    assert extracted_j.get("Reported_Issues") == 3, f"Issues failed: {extracted_j}"
    assert extracted_j.get("Service_History") == 4, f"Service failed: {extracted_j}"
    print(f"  PASS: All 3 facts extracted: {extracted_j}")

    # -------------------------------------------------------------------------
    # FULL RUNTIME TESTS (VIA API / ADK RUNNER)
    # -------------------------------------------------------------------------
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=45.0) as client:
        # TEST L: FIRST TURN NON-BLANK + ONE QUESTION
        print("\n[TEST L] Full Runtime: First Turn Non-Blank + Grounded Question")
        res_l = await client.post("/api/chat", json={"message": "My car is making a strange noise."})
        assert res_l.status_code == 200
        data_l = res_l.json()
        msg_l = data_l["message"]
        print(f"  Response ({len(msg_l)} chars): \"{msg_l}\"")
        assert len(msg_l.strip()) > 20, "Response is blank or too short!"
        assert "?" in msg_l, "First turn must contain a question!"
        conv_id = data_l["conversation_id"]
        print(f"  PASS: First turn produced non-blank grounded response with question. Conv: {conv_id[:8]}")

        # TEST K: NO REPEAT QUESTION
        print("\n[TEST K] Full Runtime: No Repeated Questions")
        # Turn 2: User answers "good brakes"
        res_k = await client.post("/api/chat", json={"message": "good brakes", "conversation_id": conv_id})
        assert res_k.status_code == 200
        data_k = res_k.json()
        msg_k = data_k["message"]
        state_k = data_k["feature_state"]["known_features"]
        print(f"  Known features: {state_k}")
        print(f"  Response: \"{msg_k}\"")
        assert "Brake_Condition" in state_k
        assert state_k["Brake_Condition"] == "Good"
        # Verify next question is NOT about brake condition
        assert "brake condition" not in msg_k.lower() or "how is your brake" not in msg_k.lower()
        print("  PASS: Brake_Condition recorded as Good and not asked again.")

        # TEST N: UNSUPPORTED RUL QUERY
        print("\n[TEST N] Full Runtime: Unsupported Remaining Useful Life (RUL)")
        res_n = await client.post("/api/chat", json={"message": "What is the remaining useful life of my vehicle?", "conversation_id": conv_id})
        assert res_n.status_code == 200
        data_n = res_n.json()
        msg_n = data_n["message"]
        print(f"  Response: \"{msg_n}\"")
        # Ensure it does NOT invent a remaining life number (e.g. "X years remaining")
        assert not re.search(r"\b\d+(?:\.\d+)?\s+years\s+(?:remaining|left|of\s+useful\s+life)\b", msg_n.lower()), "Hallucinated RUL years!"
        print("  PASS: Out-of-scope RUL clarified without hallucination.")

        # TEST O: FAILURE CAUSE QUERY
        print("\n[TEST O] Full Runtime: Unsupported Failure Cause Query")
        res_o = await client.post("/api/chat", json={"message": "What are the reasons for vehicle failure?", "conversation_id": conv_id})
        assert res_o.status_code == 200
        data_o = res_o.json()
        msg_o = data_o["message"]
        print(f"  Response: \"{msg_o}\"")
        # Ensure no hallucinated causal assertions
        assert not any(p in msg_o.lower() for p in ("caused the failure", "is why the vehicle failed"))
        print("  PASS: Failure causes addressed without unsupported causal claims.")

        # TEST M: ASSESSMENT COMPLETION AT ~8-9 FEATURES
        print("\n[TEST M] Full Runtime: 8-9 Features Assessment Completion")
        test_state = AgentState(conversation_id="completion-test-session")
        # Seed state with 8 confirmed features
        seed_features = {
            "Vehicle_Age": 5,
            "Reported_Issues": 3,
            "Service_History": 4,
            "Brake_Condition": "Good",
            "Battery_Status": "Weak",
            "Maintenance_History": "Average",
            "Tire_Condition": "Good",
            "Accident_History": 0,
        }
        for k, v in seed_features.items():
            test_state.update_feature(k, v, status="known")

        res_m = await runner.run_turn(
            user_message="My last service was 2025-08-27",
            conversation_id="completion-test-session",
            agent_state=test_state,
            conversation_history=[],
        )
        msg_m = res_m["response"]
        print(f"  Assessed count: {test_state.known_feature_count()}")
        print(f"  Assessment complete: {test_state.assessment_complete}")
        print(f"  Prediction: {res_m.get('prediction', {}).get('risk_level')}")
        print(f"  Final Response:\n\"{msg_m}\"")
        assert test_state.assessment_complete, "Assessment should be complete!"
        assert test_state.current_question_feature is None, "Should have no pending feature question!"
        assert "summary" in msg_m.lower(), "Final completion response must offer summary!"
        assert not any(f"what is your {feat.lower()}" in msg_m.lower() for feat in seed_features), "Must not ask feature questions!"
        print("  PASS: Assessment completed properly with summary offer.")

    print("\n" + "=" * 80)
    print("  ALL REGRESSION TESTS (A - O) PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())

import asyncio
import pprint
from backend.app.agent.state.state import AgentState
from backend.app.services.prediction_service import get_predictor
from backend.app.ml.explainer import (
    get_local_explainer,
    explanation_for_ui,
    format_factor_explanation_text,
    format_risk_explanation_text,
    format_score_factors_text,
)
from backend.app.agent.tools.feature_tools import classify_post_assessment_intent
from backend.app.agent.adk_agents.judge_agent import run_deterministic_guards, generate_deterministic_fallback_response


async def run_tests():
    print("=" * 70)
    print("1. TEST PREDICTOR & SHAP EXPLAINER DIRECTLY")
    print("=" * 70)
    predictor = get_predictor()
    explainer = get_local_explainer()

    # Scenario A: High Risk Vehicle
    state_high = {
        "Reported_Issues": {"value": 4, "status": "known", "source": "user"},
        "Brake_Condition": {"value": "Worn Out", "status": "known", "source": "user"},
        "Battery_Status": {"value": "Weak", "status": "known", "source": "user"},
        "Vehicle_Age": {"value": 12, "status": "known", "source": "user"},
    }

    pred_a = predictor.predict(state_high)
    expl_a = explainer.explain_prediction(state_high, pred_a)

    print(f"Prediction A Risk: {pred_a['risk_level']}, Score: {pred_a['model_score']}")
    print(f"Explanation A Available: {expl_a['available']}")
    print(f"Base Value: {expl_a['shap_base_value']}")
    print(f"Main Factor: {expl_a['main_factor']}")
    print("Top Contributors:")
    for c in expl_a["top_contributors"]:
        print(f"  • {c['display_name']} ({c['value']}): SHAP={c['contribution']:+.4f} [{c['direction']}]")

    assert expl_a["available"] is True, "Explanation A should be available"
    assert expl_a["main_factor"] is not None, "Main factor should not be None"
    assert expl_a["main_factor"]["direction"] == "higher_risk", "High risk scenario main factor should push toward higher risk"

    # Test Question 1: "What is the main factor?"
    ans_main = format_factor_explanation_text(expl_a, pred_a)
    print("\n[ANSWER: What is the main factor?]")
    print(ans_main)
    assert expl_a["main_factor"]["display_name"].lower() in ans_main.lower(), "Main factor name must appear in answer"

    # Test Question 2: "What affected the score?"
    ans_score = format_score_factors_text(expl_a, pred_a)
    print("\n[ANSWER: What affected the score?]")
    print(ans_score)

    # Test Question 3: "Why is the risk high?"
    ans_risk = format_risk_explanation_text(expl_a, pred_a)
    print("\n[ANSWER: Why is the risk high?]")
    print(ans_risk)
    assert "higher-risk" in ans_risk or "higher" in ans_risk, "Risk explanation must mention higher risk side"

    print("\n" + "=" * 70)
    print("2. TEST STATE CHANGE INVALIDATION & UPDATED SHAP")
    print("=" * 70)
    # Scenario B: Low Risk Vehicle (Replacing worn components with new)
    state_low = {
        "Reported_Issues": {"value": 0, "status": "known", "source": "user"},
        "Brake_Condition": {"value": "New", "status": "known", "source": "user"},
        "Battery_Status": {"value": "New", "status": "known", "source": "user"},
        "Maintenance_History": {"value": "Good", "status": "known", "source": "user"},
        "Vehicle_Age": {"value": 1, "status": "known", "source": "user"},
    }

    pred_b = predictor.predict(state_low)
    expl_b = explainer.explain_prediction(state_low, pred_b)

    print(f"Prediction B Risk: {pred_b['risk_level']}, Score: {pred_b['model_score']}")
    print(f"Main Factor B: {expl_b['main_factor']}")
    print("Top Contributors B:")
    for c in expl_b["top_contributors"]:
        print(f"  • {c['display_name']} ({c['value']}): SHAP={c['contribution']:+.4f} [{c['direction']}]")

    assert pred_b["risk_level"] == "LOW", "State B should have LOW risk"
    assert expl_b["main_factor"]["direction"] == "lower_risk", "Low risk main factor should push toward lower risk"

    # Test Question 4: "Why is the risk low?"
    ans_low = format_risk_explanation_text(expl_b, pred_b)
    print("\n[ANSWER: Why is the risk low?]")
    print(ans_low)
    assert "lower-risk" in ans_low or "lower" in ans_low, "Low risk explanation must describe lower risk factors"

    print("\n" + "=" * 70)
    print("3. TEST INTENT CLASSIFIER ON USER QUESTIONS")
    print("=" * 70)
    test_queries = [
        ("what is the main factor?", "FACTOR_EXPLANATION"),
        ("what is the top factor", "FACTOR_EXPLANATION"),
        ("which factor affected the score?", "FACTOR_EXPLANATION"),
        ("what affected the score", "FACTOR_EXPLANATION"),
        ("what drove the score", "FACTOR_EXPLANATION"),
        ("why is the risk high?", "RISK_EXPLANATION"),
        ("why is the risk low?", "RISK_EXPLANATION"),
        ("what is the reason for this output?", "RISK_EXPLANATION"),
        ("why this result?", "RISK_EXPLANATION"),
        ("why?", "RISK_EXPLANATION"),
        ("yes show me summary", "SUMMARY_REQUEST"),
    ]
    for query, expected_intent in test_queries:
        actual_intent = classify_post_assessment_intent(query, "AWAITING_SUMMARY_CONFIRMATION")
        print(f"Query: '{query}' -> Intent: {actual_intent} (Expected: {expected_intent})")
        assert actual_intent == expected_intent, f"Expected {expected_intent} for query '{query}' but got {actual_intent}"

    print("\n" + "=" * 70)
    print("4. TEST DETERMINISTIC GUARDS & JUDGE ON SHAP ANSWERS")
    print("=" * 70)
    agent_state = AgentState()
    for feat, data in state_high.items():
        agent_state.update_feature(feat, data["value"], status="known")
    agent_state.add_prediction(pred_a)
    agent_state.add_explanation(expl_a)
    agent_state.assessment_complete = True
    agent_state.assessment_mode = "FOLLOW_UP"

    # Validate correct SHAP answer passes guards
    is_valid, violations = run_deterministic_guards(
        response_text=ans_main,
        state=agent_state,
        prediction=pred_a,
        expected_next_feature=None,
        post_assessment_intent="FACTOR_EXPLANATION",
        local_explanation=expl_a,
    )
    print(f"Guard Validation for ans_main: valid={is_valid}, violations={violations}")
    assert is_valid, f"Guard rejected valid SHAP answer: {violations}"

    # Validate hallucinated factor is rejected
    fake_response = "The main factor is definitely your engine oil level which caused the vehicle failure with 3.5 years left of life."
    is_fake_valid, fake_violations = run_deterministic_guards(
        response_text=fake_response,
        state=agent_state,
        prediction=pred_a,
        expected_next_feature=None,
        post_assessment_intent="FACTOR_EXPLANATION",
        local_explanation=expl_a,
    )
    print(f"Guard Validation for fake_response: valid={is_fake_valid}, violations={fake_violations}")
    assert not is_fake_valid, "Guard must reject hallucinated factors/causes"

    print("\n" + "=" * 70)
    print("ALL SHAP INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_tests())

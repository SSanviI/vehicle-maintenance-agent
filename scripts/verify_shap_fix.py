"""
End-to-end fix verification: simulate exact failing scenario.
User has 8 features confirmed, assessment complete, asks 'What is the main factor?'
Verifies that SHAP produces a grounded answer, NOT the fallback message.
"""
from backend.app.agent.state.state import AgentState
from backend.app.ml.explainer import get_local_explainer, explanation_for_ui
from backend.app.agent.tools.feature_tools import classify_post_assessment_intent
from backend.app.agent.adk_agents.judge_agent import generate_deterministic_fallback_response

# Build the exact state from the failing test case
state = AgentState("test-shap-fix")
for feat, val in [
    ("Brake_Condition", "Worn Out"),
    ("Battery_Status", "Weak"),
    ("Reported_Issues", 8),
    ("Maintenance_History", "Good"),
    ("Service_History", 4),
    ("Last_Service_Date", "2025-07-08"),
    ("Accident_History", 0),
    ("Mileage", 1000),
]:
    state.update_feature(feat, val)

state.assessment_complete = True
state.assessment_mode = "FOLLOW_UP"

latest_pred = {
    "risk_level": "HIGH",
    "model_score": 1.0,
    "prediction": "maintenance_required",
    "prediction_class": 1,
    "prediction_id": "test-abc123",
}
state.add_prediction(latest_pred)

# TURN: user asks "What is the main factor?"
user_message = "What is the main factor?"
prev_mode = state.assessment_mode
intent = classify_post_assessment_intent(user_message, prev_mode)
print("Intent classified:", intent)
assert intent == "FACTOR_EXPLANATION", f"WRONG INTENT: {intent}"

# Simulate runner post-assessment block
local_expl = None
if state.has_current_explanation():
    local_expl = state.get_latest_explanation()
    print("Using cached explanation")
else:
    print("Computing fresh SHAP explanation...")
    explainer = get_local_explainer()
    print("  model:", type(explainer.model).__name__ if explainer.model else None)
    print("  tree_explainer:", explainer._tree_explainer is not None)
    local_expl = explainer.explain_prediction(state.feature_state, latest_pred)
    if local_expl and local_expl.get("available"):
        state.add_explanation(local_expl)

print("SHAP available:", local_expl["available"])
mf = local_expl.get("main_factor")
print("main_factor:", mf["feature"] if mf else None)

ui_expl = explanation_for_ui(local_expl)
print("UI explanation available:", ui_expl["available"])
print("UI top_factors count:", len(ui_expl["top_factors"]))
print("UI top_factors:", [(f["display_name"], f["direction"]) for f in ui_expl["top_factors"]])

# The deterministic fallback
fallback = generate_deterministic_fallback_response(
    state, latest_pred, None,
    post_assessment_intent=intent,
    local_explanation=local_expl
)
print()
print("=== EXPECTED RESPONSE (deterministic fallback) ===")
print(fallback)
print()

BAD_MSG = "I don't have a reliable per-prediction factor breakdown"
assert BAD_MSG not in fallback, f"STILL USING FALLBACK MESSAGE!\n{fallback}"
assert "Reported issues" in fallback, f"main_factor not mentioned!\n{fallback}"
print("ASSERTION PASSED: Response is SHAP-grounded, not the fallback message.")

# TURN: user asks "Why is the risk high?"
user_message2 = "Why is the risk high?"
intent2 = classify_post_assessment_intent(user_message2, "FOLLOW_UP")
print()
print("Intent2:", intent2)
assert intent2 == "RISK_EXPLANATION", f"WRONG INTENT2: {intent2}"

fallback2 = generate_deterministic_fallback_response(
    state, latest_pred, None,
    post_assessment_intent=intent2,
    local_explanation=local_expl
)
print("=== RISK_EXPLANATION fallback ===")
print(fallback2)
assert BAD_MSG not in fallback2
print("ASSERTION2 PASSED.")

# TURN: user asks "Why did I get this result?"
intent3 = classify_post_assessment_intent("Why did I get this result?", "FOLLOW_UP")
print()
print("Intent3:", intent3)
assert intent3 == "RISK_EXPLANATION", f"WRONG INTENT3: {intent3}"

print()
print("ALL VERIFICATIONS PASSED.")

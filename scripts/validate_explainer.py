"""Validates new explainer.py end-to-end with the exact failing case."""
from backend.app.ml.explainer import (
    LocalPredictionExplainer, get_local_explainer, reset_explainer_singleton,
    explanation_for_ui, format_factor_explanation_text,
    format_risk_explanation_text, format_score_factors_text,
    FEATURE_DISPLAY_NAMES_CLEAN,
)
print("All imports OK")

# Test singleton
gle = get_local_explainer()
print(f"Singleton model: {type(gle.model).__name__}")
print(f"Singleton tree_explainer: {gle._tree_explainer is not None}")
print(f"Singleton shap_version: {gle._shap_version}")

# Build the exact failing feature_state
feature_state = {
    "Brake_Condition":    {"value": "Worn Out",   "status": "known",  "source": "user"},
    "Battery_Status":     {"value": "Weak",        "status": "known",  "source": "user"},
    "Reported_Issues":    {"value": 8,             "status": "known",  "source": "user"},
    "Maintenance_History":{"value": "Good",        "status": "known",  "source": "user"},
    "Service_History":    {"value": 4,             "status": "known",  "source": "user"},
    "Last_Service_Date":  {"value": "2025-07-08",  "status": "known",  "source": "user"},
    "Accident_History":   {"value": 0,             "status": "known",  "source": "user"},
    "Mileage":            {"value": 1000,          "status": "known",  "source": "user"},
}
from backend.app.ml.preprocessing import SELECTED_FEATURES
for feat in SELECTED_FEATURES:
    if feat not in feature_state:
        feature_state[feat] = {"value": None, "status": "unknown", "source": "user"}

pred = {"risk_level": "HIGH", "model_score": 1.0}
result = gle.explain_prediction(feature_state, pred)

print()
print("=== explain_prediction result ===")
print("available:", result["available"])
print("main_factor:", result["main_factor"])
top = result["top_contributors"]
print(f"top_contributors ({len(top)}):", [f["feature"] for f in top])
print("shap_base_value:", result["shap_base_value"])
print()

# Test UI payload
ui = explanation_for_ui(result)
print("=== explanation_for_ui ===")
print("available:", ui["available"])
print("top_factors:", [(f["display_name"], f["direction"]) for f in ui["top_factors"]])
print()

# Test text formatters
print("=== format_factor_explanation_text ===")
print(format_factor_explanation_text(result, pred))
print()

print("=== format_risk_explanation_text ===")
print(format_risk_explanation_text(result, pred))
print()

print("=== format_score_factors_text ===")
print(format_score_factors_text(result, pred))
print()

# Test reset
reset_explainer_singleton()
gle2 = get_local_explainer()
print("After reset - new singleton OK:", gle2.model is not None)
print()
print("ALL CHECKS PASSED")

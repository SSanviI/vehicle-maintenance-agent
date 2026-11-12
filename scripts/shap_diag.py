"""
SHAP Diagnostic Script — run from project root.
Reproduces the exact failing case and surfaces the real SHAP error.
"""

import sys
import traceback

print("=== SHAP DIAGNOSTIC SCRIPT ===")
print()

# Step 1: Load model and preprocessor
import joblib
from pathlib import Path

model_path = Path("models/maintenance_model.joblib")
pipe_path = Path("models/preprocessing_pipeline.joblib")

print(f"model exists : {model_path.exists()}")
print(f"pipeline exists : {pipe_path.exists()}")

if not (model_path.exists() and pipe_path.exists()):
    print("ERROR: model files missing!")
    sys.exit(1)

model = joblib.load(model_path)
print(f"model type           : {type(model).__name__}")
print(f"model.classes_       : {model.classes_}")
n_in = getattr(model, "n_features_in_", "N/A")
print(f"model.n_features_in_ : {n_in}")
fn_in = list(getattr(model, "feature_names_in_", [])) if hasattr(model, "feature_names_in_") else "N/A"
print(f"model.feature_names_in_ : {fn_in}")

from backend.app.ml.preprocessing import MaintenancePreprocessor, SELECTED_FEATURES
preprocessor = MaintenancePreprocessor.load(str(pipe_path))
print(f"preprocessor.is_fitted : {preprocessor.is_fitted}")
print(f"preprocessor.category_maps keys : {list(preprocessor.category_maps.keys())}")
print()

# Step 2: Exact failing feature_state
feature_state = {
    "Brake_Condition":    {"value": "Worn Out",    "status": "known", "source": "user"},
    "Battery_Status":     {"value": "Weak",         "status": "known", "source": "user"},
    "Reported_Issues":    {"value": 8,              "status": "known", "source": "user"},
    "Maintenance_History":{"value": "Good",         "status": "known", "source": "user"},
    "Service_History":    {"value": 4,              "status": "known", "source": "user"},
    "Last_Service_Date":  {"value": "2025-07-08",   "status": "known", "source": "user"},
    "Accident_History":   {"value": 0,              "status": "known", "source": "user"},
    "Mileage":            {"value": 1000,           "status": "known", "source": "user"},
}
for feat in SELECTED_FEATURES:
    if feat not in feature_state:
        feature_state[feat] = {"value": None, "status": "unknown", "source": "user"}

known = [k for k, v in feature_state.items() if v["status"] == "known"]
print(f"Known features ({len(known)}): {known}")
print()

# Step 3: Transform
print("=== STEP 3: TRANSFORM ===")
X = preprocessor.transform_feature_state(feature_state)
print(f"X shape     : {X.shape}")
print(f"X columns   : {list(X.columns)}")
print()

# Step 4: Prediction
print("=== STEP 4: PREDICTION ===")
raw_probs = model.predict_proba(X)[0]
pred_class = int(model.predict(X)[0])
print(f"predict_proba : {raw_probs}")
print(f"pred_class    : {pred_class}")
maint_idx = list(model.classes_).index(1) if 1 in list(model.classes_) else 1
print(f"Maintenance class idx : {maint_idx}")
print(f"Maintenance prob      : {raw_probs[maint_idx]:.4f}")
print()

# Step 5: SHAP
print("=== STEP 5: SHAP ===")
try:
    import shap
    print(f"SHAP version : {shap.__version__}")

    print("Creating TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    ev = explainer.expected_value
    print(f"expected_value type  : {type(ev)}")
    print(f"expected_value value : {ev}")
    print()

    print("Calling shap_values(X) ...")
    sv = explainer.shap_values(X)
    print(f"shap_values type  : {type(sv)}")

    if isinstance(sv, list):
        print(f"shap_values is LIST, len={len(sv)}")
        for i, arr in enumerate(sv):
            print(f"  sv[{i}] shape : {arr.shape}")
    elif hasattr(sv, "shape"):
        print(f"shap_values shape : {sv.shape}")
    print()

    # Parse class-1 shap values
    if isinstance(sv, list) and len(sv) > 1:
        shap_class1 = sv[1][0]
        method = "sv[1][0]"
    elif isinstance(sv, list) and len(sv) == 1:
        shap_class1 = sv[0][0]
        method = "sv[0][0]"
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        shap_class1 = sv[0, :, 1]
        method = "sv[0,:,1]"
    elif hasattr(sv, "ndim") and sv.ndim == 2:
        shap_class1 = sv[0, :]
        method = "sv[0,:]"
    else:
        print(f"UNKNOWN SHAP STRUCTURE: {type(sv)}")
        sys.exit(1)

    print(f"Using method       : {method}")
    print(f"shap_class1 shape  : {shap_class1.shape}")
    print(f"len(X.columns)     : {len(X.columns)}")

    if len(shap_class1) != len(X.columns):
        print(f"COLUMN MISMATCH: shap len={len(shap_class1)} vs X.columns len={len(X.columns)}")

    print()
    print("Top-10 feature contributions (|SHAP| descending):")
    pairs = sorted(zip(X.columns, shap_class1), key=lambda p: abs(p[1]), reverse=True)
    for col, val in pairs[:10]:
        direction = "higher_risk" if val > 0 else "lower_risk"
        print(f"  {col:40s}  shap={val:+.6f}  ({direction})")

    # Map back to base features (strip _missing suffix)
    feat_contribs = {}
    for col, val in zip(X.columns, shap_class1):
        base = col[:-8] if col.endswith("_missing") else col
        feat_contribs[base] = feat_contribs.get(base, 0.0) + float(val)

    # Filter to known features
    known_contribs = {
        k: v for k, v in feat_contribs.items()
        if k in feature_state and feature_state[k].get("status") == "known"
    }
    print()
    print("Known-feature contributions (|SHAP| descending):")
    for k, v in sorted(known_contribs.items(), key=lambda p: abs(p[1]), reverse=True):
        print(f"  {k:40s}  shap={v:+.6f}  ({'higher_risk' if v > 0 else 'lower_risk'})")

    print()
    if isinstance(ev, (list, type(None))):
        base_val = float(ev[1]) if ev is not None and len(ev) > 1 else None
    else:
        base_val = float(ev)
    print(f"base_value (class 1) : {base_val}")
    print()
    print("SHAP DIAGNOSTIC: SUCCESS")

except Exception as e:
    print(f"SHAP EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc()
    print()
    print("SHAP DIAGNOSTIC: FAILED")

# Step 6: Test the full LocalPredictionExplainer
print()
print("=== STEP 6: LocalPredictionExplainer ===")
try:
    from backend.app.ml.explainer import LocalPredictionExplainer
    exp = LocalPredictionExplainer(model, preprocessor)
    print(f"_tree_explainer : {exp._tree_explainer}")
    result = exp.explain_prediction(feature_state, {"risk_level": "HIGH", "model_score": raw_probs[maint_idx]})
    print(f"available        : {result.get('available')}")
    print(f"reason           : {result.get('reason')}")
    print(f"main_factor      : {result.get('main_factor')}")
    top = result.get("top_contributors", [])
    print(f"top_contributors ({len(top)}):")
    for f in top[:5]:
        print(f"  {f}")
except Exception as e:
    print(f"LocalPredictionExplainer EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc()

# Step 7: Test get_local_explainer singleton
print()
print("=== STEP 7: get_local_explainer singleton ===")
try:
    from backend.app.ml.explainer import get_local_explainer, _global_explainer
    print(f"_global_explainer before call : {_global_explainer}")
    gle = get_local_explainer()
    print(f"_global_explainer after call  : {gle}")
    print(f"gle.model        : {type(gle.model).__name__ if gle.model else None}")
    print(f"gle.preprocessor : {type(gle.preprocessor).__name__ if gle.preprocessor else None}")
    print(f"gle._tree_explainer : {gle._tree_explainer}")
    result2 = gle.explain_prediction(feature_state, {"risk_level": "HIGH", "model_score": float(raw_probs[maint_idx])})
    print(f"Singleton explain available : {result2.get('available')}")
    print(f"Singleton explain reason    : {result2.get('reason')}")
    print(f"Singleton main_factor       : {result2.get('main_factor')}")
except Exception as e:
    print(f"get_local_explainer EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("=== DIAGNOSTIC COMPLETE ===")

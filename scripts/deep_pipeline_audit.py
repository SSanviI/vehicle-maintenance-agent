"""
Deep Audit Script for Vehicle Maintenance ML Pipeline
Checks:
1. Target label semantics & class ordering
2. Model classes_ & predict_proba mapping
3. Training vs Inference feature ordering & categorical encodings
4. Preprocessing transformation on sample feature vectors
5. Controlled feature sensitivity test
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/vehicle_maintenance_data.csv")
MODEL_PATH = Path("models/maintenance_model.joblib")
PIPELINE_PATH = Path("models/preprocessing_pipeline.joblib")
META_PATH = Path("models/model_metadata.json")

print("=" * 80)
print("  STEP 1: AUDIT TARGET LABEL & DATASET")
print("=" * 80)
df = pd.read_csv(DATA_PATH)
target_col = "Need_Maintenance"
print(f"Target column: {target_col}")
print(f"Dtype: {df[target_col].dtype}")
print(f"Unique values: {df[target_col].unique().tolist()}")
print(f"Value counts:\n{df[target_col].value_counts()}")
print(f"Target rate (mean): {df[target_col].mean():.4f}")

print("\n" + "=" * 80)
print("  STEP 2: AUDIT SAVED MODEL & PREPROCESSOR ARTIFACTS")
print("=" * 80)
model = joblib.load(MODEL_PATH)
pipeline_data = joblib.load(PIPELINE_PATH)

print(f"Model object type: {type(model)}")
if hasattr(model, "classes_"):
    print(f"model.classes_: {model.classes_}")
    print(f"model.classes_ type: {type(model.classes_)}, dtype: {model.classes_.dtype if hasattr(model.classes_, 'dtype') else 'N/A'}")
else:
    print("model has NO classes_ attribute!")

if META_PATH.exists():
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"Model metadata selected_features: {meta.get('selected_features')}")
    print(f"Model metadata categorical_values: {meta.get('categorical_values')}")

print(f"\nPipeline data keys: {list(pipeline_data.keys())}")
print(f"Pipeline category_maps: {pipeline_data.get('category_maps')}")
print(f"Pipeline selected_features: {pipeline_data.get('selected_features')}")

print("\n" + "=" * 80)
print("  STEP 3: AUDIT FEATURE SCHEMAS & ORDERING")
print("=" * 80)

from backend.app.ml.preprocessing import (
    SELECTED_FEATURES as BACKEND_SELECTED,
    CATEGORICAL_FEATURES as BACKEND_CATEGORICAL,
    NUMERICAL_FEATURES as BACKEND_NUMERICAL,
    CATEGORICAL_VALUES as BACKEND_CAT_VALUES,
    MaintenancePreprocessor
)

print(f"Backend SELECTED_FEATURES: {BACKEND_SELECTED}")
print(f"Pipeline selected_features: {pipeline_data.get('selected_features')}")
features_match = (BACKEND_SELECTED == pipeline_data.get('selected_features'))
print(f"Feature ordering & names match exactly? {features_match}")

print("\n" + "=" * 80)
print("  STEP 4: AUDIT PREPROCESSING & ENCODING ON SAMPLE VECTOR")
print("=" * 80)

preprocessor = MaintenancePreprocessor.load(str(PIPELINE_PATH))

# Test sample feature state matching Problem 3 scenario:
# Turn 1: Age 10
state1 = {"Vehicle_Age": {"value": 10, "status": "known"}}
X1 = preprocessor.transform_feature_state(state1)
prob1 = model.predict_proba(X1)[0]
print(f"State 1 (Age=10):")
print(f"  Processed dataframe cols: {list(X1.columns)}")
print(f"  Processed vector values: {X1.values.tolist()[0]}")
print(f"  predict_proba raw: {prob1}")
print(f"  predict_proba[1] (currently used as model_score): {prob1[1]:.4f}")

# Turn 6: Age=10, Brakes=Good, Battery=New, Maint=Poor, Service=5, Accident=5, Tire=Worn Out
state6 = {
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
    "Battery_Status": {"value": "New", "status": "known"},
    "Maintenance_History": {"value": "Poor", "status": "known"},
    "Service_History": {"value": 5, "status": "known"},
    "Accident_History": {"value": 5, "status": "known"},
    "Tire_Condition": {"value": "Worn Out", "status": "known"},
}
X6 = preprocessor.transform_feature_state(state6)
prob6 = model.predict_proba(X6)[0]
print(f"\nState 6 (All bad conditions + Age=10, Service=5, Accident=5):")
print(f"  Processed vector values: {X6.values.tolist()[0]}")
print(f"  predict_proba raw: {prob6}")
print(f"  predict_proba[1] (currently used as model_score): {prob6[1]:.4f}")

print("\n" + "=" * 80)
print("  STEP 5: SENSITIVITY TEST ON EACH FEATURE")
print("=" * 80)

baseline_state = {
    "Vehicle_Age": {"value": 5, "status": "known"},
    "Reported_Issues": {"value": 0, "status": "known"},
    "Maintenance_History": {"value": "Good", "status": "known"},
    "Service_History": {"value": 5, "status": "known"},
    "Accident_History": {"value": 0, "status": "known"},
    "Tire_Condition": {"value": "Good", "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
    "Battery_Status": {"value": "Good", "status": "known"},
}
X_base = preprocessor.transform_feature_state(baseline_state)
base_score = model.predict_proba(X_base)[0, 1]
print(f"Baseline (all good/moderate): prob[1] = {base_score:.4f}")

# Single feature variations:
variations = [
    ("Brake_Condition = Worn Out", {"Brake_Condition": {"value": "Worn Out", "status": "known"}}),
    ("Brake_Condition = New", {"Brake_Condition": {"value": "New", "status": "known"}}),
    ("Battery_Status = Weak", {"Battery_Status": {"value": "Weak", "status": "known"}}),
    ("Battery_Status = New", {"Battery_Status": {"value": "New", "status": "known"}}),
    ("Maintenance_History = Poor", {"Maintenance_History": {"value": "Poor", "status": "known"}}),
    ("Maintenance_History = Good", {"Maintenance_History": {"value": "Good", "status": "known"}}),
    ("Tire_Condition = Worn Out", {"Tire_Condition": {"value": "Worn Out", "status": "known"}}),
    ("Reported_Issues = 5", {"Reported_Issues": {"value": 5, "status": "known"}}),
    ("Accident_History = 3", {"Accident_History": {"value": 3, "status": "known"}}),
]

for label, var_dict in variations:
    st = baseline_state.copy()
    st.update(var_dict)
    X = preprocessor.transform_feature_state(st)
    p = model.predict_proba(X)[0]
    print(f"  Variation ({label:<30s}): prob[0]={p[0]:.4f}, prob[1]={p[1]:.4f}, pred={model.predict(X)[0]}")

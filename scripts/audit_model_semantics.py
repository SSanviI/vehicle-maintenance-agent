"""
Audit model classes, predict_proba, and score semantics on existing model.
"""

import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/maintenance_model.joblib")
pipeline = joblib.load("models/preprocessing_pipeline.joblib")

print("=" * 80)
print("  MODEL AND PREPROCESSED ARTIFACT AUDIT")
print("=" * 80)

print(f"Model type: {type(model)}")
print(f"model.classes_: {getattr(model, 'classes_', 'NO CLASSES')}")
print(f"pipeline keys: {list(pipeline.keys())}")
print(f"pipeline category_maps: {pipeline.get('category_maps')}")
print(f"pipeline selected_features: {pipeline.get('selected_features')}")

from backend.app.ml.preprocessing import MaintenancePreprocessor
preprocessor = MaintenancePreprocessor.load("models/preprocessing_pipeline.joblib")

# Test baseline vector
sample_state = {
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Reported_Issues": {"value": 1, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
}

X = preprocessor.transform_feature_state(sample_state)
pred_class = model.predict(X)[0]
probs = model.predict_proba(X)[0]

print("\nSample prediction:")
print(f"  Feature state: {sample_state}")
print(f"  Processed features: {X.values.tolist()[0]}")
print(f"  model.predict(X)[0]: {pred_class}")
print(f"  model.predict_proba(X)[0]: {probs}")

# Index 0 vs Index 1 meaning:
# Class 0: no_maintenance_needed
# Class 1: maintenance_required
print(f"  P(Class 0 - No Maintenance): {probs[0]:.4f}")
print(f"  P(Class 1 - Maintenance Required): {probs[1]:.4f}")

# Check if model score returned in backend is probs[1] or probs[0]
print(f"  Current model_score used in predict.py: {probs[1]:.4f}")

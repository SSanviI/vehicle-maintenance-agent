"""
Script to test Monotonic Constraints in LightGBM and XGBoost training
to ensure higher wear/accidents/issues NEVER decrease risk.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from backend.app.ml.preprocessing import MaintenancePreprocessor

df = pd.read_csv("data/vehicle_maintenance_data.csv")

# Let's inspect category ordering for ordinal monotonicity:
# Maintenance_History: Good(0) < Average(1) < Poor(2) < UNKNOWN(3)
# Tire_Condition: New(0) < Good(1) < Worn Out(2) < UNKNOWN(3)
# Brake_Condition: New(0) < Good(1) < Worn Out(2) < UNKNOWN(3)
# Battery_Status: New(0) < Good(1) < Weak(2) < UNKNOWN(3)

preprocessor = MaintenancePreprocessor.from_config()
X_transformed = preprocessor._transform(df)
y = df["Need_Maintenance"]

print("Transformed columns:", list(X_transformed.columns))

# Let's check monotonicity behavior on LightGBM with monotone_constraints
# Feature order in X_transformed:
# 0: Maintenance_History (+1)
# 1: Maintenance_History_missing
# 2: Tire_Condition (+1)
# 3: Tire_Condition_missing
# 4: Brake_Condition (+1)
# 5: Brake_Condition_missing
# 6: Battery_Status (+1)
# 7: Battery_Status_missing
# 8: Vehicle_Age
# 9: Vehicle_Age_missing
# 10: Reported_Issues (+1)
# 11: Reported_Issues_missing
# 12: Service_History
# 13: Service_History_missing
# 14: Accident_History (+1)
# 15: Accident_History_missing

mc = [
    1, 0,  # Maintenance_History, missing
    1, 0,  # Tire_Condition, missing
    1, 0,  # Brake_Condition, missing
    1, 0,  # Battery_Status, missing
    0, 0,  # Vehicle_Age, missing
    1, 0,  # Reported_Issues, missing
    0, 0,  # Service_History, missing
    1, 0,  # Accident_History, missing
]

lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    monotone_constraints=mc,
    random_state=42,
    verbose=-1
)
lgbm.fit(X_transformed, y)

print("\n--- Testing Monotonic LightGBM Model on Problem 3 scenario ---")

# Step 1: Age 10, all else missing/UNKNOWN
s1 = preprocessor.transform_feature_state({
    "Vehicle_Age": {"value": 10, "status": "known"}
})
p1 = lgbm.predict_proba(s1)[0, 1]

# Step 2: Brakes Good
s2 = preprocessor.transform_feature_state({
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"}
})
p2 = lgbm.predict_proba(s2)[0, 1]

# Step 3: Battery New
s3 = preprocessor.transform_feature_state({
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
    "Battery_Status": {"value": "New", "status": "known"}
})
p3 = lgbm.predict_proba(s3)[0, 1]

# Step 4: Maintenance Poor
s4 = preprocessor.transform_feature_state({
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
    "Battery_Status": {"value": "New", "status": "known"},
    "Maintenance_History": {"value": "Poor", "status": "known"}
})
p4 = lgbm.predict_proba(s4)[0, 1]

# Step 5: Service 5
s5 = preprocessor.transform_feature_state({
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
    "Battery_Status": {"value": "New", "status": "known"},
    "Maintenance_History": {"value": "Poor", "status": "known"},
    "Service_History": {"value": 5, "status": "known"}
})
p5 = lgbm.predict_proba(s5)[0, 1]

# Step 6: Accident 5
s6 = preprocessor.transform_feature_state({
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
    "Battery_Status": {"value": "New", "status": "known"},
    "Maintenance_History": {"value": "Poor", "status": "known"},
    "Service_History": {"value": 5, "status": "known"},
    "Accident_History": {"value": 5, "status": "known"}
})
p6 = lgbm.predict_proba(s6)[0, 1]

# Step 7: Tire Worn Out
s7 = preprocessor.transform_feature_state({
    "Vehicle_Age": {"value": 10, "status": "known"},
    "Brake_Condition": {"value": "Good", "status": "known"},
    "Battery_Status": {"value": "New", "status": "known"},
    "Maintenance_History": {"value": "Poor", "status": "known"},
    "Service_History": {"value": 5, "status": "known"},
    "Accident_History": {"value": 5, "status": "known"},
    "Tire_Condition": {"value": "Worn Out", "status": "known"}
})
p7 = lgbm.predict_proba(s7)[0, 1]

print(f"Step 1 (Age 10): {p1:.4f}")
print(f"Step 2 (Brakes Good): {p2:.4f}")
print(f"Step 3 (Battery New): {p3:.4f}")
print(f"Step 4 (+ Maintenance Poor): {p4:.4f}")
print(f"Step 5 (+ Service 5): {p5:.4f}")
print(f"Step 6 (+ Accident 5): {p6:.4f}")
print(f"Step 7 (+ Tire Worn Out): {p7:.4f}")

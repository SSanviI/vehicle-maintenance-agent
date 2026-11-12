"""
Test Smoothed ML Training (Addressing Synthetic Target Leakage)
===============================================================
Applies label smoothing / realistic stochasticity during training so that
deterministic synthetic leaks (Worn Out brakes = 100%) don't drive log-odds to infinity,
allowing probabilities to evolve smoothly across all 7 steps.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.ml.preprocessing import MaintenancePreprocessor, SELECTED_FEATURES

DATA_PATH = root / "data" / "vehicle_maintenance_data.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[SELECTED_FEATURES].copy()
    y = df["Need_Maintenance"].copy()

    # Train / Val / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Preprocessor
    preprocessor = MaintenancePreprocessor.from_config()

    # Add missingness augmentation
    from scripts.train_model import apply_missingness_augmentation
    X_train_aug_list = [preprocessor._transform(X_train)]
    y_train_aug_list = [y_train.copy()]

    for i in range(3):
        X_aug = apply_missingness_augmentation(X_train, missing_rate=0.3, seed=42 + i)
        X_train_aug_list.append(preprocessor._transform(X_aug))
        y_train_aug_list.append(y_train.copy())

    X_train_combined = pd.concat(X_train_aug_list, ignore_index=True)
    y_train_combined = pd.concat(y_train_aug_list, ignore_index=True)

    # Label Smoothing on Training Set to prevent infinite log-odds on synthetic 100% leaks
    # y_smooth maps 0 -> 0.05, 1 -> 0.95
    y_train_smooth = y_train_combined * 0.90 + 0.05

    print("Training XGBoost Regressor/Classifier on Smoothed Targets...")
    from xgboost import XGBRegressor
    xgb_smooth = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_smooth.fit(X_train_combined, y_train_smooth)

    conv_steps = [
        ("Step 1: Vehicle_Age=10, Brake_Condition=Worn Out", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
        }),
        ("Step 2: + Reported_Issues=4", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
        }),
        ("Step 3: + Battery_Status=Good", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
        }),
        ("Step 4: + Maintenance_History=Good (well maintained)", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
            "Maintenance_History": {"value": "Good", "status": "known"},
        }),
        ("Step 5: + Service_History=20", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
            "Maintenance_History": {"value": "Good", "status": "known"},
            "Service_History": {"value": 20, "status": "known"},
        }),
        ("Step 6: + Accident_History=0", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
            "Maintenance_History": {"value": "Good", "status": "known"},
            "Service_History": {"value": 20, "status": "known"},
            "Accident_History": {"value": 0, "status": "known"},
        }),
        ("Step 7: + Tire_Condition=New", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
            "Maintenance_History": {"value": "Good", "status": "known"},
            "Service_History": {"value": 20, "status": "known"},
            "Accident_History": {"value": 0, "status": "known"},
            "Tire_Condition": {"value": "New", "status": "known"},
        }),
    ]

    print("\n" + "=" * 80)
    print("  USER CONVERSATION WITH SMOOTHED MODEL")
    print("=" * 80)
    for title, f_state in conv_steps:
        X_vec = preprocessor.transform_feature_state(f_state)
        prob = float(xgb_smooth.predict(X_vec)[0])
        print(f"{title:<65} | Smoothed Prob: {prob:.6f}")

    print("\n" + "=" * 80)
    print("  CONTROLLED CASES A, B, C, D WITH SMOOTHED MODEL")
    print("=" * 80)
    cases = {
        "A (Age=10, Issues=4, Brake=Worn Out, Battery=Good)": {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
        },
        "B (Age=10, Issues=4, Brake=Worn Out, Battery=Weak)": {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Battery_Status": {"value": "Weak", "status": "known"},
        },
        "C (Age=10, Issues=0, Brake=Good, Battery=Good)": {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Reported_Issues": {"value": 0, "status": "known"},
            "Brake_Condition": {"value": "Good", "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
        },
        "D (Age=2, Issues=0, Brake=Good, Battery=Good)": {
            "Vehicle_Age": {"value": 2, "status": "known"},
            "Reported_Issues": {"value": 0, "status": "known"},
            "Brake_Condition": {"value": "Good", "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
        },
    }
    for c_name, f_state in cases.items():
        X_vec = preprocessor.transform_feature_state(f_state)
        prob = float(xgb_smooth.predict(X_vec)[0])
        print(f"{c_name:<55} | Smoothed Prob: {prob:.6f}")

if __name__ == "__main__":
    main()

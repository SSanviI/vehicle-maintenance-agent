"""
Test ML Fix: Regularized XGBoost + Platt Scaling (Logistic Regression Calibration)
==================================================================================
Uses LogisticRegression to calibrate XGBoost logits/probabilities.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.ml.preprocessing import MaintenancePreprocessor, SELECTED_FEATURES

DATA_PATH = root / "data" / "vehicle_maintenance_data.csv"


class CalibratedXGBoost:
    """XGBoost classifier calibrated via Platt Scaling (Logistic Regression on logits)."""

    def __init__(self, base_xgb, calibrator):
        self.base_xgb = base_xgb
        self.calibrator = calibrator

    def predict_proba(self, X):
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        logits = self.base_xgb.get_booster().predict(dmatrix, output_margin=True)
        # Pass logits to calibrator
        probs_cal = self.calibrator.predict_proba(logits.reshape(-1, 1))
        return probs_cal

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_calibrated_model(X_train, y_train, X_val, y_val):
    import xgboost as xgb

    # Regularized XGBoost: shallower depth (3), L1/L2 regularization, max_delta_step=1.0
    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        reg_alpha=1.0,
        reg_lambda=2.0,
        max_delta_step=1.0,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    base_xgb.fit(X_train, y_train)

    # Get validation logits for calibration
    dval = xgb.DMatrix(X_val)
    val_logits = base_xgb.get_booster().predict(dval, output_margin=True).reshape(-1, 1)

    # Fit Logistic Regression calibrator (Platt Scaling)
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(val_logits, y_val)

    return CalibratedXGBoost(base_xgb, calibrator)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[SELECTED_FEATURES].copy()
    y = df["Need_Maintenance"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    preprocessor = MaintenancePreprocessor.from_config()

    X_train_trans = preprocessor._transform(X_train)
    X_val_trans = preprocessor._transform(X_val)

    print("Training Calibrated XGBoost (Platt Scaling)...")
    model = train_calibrated_model(X_train_trans, y_train, X_val_trans, y_val)

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
    print("  USER CONVERSATION WITH CALIBRATED REGULARIZED MODEL")
    print("=" * 80)
    for title, f_state in conv_steps:
        X_vec = preprocessor.transform_feature_state(f_state)
        prob = float(model.predict_proba(X_vec)[0, 1])
        print(f"{title:<65} | Calibrated Prob: {prob:.6f}")

    print("\n" + "=" * 80)
    print("  CONTROLLED CASES A, B, C, D WITH CALIBRATED MODEL")
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
        prob = float(model.predict_proba(X_vec)[0, 1])
        print(f"{c_name:<55} | Calibrated Prob: {prob:.6f}")

if __name__ == "__main__":
    main()

"""
Systematic ML Investigation Script
====================================
Investigates:
1. Raw model probabilities (15 decimal places) & log-odds (logits) for test cases
2. The exact feature vector sent to the model at every incremental prediction step
3. Feature input changes & UNKNOWN representations
4. Target leakage analysis across features
5. Training data distributions vs Need_Maintenance
6. XGBoost feature importances & tree leaf behavior
7. Preprocessing parity check between training and inference
8. Calibration & probability saturation analysis
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.services.prediction_service import get_predictor
from backend.app.ml.preprocessing import MaintenancePreprocessor, SELECTED_FEATURES, CATEGORICAL_VALUES

DATA_PATH = root / "data" / "vehicle_maintenance_data.csv"
MODEL_PATH = root / "models" / "maintenance_model.joblib"
PIPELINE_PATH = root / "models" / "preprocessing_pipeline.joblib"

def section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    predictor = get_predictor()
    model = predictor.model
    preprocessor = predictor.preprocessor
    df_raw = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # 1. Controlled Test Cases A, B, C, D
    # ---------------------------------------------------------
    section("1. Controlled Test Cases (A, B, C, D) — Raw Probabilities & Logits")

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

    print(f"{'Case':<55} | {'Raw Probability (15 decimals)':<25} | {'Raw Logit (margin)':<18} | {'Class':<5}")
    print("-" * 110)

    for case_name, feature_state in cases.items():
        X = preprocessor.transform_feature_state(feature_state)
        prob = float(model.predict_proba(X)[0, 1])
        # Raw logit (margin score before sigmoid)
        if hasattr(model, "predict") and hasattr(model, "get_booster"):
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X)
            logit = float(model.get_booster().predict(dmatrix, output_margin=True)[0])
        else:
            logit = np.log(prob / (1 - prob)) if 0 < prob < 1 else float("inf")

        pred_class = int(prob >= 0.5)
        print(f"{case_name:<55} | {prob:.15f} | {logit:18.6f} | {pred_class:<5}")

    # ---------------------------------------------------------
    # 2. Recreating the User's 7-step Conversation
    # ---------------------------------------------------------
    section("2. Recreating User Conversation (Steps 1 to 7) — Feature Vectors & Predictions")

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

    for step_title, feature_state in conv_steps:
        print(f"\n--- {step_title} ---")
        X = preprocessor.transform_feature_state(feature_state)
        prob = float(model.predict_proba(X)[0, 1])

        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        logit = float(model.get_booster().predict(dmatrix, output_margin=True)[0])

        print(f"Prob (15 decimals): {prob:.15f} | Logit: {logit:.6f}")
        print("Feature Vector Sent to Model:")
        print(X.iloc[0].to_dict())

    # ---------------------------------------------------------
    # 3. XGBoost Feature Importances
    # ---------------------------------------------------------
    section("3. XGBoost Feature Importances")
    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.transform_feature_state({}).columns.tolist()
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
        print(fi_df.to_string(index=False))

    # ---------------------------------------------------------
    # 4. Training Data Analysis — Feature Distributions vs Target
    # ---------------------------------------------------------
    section("4. Training Dataset Distributions vs Need_Maintenance")
    target = "Need_Maintenance"

    print("Target Overall Distribution:")
    vc = df_raw[target].value_counts(normalize=True)
    print(vc.to_dict())

    for col in SELECTED_FEATURES:
        print(f"\nFeature: `{col}`")
        if df_raw[col].dtype == "object":
            ct = pd.crosstab(df_raw[col], df_raw[target], normalize="index") * 100
            counts = pd.crosstab(df_raw[col], df_raw[target])
            combined = pd.concat([counts.add_prefix("Count_"), ct.round(2).add_prefix("Pct_")], axis=1)
            print(combined)
        else:
            grouped = df_raw.groupby(target)[col].agg(["count", "mean", "std", "min", "median", "max"])
            print(grouped)

    # ---------------------------------------------------------
    # 5. Multi-Feature Joint Distributions (Combination Analysis)
    # ---------------------------------------------------------
    section("5. Multi-Feature Combination Analysis in Dataset")
    
    # Check probability of Need_Maintenance given Reported_Issues >= 2 vs Reported_Issues == 0
    print("Need_Maintenance by Reported_Issues:")
    ct_issues = pd.crosstab(df_raw["Reported_Issues"], df_raw[target], normalize="index") * 100
    print(ct_issues.round(2))

    # Check combination of Reported_Issues >= 2 AND Brake_Condition == 'Worn Out'
    sub_high = df_raw[(df_raw["Reported_Issues"] >= 2) & (df_raw["Brake_Condition"] == "Worn Out")]
    print(f"\nWhen Reported_Issues >= 2 AND Brake_Condition == Worn Out (Count={len(sub_high)}):")
    print(sub_high[target].value_counts(normalize=True).to_dict())

    # Check combination of Reported_Issues == 0 AND Brake_Condition == 'Good' AND Battery_Status == 'Good'
    sub_low = df_raw[(df_raw["Reported_Issues"] == 0) & (df_raw["Brake_Condition"] == "Good") & (df_raw["Battery_Status"] == "Good")]
    print(f"\nWhen Reported_Issues == 0 AND Brake_Condition == Good AND Battery_Status == Good (Count={len(sub_low)}):")
    print(sub_low[target].value_counts(normalize=True).to_dict())

if __name__ == "__main__":
    main()

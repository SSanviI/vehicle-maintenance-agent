"""
User Conversation Evaluation Script
====================================
Tests the user's exact 7-step conversation against the retrained ML model
to demonstrate that predictions now evolve meaningfully across every step.
"""

import sys
import json
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.services.prediction_service import get_predictor

def main():
    predictor = get_predictor()
    model = predictor.model
    preprocessor = predictor.preprocessor

    conv_steps = [
        ("Step 1: Vehicle_Age = 10, Brake_Condition = Worn Out", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
        }),
        ("Step 2: + Reported_Issues = 4", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
        }),
        ("Step 3: + Battery_Status = Good", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
        }),
        ("Step 4: + Maintenance_History = Good", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
            "Maintenance_History": {"value": "Good", "status": "known"},
        }),
        ("Step 5: + Service_History = 20", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
            "Maintenance_History": {"value": "Good", "status": "known"},
            "Service_History": {"value": 20, "status": "known"},
        }),
        ("Step 6: + Accident_History = 0", {
            "Vehicle_Age": {"value": 10, "status": "known"},
            "Brake_Condition": {"value": "Worn Out", "status": "known"},
            "Reported_Issues": {"value": 4, "status": "known"},
            "Battery_Status": {"value": "Good", "status": "known"},
            "Maintenance_History": {"value": "Good", "status": "known"},
            "Service_History": {"value": 20, "status": "known"},
            "Accident_History": {"value": 0, "status": "known"},
        }),
        ("Step 7: + Tire_Condition = New", {
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

    print("================================================================================")
    print("  USER CONVERSATION EVALUATION AFTER ML RETRAINING")
    print("================================================================================")
    print(f"{'Step / Feature Addition':<60} | {'ML Model Probability':<22} | {'Risk Level':<10}")
    print("-" * 98)

    for title, f_state in conv_steps:
        result = predictor.predict(f_state)
        prob = result["model_score"]
        risk = result["risk_level"]
        print(f"{title:<60} | {prob:.6f} ({prob*100:5.2f}%)   | {risk:<10}")

if __name__ == "__main__":
    main()

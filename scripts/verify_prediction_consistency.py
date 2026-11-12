"""
Verification script for exact conversation test in Section 11 & 16 of prompt.
Tracks turn, prediction_id, ML call count, model_score, and features_used.
"""

import asyncio
import json
from backend.app.agent.state import AgentState
from backend.app.agent.controller import AgentController
from backend.app.services.prediction_service import get_predictor


async def run_conversation_verification():
    predictor = get_predictor()
    controller = AgentController(predictor)
    state = AgentState(conversation_id="verification-convo-1")

    turns = [
        "My car is 10 years old. Tire is making some noise.",
        "Brake is in good condition",
        "Battery is showing signs of weakness",
        "Well maintained",
        "10 times serviced"
    ]

    print("=" * 100)
    print("  EXACT CONVERSATION PREDICTION CONSISTENCY AUDIT")
    print("=" * 100)
    print(f"{'Turn':<6} | {'Prediction ID':<36} | {'ML Calls':<8} | {'Score':<8} | {'Features Used'}")
    print("-" * 100)

    for turn_idx, turn_text in enumerate(turns, 1):
        prev_calls = predictor._inference_count
        res = await controller.process_message(turn_text, state)
        new_calls = predictor._inference_count
        ml_calls = new_calls - prev_calls

        pred = res.get("prediction")
        if pred:
            pred_id = pred.get("prediction_id", "N/A")
            score = f"{pred.get('model_score'):.4f}"
            feats = ", ".join(pred.get("features_used", []))
        else:
            pred_id = "NO PREDICTION"
            score = "N/A"
            feats = "None"

        print(f"{turn_idx:<6} | {pred_id:<36} | {ml_calls:<8} | {score:<8} | {feats}")

    print("=" * 100)

if __name__ == "__main__":
    asyncio.run(run_conversation_verification())

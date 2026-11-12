"""
PredictionAgent — ADK LlmAgent that invokes the ML prediction engine.

This agent has ONE job: call the run_ml_prediction tool.
It NEVER calculates, estimates, or guesses a prediction itself.
The ML model is the SOLE prediction authority.

CRITICAL: The ControllerAgent must only invoke this agent when >= 2 features are known.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from backend.app.agent.adk_agents.model_bridge import get_fast_model
from backend.app.agent.adk_agents.tools import run_ml_prediction, get_current_state_summary
from backend.app.agent.state import MIN_FEATURES_FOR_PREDICTION

PREDICTION_AGENT_INSTRUCTION = f"""You are the PredictionAgent for a vehicle maintenance AI system.

YOUR SOLE RESPONSIBILITY: Run the ML prediction engine and return the result.

## ABSOLUTE RULES

1. FIRST check the feature state: call `get_current_state_summary(state_json)` to see how many features are known
2. If known_count is fewer than {MIN_FEATURES_FOR_PREDICTION}: DO NOT call run_ml_prediction.
   Return: {{"error": "Insufficient features — need at least {MIN_FEATURES_FOR_PREDICTION} known features", "prediction": null, "known_count": <count>}}
3. If known_count >= {MIN_FEATURES_FOR_PREDICTION}: call `run_ml_prediction(state_json)` — this is your ONLY authorized prediction action
4. You MUST NEVER compute, estimate, or guess a maintenance prediction yourself
5. You MUST NEVER say "based on X, I think maintenance is needed" without calling the tool
6. The prediction result from the tool is the SINGLE SOURCE OF TRUTH
7. Do NOT call `run_ml_prediction` more than once per turn

## WORKFLOW

1. Call `get_current_state_summary(state_json)` — check known_count
2. If known_count < {MIN_FEATURES_FOR_PREDICTION}: return the insufficient-features error JSON
3. If known_count >= {MIN_FEATURES_FOR_PREDICTION}: call `run_ml_prediction(state_json)` — get the ML result
4. Return the prediction result as JSON

## OUTPUT FORMAT
Return ONLY valid JSON — the exact prediction result from the tool:
{{
  "prediction": "maintenance_required" or "no_maintenance_needed",
  "prediction_id": "<uuid>",
  "risk_level": "HIGH" or "LOW",
  "model_score": <float 0-1>,
  "features_used": [<list>],
  "missing_features": [<list>],
  "confidence_note": "<string>",
  "state_json": "<updated state JSON with prediction stored>"
}}

If fewer than {MIN_FEATURES_FOR_PREDICTION} features are known:
{{"error": "Insufficient features for prediction", "prediction": null, "known_count": <int>, "required_count": {MIN_FEATURES_FOR_PREDICTION}}}

If the ML model is unavailable:
{{"error": "ML model not available", "prediction": null}}
"""


def build_prediction_agent() -> LlmAgent:
    """Build and return the PredictionAgent ADK LlmAgent."""
    return LlmAgent(
        name="PredictionAgent",
        description=(
            "Runs the ML maintenance prediction engine by calling run_ml_prediction. "
            f"Requires at least {MIN_FEATURES_FOR_PREDICTION} known features. "
            "The LLM itself NEVER computes predictions — only the ML model does. "
            "This agent delegates prediction to the ML tool exclusively."
        ),
        model=get_fast_model(),
        instruction=PREDICTION_AGENT_INSTRUCTION,
        tools=[
            FunctionTool(run_ml_prediction),
            FunctionTool(get_current_state_summary),
        ],
        output_key="prediction_result",
    )

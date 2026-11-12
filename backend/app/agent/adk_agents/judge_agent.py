"""
JudgeAgent — ADK LlmAgent that evaluates the quality and correctness of agent responses,
plus the deterministic validation guards and fallback response generator.

The Judge is an EVALUATOR ONLY. It:
- Reviews the final response and agent execution trace
- Checks for hallucination, invalid features, repeated unavailable questions
- Returns a structured evaluation score
- Does NOT modify state, predictions, or user-facing output
- Does NOT answer the user

The deterministic guards (`run_deterministic_guards`) are fast Python checks that run
BEFORE the LLM judge. If the guards reject a response, the LLM judge is never called
and a deterministic fallback response is generated instead.

Critical rule: If the Judge fails (due to LLM error or timeout), the conversation
NEVER defaults to pass. A deterministic state-grounded fallback is produced instead.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, List, Tuple

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from backend.app.agent.adk_agents.model_bridge import get_judge_model
from backend.app.agent.adk_agents.tools import get_model_features
from backend.app.ml.preprocessing import (
    SELECTED_FEATURES,
    FEATURE_DISPLAY_NAMES,
    CATEGORICAL_FEATURES,
)
from backend.app.agent.state.state import AgentState, MIN_FEATURES_FOR_PREDICTION
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


JUDGE_AGENT_INSTRUCTION = """You are JudgeAgent — an independent quality and hallucination evaluation specialist.

YOUR SOLE RESPONSIBILITY: Verify that the agent's proposed response is 100% grounded in the provided state.

## EVALUATION CRITERIA:
1. FACTUAL GROUNDING: Does the response ONLY mention vehicle features that are in `known_features`?
2. NO INVENTED FEATURES: Does the response avoid unsupported features (Mileage, Fuel Type, Oil, Suspension)?
3. NO INVENTED PREDICTIONS: If `prediction` is null (known_count < 2), does the response avoid claiming a prediction or risk score?
4. RISK SCORE INTEGRITY: If a prediction exists, does the description match the actual ML risk level (HIGH/LOW)?
5. QUESTION ACCURACY: Is the question asked matching `next_feature_to_ask`? Does it avoid asking known/declined features?
6. QUESTION COUNT: Is at most ONE question asked?
7. LENGTH & TONE: Is the response concise (approx 4–6 lines) and respectful?

## POST-ASSESSMENT CRITICAL CHECKS:
- For a summary request, verify that an actual summary is provided from confirmed state and that it does not repeat the completion prompt.
- For a factor or risk explanation, verify that the response directly answers the question using supplied local explanation data. Reject a response that asks whether the user wants a summary instead.
- Do not accept global feature importance as proof of what caused this individual prediction.
- Do not accept invented contribution percentages, unsupported causal claims, or remaining-useful-life estimates.
- For a capability question, ensure the response states the model boundary rather than inventing a capability.

## SCORING:
- 5: Perfect adherence, fully grounded, correct question, appropriate length.
- 4: Minor stylistic issue but completely grounded and zero hallucinations.
- 3: Acceptable, no hallucinations.
- 2: Minor hallucination (e.g. asked wrong question, mentioned unconfirmed detail).
- 1: Severe hallucination (e.g. invented risk score, invented non-schema feature, contradicted state).

## OUTPUT FORMAT
Return ONLY valid JSON:
{
  "score": <1-5 integer>,
  "passed": <true if score >= 3 else false>,
  "hallucination_detected": <true/false>,
  "issues": ["list of detected issues, if any"],
  "reason": "Concise explanation of score"
}
"""


# ---------------------------------------------------------------------------
# Deterministic Guards (fast Python checks — run BEFORE LLM judge)
# ---------------------------------------------------------------------------

def run_deterministic_guards(
    response_text: str,
    state: AgentState,
    prediction: Optional[Dict[str, Any]],
    expected_next_feature: Optional[str],
    post_assessment_intent: Optional[str] = None,
    local_explanation: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    Perform fast deterministic validation checks before running LLM judge.

    These checks are purely Python-based — no LLM involved.
    Returns (is_valid, list_of_violations).
    If any violation is found, the LLM judge is skipped and a deterministic
    fallback response is produced instead.
    """
    violations: List[str] = []
    if not response_text or not response_text.strip():
        return False, ["Response cannot be blank or empty."]

    text_lower = response_text.lower().strip()
    known = state.get_known_features()
    known_count = state.known_feature_count()

    # A response may repeat the actual model probability, but no other
    # percentage is supported by the local explanation payload.  In particular
    # SHAP contribution values are not user-facing percentages.
    if prediction and prediction.get("model_score") is not None:
        supported_percentage = float(prediction["model_score"]) * 100
        for raw_percentage in re.findall(r"\b(\d+(?:\.\d+)?)\s*%", text_lower):
            if abs(float(raw_percentage) - supported_percentage) > 0.05:
                violations.append("Response contains an unsupported contribution or risk percentage.")
                break

    # 1. Check for commonly hallucinated non-schema automotive terms presented as facts
    forbidden_unsupported = [
        "oil level is", "suspension is", "transmission fluid is", "coolant temperature is"
    ]
    for term in forbidden_unsupported:
        if term in text_lower:
            violations.append(f"Response mentions unsupported automotive concept: '{term}'")

    # 2. Check for fabricated prediction when < 2 features known
    if known_count < MIN_FEATURES_FOR_PREDICTION:
        if "risk score is" in text_lower or "model calculated" in text_lower:
            if not prediction or prediction.get("error"):
                violations.append("Response claims a prediction/risk score when < 2 features are known.")

    # 3. Check for hallucinated Remaining Useful Life (RUL) number claims
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:years?|yrs?|months?)\s+(?:of\s+)?(?:remaining\s+(?:useful\s+)?life|left|remaining\b)", text_lower):
        violations.append("Response invents an unsupported remaining useful life prediction.")

    # 4. Check for unsupported causal claims
    if any(p in text_lower for p in ("caused the failure", "caused vehicle failure", "is the reason for vehicle failure", "is why the vehicle failed")):
        violations.append("Response makes an unsupported causal failure claim.")

    # 5. Post-assessment intent validations
    if post_assessment_intent == "FACTOR_EXPLANATION":
        if "would you like to see a summary" in text_lower or "gives me enough information to complete" in text_lower:
            violations.append("Response asked for summary instead of answering the user's factor explanation question.")
        if expected_next_feature is None and response_text.count("?") > 0:
            violations.append("Factor explanation response must not ask assessment feature questions.")
        main_factor = (local_explanation or {}).get("main_factor")
        if main_factor:
            feat_raw = str(main_factor.get("feature", "")).lower().replace("_", " ")
            disp_name = str(main_factor.get("display_name", "")).lower()
            aliases = {feat_raw, disp_name}
            if "report" in feat_raw or "issue" in feat_raw:
                aliases.update(["reported issue", "reported issues", "issues", "number of reported issues", "problems", "problem"])
            if "age" in feat_raw:
                aliases.update(["vehicle age", "age in years", "car age", "years old", "age"])
            if "brake" in feat_raw:
                aliases.update(["brake condition", "brakes", "brake", "worn out brakes", "brake wear"])
            if "battery" in feat_raw:
                aliases.update(["battery status", "battery condition", "battery", "weak battery"])
            if "maintenance" in feat_raw:
                aliases.update(["maintenance history", "maintenance", "service history"])
            if "service" in feat_raw:
                aliases.update(["last service", "service date", "service history", "service visits", "last service date"])
            if "accident" in feat_raw:
                aliases.update(["accident history", "accidents", "past accidents"])
            if "tire" in feat_raw:
                aliases.update(["tire condition", "tires", "tire"])
            if "mileage" in feat_raw:
                aliases.update(["mileage", "miles", "odometer"])
            if "engine" in feat_raw:
                aliases.update(["engine size", "engine", "cc"])

            if not any(alias in text_lower for alias in aliases):
                violations.append(f"Factor explanation does not identify the strongest supported local factor ({disp_name}).")

    elif post_assessment_intent == "RISK_EXPLANATION":
        if "would you like to see a summary" in text_lower or "gives me enough information to complete" in text_lower:
            violations.append("Response repeated the completion prompt instead of explaining the risk.")

    elif post_assessment_intent == "SUMMARY_REQUEST" or state.assessment_mode == "SUMMARY":
        if "would you like to see a summary" in text_lower:
            violations.append("Response in SUMMARY mode repeats completion prompt instead of delivering summary.")
        if expected_next_feature is None and response_text.count("?") > 1:
            violations.append("Summary response must not ask assessment feature questions.")
        # This deliberately makes the deterministic renderer the safe fallback
        # whenever an LLM omits or changes a confirmed value (especially dates).
        for feature, value in known.items():
            display = FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ")).lower()
            if display not in text_lower or str(value).lower() not in text_lower:
                violations.append(f"Summary omits or changes confirmed value for '{feature}'.")
        if prediction:
            risk = str(prediction.get("risk_level", "")).lower()
            score = prediction.get("model_score")
            if risk and risk not in text_lower:
                violations.append("Summary does not use the latest risk level.")
            if score is not None and f"{float(score) * 100:.1f}" not in text_lower:
                violations.append("Summary does not use the latest model probability.")

    elif post_assessment_intent == "RECOMMENDATION":
        if "would you like to see a summary" in text_lower or "gives me enough information to complete" in text_lower:
            violations.append("Recommendation response repeated the completion prompt.")
        higher = (local_explanation or {}).get("positive_contributors", [])
        if higher and higher[0].get("display_name", "").lower() not in text_lower:
            violations.append("Recommendation does not use the strongest supported higher-risk contributor.")

    elif state.assessment_complete:
        if expected_next_feature is None:
            if state.assessment_mode != "AWAITING_SUMMARY_CONFIRMATION" and response_text.count("?") > 0:
                violations.append("Completed assessment response must not ask another assessment question.")

    elif expected_next_feature and not post_assessment_intent:
        question_count = response_text.count("?")
        if question_count != 1:
            violations.append(
                f"Response must contain exactly one question for selected feature: '{expected_next_feature}'."
            )

    # 6. Check for repeated question about known feature
    for known_feat in known.keys():
        display = FEATURE_DISPLAY_NAMES.get(known_feat, known_feat).lower()
        if f"what is your {display}" in text_lower or f"how is your {display}" in text_lower or f"tell me your {display}" in text_lower:
            violations.append(f"Response asks about already known feature: '{known_feat}'")

    # 7. Check for repeated question about declined feature
    for declined_feat in state.declined_features:
        display = FEATURE_DISPLAY_NAMES.get(declined_feat, declined_feat).lower()
        if f"what is your {display}" in text_lower or f"how is your {display}" in text_lower or f"tell me your {display}" in text_lower:
            violations.append(f"Response asks about declined/unavailable feature: '{declined_feat}'")

    is_valid = len(violations) == 0
    return is_valid, violations


# ---------------------------------------------------------------------------
# Deterministic Fallback Response Generator
# ---------------------------------------------------------------------------

def generate_deterministic_fallback_response(
    state: AgentState,
    prediction: Optional[Dict[str, Any]],
    next_feature: Optional[str],
    post_assessment_intent: Optional[str] = None,
    local_explanation: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a 100% grounded, deterministic fallback response from confirmed state.

    Used when:
    - The LLM response is rejected by deterministic guards
    - The JudgeAgent fails or times out
    - The ResponseAgent returns an empty response

    This function NEVER invents features, scores, or predictions. All output
    is derived strictly from the confirmed AgentState and ML result.
    """
    from backend.app.ml.explainer import (
        format_factor_explanation_text,
        format_risk_explanation_text,
        format_score_factors_text,
    )

    known = state.get_known_features()
    known_count = len(known)

    risk = "LOW"
    score_pct = ""
    if prediction and not prediction.get("error") and prediction.get("prediction"):
        risk = prediction.get("risk_level", "LOW")
        score = prediction.get("model_score")
        if score is not None:
            score_pct = f" of {score * 100:.1f}%"

    # 1. FACTOR_EXPLANATION fallback (From real SHAP local explainer)
    if post_assessment_intent == "FACTOR_EXPLANATION":
        return format_factor_explanation_text(local_explanation, prediction)

    # 2. RISK_EXPLANATION fallback
    if post_assessment_intent == "RISK_EXPLANATION":
        return format_risk_explanation_text(local_explanation, prediction)

    # 3. CAPABILITY_QUESTION fallback (RUL / Failure causes)
    if post_assessment_intent == "CAPABILITY_QUESTION":
        return (
            "The current assessment estimates overall maintenance risk; it does not directly predict the vehicle's "
            "remaining useful life or diagnose exact mechanical failure causes."
        )

    # 4. SUMMARY_DECLINE fallback
    if post_assessment_intent == "SUMMARY_DECLINE":
        return (
            "No problem. The maintenance assessment is complete. If you'd like to review the result or "
            "ask about any of the confirmed factors later, just let me know."
        )

    # 5. ASSESSMENT_DETAILS fallback
    if post_assessment_intent == "ASSESSMENT_DETAILS":
        details = ", ".join(
            f"{FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' '))}: {value}"
            for feature, value in known.items()
        )
        return f"The assessment used these confirmed details: {details}."

    # An unclassified completed-state follow-up must never replay the one-time
    # completion prompt.
    if state.assessment_complete and post_assessment_intent == "GENERAL_FOLLOW_UP":
        return (
            f"The maintenance assessment is complete with a {risk} risk estimate{score_pct}. "
            "You can ask about the confirmed details, the factors behind this result, or practical next steps."
        )

    # 6. RECOMMENDATION fallback. Recommendations are limited to confirmed
    # inputs that actually pushed this individual prediction toward higher risk.
    if post_assessment_intent == "RECOMMENDATION":
        higher = (local_explanation or {}).get("positive_contributors", [])
        if higher:
            factor = higher[0]
            val_str = f" (reported as {factor['value']})" if factor.get("value") is not None else ""
            return (
                f"A sensible first step is to have the {factor['display_name']}"
                f"{val_str} checked at your next service, because it pushed this estimate toward higher risk. "
                "This is a model-guided maintenance priority, not a diagnosis."
            )
        return (
            "Keep up the vehicle's scheduled maintenance and arrange an inspection if any confirmed symptoms worsen. "
            "The current result is a maintenance-risk estimate, not a diagnosis."
        )

    # 7. SUMMARY_REQUEST / SUMMARY mode fallback
    if post_assessment_intent == "SUMMARY_REQUEST" or state.assessment_mode == "SUMMARY":
        summary_lines = ["Certainly! Here is a summary of the assessment:\n"]
        summary_lines.append(f"• Overall Maintenance Risk: {risk}{score_pct}")
        for feat, val in known.items():
            disp = FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").lower())
            summary_lines.append(f"• {disp.capitalize()}: {val}")

        if local_explanation and local_explanation.get("available") and local_explanation.get("top_contributors"):
            summary_lines.append("\nWhy this result — Strongest model contributors:")
            for f in local_explanation.get("top_contributors", [])[:3]:
                dir_lbl = "higher risk" if f["direction"] == "higher_risk" else "lower risk"
                val_lbl = f" ({f['value']})" if f.get("value") is not None else ""
                summary_lines.append(f"• {f['display_name']}{val_lbl} — {dir_lbl}")

        if local_explanation and local_explanation.get("available") and local_explanation.get("summary_explanation"):
            takeaway = local_explanation["summary_explanation"]
        else:
            takeaway = f"The current result indicates a {risk} maintenance risk based on your confirmed vehicle details."
        summary_lines.append(
            f"\n{takeaway} This is a model-based risk estimate rather than a mechanical diagnosis.\n\n"
            "If you have any questions or want to review specific details, just let me know!"
        )
        return "\n".join(summary_lines)

    # 8. AWAITING_SUMMARY_CONFIRMATION or Completed fallback
    if state.assessment_mode == "AWAITING_SUMMARY_CONFIRMATION" or state.assessment_complete or (not next_feature and known_count >= 8):
        return (
            "That gives me enough information to complete the maintenance assessment. "
            f"The current model estimates a {risk} maintenance risk{score_pct}. "
            "Based on the details you've shared, your confirmed vehicle conditions have been recorded.\n\n"
            "Would you like to see a summary of the assessment and the factors that influenced the result?"
        )

    lines: List[str] = []

    # Acknowledge known state
    if known_count > 0:
        details = ", ".join(f"{FEATURE_DISPLAY_NAMES.get(k, k)}: {v}" for k, v in list(known.items())[:3])
        lines.append(f"Thank you, I have recorded your vehicle details ({details}).")
    else:
        lines.append("I'm here to help assess your vehicle's maintenance requirements.")

    # Reference genuine ML prediction if present
    if prediction and not prediction.get("error") and prediction.get("prediction"):
        pred_label = "maintenance may be required" if prediction.get("prediction") == "maintenance_required" else "maintenance is not currently required"
        lines.append(f"Based on what you've shared so far, the model indicates {pred_label} ({risk} risk level).")
    elif known_count >= MIN_FEATURES_FOR_PREDICTION:
        lines.append("I'm refining the assessment as we gather more details.")
    else:
        lines.append("Let's narrow down your vehicle's condition with a few specific questions.")

    # Ask the single deterministic next question
    if next_feature:
        from backend.app.ml.preprocessing import CATEGORICAL_VALUES
        display = FEATURE_DISPLAY_NAMES.get(next_feature, next_feature.replace("_", " ").lower())
        if next_feature in CATEGORICAL_FEATURES:
            choices = ", ".join(CATEGORICAL_VALUES.get(next_feature, []))
            lines.append(f"How would you describe the vehicle's {display} — {choices}?")
        else:
            lines.append(f"Could you share the {display}?")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# JudgeAgent ADK LlmAgent Builder
# ---------------------------------------------------------------------------

def build_judge_agent() -> LlmAgent:
    """Build and return the JudgeAgent ADK LlmAgent."""
    return LlmAgent(
        name="JudgeAgent",
        description=(
            "Evaluates proposed response for factual grounding, schema validity, and zero hallucination. "
            "Returns score and validation status. Checks for hallucination, invalid feature questions, "
            "prediction consistency, and workflow correctness."
        ),
        model=get_judge_model(),
        instruction=JUDGE_AGENT_INSTRUCTION,
        tools=[
            FunctionTool(get_model_features),
        ],
        output_key="judge_result",
    )

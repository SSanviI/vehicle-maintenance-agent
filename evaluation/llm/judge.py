"""
Judge LLM & Guardrails Module
=============================
Evaluates response quality, grounding, faithfulness, dataset compliance,
state consistency, and hallucination with strict semantic evaluation and hard gates.

Reuses the production Judge LLM instructions and deterministic guards.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Tuple

from backend.app.llm.router import get_llm_router
from backend.app.llm.base import LLMMessage, LLMRole
from backend.app.agent.adk_agents.judge_agent import run_deterministic_guards
from backend.app.agent.state.state import AgentState
from backend.app.ml.feature_schema import MODEL_FEATURE_SCHEMA as DATASET_FEATURES


EVAL_JUDGE_SYSTEM_PROMPT = """You are an expert impartial evaluation judge for the AutoCare AI vehicle maintenance system.
Evaluate the assistant's response against the user message, context, canonical vehicle state, ML model prediction, SHAP local explanation, supported dataset schema, and ground truth expectations.

CRITICAL EVALUATION RULES:

1. RELEVANCE & INTENT MATCHING:
   - Does the response directly and meaningfully address the user's intent or question?
   - If the user asks a question (e.g. "What is the remaining useful life?", "Why is my risk high?", "What is the main factor?"), the response MUST address that question.
   - DEFLECTION IS A FAILURE: If the user asks an unsupported question (e.g. RUL, mechanical root cause) and the assistant merely deflects by asking an unrelated feature question without explaining system limitations, set relevance = 0, response_quality = 0, passed = false.
   - POST-ASSESSMENT QUESTIONS:
     * "what is the main factor?" -> Must identify the actual strongest local SHAP contributor when SHAP is available.
     * "why is my risk high?" -> Must explain actual model-supported contributors grounded in confirmed state.
     * "what influenced the prediction?" -> Must directly answer using SHAP explanation and state.
     * Answering "Would you like a summary?" to these questions is a FAILURE (relevance = 0, passed = false).

2. HALLUCINATION & FACTUAL GROUNDING:
   - Set hallucination = 1 if the response makes claims not supported by canonical state, dataset schema, model prediction, SHAP explanation, or known system capabilities.
   - Hallucinations include: invented RUL timeframes/years, invented contribution percentages (e.g. "68% responsible"), unsupported mechanical diagnoses/causes (e.g. "fuel pump failure"), invented vehicle values, or claiming confirmed state for non-schema items (e.g. turbocharger, sunroof).
   - If hallucination = 1, set faithfulness = 0 and passed = false.

3. DATASET & SCHEMA COMPLIANCE:
   - Supported schema features: Vehicle_Age, Mileage, Maintenance_History, Service_History, Accident_History, Reported_Issues, Tire_Condition, Brake_Condition, Battery_Status, Last_Service_Date, etc.
   - The assistant must stay within the dataset schema.

4. STATE CONSISTENCY:
   - The response must respect confirmed canonical state and never contradict confirmed features.

5. QUESTION RELEVANCE:
   - If the assistant asks a question, it must be relevant, within dataset schema, and must NOT re-ask already known or declined features. If no question was needed (e.g. answering a post-assessment question), question_relevance is 1.

6. PASS CRITERIA (HARD GATES):
   - passed is true ONLY IF: hallucination == 0 AND dataset_compliance == 1 AND state_consistency == 1 AND faithfulness == 1 AND relevance == 1 AND question_relevance == 1 AND response_quality == 1.

OUTPUT FORMAT:
Output ONLY valid JSON matching this schema:
{
  "faithfulness": <1 or 0>,
  "relevance": <1 or 0>,
  "dataset_compliance": <1 or 0>,
  "state_consistency": <1 or 0>,
  "question_relevance": <1 or 0>,
  "hallucination": <1 or 0>,
  "response_quality": <1 or 0>,
  "passed": <true or false>,
  "reason": "<1-2 sentence concise explanation>"
}
"""


def _parse_judge_json(raw_text: str) -> Dict[str, Any]:
    """Robustly parse JSON from LLM response, handling markdown blocks or extra text."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed JSON is not a dictionary: {type(parsed)}")
    return parsed


def _extract_retry_delay(error_msg: str, default: float = 12.0) -> float:
    """Extract recommended cooldown from rate limit error message if available."""
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", error_msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.5
        except Exception:
            pass
    return default


class EvaluationJudge:
    """
    Evaluator judge combining deterministic checks and Judge LLM semantic evaluation.
    """

    def __init__(self):
        self.router = get_llm_router()

    async def evaluate_turn(
        self,
        user_input: str,
        response_text: str,
        agent_state: AgentState,
        prediction: Optional[Dict[str, Any]] = None,
        expected_next_feature: Optional[str] = None,
        post_assessment_intent: Optional[str] = None,
        local_explanation: Optional[Dict[str, Any]] = None,
        expected_behavior: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single conversation turn with separate deterministic checks and Judge LLM.
        """
        # ---------------------------------------------------------------------
        # 1. Deterministic Checks (Objective ground truth validation)
        # ---------------------------------------------------------------------
        guard_violations: List[str] = []
        is_guard_valid, base_violations = run_deterministic_guards(
            response_text=response_text,
            state=agent_state,
            prediction=prediction,
            expected_next_feature=expected_next_feature,
            post_assessment_intent=post_assessment_intent,
            local_explanation=local_explanation,
        )
        guard_violations.extend(base_violations)

        # Dataset Compliance Check
        dataset_compliance = True
        known_features = agent_state.get_known_features()
        for k in known_features.keys():
            if k not in DATASET_FEATURES:
                dataset_compliance = False
                guard_violations.append(f"Non-dataset feature found in state: '{k}'")

        if expected_next_feature and expected_next_feature not in DATASET_FEATURES:
            dataset_compliance = False
            guard_violations.append(f"Expected next question feature '{expected_next_feature}' is outside dataset schema.")

        # State Consistency Check
        state_consistency = True
        if expected_behavior and "expected_state" in expected_behavior:
            for exp_k, exp_v in expected_behavior["expected_state"].items():
                actual_v = known_features.get(exp_k)
                if actual_v != exp_v:
                    state_consistency = False
                    guard_violations.append(
                        f"State inconsistency for '{exp_k}': expected '{exp_v}', got '{actual_v}'"
                    )

        # Deterministic Forbidden Patterns Check
        deterministic_hallucination = False
        if expected_behavior:
            forbidden = expected_behavior.get("forbidden_in_response", [])
            for forbidden_term in forbidden:
                if forbidden_term.lower() in response_text.lower():
                    deterministic_hallucination = True
                    guard_violations.append(f"Forbidden term detected in response: '{forbidden_term}'")

            forbidden_patterns = expected_behavior.get("forbidden_patterns", [])
            for pattern in forbidden_patterns:
                if re.search(pattern, response_text, re.IGNORECASE):
                    deterministic_hallucination = True
                    guard_violations.append(f"Forbidden regex pattern matched: '{pattern}'")

        deterministic_passed = (
            is_guard_valid
            and dataset_compliance
            and state_consistency
            and not deterministic_hallucination
        )

        deterministic_checks = {
            "dataset_compliance": dataset_compliance,
            "state_consistency": state_consistency,
            "guard_valid": is_guard_valid and not deterministic_hallucination,
            "passed": deterministic_passed,
            "guard_violations": guard_violations,
        }

        # ---------------------------------------------------------------------
        # 2. Judge LLM Semantic Evaluation
        # ---------------------------------------------------------------------
        prompt_context = {
            "user_message": user_input,
            "assistant_response": response_text,
            "canonical_state": known_features,
            "available_dataset_features": list(DATASET_FEATURES),
            "next_question_feature": expected_next_feature,
            "post_assessment_intent": post_assessment_intent,
            "prediction": (
                {
                    "risk_level": prediction.get("risk_level"),
                    "model_score": prediction.get("model_score"),
                }
                if prediction else None
            ),
            "local_explanation": (
                {
                    "main_factor": local_explanation.get("main_factor"),
                    "top_contributors": [
                        c.get("feature") for c in local_explanation.get("top_contributors", [])[:3]
                    ],
                }
                if local_explanation else None
            ),
            "expected_behavior": expected_behavior or {},
        }

        judge_messages = [
            LLMMessage(role="system", content=EVAL_JUDGE_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"Evaluate this interaction and respond strictly in JSON:\n{json.dumps(prompt_context, indent=2)}",
            ),
        ]

        llm_judge_data: Optional[Dict[str, Any]] = None
        judge_error: Optional[str] = None
        used_model: str = "unknown"
        used_provider: str = "unknown"

        # Call Judge LLM with retry for rate limits / transient errors
        for attempt in range(4):
            # Reset circuit breaker cooldown if set from previous rapid requests
            for h in self.router.health.values():
                h.is_circuit_open = False
                h.cooldown_until = 0.0
                h.consecutive_failures = 0

            try:
                response = await self.router.generate(
                    messages=judge_messages,
                    role=LLMRole.FAST,
                    temperature=0.0,
                    max_tokens=1000,
                )
                used_model = response.model or "unknown"
                used_provider = response.provider or "unknown"

                if response and response.content:
                    llm_judge_data = _parse_judge_json(response.content)
                    judge_error = None
                    break
                else:
                    raise ValueError("Empty response content from Judge LLM.")

            except Exception as exc:
                judge_error = str(exc)
                if "429" in judge_error or "rate limit" in judge_error.lower():
                    delay = _extract_retry_delay(judge_error, default=10.0 * (attempt + 1))
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(2)

        # ---------------------------------------------------------------------
        # 3. Assemble Judge Result
        # ---------------------------------------------------------------------
        if llm_judge_data:
            faithfulness = int(llm_judge_data.get("faithfulness", 1))
            relevance = int(llm_judge_data.get("relevance", 1))
            llm_dataset = int(llm_judge_data.get("dataset_compliance", 1))
            llm_state = int(llm_judge_data.get("state_consistency", 1))
            q_relevance = int(llm_judge_data.get("question_relevance", 1))
            hallucination = int(llm_judge_data.get("hallucination", 0))
            quality = int(llm_judge_data.get("response_quality", 1))
            reason = str(llm_judge_data.get("reason", "Semantic evaluation completed by Judge LLM."))

            # Enforce hard deterministic overrides
            if not dataset_compliance:
                llm_dataset = 0
            if not state_consistency:
                llm_state = 0
            if deterministic_hallucination:
                hallucination = 1
                faithfulness = 0

            passed = bool(
                hallucination == 0
                and llm_dataset == 1
                and llm_state == 1
                and faithfulness == 1
                and relevance == 1
                and q_relevance == 1
                and quality == 1
                and is_guard_valid
            )

            if not is_guard_valid and guard_violations:
                reason = f"Guard violation: {'; '.join(guard_violations)}. " + reason

            judge_result = {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "dataset_compliance": llm_dataset,
                "state_consistency": llm_state,
                "question_relevance": q_relevance,
                "hallucination": hallucination,
                "response_quality": quality,
                "passed": passed,
                "reason": reason,
                "judge_source": "llm",
                "model": used_model,
                "provider": used_provider,
                "error": None,
            }
        else:
            # Judge LLM failed: record error explicitly, do NOT silently fall back
            judge_result = {
                "faithfulness": 0,
                "relevance": 0,
                "dataset_compliance": 1 if dataset_compliance else 0,
                "state_consistency": 1 if state_consistency else 0,
                "question_relevance": 0,
                "hallucination": 1 if deterministic_hallucination else 0,
                "response_quality": 0,
                "passed": False,
                "reason": f"Judge LLM failed to evaluate: {judge_error}",
                "judge_source": "llm",
                "model": used_model,
                "provider": used_provider,
                "error": judge_error,
            }

        return {
            "deterministic_checks": deterministic_checks,
            "judge": judge_result,
        }

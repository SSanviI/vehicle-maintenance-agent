"""
LLM Evaluation Runner
=====================
Executes test cases from evaluation datasets through the live ADK agents
in fully isolated test sessions. Runs the EvaluationJudge on each output.

Outputs:
  evaluation/reports/llm_report.json
  evaluation/reports/raw_results.json
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from backend.app.agent.adk_agents.runner import ADKAgentRunner
from backend.app.agent.state.state import AgentState
from backend.app.agent.tools.feature_tools import (
    classify_post_assessment_intent,
)
from backend.app.llm.router import get_llm_router
from evaluation.llm.judge import EvaluationJudge
from evaluation.llm.metrics import calculate_llm_aggregate_metrics


def _build_state_from_dict(raw_state: Dict[str, Any]) -> AgentState:
    """Construct an AgentState instance from a feature dict."""
    state = AgentState()
    for feat, val in raw_state.items():
        if isinstance(val, dict):
            state.feature_state[feat] = val
        else:
            state.feature_state[feat] = {
                "value": val,
                "status": "known",
                "source": "user",
            }
    return state


def _extract_retry_delay(error_msg: str, default: float = 12.0) -> float:
    """Extract recommended cooldown from rate limit error message if available."""
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", error_msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.5
        except Exception:
            pass
    return default


async def _run_agent_turn_with_retry(
    runner: ADKAgentRunner,
    user_message: str,
    conversation_id: str,
    agent_state: AgentState,
    conversation_history: List[Dict[str, str]],
    max_retries: int = 4,
) -> Dict[str, Any]:
    """Run an agent turn with rate limit retry and router health management."""
    router = get_llm_router()
    last_res = {}
    for attempt in range(max_retries):
        for h in router.health.values():
            h.is_circuit_open = False
            h.cooldown_until = 0.0
            h.consecutive_failures = 0

        try:
            res = await runner.run_turn(
                user_message=user_message,
                conversation_id=conversation_id,
                agent_state=agent_state,
                conversation_history=conversation_history,
            )
            resp_text = res.get("response", "")
            if resp_text and not resp_text.startswith("Error:"):
                return res
            last_res = res
        except Exception as exc:
            err_str = str(exc)
            last_res = {"response": f"Error: {err_str}", "prediction": None}
            if "429" in err_str or "rate limit" in err_str.lower():
                delay = _extract_retry_delay(err_str, default=10.0 * (attempt + 1))
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(2)

    return last_res


async def run_llm_evaluation(
    datasets_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    cases_limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run evaluation across all dataset files.
    """
    workspace_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = datasets_dir or (workspace_root / "evaluation" / "datasets")
    output_dir = output_dir or (workspace_root / "evaluation" / "reports")

    judge = EvaluationJudge()
    runner = ADKAgentRunner()
    runner.initialize()

    # Load dataset files
    dataset_files = [
        datasets_dir / "llm_eval_cases.json",
        datasets_dir / "hallucination_cases.json",
        datasets_dir / "question_cases.json",
        datasets_dir / "post_assessment_cases.json",
    ]

    all_cases: List[Dict[str, Any]] = []
    for fpath in dataset_files:
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                cases = json.load(f)
                all_cases.extend(cases)

    if cases_limit and cases_limit > 0:
        all_cases = all_cases[:cases_limit]

    case_results: List[Dict[str, Any]] = []

    for case in all_cases:
        case_id = case.get("id", f"case_{uuid.uuid4().hex[:6]}")
        category = case.get("category", "general")
        description = case.get("description", "")

        # Pacing to avoid token-per-minute bursts on LLM providers
        await asyncio.sleep(2.0)

        # ---------------------------------------------------------------------
        # Multi-Turn Case (e.g. State Correction)
        # ---------------------------------------------------------------------
        if "multi_turn" in case:
            conv_id = f"eval_{case_id}_{uuid.uuid4().hex[:6]}"
            agent_state = AgentState()
            conv_history: List[Dict[str, str]] = []
            turn_results = []
            multi_turn_passed = True
            failure_reasons = []

            for turn_spec in case["multi_turn"]:
                user_msg = turn_spec["input"]
                res = await _run_agent_turn_with_retry(
                    runner=runner,
                    user_message=user_msg,
                    conversation_id=conv_id,
                    agent_state=agent_state,
                    conversation_history=conv_history,
                )
                conv_history.append({"role": "user", "content": user_msg})
                conv_history.append({"role": "assistant", "content": res.get("response", "")})

                expected_state = turn_spec.get("expected_state", {})
                known_now = agent_state.get_known_features()
                state_ok = True
                for exp_k, exp_v in expected_state.items():
                    if known_now.get(exp_k) != exp_v:
                        state_ok = False
                        multi_turn_passed = False
                        failure_reasons.append(f"Turn {turn_spec.get('turn')} state mismatch: expected {exp_k}={exp_v}, got {known_now.get(exp_k)}")

                eval_res = await judge.evaluate_turn(
                    user_input=user_msg,
                    response_text=res.get("response", ""),
                    agent_state=agent_state,
                    prediction=res.get("prediction"),
                    expected_behavior=turn_spec,
                )
                det_checks = eval_res["deterministic_checks"]
                j_res = eval_res["judge"]

                if not j_res.get("passed", False):
                    multi_turn_passed = False
                    failure_reasons.append(f"Turn {turn_spec.get('turn')} judge failed: {j_res.get('reason')}")

                turn_results.append({
                    "turn": turn_spec.get("turn"),
                    "input": user_msg,
                    "response": res.get("response"),
                    "state": known_now,
                    "deterministic_checks": det_checks,
                    "judge": j_res,
                })

            last_turn = turn_results[-1] if turn_results else {}
            case_results.append({
                "case_id": case_id,
                "category": category,
                "description": description,
                "input": [t["input"] for t in turn_results],
                "response": [t["response"] for t in turn_results],
                "expected": {
                    "multi_turn": case["multi_turn"],
                },
                "actual": {
                    "final_state": agent_state.get_known_features(),
                    "turn_responses": [t["response"] for t in turn_results],
                },
                "is_multi_turn": True,
                "turns": turn_results,
                "deterministic_checks": {
                    "state_consistency": multi_turn_passed,
                    "dataset_compliance": all(t["deterministic_checks"].get("dataset_compliance", True) for t in turn_results),
                    "guard_valid": all(t["deterministic_checks"].get("guard_valid", True) for t in turn_results),
                    "passed": multi_turn_passed,
                },
                "judge": last_turn.get("judge", {}),
                "judge_source": last_turn.get("judge", {}).get("judge_source", "llm"),
                "passed": multi_turn_passed,
                "failure_reason": "; ".join(failure_reasons) if not multi_turn_passed else None,
            })
            continue

        # ---------------------------------------------------------------------
        # Single-Turn Case
        # ---------------------------------------------------------------------
        user_input = case.get("input", "")
        current_q = case.get("current_question")
        initial_state_dict = case.get("initial_state", {})
        agent_state = _build_state_from_dict(initial_state_dict)

        if case.get("assessment_complete"):
            agent_state.assessment_complete = True
            agent_state.assessment_mode = case.get("initial_mode", "AWAITING_SUMMARY_CONFIRMATION")
            # If state has >= 2 features, generate real prediction for post-assessment context
            if agent_state.known_feature_count() >= 2:
                from backend.app.ml.predict import MaintenancePredictor
                pred_engine = MaintenancePredictor()
                ws = Path(__file__).resolve().parent.parent.parent
                pred_engine.load(
                    str(ws / "models" / "maintenance_model.joblib"),
                    str(ws / "models" / "preprocessing_pipeline.joblib"),
                )
                pred = pred_engine.predict(agent_state.feature_state)
                expl = pred_engine.explain(agent_state.feature_state, pred)
                agent_state.add_prediction(pred)
                if expl and expl.get("available"):
                    agent_state.add_explanation(expl)

        if current_q:
            agent_state.current_question_feature = current_q
            agent_state.mark_asked(current_q)

        conv_id = f"eval_{case_id}_{uuid.uuid4().hex[:6]}"
        conv_history = []

        # Execute through ADK Runner with retry
        result = await _run_agent_turn_with_retry(
            runner=runner,
            user_message=user_input,
            conversation_id=conv_id,
            agent_state=agent_state,
            conversation_history=conv_history,
        )

        response_text = result.get("response", "")
        prediction = result.get("prediction")
        known_after = agent_state.get_known_features()

        # Input extraction verification
        extraction_expected_count = 0
        extraction_correct_count = 0
        extraction_correct = True
        if "expected_extracted" in case:
            expected_extracted = case["expected_extracted"]
            extraction_expected_count = len(expected_extracted)
            for exp_f, exp_val in expected_extracted.items():
                if known_after.get(exp_f) == exp_val:
                    extraction_correct_count += 1
                else:
                    extraction_correct = False

        # Post-assessment intent classification verification
        routing_correct = True
        post_intent = None
        if case.get("assessment_complete") or case.get("expected_intent"):
            post_intent = classify_post_assessment_intent(user_input, case.get("initial_mode", "AWAITING_SUMMARY_CONFIRMATION"))
            if case.get("expected_intent") and post_intent != case["expected_intent"]:
                routing_correct = False

        # Evaluate via Judge
        eval_res = await judge.evaluate_turn(
            user_input=user_input,
            response_text=response_text,
            agent_state=agent_state,
            prediction=prediction,
            expected_next_feature=agent_state.current_question_feature,
            post_assessment_intent=post_intent,
            local_explanation=agent_state.get_latest_explanation() if hasattr(agent_state, "get_latest_explanation") else None,
            expected_behavior=case,
        )

        det_checks = eval_res["deterministic_checks"]
        det_checks["extraction_correct"] = extraction_correct
        det_checks["routing_correct"] = routing_correct

        judge_result = eval_res["judge"]

        case_passed = bool(
            judge_result.get("passed", False)
            and det_checks.get("passed", False)
            and (extraction_expected_count == 0 or extraction_correct)
            and routing_correct
        )

        failure_reasons = []
        if not judge_result.get("passed", False):
            failure_reasons.append(f"Judge: {judge_result.get('reason')}")
        if not det_checks.get("passed", False):
            failure_reasons.append(f"Deterministic: {'; '.join(det_checks.get('guard_violations', []))}")
        if extraction_expected_count > 0 and not extraction_correct:
            failure_reasons.append(f"Extraction mismatch: expected {case.get('expected_extracted')}, got {known_after}")
        if not routing_correct:
            failure_reasons.append(f"Routing mismatch: expected {case.get('expected_intent')}, got {post_intent}")

        case_results.append({
            "case_id": case_id,
            "category": category,
            "description": description,
            "input": user_input,
            "response": response_text,
            "expected": {
                "expected_extracted": case.get("expected_extracted"),
                "expected_state": case.get("expected_state"),
                "expected_intent": case.get("expected_intent"),
                "current_question": current_q,
            },
            "actual": {
                "known_features": known_after,
                "detected_intent": post_intent,
                "current_question_feature": agent_state.current_question_feature,
            },
            "extraction_expected_count": extraction_expected_count,
            "extraction_correct_count": extraction_correct_count,
            "routing_correct": routing_correct,
            "deterministic_checks": det_checks,
            "judge": judge_result,
            "judge_source": judge_result.get("judge_source", "llm"),
            "passed": case_passed,
            "failure_reason": "; ".join(failure_reasons) if not case_passed else None,
        })

    # Compute aggregate LLM metrics
    report = calculate_llm_aggregate_metrics(case_results)
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat()

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "llm_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(case_results, f, indent=2)

    return report, case_results

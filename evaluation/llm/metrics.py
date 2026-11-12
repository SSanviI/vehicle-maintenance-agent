"""
LLM Evaluation Metrics Aggregation
==================================
Computes actual rates, deterministic vs LLM statistics, and weighted overall score
from raw evaluation cases.
"""

from __future__ import annotations

from typing import Dict, Any, List


METRIC_WEIGHTS = {
    "hallucination_free_rate": 0.20,
    "faithfulness_rate": 0.15,
    "dataset_compliance_rate": 0.15,
    "state_consistency_rate": 0.15,
    "response_relevance_rate": 0.15,
    "question_relevance_rate": 0.10,
    "response_quality_rate": 0.10,
}


def calculate_llm_aggregate_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics across all evaluated test cases.
    """
    total_cases = len(case_results)
    if total_cases == 0:
        return {
            "total_test_cases": 0,
            "overall_score": 0.0,
            "judge_pass_rate": 0.0,
            "deterministic_compliance_rate": 0.0,
        }

    # Accumulators
    input_extraction_expected = 0
    input_extraction_correct = 0

    dataset_compliance_count = 0
    state_consistency_count = 0
    state_consistency_total = 0

    faithfulness_count = 0
    relevance_count = 0
    question_relevance_count = 0
    quality_count = 0
    hallucination_free_count = 0

    post_assessment_total = 0
    post_assessment_correct = 0

    overall_passed_count = 0
    deterministic_passed_count = 0
    judge_passed_count = 0

    llm_judge_cases = 0
    deterministic_only_cases = 0
    judge_llm_failures = 0
    fallback_count = 0

    for case in case_results:
        judge = case.get("judge", {})
        det = case.get("deterministic_checks", {})
        j_source = case.get("judge_source", judge.get("judge_source", "llm"))

        # Judge source tracking
        if j_source == "llm":
            llm_judge_cases += 1
            if judge.get("error"):
                judge_llm_failures += 1
        elif j_source == "deterministic_fallback":
            fallback_count += 1
            deterministic_only_cases += 1
        else:
            deterministic_only_cases += 1

        # Pass / fail tracking
        if case.get("passed", False):
            overall_passed_count += 1

        if det.get("passed", False):
            deterministic_passed_count += 1

        if judge.get("passed", False):
            judge_passed_count += 1

        # Faithfulness
        if judge.get("faithfulness", 0) == 1:
            faithfulness_count += 1

        # Relevance
        if judge.get("relevance", 0) == 1:
            relevance_count += 1

        # Dataset Compliance
        if det.get("dataset_compliance", False) and judge.get("dataset_compliance", 0) == 1:
            dataset_compliance_count += 1

        # State Consistency
        if "state_consistent" in case:
            state_consistency_total += 1
            if case["state_consistent"]:
                state_consistency_count += 1
        elif det.get("state_consistency", False) and judge.get("state_consistency", 1) == 1:
            state_consistency_total += 1
            state_consistency_count += 1
        else:
            state_consistency_total += 1

        # Question Relevance
        if judge.get("question_relevance", 0) == 1:
            question_relevance_count += 1

        # Hallucination-free
        if judge.get("hallucination", 0) == 0:
            hallucination_free_count += 1

        # Response Quality
        if judge.get("response_quality", 0) == 1:
            quality_count += 1

        # Input Extraction
        if "extraction_expected_count" in case:
            input_extraction_expected += case["extraction_expected_count"]
            input_extraction_correct += case.get("extraction_correct_count", 0)

        # Post-assessment routing
        if case.get("category") in ("summary_request", "factor_explanation", "risk_explanation", "factor_explanation_variation", "new_assessment_request"):
            post_assessment_total += 1
            if case.get("routing_correct", False):
                post_assessment_correct += 1

    # Calculate rates
    extraction_accuracy = (
        (input_extraction_correct / input_extraction_expected)
        if input_extraction_expected > 0
        else 1.0
    )
    dataset_compliance_rate = dataset_compliance_count / total_cases
    state_consistency_rate = (
        (state_consistency_count / state_consistency_total)
        if state_consistency_total > 0
        else 1.0
    )
    faithfulness_rate = faithfulness_count / total_cases
    response_relevance_rate = relevance_count / total_cases
    question_relevance_rate = question_relevance_count / total_cases
    hallucination_free_rate = hallucination_free_count / total_cases
    response_quality_rate = quality_count / total_cases
    post_assessment_accuracy = (
        (post_assessment_correct / post_assessment_total)
        if post_assessment_total > 0
        else 1.0
    )

    judge_pass_rate = judge_passed_count / total_cases
    deterministic_compliance_rate = deterministic_passed_count / total_cases

    # Weighted Overall Score based on semantic judge metrics and compliance
    overall_score = (
        METRIC_WEIGHTS["hallucination_free_rate"] * hallucination_free_rate
        + METRIC_WEIGHTS["faithfulness_rate"] * faithfulness_rate
        + METRIC_WEIGHTS["dataset_compliance_rate"] * dataset_compliance_rate
        + METRIC_WEIGHTS["state_consistency_rate"] * state_consistency_rate
        + METRIC_WEIGHTS["response_relevance_rate"] * response_relevance_rate
        + METRIC_WEIGHTS["question_relevance_rate"] * question_relevance_rate
        + METRIC_WEIGHTS["response_quality_rate"] * response_quality_rate
    )

    return {
        "total_test_cases": total_cases,
        "passed_cases": overall_passed_count,
        "failed_cases": total_cases - overall_passed_count,
        "overall_pass_rate": overall_passed_count / total_cases,
        "judge_pass_rate": judge_pass_rate,
        "deterministic_compliance_rate": deterministic_compliance_rate,
        "llm_judge_cases": llm_judge_cases,
        "deterministic_only_cases": deterministic_only_cases,
        "judge_llm_failures": judge_llm_failures,
        "fallback_count": fallback_count,
        "input_extraction_accuracy": extraction_accuracy,
        "question_relevance_rate": question_relevance_rate,
        "dataset_compliance_rate": dataset_compliance_rate,
        "faithfulness_rate": faithfulness_rate,
        "state_consistency_rate": state_consistency_rate,
        "hallucination_free_rate": hallucination_free_rate,
        "response_relevance_rate": response_relevance_rate,
        "response_quality_rate": response_quality_rate,
        "post_assessment_routing_accuracy": post_assessment_accuracy,
        "overall_score": float(round(overall_score, 4)),
        "score_weights": METRIC_WEIGHTS,
        "score_formula": "0.20*hallucination_free + 0.15*faithfulness + 0.15*dataset_compliance + 0.15*state_consistency + 0.15*relevance + 0.10*question_relevance + 0.10*quality",
    }

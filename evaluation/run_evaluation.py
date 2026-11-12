"""
AutoCare AI — Main Evaluation Entrypoint
=======================================
Runs complete, isolated evaluation across ML, SHAP, and LLM multi-agent pipelines.

Usage:
  python evaluation/run_evaluation.py
  python evaluation/run_evaluation.py --ml-only
  python evaluation/run_evaluation.py --llm-only
  python evaluation/run_evaluation.py --shap-only
"""

from __future__ import annotations

import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Ensure project root is on sys.path and remove script directory from sys.path[0] to prevent package shadowing
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
while sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Pre-import real shap to ensure it's in sys.modules
try:
    import shap
except ImportError:
    pass

from evaluation.ml.evaluate_model import run_ml_evaluation
from evaluation.shap.evaluate_explanations import run_shap_evaluation
from evaluation.llm.runner import run_llm_evaluation


def generate_markdown_report(report_data: Dict[str, Any], output_path: Path) -> str:
    """Generate human-readable markdown report from consolidated evaluation data."""
    ml = report_data.get("ml", {})
    llm = report_data.get("llm", {})
    shap = report_data.get("shap", {})
    evaluated_at = report_data.get("evaluated_at", datetime.now(timezone.utc).isoformat())

    md = f"""# AutoCare AI — Evaluation Report

*Evaluated at: {evaluated_at}*

---

## 1. ML Model Performance (Random Forest)

| Metric | Measured Value |
| :--- | :--- |
| **Model Type** | `{ml.get('model_type', 'RandomForestClassifier')}` |
| **Test Samples** | {ml.get('test_samples', 0):,} |
| **Accuracy** | **{ml.get('accuracy', 0.0):.4f}** ({ml.get('accuracy', 0.0)*100:.2f}%) |
| **Precision** | **{ml.get('precision', 0.0):.4f}** ({ml.get('precision', 0.0)*100:.2f}%) |
| **Recall** | **{ml.get('recall', 0.0):.4f}** ({ml.get('recall', 0.0)*100:.2f}%) |
| **F1 Score** | **{ml.get('f1', 0.0):.4f}** ({ml.get('f1', 0.0)*100:.2f}%) |
| **ROC-AUC** | **{ml.get('roc_auc', 0.0):.4f}** |
| **PR-AUC (Avg Precision)** | **{ml.get('pr_auc', 0.0):.4f}** |
| **Expected Calibration Error (ECE)** | {ml.get('calibration_error', 0.0):.4f} |
| **Brier Score** | {ml.get('brier_score', 0.0):.4f} |
| **False Positives / Negatives** | FP: {ml.get('false_positives', 0)} | FN: {ml.get('false_negatives', 0)} |

### Confusion Matrix
```
{json.dumps(ml.get('confusion_matrix', []), indent=2)}
```

---

## 2. LLM & Multi-Agent Pipeline Performance

| Metric | Measured Rate | Applicable Scope |
| :--- | :--- | :--- |
| **Input Extraction Accuracy** | **{llm.get('input_extraction_accuracy', 0.0)*100:.2f}%** | Explicit facts, spelling errors, short answers |
| **Question Relevance Rate** | **{llm.get('question_relevance_rate', 0.0)*100:.2f}%** | Dynamic & relevant questioning (no re-asking) |
| **Dataset Compliance Rate** | **{llm.get('dataset_compliance_rate', 0.0)*100:.2f}%** | Strictly within 19 dataset schema features |
| **Faithfulness Rate** | **{llm.get('faithfulness_rate', 0.0)*100:.2f}%** | Grounded in canonical state & predictions |
| **State Consistency Rate** | **{llm.get('state_consistency_rate', 0.0)*100:.2f}%** | Multi-turn state tracking & updates |
| **Hallucination-Free Rate** | **{llm.get('hallucination_free_rate', 0.0)*100:.2f}%** | Zero invented facts / unsupported claims |
| **Response Relevance Rate** | **{llm.get('response_relevance_rate', 0.0)*100:.2f}%** | Direct answers to user intents |
| **Response Quality Rate** | **{llm.get('response_quality_rate', 0.0)*100:.2f}%** | Clarity, tone, formatting |
| **Post-Assessment Routing** | **{llm.get('post_assessment_routing_accuracy', 0.0)*100:.2f}%** | Summary & explanation navigation |
| **Judge LLM Pass Rate** | **{llm.get('judge_pass_rate', 0.0)*100:.2f}%** | Semantic LLM Judge evaluation |
| **Deterministic Compliance Rate** | **{llm.get('deterministic_compliance_rate', 0.0)*100:.2f}%** | Objective schema & guardrails compliance |

### Evaluation Subsystem Diagnostics
- **Total Test Cases**: {llm.get('total_test_cases', 0)}
- **LLM Judge Cases**: {llm.get('llm_judge_cases', 0)}
- **Deterministic-Only Cases**: {llm.get('deterministic_only_cases', 0)}
- **Judge LLM Failures / Errors**: {llm.get('judge_llm_failures', 0)}
- **Deterministic Fallback Count**: {llm.get('fallback_count', 0)}
- **Passed Cases**: {llm.get('passed_cases', 0)}
- **Failed Cases**: {llm.get('failed_cases', 0)}

### **Overall LLM Score**: `{llm.get('overall_score', 0.0)*100:.2f}%`
*Score Formula: `{llm.get('score_formula', '')}`*

---

## 3. SHAP Local Explainability Quality

| Metric | Measured Value |
| :--- | :--- |
| **Total Test Scenarios** | {shap.get('total_test_cases', 0)} |
| **Successful Explanations** | {shap.get('successful_explanations', 0)} / {shap.get('total_test_cases', 0)} |
| **SHAP Explanation Success Rate** | **{shap.get('explanation_success_rate', 0.0)*100:.2f}%** |
| **Top Factor Validity Rate** | **{shap.get('top_factor_validity_rate', 0.0)*100:.2f}%** |
| **Schema Compliance Rate** | **{shap.get('feature_schema_compliance_rate', 0.0)*100:.2f}%** |

---

## 4. Overall Evaluation Summary

- **Total Cases Evaluated**: {llm.get('total_test_cases', 0)}
- **Judge LLM Pass Rate**: **{llm.get('judge_pass_rate', 0.0)*100:.2f}%**
- **Deterministic Compliance**: **{llm.get('deterministic_compliance_rate', 0.0)*100:.2f}%**
- **Overall System Status**: **{'PASSED' if llm.get('passed_cases', 0) == llm.get('total_test_cases', 0) else 'COMPLETED WITH AUDIT FINDINGS'}**
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")
    return md


def print_ascii_summary(report_data: Dict[str, Any]):
    """Print clean ASCII summary to console."""
    ml = report_data.get("ml", {})
    llm = report_data.get("llm", {})
    shap = report_data.get("shap", {})

    print("\n" + "=" * 62)
    print("           AUTOCARE AI — EVALUATION REPORT")
    print("=" * 62)

    print("\nML MODEL")
    print("-" * 62)
    print(f"Accuracy:                 {ml.get('accuracy', 0.0)*100:.2f}%")
    print(f"Precision:                {ml.get('precision', 0.0)*100:.2f}%")
    print(f"Recall:                   {ml.get('recall', 0.0)*100:.2f}%")
    print(f"F1 Score:                 {ml.get('f1', 0.0)*100:.2f}%")
    print(f"ROC-AUC:                  {ml.get('roc_auc', 0.0):.4f}")
    print(f"PR-AUC:                   {ml.get('pr_auc', 0.0):.4f}")
    print(f"Calibration Error (ECE):  {ml.get('calibration_error', 0.0):.4f}")

    print("\nLLM / AGENT")
    print("-" * 62)
    print(f"Test cases evaluated:     {llm.get('total_test_cases', 0)}")
    print(f"Input Extraction:         {llm.get('input_extraction_accuracy', 0.0)*100:.2f}%")
    print(f"Question Relevance:       {llm.get('question_relevance_rate', 0.0)*100:.2f}%")
    print(f"Dataset Compliance:       {llm.get('dataset_compliance_rate', 0.0)*100:.2f}%")
    print(f"Faithfulness:             {llm.get('faithfulness_rate', 0.0)*100:.2f}%")
    print(f"State Consistency:        {llm.get('state_consistency_rate', 0.0)*100:.2f}%")
    print(f"Hallucination-Free:       {llm.get('hallucination_free_rate', 0.0)*100:.2f}%")
    print(f"Response Relevance:       {llm.get('response_relevance_rate', 0.0)*100:.2f}%")
    print(f"Response Quality:         {llm.get('response_quality_rate', 0.0)*100:.2f}%")
    print(f"Post-Assessment Routing:  {llm.get('post_assessment_routing_accuracy', 0.0)*100:.2f}%")

    print("\nJUDGE & AUDIT DIAGNOSTICS")
    print("-" * 62)
    print(f"LLM Judge Cases:          {llm.get('llm_judge_cases', 0)}")
    print(f"Deterministic-Only Cases: {llm.get('deterministic_only_cases', 0)}")
    print(f"Judge LLM Failures:       {llm.get('judge_llm_failures', 0)}")
    print(f"Fallback Count:           {llm.get('fallback_count', 0)}")
    print(f"Deterministic Compliance: {llm.get('deterministic_compliance_rate', 0.0)*100:.2f}%")
    print(f"Judge LLM Pass Rate:      {llm.get('judge_pass_rate', 0.0)*100:.2f}%")
    print(f"Overall Passed Cases:     {llm.get('passed_cases', 0)} / {llm.get('total_test_cases', 0)}")

    print("\nSHAP EXPLAINABILITY")
    print("-" * 62)
    print(f"Explanation Success:      {shap.get('explanation_success_rate', 0.0)*100:.2f}%")
    print(f"Top Factor Validity:      {shap.get('top_factor_validity_rate', 0.0)*100:.2f}%")
    print(f"Schema Compliance:        {shap.get('feature_schema_compliance_rate', 0.0)*100:.2f}%")

    print("\nOVERALL")
    print("-" * 62)
    print(f"Overall LLM Score:        {llm.get('overall_score', 0.0)*100:.2f}%")
    print("=" * 62 + "\n")


async def main_async():
    parser = argparse.ArgumentParser(description="AutoCare AI Evaluation System")
    parser.add_argument("--ml-only", action="store_true", help="Run only ML evaluation")
    parser.add_argument("--llm-only", action="store_true", help="Run only LLM evaluation")
    parser.add_argument("--shap-only", action="store_true", help="Run only SHAP evaluation")
    parser.add_argument("--cases-limit", type=int, default=None, help="Limit number of LLM test cases")
    args = parser.parse_args()

    reports_dir = WORKSPACE_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    consolidated: Dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "ml": {},
        "shap": {},
        "llm": {},
    }

    run_all = not (args.ml_only or args.llm_only or args.shap_only)

    # 1. ML Model Evaluation
    if run_all or args.ml_only:
        print("[EVALUATION] Running ML model evaluation...")
        ml_report = run_ml_evaluation()
        consolidated["ml"] = ml_report
        print(f"[EVALUATION] ML model evaluation complete. Accuracy: {ml_report.get('accuracy', 0):.4f}")

    # 2. SHAP Explainability Evaluation
    if run_all or args.shap_only:
        print("[EVALUATION] Running SHAP explainability evaluation...")
        shap_report = run_shap_evaluation()
        consolidated["shap"] = shap_report
        print(f"[EVALUATION] SHAP evaluation complete. Success Rate: {shap_report.get('explanation_success_rate', 0)*100:.2f}%")

    # 3. LLM Multi-Agent Evaluation
    if run_all or args.llm_only:
        print("[EVALUATION] Running LLM & Agent pipeline evaluation...")
        llm_report, _ = await run_llm_evaluation(cases_limit=args.cases_limit)
        consolidated["llm"] = llm_report
        print(f"[EVALUATION] LLM evaluation complete. Overall Score: {llm_report.get('overall_score', 0)*100:.2f}%")

    # 4. Save Consolidated JSON Report
    json_path = reports_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    # 5. Generate Markdown Report
    md_path = reports_dir / "evaluation_report.md"
    generate_markdown_report(consolidated, md_path)

    # 6. Print ASCII Summary
    print_ascii_summary(consolidated)
    print(f"Reports saved to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

# AutoCare AI — Evaluation Report

*Evaluated at: 2026-08-29T11:29:44.763619+00:00*

---

## 1. ML Model Performance (Random Forest)

| Metric | Measured Value |
| :--- | :--- |
| **Model Type** | `RandomForestClassifier` |
| **Test Samples** | 7,500 |
| **Accuracy** | **0.9645** (96.45%) |
| **Precision** | **1.0000** (100.00%) |
| **Recall** | **0.9562** (95.62%) |
| **F1 Score** | **0.9776** (97.76%) |
| **ROC-AUC** | **1.0000** |
| **PR-AUC (Avg Precision)** | **1.0000** |
| **Expected Calibration Error (ECE)** | 0.1006 |
| **Brier Score** | 0.0360 |
| **False Positives / Negatives** | FP: 0 | FN: 266 |

### Confusion Matrix
```
[
  [
    1425,
    0
  ],
  [
    266,
    5809
  ]
]
```

---

## 2. LLM & Multi-Agent Pipeline Performance

| Metric | Measured Rate | Applicable Scope |
| :--- | :--- | :--- |
| **Input Extraction Accuracy** | **100.00%** | Explicit facts, spelling errors, short answers |
| **Question Relevance Rate** | **91.67%** | Dynamic & relevant questioning (no re-asking) |
| **Dataset Compliance Rate** | **97.22%** | Strictly within 19 dataset schema features |
| **Faithfulness Rate** | **100.00%** | Grounded in canonical state & predictions |
| **State Consistency Rate** | **100.00%** | Multi-turn state tracking & updates |
| **Hallucination-Free Rate** | **100.00%** | Zero invented facts / unsupported claims |
| **Response Relevance Rate** | **86.11%** | Direct answers to user intents |
| **Response Quality Rate** | **83.33%** | Clarity, tone, formatting |
| **Post-Assessment Routing** | **100.00%** | Summary & explanation navigation |
| **Judge LLM Pass Rate** | **83.33%** | Semantic LLM Judge evaluation |
| **Deterministic Compliance Rate** | **100.00%** | Objective schema & guardrails compliance |

### Evaluation Subsystem Diagnostics
- **Total Test Cases**: 36
- **LLM Judge Cases**: 36
- **Deterministic-Only Cases**: 0
- **Judge LLM Failures / Errors**: 0
- **Deterministic Fallback Count**: 0
- **Passed Cases**: 30
- **Failed Cases**: 6

### **Overall LLM Score**: `95.00%`
*Score Formula: `0.20*hallucination_free + 0.15*faithfulness + 0.15*dataset_compliance + 0.15*state_consistency + 0.15*relevance + 0.10*question_relevance + 0.10*quality`*

---

## 3. SHAP Local Explainability Quality

| Metric | Measured Value |
| :--- | :--- |
| **Total Test Scenarios** | 6 |
| **Successful Explanations** | 6 / 6 |
| **SHAP Explanation Success Rate** | **100.00%** |
| **Top Factor Validity Rate** | **100.00%** |
| **Schema Compliance Rate** | **100.00%** |

---

## 4. Overall Evaluation Summary

- **Total Cases Evaluated**: 36
- **Judge LLM Pass Rate**: **83.33%**
- **Deterministic Compliance**: **100.00%**
- **Overall System Status**: **COMPLETED WITH AUDIT FINDINGS**

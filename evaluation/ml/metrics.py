"""
ML Metrics Calculation Module
=============================
Computes strict, un-invented ML performance metrics:
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC, PR-AUC (Average Precision)
- Confusion Matrix, False Positives, False Negatives
- Expected Calibration Error (ECE), Brier Score, Class Distribution
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def calculate_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    ECE = sum_{b=1}^B (|acc(b) - conf(b)| * |B_b| / N)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    n_total = len(y_true)
    for b in range(len(prob_true)):
        bin_mask = bin_assignments == b
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            ece += np.abs(prob_true[b] - prob_pred[b]) * (bin_count / n_total)

    return float(ece)


def calculate_comprehensive_ml_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute all required ML metrics from true values, predicted classes, and probabilities.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    ece = calculate_calibration_error(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))

    # Class distribution
    total_samples = len(y_true)
    class_0_count = int(np.sum(y_true == 0))
    class_1_count = int(np.sum(y_true == 1))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "calibration_error": ece,
        "brier_score": brier,
        "class_distribution": {
            "total_samples": total_samples,
            "class_0_no_maintenance": class_0_count,
            "class_1_maintenance_required": class_1_count,
            "positive_ratio": float(class_1_count / total_samples) if total_samples > 0 else 0.0,
        },
    }

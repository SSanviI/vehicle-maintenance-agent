"""
ML Model Evaluator
==================
Evaluates the EXISTING trained Random Forest model and preprocessing pipeline
on the actual dataset. Does NOT retrain the model.

Outputs:
  evaluation/reports/ml_report.json
"""

from __future__ import annotations

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split
from backend.app.ml.preprocessing import MaintenancePreprocessor, SELECTED_FEATURES
from evaluation.ml.metrics import calculate_comprehensive_ml_metrics


def run_ml_evaluation(
    data_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    pipeline_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Load existing trained model and dataset, perform inference on the test split,
    and compute real performance metrics.
    """
    workspace_root = Path(__file__).resolve().parent.parent.parent
    data_path = data_path or (workspace_root / "data" / "vehicle_maintenance_data.csv")
    model_path = model_path or (workspace_root / "models" / "maintenance_model.joblib")
    pipeline_path = pipeline_path or (workspace_root / "models" / "preprocessing_pipeline.joblib")
    output_path = output_path or (workspace_root / "evaluation" / "reports" / "ml_report.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}")

    # 1. Load data
    df = pd.read_csv(data_path)
    target_col = "Need_Maintenance"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df[SELECTED_FEATURES]
    y = df[target_col].values

    # 2. Split (matching standard training split)
    # First 85% train+val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Load preprocessor & model
    preprocessor = MaintenancePreprocessor.load(str(pipeline_path))
    model = joblib.load(model_path)

    # 4. Preprocess test data
    if hasattr(preprocessor, "transform"):
        X_test_trans = preprocessor.transform(X_test)
    else:
        X_test_trans = preprocessor._transform(X_test)

    # 5. Predict
    y_pred = model.predict(X_test_trans)
    y_prob = model.predict_proba(X_test_trans)[:, 1]

    # 6. Calculate real metrics
    metrics = calculate_comprehensive_ml_metrics(y_test, y_pred, y_prob)

    report: Dict[str, Any] = {
        "model": "RandomForest",
        "model_type": type(model).__name__,
        "dataset_path": str(data_path.name),
        "test_samples": int(len(y_test)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "calibration_error": metrics["calibration_error"],
        "brier_score": metrics["brier_score"],
        "class_distribution": metrics["class_distribution"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 7. Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_ml_evaluation()
    print("=== ML Evaluation Report ===")
    print(json.dumps(report, indent=2))

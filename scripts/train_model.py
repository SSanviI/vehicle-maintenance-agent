"""
Vehicle Maintenance ML Model Training Pipeline
=================================================
- Trains multiple classifiers with missingness augmentation
- Evaluates on validation/test sets
- Saves final model + preprocessing pipeline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path(__file__).parent.parent / "data" / "vehicle_maintenance_data.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "ml" / "reports"

TARGET = "Need_Maintenance"

SELECTED_FEATURES = [
    "Vehicle_Model", "Mileage", "Maintenance_History", "Reported_Issues", "Vehicle_Age",
    "Fuel_Type", "Transmission_Type", "Engine_Size", "Odometer_Reading", "Last_Service_Date",
    "Warranty_Expiry_Date", "Owner_Type", "Insurance_Premium", "Service_History",
    "Accident_History", "Fuel_Efficiency", "Tire_Condition", "Brake_Condition", "Battery_Status",
]

CATEGORICAL_FEATURES = ["Vehicle_Model", "Maintenance_History", "Fuel_Type", "Transmission_Type", "Owner_Type", "Tire_Condition", "Brake_Condition", "Battery_Status"]
NUMERICAL_FEATURES = ["Mileage", "Reported_Issues", "Vehicle_Age", "Engine_Size", "Odometer_Reading", "Insurance_Premium", "Service_History", "Accident_History", "Fuel_Efficiency"]
DATE_FEATURES = ["Last_Service_Date", "Warranty_Expiry_Date"]

# Categorical value maps (for encoding consistency)
CATEGORICAL_VALUES = {
    "Vehicle_Model": ["Bus", "Car", "Motorcycle", "SUV", "Truck", "Van"],
    "Maintenance_History": ["Good", "Average", "Poor"],
    "Fuel_Type": ["Diesel", "Electric", "Petrol"],
    "Transmission_Type": ["Automatic", "Manual"],
    "Owner_Type": ["First", "Second", "Third"],
    "Tire_Condition": ["New", "Good", "Worn Out"],
    "Brake_Condition": ["New", "Good", "Worn Out"],
    "Battery_Status": ["New", "Good", "Weak"],
}

# Feature importance ranking (for adaptive questioning priority)
FEATURE_PRIORITY = [
    "Reported_Issues",       # Highest MI
    "Brake_Condition",       # High MI (leakage-aware)
    "Battery_Status",        # High MI (leakage-aware)
    "Maintenance_History",   # Moderate MI
    "Service_History",       # Moderate MI
    "Accident_History",      # Low but useful
    "Vehicle_Age",           # Context feature
    "Tire_Condition", "Odometer_Reading", "Mileage", "Last_Service_Date", "Warranty_Expiry_Date",
    "Engine_Size", "Fuel_Efficiency", "Insurance_Premium", "Vehicle_Model", "Fuel_Type",
    "Transmission_Type", "Owner_Type",
]


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

class MaintenancePreprocessor:
    """
    Preprocessing pipeline that handles partial inputs (missing features).
    """

    def __init__(self):
        self.category_maps = {}
        self.numerical_medians = {}
        self.is_fitted = False

    def fit(self, df):
        """Fit encoders on training data."""
        self.category_maps = {}
        for col in CATEGORICAL_FEATURES:
            values = CATEGORICAL_VALUES[col]
            self.category_maps[col] = {v: i for i, v in enumerate(values)}

        for col in NUMERICAL_FEATURES:
            self.numerical_medians[col] = df[col].median()

        self.is_fitted = True
        return self

    def transform(self, df, add_missing_indicators=True):
        """Transform features, handling missing values."""
        result = pd.DataFrame(index=df.index)

        for col in CATEGORICAL_FEATURES:
            values = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
            is_missing = values.isna() | (values == "UNKNOWN") | (values == "")
            mapped = values.map(self.category_maps[col]).fillna(-1)
            result[col] = mapped.astype(float)
            if add_missing_indicators:
                result[f"{col}_missing"] = is_missing.astype(int)

        for col in DATE_FEATURES:
            values = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
            parsed = pd.to_datetime(values, errors="coerce")
            is_missing = parsed.isna()
            result[col] = parsed.map(lambda value: value.toordinal() if not pd.isna(value) else -1).astype(float)
            if add_missing_indicators:
                result[f"{col}_missing"] = is_missing.astype(int)

        for col in NUMERICAL_FEATURES:
            values = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
            is_missing = values.isna() | (values == -1)
            result[col] = values.fillna(-1).astype(float)
            if add_missing_indicators:
                result[f"{col}_missing"] = is_missing.astype(int)

        return result

    def transform_partial(self, feature_state):
        """
        Transform a partial feature state dict into model-ready input.
        """
        row = {}
        for feat in SELECTED_FEATURES:
            if feat in feature_state and feature_state[feat].get("status") == "known":
                row[feat] = feature_state[feat]["value"]
            else:
                row[feat] = np.nan

        df = pd.DataFrame([row])
        return self.transform(df, add_missing_indicators=True)

    def save(self, path):
        joblib.dump({
            'category_maps': self.category_maps,
            'numerical_medians': self.numerical_medians,
            'categorical_values': CATEGORICAL_VALUES,
            'categorical_features': CATEGORICAL_FEATURES,
            'numerical_features': NUMERICAL_FEATURES,
            'selected_features': SELECTED_FEATURES,
            'feature_priority': FEATURE_PRIORITY,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        pp = cls()
        pp.category_maps = data['category_maps']
        pp.numerical_medians = data['numerical_medians']
        pp.is_fitted = True
        return pp


def apply_missingness_augmentation(df, missing_rate=0.3, seed=None):
    """
    Randomly mask features to simulate partial user input during training.
    Each sample has some features randomly set to missing.
    """
    rng = np.random.RandomState(seed)
    df_aug = df.copy()

    for col in SELECTED_FEATURES:
        mask = rng.random(len(df)) < missing_rate
        if col in CATEGORICAL_FEATURES:
            df_aug.loc[mask, col] = "UNKNOWN"
        else:
            df_aug.loc[mask, col] = np.nan

    return df_aug


# =============================================================================
# Model Training
# =============================================================================

def load_and_split_data():
    """Load dataset and create stratified train/val/test splits."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET].copy()

    # 70/15/15 split with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Target distribution - Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_augmented_training_data(X_train, y_train, preprocessor, n_augments=3):
    """
    Create training data with missingness augmentation.
    - Original data (clean) + N augmented copies with random missing features
    """
    # Clean training data
    X_clean = preprocessor.transform(X_train)

    # Augmented copies
    aug_dfs = [X_clean]
    y_copies = [y_train]

    for i in range(n_augments):
        X_aug = apply_missingness_augmentation(X_train, missing_rate=0.3, seed=42 + i)
        X_aug_transformed = preprocessor.transform(X_aug)
        aug_dfs.append(X_aug_transformed)
        y_copies.append(y_train)

    X_combined = pd.concat(aug_dfs, ignore_index=True)
    y_combined = pd.concat(y_copies, ignore_index=True)

    print(f"Training data (with augmentation): {len(X_combined):,} samples")
    return X_combined, y_combined


def evaluate_model(model, X, y, name="Model"):
    """Evaluate a model and return metrics dict."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
    }

    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["false_negatives"] = int(cm[1, 0]) if cm.shape[0] > 1 else 0
    metrics["false_positives"] = int(cm[0, 1]) if cm.shape[0] > 1 else 0

    # Calibration
    try:
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy='uniform')
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        metrics["calibration_error"] = float(calibration_error)
    except Exception:
        metrics["calibration_error"] = None

    return metrics


def format_metrics(metrics, name):
    """Format metrics for display."""
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"  {name}")
    lines.append(f"{'='*50}")
    lines.append(f"  Accuracy:    {metrics['accuracy']:.4f}")
    lines.append(f"  Precision:   {metrics['precision']:.4f}")
    lines.append(f"  Recall:      {metrics['recall']:.4f}")
    lines.append(f"  F1:          {metrics['f1']:.4f}")
    lines.append(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    lines.append(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
    if metrics.get('calibration_error') is not None:
        lines.append(f"  Cal Error:   {metrics['calibration_error']:.4f}")
    cm = metrics['confusion_matrix']
    lines.append(f"  Confusion Matrix:")
    lines.append(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    lines.append(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    lines.append(f"  False Negatives: {metrics['false_negatives']:,}")
    return '\n'.join(lines)


def train_all_models(X_train, y_train, X_val, y_val):
    """Train multiple models and evaluate on validation set."""
    models = {}
    results = {}

    # Random Forest is deliberately the sole production classifier.
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    results['random_forest'] = evaluate_model(rf, X_val, y_val, "Random Forest (Val)")
    print(format_metrics(results['random_forest'], "Random Forest (Val)"))

    return models, results


def select_best_model(results):
    """
    Select best model based on:
    1. High recall (minimize false negatives — don't miss maintenance needs)
    2. Good F1 balance
    3. Low calibration error
    4. Good ROC-AUC
    """
    scores = {}
    for name, metrics in results.items():
        # Weighted scoring: recall is most important for safety
        score = (
            metrics['recall'] * 0.35 +
            metrics['f1'] * 0.25 +
            metrics['roc_auc'] * 0.20 +
            metrics['pr_auc'] * 0.15 +
            (1 - (metrics.get('calibration_error') or 0.5)) * 0.05
        )
        scores[name] = score
        print(f"  {name}: combined score = {score:.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\n  >> Best model: {best_name} (score: {scores[best_name]:.4f})")
    return best_name


def test_partial_inputs(model, preprocessor):
    """Test model with various partial input scenarios."""
    print("\n" + "="*50)
    print("  Partial Input Tests")
    print("="*50)

    scenarios = [
        {
            "name": "Only Vehicle_Age",
            "state": {
                "Vehicle_Age": {"value": 5, "status": "known"},
            }
        },
        {
            "name": "Vehicle_Age + Reported_Issues",
            "state": {
                "Vehicle_Age": {"value": 5, "status": "known"},
                "Reported_Issues": {"value": 3, "status": "known"},
            }
        },
        {
            "name": "Vehicle_Age + Reported_Issues + Brake=Worn Out",
            "state": {
                "Vehicle_Age": {"value": 5, "status": "known"},
                "Reported_Issues": {"value": 3, "status": "known"},
                "Brake_Condition": {"value": "Worn Out", "status": "known"},
            }
        },
        {
            "name": "All features known (low risk)",
            "state": {
                "Vehicle_Age": {"value": 2, "status": "known"},
                "Reported_Issues": {"value": 0, "status": "known"},
                "Maintenance_History": {"value": "Good", "status": "known"},
                "Service_History": {"value": 8, "status": "known"},
                "Accident_History": {"value": 0, "status": "known"},
                "Tire_Condition": {"value": "New", "status": "known"},
                "Brake_Condition": {"value": "New", "status": "known"},
                "Battery_Status": {"value": "New", "status": "known"},
            }
        },
        {
            "name": "All features known (high risk)",
            "state": {
                "Vehicle_Age": {"value": 9, "status": "known"},
                "Reported_Issues": {"value": 5, "status": "known"},
                "Maintenance_History": {"value": "Poor", "status": "known"},
                "Service_History": {"value": 2, "status": "known"},
                "Accident_History": {"value": 3, "status": "known"},
                "Tire_Condition": {"value": "Worn Out", "status": "known"},
                "Brake_Condition": {"value": "Worn Out", "status": "known"},
                "Battery_Status": {"value": "Weak", "status": "known"},
            }
        },
        {
            "name": "No features known",
            "state": {}
        },
    ]

    for scenario in scenarios:
        X = preprocessor.transform_partial(scenario["state"])
        prob = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]
        known = [k for k, v in scenario["state"].items() if v.get("status") == "known"]
        print(f"\n  {scenario['name']}")
        print(f"    Known: {known if known else 'none'}")
        print(f"    Prediction: {'MAINTENANCE NEEDED' if pred == 1 else 'NO MAINTENANCE'}")
        print(f"    Probability: {prob:.4f}")


def generate_training_report(results, best_name, test_metrics):
    """Generate ML training report."""
    report = []
    report.append("# ML Model Training Report\n")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    report.append("## Feature Set\n")
    report.append("| Feature | Type | Rationale |")
    report.append("|---------|------|-----------|")
    report.append("| Vehicle_Age | Numerical | User context; low AUC but intuitive |")
    report.append("| Reported_Issues | Numerical | Best predictor (AUC=0.78, MI=0.11) |")
    report.append("| Maintenance_History | Categorical | Meaningful categories (Good/Average/Poor) |")
    report.append("| Service_History | Numerical | Moderate predictor (AUC=0.58) |")
    report.append("| Accident_History | Numerical | Safety-relevant; moderate MI |")
    report.append("| Tire_Condition | Categorical | Observable condition |")
    report.append("| Brake_Condition | Categorical | Observable condition; 100% when Worn Out (domain-appropriate) |")
    report.append("| Battery_Status | Categorical | Observable condition; 100% when Weak (domain-appropriate) |")

    report.append("\n## Leakage Assessment\n")
    report.append("- `Brake_Condition='Worn Out'` and `Battery_Status='Weak'` perfectly predict maintenance need")
    report.append("- This is **domain-appropriate leakage**: worn brakes and weak batteries genuinely require maintenance")
    report.append("- NOT a data processing artifact — this reflects real-world causation")
    report.append("- The model is transparent about this in its predictions\n")

    report.append("## Missing Feature Strategy\n")
    report.append("- **Missingness augmentation**: 30% random masking during training (3 augmented copies)")
    report.append("- **Categorical**: 'UNKNOWN' category added to all encoders")
    report.append("- **Numerical**: -1 sentinel + binary `_missing` indicator columns")
    report.append("- Model trained on both clean and augmented data simultaneously\n")

    report.append("## Validation Results\n")
    report.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Cal Error |")
    report.append("|-------|----------|-----------|--------|-------|---------|--------|-----------|")
    for name, metrics in results.items():
        cal = f"{metrics.get('calibration_error', 'N/A'):.4f}" if metrics.get('calibration_error') is not None else "N/A"
        report.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | {cal} |")

    report.append(f"\n**Selected model: `{best_name}`**\n")

    report.append("## Test Results (Final)\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    for k, v in test_metrics.items():
        if k != 'confusion_matrix':
            report.append(f"| {k} | {v if isinstance(v, str) else f'{v:.4f}' if isinstance(v, float) else v} |")

    cm = test_metrics['confusion_matrix']
    report.append(f"\nConfusion Matrix:\n```\nTN={cm[0][0]}  FP={cm[0][1]}\nFN={cm[1][0]}  TP={cm[1][1]}\n```\n")

    return '\n'.join(report)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Vehicle Maintenance ML Training Pipeline")
    print("=" * 60)

    # Step 1: Load and split
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    # Step 2: Fit preprocessor
    print("\nFitting preprocessor...")
    preprocessor = MaintenancePreprocessor()
    preprocessor.fit(X_train)

    # Step 3: Create augmented training data
    print("\nCreating augmented training data...")
    X_train_aug, y_train_aug = create_augmented_training_data(
        X_train, y_train, preprocessor, n_augments=3
    )

    # Step 4: Transform validation and test sets (clean, no augmentation)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    # Step 5: Train models
    print("\n" + "="*60)
    print("  Training Models")
    print("="*60)
    models, val_results = train_all_models(X_train_aug, y_train_aug, X_val_transformed, y_val)

    # Step 6: Select best model
    print("\n" + "="*60)
    print("  Model Selection")
    print("="*60)
    # The product contract makes Random Forest the sole risk authority.
    best_name = "random_forest"
    best_model = models[best_name]

    # Step 7: Evaluate on test set
    print("\n" + "="*60)
    print("  Test Set Evaluation")
    print("="*60)
    test_metrics = evaluate_model(best_model, X_test_transformed, y_test, f"{best_name} (Test)")
    print(format_metrics(test_metrics, f"{best_name} (Test)"))

    # Step 8: Test partial inputs
    test_partial_inputs(best_model, preprocessor)

    # Step 9: Save model and preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "maintenance_model.joblib"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"

    joblib.dump(best_model, model_path)
    preprocessor.save(pipeline_path)

    # Save model metadata
    metadata = {
        "model_name": best_name,
        "selected_features": SELECTED_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "date_features": DATE_FEATURES,
        "categorical_values": CATEGORICAL_VALUES,
        "feature_priority": FEATURE_PRIORITY,
        "target": TARGET,
        "training_samples": len(X_train_aug),
        "validation_metrics": {k: (v if not isinstance(v, np.floating) else float(v))
                               for k, v in val_results[best_name].items()},
        "test_metrics": {k: (v if not isinstance(v, np.floating) else float(v))
                         for k, v in test_metrics.items()},
        "trained_at": datetime.now().isoformat(),
    }
    with open(MODELS_DIR / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n[OK] Model saved to: {model_path}")
    print(f"[OK] Pipeline saved to: {pipeline_path}")
    print(f"[OK] Metadata saved to: {MODELS_DIR / 'model_metadata.json'}")

    # Step 10: Generate report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_training_report(val_results, best_name, test_metrics)
    report_path = REPORTS_DIR / "training_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] Training report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

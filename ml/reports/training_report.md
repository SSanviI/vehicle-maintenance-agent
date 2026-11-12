# ML Model Training Report

*Generated: 2026-08-26 16:25:21*

## Feature Set

| Feature | Type | Rationale |
|---------|------|-----------|
| Vehicle_Age | Numerical | User context; low AUC but intuitive |
| Reported_Issues | Numerical | Best predictor (AUC=0.78, MI=0.11) |
| Maintenance_History | Categorical | Meaningful categories (Good/Average/Poor) |
| Service_History | Numerical | Moderate predictor (AUC=0.58) |
| Accident_History | Numerical | Safety-relevant; moderate MI |
| Tire_Condition | Categorical | Observable condition |
| Brake_Condition | Categorical | Observable condition; 100% when Worn Out (domain-appropriate) |
| Battery_Status | Categorical | Observable condition; 100% when Weak (domain-appropriate) |

## Leakage Assessment

- `Brake_Condition='Worn Out'` and `Battery_Status='Weak'` perfectly predict maintenance need
- This is **domain-appropriate leakage**: worn brakes and weak batteries genuinely require maintenance
- NOT a data processing artifact — this reflects real-world causation
- The model is transparent about this in its predictions

## Missing Feature Strategy

- **Missingness augmentation**: 30% random masking during training (3 augmented copies)
- **Categorical**: 'UNKNOWN' category added to all encoders
- **Numerical**: -1 sentinel + binary `_missing` indicator columns
- Model trained on both clean and augmented data simultaneously

## Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Cal Error |
|-------|----------|-----------|--------|-------|---------|--------|-----------|
| random_forest | 0.9651 | 1.0000 | 0.9569 | 0.9780 | 1.0000 | 1.0000 | 0.3029 |

**Selected model: `random_forest`**

## Test Results (Final)

| Metric | Value |
|--------|-------|
| accuracy | 0.9628 |
| precision | 1.0000 |
| recall | 0.9541 |
| f1 | 0.9765 |
| roc_auc | 1.0000 |
| pr_auc | 1.0000 |
| false_negatives | 279 |
| false_positives | 0 |
| calibration_error | 0.3031 |

Confusion Matrix:
```
TN=1426  FP=0
FN=279  TP=5795
```

"""
Vehicle Maintenance Dataset — Exploratory Data Analysis & Leakage Report
=========================================================================
Generates docs/eda_report.md with full findings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "vehicle_maintenance_data.csv"
REPORT_PATH = Path(__file__).parent.parent / "docs" / "eda_report.md"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def basic_info(df):
    lines = []
    lines.append("## 1. Basic Dataset Information\n")
    lines.append(f"- **Rows**: {df.shape[0]:,}")
    lines.append(f"- **Columns**: {df.shape[1]}")
    lines.append(f"- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    lines.append("")

    lines.append("### Column Names & Data Types\n")
    lines.append("| Column | Dtype | Non-Null | Null % |")
    lines.append("|--------|-------|----------|--------|")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        lines.append(f"| {col} | {df[col].dtype} | {df[col].notna().sum():,} | {null_pct:.1f}% |")
    lines.append("")

    dup = df.duplicated().sum()
    lines.append(f"### Duplicate Rows: **{dup:,}** ({dup/len(df)*100:.2f}%)\n")
    return lines


def target_analysis(df):
    lines = []
    lines.append("## 2. Target Variable: `Need_Maintenance`\n")

    vc = df['Need_Maintenance'].value_counts()
    lines.append("| Value | Count | Percentage |")
    lines.append("|-------|-------|------------|")
    for val, count in vc.items():
        lines.append(f"| {val} | {count:,} | {count/len(df)*100:.1f}% |")
    lines.append("")

    ratio = vc.min() / vc.max()
    if ratio > 0.8:
        lines.append("> **Observation**: Target is fairly balanced.\n")
    elif ratio > 0.4:
        lines.append("> **Observation**: Target has moderate imbalance.\n")
    else:
        lines.append(f"> **⚠️ Warning**: Target is imbalanced (ratio {ratio:.2f}).\n")
    return lines


def numerical_analysis(df):
    lines = []
    lines.append("## 3. Numerical Feature Analysis\n")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Need_Maintenance' in num_cols:
        num_cols.remove('Need_Maintenance')

    if not num_cols:
        lines.append("No numerical features found.\n")
        return lines

    desc = df[num_cols].describe().T
    lines.append("| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |")
    lines.append("|---------|------|-----|-----|-----|-----|-----|-----|")
    for col in desc.index:
        row = desc.loc[col]
        lines.append(f"| {col} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} |")
    lines.append("")

    # Outlier detection (IQR method)
    lines.append("### Outliers (IQR Method)\n")
    lines.append("| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % |")
    lines.append("|---------|-------------|-------------|---------------|-----------|")
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        lines.append(f"| {col} | {lower:.2f} | {upper:.2f} | {outliers:,} | {outliers/len(df)*100:.1f}% |")
    lines.append("")
    return lines


def categorical_analysis(df):
    lines = []
    lines.append("## 4. Categorical Feature Analysis\n")

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        lines.append(f"### `{col}`\n")
        vc = df[col].value_counts()
        lines.append(f"- Unique values: **{df[col].nunique()}**")
        lines.append("")
        lines.append("| Value | Count | % |")
        lines.append("|-------|-------|---|")
        for val, count in vc.head(15).items():
            lines.append(f"| {val} | {count:,} | {count/len(df)*100:.1f}% |")
        if len(vc) > 15:
            lines.append(f"| ... | ... | ... |")
        lines.append("")
    return lines


def target_relationship(df):
    """Analyze relationship between each feature and the target."""
    lines = []
    lines.append("## 5. Feature-Target Relationships\n")
    target = 'Need_Maintenance'

    # Categorical features vs target
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        lines.append("### Categorical Features vs Target\n")
        for col in cat_cols:
            ct = pd.crosstab(df[col], df[target], normalize='index') * 100
            lines.append(f"#### `{col}`\n")
            lines.append(f"| {col} | % Need_Maintenance=0 | % Need_Maintenance=1 |")
            lines.append("|-------|----------------------|----------------------|")
            for idx in ct.index:
                v0 = ct.loc[idx, 0] if 0 in ct.columns else 0
                v1 = ct.loc[idx, 1] if 1 in ct.columns else 0
                lines.append(f"| {idx} | {v0:.1f}% | {v1:.1f}% |")
            lines.append("")

    # Numerical features vs target
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    if num_cols:
        lines.append("### Numerical Features vs Target\n")
        lines.append("| Feature | Mean (0) | Mean (1) | Diff | Diff % |")
        lines.append("|---------|----------|----------|------|--------|")
        for col in num_cols:
            m0 = df[df[target] == 0][col].mean()
            m1 = df[df[target] == 1][col].mean()
            diff = m1 - m0
            diff_pct = (diff / m0 * 100) if m0 != 0 else float('inf')
            lines.append(f"| {col} | {m0:.2f} | {m1:.2f} | {diff:.2f} | {diff_pct:.1f}% |")
        lines.append("")
    return lines


def leakage_analysis(df):
    """Critical leakage analysis for suspect features."""
    lines = []
    lines.append("## 6. ⚠️ LEAKAGE ANALYSIS (Critical)\n")
    lines.append("Investigating whether features are suspiciously deterministic with the target.\n")

    target = 'Need_Maintenance'
    suspect_features = [
        'Brake_Condition', 'Battery_Status', 'Reported_Issues',
        'Maintenance_History', 'Service_History', 'Tire_Condition'
    ]

    existing_suspects = [f for f in suspect_features if f in df.columns]

    for col in existing_suspects:
        lines.append(f"### `{col}`\n")

        if df[col].dtype == 'object':
            # Categorical: check if any value perfectly predicts the target
            ct = pd.crosstab(df[col], df[target])
            ct_norm = pd.crosstab(df[col], df[target], normalize='index') * 100

            lines.append("| Value | Count(0) | Count(1) | P(target=1) |")
            lines.append("|-------|----------|----------|-------------|")
            for idx in ct.index:
                c0 = ct.loc[idx, 0] if 0 in ct.columns else 0
                c1 = ct.loc[idx, 1] if 1 in ct.columns else 0
                p1 = ct_norm.loc[idx, 1] if 1 in ct_norm.columns else 0
                lines.append(f"| {idx} | {c0:,} | {c1:,} | {p1:.1f}% |")
            lines.append("")

            # Check if any single value has >95% association with one class
            for idx in ct_norm.index:
                for col_val in ct_norm.columns:
                    if ct_norm.loc[idx, col_val] > 95:
                        lines.append(f"> **🚨 LEAKAGE RISK**: `{col}` = `{idx}` predicts target={col_val} with {ct_norm.loc[idx, col_val]:.1f}% accuracy\n")
        else:
            # Numerical: check correlation
            corr = df[col].corr(df[target])
            lines.append(f"- Correlation with target: **{corr:.4f}**")

            # Check if there's a threshold that perfectly separates
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(df[target], df[col])
                lines.append(f"- Single-feature AUC: **{auc:.4f}**")
                if auc > 0.95 or auc < 0.05:
                    lines.append(f"\n> **🚨 LEAKAGE RISK**: `{col}` alone achieves AUC={auc:.4f}\n")
                elif auc > 0.85 or auc < 0.15:
                    lines.append(f"\n> **⚠️ SUSPICIOUS**: `{col}` alone achieves AUC={auc:.4f}\n")
            except Exception:
                pass
            lines.append("")

    # Mutual Information
    lines.append("### Mutual Information Scores\n")
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    features = [c for c in df_encoded.columns if c != target]
    X = df_encoded[features].fillna(0)
    y = df_encoded[target]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': features, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)

    lines.append("| Feature | Mutual Information |")
    lines.append("|---------|-------------------|")
    for _, row in mi_df.iterrows():
        flag = " 🚨" if row['MI_Score'] > 0.5 else (" ⚠️" if row['MI_Score'] > 0.3 else "")
        lines.append(f"| {row['Feature']} | {row['MI_Score']:.4f}{flag} |")
    lines.append("")

    # Synthetic data check
    lines.append("### Synthetic Data Check\n")

    # Check for perfect uniform distributions in categoricals
    synthetic_signs = []
    for col in df.select_dtypes(include=['object']).columns:
        vc = df[col].value_counts(normalize=True)
        # If all categories are within 5% of each other, possibly synthetic
        if vc.std() < 0.02:
            synthetic_signs.append(f"- `{col}`: Nearly uniform distribution (std={vc.std():.4f})")

    # Check for suspiciously round numbers in numericals
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.01 and df[col].nunique() > 2:
            synthetic_signs.append(f"- `{col}`: Very low unique ratio ({unique_ratio:.4f})")

    if synthetic_signs:
        lines.append("> **Synthetic data indicators detected:**\n")
        lines.extend(synthetic_signs)
    else:
        lines.append("No strong indicators of synthetic data detected.\n")
    lines.append("")

    # Overall leakage summary
    lines.append("### Leakage Summary\n")
    lines.append("Based on the analysis above, features are classified as:\n")
    lines.append("| Feature | Risk Level | Recommendation |")
    lines.append("|---------|------------|----------------|")

    for col in existing_suspects:
        if df[col].dtype == 'object':
            ct_norm = pd.crosstab(df[col], df[target], normalize='index') * 100
            max_assoc = ct_norm.max().max()
            if max_assoc > 95:
                lines.append(f"| {col} | 🚨 HIGH | Likely leaks target — investigate before using |")
            elif max_assoc > 80:
                lines.append(f"| {col} | ⚠️ MEDIUM | Strong predictor — may be legitimate or leaky |")
            else:
                lines.append(f"| {col} | ✅ LOW | Appears safe to use |")
        else:
            try:
                auc = roc_auc_score(df[target], df[col])
                if auc > 0.9:
                    lines.append(f"| {col} | 🚨 HIGH | Single-feature AUC={auc:.3f} |")
                elif auc > 0.75:
                    lines.append(f"| {col} | ⚠️ MEDIUM | Moderate predictive power |")
                else:
                    lines.append(f"| {col} | ✅ LOW | Normal predictive power |")
            except Exception:
                lines.append(f"| {col} | ❓ UNKNOWN | Could not compute |")
    lines.append("")

    return lines


def correlation_analysis(df):
    lines = []
    lines.append("## 7. Correlation Analysis\n")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        lines.append("Not enough numerical features for correlation analysis.\n")
        return lines

    corr = df[num_cols].corr()

    # Show strong correlations (>0.5)
    lines.append("### Strong Correlations (|r| > 0.5)\n")
    lines.append("| Feature 1 | Feature 2 | Correlation |")
    lines.append("|-----------|-----------|-------------|")
    found = False
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                lines.append(f"| {num_cols[i]} | {num_cols[j]} | {r:.4f} |")
                found = True
    if not found:
        lines.append("| — | — | No strong correlations found |")
    lines.append("")
    return lines


def feature_selection_recommendation(df):
    """Based on the analysis, recommend the final feature set."""
    lines = []
    lines.append("## 8. Feature Selection Recommendation\n")
    lines.append("### Candidate Features (from requirements)\n")
    lines.append("```")
    lines.append("Maintenance_History, Reported_Issues, Vehicle_Age,")
    lines.append("Service_History, Accident_History, Tire_Condition,")
    lines.append("Brake_Condition, Battery_Status")
    lines.append("```\n")

    lines.append("### Analysis per Feature\n")

    target = 'Need_Maintenance'
    candidates = {
        'Vehicle_Age': 'Numerical — vehicle age in years',
        'Reported_Issues': 'Numerical — count of reported issues',
        'Maintenance_History': 'Categorical — Good/Average/Poor',
        'Service_History': 'Numerical — number of service records',
        'Accident_History': 'Numerical — number of accidents',
        'Tire_Condition': 'Categorical — New/Good/Worn Out',
        'Brake_Condition': 'Categorical — New/Good/Worn Out',
        'Battery_Status': 'Categorical — New/Good/Weak',
    }

    for feat, desc in candidates.items():
        if feat not in df.columns:
            lines.append(f"#### `{feat}` — NOT IN DATASET\n")
            continue

        lines.append(f"#### `{feat}` — {desc}\n")

        if df[feat].dtype == 'object':
            ct_norm = pd.crosstab(df[feat], df[target], normalize='index') * 100
            max_assoc = ct_norm.max().max()
            if max_assoc > 95:
                lines.append(f"- ⚠️ Max single-value → target association: {max_assoc:.1f}%")
                lines.append(f"- **RISK**: May leak the target. Consider removing or using with caution.\n")
            elif max_assoc > 80:
                lines.append(f"- Strong predictor: max association {max_assoc:.1f}%")
                lines.append(f"- **STATUS**: Usable, but monitor for leakage.\n")
            else:
                lines.append(f"- Moderate predictor: max association {max_assoc:.1f}%")
                lines.append(f"- **STATUS**: Safe to include.\n")
        else:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(df[target], df[feat])
                corr = df[feat].corr(df[target])
                lines.append(f"- Correlation with target: {corr:.4f}")
                lines.append(f"- Single-feature AUC: {auc:.4f}")
                if auc > 0.9:
                    lines.append(f"- **RISK**: Suspiciously high. Possible leakage.\n")
                elif auc > 0.7:
                    lines.append(f"- **STATUS**: Good predictor. Include.\n")
                else:
                    lines.append(f"- **STATUS**: Moderate predictor. Include.\n")
            except Exception:
                lines.append(f"- Could not compute AUC.\n")

    # Non-candidate features analysis
    lines.append("### Non-candidate Features (not in initial set)\n")
    non_candidates = [c for c in df.columns if c not in candidates and c != target]
    for col in non_candidates:
        lines.append(f"- **`{col}`**: Not selected. ")
        if col in ['Mileage', 'Odometer_Reading']:
            lines.append("Highly correlated with Vehicle_Age. Redundant.")
        elif col in ['Last_Service_Date', 'Warranty_Expiry_Date']:
            lines.append("Date features — require complex engineering with uncertain value.")
        elif col in ['Vehicle_Model', 'Fuel_Type', 'Transmission_Type']:
            lines.append("Vehicle characteristics — may add some value but increases complexity.")
        elif col in ['Engine_Size', 'Owner_Type', 'Insurance_Premium', 'Fuel_Efficiency']:
            lines.append("Secondary feature — not directly related to maintenance need.")
        lines.append("")

    return lines


def main():
    print("=" * 60)
    print("Vehicle Maintenance Dataset — EDA")
    print("=" * 60)

    df = load_data()
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    report_lines = []
    report_lines.append("# Exploratory Data Analysis — Vehicle Maintenance Dataset\n")
    report_lines.append(f"*Generated automatically by `scripts/eda.py`*\n")
    report_lines.append(f"*Dataset: {DATA_PATH.name} — {df.shape[0]:,} rows, {df.shape[1]} columns*\n")
    report_lines.append("---\n")

    # Run all analyses
    sections = [
        ("Basic Info", basic_info),
        ("Target Analysis", target_analysis),
        ("Numerical Analysis", numerical_analysis),
        ("Categorical Analysis", categorical_analysis),
        ("Target Relationships", target_relationship),
        ("Leakage Analysis", leakage_analysis),
        ("Correlations", correlation_analysis),
        ("Feature Selection", feature_selection_recommendation),
    ]

    for name, func in sections:
        print(f"  Analyzing: {name}...")
        report_lines.extend(func(df))
        report_lines.append("---\n")

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n[OK] EDA report saved to: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

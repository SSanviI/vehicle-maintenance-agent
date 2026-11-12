"""Comprehensive leakage analysis for the vehicle maintenance dataset."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("data/vehicle_maintenance_data.csv")

print("=" * 60)
print("  LEAKAGE & FEATURE ANALYSIS")
print("=" * 60)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target distribution: {df['Need_Maintenance'].value_counts().to_dict()}")
print(f"Target rate: {df['Need_Maintenance'].mean():.4f}")

# 1. Deterministic leakage checks
print("\n" + "=" * 60)
print("  DETERMINISTIC LEAKAGE CHECKS")
print("=" * 60)

# Brake
worn_brake = df[df['Brake_Condition'] == 'Worn Out']
all_maint = (worn_brake['Need_Maintenance'] == 1).all()
print(f"Brake_Condition='Worn Out': N={len(worn_brake)}, ALL Need_Maintenance=1? {all_maint}")

# Battery
weak_bat = df[df['Battery_Status'] == 'Weak']
all_maint_bat = (weak_bat['Need_Maintenance'] == 1).all()
print(f"Battery_Status='Weak':     N={len(weak_bat)}, ALL Need_Maintenance=1? {all_maint_bat}")

# Reported_Issues >= 4
hi = df[df['Reported_Issues'] >= 4]
all_maint_hi = (hi['Need_Maintenance'] == 1).all()
print(f"Reported_Issues>=4:        N={len(hi)}, ALL Need_Maintenance=1? {all_maint_hi}")

# Reported_Issues == 3
iss3 = df[df['Reported_Issues'] == 3]
print(f"Reported_Issues==3:        N={len(iss3)}, maint_rate={iss3['Need_Maintenance'].mean():.4f}")

# Tire
worn_tire = df[df['Tire_Condition'] == 'Worn Out']
print(f"Tire_Condition='Worn Out': N={len(worn_tire)}, maint_rate={worn_tire['Need_Maintenance'].mean():.4f}")

# 2. Check overlap of leak features
print("\n" + "=" * 60)
print("  LEAKAGE OVERLAP")
print("=" * 60)

leak_mask = (
    (df['Brake_Condition'] == 'Worn Out') | 
    (df['Battery_Status'] == 'Weak') | 
    (df['Reported_Issues'] >= 4)
)
print(f"Rows with ANY leak condition: {leak_mask.sum()}")
print(f"Maint rate with leak: {df[leak_mask]['Need_Maintenance'].mean():.4f}")

no_leak = df[~leak_mask]
print(f"Rows WITHOUT any leak condition: {len(no_leak)}")
print(f"Maint rate without leak: {no_leak['Need_Maintenance'].mean():.4f}")
print(f"Need_Maintenance dist (no leak): {no_leak['Need_Maintenance'].value_counts().to_dict()}")

# 3. Feature analysis on NON-LEAKING subset
print("\n" + "=" * 60)
print("  FEATURE ANALYSIS (NON-LEAKING SUBSET)")
print("=" * 60)

categorical_feats = ['Maintenance_History', 'Tire_Condition']
numerical_feats = ['Vehicle_Age', 'Service_History', 'Accident_History', 'Reported_Issues']

for col in categorical_feats:
    ct = pd.crosstab(no_leak[col], no_leak['Need_Maintenance'], normalize='index')
    print(f"\n{col} (non-leaking subset):")
    print(ct.to_string())

for col in numerical_feats:
    try:
        auc = roc_auc_score(no_leak['Need_Maintenance'], no_leak[col])
        mean_0 = no_leak[no_leak['Need_Maintenance']==0][col].mean()
        mean_1 = no_leak[no_leak['Need_Maintenance']==1][col].mean()
        print(f"\n{col}: AUC={auc:.4f}, mean(0)={mean_0:.2f}, mean(1)={mean_1:.2f}")
    except Exception as e:
        print(f"\n{col}: Error - {e}")

# 4. Mutual information
print("\n" + "=" * 60)
print("  MUTUAL INFORMATION (FULL DATASET)")
print("=" * 60)

# Encode categoricals for MI calculation
df_enc = df.copy()
for col in ['Brake_Condition', 'Battery_Status', 'Tire_Condition', 'Maintenance_History']:
    df_enc[col] = df_enc[col].astype('category').cat.codes

features_for_mi = ['Vehicle_Age', 'Reported_Issues', 'Maintenance_History', 
                   'Service_History', 'Accident_History', 'Tire_Condition',
                   'Brake_Condition', 'Battery_Status']

mi_scores = mutual_info_classif(
    df_enc[features_for_mi], df['Need_Maintenance'], 
    discrete_features=[False, False, True, False, False, True, True, True],
    random_state=42
)

for feat, mi in sorted(zip(features_for_mi, mi_scores), key=lambda x: -x[1]):
    print(f"  {feat:25s}: MI = {mi:.4f}")

# 5. Target was likely generated using rules
print("\n" + "=" * 60)
print("  SYNTHETIC TARGET DETECTION")
print("=" * 60)

# Check if the target can be reconstructed from a simple rule
rule_pred = (
    (df['Brake_Condition'] == 'Worn Out') | 
    (df['Battery_Status'] == 'Weak') | 
    (df['Reported_Issues'] >= 4) |
    (df['Reported_Issues'] == 3)
)
agreement = (rule_pred == (df['Need_Maintenance'] == 1))
print(f"Simple rule agreement: {agreement.mean():.4f}")
print(f"  Rule=1, Target=1: {((rule_pred) & (df['Need_Maintenance']==1)).sum()}")
print(f"  Rule=1, Target=0: {((rule_pred) & (df['Need_Maintenance']==0)).sum()}")
print(f"  Rule=0, Target=1: {((~rule_pred) & (df['Need_Maintenance']==1)).sum()}")
print(f"  Rule=0, Target=0: {((~rule_pred) & (df['Need_Maintenance']==0)).sum()}")

# Also check Issues==3 separately
iss3_mask = (df['Reported_Issues'] == 3) & (df['Brake_Condition'] != 'Worn Out') & (df['Battery_Status'] != 'Weak')
iss3_only = df[iss3_mask]
print(f"\nReported_Issues==3 (no brake/battery leak): N={len(iss3_only)}, rate={iss3_only['Need_Maintenance'].mean():.4f}")

# When NO deterministic condition and Issues < 3
clean_mask = (
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] < 3)
)
clean = df[clean_mask]
print(f"\nClean subset (Issues<3, no brake/battery leak): N={len(clean)}, rate={clean['Need_Maintenance'].mean():.4f}")

# Try even further: Issues < 2
clean2 = df[(df['Brake_Condition'] != 'Worn Out') & (df['Battery_Status'] != 'Weak') & (df['Reported_Issues'] < 2)]
print(f"Clean subset (Issues<2, no brake/battery leak): N={len(clean2)}, rate={clean2['Need_Maintenance'].mean():.4f}")

# 6. Permutation importance simulation
print("\n" + "=" * 60)
print("  CONCLUSION")
print("=" * 60)
print("Brake_Condition='Worn Out': DETERMINISTIC (100% → maintenance=1)")
print("Battery_Status='Weak':      DETERMINISTIC (100% → maintenance=1)")  
print("Reported_Issues>=4:          DETERMINISTIC (100% → maintenance=1)")
print("Reported_Issues==3:          NEAR-DETERMINISTIC (87% → maintenance=1)")
print("")
print("These features were clearly used to GENERATE the synthetic target.")
print("Using them as predictors creates circular logic / leakage.")
print("")
print("RECOMMENDATION:")
print("  REMOVE: Brake_Condition, Battery_Status from prediction features")
print("  KEEP but TRANSFORM: Reported_Issues (cap or bin)")
print("  KEEP: Maintenance_History, Vehicle_Age, Service_History, Accident_History, Tire_Condition")

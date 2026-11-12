"""Deep analysis of Service_History and all features for leakage/semantic issues."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/vehicle_maintenance_data.csv")

print("=" * 60)
print("  SERVICE_HISTORY DEEP ANALYSIS")
print("=" * 60)

print(f"Range: {df['Service_History'].min()} - {df['Service_History'].max()}")
print(f"Unique values: {sorted(df['Service_History'].unique())}")
print(f"\nDistribution:")
print(df['Service_History'].value_counts().sort_index().to_string())

print("\n=== Target rate by Service_History ===")
for v in sorted(df['Service_History'].unique()):
    s = df[df['Service_History'] == v]
    rate = s['Need_Maintenance'].mean()
    print(f"  SH={v:2d}: N={len(s):5d}, maint_rate={rate:.4f}")

# AUC
auc = roc_auc_score(df['Need_Maintenance'], df['Service_History'])
print(f"\nService_History AUC vs Need_Maintenance: {auc:.4f}")

# Is higher service = higher maintenance? (counter-intuitive)
print("\n=== SEMANTIC ANALYSIS ===")
print("Direction: HIGHER Service_History -> HIGHER maintenance rate")
print("This is COUNTER-INTUITIVE if Service_History means 'number of services'")
print("Possible interpretations:")
print("  1. Vehicles that need more maintenance get serviced more (reverse causality)")
print("  2. Service_History is a SCORE (1=few, 10=many) that was used to GENERATE the target")
print("  3. Service_History correlates with vehicle age/wear")

# Cross-tab with Vehicle_Age
print("\n=== Service_History vs Vehicle_Age ===")
corr = df['Service_History'].corr(df['Vehicle_Age'])
print(f"Correlation: {corr:.4f}")

# Check: in the non-leaking subset
print("\n=== NON-LEAKING SUBSET ANALYSIS ===")
no_leak = df[
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] < 4)
]
print(f"Non-leak subset: N={len(no_leak)}")

print("\nService_History target rate (non-leak subset):")
for v in sorted(no_leak['Service_History'].unique()):
    s = no_leak[no_leak['Service_History'] == v]
    rate = s['Need_Maintenance'].mean()
    print(f"  SH={v:2d}: N={len(s):5d}, maint_rate={rate:.4f}")

auc_noleak = roc_auc_score(no_leak['Need_Maintenance'], no_leak['Service_History'])
print(f"\nService_History AUC (non-leak subset): {auc_noleak:.4f}")

# Check ALL features in non-leak subset
print("\n" + "=" * 60)
print("  ALL FEATURES - NON-LEAK SUBSET")
print("=" * 60)

for col in ['Maintenance_History', 'Tire_Condition']:
    ct = pd.crosstab(no_leak[col], no_leak['Need_Maintenance'], normalize='index')
    print(f"\n{col}:")
    print(ct.to_string())

for col in ['Vehicle_Age', 'Reported_Issues', 'Service_History', 'Accident_History']:
    try:
        auc_val = roc_auc_score(no_leak['Need_Maintenance'], no_leak[col])
        m0 = no_leak[no_leak['Need_Maintenance'] == 0][col].mean()
        m1 = no_leak[no_leak['Need_Maintenance'] == 1][col].mean()
        print(f"\n{col}: AUC={auc_val:.4f}, mean(0)={m0:.2f}, mean(1)={m1:.2f}")
    except Exception as e:
        print(f"\n{col}: Error - {e}")

# Check if Service_History has a MONOTONIC relationship with target
print("\n=== MONOTONICITY CHECK ===")
rates = []
for v in sorted(df['Service_History'].unique()):
    rate = df[df['Service_History'] == v]['Need_Maintenance'].mean()
    rates.append((v, rate))
    
print("Full dataset:")
for v, r in rates:
    bar = "#" * int(r * 50)
    print(f"  SH={v:2d}: {r:.4f} {bar}")

is_monotonic_inc = all(rates[i][1] <= rates[i+1][1] for i in range(len(rates)-1))
print(f"\nStrictly monotonic increasing? {is_monotonic_inc}")

# Check Reported_Issues more carefully
print("\n=== REPORTED_ISSUES ANALYSIS (NON-LEAK) ===")
for v in sorted(no_leak['Reported_Issues'].unique()):
    s = no_leak[no_leak['Reported_Issues'] == v]
    rate = s['Need_Maintenance'].mean()
    print(f"  Issues={v}: N={len(s):5d}, maint_rate={rate:.4f}")

# Check Accident_History  
print("\n=== ACCIDENT_HISTORY ANALYSIS (NON-LEAK) ===")
for v in sorted(no_leak['Accident_History'].unique()):
    s = no_leak[no_leak['Accident_History'] == v]
    rate = s['Need_Maintenance'].mean()
    print(f"  Accidents={v}: N={len(s):5d}, maint_rate={rate:.4f}")

# Final: Check if Service_History is ALSO used to generate target
print("\n=== SYNTHETIC TARGET RULE DETECTION ===")
# Test: does Service_History >= 8 make target deterministic?
sh_high = df[df['Service_History'] >= 8]
print(f"Service_History >= 8: N={len(sh_high)}, maint_rate={sh_high['Need_Maintenance'].mean():.4f}")

# Test without other leak features
sh_high_clean = no_leak[no_leak['Service_History'] >= 8]
print(f"Service_History >= 8 (no-leak): N={len(sh_high_clean)}, maint_rate={sh_high_clean['Need_Maintenance'].mean():.4f}")

sh_low_clean = no_leak[no_leak['Service_History'] <= 3]
print(f"Service_History <= 3 (no-leak): N={len(sh_low_clean)}, maint_rate={sh_low_clean['Need_Maintenance'].mean():.4f}")

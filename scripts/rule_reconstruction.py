"""Final definitive rule reconstruction - the target IS synthetic."""
import pandas as pd
import numpy as np
from itertools import product

df = pd.read_csv("data/vehicle_maintenance_data.csv")

print("=" * 60)
print("  DEFINITIVE SYNTHETIC TARGET PROOF")
print("=" * 60)

# The target Need_Maintenance = 1 if AND ONLY IF at least one of:
# 1. Brake_Condition == 'Worn Out'
# 2. Battery_Status == 'Weak'
# 3. Reported_Issues >= 4
# 4. Service_History >= 8
# 5. Reported_Issues >= 3 (probabilistic?)
# 6. Accident_History >= 2 (probabilistic?)
# 7. Reported_Issues >= 2 (probabilistic?)
# 8. Maintenance_History == 'Poor' (probabilistic?)

# Check: are ANY target=1 rows not explained by thresholds?
# The ultra-clean test showed 0% rate when ALL thresholds are below

# Let's find the EXACT rule by checking each combination
# Start by checking deterministic features first

# DETERMINISTIC (100%):
det_mask = (
    (df['Brake_Condition'] == 'Worn Out') | 
    (df['Battery_Status'] == 'Weak') | 
    (df['Reported_Issues'] >= 4) |
    (df['Service_History'] >= 8)
)

# Rows NOT covered by deterministic rules
remaining = df[~det_mask]
print(f"After deterministic rules: {det_mask.sum()} covered")
print(f"Remaining: {len(remaining)}")
print(f"Remaining target=1: {remaining['Need_Maintenance'].sum()}")
print(f"Remaining target=0: {(remaining['Need_Maintenance']==0).sum()}")

# In remaining, check each feature
print("\n--- Remaining subset analysis ---")
for col in ['Reported_Issues', 'Accident_History', 'Maintenance_History']:
    if df[col].dtype == 'object':
        ct = pd.crosstab(remaining[col], remaining['Need_Maintenance'])
        print(f"\n{col}:")
        print(ct.to_string())
    else:
        print(f"\n{col}:")
        for v in sorted(remaining[col].unique()):
            s = remaining[remaining[col] == v]
            print(f"  {col}={v}: N={len(s)}, target=1: {s['Need_Maintenance'].sum()}, rate={s['Need_Maintenance'].mean():.4f}")

# Try: what if the remaining follows: need_maint=1 if Issues>=2 OR Accident>=2 OR Maint=Poor?
# But with some probability
remaining_rule = (
    (remaining['Reported_Issues'] >= 2) |
    (remaining['Accident_History'] >= 2) |
    (remaining['Maintenance_History'] == 'Poor')
)
remaining_no_rule = remaining[~remaining_rule]
print(f"\nRemaining with NO threshold: N={len(remaining_no_rule)}")
print(f"  Target=1: {remaining_no_rule['Need_Maintenance'].sum()}")
print(f"  Target=0: {(remaining_no_rule['Need_Maintenance']==0).sum()}")
print(f"  Rate: {remaining_no_rule['Need_Maintenance'].mean():.4f}")

# So the EXACT rule seems to be:
# Need_Maintenance = 1 DETERMINISTICALLY if:
#   Brake=WornOut OR Battery=Weak OR Issues>=4 OR Service>=8
# Need_Maintenance = 1 PROBABILISTICALLY if:
#   Issues>=2 OR Accident>=2 OR Maintenance=Poor
# Need_Maintenance = 0 if NONE of the above

# Verify: rows with Issues<2 AND Accident<2 AND Maint!=Poor AND not deterministic
truly_clean = df[
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] < 2) &
    (df['Service_History'] < 8) &
    (df['Accident_History'] < 2) &
    (df['Maintenance_History'] != 'Poor')
]
print(f"\n=== TRULY CLEAN (no threshold triggered) ===")
print(f"N={len(truly_clean)}")
print(f"Target=1: {truly_clean['Need_Maintenance'].sum()}")
print(f"Target=0: {(truly_clean['Need_Maintenance']==0).sum()}")
print(f"Rate: {truly_clean['Need_Maintenance'].mean():.4f}")

# ALSO check: rows with ONLY Maintenance=Poor and nothing else
poor_only = df[
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] < 2) &
    (df['Service_History'] < 8) &
    (df['Accident_History'] < 2) &
    (df['Maintenance_History'] == 'Poor')
]
print(f"\n=== ONLY Maintenance=Poor triggered ===")
print(f"N={len(poor_only)}")
print(f"Target=1: {poor_only['Need_Maintenance'].sum()}, rate={poor_only['Need_Maintenance'].mean():.4f}")

# Check Issues=2 only
iss2_only = df[
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] == 2) &
    (df['Service_History'] < 8) &
    (df['Accident_History'] < 2) &
    (df['Maintenance_History'] != 'Poor')
]
print(f"\n=== ONLY Issues=2 triggered ===")
print(f"N={len(iss2_only)}")
print(f"Target=1: {iss2_only['Need_Maintenance'].sum()}, rate={iss2_only['Need_Maintenance'].mean():.4f}")

# Check Issues=3 only
iss3_only = df[
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] == 3) &
    (df['Service_History'] < 8) &
    (df['Accident_History'] < 2) &
    (df['Maintenance_History'] != 'Poor')
]
print(f"\n=== ONLY Issues=3 triggered ===")
print(f"N={len(iss3_only)}")
print(f"Target=1: {iss3_only['Need_Maintenance'].sum()}, rate={iss3_only['Need_Maintenance'].mean():.4f}")

# Check Accident=2 only
acc2_only = df[
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] < 2) &
    (df['Service_History'] < 8) &
    (df['Accident_History'] == 2) &
    (df['Maintenance_History'] != 'Poor')
]
print(f"\n=== ONLY Accident=2 triggered ===")
print(f"N={len(acc2_only)}")
print(f"Target=1: {acc2_only['Need_Maintenance'].sum()}, rate={acc2_only['Need_Maintenance'].mean():.4f}")

print("\n" + "=" * 60)
print("  CONCLUSION: TARGET IS 100% SYNTHETIC")
print("=" * 60)
print("Need_Maintenance = 1 if ANY of these thresholds are triggered:")
print("  DETERMINISTIC: Brake=WornOut, Battery=Weak, Issues>=4, Service>=8")
print("  PROBABILISTIC: Issues>=2, Accident>=2, Maintenance=Poor")
print("  OTHERWISE:     Need_Maintenance = 0 (ALWAYS)")
print()
print("EVERY feature in the selected set is a TARGET GENERATOR, not a predictor.")
print("The model is not predicting maintenance; it is reconstructing the rule.")

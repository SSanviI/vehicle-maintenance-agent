"""Final exact rule identification."""
import pandas as pd
import numpy as np

df = pd.read_csv("data/vehicle_maintenance_data.csv")

# From previous analysis, after removing deterministic triggers:
# Brake=WornOut, Battery=Weak, Issues>=4, Service>=8
# In the remaining subset:
# Maint=Good -> ALWAYS 0
# Maint=Poor AND Issues<2 AND Accident<2 -> rate=0.0000
# Issues=2 alone -> rate=0.0000  
# Accident=2 alone -> rate=0.0000
# Issues=3 alone -> rate=0.4793
# Need multiple threshold triggers for probabilistic target=1

# Let me find the EXACT combinatorial rule
remaining = df[
    (df['Brake_Condition'] != 'Worn Out') & 
    (df['Battery_Status'] != 'Weak') & 
    (df['Reported_Issues'] < 4) &
    (df['Service_History'] < 8)
]

print("Remaining (no deterministic trigger):", len(remaining))
print("Target=1:", remaining['Need_Maintenance'].sum())

# Check all combinations of the soft triggers
# Soft triggers: Issues>=2, Issues>=3, Accident>=2, Maint=Poor, Maint=Average
print("\n--- Detailed combinations ---")

for mh in ['Good', 'Average', 'Poor']:
    for iss in range(4):  # 0,1,2,3
        for acc in range(4):  # 0,1,2,3
            subset = remaining[
                (remaining['Maintenance_History'] == mh) &
                (remaining['Reported_Issues'] == iss) &
                (remaining['Accident_History'] == acc)
            ]
            if len(subset) > 0:
                rate = subset['Need_Maintenance'].mean()
                if rate > 0:
                    print(f"  MH={mh:7s}, Issues={iss}, Acc={acc}: N={len(subset):4d}, rate={rate:.4f}")

print("\n--- Summary: which combos produce target=1? ---")
combos_with_target = []
for mh in ['Good', 'Average', 'Poor']:
    for iss in range(4):
        for acc in range(4):
            subset = remaining[
                (remaining['Maintenance_History'] == mh) &
                (remaining['Reported_Issues'] == iss) &
                (remaining['Accident_History'] == acc)
            ]
            if len(subset) > 0 and subset['Need_Maintenance'].sum() > 0:
                rate = subset['Need_Maintenance'].mean()
                combos_with_target.append((mh, iss, acc, len(subset), rate))

print(f"Total combos with any target=1: {len(combos_with_target)}")
for mh, iss, acc, n, rate in combos_with_target:
    print(f"  MH={mh:7s}, Issues={iss}, Acc={acc}: N={n:4d}, rate={rate:.4f}")

# KEY INSIGHT: The rule appears to be a "score" system
# Each feature contributes a score, and when total >= threshold, Need_Maintenance = 1
# Let's test: 
# Score = (1 if Issues>=2) + (1 if Issues>=3) + (1 if Acc>=2) + (1 if Maint=Poor) + (1 if Maint=Average)
# Need_Maint = 1 if score >= 2 (probabilistically)

print("\n--- Testing score-based rule ---")
remaining['score'] = (
    (remaining['Reported_Issues'] >= 2).astype(int) +
    (remaining['Reported_Issues'] >= 3).astype(int) +
    (remaining['Accident_History'] >= 2).astype(int) +
    (remaining['Maintenance_History'] == 'Poor').astype(int)
)

for s in sorted(remaining['score'].unique()):
    subset = remaining[remaining['score'] == s]
    rate = subset['Need_Maintenance'].mean()
    print(f"  Score={s}: N={len(subset):5d}, rate={rate:.4f}")

"""
Script to test Proper Missing Value Handling + Monotonic Constraints
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

df = pd.read_csv("data/vehicle_maintenance_data.csv")

class ProperPreprocessor:
    def __init__(self):
        self.category_maps = {
            "Maintenance_History": {"Good": 0, "Average": 1, "Poor": 2},
            "Tire_Condition": {"New": 0, "Good": 1, "Worn Out": 2},
            "Brake_Condition": {"New": 0, "Good": 1, "Worn Out": 2},
            "Battery_Status": {"New": 0, "Good": 1, "Weak": 2},
        }

    def transform(self, df_in):
        res = pd.DataFrame(index=df_in.index)
        for col in ["Maintenance_History", "Tire_Condition", "Brake_Condition", "Battery_Status"]:
            vals = df_in[col] if col in df_in.columns else pd.Series(np.nan, index=df_in.index)
            mapped = vals.map(self.category_maps[col]) # missing becomes NaN
            res[col] = mapped
            res[f"{col}_missing"] = mapped.isna().astype(int)

        for col in ["Vehicle_Age", "Reported_Issues", "Service_History", "Accident_History"]:
            vals = df_in[col] if col in df_in.columns else pd.Series(np.nan, index=df_in.index)
            is_miss = vals.isna() | (vals == -1)
            res[col] = vals.where(~is_miss, np.nan)
            res[f"{col}_missing"] = is_miss.astype(int)

        return res

pp = ProperPreprocessor()
X_trans = pp.transform(df)
y = df["Need_Maintenance"]

mc = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]

lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    monotone_constraints=mc,
    random_state=42,
    verbose=-1
)
lgbm.fit(X_trans, y)

print("\n--- Testing Model Behavior Across Conversation Turns ---")

def test_turn(desc, row_dict):
    df_single = pd.DataFrame([row_dict])
    X_single = pp.transform(df_single)
    prob = lgbm.predict_proba(X_single)[0, 1]
    print(f"{desc:<50s} -> prob(Need_Maintenance=1): {prob:.4f}")
    return prob

t1 = test_turn("1. Age 10", {"Vehicle_Age": 10})
t2 = test_turn("2. + Brakes Good", {"Vehicle_Age": 10, "Brake_Condition": "Good"})
t3 = test_turn("3. + Battery New", {"Vehicle_Age": 10, "Brake_Condition": "Good", "Battery_Status": "New"})
t4 = test_turn("4. + Maintenance Poor", {"Vehicle_Age": 10, "Brake_Condition": "Good", "Battery_Status": "New", "Maintenance_History": "Poor"})
t5 = test_turn("5. + Service 5", {"Vehicle_Age": 10, "Brake_Condition": "Good", "Battery_Status": "New", "Maintenance_History": "Poor", "Service_History": 5})
t6 = test_turn("6. + Accident 5", {"Vehicle_Age": 10, "Brake_Condition": "Good", "Battery_Status": "New", "Maintenance_History": "Poor", "Service_History": 5, "Accident_History": 5})
t7 = test_turn("7. + Tire Worn Out", {"Vehicle_Age": 10, "Brake_Condition": "Good", "Battery_Status": "New", "Maintenance_History": "Poor", "Service_History": 5, "Accident_History": 5, "Tire_Condition": "Worn Out"})

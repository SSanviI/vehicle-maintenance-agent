"""
Feature importance and correlation analysis script
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("data/vehicle_maintenance_data.csv")

# Encode categorical variables
cat_cols = ['Vehicle_Model', 'Maintenance_History', 'Fuel_Type', 'Transmission_Type', 
            'Owner_Type', 'Tire_Condition', 'Brake_Condition', 'Battery_Status']

df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = OrdinalEncoder().fit_transform(df_encoded[[col]])

# Drop date columns for feature importance test
drop_cols = ['Last_Service_Date', 'Warranty_Expiry_Date', 'Need_Maintenance']
X = df_encoded.drop(columns=drop_cols)
y = df_encoded['Need_Maintenance']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCES:")
print("=" * 80)
print(importances)

print("\n" + "=" * 80)
print("CORRELATION WITH Need_Maintenance:")
print("=" * 80)
corrs = df_encoded.corr()['Need_Maintenance'].sort_values(ascending=False)
print(corrs)

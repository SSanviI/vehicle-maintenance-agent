"""
Comprehensive EDA script for data/vehicle_maintenance_data.csv
"""

import pandas as pd
import numpy as np

def run_eda():
    df = pd.read_csv("data/vehicle_maintenance_data.csv")
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("COLUMNS & DATA TYPES:")
    print(df.dtypes)
    print("\n" + "-" * 50)
    
    print("MISSING VALUES:")
    print(df.isnull().sum())
    print("\n" + "-" * 50)
    
    print("UNIQUE VALUES PER COLUMN:")
    for col in df.columns:
        print(f"{col:<25s}: {df[col].nunique()} unique -> {df[col].unique()[:5]}")
    print("\n" + "-" * 50)
    
    if "Need_Maintenance" in df.columns:
        print("TARGET DISTRIBUTION (Need_Maintenance):")
        print(df["Need_Maintenance"].value_counts(dropna=False, normalize=True))
        print(df["Need_Maintenance"].value_counts(dropna=False))
        print("\n" + "-" * 50)
        
        print("FEATURE RELATIONSHIPS WITH TARGET:")
        for col in df.columns:
            if col != "Need_Maintenance":
                print(f"\n--- {col} vs Need_Maintenance ---")
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    ct = pd.crosstab(df[col], df["Need_Maintenance"], normalize='index')
                    print(ct)
                else:
                    print(df.groupby("Need_Maintenance")[col].describe())

if __name__ == "__main__":
    run_eda()

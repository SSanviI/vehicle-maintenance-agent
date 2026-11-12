# VEHICLE MAINTENANCE DATASET — EXPLORATORY DATA ANALYSIS (EDA) REPORT

## 1. Dataset Overview
- **Location**: `data/vehicle_maintenance_data.csv`
- **Shape**: 50,000 rows × 20 columns
- **Missing Values**: 0 missing values across all 20 columns.
- **Target Column**: `Need_Maintenance` (Binary classification: `0 = no_maintenance_needed`, `1 = maintenance_required`).

---

## 2. Column Summary & Data Types

| Column Name | Data Type | Unique Count | Sample Values | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `Vehicle_Model` | `object` | 6 | Truck, Van, Bus, Motorcycle, SUV, Car | Uninformative (~81% target positive across all models) |
| `Mileage` | `int64` | 31,498 | 58765, 60353, 68072 | Uninformative (Mean = 54,910 for 0 vs 54,936 for 1) |
| `Maintenance_History` | `object` | 3 | Good, Average, Poor | **Highly Predictive** (Good=74.7% maint, Poor=87.1% maint) |
| `Reported_Issues` | `int64` | 6 | 0, 1, 2, 3, 4, 5 | **Highly Predictive** (4-5 issues = 100% maint required) |
| `Vehicle_Age` | `int64` | 10 | 1 to 10 years | **Domain Essential** (Included in initial candidate set) |
| `Fuel_Type` | `object` | 3 | Electric, Petrol, Diesel | Uninformative (~81% positive across all fuel types) |
| `Transmission_Type` | `object` | 2 | Automatic, Manual | Uninformative (~81% positive across all types) |
| `Engine_Size` | `int64` | 5 | 800, 1000, 1500, 2000, 2500 | Uninformative (~81% positive across all sizes) |
| `Odometer_Reading` | `int64` | 42,420 | 28524, 133630 | Noise / Uninformative |
| `Last_Service_Date` | `object` | 336 | YYYY-MM-DD | Uninformative date string |
| `Warranty_Expiry_Date` | `object` | 701 | YYYY-MM-DD | Uninformative date string |
| `Owner_Type` | `object` | 3 | First, Second, Third | Uninformative (~81% positive across all types) |
| `Insurance_Premium` | `int64` | 21,601 | 20782, 23489 | Noise / Uninformative |
| `Service_History` | `int64` | 10 | 1 to 10 counts | **Predictive** (Mean = 4.9 for 0 vs 5.7 for 1) |
| `Accident_History` | `int64` | 4 | 0, 1, 2, 3 | **Predictive** (0-1 = 77.3% maint, 2-3 = 84.5% maint) |
| `Fuel_Efficiency` | `float64` | 50,000 | 13.62, 18.70 | Noise / Uninformative |
| `Tire_Condition` | `object` | 3 | New, Good, Worn Out | **Selected Feature** (New, Good, Worn Out) |
| `Brake_Condition` | `object` | 3 | New, Good, Worn Out | **Deterministic Threshold** (Worn Out = 100% maint required) |
| `Battery_Status` | `object` | 3 | New, Good, Weak | **Deterministic Threshold** (Weak = 100% maint required) |
| `Need_Maintenance` | `int64` | 2 | 0, 1 | **Target Variable** (0 = 19.0%, 1 = 81.0%) |

---

## 3. Target Distribution & Leakage Analysis

### Target Imbalance
- `Need_Maintenance = 1`: **40,498 (80.996%)**
- `Need_Maintenance = 0`: **9,502 (19.004%)**

### Target-Construction Artifacts & Leakage Investigation
Analysis reveals that `Need_Maintenance` in `data/vehicle_maintenance_data.csv` was synthetically generated using clear deterministic rule thresholds:
1. `Brake_Condition = "Worn Out"` $\implies$ **100% `Need_Maintenance = 1`** (0 out of 5,500 rows are Class 0).
2. `Battery_Status = "Weak"` $\implies$ **100% `Need_Maintenance = 1`** (0 out of 5,512 rows are Class 0).
3. `Reported_Issues >= 4` $\implies$ **100% `Need_Maintenance = 1`** (0 out of 5,420 rows are Class 0).

These thresholds reflect realistic mechanical safety rules (e.g. worn out brakes or weak battery demand immediate maintenance), but must be handled carefully in ML preprocessing so ordinal encodings preserve strict monotonicity without artificial score anomalies.

---

## 4. Final Proposed ML Feature Set

The final ML feature schema includes **8 selected features**:

1. `Vehicle_Age` (Numerical: 1-10 years)
2. `Reported_Issues` (Numerical: 0-5 counts)
3. `Maintenance_History` (Categorical: Good, Average, Poor)
4. `Service_History` (Numerical: 1-10 visits)
5. `Accident_History` (Numerical: 0-3 counts)
6. `Tire_Condition` (Categorical: New, Good, Worn Out)
7. `Brake_Condition` (Categorical: New, Good, Worn Out)
8. `Battery_Status` (Categorical: New, Good, Weak)

### Why Included:
- **`Brake_Condition` & `Battery_Status`**: Primary mechanical failure indicators with strong predictive weight.
- **`Reported_Issues` & `Maintenance_History`**: Core indicators of vehicle reliability and past maintenance record.
- **`Accident_History` & `Service_History`**: Historical operational metrics that influence wear and tear.
- **`Tire_Condition` & `Vehicle_Age`**: Core domain candidate features for safety and wear assessment.

### Why Excluded:
- `Vehicle_Model`, `Fuel_Type`, `Transmission_Type`, `Engine_Size`, `Owner_Type`: Uniform ~81% target positive distribution across all categories. Zero predictive feature importance.
- `Mileage`, `Odometer_Reading`, `Fuel_Efficiency`, `Insurance_Premium`: Continuous noise variables with identical distributions across Class 0 and Class 1.
- `Last_Service_Date`, `Warranty_Expiry_Date`: Raw date strings without independent predictive signal beyond vehicle age.

---

## 5. Partial-Information Training Strategy

To support incremental, step-by-step predictions as users provide vehicle information over conversation:
1. **Missing Indicator & Ordinal Sentinel Encoding**: Categorical features are encoded with `Missing / UNKNOWN = -1`, placing missing values strictly below severity levels (`Good=0 < Average=1 < Poor=2`).
2. **Numerical Imputation & Missing Masking**: Numerical features use median imputation coupled with explicit missing indicators (`_isnan` columns) during preprocessing.
3. **Partial Training Masking**: The model is trained on complete and partial feature subsets to learn reliable probability estimation when only 1 or 2 features (e.g. `Vehicle_Age` and `Reported_Issues`) are available.

---

## 6. Candidate Models & Evaluation Plan
- **Candidate Models**: Logistic Regression, Random Forest Classifier, LightGBM / Gradient Boosting.
- **Data Split**: 70% Train, 15% Validation, 15% Test (Stratified split with fixed seed `42`).
- **Metrics**: ROC-AUC, PR-AUC, F1-Score, Recall (minimizing False Negatives), Brier Score / Calibration Error.

"""
churn_pipeline.py
Customer churn prediction. Trains a model on customers.csv and reports accuracy.

Setup:
    python generate_data.py     # creates customers.csv (run once)
    python churn_pipeline.py

Goal for the session: this script works, but it is slow and uses a lot of
memory. Profile it, find out why, and rewrite it to be fast and correct.
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

t0 = time.time()
_last = t0

def stage(name):
    global _last
    now = time.time()
    print(f"  [{name:<10}] {now - _last:6.1f}s")
    _last = now


# ---- load -------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "customers.csv")
df = pd.read_csv(csv_path)
stage("load")


# ---- feature engineering ----------------------------------------------------
# Vectorized cleaning of money fields using float32 to save memory
df["monthly_charges"] = (
    df["monthly_charges"]
    .str.replace("฿", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float32")
)
df["total_charges"] = (
    df["total_charges"]
    .str.replace("฿", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype("float32")
)

# Vectorized tenure days calculation downcast to int16
df["tenure_days"] = (
    pd.to_datetime(df["last_active_date"]) - pd.to_datetime(df["signup_date"])
).dt.days.astype("int16")

# Drop unused heavy string columns and index columns immediately to save memory
df.drop(columns=["customer_id", "signup_date", "last_active_date"], inplace=True)

# Vectorized region average monthly charges downcast to float32
df["region_avg_charge"] = df.groupby("region")["monthly_charges"].transform("mean").astype("float32")

# Vectorized risk score calculation downcast to int8 (values range 0-6)
df["risk_score"] = (
    (df["monthly_charges"] > 1500).astype("int8") * 2 +
    (df["tenure_days"] < 180).astype("int8") * 2 +
    (df["num_transactions"] < 5).astype("int8") * 1 +
    (df["age"] < 25).astype("int8") * 1
)

# Vectorized high value field downcast to int8
df["high_value"] = (df["monthly_charges"] > 1500).astype("int8")

stage("features")


# ---- encode + scale ---------------------------------------------------------
# Label encoding for categorical fields downcast to int8
for col in ["region", "plan"]:
    df[col] = LabelEncoder().fit_transform(df[col]).astype("int8")

# Downcast remaining numerical columns to minimize memory footprint
df["age"] = df["age"].astype("uint8")
df["num_transactions"] = df["num_transactions"].astype("int16")
df["churned"] = df["churned"].astype("int8")

features = [
    "age", "monthly_charges", "total_charges", "num_transactions",
    "tenure_days", "region", "plan", "region_avg_charge",
    "risk_score", "high_value"
]
X = df[features]
y = df["churned"]

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
stage("prep")


# ---- train ------------------------------------------------------------------
# Parallelized Random Forest using all CPU cores (n_jobs=-1)
model = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1)
model.fit(X_train, y_train)
stage("train")


# ---- evaluate ---------------------------------------------------------------
# Vectorized accuracy evaluation
preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
stage("evaluate")

print(f"Accuracy:  {accuracy:.4f}")
print(f"df memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"TOTAL:     {time.time() - t0:.1f}s")

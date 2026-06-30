"""
generate_data.py
Creates a customers CSV used by every script in this kit.

Usage:
    python generate_data.py                  # 150,000 rows -> customers.csv
    python generate_data.py --rows 2000000   # 2M rows (for the Spark demo)
    python generate_data.py --rows 5000000 --out customers_big.csv
"""
import argparse
import numpy as np
import pandas as pd

import os

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=150_000)
p.add_argument("--out", type=str, default="customers.csv")
args = p.parse_args()

out_path = args.out
if not os.path.isabs(out_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, out_path)

N = args.rows
rng = np.random.default_rng(42)
regions = ["Bangkok", "Chiang Mai", "Phuket", "Khon Kaen", "Hat Yai"]
plans = ["basic", "standard", "premium"]

signup = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1200, N), unit="D")
last_active = signup + pd.to_timedelta(rng.integers(1, 900, N), unit="D")
age = rng.integers(18, 75, N)
monthly = rng.uniform(150, 2500, N).round(2)
tenure_days = (last_active - signup).days.to_numpy()
total = (monthly * (tenure_days / 30.0) * rng.uniform(0.8, 1.1, N)).round(2)
num_tx = rng.poisson(tenure_days / 20 + 1)
score = -0.004 * tenure_days + 0.0008 * monthly - 0.05 * num_tx + rng.normal(0, 1, N)
churned = (score > np.quantile(score, 0.72)).astype(int)

# vectorized "฿1,234.50" formatting (fast even at millions of rows)
def money(arr):
    s = pd.Series(arr).map(lambda x: f"{x:,.2f}")
    return "฿" + s

df = pd.DataFrame({
    "customer_id": np.arange(N),
    "signup_date": signup.strftime("%Y-%m-%d"),
    "last_active_date": last_active.strftime("%Y-%m-%d"),
    "region": rng.choice(regions, N),
    "plan": rng.choice(plans, N),
    "age": age,
    "monthly_charges": money(monthly),
    "total_charges": money(total),
    "num_transactions": num_tx,
    "churned": churned,
})
df.to_csv(out_path, index=False)
print(f"Wrote {out_path}  rows={len(df):,}  churn_rate={churned.mean():.2%}")

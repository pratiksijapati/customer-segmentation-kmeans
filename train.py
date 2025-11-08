"""
Train a KMeans model on mall customer data (AnnualIncome, SpendingScore).
Saves: unsupervised/kmeans.joblib and unsupervised/scaler.joblib
If data/mall_customers_sample.csv is missing, it auto-generates a synthetic dataset.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "mall_customers_sample.csv")
MODEL_PATH = os.path.join(HERE, "kmeans.joblib")
SCALER_PATH = os.path.join(HERE, "scaler.joblib")

def generate_data(n=200, seed=42):
    """Generate a realistic synthetic dataset with 4 natural clusters."""
    rng = np.random.default_rng(seed)
    # Two lower-income clusters with low/high spending
    g1_income = rng.normal(30000, 5000, n//4)
    g1_score  = rng.normal(25, 10,   n//4)
    g2_income = rng.normal(35000, 6000, n//4)
    g2_score  = rng.normal(75, 10,     n//4)
    # Two higher-income clusters with low/high spending
    g3_income = rng.normal(90000, 10000, n//4)
    g3_score  = rng.normal(30, 10,      n//4)
    g4_income = rng.normal(95000, 11000, n - 3*(n//4))
    g4_score  = rng.normal(80, 10,       n - 3*(n//4))

    income = np.concatenate([g1_income, g2_income, g3_income, g4_income]).clip(10000, 200000)
    score  = np.concatenate([g1_score,  g2_score,  g3_score,  g4_score]).clip(1, 100)

    df = pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "AnnualIncome": income.astype(int),
        "SpendingScore": score.astype(int)
    })
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[Data] Generated synthetic dataset → {os.path.abspath(DATA_PATH)}")

def main():
    # Ensure dataset exists
    if not os.path.exists(DATA_PATH):
        print("[Data] mall_customers_sample.csv not found; generating synthetic data...")
        generate_data()

    # Load data
    df = pd.read_csv(DATA_PATH)
    if not {"AnnualIncome", "SpendingScore"}.issubset(df.columns):
        print("[Error] CSV must contain columns: AnnualIncome, SpendingScore")
        sys.exit(1)

    X = df[["AnnualIncome", "SpendingScore"]].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    kmeans.fit(X_scaled)

    # Save artifacts
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Model] Saved KMeans → {os.path.abspath(MODEL_PATH)} | Inertia: {kmeans.inertia_:.2f}")
    print(f"[Model] Saved Scaler → {os.path.abspath(SCALER_PATH)}")

if __name__ == "__main__":
    main()

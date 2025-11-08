import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import subprocess

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "kmeans.joblib")
SCALER_PATH = os.path.join(HERE, "scaler.joblib")
DATA_PATH = os.path.join(HERE, "..", "data", "mall_customers_sample.csv")
TRAIN_SCRIPT = os.path.join(HERE, "train.py")

st.set_page_config(page_title="Customer Segmentation (K-Means)", layout="centered")
st.title("🛍️ Unsupervised Learning — Customer Segmentation")
st.caption("K-Means on Annual Income & Spending Score")

def ensure_model():
    """Return True if model + scaler exist; otherwise offer a one-click trainer."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return True

    st.warning("Model not found. Click below to train it.")
    if st.button("Train model now"):
        with st.spinner("Training..."):
            result = subprocess.run([sys.executable, TRAIN_SCRIPT], capture_output=True, text=True)
        st.code(result.stdout)
        if result.stderr:
            st.error(result.stderr)
        # Rerun the app so the new artifacts are picked up
        st.rerun()
    return False

if ensure_model():
    # Load artifacts
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Input area
    st.subheader("🔮 Predict Cluster for a New Customer")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income", min_value=1000, max_value=200000, value=60000, step=1000)
    with col2:
        score = st.slider("Spending Score", 1, 100, 50)

    if st.button("Predict Cluster"):
        X_new = np.array([[income, score]])
        Xs = scaler.transform(X_new)
        cluster = int(kmeans.predict(Xs)[0])
        st.success(f"Cluster ID: **{cluster}**")

    st.divider()

    # Visualization
    st.subheader("📊 Cluster Visualization")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if {"AnnualIncome", "SpendingScore"}.issubset(df.columns):
            X_all = df[["AnnualIncome", "SpendingScore"]].values
            labels = kmeans.predict(scaler.transform(X_all))

            # Scatter plot
            fig, ax = plt.subplots()
            scatter = ax.scatter(df["AnnualIncome"], df["SpendingScore"], c=labels, cmap="tab10", alpha=0.7)
            ax.set_xlabel("Annual Income")
            ax.set_ylabel("Spending Score")
            ax.set_title("K-Means Clusters")
            st.pyplot(fig)

            # Cluster counts
            st.subheader("📈 Cluster Counts")
            counts = pd.Series(labels).value_counts().sort_index()
            st.bar_chart(counts)
        else:
            st.error("CSV missing required columns: AnnualIncome, SpendingScore")
    else:
        st.info("Dataset not found yet — train the model to auto-generate one.")

# Customer Segmentation with K-Means

Groups shopping-mall customers into segments by **annual income** and **spending score** using K-Means clustering, with a small Streamlit app to explore the clusters and assign a new customer to one.

> **Status:** small machine-learning exercise.

## What it does

- `train.py` scales the two features with `StandardScaler`, fits K-Means (4 clusters) and saves the model with joblib. If `data/mall_customers_sample.csv` is missing, it generates a synthetic sample dataset.
- `streamlit_app.py` trains the model on first run if needed, plots the clusters and predicts the segment for a customer you enter.

## Tech stack

Python · scikit-learn · pandas · NumPy · Matplotlib · Streamlit · joblib

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run streamlit_app.py
```

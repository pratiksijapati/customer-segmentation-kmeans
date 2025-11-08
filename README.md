# ML Combo Project — Supervised & Unsupervised (Ready-to-Run)

Two small projects with training scripts and Streamlit apps.

## Projects
1) **Supervised — Salary Prediction (Linear Regression)**
   - Predict salary from years of experience.
   - Files: `supervised/train.py`, `supervised/streamlit_app.py`.
   - Data: `data/salary_data.csv` (auto-download if missing).

2) **Unsupervised — Customer Segmentation (K-Means)**
   - Cluster customers using `AnnualIncome` and `SpendingScore`.
   - Files: `unsupervised/train.py`, `unsupervised/streamlit_app.py`.
   - Data: `data/mall_customers_sample.csv` (auto-generate if missing).

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train
python supervised/train.py
python unsupervised/train.py

# Run the apps (use separate terminals or run one-by-one)
streamlit run supervised/streamlit_app.py
streamlit run unsupervised/streamlit_app.py

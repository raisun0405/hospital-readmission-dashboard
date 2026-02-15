# Hospital Readmission Dashboard

Predictive dashboard for hospital readmission risk using machine learning.

## Project Overview
- **Dataset:** UCI Diabetes Dataset
- **Goal:** Predict 30-day hospital readmission risk
- **Model:** Logistic Regression + Random Forest
- **Dashboard:** Streamlit

## Project Structure
```
hospital-readmission-dashboard/
├── data/               # Dataset files
├── models/             # Trained ML models
├── notebooks/          # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── data_prep.py   # Data cleaning & preprocessing
│   ├── model.py       # ML model training
│   └── utils.py       # Utility functions
├── app.py             # Streamlit dashboard
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run dashboard: `streamlit run app.py`

## Features
- Patient risk score prediction
- Feature importance visualization
- Risk factor analysis
- Actionable insights for doctors

---

*Project Start: February 15, 2026*

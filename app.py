"""
Hospital Readmission Dashboard
Main application file (Streamlit)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Hospital Readmission Dashboard")
st.subheader("Predict 30-Day Readmission Risk")

st.markdown("""
This dashboard helps doctors identify patients at high risk of being readmitted 
within 30 days of discharge, allowing for proactive intervention.
""")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
st.sidebar.info("Upload patient data or use demo data to get started.")

# Main content
st.write("### 🚀 Getting Started")
st.write("""
1. **Upload Data:** Upload patient data CSV file
2. **Train Model:** Build prediction model
3. **Predict Risk:** Get readmission risk scores
4. **View Insights:** See feature importance and recommendations
""")

st.write("### 📁 Data Requirements")
st.write("""
The dataset should include:
- Patient demographics (age, gender)
- Medical history (diagnoses, medications)
- Lab results (HbA1c, glucose levels)
- Previous admissions
- Discharge disposition
""")

st.success("👈 Use the sidebar to upload data and start analysis!")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit + Scikit-learn | February 2026*")

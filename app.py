"""
Enhanced Hospital Readmission Dashboard
Improved UI with multiple pages and better visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config - MUST be first
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #ffffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9900;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Predict Risk", "📊 Analytics", "📈 Model Performance", "ℹ️ About"]
)

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        with open('models/random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        with open('models/dataset_stats.json', 'r') as f:
            dataset_stats = json.load(f)
        return rf_model, lr_model, metrics, dataset_stats
    except Exception as e:
        return None, None, None, None

# Load everything
rf_model, lr_model, metrics, dataset_stats = load_models()

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏥 Hospital Readmission Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Predict and Analyze 30-Day Hospital Readmission Risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Patients", f"{dataset_stats.get('total_patients', 101766):,}" if dataset_stats else "101,766")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", "67.9%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ROC-AUC Score", "0.672")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.write("""
    ## 🎯 What This Dashboard Does
    
    This machine learning-powered dashboard helps healthcare providers identify patients 
    at high risk of being readmitted to the hospital within 30 days of discharge.
    
    ### Key Features:
    - 🔮 **Risk Prediction**: Get instant readmission risk scores
    - 📊 **Analytics**: Explore patient data patterns
    - 📈 **Model Insights**: View feature importance and model performance
    - 💡 **Recommendations**: Get actionable intervention strategies
    
    ### Why It Matters:
    - Hospitals lose money when patients are readmitted within 30 days
    - Early identification allows for proactive care
    - Reduces healthcare costs and improves patient outcomes
    """)
    
    st.info("👈 Use the sidebar to navigate to different sections of the dashboard.")

# PREDICTION PAGE
elif page == "🔮 Predict Risk":
    st.header("🔮 Patient Risk Assessment")
    st.markdown("Enter patient details to predict 30-day readmission risk.")
    
    # Create form with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographics")
        age = st.selectbox("Age Group", 
                          ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                           '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                          index=6)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        race = st.selectbox("Race", ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'])
    
    with col2:
        st.subheader("🏥 Hospital Stay")
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 3, help="Length of current admission")
        num_lab_procedures = st.slider("Lab Procedures", 0, 100, 40, help="Number of lab tests performed")
        num_procedures = st.slider("Procedures", 0, 10, 1, help="Number of procedures performed")
        num_medications = st.slider("Medications", 0, 50, 15, help="Number of medications prescribed")
    
    with col3:
        st.subheader("📋 Medical History")
        number_outpatient = st.number_input("Outpatient Visits (prior year)", 0, 50, 0, help="Visits in past year")
        number_emergency = st.number_input("Emergency Visits (prior year)", 0, 50, 0, help="ER visits in past year")
        number_inpatient = st.number_input("Inpatient Visits (prior year)", 0, 50, 0, help="Prior admissions")
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7, help="Total diagnoses recorded")
    
    # Predict button
    if st.button("🔮 Calculate Risk", type="primary", use_container_width=True):
        if rf_model is not None:
            from src.predict import predict_readmission_risk
            
            patient_data = {
                'age': age, 'gender': gender, 'race': race,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses
            }
            
            risk_score, risk_level = predict_readmission_risk(patient_data)
            
            # Display results in columns
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Readmission Risk", f"{risk_score:.1%}")
                
                if risk_level == 'High':
                    st.markdown('<div class="risk-high"><h3>🔴 HIGH RISK</h3><p>Immediate intervention recommended</p></div>', unsafe_allow_html=True)
                elif risk_level == 'Medium':
                    st.markdown('<div class="risk-medium"><h3>🟡 MEDIUM RISK</h3><p>Close monitoring advised</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-low"><h3>🟢 LOW RISK</h3><p>Standard discharge procedures</p></div>', unsafe_allow_html=True)
            
            with col_res2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score (%)", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#ccffcc'},
                            {'range': [30, 60], 'color': '#ffffcc'},
                            {'range': [60, 100], 'color': '#ffcccc'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("📋 Clinical Recommendations")
            
            if risk_level == 'High':
                st.error("""
                ### 🔴 High Risk Interventions
                
                - ✅ **Schedule follow-up within 3-5 days**
                - ✅ **Medication reconciliation** - Review all medications
                - ✅ **Home health evaluation** - Consider home care services
                - ✅ **Patient education** - Provide clear discharge instructions
                - ✅ **Care coordinator assignment** - Assign case manager
                - ✅ **Phone check-in** - Call within 24-48 hours
                """)
            elif risk_level == 'Medium':
                st.warning("""
                ### 🟡 Medium Risk Interventions
                
                - ✅ **Schedule follow-up within 7-14 days**
                - ✅ **Medication review** - Ensure understanding
                - ✅ **Written instructions** - Provide clear documentation
                - ✅ **Phone check-in** - Call within 3-5 days
                """)
            else:
                st.success("""
                ### 🟢 Low Risk Protocol
                
                - ✅ **Standard discharge procedures**
                - ✅ **Routine follow-up** - Schedule within 30 days
                - ✅ **Patient education** - Standard materials
                """)

# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.header("📊 Data Analytics")
    st.markdown("Explore patterns in the hospital readmission data.")
    
    if dataset_stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", f"{dataset_stats['total_patients']:,}")
        col2.metric("Readmitted <30 days", f"{dataset_stats['readmitted_30_pct']:.1f}%")
        col3.metric("Avg Stay", f"{dataset_stats['avg_stay']:.1f} days")
        col4.metric("Avg Medications", f"{dataset_stats['avg_medications']:.1f}")
    
    st.markdown("---")
    
    # Display visualizations
    viz_tabs = st.tabs(["Age Distribution", "Length of Stay", "Medications", "Demographics"])
    
    with viz_tabs[0]:
        st.subheader("Readmission by Age Group")
        try:
            st.image('models/viz_age_readmission.png', use_container_width=True)
        except:
            st.info("Visualization loading...")
    
    with viz_tabs[1]:
        st.subheader("Length of Stay vs Readmission")
        try:
            st.image('models/viz_stay_readmission.png', use_container_width=True)
        except:
            st.info("Visualization loading...")
    
    with viz_tabs[2]:
        st.subheader("Medications vs Readmission")
        try:
            st.image('models/viz_medications_readmission.png', use_container_width=True)
        except:
            st.info("Visualization loading...")
    
    with viz_tabs[3]:
        st.subheader("Readmission by Demographics")
        try:
            st.image('models/viz_demographics.png', use_container_width=True)
        except:
            st.info("Visualization loading...")

# MODEL PERFORMANCE PAGE
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    
    if metrics is not None:
        # Display metrics table
        st.subheader("Model Comparison")
        
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(
            metrics_df.style.format("{:.3f}").highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        # Best model highlight
        best_model = metrics_df['roc_auc'].idxmax()
        st.success(f"🏆 **Best Model**: {best_model} (ROC-AUC: {metrics_df.loc[best_model, 'roc_auc']:.3f})")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC-AUC Comparison")
            fig = px.bar(
                x=metrics_df.index,
                y=metrics_df['roc_auc'],
                color=metrics_df['roc_auc'],
                color_continuous_scale='viridis',
                labels={'x': 'Model', 'y': 'ROC-AUC'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("All Metrics")
            metrics_plot = metrics_df[['accuracy', 'precision', 'recall', 'f1']].reset_index()
            metrics_plot_melted = metrics_plot.melt(id_vars='index', var_name='Metric', value_name='Score')
            fig = px.bar(
                metrics_plot_melted,
                x='Metric',
                y='Score',
                color='index',
                barmode='group',
                labels={'index': 'Model', 'Score': 'Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        try:
            st.image('models/confusion_matrices.png', use_container_width=True)
        except:
            st.info("Confusion matrices loading...")
        
        # Feature importance
        st.subheader("Feature Importance")
        try:
            st.image('models/feature_importance.png', use_container_width=True)
        except:
            st.info("Feature importance loading...")
    else:
        st.error("⚠️ Model metrics not available.")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    
    st.write("""
    ## Hospital Readmission Prediction Dashboard
    
    This project uses machine learning to predict the risk of hospital readmission 
    within 30 days of discharge, helping healthcare providers identify high-risk 
    patients for proactive intervention.
    
    ### 🎯 Objective
    Reduce 30-day hospital readmissions through early risk identification
    
    ### 📊 Dataset
    - **Source**: UCI Machine Learning Repository
    - **Size**: 101,766 patient records
    - **Period**: 1999-2008
    - **Hospitals**: 130 US hospitals
    
    ### 🤖 Models Used
    - Random Forest Classifier (Primary)
    - Logistic Regression (Baseline)
    - Gradient Boosting (Enhanced)
    
    ### 🛠️ Technologies
    - Python 3.13
    - scikit-learn
    - Streamlit
    - Plotly
    - pandas
    
    ### 📈 Performance
    - ROC-AUC: 0.672 (Random Forest)
    - Accuracy: 67.9%
    - Recall: 54.9% (catches 55% of high-risk patients)
    
    ### 👨‍💻 Author
    **Rohan Vishwakarma**
    - GitHub: [@raisun0405](https://github.com/raisun0405)
    
    ### 📄 License
    This project is licensed under the MIT License.
    
    ---
    
    **⚠️ Disclaimer**: This tool is for educational purposes only. 
    Predictions should not replace clinical judgment.
    """)

# Footer (all pages)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit + Scikit-learn | © 2026 Rohan Vishwakarma</p>
    <p>Dataset: UCI Diabetes 130-US Hospitals (1999-2008)</p>
</div>
""", unsafe_allow_html=True)

"""
Hospital Readmission Dashboard - Streamlit App
Interactive dashboard for predicting readmission risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏥 Hospital Readmission Dashboard")
st.markdown("### Predict 30-Day Readmission Risk Using Machine Learning")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")

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
        return rf_model, lr_model, metrics
    except:
        return None, None, None

@st.cache_data
def load_feature_importance():
    """Load feature importance"""
    try:
        rf_imp = pd.read_csv('models/random_forest_feature_importance.csv')
        return rf_imp
    except:
        return None

# Load everything
rf_model, lr_model, metrics = load_models()
rf_importance = load_feature_importance()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🎯 Risk Prediction", "📊 Model Performance", "📈 Feature Analysis"])

with tab1:
    st.header("🎯 Predict Readmission Risk")
    st.markdown("Enter patient details to get a readmission risk score.")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.selectbox("Age Group", 
                          ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                           '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        race = st.selectbox("Race", ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'])
    
    with col2:
        st.subheader("Hospital Stay")
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 3)
        num_lab_procedures = st.slider("Lab Procedures", 0, 100, 40)
        num_procedures = st.slider("Procedures", 0, 10, 1)
        num_medications = st.slider("Medications", 0, 50, 15)
    
    with col3:
        st.subheader("History")
        number_outpatient = st.number_input("Outpatient Visits (prior year)", 0, 50, 0)
        number_emergency = st.number_input("Emergency Visits (prior year)", 0, 50, 0)
        number_inpatient = st.number_input("Inpatient Visits (prior year)", 0, 50, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
    
    # Predict button
    if st.button("🔮 Predict Risk", type="primary"):
        if rf_model is not None:
            # Create patient data dict
            patient_data = {
                'age': age,
                'gender': gender,
                'race': race,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses
            }
            
            # Make prediction
            from src.predict import predict_readmission_risk
            risk_score, risk_level = predict_readmission_risk(patient_data)
            
            # Display result
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric("Readmission Risk", f"{risk_score:.1%}")
                
                if risk_score < 0.3:
                    st.success("🟢 LOW RISK")
                    st.write("Patient has low probability of readmission within 30 days.")
                elif risk_score < 0.6:
                    st.warning("🟡 MEDIUM RISK")
                    st.write("Patient has moderate risk. Consider follow-up care.")
                else:
                    st.error("🔴 HIGH RISK")
                    st.write("⚠️ High risk of readmission! Immediate intervention recommended.")
            
            with col_res2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "salmon"}
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
            st.subheader("📋 Recommendations")
            if risk_score >= 0.6:
                st.markdown("""
                - ✅ Schedule follow-up appointment within 7 days
                - ✅ Review medication adherence
                - ✅ Provide patient education materials
                - ✅ Consider home health visit
                - ✅ Coordinate with primary care physician
                """)
            elif risk_score >= 0.3:
                st.markdown("""
                - ✅ Schedule follow-up within 14 days
                - ✅ Provide discharge instructions
                - ✅ Phone call check-in at 7 days
                """)
            else:
                st.markdown("""
                - ✅ Standard discharge procedures
                - ✅ Routine follow-up in 30 days
                """)
        else:
            st.error("⚠️ Models not loaded. Please check if model files exist.")

with tab2:
    st.header("📊 Model Performance")
    
    if metrics is not None:
        # Display metrics
        st.subheader("Model Comparison")
        
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0), use_container_width=True)
        
        # ROC-AUC comparison
        fig = px.bar(
            x=metrics_df.index,
            y=metrics_df['roc_auc'],
            title='ROC-AUC Score Comparison',
            labels={'x': 'Model', 'y': 'ROC-AUC'},
            color=metrics_df['roc_auc'],
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # All metrics radar chart
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Random Forest Metrics:**")
            rf_metrics = metrics_df.loc['Random Forest']
            for metric, value in rf_metrics.items():
                st.write(f"- {metric.capitalize()}: {value:.3f}")
        
        with col2:
            st.write("**Logistic Regression Metrics:**")
            lr_metrics = metrics_df.loc['Logistic Regression']
            for metric, value in lr_metrics.items():
                st.write(f"- {metric.capitalize()}: {value:.3f}")
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        st.image('models/confusion_matrices.png', use_container_width=True)
        
    else:
        st.info("📊 Model metrics not available. Please train models first.")

with tab3:
    st.header("📈 Feature Importance Analysis")
    
    if rf_importance is not None:
        st.write("Top features that influence readmission risk:")
        
        # Top 10 features
        top_10 = rf_importance.head(10)
        
        fig = px.bar(
            top_10,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features (Random Forest)',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance image
        st.subheader("Feature Importance Visualization")
        st.image('models/feature_importance.png', use_container_width=True)
        
        # Full feature table
        st.subheader("All Feature Importances")
        st.dataframe(rf_importance, use_container_width=True)
        
    else:
        st.info("📈 Feature importance data not available.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit + Scikit-learn | Dataset: UCI Diabetes 130-US Hospitals*")
st.markdown("**Note:** This is a demo application. Predictions should not replace clinical judgment.")

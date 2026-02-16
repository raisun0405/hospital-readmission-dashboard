"""
Real Prediction Function for Hospital Readmission Dashboard
Uses trained models to make actual predictions
"""

import pandas as pd
import numpy as np
import pickle

def load_prediction_artifacts():
    """Load model and preprocessing artifacts"""
    with open('models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/processed/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('data/processed/features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, label_encoders, feature_names

def predict_readmission_risk(patient_data):
    """
    Predict readmission risk for a patient
    
    Args:
        patient_data: dict with patient information
    
    Returns:
        risk_score: float between 0 and 1
        risk_level: str ('Low', 'Medium', 'High')
    """
    model, label_encoders, feature_names = load_prediction_artifacts()
    
    # Create feature vector
    features = {}
    
    # Map age group to numeric
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    features['age'] = age_mapping.get(patient_data['age'], 55)
    
    # Encode gender
    gender_map = {'Male': 1, 'Female': 0}
    features['gender'] = gender_map.get(patient_data['gender'], 0)
    
    # Encode race
    race_map = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4}
    features['race'] = race_map.get(patient_data['race'], 4)
    
    # Hospital stay features
    features['time_in_hospital'] = patient_data['time_in_hospital']
    features['num_lab_procedures'] = patient_data['num_lab_procedures']
    features['num_procedures'] = patient_data['num_procedures']
    features['num_medications'] = patient_data['num_medications']
    
    # History features
    features['number_outpatient'] = patient_data['number_outpatient']
    features['number_emergency'] = patient_data['number_emergency']
    features['number_inpatient'] = patient_data['number_inpatient']
    features['number_diagnoses'] = patient_data['number_diagnoses']
    
    # Engineered features
    features['total_visits'] = (patient_data['number_outpatient'] + 
                                patient_data['number_emergency'] + 
                                patient_data['number_inpatient'])
    features['multiple_diagnoses'] = 1 if patient_data['number_diagnoses'] > 3 else 0
    
    # Categorize medications
    med_count = patient_data['num_medications']
    if med_count <= 5:
        features['medication_load'] = 0  # Low
    elif med_count <= 10:
        features['medication_load'] = 1  # Medium
    elif med_count <= 20:
        features['medication_load'] = 2  # High
    else:
        features['medication_load'] = 3  # Very High
    
    # Categorize stay
    stay = patient_data['time_in_hospital']
    if stay <= 2:
        features['stay_category'] = 0  # Short
    elif stay <= 5:
        features['stay_category'] = 1  # Medium
    elif stay <= 10:
        features['stay_category'] = 2  # Long
    else:
        features['stay_category'] = 3  # Very Long
    
    # Default values for other features
    features['admission_type_id'] = 1
    features['discharge_disposition_id'] = 1
    features['admission_source_id'] = 7
    features['change'] = 0
    features['diabetesMed'] = 1
    
    # Create DataFrame with correct column order
    X = pd.DataFrame([features])
    
    # Ensure all feature names are present
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    
    X = X[feature_names]
    
    # Predict
    risk_score = model.predict_proba(X)[0][1]
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = 'Low'
    elif risk_score < 0.6:
        risk_level = 'Medium'
    else:
        risk_level = 'High'
    
    return risk_score, risk_level

if __name__ == "__main__":
    # Test prediction
    test_patient = {
        'age': '[70-80)',
        'gender': 'Female',
        'race': 'Caucasian',
        'time_in_hospital': 5,
        'num_lab_procedures': 50,
        'num_procedures': 2,
        'num_medications': 15,
        'number_outpatient': 2,
        'number_emergency': 1,
        'number_inpatient': 1,
        'number_diagnoses': 8
    }
    
    score, level = predict_readmission_risk(test_patient)
    print(f"Test Patient Risk: {score:.2%} ({level})")

"""
API Endpoint for Hospital Readmission Model
Flask-based REST API for model predictions
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

app = Flask(__name__)

# Load model at startup
try:
    with open('models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/processed/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('data/processed/features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    MODEL_LOADED = True
    print("✅ Model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Error loading model: {e}")

def prepare_features(patient_data):
    """Convert patient data to model features"""
    features = {}
    
    # Age mapping
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    features['age'] = age_mapping.get(patient_data.get('age', '[50-60)'), 55)
    
    # Gender encoding
    gender_map = {'Male': 1, 'Female': 0}
    features['gender'] = gender_map.get(patient_data.get('gender', 'Female'), 0)
    
    # Race encoding
    race_map = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4}
    features['race'] = race_map.get(patient_data.get('race', 'Caucasian'), 4)
    
    # Numeric features
    features['time_in_hospital'] = patient_data.get('time_in_hospital', 3)
    features['num_lab_procedures'] = patient_data.get('num_lab_procedures', 40)
    features['num_procedures'] = patient_data.get('num_procedures', 1)
    features['num_medications'] = patient_data.get('num_medications', 15)
    features['number_outpatient'] = patient_data.get('number_outpatient', 0)
    features['number_emergency'] = patient_data.get('number_emergency', 0)
    features['number_inpatient'] = patient_data.get('number_inpatient', 0)
    features['number_diagnoses'] = patient_data.get('number_diagnoses', 7)
    
    # Engineered features
    features['total_visits'] = (features['number_outpatient'] + 
                                features['number_emergency'] + 
                                features['number_inpatient'])
    features['multiple_diagnoses'] = 1 if features['number_diagnoses'] > 3 else 0
    
    # Categorize medications
    med_count = features['num_medications']
    if med_count <= 5:
        features['medication_load'] = 0
    elif med_count <= 10:
        features['medication_load'] = 1
    elif med_count <= 20:
        features['medication_load'] = 2
    else:
        features['medication_load'] = 3
    
    # Categorize stay
    stay = features['time_in_hospital']
    if stay <= 2:
        features['stay_category'] = 0
    elif stay <= 5:
        features['stay_category'] = 1
    elif stay <= 10:
        features['stay_category'] = 2
    else:
        features['stay_category'] = 3
    
    # Default values
    features['admission_type_id'] = 1
    features['discharge_disposition_id'] = 1
    features['admission_source_id'] = 7
    features['change'] = 0
    features['diabetesMed'] = 1
    
    # Create DataFrame with correct column order
    X = pd.DataFrame([features])
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    X = X[feature_names]
    
    return X

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Hospital Readmission Prediction API',
        'version': '1.0.0',
        'model_loaded': MODEL_LOADED,
        'endpoints': {
            'predict': '/predict (POST) - Single patient prediction',
            'predict_batch': '/predict_batch (POST) - Batch predictions',
            'health': '/health - API health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single patient prediction endpoint"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare features
        X = prepare_features(data)
        
        # Predict
        risk_score = model.predict_proba(X)[0][1]
        risk_class = model.predict(X)[0]
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'Low'
        elif risk_score < 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Generate recommendations
        if risk_level == 'High':
            recommendations = [
                'Schedule follow-up within 3-5 days',
                'Medication reconciliation required',
                'Consider home health evaluation',
                'Assign care coordinator',
                'Phone check-in within 24-48 hours'
            ]
        elif risk_level == 'Medium':
            recommendations = [
                'Schedule follow-up within 7-14 days',
                'Medication review',
                'Phone check-in within 3-5 days'
            ]
        else:
            recommendations = [
                'Standard discharge procedures',
                'Routine follow-up in 30 days'
            ]
        
        return jsonify({
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_percentage': f"{risk_score:.1%}",
            'recommendations': recommendations,
            'model': 'Random Forest',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'No patients data provided'}), 400
        
        patients = data['patients']
        results = []
        
        for i, patient in enumerate(patients):
            try:
                X = prepare_features(patient)
                risk_score = model.predict_proba(X)[0][1]
                
                if risk_score < 0.3:
                    risk_level = 'Low'
                elif risk_score < 0.6:
                    risk_level = 'Medium'
                else:
                    risk_level = 'High'
                
                results.append({
                    'patient_id': patient.get('patient_id', i),
                    'risk_score': float(risk_score),
                    'risk_level': risk_level,
                    'risk_percentage': f"{risk_score:.1%}"
                })
            except Exception as e:
                results.append({
                    'patient_id': patient.get('patient_id', i),
                    'error': str(e)
                })
        
        return jsonify({
            'total_patients': len(patients),
            'predictions': results,
            'model': 'Random Forest',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Hospital Readmission API...")
    print("📍 Endpoints:")
    print("   - http://localhost:5000/")
    print("   - http://localhost:5000/health")
    print("   - http://localhost:5000/predict")
    print("   - http://localhost:5000/predict_batch")
    app.run(host='0.0.0.0', port=5000, debug=False)

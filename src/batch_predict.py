"""
Batch Prediction Script
Process multiple patients from CSV file
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime

def load_model():
    """Load trained model"""
    with open('models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/processed/features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

def prepare_patient_features(row):
    """Convert DataFrame row to features"""
    features = {}
    
    # Age mapping
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    features['age'] = age_mapping.get(row.get('age', '[50-60)'), 55)
    
    # Gender
    gender_map = {'Male': 1, 'Female': 0}
    features['gender'] = gender_map.get(row.get('gender', 'Female'), 0)
    
    # Race
    race_map = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4}
    features['race'] = race_map.get(row.get('race', 'Caucasian'), 4)
    
    # Numeric features
    features['time_in_hospital'] = row.get('time_in_hospital', 3)
    features['num_lab_procedures'] = row.get('num_lab_procedures', 40)
    features['num_procedures'] = row.get('num_procedures', 1)
    features['num_medications'] = row.get('num_medications', 15)
    features['number_outpatient'] = row.get('number_outpatient', 0)
    features['number_emergency'] = row.get('number_emergency', 0)
    features['number_inpatient'] = row.get('number_inpatient', 0)
    features['number_diagnoses'] = row.get('number_diagnoses', 7)
    
    # Engineered features
    features['total_visits'] = (features['number_outpatient'] + 
                                features['number_emergency'] + 
                                features['number_inpatient'])
    features['multiple_diagnoses'] = 1 if features['number_diagnoses'] > 3 else 0
    
    # Medication load
    med_count = features['num_medications']
    if med_count <= 5:
        features['medication_load'] = 0
    elif med_count <= 10:
        features['medication_load'] = 1
    elif med_count <= 20:
        features['medication_load'] = 2
    else:
        features['medication_load'] = 3
    
    # Stay category
    stay = features['time_in_hospital']
    if stay <= 2:
        features['stay_category'] = 0
    elif stay <= 5:
        features['stay_category'] = 1
    elif stay <= 10:
        features['stay_category'] = 2
    else:
        features['stay_category'] = 3
    
    # Defaults
    features['admission_type_id'] = 1
    features['discharge_disposition_id'] = 1
    features['admission_source_id'] = 7
    features['change'] = 0
    features['diabetesMed'] = 1
    
    return features

def batch_predict(input_file, output_file=None):
    """Process batch predictions from CSV"""
    print(f"📊 Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} patients")
    
    # Load model
    print("🤖 Loading model...")
    model, feature_names = load_model()
    
    # Prepare features
    print("⚙️ Preparing features...")
    all_features = []
    for idx, row in df.iterrows():
        features = prepare_patient_features(row)
        all_features.append(features)
    
    X = pd.DataFrame(all_features)
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    X = X[feature_names]
    
    # Predict
    print("🔮 Making predictions...")
    risk_scores = model.predict_proba(X)[:, 1]
    risk_classes = model.predict(X)
    
    # Add to dataframe
    df['risk_score'] = risk_scores
    df['risk_percentage'] = (risk_scores * 100).round(1).astype(str) + '%'
    df['risk_level'] = pd.cut(risk_scores, 
                              bins=[0, 0.3, 0.6, 1.0], 
                              labels=['Low', 'Medium', 'High'])
    
    # Summary statistics
    print("\n📈 Summary:")
    print(f"   High Risk: {(risk_scores >= 0.6).sum()} ({(risk_scores >= 0.6).mean()*100:.1f}%)")
    print(f"   Medium Risk: {((risk_scores >= 0.3) & (risk_scores < 0.6)).sum()} ({((risk_scores >= 0.3) & (risk_scores < 0.6)).mean()*100:.1f}%)")
    print(f"   Low Risk: {(risk_scores < 0.3).sum()} ({(risk_scores < 0.3).mean()*100:.1f}%)")
    print(f"   Average Risk: {risk_scores.mean():.1%}")
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'predictions_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch prediction for hospital readmission')
    parser.add_argument('input', help='Input CSV file with patient data')
    parser.add_argument('--output', '-o', help='Output CSV file (optional)')
    
    args = parser.parse_args()
    
    batch_predict(args.input, args.output)
    print("\n✅ Batch prediction complete!")

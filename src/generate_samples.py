"""
Sample Patient Data Generator
Creates test data for the dashboard
"""

import pandas as pd
import numpy as np
import random

def generate_sample_patients(n=10):
    """Generate sample patient data"""
    
    age_groups = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                  '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    genders = ['Male', 'Female']
    races = ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other']
    
    patients = []
    
    for i in range(n):
        patient = {
            'patient_id': f'P{i+1:04d}',
            'age': random.choice(age_groups),
            'gender': random.choice(genders),
            'race': random.choice(races),
            'time_in_hospital': random.randint(1, 14),
            'num_lab_procedures': random.randint(10, 100),
            'num_procedures': random.randint(0, 6),
            'num_medications': random.randint(1, 40),
            'number_outpatient': random.randint(0, 10),
            'number_emergency': random.randint(0, 5),
            'number_inpatient': random.randint(0, 5),
            'number_diagnoses': random.randint(1, 16)
        }
        patients.append(patient)
    
    return pd.DataFrame(patients)

def generate_risk_profiles():
    """Generate patients with specific risk profiles for testing"""
    
    patients = []
    
    # Low risk patient
    patients.append({
        'patient_id': 'LOW001',
        'age': '[20-30)',
        'gender': 'Male',
        'race': 'Caucasian',
        'time_in_hospital': 2,
        'num_lab_procedures': 20,
        'num_procedures': 0,
        'num_medications': 5,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 0,
        'number_diagnoses': 2
    })
    
    # Medium risk patient
    patients.append({
        'patient_id': 'MED001',
        'age': '[50-60)',
        'gender': 'Female',
        'race': 'African American',
        'time_in_hospital': 5,
        'num_lab_procedures': 45,
        'num_procedures': 2,
        'num_medications': 12,
        'number_outpatient': 2,
        'number_emergency': 1,
        'number_inpatient': 1,
        'number_diagnoses': 5
    })
    
    # High risk patient
    patients.append({
        'patient_id': 'HIGH001',
        'age': '[70-80)',
        'gender': 'Female',
        'race': 'Caucasian',
        'time_in_hospital': 10,
        'num_lab_procedures': 80,
        'num_procedures': 4,
        'num_medications': 25,
        'number_outpatient': 5,
        'number_emergency': 3,
        'number_inpatient': 2,
        'number_diagnoses': 9
    })
    
    return pd.DataFrame(patients)

if __name__ == '__main__':
    import os
    os.makedirs('data/samples', exist_ok=True)
    
    # Generate random samples
    print("🎲 Generating random sample patients...")
    random_samples = generate_sample_patients(20)
    random_samples.to_csv('data/samples/random_patients.csv', index=False)
    print(f"   Saved {len(random_samples)} random patients to data/samples/random_patients.csv")
    
    # Generate risk profile samples
    print("\n🎯 Generating risk profile samples...")
    risk_samples = generate_risk_profiles()
    risk_samples.to_csv('data/samples/risk_profile_patients.csv', index=False)
    print(f"   Saved {len(risk_samples)} risk profile patients to data/samples/risk_profile_patients.csv")
    
    print("\n✅ Sample data generated!")
    print("\nUsage:")
    print("   python src/batch_predict.py data/samples/random_patients.csv")
    print("   python src/batch_predict.py data/samples/risk_profile_patients.csv")

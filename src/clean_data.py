"""
Data Cleaning Script for Hospital Readmission Project
Processes the raw UCI Diabetes dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def load_and_clean_data(filepath='data/diabetic_data.csv'):
    """Load and clean the diabetes dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features")
    
    # Initial info
    print(f"\n🔍 Initial Analysis:")
    print(f"   - Missing values per column: {(df == '?').sum().sum():,}")
    print(f"   - Duplicates: {df.duplicated().sum():,}")
    
    # Step 1: Handle missing values (marked as '?')
    print("\n🧹 Cleaning missing values...")
    df = df.replace('?', np.nan)
    
    # Drop columns with too many missing values (>50%)
    missing_pct = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    print(f"   - Dropping columns with >50% missing: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    # Step 2: Convert age groups to numeric midpoints
    print("\n📈 Processing age groups...")
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_mapping)
    
    # Step 3: Create readmission target (binary: <30 vs rest)
    print("\n🎯 Creating target variable...")
    # <30 = readmitted within 30 days (high risk)
    # NO or >30 = not readmitted within 30 days (low risk)
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    print(f"   - High risk (readmitted <30 days): {df['readmitted_30'].sum():,} ({df['readmitted_30'].mean()*100:.1f}%)")
    print(f"   - Low risk: {(1-df['readmitted_30']).sum():,} ({(1-df['readmitted_30']).mean()*100:.1f}%)")
    
    # Step 4: Feature engineering
    print("\n⚙️ Feature engineering...")
    
    # Number of medications categories
    df['medication_load'] = pd.cut(df['num_medications'], 
                                   bins=[0, 5, 10, 20, 100], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Length of stay categories
    df['stay_category'] = pd.cut(df['time_in_hospital'],
                                 bins=[0, 2, 5, 10, 100],
                                 labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    # Total visits (emergency + outpatient + inpatient)
    df['total_visits'] = df['number_emergency'] + df['number_outpatient'] + df['number_inpatient']
    
    # Has multiple diagnoses
    df['multiple_diagnoses'] = (df['number_diagnoses'] > 3).astype(int)
    
    print(f"   - Created features: medication_load, stay_category, total_visits, multiple_diagnoses")
    
    # Step 5: Select relevant features
    print("\n📋 Selecting features...")
    
    # Categorical features to encode
    categorical_features = ['race', 'gender', 'age', 'admission_type_id', 
                           'discharge_disposition_id', 'admission_source_id',
                           'medication_load', 'stay_category', 'change', 'diabetesMed']
    
    # Numeric features
    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                       'num_medications', 'number_outpatient', 'number_emergency',
                       'number_inpatient', 'number_diagnoses', 'total_visits',
                       'multiple_diagnoses']
    
    # Keep only existing columns
    categorical_features = [f for f in categorical_features if f in df.columns]
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    feature_cols = categorical_features + numeric_features
    
    # Step 6: Encode categorical variables
    print("\n🔤 Encoding categorical variables...")
    label_encoders = {}
    
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            # Handle missing values
            df[col] = df[col].fillna('Unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Step 7: Handle missing numeric values
    print("\n📝 Filling missing numeric values...")
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Prepare final dataset
    X = df[feature_cols].copy()
    y = df['readmitted_30'].copy()
    
    print(f"\n✅ Final dataset:")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Records: {X.shape[0]:,}")
    print(f"   - Missing values: {X.isnull().sum().sum()}")
    
    return X, y, feature_cols, label_encoders

def save_cleaned_data(X, y, feature_cols, label_encoders, 
                      output_dir='data/processed'):
    """Save cleaned data for model training"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features and target
    X.to_csv(f'{output_dir}/X_train.csv', index=False)
    y.to_csv(f'{output_dir}/y_train.csv', index=False)
    
    # Save feature list
    with open(f'{output_dir}/features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save label encoders
    with open(f'{output_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print(f"\n💾 Saved cleaned data to {output_dir}/")

if __name__ == "__main__":
    X, y, features, encoders = load_and_clean_data()
    save_cleaned_data(X, y, features, encoders)
    print("\n🎉 Data cleaning complete!")

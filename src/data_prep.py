"""
Data Preparation Module
Handles loading, cleaning, and preprocessing of hospital data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """Load dataset from CSV"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean and preprocess the data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Convert categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'readmitted':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

def feature_engineering(df):
    """Create new features from existing data"""
    # Age group
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 30, 50, 70, 100], 
                                 labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # Number of medications categories
    if 'num_medications' in df.columns:
        df['medication_load'] = pd.cut(df['num_medications'],
                                       bins=[0, 5, 10, 20, 50],
                                       labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df

def prepare_features(df, target_col='readmitted'):
    """Prepare X and y for modeling"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, scaler

if __name__ == "__main__":
    print("Data preparation module ready!")

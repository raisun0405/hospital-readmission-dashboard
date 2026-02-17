"""
Advanced Model Explainability with SHAP Values
Explains why the model makes specific predictions
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def calculate_shap_values(sample_size=1000):
    """Calculate SHAP values for model explainability"""
    try:
        import shap
    except ImportError:
        print("📦 Installing SHAP...")
        import subprocess
        subprocess.run(['pip', 'install', 'shap', '-q'])
        import shap
    
    print("📊 Loading data and model...")
    # Load sample of data
    X = pd.read_csv('data/processed/X_train.csv')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    # Sample for SHAP (too slow on full dataset)
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[indices]
    else:
        X_sample = X
    
    # Load model
    with open('models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('data/processed/features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"🔍 Calculating SHAP values for {len(X_sample)} samples...")
    print("   (This may take a few minutes...)")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, use values for class 1 (positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Calculate mean absolute SHAP values for feature importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_shap_value': mean_shap
    }).sort_values('mean_shap_value', ascending=False)
    
    # Save results
    shap_importance.to_csv('models/shap_feature_importance.csv', index=False)
    
    # Save SHAP values for later use
    np.save('models/shap_values.npy', shap_values)
    X_sample.to_csv('models/shap_sample_data.csv', index=False)
    
    print("\n📈 SHAP Feature Importance (Top 10):")
    print(shap_importance.head(10))
    
    print("\n💾 Saved SHAP analysis to models/")
    print("   - shap_feature_importance.csv")
    print("   - shap_values.npy")
    print("   - shap_sample_data.csv")
    
    return shap_importance

def explain_prediction(patient_features):
    """Explain a single prediction using SHAP"""
    try:
        import shap
    except:
        print("SHAP not installed")
        return None
    
    # Load model
    with open('models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for this patient
    shap_values = explainer.shap_values(patient_features)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Get feature names
    with open('data/processed/features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Create explanation
    explanation = []
    for i, (feature, value) in enumerate(zip(feature_names, shap_values[0])):
        explanation.append({
            'feature': feature,
            'shap_value': value,
            'impact': 'increases' if value > 0 else 'decreases'
        })
    
    # Sort by absolute impact
    explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    return explanation

if __name__ == '__main__':
    print("="*60)
    print("🔍 SHAP MODEL EXPLAINABILITY")
    print("="*60)
    
    shap_imp = calculate_shap_values(sample_size=500)
    
    print("\n🎉 SHAP analysis complete!")
    print("\nThis shows which features push predictions higher or lower.")

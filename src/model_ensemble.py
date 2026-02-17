"""
Model Ensemble and Comparison
Combines multiple models for better predictions
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def create_ensemble_model():
    """Create ensemble of Random Forest, XGBoost, and Logistic Regression"""
    
    print("📊 Loading all trained models...")
    
    # Load models
    models = {}
    
    try:
        with open('models/random_forest.pkl', 'rb') as f:
            models['rf'] = pickle.load(f)
        print("   ✅ Random Forest loaded")
    except:
        print("   ❌ Random Forest not found")
    
    try:
        with open('models/logistic_regression.pkl', 'rb') as f:
            models['lr'] = pickle.load(f)
        print("   ✅ Logistic Regression loaded")
    except:
        print("   ❌ Logistic Regression not found")
    
    try:
        with open('models/xgboost.pkl', 'rb') as f:
            models['xgb'] = pickle.load(f)
        print("   ✅ XGBoost loaded")
    except:
        print("   ❌ XGBoost not found")
    
    if len(models) < 2:
        print("❌ Need at least 2 models for ensemble")
        return None
    
    # Load data
    X = pd.read_csv('data/processed/X_train.csv')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🔀 Creating ensemble with {len(models)} models...")
    
    # Create voting ensemble
    estimators = [(name, model) for name, model in models.items()]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft'  # Use probabilities
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    ensemble_score = roc_auc_score(y_test, y_prob)
    ensemble_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Ensemble Results:")
    print(f"   ✅ ROC-AUC: {ensemble_score:.3f}")
    print(f"   ✅ Accuracy: {ensemble_acc:.3f}")
    
    # Compare individual models
    print("\n📊 Individual Model Comparison:")
    for name, model in models.items():
        y_prob_ind = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_prob_ind)
        print(f"   - {name}: {score:.3f}")
    print(f"   - Ensemble: {ensemble_score:.3f}")
    
    # Save ensemble
    with open('models/ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    
    print("\n💾 Ensemble model saved to models/ensemble.pkl")
    
    return ensemble

def compare_all_models_detailed():
    """Detailed comparison of all models"""
    
    print("📊 Loading all model metrics...")
    
    # Load all metrics files
    all_metrics = {}
    
    files = ['metrics.json', 'enhanced_metrics.json', 'all_metrics.json']
    for file in files:
        try:
            with open(f'models/{file}', 'r') as f:
                metrics = json.load(f)
                all_metrics.update(metrics)
        except:
            pass
    
    if not all_metrics:
        print("❌ No metrics found")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics).T
    
    print("\n" + "="*70)
    print("📊 COMPLETE MODEL COMPARISON (All Models)")
    print("="*70)
    print(comparison_df.round(3))
    
    # Find best model for each metric
    print("\n🏆 Best Models by Metric:")
    for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']:
        if metric in comparison_df.columns:
            best = comparison_df[metric].idxmax()
            score = comparison_df.loc[best, metric]
            print(f"   {metric.upper():12s}: {best:25s} ({score:.3f})")
    
    # Overall best
    best_overall = comparison_df['roc_auc'].idxmax()
    print(f"\n🥇 OVERALL BEST: {best_overall}")
    print(f"   ROC-AUC: {comparison_df.loc[best_overall, 'roc_auc']:.3f}")
    
    # Save comprehensive comparison
    comparison_df.to_csv('models/model_comparison.csv')
    print("\n💾 Saved comparison to models/model_comparison.csv")
    
    return comparison_df

if __name__ == '__main__':
    print("="*70)
    print("🎯 MODEL ENSEMBLE & COMPARISON")
    print("="*70)
    
    # Compare all models
    comparison = compare_all_models_detailed()
    
    # Try to create ensemble
    print("\n" + "="*70)
    ensemble = create_ensemble_model()
    
    print("\n🎉 Analysis complete!")

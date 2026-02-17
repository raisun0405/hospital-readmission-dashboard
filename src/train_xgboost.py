"""
XGBoost Model Training (Fixed Version)
Advanced gradient boosting without early stopping conflicts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def train_xgboost_model():
    """Train XGBoost model with optimized parameters"""
    
    try:
        import xgboost as xgb
    except ImportError:
        print("📦 Installing XGBoost...")
        import subprocess
        subprocess.run(['pip', 'install', 'xgboost', '-q'])
        import xgboost as xgb
    
    print("📊 Loading processed data...")
    X = pd.read_csv('data/processed/X_train.csv')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    with open('data/processed/features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    # XGBoost with optimized parameters (no early stopping for CV compatibility)
    print("\n🚀 Training XGBoost model...")
    print("   (This is computationally intensive...)")
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=2,
        eval_metric='logloss',
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbosity=0
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    results = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'cv_score': cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=2).mean()
    }
    
    print("\n" + "="*60)
    print("🎯 XGBOOST MODEL RESULTS")
    print("="*60)
    print(f"   ✅ Accuracy:  {results['accuracy']:.3f}")
    print(f"   ✅ Precision: {results['precision']:.3f}")
    print(f"   ✅ Recall:    {results['recall']:.3f}")
    print(f"   ✅ F1-Score:  {results['f1']:.3f}")
    print(f"   ✅ ROC-AUC:   {results['roc_auc']:.3f}")
    print(f"   ✅ CV Score:  {results['cv_score']:.3f}")
    
    # Save model
    print("\n💾 Saving model...")
    with open('models/xgboost.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance.to_csv('models/xgboost_feature_importance.csv', index=False)
    
    # Update metrics
    metrics = {
        'XGBoost': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'roc_auc': results['roc_auc'],
            'cv_score': results['cv_score']
        }
    }
    
    try:
        with open('models/all_metrics.json', 'r') as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing
    except:
        pass
    
    with open('models/all_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n📊 Feature Importance (Top 10):")
    print(importance.head(10))
    
    print("\n💾 Saved:")
    print("   - models/xgboost.pkl")
    print("   - models/xgboost_feature_importance.csv")
    print("   - models/all_metrics.json")
    
    return results

if __name__ == '__main__':
    print("="*60)
    print("🚀 XGBOOST MODEL TRAINING")
    print("="*60)
    print("\n⚠️  This is computationally intensive!")
    print("   Training on 81K samples with 150 trees...")
    print("="*60)
    
    results = train_xgboost_model()
    
    print("\n🎉 XGBoost training complete!")
    print(f"\n🏆 ROC-AUC: {results['roc_auc']:.3f}")

"""
Enhanced Model Training (Lightweight Version)
Trains models without heavy GridSearch to avoid timeouts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(data_dir='data/processed'):
    """Load cleaned data"""
    X = pd.read_csv(f'{data_dir}/X_train.csv')
    y = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    
    with open(f'{data_dir}/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return X, y, features

def train_enhanced_models(X, y, features):
    """Train enhanced models with better parameters (no GridSearch)"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {len(X_train):,} samples")
    print(f"📊 Test set: {len(X_test):,} samples")
    
    # Enhanced Random Forest (tuned parameters)
    models = {
        'Random Forest (Enhanced)': RandomForestClassifier(
            n_estimators=200,        # More trees
            max_depth=15,            # Deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("🤖 TRAINING ENHANCED MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n📌 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'cv_score': cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean(),
            'predictions': y_pred,
            'probabilities': y_prob,
            'y_test': y_test
        }
        
        print(f"   ✅ Accuracy: {results[name]['accuracy']:.3f}")
        print(f"   ✅ Precision: {results[name]['precision']:.3f}")
        print(f"   ✅ Recall: {results[name]['recall']:.3f}")
        print(f"   ✅ F1-Score: {results[name]['f1']:.3f}")
        print(f"   ✅ ROC-AUC: {results[name]['roc_auc']:.3f}")
        print(f"   ✅ CV ROC-AUC: {results[name]['cv_score']:.3f}")
    
    return results, features, X_train, X_test

def save_enhanced_models(results, output_dir='models'):
    """Save enhanced models"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {}
    for name, result in results.items():
        model_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        
        # Save model
        with open(f'{output_dir}/{model_name}.pkl', 'wb') as f:
            pickle.dump(result['model'], f)
        
        # Save feature importance
        if hasattr(result['model'], 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'feature': result['features'] if 'features' in result else range(len(result['model'].feature_importances_)),
                'importance': result['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            feat_imp.to_csv(f'{output_dir}/{model_name}_feature_importance.csv', index=False)
        
        metrics[name] = {
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'roc_auc': result['roc_auc'],
            'cv_score': result['cv_score']
        }
    
    # Save metrics
    with open(f'{output_dir}/enhanced_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Saved enhanced models to {output_dir}/")
    
    return metrics

def compare_all_models(output_dir='models'):
    """Compare all models"""
    
    # Load basic metrics
    try:
        with open(f'{output_dir}/metrics.json', 'r') as f:
            basic_metrics = json.load(f)
    except:
        basic_metrics = {}
    
    # Load enhanced metrics
    try:
        with open(f'{output_dir}/enhanced_metrics.json', 'r') as f:
            enhanced_metrics = json.load(f)
    except:
        enhanced_metrics = {}
    
    # Combine
    all_metrics = {**basic_metrics, **enhanced_metrics}
    
    print("\n" + "="*60)
    print("📊 COMPLETE MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(all_metrics).T
    print(comparison_df.round(3))
    
    # Best overall
    best_model = comparison_df['roc_auc'].idxmax()
    best_score = comparison_df.loc[best_model, 'roc_auc']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   ROC-AUC: {best_score:.3f}")
    
    return comparison_df

if __name__ == "__main__":
    # Load data
    X, y, features = load_processed_data()
    
    # Train enhanced models
    results, features, X_train, X_test = train_enhanced_models(X, y, features)
    
    # Add features to results for feature importance
    for name in results:
        results[name]['features'] = features
    
    # Save models
    metrics = save_enhanced_models(results)
    
    # Compare all models
    comparison = compare_all_models()
    
    print("\n🎉 Enhanced model training complete!")

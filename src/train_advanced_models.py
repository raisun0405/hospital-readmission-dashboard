"""
Advanced Model Training with XGBoost and Hyperparameter Tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
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

def train_advanced_models(X, y, features):
    """Train models with hyperparameter tuning"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {len(X_train):,} samples")
    print(f"📊 Test set: {len(X_test):,} samples")
    
    # Define models with hyperparameter grids
    models_config = {
        'Random Forest (Tuned)': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.05]
            }
        }
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("🤖 TRAINING ADVANCED MODELS")
    print("="*70)
    
    for name, config in models_config.items():
        print(f"\n📌 Training {name}...")
        print(f"   Grid Search with {len(config['params'])} parameter combinations...")
        
        # Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        print(f"   Best parameters: {grid_search.best_params_}")
        
        # Predict
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'cv_score': cross_val_score(best_model, X, y, cv=5, scoring='roc_auc').mean(),
            'predictions': y_pred,
            'probabilities': y_prob,
            'y_test': y_test
        }
        
        print(f"   ✅ Accuracy: {results[name]['accuracy']:.3f}")
        print(f"   ✅ ROC-AUC: {results[name]['roc_auc']:.3f}")
        print(f"   ✅ CV ROC-AUC: {results[name]['cv_score']:.3f}")
    
    return results, features, X_train, X_test

def save_advanced_models(results, output_dir='models'):
    """Save advanced models"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {}
    for name, result in results.items():
        model_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        
        # Save model
        with open(f'{output_dir}/{model_name}.pkl', 'wb') as f:
            pickle.dump(result['model'], f)
        
        # Save metrics
        metrics[name] = {
            'best_params': result['best_params'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'roc_auc': result['roc_auc'],
            'cv_score': result['cv_score']
        }
    
    # Append to existing metrics
    try:
        with open(f'{output_dir}/advanced_metrics.json', 'r') as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing
    except:
        pass
    
    with open(f'{output_dir}/advanced_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Saved advanced models to {output_dir}/")
    
    return metrics

def compare_all_models(output_dir='models'):
    """Compare all trained models (basic + advanced)"""
    
    # Load basic metrics
    try:
        with open(f'{output_dir}/metrics.json', 'r') as f:
            basic_metrics = json.load(f)
    except:
        basic_metrics = {}
    
    # Load advanced metrics
    try:
        with open(f'{output_dir}/advanced_metrics.json', 'r') as f:
            advanced_metrics = json.load(f)
    except:
        advanced_metrics = {}
    
    # Combine
    all_metrics = {**basic_metrics, **advanced_metrics}
    
    print("\n" + "="*70)
    print("📊 COMPLETE MODEL COMPARISON (Basic + Advanced)")
    print("="*70)
    
    comparison_df = pd.DataFrame(all_metrics).T
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_score']].round(3))
    
    # Best overall
    best_model = comparison_df['roc_auc'].idxmax()
    best_score = comparison_df.loc[best_model, 'roc_auc']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   ROC-AUC: {best_score:.3f}")
    
    return comparison_df

if __name__ == "__main__":
    # Load data
    X, y, features = load_processed_data()
    
    # Train advanced models
    results, features, X_train, X_test = train_advanced_models(X, y, features)
    
    # Save models
    metrics = save_advanced_models(results)
    
    # Compare all models
    comparison = compare_all_models()
    
    print("\n🎉 Advanced model training complete!")

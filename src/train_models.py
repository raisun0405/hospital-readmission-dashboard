"""
Model Training Script
Trains Logistic Regression and Random Forest for readmission prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

def load_processed_data(data_dir='data/processed'):
    """Load cleaned data"""
    X = pd.read_csv(f'{data_dir}/X_train.csv')
    y = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    
    with open(f'{data_dir}/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return X, y, features

def train_and_evaluate_models(X, y, features):
    """Train and evaluate multiple models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {len(X_train):,} samples")
    print(f"📊 Test set: {len(X_test):,} samples")
    print(f"   - Class distribution in train: {np.bincount(y_train)}")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("🤖 TRAINING MODELS")
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
            'cv_score': cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean(),
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

def get_feature_importance(model, features, model_name):
    """Extract feature importance"""
    if hasattr(model, 'feature_importances_'):
        # Random Forest
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feat_imp

def plot_feature_importance(results, features, output_dir='models'):
    """Plot feature importance for both models"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, (name, result) in enumerate(results.items()):
        model = result['model']
        feat_imp = get_feature_importance(model, features, name)
        
        if feat_imp is not None:
            # Plot top 10 features
            top_10 = feat_imp.head(10)
            axes[idx].barh(range(len(top_10)), top_10['importance'])
            axes[idx].set_yticks(range(len(top_10)))
            axes[idx].set_yticklabels(top_10['feature'])
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{name}\nTop 10 Features')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved feature importance plot to {output_dir}/feature_importance.png")

def plot_confusion_matrices(results, output_dir='models'):
    """Plot confusion matrices"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['y_test'], result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved confusion matrices to {output_dir}/confusion_matrices.png")

def save_models(results, features, output_dir='models'):
    """Save trained models and metadata"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each model
    for name, result in results.items():
        model_name = name.lower().replace(' ', '_')
        
        # Save model
        with open(f'{output_dir}/{model_name}.pkl', 'wb') as f:
            pickle.dump(result['model'], f)
        
        # Save feature importance
        feat_imp = get_feature_importance(result['model'], features, name)
        if feat_imp is not None:
            feat_imp.to_csv(f'{output_dir}/{model_name}_feature_importance.csv', index=False)
    
    # Save comparison metrics
    metrics = {}
    for name, result in results.items():
        metrics[name] = {
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'roc_auc': result['roc_auc'],
            'cv_score': result['cv_score']
        }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Saved models and metrics to {output_dir}/")
    
    return metrics

def print_summary(metrics):
    """Print final summary"""
    print("\n" + "="*60)
    print("📊 FINAL MODEL COMPARISON")
    print("="*60)
    
    comparison = pd.DataFrame(metrics).T
    print(comparison.round(3))
    
    # Best model
    best_model = comparison['roc_auc'].idxmax()
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   ROC-AUC: {comparison.loc[best_model, 'roc_auc']:.3f}")

if __name__ == "__main__":
    # Load data
    X, y, features = load_processed_data()
    
    # Train models
    results, features, X_train, X_test = train_and_evaluate_models(X, y, features)
    
    # Generate plots
    plot_feature_importance(results, features)
    plot_confusion_matrices(results)
    
    # Save everything
    metrics = save_models(results, features)
    
    # Print summary
    print_summary(metrics)
    
    print("\n🎉 Model training complete!")

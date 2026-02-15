"""
Model Training Module
Trains and evaluates ML models for readmission prediction
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

def train_models(X, y):
    """Train multiple models and return the best one"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cv_score': cross_val_score(model, X, y, cv=5).mean()
        }
    
    # Select best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    return best_model, results, X_train, X_test, y_train, y_test

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    return dict(zip(feature_names, importance))

def save_model(model, filepath='models/readmission_model.pkl'):
    """Save trained model to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath='models/readmission_model.pkl'):
    """Load trained model from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    print("Model training module ready!")

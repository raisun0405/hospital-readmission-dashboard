"""
Advanced Feature Engineering
Creates sophisticated features for better model performance
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features():
    """Create advanced engineered features"""
    
    print("📊 Loading processed data...")
    X = pd.read_csv('data/processed/X_train.csv')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    with open('data/processed/features.pkl', 'rb') as f:
        original_features = pickle.load(f)
    
    print(f"   Original features: {len(original_features)}")
    
    # Create copy for modification
    X_advanced = X.copy()
    
    # 1. Interaction features
    print("\n⚙️ Creating interaction features...")
    
    # Age x Length of stay (elderly patients with long stays)
    X_advanced['age_x_stay'] = X['age'] * X['time_in_hospital']
    
    # Medications x Diagnoses (polypharmacy with comorbidities)
    X_advanced['meds_x_diagnoses'] = X['num_medications'] * X['number_diagnoses']
    
    # Total visits x Inpatient (frequent flyer indicator)
    X_advanced['visits_x_inpatient'] = (X['number_outpatient'] + 
                                        X['number_emergency']) * X['number_inpatient']
    
    # 2. Ratio features
    print("⚙️ Creating ratio features...")
    
    # Medications per day of stay
    X_advanced['meds_per_day'] = X['num_medications'] / (X['time_in_hospital'] + 1)
    
    # Procedures per lab procedure
    X_advanced['proc_per_lab'] = X['num_procedures'] / (X['num_lab_procedures'] + 1)
    
    # Emergency ratio (emergency visits / total visits)
    total_visits = X['number_outpatient'] + X['number_emergency'] + X['number_inpatient']
    X_advanced['emergency_ratio'] = X['number_emergency'] / (total_visits + 1)
    
    # 3. Binning features
    print("⚙️ Creating binned features...")
    
    # Age groups (more granular)
    X_advanced['age_young'] = (X['age'] < 40).astype(int)
    X_advanced['age_middle'] = ((X['age'] >= 40) & (X['age'] < 65)).astype(int)
    X_advanced['age_senior'] = ((X['age'] >= 65) & (X['age'] < 80)).astype(int)
    X_advanced['age_elderly'] = (X['age'] >= 80).astype(int)
    
    # High medication indicator
    X_advanced['high_medication'] = (X['num_medications'] > 20).astype(int)
    
    # High lab procedures
    X_advanced['high_lab'] = (X['num_lab_procedures'] > 60).astype(int)
    
    # Frequent emergency visitor
    X_advanced['frequent_emergency'] = (X['number_emergency'] > 2).astype(int)
    
    # 4. Risk score features
    print("⚙️ Creating risk score features...")
    
    # Composite risk score (weighted sum of risk factors)
    X_advanced['composite_risk'] = (
        X['age'] / 100 * 0.2 +
        X['time_in_hospital'] / 14 * 0.2 +
        X['num_medications'] / 50 * 0.2 +
        X['number_diagnoses'] / 16 * 0.2 +
        X['number_inpatient'] / 10 * 0.2
    )
    
    # Severity index
    X_advanced['severity_index'] = (
        X['number_diagnoses'] + 
        X['num_procedures'] + 
        X['number_inpatient']
    )
    
    # 5. Statistical features
    print("⚙️ Creating statistical features...")
    
    # Clinical intensity (sum of all clinical activities)
    X_advanced['clinical_intensity'] = (
        X['num_lab_procedures'] + 
        X['num_procedures'] * 10 + 
        X['num_medications']
    )
    
    # Care complexity (interactions between different care types)
    X_advanced['care_complexity'] = (
        X['number_outpatient'] * X['number_emergency'] + 
        X['number_emergency'] * X['number_inpatient'] + 
        X['number_inpatient'] * X['number_outpatient']
    )
    
    # Get new feature names
    new_features = [col for col in X_advanced.columns if col not in original_features]
    print(f"\n✅ Created {len(new_features)} new features:")
    for feat in new_features:
        print(f"   - {feat}")
    
    # Save advanced features
    X_advanced.to_csv('data/processed/X_train_advanced.csv', index=False)
    
    # Save feature list
    all_features = list(X_advanced.columns)
    with open('data/processed/features_advanced.pkl', 'wb') as f:
        pickle.dump(all_features, f)
    
    # Calculate correlation with target for new features
    print("\n📈 Correlation with target (new features):")
    correlations = []
    for feat in new_features:
        corr = np.corrcoef(X_advanced[feat], y)[0, 1]
        correlations.append((feat, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    for feat, corr in correlations[:10]:
        print(f"   {feat:25s}: {corr:.3f}")
    
    print(f"\n💾 Saved to data/processed/")
    print(f"   - X_train_advanced.csv ({len(all_features)} features)")
    print(f"   - features_advanced.pkl")
    
    return X_advanced, all_features

def train_on_advanced_features():
    """Train model on advanced features"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    print("\n" + "="*60)
    print("🤖 TRAINING ON ADVANCED FEATURES")
    print("="*60)
    
    # Load advanced features
    X = pd.read_csv('data/processed/X_train_advanced.csv')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training with {X.shape[1]} features...")
    
    # Train Random Forest on advanced features
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=-1).mean()
    
    print(f"\n🎯 Results with Advanced Features:")
    print(f"   ✅ ROC-AUC: {roc_auc:.3f}")
    print(f"   ✅ Accuracy: {accuracy:.3f}")
    print(f"   ✅ CV Score: {cv_score:.3f}")
    
    # Save model
    with open('models/random_forest_advanced.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print(importance.head(10))
    
    importance.to_csv('models/random_forest_advanced_importance.csv', index=False)
    
    print("\n💾 Model saved to models/random_forest_advanced.pkl")
    
    return roc_auc

if __name__ == '__main__':
    print("="*60)
    print("🔧 ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    # Create features
    X_adv, features = create_advanced_features()
    
    # Train model
    score = train_on_advanced_features()
    
    print("\n🎉 Advanced feature engineering complete!")
    print(f"\n🏆 New model ROC-AUC: {score:.3f}")

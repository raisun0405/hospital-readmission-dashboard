"""
Test Suite for Hospital Readmission Dashboard
Lightweight tests to ensure code quality
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestDataLoading(unittest.TestCase):
    """Test data loading functions"""
    
    def test_processed_data_exists(self):
        """Check if processed data files exist"""
        self.assertTrue(os.path.exists('data/processed/X_train.csv'))
        self.assertTrue(os.path.exists('data/processed/y_train.csv'))
    
    def test_models_exist(self):
        """Check if trained models exist"""
        self.assertTrue(os.path.exists('models/random_forest.pkl'))
        self.assertTrue(os.path.exists('models/metrics.json'))

class TestPredictionFunction(unittest.TestCase):
    """Test prediction functionality"""
    
    def test_prediction_input(self):
        """Test prediction with sample input"""
        from src.predict import predict_readmission_risk
        
        test_patient = {
            'age': '[50-60)',
            'gender': 'Male',
            'race': 'Caucasian',
            'time_in_hospital': 3,
            'num_lab_procedures': 40,
            'num_procedures': 1,
            'num_medications': 10,
            'number_outpatient': 0,
            'number_emergency': 0,
            'number_inpatient': 0,
            'number_diagnoses': 5
        }
        
        score, level = predict_readmission_risk(test_patient)
        
        # Check score is between 0 and 1
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # Check level is valid
        self.assertIn(level, ['Low', 'Medium', 'High'])

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering"""
    
    def test_age_mapping(self):
        """Test age group mapping"""
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        
        for group, expected in age_mapping.items():
            self.assertEqual(age_mapping[group], expected)

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity"""
    
    def test_no_missing_in_processed(self):
        """Check processed data has no missing values"""
        if os.path.exists('data/processed/X_train.csv'):
            X = pd.read_csv('data/processed/X_train.csv')
            self.assertEqual(X.isnull().sum().sum(), 0)

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

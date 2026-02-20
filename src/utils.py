"""
Utility functions for the Hospital Readmission Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime

def format_risk_score(score):
    """Format risk score as percentage with color indicator"""
    percentage = score * 100
    if score >= 0.6:
        return f"🔴 {percentage:.1f}% (High Risk)"
    elif score >= 0.3:
        return f"🟡 {percentage:.1f}% (Medium Risk)"
    else:
        return f"🟢 {percentage:.1f}% (Low Risk)"

def get_risk_category(score):
    """Get risk category based on score"""
    if score >= 0.6:
        return 'High'
    elif score >= 0.3:
        return 'Medium'
    else:
        return 'Low'

def calculate_age_group(age):
    """Convert age to age group"""
    if age < 10:
        return '[0-10)'
    elif age < 20:
        return '[10-20)'
    elif age < 30:
        return '[20-30)'
    elif age < 40:
        return '[30-40)'
    elif age < 50:
        return '[40-50)'
    elif age < 60:
        return '[50-60)'
    elif age < 70:
        return '[60-70)'
    elif age < 80:
        return '[70-80)'
    elif age < 90:
        return '[80-90)'
    else:
        return '[90-100)'

def generate_patient_summary(patient_data, risk_score):
    """Generate a summary report for a patient"""
    summary = f"""
    PATIENT RISK ASSESSMENT REPORT
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    RISK SCORE: {risk_score:.1%}
    RISK LEVEL: {get_risk_category(risk_score)}
    
    PATIENT DETAILS:
    - Age: {patient_data.get('age', 'N/A')}
    - Gender: {patient_data.get('gender', 'N/A')}
    - Length of Stay: {patient_data.get('time_in_hospital', 'N/A')} days
    - Medications: {patient_data.get('num_medications', 'N/A')}
    - Diagnoses: {patient_data.get('number_diagnoses', 'N/A')}
    
    RECOMMENDATIONS:
    """
    
    if risk_score >= 0.6:
        summary += """
    ⚠️ HIGH RISK - Immediate Actions Required:
       • Schedule follow-up within 3-5 days
       • Assign care coordinator
       • Medication reconciliation
       • Consider home health services
        """
    elif risk_score >= 0.3:
        summary += """
    ⚡ MEDIUM RISK - Close Monitoring:
       • Schedule follow-up within 7-14 days
       • Phone check-in within 3-5 days
       • Review medications
        """
    else:
        summary += """
    ✓ LOW RISK - Standard Protocol:
       • Routine follow-up in 30 days
       • Standard discharge procedures
        """
    
    return summary

def validate_patient_data(data):
    """Validate patient data before prediction"""
    required_fields = ['age', 'gender', 'time_in_hospital', 'num_medications']
    
    missing = [field for field in required_fields if field not in data]
    
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    
    # Validate ranges
    if data.get('time_in_hospital', 0) < 1 or data.get('time_in_hospital', 0) > 14:
        return False, "time_in_hospital must be between 1 and 14"
    
    if data.get('num_medications', 0) < 0 or data.get('num_medications', 0) > 50:
        return False, "num_medications must be between 0 and 50"
    
    return True, "Valid"

def batch_process_status(current, total):
    """Display progress for batch processing"""
    percentage = (current / total) * 100
    bar_length = 30
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    return f"Progress: |{bar}| {percentage:.1f}% ({current}/{total})"

if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test risk formatting
    print(format_risk_score(0.75))
    print(format_risk_score(0.45))
    print(format_risk_score(0.15))
    
    # Test validation
    valid_data = {'age': '[50-60)', 'gender': 'Male', 'time_in_hospital': 3, 'num_medications': 10}
    is_valid, msg = validate_patient_data(valid_data)
    print(f"Validation: {is_valid}, {msg}")
    
    print("✅ All utilities working!")

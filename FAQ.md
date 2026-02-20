# FAQ - Hospital Readmission Dashboard

## General Questions

### What is this project?
A machine learning dashboard that predicts the risk of hospital readmission within 30 days of discharge. It helps healthcare providers identify high-risk patients for proactive care.

### Who is this for?
- Healthcare administrators
- Hospital quality improvement teams
- Medical researchers
- Data science students
- Anyone interested in healthcare analytics

### Is it free to use?
Yes! This is an open-source project under MIT License. You can use, modify, and distribute it freely.

## Technical Questions

### What technologies are used?
- **Python 3.13** - Programming language
- **scikit-learn** - Machine learning
- **Streamlit** - Web dashboard
- **Plotly** - Visualizations
- **pandas** - Data manipulation

### What is the model accuracy?
- **Random Forest**: 67.9% accuracy, 0.672 ROC-AUC
- **Logistic Regression**: 64.4% accuracy, 0.652 ROC-AUC
- **XGBoost**: Coming soon

### How was the model trained?
On the UCI Diabetes 130-US Hospitals dataset with 101,766 patient records from 1999-2008.

### Can I deploy this?
Yes! See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions on deploying to Streamlit Cloud.

### Is there an API?
Yes! We provide a Flask-based REST API. See `api.py` for details.

## Usage Questions

### How do I make a prediction?
1. Go to the "Predict Risk" page
2. Enter patient details (age, gender, hospital stay info)
3. Click "Calculate Risk"
4. View the risk score and recommendations

### What does the risk score mean?
- **0-30% (Green)**: Low risk - Standard discharge procedures
- **30-60% (Yellow)**: Medium risk - Close monitoring recommended
- **60-100% (Red)**: High risk - Immediate intervention required

### Can I process multiple patients?
Yes! Use the batch prediction feature:
```bash
python src/batch_predict.py your_data.csv
```

### How do I interpret the results?
The dashboard provides:
1. Risk percentage and level
2. Visual gauge
3. Personalized recommendations based on risk level
4. Feature importance (what factors influenced the prediction)

## Data Questions

### What data is needed?
Required fields:
- Age, gender, race
- Time in hospital
- Number of medications
- Number of diagnoses
- Prior visits (outpatient, emergency, inpatient)

### Where does the data come from?
The UCI Machine Learning Repository - Diabetes 130-US Hospitals dataset.

### Is patient data stored?
No! All predictions are made in real-time. No patient data is stored or logged.

### Can I use my own data?
Yes! Format your CSV with the same columns as our sample data and use the batch prediction tool.

## Troubleshooting

### The dashboard won't start
Make sure you've installed all requirements:
```bash
pip install -r requirements.txt
```

### Model file not found
Run the training script first:
```bash
python src/train_models.py
```

### Out of memory error
Try reducing the sample size in visualization scripts or use a machine with more RAM.

### API not responding
Check if the Flask server is running:
```bash
python api.py
```

## Contributing

### Can I contribute?
Yes! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### What can I work on?
Check the [CHANGELOG.md](CHANGELOG.md) for the roadmap or open issues for suggestions.

## Contact

### Who made this?
Rohan Vishwakarma - [@raisun0405](https://github.com/raisun0405)

### Where can I report bugs?
Open an issue on GitHub: https://github.com/raisun0405/hospital-readmission-dashboard/issues

### Is there a demo?
Coming soon! Will be deployed to Streamlit Cloud.

---

**Still have questions?** Open an issue on GitHub!

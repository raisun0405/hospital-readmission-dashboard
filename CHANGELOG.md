# Hospital Readmission Dashboard - Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-16

### Added
- Initial project setup with complete ML pipeline
- Data preprocessing and cleaning module
- Random Forest and Logistic Regression models
- Interactive Streamlit dashboard with 5 pages
- Real-time risk prediction functionality
- Data visualizations (age, stay, medications, demographics)
- Feature importance analysis
- Model performance metrics and comparison
- Comprehensive README documentation
- Deployment guide for Streamlit Cloud
- Unit tests for core functionality
- MIT License

### Models
- **Random Forest**: 67.9% accuracy, 0.672 ROC-AUC
- **Logistic Regression**: 64.4% accuracy, 0.652 ROC-AUC

### Dataset
- UCI Diabetes 130-US Hospitals dataset
- 101,766 patient records
- 50+ features
- 11.2% readmitted within 30 days

### Features
- Patient risk score prediction (0-100%)
- Risk categorization (Low/Medium/High)
- Clinical recommendations based on risk level
- Interactive data visualizations
- Model performance dashboard

## Future Roadmap

### [1.1.0] - Planned
- [ ] XGBoost model implementation
- [ ] Hyperparameter tuning
- [ ] SHAP values for explainability
- [ ] API endpoint for integration
- [ ] Time-series analysis

### [1.2.0] - Planned
- [ ] Patient similarity search
- [ ] Batch prediction upload
- [ ] Export predictions to PDF
- [ ] Email alerts for high-risk patients

### [2.0.0] - Planned
- [ ] Multi-hospital deployment
- [ ] Real-time data integration
- [ ] Mobile app companion
- [ ] Advanced analytics dashboard

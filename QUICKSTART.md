# Quick Start Guide

Get the Hospital Readmission Dashboard running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

## Installation (3 Steps)

### Step 1: Clone Repository
```bash
git clone https://github.com/raisun0405/hospital-readmission-dashboard.git
cd hospital-readmission-dashboard
```

### Step 2: Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Step 3: Run Dashboard
```bash
streamlit run app.py
```

🎉 **Done!** Open http://localhost:8501 in your browser.

## What's Next?

### Make Predictions
1. Click "Predict Risk" in the sidebar
2. Enter patient details
3. Get instant risk score

### Try Batch Processing
```bash
python src/batch_predict.py data/samples/random_patients.csv
```

### Use the API
```bash
python api.py
# API runs at http://localhost:5000
```

## Common Issues

**Import Error?**
```bash
pip install -r requirements.txt
```

**Model not found?**
The models are included in the repo. If missing:
```bash
python src/train_models.py
```

**Port already in use?**
Change the port:
```bash
streamlit run app.py --server.port 8502
```

## Learn More

- [README.md](README.md) - Full documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deploy to cloud
- [FAQ.md](FAQ.md) - Common questions

## Need Help?

Open an issue on GitHub: https://github.com/raisun0405/hospital-readmission-dashboard/issues

---

**Happy Predicting! 🏥🔮**

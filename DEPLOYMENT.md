# Deployment Guide for Hospital Readmission Dashboard

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1: Prepare Repository
Ensure your GitHub repo has:
- ✅ `app.py` in root directory
- ✅ `requirements.txt` with all dependencies
- ✅ `README.md`
- ✅ Models in `models/` folder (tracked by Git)

### Step 2: Sign Up
1. Go to https://streamlit.io/cloud
2. Sign up with GitHub account (@raisun0405)
3. Authorize Streamlit to access your repos

### Step 3: Deploy
1. Click "New app"
2. Select repository: `raisun0405/hospital-readmission-dashboard`
3. Branch: `main`
4. Main file path: `app.py`
5. Click "Deploy"

### Step 4: Wait for Build
- Streamlit will install dependencies from `requirements.txt`
- Build typically takes 2-3 minutes
- You'll get a URL like: `https://hospital-readmission-dashboard-raisun0405.streamlit.app`

### Step 5: Share
- Share the URL on your resume
- Add to LinkedIn projects
- Include in job applications

---

## 🔧 Local Development

```bash
# Navigate to project
cd hospital-readmission-dashboard

# Activate virtual environment
source ~/.openclaw/workspace/venv/bin/activate

# Run Streamlit
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📦 Requirements

Make sure `requirements.txt` includes:
```
streamlit==1.31.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
kaleido==0.2.3
```

---

## 🐛 Troubleshooting

### Issue: Module not found
**Fix:** Add to `requirements.txt` and redeploy

### Issue: Model files not loading
**Fix:** Ensure models are pushed to GitHub:
```bash
git add models/*.pkl models/*.json
git commit -m "Add model files"
git push origin main
```

### Issue: Data files too large
**Fix:** Large CSV files should be downloaded at runtime, not stored in Git

---

## 📊 Post-Deployment

### Add to Resume:
```
Hospital Readmission Prediction Dashboard
- Built ML model to predict 30-day hospital readmission risk
- Processed 101K+ patient records; achieved 67% ROC-AUC
- Created interactive Streamlit dashboard with risk scoring
- Live Demo: [your-streamlit-url]
- GitHub: github.com/raisun0405/hospital-readmission-dashboard
```

### Add to LinkedIn:
- Project URL: [your-streamlit-url]
- GitHub: github.com/raisun0405/hospital-readmission-dashboard
- Screenshot: models/feature_importance.png

---

🎉 **Ready to deploy!**

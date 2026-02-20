#!/bin/bash
# Quick setup script for Hospital Readmission Dashboard

echo "🏥 Hospital Readmission Dashboard - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.10+"; exit 1; }

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo ""
echo "🚀 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install requirements
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q
echo "✅ Dependencies installed"

# Check if data exists
echo ""
echo "📊 Checking data..."
if [ ! -f "data/diabetic_data.csv" ]; then
    echo "📥 Downloading dataset..."
    python3 data/download_data.py
else
    echo "✅ Dataset already exists"
fi

# Check if models exist
echo ""
echo "🤖 Checking models..."
if [ ! -f "models/random_forest.pkl" ]; then
    echo "⚠️  Models not found. Training will be needed."
    echo "   Run: python3 src/train_models.py"
else
    echo "✅ Models found"
fi

echo ""
echo "================================================"
echo "🎉 Setup complete!"
echo ""
echo "To start the dashboard:"
echo "   source venv/bin/activate"
echo "   streamlit run app.py"
echo ""
echo "To start the API:"
echo "   source venv/bin/activate"
echo "   python api.py"
echo ""

#!/bin/bash
# Comprehensive Protein Burial Analysis Runner
# This script runs the complete analysis pipeline

echo "========================================"
echo "PROTEIN BURIAL CLASSIFICATION ANALYSIS"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================"
echo "PHASE 1: COMPREHENSIVE ANALYSIS"
echo "========================================"
python3 comprehensive_burial_analysis.py

echo ""
echo "========================================"
echo "PHASE 2: GENERATING VISUALIZATIONS"
echo "========================================"
python3 visualization_module.py

echo ""
echo "========================================"
echo "ANALYSIS COMPLETE!"
echo "========================================"
echo ""
echo "Check results/comprehensive_analysis/ for outputs"
echo ""
numpy>=1.21.0
pandas>=1.3.0
biopython>=1.79
scikit-learn>=1.0.0
optuna>=3.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0


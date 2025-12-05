#!/bin/bash
# Setup script for Final Analysis
# This script creates the necessary folder structure and tests the installation

echo "=========================================="
echo "Final Analysis - Setup Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "final_analysis.py" ]; then
    echo "ERROR: final_analysis.py not found"
    echo "Please run this script from the FINAL folder"
    exit 1
fi

echo "✓ Found final_analysis.py"
echo ""

# Go up one level to the main project folder
cd ..

# Create pdbexamples folder if it doesn't exist
if [ ! -d "pdbexamples" ]; then
    echo "Creating pdbexamples folder..."
    mkdir -p pdbexamples
    echo "✓ Created pdbexamples folder"
else
    echo "✓ pdbexamples folder already exists"
fi

# Check for PDB files in pdbexamples
PDB_COUNT=$(find pdbexamples -name "*.pdb" -o -name "*.ent" | wc -l)
if [ $PDB_COUNT -eq 0 ]; then
    echo ""
    echo "⚠️  No PDB files found in pdbexamples folder"
    echo "Please add PDB files to the pdbexamples folder before running analysis"
    echo ""
    echo "Example:"
    echo "  cp my_protein.pdb pdbexamples/"
    echo "  cp my_protein.ent pdbexamples/"
else
    echo "✓ Found $PDB_COUNT PDB file(s) in pdbexamples"
fi

echo ""

# Check Python installation
echo "Checking Python packages..."

python3 << 'EOF'
import sys

required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'Bio': 'biopython'
}

missing = []
for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
    except ImportError:
        print(f"  ✗ {package_name} - NOT INSTALLED")
        missing.append(package_name)

if missing:
    print("")
    print("Missing packages! Install with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "✓ All required packages installed"
echo ""

# Check for optional tools
echo "Checking optional tools..."
if command -v dssp &> /dev/null; then
    echo "  ✓ DSSP is installed"
else
    echo "  ⚠️  DSSP not found (optional, analysis will still work)"
fi

if command -v stride &> /dev/null; then
    echo "  ✓ STRIDE is installed"
else
    echo "  ⚠️  STRIDE not found (optional, analysis will still work)"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Add PDB files to the pdbexamples folder"
echo "  2. Edit configuration in FINAL/final_analysis.py if needed"
echo "  3. Run: cd FINAL && python final_analysis.py"
echo ""


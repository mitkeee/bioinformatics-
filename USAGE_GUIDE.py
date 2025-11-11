#!/usr/bin/env python3
"""
SIMPLE USAGE EXAMPLE - Run this to see the system in action
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║        COMPREHENSIVE PROTEIN BURIAL CLASSIFICATION ANALYSIS SYSTEM         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

QUICK START GUIDE
=================

1. BASIC ANALYSIS (Default Parameters)
   Run: python3 quick_analysis.py
   
   This will:
   ✓ Process all PDB files in the workspace
   ✓ Generate 2 confusion matrices per protein (vs DSSP and STRIDE)
   ✓ Calculate accuracy, precision, recall, F1-score
   ✓ Create visualizations
   ✓ Save detailed results

2. WITH PARAMETER OPTIMIZATION (Recommended for best accuracy)
   Run: python3 quick_analysis.py --optimize --trials 50
   
   This will:
   ✓ Test 50 different parameter combinations
   ✓ Use cross-validation to find best parameters
   ✓ Re-run analysis with optimized parameters
   ✓ Save optimization history

3. WITH CROSS-VALIDATION ONLY
   Run: python3 quick_analysis.py --cv-folds 5
   
   This will:
   ✓ Perform 5-fold cross-validation
   ✓ Report accuracy per fold
   ✓ Calculate mean ± std deviation

4. FOR DUDE DATASET (100 proteins)
   - Place all 100 PDB files in this directory
   - Run: python3 quick_analysis.py --optimize --trials 100 --cv-folds 5
   - Wait for analysis to complete (~10-30 minutes depending on size)
   - Check results/comprehensive_analysis/ for all outputs

═══════════════════════════════════════════════════════════════════════════

OUTPUT FILES EXPLAINED
======================

results/comprehensive_analysis/
├── baseline_summary_report.txt          ← Overall statistics, aggregate metrics
├── best_parameters.txt                  ← Optimized parameter values
├── confusion_matrices/                  ← Individual matrices per protein
│   ├── [protein]_confusion_matrix_dssp.csv
│   └── [protein]_confusion_matrix_stride.csv
├── [protein]_detailed_results.csv       ← Full data per protein
└── visualizations/                      ← All plots and graphs

═══════════════════════════════════════════════════════════════════════════

WHAT THE SYSTEM DOES
====================

✓ Extracts CA atoms from PDB files
✓ Calculates neighbor counts (6Å and 10Å spheres)
✓ Measures uniformity (homogeneous distribution)
✓ Classifies residues as Interior(0) or Exterior(1)
✓ Compares predictions vs DSSP reference
✓ Compares predictions vs STRIDE reference
✓ Generates 2 confusion matrices per protein
✓ Calculates accuracy for whole dataset
✓ Calculates accuracy per individual protein
✓ Identifies outliers (proteins with unusual accuracy)
✓ Optimizes 4 parameters to maximize accuracy
✓ Performs k-fold cross-validation
✓ Creates comprehensive visualizations

═══════════════════════════════════════════════════════════════════════════

UNDERSTANDING RESULTS
=====================

CONFUSION MATRIX:
                 Predicted
              Interior  Exterior
True Interior    TN        FP      (FP = False Positive, Type I error)
True Exterior    FN        TP      (FN = False Negative, Type II error)

METRICS:
- Accuracy = (TP + TN) / Total              [Overall correctness]
- Precision = TP / (TP + FP)                [Exterior prediction quality]
- Recall = TP / (TP + FN)                   [Exterior detection rate]
- F1-Score = 2*(Precision*Recall)/(P+R)     [Balanced metric]

CLASSIFICATION:
- Predictive Positive (1) = Exterior/Surface (your algorithm)
- Predictive Negative (0) = Interior/Buried (your algorithm)
- Real Positive (1) = DSSP/STRIDE says Exterior
- Real Negative (0) = DSSP/STRIDE says Interior

═══════════════════════════════════════════════════════════════════════════

READY TO START!

Current workspace has these PDB files ready:
- 3PTE.pdb
- 4d05.pdb
- 6wti.pdb
- 7upo.pdb

To begin, run:
    python3 quick_analysis.py

For help:
    python3 quick_analysis.py --help

═══════════════════════════════════════════════════════════════════════════
""")


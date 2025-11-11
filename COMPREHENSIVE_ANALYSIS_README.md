# Comprehensive Protein Burial Classification Analysis

## Overview

This comprehensive analysis system implements a **neighbor-based burial classification algorithm** for proteins and compares it against standard methods (DSSP and STRIDE). The system is designed to:

1. **Process multiple proteins** (DUDE dataset or custom sets)
2. **Generate 2 confusion matrices per protein** (vs DSSP and vs STRIDE)
3. **Optimize parameters** using Optuna and cross-validation
4. **Calculate accuracy metrics** for whole dataset and per-protein
5. **Identify outliers** and analyze performance patterns
6. **Generate comprehensive visualizations**

---

## Key Features

### 🔬 **Algorithm Description**

Our burial classification algorithm uses **neighbor-based geometric features**:

- **Neighbor counts** at 6Å and 10Å radii (`ncps_sphere_6`, `ncps_sphere_10`)
- **Uniformity metrics** (spherical variance) measuring homogeneous distribution
- **Classification logic**:
  - **Interior (buried) = 0**: Many neighbors, uniformly distributed (high uniformity)
  - **Exterior (surface) = 1**: Few neighbors OR one-sided distribution (low uniformity)

### 📊 **Comparison Methods**

- **DSSP**: Standard secondary structure and accessibility (cutoff: 30 Ų ASA)
- **STRIDE**: Alternative structure/accessibility method (cutoff: 24 Ų ASA)

### 🎯 **Metrics Calculated**

For each protein, against both DSSP and STRIDE:
- **Confusion Matrix** (2x2: True/Predicted Interior/Exterior)
- **Accuracy**
- **Precision** (positive predictive value)
- **Recall** (sensitivity)
- **F1-Score**

### 🔧 **Optimization**

- **Optuna framework** for hyperparameter optimization
- **k-fold cross-validation** (5-fold or 10-fold)
- **Parameters optimized**:
  - `nc6_threshold`: Neighbor count threshold at 6Å
  - `nc10_threshold`: Neighbor count threshold at 10Å
  - `uni6_threshold`: Uniformity threshold at 6Å
  - `uni10_threshold`: Uniformity threshold at 10Å

---

## Installation

### Prerequisites

- Python 3.8+
- DSSP (optional, for reference data)
- STRIDE (optional, for reference data)

### Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `biopython` - PDB parsing, DSSP integration
- `scikit-learn` - Machine learning metrics
- `optuna` - Hyperparameter optimization
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations

---

## Usage

### 🚀 **Quick Start**

```bash
# Basic analysis with default parameters
python3 quick_analysis.py

# Full optimization with 100 trials
python3 quick_analysis.py --optimize --trials 100

# 10-fold cross-validation
python3 quick_analysis.py --cv-folds 10

# Optimize using STRIDE as reference
python3 quick_analysis.py --optimize --reference stride

# Skip visualizations (faster)
python3 quick_analysis.py --no-viz
```

### 📝 **Full Pipeline**

```bash
# Run complete analysis (optimization + visualization)
./run_comprehensive_analysis.sh
```

Or manually:

```bash
# Step 1: Run comprehensive analysis
python3 comprehensive_burial_analysis.py

# Step 2: Generate visualizations
python3 visualization_module.py
```

---

## Output Structure

```
results/comprehensive_analysis/
├── baseline_summary_report.txt          # Overall statistics (default params)
├── optimized_summary_report.txt         # Overall statistics (optimized params)
├── best_parameters.txt                  # Optimized parameter values
├── optuna_optimization_trials.csv       # All optimization trials
├── cv_results_dssp.json                 # Cross-validation results
│
├── confusion_matrices/                  # Per-protein confusion matrices
│   ├── 3pte_confusion_matrix_dssp.csv
│   ├── 3pte_confusion_matrix_stride.csv
│   ├── 4d05_confusion_matrix_dssp.csv
│   └── ...
│
├── *_detailed_results.csv               # Per-protein full data
│   ├── 3pte_detailed_results.csv
│   ├── 4d05_detailed_results.csv
│   └── ...
│
└── visualizations/                      # Plots and graphs
    ├── accuracy_distribution_dssp.png
    ├── per_protein_accuracy_dssp.png
    ├── aggregate_confusion_matrix_dssp.png
    ├── f1_scores_comparison.png
    ├── outlier_analysis_dssp.png
    ├── cross_validation_results_dssp.png
    └── optuna_optimization_analysis.png
```

---

## Understanding the Output

### 📄 **Summary Reports**

The summary report contains:

1. **Overall Statistics**
   - Total proteins and residues analyzed
   - Mean accuracy across all proteins
   - Standard deviation, min, max, median
   - Aggregate confusion matrices

2. **Per-Protein Results**
   - Accuracy, Precision, Recall, F1-Score
   - For both DSSP and STRIDE comparisons

3. **Outlier Analysis**
   - Low-performance proteins (< mean - 1σ)
   - High-performance proteins (> mean + 1σ)

### 🎨 **Visualizations**

1. **Accuracy Distribution**: Histogram and box plot showing accuracy spread
2. **Per-Protein Accuracy**: Bar chart with color-coded performance
3. **Confusion Matrix Heatmaps**: Aggregate matrices for DSSP/STRIDE
4. **F1-Score Comparison**: Side-by-side comparison of DSSP vs STRIDE
5. **Outlier Analysis**: Scatter plot identifying unusual proteins
6. **Cross-Validation Results**: Fold-by-fold performance
7. **Optuna Optimization**: Parameter evolution and importance

### 📊 **Confusion Matrix Interpretation**

```
                Predicted
             Interior  Exterior
True Interior    TN        FP      (FP = Type I error)
True Exterior    FN        TP      (FN = Type II error)
```

- **TN (True Negative)**: Correctly predicted interior/buried
- **TP (True Positive)**: Correctly predicted exterior/surface
- **FP (False Positive)**: Interior wrongly predicted as exterior
- **FN (False Negative)**: Exterior wrongly predicted as interior

---

## Key Parameters

### **Classification Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nc6_threshold` | 10.0 | Minimum neighbors at 6Å for interior classification |
| `nc10_threshold` | 18.0 | Minimum neighbors at 10Å for interior classification |
| `uni6_threshold` | 0.40 | Minimum uniformity at 6Å for interior classification |
| `uni10_threshold` | 0.50 | Minimum uniformity at 10Å for interior classification |
| `dssp_asa_cutoff` | 30.0 | ASA threshold for DSSP classification (Ų) |
| `stride_asa_cutoff` | 24.0 | ASA threshold for STRIDE classification (Ų) |

### **Optimization Settings**

- **n_trials**: Number of parameter combinations to test (default: 50)
- **n_folds**: Cross-validation folds (default: 5)
- **reference**: Which method to optimize against ('dssp' or 'stride')

---

## Workflow Explanation

### **Phase 1: Baseline Analysis**
1. Extract CA atoms from PDB files
2. Calculate distance matrices
3. Compute neighbor counts (6Å, 10Å spheres)
4. Calculate uniformity (spherical variance)
5. Extract DSSP and STRIDE reference data
6. Classify using default parameters
7. Generate confusion matrices (2 per protein)
8. Calculate accuracy metrics

### **Phase 2: Parameter Optimization**
1. Define parameter search space
2. Run Optuna optimization:
   - For each trial:
     - Sample parameter values
     - Perform k-fold cross-validation
     - Calculate mean accuracy
   - Track best parameters
3. Save optimization history

### **Phase 3: Final Analysis**
1. Re-run analysis with optimized parameters
2. Generate comprehensive reports
3. Identify outliers
4. Create visualizations

### **Phase 4: Statistical Analysis**
1. Compare whole-dataset accuracy vs per-protein
2. Analyze why some proteins have low accuracy
3. Check correlations with protein size, structure, etc.

---

## Cross-Validation Details

**Classic k-fold cross-validation** at the protein level:

1. Split proteins into k folds (e.g., 5 folds)
2. For each fold:
   - Use 80% proteins as training context
   - Test on remaining 20%
   - Calculate accuracy
3. Average accuracy across all folds
4. Report mean ± standard deviation

This allows us to:
- Assess generalization performance
- Identify overfitting
- Compare different parameter sets

---

## Interpreting Results

### ✅ **Good Performance Indicators**
- **High accuracy** (>80%) on most proteins
- **Low standard deviation** across proteins
- **Balanced confusion matrix** (no extreme FP or FN)
- **Consistent cross-validation** scores

### ⚠️ **Warning Signs**
- **Large variance** between proteins
- **Many outliers** (proteins with very low accuracy)
- **Unbalanced confusion matrix** (many FP or FN)
- **Poor cross-validation** consistency

### 🔍 **Investigating Low-Accuracy Proteins**

When proteins show low accuracy:
1. **Check protein size**: Very small/large proteins may behave differently
2. **Examine structure**: Unusual folds, membrane proteins, etc.
3. **Review DSSP/STRIDE agreement**: If they disagree, our method may be correct
4. **Visualize residues**: Look at 3D structure to understand errors
5. **Check feature distributions**: Unusual neighbor counts or uniformity

---

## Advanced Usage

### **Custom Protein Dataset**

Place PDB files in the workspace directory:
```bash
/Users/famnit/Desktop/pythonProject/
├── protein1.pdb
├── protein2.pdb
└── ...
```

### **Adjust Optimization**

Edit `quick_analysis.py` to modify:
- Parameter search ranges
- Number of trials
- Cross-validation strategy
- Optimization objectives (accuracy, F1, etc.)

### **Use Different Reference Methods**

```python
# In comprehensive_burial_analysis.py
params = BurialParameters(
    dssp_asa_cutoff=25.0,  # Adjust DSSP threshold
    stride_asa_cutoff=20.0  # Adjust STRIDE threshold
)
```

---

## Troubleshooting

### **No DSSP/STRIDE data**
- Install DSSP: See `INSTALL_STRIDE.md`
- Generate STRIDE files: `python3 generate_stride_files.py`
- Pre-generate files for faster processing

### **Memory issues with large datasets**
- Process proteins in batches
- Reduce optimization trials
- Skip visualization generation

### **Low accuracy across all proteins**
- Check parameter ranges
- Increase optimization trials
- Try different reference thresholds
- Validate PDB file quality

---

## Citation and References

If you use this analysis system, please reference:

- **DSSP**: Kabsch & Sander (1983)
- **STRIDE**: Frishman & Argos (1995)
- **Optuna**: Akiba et al. (2019)

---

## Contact and Support

For questions or issues, check:
1. `ANALYSIS_SUMMARY.md` - Overview of methodology
2. `COMPLETE_FEATURE_LIST.md` - Feature checklist
3. `MARKO_QUESTIONS_ANSWERED.txt` - FAQ

---

## Summary

This comprehensive analysis system provides:

✅ **Automated processing** of protein datasets  
✅ **Dual comparison** against DSSP and STRIDE  
✅ **Parameter optimization** using state-of-art methods  
✅ **Cross-validation** for robust evaluation  
✅ **Per-protein AND whole-dataset metrics**  
✅ **Outlier detection** and analysis  
✅ **Rich visualizations** for interpretation  
✅ **Detailed reports** and confusion matrices  

**Goal**: Optimize neighbor-based burial classification to achieve highest accuracy when compared to standard methods, while understanding why certain proteins perform differently.
#!/usr/bin/env python3
"""
Quick Analysis Script - Run comprehensive burial analysis with simple options
Usage: python quick_analysis.py [--optimize] [--cv-folds N] [--trials N]
"""

import argparse
import sys
from pathlib import Path

# Import main analysis module
from comprehensive_burial_analysis import (
    BurialParameters,
    process_protein_dataset,
    save_confusion_matrices,
    generate_summary_report,
    cross_validate_parameters,
    optimize_parameters_optuna
)

# Import visualization
from visualization_module import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(description='Quick Protein Burial Analysis')
    parser.add_argument('--optimize', action='store_true', 
                        help='Run parameter optimization with Optuna')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials for optimization (default: 50)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--reference', choices=['dssp', 'stride'], default='dssp',
                        help='Reference method for optimization (default: dssp)')
    
    args = parser.parse_args()
    
    # Setup
    workspace_dir = Path.cwd()
    output_dir = workspace_dir / "results" / "comprehensive_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDB files
    pdb_files = sorted(workspace_dir.glob("*.pdb"))
    print(f"\n{'='*80}")
    print(f"Found {len(pdb_files)} PDB files to analyze")
    print(f"{'='*80}\n")
    
    if len(pdb_files) == 0:
        print("ERROR: No PDB files found in workspace!")
        sys.exit(1)
    
    for pdb in pdb_files[:5]:  # Show first 5
        print(f"  - {pdb.name}")
    if len(pdb_files) > 5:
        print(f"  ... and {len(pdb_files) - 5} more")
    
    # Default parameters
    params = BurialParameters(
        nc6_threshold=10.0,
        nc10_threshold=18.0,
        uni6_threshold=0.40,
        uni10_threshold=0.50,
        dssp_asa_cutoff=30.0,
        stride_asa_cutoff=24.0
    )
    
    if args.optimize:
        print(f"\n{'='*80}")
        print("RUNNING PARAMETER OPTIMIZATION")
        print(f"{'='*80}\n")
        print(f"Configuration:")
        print(f"  - Optimization trials: {args.trials}")
        print(f"  - Cross-validation folds: {args.cv_folds}")
        print(f"  - Reference method: {args.reference.upper()}")
        print()
        
        # Optimize
        best_params, study = optimize_parameters_optuna(
            pdb_files,
            n_trials=args.trials,
            reference=args.reference,
            n_folds=args.cv_folds
        )
        
        # Save optimization results
        optuna_df = study.trials_dataframe()
        optuna_df.to_csv(output_dir / "optuna_optimization_trials.csv", index=False)
        
        # Use optimized parameters
        params = best_params
        
        # Save best parameters
        with open(output_dir / "best_parameters.txt", 'w') as f:
            f.write("OPTIMIZED PARAMETERS\n")
            f.write("="*80 + "\n\n")
            f.write(f"nc6_threshold: {params.nc6_threshold:.4f}\n")
            f.write(f"nc10_threshold: {params.nc10_threshold:.4f}\n")
            f.write(f"uni6_threshold: {params.uni6_threshold:.4f}\n")
            f.write(f"uni10_threshold: {params.uni10_threshold:.4f}\n")
            f.write(f"dssp_asa_cutoff: {params.dssp_asa_cutoff:.4f}\n")
            f.write(f"stride_asa_cutoff: {params.stride_asa_cutoff:.4f}\n")
    else:
        print(f"\n{'='*80}")
        print("RUNNING ANALYSIS WITH DEFAULT PARAMETERS")
        print(f"{'='*80}\n")
        print("Default parameters:")
        print(f"  nc6_threshold: {params.nc6_threshold}")
        print(f"  nc10_threshold: {params.nc10_threshold}")
        print(f"  uni6_threshold: {params.uni6_threshold}")
        print(f"  uni10_threshold: {params.uni10_threshold}")
        print()
        print("(Use --optimize flag to run parameter optimization)")
    
    # Process all proteins
    print(f"\n{'='*80}")
    print("PROCESSING PROTEINS")
    print(f"{'='*80}\n")
    
    results = process_protein_dataset(pdb_files, params)
    
    # Save confusion matrices
    print("\nSaving confusion matrices...")
    save_confusion_matrices(results, output_dir / "confusion_matrices")
    
    # Generate summary report
    print("Generating summary report...")
    report_name = "optimized_summary_report.txt" if args.optimize else "baseline_summary_report.txt"
    generate_summary_report(results, output_dir / report_name)
    
    # Save per-protein CSV files
    print("Saving per-protein detailed results...")
    for result in results:
        csv_file = output_dir / f"{result.protein_id}_detailed_results.csv"
        result.dataframe.to_csv(csv_file, index=False)
    
    # Cross-validation analysis
    if not args.optimize and args.cv_folds > 1:
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION ANALYSIS ({args.cv_folds}-fold)")
        print(f"{'='*80}\n")
        
        cv_results = cross_validate_parameters(
            pdb_files,
            params,
            n_folds=args.cv_folds,
            reference=args.reference
        )
        
        # Save CV results
        import json
        with open(output_dir / f"cv_results_{args.reference}.json", 'w') as f:
            json.dump(cv_results, f, indent=2)
    
    # Generate visualizations
    if not args.no_viz:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        generate_all_visualizations(output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {report_name}")
    print("  - confusion_matrices/ (per-protein matrices)")
    print("  - *_detailed_results.csv (per-protein data)")
    if args.optimize:
        print("  - best_parameters.txt")
        print("  - optuna_optimization_trials.csv")
    if not args.no_viz:
        print("  - visualizations/ (plots and graphs)")
    print()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Quick test of DUDE analysis system with current 4 proteins
"""

from pathlib import Path
from dude_complete_analysis import DUDEDatasetAnalyzer, DUDEAnalysisConfig
from comprehensive_burial_analysis import BurialParameters

print("\n" + "="*80)
print("TESTING DUDE ANALYSIS SYSTEM")
print("="*80 + "\n")

# Quick config for testing
config = DUDEAnalysisConfig(
    n_folds=3,  # 3-fold for faster testing
    n_optimization_trials=10,  # Only 10 trials for quick test
    optimization_reference='dssp'
)

# Initialize analyzer
workspace = Path.cwd()
analyzer = DUDEDatasetAnalyzer(workspace, config)

print(f"Found {len(analyzer.pdb_files)} proteins:")
for pdb in analyzer.pdb_files:
    print(f"  - {pdb.name}")

# Test with default parameters (no optimization for quick test)
print("\n" + "="*80)
print("RUNNING ANALYSIS WITH DEFAULT PARAMETERS")
print("="*80 + "\n")

params = BurialParameters()
result = analyzer.run_analysis_with_params(params)

# Display results
print("\n" + "="*80)
print("RESULTS")
print("="*80 + "\n")

stats = result['statistics']

if 'dssp' in stats:
    print("DSSP Comparison:")
    print(f"  Proteins analyzed: {stats['dssp']['n_proteins_with_data']}")
    print(f"  Mean Accuracy: {stats['dssp']['mean_accuracy']:.4f} ± {stats['dssp']['std_accuracy']:.4f}")


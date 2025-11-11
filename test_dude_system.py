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
    print(f"  Mean F1-Score: {stats['dssp']['mean_f1']:.4f}")
    print(f"  Range: [{stats['dssp']['min_accuracy']:.4f}, {stats['dssp']['max_accuracy']:.4f}]")

if 'stride' in stats:
    print("\nSTRIDE Comparison:")
    print(f"  Proteins analyzed: {stats['stride']['n_proteins_with_data']}")
    print(f"  Mean Accuracy: {stats['stride']['mean_accuracy']:.4f} ± {stats['stride']['std_accuracy']:.4f}")
    print(f"  Mean F1-Score: {stats['stride']['mean_f1']:.4f}")
    print(f"  Range: [{stats['stride']['min_accuracy']:.4f}, {stats['stride']['max_accuracy']:.4f}]")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80 + "\n")

analyzer.save_results(result, "test_run")

print("\n✓ Test complete!")
print(f"\nResults saved to: {analyzer.output_dir / 'test_run'}")
print("\nCheck these files:")
print(f"  - confusion_matrices/ (8 files: 2 per protein)")
print(f"  - summary_report.txt")
print(f"  - per_protein_accuracy.csv")
print()

# Show confusion matrices
print("="*80)
print("CONFUSION MATRICES GENERATED")
print("="*80 + "\n")

cm_dir = analyzer.output_dir / "test_run" / "confusion_matrices"
for cm_file in sorted(cm_dir.glob("*.csv")):
    print(f"  ✓ {cm_file.name}")

print(f"\nTotal: {len(list(cm_dir.glob('*.csv')))} confusion matrices")
print(f"Expected: {len(analyzer.pdb_files) * 2} (2 per protein)")


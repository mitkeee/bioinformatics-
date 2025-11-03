#!/usr/bin/env python3
"""
Interactive Analysis Script for Protein Burial Classification

This script provides easy-to-use functions for:
1. Testing different parameter combinations
2. Analyzing misclassifications
3. Visualizing specific amino acids
4. Finding optimal parameters

Usage:
    python interactive_analysis.py
"""

from pathlib import Path
import pandas as pd
from extract_ca import (
    run_pipeline,
    test_parameter_set,
    analyze_misclassifications,
    visualize_residue_by_name,
    optimize_parameters_against_reference,
    DEFAULT_PDB_PATH
)

def main():
    """
    Interactive analysis workflow for 3pte protein.
    """
    print("="*70)
    print("PROTEIN BURIAL CLASSIFICATION - INTERACTIVE ANALYSIS")
    print("="*70)

    # Load data (uses 3pte.pdb by default)
    print("\n1. Loading protein data and running baseline classification...")
    df = run_pipeline(do_dssp=True, do_stride=True, optimize_params=False, visualize=False)

    print("\n" + "="*70)
    print("BASELINE RESULTS (3PTE protein)")
    print("="*70)

    # Analyze misclassifications
    print("\n2. Analyzing misclassifications vs DSSP...")
    fp, fn = analyze_misclassifications(df, reference='dssp_label')

    print("\n3. Analyzing misclassifications vs STRIDE...")
    fp_stride, fn_stride = analyze_misclassifications(df, reference='stride_label')

    # Interactive parameter testing
    print("\n" + "="*70)
    print("INTERACTIVE PARAMETER TESTING")
    print("="*70)

    print("\nTesting alternative parameter sets to improve accuracy...")

    # Test 1: More lenient (reduce false positives)
    print("\n--- TEST 1: More lenient thresholds (reduce false positives) ---")
    acc1 = test_parameter_set(df, z_low=-0.8, z_high=0.3,
                              homog_low=0.30, homog_high=0.70,
                              reference='dssp_label')

    # Test 2: Stricter (reduce false negatives)
    print("\n--- TEST 2: Stricter thresholds ---")
    acc2 = test_parameter_set(df, z_low=-0.3, z_high=0.7,
                              homog_low=0.40, homog_high=0.60,
                              reference='dssp_label')

    # Test 3: Balanced
    print("\n--- TEST 3: Balanced approach ---")
    acc3 = test_parameter_set(df, z_low=-0.6, z_high=0.4,
                              homog_low=0.32, homog_high=0.68,
                              reference='dssp_label')

    # Find best manual test
    best_acc = max(acc1, acc2, acc3)
    baseline_acc = 0.608  # Current 3pte baseline

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline accuracy: {baseline_acc:.3f} (60.8%)")
    print(f"Best manual test:  {best_acc:.3f} ({100*best_acc:.1f}%)")

    if best_acc > baseline_acc:
        improvement = (best_acc - baseline_acc) * 100
        print(f"✅ IMPROVEMENT: +{improvement:.1f} percentage points!")
    else:
        print("No improvement with manual tests. Try optimization:")
        print("  Run with: OPTIMIZE_PARAMS = True")

    # Examples for visualization
    print("\n" + "="*70)
    print("VISUALIZATION EXAMPLES")
    print("="*70)
    print("\nTo visualize specific residues, use:")
    print("  from extract_ca import visualize_residue_by_name")
    print("  visualize_residue_by_name(df, 'A:50', sphere_radius=6.0)")
    print("\nTo visualize all interesting cases:")
    print("  Run with: VISUALIZE = True")

    # Show some interesting residues
    print("\n" + "="*70)
    print("INTERESTING RESIDUES TO INVESTIGATE")
    print("="*70)

    if len(fp) > 0:
        print(f"\nFalse Positives (we called exterior, DSSP says interior):")
        print(fp[['res_label', 'z_6A', 'z_10A', 'sph_var_6A', 'burial_label', 'dssp_label']].head(5).to_string(index=False))

    if len(fn) > 0:
        print(f"\nFalse Negatives (we called interior, DSSP says exterior):")
        print(fn[['res_label', 'z_6A', 'z_10A', 'sph_var_6A', 'burial_label', 'dssp_label']].head(5).to_string(index=False))

    print("\n" + "="*70)
    print("Next steps:")
    print("1. Visualize misclassified residues to understand why")
    print("2. Run full optimization: OPTIMIZE_PARAMS = True")
    print("3. Test on more proteins using batch_process_proteins()")
    print("4. Generate PyMOL visualization: @color_by_burial.pml")
    print("="*70)

    return df


if __name__ == "__main__":
    df = main()

    print("\n✅ Analysis complete! DataFrame saved in variable 'df'")
    print("You can now run additional commands like:")
    print("  - visualize_residue_by_name(df, 'A:100')")
    print("  - test_parameter_set(df, z_low=-0.7, z_high=0.6, ...)")


#!/usr/bin/env python3
"""
Parameter Tuning Script for Protein Burial Classification
Tests different threshold combinations to find optimal parameters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools

def load_results():
    """Load the combined results from previous run."""
    df = pd.read_csv("results/combined_results.csv")
    # Filter to only residues with DSSP reference data
    df = df[df['dssp_class'].notna()].copy()
    return df

def classify_with_params(df, nc6_threshold, nc10_threshold, uni6_threshold, uni10_threshold):
    """Re-classify residues with new parameters."""
    ncps_class = []

    for _, row in df.iterrows():
        nc6 = row['ncps_sphere_6']
        nc10 = row['ncps_sphere_10']
        uni6 = row['ncps_sphere_6_uni']
        uni10 = row['ncps_sphere_10_uni']

        # Default to interior
        is_exterior = False

        # Exterior if: few neighbors (below threshold)
        if nc6 < nc6_threshold or nc10 < nc10_threshold:
            is_exterior = True
        # Exterior if: low uniformity (neighbors not surrounding, one-sided)
        elif pd.notna(uni6) and uni6 < uni6_threshold:
            is_exterior = True
        elif pd.notna(uni10) and uni10 < uni10_threshold:
            is_exterior = True

        ncps_class.append(1 if is_exterior else 0)

    return np.array(ncps_class)

def test_parameter_combination(df, params):
    """Test a specific parameter combination."""
    nc6, nc10, uni6, uni10 = params

    predictions = classify_with_params(df, nc6, nc10, uni6, uni10)
    accuracy = accuracy_score(df['dssp_class'], predictions)

    return {
        'nc6_threshold': nc6,
        'nc10_threshold': nc10,
        'uni6_threshold': uni6,
        'uni10_threshold': uni10,
        'accuracy': accuracy,
        'predictions': predictions
    }

def grid_search():
    """Perform grid search over parameter space."""
    print("Loading data...")
    df = load_results()
    print(f"Loaded {len(df)} residues with DSSP reference data")

    # Define parameter ranges to test
    nc6_range = [6, 7, 8, 9, 10, 11, 12]
    nc10_range = [12, 14, 16, 18, 20, 22, 24]
    uni6_range = [0.30, 0.35, 0.40, 0.45, 0.50]
    uni10_range = [0.40, 0.45, 0.50, 0.55, 0.60]

    print(f"\nTesting {len(nc6_range) * len(nc10_range) * len(uni6_range) * len(uni10_range)} combinations...")

    results = []
    best_accuracy = 0
    best_params = None

    total_combinations = len(nc6_range) * len(nc10_range) * len(uni6_range) * len(uni10_range)
    count = 0

    for nc6, nc10, uni6, uni10 in itertools.product(nc6_range, nc10_range, uni6_range, uni10_range):
        count += 1
        if count % 100 == 0:
            print(f"  Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%)")

        result = test_parameter_combination(df, (nc6, nc10, uni6, uni10))
        results.append(result)

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_params = result

    # Convert to DataFrame
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in results])
    results_df = results_df.sort_values('accuracy', ascending=False)

    print(f"\n{'='*80}")
    print("PARAMETER TUNING RESULTS")
    print(f"{'='*80}")
    print(f"\nBest Accuracy: {best_accuracy:.1%}")
    print(f"\nOptimal Parameters:")
    print(f"  nc6_threshold (6Å sphere): {best_params['nc6_threshold']}")
    print(f"  nc10_threshold (10Å sphere): {best_params['nc10_threshold']}")
    print(f"  uni6_threshold (6Å uniformity): {best_params['uni6_threshold']:.2f}")
    print(f"  uni10_threshold (10Å uniformity): {best_params['uni10_threshold']:.2f}")

    # Show top 10 parameter combinations
    print(f"\n{'='*80}")
    print("TOP 10 PARAMETER COMBINATIONS")
    print(f"{'='*80}")
    print(results_df.head(10).to_string(index=False))

    # Detailed analysis with best parameters
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS WITH BEST PARAMETERS")
    print(f"{'='*80}")

    predictions = best_params['predictions']
    cm = confusion_matrix(df['dssp_class'], predictions)

    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Int(0)  Ext(1)")
    print(f"   Actual Int(0)    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"   Actual Ext(1)    {cm[1,0]:4d}   {cm[1,1]:4d}")

    print(f"\nClassification Report:")
    print(classification_report(df['dssp_class'], predictions,
                                target_names=['Interior', 'Exterior']))

    # Save results
    results_df.to_csv('results/parameter_tuning_results.csv', index=False)
    print(f"\n✅ Full results saved to: results/parameter_tuning_results.csv")

    # Save best parameters
    with open('results/best_parameters.txt', 'w') as f:
        f.write("BEST PARAMETERS FOR PROTEIN BURIAL CLASSIFICATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Accuracy: {best_accuracy:.3%}\n\n")
        f.write(f"nc6_threshold = {best_params['nc6_threshold']}\n")
        f.write(f"nc10_threshold = {best_params['nc10_threshold']}\n")
        f.write(f"uni6_threshold = {best_params['uni6_threshold']:.2f}\n")
        f.write(f"uni10_threshold = {best_params['uni10_threshold']:.2f}\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                  Predicted\n")
        f.write(f"                  Int(0)  Ext(1)\n")
        f.write(f"   Actual Int(0)    {cm[0,0]:4d}   {cm[0,1]:4d}\n")
        f.write(f"   Actual Ext(1)    {cm[1,0]:4d}   {cm[1,1]:4d}\n\n")
        f.write(classification_report(df['dssp_class'], predictions,
                                     target_names=['Interior', 'Exterior']))

    print(f"✅ Best parameters saved to: results/best_parameters.txt")

    return best_params, results_df

if __name__ == "__main__":
    print("="*80)
    print("PARAMETER TUNING FOR PROTEIN BURIAL CLASSIFICATION")
    print("="*80)

    best_params, results_df = grid_search()

    print("\n" + "="*80)
    print("✅ PARAMETER TUNING COMPLETE!")
    print("="*80)


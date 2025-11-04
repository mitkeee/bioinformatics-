#!/usr/bin/env python3
"""
Advanced Parameter Optimization with Multiple Strategies
Tests various approaches to improve accuracy beyond 78.2%
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import itertools
from pathlib import Path

def load_data():
    """Load and prepare data."""
    df = pd.read_csv("results/combined_results.csv")

    # Only residues with DSSP data
    df_dssp = df[df['dssp_class'].notna()].copy()

    # Residues with both DSSP and STRIDE
    df_both = df[(df['dssp_class'].notna()) & (df['stride_class'].notna())].copy()

    # Check DSSP-STRIDE agreement
    if len(df_both) > 0:
        agreement = (df_both['dssp_class'] == df_both['stride_class']).mean()
        print(f"DSSP-STRIDE Agreement: {agreement:.1%} (on {len(df_both)} residues)")

        # Use consensus where they agree
        df_consensus = df_both[df_both['dssp_class'] == df_both['stride_class']].copy()
        print(f"Consensus subset: {len(df_consensus)} residues (high confidence)")
    else:
        df_consensus = df_dssp

    return df_dssp, df_both, df_consensus

def classify_with_params(df, nc6_thresh, nc10_thresh, uni6_thresh, uni10_thresh):
    """Classify with given parameters."""
    predictions = []

    for _, row in df.iterrows():
        nc6 = row['ncps_sphere_6']
        nc10 = row['ncps_sphere_10']
        uni6 = row['ncps_sphere_6_uni']
        uni10 = row['ncps_sphere_10_uni']

        # Exterior if: few neighbors OR low uniformity
        is_exterior = False
        if nc6 < nc6_thresh or nc10 < nc10_thresh:
            is_exterior = True
        elif pd.notna(uni6) and uni6 < uni6_thresh:
            is_exterior = True
        elif pd.notna(uni10) and uni10 < uni10_thresh:
            is_exterior = True

        predictions.append(1 if is_exterior else 0)

    return np.array(predictions)

def weighted_consensus_classify(df, nc6_thresh, nc10_thresh, uni6_thresh, uni10_thresh,
                                dssp_weight=0.5, stride_weight=0.5):
    """Use weighted consensus between DSSP and STRIDE as ground truth."""
    predictions = classify_with_params(df, nc6_thresh, nc10_thresh, uni6_thresh, uni10_thresh)

    # Create weighted ground truth
    ground_truth = []
    for _, row in df.iterrows():
        dssp = row['dssp_class']
        stride = row['stride_class']

        if pd.isna(stride):
            ground_truth.append(dssp)
        else:
            # Weighted vote
            weighted = dssp * dssp_weight + stride * stride_weight
            ground_truth.append(1 if weighted >= 0.5 else 0)

    return accuracy_score(ground_truth, predictions)

def advanced_grid_search():
    """Advanced grid search with multiple optimization strategies."""

    print("="*80)
    print("ADVANCED PARAMETER OPTIMIZATION")
    print("="*80)
    print()

    df_dssp, df_both, df_consensus = load_data()

    # Expanded parameter ranges (finer granularity)
    nc6_range = [5, 6, 7, 8, 9]
    nc10_range = [10, 11, 12, 13, 14, 15, 16]
    uni6_range = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]
    uni10_range = [0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"Testing {len(nc6_range) * len(nc10_range) * len(uni6_range) * len(uni10_range)} combinations...")
    print()

    results = []
    best_dssp = {'acc': 0, 'params': None}
    best_consensus = {'acc': 0, 'params': None}
    best_f1 = {'score': 0, 'params': None}

    count = 0
    total = len(nc6_range) * len(nc10_range) * len(uni6_range) * len(uni10_range)

    for nc6, nc10, uni6, uni10 in itertools.product(nc6_range, nc10_range, uni6_range, uni10_range):
        count += 1
        if count % 200 == 0:
            print(f"  Progress: {count}/{total} ({count/total*100:.1f}%)")

        # Test on full DSSP dataset
        pred_dssp = classify_with_params(df_dssp, nc6, nc10, uni6, uni10)
        acc_dssp = accuracy_score(df_dssp['dssp_class'], pred_dssp)
        f1_dssp = f1_score(df_dssp['dssp_class'], pred_dssp, average='weighted')

        # Test on consensus subset
        if len(df_consensus) > 0:
            pred_consensus = classify_with_params(df_consensus, nc6, nc10, uni6, uni10)
            acc_consensus = accuracy_score(df_consensus['dssp_class'], pred_consensus)
        else:
            acc_consensus = acc_dssp

        # Store result
        results.append({
            'nc6': nc6, 'nc10': nc10, 'uni6': uni6, 'uni10': uni10,
            'acc_dssp': acc_dssp,
            'acc_consensus': acc_consensus,
            'f1_score': f1_dssp
        })

        # Track best
        if acc_dssp > best_dssp['acc']:
            best_dssp = {'acc': acc_dssp, 'params': (nc6, nc10, uni6, uni10), 'pred': pred_dssp}

        if acc_consensus > best_consensus['acc']:
            best_consensus = {'acc': acc_consensus, 'params': (nc6, nc10, uni6, uni10)}

        if f1_dssp > best_f1['score']:
            best_f1 = {'score': f1_dssp, 'params': (nc6, nc10, uni6, uni10)}

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('acc_dssp', ascending=False)

    print()
    print("="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print()

    # Strategy 1: Best DSSP accuracy
    print("STRATEGY 1: Optimize for DSSP Accuracy")
    print("-"*80)
    nc6, nc10, uni6, uni10 = best_dssp['params']
    print(f"Best DSSP Accuracy: {best_dssp['acc']:.3%}")
    print(f"Parameters: nc6={nc6}, nc10={nc10}, uni6={uni6:.2f}, uni10={uni10:.2f}")

    cm = confusion_matrix(df_dssp['dssp_class'], best_dssp['pred'])
    print(f"\nConfusion Matrix:")
    print(f"  True Int -> Pred Int: {cm[0,0]} | Pred Ext: {cm[0,1]}")
    print(f"  True Ext -> Pred Int: {cm[1,0]} | Pred Ext: {cm[1,1]}")

    # Calculate precision and recall
    precision_int = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
    recall_int = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    precision_ext = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
    recall_ext = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0

    print(f"\nPer-Class Performance:")
    print(f"  Interior: Precision={precision_int:.1%}, Recall={recall_int:.1%}")
    print(f"  Exterior: Precision={precision_ext:.1%}, Recall={recall_ext:.1%}")
    print()

    # Strategy 2: Best F1 score (balanced)
    print("STRATEGY 2: Optimize for Balanced F1-Score")
    print("-"*80)
    nc6, nc10, uni6, uni10 = best_f1['params']
    print(f"Best F1-Score: {best_f1['score']:.3%}")
    print(f"Parameters: nc6={nc6}, nc10={nc10}, uni6={uni6:.2f}, uni10={uni10:.2f}")
    print()

    # Strategy 3: Consensus optimization
    print("STRATEGY 3: Optimize for Consensus (DSSP+STRIDE agree)")
    print("-"*80)
    nc6, nc10, uni6, uni10 = best_consensus['params']
    print(f"Best Consensus Accuracy: {best_consensus['acc']:.3%}")
    print(f"Parameters: nc6={nc6}, nc10={nc10}, uni6={uni6:.2f}, uni10={uni10:.2f}")
    print()

    # Show top 10 combinations
    print("="*80)
    print("TOP 10 PARAMETER COMBINATIONS (by DSSP accuracy)")
    print("="*80)
    print(results_df.head(10).to_string(index=False))
    print()

    # Save results
    results_df.to_csv('results/advanced_parameter_tuning.csv', index=False)

    # Compare to current parameters (6, 12, 0.30, 0.60)
    current_params = results_df[(results_df['nc6'] == 6) &
                                (results_df['nc10'] == 12) &
                                (results_df['uni6'] == 0.30) &
                                (results_df['uni10'] == 0.60)]

    if len(current_params) > 0:
        current_acc = current_params.iloc[0]['acc_dssp']
        improvement = (best_dssp['acc'] - current_acc) * 100

        print("="*80)
        print("COMPARISON TO CURRENT PARAMETERS")
        print("="*80)
        print(f"Current (nc6=6, nc10=12, uni6=0.30, uni10=0.60): {current_acc:.3%}")
        print(f"Best found: {best_dssp['acc']:.3%}")
        print(f"Improvement: {improvement:+.2f} percentage points")

        if improvement > 0.1:
            print("\n✓ Better parameters found!")
        else:
            print("\n✓ Current parameters are already near-optimal!")

    print("="*80)

    return best_dssp, best_f1, best_consensus, results_df

if __name__ == "__main__":
    best_dssp, best_f1, best_consensus, results_df = advanced_grid_search()

    print("\n✅ Advanced optimization complete!")
    print("Results saved to: results/advanced_parameter_tuning.csv")


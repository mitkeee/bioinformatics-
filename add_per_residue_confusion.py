#!/usr/bin/env python3
"""
Add per-residue confusion matrix classification to CSV.
For each residue, determine if it's TN, FP, FN, or TP based on:
- True label (DSSP or STRIDE)
- Predicted label (NCPS - your method)
"""

import pandas as pd
from pathlib import Path

def calculate_per_residue_confusion(df):
    """
    Calculate per-residue confusion matrix classification.

    For DSSP vs NCPS:
    - TN: predicted=0 (interior) AND true=0 (interior) ✓ correct
    - FP: predicted=1 (exterior) AND true=0 (interior) ✗ wrong
    - FN: predicted=0 (interior) AND true=1 (exterior) ✗ wrong
    - TP: predicted=1 (exterior) AND true=1 (exterior) ✓ correct
    """

    # DSSP confusion per residue
    dssp_confusion = []
    dssp_correct = []

    for idx, row in df.iterrows():
        true_label = row['dssp_class']
        pred_label = row['ncps_class']

        # Skip if missing DSSP data
        if pd.isna(true_label) or pd.isna(pred_label):
            dssp_confusion.append('N/A')
            dssp_correct.append(None)
            continue

        true_label = int(true_label)
        pred_label = int(pred_label)

        # Determine confusion matrix category
        if pred_label == 0 and true_label == 0:
            dssp_confusion.append('TN')
            dssp_correct.append(True)
        elif pred_label == 1 and true_label == 0:
            dssp_confusion.append('FP')
            dssp_correct.append(False)
        elif pred_label == 0 and true_label == 1:
            dssp_confusion.append('FN')
            dssp_correct.append(False)
        elif pred_label == 1 and true_label == 1:
            dssp_confusion.append('TP')
            dssp_correct.append(True)
        else:
            dssp_confusion.append('N/A')
            dssp_correct.append(None)

    # STRIDE confusion per residue
    stride_confusion = []
    stride_correct = []

    for idx, row in df.iterrows():
        true_label = row['stride_class']
        pred_label = row['ncps_class']

        # Skip if missing STRIDE data
        if pd.isna(true_label) or pd.isna(pred_label):
            stride_confusion.append('N/A')
            stride_correct.append(None)
            continue

        true_label = int(true_label)
        pred_label = int(pred_label)

        # Determine confusion matrix category
        if pred_label == 0 and true_label == 0:
            stride_confusion.append('TN')
            stride_correct.append(True)
        elif pred_label == 1 and true_label == 0:
            stride_confusion.append('FP')
            stride_correct.append(False)
        elif pred_label == 0 and true_label == 1:
            stride_confusion.append('FN')
            stride_correct.append(False)
        elif pred_label == 1 and true_label == 1:
            stride_confusion.append('TP')
            stride_correct.append(True)
        else:
            stride_confusion.append('N/A')
            stride_correct.append(None)

    df['dssp_confusion_type'] = dssp_confusion
    df['dssp_correct'] = dssp_correct
    df['stride_confusion_type'] = stride_confusion
    df['stride_correct'] = stride_correct

    return df

def main():
    csv_dir = Path('/holder/results_dude/detailed_reports')

    print("\n" + "=" * 100)
    print("Adding Per-Residue Confusion Matrix Classification")
    print("=" * 100 + "\n")

    count = 0
    csv_files = sorted(csv_dir.glob('*_detailed_results.csv'))

    for csv_file in csv_files:
        protein_id = csv_file.stem.replace('_detailed_results', '')

        try:
            df = pd.read_csv(csv_file)

            # Remove old confusion matrix columns if they exist
            cols_to_drop = [col for col in df.columns
                           if col in ['dssp_TN', 'dssp_FP', 'dssp_FN', 'dssp_TP',
                                     'stride_TN', 'stride_FP', 'stride_FN', 'stride_TP',
                                     'dssp_confusion_matrix', 'stride_confusion_matrix']]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # Calculate per-residue confusion
            df = calculate_per_residue_confusion(df)

            # Save updated CSV
            df.to_csv(csv_file, index=False)

            # Count accuracy
            dssp_correct = df['dssp_correct'].sum()
            dssp_total = df['dssp_correct'].notna().sum()
            stride_correct = df['stride_correct'].sum()
            stride_total = df['stride_correct'].notna().sum()

            print(f"✓ {protein_id:10s} - DSSP: {dssp_correct}/{dssp_total} correct | STRIDE: {stride_correct}/{stride_total} correct")
            count += 1
        except Exception as e:
            print(f"✗ {protein_id:10s} - Error: {str(e)[:50]}")

    print(f"\n{'=' * 100}")
    print(f"Complete! {count}/{len(csv_files)} CSV files updated")
    print(f"{'=' * 100}\n")

if __name__ == "__main__":
    main()


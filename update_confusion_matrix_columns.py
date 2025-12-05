#!/usr/bin/env python3
"""
Extract confusion matrices from detailed_report.txt and add as separate columns to CSV.
Each confusion matrix value gets its own column (TN, FP, FN, TP).
"""

import os
import re
import pandas as pd
from pathlib import Path

def extract_confusion_matrices_detailed(report_path):
    """Extract DSSP and STRIDE confusion matrices with separate columns."""
    matrices = {
        'dssp_TN': None, 'dssp_FP': None, 'dssp_FN': None, 'dssp_TP': None,
        'stride_TN': None, 'stride_FP': None, 'stride_FN': None, 'stride_TP': None
    }

    try:
        with open(report_path, 'r') as f:
            content = f.read()

        # Extract DSSP confusion matrix
        dssp_match = re.search(
            r'ACCORDING TO DSSP:.*?Confusion Matrix:(.*?)Accuracy:',
            content,
            re.DOTALL
        )
        if dssp_match:
            dssp_section = dssp_match.group(1)
            tn = re.search(r'True Negatives \(TN\):\s*(\d+)', dssp_section)
            fp = re.search(r'False Positives \(FP\):\s*(\d+)', dssp_section)
            fn = re.search(r'False Negatives \(FN\):\s*(\d+)', dssp_section)
            tp = re.search(r'True Positives \(TP\):\s*(\d+)', dssp_section)

            if all([tn, fp, fn, tp]):
                matrices['dssp_TN'] = int(tn.group(1))
                matrices['dssp_FP'] = int(fp.group(1))
                matrices['dssp_FN'] = int(fn.group(1))
                matrices['dssp_TP'] = int(tp.group(1))

        # Extract STRIDE confusion matrix
        stride_match = re.search(
            r'ACCORDING TO STRIDE:.*?Confusion Matrix:(.*?)Accuracy:',
            content,
            re.DOTALL
        )
        if stride_match:
            stride_section = stride_match.group(1)
            tn = re.search(r'True Negatives \(TN\):\s*(\d+)', stride_section)
            fp = re.search(r'False Positives \(FP\):\s*(\d+)', stride_section)
            fn = re.search(r'False Negatives \(FN\):\s*(\d+)', stride_section)
            tp = re.search(r'True Positives \(TP\):\s*(\d+)', stride_section)

            if all([tn, fp, fn, tp]):
                matrices['stride_TN'] = int(tn.group(1))
                matrices['stride_FP'] = int(fp.group(1))
                matrices['stride_FN'] = int(fn.group(1))
                matrices['stride_TP'] = int(tp.group(1))

    except Exception as e:
        print(f"Error reading {report_path}: {e}")

    return matrices

def update_csv_with_detailed_matrices(csv_path, report_path):
    """Add confusion matrix columns (separate for each value) to CSV."""
    try:
        df = pd.read_csv(csv_path)
        matrices = extract_confusion_matrices_detailed(report_path)

        # Remove old concatenated columns if they exist
        cols_to_drop = [col for col in ['dssp_confusion_matrix', 'stride_confusion_matrix']
                       if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Add new individual columns
        for key, value in matrices.items():
            df[key] = value if value is not None else None

        df.to_csv(csv_path, index=False)
        return True, matrices
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return False, None

def main():
    reports_dir = Path('/holder/results_dude/detailed_reports')

    print("\n" + "=" * 100)
    print("Converting Confusion Matrices to Separate Columns")
    print("=" * 100 + "\n")

    count = 0
    failed = 0

    # Get all CSV files
    csv_files = sorted(reports_dir.glob('*_detailed_results.csv'))

    for csv_path in csv_files:
        pdb_id = csv_path.stem.replace('_detailed_results', '')
        report_path = csv_path.parent / f'{pdb_id}_detailed_report.txt'

        if report_path.exists():
            ok, matrices = update_csv_with_detailed_matrices(csv_path, report_path)
            if ok and matrices:
                print(f"✓ {pdb_id:10s} - DSSP: TN={matrices['dssp_TN']} FP={matrices['dssp_FP']} FN={matrices['dssp_FN']} TP={matrices['dssp_TP']} | STRIDE: TN={matrices['stride_TN']} FP={matrices['stride_FP']} FN={matrices['stride_FN']} TP={matrices['stride_TP']}")
                count += 1
            else:
                print(f"✗ {pdb_id:10s} - Failed to update")
                failed += 1
        else:
            print(f"✗ {pdb_id:10s} - Report file not found")
            failed += 1

    print(f"\n{'=' * 100}")
    print(f"Complete! {count} successful, {failed} failed")
    print(f"CSV files updated at: {reports_dir}")
    print(f"{'=' * 100}\n")

if __name__ == "__main__":
    main()


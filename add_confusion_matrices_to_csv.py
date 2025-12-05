#!/usr/bin/env python3
"""
Extract confusion matrices from detailed_report.txt files and add them to CSV files.
Adds confusion matrix data for both DSSP and STRIDE comparisons.
"""

import os
import re
import pandas as pd
from pathlib import Path

def extract_confusion_matrices(report_path):
    """Extract DSSP and STRIDE confusion matrices from detailed_report.txt"""
    matrices = {'dssp': None, 'stride': None}

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
                matrices['dssp'] = f"TN:{tn.group(1)} FP:{fp.group(1)} FN:{fn.group(1)} TP:{tp.group(1)}"

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
                matrices['stride'] = f"TN:{tn.group(1)} FP:{fp.group(1)} FN:{fn.group(1)} TP:{tp.group(1)}"

    except Exception as e:
        print(f"Error reading {report_path}: {e}")

    return matrices

def add_confusion_matrices_to_csv(csv_path, report_path):
    """Add confusion matrix columns to CSV"""
    try:
        df = pd.read_csv(csv_path)
        matrices = extract_confusion_matrices(report_path)

        # Add confusion matrix columns - same value for all rows in the protein
        df['dssp_confusion_matrix'] = matrices['dssp'] if matrices['dssp'] else 'N/A'
        df['stride_confusion_matrix'] = matrices['stride'] if matrices['stride'] else 'N/A'

        df.to_csv(csv_path, index=False)
        return True, matrices
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return False, None

def main():
    reports_dir = Path('/holder/results_dude/detailed_reports')

    print("\n" + "=" * 80)
    print("Adding Confusion Matrices to CSV Files")
    print("=" * 80 + "\n")

    count = 0
    failed = 0

    # Get all CSV files
    csv_files = sorted(reports_dir.glob('*_detailed_results.csv'))

    for csv_path in csv_files:
        pdb_id = csv_path.stem.replace('_detailed_results', '')
        report_path = csv_path.parent / f'{pdb_id}_detailed_report.txt'

        if report_path.exists():
            ok, matrices = add_confusion_matrices_to_csv(csv_path, report_path)
            if ok:
                if matrices['dssp'] and matrices['stride']:
                    print(f"✓ {pdb_id:10s} - DSSP: {matrices['dssp']}, STRIDE: {matrices['stride']}")
                    count += 1
                else:
                    print(f"⚠ {pdb_id:10s} - Missing matrices")
                    failed += 1
            else:
                print(f"✗ {pdb_id:10s} - Failed to update")
                failed += 1
        else:
            print(f"✗ {pdb_id:10s} - Report file not found")
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"Complete! {count} successful, {failed} failed")
    print(f"CSV files updated at: {reports_dir}")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Final verification script - confirms all 53 CSV files have confusion matrices
"""

from pathlib import Path
import pandas as pd

def verify_confusion_matrices():
    csv_dir = Path("/holder/results_dude/detailed_reports")

    print("\n" + "=" * 100)
    print("VERIFICATION: Confusion Matrices Added to All CSV Files")
    print("=" * 100 + "\n")

    count = 0
    csv_files = sorted(csv_dir.glob("*_detailed_results.csv"))

    for csv_file in csv_files:
        protein_id = csv_file.stem.replace("_detailed_results", "")

        try:
            df = pd.read_csv(csv_file)

            # Check if columns exist
            has_dssp = 'dssp_confusion_matrix' in df.columns
            has_stride = 'stride_confusion_matrix' in df.columns

            if has_dssp and has_stride:
                # Get the values (should be same for all rows)
                dssp_val = df['dssp_confusion_matrix'].iloc[0]
                stride_val = df['stride_confusion_matrix'].iloc[0]

                print(f"✓ {protein_id:10s} | DSSP: {dssp_val:40s} | STRIDE: {stride_val}")
                count += 1
            else:
                print(f"✗ {protein_id:10s} | Missing columns - DSSP:{has_dssp}, STRIDE:{has_stride}")

        except Exception as e:
            print(f"✗ {protein_id:10s} | Error: {str(e)[:40]}")

    print(f"\n{'=' * 100}")
    print(f"Complete! {count}/{len(csv_files)} CSV files have confusion matrices")
    print(f"{'=' * 100}\n")

    if count == len(csv_files):
        print("✅ SUCCESS! All confusion matrices added!\n")
        return True
    else:
        print("❌ Some files are missing matrices\n")
        return False

if __name__ == "__main__":
    verify_confusion_matrices()


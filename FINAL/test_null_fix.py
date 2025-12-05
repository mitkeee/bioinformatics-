#!/usr/bin/env python3
"""
Quick test to verify NULL VALUES FIX in final_analysis.py
Tests that all CSV output contains NO null values
"""

import sys
from pathlib import Path
import pandas as pd

def test_null_values():
    """Test that the script produces CSVs with no null values"""

    print("=" * 80)
    print("NULL VALUES FIX - VERIFICATION TEST")
    print("=" * 80)
    print()

    # Check if any CSV files exist in final_reports
    reports_dir = Path("/Users/famnit/Desktop/pythonProject/final_reports")

    if not reports_dir.exists():
        print("ℹ️  No final_reports folder yet - run the script first!")
        print()
        print("Quick test:")
        print("  1. mkdir pdbexamples")
        print("  2. cp your_protein.pdb pdbexamples/")
        print("  3. cd FINAL")
        print("  4. python final_analysis.py")
        print("  5. python test_null_fix.py")
        return False

    csv_files = sorted(reports_dir.glob("*_detailed_results.csv"))

    if not csv_files:
        print("ℹ️  No CSV files found in final_reports/")
        print("    Run: python final_analysis.py")
        return False

    print(f"✓ Found {len(csv_files)} CSV file(s)")
    print()

    all_clean = True

    for csv_file in csv_files:
        print(f"Checking: {csv_file.name}")
        print("-" * 80)

        try:
            df = pd.read_csv(csv_file)

            # Count null values per column
            null_counts = df.isnull().sum()
            has_nulls = null_counts.sum() > 0

            if has_nulls:
                print(f"  ✗ FOUND NULL VALUES:")
                for col, count in null_counts[null_counts > 0].items():
                    print(f"    - {col}: {count} null values")
                all_clean = False
            else:
                print(f"  ✓ NO NULL VALUES - Clean!")
                print(f"    Rows: {len(df)}")
                print(f"    Columns: {len(df.columns)}")

                # Show key columns
                key_cols = ['dssp_asa', 'dssp_class', 'stride_asa', 'stride_class', 'ncps_class']
                print(f"    Key columns:")
                for col in key_cols:
                    if col in df.columns:
                        print(f"      - {col}: dtype={df[col].dtype}, min={df[col].min()}, max={df[col].max()}")

        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            all_clean = False

        print()

    # Summary
    print("=" * 80)
    if all_clean:
        print("✅ SUCCESS! All CSV files have NO null values!")
        print("=" * 80)
        return True
    else:
        print("❌ ISSUE: Some files still have null values")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = test_null_values()
    sys.exit(0 if success else 1)


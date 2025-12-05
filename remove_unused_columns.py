#!/usr/bin/env python3
"""
Remove pdb_num and icode columns from all CSV files.
These columns are not needed for computations.
"""

from pathlib import Path
import pandas as pd

def remove_columns_from_csv(csv_path, columns_to_remove=['pdb_num', 'icode']):
    """Remove specified columns from CSV file."""
    try:
        df = pd.read_csv(csv_path)

        # Check which columns exist
        existing_cols = [col for col in columns_to_remove if col in df.columns]

        if existing_cols:
            df = df.drop(columns=existing_cols)
            df.to_csv(csv_path, index=False)
            return True, existing_cols
        else:
            return False, []
    except Exception as e:
        return False, str(e)

def main():
    csv_dir = Path('/holder/results_dude/detailed_reports')

    print("\n" + "=" * 80)
    print("Removing pdb_num and icode columns from all CSV files")
    print("=" * 80 + "\n")

    count = 0
    csv_files = sorted(csv_dir.glob('*_detailed_results.csv'))

    for csv_file in csv_files:
        protein_id = csv_file.stem.replace('_detailed_results', '')
        ok, cols_removed = remove_columns_from_csv(csv_file)

        if ok:
            cols_str = ', '.join(cols_removed)
            print(f"✓ {protein_id:10s} - Removed: {cols_str}")
            count += 1
        else:
            print(f"✗ {protein_id:10s} - Failed")

    print(f"\n{'=' * 80}")
    print(f"Complete! {count}/{len(csv_files)} CSV files updated")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()


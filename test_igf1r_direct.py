#!/usr/bin/env python3
"""
Simple direct regeneration of IGF1R detailed results with STRIDE data.
Tests the fixed extract_stride_data function.
"""

import sys
import os
sys.path.insert(0, '/Users/famnit/Desktop/pythonProject')
os.chdir('/Users/famnit/Desktop/pythonProject')

from pathlib import Path
import pandas as pd
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)

def test_igf1r():
    """Test IGF1R extraction."""
    pdb_path = Path("dude_extracted/dude_1_2/igf1r/receptor.pdb")

    with open('/tmp/igf1r_test_output.txt', 'w') as log:
        log.write("IGF1R STRIDE Extraction Test\n")
        log.write("=" * 60 + "\n\n")

        # Extract CA atoms
        log.write("Step 1: Extracting CA atoms...\n")
        df = extract_ca_atoms(str(pdb_path))
        log.write(f"  Result: {len(df)} atoms\n\n")

        # Set parameters
        params = BurialParameters(
            nc6_threshold=6.0,
            nc10_threshold=12.0,
            uni6_threshold=0.30,
            uni10_threshold=0.60,
            dssp_asa_cutoff=25.0,
            stride_asa_cutoff=20.0
        )

        # Extract STRIDE
        log.write("Step 2: Extracting STRIDE...\n")
        df = extract_stride_data(str(pdb_path), df, params.stride_asa_cutoff)

        stride_count = df['stride_asa'].notna().sum()
        stride_class_count = df['stride_class'].notna().sum()

        log.write(f"  stride_asa non-null: {stride_count}/{len(df)}\n")
        log.write(f"  stride_class non-null: {stride_class_count}/{len(df)}\n\n")

        if stride_count > 0:
            log.write("✓ SUCCESS - STRIDE data extracted!\n\n")
            log.write("First 10 rows:\n")
            for idx in range(min(10, len(df))):
                row = df.iloc[idx]
                asa_str = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else "---"
                ss_str = row['stride_ss'] if pd.notna(row['stride_ss']) else "-"
                class_str = str(int(row['stride_class'])) if pd.notna(row['stride_class']) else "-"
                log.write(f"  {row['resseq']:4d} {row['resname']:3s}: ASA={asa_str:7s} SS={ss_str:1s} Class={class_str}\n")

            # Save CSV
            output_file = Path("holder/results_dude/detailed_reports/igf1r_detailed_results.csv")
            df.to_csv(output_file, index=False)
            log.write(f"\n✓ Saved CSV: {output_file}\n")
        else:
            log.write("✗ FAILED - No STRIDE data\n")
            log.write(f"stride_asa sample: {df['stride_asa'].iloc[:3].tolist()}\n")

    print("Test complete - output written to /tmp/igf1r_test_output.txt")

if __name__ == "__main__":
    test_igf1r()


#!/usr/bin/env python3
"""
STRIDE Data Fix - Verification and Regeneration Script
Tests the improved extract_stride_data function and regenerates all DUDE reports
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/Users/famnit/Desktop/pythonProject')

import pandas as pd
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)

def test_single_protein(pdb_path, protein_id):
    """Test STRIDE extraction for a single protein."""
    print(f"\n{'='*70}")
    print(f"Testing: {protein_id}")
    print(f"{'='*70}")

    try:
        # Extract CA atoms
        df = extract_ca_atoms(pdb_path)  # Pass Path object, not string
        if df is None or len(df) == 0:
            print("✗ No CA atoms found")
            return False

        print(f"✓ Extracted {len(df)} CA atoms")

        # Set parameters
        params = BurialParameters(
            nc6_threshold=6.0,
            nc10_threshold=12.0,
            uni6_threshold=0.30,
            uni10_threshold=0.60,
            dssp_asa_cutoff=25.0,
            stride_asa_cutoff=20.0
        )

        # Extract STRIDE data
        print("Extracting STRIDE data...")
        df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)  # Pass Path object

        # Check results
        stride_asa_count = df['stride_asa'].notna().sum()
        stride_class_count = df['stride_class'].notna().sum()

        print(f"  stride_asa non-null: {stride_asa_count}/{len(df)}")
        print(f"  stride_class non-null: {stride_class_count}/{len(df)}")

        if stride_asa_count > 0:
            print(f"\n✓ SUCCESS - STRIDE data extracted!")

            # Show samples
            print(f"\nFirst 5 residues with STRIDE data:")
            for idx in range(min(5, stride_asa_count)):
                row = df[df['stride_asa'].notna()].iloc[idx]
                asa_str = f"{row['stride_asa']:.1f}"
                class_str = str(int(row['stride_class']))
                print(f"  {row['resseq']:4d} {row['resname']:3s}: ASA={asa_str:7s} Class={class_str}")

            return True
        else:
            print(f"\n✗ FAILED - No STRIDE data extracted")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("STRIDE DATA FIX - VERIFICATION & REGENERATION")
    print("="*70)

    # Test IGF1R
    workspace = Path('/Users/famnit/Desktop/pythonProject')
    igf1r_pdb = workspace / "dude_extracted/dude_1_2/igf1r/receptor.pdb"

    print(f"\nWorkspace: {workspace}")
    print(f"IGF1R PDB: {igf1r_pdb}")
    print(f"IGF1R PDB exists: {igf1r_pdb.exists()}")

    # Run test
    success = test_single_protein(igf1r_pdb, "IGF1R")

    if success:
        print(f"\n{'='*70}")
        print("VERIFICATION PASSED")
        print("="*70)
        print("\nThe STRIDE extraction fix is working correctly!")
        print("\nNow run: python3 generate_all_dude_reports.py")
        print("to regenerate all DUDE protein reports with STRIDE data.")
    else:
        print(f"\n{'='*70}")
        print("VERIFICATION FAILED")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()


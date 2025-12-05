#!/usr/bin/env python3
"""Test improved STRIDE extraction and regenerate reports."""

import pandas as pd
from pathlib import Path
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_dssp_data,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)

# Test with IGF1R
pdb_path = Path("dude_extracted/dude_1_2/igf1r/receptor.pdb")
print(f"Testing IGF1R with improved STRIDE extraction...")
print(f"PDB path: {pdb_path}\n")

# Extract CA atoms
df = extract_ca_atoms(str(pdb_path))
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

# Extract STRIDE with improved function
print("Extracting STRIDE data...")
df = extract_stride_data(str(pdb_path), df, params.stride_asa_cutoff)

# Check results
stride_has_data = df['stride_asa'].notna().sum()
stride_class_data = df['stride_class'].notna().sum()

print(f"\n✓ STRIDE Results:")
print(f"  stride_asa values: {stride_has_data}/{len(df)}")
print(f"  stride_class values: {stride_class_data}/{len(df)}")

if stride_has_data > 0:
    print(f"\nFirst 10 rows with STRIDE data:")
    for idx, row in df[['resseq', 'resname', 'stride_asa', 'stride_ss', 'stride_class']].head(10).iterrows():
        print(f"  {row['resseq']:4d} {row['resname']:3s}: ASA={row['stride_asa']:7.1f} SS={row['stride_ss']:1s} Class={row['stride_class']}")
    print("\n✓✓✓ SUCCESS! STRIDE data is now being extracted! ✓✓✓")
else:
    print("\n✗✗✗ FAILED - No STRIDE data found")


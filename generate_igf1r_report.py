#!/usr/bin/env python3
"""Generate IGF1R report with STRIDE data."""

from pathlib import Path
import pandas as pd
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_dssp_data,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)

# IGF1R paths
pdb_path = Path('dude_extracted/dude_1_2/igf1r/receptor.pdb')
output_dir = Path('holder/results_dude/detailed_reports')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating IGF1R report with STRIDE data...")
print(f"PDB: {pdb_path}")
print()

# Extract CA atoms
df = extract_ca_atoms(pdb_path)
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
df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)

stride_count = df['stride_asa'].notna().sum()
print(f"✓ STRIDE data: {stride_count}/{len(df)} residues")

# Extract DSSP data
print("Extracting DSSP data...")
df = extract_dssp_data(pdb_path, df, params.dssp_asa_cutoff)

dssp_count = df['dssp_asa'].notna().sum()
print(f"✓ DSSP data: {dssp_count}/{len(df)} residues")

# Add neighbor features
coords = df[['x', 'y', 'z']].values
df = add_neighbor_features(df, coords)

# Classify
df['ncps_class'] = classify_burial(df, params)

# Save CSV
csv_path = output_dir / 'igf1r_detailed_results.csv'
df.to_csv(csv_path, index=False)
print(f"✓ Saved CSV: {csv_path}")

# Show sample
print(f"\nFirst 5 rows with STRIDE and DSSP data:")
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else "---"
    stride_ss = row['stride_ss'] if pd.notna(row['stride_ss']) else "-"
    dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) else "---"
    dssp_ss = row['dssp_ss'] if pd.notna(row['dssp_ss']) else "-"
    print(f"  {row['resseq']:4d} {row['resname']:3s}: STRIDE ASA={stride_asa:7s} SS={stride_ss:1s} | DSSP ASA={dssp_asa:7s} SS={dssp_ss:1s}")

print(f"\n✓ Complete! IGF1R report generated with STRIDE and DSSP data.")


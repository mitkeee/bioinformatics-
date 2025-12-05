#!/usr/bin/env python3
"""Simple test of the STRIDE fix."""

from pathlib import Path
from comprehensive_burial_analysis import extract_ca_atoms, extract_stride_data, BurialParameters

pdb_path = Path('/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.pdb')

print("Testing STRIDE extraction...")
print(f"PDB path: {pdb_path}")
print(f"PDB exists: {pdb_path.exists()}\n")

# Test 1: Extract CA atoms
print("Step 1: Extracting CA atoms...")
try:
    df = extract_ca_atoms(pdb_path)
    print(f"✓ Success: {len(df)} atoms\n")
except Exception as e:
    print(f"✗ Failed: {e}\n")
    exit(1)

# Test 2: Extract STRIDE
print("Step 2: Extracting STRIDE data...")
try:
    params = BurialParameters(stride_asa_cutoff=20.0)
    df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)

    stride_count = df['stride_asa'].notna().sum()
    print(f"✓ Success: {stride_count}/{len(df)} residues with STRIDE data\n")

    if stride_count > 0:
        print("✓✓✓ FIX IS WORKING! ✓✓✓")
        print("\nFirst 5 residues with STRIDE data:")
        for idx in range(min(5, stride_count)):
            row = df[df['stride_asa'].notna()].iloc[idx]
            print(f"  {row['resseq']:4d} {row['resname']:3s}: ASA={row['stride_asa']:7.1f}")
    else:
        print("✗ No STRIDE data extracted")
        exit(1)
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


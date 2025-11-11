#!/usr/bin/env python3
"""Test if STRIDE parsing is now working"""

from pathlib import Path
from comprehensive_burial_analysis import extract_ca_atoms, extract_stride_data

pdb_path = Path('3PTE.pdb')
df = extract_ca_atoms(pdb_path)
print(f"Extracted {len(df)} CA atoms")

df = extract_stride_data(pdb_path, df, 24.0)
stride_available = df['stride_class'].notna().sum()
print(f"STRIDE data available: {stride_available}/{len(df)}")

if stride_available > 0:
    print("✓ SUCCESS: STRIDE parsing is working!")
    print(f"  Interior residues: {(df['stride_class'] == 0).sum()}")
    print(f"  Exterior residues: {(df['stride_class'] == 1).sum()}")
else:
    print("✗ FAILED: STRIDE parsing still not working")


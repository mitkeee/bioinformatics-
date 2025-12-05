#!/usr/bin/env python3
"""Direct test of STRIDE extraction fix."""

import sys
sys.path.insert(0, '/Users/famnit/Desktop/pythonProject')

from pathlib import Path
from comprehensive_burial_analysis import extract_ca_atoms, extract_stride_data, BurialParameters

pdb_path = Path("/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.pdb")

print("Step 1: Extract CA atoms")
df = extract_ca_atoms(str(pdb_path))
print(f"  Result: {len(df)} atoms")

print("\nStep 2: Extract STRIDE")
df = extract_stride_data(str(pdb_path), df, 20.0)

print(f"  stride_asa non-null: {df['stride_asa'].notna().sum()}/{len(df)}")
print(f"  stride_class non-null: {df['stride_class'].notna().sum()}/{len(df)}")

if df['stride_asa'].notna().sum() > 0:
    print("\n✓ SUCCESS - STRIDE data extracted!")
    print(f"\nFirst 5 rows:")
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        print(f"  {row['resseq']:4d} {row['resname']:3s}: ASA={row['stride_asa']:7.1f} SS={row['stride_ss']:1s}")
else:
    print("\n✗ FAILED - No STRIDE data")
    print(f"stride_asa sample: {df['stride_asa'].iloc[:5].tolist()}")


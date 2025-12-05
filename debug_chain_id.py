#!/usr/bin/env python3
"""Debug STRIDE extraction - check chain ID matching."""

from pathlib import Path
from comprehensive_burial_analysis import extract_ca_atoms

pdb_path = Path('/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.pdb')

# Extract CA atoms
df = extract_ca_atoms(pdb_path)

print("DataFrame chain_id values:")
print(df['chain_id'].unique())
print(f"\nFirst 10 chain_id values:")
print(df['chain_id'].head(10).tolist())

print(f"\nFirst 10 resseq values:")
print(df['resseq'].head(10).tolist())

print(f"\nDataFrame columns: {df.columns.tolist()}")
print(f"\nDataFrame shape: {df.shape}")
print(f"\nFirst row:")
print(df.iloc[0])


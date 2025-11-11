#!/usr/bin/env python3
from pathlib import Path
from comprehensive_burial_analysis import extract_ca_atoms

pdb_path = Path('3PTE.pdb')
df = extract_ca_atoms(pdb_path)

print(f"Total CA atoms: {len(df)}")
print(f"\nFirst 5 residues:")
print(df[['chain_id', 'resseq', 'resname']].head())
print(f"\nUnique chain IDs: {df['chain_id'].unique()}")

# Now check STRIDE file
stride_file = Path('3pte.stride')
print(f"\nFirst 5 STRIDE ASG lines:")
with open(stride_file) as f:
    count = 0
    for line in f:
        if line.startswith('ASG') and count < 5:
            chain = line[9:10]
            resseq = line[11:15].strip()
            resname = line[5:8].strip()
            asa = line.split()[-1]
            print(f"  Chain: '{chain}' (len={len(chain)}), ResSeq: {resseq}, ResName: {resname}, ASA: {asa}")
            count += 1


#!/usr/bin/env python3
"""Debug STRIDE parsing issue"""

from pathlib import Path
from Bio.PDB import PDBParser
import pandas as pd

# Test with 3PTE
pdb_path = Path('3PTE.pdb')
stride_file = Path('3pte.stride')

print("="*80)
print("DEBUGGING STRIDE PARSING")
print("="*80)

# 1. Check what chains are in the PDB
print("\n1. Chains in PDB file:")
parser = PDBParser(QUIET=True)
structure = parser.get_structure('protein', str(pdb_path))

ca_residues = []
for model in structure:
    for chain in model:
        residues = [res for res in chain if res.id[0] == ' ' and 'CA' in res]
        print(f"   Chain '{chain.id}': {len(residues)} CA atoms")
        for res in residues[:3]:
            ca_residues.append({
                'chain': chain.id,
                'resseq': res.id[1],
                'resname': res.resname
            })
            print(f"      {res.resname} {res.id[1]}")

# 2. Check what's in STRIDE file
print("\n2. STRIDE file ASG records:")
stride_data = []
with open(stride_file, 'r') as f:
    for line in f:
        if line.startswith('ASG'):
            # Parse STRIDE format
            chain_id = line[9:10].strip()
            resseq_str = line[11:15].strip()
            try:
                resseq = int(resseq_str)
                parts = line.split()
                asa = float(parts[-1])
                stride_data.append({
                    'chain': chain_id,
                    'resseq': resseq,
                    'asa': asa
                })
                if len(stride_data) <= 3:
                    print(f"      Chain '{chain_id}', ResSeq {resseq}, ASA {asa}")
            except (ValueError, IndexError) as e:
                print(f"   Error parsing line: {e}")
                print(f"      Line: {line[:60]}")

print(f"\n   Total STRIDE records: {len(stride_data)}")

# 3. Try to match them
print("\n3. Matching PDB CA residues with STRIDE data:")
matches = 0
for ca in ca_residues[:10]:
    matching = [s for s in stride_data if s['chain'] == ca['chain'] and s['resseq'] == ca['resseq']]
    if matching:
        matches += 1
        print(f"   ✓ Match: Chain '{ca['chain']}', ResSeq {ca['resseq']}, ASA {matching[0]['asa']}")
    else:
        # Try empty chain match
        matching_empty = [s for s in stride_data if s['chain'] == '' and s['resseq'] == ca['resseq']]
        if matching_empty:
            matches += 1
            print(f"   ✓ Match (empty chain): Chain '{ca['chain']}' -> '', ResSeq {ca['resseq']}, ASA {matching_empty[0]['asa']}")
        else:
            print(f"   ✗ No match: Chain '{ca['chain']}', ResSeq {ca['resseq']}")

print(f"\n   Total matches: {matches}/{len(ca_residues)}")

# 4. Test the actual extraction function
print("\n4. Testing extract_stride_data function:")
from comprehensive_burial_analysis import extract_ca_atoms, extract_stride_data, BurialParameters

params = BurialParameters()
df = extract_ca_atoms(pdb_path)
print(f"   Extracted {len(df)} CA atoms")
print(f"   First 3 rows:")
for idx, row in df.head(3).iterrows():
    print(f"      {idx}: chain='{row['chain_id']}', resseq={row['resseq']}, resname={row['resname']}")

df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)
stride_available = df['stride_class'].notna().sum()
print(f"\n   STRIDE data available for {stride_available}/{len(df)} residues")

if stride_available == 0:
    print("\n   ⚠️ PROBLEM: No STRIDE data was matched!")
    print("   Checking for chain ID mismatch...")

    # Check what chain IDs are in the dataframe
    print(f"\n   Chain IDs in dataframe: {df['chain_id'].unique()}")

    # Check first few STRIDE records again
    print(f"\n   First few STRIDE chain IDs:")
    with open(stride_file, 'r') as f:
        count = 0
        for line in f:
            if line.startswith('ASG') and count < 5:
                chain_id = line[9:10]
                print(f"      '{chain_id}' (repr: {repr(chain_id)})")
                count += 1

print("\n" + "="*80)


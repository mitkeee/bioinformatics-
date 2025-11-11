#!/usr/bin/env python3
"""Diagnose why STRIDE parsing is failing"""

from pathlib import Path
from Bio.PDB import PDBParser

pdb_path = Path('3PTE.pdb')
stride_file = Path('3pte.stride')

print("="*80)
print("STRIDE PARSING DIAGNOSTIC")
print("="*80)

# 1. Check PDB chain IDs
print("\n1. Checking PDB file chain IDs:")
parser = PDBParser(QUIET=True)
structure = parser.get_structure('protein', str(pdb_path))

pdb_residues = []
for model in structure:
    for chain in model:
        residues = [res for res in chain if res.id[0] == ' ' and 'CA' in res]
        print(f"   Chain: '{chain.id}' (repr: {repr(chain.id)}, len={len(chain.id)})")
        for res in residues[:5]:
            pdb_residues.append({
                'chain': chain.id,
                'resseq': res.id[1],
                'resname': res.resname
            })
            print(f"      ResSeq {res.id[1]:4d}, ResName {res.resname}")

# 2. Check STRIDE file
print("\n2. Checking STRIDE file chain IDs:")
stride_residues = []
with open(stride_file) as f:
    for line in f:
        if line.startswith('ASG'):
            chain_raw = line[9:10]
            chain_stripped = chain_raw.strip()
            resseq_str = line[11:15].strip()
            resname = line[5:8].strip()

            try:
                resseq = int(resseq_str)
                parts = line.split()
                asa = float(parts[-1])

                stride_residues.append({
                    'chain_raw': chain_raw,
                    'chain_stripped': chain_stripped,
                    'resseq': resseq,
                    'resname': resname,
                    'asa': asa
                })

                if len(stride_residues) <= 5:
                    print(f"   Chain raw: '{chain_raw}' (repr: {repr(chain_raw)}, len={len(chain_raw)})")
                    print(f"   Chain stripped: '{chain_stripped}' (repr: {repr(chain_stripped)})")
                    print(f"      ResSeq {resseq:4d}, ResName {resname}, ASA {asa}")
            except:
                pass

print(f"\n   Total STRIDE records parsed: {len(stride_residues)}")

# 3. Try to match them
print("\n3. Attempting to match PDB and STRIDE residues:")
matches = 0
for i, pdb_res in enumerate(pdb_residues[:10]):
    pdb_chain = pdb_res['chain']
    pdb_resseq = pdb_res['resseq']

    # Try different matching strategies
    matching = None
    match_method = None

    # Strategy 1: Exact match
    for stride_res in stride_residues:
        if stride_res['chain_stripped'] == pdb_chain and stride_res['resseq'] == pdb_resseq:
            matching = stride_res
            match_method = "exact"
            break

    # Strategy 2: Match with raw chain (might be space)
    if not matching:
        for stride_res in stride_residues:
            if stride_res['chain_raw'] == pdb_chain and stride_res['resseq'] == pdb_resseq:
                matching = stride_res
                match_method = "raw"
                break

    # Strategy 3: Match by resseq only (if chains are different)
    if not matching:
        for stride_res in stride_residues:
            if stride_res['resseq'] == pdb_resseq:
                matching = stride_res
                match_method = "resseq_only"
                break

    if matching:
        matches += 1
        print(f"   ✓ Match #{i+1} ({match_method}): PDB chain '{pdb_chain}' ResSeq {pdb_resseq} -> "
              f"STRIDE chain '{matching['chain_stripped']}' (raw: '{matching['chain_raw']}'), ASA {matching['asa']}")
    else:
        print(f"   ✗ No match #{i+1}: PDB chain '{pdb_chain}' (repr: {repr(pdb_chain)}), ResSeq {pdb_resseq}")

print(f"\n   Total matches: {matches}/{len(pdb_residues)}")

# 4. Show what keys are being created in stride_map
print("\n4. Checking key format in stride_map:")
print("   First 3 keys that would be created:")
for i, stride_res in enumerate(stride_residues[:3]):
    key = (stride_res['chain_stripped'], stride_res['resseq'], '')
    print(f"      Key {i+1}: {repr(key)}")

print("\n5. Checking key format PDB would look for:")
print("   First 3 keys PDB would search for:")
for i, pdb_res in enumerate(pdb_residues[:3]):
    key = (pdb_res['chain'], int(pdb_res['resseq']), '')
    print(f"      Key {i+1}: {repr(key)}")

print("\n" + "="*80)
print("CONCLUSION:")
if matches == len(pdb_residues):
    print("✓ Chain IDs match! Issue is elsewhere in the code.")
else:
    print(f"✗ Chain ID mismatch detected! Only {matches}/{len(pdb_residues)} matched.")
    if matches > 0:
        print(f"   Use match method: {match_method}")
    else:
        print("   Need to investigate chain ID encoding further.")
print("="*80)


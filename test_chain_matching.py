#!/usr/bin/env python3
"""Minimal test to check chain ID matching"""

# Simulate what the code does
stride_line = "ASG  GLU A  182  182    H    AlphaHelix    -63.07    -42.39       6.2      3PTE"

# Current parsing
chain_id = stride_line[9:10].strip()
chain_id_raw = stride_line[9:10]
resseq = int(stride_line[11:15].strip())

print("STRIDE parsing:")
print(f"  chain_id (stripped): '{chain_id}' (repr: {repr(chain_id)})")
print(f"  chain_id_raw: '{chain_id_raw}' (repr: {repr(chain_id_raw)})")
print(f"  resseq: {resseq}")

# What keys get created?
print("\nKeys created in stride_map:")
print(f"  Key 1: {repr((chain_id, resseq, ''))}")
print(f"  Key 2: {repr((chain_id_raw, resseq, ''))}")

# What would PDB look for?
# Assuming PDB chain is 'A'
pdb_chain = 'A'
pdb_resseq = 182

print("\nKey PDB would search for:")
print(f"  Key: {repr((pdb_chain, pdb_resseq, ''))}")

# Do they match?
print("\nMatching:")
print(f"  Exact match: {(chain_id, resseq, '') == (pdb_chain, pdb_resseq, '')}")
print(f"  Raw match: {(chain_id_raw, resseq, '') == (pdb_chain, pdb_resseq, '')}")

# Test the lookup logic that was added
possible_keys = [
    (pdb_chain, pdb_resseq, ''),           # Original chain ID
    ('', pdb_resseq, ''),                  # Empty chain ID
    ('A', pdb_resseq, ''),                 # Default chain A
    (' ', pdb_resseq, ''),                 # Space as chain ID
]

stride_map = {
    (chain_id, resseq, ''): {'asa': 6.2},
    (chain_id_raw, resseq, ''): {'asa': 6.2},
}

if chain_id == '':
    stride_map[('A', resseq, '')] = {'asa': 6.2}
    stride_map[(' ', resseq, '')] = {'asa': 6.2}

print("\nstride_map keys:")
for k in stride_map.keys():
    print(f"  {repr(k)}")

print("\nTrying to find match:")
found = False
for key in possible_keys:
    if key in stride_map:
        print(f"  ✓ Found with key: {repr(key)}")
        found = True
        break
    else:
        print(f"  ✗ Not found with key: {repr(key)}")

if found:
    print("\n✓ SUCCESS: Match found!")
else:
    print("\n✗ FAILURE: No match found!")


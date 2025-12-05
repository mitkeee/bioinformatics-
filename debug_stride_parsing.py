#!/usr/bin/env python3
                break
            if count >= 10:
            count += 1
            print(f"    resseq[11:15]: '{line[11:15]}' -> {int(line[11:15].strip())}")
            print(f"    chain_id[9:10].strip(): '{line[9:10].strip()}' (repr: {repr(line[9:10].strip())})")
            print(f"    chain_id[9:10]: '{line[9:10]}' (repr: {repr(line[9:10])})")
            print(f"  Line {count+1}: {line.rstrip()}")
        if line.startswith('ASG'):
    for line in f:
    count = 0
with open(stride_file, 'r') as f:
print(f"\nFirst 10 ASG lines:")

print(f"  {chain_ids}")
chain_ids = set(key[0] for key in stride_map.keys())
print(f"\nUnique chain_ids in stride_map:")

    print(f"  {key}: {stride_map[key]}")
for i, key in enumerate(list(stride_map.keys())[:10]):
print(f"\nFirst 10 keys in stride_map:")
print(f"Total stride_map keys: {len(stride_map)}")

                continue
                print(f"Error parsing: {e}")
            except (ValueError, IndexError) as e:

                    stride_map[(' ', resseq, '')] = data
                    stride_map[('A', resseq, '')] = data
                if chain_id == '':
                stride_map[(chain_id_raw, resseq, '')] = data
                stride_map[(chain_id, resseq, '')] = data

                data = {'asa': asa, 'ss': ss if ss else 'C'}

                asa = float(parts[-2]) if len(parts) >= 10 else 0.0
                parts = line.split()
                ss = line[24:25].strip() if len(line) > 24 else 'C'
                resseq = int(line[11:15].strip())
                chain_id_raw = line[9:10]
                chain_id = line[9:10].strip()
            try:
        if line.startswith('ASG'):
    for line in f:
with open(stride_file, 'r') as f:
stride_map = {}

print("Parsing STRIDE file for keys...\n")

stride_file = Path('/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.stride')

from pathlib import Path

"""Debug STRIDE parsing - check what keys are created."""


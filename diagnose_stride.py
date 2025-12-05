#!/usr/bin/env python3
"""Minimal diagnostic to test STRIDE parsing."""

from pathlib import Path

stride_path = Path("holder/dude_1_2/abl1/receptor.stride")

print("Testing ASG line extraction:\n")

with open(stride_path, 'r') as f:
    count = 0
    for line in f:
        if line.startswith('ASG'):
            count += 1
            if count <= 5:
                print(f"Line {count}:")
                print(f"  Raw: {repr(line)}")
                parts = line.split()
                print(f"  Parts ({len(parts)}): {parts}")

                # Find numeric ASA
                asa_candidates = []
                for i, part in enumerate(parts):
                    try:
                        val = float(part)
                        asa_candidates.append((i, part, val))
                    except ValueError:
                        pass

                print(f"  Numeric parts: {asa_candidates}")
                print()

print(f"Total ASG records: {count}")


#!/usr/bin/env python3
"""Analyze exact column positions in STRIDE file."""

stride_file = "/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.stride"

with open(stride_file, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith('ASG'):
            print(f"Line {i}: {line.rstrip()}")
            print(f"Positions:")
            print(f"  0-3: '{line[0:4]}'")
            print(f"  4-7: '{line[4:8]}'")
            print(f"  8-10: '{line[8:11]}'")
            print(f"  9-10: '{line[9:10]}'")
            print(f"  10-14: '{line[10:14]}'")
            print(f"  11-15: '{line[11:15]}'")
            print(f"  15-19: '{line[15:19]}'")
            print(f"  19-23: '{line[19:23]}'")
            print(f"  23-32: '{line[23:32]}'")
            print(f"  32-45: '{line[32:45]}'")

            # Try parsing
            print(f"\nCurrent parsing:")
            print(f"  chain_id [9:10]: '{line[9:10]}' -> '.strip() = '{line[9:10].strip()}'")
            print(f"  resseq [11:15]: '{line[11:15]}' -> int = {int(line[11:15].strip())}")
            print(f"  ss [24:25]: '{line[24:25]}' -> '.strip() = '{line[24:25].strip()}'")

            parts = line.split()
            print(f"\nSplit parts: {parts}")
            print(f"  parts[-2]: {parts[-2]} (ASA)")
            print(f"  parts[-1]: {parts[-1]} (ID marker)")

            if i >= 5:
                break


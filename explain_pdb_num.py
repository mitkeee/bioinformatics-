#!/usr/bin/env python3
"""
EXPLANATION: What is pdb_num in the CSV file?

pdb_num is the RESIDUE SEQUENCE NUMBER from the PDB file.
It represents the original numbering of residues as they appear in the crystallographic structure.
"""

import pandas as pd

# DETAILED EXPLANATION
explanation = """
================================================================================
WHAT IS pdb_num IN THE CSV?
================================================================================

pdb_num = PDB Residue Sequence Number (residue number from the PDB structure)

DEFINITION:
-----------
The original residue numbering as stored in the PDB file when the protein
structure was experimentally determined (usually by X-ray crystallography or 
cryo-EM).

IMPORTANT DISTINCTION:
---------------------
There are TWO ways to refer to residues:

1. ARRAY INDEX (row number in table)
   - Sequential numbering: 1, 2, 3, 4, 5, ...
   - What you see as "row number" when you open in Excel
   - Used internally by Python (df.index)

2. PDB SEQUENCE NUMBER (pdb_num)
   - Original numbering from the PDB file
   - Can have gaps, negatives, or insertion codes (A, B, C)
   - Example: 954, 955, 956, 957, ... (IGF1R starts at 954)
   - This is what experimentalists use

WHY BOTH?
---------
Proteins can have unusual numbering because:
- Domains may be numbered separately in the PDB file
- Some residues may be missing (not in crystal structure)
- Scientists may use domain-based numbering
- Some structures have N-terminal His-tags with negative numbers (-1, 0)

EXAMPLES:
---------

IGF1R:
  Row 1:  pdb_num=954  (residue 954 in PDB file)
  Row 2:  pdb_num=955  (residue 955 in PDB file)
  Row 3:  pdb_num=956  (residue 956 in PDB file)
  ...
  Row 256: pdb_num=1209 (residue 1209 in PDB file)

ACE (different protein):
  Row 1:  pdb_num=2    (starts at residue 2)
  Row 2:  pdb_num=3
  Row 3:  pdb_num=4
  ...
  (notice it starts at 2, not 1 - first residue missing from structure)

WHEN TO USE:
-----------
- Use pdb_num when referring to your results in scientific papers
- Use row index when working with code/Python (array indexing)
- pdb_num is what appears in the detailed_report.txt

RELATIONSHIP WITH resseq:
-------------------------
In this project, pdb_num and resseq are IDENTICAL.
They both store the same PDB residue number.

pdb_num:  Added for convenience (same as resseq)
resseq:   Original residue sequence from PDB extraction
"""

print(explanation)

# SHOW IN ACTUAL CSV DATA
print("\n" + "=" * 80)
print("PRACTICAL EXAMPLE - IGF1R CSV")
print("=" * 80 + "\n")

df = pd.read_csv('/holder/results_dude/detailed_reports/igf1r_detailed_results.csv')

print("Row | pdb_num | resname | stride_asa | stride_class")
print("-" * 60)
for i in range(10):
    row = df.iloc[i]
    print(f"{i+1:3d} | {int(row['pdb_num']):7d} | {row['resname']:7s} | {row['stride_asa']:10.1f} | {int(row['stride_class']):12d}")

print("\n" + "=" * 80)
print("KEY POINT:")
print("=" * 80)
print(f"""
The 'Row' column (1, 2, 3, ...) is just the array index
The 'pdb_num' column (954, 955, 956, ...) is the ACTUAL residue number from PDB

This is why Row 1 has pdb_num=954, not pdb_num=1
""")


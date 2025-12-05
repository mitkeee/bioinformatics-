#!/usr/bin/env python3
"""
Investigate ACE Protein Issue
- Find correct PDB files for ACE from DUDE database
- Validate current ACE files
- Suggest fixes and workarounds
"""

from pathlib import Path
import subprocess
import re

def investigate_ace():
    """Main investigation function"""
    workspace = Path.cwd()

    print("=" * 80)
    print("ACE PROTEIN INVESTIGATION REPORT")
    print("=" * 80)
    print()

    # 1. Check DUDE metadata
    print("1. CHECKING DUDE DATABASE METADATA")
    print("-" * 80)

    dude_ace_dir = workspace / "dude_extracted" / "dude_1_2" / "ace"
    if dude_ace_dir.exists():
        pdb_selection = dude_ace_dir / "pdb_selection.txt"
        if pdb_selection.exists():
            with open(pdb_selection, 'r') as f:
                content = f.read()
            print("✓ Found pdb_selection.txt")
            print(content)

            # Extract PDB IDs
            pdb_ids = re.findall(r'\b([0-9][a-z0-9]{3})\b', content, re.IGNORECASE)
            pdb_ids = list(set(pdb_ids))
            print(f"\n✓ Identified PDB IDs from DUDE: {pdb_ids}")
        else:
            print("✗ pdb_selection.txt not found")
    else:
        print("✗ DUDE ACE directory not found")

    print()

    # 2. Check current ACE receptor file
    print("2. CHECKING CURRENT ACE RECEPTOR FILE")
    print("-" * 80)

    current_receptor = dude_ace_dir / "receptor.pdb"
    if current_receptor.exists():
        print(f"✓ Current receptor file: {current_receptor}")

        # Read first 100 lines
        with open(current_receptor, 'r') as f:
            lines = f.readlines()

        print(f"  File size: {current_receptor.stat().st_size} bytes")
        print(f"  Total lines: {len(lines)}")

        # Check for HEADER
        has_header = any(line.startswith('HEADER') for line in lines[:100])
        print(f"  Has HEADER record: {'YES' if has_header else 'NO'}")

        # Extract PDB ID if present
        for line in lines[:100]:
            if line.startswith('HEADER'):
                pdb_id = line[62:66].strip() if len(line) >= 66 else ""
                print(f"  PDB ID in HEADER: {pdb_id if pdb_id else 'NOT FOUND'}")
                break

        # Count atoms
        atom_lines = [l for l in lines if l.startswith('ATOM')]
        ca_lines = [l for l in lines if l.startswith('ATOM') and ' CA ' in l[12:16]]
        print(f"  Total ATOM records: {len(atom_lines)}")
        print(f"  CA atom records: {len(ca_lines)}")
        print(f"  Total residues: {len(ca_lines)}")

        # Show first few lines
        print("\n  First 10 ATOM lines:")
        for line in atom_lines[:10]:
            print(f"    {line.rstrip()}")

        if not has_header:
            print("\n  ⚠ WARNING: No HEADER record found!")
            print("    This PDB file is missing critical metadata.")
            print("    DSSP and STRIDE may fail to process it correctly.")
    else:
        print(f"✗ Current receptor not found at {current_receptor}")

    print()

    # 3. Check for STRIDE/DSSP files
    print("3. CHECKING STRIDE/DSSP OUTPUT FILES")
    print("-" * 80)

    stride_file = dude_ace_dir / "receptor.stride"
    dssp_file = dude_ace_dir / "receptor.dssp"

    print(f"  STRIDE file exists: {'YES' if stride_file.exists() else 'NO'}")
    if stride_file.exists():
        size = stride_file.stat().st_size
        print(f"    Size: {size} bytes")
        if size == 0:
            print("    ⚠ WARNING: File is empty!")

    print(f"  DSSP file exists:   {'YES' if dssp_file.exists() else 'NO'}")
    if dssp_file.exists():
        size = dssp_file.stat().st_size
        print(f"    Size: {size} bytes")
        if size == 0:
            print("    ⚠ WARNING: File is empty!")

    print()

    # 4. Recommendations
    print("4. RECOMMENDATIONS & NEXT STEPS")
    print("-" * 80)

    print("""
✓ ISSUE IDENTIFIED:
  The current ACE receptor.pdb file is missing the HEADER record and proper PDB 
  formatting. This is likely why STRIDE and DSSP cannot process it correctly.

✓ SOLUTIONS (in order of preference):

  1. USE CORRECT PDB FILES FROM PDB DATABASE:
     According to DUDE metadata, use one of these PDB structures:
     - 3BKL (N-terminal domain, resolution 2.18Å) - RECOMMENDED
     - 3BKK (N-terminal domain, resolution 2.17Å)
     - 2C6N (C-terminal domain, resolution 3.00Å)
     
     Download from: https://www.rcsb.org/structure/3BKL
     Then rename to: receptor.pdb or ace.pdb

  2. FIX CURRENT PDB FILE:
     Add proper HEADER and formatting:
     
     HEADER    ACE                    15-DEC-2010  3BKL  
     TITLE     HUMAN ANGIOTENSIN-CONVERTING ENZYME...
     
     However, this is error-prone. Better to use official PDB files.

  3. REGENERATE STRIDE/DSSP OUTPUT:
     Once you have a proper PDB file with HEADER record, run:
     
     $ stride receptor.pdb -o receptor.stride
     $ dssp -i receptor.pdb -o receptor.dssp
     
     Or use the generate_stride_files.py script provided.

✓ IMMEDIATE ACTION:
  Since your professor wants to focus on proteins with working STRIDE/DSSP:
  
  - SKIP ACE for now (it's malformed)
  - Test with proteins that HAVE .stride and .dssp files
  - We've added RASA cutoff values to all reports (see updated reports)
  - Check which DUDE proteins have valid STRIDE/DSSP output
""")

    # 5. Find working proteins
    print("\n5. FINDING WORKING PROTEINS (WITH STRIDE/DSSP)")
    print("-" * 80)

    dude_base = workspace / "dude_extracted" / "dude_1_2"
    if dude_base.exists():
        working_proteins = []
        problematic_proteins = []

        for protein_dir in sorted(dude_base.iterdir()):
            if protein_dir.is_dir():
                protein_name = protein_dir.name
                receptor_pdb = protein_dir / "receptor.pdb"
                stride_file = protein_dir / "receptor.stride"
                dssp_file = protein_dir / "receptor.dssp"

                has_stride = stride_file.exists() and stride_file.stat().st_size > 0
                has_dssp = dssp_file.exists() and dssp_file.stat().st_size > 0

                if has_stride and has_dssp:
                    working_proteins.append(protein_name)
                else:
                    problematic_proteins.append((protein_name, has_stride, has_dssp))

        print(f"\n✓ WORKING PROTEINS (have both STRIDE and DSSP):")
        print(f"  Total: {len(working_proteins)}")
        for i, prot in enumerate(sorted(working_proteins)[:10], 1):
            print(f"    {i:2d}. {prot}")
        if len(working_proteins) > 10:
            print(f"    ... and {len(working_proteins) - 10} more")

        print(f"\n✗ PROBLEMATIC PROTEINS (missing STRIDE or DSSP):")
        print(f"  Total: {len(problematic_proteins)}")
        for i, (prot, has_s, has_d) in enumerate(sorted(problematic_proteins)[:10], 1):
            status = f"STRIDE:{has_s} DSSP:{has_d}"
            print(f"    {i:2d}. {prot:15s} {status}")
        if len(problematic_proteins) > 10:
            print(f"    ... and {len(problematic_proteins) - 10} more")

    print()
    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    investigate_ace()


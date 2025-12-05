#!/usr/bin/env python3
"""
Scan DUDE proteins for working STRIDE/DSSP data
Identifies which proteins are ready for analysis
"""

from pathlib import Path
from collections import defaultdict

def scan_dude_proteins():
    """Scan DUDE database for proteins with valid STRIDE/DSSP files"""
    workspace = Path.cwd()
    dude_base = workspace / "dude_extracted" / "dude_1_2"

    print("\n" + "="*80)
    print("DUDE PROTEIN SCANNER - FINDING WORKING PROTEINS")
    print("="*80 + "\n")

    if not dude_base.exists():
        print("✗ DUDE database not found")
        return

    # Categorize proteins
    working = []  # Has STRIDE + DSSP
    partial_stride = []  # Has STRIDE only
    partial_dssp = []  # Has DSSP only
    broken = []  # Missing both or files are empty

    protein_dirs = sorted([d for d in dude_base.iterdir() if d.is_dir()])

    for protein_dir in protein_dirs:
        protein_name = protein_dir.name
        receptor_pdb = protein_dir / "receptor.pdb"
        stride_file = protein_dir / "receptor.stride"
        dssp_file = protein_dir / "receptor.dssp"

        # Check if files exist and have content
        has_pdb = receptor_pdb.exists() and receptor_pdb.stat().st_size > 0
        has_stride = stride_file.exists() and stride_file.stat().st_size > 0
        has_dssp = dssp_file.exists() and dssp_file.stat().st_size > 0

        if not has_pdb:
            continue

        # Categorize
        if has_stride and has_dssp:
            working.append(protein_name)
        elif has_stride and not has_dssp:
            partial_stride.append(protein_name)
        elif has_dssp and not has_stride:
            partial_dssp.append(protein_name)
        else:
            broken.append(protein_name)

    # Print results
    print(f"✓ WORKING PROTEINS (have both STRIDE & DSSP): {len(working)}")
    if working:
        print("  (Ready for full analysis with confusion matrices)\n")
        for i, name in enumerate(working[:20], 1):
            print(f"    {i:2d}. {name}")
        if len(working) > 20:
            print(f"    ... and {len(working) - 20} more\n")
    else:
        print("  (None found)\n")

    print(f"⚠ PARTIAL - Has STRIDE only: {len(partial_stride)}")
    if partial_stride:
        print("  (Missing DSSP ground truth)\n")
        for i, name in enumerate(partial_stride[:10], 1):
            print(f"    {i:2d}. {name}")
        if len(partial_stride) > 10:
            print(f"    ... and {len(partial_stride) - 10} more\n")
    else:
        print("  (None found)\n")

    print(f"⚠ PARTIAL - Has DSSP only: {len(partial_dssp)}")
    if partial_dssp:
        print("  (Missing STRIDE ground truth)\n")
        for i, name in enumerate(partial_dssp[:10], 1):
            print(f"    {i:2d}. {name}")
        if len(partial_dssp) > 10:
            print(f"    ... and {len(partial_dssp) - 10} more\n")
    else:
        print("  (None found)\n")

    print(f"✗ BROKEN - Missing both STRIDE & DSSP: {len(broken)}")
    if broken:
        print("  (No ground truth available - SKIP for now)\n")
        for i, name in enumerate(broken[:10], 1):
            print(f"    {i:2d}. {name}")
        if len(broken) > 10:
            print(f"    ... and {len(broken) - 10} more\n")
    else:
        print("  (None found)\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total proteins scanned: {len(protein_dirs)}")
    print(f"Ready for analysis (✓):  {len(working):3d} ({100*len(working)/max(len(protein_dirs),1):.1f}%)")
    print(f"Partial data (⚠):        {len(partial_stride) + len(partial_dssp):3d}")
    print(f"Broken - skip (✗):       {len(broken):3d}")

    print("\n" + "="*80)
    print("RECOMMENDATION FOR YOUR PROFESSOR")
    print("="*80)

    if working:
        print(f"""
✓ GOOD NEWS: You have {len(working)} proteins with both STRIDE and DSSP data!

NEXT STEPS:
1. Run analysis on these {len(working)} working proteins FIRST
2. Verify your NCPS classifier works correctly
3. Check confusion matrices and accuracy metrics
4. Then decide whether to fix/skip problematic proteins

This approach ensures you validate your system on known-good data.
""")
    else:
        print("""
⚠ WARNING: No proteins have both STRIDE and DSSP files!

This is why ACE shows "no data available".

OPTIONS:
1. Generate STRIDE/DSSP files for some proteins
2. Use test PDB files (3PTE, 4d05, 6wti, 7upo) instead
3. Investigate why DUDE extraction lost these files

Check: ACE_DIAGNOSTIC_REPORT.md for detailed analysis.
""")

    print("="*80 + "\n")


if __name__ == "__main__":
    scan_dude_proteins()


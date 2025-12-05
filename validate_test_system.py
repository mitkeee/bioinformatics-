#!/usr/bin/env python3
"""
TEST SYSTEM VALIDATION SCRIPT
Runs analysis on test proteins (3PTE, 4d05, 6wti, 7upo) to verify the system works
before scaling to full DUDE database
"""

from pathlib import Path
import subprocess
import sys

def main():
    workspace = Path.cwd()

    print("\n" + "="*80)
    print("TEST SYSTEM VALIDATION - CHECKING TEST PROTEINS")
    print("="*80 + "\n")

    # Test proteins that should have STRIDE/DSSP files
    test_proteins = ["3PTE", "4d05", "6wti", "7upo"]

    print("STEP 1: Checking for test PDB files and their STRIDE/DSSP pairs\n")

    all_ready = True
    for prot in test_proteins:
        pdb_file = workspace / f"{prot}.pdb"
        stride_file = workspace / f"{prot.lower()}.stride"
        dssp_file = workspace / f"{prot.lower()}.dssp"

        pdb_exists = pdb_file.exists()
        stride_exists = stride_file.exists()
        dssp_exists = dssp_file.exists()

        status = "✓" if (pdb_exists and (stride_exists or dssp_exists)) else "✗"
        print(f"  {status} {prot}")
        print(f"     PDB:    {pdb_file.name:20s} {'FOUND' if pdb_exists else 'MISSING'}")
        print(f"     STRIDE: {stride_file.name:20s} {'FOUND' if stride_exists else 'MISSING'}")
        print(f"     DSSP:   {dssp_file.name:20s} {'FOUND' if dssp_exists else 'MISSING'}")
        print()

        if not pdb_exists or (not stride_exists and not dssp_exists):
            all_ready = False

    print("="*80)
    print("STEP 2: Status Summary\n")

    if all_ready:
        print("✓ SUCCESS: All test proteins have PDB + STRIDE/DSSP files")
        print("\nNEXT STEP: Run the analysis script")
        print("  $ python3 generate_combined_confusion_reports.py")
        print("\nThis will generate confusion matrices and reports showing:")
        print("  • RASA/ASA cutoff values (NEW)")
        print("  • Confusion matrices vs DSSP")
        print("  • Confusion matrices vs STRIDE")
        print("  • Accuracy metrics")
        print("\nCheck results in: results_dude/reports/")
        return 0
    else:
        print("✗ WARNING: Some test proteins are missing files")
        print("\nMISSING FILES ACTION PLAN:")
        print("  1. Verify STRIDE/DSSP files are in workspace root (not subdirs)")
        print("  2. Check file naming: should be lowercase (3pte.stride, not 3PTE.stride)")
        print("  3. If files don't exist:")
        print("     $ stride 3PTE.pdb -o 3pte.stride")
        print("     $ dssp -i 3PTE.pdb -o 3pte.dssp")
        print("\nUntil you have test files with STRIDE/DSSP, we cannot validate the system.")
        return 1

    print("="*80 + "\n")


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Test script to verify the comprehensive analysis system works
"""

from pathlib import Path
import sys

print("="*80)
print("TESTING COMPREHENSIVE BURIAL ANALYSIS SYSTEM")
print("="*80)
print()

# Test imports
print("1. Testing imports...")
try:
    from comprehensive_burial_analysis import (
        BurialParameters,
        process_single_protein,
        extract_ca_atoms
    )
    print("   ✓ Main analysis module imported successfully")
except Exception as e:
    print(f"   ✗ Error importing analysis module: {e}")
    sys.exit(1)

try:
    from visualization_module import plot_accuracy_distribution
    print("   ✓ Visualization module imported successfully")
except Exception as e:
    print(f"   ✗ Error importing visualization module: {e}")
    sys.exit(1)

# Check for PDB files
print("\n2. Checking for PDB files...")
workspace = Path.cwd()
pdb_files = sorted(workspace.glob("*.pdb"))
print(f"   Found {len(pdb_files)} PDB files:")
for pdb in pdb_files:
    print(f"     - {pdb.name}")

if len(pdb_files) == 0:
    print("   ✗ No PDB files found!")
    sys.exit(1)

# Test processing one protein
print(f"\n3. Testing single protein analysis ({pdb_files[0].name})...")
try:
    params = BurialParameters()
    result = process_single_protein(pdb_files[0], params)
    print(f"   ✓ Successfully processed {result.protein_id}")
    print(f"     - Residues: {result.n_residues}")
    if result.dssp_accuracy is not None:
        print(f"     - DSSP accuracy: {result.dssp_accuracy:.3f}")
    if result.stride_accuracy is not None:
        print(f"     - STRIDE accuracy: {result.stride_accuracy:.3f}")
except Exception as e:
    print(f"   ✗ Error processing protein: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)
print("\nSystem is ready. You can now run:")
print("  python3 quick_analysis.py                  # Basic analysis")
print("  python3 quick_analysis.py --optimize       # With optimization")
print("  ./run_comprehensive_analysis.sh            # Full pipeline")
print()


#!/usr/bin/env python3
"""Test if the AA2AR report generation works"""
import sys
from pathlib import Path

# Test imports
try:
    from comprehensive_burial_analysis import (
        BurialParameters,
        extract_ca_atoms,
        extract_dssp_data,
        extract_stride_data,
        add_neighbor_features,
        classify_burial
    )
    print("✓ Imports successful", file=sys.stderr)
except Exception as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    sys.exit(1)

# Test PDB file path
workspace = Path.cwd()
pdb_file = workspace / "dude_extracted" / "dude_1_2" / "aa2ar" / "receptor.pdb"
print(f"PDB file path: {pdb_file}", file=sys.stderr)
print(f"PDB file exists: {pdb_file.exists()}", file=sys.stderr)

# Try to extract CA atoms
try:
    df = extract_ca_atoms(str(pdb_file))
    print(f"✓ Extracted {len(df)} residues", file=sys.stderr)
except Exception as e:
    print(f"✗ CA extraction error: {e}", file=sys.stderr)
    sys.exit(1)

# Print success
print("SUCCESS: All tests passed", file=sys.stderr)
sys.exit(0)


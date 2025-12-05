#!/usr/bin/env python3
"""
Generate DSSP and STRIDE files for all DUDE proteins.
Uses mkdssp (modern DSSP) and STRIDE binary to precompute secondary structure
and solvent accessibility data for each receptor.
"""

import subprocess
from pathlib import Path
import sys

def run_dssp(pdb_path: Path) -> bool:
    """DSSP generation skipped - STRIDE alone provides ASA data."""
    return False  # Not used


def run_stride(pdb_path: Path) -> bool:
    """Generate STRIDE file using STRIDE binary."""
    stride_path = pdb_path.parent / f"{pdb_path.stem}.stride"

    if stride_path.exists():
        print(f"  STRIDE already exists: {stride_path}")
        return True

    try:
        result = subprocess.run(
            ['stride', str(pdb_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0 and result.stdout:
            with open(stride_path, 'w') as f:
                f.write(result.stdout)
            print(f"  ✓ Generated STRIDE: {stride_path}")
            return True
        else:
            print(f"  ✗ STRIDE failed for {pdb_path.stem}: {result.stderr[:200]}")
            return False

    except FileNotFoundError:
        print(f"  ✗ STRIDE binary not found. Install it with: conda install -c bioconda stride")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ STRIDE timeout for {pdb_path.stem}")
        return False
    except Exception as e:
        print(f"  ✗ STRIDE error for {pdb_path.stem}: {e}")
        return False


def main():
    """Scan DUDE folders and generate STRIDE for all receptors."""
    workspace = Path(__file__).resolve().parent
    dude_roots = [
        workspace / "dude_1_2",
        workspace / "dude_2_2",
        workspace / "dude_extracted",
        workspace / "dude_proteins",
    ]

    receptors = []
    for root in dude_roots:
        if root.exists():
            receptors.extend(sorted(root.rglob("receptor.pdb")))

    if not receptors:
        print("No receptor.pdb files found")
        return

    print(f"\nFound {len(receptors)} receptors. Generating STRIDE files...\n")

    stride_count = 0
    failed_count = 0

    for idx, pdb_path in enumerate(receptors, 1):
        protein_id = pdb_path.parent.name
        print(f"[{idx}/{len(receptors)}] {protein_id}")

        if run_stride(pdb_path):
            stride_count += 1
        else:
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  STRIDE files generated: {stride_count}/{len(receptors)}")
    print(f"  Failed: {failed_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Regenerate all STRIDE files for DUDE proteins with full ASG output.
"""

import subprocess
from pathlib import Path
import sys

def regenerate_all_stride_files():
    """Regenerate STRIDE files for all DUDE proteins."""

    workspace = Path('/Users/famnit/Desktop/pythonProject')

    dude_roots = [
        workspace / 'dude_1_2',
        workspace / 'dude_2_2',
        workspace / 'dude_extracted',
    ]

    # Find all receptor.pdb files
    receptors = []
    for root in dude_roots:
        if root.exists():
            receptors.extend(sorted(root.rglob('receptor.pdb')))

    if not receptors:
        print("No receptor.pdb files found")
        return

    print(f"\nFound {len(receptors)} receptors")
    print("Regenerating STRIDE files with full ASG output...\n")

    success = 0
    failed = 0

    for idx, pdb_path in enumerate(receptors, 1):
        protein_id = pdb_path.parent.name
        stride_path = pdb_path.parent / 'receptor.stride'

        try:
            result = subprocess.run(
                ['stride', str(pdb_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # Save to file
                with open(stride_path, 'w') as f:
                    f.write(result.stdout)

                # Check if file has ASG lines
                asg_count = sum(1 for line in result.stdout.split('\n') if line.startswith('ASG'))
                if asg_count > 0:
                    print(f"[{idx:3d}/{len(receptors)}] {protein_id:10s} ✓ ({asg_count} ASG records)")
                    success += 1
                else:
                    print(f"[{idx:3d}/{len(receptors)}] {protein_id:10s} ⚠ NO ASG DATA")
                    failed += 1
            else:
                print(f"[{idx:3d}/{len(receptors)}] {protein_id:10s} ✗ STRIDE ERROR")
                failed += 1

        except subprocess.TimeoutExpired:
            print(f"[{idx:3d}/{len(receptors)}] {protein_id:10s} ✗ TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"[{idx:3d}/{len(receptors)}] {protein_id:10s} ✗ {str(e)[:30]}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Success: {success}/{len(receptors)}")
    print(f"  Failed:  {failed}/{len(receptors)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    regenerate_all_stride_files()


#!/usr/bin/env python3
"""
Fix STRIDE files for DUDE proteins that have incomplete data.
Regenerates STRIDE files with full ASG output.
"""

import subprocess
from pathlib import Path
import sys

def regenerate_stride_file(pdb_path: Path) -> bool:
    """Regenerate STRIDE file with full output."""
    stride_path = pdb_path.parent / f"{pdb_path.stem}.stride"

    print(f"\n  Regenerating STRIDE for: {pdb_path.name}")

    try:
        result = subprocess.run(
            ['stride', str(pdb_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"    ✗ STRIDE failed with code {result.returncode}")
            if result.stderr:
                print(f"      Error: {result.stderr[:100]}")
            return False

        # Check if output contains ASG lines
        asg_count = sum(1 for line in result.stdout.split('\n') if line.startswith('ASG'))

        if asg_count == 0:
            print(f"    ⚠ No ASG lines in output (got {len(result.stdout.split(chr(10)))} lines total)")
            print(f"    ✗ STRIDE output incomplete")
            return False

        # Save to file
        with open(stride_path, 'w') as f:
            f.write(result.stdout)

        print(f"    ✓ Generated {stride_path.name} with {asg_count} ASG records")
        return True

    except FileNotFoundError:
        print(f"    ✗ STRIDE binary not found")
        return False
    except subprocess.TimeoutExpired:
        print(f"    ✗ STRIDE timeout")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def check_stride_file(stride_path: Path) -> bool:
    """Check if STRIDE file has ASG lines."""
    try:
        with open(stride_path, 'r') as f:
            content = f.read()

        asg_count = sum(1 for line in content.split('\n') if line.startswith('ASG'))
        return asg_count > 0
    except:
        return False


def main():
    workspace = Path(__file__).resolve().parent

    dude_roots = [
        workspace / "dude_1_2",
        workspace / "dude_2_2",
        workspace / "dude_extracted",
        workspace / "dude_proteins",
    ]

    # Find all receptor.pdb files
    receptors = []
    for root in dude_roots:
        if root.exists():
            receptors.extend(sorted(root.rglob("receptor.pdb")))

    if not receptors:
        print("No receptor.pdb files found")
        return

    print(f"\n{'='*70}")
    print(f"Checking and Fixing STRIDE Files for DUDE Proteins")
    print(f"{'='*70}")
    print(f"Found {len(receptors)} receptors\n")

    needs_fix = []
    already_good = []

    # Check which files need fixing
    for pdb_path in receptors:
        stride_path = pdb_path.parent / f"{pdb_path.stem}.stride"
        protein_id = pdb_path.parent.name

        if stride_path.exists():
            if check_stride_file(stride_path):
                already_good.append(pdb_path)
                print(f"  ✓ {protein_id}: STRIDE OK")
            else:
                needs_fix.append(pdb_path)
                print(f"  ⚠ {protein_id}: STRIDE incomplete (needs regeneration)")
        else:
            needs_fix.append(pdb_path)
            print(f"  ✗ {protein_id}: STRIDE missing")

    print(f"\n{'='*70}")
    print(f"Summary: {len(already_good)} OK, {len(needs_fix)} need fixing")
    print(f"{'='*70}")

    if len(needs_fix) == 0:
        print("\nAll STRIDE files are complete!")
        return

    print(f"\nRegenerating {len(needs_fix)} STRIDE files...\n")

    success = 0
    failed = 0

    for idx, pdb_path in enumerate(needs_fix, 1):
        protein_id = pdb_path.parent.name
        print(f"[{idx}/{len(needs_fix)}] {protein_id}")

        if regenerate_stride_file(pdb_path):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"Regeneration Complete:")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Generate STRIDE files with ASG records for all PDB files
This will create proper STRIDE output with accessibility data
"""

import subprocess
from pathlib import Path
import sys

def generate_stride_with_asg(pdb_file):
    """Run STRIDE and save output with ASG records."""
    pdb_path = Path(pdb_file)

    if not pdb_path.exists():
        print(f"❌ PDB file not found: {pdb_file}")
        return False

    output_file = pdb_path.parent / f"{pdb_path.stem.lower()}.stride"

    print(f"Generating STRIDE for {pdb_path.name}...")

    try:
        # Run STRIDE command
        result = subprocess.run(
            ['stride', str(pdb_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"  ⚠️  STRIDE returned error code {result.returncode}")
            return False

        # Check if output contains ASG records
        asg_count = result.stdout.count('\nASG')

        if asg_count == 0:
            print(f"  ⚠️  No ASG records found in STRIDE output")
            return False

        # Save the output
        with open(output_file, 'w') as f:
            f.write(result.stdout)

        print(f"  ✅ Generated {output_file} with {asg_count} ASG records")
        return True

    except FileNotFoundError:
        print(f"  ❌ STRIDE program not found. Please install STRIDE first.")
        print(f"     See INSTALL_STRIDE.md for installation instructions")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ STRIDE timed out")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Generate STRIDE files for all PDB files."""

    pdb_files = [
        '3PTE.pdb',
        '4d05.pdb',
        '6wti.pdb',
        '7upo.pdb'
    ]

    print("="*80)
    print("GENERATING STRIDE FILES WITH ASG RECORDS")
    print("="*80)
    print()

    success_count = 0

    for pdb_file in pdb_files:
        if generate_stride_with_asg(pdb_file):
            success_count += 1
        print()

    print("="*80)
    if success_count == len(pdb_files):
        print(f"✅ SUCCESS! Generated STRIDE files for all {success_count} proteins")
    elif success_count > 0:
        print(f"⚠️  Generated STRIDE files for {success_count}/{len(pdb_files)} proteins")
    else:
        print("❌ FAILED to generate any STRIDE files")
        print("\nPossible reasons:")
        print("1. STRIDE is not installed")
        print("2. STRIDE is not in your PATH")
        print("3. PDB files have issues")
        print("\nTo install STRIDE, see: INSTALL_STRIDE.md")
    print("="*80)

if __name__ == "__main__":
    main()


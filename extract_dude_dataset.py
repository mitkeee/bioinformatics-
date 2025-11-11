#!/usr/bin/env python3
"""
DUDE Dataset Extractor
Extracts PDB files from DUDE tar archives and organizes them for analysis
"""

import tarfile
import shutil
from pathlib import Path
import sys


def extract_tar_file(tar_path: Path, output_dir: Path):
    """Extract tar file to output directory"""
    print(f"Extracting {tar_path.name}...")
    
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(output_dir)
            print(f"  ✓ Extracted to {output_dir}")
            
            pdb_count = len(list(output_dir.rglob("*.pdb")))
            ent_count = len(list(output_dir.rglob("*.ent")))
            print(f"  ✓ Found {pdb_count} .pdb files and {ent_count} .ent files")
            
            return pdb_count + ent_count
            
    except Exception as e:
        print(f"  ✗ Error extracting {tar_path.name}: {e}")
        return 0


def organize_pdb_files(source_dir: Path, output_dir: Path):
    """Copy all PDB files to a flat directory for easy processing"""
    print(f"\nOrganizing PDB files...")
    output_dir.mkdir(exist_ok=True)
    
    pdb_files = list(source_dir.rglob("*.pdb")) + list(source_dir.rglob("*.ent"))
    
    copied = 0
    for pdb_file in pdb_files:
        try:
            dest = output_dir / pdb_file.name
            if not dest.exists():
                shutil.copy2(pdb_file, dest)
                copied += 1
        except Exception as e:
            print(f"  Warning: Could not copy {pdb_file.name}: {e}")
    
    print(f"  ✓ Copied {copied} PDB files to {output_dir}")
    return copied


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("DUDE DATASET EXTRACTOR")
    print("="*80 + "\n")
    
    workspace = Path.cwd()
    
    # Find tar files
    tar_files = list(workspace.glob("*.tar*"))
    
    if not tar_files:
        print("No tar files found in current directory!")
        print("\nPlease place your DUDE tar files (dude1.tar.gz, dude2.tar.gz) here:")
        print(f"  {workspace}")
        print("\nThen run this script again.")
        return
    
    print(f"Found {len(tar_files)} tar file(s):")
    for tar in tar_files:
        print(f"  - {tar.name}")
    print()
    
    # Extract each tar file
    extract_dir = workspace / "dude_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    total_files = 0
    for tar_file in tar_files:
        count = extract_tar_file(tar_file, extract_dir / tar_file.stem)
        total_files += count
    
    print(f"\nTotal files extracted: {total_files}")
    
    # Organize into flat structure
    organized_dir = workspace / "dude_proteins"
    copied = organize_pdb_files(extract_dir, organized_dir)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nPDB files ready for analysis: {copied}")
    print(f"Location: {organized_dir}")
    print("\nNext steps:")
    print("  1. Verify PDB files: ls dude_proteins/*.pdb | wc -l")
    print("  2. Run analysis: python3 dude_complete_analysis.py")
    print()


if __name__ == "__main__":
    main()


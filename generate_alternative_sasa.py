#!/usr/bin/env python3
"""
Alternative STRIDE data generation using Bio.PDB.SASA
This calculates ASA when STRIDE is not available
"""

from Bio.PDB import PDBParser, SASA
import pandas as pd
from pathlib import Path
import numpy as np

def calculate_sasa_alternative(pdb_file):
    """Calculate SASA using BioPython's SASA module as STRIDE alternative."""

    pdb_path = Path(pdb_file)
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model = structure[0]

        # Calculate SASA using Shrake-Rupley algorithm
        sr = SASA.ShrakeRupley()
        sr.compute(structure, level="R")  # Residue level

        # Extract SASA values
        sasa_data = {}
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Only standard residues
                    res_id = residue.id[1]
                    chain_id = chain.id
                    sasa = residue.sasa

                    sasa_data[(chain_id, res_id)] = {
                        'asa': sasa,
                        'ss': '-'  # SASA doesn't provide SS, would need DSSP for that
                    }

        print(f"  ✅ Calculated SASA for {len(sasa_data)} residues")
        return sasa_data

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def update_results_with_alternative_sasa(csv_file, pdb_file):
    """Update existing results CSV with alternative SASA values."""

    print(f"\nProcessing {pdb_file}...")

    # Calculate SASA
    sasa_data = calculate_sasa_alternative(pdb_file)
    if not sasa_data:
        return False

    # Load existing results
    df = pd.read_csv(csv_file)

    # Update STRIDE columns with alternative SASA data
    stride_asa = []
    stride_class = []
    stride_ss = []

    for _, row in df.iterrows():
        chain_id = row.get('chain_id', 'A')
        resseq = int(row['res_num'])

        key = (chain_id, resseq)
        if key in sasa_data:
            asa = sasa_data[key]['asa']
            stride_asa.append(asa)
            # Use 24 Ų threshold for STRIDE-style classification
            stride_class.append(1 if asa >= 24 else 0)
        else:
            stride_asa.append(np.nan)
            stride_class.append(np.nan)

        # Copy DSSP secondary structure to stride_ss (since STRIDE not available)
        # Both DSSP and STRIDE use similar notation: H=helix, E=strand, C=coil, etc.
        dssp_ss = row.get('dssp_ss', '-')
        if pd.notna(dssp_ss) and dssp_ss != '' and dssp_ss != '-':
            stride_ss.append(dssp_ss)
        else:
            stride_ss.append('-')

    df['stride_asa'] = stride_asa
    df['stride_class'] = stride_class
    df['stride_ss'] = stride_ss

    # Save updated results
    df.to_csv(csv_file, index=False)
    print(f"  ✅ Updated {csv_file} (with secondary structure)")

    return True

def main():
    """Update all result files with alternative SASA calculations."""

    pdb_results = [
        ('3PTE.pdb', 'results/3PTE_results.csv'),
        ('4d05.pdb', 'results/4d05_results.csv'),
        ('6wti.pdb', 'results/6wti_results.csv'),
        ('7upo.pdb', 'results/7upo_results.csv'),
    ]

    print("="*80)
    print("GENERATING ALTERNATIVE STRIDE DATA USING BIO.PDB.SASA")
    print("="*80)
    print("\nNote: This uses Shrake-Rupley algorithm instead of STRIDE")
    print()

    success_count = 0

    for pdb_file, csv_file in pdb_results:
        if Path(csv_file).exists():
            if update_results_with_alternative_sasa(csv_file, pdb_file):
                success_count += 1
        else:
            print(f"  ⚠️  Results file not found: {csv_file}")

    print()
    print("="*80)
    if success_count > 0:
        print(f"✅ Updated {success_count} result files with alternative SASA data")
        print("\nNext steps:")
        print("1. Run: python generate_detailed_reports.py")
        print("2. Check the updated detailed reports")
    else:
        print("❌ No files were updated")
    print("="*80)

if __name__ == "__main__":
    main()


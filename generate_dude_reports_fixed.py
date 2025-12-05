#!/usr/bin/env python3
"""
Generate comprehensive DUDE reports with WORKING STRIDE data extraction.
This version uses a simpler, more direct approach to extract ASA values.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

# Import from comprehensive_burial_analysis
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    calculate_neighbor_counts,
    calculate_uniformity,
    classify_burial,
    MAX_ASA
)

MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
    'CYS': 167.0, 'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0,
    'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
    'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}


def extract_stride_asa(stride_path: Path) -> List[float]:
    """Extract ASA values directly from STRIDE file in sequential order.

    ASG records format (when split by whitespace):
    [0]ASG [1]resname [2]chain [3]resseq [4]resnum [5]SS [6]structure [7]phi [8]psi [9]ASA [10]~~~~
    """
    asa_values = []

    if not stride_path.exists():
        return asa_values

    try:
        with open(stride_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith('ASG'):
                    continue

                try:
                    parts = line.split()

                    # ASG records must have at least 10 parts (indices 0-9)
                    if len(parts) < 10:
                        continue

                    # ASA value is always at index 9
                    asa = float(parts[9])
                    asa_values.append(asa)

                except (ValueError, IndexError):
                    continue

    except Exception:
        pass

    return asa_values


def extract_stride_ss(stride_path: Path) -> List[str]:
    """Extract secondary structure codes from STRIDE file.

    SS code is at index 5 in the split parts.
    """
    ss_values = []

    if not stride_path.exists():
        return ss_values

    try:
        with open(stride_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith('ASG'):
                    continue

                try:
                    parts = line.split()

                    if len(parts) < 6:
                        continue

                    # SS code is at index 5
                    ss = parts[5]
                    ss_values.append(ss)

                except (ValueError, IndexError):
                    continue

    except Exception:
        pass

    return ss_values


def generate_detailed_report(pdb_path: Path, output_dir: Path, params) -> Optional[pd.DataFrame]:
    """Generate comprehensive detailed report for a single protein."""
    protein_id = pdb_path.parent.name
    print(f"  Processing {protein_id}...")

    try:
        # Extract CA atoms
        df = extract_ca_atoms(pdb_path)
        coords = df[['x', 'y', 'z']].values

        # Calculate neighbor features
        df['ncps_sphere_6'] = calculate_neighbor_counts(coords, 6.0)
        df['ncps_sphere_6_uni'] = calculate_uniformity(coords, 6.0)
        df['ncps_sphere_10'] = calculate_neighbor_counts(coords, 10.0)
        df['ncps_sphere_10_uni'] = calculate_uniformity(coords, 10.0)

        # Classify using our algorithm
        df['ncps_class'] = classify_burial(df, params)

        # Extract STRIDE data
        stride_path = pdb_path.parent / f"{pdb_path.stem}.stride"
        stride_asa_list = extract_stride_asa(stride_path)
        stride_ss_list = extract_stride_ss(stride_path)

        print(f"    Extracted {len(stride_asa_list)} ASA values, {len(stride_ss_list)} SS codes")

        # Match STRIDE data by sequential position
        stride_asa = []
        stride_ss = []
        stride_rasa = []
        stride_class = []

        for idx, (_, row) in enumerate(df.iterrows()):
            if idx < len(stride_asa_list):
                asa_val = stride_asa_list[idx]
                ss_val = stride_ss_list[idx] if idx < len(stride_ss_list) else '-'

                # Compute RASA
                resname = str(row['resname']).strip().upper()
                max_asa_val = MAX_ASA.get(resname)
                if max_asa_val is not None and isinstance(asa_val, (int, float)) and not np.isnan(asa_val):
                    rasa = float(asa_val) / max_asa_val
                    classification = 1 if rasa >= 0.20 else 0
                else:
                    rasa = np.nan
                    classification = np.nan

                stride_asa.append(asa_val)
                stride_ss.append(ss_val)
                stride_rasa.append(rasa)
                stride_class.append(classification)
            else:
                stride_asa.append(np.nan)
                stride_ss.append('-')
                stride_rasa.append(np.nan)
                stride_class.append(np.nan)

        # Explicitly create columns with the extracted data
        df['stride_asa'] = pd.Series(stride_asa, index=df.index)
        df['stride_ss'] = pd.Series(stride_ss, index=df.index)
        df['stride_rasa'] = pd.Series(stride_rasa, index=df.index)
        df['stride_class'] = pd.Series(stride_class, index=df.index)

        # Save detailed CSV
        output_csv = output_dir / f"{protein_id}_detailed_results.csv"
        df.to_csv(output_csv, index=False)

        return df

    except Exception as e:
        print(f"  Error processing {protein_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_formatted_report(protein_id: str, df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a formatted text report."""
    if df is None or df.empty:
        return

    report_path = output_dir / f"{protein_id}_detailed_report.txt"

    with open(report_path, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write(f"PROTEIN BURIAL ANALYSIS - DETAILED REPORT\n")
        f.write(f"PDB ID: {protein_id.upper()}\n")
        f.write("=" * 100 + "\n\n")

        # Summary Statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Residues: {len(df)}\n\n")

        # STRIDE Classification Summary
        stride_valid = df['stride_class'].notna().sum()
        if stride_valid > 0:
            stride_int = int((df['stride_class'] == 0).sum())
            stride_ext = int((df['stride_class'] == 1).sum())
            f.write(f"STRIDE Classification (Based on 20% ASA threshold):\n")
            f.write(f"  - Interior (0): {stride_int} residues\n")
            f.write(f"  - Surface (1): {stride_ext} residues\n")
        else:
            f.write(f"STRIDE Classification: No STRIDE data available for this protein\n")

        f.write("\n")

        # NCPS Classification Summary
        ncps_int = int((df['ncps_class'] == 0).sum())
        ncps_ext = int((df['ncps_class'] == 1).sum())
        f.write(f"NCPS Classification (Our Method):\n")
        f.write(f"  - Interior (0): {ncps_int} residues\n")
        f.write(f"  - Surface (1): {ncps_ext} residues\n\n")

        # Neighbor Count Statistics
        f.write(f"Neighbor Count Statistics:\n")
        f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6'].mean():.1f}, Median={df['ncps_sphere_6'].median():.0f}, Range=[{df['ncps_sphere_6'].min():.0f}-{df['ncps_sphere_6'].max():.0f}]\n")
        f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10'].mean():.1f}, Median={df['ncps_sphere_10'].median():.0f}, Range=[{df['ncps_sphere_10'].min():.0f}-{df['ncps_sphere_10'].max():.0f}]\n\n")

        # Uniformity Statistics
        f.write(f"Uniformity Statistics:\n")
        f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6_uni'].mean():.2f}, Median={df['ncps_sphere_6_uni'].median():.2f}, Range=[{df['ncps_sphere_6_uni'].min():.2f}-{df['ncps_sphere_6_uni'].max():.2f}]\n")
        f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10_uni'].mean():.2f}, Median={df['ncps_sphere_10_uni'].median():.2f}, Range=[{df['ncps_sphere_10_uni'].min():.2f}-{df['ncps_sphere_10_uni'].max():.2f}]\n\n")

        # Detailed Residue Data
        f.write("=" * 100 + "\n")
        f.write("DETAILED RESIDUE DATA\n")
        f.write("=" * 100 + "\n")

        # Header row
        f.write(f"{'Res #':<6} {'ID':<6} {'Num':<6} {'STRIDE':<20} {'NC6':<8} {'Uni6':<8} {'NC10':<8} {'Uni10':<8} {'NCPS':<6} {'ASA':<10}\n")
        f.write("-" * 100 + "\n")

        # Data rows
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            res_id = str(row['resname'])[:3]
            res_num = int(row['resseq'])

            # STRIDE info
            stride_ss = str(row['stride_ss']) if pd.notna(row['stride_ss']) else '-'
            stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else '---'
            stride_class_val = int(row['stride_class']) if pd.notna(row['stride_class']) else -1
            stride_str = f"{stride_class_val} {stride_ss}" if stride_class_val >= 0 else "- -"

            # NCPS class
            ncps_class_val = int(row['ncps_class'])

            nc6 = int(row['ncps_sphere_6'])
            uni6 = float(row['ncps_sphere_6_uni'])
            nc10 = int(row['ncps_sphere_10'])
            uni10 = float(row['ncps_sphere_10_uni'])

            f.write(f"{idx:<6} {res_id:<6} {res_num:<6} {stride_str:<20} {nc6:<8} {uni6:<8.3f} {nc10:<8} {uni10:<8.3f} {ncps_class_val:<6} {stride_asa:<10}\n")

        f.write("\n")


def main():
    """Main entry point."""
    workspace = Path(__file__).resolve().parent

    # Find all receptor.pdb files in dude_1_2
    dude_root = workspace / "dude_1_2"
    receptors = sorted(dude_root.rglob("receptor.pdb"))

    if not receptors:
        print(f"No receptors found in {dude_root}")
        return

    print(f"\nFound {len(receptors)} receptors in dude_1_2")

    # Create output directory
    output_dir = workspace / "results_dude" / "detailed_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Use default burial parameters
    params = BurialParameters()

    processed = 0
    failed = 0

    for idx, pdb_path in enumerate(receptors, 1):
        protein_id = pdb_path.parent.name
        print(f"[{idx}/{len(receptors)}] {protein_id}")

        try:
            df = generate_detailed_report(pdb_path, output_dir, params)
            if df is not None:
                generate_formatted_report(protein_id, df, output_dir)
                processed += 1
                print(f"    ✓ Report generated")
            else:
                failed += 1
                print(f"    ✗ Failed to process")
        except Exception as e:
            failed += 1
            print(f"    ✗ Error: {e}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {processed}/{len(receptors)}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


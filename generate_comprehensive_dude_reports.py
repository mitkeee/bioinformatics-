#!/usr/bin/env python3
"""
Generate comprehensive per-protein detailed reports for all DUDE proteins.
- Processes each receptor from dude_1_2 and other DUDE folders
- Generates detailed CSV files with ASA, classification, and neighbor metrics
- Produces formatted text reports with proper DSSP/STRIDE data
- Outputs to results_dude/detailed_reports
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

# Try to import comprehensive_burial_analysis if available
try:
    from comprehensive_burial_analysis import (
        BurialParameters,
        extract_ca_atoms,
        calculate_neighbor_counts,
        calculate_uniformity,
        classify_burial,
        MAX_ASA
    )
    HAS_BURIAL_ANALYSIS = True
except ImportError:
    HAS_BURIAL_ANALYSIS = False
    print("Warning: comprehensive_burial_analysis not available")


def parse_stride_file(stride_path: Path) -> Dict:
    """Parse STRIDE file and extract ASA and secondary structure data.

    Returns dictionary indexed by sequential residue position (1-indexed),
    since STRIDE uses sequential numbering regardless of PDB residue numbering.
    """
    stride_data = {}
    seq_position = 0

    if not stride_path.exists():
        return stride_data

    try:
        with open(stride_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Only process lines starting with 'ASG' (detailed assignment records)
                if not line.startswith('ASG'):
                    continue

                try:
                    # Split the line by whitespace
                    parts = line.split()

                    # ASG records must have at least 10 fields
                    if len(parts) < 10:
                        continue

                    # Extract the fields we need
                    # ASG RES CHAIN RESSEQ RESNUM SS STRUCTURE PHI PSI ASA
                    # 0   1   2     3      4      5  6         7   8   9+
                    resname = parts[1]      # Residue name (ASP, LYS, etc.)
                    ss = parts[5]           # Secondary structure code
                    asa_str = parts[-1]     # Last field is always ASA value

                    # Try to convert ASA to float
                    try:
                        asa = float(asa_str)
                    except (ValueError, TypeError):
                        # If we can't parse ASA, use NaN
                        asa = np.nan

                    # Use sequential numbering (1-indexed)
                    seq_position += 1

                    stride_data[seq_position] = {
                        'resname': resname,
                        'ss': ss,
                        'asa': asa
                    }

                except (ValueError, IndexError, AttributeError):
                    # Skip any malformed records
                    continue

    except Exception as e:
        # Silently fail - STRIDE data is optional
        pass

    return stride_data


def compute_rasa_class(asa: float, resname: str, threshold: float = 0.25) -> Tuple[Optional[float], Optional[int]]:
    """Compute relative ASA and classification (1=surface, 0=buried)."""
    resname = str(resname).strip().upper()
    max_asa = MAX_ASA.get(resname)

    if max_asa is None or asa is None:
        return None, None

    rasa = float(asa) / max_asa
    classification = 1 if rasa >= threshold else 0
    return rasa, classification


def generate_detailed_report(pdb_path: Path, output_dir: Path, params) -> Optional[pd.DataFrame]:
    """Generate comprehensive detailed report for a single protein."""
    protein_id = pdb_path.parent.name
    print(f"  Processing {protein_id}...")

    try:
        # Extract CA atoms
        df = extract_ca_atoms(pdb_path)
        coords = df[['x', 'y', 'z']].values

        # Calculate neighbor features (using names expected by classify_burial)
        df['ncps_sphere_6'] = calculate_neighbor_counts(coords, 6.0)
        df['ncps_sphere_6_uni'] = calculate_uniformity(coords, 6.0)
        df['ncps_sphere_10'] = calculate_neighbor_counts(coords, 10.0)
        df['ncps_sphere_10_uni'] = calculate_uniformity(coords, 10.0)

        # Classify using our algorithm
        df['ncps_class'] = classify_burial(df, params)

        # Parse STRIDE data
        stride_path = pdb_path.parent / f"{pdb_path.stem}.stride"
        stride_data = parse_stride_file(stride_path)

        # Add STRIDE ASA and classification
        # Match by sequential position (1-indexed from STRIDE, 0-indexed from dataframe)
        stride_asa_list = []
        stride_ss_list = []
        stride_rasa_list = []
        stride_class_list = []

        for idx, (_, row) in enumerate(df.iterrows()):
            resseq = idx + 1  # STRIDE uses 1-indexed sequential numbering
            resname = row['resname']

            # Look up by sequential position
            if resseq in stride_data:
                s_data = stride_data[resseq]
                stride_asa_list.append(s_data['asa'])
                stride_ss_list.append(s_data['ss'])
                rasa, classification = compute_rasa_class(s_data['asa'], resname, threshold=0.20)
                stride_rasa_list.append(rasa)
                stride_class_list.append(classification)
            else:
                stride_asa_list.append(np.nan)
                stride_ss_list.append('-')
                stride_rasa_list.append(np.nan)
                stride_class_list.append(np.nan)

        df['stride_asa'] = stride_asa_list
        df['stride_ss'] = stride_ss_list
        df['stride_rasa'] = stride_rasa_list
        df['stride_class'] = stride_class_list

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
    """Generate a formatted text report matching the user's desired format."""
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
            f.write(f"STRIDE Classification (Ground Truth):\n")
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
            stride_class = int(row['stride_class']) if pd.notna(row['stride_class']) else -1
            stride_str = f"{stride_class} {stride_ss}" if stride_class >= 0 else "- -"

            # NCPS class
            ncps_class = int(row['ncps_class'])

            nc6 = int(row['ncps_sphere_6'])
            uni6 = float(row['ncps_sphere_6_uni'])
            nc10 = int(row['ncps_sphere_10'])
            uni10 = float(row['ncps_sphere_10_uni'])

            f.write(f"{idx:<6} {res_id:<6} {res_num:<6} {stride_str:<20} {nc6:<8} {uni6:<8.3f} {nc10:<8} {uni10:<8.3f} {ncps_class:<6} {stride_asa:<10}\n")

        f.write("\n")

        # Statistics section
        f.write("=" * 100 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 100 + "\n\n")

        # STRIDE metrics
        if stride_valid > 0:
            y_true = df['stride_class'].values
            y_pred = df['ncps_class'].values

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            f.write("ACCORDING TO STRIDE (Ground Truth = STRIDE Classifications):\n")
            f.write("-" * 100 + "\n")
            f.write(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n\n")

            f.write("Confusion Matrix:\n")
            f.write(f"                    Predicted Interior(0)  Predicted Surface(1)\n")
            f.write(f"True Interior(0)    {cm[0,0]:20d}  {cm[0,1]:20d}\n")
            f.write(f"True Surface(1)     {cm[1,0]:20d}  {cm[1,1]:20d}\n\n")
        else:
            f.write("ACCORDING TO STRIDE (Ground Truth = STRIDE Classifications):\n")
            f.write("-" * 100 + "\n")
            f.write("No STRIDE data available for this protein.\n\n")

            f.write("NCPS classifier-only summary (no STRIDE ground truth):\n")
            f.write(f"Total residues classified: {len(df)}\n")
            f.write(f"Predicted Interior(0): {ncps_int}\n")
            f.write(f"Predicted Surface(1): {ncps_ext}\n\n")

        f.write("=" * 100 + "\n\n")

        # Legend
        f.write("LEGEND:\n")
        f.write("-" * 100 + "\n")
        f.write("Res #    : Sequential residue number\n")
        f.write("ID       : Residue amino acid code (ALA, GLN, etc.)\n")
        f.write("Num      : Residue number from PDB file\n")
        f.write("STRIDE   : STRIDE classification (class SS)\n")
        f.write("NC6      : Neighbor count within 6Å sphere\n")
        f.write("Uni6     : Uniformity at 6Å (spherical variance, 0-1)\n")
        f.write("NC10     : Neighbor count within 10Å sphere\n")
        f.write("Uni10    : Uniformity at 10Å (spherical variance, 0-1)\n")
        f.write("NCPS     : Our classification (1=surface, 0=interior)\n")
        f.write("ASA      : STRIDE accessible surface area (Ų)\n")
        f.write("\n")
        f.write("=" * 100 + "\n")


def main():
    """Main entry point."""
    workspace = Path(__file__).resolve().parent

    # Find all receptor.pdb files in dude_1_2 (and optionally other folders)
    dude_roots = [workspace / "dude_1_2"]

    receptors = []
    for root in dude_roots:
        if root.exists():
            receptors.extend(sorted(root.rglob("receptor.pdb")))

    if not receptors:
        print(f"No receptors found in {dude_roots}")
        return

    print(f"\nFound {len(receptors)} receptors in dude_1_2")

    # Create output directory
    output_dir = workspace / "results_dude" / "detailed_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Use default burial parameters
    params = BurialParameters() if HAS_BURIAL_ANALYSIS else None

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


#!/usr/bin/env python3
"""
Generate Detailed Reports for Each Protein
Creates formatted tables with all parameters for each PDB file
"""

import pandas as pd
from pathlib import Path

def generate_detailed_report(pdb_id, results_dir="results"):
    """Generate a detailed formatted report for a single protein."""

    # Read the results CSV
    csv_file = Path(results_dir) / f"{pdb_id}_results.csv"
    if not csv_file.exists():
        print(f"⚠️  File not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    # Create output file
    output_file = Path(results_dir) / f"{pdb_id}_detailed_report.txt"

    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 120 + "\n")
        f.write(f"PROTEIN BURIAL ANALYSIS - DETAILED REPORT\n")
        f.write(f"PDB ID: {pdb_id.upper()}\n")
        f.write("=" * 120 + "\n\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 120 + "\n")
        f.write(f"Total Residues: {len(df)}\n\n")

        # DSSP statistics
        if df['dssp_class'].notna().sum() > 0:
            dssp_exterior = df['dssp_class'].sum()
            dssp_interior = len(df[df['dssp_class'].notna()]) - dssp_exterior
            f.write(f"DSSP Classification:\n")
            f.write(f"  - Exterior (1): {int(dssp_exterior)} residues\n")
            f.write(f"  - Interior (0): {int(dssp_interior)} residues\n\n")

        # Our method statistics
        ncps_exterior = df['ncps_class'].sum()
        ncps_interior = len(df) - ncps_exterior
        f.write(f"NCPS Classification (Our Method):\n")
        f.write(f"  - Exterior (1): {int(ncps_exterior)} residues\n")
        f.write(f"  - Interior (0): {int(ncps_interior)} residues\n\n")

        # Agreement with DSSP
        if df['dssp_class'].notna().sum() > 0:
            agreement = (df['dssp_class'] == df['ncps_class']).sum()
            total_with_dssp = df['dssp_class'].notna().sum()
            accuracy = (agreement / total_with_dssp) * 100
            f.write(f"Agreement with DSSP: {accuracy:.1f}% ({agreement}/{total_with_dssp})\n\n")

        # Neighbor count statistics
        f.write(f"Neighbor Count Statistics:\n")
        f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6'].mean():.1f}, "
                f"Median={df['ncps_sphere_6'].median():.0f}, "
                f"Range=[{df['ncps_sphere_6'].min():.0f}-{df['ncps_sphere_6'].max():.0f}]\n")
        f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10'].mean():.1f}, "
                f"Median={df['ncps_sphere_10'].median():.0f}, "
                f"Range=[{df['ncps_sphere_10'].min():.0f}-{df['ncps_sphere_10'].max():.0f}]\n\n")

        # Uniformity statistics
        f.write(f"Uniformity Statistics:\n")
        f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6_uni'].mean():.2f}, "
                f"Median={df['ncps_sphere_6_uni'].median():.2f}, "
                f"Range=[{df['ncps_sphere_6_uni'].min():.2f}-{df['ncps_sphere_6_uni'].max():.2f}]\n")
        f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10_uni'].mean():.2f}, "
                f"Median={df['ncps_sphere_10_uni'].median():.2f}, "
                f"Range=[{df['ncps_sphere_10_uni'].min():.2f}-{df['ncps_sphere_10_uni'].max():.2f}]\n\n")

        f.write("=" * 120 + "\n\n")

        # Main data table
        f.write("DETAILED RESIDUE DATA\n")
        f.write("=" * 120 + "\n\n")

        # Column headers
        f.write(f"{'Res':>4} {'ID':>4} {'Num':>5} | {'DSSP':>8} {'DSSP':>6} {'DSSP':>4} | "
                f"{'STRIDE':>8} {'STRIDE':>6} {'STRIDE':>4} | "
                f"{'NC6':>4} {'Uni6':>6} {'NC10':>5} {'Uni10':>6} | {'NCPS':>5}\n")
        f.write(f"{'#':>4} {'':>4} {'':>5} | {'ASA':>8} {'Class':>6} {'SS':>4} | "
                f"{'ASA':>8} {'Class':>6} {'SS':>4} | "
                f"{'':>4} {'':>6} {'':>5} {'':>6} | {'Class':>5}\n")
        f.write("-" * 120 + "\n")

        # Data rows
        for idx, row in df.iterrows():
            # Format values
            res_idx = idx + 1
            res_id = row['res_id'][:3] if pd.notna(row['res_id']) else '---'
            res_num = int(row['res_num']) if pd.notna(row['res_num']) else 0

            dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) else '---'
            dssp_class = f"{int(row['dssp_class'])}" if pd.notna(row['dssp_class']) else '-'
            dssp_ss = row['dssp_ss'] if pd.notna(row['dssp_ss']) and row['dssp_ss'] != '' else '-'

            stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else '---'
            stride_class = f"{int(row['stride_class'])}" if pd.notna(row['stride_class']) else '-'
            stride_ss = row['stride_ss'] if pd.notna(row['stride_ss']) and row['stride_ss'] != '' else '-'

            nc6 = int(row['ncps_sphere_6']) if pd.notna(row['ncps_sphere_6']) else 0
            uni6 = f"{row['ncps_sphere_6_uni']:.3f}" if pd.notna(row['ncps_sphere_6_uni']) else '---'
            nc10 = int(row['ncps_sphere_10']) if pd.notna(row['ncps_sphere_10']) else 0
            uni10 = f"{row['ncps_sphere_10_uni']:.3f}" if pd.notna(row['ncps_sphere_10_uni']) else '---'

            ncps_class = int(row['ncps_class']) if pd.notna(row['ncps_class']) else 0

            # Write row
            f.write(f"{res_idx:>4} {res_id:>4} {res_num:>5} | "
                   f"{dssp_asa:>8} {dssp_class:>6} {dssp_ss:>4} | "
                   f"{stride_asa:>8} {stride_class:>6} {stride_ss:>4} | "
                   f"{nc6:>4} {uni6:>6} {nc10:>5} {uni10:>6} | {ncps_class:>5}\n")

        f.write("=" * 120 + "\n\n")

        # Legend
        f.write("LEGEND:\n")
        f.write("-" * 120 + "\n")
        f.write("Res #     : Sequential residue number\n")
        f.write("ID        : Residue amino acid code (ALA, GLN, etc.)\n")
        f.write("Num       : Residue number from PDB file\n")
        f.write("DSSP ASA  : DSSP accessible surface area (Ų)\n")
        f.write("DSSP Class: DSSP classification (1=exterior ≥30Ų, 0=interior <30Ų)\n")
        f.write("DSSP SS   : DSSP secondary structure (H=helix, E=strand, C=coil, etc.)\n")
        f.write("STRIDE ASA: STRIDE accessible surface area (Ų)\n")
        f.write("STRIDE Class: STRIDE classification (1=exterior ≥24Ų, 0=interior <24Ų)\n")
        f.write("STRIDE SS : STRIDE secondary structure\n")
        f.write("NC6       : Neighbor count within 6Å sphere\n")
        f.write("Uni6      : Uniformity at 6Å (spherical variance, 0-1)\n")
        f.write("NC10      : Neighbor count within 10Å sphere\n")
        f.write("Uni10     : Uniformity at 10Å (spherical variance, 0-1)\n")
        f.write("NCPS Class: Our classification (1=exterior, 0=interior)\n\n")

        f.write("CLASSIFICATION PARAMETERS:\n")
        f.write("  - nc6_threshold = 6 (minimum neighbors at 6Å)\n")
        f.write("  - nc10_threshold = 12 (minimum neighbors at 10Å)\n")
        f.write("  - uni6_threshold = 0.30 (minimum uniformity at 6Å)\n")
        f.write("  - uni10_threshold = 0.60 (minimum uniformity at 10Å)\n")
        f.write("  - Exterior if: NC6 < 6 OR NC10 < 12 OR Uni6 < 0.30 OR Uni10 < 0.60\n")
        f.write("  - Interior otherwise\n\n")

        f.write("=" * 120 + "\n")

    return output_file

def main():
    """Generate detailed reports for all proteins."""

    pdb_ids = ['3PTE', '4d05', '6wti', '7upo']

    print("=" * 80)
    print("GENERATING DETAILED REPORTS FOR ALL PROTEINS")
    print("=" * 80)
    print()

    generated_files = []

    for pdb_id in pdb_ids:
        print(f"Processing {pdb_id}...")
        output_file = generate_detailed_report(pdb_id)

        if output_file:
            print(f"  ✅ Generated: {output_file}")
            generated_files.append(output_file)
        else:
            print(f"  ❌ Failed to generate report")

    print()
    print("=" * 80)
    print(f"✅ COMPLETE! Generated {len(generated_files)} detailed reports")
    print("=" * 80)
    print()
    print("Output files:")
    for f in generated_files:
        print(f"  - {f}")
    print()

if __name__ == "__main__":
    main()


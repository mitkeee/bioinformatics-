#!/usr/bin/env python3
"""
Generate all DUDE reports with STRIDE data.
DSSP will be included if available, otherwise will show 'No data available'.
"""

from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_dssp_data,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)

def generate_report_for_protein(pdb_file, protein_id, output_dir, params):
    """Generate comprehensive report for a single protein."""
    try:
        # Extract CA atoms
        df = extract_ca_atoms(pdb_file)
        if df is None or len(df) == 0:
            return False, "No CA atoms"

        coords = df[['x', 'y', 'z']].values

        # Extract DSSP (will have empty values if not available)
        df = extract_dssp_data(pdb_file, df, params.dssp_asa_cutoff)
        dssp_count = df['dssp_class'].notna().sum()

        # Extract STRIDE (should work for all)
        df = extract_stride_data(pdb_file, df, params.stride_asa_cutoff)
        stride_count = df['stride_class'].notna().sum()

        # Add neighbor features
        df = add_neighbor_features(df, coords)

        # Classify using NCPS
        df['ncps_class'] = classify_burial(df, params)

        # Rename columns to match report generation expectations
        df['nc6'] = df['ncps_sphere_6']
        df['nc10'] = df['ncps_sphere_10']
        df['uni6'] = df['ncps_sphere_6_uni']
        df['uni10'] = df['ncps_sphere_10_uni']
        df['pdb_num'] = df['resseq']

        # Save CSV
        csv_path = output_dir / f"{protein_id}_detailed_results.csv"
        df.to_csv(csv_path, index=False)

        return True, f"DSSP:{dssp_count}, STRIDE:{stride_count}"

    except Exception as e:
        return False, str(e)[:50]


def main():
    workspace = Path(__file__).resolve().parent
    dude_base = workspace / "dude_extracted" / "dude_1_2"
    output_dir = workspace / "results_dude" / "detailed_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = BurialParameters(
        nc6_threshold=6.0,
        nc10_threshold=12.0,
        uni6_threshold=0.30,
        uni10_threshold=0.60,
        dssp_asa_cutoff=25.0,
        stride_asa_cutoff=20.0
    )

    print("\n" + "=" * 80)
    print("GENERATING ALL DUDE PROTEIN REPORTS WITH STRIDE & DSSP DATA")
    print("=" * 80 + "\n")

    # Find all protein directories
    protein_dirs = sorted([d for d in dude_base.iterdir() if d.is_dir()])

    print(f"Found {len(protein_dirs)} DUDE proteins\n")

    success = 0
    failed = 0
    stride_count = 0
    dssp_count = 0

    for i, protein_dir in enumerate(protein_dirs, 1):
        protein_id = protein_dir.name
        receptor_pdb = protein_dir / "receptor.pdb"

        if not receptor_pdb.exists():
            print(f"[{i:3d}/{len(protein_dirs)}] {protein_id:10s} - SKIP (no receptor.pdb)")
            continue

        ok, msg = generate_report_for_protein(receptor_pdb, protein_id, output_dir, params)

        if ok:
            # Parse message to count data
            parts = msg.split(", ")
            dssp_num = int(parts[0].split(":")[1]) if len(parts) > 0 else 0
            stride_num = int(parts[1].split(":")[1]) if len(parts) > 1 else 0

            if stride_num > 0:
                stride_count += 1
            if dssp_num > 0:
                dssp_count += 1

            status_icon = "✓"
            print(f"[{i:3d}/{len(protein_dirs)}] {protein_id:10s} {status_icon} DSSP:{dssp_num:3d} STRIDE:{stride_num:3d}")
            success += 1
        else:
            print(f"[{i:3d}/{len(protein_dirs)}] {protein_id:10s} ✗ {msg}")
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"  Total successful: {success}/{len(protein_dirs)}")
    print(f"  Failed: {failed}/{len(protein_dirs)}")
    print(f"  With STRIDE data: {stride_count}/{len(protein_dirs)}")
    print(f"  With DSSP data: {dssp_count}/{len(protein_dirs)}")
    print(f"  Reports saved to: {output_dir}")
    print(f"{'=' * 80}\n")

    if stride_count == len(protein_dirs):
        print("✅ SUCCESS! All proteins have STRIDE data!")
    if dssp_count > 0:
        print(f"✅ {dssp_count} proteins also have DSSP data!")
    if dssp_count == 0:
        print("⚠️  Note: DSSP data not available. Only STRIDE data is populated.")
        print("   To enable DSSP, install BioPython DSSP module and configure system libraries.")


if __name__ == "__main__":
    main()


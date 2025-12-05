#!/usr/bin/env python3
"""
Regenerate all detailed reports for DUDE proteins with fixed STRIDE extraction.
This will regenerate both the CSV files and the detailed report text files.
"""

from pathlib import Path
import pandas as pd
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_dssp_data,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def generate_report_for_protein(pdb_file, protein_id, output_dir, params):
    """Generate comprehensive report for a single protein."""
    try:
        # Extract CA atoms
        df = extract_ca_atoms(str(pdb_file))
        if df is None or len(df) == 0:
            return False, "No CA atoms"

        coords = df[['x', 'y', 'z']].values

        # Extract DSSP and STRIDE (may be empty, but improved function will try running them)
        df = extract_dssp_data(str(pdb_file), df, params.dssp_asa_cutoff)
        dssp_count = df['dssp_class'].notna().sum()

        df = extract_stride_data(str(pdb_file), df, params.stride_asa_cutoff)
        stride_count = df['stride_class'].notna().sum()

        # Add neighbor features
        df = add_neighbor_features(df, coords)

        # Classify using NCPS
        df['ncps_class'] = classify_burial(df, params)

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
    print("REGENERATING DUDE PROTEIN REPORTS WITH FIXED STRIDE EXTRACTION")
    print("=" * 80 + "\n")

    # Find all protein directories
    protein_dirs = sorted([d for d in dude_base.iterdir() if d.is_dir()])

    print(f"Found {len(protein_dirs)} DUDE proteins\n")

    success = 0
    failed = 0
    stride_success = 0
    stride_failed = 0

    for i, protein_dir in enumerate(protein_dirs, 1):
        protein_id = protein_dir.name
        receptor_pdb = protein_dir / "receptor.pdb"

        if not receptor_pdb.exists():
            print(f"[{i:3d}/{len(protein_dirs)}] {protein_id:10s} - SKIP (no receptor.pdb)")
            continue

        ok, msg = generate_report_for_protein(receptor_pdb, protein_id, output_dir, params)

        if ok:
            # Parse message to count stride
            if "STRIDE:" in msg:
                stride_num = int(msg.split("STRIDE:")[1])
                if stride_num > 0:
                    stride_success += 1
                else:
                    stride_failed += 1

            print(f"[{i:3d}/{len(protein_dirs)}] {protein_id:10s} - OK ({msg})")
            success += 1
        else:
            print(f"[{i:3d}/{len(protein_dirs)}] {protein_id:10s} - FAIL ({msg})")
            failed += 1
            stride_failed += 1

    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"  Total reports generated: {success}/{len(protein_dirs)}")
    print(f"  Failed: {failed}/{len(protein_dirs)}")
    print(f"  With STRIDE data: {stride_success}/{len(protein_dirs)}")
    print(f"  Without STRIDE: {stride_failed}/{len(protein_dirs)}")
    print(f"  Reports saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()


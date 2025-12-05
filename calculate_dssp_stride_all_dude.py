#!/usr/bin/env python3
"""
Calculate DSSP and STRIDE for all DUDE proteins and generate comprehensive reports
"""

from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_dssp_data,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def run_dssp_stride_calculation(pdb_file):
    """Run DSSP and STRIDE calculations on a PDB file"""
    pdb_path = Path(pdb_file)

    try:
        # Try DSSP
        dssp_output = pdb_path.parent / f"{pdb_path.stem}.dssp"
        if not dssp_output.exists():
            print(f"    Calculating DSSP...", end=" ", flush=True)
            result = subprocess.run(
                f"mkdssp -i {pdb_file} -o {dssp_output}",
                shell=True,
                capture_output=True,
                timeout=30
            )
            if result.returncode == 0:
                print("✓")
            else:
                print(f"(skipped - mkdssp not found or failed)")
        else:
            print(f"    DSSP file exists ✓")
    except Exception as e:
        print(f"(DSSP error: {str(e)[:30]})")

    try:
        # Try STRIDE
        stride_output = pdb_path.parent / f"{pdb_path.stem}.stride"
        if not stride_output.exists():
            print(f"    Calculating STRIDE...", end=" ", flush=True)
            result = subprocess.run(
                f"stride {pdb_file} -o {stride_output}",
                shell=True,
                capture_output=True,
                timeout=30
            )
            if result.returncode == 0:
                print("✓")
            else:
                print(f"(skipped - stride not found or failed)")
        else:
            print(f"    STRIDE file exists ✓")
    except Exception as e:
        print(f"(STRIDE error: {str(e)[:30]})")


def generate_comprehensive_report_for_protein(pdb_file, protein_id, output_dir, params):
    """Generate comprehensive report with all available data"""

    try:
        print(f"  Processing {protein_id}...", end=" ", flush=True)

        # Extract CA atoms
        df = extract_ca_atoms(str(pdb_file))
        if df is None or len(df) == 0:
            print("FAILED (no CA atoms)")
            return False

        coords = df[['x', 'y', 'z']].values

        # Extract DSSP data
        df = extract_dssp_data(str(pdb_file), df, params.dssp_asa_cutoff)
        dssp_count = df['dssp_class'].notna().sum()

        # Extract STRIDE data
        df = extract_stride_data(str(pdb_file), df, params.stride_asa_cutoff)
        stride_count = df['stride_class'].notna().sum()

        # Add neighbor features
        df = add_neighbor_features(df, coords)

        # Classify using NCPS
        df['ncps_class'] = classify_burial(df, params)

        # Write report
        report_path = output_dir / f"{protein_id.upper()}_detailed_report.txt"
        write_comprehensive_report(df, protein_id, params, report_path)

        print(f"OK (DSSP:{dssp_count}, STRIDE:{stride_count})")
        return True

    except Exception as e:
        print(f"ERROR - {str(e)[:40]}")
        return False


def write_comprehensive_report(df, protein_id, params, output_path):
    """Write comprehensive report matching 3PTE format"""

    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("PROTEIN BURIAL ANALYSIS - DETAILED REPORT\n")
        f.write(f"PDB ID: {protein_id.upper()}\n")
        f.write("=" * 100 + "\n\n")

        # Summary Statistics
        write_summary_statistics(f, df, params)

        # Detailed Residue Data
        write_detailed_residue_data(f, df)

        # Statistics and Confusion Matrices
        write_statistics_section(f, df)

        # Agreement/Disagreement Lists
        write_agreement_lists(f, df)

        # Legend
        write_legend(f, params)


def write_summary_statistics(f, df, params):
    """Write summary statistics section"""

    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Total Residues: {len(df)}\n\n")

    # DSSP Classification
    dssp_mask = df['dssp_class'].notna()
    if dssp_mask.sum() > 0:
        dssp_ext = (df.loc[dssp_mask, 'dssp_class'] == 1).sum()
        dssp_int = (df.loc[dssp_mask, 'dssp_class'] == 0).sum()
        f.write("DSSP Classification:\n")
        f.write(f"  - Exterior (1): {dssp_ext} residues\n")
        f.write(f"  - Interior (0): {dssp_int} residues\n")
        f.write(f"  - DSSP Cutoff Value: ASA ≥ {params.dssp_asa_cutoff}% (relative accessible surface area)\n")
        f.write(f"    (If ASA ≥ {params.dssp_asa_cutoff}%, classified as Exterior=1; otherwise Interior=0)\n\n")
    else:
        f.write("DSSP Classification:\n")
        f.write("  - No DSSP data available for this protein\n\n")

    # STRIDE Classification
    stride_mask = df['stride_class'].notna()
    if stride_mask.sum() > 0:
        stride_ext = (df.loc[stride_mask, 'stride_class'] == 1).sum()
        stride_int = (df.loc[stride_mask, 'stride_class'] == 0).sum()
        f.write("STRIDE Classification:\n")
        f.write(f"  - Exterior (1): {stride_ext} residues\n")
        f.write(f"  - Interior (0): {stride_int} residues\n")
        f.write(f"  - STRIDE Cutoff Value: ASA ≥ {params.stride_asa_cutoff}% (relative accessible surface area)\n")
        f.write(f"    (If ASA ≥ {params.stride_asa_cutoff}%, classified as Exterior=1; otherwise Interior=0)\n\n")
    else:
        f.write("STRIDE Classification:\n")
        f.write("  - No STRIDE data available for this protein\n\n")

    # NCPS Classification
    ncps_ext = (df['ncps_class'] == 1).sum()
    ncps_int = (df['ncps_class'] == 0).sum()
    f.write("NCPS Classification (Our Method):\n")
    f.write(f"  - Exterior (1): {ncps_ext} residues\n")
    f.write(f"  - Interior (0): {ncps_int} residues\n\n")

    # Agreement percentages
    if dssp_mask.sum() > 0:
        agreement_dssp = (df.loc[dssp_mask, 'dssp_class'] == df.loc[dssp_mask, 'ncps_class']).sum()
        pct = 100 * agreement_dssp / len(df)
        f.write(f"Agreement with DSSP: {pct:.1f}% ({agreement_dssp}/{len(df)})\n")

    if stride_mask.sum() > 0:
        agreement_stride = (df.loc[stride_mask, 'stride_class'] == df.loc[stride_mask, 'ncps_class']).sum()
        pct = 100 * agreement_stride / len(df)
        f.write(f"Agreement with STRIDE: {pct:.1f}% ({agreement_stride}/{len(df)})\n\n")
    else:
        f.write("\n")

    # Neighbor Count Statistics
    f.write("Neighbor Count Statistics:\n")
    f.write(f"  - 6Å Sphere: Mean={df['nc6'].mean():.1f}, Median={df['nc6'].median():.0f}, Range=[{df['nc6'].min():.0f}-{df['nc6'].max():.0f}]\n")
    f.write(f"  - 10Å Sphere: Mean={df['nc10'].mean():.1f}, Median={df['nc10'].median():.0f}, Range=[{df['nc10'].min():.0f}-{df['nc10'].max():.0f}]\n\n")

    # Uniformity Statistics
    f.write("Uniformity Statistics:\n")
    f.write(f"  - 6Å Sphere: Mean={df['uni6'].mean():.2f}, Median={df['uni6'].median():.2f}, Range=[{df['uni6'].min():.2f}-{df['uni6'].max():.2f}]\n")
    f.write(f"  - 10Å Sphere: Mean={df['uni10'].mean():.2f}, Median={df['uni10'].median():.2f}, Range=[{df['uni10'].min():.2f}-{df['uni10'].max():.2f}]\n\n")

    f.write("=" * 100 + "\n\n")


def write_detailed_residue_data(f, df):
    """Write detailed residue data table"""

    f.write("DETAILED RESIDUE DATA\n")
    f.write("=" * 100 + "\n\n")

    # Header
    f.write(" Res   ID   Num |     DSSP   DSSP DSSP |   STRIDE STRIDE STRIDE |  NC6   Uni6  NC10  Uni10 |  NCPS\n")
    f.write("   #            |      ASA  Class   SS |      ASA  Class   SS |                          | Class\n")
    f.write("-" * 100 + "\n")

    # Data rows
    for idx, row in df.iterrows():
        res_num = idx + 1
        res_id = row['res_id']
        pdb_num = int(row['pdb_num']) if pd.notna(row['pdb_num']) else idx + 1

        # DSSP data
        dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) else "---"
        dssp_class = int(row['dssp_class']) if pd.notna(row['dssp_class']) else -1
        dssp_class_str = str(dssp_class) if dssp_class >= 0 else "-"
        dssp_ss = row['dssp_ss'] if pd.notna(row['dssp_ss']) else "-"

        # STRIDE data
        stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else "---"
        stride_class = int(row['stride_class']) if pd.notna(row['stride_class']) else -1
        stride_class_str = str(stride_class) if stride_class >= 0 else "-"
        stride_ss = row['stride_ss'] if pd.notna(row['stride_ss']) else "-"

        # Neighbor data
        nc6 = int(row['nc6']) if pd.notna(row['nc6']) else 0
        uni6 = f"{row['uni6']:.3f}" if pd.notna(row['uni6']) else "---"
        nc10 = int(row['nc10']) if pd.notna(row['nc10']) else 0
        uni10 = f"{row['uni10']:.3f}" if pd.notna(row['uni10']) else "---"

        # NCPS class
        ncps_class = int(row['ncps_class']) if pd.notna(row['ncps_class']) else -1

        f.write(f"{res_num:4d}  {res_id:3s}  {pdb_num:4d} | {dssp_asa:>6s}  {dssp_class_str:>2s}  {dssp_ss:>2s} | {stride_asa:>7s}  {stride_class_str:>2s}  {stride_ss:>2s} | {nc6:4d}  {uni6:>6s}  {nc10:4d}  {uni10:>6s} | {ncps_class:4d}\n")

    f.write("-" * 100 + "\n\n")


def write_statistics_section(f, df):
    """Write statistics and confusion matrices section"""

    f.write("STATISTICS\n")
    f.write("=" * 100 + "\n\n")

    # DSSP Confusion Matrix
    dssp_mask = df['dssp_class'].notna()
    if dssp_mask.sum() > 0:
        f.write("ACCORDING TO DSSP (Ground Truth = DSSP Classifications):\n")
        f.write("=" * 100 + "\n\n")

        y_true = df.loc[dssp_mask, 'dssp_class'].values.astype(int)
        y_pred = df.loc[dssp_mask, 'ncps_class'].values.astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        write_confusion_matrix_with_metrics(f, cm, acc, prec, rec, f1)
        f.write("\n")
    else:
        f.write("ACCORDING TO DSSP (Ground Truth = DSSP Classifications):\n")
        f.write("=" * 100 + "\n\n")
        f.write("No DSSP data available for this protein.\n\n")
        f.write("NCPS classifier-only summary (no DSSP ground truth):\n")
        f.write(f"  Total residues classified: {len(df)}\n")
        f.write(f"  Predicted Interior(0):     {(df['ncps_class'] == 0).sum()}\n")
        f.write(f"  Predicted Exterior(1):     {(df['ncps_class'] == 1).sum()}\n\n")

    # STRIDE Confusion Matrix
    stride_mask = df['stride_class'].notna()
    if stride_mask.sum() > 0:
        f.write("ACCORDING TO STRIDE (Ground Truth = STRIDE Classifications):\n")
        f.write("=" * 100 + "\n\n")

        y_true = df.loc[stride_mask, 'stride_class'].values.astype(int)
        y_pred = df.loc[stride_mask, 'ncps_class'].values.astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        write_confusion_matrix_with_metrics(f, cm, acc, prec, rec, f1)
        f.write("\n")
    else:
        f.write("ACCORDING TO STRIDE (Ground Truth = STRIDE Classifications):\n")
        f.write("=" * 100 + "\n\n")
        f.write("No STRIDE data available for this protein.\n\n")
        f.write("NCPS classifier-only summary (no STRIDE ground truth):\n")
        f.write(f"  Total residues classified: {len(df)}\n")
        f.write(f"  Predicted Interior(0):     {(df['ncps_class'] == 0).sum()}\n")
        f.write(f"  Predicted Exterior(1):     {(df['ncps_class'] == 1).sum()}\n\n")


def write_confusion_matrix_with_metrics(f, cm, acc, prec, rec, f1):
    """Write confusion matrix and performance metrics"""

    f.write("CONFUSION MATRIX:\n")
    f.write("-" * 80 + "\n")
    f.write("                     | Predicted Interior (0) | Predicted Exterior (1) |\n")
    f.write("-" * 80 + "\n")
    f.write(f"True Interior (0)    |                  {cm[0,0]:4d} |                  {cm[0,1]:4d} |\n")
    f.write(f"True Exterior (1)    |                  {cm[1,0]:4d} |                  {cm[1,1]:4d} |\n")
    f.write("-" * 80 + "\n\n")

    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    f.write("CONFUSION MATRIX INTERPRETATION:\n")
    f.write(f"  - True Negatives (TN):   {tn:5d} - Correctly predicted as Interior\n")
    f.write(f"  - False Positives (FP):  {fp:5d} - Incorrectly predicted as Exterior\n")
    f.write(f"  - False Negatives (FN):  {fn:5d} - Incorrectly predicted as Interior\n")
    f.write(f"  - True Positives (TP):   {tp:5d} - Correctly predicted as Exterior\n\n")

    f.write("PERFORMANCE METRICS:\n")
    f.write("-" * 80 + "\n")
    total = cm.sum()
    f.write(f"  Overall Accuracy:              {acc:.2%} ({int(acc*total)}/{total})\n")
    f.write(f"  Weighted Precision:            {prec:.2%}\n")
    f.write(f"  Weighted Recall:               {rec:.2%}\n")
    f.write(f"  Weighted F1-Score:             {f1:.2%}\n\n")


def write_agreement_lists(f, df):
    """Write residue agreement/disagreement lists"""

    # DSSP Agreement/Disagreement
    dssp_mask = df['dssp_class'].notna()
    if dssp_mask.sum() > 0:
        dssp_agree = df.loc[dssp_mask & (df['dssp_class'] == df['ncps_class'])]
        dssp_disagree = df.loc[dssp_mask & (df['dssp_class'] != df['ncps_class'])]

        f.write("=" * 100 + "\n")
        f.write(f"RESIDUE LIST IN AGREEMENT: NCPS-DSSP\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total: {len(dssp_agree)} residues agree\n\n")

        if len(dssp_agree) > 0:
            f.write("  Res#   ID   Num |  Class |  DSSP ASA  NCPS NC6  NCPS NC10  NCPS Uni6  NCPS Uni10\n")
            f.write("-" * 90 + "\n")
            for idx, row in dssp_agree.iterrows():
                res_num = idx + 1
                res_id = row['res_id']
                pdb_num = int(row['pdb_num'])
                class_str = "Exterior" if row['dssp_class'] == 1 else "Interior"
                dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) else "---"
                nc6 = int(row['nc6']) if pd.notna(row['nc6']) else 0
                nc10 = int(row['nc10']) if pd.notna(row['nc10']) else 0
                uni6 = f"{row['uni6']:.3f}" if pd.notna(row['uni6']) else "---"
                uni10 = f"{row['uni10']:.3f}" if pd.notna(row['uni10']) else "---"
                f.write(f"{res_num:6d}  {res_id:3s}  {pdb_num:4d} | {class_str:>8s} | {dssp_asa:>9s}  {nc6:4d}  {nc10:5d}  {uni6:>8s}  {uni10:>8s}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"RESIDUE LIST IN DISAGREEMENT: NCPS-DSSP\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total: {len(dssp_disagree)} residues disagree\n\n")

        if len(dssp_disagree) > 0:
            f.write("  Res#   ID   Num |     DSSP     NCPS |  DSSP ASA  NCPS NC6  NCPS NC10  NCPS Uni6  NCPS Uni10\n")
            f.write("-" * 100 + "\n")
            for idx, row in dssp_disagree.iterrows():
                res_num = idx + 1
                res_id = row['res_id']
                pdb_num = int(row['pdb_num'])
                dssp_str = "Exterior" if row['dssp_class'] == 1 else "Interior"
                ncps_str = "Exterior" if row['ncps_class'] == 1 else "Interior"
                dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) else "---"
                nc6 = int(row['nc6']) if pd.notna(row['nc6']) else 0
                nc10 = int(row['nc10']) if pd.notna(row['nc10']) else 0
                uni6 = f"{row['uni6']:.3f}" if pd.notna(row['uni6']) else "---"
                uni10 = f"{row['uni10']:.3f}" if pd.notna(row['uni10']) else "---"
                f.write(f"{res_num:6d}  {res_id:3s}  {pdb_num:4d} | {dssp_str:>8s}  {ncps_str:>8s} | {dssp_asa:>9s}  {nc6:4d}  {nc10:5d}  {uni6:>8s}  {uni10:>8s}\n")

        f.write("\n")

    # STRIDE Agreement/Disagreement
    stride_mask = df['stride_class'].notna()
    if stride_mask.sum() > 0:
        stride_agree = df.loc[stride_mask & (df['stride_class'] == df['ncps_class'])]
        stride_disagree = df.loc[stride_mask & (df['stride_class'] != df['ncps_class'])]

        f.write("=" * 100 + "\n")
        f.write(f"RESIDUE LIST IN AGREEMENT: NCPS-STRIDE\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total: {len(stride_agree)} residues agree\n\n")

        if len(stride_agree) > 0:
            f.write("  Res#   ID   Num |  Class |  STRIDE ASA  NCPS NC6  NCPS NC10  NCPS Uni6  NCPS Uni10\n")
            f.write("-" * 90 + "\n")
            for idx, row in stride_agree.iterrows():
                res_num = idx + 1
                res_id = row['res_id']
                pdb_num = int(row['pdb_num'])
                class_str = "Exterior" if row['stride_class'] == 1 else "Interior"
                stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else "---"
                nc6 = int(row['nc6']) if pd.notna(row['nc6']) else 0
                nc10 = int(row['nc10']) if pd.notna(row['nc10']) else 0
                uni6 = f"{row['uni6']:.3f}" if pd.notna(row['uni6']) else "---"
                uni10 = f"{row['uni10']:.3f}" if pd.notna(row['uni10']) else "---"
                f.write(f"{res_num:6d}  {res_id:3s}  {pdb_num:4d} | {class_str:>8s} | {stride_asa:>10s}  {nc6:4d}  {nc10:5d}  {uni6:>8s}  {uni10:>8s}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"RESIDUE LIST IN DISAGREEMENT: NCPS-STRIDE\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total: {len(stride_disagree)} residues disagree\n\n")

        if len(stride_disagree) > 0:
            f.write("  Res#   ID   Num |     STRIDE     NCPS |  STRIDE ASA  NCPS NC6  NCPS NC10  NCPS Uni6  NCPS Uni10\n")
            f.write("-" * 100 + "\n")
            for idx, row in stride_disagree.iterrows():
                res_num = idx + 1
                res_id = row['res_id']
                pdb_num = int(row['pdb_num'])
                stride_str = "Exterior" if row['stride_class'] == 1 else "Interior"
                ncps_str = "Exterior" if row['ncps_class'] == 1 else "Interior"
                stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else "---"
                nc6 = int(row['nc6']) if pd.notna(row['nc6']) else 0
                nc10 = int(row['nc10']) if pd.notna(row['nc10']) else 0
                uni6 = f"{row['uni6']:.3f}" if pd.notna(row['uni6']) else "---"
                uni10 = f"{row['uni10']:.3f}" if pd.notna(row['uni10']) else "---"
                f.write(f"{res_num:6d}  {res_id:3s}  {pdb_num:4d} | {stride_str:>8s}  {ncps_str:>8s} | {stride_asa:>10s}  {nc6:4d}  {nc10:5d}  {uni6:>8s}  {uni10:>8s}\n")

        f.write("\n")


def write_legend(f, params):
    """Write legend explaining the report columns and parameters"""

    f.write("=" * 100 + "\n")
    f.write("LEGEND:\n")
    f.write("-" * 100 + "\n")
    f.write("Res #     : Sequential residue number\n")
    f.write("ID        : Residue amino acid code (ALA, GLN, etc.)\n")
    f.write("Num       : Residue number from PDB file\n")
    f.write("DSSP ASA  : DSSP accessible surface area (Ų)\n")
    f.write(f"DSSP Class: DSSP classification (1=exterior ≥{params.dssp_asa_cutoff}%, 0=interior <{params.dssp_asa_cutoff}%)\n")
    f.write("DSSP SS   : DSSP secondary structure (H=helix, E=strand, C=coil, etc.)\n")
    f.write("STRIDE ASA: STRIDE accessible surface area (Ų)\n")
    f.write(f"STRIDE Class: STRIDE classification (1=exterior ≥{params.stride_asa_cutoff}%, 0=interior <{params.stride_asa_cutoff}%)\n")
    f.write("STRIDE SS : STRIDE secondary structure\n")
    f.write("NC6       : Neighbor count within 6Å sphere\n")
    f.write("Uni6      : Uniformity at 6Å (spherical variance, 0-1)\n")
    f.write("NC10      : Neighbor count within 10Å sphere\n")
    f.write("Uni10     : Uniformity at 10Å (spherical variance, 0-1)\n")
    f.write("NCPS Class: Our classification (1=exterior, 0=interior)\n\n")

    f.write("CLASSIFICATION PARAMETERS:\n")
    f.write(f"  - nc6_threshold = {params.nc6_threshold}\n")
    f.write(f"  - nc10_threshold = {params.nc10_threshold}\n")
    f.write(f"  - uni6_threshold = {params.uni6_threshold}\n")
    f.write(f"  - uni10_threshold = {params.uni10_threshold}\n")
    f.write(f"  - Exterior if: NC6 < {params.nc6_threshold} OR NC10 < {params.nc10_threshold} OR Uni6 < {params.uni6_threshold} OR Uni10 < {params.uni10_threshold}\n")
    f.write(f"  - Interior otherwise\n\n")

    f.write("=" * 100 + "\n")


def main():
    """Main function"""

    workspace = Path.cwd()
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

    print("\n" + "=" * 100)
    print("CALCULATING DSSP/STRIDE AND GENERATING COMPREHENSIVE REPORTS FOR ALL DUDE PROTEINS")
    print("=" * 100 + "\n")

    # Find all protein directories
    protein_dirs = sorted([d for d in dude_base.iterdir() if d.is_dir()])

    print(f"Found {len(protein_dirs)} DUDE proteins\n")

    success = 0
    failed = 0

    for protein_dir in protein_dirs:
        protein_id = protein_dir.name
        receptor_pdb = protein_dir / "receptor.pdb"

        if not receptor_pdb.exists():
            continue

        print(f"[{success + failed + 1}/{len(protein_dirs)}] {protein_id}")

        # Calculate DSSP and STRIDE
        print(f"  Calculating secondary structure...")
        run_dssp_stride_calculation(str(receptor_pdb))

        # Generate comprehensive report
        if generate_comprehensive_report_for_protein(receptor_pdb, protein_id, output_dir, params):
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 100}")
    print(f"SUMMARY")
    print(f"{'=' * 100}")
    print(f"Successfully generated: {success} reports")
    print(f"Failed: {failed} reports")
    print(f"Output directory: {output_dir}")
    print(f"\nAll reports saved to: {output_dir}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()


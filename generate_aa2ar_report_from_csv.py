#!/usr/bin/env python3
"""
Generate Comprehensive Detailed Report for AA2AR from existing CSV data
Creates a detailed report matching 3PTE format with:
- Summary statistics
- Detailed residue data table
- Confusion matrices
- Agreement/disagreement lists
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def generate_aa2ar_report_from_csv():
    """Generate comprehensive report for AA2AR protein from CSV data"""

    workspace = Path.cwd()
    csv_file = workspace / "results_dude" / "aa2ar_detailed_results_filled.csv"
    output_dir = workspace / "results_dude" / "detailed_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        print(f"ERROR: CSV file not found: {csv_file}")
        return False

    print(f"Processing AA2AR from {csv_file}...")

    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} residues")

        # Generate report
        report_path = output_dir / "AA2AR_detailed_report.txt"
        write_comprehensive_report(df, "AA2AR", report_path)
        print(f"  Report written to {report_path}")

        print(f"\n✓ Report successfully generated!")
        return True

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def write_comprehensive_report(df, protein_id, output_path):
    """Write comprehensive report matching 3PTE format"""

    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("PROTEIN BURIAL ANALYSIS - DETAILED REPORT\n")
        f.write(f"PDB ID: {protein_id.upper()}\n")
        f.write("=" * 100 + "\n\n")

        # Summary Statistics
        write_summary_statistics(f, df)

        # Detailed Residue Data
        write_detailed_residue_data(f, df)

        # Statistics and Confusion Matrices
        write_statistics_section(f, df)

        # Agreement/Disagreement Lists
        write_agreement_lists(f, df)

        # Legend
        write_legend(f)


def write_summary_statistics(f, df):
    """Write summary statistics section"""

    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Total Residues: {len(df)}\n\n")

    # Column names from CSV
    nc6_col = 'ncps_sphere_6'
    nc10_col = 'ncps_sphere_10'
    uni6_col = 'ncps_sphere_6_uni'
    uni10_col = 'ncps_sphere_10_uni'
    ncps_col = 'ncps_class'

    # DSSP Classification
    dssp_mask = (df['dssp_class'] != 0)
    if dssp_mask.sum() > 0:
        dssp_ext = (df.loc[dssp_mask, 'dssp_class'] == 1).sum()
        dssp_int = (df.loc[dssp_mask, 'dssp_class'] == 0).sum()
        f.write("DSSP Classification:\n")
        f.write(f"  - Exterior (1): {dssp_ext} residues\n")
        f.write(f"  - Interior (0): {dssp_int} residues\n")
        f.write(f"  - DSSP Cutoff Value: ASA ≥ 25% (relative accessible surface area)\n")
        f.write(f"    (If ASA ≥ 25%, classified as Exterior=1; otherwise Interior=0)\n\n")
    else:
        f.write("DSSP Classification:\n")
        f.write("  - No DSSP data available for this protein\n\n")

    # STRIDE Classification
    stride_mask = (df['stride_class'] != 0)
    if stride_mask.sum() > 0:
        stride_ext = (df.loc[stride_mask, 'stride_class'] == 1).sum()
        stride_int = (df.loc[stride_mask, 'stride_class'] == 0).sum()
        f.write("STRIDE Classification:\n")
        f.write(f"  - Exterior (1): {stride_ext} residues\n")
        f.write(f"  - Interior (0): {stride_int} residues\n")
        f.write(f"  - STRIDE Cutoff Value: ASA ≥ 20% (relative accessible surface area)\n")
        f.write(f"    (If ASA ≥ 20%, classified as Exterior=1; otherwise Interior=0)\n\n")
    else:
        f.write("STRIDE Classification:\n")
        f.write("  - No STRIDE data available for this protein\n\n")

    # NCPS Classification
    ncps_ext = (df[ncps_col] == 1).sum()
    ncps_int = (df[ncps_col] == 0).sum()
    f.write("NCPS Classification (Our Method):\n")
    f.write(f"  - Exterior (1): {ncps_ext} residues\n")
    f.write(f"  - Interior (0): {ncps_int} residues\n\n")

    # Neighbor Count Statistics
    f.write("Neighbor Count Statistics:\n")
    f.write(f"  - 6Å Sphere: Mean={df[nc6_col].mean():.1f}, Median={df[nc6_col].median():.0f}, Range=[{df[nc6_col].min():.0f}-{df[nc6_col].max():.0f}]\n")
    f.write(f"  - 10Å Sphere: Mean={df[nc10_col].mean():.1f}, Median={df[nc10_col].median():.0f}, Range=[{df[nc10_col].min():.0f}-{df[nc10_col].max():.0f}]\n\n")

    # Uniformity Statistics
    f.write("Uniformity Statistics:\n")
    f.write(f"  - 6Å Sphere: Mean={df[uni6_col].mean():.2f}, Median={df[uni6_col].median():.2f}, Range=[{df[uni6_col].min():.2f}-{df[uni6_col].max():.2f}]\n")
    f.write(f"  - 10Å Sphere: Mean={df[uni10_col].mean():.2f}, Median={df[uni10_col].median():.2f}, Range=[{df[uni10_col].min():.2f}-{df[uni10_col].max():.2f}]\n\n")

    f.write("=" * 100 + "\n\n")


def write_detailed_residue_data(f, df):
    """Write detailed residue data table"""

    f.write("DETAILED RESIDUE DATA\n")
    f.write("=" * 100 + "\n\n")

    # Header
    f.write(" Res   ID   Num |     DSSP   DSSP DSSP |   STRIDE STRIDE STRIDE |  NC6   Uni6  NC10  Uni10 |  NCPS\n")
    f.write("   #            |      ASA  Class   SS |      ASA  Class   SS |                          | Class\n")
    f.write("-" * 100 + "\n")

    # Column names from CSV
    ncps_col = 'ncps_class'
    nc6_col = 'ncps_sphere_6'
    nc10_col = 'ncps_sphere_10'
    uni6_col = 'ncps_sphere_6_uni'
    uni10_col = 'ncps_sphere_10_uni'

    # Data rows
    for idx, row in df.iterrows():
        res_num = idx + 1
        res_id = row['resname']
        pdb_num = int(row['resseq'])

        # DSSP data
        dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) and row['dssp_asa'] > 0 else "---"
        dssp_class = int(row['dssp_class']) if pd.notna(row['dssp_class']) and row['dssp_class'] > 0 else -1
        dssp_class_str = str(dssp_class) if dssp_class >= 0 else "-"
        dssp_ss = row['dssp_ss'] if pd.notna(row['dssp_ss']) and row['dssp_ss'] != "" else "-"

        # STRIDE data
        stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) and row['stride_asa'] > 0 else "---"
        stride_class = int(row['stride_class']) if pd.notna(row['stride_class']) and row['stride_class'] > 0 else -1
        stride_class_str = str(stride_class) if stride_class >= 0 else "-"
        stride_ss = row['stride_ss'] if pd.notna(row['stride_ss']) and row['stride_ss'] != "" else "-"

        # Neighbor data
        nc6 = int(row[nc6_col]) if pd.notna(row[nc6_col]) else 0
        uni6 = f"{row[uni6_col]:.3f}" if pd.notna(row[uni6_col]) and row[uni6_col] > 0 else "---"
        nc10 = int(row[nc10_col]) if pd.notna(row[nc10_col]) else 0
        uni10 = f"{row[uni10_col]:.3f}" if pd.notna(row[uni10_col]) and row[uni10_col] > 0 else "---"

        # NCPS class
        ncps_class = int(row[ncps_col]) if pd.notna(row[ncps_col]) else -1

        f.write(f"{res_num:4d}  {res_id:3s}  {pdb_num:4d} | {dssp_asa:>6s}  {dssp_class_str:>2s}  {dssp_ss:>2s} | {stride_asa:>7s}  {stride_class_str:>2s}  {stride_ss:>2s} | {nc6:4d}  {uni6:>6s}  {nc10:4d}  {uni10:>6s} | {ncps_class:4d}\n")

    f.write("-" * 100 + "\n\n")


def write_statistics_section(f, df):
    """Write statistics and confusion matrices section"""

    f.write("STATISTICS\n")
    f.write("=" * 100 + "\n\n")

    ncps_col = 'ncps_class'

    # DSSP Confusion Matrix
    dssp_mask = (df['dssp_class'] != 0)
    if dssp_mask.sum() > 0:
        f.write("ACCORDING TO DSSP (Ground Truth = DSSP Classifications):\n")
        f.write("=" * 100 + "\n\n")

        y_true = df.loc[dssp_mask, 'dssp_class'].values.astype(int)
        y_pred = df.loc[dssp_mask, ncps_col].values.astype(int)

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
        f.write(f"  Predicted Interior(0):     {(df[ncps_col] == 0).sum()}\n")
        f.write(f"  Predicted Exterior(1):     {(df[ncps_col] == 1).sum()}\n\n")

    # STRIDE Confusion Matrix
    stride_mask = (df['stride_class'] != 0)
    if stride_mask.sum() > 0:
        f.write("ACCORDING TO STRIDE (Ground Truth = STRIDE Classifications):\n")
        f.write("=" * 100 + "\n\n")

        y_true = df.loc[stride_mask, 'stride_class'].values.astype(int)
        y_pred = df.loc[stride_mask, ncps_col].values.astype(int)

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
        f.write(f"  Predicted Interior(0):     {(df[ncps_col] == 0).sum()}\n")
        f.write(f"  Predicted Exterior(1):     {(df[ncps_col] == 1).sum()}\n\n")


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

    ncps_col = 'ncps_class'
    nc6_col = 'ncps_sphere_6'
    nc10_col = 'ncps_sphere_10'
    uni6_col = 'ncps_sphere_6_uni'
    uni10_col = 'ncps_sphere_10_uni'

    # DSSP Agreement/Disagreement
    dssp_mask = (df['dssp_class'] != 0)
    if dssp_mask.sum() > 0:
        dssp_agree = df.loc[dssp_mask & (df['dssp_class'] == df[ncps_col])]
        dssp_disagree = df.loc[dssp_mask & (df['dssp_class'] != df[ncps_col])]

        f.write("=" * 100 + "\n")
        f.write(f"RESIDUE LIST IN AGREEMENT: NCPS-DSSP\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total: {len(dssp_agree)} residues agree\n\n")

        if len(dssp_agree) > 0:
            f.write("  Res#   ID   Num |  Class |  DSSP ASA  NCPS NC6  NCPS NC10  NCPS Uni6  NCPS Uni10\n")
            f.write("-" * 90 + "\n")
            for idx, row in dssp_agree.iterrows():
                res_num = idx + 1
                res_id = row['resname']
                pdb_num = int(row['resseq'])
                class_str = "Exterior" if row['dssp_class'] == 1 else "Interior"
                dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) and row['dssp_asa'] > 0 else "---"
                nc6 = int(row[nc6_col]) if pd.notna(row[nc6_col]) else 0
                nc10 = int(row[nc10_col]) if pd.notna(row[nc10_col]) else 0
                uni6 = f"{row[uni6_col]:.3f}" if pd.notna(row[uni6_col]) and row[uni6_col] > 0 else "---"
                uni10 = f"{row[uni10_col]:.3f}" if pd.notna(row[uni10_col]) and row[uni10_col] > 0 else "---"
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
                res_id = row['resname']
                pdb_num = int(row['resseq'])
                dssp_str = "Exterior" if row['dssp_class'] == 1 else "Interior"
                ncps_str = "Exterior" if row[ncps_col] == 1 else "Interior"
                dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) and row['dssp_asa'] > 0 else "---"
                nc6 = int(row[nc6_col]) if pd.notna(row[nc6_col]) else 0
                nc10 = int(row[nc10_col]) if pd.notna(row[nc10_col]) else 0
                uni6 = f"{row[uni6_col]:.3f}" if pd.notna(row[uni6_col]) and row[uni6_col] > 0 else "---"
                uni10 = f"{row[uni10_col]:.3f}" if pd.notna(row[uni10_col]) and row[uni10_col] > 0 else "---"
                f.write(f"{res_num:6d}  {res_id:3s}  {pdb_num:4d} | {dssp_str:>8s}  {ncps_str:>8s} | {dssp_asa:>9s}  {nc6:4d}  {nc10:5d}  {uni6:>8s}  {uni10:>8s}\n")

        f.write("\n")

    # STRIDE Agreement/Disagreement
    stride_mask = (df['stride_class'] != 0)
    if stride_mask.sum() > 0:
        stride_agree = df.loc[stride_mask & (df['stride_class'] == df[ncps_col])]
        stride_disagree = df.loc[stride_mask & (df['stride_class'] != df[ncps_col])]

        f.write("=" * 100 + "\n")
        f.write(f"RESIDUE LIST IN AGREEMENT: NCPS-STRIDE\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total: {len(stride_agree)} residues agree\n\n")

        if len(stride_agree) > 0:
            f.write("  Res#   ID   Num |  Class |  STRIDE ASA  NCPS NC6  NCPS NC10  NCPS Uni6  NCPS Uni10\n")
            f.write("-" * 90 + "\n")
            for idx, row in stride_agree.iterrows():
                res_num = idx + 1
                res_id = row['resname']
                pdb_num = int(row['resseq'])
                class_str = "Exterior" if row['stride_class'] == 1 else "Interior"
                stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) and row['stride_asa'] > 0 else "---"
                nc6 = int(row[nc6_col]) if pd.notna(row[nc6_col]) else 0
                nc10 = int(row[nc10_col]) if pd.notna(row[nc10_col]) else 0
                uni6 = f"{row[uni6_col]:.3f}" if pd.notna(row[uni6_col]) and row[uni6_col] > 0 else "---"
                uni10 = f"{row[uni10_col]:.3f}" if pd.notna(row[uni10_col]) and row[uni10_col] > 0 else "---"
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
                res_id = row['resname']
                pdb_num = int(row['resseq'])
                stride_str = "Exterior" if row['stride_class'] == 1 else "Interior"
                ncps_str = "Exterior" if row[ncps_col] == 1 else "Interior"
                stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) and row['stride_asa'] > 0 else "---"
                nc6 = int(row[nc6_col]) if pd.notna(row[nc6_col]) else 0
                nc10 = int(row[nc10_col]) if pd.notna(row[nc10_col]) else 0
                uni6 = f"{row[uni6_col]:.3f}" if pd.notna(row[uni6_col]) and row[uni6_col] > 0 else "---"
                uni10 = f"{row[uni10_col]:.3f}" if pd.notna(row[uni10_col]) and row[uni10_col] > 0 else "---"
                f.write(f"{res_num:6d}  {res_id:3s}  {pdb_num:4d} | {stride_str:>8s}  {ncps_str:>8s} | {stride_asa:>10s}  {nc6:4d}  {nc10:5d}  {uni6:>8s}  {uni10:>8s}\n")

        f.write("\n")


def write_legend(f):
    """Write legend explaining the report columns and parameters"""

    f.write("=" * 100 + "\n")
    f.write("LEGEND:\n")
    f.write("-" * 100 + "\n")
    f.write("Res #     : Sequential residue number\n")
    f.write("ID        : Residue amino acid code (ALA, GLN, etc.)\n")
    f.write("Num       : Residue number from PDB file\n")
    f.write("DSSP ASA  : DSSP accessible surface area (Ų)\n")
    f.write("DSSP Class: DSSP classification (1=exterior ≥25%, 0=interior <25%)\n")
    f.write("DSSP SS   : DSSP secondary structure (H=helix, E=strand, C=coil, etc.)\n")
    f.write("STRIDE ASA: STRIDE accessible surface area (Ų)\n")
    f.write("STRIDE Class: STRIDE classification (1=exterior ≥20%, 0=interior <20%)\n")
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

    f.write("=" * 100 + "\n")


if __name__ == "__main__":
    import sys
    success = generate_aa2ar_report_from_csv()
    sys.exit(0 if success else 1)


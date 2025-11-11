#!/usr/bin/env python3
"""
Generate Detailed Reports for Each Protein
Creates formatted tables with all parameters for each PDB file
Includes complete confusion matrices and agreement/disagreement lists
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
            dssp_exterior = int(df['dssp_class'].sum())
            dssp_interior = int(len(df[df['dssp_class'].notna()]) - dssp_exterior)
            f.write(f"DSSP Classification:\n")
            f.write(f"  - Exterior (1): {dssp_exterior} residues\n")
            f.write(f"  - Interior (0): {dssp_interior} residues\n")
            f.write(f"  - DSSP Cutoff Value: ASA ≥ 25% (relative accessible surface area)\n")
            f.write(f"    (If ASA ≥ 25%, classified as Exterior=1; otherwise Interior=0)\n\n")

        # STRIDE statistics
        if df['stride_class'].notna().sum() > 0:
            stride_exterior = int(df['stride_class'].sum())
            stride_interior = int(len(df[df['stride_class'].notna()]) - stride_exterior)
            f.write(f"STRIDE Classification:\n")
            f.write(f"  - Exterior (1): {stride_exterior} residues\n")
            f.write(f"  - Interior (0): {stride_interior} residues\n")
            f.write(f"  - STRIDE Cutoff Value: ASA ≥ 20% (relative accessible surface area)\n")
            f.write(f"    (If ASA ≥ 20%, classified as Exterior=1; otherwise Interior=0)\n\n")

        # Our method statistics
        ncps_exterior = int(df['ncps_class'].sum())
        ncps_interior = int(len(df) - ncps_exterior)
        f.write(f"NCPS Classification (Our Method):\n")
        f.write(f"  - Exterior (1): {ncps_exterior} residues\n")
        f.write(f"  - Interior (0): {ncps_interior} residues\n\n")

        # Agreement with DSSP
        if df['dssp_class'].notna().sum() > 0:
            df_dssp = df[df['dssp_class'].notna()].copy()
            agreement_dssp = (df_dssp['dssp_class'] == df_dssp['ncps_class']).sum()
            total_with_dssp = len(df_dssp)
            accuracy_dssp = (agreement_dssp / total_with_dssp) * 100
            f.write(f"Agreement with DSSP: {accuracy_dssp:.1f}% ({agreement_dssp}/{total_with_dssp})\n")

        # Agreement with STRIDE
        if df['stride_class'].notna().sum() > 0:
            df_stride = df[df['stride_class'].notna()].copy()
            agreement_stride = (df_stride['stride_class'] == df_stride['ncps_class']).sum()
            total_with_stride = len(df_stride)
            accuracy_stride = (agreement_stride / total_with_stride) * 100
            f.write(f"Agreement with STRIDE: {accuracy_stride:.1f}% ({agreement_stride}/{total_with_stride})\n\n")
        else:
            f.write("\n")

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

        f.write("=" * 120 + "\n\n")

        # ==================== STATISTICS SECTION ====================
        f.write("STATISTICS\n")
        f.write("=" * 120 + "\n\n")

        # DSSP Statistics
        if df['dssp_class'].notna().sum() > 0:
            df_dssp = df[df['dssp_class'].notna()].copy()
            y_true_dssp = df_dssp['dssp_class'].astype(int)
            y_pred_dssp = df_dssp['ncps_class'].astype(int)

            f.write("ACCORDING TO DSSP (Ground Truth = DSSP Classifications):\n")
            f.write("=" * 120 + "\n\n")

            # Confusion Matrix
            cm_dssp = confusion_matrix(y_true_dssp, y_pred_dssp)
            tn, fp, fn, tp = cm_dssp.ravel()

            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'':20} | {'Predicted Interior (0)':>20} | {'Predicted Exterior (1)':>20} |\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'True Interior (0)':20} | {tn:>20} | {fp:>20} | {tn+fp:>10}\n")
            f.write(f"{'True Exterior (1)':20} | {fn:>20} | {tp:>20} | {fn+tp:>10}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Total':20} | {tn+fn:>20} | {fp+tp:>20} | {tn+fp+fn+tp:>10}\n")
            f.write("-" * 60 + "\n\n")

            f.write("CONFUSION MATRIX INTERPRETATION:\n")
            f.write(f"  - True Negatives (TN):  {tn:4d} - Correctly predicted as Interior (both DSSP and NCPS agree on Interior)\n")
            f.write(f"  - False Positives (FP): {fp:4d} - Incorrectly predicted as Exterior (DSSP=Interior, NCPS=Exterior)\n")
            f.write(f"  - False Negatives (FN): {fn:4d} - Incorrectly predicted as Interior (DSSP=Exterior, NCPS=Interior)\n")
            f.write(f"  - True Positives (TP):  {tp:4d} - Correctly predicted as Exterior (both DSSP and NCPS agree on Exterior)\n\n")

            # Metrics
            acc_dssp = accuracy_score(y_true_dssp, y_pred_dssp)
            prec_dssp = precision_score(y_true_dssp, y_pred_dssp, average='weighted', zero_division=0)
            rec_dssp = recall_score(y_true_dssp, y_pred_dssp, average='weighted', zero_division=0)
            f1_dssp = f1_score(y_true_dssp, y_pred_dssp, average='weighted', zero_division=0)

            # Per-class metrics
            prec_per_class = precision_score(y_true_dssp, y_pred_dssp, average=None, zero_division=0)
            rec_per_class = recall_score(y_true_dssp, y_pred_dssp, average=None, zero_division=0)
            f1_per_class = f1_score(y_true_dssp, y_pred_dssp, average=None, zero_division=0)

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Overall Accuracy:              {acc_dssp:6.2%} ({int(acc_dssp*len(y_true_dssp))}/{len(y_true_dssp)})\n")
            f.write(f"  Weighted Precision:            {prec_dssp:6.2%}\n")
            f.write(f"  Weighted Recall:               {rec_dssp:6.2%}\n")
            f.write(f"  Weighted F1-Score:             {f1_dssp:6.2%}\n\n")

            f.write("PER-CLASS METRICS:\n")
            f.write(f"  Interior (0) - Precision:      {prec_per_class[0]:6.2%}\n")
            f.write(f"  Interior (0) - Recall:         {rec_per_class[0]:6.2%}\n")
            f.write(f"  Interior (0) - F1-Score:       {f1_per_class[0]:6.2%}\n\n")
            f.write(f"  Exterior (1) - Precision:      {prec_per_class[1]:6.2%}\n")
            f.write(f"  Exterior (1) - Recall:         {rec_per_class[1]:6.2%}\n")
            f.write(f"  Exterior (1) - F1-Score:       {f1_per_class[1]:6.2%}\n\n")

            f.write("=" * 120 + "\n\n")

        # STRIDE Statistics
        if df['stride_class'].notna().sum() > 0:
            df_stride = df[df['stride_class'].notna()].copy()
            y_true_stride = df_stride['stride_class'].astype(int)
            y_pred_stride = df_stride['ncps_class'].astype(int)

            f.write("ACCORDING TO STRIDE (Ground Truth = STRIDE Classifications):\n")
            f.write("=" * 120 + "\n\n")

            # Confusion Matrix
            cm_stride = confusion_matrix(y_true_stride, y_pred_stride)
            tn, fp, fn, tp = cm_stride.ravel()

            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'':20} | {'Predicted Interior (0)':>20} | {'Predicted Exterior (1)':>20} |\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'True Interior (0)':20} | {tn:>20} | {fp:>20} | {tn+fp:>10}\n")
            f.write(f"{'True Exterior (1)':20} | {fn:>20} | {tp:>20} | {fn+tp:>10}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Total':20} | {tn+fn:>20} | {fp+tp:>20} | {tn+fp+fn+tp:>10}\n")
            f.write("-" * 60 + "\n\n")

            f.write("CONFUSION MATRIX INTERPRETATION:\n")
            f.write(f"  - True Negatives (TN):  {tn:4d} - Correctly predicted as Interior (both STRIDE and NCPS agree on Interior)\n")
            f.write(f"  - False Positives (FP): {fp:4d} - Incorrectly predicted as Exterior (STRIDE=Interior, NCPS=Exterior)\n")
            f.write(f"  - False Negatives (FN): {fn:4d} - Incorrectly predicted as Interior (STRIDE=Exterior, NCPS=Interior)\n")
            f.write(f"  - True Positives (TP):  {tp:4d} - Correctly predicted as Exterior (both STRIDE and NCPS agree on Exterior)\n\n")

            # Metrics
            acc_stride = accuracy_score(y_true_stride, y_pred_stride)
            prec_stride = precision_score(y_true_stride, y_pred_stride, average='weighted', zero_division=0)
            rec_stride = recall_score(y_true_stride, y_pred_stride, average='weighted', zero_division=0)
            f1_stride = f1_score(y_true_stride, y_pred_stride, average='weighted', zero_division=0)

            # Per-class metrics
            prec_per_class = precision_score(y_true_stride, y_pred_stride, average=None, zero_division=0)
            rec_per_class = recall_score(y_true_stride, y_pred_stride, average=None, zero_division=0)
            f1_per_class = f1_score(y_true_stride, y_pred_stride, average=None, zero_division=0)

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Overall Accuracy:              {acc_stride:6.2%} ({int(acc_stride*len(y_true_stride))}/{len(y_true_stride)})\n")
            f.write(f"  Weighted Precision:            {prec_stride:6.2%}\n")
            f.write(f"  Weighted Recall:               {rec_stride:6.2%}\n")
            f.write(f"  Weighted F1-Score:             {f1_stride:6.2%}\n\n")

            f.write("PER-CLASS METRICS:\n")
            f.write(f"  Interior (0) - Precision:      {prec_per_class[0]:6.2%}\n")
            f.write(f"  Interior (0) - Recall:         {rec_per_class[0]:6.2%}\n")
            f.write(f"  Interior (0) - F1-Score:       {f1_per_class[0]:6.2%}\n\n")
            f.write(f"  Exterior (1) - Precision:      {prec_per_class[1]:6.2%}\n")
            f.write(f"  Exterior (1) - Recall:         {rec_per_class[1]:6.2%}\n")
            f.write(f"  Exterior (1) - F1-Score:       {f1_per_class[1]:6.2%}\n\n")

            f.write("=" * 120 + "\n\n")

        # ==================== AGREEMENT/DISAGREEMENT LISTS ====================

        # DSSP Agreement/Disagreement
        if df['dssp_class'].notna().sum() > 0:
            df_dssp = df[df['dssp_class'].notna()].copy()

            # Agreement
            df_agree_dssp = df_dssp[df_dssp['dssp_class'] == df_dssp['ncps_class']].copy()
            f.write("RESIDUE LIST IN AGREEMENT: NCPS-DSSP\n")
            f.write("=" * 120 + "\n")
            f.write(f"Total: {len(df_agree_dssp)} residues agree\n\n")

            if len(df_agree_dssp) > 0:
                f.write(f"{'Res#':>5} {'ID':>4} {'Num':>5} | {'Class':>6} | {'DSSP ASA':>9} {'NCPS NC6':>9} {'NCPS NC10':>10} {'NCPS Uni6':>10} {'NCPS Uni10':>11}\n")
                f.write("-" * 120 + "\n")
                for idx, row in df_agree_dssp.iterrows():
                    res_idx = df.index.get_loc(idx) + 1
                    class_str = "Interior" if row['dssp_class'] == 0 else "Exterior"
                    f.write(f"{res_idx:>5} {row['res_id']:>4} {int(row['res_num']):>5} | {class_str:>8} | "
                           f"{row['dssp_asa']:>8.1f} {int(row['ncps_sphere_6']):>9} {int(row['ncps_sphere_10']):>10} "
                           f"{row['ncps_sphere_6_uni']:>10.3f} {row['ncps_sphere_10_uni']:>11.3f}\n")
            f.write("\n" + "=" * 120 + "\n\n")

            # Disagreement
            df_disagree_dssp = df_dssp[df_dssp['dssp_class'] != df_dssp['ncps_class']].copy()
            f.write("RESIDUE LIST IN DISAGREEMENT: NCPS-DSSP\n")
            f.write("=" * 120 + "\n")
            f.write(f"Total: {len(df_disagree_dssp)} residues disagree\n\n")

            if len(df_disagree_dssp) > 0:
                f.write(f"{'Res#':>5} {'ID':>4} {'Num':>5} | {'DSSP':>8} {'NCPS':>8} | {'DSSP ASA':>9} {'NCPS NC6':>9} {'NCPS NC10':>10} {'NCPS Uni6':>10} {'NCPS Uni10':>11}\n")
                f.write("-" * 120 + "\n")
                for idx, row in df_disagree_dssp.iterrows():
                    res_idx = df.index.get_loc(idx) + 1
                    dssp_str = "Interior" if row['dssp_class'] == 0 else "Exterior"
                    ncps_str = "Interior" if row['ncps_class'] == 0 else "Exterior"
                    f.write(f"{res_idx:>5} {row['res_id']:>4} {int(row['res_num']):>5} | {dssp_str:>8} {ncps_str:>8} | "
                           f"{row['dssp_asa']:>8.1f} {int(row['ncps_sphere_6']):>9} {int(row['ncps_sphere_10']):>10} "
                           f"{row['ncps_sphere_6_uni']:>10.3f} {row['ncps_sphere_10_uni']:>11.3f}\n")
            f.write("\n" + "=" * 120 + "\n\n")

        # STRIDE Agreement/Disagreement
        if df['stride_class'].notna().sum() > 0:
            df_stride = df[df['stride_class'].notna()].copy()

            # Agreement
            df_agree_stride = df_stride[df_stride['stride_class'] == df_stride['ncps_class']].copy()
            f.write("RESIDUE LIST IN AGREEMENT: NCPS-STRIDE\n")
            f.write("=" * 120 + "\n")
            f.write(f"Total: {len(df_agree_stride)} residues agree\n\n")

            if len(df_agree_stride) > 0:
                f.write(f"{'Res#':>5} {'ID':>4} {'Num':>5} | {'Class':>6} | {'STRIDE ASA':>11} {'NCPS NC6':>9} {'NCPS NC10':>10} {'NCPS Uni6':>10} {'NCPS Uni10':>11}\n")
                f.write("-" * 120 + "\n")
                for idx, row in df_agree_stride.iterrows():
                    res_idx = df.index.get_loc(idx) + 1
                    class_str = "Interior" if row['stride_class'] == 0 else "Exterior"
                    f.write(f"{res_idx:>5} {row['res_id']:>4} {int(row['res_num']):>5} | {class_str:>8} | "
                           f"{row['stride_asa']:>10.1f} {int(row['ncps_sphere_6']):>9} {int(row['ncps_sphere_10']):>10} "
                           f"{row['ncps_sphere_6_uni']:>10.3f} {row['ncps_sphere_10_uni']:>11.3f}\n")
            f.write("\n" + "=" * 120 + "\n\n")

            # Disagreement
            df_disagree_stride = df_stride[df_stride['stride_class'] != df_stride['ncps_class']].copy()
            f.write("RESIDUE LIST IN DISAGREEMENT: NCPS-STRIDE\n")
            f.write("=" * 120 + "\n")
            f.write(f"Total: {len(df_disagree_stride)} residues disagree\n\n")

            if len(df_disagree_stride) > 0:
                f.write(f"{'Res#':>5} {'ID':>4} {'Num':>5} | {'STRIDE':>8} {'NCPS':>8} | {'STRIDE ASA':>11} {'NCPS NC6':>9} {'NCPS NC10':>10} {'NCPS Uni6':>10} {'NCPS Uni10':>11}\n")
                f.write("-" * 120 + "\n")
                for idx, row in df_disagree_stride.iterrows():
                    res_idx = df.index.get_loc(idx) + 1
                    stride_str = "Interior" if row['stride_class'] == 0 else "Exterior"
                    ncps_str = "Interior" if row['ncps_class'] == 0 else "Exterior"
                    f.write(f"{res_idx:>5} {row['res_id']:>4} {int(row['res_num']):>5} | {stride_str:>8} {ncps_str:>8} | "
                           f"{row['stride_asa']:>10.1f} {int(row['ncps_sphere_6']):>9} {int(row['ncps_sphere_10']):>10} "
                           f"{row['ncps_sphere_6_uni']:>10.3f} {row['ncps_sphere_10_uni']:>11.3f}\n")
            f.write("\n" + "=" * 120 + "\n\n")

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


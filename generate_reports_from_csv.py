#!/usr/bin/env python3
"""
Simple report generator from pre-computed CSV files.
Reads the CSV files with DSSP and STRIDE data and generates detailed_report.txt files.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def write_report(df, protein_id, output_path, params):
    """Write detailed report from DataFrame."""

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("PROTEIN BURIAL ANALYSIS - DETAILED REPORT\n")
        f.write(f"PDB ID: {protein_id.upper()}\n")
        f.write("=" * 100 + "\n\n")

        # SUMMARY STATISTICS
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
            f.write(f"  - DSSP Cutoff Value: ASA ≥ 25.0%\n\n")
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
            f.write(f"  - STRIDE Cutoff Value: ASA ≥ 20.0%\n\n")
        else:
            f.write("STRIDE Classification:\n")
            f.write("  - No STRIDE data available for this protein\n\n")

        # NCPS Classification
        ncps_ext = (df['ncps_class'] == 1).sum()
        ncps_int = (df['ncps_class'] == 0).sum()
        f.write("NCPS Classification (Our Method):\n")
        f.write(f"  - Exterior (1): {ncps_ext} residues\n")
        f.write(f"  - Interior (0): {ncps_int} residues\n\n")

        # Agreements
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

        # Neighbor statistics
        f.write("Neighbor Count Statistics:\n")
        f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6'].mean():.1f}, Median={df['ncps_sphere_6'].median():.0f}\n")
        f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10'].mean():.1f}, Median={df['ncps_sphere_10'].median():.0f}\n\n")

        f.write("Uniformity Statistics:\n")
        f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6_uni'].mean():.2f}, Median={df['ncps_sphere_6_uni'].median():.2f}\n")
        f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10_uni'].mean():.2f}, Median={df['ncps_sphere_10_uni'].median():.2f}\n\n")

        f.write("=" * 100 + "\n\n")

        # DETAILED RESIDUE DATA
        f.write("DETAILED RESIDUE DATA\n")
        f.write("=" * 100 + "\n\n")
        f.write(" Res   ID   Num |     DSSP   DSSP DSSP |   STRIDE STRIDE STRIDE |  NC6   Uni6  NC10  Uni10 |  NCPS\n")
        f.write("   #            |      ASA  Class   SS |      ASA  Class   SS |                          | Class\n")
        f.write("-" * 100 + "\n")

        for idx, row in df.iterrows():
            res_num = idx + 1
            res_id = row['resname'] if pd.notna(row['resname']) else "---"
            pdb_num = int(row['resseq']) if pd.notna(row['resseq']) else idx + 1

            dssp_asa = f"{row['dssp_asa']:.1f}" if pd.notna(row['dssp_asa']) else "---"
            dssp_class = int(row['dssp_class']) if pd.notna(row['dssp_class']) else -1
            dssp_class_str = str(dssp_class) if dssp_class >= 0 else "-"
            dssp_ss = row['dssp_ss'] if pd.notna(row['dssp_ss']) and row['dssp_ss'] else "-"

            stride_asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else "---"
            stride_class = int(row['stride_class']) if pd.notna(row['stride_class']) else -1
            stride_class_str = str(stride_class) if stride_class >= 0 else "-"
            stride_ss = row['stride_ss'] if pd.notna(row['stride_ss']) and row['stride_ss'] else "-"

            nc6 = int(row['ncps_sphere_6']) if pd.notna(row['ncps_sphere_6']) else 0
            uni6 = f"{row['ncps_sphere_6_uni']:.3f}" if pd.notna(row['ncps_sphere_6_uni']) else "---"
            nc10 = int(row['ncps_sphere_10']) if pd.notna(row['ncps_sphere_10']) else 0
            uni10 = f"{row['ncps_sphere_10_uni']:.3f}" if pd.notna(row['ncps_sphere_10_uni']) else "---"

            ncps_class = int(row['ncps_class']) if pd.notna(row['ncps_class']) else -1

            f.write(f"{res_num:4d}  {res_id:3s}  {pdb_num:4d} | {dssp_asa:>6s}  {dssp_class_str:>2s}  {dssp_ss:>2s} | {stride_asa:>7s}  {stride_class_str:>2s}  {stride_ss:>2s} | {nc6:4d}  {uni6:>6s}  {nc10:4d}  {uni10:>6s} | {ncps_class:4d}\n")

        f.write("-" * 100 + "\n\n")

        # STATISTICS
        f.write("STATISTICS\n")
        f.write("=" * 100 + "\n\n")

        if dssp_mask.sum() > 0:
            f.write("ACCORDING TO DSSP:\n")
            f.write("=" * 100 + "\n\n")
            y_true = df.loc[dssp_mask, 'dssp_class'].values.astype(int)
            y_pred = df.loc[dssp_mask, 'ncps_class'].values.astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            f.write(f"Confusion Matrix:\n")
            f.write(f"  True Negatives (TN):  {cm[0,0]}\n")
            f.write(f"  False Positives (FP): {cm[0,1]}\n")
            f.write(f"  False Negatives (FN): {cm[1,0]}\n")
            f.write(f"  True Positives (TP):  {cm[1,1]}\n\n")
            f.write(f"Accuracy:  {acc:.3f}\n")
            f.write(f"Precision: {prec:.3f}\n")
            f.write(f"Recall:    {rec:.3f}\n")
            f.write(f"F1-Score:  {f1:.3f}\n\n")

        if stride_mask.sum() > 0:
            f.write("ACCORDING TO STRIDE:\n")
            f.write("=" * 100 + "\n\n")
            y_true = df.loc[stride_mask, 'stride_class'].values.astype(int)
            y_pred = df.loc[stride_mask, 'ncps_class'].values.astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            f.write(f"Confusion Matrix:\n")
            f.write(f"  True Negatives (TN):  {cm[0,0]}\n")
            f.write(f"  False Positives (FP): {cm[0,1]}\n")
            f.write(f"  False Negatives (FN): {cm[1,0]}\n")
            f.write(f"  True Positives (TP):  {cm[1,1]}\n\n")
            f.write(f"Accuracy:  {acc:.3f}\n")
            f.write(f"Precision: {prec:.3f}\n")
            f.write(f"Recall:    {rec:.3f}\n")
            f.write(f"F1-Score:  {f1:.3f}\n\n")

def main():
    csv_dir = Path("/holder/results_dude/detailed_reports")

    print("\nGenerating detailed_report.txt files from CSVs...\n")

    params = type('params', (), {'dssp_asa_cutoff': 25.0, 'stride_asa_cutoff': 20.0})()

    count = 0
    for csv_file in sorted(csv_dir.glob("*_detailed_results.csv")):
        protein_id = csv_file.stem.replace("_detailed_results", "")
        output_file = csv_dir / f"{protein_id}_detailed_report.txt"

        try:
            df = pd.read_csv(csv_file)
            write_report(df, protein_id, output_file, params)
            print(f"✓ {protein_id:10s} - report generated")
            count += 1
        except Exception as e:
            print(f"✗ {protein_id:10s} - {str(e)[:40]}")

    print(f"\n{count} reports generated!\n")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Generate DUDE model-development CSVs and detailed text reports.

This script builds on the outputs in `results/dude_reports` produced by
`generate_dude_reports.py`.

For each DUDE receptor (e.g. `aa2ar`), it will:

1. Read `<protein_id>_detailed_results.csv` from `results/dude_reports/`.
2. Create a 3PTE-style model-development CSV:
   - `<protein_id>_model_development.csv`
3. Create a 3PTE-style detailed text report:
   - `<protein_id>_detailed_report.txt`

Additionally, it will create:

- `dude_combined_model_development.csv`: concatenation of all
  per-protein model-development CSVs.
- `dude_all_detailed_reports_summary.txt`: textual summary across all
  DUDE receptors, including aggregate confusion matrices and metrics vs
  DSSP and vs STRIDE.

The confusion matrix convention follows the standard:
- Real positive (P): reference class == 1 (surface)
- Real negative (N): reference class == 0 (interior)
- Predicted positive: ncps_class == 1
- Predicted negative: ncps_class == 0

Thus:
- TP: real=1, pred=1
- FP: real=0, pred=1
- FN: real=1, pred=0
- TN: real=0, pred=0

and derived metrics (accuracy, precision, recall, F1, balanced
accuracy) are computed accordingly.
"""

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


RESULTS_DIR = Path("holder/results/dude_reports")


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute TP/FP/FN/TN and common metrics from binary labels.

    Labels: 0 = interior (negative), 1 = surface (positive).
    """
    if y_true.size == 0:
        return {}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = tp + tn + fp + fn

    # Convert to float for divisions, guard against zero denominators
    tp_f, fp_f, fn_f, tn_f = map(float, (tp, fp, fn, tn))
    total_f = float(total) if total > 0 else 1.0

    # Basic rates
    tpr = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0  # recall, sensitivity
    tnr = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0  # specificity
    fpr = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
    fnr = fn_f / (fn_f + tp_f) if (fn_f + tp_f) > 0 else 0.0

    # Predictive values
    ppv = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0  # precision
    npv = tn_f / (tn_f + fn_f) if (tn_f + fn_f) > 0 else 0.0

    # Accuracy & F1
    acc = (tp_f + tn_f) / total_f
    f1 = (2 * ppv * tpr / (ppv + tpr)) if (ppv + tpr) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "accuracy": acc,
        "precision": ppv,
        "recall": tpr,
        "specificity": tnr,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "npv": npv,
        "support": float(total),
    }


def build_model_development_csv(df: pd.DataFrame, protein_id: str, out_path: Path) -> pd.DataFrame:
    """Build a 3PTE-style model-development CSV for a single protein.

    We mirror `results/3pte_model_development.csv` layout and add DSSP/STRIDE
    class columns at the end for convenience.
    """
    dev_df = pd.DataFrame({
        "protein": protein_id,
        "res_id": df.get("resname", df.get("res_id")),
        "res_num": df.get("res_num", df.get("resseq")),
        "ncps_sphere_6": df["ncps_sphere_6"],
        "ncps_sphere_6_uni": df["ncps_sphere_6_uni"],
        "ncps_sphere_10": df["ncps_sphere_10"],
        "ncps_sphere_10_uni": df["ncps_sphere_10_uni"],
        "ncps_class": df["ncps_class"],
    })

    # Optionally include reference classes if present
    if "dssp_class" in df.columns:
        dev_df["dssp_class"] = df["dssp_class"]
    if "stride_class" in df.columns:
        dev_df["stride_class"] = df["stride_class"]

    dev_df.to_csv(out_path, index=False)
    return dev_df


def write_detailed_report(df: pd.DataFrame, protein_id: str, out_path: Path) -> Dict[str, Dict[str, float]]:
    """Write a 3PTE-style detailed report text file for one protein.

    Returns a dict with per-reference metrics (keys: 'dssp', 'stride') for
    aggregation at the summary level.
    """
    n_residues = len(df)

    # DSSP stats
    dssp_metrics = None
    if "dssp_class" in df.columns and df["dssp_class"].notna().any():
        mask = df["dssp_class"].notna()
        y_true = df.loc[mask, "dssp_class"].astype(int).to_numpy()
        y_pred = df.loc[mask, "ncps_class"].astype(int).to_numpy()
        dssp_metrics = compute_confusion_metrics(y_true, y_pred)

    # STRIDE stats
    stride_metrics = None
    if "stride_class" in df.columns and df["stride_class"].notna().any():
        mask = df["stride_class"].notna()
        y_true = df.loc[mask, "stride_class"].astype(int).to_numpy()
        y_pred = df.loc[mask, "ncps_class"].astype(int).to_numpy()
        stride_metrics = compute_confusion_metrics(y_true, y_pred)

    # Basic counts
    dssp_counts = df["dssp_class"].value_counts(dropna=True).to_dict() if "dssp_class" in df.columns else {}
    stride_counts = df["stride_class"].value_counts(dropna=True).to_dict() if "stride_class" in df.columns else {}
    ncps_counts = df["ncps_class"].value_counts(dropna=True).to_dict()

    # Neighbor/uniformity stats
    def stat_line(series: pd.Series) -> str:
        if series.empty:
            return "Mean=NA, Median=NA, Range=[NA-NA]"
        return f"Mean={series.mean():.2f}, Median={series.median():.2f}, Range=[{series.min():.0f}-{series.max():.0f}]"

    nc6_stats = stat_line(df["ncps_sphere_6"])
    nc10_stats = stat_line(df["ncps_sphere_10"])
    uni6_stats = stat_line(df["ncps_sphere_6_uni"])
    uni10_stats = stat_line(df["ncps_sphere_10_uni"])

    with out_path.open("w") as f:
        f.write("=" * 120 + "\n")
        f.write("PROTEIN BURIAL ANALYSIS - DETAILED REPORT\n")
        f.write(f"Receptor ID: {protein_id}\n")
        f.write("=" * 120 + "\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 120 + "\n")
        f.write(f"Total Residues: {n_residues}\n\n")

        # DSSP summary
        if dssp_metrics is not None:
            ext = int(dssp_counts.get(1, 0))
            inte = int(dssp_counts.get(0, 0))
            f.write("DSSP Classification (ASA ≥ 30 Å² → Exterior=1):\n")
            f.write(f"  - Exterior (1): {ext} residues\n")
            f.write(f"  - Interior (0): {inte} residues\n")
            f.write("  Metrics vs DSSP (NCPS vs DSSP):\n")
            f.write(f"    Accuracy: {dssp_metrics['accuracy']:.3f}\n")
            f.write(f"    Precision (PPV): {dssp_metrics['precision']:.3f}\n")
            f.write(f"    Recall (TPR): {dssp_metrics['recall']:.3f}\n")
            f.write(f"    Specificity (TNR): {dssp_metrics['specificity']:.3f}\n")
            f.write(f"    F1-Score: {dssp_metrics['f1']:.3f}\n")
            f.write(f"    Balanced Accuracy: {dssp_metrics['balanced_accuracy']:.3f}\n")
            f.write("    Confusion Matrix (DSSP as truth):\n")
            f.write(f"      TP: {int(dssp_metrics['tp'])}, FP: {int(dssp_metrics['fp'])}, "
                    f"FN: {int(dssp_metrics['fn'])}, TN: {int(dssp_metrics['tn'])}\n\n")
        else:
            f.write("DSSP Classification: no valid DSSP labels for this receptor.\n\n")

        # STRIDE summary
        if stride_metrics is not None:
            ext = int(stride_counts.get(1, 0))
            inte = int(stride_counts.get(0, 0))
            f.write("STRIDE Classification (ASA ≥ 24 Å² → Exterior=1):\n")
            f.write(f"  - Exterior (1): {ext} residues\n")
            f.write(f"  - Interior (0): {inte} residues\n")
            f.write("  Metrics vs STRIDE (NCPS vs STRIDE):\n")
            f.write(f"    Accuracy: {stride_metrics['accuracy']:.3f}\n")
            f.write(f"    Precision (PPV): {stride_metrics['precision']:.3f}\n")
            f.write(f"    Recall (TPR): {stride_metrics['recall']:.3f}\n")
            f.write(f"    Specificity (TNR): {stride_metrics['specificity']:.3f}\n")
            f.write(f"    F1-Score: {stride_metrics['f1']:.3f}\n")
            f.write(f"    Balanced Accuracy: {stride_metrics['balanced_accuracy']:.3f}\n")
            f.write("    Confusion Matrix (STRIDE as truth):\n")
            f.write(f"      TP: {int(stride_metrics['tp'])}, FP: {int(stride_metrics['fp'])}, "
                    f"FN: {int(stride_metrics['fn'])}, TN: {int(stride_metrics['tn'])}\n\n")
        else:
            f.write("STRIDE Classification: no valid STRIDE labels for this receptor.\n\n")

        # NCPS summary
        f.write("NCPS Classification (Our Method):\n")
        f.write(f"  - Exterior (1): {int(ncps_counts.get(1, 0))} residues\n")
        f.write(f"  - Interior (0): {int(ncps_counts.get(0, 0))} residues\n\n")

        f.write("Neighbor Count Statistics:\n")
        f.write(f"  - 6Å Sphere: {nc6_stats}\n")
        f.write(f"  - 10Å Sphere: {nc10_stats}\n\n")
        f.write("Uniformity Statistics:\n")
        f.write(f"  - 6Å Sphere: {uni6_stats}\n")
        f.write(f"  - 10Å Sphere: {uni10_stats}\n\n")

        # Detailed table
        f.write("""========================================================================================================================

DETAILED RESIDUE DATA
========================================================================================================================

 Res   ID   Num |   DSSP    DSSP DSSP |  STRIDE  STRIDE STRIDE |  NC6   Uni6  NC10  Uni10 |  NCPS
   #            |    ASA   Class   SS |    ASA   Class   SS |                           | Class
------------------------------------------------------------------------------------------------------------------------
""")
        for idx, row in df.iterrows():
            resnum = row.get("res_num", row.get("resseq", ""))
            resname = row.get("resname", row.get("res_id", ""))
            d_asa = row.get("dssp_asa", np.nan)
            d_cls = row.get("dssp_class", np.nan)
            d_ss = row.get("dssp_ss", "-")
            s_asa = row.get("stride_asa", np.nan)
            s_cls = row.get("stride_class", np.nan)
            s_ss = row.get("stride_ss", "-")
            nc6 = row.get("ncps_sphere_6", np.nan)
            nc10 = row.get("ncps_sphere_10", np.nan)
            u6 = row.get("ncps_sphere_6_uni", np.nan)
            u10 = row.get("ncps_sphere_10_uni", np.nan)
            ncps = row.get("ncps_class", np.nan)

            def fmt(x, fmt_str="{:.1f}"):
                return fmt_str.format(x) if pd.notna(x) else "  NA"

            f.write(
                f"{idx+1:4d} {resname:>3} {int(resnum):5d} | "
                f"{fmt(d_asa, '{:6.1f}')} {int(d_cls) if pd.notna(d_cls) else -1:6d} {d_ss:>4} | "
                f"{fmt(s_asa, '{:6.1f}')} {int(s_cls) if pd.notna(s_cls) else -1:6d} {s_ss:>4} | "
                f"{fmt(nc6, '{:4.0f}')} {fmt(u6, '{:5.3f}')} {fmt(nc10, '{:4.0f}')} {fmt(u10, '{:5.3f}')} | "
                f"{int(ncps) if pd.notna(ncps) else -1:5d}\n"
            )

    metrics_bundle: Dict[str, Dict[str, float]] = {}
    if dssp_metrics is not None:
        metrics_bundle["dssp"] = dssp_metrics
    if stride_metrics is not None:
        metrics_bundle["stride"] = stride_metrics
    return metrics_bundle


def main() -> None:
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return

    detailed_files = sorted(p for p in RESULTS_DIR.glob("*_detailed_results.csv") if p.is_file())
    if not detailed_files:
        print(f"No *_detailed_results.csv files found in {RESULTS_DIR}")
        return

    print(f"Found {len(detailed_files)} DUDE detailed CSV files.")

    combined_dev_rows: List[pd.DataFrame] = []
    aggregate_metrics: Dict[str, Dict[str, float]] = {"dssp": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                                                      "stride": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}

    for csv_path in detailed_files:
        protein_id = csv_path.stem.replace("_detailed_results", "")
        print(f"Processing DUDE receptor for reports: {protein_id}")
        df = pd.read_csv(csv_path)

        # Per-protein model development CSV
        dev_out = RESULTS_DIR / f"{protein_id}_model_development.csv"
        dev_df = build_model_development_csv(df, protein_id, dev_out)
        combined_dev_rows.append(dev_df)

        # Per-protein detailed text report
        report_out = RESULTS_DIR / f"{protein_id}_detailed_report.txt"
        metrics_bundle = write_detailed_report(df, protein_id, report_out)

        # Accumulate confusion counts for aggregate metrics
        for ref in ("dssp", "stride"):
            if ref in metrics_bundle:
                for key in ("tp", "fp", "fn", "tn"):
                    aggregate_metrics[ref][key] += metrics_bundle[ref][key]

    # Combined model-development CSV
    if combined_dev_rows:
        combined_df = pd.concat(combined_dev_rows, ignore_index=True)
        combined_out = RESULTS_DIR / "dude_combined_model_development.csv"
        combined_df.to_csv(combined_out, index=False)
        print(f"Combined model-development CSV saved to: {combined_out}")

    # Aggregate summary report
    summary_out = RESULTS_DIR / "dude_all_detailed_reports_summary.txt"
    with summary_out.open("w") as f:
        f.write("=" * 120 + "\n")
        f.write("DUDE DATASET - AGGREGATED DETAILED REPORT SUMMARY\n")
        f.write("=" * 120 + "\n\n")

        def write_agg_block(ref: str, label: str) -> None:
            tp = aggregate_metrics[ref]["tp"]
            fp = aggregate_metrics[ref]["fp"]
            fn = aggregate_metrics[ref]["fn"]
            tn = aggregate_metrics[ref]["tn"]
            total = tp + fp + fn + tn
            if total == 0:
                f.write(f"No aggregate {label} metrics (no valid labels).\n\n")
                return
            y_true = np.array([0] * int(tn + fp) + [1] * int(fn + tp))
            y_pred = np.array([0] * int(tn) + [1] * int(fp) + [0] * int(fn) + [1] * int(tp))
            m = compute_confusion_metrics(y_true, y_pred)

            f.write(f"Aggregate metrics vs {label}:\n")
            f.write(f"  TP: {int(m['tp'])}, FP: {int(m['fp'])}, FN: {int(m['fn'])}, TN: {int(m['tn'])}\n")
            f.write(f"  Accuracy: {m['accuracy']:.3f}\n")
            f.write(f"  Precision (PPV): {m['precision']:.3f}\n")
            f.write(f"  Recall (TPR): {m['recall']:.3f}\n")
            f.write(f"  Specificity (TNR): {m['specificity']:.3f}\n")
            f.write(f"  F1-Score: {m['f1']:.3f}\n")
            f.write(f"  Balanced Accuracy: {m['balanced_accuracy']:.3f}\n\n")

        write_agg_block("dssp", "DSSP")
        write_agg_block("stride", "STRIDE")

    print(f"Aggregate summary report saved to: {summary_out}")


if __name__ == "__main__":
    main()


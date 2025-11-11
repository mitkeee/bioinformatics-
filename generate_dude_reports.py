#!/usr/bin/env python3
"""Generate per-protein DUDE CSV reports and a summary with confusion-matrix metrics.

- Scans DUDE directories for receptor PDB files.
- Runs the existing comprehensive burial analysis on each receptor.
- Writes one `*_detailed_results.csv` per protein.
- Writes `dude_summary_results.csv` with per-protein metrics vs DSSP and STRIDE.
- Optionally triggers visualization generation using `visualization_module.generate_all_visualizations`.
"""

from pathlib import Path
from typing import List, Dict
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

from comprehensive_burial_analysis import (
    BurialParameters,
    process_protein_dataset,
)

try:
    from visualization_module import generate_all_visualizations
except Exception:
    generate_all_visualizations = None


def find_receptor_pdbs(dude_roots: List[Path]) -> List[Path]:
    """Return all receptor PDB files under the given DUDE root directories.

    We treat any file named `receptor.pdb` as a DUDE system receptor.
    """
    receptors: List[Path] = []
    for root in dude_roots:
        if not root.exists():
            continue
        for path in root.rglob("receptor.pdb"):
            receptors.append(path)
    # de-duplicate and sort for stable ordering
    receptors = sorted({p.resolve() for p in receptors})
    return receptors


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute common binary classification metrics from labels 0/1.

    Labels follow the convention:
    - 0: interior
    - 1: surface (povrsina)
    """
    metrics: Dict = {}
    if y_true.size == 0:
        return metrics

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    metrics["support"] = int(cm.sum())
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def run_dude_on_receptors(receptors: List[Path], output_dir: Path, params: BurialParameters) -> None:
    """Run NCPI/comprehensive analysis on each receptor and write CSV + summary.

    This function uses `process_protein_dataset` but passes one PDB at a time so we
    get one `ProteinResults` and can easily write `*_detailed_results.csv`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    per_protein_rows: List[Dict] = []

    for idx, pdb_path in enumerate(receptors, 1):
        protein_id = pdb_path.parent.name  # DUDE system folder name
        print(f"[{idx}/{len(receptors)}] Processing DUDE receptor: {protein_id} ({pdb_path})")

        # Run analysis on this single protein
        results = process_protein_dataset([pdb_path], params)
        if not results:
            print(f"  Warning: no results produced for {protein_id}")
            continue

        protein_result = results[0]
        df = protein_result.dataframe

        # Save detailed per-residue CSV
        detailed_csv = output_dir / f"{protein_id}_detailed_results.csv"
        df.to_csv(detailed_csv, index=False)

        row: Dict = {
            "protein_id": protein_result.protein_id,
            "n_residues": int(protein_result.n_residues),
        }

        # Metrics vs DSSP
        if "dssp_class" in df.columns and df["dssp_class"].notna().any():
            mask = df["dssp_class"].notna()
            y_true = df.loc[mask, "dssp_class"].astype(int).to_numpy()
            y_pred = df.loc[mask, "ncps_class"].astype(int).to_numpy()
            m = compute_binary_metrics(y_true, y_pred)
            for k, v in m.items():
                row[f"dssp_{k}"] = v

        # Metrics vs STRIDE
        if "stride_class" in df.columns and df["stride_class"].notna().any():
            mask = df["stride_class"].notna()
            y_true = df.loc[mask, "stride_class"].astype(int).to_numpy()
            y_pred = df.loc[mask, "ncps_class"].astype(int).to_numpy()
            m = compute_binary_metrics(y_true, y_pred)
            for k, v in m.items():
                row[f"stride_{k}"] = v

        per_protein_rows.append(row)

    # Write summary CSV and JSON
    if per_protein_rows:
        summary_df = pd.DataFrame(per_protein_rows)
        summary_csv = output_dir / "dude_summary_results.csv"
        summary_df.to_csv(summary_csv, index=False)

        summary_json = output_dir / "dude_summary_results.json"
        summary_json.write_text(json.dumps(per_protein_rows, indent=2))

        print(f"\nSaved DUDE summary CSV to: {summary_csv}")
        print(f"Saved DUDE summary JSON to: {summary_json}")

        # Optionally trigger visualizations (will reuse detailed CSVs)
        if generate_all_visualizations is not None:
            try:
                generate_all_visualizations(output_dir)
            except Exception as exc:
                print(f"Warning: could not generate visualizations: {exc}")


def main() -> None:
    """Entry point: scan DUDE roots, run analysis, write CSVs & reports."""
    # These are the DUDE roots present in your workspace; adjust if needed.
    workspace = Path(__file__).resolve().parent
    dude_roots = [
        workspace / "dude_1_2",
        workspace / "dude_2_2",
        workspace / "dude_extracted",
        workspace / "dude_proteins",
    ]

    receptors = find_receptor_pdbs(dude_roots)
    if not receptors:
        print("No receptor.pdb files found under DUDE roots:")
        for r in dude_roots:
            print(f"  - {r}")
        return

    print(f"Found {len(receptors)} DUDE receptors.")

    # Use default burial parameters; you can tune these or plug in NCPI/Optuna params.
    default_params = BurialParameters()

    output_dir = workspace / "results" / "dude_reports"
    run_dude_on_receptors(receptors, output_dir, default_params)


if __name__ == "__main__":
    main()


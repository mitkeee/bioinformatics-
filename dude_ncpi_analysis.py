#!/usr/bin/env python3
"""Run NCPI/comprehensive burial analysis on the DUDE dataset.

- Finds all `receptor.pdb` files under `dude_extracted/` (both dude_1_2 and dude_2_2)
- Maps each receptor to its DUDE system name (parent directory name)
- Runs the same pipeline as `comprehensive_burial_analysis` / `NCPIProtocol`
- Writes outputs into a separate `results_dude/` tree so they don't mix with 3PTE/4d05/etc.

Outputs (per DUDE system):
- detailed CSV with RASA_dssp, RASA_stride, dssp_class, stride_class, ncps_class, neighbor features
- confusion matrices vs DSSP and vs STRIDE
- per-protein text report and overall summary
"""

from pathlib import Path
from typing import List

from comprehensive_burial_analysis import (
    BurialParameters,
    process_protein_dataset,
    save_confusion_matrices,
    generate_summary_report,
    ProteinResults,
    save_combined_confusion_matrix_report,
)


def find_dude_receptors(base_dir: Path) -> List[Path]:
    """Return a sorted list of all DUDE receptor PDB files under `dude_extracted/`.

    We expect them as `<base_dir>/dude_extracted/**/receptor.pdb`.
    """
    dude_root = base_dir / "dude_extracted"
    if not dude_root.exists():
        print(f"DUDE root directory not found: {dude_root}")
        return []

    receptors = sorted(dude_root.glob("**/receptor.pdb"))
    print(f"Found {len(receptors)} DUDE receptors under {dude_root}")
    return receptors


def relabel_results_with_system_names(results: List[ProteinResults]) -> List[ProteinResults]:
    """Replace generic protein_id 'receptor' with DUDE system directory name.

    For a path like .../dude_1_2/adrb1/receptor.pdb, the system name is 'adrb1'.
    We update ProteinResults.protein_id and also add a 'system_id' column
    to the underlying dataframe for traceability.
    """
    for res in results:
        # Infer system name from first row's source if present, else from stem
        # We rely on the fact that process_protein_dataset used the PDB stem
        # as protein_id, which is always 'receptor' here, so we re-derive from path.
        # To get the path, we store it in the dataframe when running this script.
        df = res.dataframe
        if 'source_pdb_path' in df.columns and len(df) > 0:
            pdb_path = Path(df['source_pdb_path'].iloc[0])
            system_name = pdb_path.parent.name  # e.g., 'adrb1'
        else:
            # Fallback: keep existing id
            system_name = res.protein_id

        res.protein_id = system_name
        df['system_id'] = system_name
        res.dataframe = df

    return results


def main() -> None:
    workspace = Path.cwd()
    output_root = workspace / "results_dude"
    output_root.mkdir(parents=True, exist_ok=True)

    pdb_files = find_dude_receptors(workspace)
    if not pdb_files:
        print("No DUDE receptor PDB files found. Nothing to do.")
        return

    # Use baseline NCPI/burial parameters; you can later plug in optimized ones
    params = BurialParameters(
        nc6_threshold=10.0,
        nc10_threshold=18.0,
        uni6_threshold=0.40,
        uni10_threshold=0.50,
        dssp_asa_cutoff=30.0,
        stride_asa_cutoff=24.0,
    )

    print("\n" + "=" * 80)
    print("DUDE NCPI / BURIAL ANALYSIS")
    print("=" * 80)
    print(f"Total DUDE receptors: {len(pdb_files)}")

    # Before passing to the generic processor, attach the PDB path as a column
    # after processing so we can recover the system name.
    results = process_protein_dataset(pdb_files, params)

    # Attach the original PDB path to each dataframe so relabeling works
    for res, pdb in zip(results, pdb_files):
        res.dataframe['source_pdb_path'] = str(pdb)

    # Replace generic protein_id ('receptor') with DUDE system name (parent dir)
    results = relabel_results_with_system_names(results)

    # Save confusion matrices (per DUDE receptor/system)
    cm_dir = output_root / "confusion_matrices"
    save_confusion_matrices(results, cm_dir)

    # Also save detailed confusion-matrix reports per system (DSSP & STRIDE)
    reports_dir = output_root / "reports"
    save_combined_confusion_matrix_report(results, reports_dir)

    # Summary report (all DUDE receptors combined)
    summary_file = output_root / "summary_report.txt"
    generate_summary_report(results, summary_file)

    # Per-receptor detailed CSVs (same layout as 3PTE_detailed_results.csv,
    # including RASA_dssp / RASA_stride and classification columns)
    for res in results:
        df = res.dataframe.copy()
        # Drop helper path column from final CSVs; keep system_id for clarity
        if 'source_pdb_path' in df.columns:
            df = df.drop(columns=['source_pdb_path'])
        # Fill NaNs with zeros and also replace empty strings with '0' so there
        # are no visually blank cells in the main detailed CSV.
        df_filled = df.fillna(0).replace('', '0')
        df_filled.to_csv(csv_path, index=False)

        # Also keep the original (with NaNs/empty strings) for advanced analysis
        raw_csv_path = output_root / f"{res.protein_id}_detailed_results_raw.csv"
        df.to_csv(raw_csv_path, index=False)
        df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("DUDE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results written under: {output_root}")


if __name__ == "__main__":
    main()

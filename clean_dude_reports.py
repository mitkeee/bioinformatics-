#!/usr/bin/env python3
"""Clean DUDE detailed result CSVs by removing mostly-null DSSP/STRIDE columns.

This script walks `results/dude_reports` and, for each `*_detailed_results.csv`,
removes columns related to DSSP/STRIDE that are entirely (or almost entirely)
NaN. This keeps the per-residue reports focused on our algorithm and
geometric features.
"""

from pathlib import Path
import pandas as pd


# Column prefixes we consider optional/reference-only
DSSP_COLS = ["dssp_asa", "dssp_ss", "dssp_class"]
STRIDE_COLS = ["stride_asa", "stride_ss", "stride_class"]


def clean_detailed_csv(csv_path: Path, null_threshold: float = 1.0) -> None:
    """Load a detailed DUDE CSV and drop DSSP/STRIDE columns that are mostly null.

    Additionally, always drop the `stride_ss` column if present, since you
    requested to exclude that field entirely from DUDE detailed reports.
    """
    df = pd.read_csv(csv_path)
    original_cols = list(df.columns)

    # First, drop stride_ss unconditionally if present
    if "stride_ss" in df.columns:
        df = df.drop(columns=["stride_ss"])

    # Then drop other DSSP/STRIDE columns that are fully (or mostly) null
    for col in DSSP_COLS + STRIDE_COLS:
        if col in df.columns:
            frac_null = df[col].isna().mean()
            if frac_null >= null_threshold:
                df = df.drop(columns=[col])

    if list(df.columns) != original_cols:
        df.to_csv(csv_path, index=False)
        print(f"Cleaned: {csv_path.name}")
    else:
        print(f"Unchanged: {csv_path.name}")


def main() -> None:
    root = Path(__file__).resolve().parent
    reports_dir = root / "results" / "dude_reports"
    if not reports_dir.exists():
        print(f"No DUDE reports directory found at: {reports_dir}")
        return

    detailed_files = sorted(reports_dir.glob("*_detailed_results.csv"))
    if not detailed_files:
        print(f"No *_detailed_results.csv files found in: {reports_dir}")
        return

    print(f"Found {len(detailed_files)} detailed DUDE CSV files to clean.")
    for csv_file in detailed_files:
        clean_detailed_csv(csv_file, null_threshold=1.0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Replace NULL/NaN values in CSV files with 0.000000
Preserves formatting and ensures no empty cells
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_file(csv_path):
    """Replace all NaN/null values with 0.000000 in CSV file."""
    print(f"\nProcessing: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Count nulls before
    null_count_before = df.isnull().sum().sum()
    print(f"  Null values before: {null_count_before}")

    # Replace NaN with 0.0 for numeric columns
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(0.0)
        elif df[col].dtype == 'object':
            # For string columns, replace NaN with '-' or empty string
            df[col] = df[col].fillna('-')

    # Count nulls after
    null_count_after = df.isnull().sum().sum()
    print(f"  Null values after: {null_count_after}")

    # Save back to CSV with proper formatting
    # Use float_format to ensure consistent decimal places
    df.to_csv(csv_path, index=False, float_format='%.10f')
    print(f"  ✅ Saved: {csv_path}")

    return null_count_before, null_count_after

def main():
    """Clean all CSV result files."""
    print("="*80)
    print("CLEANING CSV FILES - REPLACING NULL VALUES WITH 0.000000")
    print("="*80)

    # List of CSV files to clean
    csv_files = [
        'results/3pte_results.csv',
        'results/4d05_results.csv',
        'results/6wti_results.csv',
        'results/7upo_results.csv',
        'results/combined_results.csv',
        'results/decision_tree/combined_normalized.csv'
    ]

    total_before = 0
    total_after = 0

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if csv_path.exists():
            before, after = clean_csv_file(csv_path)
            total_before += before
            total_after += after
        else:
            print(f"\n⚠️  File not found: {csv_file}")

    print("\n" + "="*80)
    print(f"COMPLETE!")
    print(f"Total null values replaced: {total_before}")
    print(f"Remaining null values: {total_after}")
    print("="*80)

    # Show example of cleaned data
    print("\nExample from 3pte_results.csv (first 5 rows):")
    df_sample = pd.read_csv('results/3pte_results.csv')
    print(df_sample.head(5).to_string())

if __name__ == "__main__":
    main()


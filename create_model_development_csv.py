#!/usr/bin/env python3
"""
Create Model Development CSV Files
Extracts ONLY features for model development (no DSSP/STRIDE values)
Creates clean datasets with only neighbor-based features
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_model_development_csv(results_dir="results"):
    """
    Create CSV files containing ONLY features for model development.

    Excludes ALL DSSP/STRIDE values:
    - No dssp_asa, dssp_class, dssp_ss
    - No stride_asa, stride_class, stride_ss

    Includes ONLY:
    - Basic residue info (res_id, res_num, protein)
    - Neighbor-based features (counts and uniformity)
    """

    print("="*80)
    print("CREATING MODEL DEVELOPMENT CSV FILES")
    print("(DSSP/STRIDE FREE - ONLY NEIGHBOR-BASED FEATURES)")
    print("="*80)

    # Columns to KEEP for model development
    model_columns = [
        'res_id',                    # Amino acid type
        'res_num',                   # Residue number
        'ncps_sphere_6',             # Neighbor count at 6Å
        'ncps_sphere_6_uni',         # Uniformity at 6Å
        'ncps_sphere_10',            # Neighbor count at 10Å
        'ncps_sphere_10_uni',        # Uniformity at 10Å
        'ncps_class'                 # Your algorithm's classification
    ]

    # Process individual protein files
    proteins = ['3pte', '4d05', '6wti', '7upo']
    individual_files = []

    for protein in proteins:
        input_file = Path(results_dir) / f"{protein}_results.csv"
        output_file = Path(results_dir) / f"{protein}_model_development.csv"

        if input_file.exists():
            print(f"\nProcessing: {protein.upper()}")

            # Read full results
            df = pd.read_csv(input_file)
            print(f"  Input: {len(df)} residues with {len(df.columns)} columns")

            # Extract only model development columns
            df_model = df[model_columns].copy()

            # Add protein identifier
            df_model.insert(0, 'protein', protein.upper())

            # Save
            df_model.to_csv(output_file, index=False, float_format='%.10f')
            print(f"  Output: {len(df_model)} residues with {len(df_model.columns)} columns")
            print(f"  ✅ Saved: {output_file}")

            individual_files.append((protein, output_file, df_model))
        else:
            print(f"\n⚠️  File not found: {input_file}")

    # Create combined model development file
    print("\n" + "="*80)
    print("CREATING COMBINED MODEL DEVELOPMENT FILE")
    print("="*80)

    combined_dfs = [df for _, _, df in individual_files]
    df_combined = pd.concat(combined_dfs, ignore_index=True)

    combined_file = Path(results_dir) / "combined_model_development.csv"
    df_combined.to_csv(combined_file, index=False, float_format='%.10f')

    print(f"\nCombined dataset:")
    print(f"  Total residues: {len(df_combined)}")
    print(f"  Proteins: {df_combined['protein'].unique().tolist()}")
    print(f"  Features: {len(df_combined.columns)} columns")
    print(f"  ✅ Saved: {combined_file}")

    # Create normalized version for machine learning
    print("\n" + "="*80)
    print("CREATING NORMALIZED MODEL DEVELOPMENT FILE")
    print("="*80)

    df_normalized = df_combined.copy()

    # Features to normalize (only neighbor-based)
    features_to_normalize = [
        'ncps_sphere_6',
        'ncps_sphere_6_uni',
        'ncps_sphere_10',
        'ncps_sphere_10_uni'
    ]

    # Z-score normalization per protein
    for protein in df_normalized['protein'].unique():
        mask = df_normalized['protein'] == protein
        print(f"\nNormalizing {protein}:")

        for feat in features_to_normalize:
            values = df_normalized.loc[mask, feat]
            if values.notna().sum() > 0:
                mean = values.mean()
                std = values.std()
                if std > 0:
                    df_normalized.loc[mask, f'{feat}_norm'] = (values - mean) / std
                else:
                    df_normalized.loc[mask, f'{feat}_norm'] = 0.0
                print(f"  {feat}: mean={mean:.2f}, std={std:.2f}")
            else:
                df_normalized.loc[mask, f'{feat}_norm'] = 0.0

    # Reorder columns: basic info, original features, normalized features, classification
    column_order = [
        'protein',
        'res_id',
        'res_num',
        'ncps_sphere_6',
        'ncps_sphere_6_uni',
        'ncps_sphere_10',
        'ncps_sphere_10_uni',
        'ncps_sphere_6_norm',
        'ncps_sphere_6_uni_norm',
        'ncps_sphere_10_norm',
        'ncps_sphere_10_uni_norm',
        'ncps_class'
    ]

    df_normalized = df_normalized[column_order]

    normalized_file = Path(results_dir) / "combined_model_development_normalized.csv"
    df_normalized.to_csv(normalized_file, index=False, float_format='%.10f')

    print(f"\nNormalized dataset:")
    print(f"  Total residues: {len(df_normalized)}")
    print(f"  Features: {len(df_normalized.columns)} columns")
    print(f"    - 4 original neighbor features")
    print(f"    - 4 normalized neighbor features (z-score per protein)")
    print(f"  ✅ Saved: {normalized_file}")

    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n📊 INDIVIDUAL PROTEIN FILES (for single-protein analysis):")
    for protein, filepath, df in individual_files:
        print(f"  - {filepath} ({len(df)} residues)")

    print("\n📊 COMBINED FILES (for multi-protein analysis):")
    print(f"  - {combined_file}")
    print(f"    → {len(df_combined)} residues, raw features")
    print(f"  - {normalized_file}")
    print(f"    → {len(df_normalized)} residues, z-score normalized per protein")

    print("\n✅ COLUMNS IN MODEL DEVELOPMENT FILES:")
    print("  Basic Info:")
    print("    - protein: Protein identifier (3PTE, 4d05, 6wti, 7upo)")
    print("    - res_id: Amino acid type (ALA, LEU, etc.)")
    print("    - res_num: Residue number in sequence")
    print("\n  Features for Classification (Raw):")
    print("    - ncps_sphere_6: Neighbor count at 6Å radius")
    print("    - ncps_sphere_6_uni: Uniformity at 6Å radius")
    print("    - ncps_sphere_10: Neighbor count at 10Å radius")
    print("    - ncps_sphere_10_uni: Uniformity at 10Å radius")
    print("\n  Features for Classification (Normalized - in normalized file only):")
    print("    - ncps_sphere_6_norm: Z-score normalized neighbor count at 6Å")
    print("    - ncps_sphere_6_uni_norm: Z-score normalized uniformity at 6Å")
    print("    - ncps_sphere_10_norm: Z-score normalized neighbor count at 10Å")
    print("    - ncps_sphere_10_uni_norm: Z-score normalized uniformity at 10Å")
    print("\n  Classification Result:")
    print("    - ncps_class: Your algorithm's prediction (0=interior, 1=exterior)")

    print("\n⚠️  EXCLUDED (not in these files):")
    print("    - NO dssp_asa, dssp_class, dssp_ss")
    print("    - NO stride_asa, stride_class, stride_ss")
    print("    → Use original *_results.csv files for validation against DSSP/STRIDE")

    print("\n💡 USAGE:")
    print("  For model development/training:")
    print("    → Use: combined_model_development_normalized.csv")
    print("    → Features: 4 normalized features (ncps_*_norm)")
    print("    → Target: ncps_class (your classification)")
    print("\n  For validation against DSSP/STRIDE:")
    print("    → Use: [protein]_results.csv (original files)")
    print("    → Contains both model features AND reference methods")

    # Show sample of normalized file
    print("\n" + "="*80)
    print("SAMPLE DATA (first 5 rows from normalized file):")
    print("="*80)
    print(df_normalized.head(5).to_string())

    print("\n" + "="*80)
    print("✅ COMPLETE! Model development files created successfully.")
    print("="*80)

    return combined_file, normalized_file

if __name__ == "__main__":
    combined, normalized = create_model_development_csv()
    print(f"\n🎯 PRIMARY FILE FOR MODEL DEVELOPMENT:")
    print(f"   {normalized}")
    print(f"\n   This file contains ONLY neighbor-based features.")
    print(f"   NO DSSP or STRIDE values - completely agnostic!")


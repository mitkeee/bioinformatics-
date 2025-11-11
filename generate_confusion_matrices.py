#!/usr/bin/env python3
"""
Quick script to generate 2 confusion matrices per protein
"""

from pathlib import Path
from comprehensive_burial_analysis import (
    BurialParameters,
    process_protein_dataset,
    save_confusion_matrices,
    generate_summary_report
)

print("\n" + "="*80)
print("GENERATING CONFUSION MATRICES")
print("="*80 + "\n")

# Setup
workspace_dir = Path.cwd()
output_dir = workspace_dir / "results" / "comprehensive_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# Find PDB files
pdb_files = sorted(workspace_dir.glob("*.pdb"))
print(f"Found {len(pdb_files)} PDB files to analyze:\n")
for pdb in pdb_files:
    print(f"  - {pdb.name}")

# Default parameters
params = BurialParameters(
    nc6_threshold=10.0,
    nc10_threshold=18.0,
    uni6_threshold=0.40,
    uni10_threshold=0.50,
    dssp_asa_cutoff=30.0,
    stride_asa_cutoff=24.0
)

print(f"\n{'='*80}")
print("PROCESSING PROTEINS...")
print(f"{'='*80}\n")

# Process all proteins
results = process_protein_dataset(pdb_files, params)

# Save confusion matrices
print("\n" + "="*80)
print("SAVING CONFUSION MATRICES")
print("="*80 + "\n")

cm_dir = output_dir / "confusion_matrices"
save_confusion_matrices(results, cm_dir)

# Save combined confusion matrix reports (both DSSP and STRIDE in readable format)
save_combined_confusion_matrix_report(results, cm_dir)

# List generated files
print("\nGenerated confusion matrix CSV files:")
print("-" * 80)
for cm_file in sorted(cm_dir.glob("*.csv")):
    print(f"  ✓ {cm_file.name}")

print("\nGenerated confusion matrix report files:")
print("-" * 80)
for report_file in sorted(cm_dir.glob("*_report.txt")):
    print(f"  ✓ {report_file.name}")

# Display the confusion matrices
print("\n" + "="*80)
print("CONFUSION MATRICES CONTENT")
print("="*80 + "\n")

import pandas as pd

for result in results:
    protein_id = result.protein_id

    # DSSP confusion matrix
    if result.dssp_confusion_matrix is not None:
        print(f"\n{protein_id.upper()} - CONFUSION MATRIX vs DSSP:")
        print("-" * 60)
        cm_df = pd.DataFrame(
            result.dssp_confusion_matrix,
            index=['True_Interior(0)', 'True_Exterior(1)'],
            columns=['Pred_Interior(0)', 'Pred_Exterior(1)']
        )
        print(cm_df)
        print(f"Accuracy: {result.dssp_accuracy:.4f}")
        print(f"F1-Score: {result.dssp_f1:.4f}")

    # STRIDE confusion matrix
    if result.stride_confusion_matrix is not None:
        print(f"\n{protein_id.upper()} - CONFUSION MATRIX vs STRIDE:")
        print("-" * 60)
        cm_df = pd.DataFrame(
            result.stride_confusion_matrix,
            index=['True_Interior(0)', 'True_Exterior(1)'],
            columns=['Pred_Interior(0)', 'Pred_Exterior(1)']
        )
        print(cm_df)
        print(f"Accuracy: {result.stride_accuracy:.4f}")
        print(f"F1-Score: {result.stride_f1:.4f}")

    print()

# Save summary report
generate_summary_report(results, output_dir / "summary_report.txt")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"\nAll confusion matrices saved to:")
print(f"  {cm_dir}")
print(f"\nTotal matrices generated: {len(list(cm_dir.glob('*.csv')))}")
print(f"  - {len(pdb_files)} proteins × 2 references (DSSP + STRIDE)")
print()
#!/usr/bin/env python3
"""
Quick script to generate 2 confusion matrices per protein
"""

from pathlib import Path
from comprehensive_burial_analysis import (
    BurialParameters,
    process_protein_dataset,
    save_confusion_matrices,
    save_combined_confusion_matrix_report,
    generate_summary_report
)

print("\n" + "="*80)
print("GENERATING CONFUSION MATRICES")
print("="*80 + "\n")

# Setup
workspace_dir = Path.cwd()
output_dir = workspace_dir / "results" / "comprehensive_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# Find PDB files
pdb_files = sorted(workspace_dir.glob("*.pdb"))
print(f"Found {len(pdb_files)} PDB files to analyze:\n")
for pdb in pdb_files:
    print(f"  - {pdb.name}")

# Default parameters
params = BurialParameters(
    nc6_threshold=10.0,
    nc10_threshold=18.0,
    uni6_threshold=0.40,
    uni10_threshold=0.50,
    dssp_asa_cutoff=30.0,
    stride_asa_cutoff=24.0
)

print(f"\n{'='*80}")
print("PROCESSING PROTEINS...")
print(f"{'='*80}\n")

# Process all proteins
results = process_protein_dataset(pdb_files, params)

# Save confusion matrices
print("\n" + "="*80)
print("SAVING CONFUSION MATRICES")
print("="*80 + "\n")

cm_dir = output_dir / "confusion_matrices"
save_confusion_matrices(results, cm_dir)

# List generated files
print("\nGenerated confusion matrix files:")
print("-" * 80)
for cm_file in sorted(cm_dir.glob("*.csv")):
    print(f"  ✓ {cm_file.name}")

# Display the confusion matrices
print("\n" + "="*80)
print("CONFUSION MATRICES CONTENT")
print("="*80 + "\n")

import pandas as pd

for result in results:
    protein_id = result.protein_id

    # DSSP confusion matrix
    if result.dssp_confusion_matrix is not None:
        print(f"\n{protein_id.upper()} - CONFUSION MATRIX vs DSSP:")
        print("-" * 60)
        cm_df = pd.DataFrame(
            result.dssp_confusion_matrix,
            index=['True_Interior(0)', 'True_Exterior(1)'],
            columns=['Pred_Interior(0)', 'Pred_Exterior(1)']
        )
        print(cm_df)
        print(f"Accuracy: {result.dssp_accuracy:.4f}")
        print(f"F1-Score: {result.dssp_f1:.4f}")

    # STRIDE confusion matrix
    if result.stride_confusion_matrix is not None:
        print(f"\n{protein_id.upper()} - CONFUSION MATRIX vs STRIDE:")
        print("-" * 60)
        cm_df = pd.DataFrame(
            result.stride_confusion_matrix,
            index=['True_Interior(0)', 'True_Exterior(1)'],
            columns=['Pred_Interior(0)', 'Pred_Exterior(1)']
        )
        print(cm_df)
        print(f"Accuracy: {result.stride_accuracy:.4f}")
        print(f"F1-Score: {result.stride_f1:.4f}")

    print()

# Save summary report
generate_summary_report(results, output_dir / "summary_report.txt")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"\nAll confusion matrices saved to:")
print(f"  {cm_dir}")
print(f"\nTotal matrices generated: {len(list(cm_dir.glob('*.csv')))}")
print(f"  - {len(pdb_files)} proteins × 2 references (DSSP + STRIDE)")
print()


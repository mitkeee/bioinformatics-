#!/usr/bin/env python3
"""
Standalone script to generate confusion matrix reports
Creates readable text files showing both DSSP and STRIDE confusion matrices
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

print("\n" + "="*80)
print("GENERATING CONFUSION MATRIX REPORTS")
print("="*80 + "\n")

# Setup paths
workspace = Path.cwd()
input_dir = workspace / "results" / "comprehensive_analysis" / "confusion_matrices"
output_dir = workspace / "results" / "confusion_matrix_reports"
output_dir.mkdir(parents=True, exist_ok=True)

# Check if confusion matrix CSV files exist
csv_files = list(input_dir.glob("*_confusion_matrix_*.csv"))
if not csv_files:
    print(f"ERROR: No confusion matrix CSV files found in {input_dir}")
    print("\nPlease run 'python3 generate_confusion_matrices.py' first to create the confusion matrices.")
    exit(1)

print(f"Found {len(csv_files)} confusion matrix CSV files\n")

# Group by protein
proteins = {}
for csv_file in csv_files:
    filename = csv_file.stem
    if '_confusion_matrix_dssp' in filename:
        protein_id = filename.replace('_confusion_matrix_dssp', '')
        if protein_id not in proteins:
            proteins[protein_id] = {}
        proteins[protein_id]['dssp'] = csv_file
    elif '_confusion_matrix_stride' in filename:
        protein_id = filename.replace('_confusion_matrix_stride', '')
        if protein_id not in proteins:
            proteins[protein_id] = {}
        proteins[protein_id]['stride'] = csv_file

print(f"Identified {len(proteins)} unique proteins\n")

# Generate reports for each protein
all_results = []

for protein_id, files in proteins.items():
    print(f"Generating report for {protein_id}...")
    
    report_file = output_dir / f"{protein_id}_confusion_matrices_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CONFUSION MATRICES FOR PROTEIN: {protein_id.upper()}\n")
        f.write("="*80 + "\n\n")
        
        result = {'protein_id': protein_id}
        
        # DSSP Confusion Matrix
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX #1: vs DSSP (Ground Truth)\n")
        f.write("="*80 + "\n\n")
        
        if 'dssp' in files:
            cm_df = pd.read_csv(files['dssp'], index_col=0)
            cm = cm_df.values
            
            f.write("Classification Key:\n")
            f.write("  0 = Interior (Buried)\n")
            f.write("  1 = Exterior (Surface/Exposed)\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {cm[0,0]:20d}  {cm[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {cm[1,0]:20d}  {cm[1,1]:20d}\n\n")
            
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f.write("Metrics:\n")
            f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1-Score:  {f1:.4f}\n\n")
            
            f.write("Breakdown:\n")
            f.write(f"  True Negatives (TN):  {tn:5d} - Correctly predicted as Interior\n")
            f.write(f"  False Positives (FP): {fp:5d} - Interior wrongly predicted as Exterior\n")
            f.write(f"  False Negatives (FN): {fn:5d} - Exterior wrongly predicted as Interior\n")
            f.write(f"  True Positives (TP):  {tp:5d} - Correctly predicted as Exterior\n")
            f.write(f"  Total:                {total:5d}\n\n")
            
            result['dssp_cm'] = cm
            result['dssp_accuracy'] = accuracy
            result['dssp_f1'] = f1
        else:
            f.write("  No DSSP confusion matrix found.\n\n")
        
        # STRIDE Confusion Matrix
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX #2: vs STRIDE (Ground Truth)\n")
        f.write("="*80 + "\n\n")
        
        if 'stride' in files:
            cm_df = pd.read_csv(files['stride'], index_col=0)
            cm = cm_df.values
            
            f.write("Classification Key:\n")
            f.write("  0 = Interior (Buried)\n")
            f.write("  1 = Exterior (Surface/Exposed)\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {cm[0,0]:20d}  {cm[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {cm[1,0]:20d}  {cm[1,1]:20d}\n\n")
            
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f.write("Metrics:\n")
            f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1-Score:  {f1:.4f}\n\n")
            
            f.write("Breakdown:\n")
            f.write(f"  True Negatives (TN):  {tn:5d} - Correctly predicted as Interior\n")
            f.write(f"  False Positives (FP): {fp:5d} - Interior wrongly predicted as Exterior\n")
            f.write(f"  False Negatives (FN): {fn:5d} - Exterior wrongly predicted as Interior\n")
            f.write(f"  True Positives (TP):  {tp:5d} - Correctly predicted as Exterior\n")
            f.write(f"  Total:                {total:5d}\n\n")
            
            result['stride_cm'] = cm
            result['stride_accuracy'] = accuracy
            result['stride_f1'] = f1
        else:
            f.write("  No STRIDE confusion matrix found.\n\n")
        
        # Comparison
        if 'dssp' in files and 'stride' in files:
            f.write("="*80 + "\n")
            f.write("COMPARISON: DSSP vs STRIDE\n")
            f.write("="*80 + "\n\n")
            dssp_acc = result['dssp_accuracy']
            stride_acc = result['stride_accuracy']
            f.write(f"DSSP Accuracy:   {dssp_acc:.4f} ({dssp_acc*100:.2f}%)\n")
            f.write(f"STRIDE Accuracy: {stride_acc:.4f} ({stride_acc*100:.2f}%)\n")
            f.write(f"Difference:      {abs(dssp_acc - stride_acc):.4f}\n\n")
            
            if dssp_acc > stride_acc:
                f.write("→ Better agreement with DSSP\n")
            elif stride_acc > dssp_acc:
                f.write("→ Better agreement with STRIDE\n")
            else:
                f.write("→ Equal agreement with both methods\n")
        
        all_results.append(result)

# Generate master summary
print("\nGenerating master summary...")
master_file = output_dir / "ALL_PROTEINS_confusion_matrices_summary.txt"

with open(master_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MASTER SUMMARY: ALL PROTEINS CONFUSION MATRICES\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total Proteins Analyzed: {len(all_results)}\n\n")
    
    # DSSP Summary
    f.write("="*80 + "\n")
    f.write("AGGREGATE RESULTS vs DSSP\n")
    f.write("="*80 + "\n\n")
    
    dssp_results = [r for r in all_results if 'dssp_cm' in r]
    if dssp_results:
        total_cm = sum(r['dssp_cm'] for r in dssp_results)
        accuracies = [r['dssp_accuracy'] for r in dssp_results]
        
        f.write(f"Proteins with DSSP data: {len(dssp_results)}\n\n")
        f.write("Aggregate Confusion Matrix (All Proteins Combined):\n")
        f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
        f.write(f"True Interior(0)    {total_cm[0,0]:20.0f}  {total_cm[0,1]:20.0f}\n")
        f.write(f"True Exterior(1)    {total_cm[1,0]:20.0f}  {total_cm[1,1]:20.0f}\n\n")
        
        f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Min Accuracy:  {np.min(accuracies):.4f}\n")
        f.write(f"Max Accuracy:  {np.max(accuracies):.4f}\n\n")
    
    # STRIDE Summary
    f.write("="*80 + "\n")
    f.write("AGGREGATE RESULTS vs STRIDE\n")
    f.write("="*80 + "\n\n")
    
    stride_results = [r for r in all_results if 'stride_cm' in r]
    if stride_results:
        total_cm = sum(r['stride_cm'] for r in stride_results)
        accuracies = [r['stride_accuracy'] for r in stride_results]
        
        f.write(f"Proteins with STRIDE data: {len(stride_results)}\n\n")
        f.write("Aggregate Confusion Matrix (All Proteins Combined):\n")
        f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
        f.write(f"True Interior(0)    {total_cm[0,0]:20.0f}  {total_cm[0,1]:20.0f}\n")
        f.write(f"True Exterior(1)    {total_cm[1,0]:20.0f}  {total_cm[1,1]:20.0f}\n\n")
        
        f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Min Accuracy:  {np.min(accuracies):.4f}\n")
        f.write(f"Max Accuracy:  {np.max(accuracies):.4f}\n\n")
    
    # Per-protein table
    f.write("="*80 + "\n")
    f.write("PER-PROTEIN ACCURACY TABLE\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Protein ID':<15} {'DSSP Accuracy':>15} {'STRIDE Accuracy':>15}\n")
    f.write("-"*80 + "\n")
    
    for r in all_results:
        dssp_acc = f"{r['dssp_accuracy']:.4f}" if 'dssp_accuracy' in r else "N/A"
        stride_acc = f"{r['stride_accuracy']:.4f}" if 'stride_accuracy' in r else "N/A"
        f.write(f"{r['protein_id']:<15} {dssp_acc:>15} {stride_acc:>15}\n")

print("\n" + "="*80)
print("✓ CONFUSION MATRIX REPORTS GENERATED")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated files:")
print(f"  - {len(all_results)} individual protein reports (*_confusion_matrices_report.txt)")
print(f"  - 1 master summary (ALL_PROTEINS_confusion_matrices_summary.txt)")
print(f"\nEach report contains:")
print(f"  ✓ Confusion Matrix #1 vs DSSP (resnica DSSP)")
print(f"  ✓ Confusion Matrix #2 vs STRIDE (resnica STRIDE)")
print(f"  ✓ Metrics: Accuracy, Precision, Recall, F1-Score")
print(f"  ✓ Detailed breakdown (TP, TN, FP, FN)")
print()


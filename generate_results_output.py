#!/usr/bin/env python3
"""
Generate a comprehensive results output file
Creates a single readable file with all confusion matrix results
"""

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

print("\n" + "="*80)
print("GENERATING COMPREHENSIVE RESULTS OUTPUT FILE")
print("="*80 + "\n")

# Setup paths
workspace = Path.cwd()
input_dir = workspace / "results" / "comprehensive_analysis" / "confusion_matrices"
output_dir = workspace / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Output file
output_file = output_dir / "RESULTS_OUTPUT.txt"

# Check if confusion matrix CSV files exist
csv_files = list(input_dir.glob("*_confusion_matrix_*.csv"))
if not csv_files:
    print(f"ERROR: No confusion matrix CSV files found in {input_dir}")
    print("\nPlease run 'python3 generate_confusion_matrices.py' first.")
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

print(f"Processing {len(proteins)} proteins...\n")

# Collect all results
all_results = []

for protein_id, files in proteins.items():
    result = {'protein_id': protein_id}
    
    # Process DSSP
    if 'dssp' in files:
        cm_df = pd.read_csv(files['dssp'], index_col=0)
        cm = cm_df.values
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        result['dssp_cm'] = cm
        result['dssp_tn'] = tn
        result['dssp_fp'] = fp
        result['dssp_fn'] = fn
        result['dssp_tp'] = tp
        result['dssp_total'] = total
        result['dssp_accuracy'] = accuracy
        result['dssp_precision'] = precision
        result['dssp_recall'] = recall
        result['dssp_f1'] = f1
    
    # Process STRIDE
    if 'stride' in files:
        cm_df = pd.read_csv(files['stride'], index_col=0)
        cm = cm_df.values
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        result['stride_cm'] = cm
        result['stride_tn'] = tn
        result['stride_fp'] = fp
        result['stride_fn'] = fn
        result['stride_tp'] = tp
        result['stride_total'] = total
        result['stride_accuracy'] = accuracy
        result['stride_precision'] = precision
        result['stride_recall'] = recall
        result['stride_f1'] = f1
    
    all_results.append(result)

# Generate comprehensive output file
print(f"Writing results to: {output_file}")

with open(output_file, 'w') as f:
    # Header
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE RESULTS OUTPUT\n")
    f.write("Protein Burial Classification Analysis\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Proteins: {len(all_results)}\n\n")
    
    # Executive Summary
    f.write("="*80 + "\n")
    f.write("EXECUTIVE SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    dssp_results = [r for r in all_results if 'dssp_accuracy' in r]
    stride_results = [r for r in all_results if 'stride_accuracy' in r]
    
    if dssp_results:
        dssp_accuracies = [r['dssp_accuracy'] for r in dssp_results]
        f.write(f"DSSP Comparison:\n")
        f.write(f"  Proteins analyzed: {len(dssp_results)}\n")
        f.write(f"  Mean accuracy: {np.mean(dssp_accuracies):.4f} ({np.mean(dssp_accuracies)*100:.2f}%)\n")
        f.write(f"  Std deviation: {np.std(dssp_accuracies):.4f}\n")
        f.write(f"  Best accuracy: {np.max(dssp_accuracies):.4f} ({np.max(dssp_accuracies)*100:.2f}%)\n")
        f.write(f"  Worst accuracy: {np.min(dssp_accuracies):.4f} ({np.min(dssp_accuracies)*100:.2f}%)\n\n")
    
    if stride_results:
        stride_accuracies = [r['stride_accuracy'] for r in stride_results]
        f.write(f"STRIDE Comparison:\n")
        f.write(f"  Proteins analyzed: {len(stride_results)}\n")
        f.write(f"  Mean accuracy: {np.mean(stride_accuracies):.4f} ({np.mean(stride_accuracies)*100:.2f}%)\n")
        f.write(f"  Std deviation: {np.std(stride_accuracies):.4f}\n")
        f.write(f"  Best accuracy: {np.max(stride_accuracies):.4f} ({np.max(stride_accuracies)*100:.2f}%)\n")
        f.write(f"  Worst accuracy: {np.min(stride_accuracies):.4f} ({np.min(stride_accuracies)*100:.2f}%)\n\n")
    
    # Aggregate Confusion Matrices
    f.write("="*80 + "\n")
    f.write("AGGREGATE CONFUSION MATRICES (ALL PROTEINS COMBINED)\n")
    f.write("="*80 + "\n\n")
    
    if dssp_results:
        total_cm_dssp = sum(r['dssp_cm'] for r in dssp_results)
        total_tn = sum(r['dssp_tn'] for r in dssp_results)
        total_fp = sum(r['dssp_fp'] for r in dssp_results)
        total_fn = sum(r['dssp_fn'] for r in dssp_results)
        total_tp = sum(r['dssp_tp'] for r in dssp_results)
        total_residues = sum(r['dssp_total'] for r in dssp_results)
        
        f.write("DSSP Aggregate Confusion Matrix:\n")
        f.write("-" * 80 + "\n")
        f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)     Total\n")
        f.write(f"True Interior(0)    {total_cm_dssp[0,0]:20.0f}  {total_cm_dssp[0,1]:20.0f}  {total_cm_dssp[0,0]+total_cm_dssp[0,1]:9.0f}\n")
        f.write(f"True Exterior(1)    {total_cm_dssp[1,0]:20.0f}  {total_cm_dssp[1,1]:20.0f}  {total_cm_dssp[1,0]+total_cm_dssp[1,1]:9.0f}\n")
        f.write(f"Total               {total_cm_dssp[0,0]+total_cm_dssp[1,0]:20.0f}  {total_cm_dssp[0,1]+total_cm_dssp[1,1]:20.0f}  {total_residues:9.0f}\n\n")
        
        overall_accuracy = (total_tn + total_tp) / total_residues
        f.write(f"Overall Metrics:\n")
        f.write(f"  Total residues: {total_residues}\n")
        f.write(f"  Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
        f.write(f"  True Negatives (TN): {total_tn} (correctly predicted interior)\n")
        f.write(f"  True Positives (TP): {total_tp} (correctly predicted exterior)\n")
        f.write(f"  False Positives (FP): {total_fp} (interior predicted as exterior)\n")
        f.write(f"  False Negatives (FN): {total_fn} (exterior predicted as interior)\n\n")
    
    if stride_results:
        total_cm_stride = sum(r['stride_cm'] for r in stride_results)
        total_tn = sum(r['stride_tn'] for r in stride_results)
        total_fp = sum(r['stride_fp'] for r in stride_results)
        total_fn = sum(r['stride_fn'] for r in stride_results)
        total_tp = sum(r['stride_tp'] for r in stride_results)
        total_residues = sum(r['stride_total'] for r in stride_results)
        
        f.write("STRIDE Aggregate Confusion Matrix:\n")
        f.write("-" * 80 + "\n")
        f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)     Total\n")
        f.write(f"True Interior(0)    {total_cm_stride[0,0]:20.0f}  {total_cm_stride[0,1]:20.0f}  {total_cm_stride[0,0]+total_cm_stride[0,1]:9.0f}\n")
        f.write(f"True Exterior(1)    {total_cm_stride[1,0]:20.0f}  {total_cm_stride[1,1]:20.0f}  {total_cm_stride[1,0]+total_cm_stride[1,1]:9.0f}\n")
        f.write(f"Total               {total_cm_stride[0,0]+total_cm_stride[1,0]:20.0f}  {total_cm_stride[0,1]+total_cm_stride[1,1]:20.0f}  {total_residues:9.0f}\n\n")
        
        overall_accuracy = (total_tn + total_tp) / total_residues
        f.write(f"Overall Metrics:\n")
        f.write(f"  Total residues: {total_residues}\n")
        f.write(f"  Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
        f.write(f"  True Negatives (TN): {total_tn} (correctly predicted interior)\n")
        f.write(f"  True Positives (TP): {total_tp} (correctly predicted exterior)\n")
        f.write(f"  False Positives (FP): {total_fp} (interior predicted as exterior)\n")
        f.write(f"  False Negatives (FN): {total_fn} (exterior predicted as interior)\n\n")
    
    # Per-Protein Results Table
    f.write("="*80 + "\n")
    f.write("PER-PROTEIN DETAILED RESULTS\n")
    f.write("="*80 + "\n\n")
    
    # Sort by protein ID
    all_results.sort(key=lambda x: x['protein_id'])
    
    for result in all_results:
        f.write("-" * 80 + "\n")
        f.write(f"PROTEIN: {result['protein_id'].upper()}\n")
        f.write("-" * 80 + "\n\n")
        
        # DSSP Results
        if 'dssp_accuracy' in result:
            f.write("Results vs DSSP:\n")
            f.write(f"  Accuracy:  {result['dssp_accuracy']:.4f} ({result['dssp_accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {result['dssp_precision']:.4f}\n")
            f.write(f"  Recall:    {result['dssp_recall']:.4f}\n")
            f.write(f"  F1-Score:  {result['dssp_f1']:.4f}\n\n")
            
            f.write(f"  Confusion Matrix:\n")
            f.write(f"                      Predicted Interior  Predicted Exterior\n")
            f.write(f"    True Interior     {result['dssp_tn']:18d}  {result['dssp_fp']:18d}\n")
            f.write(f"    True Exterior     {result['dssp_fn']:18d}  {result['dssp_tp']:18d}\n\n")
            
            f.write(f"  Breakdown:\n")
            f.write(f"    TN (True Negatives):  {result['dssp_tn']:5d}\n")
            f.write(f"    TP (True Positives):  {result['dssp_tp']:5d}\n")
            f.write(f"    FP (False Positives): {result['dssp_fp']:5d}\n")
            f.write(f"    FN (False Negatives): {result['dssp_fn']:5d}\n")
            f.write(f"    Total residues:       {result['dssp_total']:5d}\n\n")
        else:
            f.write("Results vs DSSP: No data available\n\n")
        
        # STRIDE Results
        if 'stride_accuracy' in result:
            f.write("Results vs STRIDE:\n")
            f.write(f"  Accuracy:  {result['stride_accuracy']:.4f} ({result['stride_accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {result['stride_precision']:.4f}\n")
            f.write(f"  Recall:    {result['stride_recall']:.4f}\n")
            f.write(f"  F1-Score:  {result['stride_f1']:.4f}\n\n")
            
            f.write(f"  Confusion Matrix:\n")
            f.write(f"                      Predicted Interior  Predicted Exterior\n")
            f.write(f"    True Interior     {result['stride_tn']:18d}  {result['stride_fp']:18d}\n")
            f.write(f"    True Exterior     {result['stride_fn']:18d}  {result['stride_tp']:18d}\n\n")
            
            f.write(f"  Breakdown:\n")
            f.write(f"    TN (True Negatives):  {result['stride_tn']:5d}\n")
            f.write(f"    TP (True Positives):  {result['stride_tp']:5d}\n")
            f.write(f"    FP (False Positives): {result['stride_fp']:5d}\n")
            f.write(f"    FN (False Negatives): {result['stride_fn']:5d}\n")
            f.write(f"    Total residues:       {result['stride_total']:5d}\n\n")
        else:
            f.write("Results vs STRIDE: No data available\n\n")
    
    # Quick Reference Table
    f.write("="*80 + "\n")
    f.write("QUICK REFERENCE TABLE - ALL PROTEINS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Protein':<12} {'DSSP Acc':<12} {'DSSP F1':<12} {'STRIDE Acc':<12} {'STRIDE F1':<12}\n")
    f.write("-" * 80 + "\n")
    
    for result in all_results:
        dssp_acc = f"{result['dssp_accuracy']:.4f}" if 'dssp_accuracy' in result else "N/A"
        dssp_f1 = f"{result['dssp_f1']:.4f}" if 'dssp_f1' in result else "N/A"
        stride_acc = f"{result['stride_accuracy']:.4f}" if 'stride_accuracy' in result else "N/A"
        stride_f1 = f"{result['stride_f1']:.4f}" if 'stride_f1' in result else "N/A"
        
        f.write(f"{result['protein_id']:<12} {dssp_acc:<12} {dssp_f1:<12} {stride_acc:<12} {stride_f1:<12}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print("\n" + "="*80)
print("✓ RESULTS OUTPUT FILE GENERATED")
print("="*80)
print(f"\nOutput file: {output_file}")
print(f"\nThis file contains:")
print(f"  ✓ Executive summary with mean accuracies")
print(f"  ✓ Aggregate confusion matrices (all proteins combined)")
print(f"  ✓ Per-protein detailed results")
print(f"  ✓ Quick reference table")
print(f"\nTotal proteins processed: {len(all_results)}")
print()


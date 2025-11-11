#!/usr/bin/env python3
"""
Generate Combined Confusion Matrix Reports
Creates readable reports showing both DSSP and STRIDE confusion matrices for each protein
"""

from pathlib import Path
import numpy as np
import pandas as pd
from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_dssp_data,
    extract_stride_data,
    add_neighbor_features,
    classify_burial
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from Bio.PDB import PDBParser
import re


def validate_pdb_file(pdb_path: Path) -> dict:
    """Validate if file is a proper PDB and extract metadata"""
    diagnostics = {
        'is_valid_pdb': False,
        'pdb_id': None,
        'has_header': False,
        'has_atom_records': False,
        'num_atoms': 0,
        'num_ca_atoms': 0,
        'file_size_kb': 0,
        'error': None
    }

    try:
        # Check file size
        diagnostics['file_size_kb'] = pdb_path.stat().st_size / 1024

        # Read and parse file
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        # Extract PDB ID from HEADER line
        for line in lines[:50]:  # Check first 50 lines
            if line.startswith('HEADER'):
                diagnostics['has_header'] = True
                # PDB ID is typically at positions 62-66
                if len(line) >= 66:
                    pdb_id = line[62:66].strip()
                    if pdb_id:
                        diagnostics['pdb_id'] = pdb_id.upper()
                break

        # If no PDB ID in header, try to extract from filename or other records
        if not diagnostics['pdb_id']:
            for line in lines[:100]:
                if line.startswith('COMPND') or line.startswith('SOURCE'):
                    match = re.search(r'\b([0-9][A-Z0-9]{3})\b', line)
                    if match:
                        diagnostics['pdb_id'] = match.group(1).upper()
                        break

        # Count ATOM records and CA atoms
        for line in lines:
            if line.startswith('ATOM'):
                diagnostics['has_atom_records'] = True
                diagnostics['num_atoms'] += 1
                if ' CA ' in line[12:16]:
                    diagnostics['num_ca_atoms'] += 1

        # Try to parse with BioPython as final validation
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', str(pdb_path))

        if diagnostics['has_atom_records'] and diagnostics['num_atoms'] > 0:
            diagnostics['is_valid_pdb'] = True

    except Exception as e:
        diagnostics['error'] = str(e)

    return diagnostics


def check_dssp_stride_files(pdb_path: Path) -> dict:
    """Check if DSSP and STRIDE output files exist"""
    stem = pdb_path.stem.lower()
    parent = pdb_path.parent

    return {
        'dssp_file': parent / f"{stem}.dssp",
        'stride_file': parent / f"{stem}.stride",
        'has_dssp': (parent / f"{stem}.dssp").exists(),
        'has_stride': (parent / f"{stem}.stride").exists(),
        'dssp_size': (parent / f"{stem}.dssp").stat().st_size if (parent / f"{stem}.dssp").exists() else 0,
        'stride_size': (parent / f"{stem}.stride").stat().st_size if (parent / f"{stem}.stride").exists() else 0,
    }


def process_protein(pdb_path: Path, params: BurialParameters):
    """Process a single protein and return metrics with diagnostics"""
    try:
        protein_id = pdb_path.stem
        
        # Validate PDB file
        pdb_validation = validate_pdb_file(pdb_path)

        # Check for DSSP/STRIDE files
        auxiliary_files = check_dssp_stride_files(pdb_path)

        # Extract CA atoms
        df = extract_ca_atoms(pdb_path)
        coords = df[['x', 'y', 'z']].values
        
        # Extract reference data
        df = extract_dssp_data(pdb_path, df, params.dssp_asa_cutoff)
        df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)
        
        # Add neighbor features
        df = add_neighbor_features(df, coords)
        
        # Classify using our algorithm
        df['ncps_class'] = classify_burial(df, params)
        
        # Count NCPS classifications
        ncps_interior = (df['ncps_class'] == 0).sum()
        ncps_exterior = (df['ncps_class'] == 1).sum()

        # Calculate DSSP metrics
        dssp_metrics = None
        dssp_mask = df['dssp_class'].notna()
        if dssp_mask.sum() > 0:
            y_true = df.loc[dssp_mask, 'dssp_class'].values.astype(int)
            y_pred = df.loc[dssp_mask, 'ncps_class'].values.astype(int)
            
            dssp_metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
        
        # Calculate STRIDE metrics
        stride_metrics = None
        stride_mask = df['stride_class'].notna()
        if stride_mask.sum() > 0:
            y_true = df.loc[stride_mask, 'stride_class'].values.astype(int)
            y_pred = df.loc[stride_mask, 'ncps_class'].values.astype(int)
            
            stride_metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
        
        return {
            'protein_id': protein_id,
            'n_residues': len(df),
            'dssp_metrics': dssp_metrics,
            'stride_metrics': stride_metrics,
            'pdb_validation': pdb_validation,
            'auxiliary_files': auxiliary_files,
            'ncps_summary': {
                'total_residues': len(df),
                'ncps_interior': ncps_interior,
                'ncps_exterior': ncps_exterior
            }
        }
    except Exception as e:
        print(f"  Error processing {pdb_path.stem}: {e}")
        return None


def save_individual_report(result, params: BurialParameters, output_dir: Path):
    """Save individual protein report with both confusion matrices"""
    protein_id = result['protein_id']
    report_file = output_dir / f"{protein_id}_confusion_matrices_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CONFUSION MATRICES FOR PROTEIN: {protein_id.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # PDB Validation Information
        pdb_val = result.get('pdb_validation', {})
        f.write("PDB FILE VALIDATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  File is valid PDB:     {'YES' if pdb_val.get('is_valid_pdb') else 'NO'}\n")
        f.write(f"  PDB ID from file:      {pdb_val.get('pdb_id', 'Not found')}\n")
        f.write(f"  Has HEADER record:     {'YES' if pdb_val.get('has_header') else 'NO'}\n")
        f.write(f"  Has ATOM records:      {'YES' if pdb_val.get('has_atom_records') else 'NO'}\n")
        f.write(f"  Total atoms:           {pdb_val.get('num_atoms', 0)}\n")
        f.write(f"  CA atoms:              {pdb_val.get('num_ca_atoms', 0)}\n")
        f.write(f"  File size:             {pdb_val.get('file_size_kb', 0):.2f} KB\n")
        if pdb_val.get('error'):
            f.write(f"  Error:                 {pdb_val.get('error')}\n")
        f.write("\n")

        # Auxiliary Files Information
        aux = result.get('auxiliary_files', {})
        f.write("DSSP/STRIDE OUTPUT FILES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  DSSP file exists:      {'YES' if aux.get('has_dssp') else 'NO'}\n")
        if aux.get('has_dssp'):
            f.write(f"    Location: {aux.get('dssp_file')}\n")
            f.write(f"    Size: {aux.get('dssp_size', 0)} bytes\n")
        else:
            f.write(f"    Expected: {aux.get('dssp_file')}\n")
            f.write(f"    ⚠ DSSP file missing - run 'mkdssp {protein_id}.pdb > {protein_id}.dssp'\n")
        f.write("\n")
        f.write(f"  STRIDE file exists:    {'YES' if aux.get('has_stride') else 'NO'}\n")
        if aux.get('has_stride'):
            f.write(f"    Location: {aux.get('stride_file')}\n")
            f.write(f"    Size: {aux.get('stride_size', 0)} bytes\n")
        else:
            f.write(f"    Expected: {aux.get('stride_file')}\n")
            f.write(f"    ⚠ STRIDE file missing - run 'stride {protein_id}.pdb > {protein_id}.stride'\n")
        f.write("\n")

        # Classification Parameters (RASA cutoffs)
        f.write("CLASSIFICATION PARAMETERS (RASA CUTOFFS):\n")
        f.write("-" * 80 + "\n")
        f.write(f"  DSSP ASA cutoff:       {params.dssp_asa_cutoff:.1f} Ų\n")
        f.write(f"    (Residues with ASA ≤ {params.dssp_asa_cutoff:.1f} Ų classified as Interior)\n")
        f.write(f"  STRIDE ASA cutoff:     {params.stride_asa_cutoff:.1f} Ų\n")
        f.write(f"    (Residues with ASA ≤ {params.stride_asa_cutoff:.1f} Ų classified as Interior)\n")
        f.write("\n")
        f.write(f"  NC6 threshold:         {params.nc6_threshold:.1f} neighbors\n")
        f.write(f"  NC10 threshold:        {params.nc10_threshold:.1f} neighbors\n")
        f.write(f"  UNI6 threshold:        {params.uni6_threshold:.2f}\n")
        f.write(f"  UNI10 threshold:       {params.uni10_threshold:.2f}\n")
        f.write("\n")

        f.write(f"Total Residues: {result['n_residues']}\n\n")
        
        # DSSP Confusion Matrix
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX vs DSSP (Ground Truth)\n")
        f.write("="*80 + "\n\n")
        
        if result['dssp_metrics']:
            cm = result['dssp_metrics']['confusion_matrix']
            f.write("Classification Key:\n")
            f.write("  0 = Interior (Buried)\n")
            f.write("  1 = Exterior (Surface/Exposed)\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {cm[0,0]:20d}  {cm[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {cm[1,0]:20d}  {cm[1,1]:20d}\n\n")
            
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            
            f.write("Metrics:\n")
            f.write(f"  Accuracy:  {result['dssp_metrics']['accuracy']:.4f} ({result['dssp_metrics']['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {result['dssp_metrics']['precision']:.4f}\n")
            f.write(f"  Recall:    {result['dssp_metrics']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {result['dssp_metrics']['f1']:.4f}\n\n")
            
            f.write("Breakdown:\n")
            f.write(f"  True Negatives (TN):  {tn:5d} - Correctly predicted as Interior\n")
            f.write(f"  False Positives (FP): {fp:5d} - Interior wrongly predicted as Exterior\n")
            f.write(f"  False Negatives (FN): {fn:5d} - Exterior wrongly predicted as Interior\n")
            f.write(f"  True Positives (TP):  {tp:5d} - Correctly predicted as Exterior\n\n")
        else:
            # No DSSP ground truth; print message and, if available, NCPS-only summary
            f.write("  No DSSP data available.\n")
            # If the caller attached classifier-only info, surface it here
            ncps_info = result.get('ncps_summary')
            if ncps_info is not None:
                total = ncps_info.get('total_residues')
                interior = ncps_info.get('ncps_interior')
                exterior = ncps_info.get('ncps_exterior')
                f.write("\n  NCPS classifier-only summary (no DSSP ground truth):\n")
                if total is not None:
                    f.write(f"    Total residues classified: {int(total)}\n")
                if interior is not None:
                    f.write(f"    Predicted Interior(0):     {int(interior)}\n")
                if exterior is not None:
                    f.write(f"    Predicted Exterior(1):     {int(exterior)}\n")
            f.write("\n")

        # STRIDE Confusion Matrix
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX vs STRIDE (Ground Truth)\n")
        f.write("="*80 + "\n\n")
        
        if result['stride_metrics']:
            cm = result['stride_metrics']['confusion_matrix']
            f.write("Classification Key:\n")
            f.write("  0 = Interior (Buried)\n")
            f.write("  1 = Exterior (Surface/Exposed)\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {cm[0,0]:20d}  {cm[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {cm[1,0]:20d}  {cm[1,1]:20d}\n\n")
            
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            
            f.write("Metrics:\n")
            f.write(f"  Accuracy:  {result['stride_metrics']['accuracy']:.4f} ({result['stride_metrics']['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {result['stride_metrics']['precision']:.4f}\n")
            f.write(f"  Recall:    {result['stride_metrics']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {result['stride_metrics']['f1']:.4f}\n\n")
            
            f.write("Breakdown:\n")
            f.write(f"  True Negatives (TN):  {tn:5d} - Correctly predicted as Interior\n")
            f.write(f"  False Positives (FP): {fp:5d} - Interior wrongly predicted as Exterior\n")
            f.write(f"  False Negatives (FN): {fn:5d} - Exterior wrongly predicted as Interior\n")
            f.write(f"  True Positives (TP):  {tp:5d} - Correctly predicted as Exterior\n\n")
        else:
            # No STRIDE ground truth; print message and reuse NCPS-only summary
            f.write("  No STRIDE data available.\n")
            ncps_info = result.get('ncps_summary')
            if ncps_info is not None:
                total = ncps_info.get('total_residues')
                interior = ncps_info.get('ncps_interior')
                exterior = ncps_info.get('ncps_exterior')
                f.write("\n  NCPS classifier-only summary (no STRIDE ground truth):\n")
                if total is not None:
                    f.write(f"    Total residues classified: {int(total)}\n")
                if interior is not None:
                    f.write(f"    Predicted Interior(0):     {int(interior)}\n")
                if exterior is not None:
                    f.write(f"    Predicted Exterior(1):     {int(exterior)}\n")
            f.write("\n")

        # Comparison
        if result['dssp_metrics'] and result['stride_metrics']:
            f.write("="*80 + "\n")
            f.write("COMPARISON: DSSP vs STRIDE\n")
            f.write("="*80 + "\n\n")
            dssp_acc = result['dssp_metrics']['accuracy']
            stride_acc = result['stride_metrics']['accuracy']
            f.write(f"DSSP Accuracy:   {dssp_acc:.4f} ({dssp_acc*100:.2f}%)\n")
            f.write(f"STRIDE Accuracy: {stride_acc:.4f} ({stride_acc*100:.2f}%)\n")
            f.write(f"Difference:      {abs(dssp_acc - stride_acc):.4f}\n\n")
            
            if dssp_acc > stride_acc:
                f.write("→ Better agreement with DSSP\n")
            elif stride_acc > dssp_acc:
                f.write("→ Better agreement with STRIDE\n")
            else:
                f.write("→ Equal agreement with both methods\n")


def save_master_summary(results, output_dir: Path):
    """Save master summary with all proteins"""
    master_file = output_dir / "ALL_PROTEINS_confusion_matrices_summary.txt"
    
    with open(master_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MASTER SUMMARY: ALL PROTEINS CONFUSION MATRICES\n")
        f.write("="*80 + "\n\n")
        
        total_proteins = len(results)
        total_residues = sum(r['n_residues'] for r in results)
        
        f.write(f"Total Proteins Analyzed: {total_proteins}\n")
        f.write(f"Total Residues: {total_residues}\n\n")
        
        # DSSP Summary
        f.write("="*80 + "\n")
        f.write("AGGREGATE RESULTS vs DSSP\n")
        f.write("="*80 + "\n\n")
        
        dssp_results = [r for r in results if r['dssp_metrics']]
        if dssp_results:
            total_cm = sum(r['dssp_metrics']['confusion_matrix'] for r in dssp_results)
            accuracies = [r['dssp_metrics']['accuracy'] for r in dssp_results]
            
            f.write(f"Proteins with DSSP data: {len(dssp_results)}\n\n")
            f.write("Aggregate Confusion Matrix:\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {total_cm[0,0]:20d}  {total_cm[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {total_cm[1,0]:20d}  {total_cm[1,1]:20d}\n\n")
            
            f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
            f.write(f"Min Accuracy:  {np.min(accuracies):.4f}\n")
            f.write(f"Max Accuracy:  {np.max(accuracies):.4f}\n\n")
        
        # STRIDE Summary
        f.write("="*80 + "\n")
        f.write("AGGREGATE RESULTS vs STRIDE\n")
        f.write("="*80 + "\n\n")
        
        stride_results = [r for r in results if r['stride_metrics']]
        if stride_results:
            total_cm = sum(r['stride_metrics']['confusion_matrix'] for r in stride_results)
            accuracies = [r['stride_metrics']['accuracy'] for r in stride_results]
            
            f.write(f"Proteins with STRIDE data: {len(stride_results)}\n\n")
            f.write("Aggregate Confusion Matrix:\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {total_cm[0,0]:20d}  {total_cm[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {total_cm[1,0]:20d}  {total_cm[1,1]:20d}\n\n")
            
            f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
            f.write(f"Min Accuracy:  {np.min(accuracies):.4f}\n")
            f.write(f"Max Accuracy:  {np.max(accuracies):.4f}\n\n")
        
        # Per-protein table
        f.write("="*80 + "\n")
        f.write("PER-PROTEIN ACCURACY TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Protein ID':<15} {'Residues':>10} {'DSSP Acc':>12} {'STRIDE Acc':>12}\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            dssp_acc = f"{r['dssp_metrics']['accuracy']:.4f}" if r['dssp_metrics'] else "N/A"
            stride_acc = f"{r['stride_metrics']['accuracy']:.4f}" if r['stride_metrics'] else "N/A"
            f.write(f"{r['protein_id']:<15} {r['n_residues']:>10} {dssp_acc:>12} {stride_acc:>12}\n")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("GENERATING COMBINED CONFUSION MATRIX REPORTS")
    print("="*80 + "\n")
    
    workspace = Path.cwd()
    output_dir = workspace / "results" / "confusion_matrix_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDB files
    pdb_files = sorted(workspace.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files\n")
    
    # Default parameters
    params = BurialParameters()
    
    # Process all proteins
    results = []
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"Processing {i}/{len(pdb_files)}: {pdb_file.stem}...")
        result = process_protein(pdb_file, params)
        if result:
            results.append(result)
            save_individual_report(result, params, output_dir)

    # Save master summary
    save_master_summary(results, output_dir)
    
    print("\n" + "="*80)
    print("✓ REPORTS GENERATED")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - {len(results)} individual protein reports (*_confusion_matrices_report.txt)")
    print(f"  - 1 master summary (ALL_PROTEINS_confusion_matrices_summary.txt)")
    print()


if __name__ == "__main__":
    main()


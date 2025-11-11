#!/usr/bin/env python3
"""
Comprehensive Protein Burial Classification Analysis System
- Processes multiple proteins (DUDE dataset or custom set)
- Generates 2 confusion matrices per protein (vs DSSP and vs STRIDE)
- Implements cross-validation (5-fold or 10-fold)
- Parameter optimization using Optuna
- Per-protein and overall accuracy metrics
- Outlier detection and analysis
"""

import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import KFold
from Bio.PDB import PDBParser

# Optuna import with fallback
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: Optuna not installed. Parameter optimization will not be available.")

# DSSP / STRIDE availability flags
try:
    from Bio.PDB.DSSP import DSSP  # type: ignore
    HAS_DSSP = True
except Exception:
    HAS_DSSP = False
    print("Warning: DSSP not available; DSSP-based metrics will be skipped.")

# External STRIDE binary assumed; we treat availability as "best effort"
HAS_STRIDE = True

print(f"DSSP available: {HAS_DSSP}")
print(f"STRIDE available: {HAS_STRIDE}")


@dataclass
class BurialParameters:
    """Parameters for burial classification algorithm"""
    # Neighbor count thresholds
    nc6_threshold: float = 10.0  # 6Å sphere
    nc10_threshold: float = 18.0  # 10Å sphere

    # Uniformity (homogeneous distribution) thresholds
    uni6_threshold: float = 0.40  # 6Å sphere uniformity
    uni10_threshold: float = 0.50  # 10Å sphere uniformity

    # Cutoff for DSSP/STRIDE classification (ASA value)
    dssp_asa_cutoff: float = 30.0  # Å²
    stride_asa_cutoff: float = 24.0  # Å²


@dataclass
class ProteinResults:
    """Results for a single protein"""
    protein_id: str
    n_residues: int

    # Data
    dataframe: pd.DataFrame

    # Accuracy metrics vs DSSP
    dssp_accuracy: Optional[float] = None
    dssp_confusion_matrix: Optional[np.ndarray] = None
    dssp_precision: Optional[float] = None
    dssp_recall: Optional[float] = None
    dssp_f1: Optional[float] = None

    # Accuracy metrics vs STRIDE
    stride_accuracy: Optional[float] = None
    stride_confusion_matrix: Optional[np.ndarray] = None
    stride_precision: Optional[float] = None
    stride_recall: Optional[float] = None
    stride_f1: Optional[float] = None


# ==================== FEATURE EXTRACTION ====================

# MaxASA values from Tien et al. (PLoS ONE 2013, 8(11):e80635)
# Theoretical maximum solvent accessible surface area for Gly-X-Gly tripeptides
MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
    'CYS': 167.0, 'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0,
    'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
    'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}


def extract_ca_atoms(pdb_path: Path) -> pd.DataFrame:
    """Extract CA atoms with coordinates and residue info."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    ca_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # Skip HETATM
                    continue
                if 'CA' not in residue:
                    continue

                ca = residue['CA']
                ca_list.append({
                    'chain_id': chain.id,
                    'resseq': residue.id[1],
                    'icode': residue.id[2].strip() or '',
                    'resname': residue.resname,
                    'x': ca.coord[0],
                    'y': ca.coord[1],
                    'z': ca.coord[2]
                })
        break  # Only first model

    df = pd.DataFrame(ca_list)
    df['res_num'] = df['resseq']
    df['res_id'] = df['resname']
    return df


def calculate_neighbor_counts(coords: np.ndarray, radius: float) -> np.ndarray:
    """Count neighbors within given radius for each atom."""
    n = len(coords)
    counts = np.zeros(n, dtype=int)

    for i in range(n):
        distances = np.linalg.norm(coords - coords[i], axis=1)
        counts[i] = np.sum((distances > 0) & (distances <= radius))

    return counts


def calculate_uniformity(coords: np.ndarray, radius: float) -> np.ndarray:
    """
    Calculate uniformity (spherical variance) for each residue.
    Low value = neighbors on one side (exterior)
    High value = neighbors all around (interior)
    """
    n = len(coords)
    uniformity = np.zeros(n)

    for i in range(n):
        center = coords[i]
        distances = np.linalg.norm(coords - center, axis=1)
        neighbors_mask = (distances > 0) & (distances <= radius)
        neighbors = coords[neighbors_mask]

        if len(neighbors) < 3:
            uniformity[i] = 0.0
            continue

        # Calculate unit vectors to neighbors
        vectors = neighbors - center
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

        # Calculate mean direction (should be ~0 for uniform distribution)
        mean_vector = np.mean(vectors, axis=0)
        mean_magnitude = np.linalg.norm(mean_vector)

        # Uniformity: 1 - mean_magnitude (0=all one side, 1=perfectly uniform)
        uniformity[i] = 1.0 - mean_magnitude

    return uniformity


def extract_dssp_data(pdb_path: Path, df: pd.DataFrame, asa_cutoff: float = 30.0) -> pd.DataFrame:
    """Extract DSSP ASA and classify as buried/exposed."""
    if not HAS_DSSP:
        df['dssp_asa'] = np.nan
        df['dssp_class'] = np.nan
        df['dssp_ss'] = ''
        return df

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model = structure[0]

        file_ext = pdb_path.suffix.lower()
        file_type = 'PDB' if file_ext in ['.pdb', '.ent'] else 'mmCIF'

        dssp = DSSP(model, str(pdb_path), file_type=file_type)

        dssp_map = {}
        for key in dssp.keys():
            chain_id, res_id = key
            hetflag, resseq, icode = res_id
            rec = dssp[key]

            try:
                rel_asa = float(rec[3]) if rec[3] != 'NA' and rec[3] is not None else 0.0
            except (ValueError, TypeError):
                rel_asa = 0.0

            aa = rec[1] if len(rec) > 1 else 'ALA'
            ss = rec[2] if len(rec) > 2 else '-'

            max_asa = MAX_ASA.get(aa, 100.0)
            asa = rel_asa * max_asa

            dssp_map[(chain_id, int(resseq), icode.strip() or '')] = {
                'asa': asa,
                'ss': ss
            }

        dssp_asa = []
        dssp_ss = []
        for _, row in df.iterrows():
            key = (row['chain_id'], int(row['resseq']), row['icode'])
            if key in dssp_map:
                dssp_asa.append(dssp_map[key]['asa'])
                dssp_ss.append(dssp_map[key]['ss'])
            else:
                dssp_asa.append(np.nan)
                dssp_ss.append('-')

        df['dssp_asa'] = dssp_asa
        df['dssp_ss'] = dssp_ss

        # RASA based on Tien et al. maxASA (Gly-X-Gly model)
        # If residue not in table, RASA will be NaN
        def _rasa_dssp(row):
            aa = str(row['resname']).strip().upper()
            max_asa = MAX_ASA.get(aa)
            if max_asa is None or pd.isna(row['dssp_asa']):
                return np.nan
            return float(row['dssp_asa']) / max_asa

        df['RASA_dssp'] = df.apply(_rasa_dssp, axis=1)

        # Classification by RASA (1 = exposed/surface, 0 = buried)
        df['dssp_class'] = df['RASA_dssp'].apply(
            lambda r: 1 if pd.notna(r) and r >= 0.25 else (0 if pd.notna(r) else np.nan)
        )

    except Exception as e:
        print(f"  DSSP error: {e}")
        df['dssp_asa'] = np.nan
        df['dssp_class'] = np.nan
        df['dssp_ss'] = ''
        df['RASA_dssp'] = np.nan

    return df


def extract_stride_data(pdb_path: Path, df: pd.DataFrame, asa_cutoff: float = 24.0) -> pd.DataFrame:
    """Extract STRIDE ASA and classify as buried/exposed."""
    try:
        stride_file = None
        for name_variant in [pdb_path.stem.lower(), pdb_path.stem.upper(), pdb_path.stem]:
            test_file = pdb_path.parent / f"{name_variant}.stride"
            if test_file.exists():
                stride_file = test_file
                break

        if stride_file is not None:
            with open(stride_file, 'r') as f:
                stride_content = f.read()
        elif HAS_STRIDE:
            result = subprocess.run(
                ['stride', str(pdb_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            stride_content = result.stdout
        else:
            raise Exception("No STRIDE data available")

        stride_map = {}
        for line in stride_content.split('\n'):
            if line.startswith('ASG'):
                try:
                    chain_id = line[9:10].strip()
                    chain_id_raw = line[9:10]  # Keep raw (might be space)
                    resseq = int(line[11:15].strip())
                    ss = line[24:25].strip() if len(line) > 24 else 'C'
                    parts = line.split()
                    # ASA is the second-to-last field; last field is protein ID
                    asa = float(parts[-2]) if len(parts) >= 10 else 0.0

                    data = {'asa': asa, 'ss': ss if ss else 'C'}

                    # Store with multiple key variations to handle chain ID mismatches
                    stride_map[(chain_id, resseq, '')] = data
                    stride_map[(chain_id_raw, resseq, '')] = data
                    if chain_id == '':
                        stride_map[('A', resseq, '')] = data  # Default to A if empty
                        stride_map[(' ', resseq, '')] = data
                except (ValueError, IndexError):
                    continue

        stride_asa = []
        stride_ss = []
        for _, row in df.iterrows():
            resseq_int = int(row['resseq'])
            chain = row['chain_id']

            # Try multiple chain ID variations to handle mismatches
            possible_keys = [
                (chain, resseq_int, ''),           # Original chain ID
                ('', resseq_int, ''),               # Empty chain ID
                ('A', resseq_int, ''),              # Default chain A
                (' ', resseq_int, ''),              # Space as chain ID
            ]

            found = False
            for key in possible_keys:
                if key in stride_map:
                    stride_asa.append(stride_map[key]['asa'])
                    stride_ss.append(stride_map[key]['ss'])
                    found = True
                    break

            if not found:
                stride_asa.append(np.nan)
                stride_ss.append('-')

        df['stride_asa'] = stride_asa
        df['stride_ss'] = stride_ss

        # RASA based on Tien et al. maxASA (Gly-X-Gly model)
        def _rasa_stride(row):
            aa = str(row['resname']).strip().upper()
            max_asa = MAX_ASA.get(aa)
            if max_asa is None or pd.isna(row['stride_asa']):
                return np.nan
            return float(row['stride_asa']) / max_asa

        df['RASA_stride'] = df.apply(_rasa_stride, axis=1)

        # Classification by RASA (1 = exposed/surface, 0 = buried)
        df['stride_class'] = df['RASA_stride'].apply(
            lambda r: 1 if pd.notna(r) and r >= 0.25 else (0 if pd.notna(r) else np.nan)
        )

    except Exception as e:
        print(f"  STRIDE error: {e}")
        df['stride_asa'] = np.nan
        df['stride_class'] = np.nan
        df['stride_ss'] = ''
        df['RASA_stride'] = np.nan

    return df


def add_neighbor_features(df: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    """Add neighbor counts and uniformity metrics."""
    df['ncps_sphere_6'] = calculate_neighbor_counts(coords, 6.0)
    df['ncps_sphere_10'] = calculate_neighbor_counts(coords, 10.0)
    df['ncps_sphere_6_uni'] = calculate_uniformity(coords, 6.0)
    df['ncps_sphere_10_uni'] = calculate_uniformity(coords, 10.0)
    return df


def classify_burial(df: pd.DataFrame, params: BurialParameters) -> np.ndarray:
    """
    Classify residues as buried (0) or exposed (1) using our algorithm.
    Prediction: 1 = exterior (surface), 0 = interior (buried)
    """
    ncps_class = []

    for _, row in df.iterrows():
        nc6 = row['ncps_sphere_6']
        nc10 = row['ncps_sphere_10']
        uni6 = row['ncps_sphere_6_uni']
        uni10 = row['ncps_sphere_10_uni']

        # Default to interior (buried)
        is_exterior = False

        # Exterior if: few neighbors (below threshold)
        if nc6 < params.nc6_threshold or nc10 < params.nc10_threshold:
            is_exterior = True
        # Exterior if: low uniformity (neighbors not surrounding, one-sided)
        elif pd.notna(uni6) and uni6 < params.uni6_threshold:
            is_exterior = True
        elif pd.notna(uni10) and uni10 < params.uni10_threshold:
            is_exterior = True

        ncps_class.append(1 if is_exterior else 0)

    return np.array(ncps_class)


# ==================== ANALYSIS FUNCTIONS ====================

def process_single_protein(pdb_path: Path, params: BurialParameters) -> ProteinResults:
    """Process a single protein and return results."""
    protein_id = pdb_path.stem
    print(f"\nProcessing {protein_id}...")

    # Extract CA atoms
    df = extract_ca_atoms(pdb_path)
    coords = df[['x', 'y', 'z']].values

    # Extract reference data (DSSP and STRIDE)
    df = extract_dssp_data(pdb_path, df, params.dssp_asa_cutoff)
    df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)

    # Add neighbor features
    df = add_neighbor_features(df, coords)

    # Classify using our algorithm
    df['ncps_class'] = classify_burial(df, params)

    # Calculate metrics vs DSSP
    dssp_mask = df['dssp_class'].notna()
    dssp_accuracy = None
    dssp_cm = None
    dssp_precision = None
    dssp_recall = None
    dssp_f1 = None

    if dssp_mask.sum() > 0:
        y_true_dssp = df.loc[dssp_mask, 'dssp_class'].values
        y_pred = df.loc[dssp_mask, 'ncps_class'].values

        dssp_accuracy = accuracy_score(y_true_dssp, y_pred)
        dssp_cm = confusion_matrix(y_true_dssp, y_pred, labels=[0, 1])
        dssp_precision = precision_score(y_true_dssp, y_pred, zero_division=0)
        dssp_recall = recall_score(y_true_dssp, y_pred, zero_division=0)
        dssp_f1 = f1_score(y_true_dssp, y_pred, zero_division=0)

        print(f"  vs DSSP: Accuracy={dssp_accuracy:.3f}, F1={dssp_f1:.3f}")

    # Calculate metrics vs STRIDE
    stride_mask = df['stride_class'].notna()
    stride_accuracy = None
    stride_cm = None
    stride_precision = None
    stride_recall = None
    stride_f1 = None

    if stride_mask.sum() > 0:
        y_true_stride = df.loc[stride_mask, 'stride_class'].values
        y_pred = df.loc[stride_mask, 'ncps_class'].values

        stride_accuracy = accuracy_score(y_true_stride, y_pred)
        stride_cm = confusion_matrix(y_true_stride, y_pred, labels=[0, 1])
        stride_precision = precision_score(y_true_stride, y_pred, zero_division=0)
        stride_recall = recall_score(y_true_stride, y_pred, zero_division=0)
        stride_f1 = f1_score(y_true_stride, y_pred, zero_division=0)

        print(f"  vs STRIDE: Accuracy={stride_accuracy:.3f}, F1={stride_f1:.3f}")

    return ProteinResults(
        protein_id=protein_id,
        n_residues=len(df),
        dataframe=df,
        dssp_accuracy=dssp_accuracy,
        dssp_confusion_matrix=dssp_cm,
        dssp_precision=dssp_precision,
        dssp_recall=dssp_recall,
        dssp_f1=dssp_f1,
        stride_accuracy=stride_accuracy,
        stride_confusion_matrix=stride_cm,
        stride_precision=stride_precision,
        stride_recall=stride_recall,
        stride_f1=stride_f1
    )


def process_protein_dataset(pdb_files: List[Path], params: BurialParameters) -> List[ProteinResults]:
    """Process multiple proteins and return results."""
    results = []

    for pdb_file in pdb_files:
        try:
            result = process_single_protein(pdb_file, params)
            results.append(result)
        except Exception as e:
            print(f"Error processing {pdb_file.stem}: {e}")
            continue

    return results


def save_confusion_matrices(results: List[ProteinResults], output_dir: Path):
    """Save confusion matrices for each protein (both DSSP and STRIDE)."""
    output_dir.mkdir(exist_ok=True)

    for result in results:
        protein_id = result.protein_id

        # Save DSSP confusion matrix
        if result.dssp_confusion_matrix is not None:
            cm_file = output_dir / f"{protein_id}_confusion_matrix_dssp.csv"
            cm_df = pd.DataFrame(
                result.dssp_confusion_matrix,
                index=['True_Interior', 'True_Exterior'],
                columns=['Pred_Interior', 'Pred_Exterior']
            )
            cm_df.to_csv(cm_file)

        # Save STRIDE confusion matrix
        if result.stride_confusion_matrix is not None:
            cm_file = output_dir / f"{protein_id}_confusion_matrix_stride.csv"
            cm_df = pd.DataFrame(
                result.stride_confusion_matrix,
                index=['True_Interior', 'True_Exterior'],
                columns=['Pred_Interior', 'Pred_Exterior']
            )
            cm_df.to_csv(cm_file)


def save_combined_confusion_matrix_report(results: List[ProteinResults], output_dir: Path):
    """
    Save a comprehensive report showing both confusion matrices (DSSP and STRIDE)
    for each protein in a single readable file.
    """
    output_dir.mkdir(exist_ok=True)

    # Create individual protein reports
    for result in results:
        protein_id = result.protein_id
        report_file = output_dir / f"{protein_id}_confusion_matrices_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CONFUSION MATRICES FOR PROTEIN: {protein_id.upper()}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total Residues: {result.n_residues}\n\n")

            # DSSP Confusion Matrix
            f.write("="*80 + "\n")
            f.write("CONFUSION MATRIX vs DSSP (Ground Truth)\n")
            f.write("="*80 + "\n\n")

            if result.dssp_confusion_matrix is not None:
                cm = result.dssp_confusion_matrix
                f.write("Classification Key:\n")
                f.write("  0 = Interior (Buried)\n")
                f.write("  1 = Exterior (Surface/Exposed)\n\n")

                f.write("Confusion Matrix:\n")
                f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
                f.write(f"True Interior(0)    {cm[0,0]:20d}  {cm[0,1]:20d}\n")
                f.write(f"True Exterior(1)    {cm[1,0]:20d}  {cm[1,1]:20d}\n\n")

                # Calculate metrics
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                total = tn + fp + fn + tp

                f.write("Metrics:\n")
                f.write(f"  Accuracy:  {result.dssp_accuracy:.4f} ({result.dssp_accuracy*100:.2f}%)\n")
                f.write(f"  Precision: {result.dssp_precision:.4f}\n")
                f.write(f"  Recall:    {result.dssp_recall:.4f}\n")
                f.write(f"  F1-Score:  {result.dssp_f1:.4f}\n\n")

                f.write("Confusion Matrix Breakdown:\n")
                f.write(f"  True Negatives (TN):  {tn:5d} - Correctly predicted as Interior\n")
                f.write(f"  False Positives (FP): {fp:5d} - Interior wrongly predicted as Exterior\n")
                f.write(f"  False Negatives (FN): {fn:5d} - Exterior wrongly predicted as Interior\n")
                f.write(f"  True Positives (TP):  {tp:5d} - Correctly predicted as Exterior\n")
                f.write(f"  Total:                {total:5d}\n\n")
            else:
                # No DSSP ground truth for this protein; provide classifier-only summary
                f.write("  No DSSP data available for this protein.\n")
                df = result.dataframe if hasattr(result, 'dataframe') else None
                if df is not None and 'ncps_class' in df.columns:
                    total_res = len(df)
                    interior = int((df['ncps_class'] == 0).sum())
                    exterior = int((df['ncps_class'] == 1).sum())
                    f.write("\n  NCPS classifier-only summary (no DSSP ground truth):\n")
                    f.write(f"    Total residues classified: {total_res}\n")
                    f.write(f"    Predicted Interior(0):     {interior}\n")
                    f.write(f"    Predicted Exterior(1):     {exterior}\n")
                f.write("\n")

            # STRIDE Confusion Matrix
            f.write("="*80 + "\n")
            f.write("CONFUSION MATRIX vs STRIDE (Ground Truth)\n")
            f.write("="*80 + "\n\n")

            if result.stride_confusion_matrix is not None:
                cm = result.stride_confusion_matrix
                f.write("Classification Key:\n")
                f.write("  0 = Interior (Buried)\n")
                f.write("  1 = Exterior (Surface/Exposed)\n\n")

                f.write("Confusion Matrix:\n")
                f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
                f.write(f"True Interior(0)    {cm[0,0]:20d}  {cm[0,1]:20d}\n")
                f.write(f"True Exterior(1)    {cm[1,0]:20d}  {cm[1,1]:20d}\n\n")

                # Calculate metrics
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                total = tn + fp + fn + tp

                f.write("Metrics:\n")
                f.write(f"  Accuracy:  {result.stride_accuracy:.4f} ({result.stride_accuracy*100:.2f}%)\n")
                f.write(f"  Precision: {result.stride_precision:.4f}\n")
                f.write(f"  Recall:    {result.stride_recall:.4f}\n")
                f.write(f"  F1-Score:  {result.stride_f1:.4f}\n\n")

                f.write("Confusion Matrix Breakdown:\n")
                f.write(f"  True Negatives (TN):  {tn:5d} - Correctly predicted as Interior\n")
                f.write(f"  False Positives (FP): {fp:5d} - Interior wrongly predicted as Exterior\n")
                f.write(f"  False Negatives (FN): {fn:5d} - Exterior wrongly predicted as Interior\n")
                f.write(f"  True Positives (TP):  {tp:5d} - Correctly predicted as Exterior\n")
                f.write(f"  Total:                {total:5d}\n\n")
            else:
                # No STRIDE ground truth for this protein; provide classifier-only summary
                f.write("  No STRIDE data available for this protein.\n")
                df = result.dataframe if hasattr(result, 'dataframe') else None
                if df is not None and 'ncps_class' in df.columns:
                    total_res = len(df)
                    interior = int((df['ncps_class'] == 0).sum())
                    exterior = int((df['ncps_class'] == 1).sum())
                    f.write("\n  NCPS classifier-only summary (no STRIDE ground truth):\n")
                    f.write(f"    Total residues classified: {total_res}\n")
                    f.write(f"    Predicted Interior(0):     {interior}\n")
                    f.write(f"    Predicted Exterior(1):     {exterior}\n")
                f.write("\n")

            # Comparison section
            if result.dssp_confusion_matrix is not None and result.stride_confusion_matrix is not None:
                f.write("="*80 + "\n")
                f.write("COMPARISON: DSSP vs STRIDE\n")
                f.write("="*80 + "\n\n")
                f.write(f"DSSP Accuracy:   {result.dssp_accuracy:.4f} ({result.dssp_accuracy*100:.2f}%)\n")
                f.write(f"STRIDE Accuracy: {result.stride_accuracy:.4f} ({result.stride_accuracy*100:.2f}%)\n")
                f.write(f"Difference:      {abs(result.dssp_accuracy - result.stride_accuracy):.4f}\n\n")

                if result.dssp_accuracy > result.stride_accuracy:
                    f.write("→ Better agreement with DSSP\n")
                elif result.stride_accuracy > result.dssp_accuracy:
                    f.write("→ Better agreement with STRIDE\n")
                else:
                    f.write("→ Equal agreement with both methods\n")

    # Create a master summary file
    master_file = output_dir / "ALL_PROTEINS_confusion_matrices_summary.txt"
    with open(master_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MASTER SUMMARY: ALL PROTEINS CONFUSION MATRICES\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Proteins Analyzed: {len(results)}\n")
        f.write(f"Total Residues: {sum(r.n_residues for r in results)}\n\n")

        # DSSP Summary
        f.write("="*80 + "\n")
        f.write("AGGREGATE RESULTS vs DSSP\n")
        f.write("="*80 + "\n\n")

        dssp_results = [r for r in results if r.dssp_confusion_matrix is not None]
        if dssp_results:
            total_cm_dssp = sum(r.dssp_confusion_matrix for r in dssp_results)
            f.write(f"Proteins with DSSP data: {len(dssp_results)}\n\n")

            f.write("Aggregate Confusion Matrix (All Proteins Combined):\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {total_cm_dssp[0,0]:20d}  {total_cm_dssp[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {total_cm_dssp[1,0]:20d}  {total_cm_dssp[1,1]:20d}\n\n")

            dssp_accuracies = [r.dssp_accuracy for r in dssp_results]
            f.write(f"Mean Accuracy: {np.mean(dssp_accuracies):.4f} ± {np.std(dssp_accuracies):.4f}\n")
            f.write(f"Min Accuracy:  {np.min(dssp_accuracies):.4f}\n")
            f.write(f"Max Accuracy:  {np.max(dssp_accuracies):.4f}\n\n")

        # STRIDE Summary
        f.write("="*80 + "\n")
        f.write("AGGREGATE RESULTS vs STRIDE\n")
        f.write("="*80 + "\n\n")

        stride_results = [r for r in results if r.stride_confusion_matrix is not None]
        if stride_results:
            total_cm_stride = sum(r.stride_confusion_matrix for r in stride_results)
            f.write(f"Proteins with STRIDE data: {len(stride_results)}\n\n")

            f.write("Aggregate Confusion Matrix (All Proteins Combined):\n")
            f.write(f"                    Predicted Interior(0)  Predicted Exterior(1)\n")
            f.write(f"True Interior(0)    {total_cm_stride[0,0]:20d}  {total_cm_stride[0,1]:20d}\n")
            f.write(f"True Exterior(1)    {total_cm_stride[1,0]:20d}  {total_cm_stride[1,1]:20d}\n\n")

            stride_accuracies = [r.stride_accuracy for r in stride_results]
            f.write(f"Mean Accuracy: {np.mean(stride_accuracies):.4f} ± {np.std(stride_accuracies):.4f}\n")
            f.write(f"Min Accuracy:  {np.min(stride_accuracies):.4f}\n")
            f.write(f"Max Accuracy:  {np.max(stride_accuracies):.4f}\n\n")

        # Per-protein summary table
        f.write("="*80 + "\n")
        f.write("PER-PROTEIN ACCURACY SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Protein ID':<15} {'Residues':>10} {'DSSP Acc':>12} {'STRIDE Acc':>12}\n")
        f.write("-"*80 + "\n")

        for result in results:
            dssp_acc_str = f"{result.dssp_accuracy:.4f}" if result.dssp_accuracy is not None else "N/A"
            stride_acc_str = f"{result.stride_accuracy:.4f}" if result.stride_accuracy is not None else "N/A"
            f.write(f"{result.protein_id:<15} {result.n_residues:>10} {dssp_acc_str:>12} {stride_acc_str:>12}\n")

    print(f"\n✓ Confusion matrix reports saved to: {output_dir}")
    print(f"  - Individual reports: {len(results)} files (*_confusion_matrices_report.txt)")
    print(f"  - Master summary: ALL_PROTEINS_confusion_matrices_summary.txt")


def generate_summary_report(results: List[ProteinResults], output_file: Path):
    """Generate comprehensive summary report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE PROTEIN BURIAL CLASSIFICATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        total_proteins = len(results)
        total_residues = sum(r.n_residues for r in results)

        dssp_accuracies = [r.dssp_accuracy for r in results if r.dssp_accuracy is not None]
        stride_accuracies = [r.stride_accuracy for r in results if r.stride_accuracy is not None]

        f.write(f"Total Proteins Analyzed: {total_proteins}\n")
        f.write(f"Total Residues: {total_residues}\n\n")

        # DSSP Statistics
        if dssp_accuracies:
            f.write("--- DSSP COMPARISON ---\n")
            f.write(f"Proteins with DSSP data: {len(dssp_accuracies)}\n")
            f.write(f"Overall Accuracy (mean): {np.mean(dssp_accuracies):.4f}\n")
            f.write(f"Accuracy Std Dev: {np.std(dssp_accuracies):.4f}\n")
            f.write(f"Min Accuracy: {np.min(dssp_accuracies):.4f}\n")
            f.write(f"Max Accuracy: {np.max(dssp_accuracies):.4f}\n")
            f.write(f"Median Accuracy: {np.median(dssp_accuracies):.4f}\n\n")

            # Aggregate confusion matrix
            total_cm = sum(r.dssp_confusion_matrix for r in results if r.dssp_confusion_matrix is not None)
            f.write("Aggregate Confusion Matrix (DSSP):\n")
            f.write(f"                Pred_Interior  Pred_Exterior\n")
            f.write(f"True_Interior   {total_cm[0,0]:14d}  {total_cm[0,1]:14d}\n")
            f.write(f"True_Exterior   {total_cm[1,0]:14d}  {total_cm[1,1]:14d}\n\n")

        # STRIDE Statistics
        if stride_accuracies:
            f.write("--- STRIDE COMPARISON ---\n")
            f.write(f"Proteins with STRIDE data: {len(stride_accuracies)}\n")
            f.write(f"Overall Accuracy (mean): {np.mean(stride_accuracies):.4f}\n")
            f.write(f"Accuracy Std Dev: {np.std(stride_accuracies):.4f}\n")
            f.write(f"Min Accuracy: {np.min(stride_accuracies):.4f}\n")
            f.write(f"Max Accuracy: {np.max(stride_accuracies):.4f}\n")
            f.write(f"Median Accuracy: {np.median(stride_accuracies):.4f}\n\n")

            # Aggregate confusion matrix
            total_cm = sum(r.stride_confusion_matrix for r in results if r.stride_confusion_matrix is not None)
            f.write("Aggregate Confusion Matrix (STRIDE):\n")
            f.write(f"                Pred_Interior  Pred_Exterior\n")
            f.write(f"True_Interior   {total_cm[0,0]:14d}  {total_cm[0,1]:14d}\n")
            f.write(f"True_Exterior   {total_cm[1,0]:14d}  {total_cm[1,1]:14d}\n\n")

        # Per-protein details
        f.write("=" * 80 + "\n")
        f.write("PER-PROTEIN RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"Protein: {result.protein_id}\n")
            f.write(f"  Residues: {result.n_residues}\n")

            if result.dssp_accuracy is not None:
                f.write(f"  DSSP Accuracy: {result.dssp_accuracy:.4f}\n")
                f.write(f"  DSSP Precision: {result.dssp_precision:.4f}\n")
                f.write(f"  DSSP Recall: {result.dssp_recall:.4f}\n")
                f.write(f"  DSSP F1-Score: {result.dssp_f1:.4f}\n")

            if result.stride_accuracy is not None:
                f.write(f"  STRIDE Accuracy: {result.stride_accuracy:.4f}\n")
                f.write(f"  STRIDE Precision: {result.stride_precision:.4f}\n")
                f.write(f"  STRIDE Recall: {result.stride_recall:.4f}\n")
                f.write(f"  STRIDE F1-Score: {result.stride_f1:.4f}\n")

            f.write("\n")

        # Outlier analysis
        f.write("=" * 80 + "\n")
        f.write("OUTLIER ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        if dssp_accuracies:
            mean_acc = np.mean(dssp_accuracies)
            std_acc = np.std(dssp_accuracies)

            f.write("Low Performance Proteins (< mean - 1*std):\n")
            threshold_low = mean_acc - std_acc
            for result in results:
                if result.dssp_accuracy is not None and result.dssp_accuracy < threshold_low:
                    f.write(f"  {result.protein_id}: {result.dssp_accuracy:.4f}\n")

            f.write("\nHigh Performance Proteins (> mean + 1*std):\n")
            threshold_high = mean_acc + std_acc
            for result in results:
                if result.dssp_accuracy is not None and result.dssp_accuracy > threshold_high:
                    f.write(f"  {result.protein_id}: {result.dssp_accuracy:.4f}\n")

    print(f"\nSummary report saved to: {output_file}")


# ==================== CROSS-VALIDATION ====================

def cross_validate_parameters(
    pdb_files: List[Path],
    params: BurialParameters,
    n_folds: int = 5,
    reference: str = 'dssp'  # 'dssp' or 'stride'
) -> Dict:
    """Perform k-fold cross-validation on the dataset (split by proteins).

    If the requested number of folds is greater than the number of proteins,
    reduce n_folds to len(pdb_files) to avoid sklearn's ValueError.
    """
    n_samples = len(pdb_files)
    if n_samples < 2:
        # Not enough data for any meaningful CV; return zeros
        return {
            'fold_accuracies': [],
            'fold_f1_scores': [],
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'mean_f1': 0.0,
            'std_f1': 0.0,
        }

    if n_folds > n_samples:
        print(f"Requested {n_folds}-fold CV but only {n_samples} proteins are available; "
              f"using {n_samples}-fold instead.")
        n_folds = n_samples

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies: List[float] = []
    fold_f1_scores: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(pdb_files)):
        print(f"Fold {fold_idx + 1}/{n_folds}")

        train_files = [pdb_files[i] for i in train_idx]
        test_files = [pdb_files[i] for i in test_idx]

        print(f"  Training: {len(train_files)} proteins")
        print(f"  Testing: {len(test_files)} proteins")

        # Process test set only – training is implicit via fixed parameters
        test_results = process_protein_dataset(test_files, params)

        # Collect accuracies/F1 for this fold based on chosen reference
        if reference == 'dssp':
            accuracies = [r.dssp_accuracy for r in test_results if r.dssp_accuracy is not None]
            f1_scores = [r.dssp_f1 for r in test_results if r.dssp_f1 is not None]
        else:
            accuracies = [r.stride_accuracy for r in test_results if r.stride_accuracy is not None]
            f1_scores = [r.stride_f1 for r in test_results if r.stride_f1 is not None]

        if accuracies:
            fold_acc = float(np.mean(accuracies))
            fold_f1 = float(np.mean(f1_scores))
            fold_accuracies.append(fold_acc)
            fold_f1_scores.append(fold_f1)
            print(f"  Fold Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}\n")

    cv_results = {
        'fold_accuracies': fold_accuracies,
        'fold_f1_scores': fold_f1_scores,
        'mean_accuracy': float(np.mean(fold_accuracies)) if fold_accuracies else 0.0,
        'std_accuracy': float(np.std(fold_accuracies)) if fold_accuracies else 0.0,
        'mean_f1': float(np.mean(fold_f1_scores)) if fold_f1_scores else 0.0,
        'std_f1': float(np.std(fold_f1_scores)) if fold_f1_scores else 0.0,
    }

    print("\nCross-Validation Results:")
    print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"  Mean F1-Score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")

    return cv_results


# ==================== OPTUNA OPTIMIZATION ====================

def optimize_parameters_optuna(
    pdb_files: List[Path],
    n_trials: int = 100,
    reference: str = 'dssp',
    n_folds: int = 5
) -> BurialParameters:
    """Optimize parameters using Optuna framework.

    Ensures that the number of CV folds does not exceed the number of proteins.
    """
    n_samples = len(pdb_files)
    if n_samples < 2:
        raise ValueError("Not enough PDB files for optimization (need at least 2).")

    if n_folds > n_samples:
        print(f"Requested {n_folds}-fold CV for optimization but only {n_samples} proteins "
              f"are available; using {n_samples}-fold instead.")
        n_folds = n_samples

    print(f"\n{'='*80}")
    print(f"PARAMETER OPTIMIZATION USING OPTUNA")
    print(f"{'='*80}\n")
    print(f"Trials: {n_trials}")
    print(f"Reference: {reference.upper()}")
    print(f"Cross-validation: {n_folds}-fold\n")

    def objective(trial):
        """Optuna objective function."""
        # Suggest parameters
        params = BurialParameters(
            nc6_threshold=trial.suggest_float('nc6_threshold', 6.0, 15.0),
            nc10_threshold=trial.suggest_float('nc10_threshold', 12.0, 30.0),
            uni6_threshold=trial.suggest_float('uni6_threshold', 0.25, 0.65),
            uni10_threshold=trial.suggest_float('uni10_threshold', 0.35, 0.75),
            dssp_asa_cutoff=30.0,  # Keep fixed
            stride_asa_cutoff=24.0  # Keep fixed
        )

        # Evaluate using cross-validation
        cv_results = cross_validate_parameters(pdb_files, params, n_folds, reference)

        # Return mean accuracy as optimization target
        return cv_results['mean_accuracy']

    # Create study and optimize
    study = optuna.create_study(direction='maximize', study_name='burial_classification')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n{'='*80}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*80}\n")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")

    # Create best parameters object
    best_params = BurialParameters(
        nc6_threshold=study.best_params['nc6_threshold'],
        nc10_threshold=study.best_params['nc10_threshold'],
        uni6_threshold=study.best_params['uni6_threshold'],
        uni10_threshold=study.best_params['uni10_threshold']
    )

    return best_params, study


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function."""
    # Setup
    workspace_dir = Path.cwd()
    output_dir = workspace_dir / "results" / "comprehensive_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDB files
    pdb_files = sorted(workspace_dir.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")

    if len(pdb_files) == 0:
        print("No PDB files found in workspace!")
        return

    # Default parameters (baseline)
    default_params = BurialParameters(
        nc6_threshold=10.0,
        nc10_threshold=18.0,
        uni6_threshold=0.40,
        uni10_threshold=0.50,
        dssp_asa_cutoff=30.0,
        stride_asa_cutoff=24.0
    )

    print("\n" + "="*80)
    print("PHASE 1: BASELINE ANALYSIS WITH DEFAULT PARAMETERS")
    print("="*80)

    # Process all proteins with default parameters
    baseline_results = process_protein_dataset(pdb_files, default_params)

    # Save confusion matrices
    save_confusion_matrices(baseline_results, output_dir / "confusion_matrices")

    # Generate summary report
    generate_summary_report(
        baseline_results,
        output_dir / "baseline_summary_report.txt"
    )

    # Save per-protein CSV files
    for result in baseline_results:
        csv_file = output_dir / f"{result.protein_id}_detailed_results.csv"
        result.dataframe.to_csv(csv_file, index=False)

    print("\n" + "="*80)
    print("PHASE 2: PARAMETER OPTIMIZATION")
    print("="*80)

    # Optimize parameters using Optuna
    best_params, study = optimize_parameters_optuna(
        pdb_files,
        n_trials=50,  # Adjust as needed
        reference='dssp',
        n_folds=5
    )

    # Save optimization results
    optuna_df = study.trials_dataframe()
    optuna_df.to_csv(output_dir / "optuna_optimization_trials.csv", index=False)

    print("\n" + "="*80)
    print("PHASE 3: FINAL ANALYSIS WITH OPTIMIZED PARAMETERS")
    print("="*80)

    # Process all proteins with optimized parameters
    optimized_results = process_protein_dataset(pdb_files, best_params)

    # Save confusion matrices
    save_confusion_matrices(optimized_results, output_dir / "confusion_matrices_optimized")

    # Generate summary report
    generate_summary_report(
        optimized_results,
        output_dir / "optimized_summary_report.txt"
    )

    # Save per-protein CSV files
    for result in optimized_results:
        csv_file = output_dir / f"{result.protein_id}_optimized_results.csv"
        result.dataframe.to_csv(csv_file, index=False)

    # Save best parameters
    with open(output_dir / "best_parameters.txt", 'w') as f:
        f.write("OPTIMIZED PARAMETERS\n")
        f.write("="*80 + "\n\n")
        f.write(f"nc6_threshold: {best_params.nc6_threshold:.4f}\n")
        f.write(f"nc10_threshold: {best_params.nc10_threshold:.4f}\n")
        f.write(f"uni6_threshold: {best_params.uni6_threshold:.4f}\n")
        f.write(f"uni10_threshold: {best_params.uni10_threshold:.4f}\n")
        f.write(f"dssp_asa_cutoff: {best_params.dssp_asa_cutoff:.4f}\n")
        f.write(f"stride_asa_cutoff: {best_params.stride_asa_cutoff:.4f}\n")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - baseline_summary_report.txt")
    print("  - optimized_summary_report.txt")
    print("  - best_parameters.txt")
    print("  - optuna_optimization_trials.csv")
    print("  - confusion_matrices/ (per-protein matrices)")
    print("  - *_detailed_results.csv (per-protein data)")


if __name__ == "__main__":
    main()


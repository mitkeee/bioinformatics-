import sys
from typing import Dict, List, Optional
from pathlib import Path
import subprocess
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    # Input/Output paths
    'pdb_input_folder': 'pdbexamples',  # Folder containing PDB files to analyze
    'output_folder': 'final_reports',    # Output folder for CSV and TXT files

    # Burial classification thresholds - BALANCED FOR BEST OVERALL PERFORMANCE
    # After testing, these values provide the best balance
    'nc6_threshold': 5.0,                # Min neighbors at 6Å sphere (optimal value)
    'nc10_threshold': 16.0,              # Min neighbors at 10Å sphere (optimal value)
    'uni6_threshold': 0.38,              # Min uniformity at 6Å sphere (slightly decreased for better recall)
    'uni10_threshold': 0.48,             # Min uniformity at 10Å sphere (slightly decreased for better recall)

    # DSSP/STRIDE cutoffs (accessible surface area)
    'dssp_asa_cutoff': 25.0,             # DSSP: ASA ≥ 25% = exterior
    'stride_asa_cutoff': 20.0,           # STRIDE: ASA ≥ 20% = exterior

    # Processing options
    'search_subdirectories': True,       # Recursively search subdirectories for PDB files
    'skip_missing_dssp_stride': True,    # Skip proteins if DSSP/STRIDE unavailable (True) or continue anyway (False)
    'verbose': True,                     # Print detailed progress messages
}

# ================================================================================
# CONSTANTS
# ================================================================================

# MaxASA values from Tien et al. (PLoS ONE 2013, 8(11):e80635)
MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
    'CYS': 167.0, 'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0,
    'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
    'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}

# Check for DSSP availability
try:
    from Bio.PDB.DSSP import DSSP
    HAS_DSSP = True
except Exception:
    HAS_DSSP = False

# ================================================================================
# DATA CLASSES
# ================================================================================

@dataclass
class BurialParameters:
    """Parameters for burial classification algorithm"""
    nc6_threshold: float = 10.0
    nc10_threshold: float = 18.0
    uni6_threshold: float = 0.40
    uni10_threshold: float = 0.50
    dssp_asa_cutoff: float = 30.0
    stride_asa_cutoff: float = 24.0


@dataclass
class ProteinAnalysis:
    """Results for a single protein"""
    protein_id: str
    pdb_path: Path
    n_residues: int
    dataframe: pd.DataFrame
    dssp_available: bool = False
    stride_available: bool = False
    dssp_accuracy: Optional[float] = None
    stride_accuracy: Optional[float] = None


# ================================================================================
# METRICS CALCULATION FUNCTION
# ================================================================================

def confusion_metrics(TP, FP, TN, FN):
    """Calculate confusion matrix metrics - exact implementation."""
    P = TP + FN
    N = TN + FP
    total = P + N

    BM = (TP / P) + (TN / N) - 1
    prevalence = P / (P + N)
    TPR = TP / P
    FNR = FN / P
    FPR = FP / N
    TNR = TN / N
    RECC = TP / (TP + FN)
    PPV = TP / (TP + FP)
    FOR = FN / (TN + FN)
    LR_plus = (TP / P) / (FP / N)
    LR_minus = (FN / P) / (TN / N)
    ACC = (TP + TN) / total
    FDR = FP / (TP + FP)
    NPV = TN / (TN + FN)
    MK = PPV + NPV - 1
    DOR = LR_plus / LR_minus
    BA = ((TP / P) + (TN / N)) / 2
    F1 = (2 * TP) / (2 * TP + FP + FN)
    FM = math.sqrt(PPV * TPR)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    Jaccard = TP / (TP + FP + FN)

    return {
        "Informedness (BM)": BM,
        "Prevalence": prevalence,
        "TPR": TPR,
        "FNR": FNR,
        "FPR": FPR,
        "TNR": TNR,
        "RECC": RECC,
        "PPV": PPV,
        "FOR": FOR,
        "LR+": LR_plus,
        "LR-": LR_minus,
        "Accuracy": ACC,
        "FDR": FDR,
        "NPV": NPV,
        "Markedness (MK)": MK,
        "DOR": DOR,
        "Balanced Accuracy": BA,
        "F1 Score": F1,
        "Fowlkes–Mallows": FM,
        "MCC": MCC,
        "Jaccard Index": Jaccard
    }


# ================================================================================
# LOCAL SKLEARN-LIKE METRICS IMPLEMENTATION
# ================================================================================

def confusion_matrix_local(y_true, y_pred):
    """
    Compute confusion matrix (local reimplementation of sklearn.metrics.confusion_matrix).

    Returns:
        tuple: (matrix_2x2, (tn, fp, fn, tp))
            - matrix_2x2: numpy array [[tn, fp], [fn, tp]]
            - tuple: (tn, fp, fn, tp) for convenience

    Formula:
        TN = True Negatives  (predicted 0, actual 0)
        FP = False Positives (predicted 1, actual 0)
        FN = False Negatives (predicted 0, actual 1)
        TP = True Positives  (predicted 1, actual 1)
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    return matrix, (tn, fp, fn, tp)


def accuracy_score_local(y_true, y_pred):
    """
    Compute accuracy score (local reimplementation of sklearn.metrics.accuracy_score).

    Formula:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    if n == 0:
        return 0.0
    correct = int(np.sum(yt == yp))
    return float(correct) / n


def precision_score_local(y_true, y_pred, zero_division=0.0):
    """
    Compute precision score (local reimplementation of sklearn.metrics.precision_score).

    Formula:
        Precision = TP / (TP + FP)

    Also known as Positive Predictive Value (PPV).
    """
    _, (_, fp, _, tp) = confusion_matrix_local(y_true, y_pred)
    denom = tp + fp
    if denom == 0:
        return float(zero_division)
    return float(tp) / denom


def recall_score_local(y_true, y_pred, zero_division=0.0):
    """
    Compute recall score (local reimplementation of sklearn.metrics.recall_score).

    Formula:
        Recall = TP / (TP + FN)

    Also known as Sensitivity or True Positive Rate (TPR).
    """
    _, (_, _, fn, tp) = confusion_matrix_local(y_true, y_pred)
    denom = tp + fn
    if denom == 0:
        return float(zero_division)
    return float(tp) / denom


def f1_score_local(y_true, y_pred, zero_division=0.0):
    """
    Compute F1 score (local reimplementation of sklearn.metrics.f1_score).

    Formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Harmonic mean of precision and recall.
    """
    prec = precision_score_local(y_true, y_pred, zero_division=zero_division)
    rec = recall_score_local(y_true, y_pred, zero_division=zero_division)
    denom = prec + rec
    if denom == 0:
        return float(zero_division)
    return 2.0 * (prec * rec) / denom


def balanced_accuracy_score_local(y_true, y_pred):
    """
    Compute balanced accuracy (local reimplementation of sklearn.metrics.balanced_accuracy_score).

    Formula:
        Balanced Accuracy = (TPR + TNR) / 2

    Where:
        TPR = True Positive Rate = TP / (TP + FN)
        TNR = True Negative Rate = TN / (TN + FP)
    """
    _, (tn, fp, fn, tp) = confusion_matrix_local(y_true, y_pred)

    # TPR (Sensitivity)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # TNR (Specificity)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return (tpr + tnr) / 2.0


def matthews_corrcoef_local(y_true, y_pred):
    """
    Compute Matthews Correlation Coefficient (local reimplementation of sklearn.metrics.matthews_corrcoef).

    Formula:
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Returns value between -1 and +1:
        +1 = perfect prediction
         0 = no better than random
        -1 = total disagreement
    """
    _, (tn, fp, fn, tp) = confusion_matrix_local(y_true, y_pred)

    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def validate_local_metrics_vs_sklearn(y_true, y_pred, metric_name="Test", verbose=True):
    """
    Validate local metric implementations against sklearn.

    This function compares all local implementations with sklearn to ensure correctness.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_name: Name for this validation test
        verbose: Print detailed comparison

    Returns:
        dict: Validation results with match status and values
    """
    results = {
        'metric_name': metric_name,
        'all_match': False,
        'metrics': {}
    }

    try:
        # Import sklearn metrics
        from sklearn.metrics import (
            confusion_matrix as cm_sklearn,
            accuracy_score as acc_sklearn,
            precision_score as prec_sklearn,
            recall_score as rec_sklearn,
            f1_score as f1_sklearn,
            balanced_accuracy_score as ba_sklearn,
            matthews_corrcoef as mcc_sklearn
        )

        # Confusion Matrix
        cm_local, (tn_l, fp_l, fn_l, tp_l) = confusion_matrix_local(y_true, y_pred)
        cm_sk = cm_sklearn(y_true, y_pred)
        cm_match = np.allclose(cm_local, cm_sk)
        results['metrics']['confusion_matrix'] = {
            'local': cm_local,
            'sklearn': cm_sk,
            'match': cm_match
        }

        # Accuracy
        acc_local = accuracy_score_local(y_true, y_pred)
        acc_sk = acc_sklearn(y_true, y_pred)
        acc_match = abs(acc_local - acc_sk) < 1e-10
        results['metrics']['accuracy'] = {
            'local': acc_local,
            'sklearn': acc_sk,
            'match': acc_match,
            'diff': abs(acc_local - acc_sk)
        }

        # Precision
        prec_local = precision_score_local(y_true, y_pred, zero_division=0)
        prec_sk = prec_sklearn(y_true, y_pred, zero_division=0)
        prec_match = abs(prec_local - prec_sk) < 1e-10
        results['metrics']['precision'] = {
            'local': prec_local,
            'sklearn': prec_sk,
            'match': prec_match,
            'diff': abs(prec_local - prec_sk)
        }

        # Recall
        rec_local = recall_score_local(y_true, y_pred, zero_division=0)
        rec_sk = rec_sklearn(y_true, y_pred, zero_division=0)
        rec_match = abs(rec_local - rec_sk) < 1e-10
        results['metrics']['recall'] = {
            'local': rec_local,
            'sklearn': rec_sk,
            'match': rec_match,
            'diff': abs(rec_local - rec_sk)
        }

        # F1 Score
        f1_local = f1_score_local(y_true, y_pred, zero_division=0)
        f1_sk = f1_sklearn(y_true, y_pred, zero_division=0)
        f1_match = abs(f1_local - f1_sk) < 1e-10
        results['metrics']['f1_score'] = {
            'local': f1_local,
            'sklearn': f1_sk,
            'match': f1_match,
            'diff': abs(f1_local - f1_sk)
        }

        # Balanced Accuracy
        ba_local = balanced_accuracy_score_local(y_true, y_pred)
        ba_sk = ba_sklearn(y_true, y_pred)
        ba_match = abs(ba_local - ba_sk) < 1e-10
        results['metrics']['balanced_accuracy'] = {
            'local': ba_local,
            'sklearn': ba_sk,
            'match': ba_match,
            'diff': abs(ba_local - ba_sk)
        }

        # Matthews Correlation Coefficient
        mcc_local = matthews_corrcoef_local(y_true, y_pred)
        mcc_sk = mcc_sklearn(y_true, y_pred)
        mcc_match = abs(mcc_local - mcc_sk) < 1e-10
        results['metrics']['mcc'] = {
            'local': mcc_local,
            'sklearn': mcc_sk,
            'match': mcc_match,
            'diff': abs(mcc_local - mcc_sk)
        }

        # Check if all metrics match
        all_matches = [
            cm_match, acc_match, prec_match, rec_match,
            f1_match, ba_match, mcc_match
        ]
        results['all_match'] = all(all_matches)
        results['sklearn_available'] = True

        # Print verbose output
        if verbose:
            print(f"\n{'='*80}")
            print(f"LOCAL METRICS VALIDATION: {metric_name}")
            print(f"{'='*80}")
            print(f"Dataset size: {len(y_true)} samples")
            print(f"Confusion Matrix (TN={tn_l}, FP={fp_l}, FN={fn_l}, TP={tp_l})")
            print(f"{'-'*80}")

            for name, data in results['metrics'].items():
                if name == 'confusion_matrix':
                    status = "✓ MATCH" if data['match'] else "✗ MISMATCH"
                    print(f"{name:25s}: {status}")
                else:
                    status = "✓ MATCH" if data['match'] else "✗ MISMATCH"
                    local_val = data['local']
                    sklearn_val = data['sklearn']
                    diff = data['diff']
                    print(f"{name:25s}: {status}  (local={local_val:.6f}, sklearn={sklearn_val:.6f}, diff={diff:.2e})")

            print(f"{'-'*80}")
            if results['all_match']:
                print("✓ ALL METRICS MATCH - Local implementation is correct!")
            else:
                print("✗ SOME METRICS MISMATCH - Check implementation!")
            print(f"{'='*80}\n")

    except ImportError:
        results['sklearn_available'] = False
        if verbose:
            print(f"\n⚠ sklearn not available - cannot validate {metric_name}")

    except Exception as e:
        results['error'] = str(e)
        if verbose:
            print(f"\n⚠ Error during validation of {metric_name}: {e}")

    return results

# ================================================================================
# FEATURE EXTRACTION FUNCTIONS
# ================================================================================

def extract_ca_atoms(pdb_path: Path) -> pd.DataFrame:
    """Extract CA atoms with coordinates and residue info."""
    if isinstance(pdb_path, str):
        pdb_path = Path(pdb_path)

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

        # Calculate mean direction
        mean_vector = np.mean(vectors, axis=0)
        mean_magnitude = np.linalg.norm(mean_vector)

        # Uniformity: 1 - mean_magnitude
        uniformity[i] = 1.0 - mean_magnitude

    return uniformity


def extract_dssp_data(pdb_path: Path, df: pd.DataFrame, asa_cutoff: float = 30.0) -> pd.DataFrame:
    """Extract DSSP ASA and classify as buried/exposed."""
    if isinstance(pdb_path, str):
        pdb_path = Path(pdb_path)

    # Try BioPython first
    if HAS_DSSP:
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
                    # Use NaN for missing ASA values instead of 0.0
                    rel_asa = float(rec[3]) if rec[3] != 'NA' and rec[3] is not None else np.nan
                except (ValueError, TypeError):
                    rel_asa = np.nan
                aa = rec[1] if len(rec) > 1 else 'ALA'
                ss = rec[2] if len(rec) > 2 else '-'
                max_asa = MAX_ASA.get(aa, 100.0)
                asa = rel_asa * max_asa
                dssp_map[(chain_id, int(resseq), icode.strip() or '')] = {'asa': asa, 'ss': ss}

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

            df['dssp_asa'] = pd.Series(dssp_asa, index=df.index, dtype='float64')
            df['dssp_ss'] = dssp_ss

            def _rasa_dssp(row):
                aa = str(row['resname']).strip().upper()
                max_asa = MAX_ASA.get(aa)
                if max_asa is None or pd.isna(row['dssp_asa']):
                    return np.nan
                return float(row['dssp_asa']) / max_asa

            df['RASA_dssp'] = df.apply(_rasa_dssp, axis=1)
            df['dssp_class'] = df['RASA_dssp'].apply(
                lambda r: 1 if pd.notna(r) and r >= 0.25 else (0 if pd.notna(r) else np.nan)
            )
            return df
        except Exception:
            pass

    # Next: try to read an existing .dssp file similar to STRIDE handling
    try:
        dssp_content = None
        for name_variant in [pdb_path.stem.lower(), pdb_path.stem.upper(), pdb_path.stem]:
            test_file = pdb_path.parent / f"{name_variant}.dssp"
            if test_file.exists():
                with open(test_file, 'r') as f:
                    content = f.read()
                # DSSP files have a header ending with "  #  RESIDUE AA STRUCTURE BP1 BP2  ACC ..."
                # Data lines start after this header
                if '  #  RESIDUE AA STRUCTURE' in content or 'ACC' in content:
                    dssp_content = content
                    break
        if dssp_content is not None:
            dssp_map = {}
            parsing_data = False
            for line in dssp_content.split('\n'):
                # Skip until we find the header line
                if '  #  RESIDUE AA STRUCTURE' in line:
                    parsing_data = True
                    continue
                if not parsing_data or not line.strip():
                    continue

                # DSSP format is fixed-width, not space-delimited
                # Format: "  143  143 A K  H  > S+     0   0  135"
                # Position 0-4: residue number (right-aligned)
                # Position 5-9: PDB residue number
                # Position 10-11: chain ID
                # Position 13: amino acid
                # Position 16: secondary structure
                # Position 35-38: ACC (accessible surface area)
                try:
                    if len(line) < 35:
                        continue

                    # Extract residue number (PDB numbering at position 5-10, can have insertion code)
                    resseq_str = line[5:10].strip()
                    if not resseq_str or resseq_str == '!':
                        continue
                    resseq = int(resseq_str)

                    # Extract chain ID (position 11)
                    chain = line[11:12] if len(line) > 11 else ' '
                    if chain == '!':
                        chain = ' '

                    # Extract secondary structure (position 16)
                    ss = line[16:17] if len(line) > 16 else '-'
                    if ss == ' ':
                        ss = 'C'  # Coil

                    # Extract ASA (position 35-38 or nearby, typically 4 digits right-aligned)
                    asa_val = np.nan
                    if len(line) > 35:
                        # Try to extract ASA from multiple possible positions
                        for start_pos in [34, 35, 36]:
                            try:
                                asa_str = line[start_pos:start_pos+4].strip()
                                if asa_str:
                                    asa_val = float(asa_str)
                                    break
                            except (ValueError, IndexError):
                                continue

                    data = {'asa': asa_val, 'ss': ss if ss else '-'}
                    dssp_map[(chain, resseq, '')] = data
                    dssp_map[(chain.strip(), resseq, '')] = data
                    if chain.strip() == '' or chain.strip() == '-':
                        dssp_map[(' ', resseq, '')] = data
                        dssp_map[('', resseq, '')] = data
                        dssp_map[('A', resseq, '')] = data
                except Exception:
                    continue

            dssp_asa = []
            dssp_ss = []
            for _, row in df.iterrows():
                resseq_int = int(row['resseq'])
                chain = row['chain_id']
                possible_keys = [
                    (chain, resseq_int, ''),
                    (chain.strip(), resseq_int, ''),
                    ('', resseq_int, ''),
                    (' ', resseq_int, ''),
                    ('A', resseq_int, ''),
                    ('-', resseq_int, ''),
                ]
                found = False
                for key in possible_keys:
                    if key in dssp_map:
                        dssp_asa.append(dssp_map[key]['asa'])
                        dssp_ss.append(dssp_map[key]['ss'])
                        found = True
                        break
                if not found:
                    dssp_asa.append(np.nan)
                    dssp_ss.append('-')

            df['dssp_asa'] = pd.Series(dssp_asa, index=df.index, dtype='float64')
            df['dssp_ss'] = dssp_ss

            def _rasa_dssp(row):
                aa = str(row['resname']).strip().upper()
                max_asa = MAX_ASA.get(aa)
                if max_asa is None or pd.isna(row['dssp_asa']):
                    return np.nan
                return float(row['dssp_asa']) / max_asa

            df['RASA_dssp'] = df.apply(_rasa_dssp, axis=1)
            df['dssp_class'] = df['RASA_dssp'].apply(
                lambda r: 1 if pd.notna(r) and r >= 0.25 else (0 if pd.notna(r) else np.nan)
            )
            return df
    except Exception:
        pass

    # Fallback: Try command-line DSSP
    try:
        result = subprocess.run(
            ['dssp', '-i', str(pdb_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            dssp_map = {}
            for line in result.stdout.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                try:
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    resseq = int(parts[1])
                    chain = parts[2] if len(parts) > 2 else ' '
                    ss = parts[4] if len(parts) > 4 else 'C'
                    # Use NaN for missing ASA values instead of 0.0
                    asa = float(parts[5]) if len(parts) > 5 else np.nan

                    for c in [chain, ' ', '', 'A']:
                        dssp_map[(c, resseq, '')] = {'asa': asa, 'ss': ss}
                except (ValueError, IndexError):
                    continue

            if dssp_map:
                dssp_asa = []
                dssp_ss = []
                for _, row in df.iterrows():
                    resseq_int = int(row['resseq'])
                    chain = row['chain_id']
                    found = False
                    for try_chain in [chain, chain.strip(), '', ' ', 'A', '-']:
                        if (try_chain, resseq_int, '') in dssp_map:
                            dssp_asa.append(dssp_map[(try_chain, resseq_int, '')]['asa'])
                            dssp_ss.append(dssp_map[(try_chain, resseq_int, '')]['ss'])
                            found = True
                            break
                    if not found:
                        dssp_asa.append(np.nan)
                        dssp_ss.append('-')

                df['dssp_asa'] = pd.Series(dssp_asa, index=df.index, dtype='float64')
                df['dssp_ss'] = dssp_ss

                def _rasa_dssp(row):
                    aa = str(row['resname']).strip().upper()
                    max_asa = MAX_ASA.get(aa)
                    if max_asa is None or pd.isna(row['dssp_asa']):
                        return np.nan
                    return float(row['dssp_asa']) / max_asa

                df['RASA_dssp'] = df.apply(_rasa_dssp, axis=1)
                df['dssp_class'] = df['RASA_dssp'].apply(
                    lambda r: 1 if pd.notna(r) and r >= 0.25 else (0 if pd.notna(r) else np.nan)
                )
                return df
    except Exception:
        pass

    # No DSSP data available - set to NaN
    df['dssp_asa'] = np.nan
    df['dssp_class'] = np.nan
    df['dssp_ss'] = '-'
    df['RASA_dssp'] = np.nan
    return df


def extract_stride_data(pdb_path: Path, df: pd.DataFrame, asa_cutoff: float = 24.0) -> pd.DataFrame:
    """Extract STRIDE ASA and classify as buried/exposed."""
    if isinstance(pdb_path, str):
        pdb_path = Path(pdb_path)

    try:
        stride_content = None

        # Try to find existing STRIDE file
        for name_variant in [pdb_path.stem.lower(), pdb_path.stem.upper(), pdb_path.stem]:
            test_file = pdb_path.parent / f"{name_variant}.stride"
            if test_file.exists():
                with open(test_file, 'r') as f:
                    content = f.read()
                if any(line.startswith('ASG') for line in content.split('\n')):
                    stride_content = content
                    break

        # If no valid STRIDE file, try running STRIDE
        if stride_content is None:
            try:
                result = subprocess.run(
                    ['stride', str(pdb_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    stride_content = result.stdout
                    if any(line.startswith('ASG') for line in stride_content.split('\n')):
                        try:
                            with open(pdb_path.parent / f"{pdb_path.stem}.stride", 'w') as f:
                                f.write(stride_content)
                        except:
                            pass
            except:
                pass

        if stride_content is None:
            raise Exception("No STRIDE data available")

        stride_map = {}
        for line in stride_content.split('\n'):
            if line.startswith('ASG'):
                try:
                    chain_id = line[9:10]
                    resseq = int(line[11:15].strip())
                    ss = line[24:25].strip() if len(line) > 24 else 'C'
                    parts = line.split()
                    # Use NaN for missing ASA values instead of 0.0
                    asa = float(parts[-2]) if len(parts) >= 10 else np.nan

                    data = {'asa': asa, 'ss': ss if ss else 'C'}
                    stride_map[(chain_id, resseq, '')] = data
                    stride_map[(chain_id.strip(), resseq, '')] = data

                    if chain_id.strip() == '' or chain_id.strip() == '-':
                        stride_map[(' ', resseq, '')] = data
                        stride_map[('', resseq, '')] = data
                        stride_map[('A', resseq, '')] = data

                except (ValueError, IndexError):
                    continue

        stride_asa = []
        stride_ss = []
        for _, row in df.iterrows():
            resseq_int = int(row['resseq'])
            chain = row['chain_id']

            possible_keys = [
                (chain, resseq_int, ''),
                (chain.strip(), resseq_int, ''),
                ('', resseq_int, ''),
                (' ', resseq_int, ''),
                ('A', resseq_int, ''),
                ('-', resseq_int, ''),
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

        # Convert to Series to preserve NaN values
        df['stride_asa'] = pd.Series(stride_asa, index=df.index, dtype='float64')
        df['stride_ss'] = stride_ss

        def _rasa_stride(row):
            aa = str(row['resname']).strip().upper()
            max_asa = MAX_ASA.get(aa)
            if max_asa is None or pd.isna(row['stride_asa']):
                return np.nan
            return float(row['stride_asa']) / max_asa

        df['RASA_stride'] = df.apply(_rasa_stride, axis=1)
        df['stride_class'] = df['RASA_stride'].apply(
            lambda r: 1 if pd.notna(r) and r >= 0.25 else (0 if pd.notna(r) else np.nan)
        )

    except Exception as e:
        # No STRIDE data available - set to NaN
        df['stride_asa'] = np.nan
        df['stride_class'] = np.nan
        df['stride_ss'] = '-'
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

    Logic:
    - AUTOMATIC SURFACE (exterior=1): if nc6 <= 2 OR nc10 <= 2 (very few neighbors = always surface)
    - THRESHOLD LOGIC: Both nc6 AND nc10 must be below thresholds to be considered potentially buried
    - UNIFORMITY CHECK: If uniformity is low in either sphere, it indicates non-uniform distribution -> surface
    """
    ncps_class = []

    for _, row in df.iterrows():
        nc6 = row['ncps_sphere_6']
        nc10 = row['ncps_sphere_10']
        uni6 = row['ncps_sphere_6_uni']
        uni10 = row['ncps_sphere_10_uni']

        is_exterior = False

        # RULE 1: AUTOMATIC SURFACE - very few neighbors (1-2) = definite surface
        if nc6 <= 2 or nc10 <= 2:
            is_exterior = True
        # RULE 2: Check if both neighbor counts are above thresholds (buried criteria)
        # If BOTH nc6 AND nc10 are high, might be buried (need uniformity check)
        elif nc6 >= params.nc6_threshold and nc10 >= params.nc10_threshold:
            # Both neighbor counts are high - check uniformity
            # If uniformity is low in either sphere, it's still exterior despite high neighbor count
            if (pd.notna(uni6) and uni6 < params.uni6_threshold) or (pd.notna(uni10) and uni10 < params.uni10_threshold):
                is_exterior = True
            else:
                # Both neighbor counts high AND uniformity good = buried
                is_exterior = False
        # RULE 3: If neighbor counts are intermediate, check uniformity
        else:
            # At least one neighbor count is below threshold - more likely to be exterior
            # Check if uniformity is also low
            if (pd.notna(uni6) and uni6 < params.uni6_threshold) or (pd.notna(uni10) and uni10 < params.uni10_threshold):
                is_exterior = True
            else:
                # Even with lower neighbor counts, if uniformity is good, might be interior
                # Use the 6Å sphere as primary indicator (more local)
                if nc6 < params.nc6_threshold:
                    is_exterior = True
                else:
                    is_exterior = False

        ncps_class.append(1 if is_exterior else 0)

    return np.array(ncps_class)


# ================================================================================
# PROCESSING FUNCTIONS
# ================================================================================

def process_single_protein(pdb_path: Path, params: BurialParameters, verbose: bool = True) -> Optional[ProteinAnalysis]:
    """Process a single protein and return analysis results."""
    protein_id = pdb_path.stem

    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing: {protein_id}")
        print(f"{'='*80}")

    try:
        # Extract CA atoms
        if verbose:
            print(f"  ✓ Extracting CA atoms...", end=" ")
        df = extract_ca_atoms(pdb_path)
        if len(df) == 0:
            if verbose:
                print("ERROR: No CA atoms found")
            return None
        if verbose:
            print(f"Found {len(df)} residues")

        coords = df[['x', 'y', 'z']].values

        # Extract DSSP data
        if verbose:
            print(f"  ✓ Extracting DSSP data...", end=" ")
        df = extract_dssp_data(pdb_path, df, params.dssp_asa_cutoff)
        dssp_count = df['dssp_class'].notna().sum()
        dssp_available = dssp_count > 0
        if verbose:
            print(f"{dssp_count} available" if dssp_available else "Not available")

        # Extract STRIDE data
        if verbose:
            print(f"  ✓ Extracting STRIDE data...", end=" ")
        df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)
        stride_count = df['stride_class'].notna().sum()
        stride_available = stride_count > 0
        if verbose:
            print(f"{stride_count} available" if stride_available else "Not available")

        # Add neighbor features
        if verbose:
            print(f"  ✓ Computing neighbor features...", end=" ")
        df = add_neighbor_features(df, coords)
        if verbose:
            print("Done")

        # Classify burial
        if verbose:
            print(f"  ✓ Classifying burial status...", end=" ")
        df['ncps_class'] = classify_burial(df, params)

        # Flag residues with very few neighbors (automatic surface indicator)
        df['ncps_very_low_neighbors'] = ((df['ncps_sphere_6'] <= 2) | (df['ncps_sphere_10'] <= 2)).astype(int)

        if verbose:
            print("Done")

        # Calculate metrics using local implementation and validate against sklearn
        dssp_accuracy = None
        stride_accuracy = None

        if dssp_available:
            df_dssp = df[df['dssp_class'].notna()].copy()
            y_true = df_dssp['dssp_class'].values.astype(int)
            y_pred = df_dssp['ncps_class'].values.astype(int)

            # Use local metrics (primary implementation)
            dssp_accuracy = accuracy_score_local(y_true, y_pred)

            if verbose:
                print(f"  ✓ DSSP Metrics (using local implementation):")
                # Validate local implementation against sklearn
                validation = validate_local_metrics_vs_sklearn(
                    y_true, y_pred,
                    metric_name=f"DSSP ({protein_id})",
                    verbose=False  # Don't print full validation table in main flow
                )

                if validation.get('sklearn_available'):
                    # Print key metrics with comparison
                    metrics = validation['metrics']
                    print(f"    Accuracy  : {metrics['accuracy']['local']:.4f} (sklearn: {metrics['accuracy']['sklearn']:.4f}) {'✓' if metrics['accuracy']['match'] else '✗'}")
                    print(f"    Precision : {metrics['precision']['local']:.4f} (sklearn: {metrics['precision']['sklearn']:.4f}) {'✓' if metrics['precision']['match'] else '✗'}")
                    print(f"    Recall    : {metrics['recall']['local']:.4f} (sklearn: {metrics['recall']['sklearn']:.4f}) {'✓' if metrics['recall']['match'] else '✗'}")
                    print(f"    F1 Score  : {metrics['f1_score']['local']:.4f} (sklearn: {metrics['f1_score']['sklearn']:.4f}) {'✓' if metrics['f1_score']['match'] else '✗'}")
                    print(f"    MCC       : {metrics['mcc']['local']:.4f} (sklearn: {metrics['mcc']['sklearn']:.4f}) {'✓' if metrics['mcc']['match'] else '✗'}")

                    if not validation['all_match']:
                        print(f"    ⚠ WARNING: Some metrics don't match sklearn!")
                else:
                    # sklearn not available - just show local metrics
                    print(f"    Accuracy  : {dssp_accuracy:.4f}")
                    print(f"    Precision : {precision_score_local(y_true, y_pred):.4f}")
                    print(f"    Recall    : {recall_score_local(y_true, y_pred):.4f}")
                    print(f"    F1 Score  : {f1_score_local(y_true, y_pred):.4f}")
                    print(f"    MCC       : {matthews_corrcoef_local(y_true, y_pred):.4f}")

        if stride_available:
            df_stride = df[df['stride_class'].notna()].copy()
            y_true = df_stride['stride_class'].values.astype(int)
            y_pred = df_stride['ncps_class'].values.astype(int)

            # Use local metrics (primary implementation)
            stride_accuracy = accuracy_score_local(y_true, y_pred)

            if verbose:
                print(f"  ✓ STRIDE Metrics (using local implementation):")
                # Validate local implementation against sklearn
                validation = validate_local_metrics_vs_sklearn(
                    y_true, y_pred,
                    metric_name=f"STRIDE ({protein_id})",
                    verbose=False  # Don't print full validation table in main flow
                )

                if validation.get('sklearn_available'):
                    # Print key metrics with comparison
                    metrics = validation['metrics']
                    print(f"    Accuracy  : {metrics['accuracy']['local']:.4f} (sklearn: {metrics['accuracy']['sklearn']:.4f}) {'✓' if metrics['accuracy']['match'] else '✗'}")
                    print(f"    Precision : {metrics['precision']['local']:.4f} (sklearn: {metrics['precision']['sklearn']:.4f}) {'✓' if metrics['precision']['match'] else '✗'}")
                    print(f"    Recall    : {metrics['recall']['local']:.4f} (sklearn: {metrics['recall']['sklearn']:.4f}) {'✓' if metrics['recall']['match'] else '✗'}")
                    print(f"    F1 Score  : {metrics['f1_score']['local']:.4f} (sklearn: {metrics['f1_score']['sklearn']:.4f}) {'✓' if metrics['f1_score']['match'] else '✗'}")
                    print(f"    MCC       : {metrics['mcc']['local']:.4f} (sklearn: {metrics['mcc']['sklearn']:.4f}) {'✓' if metrics['mcc']['match'] else '✗'}")

                    if not validation['all_match']:
                        print(f"    ⚠ WARNING: Some metrics don't match sklearn!")
                else:
                    # sklearn not available - just show local metrics
                    print(f"    Accuracy  : {stride_accuracy:.4f}")
                    print(f"    Precision : {precision_score_local(y_true, y_pred):.4f}")
                    print(f"    Recall    : {recall_score_local(y_true, y_pred):.4f}")
                    print(f"    F1 Score  : {f1_score_local(y_true, y_pred):.4f}")
                    print(f"    MCC       : {matthews_corrcoef_local(y_true, y_pred):.4f}")

        result = ProteinAnalysis(
            protein_id=protein_id,
            pdb_path=pdb_path,
            n_residues=len(df),
            dataframe=df,
            dssp_available=dssp_available,
            stride_available=stride_available,
            dssp_accuracy=dssp_accuracy,
            stride_accuracy=stride_accuracy
        )

        if verbose:
            print(f"✓ Processing complete")

        return result

    except Exception as e:
        if verbose:
            print(f"✗ ERROR: {str(e)}")
        return None


# ================================================================================
# REPORT GENERATION FUNCTIONS
# ================================================================================

def save_detailed_csv(analysis: ProteinAnalysis, output_path: Path) -> bool:
    """Save detailed results CSV file."""
    try:
        df = analysis.dataframe.copy()

        # Calculate confusion matrix types for each row - DO NOT fill NaN
        # Keep NaN values to properly classify missing data
        dssp_confusion_types = []
        stride_confusion_types = []

        for idx, row in df.iterrows():
            # DSSP confusion matrix
            if pd.isna(row['dssp_class']):
                dssp_confusion_types.append('NaN')
            else:
                dssp_true = int(row['dssp_class'])
                dssp_pred = int(row['ncps_class'])

                if dssp_true == 1 and dssp_pred == 1:
                    dssp_confusion_types.append('TP')
                elif dssp_true == 0 and dssp_pred == 0:
                    dssp_confusion_types.append('TN')
                elif dssp_true == 1 and dssp_pred == 0:
                    dssp_confusion_types.append('FN')
                elif dssp_true == 0 and dssp_pred == 1:
                    dssp_confusion_types.append('FP')
                else:
                    dssp_confusion_types.append('N/A')

            # STRIDE confusion matrix
            if pd.isna(row['stride_class']):
                stride_confusion_types.append('NaN')
            else:
                stride_true = int(row['stride_class'])
                stride_pred = int(row['ncps_class'])

                if stride_true == 1 and stride_pred == 1:
                    stride_confusion_types.append('TP')
                elif stride_true == 0 and stride_pred == 0:
                    stride_confusion_types.append('TN')
                elif stride_true == 1 and stride_pred == 0:
                    stride_confusion_types.append('FN')
                elif stride_true == 0 and stride_pred == 1:
                    stride_confusion_types.append('FP')
                else:
                    stride_confusion_types.append('N/A')

        # Calculate global confusion matrices for DSSP and STRIDE
        # DSSP - Only use rows where dssp_class is not NaN
        df_dssp_valid = df[df['dssp_class'].notna()].copy()
        y_true_dssp = df_dssp_valid['dssp_class'].astype(int)
        y_pred_dssp = df_dssp_valid['ncps_class'].astype(int)
        #tn_dssp = int(((y_true_dssp == 0) & (y_pred_dssp == 0)).sum())
        #tn_dssp = int(((y_true_dssp == 0) & (y_pred_dssp == 0)).sum())
        #fp_dssp = int(((y_true_dssp == 0) & (y_pred_dssp == 1)).sum())
        #fn_dssp = int(((y_true_dssp == 1) & (y_pred_dssp == 0)).sum())
        #tp_dssp = int(((y_true_dssp == 1) & (y_pred_dssp == 1)).sum())

        # STRIDE - Only use rows where stride_class is not NaN
        df_stride_valid = df[df['stride_class'].notna()].copy()
        y_true_stride = df_stride_valid['stride_class'].astype(int)
        y_pred_stride = df_stride_valid['ncps_class'].astype(int)
        #tn_stride = int(((y_true_stride == 0) & (y_pred_stride == 0)).sum())
        #fp_stride = int(((y_true_stride == 0) & (y_pred_stride == 1)).sum())
        #fn_stride = int(((y_true_stride == 1) & (y_pred_stride == 0)).sum())
        #tp_stride = int(((y_true_stride == 1) & (y_pred_stride == 1)).sum())

        # Select and save all columns (preserve NaN values in dssp_class and stride_class)
        output_df = pd.DataFrame()
        output_df['resseq'] = df['resseq'].astype(int)
        output_df['resname'] = df['resname'].astype(str)
        output_df['x'] = df['x'].astype(float)
        output_df['y'] = df['y'].astype(float)
        output_df['z'] = df['z'].astype(float)
        output_df['res_num'] = df['res_num'].astype(int)
        output_df['res_id'] = df['res_id'].astype(str)
        # Preserve NaN values - do NOT convert to 0
        output_df['dssp_asa'] = df['dssp_asa']  # Keep NaN as NaN
        output_df['dssp_class'] = df['dssp_class']  # Keep as is (with NaN)
        output_df['dssp_ss'] = df['dssp_ss'].astype(str)
        output_df['RASA_dssp'] = df['RASA_dssp']  # Keep NaN as NaN
        output_df['stride_asa'] = df['stride_asa']  # Keep NaN as NaN
        output_df['stride_ss'] = df['stride_ss'].astype(str)
        output_df['RASA_stride'] = df['RASA_stride']  # Keep NaN as NaN
        output_df['stride_class'] = df['stride_class']  # Keep as is (with NaN)
        output_df['ncps_sphere_6'] = df['ncps_sphere_6'].astype(int)
        output_df['ncps_sphere_10'] = df['ncps_sphere_10'].astype(int)
        output_df['ncps_sphere_6_uni'] = df['ncps_sphere_6_uni'].astype(float)
        output_df['ncps_sphere_10_uni'] = df['ncps_sphere_10_uni'].astype(float)
        output_df['ncps_class'] = df['ncps_class'].astype(int)
        output_df['ncps_very_low_neighbors'] = df['ncps_very_low_neighbors'].astype(int)  # Flag for automatic surface (<=2 neighbors)
        # Add duplicate columns (matching old format)
        output_df['nc6'] = df['ncps_sphere_6'].astype(int)
        output_df['nc10'] = df['ncps_sphere_10'].astype(int)
        output_df['uni6'] = df['ncps_sphere_6_uni'].astype(float)
        output_df['uni10'] = df['ncps_sphere_10_uni'].astype(float)
        # Add confusion matrix columns (per-residue classification only)
        output_df['dssp_confusion_matrix'] = dssp_confusion_types
        output_df['stride_confusion_matrix'] = stride_confusion_types

        # Save CSV with NaN preservation (empty string represents NaN)
        output_df.to_csv(output_path, index=False, na_rep='')
        print(f"    ✓ CSV saved: {output_path.name}")
        return True
    except Exception as e:
        print(f"    ✗ Error saving CSV: {e}")
        return False


def compute_classification_scores(df: pd.DataFrame, truth_col: str, pred_col: str = 'ncps_class') -> Optional[Dict[str, object]]:
    """Compute confusion counts and a comprehensive metrics dict based on a truth column.
    - Filters out NaN in truth_col; assumes pred_col has integer 0/1 predictions
    - Returns None if no valid rows
    """
    if truth_col not in df.columns or pred_col not in df.columns:
        return None
    df_valid = df[df[truth_col].notna()].copy()
    if len(df_valid) == 0:
        return None
    y_true = df_valid[truth_col].astype(int)
    y_pred = df_valid[pred_col].astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    metrics = confusion_metrics(tp, fp, tn, fn)
    return {
        'counts': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp},
        'metrics': metrics,
        'n': int(len(df_valid))
    }


def save_detailed_report(analysis: ProteinAnalysis, output_path: Path, params: BurialParameters) -> bool:
    """Save detailed formatted report TXT file matching fixed_receptor format."""
    try:
        df_orig = analysis.dataframe.copy()
        df = df_orig.copy()
        # DO NOT fill NaN values - keep them as NaN for proper statistics
        # Only convert to appropriate display values as needed
        protein_id = analysis.protein_id

        with open(output_path, 'w') as f:
            # Header
            f.write("=" * 120 + "\n")
            f.write(f"PROTEIN BURIAL ANALYSIS - DETAILED REPORT\n")
            f.write(f"PDB ID: {protein_id.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 120 + "\n\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 120 + "\n")
            f.write(f"Total Residues: {len(df)}\n\n")

            # DSSP statistics
            if df['dssp_class'].notna().sum() > 0:
                dssp_exterior = int((df['dssp_class'] == 1).sum())
                dssp_interior = int((df['dssp_class'] == 0).sum())
                f.write(f"DSSP Classification:\n")
                f.write(f"  - Exterior (1): {dssp_exterior} residues\n")
                f.write(f"  - Interior (0): {dssp_interior} residues\n\n")
                #f.write(f"  - DSSP Cutoff Value: RASA ≥ 0.25 (25% relative accessible surface area)\n")
                #f.write(f"    (If RASA ≥ 0.25, classified as Exterior=1; otherwise Interior=0)\n\n")
                #f.write(f"    Note: params.dssp_asa_cutoff={params.dssp_asa_cutoff}% is for reference only\n\n")

            # STRIDE statistics
            if df['stride_class'].notna().sum() > 0:
                stride_exterior = int((df['stride_class'] == 1).sum())
                stride_interior = int((df['stride_class'] == 0).sum())
                f.write(f"STRIDE Classification:\n")
                f.write(f"  - Exterior (1): {stride_exterior} residues\n")
                f.write(f"  - Interior (0): {stride_interior} residues\n\n")
                #f.write(f"  - STRIDE Cutoff Value: RASA ≥ 0.25 (25% relative accessible surface area)\n")
                #f.write(f"    (If RASA ≥ 0.25, classified as Exterior=1; otherwise Interior=0)\n\n")
                #f.write(f"    Note: params.stride_asa_cutoff={params.stride_asa_cutoff}% is for reference only\n\n")

            # NCPS statistics
            ncps_exterior = int((df['ncps_class'] == 1).sum())
            ncps_interior = int((df['ncps_class'] == 0).sum())
            ncps_auto_surface = int((df['ncps_very_low_neighbors'] == 1).sum())
            f.write(f"NCPS Classification (Our Method):\n")
            f.write(f"  - Exterior (1): {ncps_exterior} residues\n")
            f.write(f"  - Interior (0): {ncps_interior} residues\n\n")
            #f.write(f"  - Automatic Surface (<=2 neighbors): {ncps_auto_surface} residues\n\n")
            #f.write(f"\n  Classification Rules:\n")
            #f.write(f"    Rule 1 - AUTOMATIC SURFACE: If NC6 <= 2 OR NC10 <= 2 → EXTERIOR (definite surface)\n")
            #f.write(f"    Rule 2 - HIGH NEIGHBORS: If NC6 >= {params.nc6_threshold} AND NC10 >= {params.nc10_threshold}\n")
            #f.write(f"             → Check uniformity:\n")
            #f.write(f"                - If Uni6 < {params.uni6_threshold} OR Uni10 < {params.uni10_threshold} → EXTERIOR\n")
            #f.write(f"                - Otherwise → INTERIOR\n")
            #f.write(f"    Rule 3 - INTERMEDIATE/LOW NEIGHBORS: Check uniformity\n")
            #f.write(f"             → If Uni6 < {params.uni6_threshold} OR Uni10 < {params.uni10_threshold} → EXTERIOR\n")
            #f.write(f"             → Else use NC6 as primary indicator\n\n")

            # Agreement
            if df['dssp_class'].notna().sum() > 0:
                df_dssp = df[df['dssp_class'].notna()].copy()
                agreement_dssp = (df_dssp['dssp_class'] == df_dssp['ncps_class']).sum()
                total_with_dssp = len(df_dssp)
                accuracy_dssp = (agreement_dssp / total_with_dssp) * 100
                f.write(f"Agreement with DSSP: {accuracy_dssp:.1f}% ({agreement_dssp}/{total_with_dssp})\n")

            if df['stride_class'].notna().sum() > 0:
                df_stride = df[df['stride_class'].notna()].copy()
                agreement_stride = (df_stride['stride_class'] == df_stride['ncps_class']).sum()
                total_with_stride = len(df_stride)
                accuracy_stride = (agreement_stride / total_with_stride) * 100
                f.write(f"Agreement with STRIDE: {accuracy_stride:.1f}% ({agreement_stride}/{total_with_stride})\n\n")
            else:
                f.write("\n")

            # Neighbor count statistics
            f.write(f"Neighbor Count Statistics:\n")
            f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6'].mean():.1f}, "
                    f"Median={df['ncps_sphere_6'].median():.0f}, "
                    f"Range=[{df['ncps_sphere_6'].min():.0f}-{df['ncps_sphere_6'].max():.0f}]\n")
            f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10'].mean():.1f}, "
                    f"Median={df['ncps_sphere_10'].median():.0f}, "
                    f"Range=[{df['ncps_sphere_10'].min():.0f}-{df['ncps_sphere_10'].max():.0f}]\n\n")

            # Uniformity statistics
            f.write(f"Uniformity Statistics:\n")
            f.write(f"  - 6Å Sphere: Mean={df['ncps_sphere_6_uni'].mean():.2f}, "
                    f"Median={df['ncps_sphere_6_uni'].median():.2f}, "
                    f"Range=[{df['ncps_sphere_6_uni'].min():.2f}-{df['ncps_sphere_6_uni'].max():.2f}]\n")
            f.write(f"  - 10Å Sphere: Mean={df['ncps_sphere_10_uni'].mean():.2f}, "
                    f"Median={df['ncps_sphere_10_uni'].median():.2f}, "
                    f"Range=[{df['ncps_sphere_10_uni'].min():.2f}-{df['ncps_sphere_10_uni'].max():.2f}]\n\n")

            f.write("=" * 120 + "\n\n")

            # Detailed residue data
            f.write("DETAILED RESIDUE DATA\n")
            f.write("=" * 120 + "\n\n")

            # Header
            f.write(" Res   ID   Num |     DSSP   DSSP DSSP |   STRIDE STRIDE STRIDE |  NC6   Uni6  NC10  Uni10 |  NCPS Auto\n")
            f.write("   #            |      ASA  Class   SS |      ASA  Class   SS |                          | Class Surf\n")
            f.write("-" * 120 + "\n")

            # Data rows
            try:
                for idx, row in df.iterrows():
                    try:
                        res_idx = idx + 1
                        res_id = str(row['res_id'])[:3] if pd.notna(row['res_id']) else '---'
                        res_num = int(row['res_num']) if pd.notna(row['res_num']) else 0

                        # DSSP - Handle NaN values
                        if pd.isna(row['dssp_asa']) or pd.isna(row['dssp_class']):
                            dssp_asa = "NaN"
                            dssp_class = "N"
                            dssp_ss = '-'
                        else:
                            dssp_asa = f"{float(row['dssp_asa']):.1f}"
                            dssp_class = str(int(float(row['dssp_class'])))
                            dssp_ss = str(row['dssp_ss']) if row['dssp_ss'] not in ['-', '0.0', 0.0] else '-'

                        # STRIDE - Handle NaN values
                        if pd.isna(row['stride_asa']) or pd.isna(row['stride_class']):
                            stride_asa = "NaN"
                            stride_class = "N"
                            stride_ss = '-'
                        else:
                            stride_asa = f"{float(row['stride_asa']):.1f}"
                            stride_class = str(int(float(row['stride_class'])))
                            stride_ss = str(row['stride_ss']) if row['stride_ss'] not in ['-', '0.0', 0.0] else '-'

                        nc6 = int(float(row['ncps_sphere_6']))
                        uni6 = f"{float(row['ncps_sphere_6_uni']):.3f}"
                        nc10 = int(float(row['ncps_sphere_10']))
                        uni10 = f"{float(row['ncps_sphere_10_uni']):.3f}"

                        ncps_class = int(float(row['ncps_class']))
                        # Safe access to ncps_very_low_neighbors with fallback
                        try:
                            auto_surf = int(row['ncps_very_low_neighbors'])
                        except (KeyError, ValueError, TypeError):
                            auto_surf = 0

                        f.write(f"{res_idx:4d}  {res_id:3s}  {res_num:4d} | {dssp_asa:>7s}      {dssp_class:>1s}    {dssp_ss:1s} | {stride_asa:>7s}      {stride_class:>1s}    {stride_ss:1s} |  {nc6:2d}  {uni6:6s}  {nc10:2d}  {uni10:6s} |     {ncps_class:1d}     {auto_surf:1d}\n")
                    except Exception as e:
                        # If a single row fails, write placeholder and continue
                        f.write(f"ERROR in row {idx}: {str(e)}\n")
                        continue
            except Exception as e:
                f.write(f"ERROR in residue loop: {str(e)}\n")

            f.write("-" * 120 + "\n")

            # STATISTICS SECTION
            f.write("\n" + "=" * 120 + "\n")
            f.write("STATISTICS\n")
            f.write("=" * 120 + "\n\n")

            # DSSP Confusion Matrix
            dssp_has_data = (df_orig['dssp_class'].notna().sum() > 0)
            if dssp_has_data:
                scores = compute_classification_scores(df_orig, 'dssp_class', 'ncps_class')
                tn = scores['counts']['TN']
                fp = scores['counts']['FP']
                fn = scores['counts']['FN']
                tp = scores['counts']['TP']
                dssp_metrics = scores['metrics']

                f.write("ACCORDING TO DSSP:\n")
                f.write("=" * 120 + "\n\n")
                f.write("Confusion Matrix:\n")
                f.write(f"  True Negatives (TN):  {tn:5d}\n")
                f.write(f"  False Positives (FP): {fp:5d}\n")
                f.write(f"  False Negatives (FN): {fn:5d}\n")
                f.write(f"  True Positives (TP):  {tp:5d}\n\n")
                #f.write(f"Accuracy:  {accuracy:.3f}\n")
                #f.write(f"Precision: {precision:.3f}\n")
                #f.write(f"Recall:    {recall:.3f}\n")
                #f.write(f"F1-Score:  {f1:.3f}\n\n")

                # All Metrics from confusion_metrics function
                f.write("All Metrics:\n")
                f.write("-" * 120 + "\n\n")

                for metric_name, metric_value in dssp_metrics.items():
                    if isinstance(metric_value, float):
                        if metric_value == float('inf'):
                            f.write(f"{metric_name}: ∞\n")
                        elif metric_value == float('-inf'):
                            f.write(f"{metric_name}: -∞\n")
                        else:
                            f.write(f"{metric_name}: {metric_value:.4f}\n")
                    else:
                        f.write(f"{metric_name}: {metric_value}\n")


                f.write("\n")

            # STRIDE Confusion Matrix
            stride_has_data = (df_orig['stride_class'].notna().sum() > 0)
            if stride_has_data:
                scores = compute_classification_scores(df_orig, 'stride_class', 'ncps_class')
                tn = scores['counts']['TN']
                fp = scores['counts']['FP']
                fn = scores['counts']['FN']
                tp = scores['counts']['TP']
                stride_metrics = scores['metrics']

                f.write("ACCORDING TO STRIDE:\n")
                f.write("=" * 120 + "\n\n")
                f.write("Confusion Matrix:\n")
                f.write(f"  True Negatives (TN):  {tn:5d}\n")
                f.write(f"  False Positives (FP): {fp:5d}\n")
                f.write(f"  False Negatives (FN): {fn:5d}\n")
                f.write(f"  True Positives (TP):  {tp:5d}\n\n")
                #f.write(f"Accuracy:  {accuracy:.3f}\n")
                #f.write(f"Precision: {precision:.3f}\n")
                #f.write(f"Recall:    {recall:.3f}\n")
                #f.write(f"F1-Score:  {f1:.3f}\n\n")

                # All Metrics from confusion_metrics function
                f.write("All Metrics:\n")
                f.write("-" * 120 + "\n\n")

                for metric_name, metric_value in stride_metrics.items():
                    if isinstance(metric_value, float):
                        if metric_value == float('inf'):
                            f.write(f"{metric_name}: ∞\n")
                        elif metric_value == float('-inf'):
                            f.write(f"{metric_name}: -∞\n")
                        else:
                            f.write(f"{metric_name}: {metric_value:.4f}\n")
                    else:
                        f.write(f"{metric_name}: {metric_value}\n")

                f.write("\n")
        return True

    except Exception as e:
        print(f"    ✗ Error saving report: {e}")
        import traceback
        traceback.print_exc()
        return False


# ================================================================================
# MAIN PROCESSING FUNCTIONS
# ================================================================================

def find_pdb_files(input_folder: Path, search_subdirs: bool = True) -> List[Path]:
    """Find all PDB files in input folder."""
    pdb_files = []

    # Check main directory
    pdb_files.extend(input_folder.glob("*.pdb"))
    pdb_files.extend(input_folder.glob("*.ent"))

    # Check subdirectories if enabled
    if search_subdirs:
        pdb_files.extend(input_folder.glob("*/*.pdb"))
        pdb_files.extend(input_folder.glob("*/*.ent"))

    return sorted(list(set(pdb_files)))


def generate_average_report(protein_results: Dict[str, ProteinAnalysis], output_folder: Path, params: BurialParameters) -> bool:
    """
    Generate an average report across all analyzed proteins.

    Args:
        protein_results: Dictionary of protein_name -> ProteinAnalysis objects
        output_folder: Path to output folder
        params: BurialParameters object with thresholds

    Returns:
        True if successful, False otherwise
    """
    try:
        if not protein_results:
            print("    ✗ No protein results to average")
            return False

        print(f"\n{'='*80}")
        print("GENERATING AVERAGE REPORT")
        print(f"{'='*80}")

        # Collect all metrics from all proteins
        all_metrics = {
            'dssp': {
                'total_residues': [],
                'exterior_count': [],
                'interior_count': [],
                'confusion': {'TP': [], 'TN': [], 'FP': [], 'FN': []},
                'metrics': {}
            },
            'stride': {
                'total_residues': [],
                'exterior_count': [],
                'interior_count': [],
                'confusion': {'TP': [], 'TN': [], 'FP': [], 'FN': []},
                'metrics': {}
            },
            'ncps': {
                'exterior_count': [],
                'interior_count': []
            },
            'neighbor_stats': {
                'nc6_mean': [],
                'nc6_median': [],
                'nc6_min': [],
                'nc6_max': [],
                'nc10_mean': [],
                'nc10_median': [],
                'nc10_min': [],
                'nc10_max': [],
                'uni6_mean': [],
                'uni6_median': [],
                'uni6_min': [],
                'uni6_max': [],
                'uni10_mean': [],
                'uni10_median': [],
                'uni10_min': [],
                'uni10_max': []
            }
        }

        dssp_proteins_count = 0
        stride_proteins_count = 0
        total_proteins = len(protein_results)

        # Iterate through all protein results
        for protein_name, analysis in protein_results.items():
            df = analysis.dataframe

            # NCPS stats (always available)
            all_metrics['ncps']['exterior_count'].append(int((df['ncps_class'] == 1).sum()))
            all_metrics['ncps']['interior_count'].append(int((df['ncps_class'] == 0).sum()))

            # Neighbor count statistics
            all_metrics['neighbor_stats']['nc6_mean'].append(df['ncps_sphere_6'].mean())
            all_metrics['neighbor_stats']['nc6_median'].append(df['ncps_sphere_6'].median())
            all_metrics['neighbor_stats']['nc6_min'].append(df['ncps_sphere_6'].min())
            all_metrics['neighbor_stats']['nc6_max'].append(df['ncps_sphere_6'].max())

            all_metrics['neighbor_stats']['nc10_mean'].append(df['ncps_sphere_10'].mean())
            all_metrics['neighbor_stats']['nc10_median'].append(df['ncps_sphere_10'].median())
            all_metrics['neighbor_stats']['nc10_min'].append(df['ncps_sphere_10'].min())
            all_metrics['neighbor_stats']['nc10_max'].append(df['ncps_sphere_10'].max())

            all_metrics['neighbor_stats']['uni6_mean'].append(df['ncps_sphere_6_uni'].mean())
            all_metrics['neighbor_stats']['uni6_median'].append(df['ncps_sphere_6_uni'].median())
            all_metrics['neighbor_stats']['uni6_min'].append(df['ncps_sphere_6_uni'].min())
            all_metrics['neighbor_stats']['uni6_max'].append(df['ncps_sphere_6_uni'].max())

            all_metrics['neighbor_stats']['uni10_mean'].append(df['ncps_sphere_10_uni'].mean())
            all_metrics['neighbor_stats']['uni10_median'].append(df['ncps_sphere_10_uni'].median())
            all_metrics['neighbor_stats']['uni10_min'].append(df['ncps_sphere_10_uni'].min())
            all_metrics['neighbor_stats']['uni10_max'].append(df['ncps_sphere_10_uni'].max())

            # DSSP stats
            if df['dssp_class'].notna().sum() > 0:
                dssp_proteins_count += 1
                dssp_valid = df[df['dssp_class'].notna()]

                all_metrics['dssp']['total_residues'].append(len(dssp_valid))
                all_metrics['dssp']['exterior_count'].append(int((dssp_valid['dssp_class'] == 1).sum()))
                all_metrics['dssp']['interior_count'].append(int((dssp_valid['dssp_class'] == 0).sum()))

                # Calculate confusion matrix
                scores = compute_classification_scores(df, 'dssp_class', 'ncps_class')
                if scores:
                    all_metrics['dssp']['confusion']['TP'].append(scores['counts']['TP'])
                    all_metrics['dssp']['confusion']['TN'].append(scores['counts']['TN'])
                    all_metrics['dssp']['confusion']['FP'].append(scores['counts']['FP'])
                    all_metrics['dssp']['confusion']['FN'].append(scores['counts']['FN'])

                    # Store all individual metrics
                    for metric_name, metric_value in scores['metrics'].items():
                        if metric_name not in all_metrics['dssp']['metrics']:
                            all_metrics['dssp']['metrics'][metric_name] = []
                        if isinstance(metric_value, (int, float)) and not math.isinf(metric_value) and not math.isnan(metric_value):
                            all_metrics['dssp']['metrics'][metric_name].append(metric_value)

            # STRIDE stats
            if df['stride_class'].notna().sum() > 0:
                stride_proteins_count += 1
                stride_valid = df[df['stride_class'].notna()]

                all_metrics['stride']['total_residues'].append(len(stride_valid))
                all_metrics['stride']['exterior_count'].append(int((stride_valid['stride_class'] == 1).sum()))
                all_metrics['stride']['interior_count'].append(int((stride_valid['stride_class'] == 0).sum()))

                # Calculate confusion matrix
                scores = compute_classification_scores(df, 'stride_class', 'ncps_class')
                if scores:
                    all_metrics['stride']['confusion']['TP'].append(scores['counts']['TP'])
                    all_metrics['stride']['confusion']['TN'].append(scores['counts']['TN'])
                    all_metrics['stride']['confusion']['FP'].append(scores['counts']['FP'])
                    all_metrics['stride']['confusion']['FN'].append(scores['counts']['FN'])

                    # Store all individual metrics
                    for metric_name, metric_value in scores['metrics'].items():
                        if metric_name not in all_metrics['stride']['metrics']:
                            all_metrics['stride']['metrics'][metric_name] = []
                        if isinstance(metric_value, (int, float)) and not math.isinf(metric_value) and not math.isnan(metric_value):
                            all_metrics['stride']['metrics'][metric_name].append(metric_value)

        # Write average report
        output_path = output_folder / "average.txt"

        with open(output_path, 'w') as f:
            # Header
            f.write("=" * 120 + "\n")
            f.write("PROTEIN BURIAL ANALYSIS - AVERAGE REPORT ACROSS ALL PROTEINS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 120 + "\n\n")

            # General statistics
            f.write("GENERAL STATISTICS\n")
            f.write("-" * 120 + "\n")
            f.write(f"Total Proteins Analyzed: {total_proteins}\n")
            f.write(f"Proteins with DSSP data: {dssp_proteins_count}\n")
            f.write(f"Proteins with STRIDE data: {stride_proteins_count}\n\n")

            # Parameters used
            f.write("PARAMETERS USED:\n")
            f.write("-" * 120 + "\n")
            f.write(f"NC6 Threshold:    {params.nc6_threshold}\n")
            f.write(f"NC10 Threshold:   {params.nc10_threshold}\n")
            f.write(f"Uni6 Threshold:   {params.uni6_threshold}\n")
            f.write(f"Uni10 Threshold:  {params.uni10_threshold}\n")
            f.write(f"DSSP ASA Cutoff:  {params.dssp_asa_cutoff}%\n")
            f.write(f"STRIDE ASA Cutoff: {params.stride_asa_cutoff}%\n\n")

            # NCPS Classification Averages
            f.write("=" * 120 + "\n")
            f.write("NCPS CLASSIFICATION - AVERAGE STATISTICS\n")
            f.write("=" * 120 + "\n\n")

            if all_metrics['ncps']['exterior_count']:
                avg_ncps_ext = np.mean(all_metrics['ncps']['exterior_count'])
                avg_ncps_int = np.mean(all_metrics['ncps']['interior_count'])
                f.write(f"Average Exterior Residues (NCPS): {avg_ncps_ext:.2f}\n")
                f.write(f"Average Interior Residues (NCPS): {avg_ncps_int:.2f}\n\n")

            # Neighbor Count Statistics
            f.write("NEIGHBOR COUNT STATISTICS - AVERAGES\n")
            f.write("-" * 120 + "\n\n")

            ns = all_metrics['neighbor_stats']
            if ns['nc6_mean']:
                f.write("6Å Sphere:\n")
                f.write(f"  Mean NC6:     {np.mean(ns['nc6_mean']):.2f}\n")
                f.write(f"  Median NC6:   {np.mean(ns['nc6_median']):.2f}\n")
                f.write(f"  Min NC6:      {np.mean(ns['nc6_min']):.2f}\n")
                f.write(f"  Max NC6:      {np.mean(ns['nc6_max']):.2f}\n\n")

                f.write("10Å Sphere:\n")
                f.write(f"  Mean NC10:    {np.mean(ns['nc10_mean']):.2f}\n")
                f.write(f"  Median NC10:  {np.mean(ns['nc10_median']):.2f}\n")
                f.write(f"  Min NC10:     {np.mean(ns['nc10_min']):.2f}\n")
                f.write(f"  Max NC10:     {np.mean(ns['nc10_max']):.2f}\n\n")

            # Uniformity Statistics
            f.write("UNIFORMITY STATISTICS - AVERAGES\n")
            f.write("-" * 120 + "\n\n")

            if ns['uni6_mean']:
                f.write("6Å Sphere:\n")
                f.write(f"  Mean Uni6:    {np.mean(ns['uni6_mean']):.4f}\n")
                f.write(f"  Median Uni6:  {np.mean(ns['uni6_median']):.4f}\n")
                f.write(f"  Min Uni6:     {np.mean(ns['uni6_min']):.4f}\n")
                f.write(f"  Max Uni6:     {np.mean(ns['uni6_max']):.4f}\n\n")

                f.write("10Å Sphere:\n")
                f.write(f"  Mean Uni10:   {np.mean(ns['uni10_mean']):.4f}\n")
                f.write(f"  Median Uni10: {np.mean(ns['uni10_median']):.4f}\n")
                f.write(f"  Min Uni10:    {np.mean(ns['uni10_min']):.4f}\n")
                f.write(f"  Max Uni10:    {np.mean(ns['uni10_max']):.4f}\n\n")

            # DSSP Statistics
            if dssp_proteins_count > 0:
                f.write("=" * 120 + "\n")
                f.write(f"DSSP COMPARISON - AVERAGE STATISTICS ({dssp_proteins_count} proteins)\n")
                f.write("=" * 120 + "\n\n")

                if all_metrics['dssp']['total_residues']:
                    avg_dssp_residues = np.mean(all_metrics['dssp']['total_residues'])
                    avg_dssp_ext = np.mean(all_metrics['dssp']['exterior_count'])
                    avg_dssp_int = np.mean(all_metrics['dssp']['interior_count'])

                    f.write(f"Average Total Residues (DSSP): {avg_dssp_residues:.2f}\n")
                    f.write(f"Average Exterior Residues (DSSP): {avg_dssp_ext:.2f}\n")
                    f.write(f"Average Interior Residues (DSSP): {avg_dssp_int:.2f}\n\n")

                # Confusion Matrix Averages
                if all_metrics['dssp']['confusion']['TP']:
                    avg_tp = np.mean(all_metrics['dssp']['confusion']['TP'])
                    avg_tn = np.mean(all_metrics['dssp']['confusion']['TN'])
                    avg_fp = np.mean(all_metrics['dssp']['confusion']['FP'])
                    avg_fn = np.mean(all_metrics['dssp']['confusion']['FN'])

                    f.write("Average Confusion Matrix (NCPS vs DSSP):\n")
                    f.write(f"  True Positives (TP):  {avg_tp:.2f}\n")
                    f.write(f"  True Negatives (TN):  {avg_tn:.2f}\n")
                    f.write(f"  False Positives (FP): {avg_fp:.2f}\n")
                    f.write(f"  False Negatives (FN): {avg_fn:.2f}\n\n")

                # All metrics averages
                if all_metrics['dssp']['metrics']:
                    f.write("Average Metrics (NCPS vs DSSP):\n")
                    f.write("-" * 120 + "\n")

                    for metric_name in sorted(all_metrics['dssp']['metrics'].keys()):
                        values = all_metrics['dssp']['metrics'][metric_name]
                        if values:
                            avg_value = np.mean(values)
                            std_value = np.std(values)
                            min_value = np.min(values)
                            max_value = np.max(values)
                            f.write(f"{metric_name:30s}: {avg_value:8.4f} (±{std_value:.4f}) [min={min_value:.4f}, max={max_value:.4f}]\n")
                    f.write("\n")

            # STRIDE Statistics
            if stride_proteins_count > 0:
                f.write("=" * 120 + "\n")
                f.write(f"STRIDE COMPARISON - AVERAGE STATISTICS ({stride_proteins_count} proteins)\n")
                f.write("=" * 120 + "\n\n")

                if all_metrics['stride']['total_residues']:
                    avg_stride_residues = np.mean(all_metrics['stride']['total_residues'])
                    avg_stride_ext = np.mean(all_metrics['stride']['exterior_count'])
                    avg_stride_int = np.mean(all_metrics['stride']['interior_count'])

                    f.write(f"Average Total Residues (STRIDE): {avg_stride_residues:.2f}\n")
                    f.write(f"Average Exterior Residues (STRIDE): {avg_stride_ext:.2f}\n")
                    f.write(f"Average Interior Residues (STRIDE): {avg_stride_int:.2f}\n\n")

                # Confusion Matrix Averages
                if all_metrics['stride']['confusion']['TP']:
                    avg_tp = np.mean(all_metrics['stride']['confusion']['TP'])
                    avg_tn = np.mean(all_metrics['stride']['confusion']['TN'])
                    avg_fp = np.mean(all_metrics['stride']['confusion']['FP'])
                    avg_fn = np.mean(all_metrics['stride']['confusion']['FN'])

                    f.write("Average Confusion Matrix (NCPS vs STRIDE):\n")
                    f.write(f"  True Positives (TP):  {avg_tp:.2f}\n")
                    f.write(f"  True Negatives (TN):  {avg_tn:.2f}\n")
                    f.write(f"  False Positives (FP): {avg_fp:.2f}\n")
                    f.write(f"  False Negatives (FN): {avg_fn:.2f}\n\n")

                # All metrics averages
                if all_metrics['stride']['metrics']:
                    f.write("Average Metrics (NCPS vs STRIDE):\n")
                    f.write("-" * 120 + "\n")

                    for metric_name in sorted(all_metrics['stride']['metrics'].keys()):
                        values = all_metrics['stride']['metrics'][metric_name]
                        if values:
                            avg_value = np.mean(values)
                            std_value = np.std(values)
                            min_value = np.min(values)
                            max_value = np.max(values)
                            f.write(f"{metric_name:30s}: {avg_value:8.4f} (±{std_value:.4f}) [min={min_value:.4f}, max={max_value:.4f}]\n")
                    f.write("\n")

            # Footer
            f.write("=" * 120 + "\n")
            f.write("END OF AVERAGE REPORT\n")
            f.write("=" * 120 + "\n")

        print(f"    ✓ Average report saved: {output_path.name}")
        return True

    except Exception as e:
        print(f"    ✗ Error generating average report: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("PROTEIN BURIAL ANALYSIS - FINAL ANALYSIS SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input folder:     {CONFIG['pdb_input_folder']}")
    print(f"  Output folder:    {CONFIG['output_folder']}")
    print(f"  NC6 threshold:    {CONFIG['nc6_threshold']}")
    print(f"  NC10 threshold:   {CONFIG['nc10_threshold']}")
    print(f"  Uni6 threshold:   {CONFIG['uni6_threshold']}")
    print(f"  Uni10 threshold:  {CONFIG['uni10_threshold']}")
    print()

    # Create input and output folders
    input_folder = Path(CONFIG['pdb_input_folder'])
    output_folder = Path(CONFIG['output_folder'])

    # Check if input folder exists
    if not input_folder.exists():
        print(f"ERROR: Input folder not found: {input_folder}")
        return False

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find PDB files
    pdb_files = find_pdb_files(input_folder, CONFIG['search_subdirectories'])

    if not pdb_files:
        print(f"ERROR: No PDB files found in {input_folder}")
        return False

    print(f"Found {len(pdb_files)} PDB file(s) to process:")
    for pdb_file in pdb_files:
        print(f"  - {pdb_file.relative_to(input_folder) if input_folder in pdb_file.parents or input_folder == pdb_file.parent else pdb_file.name}")
    print()

    # Create parameters object
    params = BurialParameters(
        nc6_threshold=CONFIG['nc6_threshold'],
        nc10_threshold=CONFIG['nc10_threshold'],
        uni6_threshold=CONFIG['uni6_threshold'],
        uni10_threshold=CONFIG['uni10_threshold'],
        dssp_asa_cutoff=CONFIG['dssp_asa_cutoff'],
        stride_asa_cutoff=CONFIG['stride_asa_cutoff']
    )

    # Process each PDB file and group by protein (folder name)
    successful = 0
    failed = 0
    protein_results = {}  # Store results by protein name

    for pdb_file in pdb_files:
        analysis = process_single_protein(pdb_file, params, verbose=CONFIG['verbose'])

        if analysis is None:
            failed += 1
            continue

        successful += 1

        # Get protein name from folder if it exists, otherwise use file name
        if pdb_file.parent.name != 'pdbexamples':
            protein_name = pdb_file.parent.name.lower()
        else:
            protein_name = pdb_file.stem.lower()

        # Use the first file for each protein, or combine if multiple
        if protein_name not in protein_results:
            protein_results[protein_name] = analysis
        else:
            # If multiple files for same protein, use the best one
            if CONFIG['verbose']:
                print(f"  Note: Using first file for {protein_name}")

    # Generate output files for each protein
    print(f"\nGenerating output files:")
    for protein_name, analysis in protein_results.items():
        # Save CSV with protein name
        csv_path = output_folder / f"{protein_name}_detailed_results.csv"
        save_detailed_csv(analysis, csv_path)

        # Save report with protein name
        report_path = output_folder / f"{protein_name}_detailed_report.txt"
        save_detailed_report(analysis, report_path, params)

    # Generate average report across all proteins
    if protein_results:
        generate_average_report(protein_results, output_folder, params)

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  ✓ Successfully processed: {successful} proteins")
    print(f"  ✗ Failed to process:     {failed}")
    print(f"  ✓ Unique proteins:       {len(protein_results)}")
    print(f"\nOutput folder: {output_folder.absolute()}")
    print(f"Files generated: {len(protein_results)} CSV files + {len(protein_results)} TXT reports + 1 average report")
    print("\n" + "=" * 80 + "\n")

    return successful > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


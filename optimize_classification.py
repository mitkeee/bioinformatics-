"""
Optimization Script for classify_burial() Function
Tests different parameter combinations to maximize accuracy
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Import from final_analysis.py
sys.path.insert(0, str(Path(__file__).parent))
from final_analysis import (
    extract_ca_atoms, calculate_neighbor_counts, calculate_uniformity,
    extract_dssp_data, extract_stride_data, BurialParameters,
    confusion_matrix_local, accuracy_score_local, f1_score_local,
    matthews_corrcoef_local, balanced_accuracy_score_local
)

def classify_burial_optimized(df, params):
    """
    Optimized version of classify_burial with configurable parameters.
    """
    ncps_class = []

    for _, row in df.iterrows():
        nc6 = row['ncps_sphere_6']
        nc10 = row['ncps_sphere_10']
        uni6 = row['ncps_sphere_6_uni']
        uni10 = row['ncps_sphere_10_uni']

        is_exterior = False

        # RULE 1: AUTOMATIC SURFACE - very few neighbors
        if nc6 <= params.auto_surface_nc6 or nc10 <= params.auto_surface_nc10:
            is_exterior = True
        # RULE 2: Check if both neighbor counts are above thresholds
        elif nc6 >= params.nc6_threshold and nc10 >= params.nc10_threshold:
            # Both neighbor counts are high - check uniformity
            if (pd.notna(uni6) and uni6 < params.uni6_threshold) or (pd.notna(uni10) and uni10 < params.uni10_threshold):
                is_exterior = True
            else:
                is_exterior = False
        # RULE 3: Intermediate neighbor counts
        else:
            if (pd.notna(uni6) and uni6 < params.uni6_threshold) or (pd.notna(uni10) and uni10 < params.uni10_threshold):
                is_exterior = True
            else:
                if nc6 < params.nc6_threshold:
                    is_exterior = True
                else:
                    is_exterior = False

        ncps_class.append(1 if is_exterior else 0)

    return np.array(ncps_class)


class OptimizedBurialParameters:
    """Extended parameters with auto_surface thresholds"""
    def __init__(self, nc6_threshold=5.0, nc10_threshold=16.0,
                 uni6_threshold=0.40, uni10_threshold=0.50,
                 auto_surface_nc6=2, auto_surface_nc10=2,
                 dssp_asa_cutoff=25.0, stride_asa_cutoff=20.0):
        self.nc6_threshold = nc6_threshold
        self.nc10_threshold = nc10_threshold
        self.uni6_threshold = uni6_threshold
        self.uni10_threshold = uni10_threshold
        self.auto_surface_nc6 = auto_surface_nc6
        self.auto_surface_nc10 = auto_surface_nc10
        self.dssp_asa_cutoff = dssp_asa_cutoff
        self.stride_asa_cutoff = stride_asa_cutoff


def load_and_prepare_protein(pdb_path, params):
    """Load a protein and prepare features"""
    from Bio.PDB import PDBParser

    # Extract CA atoms
    df = extract_ca_atoms(pdb_path)

    # Calculate coordinates
    coords = df[['x', 'y', 'z']].values

    # Calculate neighbor counts and uniformity
    df['ncps_sphere_6'] = calculate_neighbor_counts(coords, 6.0)
    df['ncps_sphere_10'] = calculate_neighbor_counts(coords, 10.0)
    df['ncps_sphere_6_uni'] = calculate_uniformity(coords, 6.0)
    df['ncps_sphere_10_uni'] = calculate_uniformity(coords, 10.0)

    # Extract DSSP and STRIDE data
    df = extract_dssp_data(df, pdb_path, params)
    df = extract_stride_data(df, pdb_path, params)

    return df


def evaluate_parameters(pdb_files, params, reference='dssp'):
    """Evaluate classification parameters across all proteins"""
    all_accuracies = []
    all_f1_scores = []
    all_mcc = []
    all_balanced_acc = []

    for pdb_file in pdb_files:
        try:
            df = load_and_prepare_protein(pdb_file, params)

            # Classify using optimized function
            df['ncps_class'] = classify_burial_optimized(df, params)

            # Get reference column
            ref_col = 'dssp_class' if reference == 'dssp' else 'stride_class'

            # Filter valid data
            df_valid = df[df[ref_col].notna()].copy()

            if len(df_valid) == 0:
                continue

            y_true = df_valid[ref_col].values.astype(int)
            y_pred = df_valid['ncps_class'].values.astype(int)

            # Calculate metrics
            acc = accuracy_score_local(y_true, y_pred)
            f1 = f1_score_local(y_true, y_pred)
            mcc = matthews_corrcoef_local(y_true, y_pred)
            bal_acc = balanced_accuracy_score_local(y_true, y_pred)

            all_accuracies.append(acc)
            all_f1_scores.append(f1)
            all_mcc.append(mcc)
            all_balanced_acc.append(bal_acc)

        except Exception as e:
            continue

    if not all_accuracies:
        return None

    return {
        'accuracy': np.mean(all_accuracies),
        'f1_score': np.mean(all_f1_scores),
        'mcc': np.mean(all_mcc),
        'balanced_accuracy': np.mean(all_balanced_acc),
        'n_proteins': len(all_accuracies)
    }


def grid_search_optimization(pdb_files, reference='dssp'):
    """
    Perform grid search to find optimal parameters
    """
    print(f"\n{'='*80}")
    print(f"PARAMETER OPTIMIZATION - Grid Search")
    print(f"Reference: {reference.upper()}")
    print(f"Proteins: {len(pdb_files)}")
    print(f"{'='*80}\n")

    # Define parameter ranges to test
    nc6_values = [3, 4, 5, 6, 7]
    nc10_values = [12, 14, 16, 18, 20]
    uni6_values = [0.30, 0.35, 0.40, 0.45, 0.50]
    uni10_values = [0.40, 0.45, 0.50, 0.55, 0.60]
    auto_nc6_values = [1, 2, 3]
    auto_nc10_values = [1, 2, 3]

    best_result = None
    best_params = None
    best_accuracy = 0

    total_combinations = (len(nc6_values) * len(nc10_values) *
                         len(uni6_values) * len(uni10_values) *
                         len(auto_nc6_values) * len(auto_nc10_values))

    print(f"Testing {total_combinations} parameter combinations...")
    print(f"This may take a while...\n")

    tested = 0

    # Grid search
    for nc6, nc10, uni6, uni10, auto_nc6, auto_nc10 in product(
        nc6_values, nc10_values, uni6_values, uni10_values,
        auto_nc6_values, auto_nc10_values
    ):
        tested += 1

        params = OptimizedBurialParameters(
            nc6_threshold=nc6,
            nc10_threshold=nc10,
            uni6_threshold=uni6,
            uni10_threshold=uni10,
            auto_surface_nc6=auto_nc6,
            auto_surface_nc10=auto_nc10
        )

        result = evaluate_parameters(pdb_files, params, reference)

        if result is None:
            continue

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_result = result
            best_params = params

            print(f"[{tested}/{total_combinations}] New Best! Accuracy: {best_accuracy:.4f}")
            print(f"  nc6={nc6}, nc10={nc10}, uni6={uni6:.2f}, uni10={uni10:.2f}, "
                  f"auto_nc6={auto_nc6}, auto_nc10={auto_nc10}")
            print(f"  F1={result['f1_score']:.4f}, MCC={result['mcc']:.4f}, "
                  f"Bal_Acc={result['balanced_accuracy']:.4f}\n")

        if tested % 100 == 0:
            print(f"Progress: {tested}/{total_combinations} ({100*tested/total_combinations:.1f}%)")

    return best_params, best_result


def focused_search_optimization(pdb_files, reference='dssp', initial_params=None):
    """
    More focused search around current/initial parameters - FAST VERSION
    """
    print(f"\n{'='*80}")
    print(f"PARAMETER OPTIMIZATION - Focused Search")
    print(f"Reference: {reference.upper()}")
    print(f"{'='*80}\n")

    if initial_params is None:
        # Current parameters
        initial_params = OptimizedBurialParameters(
            nc6_threshold=5.0,
            nc10_threshold=16.0,
            uni6_threshold=0.40,
            uni10_threshold=0.50,
            auto_surface_nc6=2,
            auto_surface_nc10=2
        )

    # Reduced test ranges for speed - focus on most impactful parameters
    nc6_values = [initial_params.nc6_threshold - 1,
                  initial_params.nc6_threshold,
                  initial_params.nc6_threshold + 1]

    nc10_values = [initial_params.nc10_threshold - 2,
                   initial_params.nc10_threshold,
                   initial_params.nc10_threshold + 2]

    uni6_values = [max(0.2, initial_params.uni6_threshold - 0.05),
                   initial_params.uni6_threshold,
                   min(0.8, initial_params.uni6_threshold + 0.05)]

    uni10_values = [max(0.2, initial_params.uni10_threshold - 0.05),
                    initial_params.uni10_threshold,
                    min(0.8, initial_params.uni10_threshold + 0.05)]

    auto_nc6_values = [2]  # Keep fixed
    auto_nc10_values = [2]  # Keep fixed

    best_result = None
    best_params = None
    best_accuracy = 0

    total = len(nc6_values) * len(nc10_values) * len(uni6_values) * len(uni10_values) * len(auto_nc6_values) * len(auto_nc10_values)
    tested = 0

    print(f"Testing {total} parameter combinations (optimized for speed)...\n")

    for nc6, nc10, uni6, uni10, auto_nc6, auto_nc10 in product(
        nc6_values, nc10_values, uni6_values, uni10_values,
        auto_nc6_values, auto_nc10_values
    ):
        tested += 1

        params = OptimizedBurialParameters(
            nc6_threshold=nc6,
            nc10_threshold=nc10,
            uni6_threshold=uni6,
            uni10_threshold=uni10,
            auto_surface_nc6=auto_nc6,
            auto_surface_nc10=auto_nc10
        )

        result = evaluate_parameters(pdb_files, params, reference)

        if result is None:
            continue

        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_result = result
            best_params = params

            print(f"[{tested}/{total}] ✓ New Best! Accuracy: {best_accuracy:.4f}")
            print(f"  Parameters:")
            print(f"    nc6_threshold={nc6:.1f}, nc10_threshold={nc10:.1f}")
            print(f"    uni6_threshold={uni6:.2f}, uni10_threshold={uni10:.2f}")
            print(f"    auto_surface_nc6={auto_nc6}, auto_surface_nc10={auto_nc10}")
            print(f"  Metrics:")
            print(f"    F1 Score={result['f1_score']:.4f}, MCC={result['mcc']:.4f}")
            print(f"    Balanced Accuracy={result['balanced_accuracy']:.4f}\n")

        # Print progress every 10 tests
        if tested % 10 == 0:
            print(f"Progress: {tested}/{total} ({100*tested/total:.1f}%) - Current best: {best_accuracy:.4f}")

    return best_params, best_result


def main():
    """Main optimization routine"""

    # Find PDB files
    pdb_folder = Path('pdbexamples')
    pdb_files = []

    # Get unique proteins (first file from each folder)
    seen_proteins = set()
    for pdb_file in sorted(pdb_folder.glob('*/*.pdb')):
        protein_name = pdb_file.parent.name
        if protein_name not in seen_proteins:
            pdb_files.append(pdb_file)
            seen_proteins.add(protein_name)

    print(f"\nFound {len(pdb_files)} unique proteins to analyze")

    # Test current parameters first
    print(f"\n{'='*80}")
    print("BASELINE PERFORMANCE (Current Parameters)")
    print(f"{'='*80}\n")

    current_params = OptimizedBurialParameters(
        nc6_threshold=5.0,
        nc10_threshold=16.0,
        uni6_threshold=0.40,
        uni10_threshold=0.50,
        auto_surface_nc6=2,
        auto_surface_nc10=2
    )

    print("Testing vs DSSP...")
    dssp_baseline = evaluate_parameters(pdb_files, current_params, 'dssp')
    if dssp_baseline:
        print(f"DSSP Baseline:")
        print(f"  Accuracy: {dssp_baseline['accuracy']:.4f}")
        print(f"  F1 Score: {dssp_baseline['f1_score']:.4f}")
        print(f"  MCC: {dssp_baseline['mcc']:.4f}")
        print(f"  Balanced Accuracy: {dssp_baseline['balanced_accuracy']:.4f}")

    print("\nTesting vs STRIDE...")
    stride_baseline = evaluate_parameters(pdb_files, current_params, 'stride')
    if stride_baseline:
        print(f"STRIDE Baseline:")
        print(f"  Accuracy: {stride_baseline['accuracy']:.4f}")
        print(f"  F1 Score: {stride_baseline['f1_score']:.4f}")
        print(f"  MCC: {stride_baseline['mcc']:.4f}")
        print(f"  Balanced Accuracy: {stride_baseline['balanced_accuracy']:.4f}")

    # Optimize for DSSP
    print(f"\n{'='*80}")
    print("OPTIMIZING FOR DSSP")
    print(f"{'='*80}")
    best_dssp_params, best_dssp_result = focused_search_optimization(pdb_files, 'dssp', current_params)

    # Optimize for STRIDE
    print(f"\n{'='*80}")
    print("OPTIMIZING FOR STRIDE")
    print(f"{'='*80}")
    best_stride_params, best_stride_result = focused_search_optimization(pdb_files, 'stride', current_params)

    # Print final results
    print(f"\n{'='*80}")
    print("OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*80}\n")

    print("BEST PARAMETERS FOR DSSP:")
    print(f"  nc6_threshold: {best_dssp_params.nc6_threshold}")
    print(f"  nc10_threshold: {best_dssp_params.nc10_threshold}")
    print(f"  uni6_threshold: {best_dssp_params.uni6_threshold}")
    print(f"  uni10_threshold: {best_dssp_params.uni10_threshold}")
    print(f"  auto_surface_nc6: {best_dssp_params.auto_surface_nc6}")
    print(f"  auto_surface_nc10: {best_dssp_params.auto_surface_nc10}")
    print(f"\n  Performance:")
    print(f"    Accuracy: {best_dssp_result['accuracy']:.4f} (baseline: {dssp_baseline['accuracy']:.4f})")
    print(f"    F1 Score: {best_dssp_result['f1_score']:.4f} (baseline: {dssp_baseline['f1_score']:.4f})")
    print(f"    MCC: {best_dssp_result['mcc']:.4f} (baseline: {dssp_baseline['mcc']:.4f})")
    print(f"    Balanced Accuracy: {best_dssp_result['balanced_accuracy']:.4f} (baseline: {dssp_baseline['balanced_accuracy']:.4f})")

    print(f"\nBEST PARAMETERS FOR STRIDE:")
    print(f"  nc6_threshold: {best_stride_params.nc6_threshold}")
    print(f"  nc10_threshold: {best_stride_params.nc10_threshold}")
    print(f"  uni6_threshold: {best_stride_params.uni6_threshold}")
    print(f"  uni10_threshold: {best_stride_params.uni10_threshold}")
    print(f"  auto_surface_nc6: {best_stride_params.auto_surface_nc6}")
    print(f"  auto_surface_nc10: {best_stride_params.auto_surface_nc10}")
    print(f"\n  Performance:")
    print(f"    Accuracy: {best_stride_result['accuracy']:.4f} (baseline: {stride_baseline['accuracy']:.4f})")
    print(f"    F1 Score: {best_stride_result['f1_score']:.4f} (baseline: {stride_baseline['f1_score']:.4f})")
    print(f"    MCC: {best_stride_result['mcc']:.4f} (baseline: {stride_baseline['mcc']:.4f})")
    print(f"    Balanced Accuracy: {best_stride_result['balanced_accuracy']:.4f} (baseline: {stride_baseline['balanced_accuracy']:.4f})")

    # Save results
    with open('optimization_results.txt', 'w') as f:
        f.write("OPTIMIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analyzed {len(pdb_files)} proteins\n\n")

        f.write("BEST PARAMETERS FOR DSSP:\n")
        f.write(f"  nc6_threshold = {best_dssp_params.nc6_threshold}\n")
        f.write(f"  nc10_threshold = {best_dssp_params.nc10_threshold}\n")
        f.write(f"  uni6_threshold = {best_dssp_params.uni6_threshold}\n")
        f.write(f"  uni10_threshold = {best_dssp_params.uni10_threshold}\n")
        f.write(f"  auto_surface_nc6 = {best_dssp_params.auto_surface_nc6}\n")
        f.write(f"  auto_surface_nc10 = {best_dssp_params.auto_surface_nc10}\n")
        f.write(f"\n  Accuracy: {best_dssp_result['accuracy']:.4f}\n")
        f.write(f"  Improvement: {100*(best_dssp_result['accuracy']-dssp_baseline['accuracy']):.2f}%\n\n")

        f.write("BEST PARAMETERS FOR STRIDE:\n")
        f.write(f"  nc6_threshold = {best_stride_params.nc6_threshold}\n")
        f.write(f"  nc10_threshold = {best_stride_params.nc10_threshold}\n")
        f.write(f"  uni6_threshold = {best_stride_params.uni6_threshold}\n")
        f.write(f"  uni10_threshold = {best_stride_params.uni10_threshold}\n")
        f.write(f"  auto_surface_nc6 = {best_stride_params.auto_surface_nc6}\n")
        f.write(f"  auto_surface_nc10 = {best_stride_params.auto_surface_nc10}\n")
        f.write(f"\n  Accuracy: {best_stride_result['accuracy']:.4f}\n")
        f.write(f"  Improvement: {100*(best_stride_result['accuracy']-stride_baseline['accuracy']):.2f}%\n")

    print(f"\n✓ Results saved to optimization_results.txt")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


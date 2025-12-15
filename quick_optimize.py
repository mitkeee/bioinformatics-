"""
Quick parameter optimization - tests a small set of promising parameter combinations
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from final_analysis import (
    extract_ca_atoms, calculate_neighbor_counts, calculate_uniformity,
    extract_dssp_data, extract_stride_data, BurialParameters,
    accuracy_score_local, f1_score_local, matthews_corrcoef_local,
    balanced_accuracy_score_local, precision_score_local, recall_score_local
)


class TestParams:
    def __init__(self, nc6=5.0, nc10=16.0, uni6=0.40, uni10=0.50, auto6=2, auto10=2):
        self.nc6_threshold = nc6
        self.nc10_threshold = nc10
        self.uni6_threshold = uni6
        self.uni10_threshold = uni10
        self.auto_surface_nc6 = auto6
        self.auto_surface_nc10 = auto10
        self.dssp_asa_cutoff = 25.0
        self.stride_asa_cutoff = 20.0


def classify_burial_test(df, params):
    """Test classification"""
    ncps_class = []
    for _, row in df.iterrows():
        nc6 = row['ncps_sphere_6']
        nc10 = row['ncps_sphere_10']
        uni6 = row['ncps_sphere_6_uni']
        uni10 = row['ncps_sphere_10_uni']

        is_exterior = False

        if nc6 <= params.auto_surface_nc6 or nc10 <= params.auto_surface_nc10:
            is_exterior = True
        elif nc6 >= params.nc6_threshold and nc10 >= params.nc10_threshold:
            if (pd.notna(uni6) and uni6 < params.uni6_threshold) or (pd.notna(uni10) and uni10 < params.uni10_threshold):
                is_exterior = True
            else:
                is_exterior = False
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


def test_params_on_protein(pdb_file, params, reference='dssp'):
    """Test parameters on a single protein"""
    try:
        # Load protein
        df = extract_ca_atoms(pdb_file)
        coords = df[['x', 'y', 'z']].values

        # Calculate features
        df['ncps_sphere_6'] = calculate_neighbor_counts(coords, 6.0)
        df['ncps_sphere_10'] = calculate_neighbor_counts(coords, 10.0)
        df['ncps_sphere_6_uni'] = calculate_uniformity(coords, 6.0)
        df['ncps_sphere_10_uni'] = calculate_uniformity(coords, 10.0)

        # Get reference data
        df = extract_dssp_data(df, pdb_file, params)
        df = extract_stride_data(df, pdb_file, params)

        # Classify
        df['ncps_class'] = classify_burial_test(df, params)

        # Evaluate
        ref_col = 'dssp_class' if reference == 'dssp' else 'stride_class'
        df_valid = df[df[ref_col].notna()].copy()

        if len(df_valid) == 0:
            return None

        y_true = df_valid[ref_col].values.astype(int)
        y_pred = df_valid['ncps_class'].values.astype(int)

        return {
            'accuracy': accuracy_score_local(y_true, y_pred),
            'precision': precision_score_local(y_true, y_pred),
            'recall': recall_score_local(y_true, y_pred),
            'f1': f1_score_local(y_true, y_pred),
            'mcc': matthews_corrcoef_local(y_true, y_pred),
            'balanced_acc': balanced_accuracy_score_local(y_true, y_pred)
        }
    except:
        return None


def main():
    # Find proteins
    pdb_folder = Path('pdbexamples')
    pdb_files = []
    seen = set()
    for pdb in sorted(pdb_folder.glob('*/*.pdb')):
        name = pdb.parent.name
        if name not in seen:
            pdb_files.append(pdb)
            seen.add(name)

    print(f"\nTesting on {len(pdb_files)} proteins\n")
    print(f"{'='*80}\n")

    # Test different parameter combinations
    test_configs = [
        # (nc6, nc10, uni6, uni10, auto6, auto10, description)
        (5.0, 16.0, 0.40, 0.50, 2, 2, "Current"),
        (6.0, 18.0, 0.40, 0.50, 2, 2, "Increase NC thresholds"),
        (4.0, 14.0, 0.40, 0.50, 2, 2, "Decrease NC thresholds"),
        (5.0, 16.0, 0.35, 0.45, 2, 2, "Decrease Uni thresholds"),
        (5.0, 16.0, 0.45, 0.55, 2, 2, "Increase Uni thresholds"),
        (6.0, 18.0, 0.35, 0.45, 2, 2, "Higher NC, Lower Uni"),
        (4.0, 14.0, 0.45, 0.55, 2, 2, "Lower NC, Higher Uni"),
        (5.0, 16.0, 0.40, 0.50, 1, 1, "Stricter auto-surface"),
        (5.0, 16.0, 0.40, 0.50, 3, 3, "Looser auto-surface"),
    ]

    best_dssp = {'accuracy': 0, 'params': None, 'desc': ''}
    best_stride = {'accuracy': 0, 'params': None, 'desc': ''}

    for config in test_configs:
        nc6, nc10, uni6, uni10, auto6, auto10, desc = config
        params = TestParams(nc6, nc10, uni6, uni10, auto6, auto10)

        print(f"Testing: {desc}")
        print(f"  nc6={nc6}, nc10={nc10}, uni6={uni6}, uni10={uni10}, auto6={auto6}, auto10={auto10}")

        dssp_results = []
        stride_results = []

        for pdb in pdb_files:
            res_dssp = test_params_on_protein(pdb, params, 'dssp')
            res_stride = test_params_on_protein(pdb, params, 'stride')

            if res_dssp:
                dssp_results.append(res_dssp['accuracy'])
            if res_stride:
                stride_results.append(res_stride['accuracy'])

        if dssp_results:
            dssp_acc = np.mean(dssp_results)
            print(f"  DSSP Accuracy: {dssp_acc:.4f}")
            if dssp_acc > best_dssp['accuracy']:
                best_dssp = {'accuracy': dssp_acc, 'params': params, 'desc': desc}
                print(f"    ✓ NEW BEST FOR DSSP!")

        if stride_results:
            stride_acc = np.mean(stride_results)
            print(f"  STRIDE Accuracy: {stride_acc:.4f}")
            if stride_acc > best_stride['accuracy']:
                best_stride = {'accuracy': stride_acc, 'params': params, 'desc': desc}
                print(f"    ✓ NEW BEST FOR STRIDE!")

        print()

    print(f"{'='*80}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*80}\n")

    print("BEST FOR DSSP:")
    print(f"  Description: {best_dssp['desc']}")
    p = best_dssp['params']
    print(f"  nc6_threshold: {p.nc6_threshold}")
    print(f"  nc10_threshold: {p.nc10_threshold}")
    print(f"  uni6_threshold: {p.uni6_threshold}")
    print(f"  uni10_threshold: {p.uni10_threshold}")
    print(f"  auto_surface_nc6: {p.auto_surface_nc6}")
    print(f"  auto_surface_nc10: {p.auto_surface_nc10}")
    print(f"  Accuracy: {best_dssp['accuracy']:.4f}\n")

    print("BEST FOR STRIDE:")
    print(f"  Description: {best_stride['desc']}")
    p = best_stride['params']
    print(f"  nc6_threshold: {p.nc6_threshold}")
    print(f"  nc10_threshold: {p.nc10_threshold}")
    print(f"  uni6_threshold: {p.uni6_threshold}")
    print(f"  uni10_threshold: {p.uni10_threshold}")
    print(f"  auto_surface_nc6: {p.auto_surface_nc6}")
    print(f"  auto_surface_nc10: {p.auto_surface_nc10}")
    print(f"  Accuracy: {best_stride['accuracy']:.4f}\n")

    # Save results
    with open('quick_optimization_results.txt', 'w') as f:
        f.write("QUICK OPTIMIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"BEST FOR DSSP ({best_dssp['desc']}):\n")
        p = best_dssp['params']
        f.write(f"  nc6_threshold = {p.nc6_threshold}\n")
        f.write(f"  nc10_threshold = {p.nc10_threshold}\n")
        f.write(f"  uni6_threshold = {p.uni6_threshold}\n")
        f.write(f"  uni10_threshold = {p.uni10_threshold}\n")
        f.write(f"  auto_surface_nc6 = {p.auto_surface_nc6}\n")
        f.write(f"  auto_surface_nc10 = {p.auto_surface_nc10}\n")
        f.write(f"  Accuracy: {best_dssp['accuracy']:.4f}\n\n")

        f.write(f"BEST FOR STRIDE ({best_stride['desc']}):\n")
        p = best_stride['params']
        f.write(f"  nc6_threshold = {p.nc6_threshold}\n")
        f.write(f"  nc10_threshold = {p.nc10_threshold}\n")
        f.write(f"  uni6_threshold = {p.uni6_threshold}\n")
        f.write(f"  uni10_threshold = {p.uni10_threshold}\n")
        f.write(f"  auto_surface_nc6 = {p.auto_surface_nc6}\n")
        f.write(f"  auto_surface_nc10 = {p.auto_surface_nc10}\n")
        f.write(f"  Accuracy: {best_stride['accuracy']:.4f}\n")

    print("✓ Results saved to quick_optimization_results.txt\n")


if __name__ == "__main__":
    main()


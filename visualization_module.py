#!/usr/bin/env python3
"""
Visualization Module for Protein Burial Analysis
Generates comprehensive plots and graphs for results analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_accuracy_distribution(results_data: List[Dict], output_dir: Path, reference: str = 'dssp'):
    """Plot distribution of accuracies across proteins."""
    accuracies = [r[f'{reference}_accuracy'] for r in results_data if r.get(f'{reference}_accuracy') is not None]
    protein_ids = [r['protein_id'] for r in results_data if r.get(f'{reference}_accuracy') is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.hist(accuracies, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    ax1.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(accuracies):.3f}')
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_ylabel('Number of Proteins', fontsize=12)
    ax1.set_title(f'Accuracy Distribution (vs {reference.upper()})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(accuracies, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy Box Plot (vs {reference.upper()})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'accuracy_distribution_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: accuracy_distribution_{reference}.png")


def plot_per_protein_accuracy(results_data: List[Dict], output_dir: Path, reference: str = 'dssp'):
    """Plot per-protein accuracy as bar chart."""
    data = [(r['protein_id'], r[f'{reference}_accuracy'])
            for r in results_data if r.get(f'{reference}_accuracy') is not None]

    if not data:
        return

    # Sort by accuracy
    data.sort(key=lambda x: x[1])
    protein_ids, accuracies = zip(*data)

    # Color code by performance
    colors = ['red' if acc < 0.7 else 'orange' if acc < 0.8 else 'green' for acc in accuracies]

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.4), 8))
    bars = ax.bar(range(len(protein_ids)), accuracies, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(range(len(protein_ids)))
    ax.set_xticklabels(protein_ids, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Protein ID', fontsize=12)
    ax.set_title(f'Per-Protein Accuracy (vs {reference.upper()})', fontsize=14, fontweight='bold')
    ax.axhline(np.mean(accuracies), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'per_protein_accuracy_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: per_protein_accuracy_{reference}.png")


def plot_confusion_matrix_heatmap(cm: np.ndarray, output_file: Path, title: str):
    """Plot confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Interior (0)', 'Exterior (1)'],
                yticklabels=['Interior (0)', 'Exterior (1)'],
                cbar_kws={'label': 'Count'},
                ax=ax)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file.name}")


def plot_aggregate_confusion_matrices(results_data: List[Dict], output_dir: Path):
    """Plot aggregate confusion matrices for DSSP and STRIDE."""

    # Aggregate DSSP
    dssp_cm = np.zeros((2, 2), dtype=int)
    for r in results_data:
        if r.get('dssp_confusion_matrix') is not None:
            dssp_cm += np.array(r['dssp_confusion_matrix'])

    if dssp_cm.sum() > 0:
        plot_confusion_matrix_heatmap(
            dssp_cm,
            output_dir / 'aggregate_confusion_matrix_dssp.png',
            'Aggregate Confusion Matrix (vs DSSP)'
        )

    # Aggregate STRIDE
    stride_cm = np.zeros((2, 2), dtype=int)
    for r in results_data:
        if r.get('stride_confusion_matrix') is not None:
            stride_cm += np.array(r['stride_confusion_matrix'])

    if stride_cm.sum() > 0:
        plot_confusion_matrix_heatmap(
            stride_cm,
            output_dir / 'aggregate_confusion_matrix_stride.png',
            'Aggregate Confusion Matrix (vs STRIDE)'
        )


def plot_f1_scores_comparison(results_data: List[Dict], output_dir: Path):
    """Plot F1 scores for DSSP and STRIDE comparison."""
    protein_ids = []
    dssp_f1 = []
    stride_f1 = []

    for r in results_data:
        if r.get('dssp_f1') is not None or r.get('stride_f1') is not None:
            protein_ids.append(r['protein_id'])
            dssp_f1.append(r.get('dssp_f1', 0))
            stride_f1.append(r.get('stride_f1', 0))

    if not protein_ids:
        return

    x = np.arange(len(protein_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(protein_ids) * 0.5), 8))

    bars1 = ax.bar(x - width/2, dssp_f1, width, label='vs DSSP', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, stride_f1, width, label='vs STRIDE', alpha=0.8, color='coral')

    ax.set_xlabel('Protein ID', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score Comparison (DSSP vs STRIDE)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(protein_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'f1_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: f1_scores_comparison.png")


def plot_outlier_analysis(results_data: List[Dict], output_dir: Path, reference: str = 'dssp'):
    """Identify and visualize outliers."""
    accuracies = [r[f'{reference}_accuracy'] for r in results_data if r.get(f'{reference}_accuracy') is not None]
    protein_ids = [r['protein_id'] for r in results_data if r.get(f'{reference}_accuracy') is not None]
    n_residues = [r['n_residues'] for r in results_data if r.get(f'{reference}_accuracy') is not None]

    if not accuracies:
        return

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    # Identify outliers (> 1 std away from mean)
    outlier_low = mean_acc - std_acc
    outlier_high = mean_acc + std_acc

    colors = ['red' if acc < outlier_low else 'green' if acc > outlier_high else 'gray'
              for acc in accuracies]

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(n_residues, accuracies, c=colors, s=100, alpha=0.6, edgecolors='black')

    # Add protein labels for outliers
    for i, (x, y, pid) in enumerate(zip(n_residues, accuracies, protein_ids)):
        if colors[i] != 'gray':
            ax.annotate(pid, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.axhline(mean_acc, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    ax.axhline(outlier_low, color='red', linestyle=':', linewidth=1.5, label=f'Low Threshold: {outlier_low:.3f}')
    ax.axhline(outlier_high, color='green', linestyle=':', linewidth=1.5, label=f'High Threshold: {outlier_high:.3f}')

    ax.set_xlabel('Number of Residues', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Outlier Analysis: Accuracy vs Protein Size (vs {reference.upper()})',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'outlier_analysis_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: outlier_analysis_{reference}.png")


def plot_cross_validation_results(cv_results: Dict, output_dir: Path, reference: str = 'dssp'):
    """Plot cross-validation results."""
    fold_accuracies = cv_results['fold_accuracies']
    fold_f1_scores = cv_results['fold_f1_scores']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy per fold
    folds = range(1, len(fold_accuracies) + 1)
    ax1.plot(folds, fold_accuracies, 'o-', linewidth=2, markersize=10, color='steelblue', label='Fold Accuracy')
    ax1.axhline(cv_results['mean_accuracy'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {cv_results['mean_accuracy']:.4f}")
    ax1.fill_between(folds,
                     cv_results['mean_accuracy'] - cv_results['std_accuracy'],
                     cv_results['mean_accuracy'] + cv_results['std_accuracy'],
                     alpha=0.2, color='red')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Cross-Validation Accuracy per Fold (vs {reference.upper()})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # F1-Score per fold
    ax2.plot(folds, fold_f1_scores, 'o-', linewidth=2, markersize=10, color='coral', label='Fold F1-Score')
    ax2.axhline(cv_results['mean_f1'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {cv_results['mean_f1']:.4f}")
    ax2.fill_between(folds,
                     cv_results['mean_f1'] - cv_results['std_f1'],
                     cv_results['mean_f1'] + cv_results['std_f1'],
                     alpha=0.2, color='red')
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title(f'Cross-Validation F1-Score per Fold (vs {reference.upper()})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / f'cross_validation_results_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: cross_validation_results_{reference}.png")


def plot_optuna_optimization(optuna_df: pd.DataFrame, output_dir: Path):
    """Plot Optuna optimization history."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Optimization history
    ax = axes[0, 0]
    ax.plot(optuna_df['number'], optuna_df['value'], 'o-', alpha=0.6, color='steelblue')
    ax.plot(optuna_df['number'], optuna_df['value'].cummax(), 'r-', linewidth=2, label='Best So Far')
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Optimization History', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter importance (correlations)
    ax = axes[0, 1]
    param_cols = [col for col in optuna_df.columns if col.startswith('params_')]
    if param_cols:
        correlations = []
        param_names = []
        for col in param_cols:
            if optuna_df[col].notna().sum() > 0:
                corr = optuna_df[col].corr(optuna_df['value'])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                    param_names.append(col.replace('params_', ''))

        if correlations:
            ax.barh(param_names, correlations, color='coral', alpha=0.7)
            ax.set_xlabel('Absolute Correlation with Accuracy', fontsize=12)
            ax.set_title('Parameter Importance', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

    # Distribution of best trials
    ax = axes[1, 0]
    top_trials = optuna_df.nlargest(20, 'value')
    ax.hist(top_trials['value'], bins=15, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Top 20 Trials', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Parameter values over time
    ax = axes[1, 1]
    if 'params_nc6_threshold' in optuna_df.columns:
        ax.scatter(optuna_df['number'], optuna_df['params_nc6_threshold'],
                  c=optuna_df['value'], cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('nc6_threshold', fontsize=12)
        ax.set_title('Parameter Evolution: nc6_threshold', fontsize=12, fontweight='bold')
        plt.colorbar(ax.collections[0], ax=ax, label='Accuracy')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'optuna_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: optuna_optimization_analysis.png")


def generate_all_visualizations(results_dir: Path):
    """Generate all visualizations from saved results."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Load baseline results
    baseline_file = results_dir / "baseline_summary_report.txt"

    # Try to load results from CSV files
    result_files = list(results_dir.glob("*_detailed_results.csv"))

    if result_files:
        results_data = []
        # Import metrics at top of block
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

        for csv_file in result_files:
            df = pd.read_csv(csv_file)
            protein_id = csv_file.stem.replace('_detailed_results', '')

            # Calculate metrics
            result_dict = {
                'protein_id': protein_id,
                'n_residues': len(df)
            }

            # DSSP metrics
            if 'dssp_class' in df.columns and df['dssp_class'].notna().sum() > 0:
                dssp_mask = df['dssp_class'].notna()
                y_true = df.loc[dssp_mask, 'dssp_class'].values
                y_pred = df.loc[dssp_mask, 'ncps_class'].values

                result_dict['dssp_accuracy'] = accuracy_score(y_true, y_pred)
                result_dict['dssp_confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
                result_dict['dssp_f1'] = f1_score(y_true, y_pred, zero_division=0)

            # STRIDE metrics
            if 'stride_class' in df.columns and df['stride_class'].notna().sum() > 0:
                stride_mask = df['stride_class'].notna()
                y_true = df.loc[stride_mask, 'stride_class'].values
                y_pred = df.loc[stride_mask, 'ncps_class'].values

                result_dict['stride_accuracy'] = accuracy_score(y_true, y_pred)
                result_dict['stride_confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
                result_dict['stride_f1'] = f1_score(y_true, y_pred, zero_division=0)

            results_data.append(result_dict)

        # Generate plots
        plot_accuracy_distribution(results_data, viz_dir, 'dssp')
        plot_accuracy_distribution(results_data, viz_dir, 'stride')
        plot_per_protein_accuracy(results_data, viz_dir, 'dssp')
        plot_per_protein_accuracy(results_data, viz_dir, 'stride')
        plot_aggregate_confusion_matrices(results_data, viz_dir)
        plot_f1_scores_comparison(results_data, viz_dir)
        plot_outlier_analysis(results_data, viz_dir, 'dssp')
        plot_outlier_analysis(results_data, viz_dir, 'stride')

        # Check for Optuna results
        optuna_file = results_dir / "optuna_optimization_trials.csv"
        if optuna_file.exists():
            optuna_df = pd.read_csv(optuna_file)
            plot_optuna_optimization(optuna_df, viz_dir)

    print(f"\nAll visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    # Generate visualizations from results
    results_dir = Path("results/comprehensive_analysis")
    if results_dir.exists():
        generate_all_visualizations(results_dir)
    else:
        print("No results directory found. Run comprehensive_burial_analysis.py first.")
#!/usr/bin/env python3
"""
Visualization Module for Protein Burial Analysis
Generates comprehensive plots and graphs for results analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_accuracy_distribution(results_data: List[Dict], output_dir: Path, reference: str = 'dssp'):
    """Plot distribution of accuracies across proteins."""
    accuracies = [r[f'{reference}_accuracy'] for r in results_data if r.get(f'{reference}_accuracy') is not None]
    protein_ids = [r['protein_id'] for r in results_data if r.get(f'{reference}_accuracy') is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.hist(accuracies, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    ax1.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(accuracies):.3f}')
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_ylabel('Number of Proteins', fontsize=12)
    ax1.set_title(f'Accuracy Distribution (vs {reference.upper()})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(accuracies, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy Box Plot (vs {reference.upper()})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'accuracy_distribution_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: accuracy_distribution_{reference}.png")


def plot_per_protein_accuracy(results_data: List[Dict], output_dir: Path, reference: str = 'dssp'):
    """Plot per-protein accuracy as bar chart."""
    data = [(r['protein_id'], r[f'{reference}_accuracy'])
            for r in results_data if r.get(f'{reference}_accuracy') is not None]

    if not data:
        return

    # Sort by accuracy
    data.sort(key=lambda x: x[1])
    protein_ids, accuracies = zip(*data)

    # Color code by performance
    colors = ['red' if acc < 0.7 else 'orange' if acc < 0.8 else 'green' for acc in accuracies]

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.4), 8))
    bars = ax.bar(range(len(protein_ids)), accuracies, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(range(len(protein_ids)))
    ax.set_xticklabels(protein_ids, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Protein ID', fontsize=12)
    ax.set_title(f'Per-Protein Accuracy (vs {reference.upper()})', fontsize=14, fontweight='bold')
    ax.axhline(np.mean(accuracies), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'per_protein_accuracy_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: per_protein_accuracy_{reference}.png")


def plot_confusion_matrix_heatmap(cm: np.ndarray, output_file: Path, title: str):
    """Plot confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Interior (0)', 'Exterior (1)'],
                yticklabels=['Interior (0)', 'Exterior (1)'],
                cbar_kws={'label': 'Count'},
                ax=ax)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file.name}")


def plot_aggregate_confusion_matrices(results_data: List[Dict], output_dir: Path):
    """Plot aggregate confusion matrices for DSSP and STRIDE."""

    # Aggregate DSSP
    dssp_cm = np.zeros((2, 2), dtype=int)
    for r in results_data:
        if r.get('dssp_confusion_matrix') is not None:
            dssp_cm += np.array(r['dssp_confusion_matrix'])

    if dssp_cm.sum() > 0:
        plot_confusion_matrix_heatmap(
            dssp_cm,
            output_dir / 'aggregate_confusion_matrix_dssp.png',
            'Aggregate Confusion Matrix (vs DSSP)'
        )

    # Aggregate STRIDE
    stride_cm = np.zeros((2, 2), dtype=int)
    for r in results_data:
        if r.get('stride_confusion_matrix') is not None:
            stride_cm += np.array(r['stride_confusion_matrix'])

    if stride_cm.sum() > 0:
        plot_confusion_matrix_heatmap(
            stride_cm,
            output_dir / 'aggregate_confusion_matrix_stride.png',
            'Aggregate Confusion Matrix (vs STRIDE)'
        )


def plot_f1_scores_comparison(results_data: List[Dict], output_dir: Path):
    """Plot F1 scores for DSSP and STRIDE comparison."""
    protein_ids = []
    dssp_f1 = []
    stride_f1 = []

    for r in results_data:
        if r.get('dssp_f1') is not None or r.get('stride_f1') is not None:
            protein_ids.append(r['protein_id'])
            dssp_f1.append(r.get('dssp_f1', 0))
            stride_f1.append(r.get('stride_f1', 0))

    if not protein_ids:
        return

    x = np.arange(len(protein_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(protein_ids) * 0.5), 8))

    bars1 = ax.bar(x - width/2, dssp_f1, width, label='vs DSSP', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, stride_f1, width, label='vs STRIDE', alpha=0.8, color='coral')

    ax.set_xlabel('Protein ID', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score Comparison (DSSP vs STRIDE)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(protein_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'f1_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: f1_scores_comparison.png")


def plot_outlier_analysis(results_data: List[Dict], output_dir: Path, reference: str = 'dssp'):
    """Identify and visualize outliers."""
    accuracies = [r[f'{reference}_accuracy'] for r in results_data if r.get(f'{reference}_accuracy') is not None]
    protein_ids = [r['protein_id'] for r in results_data if r.get(f'{reference}_accuracy') is not None]
    n_residues = [r['n_residues'] for r in results_data if r.get(f'{reference}_accuracy') is not None]

    if not accuracies:
        return

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    # Identify outliers (> 1 std away from mean)
    outlier_low = mean_acc - std_acc
    outlier_high = mean_acc + std_acc

    colors = ['red' if acc < outlier_low else 'green' if acc > outlier_high else 'gray'
              for acc in accuracies]

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(n_residues, accuracies, c=colors, s=100, alpha=0.6, edgecolors='black')

    # Add protein labels for outliers
    for i, (x, y, pid) in enumerate(zip(n_residues, accuracies, protein_ids)):
        if colors[i] != 'gray':
            ax.annotate(pid, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.axhline(mean_acc, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    ax.axhline(outlier_low, color='red', linestyle=':', linewidth=1.5, label=f'Low Threshold: {outlier_low:.3f}')
    ax.axhline(outlier_high, color='green', linestyle=':', linewidth=1.5, label=f'High Threshold: {outlier_high:.3f}')

    ax.set_xlabel('Number of Residues', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Outlier Analysis: Accuracy vs Protein Size (vs {reference.upper()})',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'outlier_analysis_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: outlier_analysis_{reference}.png")


def plot_cross_validation_results(cv_results: Dict, output_dir: Path, reference: str = 'dssp'):
    """Plot cross-validation results."""
    fold_accuracies = cv_results['fold_accuracies']
    fold_f1_scores = cv_results['fold_f1_scores']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy per fold
    folds = range(1, len(fold_accuracies) + 1)
    ax1.plot(folds, fold_accuracies, 'o-', linewidth=2, markersize=10, color='steelblue', label='Fold Accuracy')
    ax1.axhline(cv_results['mean_accuracy'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {cv_results['mean_accuracy']:.4f}")
    ax1.fill_between(folds,
                     cv_results['mean_accuracy'] - cv_results['std_accuracy'],
                     cv_results['mean_accuracy'] + cv_results['std_accuracy'],
                     alpha=0.2, color='red')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Cross-Validation Accuracy per Fold (vs {reference.upper()})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # F1-Score per fold
    ax2.plot(folds, fold_f1_scores, 'o-', linewidth=2, markersize=10, color='coral', label='Fold F1-Score')
    ax2.axhline(cv_results['mean_f1'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {cv_results['mean_f1']:.4f}")
    ax2.fill_between(folds,
                     cv_results['mean_f1'] - cv_results['std_f1'],
                     cv_results['mean_f1'] + cv_results['std_f1'],
                     alpha=0.2, color='red')
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title(f'Cross-Validation F1-Score per Fold (vs {reference.upper()})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / f'cross_validation_results_{reference}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: cross_validation_results_{reference}.png")


def plot_optuna_optimization(optuna_df: pd.DataFrame, output_dir: Path):
    """Plot Optuna optimization history."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Optimization history
    ax = axes[0, 0]
    ax.plot(optuna_df['number'], optuna_df['value'], 'o-', alpha=0.6, color='steelblue')
    ax.plot(optuna_df['number'], optuna_df['value'].cummax(), 'r-', linewidth=2, label='Best So Far')
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Optimization History', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter importance (correlations)
    ax = axes[0, 1]
    param_cols = [col for col in optuna_df.columns if col.startswith('params_')]
    if param_cols:
        correlations = []
        param_names = []
        for col in param_cols:
            if optuna_df[col].notna().sum() > 0:
                corr = optuna_df[col].corr(optuna_df['value'])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                    param_names.append(col.replace('params_', ''))

        if correlations:
            ax.barh(param_names, correlations, color='coral', alpha=0.7)
            ax.set_xlabel('Absolute Correlation with Accuracy', fontsize=12)
            ax.set_title('Parameter Importance', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

    # Distribution of best trials
    ax = axes[1, 0]
    top_trials = optuna_df.nlargest(20, 'value')
    ax.hist(top_trials['value'], bins=15, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Top 20 Trials', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Parameter values over time
    ax = axes[1, 1]
    if 'params_nc6_threshold' in optuna_df.columns:
        ax.scatter(optuna_df['number'], optuna_df['params_nc6_threshold'],
                  c=optuna_df['value'], cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('nc6_threshold', fontsize=12)
        ax.set_title('Parameter Evolution: nc6_threshold', fontsize=12, fontweight='bold')
        plt.colorbar(ax.collections[0], ax=ax, label='Accuracy')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'optuna_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: optuna_optimization_analysis.png")


def generate_all_visualizations(results_dir: Path):
    """Generate all visualizations from saved results."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Load baseline results
    baseline_file = results_dir / "baseline_summary_report.txt"

    # Try to load results from CSV files
    result_files = list(results_dir.glob("*_detailed_results.csv"))

    if result_files:
        results_data = []
        for csv_file in result_files:
            df = pd.read_csv(csv_file)
            protein_id = csv_file.stem.replace('_detailed_results', '')

            # Calculate metrics
            result_dict = {
                'protein_id': protein_id,
                'n_residues': len(df)
            }

            # DSSP metrics
            if 'dssp_class' in df.columns and df['dssp_class'].notna().sum() > 0:
                dssp_mask = df['dssp_class'].notna()
                y_true = df.loc[dssp_mask, 'dssp_class'].values
                y_pred = df.loc[dssp_mask, 'ncps_class'].values

                from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
                result_dict['dssp_accuracy'] = accuracy_score(y_true, y_pred)
                result_dict['dssp_confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
                result_dict['dssp_f1'] = f1_score(y_true, y_pred, zero_division=0)

            # STRIDE metrics
            if 'stride_class' in df.columns and df['stride_class'].notna().sum() > 0:
                stride_mask = df['stride_class'].notna()
                y_true = df.loc[stride_mask, 'stride_class'].values
                y_pred = df.loc[stride_mask, 'ncps_class'].values

                result_dict['stride_accuracy'] = accuracy_score(y_true, y_pred)
                result_dict['stride_confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
                result_dict['stride_f1'] = f1_score(y_true, y_pred, zero_division=0)

            results_data.append(result_dict)

        # Generate plots
        plot_accuracy_distribution(results_data, viz_dir, 'dssp')
        plot_accuracy_distribution(results_data, viz_dir, 'stride')
        plot_per_protein_accuracy(results_data, viz_dir, 'dssp')
        plot_per_protein_accuracy(results_data, viz_dir, 'stride')
        plot_aggregate_confusion_matrices(results_data, viz_dir)
        plot_f1_scores_comparison(results_data, viz_dir)
        plot_outlier_analysis(results_data, viz_dir, 'dssp')
        plot_outlier_analysis(results_data, viz_dir, 'stride')

        # Check for Optuna results
        optuna_file = results_dir / "optuna_optimization_trials.csv"
        if optuna_file.exists():
            optuna_df = pd.read_csv(optuna_file)
            plot_optuna_optimization(optuna_df, viz_dir)

    print(f"\nAll visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    # Generate visualizations from results
    results_dir = Path("results/comprehensive_analysis")
    if results_dir.exists():
        generate_all_visualizations(results_dir)
    else:
        print("No results directory found. Run comprehensive_burial_analysis.py first.")


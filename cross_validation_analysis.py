#!/usr/bin/env python3
"""
Cross-Validation Analysis for Decision Tree Classifier

Performs k-fold cross-validation to evaluate model performance
on unseen data and avoid overfitting.

This validates the decision tree models built in decision_tree_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_normalized_data():
    """
    Load the pre-normalized combined dataset.
    """
    data_file = Path('results/decision_tree/combined_normalized.csv')

    if not data_file.exists():
        print("❌ Error: combined_normalized.csv not found!")
        print("   Please run decision_tree_analysis.py first.")
        return None

    df = pd.read_csv(data_file)
    print(f"✅ Loaded normalized data: {len(df)} residues")
    return df

def cross_validate_model(X, y, feature_names, model_name, n_folds=5):
    """
    Perform k-fold cross-validation on the decision tree model.
    """
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION: {model_name}")
    print("="*80)

    # Initialize model
    clf = DecisionTreeClassifier(max_depth=6, random_state=42,
                                  min_samples_split=20, min_samples_leaf=10)

    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    # K-Fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\nPerforming {n_folds}-fold cross-validation...")
    print(f"Dataset: {len(X)} residues")
    print(f"Features: {', '.join(feature_names)}")

    # Perform cross-validation
    cv_results = cross_validate(clf, X, y, cv=kfold, scoring=scoring,
                                 return_train_score=True)

    # Print results
    print("\n" + "-"*80)
    print("CROSS-VALIDATION RESULTS")
    print("-"*80)

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results_summary = {}

    for metric in metrics:
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']

        results_summary[metric] = {
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std(),
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std()
        }

        print(f"\n{metric.upper()}:")
        print(f"  Training:   {train_scores.mean():.3%} ± {train_scores.std():.3%}")
        print(f"  Validation: {test_scores.mean():.3%} ± {test_scores.std():.3%}")
        print(f"  Per fold (validation): {[f'{s:.3%}' for s in test_scores]}")

    # Check for overfitting
    train_acc = results_summary['accuracy']['train_mean']
    test_acc = results_summary['accuracy']['test_mean']
    gap = train_acc - test_acc

    print("\n" + "-"*80)
    print("OVERFITTING ANALYSIS")
    print("-"*80)
    print(f"Training accuracy:   {train_acc:.3%}")
    print(f"Validation accuracy: {test_acc:.3%}")
    print(f"Gap (overfitting):   {gap:.3%}")

    if gap < 0.05:
        print("✅ Good generalization (gap < 5%)")
    elif gap < 0.10:
        print("⚠️  Slight overfitting (gap 5-10%)")
    else:
        print("❌ Significant overfitting (gap > 10%)")

    return results_summary, cv_results

def compare_all_models(df):
    """
    Compare cross-validation results for all three models.
    """
    print("\n" + "="*80)
    print("COMPARING ALL MODELS WITH CROSS-VALIDATION")
    print("="*80)

    results_all_models = {}

    # Model 1: All features (6Å + 10Å)
    feature_cols_all = ['stride_asa_norm', 'ncps_sphere_6_norm', 'ncps_sphere_10_norm',
                        'ncps_sphere_6_uni_norm', 'ncps_sphere_10_uni_norm']
    df_clean = df[feature_cols_all + ['dssp_class']].dropna()
    X_all = df_clean[feature_cols_all]
    y_all = df_clean['dssp_class']

    results_all, cv_all = cross_validate_model(X_all, y_all, feature_cols_all,
                                                "All Features (6Å + 10Å)", n_folds=5)
    results_all_models['all'] = results_all

    # Model 2: 6Å only
    feature_cols_6A = ['stride_asa_norm', 'ncps_sphere_6_norm', 'ncps_sphere_6_uni_norm']
    df_clean = df[feature_cols_6A + ['dssp_class']].dropna()
    X_6A = df_clean[feature_cols_6A]
    y_6A = df_clean['dssp_class']

    results_6A, cv_6A = cross_validate_model(X_6A, y_6A, feature_cols_6A,
                                              "6Å Features Only", n_folds=5)
    results_all_models['6A'] = results_6A

    # Model 3: 10Å only
    feature_cols_10A = ['stride_asa_norm', 'ncps_sphere_10_norm', 'ncps_sphere_10_uni_norm']
    df_clean = df[feature_cols_10A + ['dssp_class']].dropna()
    X_10A = df_clean[feature_cols_10A]
    y_10A = df_clean['dssp_class']

    results_10A, cv_10A = cross_validate_model(X_10A, y_10A, feature_cols_10A,
                                                "10Å Features Only", n_folds=5)
    results_all_models['10A'] = results_10A

    # Summary comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON (VALIDATION ACCURACY)")
    print("="*80)

    acc_all = results_all_models['all']['accuracy']['test_mean']
    acc_6A = results_all_models['6A']['accuracy']['test_mean']
    acc_10A = results_all_models['10A']['accuracy']['test_mean']

    print(f"\nValidation Accuracy (5-fold CV):")
    print(f"  All features (6Å + 10Å): {acc_all:.3%} ± {results_all_models['all']['accuracy']['test_std']:.3%}")
    print(f"  6Å features only:        {acc_6A:.3%} ± {results_all_models['6A']['accuracy']['test_std']:.3%}")
    print(f"  10Å features only:       {acc_10A:.3%} ± {results_all_models['10A']['accuracy']['test_std']:.3%}")

    # Determine best model
    best_acc = max(acc_all, acc_6A, acc_10A)
    if best_acc == acc_all:
        best_model = "All features (6Å + 10Å)"
    elif best_acc == acc_6A:
        best_model = "6Å features only"
    else:
        best_model = "10Å features only"

    print(f"\n✅ Best model (by CV accuracy): {best_model} ({best_acc:.3%})")

    # Visualize comparison
    visualize_cv_comparison(results_all_models)

    return results_all_models

def visualize_cv_comparison(results_all_models):
    """
    Create visualization comparing cross-validation results.
    """
    output_dir = Path('results/decision_tree')

    # Prepare data for plotting
    models = ['All Features', '6Å Only', '10Å Only']
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    test_means = {
        'All Features': [results_all_models['all'][m]['test_mean'] for m in metrics],
        '6Å Only': [results_all_models['6A'][m]['test_mean'] for m in metrics],
        '10Å Only': [results_all_models['10A'][m]['test_mean'] for m in metrics]
    }

    test_stds = {
        'All Features': [results_all_models['all'][m]['test_std'] for m in metrics],
        '6Å Only': [results_all_models['6A'][m]['test_std'] for m in metrics],
        '10Å Only': [results_all_models['10A'][m]['test_std'] for m in metrics]
    }

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.25

    for i, (model, color) in enumerate(zip(models, ['#2ecc71', '#3498db', '#e74c3c'])):
        offset = (i - 1) * width
        ax.bar(x + offset, test_means[model], width,
               yerr=test_stds[model], label=model, color=color, alpha=0.8,
               capsize=5)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Results Comparison (5-fold CV)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_validation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_dir / 'cross_validation_comparison.png'}")
    plt.close()

def save_cv_report(results_all_models):
    """
    Save detailed cross-validation report to text file.
    """
    output_dir = Path('results/decision_tree')
    output_file = output_dir / 'cross_validation_report.txt'

    with open(output_file, 'w') as f:
        f.write("CROSS-VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write("Method: 5-fold cross-validation\n")
        f.write("Strategy: Normalize each protein separately, then concatenate\n\n")

        for model_name, display_name in [('all', 'All Features (6Å + 10Å)'),
                                         ('6A', '6Å Features Only'),
                                         ('10A', '10Å Features Only')]:
            results = results_all_models[model_name]

            f.write("-"*80 + "\n")
            f.write(f"{display_name}\n")
            f.write("-"*80 + "\n\n")

            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                train_mean = results[metric]['train_mean']
                train_std = results[metric]['train_std']
                test_mean = results[metric]['test_mean']
                test_std = results[metric]['test_std']

                f.write(f"{metric.upper()}:\n")
                f.write(f"  Training:   {train_mean:.3%} ± {train_std:.3%}\n")
                f.write(f"  Validation: {test_mean:.3%} ± {test_std:.3%}\n")
                f.write(f"  Gap:        {(train_mean - test_mean):.3%}\n\n")

        # Best model summary
        acc_all = results_all_models['all']['accuracy']['test_mean']
        acc_6A = results_all_models['6A']['accuracy']['test_mean']
        acc_10A = results_all_models['10A']['accuracy']['test_mean']

        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Validation Accuracy:\n")
        f.write(f"  All features: {acc_all:.3%}\n")
        f.write(f"  6Å only:      {acc_6A:.3%}\n")
        f.write(f"  10Å only:     {acc_10A:.3%}\n\n")

        best_acc = max(acc_all, acc_6A, acc_10A)
        if best_acc == acc_all:
            f.write(f"Best model: All features (6Å + 10Å)\n")
        elif best_acc == acc_6A:
            f.write(f"Best model: 6Å features only\n")
        else:
            f.write(f"Best model: 10Å features only\n")

    print(f"✅ Report saved: {output_file}")

def main():
    """
    Main cross-validation analysis.
    """
    print("="*80)
    print("CROSS-VALIDATION ANALYSIS FOR DECISION TREE MODELS")
    print("="*80)

    # Load normalized data
    df = load_normalized_data()
    if df is None:
        return

    # Run cross-validation for all models
    results_all_models = compare_all_models(df)

    # Save report
    save_cv_report(results_all_models)

    print("\n" + "="*80)
    print("✅ CROSS-VALIDATION COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print("  - results/decision_tree/cross_validation_comparison.png")
    print("  - results/decision_tree/cross_validation_report.txt")

if __name__ == "__main__":
    main()


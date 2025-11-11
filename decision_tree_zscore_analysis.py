#!/usr/bin/env python3
"""
Decision Tree Analysis for Protein Burial Classification

Strategy:
1. Load data for each protein separately
2. Normalize each protein independently (z-score per protein)
3. Concatenate normalized proteins together
4. Build decision tree classifier
5. Visualize tree and evaluate performance

Features used:
- stride_asa (STRIDE accessible surface area)
- ncps_sphere_6 (neighbor count at 6Å)
- ncps_sphere_10 (neighbor count at 10Å)
- ncps_sphere_6_uni (uniformity at 6Å)
- ncps_sphere_10_uni (uniformity at 10Å)

Target: dssp_class (0=interior, 1=exterior)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_and_normalize_protein(csv_file, protein_name):
    """
    Load protein data and normalize features using z-score.
    Normalization is done PER PROTEIN to account for size differences.
    """
    df = pd.read_csv(csv_file)

    # Filter to only residues with DSSP ground truth
    df = df[df['dssp_class'].notna()].copy()

    print(f"\n{protein_name}:")
    print(f"  Total residues: {len(df)}")

    # Features to normalize - ONLY neighbor-based features (DSSP/STRIDE agnostic)
    # NOTE: stride_asa and dssp_asa are NOT used for model training, only for validation
    features = ['ncps_sphere_6', 'ncps_sphere_10',
                'ncps_sphere_6_uni', 'ncps_sphere_10_uni']

    # Z-score normalization per protein
    for feat in features:
        if df[feat].notna().sum() > 0:
            mean = df[feat].mean()
            std = df[feat].std()
            if std > 0:
                df[f'{feat}_norm'] = (df[feat] - mean) / std
            else:
                df[f'{feat}_norm'] = 0.0
            print(f"  {feat}: mean={mean:.2f}, std={std:.2f}")
        else:
            df[f'{feat}_norm'] = 0.0

    # Add protein identifier
    df['protein'] = protein_name

    return df

def build_decision_tree_all_features(df_combined, max_depth=5):
    """
    Build decision tree using ALL features (6Å + 10Å).
    """
    print("\n" + "="*80)
    print("DECISION TREE - ALL FEATURES (6Å + 10Å)")
    print("="*80)

    # Prepare features (normalized) - ONLY neighbor-based features (DSSP/STRIDE agnostic)
    feature_cols = ['ncps_sphere_6_norm', 'ncps_sphere_10_norm',
                    'ncps_sphere_6_uni_norm', 'ncps_sphere_10_uni_norm']

    # Remove rows with NaN
    df_clean = df_combined[feature_cols + ['dssp_class']].dropna()

    X = df_clean[feature_cols]
    y = df_clean['dssp_class']

    print(f"\nTraining on {len(X)} residues")
    print(f"  Interior (0): {(y==0).sum()}")
    print(f"  Exterior (1): {(y==1).sum()}")

    # Build decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42,
                                  min_samples_split=20, min_samples_leaf=10)
    clf.fit(X, y)

    # Predictions
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print(f"\nTraining Accuracy: {accuracy:.3%}")
    print(f"\nFeature Importances:")
    for feat, imp in zip(feature_cols, clf.feature_importances_):
        print(f"  {feat:30s}: {imp:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Interior', 'Exterior']))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Int(0)  Ext(1)")
    print(f"   Actual Int(0)    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"   Actual Ext(1)    {cm[1,0]:4d}   {cm[1,1]:4d}")

    return clf, feature_cols, X, y

def build_decision_tree_6A_only(df_combined, max_depth=5):
    """
    Build decision tree using ONLY 6Å features.
    """
    print("\n" + "="*80)
    print("DECISION TREE - 6Å FEATURES ONLY")
    print("="*80)

    # Features: 6Å neighbors + 6Å uniformity (DSSP/STRIDE agnostic)
    feature_cols = ['ncps_sphere_6_norm', 'ncps_sphere_6_uni_norm']

    df_clean = df_combined[feature_cols + ['dssp_class']].dropna()

    X = df_clean[feature_cols]
    y = df_clean['dssp_class']

    print(f"\nTraining on {len(X)} residues (6Å features only)")

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42,
                                  min_samples_split=20, min_samples_leaf=10)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print(f"\nTraining Accuracy: {accuracy:.3%}")
    print(f"\nFeature Importances:")
    for feat, imp in zip(feature_cols, clf.feature_importances_):
        print(f"  {feat:30s}: {imp:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Interior', 'Exterior']))

    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Int(0)  Ext(1)")
    print(f"   Actual Int(0)    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"   Actual Ext(1)    {cm[1,0]:4d}   {cm[1,1]:4d}")

    return clf, feature_cols, X, y

def build_decision_tree_10A_only(df_combined, max_depth=5):
    """
    Build decision tree using ONLY 10Å features.
    """
    print("\n" + "="*80)
    print("DECISION TREE - 10Å FEATURES ONLY")
    print("="*80)

    # Features: 10Å neighbors + 10Å uniformity (DSSP/STRIDE agnostic)
    feature_cols = ['ncps_sphere_10_norm', 'ncps_sphere_10_uni_norm']

    df_clean = df_combined[feature_cols + ['dssp_class']].dropna()

    X = df_clean[feature_cols]
    y = df_clean['dssp_class']

    print(f"\nTraining on {len(X)} residues (10Å features only)")

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42,
                                  min_samples_split=20, min_samples_leaf=10)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print(f"\nTraining Accuracy: {accuracy:.3%}")
    print(f"\nFeature Importances:")
    for feat, imp in zip(feature_cols, clf.feature_importances_):
        print(f"  {feat:30s}: {imp:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Interior', 'Exterior']))

    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Int(0)  Ext(1)")
    print(f"   Actual Int(0)    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"   Actual Ext(1)    {cm[1,0]:4d}   {cm[1,1]:4d}")

    return clf, feature_cols, X, y

def visualize_decision_tree(clf, feature_names, output_file, title):
    """
    Visualize the decision tree and save as image.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names,
              class_names=['Interior', 'Exterior'],
              filled=True, rounded=True, fontsize=10)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Decision tree visualization saved: {output_file}")
    plt.close()

def main():
    """
    Main analysis pipeline.
    """
    print("="*80)
    print("DECISION TREE CLASSIFICATION ANALYSIS")
    print("Strategy: Normalize each protein separately, then concatenate")
    print("="*80)

    # Load and normalize each protein separately
    proteins = {
        '3PTE': 'results/3PTE_results.csv',
        '4d05': 'results/4d05_results.csv',
        '6wti': 'results/6wti_results.csv',
        '7upo': 'results/7upo_results.csv'
    }

    dfs = []
    for name, path in proteins.items():
        if Path(path).exists():
            df = load_and_normalize_protein(path, name)
            dfs.append(df)

    # Concatenate all normalized proteins
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\n{'='*80}")
    print(f"COMBINED DATASET: {len(df_combined)} total residues")
    print(f"  Interior (0): {(df_combined['dssp_class']==0).sum()}")
    print(f"  Exterior (1): {(df_combined['dssp_class']==1).sum()}")
    print(f"{'='*80}")

    # Save normalized combined data
    output_dir = Path('results/decision_tree')
    output_dir.mkdir(exist_ok=True)
    df_combined.to_csv(output_dir / 'combined_normalized.csv', index=False)
    print(f"\n✅ Saved normalized data: {output_dir / 'combined_normalized.csv'}")

    # Build decision trees with different feature sets
    max_depth = 6  # Adjustable parameter

    # 1. All features (6Å + 10Å)
    clf_all, feat_all, X_all, y_all = build_decision_tree_all_features(df_combined, max_depth)
    visualize_decision_tree(clf_all, feat_all,
                           output_dir / 'decision_tree_all_features.png',
                           'Decision Tree - All Features (6Å + 10Å)')

    # 2. 6Å only
    clf_6A, feat_6A, X_6A, y_6A = build_decision_tree_6A_only(df_combined, max_depth)
    visualize_decision_tree(clf_6A, feat_6A,
                           output_dir / 'decision_tree_6A_only.png',
                           'Decision Tree - 6Å Features Only')

    # 3. 10Å only
    clf_10A, feat_10A, X_10A, y_10A = build_decision_tree_10A_only(df_combined, max_depth)
    visualize_decision_tree(clf_10A, feat_10A,
                           output_dir / 'decision_tree_10A_only.png',
                           'Decision Tree - 10Å Features Only')

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    acc_all = accuracy_score(y_all, clf_all.predict(X_all))
    acc_6A = accuracy_score(y_6A, clf_6A.predict(X_6A))
    acc_10A = accuracy_score(y_10A, clf_10A.predict(X_10A))

    print(f"\nTraining Accuracy:")
    print(f"  All features (6Å + 10Å): {acc_all:.3%}")
    print(f"  6Å features only:        {acc_6A:.3%}")
    print(f"  10Å features only:       {acc_10A:.3%}")

    if acc_all >= acc_6A and acc_all >= acc_10A:
        print(f"\n✅ Best: All features (combining both radii)")
    elif acc_6A > acc_10A:
        print(f"\n✅ Best: 6Å features (local neighborhood more important)")
    else:
        print(f"\n✅ Best: 10Å features (broader context more important)")

    # Save summary
    with open(output_dir / 'decision_tree_summary.txt', 'w') as f:
        f.write("DECISION TREE CLASSIFICATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total residues: {len(df_combined)}\n")
        f.write(f"Interior (0): {(df_combined['dssp_class']==0).sum()}\n")
        f.write(f"Exterior (1): {(df_combined['dssp_class']==1).sum()}\n\n")
        f.write(f"Training Accuracy:\n")
        f.write(f"  All features: {acc_all:.3%}\n")
        f.write(f"  6Å only:      {acc_6A:.3%}\n")
        f.write(f"  10Å only:     {acc_10A:.3%}\n\n")
        f.write("Strategy: Each protein normalized separately (z-score),\n")
        f.write("          then concatenated together for training.\n")

    print(f"\n✅ Summary saved: {output_dir / 'decision_tree_summary.txt'}")
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files in: {output_dir}/")
    print("  - combined_normalized.csv (normalized data)")
    print("  - decision_tree_all_features.png")
    print("  - decision_tree_6A_only.png")
    print("  - decision_tree_10A_only.png")
    print("  - decision_tree_summary.txt")

if __name__ == "__main__":
    main()


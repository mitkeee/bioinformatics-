#!/usr/bin/env python3
"""
Create Enhanced Visualizations for Split Strategy Comparison

This script creates publication-quality visualizations showing:
1. Side-by-side tree comparisons with better colors
2. Accuracy comparison bar chart
3. Feature importance comparison
4. Detailed infographic explaining each strategy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def load_data():
    """Load the combined normalized dataset."""
    data_path = Path('results/decision_tree/combined_normalized.csv')
    df = pd.read_csv(data_path)
    return df


def prepare_data(df, feature_cols):
    """Prepare clean dataset."""
    df_clean = df[feature_cols + ['dssp_class']].dropna()
    X = df_clean[feature_cols]
    y = df_clean['dssp_class']
    return X, y


def train_binary_model(df):
    """Train binary split decision tree."""
    feature_cols = ['stride_asa_norm', 'ncps_sphere_6_norm', 'ncps_sphere_10_norm',
                    'ncps_sphere_6_uni_norm', 'ncps_sphere_10_uni_norm']
    X, y = prepare_data(df, feature_cols)

    clf = DecisionTreeClassifier(max_depth=4, random_state=42,
                                  min_samples_split=20, min_samples_leaf=10)
    clf.fit(X, y)
    accuracy = accuracy_score(y, clf.predict(X))

    return clf, feature_cols, X, y, accuracy


def train_categorical_model(df, n_bins=3):
    """Train categorical split decision tree."""
    feature_cols = ['stride_asa_norm', 'ncps_sphere_6_norm', 'ncps_sphere_10_norm',
                    'ncps_sphere_6_uni_norm', 'ncps_sphere_10_uni_norm']

    df_clean = df[feature_cols + ['dssp_class']].dropna().copy()

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df_binned = df_clean.copy()
    df_binned[feature_cols] = discretizer.fit_transform(df_clean[feature_cols])

    X = df_binned[feature_cols]
    y = df_binned['dssp_class']

    clf = DecisionTreeClassifier(max_depth=4, random_state=42,
                                  min_samples_split=20, min_samples_leaf=10)
    clf.fit(X, y)
    accuracy = accuracy_score(y, clf.predict(X))

    return clf, feature_cols, X, y, accuracy, discretizer


def apply_rule_based(df):
    """Apply rule-based classification."""
    feature_cols = ['stride_asa_norm', 'ncps_sphere_6_norm', 'ncps_sphere_10_norm',
                    'ncps_sphere_6_uni_norm', 'ncps_sphere_10_uni_norm']

    df_clean = df[feature_cols + ['dssp_class']].dropna()

    predictions = []
    for idx, row in df_clean.iterrows():
        sasa = row['stride_asa_norm']
        n_neighbors_10 = row['ncps_sphere_10_norm']
        uniformity_10 = row['ncps_sphere_10_uni_norm']

        if sasa < -0.5 and n_neighbors_10 > 0.3:
            pred = 0
        elif sasa > 0.5:
            pred = 1
        elif sasa < 0 and uniformity_10 < -0.3:
            pred = 0
        elif sasa <= 0:
            pred = 0
        else:
            pred = 1

        predictions.append(pred)

    predictions = np.array(predictions)
    y_true = df_clean['dssp_class'].values
    accuracy = accuracy_score(y_true, predictions)

    return predictions, y_true, accuracy


def create_beautiful_tree_comparison(clf_binary, clf_cat, feature_names, output_dir):
    """Create beautiful side-by-side tree comparison."""
    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 12], hspace=0.05, wspace=0.15)

    # Title for binary tree
    ax_title1 = fig.add_subplot(gs[0, 0])
    ax_title1.axis('off')
    ax_title1.text(0.5, 0.5, 'Binary Splits (Continuous Thresholds)',
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#4CAF50',
                            edgecolor='#2E7D32', linewidth=2, alpha=0.9),
                   color='white')

    # Title for categorical tree
    ax_title2 = fig.add_subplot(gs[0, 1])
    ax_title2.axis('off')
    ax_title2.text(0.5, 0.5, 'Categorical Splits (Discretized Bins)',
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#2196F3',
                            edgecolor='#1565C0', linewidth=2, alpha=0.9),
                   color='white')

    # Binary tree
    ax1 = fig.add_subplot(gs[1, 0])
    plot_tree(clf_binary, feature_names=feature_names,
              class_names=['Interior', 'Exterior'],
              filled=True, rounded=True, fontsize=10, ax=ax1,
              proportion=True, precision=2,
              impurity=False)
    ax1.set_title('Questions: "Is feature ≤ threshold?"\nOptimal for continuous data',
                  fontsize=13, pad=10, style='italic', color='#2E7D32')

    # Categorical tree
    ax2 = fig.add_subplot(gs[1, 1])
    plot_tree(clf_cat, feature_names=feature_names,
              class_names=['Interior', 'Exterior'],
              filled=True, rounded=True, fontsize=10, ax=ax2,
              proportion=True, precision=2,
              impurity=False)
    ax2.set_title('Questions: "Is feature ≤ bin_value?"\nDiscretized into Low/Medium/High',
                  fontsize=13, pad=10, style='italic', color='#1565C0')

    # Add subtle background
    fig.patch.set_facecolor('#f8f9fa')

    plt.tight_layout()
    output_path = output_dir / 'enhanced_tree_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"✅ Enhanced tree comparison saved: {output_path}")
    plt.close()


def create_accuracy_comparison(results, output_dir):
    """Create beautiful accuracy comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    strategies = list(results.keys())
    accuracies = [results[s] * 100 for s in strategies]

    # Color palette
    colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF9800']

    # Create bars
    bars = ax.barh(strategies, accuracies, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        label_x = width + 1
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{acc:.2f}%',
               ha='left', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor=colors[i], linewidth=2))

    # Styling
    ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Decision Tree Split Strategy Comparison\nAccuracy on Protein Burial Classification',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Add winner badge
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    bars[best_idx].set_alpha(1.0)

    # Add winner star
    ax.text(accuracies[best_idx] - 3, best_idx, '★',
           fontsize=30, color='gold', ha='center', va='center',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                    edgecolor='gold', linewidth=3))

    plt.tight_layout()
    output_path = output_dir / 'accuracy_comparison_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Accuracy comparison chart saved: {output_path}")
    plt.close()


def create_feature_importance_comparison(clf_binary, clf_cat, feature_names, output_dir):
    """Create feature importance comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

    # Binary model importances
    importances_binary = clf_binary.feature_importances_
    colors_binary = plt.cm.Greens(np.linspace(0.4, 0.8, len(feature_names)))

    bars1 = ax1.barh(feature_names, importances_binary, color=colors_binary,
                     edgecolor='#2E7D32', linewidth=1.5)
    ax1.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax1.set_title('Binary Splits\nFeature Importance', fontsize=14, fontweight='bold',
                  color='#2E7D32')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, imp in zip(bars1, importances_binary):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # Categorical model importances
    importances_cat = clf_cat.feature_importances_
    colors_cat = plt.cm.Blues(np.linspace(0.4, 0.8, len(feature_names)))

    bars2 = ax2.barh(feature_names, importances_cat, color=colors_cat,
                     edgecolor='#1565C0', linewidth=1.5)
    ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Categorical Splits\nFeature Importance', fontsize=14, fontweight='bold',
                  color='#1565C0')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, imp in zip(bars2, importances_cat):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')

    plt.suptitle('Feature Importance: Binary vs Categorical Splits',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'feature_importance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Feature importance comparison saved: {output_path}")
    plt.close()


def create_confusion_matrix_comparison(clf_binary, clf_cat, X_binary, y_binary,
                                       X_cat, y_cat, output_dir):
    """Create confusion matrix comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # Binary confusion matrix
    y_pred_binary = clf_binary.predict(X_binary)
    cm_binary = confusion_matrix(y_binary, y_pred_binary)

    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Greens',
               cbar=True, square=True, ax=ax1,
               xticklabels=['Interior', 'Exterior'],
               yticklabels=['Interior', 'Exterior'],
               annot_kws={'fontsize': 16, 'fontweight': 'bold'},
               linewidths=2, linecolor='white')
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax1.set_title('Binary Splits\nConfusion Matrix', fontsize=14, fontweight='bold',
                  color='#2E7D32', pad=15)

    # Categorical confusion matrix
    y_pred_cat = clf_cat.predict(X_cat)
    cm_cat = confusion_matrix(y_cat, y_pred_cat)

    sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues',
               cbar=True, square=True, ax=ax2,
               xticklabels=['Interior', 'Exterior'],
               yticklabels=['Interior', 'Exterior'],
               annot_kws={'fontsize': 16, 'fontweight': 'bold'},
               linewidths=2, linecolor='white')
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title('Categorical Splits\nConfusion Matrix', fontsize=14, fontweight='bold',
                  color='#1565C0', pad=15)

    plt.suptitle('Confusion Matrix Comparison',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = output_dir / 'confusion_matrix_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Confusion matrix comparison saved: {output_path}")
    plt.close()


def create_strategy_infographic(results, output_dir):
    """Create an infographic explaining the strategies."""
    fig = plt.figure(figsize=(16, 20), facecolor='#f0f4f8')

    # Title
    fig.text(0.5, 0.97, 'Decision Tree Split Strategies Explained',
            ha='center', fontsize=24, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='white',
                     edgecolor='#1976D2', linewidth=3))

    # Strategy 1: Binary Splits
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.axis('off')
    ax1.text(0.05, 0.9, '1. BINARY SPLITS (Standard Threshold-Based)',
            fontsize=18, fontweight='bold', color='#2E7D32',
            transform=ax1.transAxes)
    ax1.text(0.05, 0.7, '✓ How it works: Asks "Is feature ≤ threshold?" → True/False',
            fontsize=14, transform=ax1.transAxes)
    ax1.text(0.05, 0.55, '✓ Example: "Is SASA ≤ 0.234?" "Is neighbors ≤ 45.6?"',
            fontsize=14, transform=ax1.transAxes, style='italic', color='#555')
    ax1.text(0.05, 0.4, f'✓ Accuracy: {results["Binary Splits"]:.1%}',
            fontsize=14, fontweight='bold', color='#2E7D32', transform=ax1.transAxes)
    ax1.text(0.05, 0.25, '✓ Pros: Optimal for continuous data, finds best thresholds automatically',
            fontsize=13, color='#2E7D32', transform=ax1.transAxes)
    ax1.text(0.05, 0.1, '✗ Cons: Many nested conditions can be complex',
            fontsize=13, color='#666', transform=ax1.transAxes)
    ax1.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                     transform=ax1.transAxes,
                                     facecolor='#E8F5E9', edgecolor='#2E7D32',
                                     linewidth=3, alpha=0.3))

    # Strategy 2: Categorical Splits
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.axis('off')
    ax2.text(0.05, 0.9, '2. CATEGORICAL SPLITS (Discretized)',
            fontsize=18, fontweight='bold', color='#1565C0',
            transform=ax2.transAxes)
    ax2.text(0.05, 0.7, '✓ How it works: Pre-bin features into Low/Medium/High, then split',
            fontsize=14, transform=ax2.transAxes)
    ax2.text(0.05, 0.55, '✓ Example: "Is SASA = Low?" "Is neighbors = High?"',
            fontsize=14, transform=ax2.transAxes, style='italic', color='#555')
    ax2.text(0.05, 0.4, f'✓ Accuracy: {results["Categorical (3 bins)"]:.1%}',
            fontsize=14, fontweight='bold', color='#1565C0', transform=ax2.transAxes)
    ax2.text(0.05, 0.25, '✓ Pros: More interpretable categories (Low/Med/High)',
            fontsize=13, color='#1565C0', transform=ax2.transAxes)
    ax2.text(0.05, 0.1, '✗ Cons: Loses precision, performance decreases',
            fontsize=13, color='#666', transform=ax2.transAxes)
    ax2.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                     transform=ax2.transAxes,
                                     facecolor='#E3F2FD', edgecolor='#1565C0',
                                     linewidth=3, alpha=0.3))

    # Strategy 3: Categorical with 5 bins
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.axis('off')
    ax3.text(0.05, 0.9, '3. CATEGORICAL SPLITS (5 bins - More Granular)',
            fontsize=18, fontweight='bold', color='#6A1B9A',
            transform=ax3.transAxes)
    ax3.text(0.05, 0.7, '✓ How it works: More bins = Very Low/Low/Med/High/Very High',
            fontsize=14, transform=ax3.transAxes)
    ax3.text(0.05, 0.55, '✓ Example: "Is SASA = VeryLow?" "Is neighbors = VeryHigh?"',
            fontsize=14, transform=ax3.transAxes, style='italic', color='#555')
    ax3.text(0.05, 0.4, f'✓ Accuracy: {results["Categorical (5 bins)"]:.1%}',
            fontsize=14, fontweight='bold', color='#6A1B9A', transform=ax3.transAxes)
    ax3.text(0.05, 0.25, '✓ Pros: More granular than 3 bins, still interpretable',
            fontsize=13, color='#6A1B9A', transform=ax3.transAxes)
    ax3.text(0.05, 0.1, '✗ Cons: Still loses precision vs continuous',
            fontsize=13, color='#666', transform=ax3.transAxes)
    ax3.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                     transform=ax3.transAxes,
                                     facecolor='#F3E5F5', edgecolor='#6A1B9A',
                                     linewidth=3, alpha=0.3))

    # Strategy 4: Rule-based
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')
    ax4.text(0.05, 0.9, '4. RULE-BASED CLASSIFICATION (Manual Rules)',
            fontsize=18, fontweight='bold', color='#E65100',
            transform=ax4.transAxes)
    ax4.text(0.05, 0.7, '✓ How it works: Hand-crafted IF-THEN rules',
            fontsize=14, transform=ax4.transAxes)
    ax4.text(0.05, 0.55, '✓ Example: "IF SASA < -0.5 AND neighbors > 50 THEN Interior"',
            fontsize=14, transform=ax4.transAxes, style='italic', color='#555')
    ax4.text(0.05, 0.4, f'✓ Accuracy: {results["Rule-Based"]:.1%}',
            fontsize=14, fontweight='bold', color='#E65100', transform=ax4.transAxes)
    ax4.text(0.05, 0.25, '✓ Pros: Maximum interpretability, domain knowledge integration',
            fontsize=13, color='#E65100', transform=ax4.transAxes)
    ax4.text(0.05, 0.1, '✗ Cons: Manual tuning required, misses complex patterns',
            fontsize=13, color='#666', transform=ax4.transAxes)
    ax4.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                     transform=ax4.transAxes,
                                     facecolor='#FFF3E0', edgecolor='#E65100',
                                     linewidth=3, alpha=0.3))

    # Recommendation box
    best_strategy = max(results.items(), key=lambda x: x[1])
    fig.text(0.5, 0.03, f'🏆 WINNER: {best_strategy[0]} ({best_strategy[1]:.1%} accuracy)',
            ha='center', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='gold',
                     edgecolor='#FF6F00', linewidth=4, alpha=0.9))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_path = output_dir / 'strategy_infographic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f0f4f8')
    print(f"✅ Strategy infographic saved: {output_path}")
    plt.close()


def main():
    """Main visualization pipeline."""
    print("="*80)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*80)

    # Load data
    df = load_data()
    output_dir = Path('results/decision_tree/enhanced_viz')
    output_dir.mkdir(exist_ok=True)

    # Train models
    print("\n📊 Training models...")
    clf_binary, feat_binary, X_binary, y_binary, acc_binary = train_binary_model(df)
    clf_cat3, feat_cat3, X_cat3, y_cat3, acc_cat3, disc3 = train_categorical_model(df, n_bins=3)
    clf_cat5, feat_cat5, X_cat5, y_cat5, acc_cat5, disc5 = train_categorical_model(df, n_bins=5)
    pred_rules, y_rules, acc_rules = apply_rule_based(df)

    results = {
        'Binary Splits': acc_binary,
        'Categorical (3 bins)': acc_cat3,
        'Categorical (5 bins)': acc_cat5,
        'Rule-Based': acc_rules
    }

    # Create visualizations
    print("\n🎨 Creating enhanced visualizations...")

    print("\n1. Tree comparison...")
    create_beautiful_tree_comparison(clf_binary, clf_cat3, feat_binary, output_dir)

    print("2. Accuracy comparison...")
    create_accuracy_comparison(results, output_dir)

    print("3. Feature importance...")
    create_feature_importance_comparison(clf_binary, clf_cat3, feat_binary, output_dir)

    print("4. Confusion matrices...")
    create_confusion_matrix_comparison(clf_binary, clf_cat3, X_binary, y_binary,
                                       X_cat3, y_cat3, output_dir)

    print("5. Strategy infographic...")
    create_strategy_infographic(results, output_dir)

    print("\n" + "="*80)
    print("✅ ALL ENHANCED VISUALIZATIONS CREATED!")
    print("="*80)
    print(f"\nLocation: {output_dir}/")
    print("\nGenerated files:")
    print("  1. enhanced_tree_comparison.png - Side-by-side tree comparison")
    print("  2. accuracy_comparison_chart.png - Accuracy bar chart")
    print("  3. feature_importance_comparison.png - Feature importance")
    print("  4. confusion_matrix_comparison.png - Confusion matrices")
    print("  5. strategy_infographic.png - Complete strategy explanation")
    print("\n🎉 Ready for presentation!")


if __name__ == '__main__':
    main()


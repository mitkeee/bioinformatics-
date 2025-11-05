#!/usr/bin/env python3
"""Simple visualization script for decision tree split strategy comparison."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import sys

# Configuration
OUTPUT_DIR = Path('results/decision_tree/beautiful_viz')
FEATURE_COLS = ['stride_asa_norm', 'ncps_sphere_6_norm', 'ncps_sphere_10_norm',
                'ncps_sphere_6_uni_norm', 'ncps_sphere_10_uni_norm']

# Better feature names for display
FEATURE_NAMES = {
    'stride_asa_norm': 'Surface Area',
    'ncps_sphere_6_norm': 'Neighbors (6Å)',
    'ncps_sphere_10_norm': 'Neighbors (10Å)',
    'ncps_sphere_6_uni_norm': 'Uniformity (6Å)',
    'ncps_sphere_10_uni_norm': 'Uniformity (10Å)'
}

def load_data():
    """Load and prepare data with error handling."""
    data_file = Path('results/decision_tree/combined_normalized.csv')

    if not data_file.exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("   Please run decision_tree_analysis.py first.")
        sys.exit(1)

    df = pd.read_csv(data_file)
    df_clean = df[FEATURE_COLS + ['dssp_class']].dropna()

    if len(df_clean) == 0:
        print("❌ Error: No valid data found after cleaning.")
        sys.exit(1)

    X = df_clean[FEATURE_COLS]
    y = df_clean['dssp_class']

    # Print data statistics
    print(f"📊 Data loaded: {len(df_clean)} residues")
    print(f"   Interior: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"   Exterior: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

    return X, y

def train_models(X, y):
    """Train binary and categorical models - calculate real accuracies."""
    print("\n🤖 Training models...")

    # Binary model
    clf_binary = DecisionTreeClassifier(max_depth=4, random_state=42,
                                        min_samples_split=20, min_samples_leaf=10)
    clf_binary.fit(X, y)
    acc_binary = accuracy_score(y, clf_binary.predict(X))
    print(f"   Binary splits: {acc_binary:.1%}")

    # Categorical model (3 bins)
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    X_binned = discretizer.fit_transform(X)
    clf_cat = DecisionTreeClassifier(max_depth=4, random_state=42,
                                     min_samples_split=20, min_samples_leaf=10)
    clf_cat.fit(X_binned, y)
    acc_cat = accuracy_score(y, clf_cat.predict(X_binned))
    print(f"   Categorical (3 bins): {acc_cat:.1%}")

    # Simple rule-based approach
    acc_rules = calculate_rule_based_accuracy(X, y)
    print(f"   Rule-based: {acc_rules:.1%}")

    return clf_binary, acc_binary, acc_cat, acc_rules

def calculate_rule_based_accuracy(X, y):
    """Calculate accuracy of simple rule-based classification."""
    predictions = []
    for idx, row in X.iterrows():
        sasa = row['stride_asa_norm']
        n_neighbors = row['ncps_sphere_10_norm']

        # Simple rules
        if sasa < -0.5 and n_neighbors > 0.3:
            pred = 0  # Interior
        elif sasa > 0.5:
            pred = 1  # Exterior
        elif sasa <= 0:
            pred = 0  # Interior
        else:
            pred = 1  # Exterior
        predictions.append(pred)

    return accuracy_score(y, predictions)

def create_accuracy_chart(acc_binary, acc_cat, acc_rules, output_dir):
    """Create accuracy comparison chart with real values."""
    strategies = ['Binary Splits\n(Optimal)', 'Categorical\n(3 bins)', 'Rule-Based\n(Manual)']
    accuracies = [acc_binary * 100, acc_cat * 100, acc_rules * 100]
    colors = ['#4CAF50', '#2196F3', '#FF9800']

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    bars = ax.barh(strategies, accuracies, color=colors, edgecolor='black', linewidth=2)

    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%',
                va='center', fontweight='bold', fontsize=12)

    # Highlight winner
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    ax.text(accuracies[best_idx] - 2, best_idx, '★', fontsize=30, color='gold', ha='center')

    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Split Strategy Comparison\nProtein Burial Classification',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_tree_visualization(clf, acc, output_dir):
    """Create tree visualization with better feature names."""
    # Use better names for display
    display_names = [FEATURE_NAMES.get(col, col) for col in FEATURE_COLS]

    fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')

    plot_tree(clf, feature_names=display_names,
              class_names=['Interior', 'Exterior'],
              filled=True, rounded=True, fontsize=10, ax=ax,
              impurity=False, proportion=True)

    ax.set_title(f'Binary Split Decision Tree\nAccuracy: {acc:.1%} | Depth: {clf.get_depth()}',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance(clf, output_dir):
    """Create feature importance chart with better names and sorting."""
    importances = clf.feature_importances_
    display_names = [FEATURE_NAMES.get(col, col) for col in FEATURE_COLS]

    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_importances = importances[sorted_idx]
    sorted_names = [display_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(sorted_names)))

    bars = ax.barh(sorted_names, sorted_importances, color=colors,
                   edgecolor='darkgreen', linewidth=2)

    # Add value labels and percentage
    total = sorted_importances.sum()
    for bar, imp in zip(bars, sorted_importances):
        percentage = (imp / total) * 100
        ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f} ({percentage:.1f}%)',
                va='center', fontweight='bold', fontsize=11)

    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance (Sorted)\nWhich features drive decisions?',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution."""
    print("=" * 60)
    print("🎨 CREATING DECISION TREE VISUALIZATIONS")
    print("=" * 60)

    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load and train
    X, y = load_data()
    clf, acc_binary, acc_cat, acc_rules = train_models(X, y)

    # Create visualizations
    print("\n📊 Generating visualizations...")
    print("  → Accuracy chart...")
    create_accuracy_chart(acc_binary, acc_cat, acc_rules, OUTPUT_DIR)

    print("  → Decision tree...")
    create_tree_visualization(clf, acc_binary, OUTPUT_DIR)

    print("  → Feature importance...")
    create_feature_importance(clf, OUTPUT_DIR)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"✅ SUCCESS! Created 3 high-quality visualizations")
    print(f"{'=' * 60}")
    print(f"\n📁 Location: {OUTPUT_DIR}/")
    print(f"\n📈 Results:")
    print(f"   Binary Splits:    {acc_binary:.1%} ⭐ (Winner)")
    print(f"   Categorical:      {acc_cat:.1%}")
    print(f"   Rule-Based:       {acc_rules:.1%}")
    print(f"\n💡 Binary splits are optimal for continuous features!")

if __name__ == '__main__':
    main()


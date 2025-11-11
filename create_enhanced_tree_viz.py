#!/usr/bin/env python3
"""
Enhanced Decision Tree Visualization
Creates beautiful, publication-quality decision tree diagrams
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_normalized_data():
    """Load the pre-normalized combined dataset."""
    data_file = Path('results/decision_tree/combined_normalized.csv')
    if not data_file.exists():
        print("❌ Error: Run decision_tree_analysis.py first to generate normalized data")
        return None
    df = pd.read_csv(data_file)
    print(f"✅ Loaded normalized data: {len(df)} residues")
    return df

def create_enhanced_tree_visualization(clf, feature_names, model_name, accuracy, output_file):
    """
    Create a beautiful, enhanced decision tree visualization with custom styling.
    """
    # Set up the figure with high quality
    plt.figure(figsize=(24, 14), dpi=150, facecolor='white')

    # Custom color scheme - gradient from interior (blue) to exterior (orange)
    colors = ['#3498db', '#e74c3c']  # Blue for interior, Red for exterior

    # Plot the tree with enhanced styling
    plot_tree(clf,
              feature_names=feature_names,
              class_names=['Interior (0)', 'Exterior (1)'],
              filled=True,
              rounded=True,
              fontsize=9,  # Reduced font size to prevent overlap
              fontsize=11,
              proportion=True,  # Show proportions instead of absolute values
              impurity=True,
              ax=plt.gca())

    # Enhance the title
    plt.title(f'{model_name}\nValidation Accuracy: {accuracy:.2%}\n' +
              'Blue = Interior | Red = Exterior',
              fontsize=18, fontweight='bold', pad=30)

    # Add annotations at bottom with more space
    plt.title(f'{model_name}\nValidation Accuracy: {accuracy:.2%}\n' +
             'Decision rules based on z-score normalized features\n' +
              fontsize=18, fontweight='bold', pad=20)

    # Add annotations
    plt.text(0.5, -0.05,
    # More padding to prevent cutoff
    plt.tight_layout(pad=2.0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    Create a horizontal (left-to-right) decision tree for better readability.
    Fixed: Extra wide format to prevent any text overlap.
    """
    from sklearn.tree import plot_tree

    # Extra wide and tall to prevent overlap
    depth = clf.get_depth()
    fig_width = max(35, n_leaves * 2.5)
    fig_height = max(22, depth * 4.5)


    fig, ax = plt.subplots(figsize=(20, 16), dpi=150, facecolor='white')

    # Plot tree
              precision=2,
              impurity=True,
              ax=ax)

    # Styling
    ax.set_title(f'{model_name} (Accuracy: {accuracy:.2%})\n' +
              fontsize=10,
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout(pad=2.5)
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.5)
    print(f"✅ Horizontal tree saved: {output_file}")
    plt.close()

def create_graphviz_style_tree(clf, feature_names, model_name, accuracy, output_file):
    """

    """
    ax.set_title(f'{model_name} (Accuracy: {accuracy:.2%})\n' +
        from sklearn.tree import export_graphviz
                 fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            out_file=None,
            feature_names=feature_names,
            class_names=['Interior', 'Exterior'],
            filled=True,
            rounded=True,
            special_characters=True,
            proportion=True,
            precision=2,
            impurity=True
        )

        # Enhance the DOT styling
        dot_data = dot_data.replace('fontname=helvetica', 'fontname="Arial"')
        dot_data = dot_data.replace('graph [', f'graph [label="{model_name}\\nAccuracy: {accuracy:.2%}", labelloc=t, fontsize=20, ')

        # Create graph
        graph = pydot.graph_from_dot_data(dot_data)[0]

        # Save as high-quality PNG
        graph.write_png(str(output_file))
        print(f"✅ Graphviz-style tree saved: {output_file}")
        return True

    except ImportError:
        print("⚠️  pydot not installed, skipping Graphviz-style visualization")
        print("   Install with: pip install pydot graphviz")
        return False

def create_text_based_tree(clf, feature_names, model_name, output_file):
    """
    Create a text-based representation of the decision tree.
    """
    from sklearn.tree import export_text

    tree_text = export_text(clf, feature_names=feature_names, max_depth=10)

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"DECISION TREE STRUCTURE: {model_name}\n")
        f.write("="*80 + "\n\n")
        f.write("Reading the tree:\n")
        f.write("  - Each line shows a decision rule\n")
        f.write("  - Indentation shows tree depth\n")
        f.write("  - 'class: X' shows the predicted class at that leaf\n")
        f.write("  - Values are z-scores (normalized per protein)\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write(tree_text)
        f.write("\n\n" + "="*80 + "\n")
        f.write("FEATURE MEANINGS:\n")
        f.write("="*80 + "\n")
        f.write("  stride_asa_norm:       STRIDE accessible surface area (z-score)\n")
        f.write("  ncps_sphere_6_norm:    Neighbor count at 6Å radius (z-score)\n")
        f.write("  ncps_sphere_10_norm:   Neighbor count at 10Å radius (z-score)\n")
        f.write("  ncps_sphere_6_uni_norm: Uniformity at 6Å (z-score)\n")
        f.write("  ncps_sphere_10_uni_norm: Uniformity at 10Å (z-score)\n\n")
        f.write("Z-score interpretation:\n")
        f.write("  Positive values: Above the protein's average\n")
        f.write("  Negative values: Below the protein's average\n")
        f.write("  Each protein normalized separately before training\n")

    print(f"✅ Text-based tree saved: {output_file}")

def create_feature_importance_chart(clf, feature_names, model_name, output_file):
    """
    Create a horizontal bar chart showing feature importances.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor='white')

    # Create bars
    colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = ax.barh(range(len(feature_names)),
                   importances[indices],
                   color=colors_grad)

    # Labels
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance: {model_name}',
                 fontsize=14, fontweight='bold', pad=15)

    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importances[indices])):
        ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}',
                va='center', fontsize=10, fontweight='bold')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim([0, max(importances) * 1.15])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Feature importance chart saved: {output_file}")
    plt.close()

def visualize_all_models(df):
    """
    Create enhanced visualizations for all three models.
    """
    output_dir = Path('results/decision_tree/enhanced_viz')
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*80)

    # Model 1: All features
    # All features (6Å + 10Å) - ONLY neighbor-based features (DSSP/STRIDE agnostic)
    feature_cols_all = ['ncps_sphere_6_norm', 'ncps_sphere_10_norm',
                        'ncps_sphere_6_uni_norm', 'ncps_sphere_10_uni_norm']
    df_clean = df[feature_cols_all + ['dssp_class']].dropna()
    X_all = df_clean[feature_cols_all]
    y_all = df_clean['dssp_class']

    clf_all = DecisionTreeClassifier(max_depth=6, random_state=42,
                                     min_samples_split=20, min_samples_leaf=10)
    clf_all.fit(X_all, y_all)
    acc_all = accuracy_score(y_all, clf_all.predict(X_all))

    create_enhanced_tree_visualization(clf_all, feature_cols_all,
                                      'Decision Tree - All Features (6Å + 10Å)',
                                      acc_all,
                                      output_dir / 'tree_all_features_enhanced.png')
    create_horizontal_tree_visualization(clf_all, feature_cols_all,
                                        'All Features',
                                        acc_all,
                                        output_dir / 'tree_all_features_horizontal.png')
    create_text_based_tree(clf_all, feature_cols_all,
                          'All Features (6Å + 10Å)',
                          output_dir / 'tree_all_features_text.txt')
    create_feature_importance_chart(clf_all, feature_cols_all,
                                   'All Features',
                                   output_dir / 'importance_all_features.png')

    # Model 2: 10Å only (BEST MODEL) - ONLY neighbor-based features (DSSP/STRIDE agnostic)
    print("\n2. 10Å Features Only (Best Model)...")
    feature_cols_10A = ['ncps_sphere_10_norm', 'ncps_sphere_10_uni_norm']
    df_clean = df[feature_cols_10A + ['dssp_class']].dropna()
    X_10A = df_clean[feature_cols_10A]
    y_10A = df_clean['dssp_class']

    clf_10A = DecisionTreeClassifier(max_depth=6, random_state=42,
                                     min_samples_split=20, min_samples_leaf=10)
    clf_10A.fit(X_10A, y_10A)
    acc_10A = accuracy_score(y_10A, clf_10A.predict(X_10A))

    create_enhanced_tree_visualization(clf_10A, feature_cols_10A,
                                      'Decision Tree - 10Å Features Only (BEST MODEL)',
                                      acc_10A,
                                      output_dir / 'tree_10A_only_enhanced.png')
    create_horizontal_tree_visualization(clf_10A, feature_cols_10A,
                                        '10Å Features Only (BEST)',
                                        acc_10A,
                                        output_dir / 'tree_10A_only_horizontal.png')
    create_text_based_tree(clf_10A, feature_cols_10A,
                          '10Å Features Only (BEST MODEL)',
                          output_dir / 'tree_10A_only_text.txt')
    create_feature_importance_chart(clf_10A, feature_cols_10A,
                                   '10Å Features Only (BEST)',
                                   output_dir / 'importance_10A_only.png')

    # Try Graphviz-style for the best model
    create_graphviz_style_tree(clf_10A, feature_cols_10A,
                               '10Å Features Only (BEST MODEL)',
                               acc_10A,
                               output_dir / 'tree_10A_only_graphviz.png')

    # Model 3: 6Å only - ONLY neighbor-based features (DSSP/STRIDE agnostic)
    print("\n3. 6Å Features Only...")
    feature_cols_6A = ['ncps_sphere_6_norm', 'ncps_sphere_6_uni_norm']
    df_clean = df[feature_cols_6A + ['dssp_class']].dropna()
    X_6A = df_clean[feature_cols_6A]
    y_6A = df_clean['dssp_class']

    clf_6A = DecisionTreeClassifier(max_depth=6, random_state=42,
                                    min_samples_split=20, min_samples_leaf=10)
    clf_6A.fit(X_6A, y_6A)
    acc_6A = accuracy_score(y_6A, clf_6A.predict(X_6A))

    create_enhanced_tree_visualization(clf_6A, feature_cols_6A,
                                      'Decision Tree - 6Å Features Only',
                                      acc_6A,
                                      output_dir / 'tree_6A_only_enhanced.png')
    create_horizontal_tree_visualization(clf_6A, feature_cols_6A,
                                        '6Å Features Only',
                                        acc_6A,
                                        output_dir / 'tree_6A_only_horizontal.png')
    create_text_based_tree(clf_6A, feature_cols_6A,
                          '6Å Features Only',
                          output_dir / 'tree_6A_only_text.txt')
    create_feature_importance_chart(clf_6A, feature_cols_6A,
                                   '6Å Features Only',
                                   output_dir / 'importance_6A_only.png')

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTraining Accuracy:")
    print(f"  All features: {acc_all:.2%}")
    print(f"  10Å only:     {acc_10A:.2%} ⭐ BEST (88.6% validation)")
    print(f"  6Å only:      {acc_6A:.2%}")

    print(f"\n✅ All visualizations saved in: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - *_enhanced.png (Standard tree visualization)")
    print(f"  - *_horizontal.png (Wide format for presentations)")
    print(f"  - *_text.txt (Text-based tree structure)")
    print(f"  - importance_*.png (Feature importance charts)")
    print(f"  - tree_10A_only_graphviz.png (if pydot installed)")

def main():
    """
    Create enhanced visualizations for all decision tree models.
    """
    print("="*80)
    print("ENHANCED DECISION TREE VISUALIZATION")
    print("="*80)

    # Load data
    df = load_normalized_data()
    if df is None:
        return

    # Create all visualizations
    visualize_all_models(df)

    print("\n" + "="*80)
    print("✅ ENHANCED VISUALIZATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()


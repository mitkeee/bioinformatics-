#!/usr/bin/env python3
"""
Create Intuitive Decision Tree Visualizations
Replaces True/False with descriptive labels and arrows
Makes the decision rules clearer and more intuitive
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def create_intuitive_tree_text(clf, feature_names, class_names, output_file):
    """
    Create a text-based tree with intuitive descriptions instead of True/False.
    """
    from sklearn.tree import export_text

    # Get the raw tree structure
    tree_text = export_text(clf, feature_names=feature_names, max_depth=10)

    # Create enhanced version with descriptions
    enhanced_lines = []
    enhanced_lines.append("="*80)
    enhanced_lines.append("DECISION TREE - INTUITIVE RULE FORMAT")
    enhanced_lines.append("="*80)
    enhanced_lines.append("")
    enhanced_lines.append("Reading the tree:")
    enhanced_lines.append("  → means 'if condition is TRUE' (left branch)")
    enhanced_lines.append("  ← means 'if condition is FALSE' (right branch)")
    enhanced_lines.append("")
    enhanced_lines.append("="*80)
    enhanced_lines.append("")

    # Process each line to make it more intuitive
    for line in tree_text.split('\n'):
        if '<=' in line:
            # Extract the condition
            parts = line.split('<=')
            if len(parts) == 2:
                indent = len(line) - len(line.lstrip())
                feature = parts[0].strip().replace('|', '').replace('---', '').strip()
                threshold = parts[1].split()[0]

                # Add intuitive descriptions
                desc = get_feature_description(feature, float(threshold))
                enhanced_lines.append(' ' * indent + f"├─ {feature} ≤ {threshold}")
                enhanced_lines.append(' ' * indent + f"│  → {desc['left']} (YES)")
                enhanced_lines.append(' ' * indent + f"│  ← {desc['right']} (NO)")
        elif 'class:' in line:
            indent = len(line) - len(line.lstrip())
            if '0' in line:
                enhanced_lines.append(' ' * indent + "└─ PREDICT: INTERIOR (buried)")
            else:
                enhanced_lines.append(' ' * indent + "└─ PREDICT: EXTERIOR (exposed)")
        elif line.strip():
            enhanced_lines.append(line)

    # Add legend
    enhanced_lines.append("")
    enhanced_lines.append("="*80)
    enhanced_lines.append("INTERPRETATION GUIDE")
    enhanced_lines.append("="*80)
    enhanced_lines.append("")
    enhanced_lines.append("Z-score values (normalized per protein):")
    enhanced_lines.append("  Negative values: BELOW the protein's average")
    enhanced_lines.append("  Positive values: ABOVE the protein's average")
    enhanced_lines.append("  Zero: AT the protein's average")
    enhanced_lines.append("")
    enhanced_lines.append("Feature meanings:")
    enhanced_lines.append("  ncps_sphere_6_norm:  Number of neighbors within 6Å")
    enhanced_lines.append("  ncps_sphere_10_norm: Number of neighbors within 10Å")
    enhanced_lines.append("  ncps_sphere_6_uni_norm:  Uniformity at 6Å (how evenly distributed)")
    enhanced_lines.append("  ncps_sphere_10_uni_norm: Uniformity at 10Å")
    enhanced_lines.append("  stride_asa_norm:     Accessible surface area")
    enhanced_lines.append("")

    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(enhanced_lines))

    print(f"✅ Intuitive text tree saved: {output_file}")

def get_feature_description(feature, threshold):
    """
    Get intuitive descriptions for each feature and threshold.
    """
    descriptions = {
        'ncps_sphere_6_norm': {
            'left': f"Few neighbors at 6Å (≤ {threshold:.2f} std from mean)",
            'right': f"Many neighbors at 6Å (> {threshold:.2f} std from mean)"
        },
        'ncps_sphere_10_norm': {
            'left': f"Few neighbors at 10Å (≤ {threshold:.2f} std from mean)",
            'right': f"Many neighbors at 10Å (> {threshold:.2f} std from mean)"
        },
        'ncps_sphere_6_uni_norm': {
            'left': f"Low uniformity at 6Å (≤ {threshold:.2f} std from mean) - neighbors on one side",
            'right': f"High uniformity at 6Å (> {threshold:.2f} std from mean) - surrounded"
        },
        'ncps_sphere_10_uni_norm': {
            'left': f"Low uniformity at 10Å (≤ {threshold:.2f} std from mean) - one-sided",
            'right': f"High uniformity at 10Å (> {threshold:.2f} std from mean) - surrounded"
        },
        'stride_asa_norm': {
            'left': f"Low surface area (≤ {threshold:.2f} std from mean) - buried",
            'right': f"High surface area (> {threshold:.2f} std from mean) - exposed"
        }
    }

    return descriptions.get(feature, {
        'left': f"Value ≤ {threshold:.2f}",
        'right': f"Value > {threshold:.2f}"
    })

def create_custom_graphical_tree(clf, feature_names, model_name, accuracy, output_file):
    """
    Create a custom graphical tree with descriptive labels instead of True/False.
    """
    try:
        from sklearn.tree import export_graphviz
        import pydot

        # Export tree to DOT format
        dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=feature_names,
            class_names=['Interior\n(Buried)', 'Exterior\n(Exposed)'],
            filled=True,
            rounded=True,
            special_characters=True,
            proportion=True,
            precision=2,
            impurity=False,  # Remove impurity to reduce clutter
        )

        # Customize the DOT data to replace True/False with descriptions
        dot_data = dot_data.replace('digraph Tree {',
                                    f'digraph Tree {{\nlabel="{model_name}\\nAccuracy: {accuracy:.2%}\\n\\n";\n' +
                                    'labelloc=t;\nfontsize=20;\nfontname="Arial Bold";\n')

        # Replace edge labels
        dot_data = dot_data.replace('label="True"', 'label="YES ✓"')
        dot_data = dot_data.replace('label="False"', 'label="NO ✗"')

        # Enhance styling
        dot_data = dot_data.replace('fontname=helvetica', 'fontname="Arial"')
        dot_data = dot_data.replace('fontsize=12', 'fontsize=11')

        # Create graph
        graph = pydot.graph_from_dot_data(dot_data)[0]
        graph.write_png(str(output_file))

        print(f"✅ Custom graphical tree saved: {output_file}")
        return True

    except ImportError:
        print("⚠️  pydot not installed, skipping custom graphical tree")
        return False

def create_flowchart_style_description(clf, feature_names, output_file):
    """
    Create a flowchart-style description of the decision tree.
    """
    lines = []
    lines.append("="*80)
    lines.append("DECISION TREE FLOWCHART")
    lines.append("="*80)
    lines.append("")
    lines.append("START → Analyzing a residue...")
    lines.append("")

    # Get tree structure
    tree = clf.tree_

    def print_node(node_id, depth=0, path=""):
        indent = "  " * depth

        if tree.feature[node_id] != -2:  # Not a leaf
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]

            desc = get_feature_description(feature_name, threshold)

            lines.append(f"{indent}{'└─ ' if depth > 0 else ''}QUESTION: Is {feature_name} ≤ {threshold:.2f}?")
            lines.append(f"{indent}   (i.e., {desc['left'].lower()}?)")
            lines.append(f"{indent}   │")
            lines.append(f"{indent}   ├─ YES → {desc['left']}")

            # Left child (YES)
            left_child = tree.children_left[node_id]
            if tree.feature[left_child] == -2:  # Leaf
                class_idx = np.argmax(tree.value[left_child])
                class_name = "INTERIOR (buried)" if class_idx == 0 else "EXTERIOR (exposed)"
                samples = int(tree.value[left_child][0].sum())
                lines.append(f"{indent}   │     └─ PREDICT: {class_name}")
            else:
                print_node(left_child, depth + 1, path + "→YES→")

            lines.append(f"{indent}   │")
            lines.append(f"{indent}   └─ NO → {desc['right']}")

            # Right child (NO)
            right_child = tree.children_right[node_id]
            if tree.feature[right_child] == -2:  # Leaf
                class_idx = np.argmax(tree.value[right_child])
                class_name = "INTERIOR (buried)" if class_idx == 0 else "EXTERIOR (exposed)"
                samples = int(tree.value[right_child][0].sum())
                lines.append(f"{indent}         └─ PREDICT: {class_name}")
            else:
                print_node(right_child, depth + 1, path + "→NO→")

            lines.append("")

    print_node(0)

    lines.append("="*80)
    lines.append("END OF DECISION TREE")
    lines.append("="*80)

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✅ Flowchart description saved: {output_file}")

def load_data_and_create_intuitive_trees():
    """
    Load data and create intuitive tree visualizations.
    """
    print("="*80)
    print("CREATING INTUITIVE DECISION TREE VISUALIZATIONS")
    print("Replacing True/False with descriptive labels")
    print("="*80)

    # Load normalized data
    data_file = Path('results/decision_tree/combined_normalized.csv')
    if not data_file.exists():
        print("❌ Error: Run decision_tree_analysis.py first")
        return

    df = pd.read_csv(data_file)
    print(f"\n✅ Loaded data: {len(df)} residues")

    output_dir = Path('results/decision_tree/intuitive_viz')
    output_dir.mkdir(exist_ok=True)

    # Create intuitive visualizations for 10Å model (BEST MODEL)
    print("\n" + "-"*80)
    print("Creating intuitive visualizations for 10Å model (BEST - 88.6% accuracy)")
    print("-"*80)

    feature_cols = ['stride_asa_norm', 'ncps_sphere_10_norm', 'ncps_sphere_10_uni_norm']
    df_clean = df[feature_cols + ['dssp_class']].dropna()
    X = df_clean[feature_cols]
    y = df_clean['dssp_class']

    clf = DecisionTreeClassifier(max_depth=5, random_state=42,
                                 min_samples_split=20, min_samples_leaf=10)
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))

    # Create different intuitive formats
    create_intuitive_tree_text(clf, feature_cols, ['Interior', 'Exterior'],
                              output_dir / 'tree_10A_intuitive_text.txt')

    create_flowchart_style_description(clf, feature_cols,
                                      output_dir / 'tree_10A_flowchart.txt')

    create_custom_graphical_tree(clf, feature_cols,
                                'Decision Tree - 10Å Features (BEST MODEL)',
                                acc,
                                output_dir / 'tree_10A_intuitive_graphic.png')

    print("\n" + "="*80)
    print("✅ INTUITIVE VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput files in: {output_dir}/")
    print("  - tree_10A_intuitive_text.txt (with YES/NO labels)")
    print("  - tree_10A_flowchart.txt (flowchart style)")
    print("  - tree_10A_intuitive_graphic.png (if pydot installed)")
    print("\n" + "="*80)
    print("EXPLANATION: Why True/False?")
    print("="*80)
    print("Decision trees use binary (True/False) splits because:")
    print("  1. Mathematically optimal - finds best single threshold")
    print("  2. Computationally efficient - one condition per split")
    print("  3. Handles continuous features (like our z-scores)")
    print("  4. Easy to implement in code")
    print("\nAlternatives:")
    print("  - Multi-way splits: Possible but less common (e.g., CART extensions)")
    print("  - Categorical splits: For non-numeric features")
    print("  - Fuzzy decision trees: Allow partial membership")
    print("\nFor YOUR data: Binary splits are BEST because:")
    print("  - Features are continuous (z-scores)")
    print("  - Clear threshold separates interior/exterior")
    print("  - Interpretable rules (e.g., 'few neighbors' vs 'many neighbors')")
    print("="*80)

if __name__ == "__main__":
    load_data_and_create_intuitive_trees()


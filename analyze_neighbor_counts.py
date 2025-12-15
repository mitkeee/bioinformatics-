"""
Detailed analysis of neighbor count distributions to validate/improve thresholds
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def analyze_neighbor_distributions():
    """Analyze neighbor count distributions across all proteins"""

    csv_files = list(Path('final_reports').glob('*_detailed_results.csv'))

    all_data = {
        'surface_nc6': [], 'surface_nc10': [],
        'buried_nc6': [], 'buried_nc10': [],
        'surface_uni6': [], 'surface_uni10': [],
        'buried_uni6': [], 'buried_uni10': []
    }

    protein_stats = []

    print("="*80)
    print("NEIGHBOR COUNT DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(csv_files)} proteins...\n")

    for csv in csv_files:
        df = pd.read_csv(csv)

        if 'dssp_class' not in df.columns or df['dssp_class'].isna().all():
            continue

        df_valid = df[df['dssp_class'].notna()].copy()

        surface = df_valid[df_valid['dssp_class'] == 1]
        buried = df_valid[df_valid['dssp_class'] == 0]

        if len(surface) > 0 and len(buried) > 0:
            all_data['surface_nc6'].extend(surface['ncps_sphere_6'].tolist())
            all_data['surface_nc10'].extend(surface['ncps_sphere_10'].tolist())
            all_data['buried_nc6'].extend(buried['ncps_sphere_6'].tolist())
            all_data['buried_nc10'].extend(buried['ncps_sphere_10'].tolist())

            all_data['surface_uni6'].extend(surface['ncps_sphere_6_uni'].tolist())
            all_data['surface_uni10'].extend(surface['ncps_sphere_10_uni'].tolist())
            all_data['buried_uni6'].extend(buried['ncps_sphere_6_uni'].tolist())
            all_data['buried_uni10'].extend(buried['ncps_sphere_10_uni'].tolist())

            protein_stats.append({
                'protein': csv.stem.replace('_detailed_results', ''),
                'surf_nc6_mean': surface['ncps_sphere_6'].mean(),
                'surf_nc10_mean': surface['ncps_sphere_10'].mean(),
                'bur_nc6_mean': buried['ncps_sphere_6'].mean(),
                'bur_nc10_mean': buried['ncps_sphere_10'].mean(),
                'separation_nc6': buried['ncps_sphere_6'].mean() - surface['ncps_sphere_6'].mean(),
                'separation_nc10': buried['ncps_sphere_10'].mean() - surface['ncps_sphere_10'].mean(),
            })

    # Calculate statistics
    print("SURFACE RESIDUES (DSSP exterior):")
    print(f"  NC6:  mean={np.mean(all_data['surface_nc6']):.2f}, median={np.median(all_data['surface_nc6']):.2f}, std={np.std(all_data['surface_nc6']):.2f}")
    print(f"  NC10: mean={np.mean(all_data['surface_nc10']):.2f}, median={np.median(all_data['surface_nc10']):.2f}, std={np.std(all_data['surface_nc10']):.2f}")
    print(f"  UNI6:  mean={np.mean(all_data['surface_uni6']):.3f}, median={np.median(all_data['surface_uni6']):.3f}")
    print(f"  UNI10: mean={np.mean(all_data['surface_uni10']):.3f}, median={np.median(all_data['surface_uni10']):.3f}")

    print("\nBURIED RESIDUES (DSSP interior):")
    print(f"  NC6:  mean={np.mean(all_data['buried_nc6']):.2f}, median={np.median(all_data['buried_nc6']):.2f}, std={np.std(all_data['buried_nc6']):.2f}")
    print(f"  NC10: mean={np.mean(all_data['buried_nc10']):.2f}, median={np.median(all_data['buried_nc10']):.2f}, std={np.std(all_data['buried_nc10']):.2f}")
    print(f"  UNI6:  mean={np.mean(all_data['buried_uni6']):.3f}, median={np.median(all_data['buried_uni6']):.3f}")
    print(f"  UNI10: mean={np.mean(all_data['buried_uni10']):.3f}, median={np.median(all_data['buried_uni10']):.3f}")

    # Calculate separation
    sep_nc6 = np.mean(all_data['buried_nc6']) - np.mean(all_data['surface_nc6'])
    sep_nc10 = np.mean(all_data['buried_nc10']) - np.mean(all_data['surface_nc10'])

    print(f"\nSEPARATION (buried - surface):")
    print(f"  NC6:  {sep_nc6:.2f} neighbors")
    print(f"  NC10: {sep_nc10:.2f} neighbors")

    # Current thresholds analysis
    print(f"\n{'='*80}")
    print("CURRENT THRESHOLDS ANALYSIS")
    print(f"{'='*80}")
    print(f"\nCurrent: NC6=5.0, NC10=16.0")

    # Calculate percentiles
    surf_nc6_percentiles = np.percentile(all_data['surface_nc6'], [25, 50, 75, 90, 95])
    surf_nc10_percentiles = np.percentile(all_data['surface_nc10'], [25, 50, 75, 90, 95])
    bur_nc6_percentiles = np.percentile(all_data['buried_nc6'], [25, 50, 75, 90, 95])
    bur_nc10_percentiles = np.percentile(all_data['buried_nc10'], [25, 50, 75, 90, 95])

    print("\nSURFACE NC6 percentiles: 25%={:.1f}, 50%={:.1f}, 75%={:.1f}, 90%={:.1f}, 95%={:.1f}".format(*surf_nc6_percentiles))
    print("BURIED NC6 percentiles:  25%={:.1f}, 50%={:.1f}, 75%={:.1f}, 90%={:.1f}, 95%={:.1f}".format(*bur_nc6_percentiles))
    print("\nSURFACE NC10 percentiles: 25%={:.1f}, 50%={:.1f}, 75%={:.1f}, 90%={:.1f}, 95%={:.1f}".format(*surf_nc10_percentiles))
    print("BURIED NC10 percentiles:  25%={:.1f}, 50%={:.1f}, 75%={:.1f}, 90%={:.1f}, 95%={:.1f}".format(*bur_nc10_percentiles))

    # Optimal threshold suggestion
    print(f"\n{'='*80}")
    print("THRESHOLD RECOMMENDATIONS")
    print(f"{'='*80}")

    # Method 1: Midpoint between means
    opt_nc6_mid = (np.mean(all_data['surface_nc6']) + np.mean(all_data['buried_nc6'])) / 2
    opt_nc10_mid = (np.mean(all_data['surface_nc10']) + np.mean(all_data['buried_nc10'])) / 2

    print(f"\nMethod 1 - Midpoint between surface and buried means:")
    print(f"  Optimal NC6:  {opt_nc6_mid:.2f} (current: 5.0)")
    print(f"  Optimal NC10: {opt_nc10_mid:.2f} (current: 16.0)")

    # Method 2: 75th percentile of surface
    opt_nc6_p75 = surf_nc6_percentiles[2]  # 75th percentile
    opt_nc10_p75 = surf_nc10_percentiles[2]

    print(f"\nMethod 2 - 75th percentile of surface residues:")
    print(f"  Optimal NC6:  {opt_nc6_p75:.2f} (current: 5.0)")
    print(f"  Optimal NC10: {opt_nc10_p75:.2f} (current: 16.0)")

    # Method 3: 25th percentile of buried
    opt_nc6_p25_bur = bur_nc6_percentiles[0]  # 25th percentile
    opt_nc10_p25_bur = bur_nc10_percentiles[0]

    print(f"\nMethod 3 - 25th percentile of buried residues:")
    print(f"  Optimal NC6:  {opt_nc6_p25_bur:.2f} (current: 5.0)")
    print(f"  Optimal NC10: {opt_nc10_p25_bur:.2f} (current: 16.0)")

    # Overlap analysis
    print(f"\n{'='*80}")
    print("DISTRIBUTION OVERLAP ANALYSIS")
    print(f"{'='*80}")

    # Find how many surface residues are above threshold
    surf_above_nc6 = sum(1 for x in all_data['surface_nc6'] if x >= 5.0)
    surf_above_nc10 = sum(1 for x in all_data['surface_nc10'] if x >= 16.0)

    # Find how many buried residues are below threshold
    bur_below_nc6 = sum(1 for x in all_data['buried_nc6'] if x < 5.0)
    bur_below_nc10 = sum(1 for x in all_data['buried_nc10'] if x < 16.0)

    print(f"\nAt current NC6=5.0:")
    print(f"  Surface residues above threshold: {surf_above_nc6}/{len(all_data['surface_nc6'])} ({100*surf_above_nc6/len(all_data['surface_nc6']):.1f}%)")
    print(f"  Buried residues below threshold:  {bur_below_nc6}/{len(all_data['buried_nc6'])} ({100*bur_below_nc6/len(all_data['buried_nc6']):.1f}%)")

    print(f"\nAt current NC10=16.0:")
    print(f"  Surface residues above threshold: {surf_above_nc10}/{len(all_data['surface_nc10'])} ({100*surf_above_nc10/len(all_data['surface_nc10']):.1f}%)")
    print(f"  Buried residues below threshold:  {bur_below_nc10}/{len(all_data['buried_nc10'])} ({100*bur_below_nc10/len(all_data['buried_nc10']):.1f}%)")

    # Create visualizations
    create_distribution_plots(all_data)

    # Save detailed stats
    save_detailed_statistics(protein_stats, all_data)

    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print("✓ Distribution plots saved to: neighbor_distribution_analysis.png")
    print("✓ Detailed statistics saved to: neighbor_statistics.txt")
    print(f"{'='*80}\n")


def create_distribution_plots(data):
    """Create distribution plots for neighbor counts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # NC6 distribution
    axes[0, 0].hist(data['surface_nc6'], bins=30, alpha=0.5, label='Surface', color='blue', density=True)
    axes[0, 0].hist(data['buried_nc6'], bins=30, alpha=0.5, label='Buried', color='red', density=True)
    axes[0, 0].axvline(5.0, color='green', linestyle='--', linewidth=2, label='Current threshold')
    axes[0, 0].set_xlabel('Neighbor Count (6Å)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('NC6 Distribution: Surface vs Buried')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # NC10 distribution
    axes[0, 1].hist(data['surface_nc10'], bins=30, alpha=0.5, label='Surface', color='blue', density=True)
    axes[0, 1].hist(data['buried_nc10'], bins=30, alpha=0.5, label='Buried', color='red', density=True)
    axes[0, 1].axvline(16.0, color='green', linestyle='--', linewidth=2, label='Current threshold')
    axes[0, 1].set_xlabel('Neighbor Count (10Å)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('NC10 Distribution: Surface vs Buried')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # UNI6 distribution
    axes[1, 0].hist(data['surface_uni6'], bins=30, alpha=0.5, label='Surface', color='blue', density=True)
    axes[1, 0].hist(data['buried_uni6'], bins=30, alpha=0.5, label='Buried', color='red', density=True)
    axes[1, 0].axvline(0.38, color='green', linestyle='--', linewidth=2, label='Current threshold')
    axes[1, 0].set_xlabel('Uniformity (6Å)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('UNI6 Distribution: Surface vs Buried')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # UNI10 distribution
    axes[1, 1].hist(data['surface_uni10'], bins=30, alpha=0.5, label='Surface', color='blue', density=True)
    axes[1, 1].hist(data['buried_uni10'], bins=30, alpha=0.5, label='Buried', color='red', density=True)
    axes[1, 1].axvline(0.48, color='green', linestyle='--', linewidth=2, label='Current threshold')
    axes[1, 1].set_xlabel('Uniformity (10Å)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('UNI10 Distribution: Surface vs Buried')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neighbor_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_statistics(protein_stats, all_data):
    """Save detailed statistics to file"""
    with open('neighbor_statistics.txt', 'w') as f:
        f.write("DETAILED NEIGHBOR COUNT STATISTICS\n")
        f.write("="*80 + "\n\n")

        f.write("PER-PROTEIN STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Protein':<20} {'Surf NC6':>10} {'Surf NC10':>10} {'Bur NC6':>10} {'Bur NC10':>10} {'Sep NC6':>10} {'Sep NC10':>10}\n")
        f.write("-"*80 + "\n")

        for stats in sorted(protein_stats, key=lambda x: x['separation_nc6'], reverse=True):
            f.write(f"{stats['protein']:<20} {stats['surf_nc6_mean']:>10.2f} {stats['surf_nc10_mean']:>10.2f} "
                   f"{stats['bur_nc6_mean']:>10.2f} {stats['bur_nc10_mean']:>10.2f} "
                   f"{stats['separation_nc6']:>10.2f} {stats['separation_nc10']:>10.2f}\n")

        f.write("\n" + "="*80 + "\n\n")
        f.write("OVERALL STATISTICS:\n\n")

        f.write(f"Surface NC6:  mean={np.mean(all_data['surface_nc6']):.2f}, std={np.std(all_data['surface_nc6']):.2f}\n")
        f.write(f"Surface NC10: mean={np.mean(all_data['surface_nc10']):.2f}, std={np.std(all_data['surface_nc10']):.2f}\n")
        f.write(f"Buried NC6:   mean={np.mean(all_data['buried_nc6']):.2f}, std={np.std(all_data['buried_nc6']):.2f}\n")
        f.write(f"Buried NC10:  mean={np.mean(all_data['buried_nc10']):.2f}, std={np.std(all_data['buried_nc10']):.2f}\n")


if __name__ == "__main__":
    analyze_neighbor_distributions()


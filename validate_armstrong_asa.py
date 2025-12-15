"""
Validate NCPS exterior predictions against Armstrong (ASA) values
Answers the question: "Is it really exterior?"
"""

import pandas as pd
from pathlib import Path
import numpy as np

def validate_armstrong_values():
    """Check if NCPS exterior predictions match ASA (Armstrong) values"""

    csv_files = list(Path('final_reports').glob('*_detailed_results.csv'))

    all_stats = []
    total_ncps_ext = 0
    total_with_high_asa = 0
    total_false_positives = 0

    print("="*80)
    print("ARMSTRONG (ASA) VALIDATION ACROSS ALL PROTEINS")
    print("="*80)
    print(f"\nAnalyzing {len(csv_files)} proteins...\n")

    for csv in csv_files:
        df = pd.read_csv(csv)

        if 'dssp_asa' not in df.columns or 'ncps_class' not in df.columns:
            continue

        df_valid = df[df['dssp_asa'].notna()].copy()

        if len(df_valid) == 0:
            continue

        # NCPS exterior residues
        ncps_ext = df_valid[df_valid['ncps_class'] == 1]

        if len(ncps_ext) == 0:
            continue

        # Count by ASA level
        low_asa = len(ncps_ext[ncps_ext['dssp_asa'] < 25])  # False positives
        med_asa = len(ncps_ext[(ncps_ext['dssp_asa'] >= 25) & (ncps_ext['dssp_asa'] < 50)])
        high_asa = len(ncps_ext[ncps_ext['dssp_asa'] >= 50])

        truly_exterior = med_asa + high_asa
        pct_correct = 100 * truly_exterior / len(ncps_ext) if len(ncps_ext) > 0 else 0

        all_stats.append({
            'protein': csv.stem.replace('_detailed_results', ''),
            'ncps_ext_total': len(ncps_ext),
            'low_asa': low_asa,
            'med_asa': med_asa,
            'high_asa': high_asa,
            'pct_correct': pct_correct,
            'mean_asa': ncps_ext['dssp_asa'].mean()
        })

        total_ncps_ext += len(ncps_ext)
        total_with_high_asa += truly_exterior
        total_false_positives += low_asa

    # Overall statistics
    overall_pct = 100 * total_with_high_asa / total_ncps_ext if total_ncps_ext > 0 else 0

    print("OVERALL RESULTS:")
    print("-"*80)
    print(f"Total NCPS exterior predictions: {total_ncps_ext}")
    print(f"Truly exterior (ASA >= 25%):     {total_with_high_asa} ({overall_pct:.1f}%)")
    print(f"False positives (ASA < 25%):     {total_false_positives} ({100*total_false_positives/total_ncps_ext:.1f}%)")
    print()

    # Categorize
    print("="*80)
    print("VERDICT:")
    print("="*80)

    if overall_pct >= 75:
        print("✅ YES - NCPS exterior predictions are MOSTLY TRUE")
        print(f"   {overall_pct:.1f}% of predicted exterior residues have significant ASA")
        print("   The method is reliable for identifying surface residues")
    elif overall_pct >= 60:
        print("⚠️  PARTIAL - NCPS has moderate accuracy")
        print(f"   {overall_pct:.1f}% of predicted exterior residues have ASA >= 25%")
        print("   Some over-prediction of exterior residues")
    else:
        print("❌ NO - NCPS over-predicts exterior")
        print(f"   Only {overall_pct:.1f}% of predicted exterior have ASA >= 25%")
        print("   Too many false positives")

    print()
    print("EXPLANATION:")
    print("-"*80)
    print(f"The {100*total_false_positives/total_ncps_ext:.1f}% false positives are residues with:")
    print("  • Few neighbors (look exterior to NCPS)")
    print("  • Low ASA <25% (classified interior by DSSP)")
    print()
    print("These are typically:")
    print("  1. Residues in shallow surface grooves")
    print("  2. Residues on protein edges/corners")
    print("  3. Residues in clefts (partially accessible)")
    print("  4. Borderline cases near the 25% ASA cutoff")
    print()
    print("This is EXPECTED because:")
    print("  • NCPS uses 3D geometry (neighbor counting)")
    print("  • DSSP uses 2D surface (rolling sphere accessibility)")
    print("  • They measure fundamentally different properties!")
    print()

    # Per-protein variation
    print("="*80)
    print("PER-PROTEIN STATISTICS:")
    print("-"*80)
    print(f"{'Protein':<15} {'NCPS Ext':>10} {'True Ext %':>12} {'Mean ASA':>10}")
    print("-"*80)

    for stat in sorted(all_stats, key=lambda x: x['pct_correct'], reverse=True)[:10]:
        print(f"{stat['protein']:<15} {stat['ncps_ext_total']:>10} {stat['pct_correct']:>11.1f}% {stat['mean_asa']:>9.1f}%")

    print("...")
    print()

    # Save report
    with open('armstrong_validation_report.txt', 'w') as f:
        f.write("ARMSTRONG (ASA) VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write("QUESTION: Are NCPS exterior predictions really exterior?\n\n")
        f.write(f"ANSWER: {overall_pct:.1f}% of NCPS exterior predictions have ASA >= 25%\n\n")
        f.write(f"Total NCPS exterior predictions: {total_ncps_ext}\n")
        f.write(f"Truly exterior (ASA >= 25%):     {total_with_high_asa} ({overall_pct:.1f}%)\n")
        f.write(f"False positives (ASA < 25%):     {total_false_positives} ({100*total_false_positives/total_ncps_ext:.1f}%)\n\n")

        if overall_pct >= 75:
            f.write("VERDICT: ✅ YES - Mostly true exterior predictions\n")
        elif overall_pct >= 60:
            f.write("VERDICT: ⚠️  PARTIAL - Moderate accuracy with some over-prediction\n")
        else:
            f.write("VERDICT: ❌ NO - Significant over-prediction of exterior\n")

        f.write("\nPER-PROTEIN DETAILS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Protein':<15} {'NCPS Ext':>10} {'Low ASA':>10} {'Med ASA':>10} {'High ASA':>10} {'% Correct':>12} {'Mean ASA':>10}\n")
        f.write("-"*80 + "\n")

        for stat in sorted(all_stats, key=lambda x: x['protein']):
            f.write(f"{stat['protein']:<15} {stat['ncps_ext_total']:>10} {stat['low_asa']:>10} "
                   f"{stat['med_asa']:>10} {stat['high_asa']:>10} {stat['pct_correct']:>11.1f}% {stat['mean_asa']:>9.1f}%\n")

    print("✓ Detailed report saved to: armstrong_validation_report.txt")
    print()

    return overall_pct


if __name__ == "__main__":
    pct = validate_armstrong_values()

    print("="*80)
    print("FINAL ANSWER:")
    print("="*80)
    print()

    if pct >= 75:
        print("✅ YES, NCPS exterior predictions are REALLY EXTERIOR")
        print(f"   {pct:.1f}% have significant Armstrong (ASA) values >= 25%")
    elif pct >= 60:
        print("⚠️  MOSTLY YES, but with some over-prediction")
        print(f"   {pct:.1f}% are truly exterior by ASA criteria")
    else:
        print("❌ NO, NCPS over-predicts exterior residues")
        print(f"   Only {pct:.1f}% are truly exterior by ASA")

    print()
    print("="*80)


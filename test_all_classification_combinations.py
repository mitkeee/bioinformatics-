"""Test All Classification Combinations"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

Z_LOW = -1.5
Z_HIGH = 0.0
HOMOG_LOW = 0.20
HOMOG_HIGH = 0.55

def zscore(series):
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sd

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    if 'ncps_sphere_6' in df.columns:
        df['z_6A'] = zscore(df['ncps_sphere_6'])
    else:
        df['z_6A'] = 0.0
    if 'ncps_sphere_10' in df.columns:
        df['z_10A'] = zscore(df['ncps_sphere_10'])
    else:
        df['z_10A'] = 0.0
    if 'ncps_sphere_6_uni' in df.columns:
        df['sph_var_6A'] = df['ncps_sphere_6_uni']
    if 'ncps_sphere_10_uni' in df.columns:
        df['sph_var_10A'] = df['ncps_sphere_10_uni']
    return df

def classify_with_combination(df, ext_variant, int_variant):
    out = df.copy()
    use_homog = False
    sv_cols = []
    for r in [6, 10]:
        c = f'sph_var_{r}A'
        if c in out.columns:
            sv_cols.append(c)
            use_homog = True
    labels = []
    for i, row in out.iterrows():
        z6 = row.get('z_6A', 0.0)
        z10 = row.get('z_10A', 0.0)
        sv_low_flag = False
        sv_high_flag = False
        if use_homog:
            vals = [row[c] for c in sv_cols if pd.notna(row[c])]
            if len(vals) > 0:
                sv_mean = float(np.mean(vals))
                sv_low_flag = sv_mean < HOMOG_LOW
                sv_high_flag = sv_mean > HOMOG_HIGH
        if ext_variant == 'current':
            is_exterior = (z6 <= Z_LOW) or (z10 <= Z_LOW) or sv_low_flag
        elif ext_variant == 'var1':
            is_exterior = ((z6 <= Z_LOW) and (z10 <= Z_LOW)) or sv_low_flag
        elif ext_variant == 'var2':
            is_exterior = (z6 <= Z_LOW) and (z10 <= Z_LOW) and sv_low_flag
        elif ext_variant == 'var3':
            is_exterior = (z6 <= Z_LOW) or ((z10 <= Z_LOW) and sv_low_flag)
        if int_variant == 'current':
            is_interior = ((z6 >= Z_HIGH) and (z10 >= Z_HIGH)) or sv_high_flag
        elif int_variant == '1var':
            is_interior = ((z6 >= Z_HIGH) or (z10 >= Z_HIGH)) or sv_high_flag
        elif int_variant == '2var':
            is_interior = ((z6 >= Z_HIGH) or (z10 >= Z_HIGH)) and sv_high_flag
        elif int_variant == '3var':
            is_interior = ((z6 >= Z_HIGH) and (z10 >= Z_HIGH)) and sv_high_flag
        if is_exterior and not is_interior:
            labels.append('exterior')
        elif is_interior and not is_exterior:
            labels.append('interior')
        else:
            labels.append('intermediate')
    out['burial_label'] = labels
    return out

def evaluate_classification(df, reference_col='dssp_class'):
    df_eval = df[df[reference_col].notna()].copy()
    if len(df_eval) == 0:
        return {'n_residues': 0, 'accuracy': 0.0, 'n_correct': 0, 'n_exterior': 0, 'n_interior': 0, 'n_intermediate': 0}
    ref_labels = df_eval[reference_col].map({0: 'interior', 1: 'exterior'})
    pred_labels = df_eval['burial_label'].replace('intermediate', 'exterior')
    correct = (pred_labels == ref_labels).sum()
    total = len(df_eval)
    accuracy = correct / total if total > 0 else 0.0
    counts = df_eval['burial_label'].value_counts().to_dict()
    return {'n_residues': total, 'accuracy': accuracy, 'n_correct': correct, 'n_exterior': counts.get('exterior', 0), 'n_interior': counts.get('interior', 0), 'n_intermediate': counts.get('intermediate', 0)}

def test_all_combinations(csv_file, output_file):
    df = load_and_prepare_data(csv_file)
    print(f"✅ Loaded {len(df)} residues from {csv_file.name}")
    ext_variants = ['current', 'var1', 'var2', 'var3']
    int_variants = ['current', '1var', '2var', '3var']
    results = []
    print(f"\n{'='*80}\nTESTING ALL CLASSIFICATION COMBINATIONS\n{'='*80}\nTotal combinations to test: {len(ext_variants) * len(int_variants)}\n")
    combo_num = 1
    for ext_var in ext_variants:
        for int_var in int_variants:
            print(f"Testing {combo_num:2d}/16: ext={ext_var:7s} + int={int_var:7s}...", end='')
            df_classified = classify_with_combination(df, ext_var, int_var)
            dssp_metrics = evaluate_classification(df_classified, 'dssp_class')
            stride_metrics = evaluate_classification(df_classified, 'stride_class')
            print(f" DSSP: {dssp_metrics['accuracy']:.1%}  STRIDE: {stride_metrics['accuracy']:.1%}")
            results.append({'combo_num': combo_num, 'combination': f"{ext_var}+{int_var}", 'ext_variant': ext_var, 'int_variant': int_var, 'dssp_accuracy': dssp_metrics['accuracy'], 'dssp_correct': dssp_metrics['n_correct'], 'dssp_total': dssp_metrics['n_residues'], 'stride_accuracy': stride_metrics['accuracy'], 'stride_correct': stride_metrics['n_correct'], 'stride_total': stride_metrics['n_residues'], 'n_exterior': dssp_metrics['n_exterior'], 'n_interior': dssp_metrics['n_interior'], 'n_intermediate': dssp_metrics['n_intermediate']})
            combo_num += 1
    results_df = pd.DataFrame(results).sort_values('dssp_accuracy', ascending=False)
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    return results_df

def print_summary(results_df, protein_name):
    print(f"\n{'='*80}\nRESULTS SUMMARY: {protein_name.upper()}\n{'='*80}\n")
    print("🏆 TOP 5 COMBINATIONS (by DSSP accuracy):")
    print(f"{'='*80}")
    for idx, row in results_df.head(5).iterrows():
        print(f"\n#{row['combo_num']:2d}. {row['combination']:20s}\n     Exterior logic: {row['ext_variant']}\n     Interior logic: {row['int_variant']}\n     DSSP accuracy:   {row['dssp_accuracy']:.3%} ({row['dssp_correct']}/{row['dssp_total']})\n     STRIDE accuracy: {row['stride_accuracy']:.3%} ({row['stride_correct']}/{row['stride_total']})\n     Predictions: {row['n_exterior']} ext, {row['n_interior']} int, {row['n_intermediate']} inter")

def main():
    print(f"{'='*80}\nCLASSIFICATION COMBINATION TESTING\n{'='*80}\n")
    results_dir = Path('results')
    csv_files = [f for f in [results_dir / '3pte_results.csv', results_dir / '4d05_results.csv', results_dir / '6wti_results.csv', results_dir / '7upo_results.csv'] if f.exists()]
    if not csv_files:
        print("❌ No result files found"); return
    print(f"Found {len(csv_files)} protein result files:")
    for f in csv_files: print(f"  - {f.name}")
    all_results = []
    for csv_file in csv_files:
        protein_name = csv_file.stem.replace('_results', '')
        output_file = results_dir / f'{protein_name}_classification_combinations.csv'
        print(f"\n{'='*80}\nTESTING PROTEIN: {protein_name.upper()}\n{'='*80}")
        try:
            results_df = test_all_combinations(csv_file, output_file)
            print_summary(results_df, protein_name)
            results_df['protein'] = protein_name
            all_results.append(results_df)
        except Exception as e:
            print(f"❌ Error: {e}")
    if len(all_results) > 1:
        print(f"\n\n{'='*80}\nCOMBINED ANALYSIS ACROSS ALL PROTEINS\n{'='*80}\n")
        combined_df = pd.concat(all_results, ignore_index=True)
        avg_by_combo = combined_df.groupby('combination').agg({'dssp_accuracy': 'mean', 'stride_accuracy': 'mean', 'protein': 'count'}).rename(columns={'protein': 'n_proteins'}).sort_values('dssp_accuracy', ascending=False)
        print("Average DSSP accuracy by combination:\n" + "="*80)
        for combo, row in avg_by_combo.head(10).iterrows():
            print(f"{combo:20s}  Avg DSSP: {row['dssp_accuracy']:.3%}  Avg STRIDE: {row['stride_accuracy']:.3%}  ({int(row['n_proteins'])} proteins)")
        combined_output = results_dir / 'ALL_classification_combinations.csv'
        combined_df.to_csv(combined_output, index=False)
        print(f"\n✅ Combined results saved to: {combined_output}")
    print(f"\n{'='*80}\n✅ ALL TESTING COMPLETE!\n{'='*80}\n")

if __name__ == "__main__":
    main()
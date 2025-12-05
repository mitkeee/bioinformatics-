#!/usr/bin/env python3
"""Manually parse STRIDE file and update IGF1R CSV with STRIDE data."""

from pathlib import Path
import pandas as pd

# Read the STRIDE file
stride_file = Path("/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.stride")
print(f"Reading STRIDE file: {stride_file}")

stride_map = {}
with open(stride_file, 'r') as f:
    for line in f:
        if line.startswith('ASG'):
            try:
                # ASG lines have format:
                # ASG  ALA -  954    1    C          Coil    360.00    156.38      75.6      ~~~~
                chain_id = line[9:10].strip()
                resseq = int(line[11:15].strip())
                ss = line[24:25].strip()
                parts = line.split()
                asa = float(parts[-2]) if len(parts) >= 10 else 0.0

                stride_map[resseq] = {'asa': asa, 'ss': ss}
                print(f"  {resseq}: ASA={asa:.1f} SS={ss}")
            except (ValueError, IndexError) as e:
                print(f"  Error parsing line: {e}")
                continue

print(f"\nExtracted {len(stride_map)} ASG records")

# Now read the CSV and update it
csv_file = Path("/holder/results_dude/detailed_reports/igf1r_detailed_results.csv")
print(f"\nReading CSV: {csv_file}")

df = pd.read_csv(csv_file)
print(f"CSV has {len(df)} rows")

# Add STRIDE columns if they don't exist
if 'stride_asa' not in df.columns:
    df['stride_asa'] = float('nan')
if 'stride_ss' not in df.columns:
    df['stride_ss'] = ''
if 'stride_rasa' not in df.columns:
    df['stride_rasa'] = float('nan')
if 'stride_class' not in df.columns:
    df['stride_class'] = float('nan')

# Max ASA for RASA calculation (Tien et al.)
MAX_ASA = {
    'ALA': 106.0, 'ARG': 248.0, 'ASN': 157.0, 'ASP': 163.0, 'CYS': 135.0,
    'GLN': 198.0, 'GLU': 194.0, 'GLY': 84.0, 'HIS': 194.0, 'ILE': 169.0,
    'LEU': 164.0, 'LYS': 205.0, 'MET': 188.0, 'PHE': 197.0, 'PRO': 136.0,
    'SER': 130.0, 'THR': 142.0, 'TRP': 227.0, 'TYR': 222.0, 'VAL': 142.0
}

# Update rows
updated = 0
for idx, row in df.iterrows():
    resseq = int(row['resseq'])
    if resseq in stride_map:
        df.at[idx, 'stride_asa'] = stride_map[resseq]['asa']
        df.at[idx, 'stride_ss'] = stride_map[resseq]['ss']

        # Calculate RASA
        resname = str(row['resname']).strip().upper()
        max_asa = MAX_ASA.get(resname)
        if max_asa:
            rasa = stride_map[resseq]['asa'] / max_asa
            df.at[idx, 'stride_rasa'] = rasa
            # Classify as exterior if RASA >= 0.25
            df.at[idx, 'stride_class'] = 1 if rasa >= 0.25 else 0
        updated += 1

print(f"\nUpdated {updated} rows with STRIDE data")

# Save the updated CSV
df.to_csv(csv_file, index=False)
print(f"\nSaved updated CSV: {csv_file}")

# Print sample
print("\nFirst 10 rows with STRIDE data:")
for idx in range(min(10, len(df))):
    row = df.iloc[idx]
    asa = f"{row['stride_asa']:.1f}" if pd.notna(row['stride_asa']) else "---"
    ss = row['stride_ss'] if pd.notna(row['stride_ss']) else "-"
    cls = str(int(row['stride_class'])) if pd.notna(row['stride_class']) else "-"
    print(f"  {row['resseq']:4d} {row['resname']:3s}: ASA={asa:7s} SS={ss:1s} CLASS={cls}")

print("\n✓ Complete!")


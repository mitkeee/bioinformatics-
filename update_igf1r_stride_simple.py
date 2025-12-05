#!/usr/bin/env python3
"""Update IGF1R CSV with STRIDE data - pure Python, no pandas."""

import csv

stride_file = "/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.stride"
csv_file = "/holder/results_dude/detailed_reports/igf1r_detailed_results.csv"

# Parse STRIDE file
stride_map = {}
try:
    with open(stride_file, 'r') as f:
        for line in f:
            if line.startswith('ASG'):
                try:
                    resseq = int(line[11:15].strip())
                    ss = line[24:25].strip()
                    parts = line.split()
                    asa = float(parts[-2]) if len(parts) >= 10 else 0.0
                    stride_map[resseq] = (asa, ss)
                except:
                    pass
except Exception as e:
    print(f"Error reading STRIDE: {e}")
    exit(1)

print(f"Read {len(stride_map)} STRIDE records")

# Max ASA values
MAX_ASA = {'ALA': 106.0, 'ARG': 248.0, 'ASN': 157.0, 'ASP': 163.0, 'CYS': 135.0,
    'GLN': 198.0, 'GLU': 194.0, 'GLY': 84.0, 'HIS': 194.0, 'ILE': 169.0,
    'LEU': 164.0, 'LYS': 205.0, 'MET': 188.0, 'PHE': 197.0, 'PRO': 136.0,
    'SER': 130.0, 'THR': 142.0, 'TRP': 227.0, 'TYR': 222.0, 'VAL': 142.0}

# Read and update CSV
try:
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from CSV")

    # Update rows with STRIDE data
    updated = 0
    for row in rows:
        try:
            resseq = int(row['resseq'])
            if resseq in stride_map:
                asa, ss = stride_map[resseq]
                row['stride_asa'] = str(asa)
                row['stride_ss'] = ss

                # Calculate RASA and classify
                resname = row['resname'].strip().upper()
                if resname in MAX_ASA:
                    max_asa = MAX_ASA[resname]
                    rasa = asa / max_asa
                    row['stride_rasa'] = str(rasa)
                    row['stride_class'] = '1' if rasa >= 0.25 else '0'
                    updated += 1
        except:
            pass

    print(f"Updated {updated} rows")

    # Write back
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {csv_file}")
    print("SUCCESS!")

except Exception as e:
    print(f"Error processing CSV: {e}")
    exit(1)


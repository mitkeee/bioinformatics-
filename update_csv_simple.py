#!/usr/bin/env python3
"""
Direct CSV update - reads STRIDE file and updates CSV.
Uses only standard library to avoid dependency issues.
"""

# First, parse STRIDE file
stride_file = "/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.stride"
stride_data = {}

print("Parsing STRIDE file...", flush=True)
with open(stride_file, 'r') as f:
    for line in f:
        if line.startswith('ASG'):
            try:
                parts = line.split()
                # Format: ASG  ALA -  954    1    C          Coil    ...      75.6      ~~~~
                # resseq is parts[3], ASA is parts[-2]
                resseq_str = parts[3]
                resseq = int(resseq_str)
                asa = float(parts[-2])
                ss = parts[5]
                stride_data[resseq] = (asa, ss)
                if len(stride_data) <= 5:
                    print(f"  {resseq}: ASA={asa:.1f} SS={ss}", flush=True)
            except Exception as e:
                print(f"  Error: {e}", flush=True)

print(f"Parsed {len(stride_data)} STRIDE records", flush=True)

# Read CSV
csv_file = "/holder/results_dude/detailed_reports/igf1r_detailed_results.csv"
print(f"\nReading CSV...", flush=True)

lines = []
with open(csv_file, 'r') as f:
    lines = f.readlines()

print(f"Read {len(lines)} lines", flush=True)

# Parse MAX ASA
MAX_ASA = {
    'ALA': 106.0, 'ARG': 248.0, 'ASN': 157.0, 'ASP': 163.0, 'CYS': 135.0,
    'GLN': 198.0, 'GLU': 194.0, 'GLY': 84.0, 'HIS': 194.0, 'ILE': 169.0,
    'LEU': 164.0, 'LYS': 205.0, 'MET': 188.0, 'PHE': 197.0, 'PRO': 136.0,
    'SER': 130.0, 'THR': 142.0, 'TRP': 227.0, 'TYR': 222.0, 'VAL': 142.0
}

# Process CSV
output_lines = []
for i, line in enumerate(lines):
    if i == 0:  # Header
        output_lines.append(line)
        continue

    parts = line.rstrip('\n').split(',')
    if len(parts) < 19:  # Must have all columns
        output_lines.append(line)
        continue

    try:
        resseq = int(parts[1])  # resseq is column 1
        resname = parts[3].strip().upper()  # resname is column 3

        if resseq in stride_data:
            asa, ss = stride_data[resseq]
            parts[14] = str(asa)  # stride_asa column
            parts[15] = ss  # stride_ss column

            # Calculate RASA
            if resname in MAX_ASA:
                max_asa = MAX_ASA[resname]
                rasa = asa / max_asa
                parts[16] = str(rasa)  # stride_rasa column
                parts[17] = '1' if rasa >= 0.25 else '0'  # stride_class column

                if len(stride_data) > 0 and i < 10:  # First 10 rows
                    print(f"  Row {i}: {resseq} {resname} -> ASA={asa:.1f} RASA={rasa:.3f} CLASS={parts[17]}", flush=True)
    except (ValueError, IndexError) as e:
        pass

    output_lines.append(','.join(parts) + '\n')

print(f"\nWriting CSV...", flush=True)
with open(csv_file, 'w') as f:
    f.writelines(output_lines)

print("✓ Complete!", flush=True)


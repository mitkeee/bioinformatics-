#!/usr/bin/env python3
"""
Alternative DSSP extraction using command-line DSSP binary.
This avoids the complex system library configuration issues.
"""

import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# MAX ASA values (Tien et al. Gly-X-Gly model)
MAX_ASA = {
    'ALA': 106.0, 'ARG': 248.0, 'ASN': 157.0, 'ASP': 163.0, 'CYS': 135.0,
    'GLN': 198.0, 'GLU': 194.0, 'GLY': 84.0, 'HIS': 194.0, 'ILE': 169.0,
    'LEU': 164.0, 'LYS': 205.0, 'MET': 188.0, 'PHE': 197.0, 'PRO': 136.0,
    'SER': 130.0, 'THR': 142.0, 'TRP': 227.0, 'TYR': 222.0, 'VAL': 142.0
}

def check_dssp_binary():
    """Check if DSSP binary is available."""
    try:
        result = subprocess.run(['dssp', '--version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def extract_dssp_from_binary(pdb_path):
    """Extract DSSP data using command-line DSSP binary."""
    try:
        pdb_path = Path(pdb_path)

        # Run DSSP command-line tool
        # dssp generates output to stdout
        result = subprocess.run(
            ['dssp', '-i', str(pdb_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return None

        # Parse DSSP output
        dssp_output = result.stdout

        # DSSP output format:
        # Residue # ...chain resseq icode ... SS ASA ...
        dssp_map = {}

        for line in dssp_output.split('\n'):
            # Skip header and comments
            if line.startswith('#') or not line.strip():
                continue

            # Parse line
            try:
                # Format: residue# chain resseq icode aa ss ...
                parts = line.split()
                if len(parts) < 5:
                    continue

                resseq = int(parts[1])
                chain = parts[2] if len(parts) > 2 else ' '
                ss = parts[4] if len(parts) > 4 else 'C'
                asa = float(parts[5]) if len(parts) > 5 else 0.0

                dssp_map[(chain, resseq, '')] = {
                    'asa': asa,
                    'ss': ss if ss != '-' else 'C'
                }

                # Also store common chain ID variations
                dssp_map[(' ', resseq, '')] = {'asa': asa, 'ss': ss if ss != '-' else 'C'}
                dssp_map[('', resseq, '')] = {'asa': asa, 'ss': ss if ss != '-' else 'C'}
                dssp_map[('-', resseq, '')] = {'asa': asa, 'ss': ss if ss != '-' else 'C'}

            except (ValueError, IndexError):
                continue

        return dssp_map if dssp_map else None

    except Exception as e:
        print(f"Error running DSSP binary: {e}")
        return None

def extract_dssp_alternative(pdb_path, df):
    """
    Extract DSSP data using command-line binary.
    If binary fails, returns empty data (shows "No DSSP data available").
    """
    pdb_path = Path(pdb_path) if isinstance(pdb_path, str) else pdb_path

    # Try command-line DSSP
    print(f"    Attempting DSSP extraction via command-line binary...")
    dssp_map = extract_dssp_from_binary(pdb_path)

    if dssp_map is None:
        print(f"    DSSP not available - skipping")
        df['dssp_asa'] = np.nan
        df['dssp_class'] = np.nan
        df['dssp_ss'] = ''
        df['RASA_dssp'] = np.nan
        return df

    # Extract values
    dssp_asa = []
    dssp_ss = []

    for _, row in df.iterrows():
        resseq_int = int(row['resseq'])
        chain = row['chain_id']

        # Try multiple chain ID variations
        possible_keys = [
            (chain, resseq_int, ''),
            (chain.strip(), resseq_int, ''),
            ('', resseq_int, ''),
            (' ', resseq_int, ''),
            ('A', resseq_int, ''),
            ('-', resseq_int, ''),
        ]

        found = False
        for key in possible_keys:
            if key in dssp_map:
                dssp_asa.append(dssp_map[key]['asa'])
                dssp_ss.append(dssp_map[key]['ss'])
                found = True
                break

        if not found:
            dssp_asa.append(np.nan)
            dssp_ss.append('-')

    df['dssp_asa'] = dssp_asa
    df['dssp_ss'] = dssp_ss

    # Calculate RASA
    def _rasa_dssp(row):
        aa = str(row['resname']).strip().upper()
        max_asa = MAX_ASA.get(aa)
        if max_asa is None or pd.isna(row['dssp_asa']):
            return np.nan
        return float(row['dssp_asa']) / max_asa

    df['RASA_dssp'] = df.apply(_rasa_dssp, axis=1)

    # Calculate class (0 = buried, 1 = exposed)
    df['dssp_class'] = df['RASA_dssp'].apply(
        lambda r: 1 if pd.notna(r) and r >= 0.25 else (0 if pd.notna(r) else np.nan)
    )

    return df

if __name__ == "__main__":
    # Test
    if check_dssp_binary():
        print("✓ DSSP binary found!")
    else:
        print("✗ DSSP binary not found")
        print("Install with: conda install -c bioconda dssp")


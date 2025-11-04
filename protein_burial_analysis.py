#!/usr/bin/env python3
"""
Protein Burial Classification Analysis
Processes multiple proteins and generates detailed CSV output with:
- Residue info (ID, number)
- DSSP data (ASA, class, secondary structure)
- STRIDE data (ASA, class, secondary structure)
- Neighbor counts (6Å, 10Å spheres)
- Classification results

Output CSV columns:
res_id, res_num, dssp_asa, dssp_class, stride_asa, stride_class,
ncps_sphere_6, ncps_sphere_6_uni, ncps_sphere_10, ncps_sphere_10_uni,
ncps_class, dssp_ss, stride_ss
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import subprocess
import re

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

# Try to import DSSP
try:
    from Bio.PDB.DSSP import DSSP
    HAS_DSSP = True
except Exception:
    HAS_DSSP = False

# Check if STRIDE is available
HAS_STRIDE = False
try:
    result = subprocess.run(['stride', '-h'], capture_output=True, timeout=2)
    HAS_STRIDE = True
except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
    HAS_STRIDE = False

print(f"DSSP available: {HAS_DSSP}")
print(f"STRIDE available: {HAS_STRIDE}")


# ==============================
# Step 1: Extract CA atoms and calculate distances
# ==============================

def extract_ca_atoms(pdb_path: Path) -> pd.DataFrame:
    """Extract CA atoms with coordinates and residue info."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    ca_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # Skip HETATM
                    continue
                if 'CA' not in residue:
                    continue

                ca = residue['CA']
                ca_list.append({
                    'chain_id': chain.id,
                    'resseq': residue.id[1],
                    'icode': residue.id[2].strip() or '',
                    'resname': residue.resname,
                    'x': ca.coord[0],
                    'y': ca.coord[1],
                    'z': ca.coord[2]
                })
        break  # Only first model

    df = pd.DataFrame(ca_list)
    df['res_num'] = df['resseq']  # Keep original numbering
    df['res_id'] = df['resname']
    return df


def calculate_neighbor_counts(coords: np.ndarray, radius: float) -> np.ndarray:
    """Count neighbors within given radius for each atom."""
    n = len(coords)
    counts = np.zeros(n, dtype=int)

    for i in range(n):
        distances = np.linalg.norm(coords - coords[i], axis=1)
        counts[i] = np.sum((distances > 0) & (distances <= radius))

    return counts


# ==============================
# Step 2: Extract DSSP data
# ==============================

# Maximum ASA values for each amino acid (Tien et al. 2013)
MAX_ASA = {
    'ALA': 121.0, 'ARG': 265.0, 'ASN': 187.0, 'ASP': 187.0,
    'CYS': 148.0, 'GLN': 214.0, 'GLU': 214.0, 'GLY': 97.0,
    'HIS': 216.0, 'ILE': 195.0, 'LEU': 191.0, 'LYS': 230.0,
    'MET': 203.0, 'PHE': 228.0, 'PRO': 154.0, 'SER': 143.0,
    'THR': 163.0, 'TRP': 264.0, 'TYR': 255.0, 'VAL': 165.0
}

def extract_dssp_data(pdb_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Extract DSSP ASA and secondary structure."""
    if not HAS_DSSP:
        df['dssp_asa'] = np.nan
        df['dssp_class'] = np.nan
        df['dssp_ss'] = ''
        return df

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model = structure[0]

        # Determine file type
        file_ext = pdb_path.suffix.lower()
        file_type = 'PDB' if file_ext in ['.pdb', '.ent'] else 'mmCIF'

        dssp = DSSP(model, str(pdb_path), file_type=file_type)

        # Map DSSP data
        dssp_map = {}
        for key in dssp.keys():
            chain_id, res_id = key
            hetflag, resseq, icode = res_id
            rec = dssp[key]

            # DSSP tuple format: (dssp_index, amino_acid, ss, rel_asa, phi, psi, ...)
            # rec[3] is RELATIVE accessibility (0-1), need to convert to absolute Ų
            # Handle 'NA' values from DSSP (for unknown residues like MSE)
            try:
                if rec[3] == 'NA' or rec[3] is None:
                    rel_asa = 0.0  # Default to 0 for NA values
                else:
                    rel_asa = float(rec[3]) if len(rec) > 3 else 0.0
            except (ValueError, TypeError):
                rel_asa = 0.0

            aa = rec[1] if len(rec) > 1 else 'ALA'  # Amino acid
            ss = rec[2] if len(rec) > 2 else '-'  # Secondary structure

            # Convert relative to absolute ASA
            max_asa = MAX_ASA.get(aa, 100.0)
            asa = rel_asa * max_asa

            dssp_map[(chain_id, int(resseq), icode.strip() or '')] = {
                'asa': asa,
                'ss': ss
            }

        # Map to dataframe
        dssp_asa = []
        dssp_ss = []
        for _, row in df.iterrows():
            key = (row['chain_id'], int(row['resseq']), row['icode'])
            if key in dssp_map:
                dssp_asa.append(dssp_map[key]['asa'])
                dssp_ss.append(dssp_map[key]['ss'])
            else:
                dssp_asa.append(np.nan)
                dssp_ss.append('-')

        df['dssp_asa'] = dssp_asa
        df['dssp_ss'] = dssp_ss

        # Classification: 1 = exterior (exposed), 0 = interior (buried)
        # Using 30 Å² threshold (common in literature)
        df['dssp_class'] = df['dssp_asa'].apply(
            lambda x: 1 if pd.notna(x) and x >= 30 else (0 if pd.notna(x) else np.nan)
        )

    except Exception as e:
        print(f"DSSP error: {e}")
        df['dssp_asa'] = np.nan
        df['dssp_class'] = np.nan
        df['dssp_ss'] = ''

    return df


# ==============================
# Step 3: Extract STRIDE data
# ==============================

def extract_stride_data(pdb_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Extract STRIDE ASA and secondary structure from ASG records."""

    try:
        # First try to read from pre-generated .stride file
        # Try multiple case variations
        stride_file = None
        for name_variant in [pdb_path.stem.lower(), pdb_path.stem.upper(), pdb_path.stem]:
            test_file = pdb_path.parent / f"{name_variant}.stride"
            if test_file.exists():
                stride_file = test_file
                break

        if stride_file is not None:
            # Read from existing file
            with open(stride_file, 'r') as f:
                stride_content = f.read()
        elif HAS_STRIDE:
            # Run STRIDE if file doesn't exist
            result = subprocess.run(
                ['stride', str(pdb_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            stride_content = result.stdout
        else:
            # No STRIDE data available
            raise Exception("No STRIDE data available")

        # Parse STRIDE ASG output
        # Format: ASG  RES CHAIN RESNUM RESNUM2 SS SSName Phi Psi Area
        stride_map = {}
        for line in stride_content.split('\n'):
            if line.startswith('ASG'):
                try:
                    # Parse fixed-width format
                    resname = line[5:8].strip()
                    chain_id = line[9:10].strip()
                    resseq = int(line[11:15].strip())
                    ss = line[24:25].strip() if len(line) > 24 else 'C'
                    # Area is typically at the end after phi/psi angles
                    parts = line.split()
                    if len(parts) >= 10:
                        asa = float(parts[-1])
                    else:
                        asa = 0.0

                    stride_map[(chain_id, resseq, '')] = {
                        'asa': asa,
                        'ss': ss if ss else 'C'
                    }
                except (ValueError, IndexError) as e:
                    continue

        # Map to dataframe
        stride_asa = []
        stride_ss = []
        for _, row in df.iterrows():
            key = (row['chain_id'], int(row['resseq']), '')
            if key in stride_map:
                stride_asa.append(stride_map[key]['asa'])
                stride_ss.append(stride_map[key]['ss'])
            else:
                stride_asa.append(np.nan)
                stride_ss.append('-')

        df['stride_asa'] = stride_asa
        df['stride_ss'] = stride_ss

        # Classification: 1 = exterior (exposed), 0 = interior (buried)
        # Using 24 Å² threshold
        df['stride_class'] = df['stride_asa'].apply(
            lambda x: 1 if pd.notna(x) and x >= 24 else (0 if pd.notna(x) else np.nan)
        )

    except Exception as e:
        print(f"STRIDE error: {e}")
        df['stride_asa'] = np.nan
        df['stride_class'] = np.nan
        df['stride_ss'] = ''

    return df


# ==============================
# Step 4: Calculate neighbor-based features
# ==============================

def add_neighbor_features(df: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    """Add neighbor counts and uniformity metrics."""

    # Neighbor counts at 6Å and 10Å
    df['ncps_sphere_6'] = calculate_neighbor_counts(coords, 6.0)
    df['ncps_sphere_10'] = calculate_neighbor_counts(coords, 10.0)

    # Calculate uniformity (spherical variance) at 6Å and 10Å
    df['ncps_sphere_6_uni'] = calculate_uniformity(coords, 6.0)
    df['ncps_sphere_10_uni'] = calculate_uniformity(coords, 10.0)

    return df


def calculate_uniformity(coords: np.ndarray, radius: float) -> np.ndarray:
    """
    Calculate uniformity (spherical variance) for each residue.
    Low value = neighbors on one side (exterior)
    High value = neighbors all around (interior)
    """
    n = len(coords)
    uniformity = np.zeros(n)

    for i in range(n):
        # Find neighbors
        distances = np.linalg.norm(coords - coords[i], axis=1)
        neighbor_mask = (distances > 0) & (distances <= radius)
        neighbor_indices = np.where(neighbor_mask)[0]

        if len(neighbor_indices) < 3:
            uniformity[i] = np.nan
            continue

        # Calculate unit vectors to neighbors
        vectors = coords[neighbor_indices] - coords[i]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / np.maximum(norms, 1e-12)

        # Mean direction
        mean_vector = unit_vectors.mean(axis=0)
        R = np.linalg.norm(mean_vector)

        # Spherical variance: 1 - R
        # High variance = more uniform distribution
        uniformity[i] = 1.0 - R

    return uniformity


# ==============================
# Step 5: Classification (without deg7, simplified)
# ==============================

def classify_residues_simple(df: pd.DataFrame,
                             nc6_threshold: int = 16,
                             nc10_threshold: int = 40,
                             uni6_threshold: float = 0.40,
                             uni10_threshold: float = 0.40) -> pd.DataFrame:
    """
    Classify residues as interior (0) or exterior (1) based on:
    - Neighbor counts at 6Å and 10Å radii
    - Uniformity (spherical variance)
    WITHOUT using deg7

    Logic:
    - Interior (0): Many neighbors AND high uniformity (surrounded from all sides)
    - Exterior (1): Few neighbors OR low uniformity (one-sided, exposed)
    """

    df['ncps_class'] = 0  # Default to interior

    for idx, row in df.iterrows():
        nc6 = row['ncps_sphere_6']
        nc10 = row['ncps_sphere_10']
        uni6 = row['ncps_sphere_6_uni']
        uni10 = row['ncps_sphere_10_uni']

        # Exterior if: few neighbors (below threshold)
        if nc6 < nc6_threshold or nc10 < nc10_threshold:
            df.at[idx, 'ncps_class'] = 1
            continue

        # Exterior if: low uniformity (neighbors not surrounding, one-sided)
        if pd.notna(uni6) and uni6 < uni6_threshold:
            df.at[idx, 'ncps_class'] = 1
            continue

        if pd.notna(uni10) and uni10 < uni10_threshold:
            df.at[idx, 'ncps_class'] = 1
            continue

        # Otherwise: interior (many neighbors + high uniformity)
        df.at[idx, 'ncps_class'] = 0

    return df


# ==============================
# Step 6: Main processing function
# ==============================

def process_protein(pdb_path: Path,
                   nc6_threshold: int = 6,
                   nc10_threshold: int = 12,
                   uni6_threshold: float = 0.30,
                   uni10_threshold: float = 0.60) -> pd.DataFrame:
    """Process a single protein and return results."""

    print(f"\n{'='*80}")
    print(f"Processing: {pdb_path.name}")
    print(f"{'='*80}")

    # Step 1: Extract CA atoms
    df = extract_ca_atoms(pdb_path)
    print(f"Extracted {len(df)} CA atoms")

    # Step 2: Get coordinates
    coords = df[['x', 'y', 'z']].to_numpy(float)

    # Step 3: Extract DSSP data
    df = extract_dssp_data(pdb_path, df)

    # Step 4: Extract STRIDE data
    df = extract_stride_data(pdb_path, df)

    # Step 5: Calculate neighbor features
    df = add_neighbor_features(df, coords)

    # Step 6: Classify residues
    df = classify_residues_simple(df, nc6_threshold, nc10_threshold, uni6_threshold, uni10_threshold)

    # Step 7: Create final output columns in exact order you specified
    output_df = pd.DataFrame({
        'res_id': df['res_id'],
        'res_num': df['res_num'],
        'dssp_asa': df['dssp_asa'],
        'dssp_class': df['dssp_class'],
        'stride_asa': df['stride_asa'],
        'stride_class': df['stride_class'],
        'ncps_sphere_6': df['ncps_sphere_6'],
        'ncps_sphere_6_uni': df['ncps_sphere_6_uni'],
        'ncps_sphere_10': df['ncps_sphere_10'],
        'ncps_sphere_10_uni': df['ncps_sphere_10_uni'],
        'ncps_class': df['ncps_class'],
        'dssp_ss': df['dssp_ss'],
        'stride_ss': df['stride_ss']
    })

    return output_df


# ==============================
# Step 7: Batch processing and optimization
# ==============================

def process_multiple_proteins(pdb_files: List[Path], output_dir: Path = Path("results")):
    """Process multiple proteins and save results."""

    output_dir.mkdir(exist_ok=True)
    all_results = []

    for pdb_file in pdb_files:
        if not pdb_file.exists():
            print(f"⚠️  File not found: {pdb_file}")
            continue

        try:
            result_df = process_protein(pdb_file)

            # Save individual result
            output_file = output_dir / f"{pdb_file.stem}_results.csv"
            result_df.to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file}")

            # Add protein ID column for combined analysis
            result_df['protein'] = pdb_file.stem
            all_results.append(result_df)

            # Print summary
            print_summary(result_df, pdb_file.stem)

        except Exception as e:
            print(f"❌ Error processing {pdb_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = output_dir / "combined_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✅ Combined results saved: {combined_file}")

        # Generate accuracy analysis
        analyze_accuracy(combined_df, output_dir)

    return all_results


def print_summary(df: pd.DataFrame, protein_name: str):
    """Print summary statistics for a protein."""
    print(f"\n📊 Summary for {protein_name}:")
    print(f"   Total residues: {len(df)}")

    if df['dssp_class'].notna().any():
        dssp_ext = (df['dssp_class'] == 1).sum()
        dssp_int = (df['dssp_class'] == 0).sum()
        print(f"   DSSP: {dssp_ext} exterior, {dssp_int} interior")

    if df['stride_class'].notna().any():
        stride_ext = (df['stride_class'] == 1).sum()
        stride_int = (df['stride_class'] == 0).sum()
        print(f"   STRIDE: {stride_ext} exterior, {stride_int} interior")

    ncps_ext = (df['ncps_class'] == 1).sum()
    ncps_int = (df['ncps_class'] == 0).sum()
    print(f"   OUR METHOD: {ncps_ext} exterior, {ncps_int} interior")

    # Agreement with references
    if df['dssp_class'].notna().any():
        agreement = (df['dssp_class'] == df['ncps_class']).sum()
        total = df['dssp_class'].notna().sum()
        acc = agreement / total if total > 0 else 0
        print(f"   Agreement with DSSP: {acc:.1%} ({agreement}/{total})")

    if df['stride_class'].notna().any():
        agreement = (df['stride_class'] == df['ncps_class']).sum()
        total = df['stride_class'].notna().sum()
        acc = agreement / total if total > 0 else 0
        print(f"   Agreement with STRIDE: {acc:.1%} ({agreement}/{total})")


def analyze_accuracy(combined_df: pd.DataFrame, output_dir: Path):
    """Generate detailed accuracy analysis."""

    print(f"\n{'='*80}")
    print("COMBINED ACCURACY ANALYSIS")
    print(f"{'='*80}")

    # Filter to residues with reference data
    df_dssp = combined_df[combined_df['dssp_class'].notna()].copy()
    df_stride = combined_df[combined_df['stride_class'].notna()].copy()

    if len(df_dssp) > 0:
        print("\n📊 DSSP Comparison:")
        dssp_acc = (df_dssp['dssp_class'] == df_dssp['ncps_class']).mean()
        print(f"   Overall accuracy: {dssp_acc:.1%}")

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(df_dssp['dssp_class'], df_dssp['ncps_class'])
        print(f"\n   Confusion Matrix (DSSP):")
        print(f"                  Predicted")
        print(f"                  Int(0)  Ext(1)")
        print(f"   Actual Int(0)    {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"   Actual Ext(1)    {cm[1,0]:4d}   {cm[1,1]:4d}")

    if len(df_stride) > 0:
        print("\n📊 STRIDE Comparison:")
        stride_acc = (df_stride['stride_class'] == df_stride['ncps_class']).mean()
        print(f"   Overall accuracy: {stride_acc:.1%}")

        # Confusion matrix
        cm = confusion_matrix(df_stride['stride_class'], df_stride['ncps_class'])
        print(f"\n   Confusion Matrix (STRIDE):")
        print(f"                  Predicted")
        print(f"                  Int(0)  Ext(1)")
        print(f"   Actual Int(0)    {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"   Actual Ext(1)    {cm[1,0]:4d}   {cm[1,1]:4d}")

    # Save detailed analysis
    analysis_file = output_dir / "accuracy_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("ACCURACY ANALYSIS\n")
        f.write("="*80 + "\n\n")

        if len(df_dssp) > 0:
            f.write(f"DSSP Accuracy: {dssp_acc:.3f}\n")
            f.write(classification_report(df_dssp['dssp_class'], df_dssp['ncps_class'],
                                         target_names=['Interior', 'Exterior']))
            f.write("\n\n")

        if len(df_stride) > 0:
            f.write(f"STRIDE Accuracy: {stride_acc:.3f}\n")
            f.write(classification_report(df_stride['stride_class'], df_stride['ncps_class'],
                                         target_names=['Interior', 'Exterior']))

    print(f"\n✅ Detailed analysis saved: {analysis_file}")


# ==============================
# Main execution
# ==============================

if __name__ == "__main__":

    # Define PDB files to process
    pdb_dir = Path("/Users/famnit/Desktop/pythonProject")
    pdb_files = [
        pdb_dir / "3PTE.pdb",
        pdb_dir / "4d05.pdb",
        pdb_dir / "6wti.pdb",
        pdb_dir / "7upo.pdb"
    ]

    print("="*80)
    print("PROTEIN BURIAL CLASSIFICATION ANALYSIS")
    print("="*80)
    print(f"Processing {len(pdb_files)} proteins...")

    # Process all proteins
    results = process_multiple_proteins(pdb_files)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput files created in 'results/' directory:")
    print("  - Individual CSV files for each protein")
    print("  - combined_results.csv (all proteins together)")
    print("  - accuracy_analysis.txt (detailed metrics)")


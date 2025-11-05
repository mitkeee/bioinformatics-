# extract_ca.py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import subprocess
import tempfile
import re

import numpy as np
import pandas as pd
import networkx as nx
from Bio.PDB import PDBParser

# Try to import DSSP for secondary structure analysis
try:
    from Bio.PDB.DSSP import DSSP
    HAS_DSSP = True
except Exception:
    HAS_DSSP = False

# Check if STRIDE is available on the system
HAS_STRIDE = False
try:
    result = subprocess.run(['stride', '-h'], capture_output=True, timeout=2)
    HAS_STRIDE = True
except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
    HAS_STRIDE = False

# ==============================
# Configuration
# ==============================

# Default path to your local file (you can override when calling run_pipeline())
DEFAULT_PDB_PATH = Path("/Users/famnit/Desktop/pythonProject/3pte.pdb")

# Cutoffs (in Å) - kept for backward compatibility
CUTOFF_GRAPH = 7.0           # for building the graph
SPHERE_RADII = (6.0, 10.0)   # for neighbor counts / Z-scores
HOMOG_RADII = (6.0, 10.0)    # for homogeneity analysis (spherical variance)
#== == == == == == == == == == == == == == ==++++++++++++++++++++++++++++++++++++++++++++
# Classification parameters (MAXIMUM OPTIMIZED for best accuracy!)
# These values achieved 81.0% accuracy on 3pte protein (tested 1296 combinations)
Z_LOW = -1.5       # VERY lenient - maximizes interior detection
Z_HIGH = 0.0       # No positive z-score required - balanced approach
HOMOG_LOW = 0.20   # Very accepting of distributions
HOMOG_HIGH = 0.55  # Moderate-low threshold for interior
#== == == == == == == == == == == == == == ==++++++++++++++++++++++++++++++++++++

# ==============================
# Tunable Parameters (for optimization)
# ==============================

@dataclass
class ClassificationParams:
    """
    Tunable parameters for classification.
    MAXIMUM OPTIMIZED: Achieved 81.0% accuracy on 3pte protein! (tested 1296 combinations)
    """
    # Sphere radii for neighbor counting
    sphere_r1: float = 6.0   # smaller sphere
    sphere_r2: float = 10.0  # larger sphere

    # Z-score thresholds (MAXIMUM OPTIMIZED through expanded grid search)
    z_low: float = -1.5      # VERY lenient - best interior detection
    z_high: float = 0.0      # No positive z-score required - balanced

    # Homogeneity (spherical variance) thresholds (MAXIMUM OPTIMIZED)
    homog_low: float = 0.20   # Very accepting of distributions
    homog_high: float = 0.55  # Moderate-low threshold for interior

    # Graph cutoff (for adjacency matrix)
    graph_cutoff: float = 7.0

    # Optional: use deg7 in classification (DISABLED per request)
    use_degree: bool = False


# Create global default params
DEFAULT_PARAMS = ClassificationParams()


# ==============================
# Step 1 — Extract CA atoms
# ==============================

def extract_ca_atoms(
    pdb_path: Optional[str | Path] = None,
    model_index: int = 0,
    include_hetatm: bool = False,
    include_waters: bool = False,
) -> List[Dict[str, Any]]:
    """
    Read a PDB/.ent file and extract Cα atoms with residue metadata.

    Returns a list of dicts with: chain_id, resseq, icode, resname, atom_name, x, y, z
    """
    pdb_path = Path(pdb_path) if pdb_path is not None else DEFAULT_PDB_PATH
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    try:
        model = structure[model_index]
    except IndexError:
        raise ValueError(f"Model index {model_index} not found. Structure has {len(structure)} model(s).")

    results: List[Dict[str, Any]] = []
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            hetflag, resseq, icode = residue.id  # e.g., (' ', 120, ' ')
            resname = residue.get_resname().strip()

            is_het = (hetflag != ' ')
            is_water = (resname == 'HOH' or hetflag == 'W')
            if not include_hetatm and is_het and not is_water:
                continue
            if not include_waters and is_water:
                continue
            if 'CA' not in residue:
                continue

            ca = residue['CA']
            coord = ca.get_coord()  # numpy array [x, y, z]

            results.append({
                'chain_id': chain_id,
                'resseq': int(resseq),
                'icode': (icode.strip() or ''),
                'resname': resname,
                'atom_name': ca.get_name(),  # 'CA'
                'x': float(coord[0]),
                'y': float(coord[1]),
                'z': float(coord[2]),
            })
    return results


# ==============================
# Step 2 — Utilities
# ==============================

def make_res_label(row: pd.Series) -> str:
    """Create a readable residue label, e.g., A:120 THR (handles insertion codes)."""
    icode = row['icode'] if row['icode'] else ''
    return f"{row['chain_id']}:{row['resseq']}{icode} {row['resname']}"

def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Fast NxN Euclidean distance matrix for Nx3 coords."""
    ss = np.sum(coords**2, axis=1, keepdims=True)
    dist2 = ss + ss.T - 2 * coords @ coords.T
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2)

def adjacency_from_distance(D: np.ndarray, cutoff: float) -> np.ndarray:
    """Binary adjacency (0/1) where A[i,j] = 1 if D[i,j] <= cutoff and i != j."""
    A = (D <= cutoff).astype(int)
    np.fill_diagonal(A, 0)
    return A

def add_neighbor_counts(df_ca: pd.DataFrame, D: np.ndarray, radii=(6.0, 10.0)) -> pd.DataFrame:
    """Add degree counts within each radius as new columns (deg_6A, deg_10A, ...)."""
    out = df_ca.copy()
    for r in radii:
        A = adjacency_from_distance(D, cutoff=r)
        out[f'deg_{int(r)}A'] = A.sum(axis=1)
    return out

def zscore(series: pd.Series) -> pd.Series:
    """Z-score normalization (mean 0, std 1); if std=0 return zeros."""
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sd


# ==============================
# Step 3 — Graph metrics
# ==============================

@dataclass
class GraphSummary:
    n_nodes: int
    n_edges: int
    n_components: int
    component_sizes: list
    component_diameters: list
    component_radii: list

def graph_metrics_from_adjacency(A: np.ndarray) -> tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int], GraphSummary]:
    """
    Build an undirected graph from adjacency and compute:
      - degree per node
      - eccentricity per node (#edges in shortest path)
      - per-component radius and diameter, mapped back to each node
      - component id per node (0..k-1)
    """
    G = nx.from_numpy_array(A)  # nodes are 0..N-1
    degrees = dict(G.degree())

    # Connected components
    components = list(nx.connected_components(G))
    comp_id: Dict[int, int] = {}
    ecc_all: Dict[int, int] = {}
    comp_radius_per_node: Dict[int, int] = {}
    comp_diameter_per_node: Dict[int, int] = {}

    comp_sizes = []
    comp_diams = []
    comp_rads = []

    for cid, nodes in enumerate(components):
        nodes = sorted(nodes)
        comp_sizes.append(len(nodes))
        H = G.subgraph(nodes).copy()
        ecc = nx.eccentricity(H)  # dict: node -> ecc
        rad = nx.radius(H)
        diam = nx.diameter(H)

        comp_diams.append(diam)
        comp_rads.append(rad)

        for n in nodes:
            comp_id[n] = cid
            ecc_all[n] = ecc[n]
            comp_radius_per_node[n] = rad
            comp_diameter_per_node[n] = diam

    summary = GraphSummary(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_components=len(components),
        component_sizes=comp_sizes,
        component_diameters=comp_diams,
        component_radii=comp_rads,
    )
    return degrees, ecc_all, comp_id, comp_radius_per_node, comp_diameter_per_node, summary


# ==============================
# Step 4 — Homogeneity (optional but useful)
# ==============================

def spherical_variance_for_neighbors(coords: np.ndarray, radius: float) -> np.ndarray:
    """
    For each residue (row in coords), compute spherical variance of neighbor directions within 'radius'.
    - For center i, take neighbors j within radius (excluding i).
    - Compute unit vectors v_ij = (coords[j] - coords[i]) / ||...||
    - Compute mean resultant length R = ||mean(v_ij)|| (0..1)
    - Spherical variance SV = 1 - R (0..1). Higher SV => more uniform (homogeneous) distribution.
      Low SV => vectors point in similar direction (neighbors on one side) -> likely surface.
    Returns: array of SV per node.
    """
    N = coords.shape[0]
    D = pairwise_distance_matrix(coords)
    SV = np.zeros(N, dtype=float)
    for i in range(N):
        mask = (D[i] > 0) & (D[i] <= radius)
        idx = np.where(mask)[0]
        if idx.size < 3:
            SV[i] = np.nan  # not enough neighbors to judge
            continue
        vecs = coords[idx] - coords[i]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / np.maximum(norms, 1e-12)
        mean_vec = unit.mean(axis=0)
        R = float(np.linalg.norm(mean_vec))  # 0..1
        SV[i] = 1.0 - R
    return SV


# ==============================
# Step 5 — Classification
# ==============================

def classify_residues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify residues into 'interior', 'exterior', or 'intermediate' using:
      - Z-scores of neighbor counts at 6 Å and 10 Å
      - Optional spherical variance (homogeneity) at 6/10 Å
      - NOTE: deg7 removed from classification per request

    Heuristic decision:
      - EXTERIOR if any (z_6A <= Z_LOW) OR (z_10A <= Z_LOW) OR (sph_var low)
      - INTERIOR if (z_6A >= Z_HIGH AND z_10A >= Z_HIGH) OR (sph_var high)
      - otherwise INTERMEDIATE
    """
    out = df.copy()

    # Evaluate homogeneity if present
    use_homog = False
    sv_cols = []
    for r in HOMOG_RADII:
        c = f'sph_var_{int(r)}A'
        if c in out.columns:
            sv_cols.append(c)
            use_homog = True

    labels = []
    reasons = []
    for i, row in out.iterrows():
        z6 = row.get('z_6A', 0.0)
        z10 = row.get('z_10A', 0.0)
        deg7 = row.get('degree_7A', 0)

        # Homog signals
        sv_low_flag = False
        sv_high_flag = False
        if use_homog:
            vals = [row[c] for c in sv_cols if pd.notna(row[c])]
            if len(vals) > 0:
                sv_mean = float(np.mean(vals))
                sv_low_flag = sv_mean < HOMOG_LOW
                sv_high_flag = sv_mean > HOMOG_HIGH
#------------------------------------------------------------------------------------------------
        is_exterior = (z6 <= Z_LOW) and (z10 <= Z_LOW) or sv_low_flag # OLD is_exterior = (z6 <= Z_LOW) or (z10 <= Z_LOW) or sv_low_flag
        is_interior = (z6 >= Z_HIGH and z10 >= Z_HIGH) or sv_high_flag
#------------------------------------------------------------------------------------------------
        if is_exterior and not is_interior:
            labels.append('exterior')
            reasons.append(f"low Z{', low homog' if sv_low_flag else ''}")
        elif is_interior and not is_exterior:
            labels.append('interior')
            reasons.append(f"high Z{', high homog' if sv_high_flag else ''}")
        else:
            labels.append('intermediate')
            reasons.append('mixed signals')

    out['burial_label'] = labels
    out['burial_reason'] = reasons
    return out


def classify_residues_with_params(df: pd.DataFrame, params: ClassificationParams) -> pd.DataFrame:
    """
    Classify residues using custom parameters (for parameter optimization).
    EXCLUDES deg7 from classification (as requested).
    """
    out = df.copy()

    # Evaluate homogeneity if present
    use_homog = False
    sv_cols = []
    for r in [params.sphere_r1, params.sphere_r2]:
        c = f'sph_var_{int(r)}A'
        if c in out.columns:
            sv_cols.append(c)
            use_homog = True

    labels = []
    reasons = []
    for i, row in out.iterrows():
        z6 = row.get('z_6A', 0.0)
        z10 = row.get('z_10A', 0.0)

        # Homog signals
        sv_low_flag = False
        sv_high_flag = False
        if use_homog:
            vals = [row[c] for c in sv_cols if pd.notna(row[c])]
            if len(vals) > 0:
                sv_mean = float(np.mean(vals))
                sv_low_flag = sv_mean < params.homog_low
                sv_high_flag = sv_mean > params.homog_high

        # Classification without deg7
        is_exterior = (z6 <= params.z_low) or (z10 <= params.z_low) or sv_low_flag
        is_interior = (z6 >= params.z_high and z10 >= params.z_high) or sv_high_flag

        if is_exterior and not is_interior:
            labels.append('exterior')
            reasons.append(f"low Z{', low homog' if sv_low_flag else ''}")
        elif is_interior and not is_exterior:
            labels.append('interior')
            reasons.append(f"high Z{', high homog' if sv_high_flag else ''}")
        else:
            labels.append('intermediate')
            reasons.append('mixed signals')

    out['burial_label'] = labels
    out['burial_reason'] = reasons
    return out


# ==============================
# Step 5b — Parameter optimization
# ==============================

def optimize_parameters_against_reference(
    df: pd.DataFrame,
    reference_col: str = 'dssp_label',  # or 'stride_label'
    param_ranges: Optional[Dict] = None
) -> ClassificationParams:
    """
    Grid search to find best parameters that maximize agreement with DSSP/STRIDE.

    param_ranges example:
    {
        'z_low': [-1.0, -0.5, 0.0],
        'z_high': [0.0, 0.5, 1.0],
        'homog_low': [0.25, 0.35, 0.45],
        'homog_high': [0.55, 0.65, 0.75]
    }
    """
    if reference_col not in df.columns or df[reference_col].isna().all():
        print(f"No {reference_col} available for optimization.")
        return DEFAULT_PARAMS

    if param_ranges is None:
        # EXPANDED search space with finer granularity - testing 1296 combinations!
        param_ranges = {
            'z_low': [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25],
            'z_high': [0.0, 0.15, 0.25, 0.35, 0.5, 0.75],
            'homog_low': [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            'homog_high': [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        }

    best_accuracy = 0.0
    best_params = DEFAULT_PARAMS

    print(f"\n=== Optimizing parameters against {reference_col} ===")
    print("This may take a minute...")

    import itertools
    total_combinations = int(np.prod([len(v) for v in param_ranges.values()]))
    print(f"Testing {total_combinations} parameter combinations...")

    # Grid search
    keys = list(param_ranges.keys())
    for values in itertools.product(*[param_ranges[k] for k in keys]):
        test_params = ClassificationParams()
        for k, v in zip(keys, values):
            setattr(test_params, k, v)

        # Reclassify with test parameters
        df_test = classify_residues_with_params(df.copy(), test_params)

        # Calculate accuracy
        df_eval = df_test[df_test[reference_col].notna()].copy()
        df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})

        accuracy = (df_eval[reference_col] == df_eval['burial_label']).mean()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = test_params

    print(f"\nBest parameters found (accuracy: {best_accuracy:.3f}):")
    print(f"  z_low: {best_params.z_low}")
    print(f"  z_high: {best_params.z_high}")
    print(f"  homog_low: {best_params.homog_low}")
    print(f"  homog_high: {best_params.homog_high}")

    return best_params


# ==============================
# Step 6 — DSSP solvent accessibility
# ==============================

def add_dssp_acc_and_rsa(
    df: pd.DataFrame,
    pdb_path: Path,
    model_index: int = 0,
) -> pd.DataFrame:
    """
    Adds DSSP solvent accessibility (ACC) and a relative percentage:
      rsa_rel_to_max = ACC / max(ACC in this protein).
    Labels:
      dssp_label = 'interior' if rsa_rel_to_max < 0.5 else 'exterior'
    NOTE: This follows your class heuristic (NOT canonical per-residue RSA).
    """
    if not HAS_DSSP:
        df = df.copy()
        df['dssp_acc'] = np.nan
        df['dssp_rsa_rel_to_max'] = np.nan
        df['dssp_label'] = pd.Series([None] * len(df), dtype=object)
        return df

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    try:
        model = structure[model_index]
    except IndexError:
        raise ValueError(f"Model index {model_index} not found. Structure has {len(structure)} model(s).")

    # Determine file type for DSSP
    file_ext = pdb_path.suffix.lower()
    if file_ext in ['.pdb', '.ent']:
        file_type = 'PDB'
    elif file_ext in ['.cif', '.mmcif']:
        file_type = 'mmCIF'
    else:
        file_type = 'PDB'  # default

    dssp = DSSP(model, str(pdb_path), file_type=file_type)

    # Map (chain_id, resseq, icode) -> ACC
    acc_map = {}
    for key in dssp.keys():
        try:
            chain_id, res_id = key
            hetflag, resseq, icode = res_id  # (' ', 42, ' ')
        except Exception:
            continue
        rec = dssp[key]
        try:
            acc = float(rec[3])  # ACC is usually at index 3 in DSSP tuple
        except Exception:
            try:
                acc = float(rec.get('ACC', np.nan))  # new dict-like API
            except Exception:
                acc = np.nan
        acc_map[(chain_id, int(resseq), (icode.strip() or ''))] = acc

    acc_vals = []
    for _, r in df.iterrows():
        acc_vals.append(acc_map.get((r['chain_id'], int(r['resseq']), r['icode']), np.nan))

    out = df.copy()
    out['dssp_acc'] = pd.to_numeric(pd.Series(acc_vals), errors='coerce')

    if out['dssp_acc'].notna().any():
        max_acc = out['dssp_acc'].max(skipna=True)
        out['dssp_rsa_rel_to_max'] = out['dssp_acc'] / max_acc if (max_acc and max_acc > 0) else np.nan
        out['dssp_label'] = out['dssp_rsa_rel_to_max'].apply(
            lambda v: ('interior' if pd.notna(v) and v < 0.5 else ('exterior' if pd.notna(v) else None))
        )
    else:
        out['dssp_rsa_rel_to_max'] = np.nan
        out['dssp_label'] = pd.Series([None] * len(out), dtype=object)

    return out


def evaluate_against_dssp(df: pd.DataFrame, treat_intermediate_as='exterior'):
    """
    Show YOUR predictions vs. DSSP in the simplest possible way.
    """
    if 'dssp_label' not in df.columns or df['dssp_label'].isna().all():
        print("⚠️  No DSSP labels available. Ensure mkdssp is installed and do_dssp=True.")
        return

    df_eval = df[df['dssp_label'].notna()].copy()
    if treat_intermediate_as == 'drop':
        df_eval = df_eval[df_eval['burial_label'].isin(['interior', 'exterior'])]
    else:
        df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})

    y_dssp = df_eval['dssp_label']     # What DSSP says
    y_yours = df_eval['burial_label']  # What YOUR algorithm says

    print("\n" + "="*80)
    print("              COMPARING: YOUR ALGORITHM  ←→  DSSP (Reference)")
    print("="*80)

    # Simple counts
    print("\n📊 WHAT EACH METHOD PREDICTS:")
    print("-"*80)
    print(f"YOUR algorithm:  {(y_yours == 'exterior').sum():>3} exterior  +  {(y_yours == 'interior').sum():>3} interior  = {len(y_yours)} total")
    print(f"DSSP reference:  {(y_dssp == 'exterior').sum():>3} exterior  +  {(y_dssp == 'interior').sum():>3} interior  = {len(y_dssp)} total")

    # Agreement
    agreement = (y_yours == y_dssp).sum()
    disagreement = (y_yours != y_dssp).sum()
    accuracy = agreement / len(y_yours)

    print("\n✓ AGREEMENT:")
    print("-"*80)
    print(f"Both agree on {agreement} residues ({100*accuracy:.1f}%)")
    print(f"They disagree on {disagreement} residues ({100*(1-accuracy):.1f}%)")

    # Show WHERE they disagree
    print("\n✗ DISAGREEMENTS (where YOUR algorithm differs from DSSP):")
    print("-"*80)

    # Type 1: You say exterior, DSSP says interior
    your_ext_dssp_int = ((y_yours == 'exterior') & (y_dssp == 'interior')).sum()
    # Type 2: You say interior, DSSP says exterior
    your_int_dssp_ext = ((y_yours == 'interior') & (y_dssp == 'exterior')).sum()

    print(f"Type 1: YOU say 'exterior' but DSSP says 'interior'  →  {your_ext_dssp_int} cases")
    print(f"Type 2: YOU say 'interior' but DSSP says 'exterior'  →  {your_int_dssp_ext} cases")

    # Interpretation
    print("\n💡 INTERPRETATION:")
    print("-"*80)
    if your_ext_dssp_int > your_int_dssp_ext * 2:
        print("❌ Problem: Your algorithm is TOO CONSERVATIVE")
        print("   You're calling many things 'exterior' that DSSP thinks are 'interior'")
        print("   → Try LOWERING z_low (e.g., -0.8) to make it easier to call interior")
        print("   → Or LOWERING homog_low (e.g., 0.25) to accept less uniform distributions")
    elif your_int_dssp_ext > your_ext_dssp_int * 2:
        print("❌ Problem: Your algorithm is TOO AGGRESSIVE")
        print("   You're calling many things 'interior' that DSSP thinks are 'exterior'")
        print("   → Try RAISING z_high (e.g., 0.8) to be more strict about interior")
        print("   → Or RAISING homog_high (e.g., 0.75) to require more uniform distributions")
    else:
        print("⚖️  Balanced: Your errors are roughly equal in both directions")
        print("   → Fine-tune both thresholds slightly")

    # Simple visual comparison
    print("\n📋 VISUAL COMPARISON:")
    print("-"*80)

    ext_ext = ((y_yours == 'exterior') & (y_dssp == 'exterior')).sum()
    ext_int = ((y_yours == 'interior') & (y_dssp == 'exterior')).sum()
    int_ext = ((y_yours == 'exterior') & (y_dssp == 'interior')).sum()
    int_int = ((y_yours == 'interior') & (y_dssp == 'interior')).sum()

    print("When DSSP says 'EXTERIOR':")
    print(f"  ✓ YOU also say 'exterior': {ext_ext} residues (CORRECT)")
    print(f"  ✗ YOU say 'interior':      {ext_int} residues (WRONG)")
    print()
    print("When DSSP says 'INTERIOR':")
    print(f"  ✗ YOU say 'exterior':      {int_ext} residues (WRONG) ← Main problem!")
    print(f"  ✓ YOU also say 'interior': {int_int} residues (CORRECT)")

    print("\n" + "="*80)
    print(f"ACCURACY: {100*accuracy:.1f}%  ({agreement} correct out of {len(y_yours)} total)")
    print("="*80)



def add_stride_acc_and_rsa(
    df: pd.DataFrame,
    pdb_path: Path,
) -> pd.DataFrame:
    """
    Runs STRIDE (external program) to get solvent accessibility (ACC).
    Similar to DSSP, calculates RSA as percentage of max ACC.
    Labels: stride_label = 'interior' if rsa < 0.5 else 'exterior'

    NOTE: STRIDE must be installed on your system.
    """
    if not HAS_STRIDE:
        df = df.copy()
        df['stride_acc'] = np.nan
        df['stride_rsa_rel_to_max'] = np.nan
        df['stride_label'] = pd.Series([None] * len(df), dtype=object)
        return df

    try:
        # Run STRIDE on the PDB file
        result = subprocess.run(
            ['stride', str(pdb_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"STRIDE failed with return code {result.returncode}")
            df = df.copy()
            df['stride_acc'] = np.nan
            df['stride_rsa_rel_to_max'] = np.nan
            df['stride_label'] = pd.Series([None] * len(df), dtype=object)
            return df

        # Parse STRIDE output
        # ASG lines contain: resname, chain, resseq, icode, secondary structure, etc.
        # ACC lines contain: accessible surface area
        acc_map = {}

        for line in result.stdout.split('\n'):
            if line.startswith('ASG'):
                # Format: ASG  RES_NAME CHAIN RES_NUM ICODE SS ... ACC
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        resname = parts[1]
                        chain_id = parts[2]
                        resseq = int(parts[3])
                        # STRIDE format may not have explicit icode field in same way
                        # Usually it's part of residue number
                        icode = ''
                        # ACC is typically in column 9 or 10
                        acc = float(parts[9]) if len(parts) > 9 else 0.0
                        acc_map[(chain_id, resseq, icode)] = acc
                    except (ValueError, IndexError):
                        continue

        # Map ACC values to dataframe
        acc_vals = []
        for _, r in df.iterrows():
            acc_vals.append(acc_map.get((r['chain_id'], int(r['resseq']), r['icode']), np.nan))

        out = df.copy()
        out['stride_acc'] = pd.to_numeric(pd.Series(acc_vals), errors='coerce')

        if out['stride_acc'].notna().any():
            max_acc = out['stride_acc'].max(skipna=True)
            out['stride_rsa_rel_to_max'] = out['stride_acc'] / max_acc if (max_acc and max_acc > 0) else np.nan
            out['stride_label'] = out['stride_rsa_rel_to_max'].apply(
                lambda v: ('interior' if pd.notna(v) and v < 0.5 else ('exterior' if pd.notna(v) else None))
            )
        else:
            out['stride_rsa_rel_to_max'] = np.nan
            out['stride_label'] = pd.Series([None] * len(out), dtype=object)

        return out

    except Exception as e:
        print(f"Error running STRIDE: {e}")
        df = df.copy()
        df['stride_acc'] = np.nan
        df['stride_rsa_rel_to_max'] = np.nan
        df['stride_label'] = pd.Series([None] * len(df), dtype=object)
        return df


def evaluate_against_stride(df: pd.DataFrame, treat_intermediate_as='exterior'):
    """
    Show YOUR predictions vs. STRIDE in the simplest possible way.
    """
    if 'stride_label' not in df.columns or df['stride_label'].isna().all():
        print("⚠️  No STRIDE labels available. Ensure STRIDE is installed and do_stride=True.")
        return

    df_eval = df[df['stride_label'].notna()].copy()
    if treat_intermediate_as == 'drop':
        df_eval = df_eval[df_eval['burial_label'].isin(['interior', 'exterior'])]
    else:
        df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})

    y_stride = df_eval['stride_label']    # What STRIDE says
    y_yours = df_eval['burial_label']    # What YOUR algorithm says

    print("\n" + "="*80)
    print("              COMPARING: YOUR ALGORITHM  ←→  STRIDE (Reference)")
    print("="*80)

    # Simple counts
    print("\n📊 WHAT EACH METHOD PREDICTS:")
    print("-"*80)
    print(f"YOUR algorithm:    {(y_yours == 'exterior').sum():>3} exterior  +  {(y_yours == 'interior').sum():>3} interior  = {len(y_yours)} total")
    print(f"STRIDE reference:  {(y_stride == 'exterior').sum():>3} exterior  +  {(y_stride == 'interior').sum():>3} interior  = {len(y_stride)} total")

    # Agreement
    agreement = (y_yours == y_stride).sum()
    disagreement = (y_yours != y_stride).sum()
    accuracy = agreement / len(y_yours)

    print("\n✓ AGREEMENT:")
    print("-"*80)
    print(f"Both agree on {agreement} residues ({100*accuracy:.1f}%)")
    print(f"They disagree on {disagreement} residues ({100*(1-accuracy):.1f}%)")

    # Show WHERE they disagree
    print("\n✗ DISAGREEMENTS (where YOUR algorithm differs from STRIDE):")
    print("-"*80)

    your_ext_stride_int = ((y_yours == 'exterior') & (y_stride == 'interior')).sum()
    your_int_stride_ext = ((y_yours == 'interior') & (y_stride == 'exterior')).sum()

    print(f"Type 1: YOU say 'exterior' but STRIDE says 'interior'  →  {your_ext_stride_int} cases")
    print(f"Type 2: YOU say 'interior' but STRIDE says 'exterior'  →  {your_int_stride_ext} cases")

    # Simple visual comparison
    print("\n📋 VISUAL COMPARISON:")
    print("-"*80)

    ext_ext = ((y_yours == 'exterior') & (y_stride == 'exterior')).sum()
    ext_int = ((y_yours == 'interior') & (y_stride == 'exterior')).sum()
    int_ext = ((y_yours == 'exterior') & (y_stride == 'interior')).sum()
    int_int = ((y_yours == 'interior') & (y_stride == 'interior')).sum()

    print("When STRIDE says 'EXTERIOR':")
    print(f"  ✓ YOU also say 'exterior': {ext_ext} residues (CORRECT)")
    print(f"  ✗ YOU say 'interior':      {ext_int} residues (WRONG)")
    print()
    print("When STRIDE says 'INTERIOR':")
    print(f"  ✗ YOU say 'exterior':      {int_ext} residues (WRONG) ← Main problem!")
    print(f"  ✓ YOU also say 'interior': {int_int} residues (CORRECT)")

    print("\n" + "="*80)
    print(f"ACCURACY: {100*accuracy:.1f}%  ({agreement} correct out of {len(y_yours)} total)")
    print("="*80)



def compare_dssp_vs_stride(df: pd.DataFrame):
    """
    Compare the two reference methods (DSSP vs STRIDE) - simplified.
    This shows if the reference methods themselves agree.
    """
    if 'dssp_label' not in df.columns or 'stride_label' not in df.columns:
        print("⚠️  Need both DSSP and STRIDE labels to compare them.")
        return

    df_both = df[(df['dssp_label'].notna()) & (df['stride_label'].notna())].copy()
    if len(df_both) == 0:
        print("⚠️  No residues have both DSSP and STRIDE labels.")
        return

    y_dssp = df_both['dssp_label']
    y_stride = df_both['stride_label']

    print("\n" + "="*80)
    print("         REFERENCE METHODS COMPARISON: DSSP  ←→  STRIDE")
    print("="*80)
    print("(This shows how well the two reference methods agree with each other)")

    # Counts
    print("\n📊 PREDICTIONS:")
    print("-"*80)
    print(f"DSSP:    {(y_dssp == 'exterior').sum():>3} exterior  +  {(y_dssp == 'interior').sum():>3} interior  = {len(y_dssp)} total")
    print(f"STRIDE:  {(y_stride == 'exterior').sum():>3} exterior  +  {(y_stride == 'interior').sum():>3} interior  = {len(y_stride)} total")

    # Agreement
    agreement = (y_dssp == y_stride).sum()
    accuracy = agreement / len(y_dssp)

    print("\n✓ AGREEMENT BETWEEN DSSP AND STRIDE:")
    print("-"*80)
    print(f"{agreement}/{len(y_dssp)} residues = {100*accuracy:.1f}% agreement")

    if accuracy > 0.9:
        print("✅ Excellent! The two reference methods strongly agree.")
        print("   This means they are reliable references for validating YOUR algorithm.")
    elif accuracy > 0.8:
        print("👍 Good agreement. The references are mostly consistent.")
    else:
        print("⚠️  Moderate agreement. The references themselves disagree significantly.")

    # Simple comparison
    ext_ext = ((y_dssp == 'exterior') & (y_stride == 'exterior')).sum()
    ext_int = ((y_dssp == 'exterior') & (y_stride == 'interior')).sum()
    int_ext = ((y_dssp == 'interior') & (y_stride == 'exterior')).sum()
    int_int = ((y_dssp == 'interior') & (y_stride == 'interior')).sum()

    print("\n📋 WHERE THEY AGREE/DISAGREE:")
    print("-"*80)
    print(f"Both say 'exterior': {ext_ext} residues")
    print(f"Both say 'interior': {int_int} residues")
    print(f"DSSP='exterior', STRIDE='interior': {ext_int} residues")
    print(f"DSSP='interior', STRIDE='exterior': {int_ext} residues")

    print("="*80)


def show_example_predictions(df: pd.DataFrame, n_examples: int = 10):
    """
    Show side-by-side examples in the SIMPLEST way possible.
    """
    print("\n" + "="*80)
    print(f"EXAMPLE RESIDUES: YOUR predictions vs. reference methods")
    print("="*80)
    print("(Showing first 10 residues)")
    print()

    # Filter to residues with reference data
    df_with_ref = df[df['dssp_label'].notna()].copy()

    if len(df_with_ref) == 0:
        print("⚠️  No residues with reference data to show.")
        return

    print("📖 COLUMN MEANINGS:")
    print("   Residue    = Amino acid name (e.g., A:50 LEU = chain A, position 50, Leucine)")
    print("   YOU        = What YOUR algorithm predicts")
    print("   DSSP       = What DSSP reference says (✓ = match, ✗ = disagree)")
    print("   STRIDE     = What STRIDE reference says (✓ = match, ✗ = disagree)")
    print("   z_6A       = Neighbor count at 6Å (negative = fewer neighbors = likely exterior)")
    print("   z_10A      = Neighbor count at 10Å (negative = fewer neighbors = likely exterior)")
    print("   homog      = Homogeneity 0-1 (low = one-sided, high = surrounded)")
    print()
    print("-"*80)

    # Build simple display table
    for idx, row in df_with_ref.head(n_examples).iterrows():
        you = row['burial_label']
        dssp = row.get('dssp_label', '?')
        stride = row.get('stride_label', '?')

        # Mark agreement/disagreement
        match_dssp = "✓" if you.replace('intermediate', 'exterior') == dssp else "✗"
        match_stride = "✓" if you.replace('intermediate', 'exterior') == stride else "✗"

        # Shorten labels for display
        you_short = you[:3].upper()  # EXT or INT
        dssp_short = dssp[:3].upper() if dssp else '?'
        stride_short = stride[:3].upper() if stride else '?'

        z6 = row.get('z_6A', 0)
        z10 = row.get('z_10A', 0)
        homog = row.get('sph_var_6A', np.nan)
        homog_str = f"{homog:.2f}" if pd.notna(homog) else "n/a"

        print(f"{row['res_label']:<15}  YOU:{you_short}  {match_dssp} DSSP:{dssp_short}  {match_stride} STRIDE:{stride_short}  │  z6={z6:>6.2f}  z10={z10:>6.2f}  homog={homog_str:>5}")

    # Show some disagreements
    print("\n" + "-"*80)
    print("❌ DISAGREEMENTS (where YOU differ from DSSP):")
    print("-"*80)

    disagreements = df_with_ref[df_with_ref['burial_label'].replace({'intermediate': 'exterior'}) != df_with_ref['dssp_label']]
    if len(disagreements) > 0:
        print("Why might these be wrong? Look at the z-scores and homogeneity:")
        print()
        for idx, row in disagreements.head(min(5, len(disagreements))).iterrows():
            you = row['burial_label']
            dssp = row['dssp_label']
            z6 = row.get('z_6A', 0)
            z10 = row.get('z_10A', 0)
            homog = row.get('sph_var_6A', np.nan)
            homog_str = f"{homog:.2f}" if pd.notna(homog) else "n/a"

            print(f"{row['res_label']:<15}  YOU:{you:<12}  DSSP:{dssp:<8}  │  z6={z6:>6.2f}  z10={z10:>6.2f}  homog={homog_str:>5}")
    else:
        print("✅ No disagreements found! Perfect match!")

    print("="*80)


# ==============================
# Step 6b — Visualization: Environment around specific amino acid
# ==============================

def visualize_amino_acid_environment(
    df: pd.DataFrame,
    residue_index: int,
    sphere_radius: float = 6.0,
    save_path: Optional[Path] = None
):
    """
    Creates a 3D visualization showing:
    - Target amino acid at center
    - Neighboring atoms within sphere
    - Vectors pointing to neighbors
    - Empty space detection

    This helps understand why an amino acid is classified as interior/exterior.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return

    # Get target residue
    target = df.iloc[residue_index]
    target_coords = np.array([target['x'], target['y'], target['z']])

    # Get all coordinates
    all_coords = df[['x', 'y', 'z']].to_numpy(float)

    # Find neighbors within sphere
    distances = np.linalg.norm(all_coords - target_coords, axis=1)
    neighbor_mask = (distances > 0) & (distances <= sphere_radius)
    neighbor_indices = np.where(neighbor_mask)[0]
    neighbor_coords = all_coords[neighbor_indices]

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot target residue (large red sphere)
    ax.scatter(*target_coords, c='red', s=200, marker='o',
               label=f'Target: {target["res_label"]} ({target["burial_label"]})')

    # Plot neighbors (blue spheres)
    if len(neighbor_coords) > 0:
        ax.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1], neighbor_coords[:, 2],
                   c='blue', s=50, alpha=0.6, label=f'Neighbors (n={len(neighbor_coords)})')

        # Draw vectors from target to neighbors
        for nc in neighbor_coords:
            ax.plot([target_coords[0], nc[0]],
                    [target_coords[1], nc[1]],
                    [target_coords[2], nc[2]],
                    'gray', alpha=0.3, linewidth=0.5)

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v)) + target_coords[0]
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v)) + target_coords[1]
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + target_coords[2]
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='green', alpha=0.1, linewidth=0.5)

    # Calculate homogeneity (spherical variance)
    if len(neighbor_coords) >= 3:
        vectors = neighbor_coords - target_coords
        unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        mean_vector = unit_vectors.mean(axis=0)
        R = np.linalg.norm(mean_vector)
        sv = 1.0 - R

        # Draw mean vector (shows which side neighbors are on)
        mean_vec_scaled = mean_vector * sphere_radius * 0.7
        ax.quiver(target_coords[0], target_coords[1], target_coords[2],
                  mean_vec_scaled[0], mean_vec_scaled[1], mean_vec_scaled[2],
                  color='orange', arrow_length_ratio=0.2, linewidth=2,
                  label=f'Mean direction (SV={sv:.2f})')

    # Labels and formatting
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Environment of {target["res_label"]}\n'
                 f'Sphere radius: {sphere_radius}Å, '
                 f'Z-scores: z6={target.get("z_6A", 0):.2f}, z10={target.get("z_10A", 0):.2f}')
    ax.legend(loc='upper left')

    # Equal aspect ratio
    max_range = sphere_radius * 1.2
    ax.set_xlim([target_coords[0] - max_range, target_coords[0] + max_range])
    ax.set_ylim([target_coords[1] - max_range, target_coords[1] + max_range])
    ax.set_zlim([target_coords[2] - max_range, target_coords[2] + max_range])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_all_interesting_residues(df: pd.DataFrame, output_dir: Path = Path("visualizations")):
    """
    Automatically visualizes:
    - Most interior residue (highest z10)
    - Most exterior residue (lowest z10)
    - Borderline cases (intermediate classification)
    """
    output_dir.mkdir(exist_ok=True)

    # Most interior
    most_interior_idx = df['z_10A'].idxmax()
    visualize_amino_acid_environment(
        df, most_interior_idx, sphere_radius=6.0,
        save_path=output_dir / f"most_interior_{df.iloc[most_interior_idx]['res_label'].replace(':', '_')}.png"
    )

    # Most exterior
    most_exterior_idx = df['z_10A'].idxmin()
    visualize_amino_acid_environment(
        df, most_exterior_idx, sphere_radius=6.0,
        save_path=output_dir / f"most_exterior_{df.iloc[most_exterior_idx]['res_label'].replace(':', '_')}.png"
    )

    # Intermediate cases
    intermediate_df = df[df['burial_label'] == 'intermediate']
    if len(intermediate_df) > 0:
        for i, (idx, row) in enumerate(intermediate_df.head(3).iterrows()):
            visualize_amino_acid_environment(
                df, idx, sphere_radius=6.0,
                save_path=output_dir / f"intermediate_{i+1}_{row['res_label'].replace(':', '_')}.png"
            )

    print(f"Generated visualizations in {output_dir}/")


# ==============================
# Step 6c — Statistics summary
# ==============================

def generate_statistics_report(df: pd.DataFrame, params: ClassificationParams, output_path: Path = Path("statistics_report.txt")):
    """
    Ultra-lightweight statistics tool for rapid analysis.
    """
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PROTEIN BURIAL STATISTICS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Basic counts
        f.write("CLASSIFICATION SUMMARY:\n")
        f.write("-" * 40 + "\n")
        counts = df['burial_label'].value_counts()
        for label, count in counts.items():
            pct = 100 * count / len(df)
            f.write(f"{label:12s}: {count:3d} ({pct:5.1f}%)\n")
        f.write(f"{'TOTAL':12s}: {len(df):3d}\n\n")

        # Z-score statistics
        f.write("Z-SCORE STATISTICS:\n")
        f.write("-" * 40 + "\n")
        for col in ['z_6A', 'z_10A']:
            if col in df.columns:
                f.write(f"{col}:\n")
                f.write(f"  Mean: {df[col].mean():7.3f}\n")
                f.write(f"  Std:  {df[col].std():7.3f}\n")
                f.write(f"  Min:  {df[col].min():7.3f}\n")
                f.write(f"  Max:  {df[col].max():7.3f}\n\n")

        # Homogeneity statistics
        f.write("HOMOGENEITY (Spherical Variance) STATISTICS:\n")
        f.write("-" * 40 + "\n")
        for col in ['sph_var_6A', 'sph_var_10A']:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    f.write(f"{col}:\n")
                    f.write(f"  Mean: {valid.mean():7.3f}\n")
                    f.write(f"  Std:  {valid.std():7.3f}\n")
                    f.write(f"  Min:  {valid.min():7.3f}\n")
                    f.write(f"  Max:  {valid.max():7.3f}\n\n")

        # Parameters used
        f.write("PARAMETERS USED:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Sphere radii: {params.sphere_r1}Å, {params.sphere_r2}Å\n")
        f.write(f"Z-score thresholds: low={params.z_low}, high={params.z_high}\n")
        f.write(f"Homogeneity thresholds: low={params.homog_low}, high={params.homog_high}\n")
        f.write(f"Graph cutoff: {params.graph_cutoff}Å\n")
        f.write(f"Use degree in classification: {params.use_degree}\n\n")

        # Validation if available
        if 'dssp_label' in df.columns:
            f.write("DSSP VALIDATION:\n")
            f.write("-" * 40 + "\n")
            df_eval = df[df['dssp_label'].notna()].copy()
            df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})
            accuracy = (df_eval['dssp_label'] == df_eval['burial_label']).mean()
            f.write(f"Accuracy: {accuracy:.3f} ({100*accuracy:.1f}%)\n\n")

        if 'stride_label' in df.columns:
            f.write("STRIDE VALIDATION:\n")
            f.write("-" * 40 + "\n")
            df_eval = df[df['stride_label'].notna()].copy()
            df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})
            accuracy = (df_eval['stride_label'] == df_eval['burial_label']).mean()
            f.write(f"Accuracy: {accuracy:.3f} ({100*accuracy:.1f}%)\n\n")

    print(f"Statistics report saved: {output_path}")


# ==============================
# Step 7 — PyMOL coloring script
# ==============================

def pymol_resi_string(rows: pd.DataFrame) -> str:
    """
    Build a PyMOL 'resi' string like '1+2+3A+10' and include insertion codes if present.
    """
    parts = []
    for _, r in rows.iterrows():
        if r['icode']:
            parts.append(f"{r['resseq']}{r['icode']}")
        else:
            parts.append(f"{r['resseq']}")
    return '+'.join(parts)

def write_pymol_coloring(df: pd.DataFrame, pdb_path: Path, out_path: Path = Path("color_by_burial.pml")):
    """
    Write a PyMOL script that:
      - loads the PDB
      - shows cartoon
      - colors interior (blue), exterior (red), intermediate (yellow)
    """
    pdb_path = Path(pdb_path)
    obj_name = pdb_path.stem.replace('.', '_')

    with open(out_path, 'w') as f:
        f.write(f'load "{str(pdb_path)}", {obj_name}\n')
        f.write(f'hide everything, {obj_name}\n')
        f.write(f'show cartoon, {obj_name}\n')
        f.write(f'color grey70, {obj_name}\n')

        # Group by chain for correct selections
        for chain_id, g in df.groupby('chain_id'):
            for label, color in [('interior', 'blue'), ('exterior', 'red'), ('intermediate', 'yellow')]:
                subset = g[g['burial_label'] == label]
                if subset.empty:
                    continue
                resistr = pymol_resi_string(subset)
                f.write(f'select {label}_{chain_id}, ({obj_name} and chain {chain_id} and resi {resistr})\n')
                f.write(f'color {color}, {label}_{chain_id}\n')

        # nice look
        f.write('set cartoon_transparency, 0.2\n')
        f.write('set ray_opaque_background, off\n')
        f.write('bg_color white\n')

    print(f'Wrote PyMOL coloring script: {out_path}')


# ==============================
# Step 8 — Main pipeline
# ==============================

def run_pipeline(
    pdb_path: Optional[str | Path] = None,
    do_dssp: bool = True,
    do_stride: bool = True,
    optimize_params: bool = False,  # NEW: enable parameter optimization
    visualize: bool = False,  # NEW: enable visualizations
    show_examples: bool = True  # NEW: show side-by-side comparisons
) -> pd.DataFrame:
    """
    Full pipeline:
      1) Extract CA atoms
      2) Pairwise distances
      3) Adjacency at 7 Å
      4) Graph metrics (degree, eccentricity, radius, diameter)
      5) Two-sphere neighbor counts (6 Å, 10 Å) and Z-scores
      6) OPTIONAL: Spherical variance (homogeneity) at 6/10 Å
      7) Classification into interior/exterior/intermediate (WITHOUT deg7)
      8) Save CSV, NPYs, PyMOL script + summary
      9) OPTIONAL: DSSP validation
      10) OPTIONAL: STRIDE validation
      11) OPTIONAL: Parameter optimization
      12) OPTIONAL: Visualizations
    """
    # 1) Extract
    ca_list = extract_ca_atoms(pdb_path=pdb_path)
    df = pd.DataFrame(ca_list)
    df['res_label'] = df.apply(make_res_label, axis=1)

    print(f"Loaded {len(df)} CA atoms from {pdb_path if pdb_path else DEFAULT_PDB_PATH}")

    # 2) Distances
    coords = df[['x', 'y', 'z']].to_numpy(float)
    D = pairwise_distance_matrix(coords)
    assert np.allclose(D, D.T, atol=1e-6)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)
    print("Distance matrix computed.")

    # 3) Adjacency (graph cutoff)
    A = adjacency_from_distance(D, cutoff=CUTOFF_GRAPH)
    print(f"Adjacency (≤{CUTOFF_GRAPH:.1f} Å) computed. Edges: {A.sum()//2}")

    # 4) Graph metrics
    deg_7, ecc, comp_id, comp_rad_node, comp_diam_node, summary = graph_metrics_from_adjacency(A)
    print(f"Graph: {summary.n_nodes} nodes, {summary.n_edges} edges, "
          f"{summary.n_components} component(s).")
    print(f"Component sizes: {summary.component_sizes}")
    print(f"Component diameters: {summary.component_diameters}")
    print(f"Component radii: {summary.component_radii}")

    # 5) Two-sphere neighbor counts & Z-scores
    df = add_neighbor_counts(df, D, radii=SPHERE_RADII)
    for r in SPHERE_RADII:
        col = f'deg_{int(r)}A'
        df[f'z_{int(r)}A'] = zscore(df[col])

    # Merge graph metrics into df
    idx = pd.Index(range(len(df)), name='node')
    df_metrics = pd.DataFrame({
        'node': idx,
        'degree_7A': pd.Series(deg_7),
        'eccentricity': pd.Series(ecc),
        'component_id': pd.Series(comp_id),
        'component_radius': pd.Series(comp_rad_node),
        'component_diameter': pd.Series(comp_diam_node),
    }).set_index('node')
    df = df.join(df_metrics, how='left')

    # 6) OPTIONAL: Homogeneity via spherical variance
    print("Computing spherical variance (homogeneity) ...")
    for r in HOMOG_RADII:
        sv = spherical_variance_for_neighbors(coords, radius=r)
        df[f'sph_var_{int(r)}A'] = sv

    # 7) Classification
    df = classify_residues(df)

    # 8) Save outputs
    out_dir = Path(".")
    csv_path = out_dir / "ca_with_metrics.csv"
    npy_D = out_dir / "distance_matrix.npy"
    npy_A = out_dir / "adjacency_7A.npy"
    df.to_csv(csv_path, index=False)
    np.save(npy_D, D)
    np.save(npy_A, A)

    print("Saved:")
    print(f" - {csv_path.name}")
    print(f" - {npy_D.name}")
    print(f" - {npy_A.name}")

    # Summary
    summary_path = out_dir / "classification_summary.txt"
    with open(summary_path, "w") as fh:
        counts = df['burial_label'].value_counts()
        fh.write("Counts per class:\n")
        fh.write(counts.to_string() + "\n\n")
        fh.write("Preview (first 12 rows):\n")
        preview_cols = [
            'res_label', 'degree_7A', 'eccentricity',
            f'deg_{int(SPHERE_RADII[0])}A', f'z_{int(SPHERE_RADII[0])}A',
            f'deg_{int(SPHERE_RADII[1])}A', f'z_{int(SPHERE_RADII[1])}A',
            f'sph_var_{int(HOMOG_RADII[0])}A', f'sph_var_{int(HOMOG_RADII[1])}A',
            'burial_label', 'burial_reason',
            'component_id', 'component_radius', 'component_diameter'
        ]
        fh.write(df[preview_cols].head(12).to_string(index=False) + "\n")
    print(f"Saved summary: {summary_path}")

    # PyMOL coloring script
    pymol_script = out_dir / "color_by_burial.pml"
    write_pymol_coloring(df, pdb_path if pdb_path else DEFAULT_PDB_PATH, pymol_script)

    # --- Optional DSSP validation ---
    if do_dssp:
        print("\n=== Running DSSP validation ===")
        df = add_dssp_acc_and_rsa(df, pdb_path=Path(pdb_path) if pdb_path else DEFAULT_PDB_PATH, model_index=0)
        df['agree_with_dssp'] = np.where(
            df['dssp_label'].notna(), df['dssp_label'] == df['burial_label'], np.nan
        )
        evaluate_against_dssp(df, treat_intermediate_as='exterior')  # or 'drop'

    # --- Optional STRIDE validation ---
    if do_stride:
        print("\n=== Running STRIDE validation ===")
        df = add_stride_acc_and_rsa(df, pdb_path=Path(pdb_path) if pdb_path else DEFAULT_PDB_PATH)
        df['agree_with_stride'] = np.where(
            df['stride_label'].notna(), df['stride_label'] == df['burial_label'], np.nan
        )
        evaluate_against_stride(df, treat_intermediate_as='exterior')  # or 'drop'

    # --- Compare DSSP vs STRIDE ---
    if do_dssp and do_stride:
        compare_dssp_vs_stride(df)

    # --- Show example predictions ---
    if show_examples and (do_dssp or do_stride):
        show_example_predictions(df, n_examples=10)

    # --- NEW: Parameter optimization ---
    params = DEFAULT_PARAMS
    if optimize_params and (do_dssp or do_stride):
        reference = 'dssp_label' if do_dssp else 'stride_label'
        print(f"\n=== Optimizing parameters against {reference} ===")
        best_params = optimize_parameters_against_reference(df, reference_col=reference)

        # Reclassify with optimized parameters
        df = classify_residues_with_params(df, best_params)
        params = best_params

        # Re-evaluate
        if do_dssp:
            df['agree_with_dssp'] = np.where(
                df['dssp_label'].notna(), df['dssp_label'] == df['burial_label'], np.nan
            )
            print("\n=== After optimization (DSSP) ===")
            evaluate_against_dssp(df, treat_intermediate_as='exterior')

        if do_stride:
            df['agree_with_stride'] = np.where(
                df['stride_label'].notna(), df['stride_label'] == df['burial_label'], np.nan
            )
            print("\n=== After optimization (STRIDE) ===")
            evaluate_against_stride(df, treat_intermediate_as='exterior')

    # Save updated CSV with validation columns
    if do_dssp or do_stride or optimize_params:
        df.to_csv(csv_path, index=False)
        validation_cols = []
        if do_dssp:
            validation_cols.append("DSSP")
        if do_stride:
            validation_cols.append("STRIDE")
        if validation_cols:
            print(f"Updated {csv_path.name} with {' and '.join(validation_cols)} columns")

    # --- NEW: Generate statistics report ---
    generate_statistics_report(df, params)

    # --- NEW: Visualizations ---
    if visualize:
        print("\n=== Generating visualizations ===")
        visualize_all_interesting_residues(df)

    # Console preview
    preview_cols = [
        'res_label', 'degree_7A', 'eccentricity',
        f'deg_{int(SPHERE_RADII[0])}A', f'z_{int(SPHERE_RADII[0])}A',
        f'deg_{int(SPHERE_RADII[1])}A', f'z_{int(SPHERE_RADII[1])}A',
        f'sph_var_{int(HOMOG_RADII[0])}A', f'sph_var_{int(HOMOG_RADII[1])}A',
        'burial_label'
    ]
    print("\nPreview:")
    print(df[preview_cols].head(12).to_string(index=False))

    return df


# ==============================
# Step 9 — Interactive analysis tools
# ==============================

def find_residue_by_label(df: pd.DataFrame, res_label: str) -> Optional[int]:
    """
    Find residue index by label (e.g., 'A:25 LEU')
    Returns the index in the dataframe, or None if not found.
    """
    matches = df[df['res_label'].str.contains(res_label, case=False)]
    if len(matches) == 0:
        print(f"Residue '{res_label}' not found")
        return None
    elif len(matches) > 1:
        print(f"Multiple matches for '{res_label}':")
        print(matches[['res_label', 'burial_label']].to_string())
        return matches.index[0]
    else:
        return matches.index[0]


def visualize_residue_by_name(df: pd.DataFrame, res_label: str, sphere_radius: float = 6.0):
    """
    Visualize a specific residue environment by its label.
    Example: visualize_residue_by_name(df, 'A:50', sphere_radius=6.0)
    """
    idx = find_residue_by_label(df, res_label)
    if idx is not None:
        visualize_amino_acid_environment(df, idx, sphere_radius)


def analyze_misclassifications(df: pd.DataFrame, reference: str = 'dssp_label'):
    """
    Detailed analysis of which residues are being misclassified and why.
    Shows patterns in false positives and false negatives.
    """
    if reference not in df.columns:
        print(f"{reference} not available")
        return

    df_eval = df[df[reference].notna()].copy()
    df_eval['burial_label_binary'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})

    # Find misclassifications
    df_eval['correct'] = df_eval[reference] == df_eval['burial_label_binary']

    false_positives = df_eval[(df_eval[reference] == 'interior') & (df_eval['burial_label_binary'] == 'exterior')]
    false_negatives = df_eval[(df_eval[reference] == 'exterior') & (df_eval['burial_label_binary'] == 'interior')]

    print(f"\n=== Misclassification Analysis (vs {reference}) ===\n")
    print(f"Total residues: {len(df_eval)}")
    print(f"Correct: {df_eval['correct'].sum()} ({100*df_eval['correct'].mean():.1f}%)")
    print(f"Incorrect: {(~df_eval['correct']).sum()} ({100*(~df_eval['correct']).mean():.1f}%)\n")

    print(f"False Positives (we said exterior, {reference} says interior): {len(false_positives)}")
    if len(false_positives) > 0:
        print("  Z-score stats for FP:")
        print(f"    z6  mean: {false_positives['z_6A'].mean():.2f} (range: {false_positives['z_6A'].min():.2f} to {false_positives['z_6A'].max():.2f})")
        print(f"    z10 mean: {false_positives['z_10A'].mean():.2f} (range: {false_positives['z_10A'].min():.2f} to {false_positives['z_10A'].max():.2f})")
        if 'sph_var_6A' in false_positives.columns:
            sv_valid = false_positives['sph_var_6A'].dropna()
            if len(sv_valid) > 0:
                print(f"    sph_var mean: {sv_valid.mean():.2f}")

    print(f"\nFalse Negatives (we said interior, {reference} says exterior): {len(false_negatives)}")
    if len(false_negatives) > 0:
        print("  Z-score stats for FN:")
        print(f"    z6  mean: {false_negatives['z_6A'].mean():.2f} (range: {false_negatives['z_6A'].min():.2f} to {false_negatives['z_6A'].max():.2f})")
        print(f"    z10 mean: {false_negatives['z_10A'].mean():.2f} (range: {false_negatives['z_10A'].min():.2f} to {false_negatives['z_10A'].max():.2f})")
        if 'sph_var_6A' in false_negatives.columns:
            sv_valid = false_negatives['sph_var_6A'].dropna()
            if len(sv_valid) > 0:
                print(f"    sph_var mean: {sv_valid.mean():.2f}")

    print("\n=== Suggested parameter adjustments ===")
    if len(false_positives) > len(false_negatives):
        print(f"Many false positives ({len(false_positives)}) → thresholds too strict")
        print(f"  Try: z_low = {false_positives['z_6A'].quantile(0.25):.2f} (currently -0.5)")
        print(f"  Try: z_high = {false_positives['z_10A'].quantile(0.75):.2f} (currently 0.5)")
    elif len(false_negatives) > len(false_positives):
        print(f"Many false negatives ({len(false_negatives)}) → thresholds too loose")
        print("  Try increasing z_high or decreasing z_low")
    else:
        print("Balanced errors - fine-tune thresholds carefully")

    return false_positives, false_negatives


def test_parameter_set(df: pd.DataFrame, z_low: float, z_high: float,
                       homog_low: float, homog_high: float,
                       reference: str = 'dssp_label') -> float:
    """
    Quick test of a parameter set. Returns accuracy.
    Use this to manually experiment with different thresholds.

    Example:
    test_parameter_set(df, z_low=-0.8, z_high=0.8, homog_low=0.30, homog_high=0.70)
    """
    params = ClassificationParams(
        z_low=z_low,
        z_high=z_high,
        homog_low=homog_low,
        homog_high=homog_high
    )

    df_test = classify_residues_with_params(df.copy(), params)

    if reference not in df_test.columns:
        print(f"{reference} not available")
        return 0.0

    df_eval = df_test[df_test[reference].notna()].copy()
    df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})

    accuracy = (df_eval[reference] == df_eval['burial_label']).mean()

    print(f"\nParameters: z_low={z_low}, z_high={z_high}, homog_low={homog_low}, homog_high={homog_high}")
    print(f"Accuracy vs {reference}: {accuracy:.3f} ({100*accuracy:.1f}%)")

    # Show confusion matrix
    cm = pd.crosstab(df_eval[reference], df_eval['burial_label'],
                     rownames=[reference], colnames=['Predicted'])
    print(cm)

    return accuracy


def batch_process_proteins(pdb_files: List[Path], output_dir: Path = Path("batch_results")):
    """
    Process multiple PDB files and generate summary statistics.
    Useful for testing on a database of proteins.

    Example:
    batch_process_proteins([Path("3pte.pdb"), Path("1crn.pdb")])
    """
    output_dir.mkdir(exist_ok=True)
    results = []

    for pdb_file in pdb_files:
        print(f"\n{'='*60}")
        print(f"Processing {pdb_file.name}...")
        print(f"{'='*60}")

        try:
            df = run_pipeline(pdb_path=pdb_file, do_dssp=True, do_stride=True,
                            optimize_params=False, visualize=False)

            # Calculate accuracies
            dssp_acc = 0.0
            stride_acc = 0.0

            if 'dssp_label' in df.columns:
                df_dssp = df[df['dssp_label'].notna()].copy()
                df_dssp['burial_label'] = df_dssp['burial_label'].replace({'intermediate': 'exterior'})
                dssp_acc = (df_dssp['dssp_label'] == df_dssp['burial_label']).mean()

            if 'stride_label' in df.columns:
                df_stride = df[df['stride_label'].notna()].copy()
                df_stride['burial_label'] = df_stride['burial_label'].replace({'intermediate': 'exterior'})
                stride_acc = (df_stride['stride_label'] == df_stride['burial_label']).mean()

            results.append({
                'pdb_file': pdb_file.name,
                'n_residues': len(df),
                'n_interior': (df['burial_label'] == 'interior').sum(),
                'n_exterior': (df['burial_label'] == 'exterior').sum(),
                'n_intermediate': (df['burial_label'] == 'intermediate').sum(),
                'dssp_accuracy': dssp_acc,
                'stride_accuracy': stride_acc
            })

        except Exception as e:
            print(f"ERROR processing {pdb_file.name}: {e}")
            results.append({
                'pdb_file': pdb_file.name,
                'error': str(e)
            })

    # Save batch results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "batch_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Results saved to: {results_path}")
    print(f"\nSummary:")
    print(results_df.to_string(index=False))

    if 'dssp_accuracy' in results_df.columns:
        print(f"\nAverage DSSP accuracy: {results_df['dssp_accuracy'].mean():.3f}")
    if 'stride_accuracy' in results_df.columns:
        print(f"Average STRIDE accuracy: {results_df['stride_accuracy'].mean():.3f}")

    return results_df


# ==============================
# Entry point
# ==============================

if __name__ == "__main__":
    # Configure analysis options
    DO_DSSP = True        # Run DSSP validation
    DO_STRIDE = True      # Run STRIDE validation
    OPTIMIZE_PARAMS = True   # Running EXPANDED optimization (1296 combinations!)
    VISUALIZE = False     # Set True to generate 3D environment plots
    SHOW_EXAMPLES = True  # Show side-by-side comparisons of YOUR vs DSSP/STRIDE

    # For testing on 3pte protein (baseline as mentioned in class notes):
    # PDB_FILE = "/path/to/3pte.pdb"
    # df = run_pipeline(pdb_path=PDB_FILE, do_dssp=DO_DSSP, do_stride=DO_STRIDE,
    #                   optimize_params=OPTIMIZE_PARAMS, visualize=VISUALIZE, show_examples=SHOW_EXAMPLES)

    # Default: use 3pte.pdb
    df = run_pipeline(
        do_dssp=DO_DSSP,
        do_stride=DO_STRIDE,
        optimize_params=OPTIMIZE_PARAMS,
        visualize=VISUALIZE,
        show_examples=SHOW_EXAMPLES
    )

    # Example: Visualize specific residue environment
    # visualize_amino_acid_environment(df, residue_index=25, sphere_radius=6.0)

    # Option 2: or pass another file explicitly
    # run_pipeline(pdb_path="path/to/another/structure.pdb", do_dssp=True, do_stride=True)

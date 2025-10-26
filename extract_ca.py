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
DEFAULT_PDB_PATH = Path("/Users/famnit/Desktop/pythonProject/pdb1crn.ent")

# Cutoffs (in Å)
CUTOFF_GRAPH = 7.0           # for building the graph
SPHERE_RADII = (6.0, 10.0)   # for neighbor counts / Z-scores
HOMOG_RADII = (6.0, 10.0)    # for homogeneity analysis (spherical variance)

# Classification parameters (tune if you like)
Z_LOW = -0.5
Z_HIGH = 0.5
HOMOG_LOW = 0.35   # spherical variance < 0.35 suggests neighbors concentrated to one side (surface)
HOMOG_HIGH = 0.65  # spherical variance > 0.65 suggests neighbors well spread (interior)


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
      - Degree on the 7 Å graph (low/high relative to protein)
      - Optional spherical variance (homogeneity) at 6/10 Å

    Heuristic decision:
      - EXTERIOR if any (z_6A <= Z_LOW) OR (z_10A <= Z_LOW) OR (degree_7A <= q25) OR (sph_var low)
      - INTERIOR if (z_6A >= Z_HIGH AND z_10A >= Z_HIGH) OR (degree_7A >= q75) OR (sph_var high)
      - otherwise INTERMEDIATE
    """
    out = df.copy()

    # Quartiles for degree_7A to adapt per-structure
    q25 = out['degree_7A'].quantile(0.25)
    q75 = out['degree_7A'].quantile(0.75)

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
        is_exterior = (z6 <= Z_LOW) or (z10 <= Z_LOW) or (deg7 <= q25) or sv_low_flag
        is_interior = ((z6 >= Z_HIGH and z10 >= Z_HIGH) or (deg7 >= q75) or sv_high_flag)
#------------------------------------------------------------------------------------------------
        if is_exterior and not is_interior:
            labels.append('exterior')
            reasons.append(f"low Z or low degree{', low homog' if sv_low_flag else ''}")
        elif is_interior and not is_exterior:
            labels.append('interior')
            reasons.append(f"high Z or high degree{', high homog' if sv_high_flag else ''}")
        else:
            labels.append('intermediate')
            reasons.append('mixed signals')

    out['burial_label'] = labels
    out['burial_reason'] = reasons
    return out


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
    Prints confusion matrix and accuracy between your 'burial_label' and DSSP 'dssp_label'.
    treat_intermediate_as: 'exterior' | 'drop'
    """
    if 'dssp_label' not in df.columns or df['dssp_label'].isna().all():
        print("No DSSP labels available. Ensure mkdssp is installed and do_dssp=True.")
        return

    df_eval = df[df['dssp_label'].notna()].copy()
    if treat_intermediate_as == 'drop':
        df_eval = df_eval[df_eval['burial_label'].isin(['interior', 'exterior'])]
    else:
        df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})

    y_true = df_eval['dssp_label']
    y_pred = df_eval['burial_label']

    cm = pd.crosstab(y_true, y_pred, rownames=['DSSP'], colnames=['Pred'])
    acc = (y_true == y_pred).mean()

    print("\n=== DSSP agreement ===")
    print(cm)
    print(f"Accuracy: {acc:.3f}")


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
    Prints confusion matrix and accuracy between your 'burial_label' and STRIDE 'stride_label'.
    treat_intermediate_as: 'exterior' | 'drop'
    """
    if 'stride_label' not in df.columns or df['stride_label'].isna().all():
        print("No STRIDE labels available. Ensure stride is installed and do_stride=True.")
        return

    df_eval = df[df['stride_label'].notna()].copy()
    if treat_intermediate_as == 'drop':
        df_eval = df_eval[df_eval['burial_label'].isin(['interior', 'exterior'])]
    else:
        df_eval['burial_label'] = df_eval['burial_label'].replace({'intermediate': 'exterior'})

    y_true = df_eval['stride_label']
    y_pred = df_eval['burial_label']

    cm = pd.crosstab(y_true, y_pred, rownames=['STRIDE'], colnames=['Pred'])
    acc = (y_true == y_pred).mean()

    print("\n=== STRIDE agreement ===")
    print(cm)
    print(f"Accuracy: {acc:.3f}")


def compare_dssp_vs_stride(df: pd.DataFrame):
    """
    Compares DSSP and STRIDE labels to see how much they agree with each other.
    """
    if 'dssp_label' not in df.columns or 'stride_label' not in df.columns:
        print("Both DSSP and STRIDE labels needed for comparison.")
        return

    df_both = df[(df['dssp_label'].notna()) & (df['stride_label'].notna())].copy()

    if len(df_both) == 0:
        print("No residues have both DSSP and STRIDE labels.")
        return

    cm = pd.crosstab(df_both['dssp_label'], df_both['stride_label'],
                     rownames=['DSSP'], colnames=['STRIDE'])
    agreement = (df_both['dssp_label'] == df_both['stride_label']).mean()

    print("\n=== DSSP vs STRIDE agreement ===")
    print(cm)
    print(f"Agreement: {agreement:.3f}")


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

def run_pipeline(pdb_path: Optional[str | Path] = None, do_dssp: bool = True, do_stride: bool = True) -> pd.DataFrame:
    """
    Full pipeline:
      1) Extract CA atoms
      2) Pairwise distances
      3) Adjacency at 7 Å
      4) Graph metrics (degree, eccentricity, radius, diameter)
      5) Two-sphere neighbor counts (6 Å, 10 Å) and Z-scores
      6) OPTIONAL: Spherical variance (homogeneity) at 6/10 Å
      7) Classification into interior/exterior/intermediate
      8) Save CSV, NPYs, PyMOL script + summary
      9) OPTIONAL: DSSP validation
      10) OPTIONAL: STRIDE validation
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

    # Save updated CSV with validation columns
    if do_dssp or do_stride:
        df.to_csv(csv_path, index=False)
        validation_cols = []
        if do_dssp:
            validation_cols.append("DSSP")
        if do_stride:
            validation_cols.append("STRIDE")
        print(f"Updated {csv_path.name} with {' and '.join(validation_cols)} columns")

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
# Entry point
# ==============================

if __name__ == "__main__":
    # Configure which validation methods to run
    DO_DSSP = True    # set to False if DSSP not available
    DO_STRIDE = True  # set to False if STRIDE not installed

    run_pipeline(do_dssp=DO_DSSP, do_stride=DO_STRIDE)

    # Option 2: or pass another file explicitly

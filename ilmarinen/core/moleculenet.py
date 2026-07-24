"""MoleculeNet loaders -- SMILES-based molecular property datasets as graphs (for the graph schema).
Chemistry / pharmacology tasks, complementing the quantum-chemistry QM7/QM9 (which are 3D-geometry).

- ESOL (delaney)  : water solubility regression (physical chemistry), 1128 molecules.
- Tox21           : 12-task toxicity classification (toxicology/pharma), ~7831 molecules, multi-label
                    with missing entries; we default to a single task (NR-AR) for a clean binary problem.

Molecules are parsed from SMILES with RDKit into graphs matching the package's graph contract:
dict(x (n_atoms, F), edge_index (2,|E|)). Atom features are a small, standard set (one-hot atom type over
a common element set + degree + aromaticity), bonds give undirected edges (both directions).
"""
from __future__ import annotations
import csv
import gzip
import numpy as np

# a compact common-organic element vocabulary; anything else -> "other"
_ELEMENTS = ["C", "N", "O", "F", "S", "Cl", "Br", "P", "I", "B", "Si"]
_ELEM_IDX = {e: i for i, e in enumerate(_ELEMENTS)}
_N_ELEM = len(_ELEMENTS) + 1                               # +1 for "other"
ATOM_FEAT_DIM = _N_ELEM + 1 + 1                            # type one-hot + degree + aromatic flag


def _atom_features(atom):
    f = np.zeros(ATOM_FEAT_DIM, np.float32)
    sym = atom.GetSymbol()
    f[_ELEM_IDX.get(sym, _N_ELEM - 1)] = 1.0              # element one-hot (last slot = other)
    f[_N_ELEM] = atom.GetDegree() / 4.0                   # normalized degree
    f[_N_ELEM + 1] = 1.0 if atom.GetIsAromatic() else 0.0
    return f


def _smiles_to_graph(smiles):
    """SMILES -> (x (n,F) float32, edge_index (2,|E|) int64) or None if unparseable / <2 atoms."""
    from rdkit import Chem
    import torch
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    x = np.stack([_atom_features(a) for a in mol.GetAtoms()], 0)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]                      # undirected: both directions
    if not src:
        return None
    edge_index = np.array([src, dst], np.int64)
    return torch.tensor(x), torch.tensor(edge_index)


def load_esol(path=None, n_max=None):
    """ESOL water-solubility regression. Returns (graphs, y) with y = measured log solubility (float32).
    graphs: list of dict(x, edge_index) matching the graph-schema contract."""
    if path is None:
        from .data_sources import moleculenet_csv; path = moleculenet_csv("delaney-processed.csv")
    graphs, ys = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = _smiles_to_graph(row["smiles"].strip())
            if g is None:
                continue
            graphs.append({"x": g[0], "edge_index": g[1]})
            ys.append(float(row["measured log solubility in mols per litre"]))
            if n_max and len(graphs) >= n_max:
                break
    return graphs, np.array(ys, np.float32)


def load_tox21(path=None, task="NR-AR", n_max=None):
    """Tox21 toxicity classification. Returns (graphs, y) for one task (default NR-AR, androgen receptor).
    Molecules with a missing label for the chosen task are skipped (Tox21 is sparsely labelled).
    y is 0/1 float32. graphs match the graph-schema contract."""
    if path is None:
        from .data_sources import moleculenet_csv; path = moleculenet_csv("tox21.csv.gz")
    graphs, ys = [], []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get(task, "")
            if lbl == "" or lbl is None:                  # missing label -> skip
                continue
            g = _smiles_to_graph(row["smiles"].strip())
            if g is None:
                continue
            graphs.append({"x": g[0], "edge_index": g[1]})
            ys.append(float(int(float(lbl))))
            if n_max and len(graphs) >= n_max:
                break
    return graphs, np.array(ys, np.float32)


def collate_mol_graphs(graphs, idx):
    """Disjoint-union batch matching collate_graphs of the graph schema:
    returns (x, edge_index, batch, n_graphs)."""
    import torch
    xs, eis, batch = [], [], []
    offset = 0
    for gi, i in enumerate(idx):
        g = graphs[i]; n = g["x"].shape[0]
        xs.append(g["x"]); eis.append(g["edge_index"] + offset)
        batch.append(torch.full((n,), gi, dtype=torch.long)); offset += n
    return torch.cat(xs, 0), torch.cat(eis, 1), torch.cat(batch, 0), len(idx)

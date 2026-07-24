"""rMD17 loader (revised MD17; Christensen & von Lilienfeld 2020). Molecular dynamics trajectories with
ENERGIES and FORCES for small molecules -- the standard benchmark for force fields / conservative
equivariant potentials. Unlike QM7/QM9 (energy only), rMD17 has forces, exercising the conservative
force-training path (F = -dE/dr) of the equivariant schema.

Native format: one .npz per molecule with keys nuclear_charges (n_atoms,), coords (n_conf,n_atoms,3),
energies (n_conf,), forces (n_conf,n_atoms,3), plus old_* variants. Energies in kcal/mol, coords in
Angstrom, forces in kcal/mol/Angstrom. Obtain from figshare 12672038 (Revised MD17).
"""
from __future__ import annotations

import numpy as np


def load_rmd17(path, max_conf=None, center=True):
    """Load an rMD17 .npz. Returns dict(Z (n_atoms,) atomic numbers, coords (n,n_atoms,3),
    energies (n,), forces (n,n_atoms,3)). If center, each conformation is centered on its centroid
    (forces are translation-invariant, unaffected)."""
    d = np.load(path)
    Z = d['nuclear_charges'].astype(np.int64)
    coords = d['coords'].astype(np.float32)
    energies = d['energies'].astype(np.float32)
    forces = d['forces'].astype(np.float32)
    if max_conf:
        coords, energies, forces = coords[:max_conf], energies[:max_conf], forces[:max_conf]
    if center:
        coords = coords - coords.mean(1, keepdims=True)
    return {"Z": Z, "coords": coords, "energies": energies, "forces": forces}


def build_rmd17_equivariant(data, indices, cutoff=3.0):
    """Build the equivariant-graph batch (x, pos, edge_index, batch, n_graphs) for a set of rMD17
    conformations, with one-hot element features and edges from a per-conformation distance cutoff
    (Angstrom). Matches the collate_equivariant contract of the equivariant schema."""
    import torch
    Z = data["Z"]; coords = data["coords"]
    n_atoms = len(Z)
    elems = sorted(set(Z.tolist())); emap = {e: i for i, e in enumerate(elems)}
    feat0 = np.zeros((n_atoms, len(elems)), np.float32)
    for i, z in enumerate(Z):
        feat0[i, emap[z]] = 1.0
    xs, ps, eis, batch = [], [], [], []
    off = 0
    for b, i in enumerate(indices):
        p = coords[i]
        D = np.linalg.norm(p[:, None] - p[None], axis=-1)
        src, dst = np.where((D < cutoff) & (D > 0))
        eis.append(np.stack([src, dst]) + off)
        xs.append(feat0); ps.append(p); batch.append(np.full(n_atoms, b)); off += n_atoms
    return (torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(ps)),
            torch.tensor(np.concatenate(eis, axis=1)), torch.tensor(np.concatenate(batch)),
            len(indices), len(elems))

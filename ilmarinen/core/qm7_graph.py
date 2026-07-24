"""Build molecular BOND GRAPHS from the local QM7 coordinates (no download needed).

QM7 coordinates are in Bohr (median nearest-neighbor ~2.07 Bohr = 1.09 A, a C-H bond). A cutoff of
~3.5 Bohr (~1.85 A) captures covalent bonds (C-C 1.54A=2.9Bohr, C-H 1.09A=2.06Bohr) without pulling
in second neighbors (avg degree ~2.5). Node features = one-hot atom type (5 elements: H,C,N,O,S).
"""

import numpy as np
import torch

from .qm7 import load_qm7


def _build_qm7(path, cutoff_bohr, n_max, with_pos):
    """Shared QM7 molecular-graph builder. Per molecule -> dict(x (n,5) one-hot atom type, edge_index (2,|E|)
    from the interatomic-distance cutoff graph, with a nearest-neighbor fallback for isolated atoms), plus
    pos (n,3) Bohr coordinates when `with_pos` (the equivariant contract needs geometry). Returns
    (graphs, y). build_qm7_graphs / build_qm7_equivariant are the thin public wrappers over this."""
    if path is None:
        from .data_sources import qm7_mat_path

        path = qm7_mat_path()
    R, T, M, y = load_qm7(path)
    graphs, ys = [], []
    N = len(R) if n_max is None else min(n_max, len(R))
    for i in range(N):
        present = T[i].sum(-1) > 0
        feat = T[i][present].astype(np.float32)  # (n_atoms, 5) one-hot type
        coords = R[i][present].astype(np.float32)
        n = len(coords)
        if n < 2:
            continue
        D = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        srcs, dsts = np.where(D < cutoff_bohr)
        if len(srcs) == 0:  # fallback: connect nearest neighbor
            dsts = D.argmin(1)
            srcs = np.arange(n)
        g = {"x": torch.tensor(feat), "edge_index": torch.tensor(np.stack([srcs, dsts], axis=0).astype(np.int64))}
        if with_pos:
            g["pos"] = torch.tensor(coords)
        graphs.append(g)
        ys.append(float(y[i]))
    return graphs, np.array(ys, dtype=np.float32)


def build_qm7_graphs(path=None, cutoff_bohr=3.5, n_max=None):
    """Per-molecule graphs dict(x (n,5), edge_index (2,|E|)) + targets y, for the graph contract."""
    return _build_qm7(path, cutoff_bohr, n_max, with_pos=False)


def collate_graphs(graphs, idx):
    """Disjoint-union batch a list of graph dicts at positions idx into flat tensors."""
    xs, eis, batch = [], [], []
    offset = 0
    for g_i, i in enumerate(idx):
        g = graphs[i]
        n = g["x"].shape[0]
        xs.append(g["x"])
        eis.append(g["edge_index"] + offset)
        batch.append(torch.full((n,), g_i, dtype=torch.long))
        offset += n
    x = torch.cat(xs, 0)
    edge_index = torch.cat(eis, 1)
    batch = torch.cat(batch, 0)
    return x, edge_index, batch, len(idx)


def build_qm7_equivariant(path=None, cutoff_bohr=3.5, n_max=None):
    """Like build_qm7_graphs but ALSO returns node POSITIONS (Bohr) for the equivariant contract:
    dict(x (n,5) one-hot type, pos (n,3), edge_index (2,|E|)) + targets y."""
    return _build_qm7(path, cutoff_bohr, n_max, with_pos=True)


def collate_equivariant(graphs, idx):
    """Disjoint-union batch for the equivariant schema (includes positions)."""
    import torch

    xs, ps, eis, batch = [], [], [], []
    offset = 0
    for g_i, i in enumerate(idx):
        g = graphs[i]
        n = g["x"].shape[0]
        xs.append(g["x"])
        ps.append(g["pos"])
        eis.append(g["edge_index"] + offset)
        batch.append(torch.full((n,), g_i, dtype=torch.long))
        offset += n
    return (torch.cat(xs, 0), torch.cat(ps, 0), torch.cat(eis, 1), torch.cat(batch, 0), len(idx))

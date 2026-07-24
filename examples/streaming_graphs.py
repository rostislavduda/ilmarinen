"""Runnable example: train a relational (graph) contract by STREAMING graphs from disk.

The relational contracts (graph / equivariant / set) normally pre-convert EVERY graph in the dataset to
tensors up front and keep them for the whole fit, which caps the dataset by host RAM. `AllData.graph_stream`
instead pulls one graph at a time from a `GraphSource`, so a corpus of millions of graphs never needs to be
resident. Here we back it with `LazyGraphSource` over a directory of per-graph `.npy` files -- the realistic
out-of-core pattern (swap the loader for HDF5, a database, or an object store as needed).

This script:
  1. writes each graph (node features + edge_index) to its own .npy files on disk;
  2. trains once RESIDENT (all graphs held in memory) and once STREAMING (one graph loaded at a time);
  3. shows the two fits produce BIT-IDENTICAL weights.

Run from the repository root:

    python -m examples.streaming_graphs
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

from ilmarinen import AllGraph, AllData, LazyGraphSource

warnings.filterwarnings("ignore")                        # silence torch index_reduce beta warning in the demo


def make_graphs(n=240, n_nodes=8, n_feat=4, seed=0):
    """A batch of small graphs: node features + a ring-plus-chord edge list, binary label on feature sum."""
    rng = np.random.RandomState(seed)
    node_feats, edges, ys = [], [], []
    for _ in range(n):
        f = rng.randn(n_nodes, n_feat).astype(np.float32)
        e = [(i, (i + 1) % n_nodes) for i in range(n_nodes)] + [(0, n_nodes // 2)]
        node_feats.append(f)
        edges.append(np.array(e, dtype=np.int64).T)      # (2, E)
        ys.append(int(f[:, 0].sum() > 0))
    return node_feats, edges, np.array(ys, dtype=np.int64)


def weights_identical(net_a, net_b):
    sa, sb = net_a.state_dict(), net_b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    node_feats, edges, y = make_graphs()
    cfg = dict(width=8, depth=1, epochs=8, seed=0, verbose=False)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for i, (f, e) in enumerate(zip(node_feats, edges)):   # one file pair per graph, on disk
            np.save(root / f"node_{i}.npy", f)
            np.save(root / f"edge_{i}.npy", e)
        print(f"wrote {len(node_feats)} graphs to {root.name}/ as per-graph .npy files")

        # ---- RESIDENT: all graphs are converted to tensors up front and kept for the whole fit ----
        mg_resident = AllGraph(**cfg)
        r_res = mg_resident.fit(AllData.graphs(node_feats, edges, y=y), task="classification", n_out=2)

        # ---- STREAMING: LazyGraphSource loads one graph's files on demand via the loader ----
        # The loader returns {'node': ..., 'edge': ...} for graph i; only the minibatch in flight is resident.
        # Declare n / n_in / has_edges up front so the builder reads shapes as metadata (never a fetch).
        def load_graph(i):
            return {"node": np.load(root / f"node_{i}.npy"),
                    "edge": np.load(root / f"edge_{i}.npy")}

        source = LazyGraphSource(load_graph, n=len(node_feats), n_in=node_feats[0].shape[1], has_edges=True)
        mg_stream = AllGraph(**cfg)
        r_stream = mg_stream.fit(
            AllData.graph_stream(source, y=y, kind_hint="graph"),
            task="classification", n_out=2, stream=True,
        )

    print()
    print(f"resident : contract={r_res['contract']:6s} value={r_res['value']:.4f}  n_params={r_res['n_params']}")
    print(f"streaming: contract={r_stream['contract']:6s} value={r_stream['value']:.4f}  n_params={r_stream['n_params']}")
    print()
    identical = weights_identical(mg_resident.net, mg_stream.net)
    print(f"trained weights bit-identical (streaming == resident): {identical}")
    print(f"reported values equal:                                 {r_res['value'] == r_stream['value']}")

    if not identical:
        raise SystemExit("streaming diverged from the resident fit -- this should never happen")
    print("\nOK: streamed one graph at a time from disk and trained the same model as the resident fit.")


if __name__ == "__main__":
    main()

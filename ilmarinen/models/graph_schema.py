"""GRAPH schema -- the arbitrary-topology metamodel, and the first irregular-structure contract
among the eight mutually-exclusive tensor contracts (the first that is not a fixed-locality grid).

Motivation and taxonomy. The four grid contracts act on ORDERED axes with fixed locality:
    schema          : (b, T, n_in)        1D sequence
    spatial_schema  : (b, C, H, W)        2D grid
    volumetric_schema:(b, C, D, H, W)     3D volume
    grid4d_schema   : (b, C, T, D, H, W)  4D grid
A graph has NO ordered axis and NO fixed locality: a node's neighborhood is given by an EDGE SET
supplied per-input at runtime, not by grid position. So the contract is fundamentally different:
    graph: node features X (N, F) + edge_index E (2, |E|)  ->  node embeddings (N, width)
             -> permutation-invariant readout -> graph-level prediction
This cannot be alpha-mixed into any grid cell (grid primitives index neighbors by fixed offset;
graph primitives index them by a runtime edge set), so it is a genuinely distinct metamodel. Message
passing GENERALIZES conv/attention (they are message passing on grid / complete graphs), but it needs
the edge set as input, which the grid contracts do not carry.

Message-passing skeleton (Gilmer et al. 2017). Every GNN layer is, for each node v:
    message   m_{u->v} = f(x_u)          (per neighbor u in N(v))
    aggregate a_v      = AGG_{u} m_{u->v} (permutation-invariant over neighbors)
    update    x_v'     = g(x_v, a_v)
The canonical layers differ MAINLY IN THE AGGREGATOR, and that aggregator is the selection axis this
schema's alpha ranges over -- a genuine antichain (sum/mean/max/attention are provably
non-equivalent in expressive power; sum > mean/max for multiset distinction; attention is anisotropic).

Primitive vocabulary (aggregator-as-primitive + no-message-passing baselines + stabilizer):
    gcn       : degree-normalized MEAN aggregation (isotropic; Kipf-Welling) -- simple strong baseline
    sage      : MAX-pool aggregation with a learnable message (GraphSAGE-style; a non-mean reduction)
    gin       : SUM aggregation + MLP update (Graph Isomorphism Network; maximally expressive MPNN)
    gat       : ATTENTION aggregation -- learned per-neighbor weights (anisotropic; content routing)
    dense     : per-node MLP, NO message passing (ignores edges) -- the no-graph-structure baseline
    norm      : per-node feature norm + linear -- the stabilizer (mitigates over-smoothing)

Graph-level readout (permutation-invariant pool over nodes): 'mean' (size-invariant), 'sum'
(multiset-preserving, GIN-style), or 'max'. Exposed as a builder arg, analogous to the sequence
schema's last/mean/flatten readout.

Batching convention. Variable node counts are handled by a flat batch: node features are concatenated
across a batch of graphs into (sum_i N_i, F), a `batch` vector maps each node to its graph, and
edge_index is offset accordingly. This is the standard PyG-style disjoint-union batching, implemented
here without external dependencies (pure torch scatter via index_add / scatter_reduce).

All prior modules are left UNTOUCHED; this is a new capability in a new module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_lin(lin, sigma_w2=1.0):
    with torch.no_grad():
        lin.weight.normal_(0, np.sqrt(sigma_w2 / lin.in_features))
        if lin.bias is not None:
            lin.bias.zero_()


def _scatter_sum(src, index, n):
    """Sum rows of src (E, F) into an (n, F) output at positions given by index (E,)."""
    out = src.new_zeros(n, src.shape[1])
    out.index_add_(0, index, src)
    return out


def _scatter_mean(src, index, n):
    s = _scatter_sum(src, index, n)
    cnt = src.new_zeros(n).index_add_(0, index, src.new_ones(src.shape[0]))
    return s / cnt.clamp(min=1).unsqueeze(1)


def _scatter_max(src, index, n):
    out = src.new_full((n, src.shape[1]), float("-inf"))
    if hasattr(out, "index_reduce"):
        out = out.index_reduce(0, index, src, "amax", include_self=True)
    else:
        out = _scatter_max_fallback(src, index, n)
    # replace unreached rows (-inf) with 0 WITHOUT in-place ops on the autograd path
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def _scatter_max_fallback(src, index, n):
    out = src.new_full((n, src.shape[1]), float("-inf"))
    for e in range(src.shape[0]):
        out[index[e]] = torch.maximum(out[index[e]], src[e])
    return out


def _scatter_min(src, index, n):
    """Min aggregation, autograd-safe (out-of-place)."""
    out = src.new_full((n, src.shape[1]), float("inf"))
    if hasattr(out, "index_reduce"):
        out = out.index_reduce(0, index, src, "amin", include_self=True)
    else:
        out = src.new_full((n, src.shape[1]), float("inf"))
        for e in range(src.shape[0]):
            out[index[e]] = torch.minimum(out[index[e]], src[e])
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def _scatter_std(src, index, n):
    """Per-node standard deviation of incoming messages (the 4th PNA aggregator). std = sqrt(E[x^2] -
    E[x]^2), clamped for numerical safety."""
    mean = _scatter_mean(src, index, n)
    mean_sq = _scatter_mean(src * src, index, n)
    var = (mean_sq - mean * mean).clamp(min=0.0)
    return torch.sqrt(var + 1e-6)


# --------------------------------------------------------------------------- primitive cores
# Each core: forward_graph(x (N,F), edge_index (2,|E|)) -> (N, width).
# edge_index rows are [src, dst]; messages flow src -> dst.


class _GCNGraph(nn.Module):
    """Degree-normalized mean aggregation (isotropic message passing)."""

    name = "gcn"

    def __init__(self, fin, width):
        super().__init__()
        self.lin = nn.Linear(fin, width)
        _init_lin(self.lin)

    def forward_graph(self, x, edge_index):
        n = x.shape[0]
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        # add self-loops implicitly via +h, then mean over incoming neighbors
        agg = _scatter_mean(h[src], dst, n)
        return h + agg


class _SAGEGraph(nn.Module):
    """GraphSAGE-style: max-pool aggregation of transformed neighbor messages, concat with self."""

    name = "sage"

    def __init__(self, fin, width):
        super().__init__()
        self.msg = nn.Linear(fin, width)
        self.self_lin = nn.Linear(fin, width)
        _init_lin(self.msg)
        _init_lin(self.self_lin)

    def forward_graph(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        m = F.relu(self.msg(x))[src]
        agg = _scatter_max(m, dst, n)
        return self.self_lin(x) + agg


class _GINGraph(nn.Module):
    """Graph Isomorphism Network: sum aggregation + MLP update ( (1+eps)*x_v + sum_u x_u )."""

    name = "gin"

    def __init__(self, fin, width):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(fin, width), nn.ReLU(), nn.Linear(width, width))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                _init_lin(m)

    def forward_graph(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        agg = _scatter_sum(x[src], dst, n)
        return self.mlp((1 + self.eps) * x + agg)


class _PNAGraph(nn.Module):
    """Principal Neighbourhood Aggregation (Corso et al. 2020). Combines MULTIPLE aggregators
    (mean, max, min, std) with degree-SCALERS (identity, amplification, attenuation), then projects.
    A single aggregator provably cannot distinguish certain neighbourhoods (the 1-WL / injectivity gap);
    PNA's multi-aggregator + degree-scaler design is the canonical expressiveness fix, and is strong on
    real molecular/benchmark graphs. This is a genuinely distinct primitive from the single-aggregator
    cores (gcn=mean, sage=mean+max, gin=sum, gat=attention)."""

    name = "pna"

    def __init__(self, fin, width):
        super().__init__()
        self.lin = nn.Linear(fin, width)
        _init_lin(self.lin)
        # 4 aggregators x 3 scalers = 12 combined channels of width, projected back to width
        self.n_agg = 4
        self.n_scale = 3
        self.proj = nn.Linear(width * self.n_agg * self.n_scale, width)
        _init_lin(self.proj)

    def forward_graph(self, x, edge_index):
        n = x.shape[0]
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        m = h[src]  # messages
        # Share the scatter passes across the four aggregators + the degree scaler. mean and std both need
        # the running sum and count, and std also needs sum-of-squares; computing each scatter ONCE and
        # deriving mean = sum/cnt, std = sqrt(E[x^2]-E[x]^2) avoids recomputing the sum/count three times
        # (previously _scatter_mean ran inside both the mean aggregator and _scatter_std, and the degree was
        # scattered a third time).
        ones = m.new_ones(m.shape[0], 1)
        cnt = _scatter_sum(ones, dst, n)  # (n,1) in-degree (== deg)
        cnt_c = cnt.clamp(min=1)
        s = _scatter_sum(m, dst, n)  # (n,width) sum of messages
        ssq = _scatter_sum(m * m, dst, n)  # (n,width) sum of squares
        mean = s / cnt_c
        var = (ssq / cnt_c - mean * mean).clamp(min=0.0)
        std = torch.sqrt(var + 1e-6)
        mx = _scatter_max(m, dst, n)
        mn = _scatter_min(m, dst, n)
        agg = torch.cat([mean, mx, mn, std], dim=1)  # (n, 4*width)
        # degree-scalers: identity, amplification (log(d+1)/delta), attenuation (delta/log(d+1))
        log_deg = torch.log(cnt + 1.0)
        delta = 1.0  # normalization constant (avg log-deg ~1)
        amp = log_deg / delta
        att = delta / (log_deg + 1e-6)
        scaled = torch.cat([agg, agg * amp, agg * att], dim=1)  # (n, 12*width)
        return h + self.proj(scaled)


class _GATGraph(nn.Module):
    """Single-head graph attention: learned per-edge weights (anisotropic aggregation)."""

    name = "gat"

    def __init__(self, fin, width):
        super().__init__()
        self.lin = nn.Linear(fin, width)
        _init_lin(self.lin)
        self.att = nn.Linear(2 * width, 1)
        _init_lin(self.att)

    def forward_graph(self, x, edge_index):
        n = x.shape[0]
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        e = F.leaky_relu(self.att(torch.cat([h[src], h[dst]], dim=1)).squeeze(1), 0.2)  # (E,)
        # softmax over incoming edges per destination node
        e = e - e.max()
        exp = e.exp()
        denom = _scatter_sum(exp.unsqueeze(1), dst, n).squeeze(1).clamp(min=1e-9)
        alpha = exp / denom[dst]
        agg = _scatter_sum(alpha.unsqueeze(1) * h[src], dst, n)
        return h + agg


class _DenseGraph(nn.Module):
    """Per-node MLP with NO message passing (ignores edges) -- the no-graph-structure baseline."""

    name = "dense"

    def __init__(self, fin, width):
        super().__init__()
        self.lin = nn.Linear(fin, width)
        _init_lin(self.lin)

    def forward_graph(self, x, edge_index):
        return self.lin(x)


class _NormGraph(nn.Module):
    """Per-node feature standardization + linear -- the stabilizer (mitigates over-smoothing)."""

    name = "norm"

    def __init__(self, fin, width):
        super().__init__()
        self.lin = nn.Linear(fin, width)
        _init_lin(self.lin)
        self.ln = nn.LayerNorm(width)

    def forward_graph(self, x, edge_index):
        return self.ln(self.lin(x))


_GRAPH_CORES = {
    "gcn": _GCNGraph,
    "sage": _SAGEGraph,
    "gin": _GINGraph,
    "pna": _PNAGraph,
    "gat": _GATGraph,
    "dense": _DenseGraph,
    "norm": _NormGraph,
}


# --------------------------------------------------------------------------- schema cell
class _GraphCell(nn.Module):
    """One meta-layer: all graph primitives in parallel, mixed by softmax(alpha), then ReLU."""

    def __init__(self, fin, width, primitives):
        super().__init__()
        self.primitives = tuple(primitives)
        self.cores = nn.ModuleList([_GRAPH_CORES[p](fin, width) for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def mixed(self, x, edge_index):  # canonical uniform name (delegates to forward)
        return self.forward(x, edge_index)

    def forward(self, x, edge_index):
        outs = torch.stack([c.forward_graph(x, edge_index) for c in self.cores], dim=0)  # (P,N,width)
        w = torch.softmax(self.alpha, dim=0)
        mixed = torch.einsum("p,pnw->nw", w, outs)
        return F.relu(mixed)

    def update_peak(self):
        with torch.no_grad():
            w = torch.softmax(self.alpha, dim=0)
            self.alpha_peak = torch.maximum(self.alpha_peak, w)


def _global_pool(x, batch, n_graphs, mode):
    if mode == "sum":
        return _scatter_sum(x, batch, n_graphs)
    if mode == "max":
        return _scatter_max(x, batch, n_graphs)
    return _scatter_mean(x, batch, n_graphs)


class GraphSchema(nn.Module):
    """Stack of graph meta-layers over an input node-embedding, then a permutation-invariant
    node-readout (sum/mean/max) + linear head for graph-level prediction.

    forward(x, edge_index, batch, n_graphs):
      x          : (N_total, F)   node features (disjoint-union batch of graphs)
      edge_index : (2, |E_total|) edges with node indices into the flat batch
      batch      : (N_total,)     graph id per node
      n_graphs   : int            number of graphs in the batch
      -> (n_graphs, n_out)
    """

    def __init__(
        self,
        fin,
        width=32,
        depth=2,
        n_out=1,
        seed=0,
        primitives=("gcn", "sage", "gin", "gat", "dense", "norm"),
        readout="mean",
    ):
        super().__init__()
        if readout not in ("mean", "sum", "max"):
            raise ValueError("readout must be 'mean', 'sum', or 'max'")
        torch.manual_seed(seed)
        self.primitives = tuple(primitives)
        self.width, self.depth, self.readout = width, depth, readout
        self.embed = nn.Linear(fin, width)
        _init_lin(self.embed)
        self.cells = nn.ModuleList([_GraphCell(width, width, self.primitives) for _ in range(depth)])
        self.head = nn.Linear(width, n_out)
        _init_lin(self.head)

    def forward(self, x, edge_index, batch, n_graphs):
        h = self.embed(x)
        for cell in self.cells:
            h = cell(h, edge_index)
        pooled = _global_pool(h, batch, n_graphs, self.readout)  # (n_graphs, width)
        return self.head(pooled)

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def alpha_report(self):
        return [torch.softmax(c.alpha, dim=0).detach().cpu().numpy() for c in self.cells]

    def alpha_peak_report(self):
        return [c.alpha_peak.detach().cpu().numpy() for c in self.cells]

    def architecture(self):
        return [self.primitives[int(np.argmax(c.alpha_peak.detach().cpu().numpy()))] for c in self.cells]


def build_graph_schema(
    n_in=None,
    width=32,
    depth=2,
    n_out=1,
    seed=0,
    primitives=("gcn", "sage", "gin", "gat", "dense", "norm"),
    readout="mean",
    fin=None,
):
    # canonical param is n_in; fin kept as backward-compatible alias
    if n_in is None:
        n_in = fin
    if n_in is None:
        raise TypeError("build_graph_schema requires n_in (input feature dim)")
    return GraphSchema(
        fin=n_in, width=width, depth=depth, n_out=n_out, seed=seed, primitives=primitives, readout=readout
    )

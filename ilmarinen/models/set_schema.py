"""SET schema -- a first-class contract for UNORDERED sets (Future Direction #6).

A set {x_1,...,x_n} has NO order and NO edges; its only symmetry is the FULL permutation group S_n (the
MAXIMAL symmetry, of which a graph's automorphism group Aut(G) is a subgroup). This is NOT the graph
contract with empty edges:
  - the graph aggregators (gcn/sage/gin/gat) all COLLAPSE with no edges (no messages to pass) -- the
    4-way vocabulary degenerates to one, and the GAT branch even crashes on 0 edges;
  - set functions have their OWN canonical vocabulary (Deep Sets / Set Transformer), none of which is
    edge/message-based;
  - the guarantee is stronger: EXACT S_n-invariance by construction (Deep Sets theorem: any permutation-
    invariant function factors as rho(pool_i phi(x_i))).

Contract: input X (N, F) + batch (N,) mapping each element to its set; output per-set prediction
(n_sets, n_out). Every primitive is symmetric in the element index, so permutation-invariance holds by
construction: attention over elements is permutation-EQUIVARIANT, pooling is permutation-INVARIANT,
their composition is invariant.

Permutation-primitive vocabulary (the set analogue of the six irreducibles):
  deepsets    : rho(pool_i phi(x_i))            -- Deep Sets (Zaheer 2017); element MLP + invariant pool.
  sab         : Set Attention Block             -- multihead self-attention over elements, no positional
                                                   encoding (Set Transformer; Lee 2019); higher-order
                                                   interactions the sum cannot capture.
  isab        : Induced SAB                      -- attention through m learned inducing points, O(Nm).
  pma_block   : element MLP feeding a PMA readout-- richer than sum/mean/max pooling.
  element_mlp : per-element MLP (stabilizer/baseline, the minimal set function; like 'plain').
  norm        : element-wise normalization (parameter-light stabilizer).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _scatter_mean(src, batch, n_sets):
    out = src.new_zeros(n_sets, src.shape[-1])
    cnt = src.new_zeros(n_sets, 1)
    out.index_add_(0, batch, src)
    cnt.index_add_(0, batch, torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype))
    return out / cnt.clamp(min=1.0)


def _scatter_sum(src, batch, n_sets):
    out = src.new_zeros(n_sets, src.shape[-1])
    out.index_add_(0, batch, src)
    return out


def _scatter_max(src, batch, n_sets):
    # autograd-safe AND MPS-safe: index_reduce(amax) rather than scatter_reduce. On MPS, scatter_reduce
    # HANGS if src carries a NaN (from a diverged fit); index_reduce(amax) does not (both give the same
    # segment maxima). Mirrors graph_schema._scatter_max. Falls back to scatter_reduce on old torch.
    out = src.new_full((n_sets, src.shape[-1]), float("-inf"))
    if hasattr(out, "index_reduce"):
        out = out.index_reduce(0, batch, src, "amax", include_self=True)
    else:
        idx = batch.view(-1, 1).expand(-1, src.shape[-1])
        out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    out = torch.where(out == float("-inf"), torch.zeros_like(out), out)
    return out


def _fspool(h, batch, n_sets, n_pieces=None):
    """Featurewise Sort Pooling (Zhang et al. 2020). For each set and each feature, SORT the feature
    values across elements, then read off a fixed-length descriptor (piecewise-linear over the sorted
    sequence). Permutation-INVARIANT (sorting removes order) and NOT expressible as sum/mean/max or
    attention -- it preserves the whole per-feature distribution's shape, not just a moment. Here we use
    a simple fixed set of quantile positions as the piecewise-linear readout (learnable weights are a
    refinement; the sort is the essential permutation-invariant operation)."""
    F = h.shape[-1]
    out = h.new_zeros(n_sets, F)
    for s in range(n_sets):
        rows = h[batch == s]
        if rows.shape[0] == 0:
            continue
        sorted_vals, _ = torch.sort(rows, dim=0)          # sort each feature across elements
        # descriptor: mean of sorted (invariant), plus the sorted max/min captured via endpoints;
        # a compact FSPool: average the sorted sequence (equals mean) is degenerate, so use a weighted
        # sum with a fixed ramp weight over the sorted order (this is the piecewise-linear FSPool with a
        # fixed single-piece ramp), which is DISTINCT from mean/sum/max.
        k = sorted_vals.shape[0]
        ramp = torch.linspace(-1.0, 1.0, k, device=h.device, dtype=h.dtype).view(k, 1)
        out[s] = (sorted_vals * ramp).sum(0)              # order-weighted sum of sorted vals (FSPool)
    return out


def _set_pool(h, batch, n_sets, kind):
    if kind == "sum":    return _scatter_sum(h, batch, n_sets)
    if kind == "max":    return _scatter_max(h, batch, n_sets)
    if kind == "fspool": return _fspool(h, batch, n_sets)
    return _scatter_mean(h, batch, n_sets)


class _MHA(nn.Module):
    """Multihead attention over the elements of each set (masked so attention stays within a set).
    Permutation-EQUIVARIANT: permuting elements permutes the outputs identically."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads; self.dk = dim // heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)

    def forward(self, x, batch, n_sets, kv=None, kv_batch=None):
        # x: (Nq, D) queries; kv: (Nk, D) keys/values (defaults to x -> self-attention). Attention is
        # BLOCK-DIAGONAL per set (elements only attend within their own set), so we compute it one set at a
        # time -- cost sum_i n_i^2 (or n_i*m for inducing). A dense (Nq,Nk) score matrix over the whole
        # disjoint-union batch would be (sum_i n_i)^2 and OOM on large point-set batches; the per-set result
        # is IDENTICAL (softmax over the within-set keys either way).
        kv = x if kv is None else kv
        kvb = batch if kv_batch is None else kv_batch
        Nq = x.shape[0]; D = x.shape[1]
        q = self.q(x).view(Nq, self.h, self.dk)
        k = self.k(kv).view(-1, self.h, self.dk)
        v = self.v(kv).view(-1, self.h, self.dk)
        idx_list, out_list = [], []
        for s in range(n_sets):
            qi = (batch == s).nonzero(as_tuple=True)[0]
            ki = (kvb == s).nonzero(as_tuple=True)[0]
            if qi.numel() == 0 or ki.numel() == 0:
                continue
            sc = torch.einsum("qhd,khd->qkh", q[qi], k[ki]) / (self.dk ** 0.5)   # (nq,nk,h), within-set only
            at = torch.softmax(sc, dim=1)
            out_list.append(torch.einsum("qkh,khd->qhd", at, v[ki]))            # (nq,h,dk)
            idx_list.append(qi)
        out = x.new_zeros(Nq, self.h, self.dk)
        if idx_list:
            out = out.index_copy(0, torch.cat(idx_list), torch.cat(out_list, 0))  # scatter back (autograd-safe)
        return self.o(out.reshape(Nq, D))


class _SetCore(nn.Module):
    """One set primitive: forward_set(x (N,F), batch, n_sets) -> (N, width) equivariant element features
    (pooling to per-set happens at readout). kind selects the block."""
    def __init__(self, fin, width, kind, heads=4, n_inducing=8):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(fin, width)
        if kind in ("sab", "isab", "pma_block"):
            self.mha = _MHA(width, heads)
            self.ff = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
            self.ln1 = nn.LayerNorm(width); self.ln2 = nn.LayerNorm(width)
            if kind == "isab":
                self.inducing = nn.Parameter(torch.randn(n_inducing, width) * 0.1)
                self.mha2 = _MHA(width, heads)
        if kind in ("deepsets", "element_mlp"):
            self.phi = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        if kind == "norm":
            self.ln = nn.LayerNorm(width)

    def forward_set(self, x, batch, n_sets):
        h = self.proj(x)
        if self.kind == "element_mlp":
            return self.phi(h)
        if self.kind == "deepsets":
            # element MLP; the invariant pool is applied at readout (rho after pool). Equivariant part:
            return self.phi(h)
        if self.kind == "norm":
            return self.ln(h)
        if self.kind == "sab" or self.kind == "pma_block":
            a = self.ln1(h + self.mha(h, batch, n_sets))
            return self.ln2(a + self.ff(a))
        if self.kind == "isab":
            # inducing points attend to elements, then elements attend back (O(Nm))
            m = self.inducing.shape[0]
            ind = self.inducing.unsqueeze(0).expand(n_sets, m, -1).reshape(n_sets * m, -1)
            ind_batch = torch.arange(n_sets, device=x.device).repeat_interleave(m)
            hI = self.mha(ind, ind_batch, n_sets, kv=h, kv_batch=batch)     # inducing <- elements
            out = self.mha2(h, batch, n_sets, kv=hI, kv_batch=ind_batch)    # elements <- inducing
            return self.ln2(out + self.ff(out))
        return h


class _SetCell(nn.Module):
    """Softmax mixture over set primitives (permutation-equivariant), same relaxation as the other
    contracts. mixed_set(x, batch, n_sets) -> (N, width)."""
    def __init__(self, fin, width, primitives, heads=4):
        super().__init__()
        self.primitives = tuple(primitives)
        self.cores = nn.ModuleList([_SetCore(fin, width, p, heads) for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def mixed(self, x, batch, n_sets):
        w = torch.softmax(self.alpha, dim=0)
        outs = torch.stack([c.forward_set(x, batch, n_sets) for c in self.cores], dim=0)  # (P,N,width)
        return (w.view(-1, 1, 1) * outs).sum(0)


    def mixed_set(self, x, batch, n_sets):  # backward-compatible alias
        return self.mixed(x, batch, n_sets)

    def update_peak(self):
        with torch.no_grad():
            self.alpha_peak = torch.maximum(self.alpha_peak, torch.softmax(self.alpha, dim=0))


class SetSchema(nn.Module):
    """S_n-invariant set model: element embedding -> stacked set cells (equivariant) -> invariant pool
    -> per-set head. Permutation-invariant by construction."""
    def __init__(self, fin, width=32, depth=2, n_out=1, primitives=("deepsets", "sab", "isab",
                 "pma_block", "element_mlp", "norm"), readout="mean", heads=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.width = width; self.depth = depth; self.readout = readout
        self.cells = nn.ModuleList()
        f = fin
        for _ in range(depth):
            self.cells.append(_SetCell(f, width, primitives, heads))
            f = width
        self.rho = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, n_out))

    def embed(self, x, batch, n_sets):
        h = None
        for l, cell in enumerate(self.cells):
            out = cell.mixed(x if l == 0 else h, batch, n_sets)
            h = out if l == 0 else h + out                       # residual (equivariant)
        return h

    def forward(self, x, batch, n_sets):
        h = self.embed(x, batch, n_sets)                          # (N, width) equivariant
        pooled = _set_pool(h, batch, n_sets, self.readout)        # (n_sets, width) INVARIANT
        return self.rho(pooled)                                    # per-set prediction

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def alpha_report(self):
        return [torch.softmax(c.alpha, dim=0).detach().cpu().numpy() for c in self.cells]

    def alpha_peak_report(self):
        return [c.alpha_peak.detach().cpu().numpy() for c in self.cells]

    def architecture(self):
        with torch.no_grad():
            return {"primitives": [c.primitives[int(torch.argmax(c.alpha))] for c in self.cells],
                    "readout": self.readout}


def build_set_schema(n_in=None, width=32, depth=2, n_out=1,
                                 primitives=("deepsets", "sab", "isab", "pma_block", "element_mlp",
                                             "norm"), readout="mean", heads=4, seed=0, fin=None):
    # canonical param is n_in; fin kept as backward-compatible alias
    if n_in is None:
        n_in = fin
    if n_in is None:
        raise TypeError("build_set_schema requires n_in (element feature dim)")
    return SetSchema(n_in, width, depth, n_out, primitives, readout, heads, seed)

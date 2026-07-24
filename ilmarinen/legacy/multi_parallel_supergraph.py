"""Generalized N-way parallel supergraph (adds ATTENTION as a third primitive).

The original ParallelSuperGraph (parallel_supergraph.py) is hardwired to the 2-way
{conv, spectral} choice: every method threads exactly two outputs and mixes w[0]*..+w[1]*..
This module GENERALIZES that to an arbitrary list of parallel primitives, adding
ATTENTION -- content-based normalized routing (primitive #5 of the minimal six), the one
primitive whose connectivity is DATA-DEPENDENT. The validated 2-way class is left untouched.

The three primitives are a genuine ANTICHAIN by their routing structure:
- conv      = FIXED LOCAL routing (translation-equivariant; a fixed local stencil).
- spectral  = FIXED GLOBAL routing in the FREQUENCY basis (position-invariant power).
- attention = DATA-DEPENDENT GLOBAL routing (softmax over content-based scores). Unlike
              conv (fixed neighborhood) and spectral (fixed frequency mixing), attention's
              aggregation weights are COMPUTED FROM THE INPUT -- the primitive that makes
              the connectivity itself input-dependent. No fixed-kernel primitive emulates
              it and it does not reduce to either.

Input convention: the flat n_in-vector is treated as a SEQUENCE of `attn_tokens` tokens
(reshaped positions x features) for the attention primitive, so self-attention can route
over positions. conv and spectral keep their existing flat-signal readouts.

Design invariants (carried from the validated parallel + recurrent supergraphs):
- FUNCTIONAL composition across depth (layer 1 reads layer 0's alpha-mixed output) -> the
  loss is non-additive across layers, coupling the selection fields (J != 0).
- SHARED per-layer head for the alpha-mix.
- Per-layer N-way alpha; deep-supervision option so early layers get a bare field.
- Peak-alpha tracking (the honest selection signal on moderate-margin tasks).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .networks import _init_linear


class _ConvCore(nn.Module):
    """Fixed-local (position-basis) primitive: 1D conv + translation-invariant pooling."""

    name = "conv"

    def __init__(self, n_in, width, n_chan=16, ksize=7):
        super().__init__()
        self.conv = nn.Conv1d(1, n_chan, kernel_size=ksize, padding=ksize // 2)
        self.proj = nn.Linear(2 * n_chan, width)
        _init_linear(self.proj, 1.0, 0.0)
        self.n_in, self.width = n_in, width

    def forward(self, x):  # x: (b, n_in)
        c = torch.relu(self.conv(x.unsqueeze(1)))
        feat = torch.cat([c.mean(dim=2), c.amax(dim=2)], dim=1)  # amax = MPS-safe over NaN
        return torch.tanh(self.proj(feat))


class _SpectralCore(nn.Module):
    """Fixed-global (frequency-basis) primitive: rFFT -> per-frequency log-power -> proj."""

    name = "spectral"

    def __init__(self, n_in, width):
        super().__init__()
        self.nf = n_in // 2 + 1
        self.proj = nn.Linear(self.nf, width)
        _init_linear(self.proj, 1.0, 0.0)
        self.n_in, self.width = n_in, width

    def forward(self, x):  # x: (b, n_in)
        Xf = torch.fft.rfft(x, dim=1)
        power = Xf.real ** 2 + Xf.imag ** 2
        return torch.tanh(self.proj(torch.log1p(power)))


class _AttentionCore(nn.Module):
    """Content-based routing (attention) primitive -- DATA-DEPENDENT global aggregation.

    The flat n_in-vector is reshaped into `n_tok` tokens of dim `d_tok` (n_in = n_tok*d_tok,
    padded if needed). Single-head self-attention computes softmax(QK^T/sqrt(d)) V -- routing
    weights COMPUTED FROM THE INPUT -- then mean-pools over tokens and projects to `width`.

    This is the irreducible primitive #5: its aggregation is a normalized convex combination
    over the token set whose weights are a learned function of the tokens (permutation-
    equivariant relational routing). Neither conv (fixed local stencil) nor spectral (fixed
    frequency mixing) can represent input-dependent routing.
    """

    name = "attention"

    def __init__(self, n_in, width, n_tok=16, d_model=32):
        super().__init__()
        # choose token layout: n_tok tokens, each of dim d_tok = ceil(n_in / n_tok)
        self.n_tok = n_tok
        self.d_tok = (n_in + n_tok - 1) // n_tok
        self.pad = self.n_tok * self.d_tok - n_in
        self.d_model = d_model
        self.embed = nn.Linear(self.d_tok, d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, width)
        for lin in (self.embed, self.Wq, self.Wk, self.Wv, self.proj):
            _init_linear(lin, 1.0, 0.0)
        self.n_in, self.width = n_in, width

    def forward(self, x):  # x: (b, n_in)
        b = x.shape[0]
        if self.pad:
            x = torch.cat([x, x.new_zeros(b, self.pad)], dim=1)
        tok = x.view(b, self.n_tok, self.d_tok)          # (b, n_tok, d_tok)
        e = torch.tanh(self.embed(tok))                  # (b, n_tok, d_model)
        q, k, v = self.Wq(e), self.Wk(e), self.Wv(e)     # (b, n_tok, d_model)
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)             # (b, n_tok, n_tok) data-dependent
        ctx = torch.matmul(attn, v)                      # (b, n_tok, d_model)
        pooled = ctx.mean(dim=1)                         # (b, d_model) permutation-invariant
        return torch.tanh(self.proj(pooled))


class _DenseCore(nn.Module):
    """Dense/affine primitive (#1 of the minimal six): a full affine map on the flat input.

    Unlike conv (pooling -> translation-invariant, lossy) and spectral (power -> position-
    invariant, lossy), the dense readout keeps ALL input coordinates -- no invariant is
    imposed. It is the least-biased aggregator and the natural optimum for tasks without
    spatial/frequency/relational structure (e.g. flat FashionMNIST, where dense beats the
    lossy-readout primitives). Carries the most parameters (n_in*width) of the primitives.
    """

    name = "dense"

    def __init__(self, n_in, width):
        super().__init__()
        self.fc = nn.Linear(n_in, width)
        _init_linear(self.fc, 1.0, 0.0)
        self.n_in, self.width = n_in, width

    def forward(self, x):  # x: (b, n_in)
        return torch.tanh(self.fc(x))


class _NormCore(nn.Module):
    """Normalization primitive (#6 of the minimal six): standardize over the feature slice,
    then an affine map. x -> LayerNorm(x) @ W.

    Normalization is the STABILIZER of the set: (x - mu)/sigma over the feature dimension
    pins the activation scale, doing at runtime what critical initialization does at t=0
    (the two are substitutes for signal propagation through depth). Unlike the other
    primitives it carries no routing/mixing content of its own -- as a value transform it is
    a normalized affine map. Its distinctive value is DEPTH STABILITY: a normalized stack
    stays on the critical fixed point where an un-normalized stack drifts off it.

    Implemented as LayerNorm over the input features followed by a learnable affine to width
    (so it is a usable value-transform slot in the mix, while its normalization is the point).
    """

    name = "norm"

    def __init__(self, n_in, width):
        super().__init__()
        self.ln = nn.LayerNorm(n_in)
        self.proj = nn.Linear(n_in, width)
        _init_linear(self.proj, 1.0, 0.0)
        self.n_in, self.width = n_in, width

    def forward(self, x):  # x: (b, n_in)
        return torch.tanh(self.proj(self.ln(x)))


_PARALLEL_CORES = {"conv": _ConvCore, "spectral": _SpectralCore,
                   "attention": _AttentionCore, "dense": _DenseCore, "norm": _NormCore}


class MultiCellPar(nn.Module):
    """A meta-cell holding N parallel primitives (stateless). alpha mixes their outputs."""

    def __init__(self, n_in, width, primitives, attn_tokens=16):
        super().__init__()
        self.primitives = list(primitives)
        cores = []
        for p in primitives:
            if p == "attention":
                cores.append(_PARALLEL_CORES[p](n_in, width, n_tok=attn_tokens))
            else:
                cores.append(_PARALLEL_CORES[p](n_in, width))
        self.cores = nn.ModuleList(cores)
        self.alpha = nn.Parameter(torch.zeros(len(primitives)))
        self.register_buffer("_alpha_peak", torch.full((len(primitives),),
                                                       1.0 / len(primitives)))

    def outputs(self, h):
        """All primitives read the same (functionally composed) input h."""
        return [core(h) for core in self.cores]

    def update_peak(self):
        with torch.no_grad():
            cur = torch.softmax(self.alpha, dim=0)
            if cur.max() > self._alpha_peak.max():
                self._alpha_peak = cur.clone()

    def alpha_weights(self):
        with torch.no_grad():
            return torch.softmax(self.alpha, dim=0).cpu().numpy()

    def alpha_peak(self):
        return self._alpha_peak.cpu().numpy()


class MultiParallelSuperGraph(nn.Module):
    """N-way parallel supergraph with functional composition over {conv, spectral, attention}.

    Layer 0 reads the raw input; layer ell reads the previous layer's alpha-mixed output
    (functional composition -> non-additive loss -> coupled selection fields). Mixing is an
    N-way softmax over the primitives through a shared per-layer head.
    """

    def __init__(self, depth, width, n_in=128, n_out=2, seed=0, deep_supervision=False,
                 primitives=("conv", "spectral", "attention"), attn_tokens=16):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        for p in primitives:
            if p not in _PARALLEL_CORES:
                raise ValueError(f"unknown primitive {p!r}; have {list(_PARALLEL_CORES)}")
        torch.manual_seed(seed)
        self.depth, self.width = depth, width
        self.primitives = list(primitives)
        self.deep_supervision = deep_supervision
        cells = []
        din = n_in
        for i in range(depth):
            # layer 0 tokenizes the raw input to attn_tokens; deeper layers operate on the
            # width-dim mixed representation, tokenized into a modest number of tokens.
            tk = attn_tokens if i == 0 else min(attn_tokens, width)
            cells.append(MultiCellPar(din, width, primitives, attn_tokens=tk))
            din = width
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(width, n_out)
        _init_linear(self.head, 1.0, 0.0)
        if deep_supervision:
            early = [nn.Linear(width, n_out) for _ in range(depth - 1)]
            for h in early:
                _init_linear(h, 1.0, 0.0)
            self._early_aux = nn.ModuleList(early)
            self.aux_heads = list(self._early_aux) + [self.head]
        else:
            self._early_aux = None
            self.aux_heads = None

    def _layer_outputs(self, x):
        mixed = []
        h = x
        for cell in self.cells:
            outs = cell.outputs(h)
            w = torch.softmax(cell.alpha, dim=0)
            m = sum(w[i] * outs[i] for i in range(len(self.primitives)))
            mixed.append(m)
            h = m
        return mixed

    def forward(self, x):
        return self.head(self._layer_outputs(x)[-1])

    def forward_all_layers(self, x):
        if self.aux_heads is None:
            raise RuntimeError("forward_all_layers requires deep_supervision=True")
        mixed = self._layer_outputs(x)
        return [self.aux_heads[l](mixed[l]) for l in range(self.depth)]

    def alpha_entropy(self):
        ent = self.cells[0].alpha.new_zeros(())
        for cell in self.cells:
            mu = torch.softmax(cell.alpha, dim=0)
            ent = ent - (mu * torch.log(mu + 1e-9)).sum()
        return ent

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def alpha_report(self):
        """Per-layer softmax(alpha) over primitives (order = self.primitives)."""
        return [cell.alpha_weights() for cell in self.cells]

    def alpha_peak_report(self):
        return [cell.alpha_peak() for cell in self.cells]

    def selected_primitive(self):
        w = self.cells[-1].alpha_peak()
        return self.primitives[int(np.argmax(w))]


def build_multi_parallel_supergraph(**kwargs):
    return MultiParallelSuperGraph(**kwargs)

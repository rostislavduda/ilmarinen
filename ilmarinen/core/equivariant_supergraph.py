"""Equivariant supergraph: closes the Family-2 discovery->architecture loop.

Given a generator L discovered by core/symmetry_discovery.py, this supergraph lets the
metaoptimizer choose, via the usual softmax-alpha, between:
  - `equivariant` : a layer constrained to the commutant of L (weight-sharing under exp(theta L))
  - `dense`       : an unconstrained linear layer (no symmetry bias)
so that the metaoptimizer DISCOVERS whether the data's symmetry (encoded by L) is worth imposing.
On data that genuinely has the symmetry, the equivariant primitive should generalize better and be
selected; on data without it, dense should win. This is the symmetry-analogue of the conv-vs-dense
selection, but with the group discovered from data rather than hand-specified.

Interface: vector inputs (batch, in_channels, n), where n = dim of the generator's coordinate
space. This is the natural home for a discovered symmetry, which acts on the coordinate space.

STATUS (documented, validated, but not on the live AllGraph path). This is the Family-2 predecessor
of today's discovered-symmetry mechanism -- the single-generator commutant + invariant-vs-dense alpha
selection, validated end-to-end on QM7 SO(3) (alpha_inv=0.61 at n=800; see the implementation report,
"Real-data validation on QM7"). The live AllGraph controller has SUPERSEDED it with the more general
emlp_layer path: `_discover_equivariant_group` -> extended_groups.discover_group -> emlp_layer.
EquivariantMLP, deployed as the `generated_equivariant` contract (and the nonlinear `latent_equivariant`
variant). That path generalizes the SAME commutant/Lie-derivative nullspace idea from a single generator
to a SET of generators with direct-sum reps and indefinite metrics (O(p,q)/U/Sp/SL, Lorentz O(1,3)).
This module is retained as the documented, validated Family-2 realization; it is exported but no longer
invoked by fit().
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .equivariant_layer import EquivariantLayer


class _DenseVec(nn.Module):
    """Unconstrained linear map over flattened (channels*n) input -> (out_channels*n)."""

    def __init__(self, n, in_channels, out_channels, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n, self.out_ch = n, out_channels
        self.lin = nn.Linear(in_channels * n, out_channels * n)

    def forward(self, x):                              # x: (b, in_ch, n)
        b = x.shape[0]
        return self.lin(x.reshape(b, -1)).reshape(b, self.out_ch, self.n)


class EquivariantSuperCell(nn.Module):
    """Mix {equivariant-to-L, dense} by softmax(alpha), then nonlinearity."""

    def __init__(self, L, n, in_channels, out_channels, seed=0):
        super().__init__()
        self.primitives = ("equivariant", "dense")
        self.equiv = EquivariantLayer(L, in_channels, out_channels, seed=seed)
        self.dense = _DenseVec(n, in_channels, out_channels, seed=seed)
        self.alpha = nn.Parameter(torch.zeros(2))
        self.register_buffer("alpha_peak", torch.zeros(2))
        self.act = nn.Tanh()

    def forward(self, x):
        w = torch.softmax(self.alpha, dim=0)
        return self.act(w[0] * self.equiv(x) + w[1] * self.dense(x))

    def update_peak(self):
        with torch.no_grad():
            self.alpha_peak = torch.maximum(self.alpha_peak, torch.softmax(self.alpha, dim=0))

    def alpha_weights(self):
        with torch.no_grad():
            return torch.softmax(self.alpha, dim=0).cpu().numpy()


class EquivariantSuperGraph(nn.Module):
    """One equivariant super-cell + a pooling readout to class logits.

    L         : discovered generator (n x n)
    n         : coordinate dimension (= L.shape[0])
    channels  : hidden channel multiplicity (each channel is an n-vector the symmetry acts on)
    n_classes : output classes
    """

    def __init__(self, L, channels=8, n_classes=2, seed=0):
        super().__init__()
        self.n = int(np.asarray(L).shape[0])
        self.channels = channels
        # lift raw input (b, n) -> (b, channels, n) equivariantly: replicate then equivariant-mix
        self.cell = EquivariantSuperCell(L, self.n, channels, channels, seed=seed)
        self.head = nn.Linear(channels * self.n, n_classes)

    def forward(self, x):
        # x: (b, n)  -> lift to (b, channels, n) by broadcasting, then the super-cell
        b = x.shape[0]
        xin = x.unsqueeze(1).expand(b, self.channels, self.n)
        h = self.cell(xin)                           # (b, channels, n)
        return self.head(h.reshape(b, -1))

    def update_peak(self):
        self.cell.update_peak()

    def architecture(self):
        peak = self.cell.alpha_peak.detach().cpu().numpy()
        return self.cell.primitives[int(np.argmax(peak))]

    def alpha_report(self):
        return self.cell.alpha_weights()


def build_equivariant_supergraph(L, channels=8, n_classes=2, seed=0):
    return EquivariantSuperGraph(L, channels=channels, n_classes=n_classes, seed=seed)

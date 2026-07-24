"""Parallel conv|spectral supergraph for the beta*J=1 coupling-transition substrate.

Non-recurrent (stateless) two-layer supergraph over the INCOMPARABLE primitive pair
{conv, spectral}. Unlike the recurrent hetero supergraph, there is no state to thread; the
subtlety here is FUNCTIONAL COMPOSITION -- layer 1 operates on layer 0's OUTPUT
representation, so the loss is NON-ADDITIVE across the two layers and the coupling J
between the two selection fields is nonzero. (If instead both primitives read the raw input
and their outputs were merely concatenated, the loss would be additive and J=0 -- a trivially
decoupled system with no transition.)

Primitive pair (conjugate-basis, mutually incomparable by the uncertainty principle):
- conv     = diagonal in the POSITION basis (local, translation-equivariant); invariant
             readout = mean+max pool over position (translation-invariant).
- spectral = diagonal in the FREQUENCY basis (rFFT -> per-frequency power); invariant
             readout = log-power spectrum (position-invariant).
Validated as a genuine antichain (double dissociation at the detection margin): conv wins
position-local features, spectral wins frequency-local features, with neither dominating.

Design invariants (carried over from the validated recurrent supergraph):
- PRODUCT PATHS across depth: each primitive path stays clean; layer 1's conv reads the
  functional composition of layer 0's mixed output, likewise its spectral.
- SHARED per-layer head for the alpha-mix at readout (separate heads let head magnitude
  absorb selection, degenerating the alpha signal).
- Per-layer alpha^(ell): both layers get an independent, identified selection field.
- Deep supervision option so BOTH fields get a bare field (else the first layer's field is
  screened by the second -- the coupled-Ising field-screening effect).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .networks import _init_linear


class _ConvCore(nn.Module):
    """Position-basis primitive: 1D conv + translation-invariant (mean+max) pooling.

    Maps a length-`n_in` feature vector to a `width` vector. The pooling makes the readout
    translation-invariant, committing this primitive to the position basis (it sees local
    waveform shape, discards absolute position/phase).
    """

    def __init__(self, n_in, width, n_chan=16, ksize=7):
        super().__init__()
        self.conv = nn.Conv1d(1, n_chan, kernel_size=ksize, padding=ksize // 2)
        self.proj = nn.Linear(2 * n_chan, width)
        _init_linear(self.proj, 1.0, 0.0)
        self.n_in, self.width = n_in, width

    def forward(self, x):  # x: (b, n_in)
        c = torch.relu(self.conv(x.unsqueeze(1)))          # (b, n_chan, n_in)
        feat = torch.cat([c.mean(dim=2), c.amax(dim=2)], dim=1)  # (b, 2*n_chan); amax = MPS-safe over NaN
        return torch.tanh(self.proj(feat))


class _SpectralCore(nn.Module):
    """Frequency-basis primitive: rFFT -> per-frequency log-power -> projection.

    The power spectrum is position-invariant, committing this primitive to the frequency
    basis (it sees which frequencies are present, discards position). A learnable per-
    frequency gain would add parameters; the discriminative content is already in the
    log-power, which the projection reads.
    """

    def __init__(self, n_in, width):
        super().__init__()
        self.nf = n_in // 2 + 1
        self.proj = nn.Linear(self.nf, width)
        _init_linear(self.proj, 1.0, 0.0)
        self.n_in, self.width = n_in, width

    def forward(self, x):  # x: (b, n_in)
        Xf = torch.fft.rfft(x, dim=1)
        power = Xf.real ** 2 + Xf.imag ** 2                # (b, nf), position-invariant
        return torch.tanh(self.proj(torch.log1p(power)))


class HeteroCellPar(nn.Module):
    """A meta-cell holding conv and spectral primitives (stateless, parallel).

    Both primitives act on the layer's input; the owning network alpha-mixes their outputs.
    alpha = [conv_weight, spectral_weight].
    """

    def __init__(self, n_in, width):
        super().__init__()
        self.conv = _ConvCore(n_in, width)
        self.spec = _SpectralCore(n_in, width)
        self.alpha = nn.Parameter(torch.zeros(2))   # [conv, spectral]; uniform -> unbiased
        self.register_buffer("_alpha_peak", torch.tensor([0.5, 0.5]))

    def outputs(self, x_conv, x_spec):
        """Product paths: conv reads the conv-path input, spectral the spectral-path input."""
        return self.conv(x_conv), self.spec(x_spec)

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


class ParallelSuperGraph(nn.Module):
    """2-layer parallel conv|spectral supergraph with FUNCTIONAL composition.

    Layer 0 reads the raw input; layer 1 reads layer 0's (alpha-mixed) output -- functional
    composition, so the loss is non-additive across layers and the two selection fields
    couple (J != 0). Product paths keep each primitive's path clean across depth: the conv
    path and spectral path each carry their own representation forward, and each layer's two
    primitives read the appropriate path input.

    NOTE on composition + product paths: to make layer 1 a genuine function of layer 0's
    SELECTED representation (the thing the loss sees), the forward feeds each layer-1
    primitive the alpha-mixed layer-0 output. This is what couples the fields: layer 1's
    best primitive depends on what layer 0 produced, which depends on layer 0's alpha.
    """

    def __init__(self, depth, width, n_in=128, n_out=2, seed=0, deep_supervision=False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        torch.manual_seed(seed)
        self.depth, self.width = depth, width
        self.deep_supervision = deep_supervision
        cells = []
        din = n_in
        for _ in range(depth):
            cells.append(HeteroCellPar(din, width))
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
        """Return the alpha-mixed output of each layer (list length `depth`).

        Functional composition: layer ell's input is the previous layer's mixed output.
        Product paths: within a layer, conv reads the mixed input and spectral reads the
        mixed input (stateless, so both read the same composed representation -- the path
        cleanliness here is across DEPTH via the mixed representation, there being no
        temporal state to keep separate)."""
        mixed = []
        h = x
        for l, cell in enumerate(self.cells):
            oc, os = cell.outputs(h, h)
            w = torch.softmax(cell.alpha, dim=0)
            m = w[0] * oc + w[1] * os
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
        """Per-layer softmax(alpha) = [conv_weight, spectral_weight]."""
        return [cell.alpha_weights() for cell in self.cells]

    def alpha_peak_report(self):
        return [cell.alpha_peak() for cell in self.cells]


def build_parallel_supergraph(**kwargs):
    return ParallelSuperGraph(**kwargs)

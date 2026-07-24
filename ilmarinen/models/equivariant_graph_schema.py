"""EQUIVARIANT-GRAPH schema -- SO(3)-equivariant message passing on molecular/point graphs.
The 5th metamodel: distinguished not by tensor RANK (like sequence/spatial/volumetric) but by the
GROUP-REPRESENTATION STRUCTURE of its features. Node features are STEERABLE (irrep-typed): a direct
sum of a type-0 part (scalars, rotation-INVARIANT) and a type-1 part (vectors, that ROTATE with the
input). Every operation commutes with the SO(3) action, so predictions of invariant targets (energy)
are invariant and predictions of equivariant targets (forces, dipoles) rotate correctly.

WHY A SEPARATE SCHEMA (see tests/equivariant_supergraph_design.md). Equivariant primitives cannot
be alpha-mixed into the plain graph schema: (1) its cell sums primitive outputs, but you cannot
add a type-1 (vector) output to a type-0 (scalar) output; (2) its ReLU is not equivariant on vector
features (ReLU(Rv) != R ReLU(v)). Equivariant nets require per-irrep mixing and NORM-GATED
nonlinearities. All primitives HERE share the same steerable contract, so alpha-mixing among them IS
valid (mixing within an irrep type, and gated nonlinearities, are equivariant).

Contract:
    forward(s (N,C0), v (N,C1,3), pos (N,3), edge_index (2,|E|), batch (N,), n_graphs)
      s   : type-0 node features (scalars)          -- e.g. one-hot atom type embedded
      v   : type-1 node features (vectors), init 0  -- built up by tensor-product messages
      pos : node POSITIONS (required; equivariance is about how outputs transform when pos rotates)
      -> invariant graph readout (n_graphs, n_out)  (type-0 pooled), for energy-like targets.
    Equivariant (type-1) readout for forces/dipoles is available via forward_vector().

Feature layout: scalars s (N, C0); vectors v (N, C1, 3). Irreps l in {0, 1} (the minimal complete
algebra: 0x0->0, 1x1->0 [dot], 0x1->1 [scale], 1x1->1 [cross]). This is the NequIP/MACE/Tensor-Field-
Networks class restricted to l<=1, with Clebsch-Gordan tensor products steered by the l=0,1 spherical
harmonics of edge directions.

Relationship to the symmetry FRONT-END. ilmarinen's symmetry_pipeline discovers a group and QUOTIENTS
it out (invariance, before the model). This schema keeps the group structure INSIDE the model
(equivariance, through depth). core/equivariant_layer.py (commutant [W,L]=0) is the linear-equivariant
seed; this generalizes it to message passing + tensor-product nonlinearities on graphs.

All prior modules are UNTOUCHED; this is a new capability in a new module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _init_lin(lin, sigma_w2=1.0):
    with torch.no_grad():
        lin.weight.normal_(0, np.sqrt(sigma_w2 / lin.in_features))
        if lin.bias is not None:
            lin.bias.zero_()


def _scatter_sum(src, index, n):
    out = src.new_zeros((n,) + src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


# --------------------------------------------------------------------------- equivariant building blocks
class _EquivLinear(nn.Module):
    """Per-irrep linear self-interaction: mixes channels WITHIN each l (the weight is shared across
    the 2l+1 components of a type, which is what makes it equivariant). type-preserving affine."""

    def __init__(self, c0_in, c1_in, c0_out, c1_out):
        super().__init__()
        self.Ws = nn.Parameter(torch.empty(c0_in, c0_out))
        self.bs = nn.Parameter(torch.zeros(c0_out))
        self.Wv = nn.Parameter(torch.empty(c1_in, c1_out)) if c1_in > 0 and c1_out > 0 else None
        nn.init.normal_(self.Ws, 0, 1.0 / max(c0_in, 1) ** 0.5)
        if self.Wv is not None:
            nn.init.normal_(self.Wv, 0, 1.0 / max(c1_in, 1) ** 0.5)
        self.c1_out = c1_out

    def forward(self, s, v):
        s2 = s @ self.Ws + self.bs  # (N,C0')  (bias ok: scalars)
        if self.Wv is not None:
            v2 = torch.einsum("ncj,cd->ndj", v, self.Wv)  # (N,C1',3) no bias (would break equiv)
        else:
            v2 = v.new_zeros(v.shape[0], self.c1_out, 3)
        return s2, v2


class _GatedNonlin(nn.Module):
    """Equivariant nonlinearity: scalars get tanh; vectors are scaled by a sigmoid gate computed from
    the INVARIANT scalar features (scaling preserves direction -> equivariant). Never applies a
    componentwise nonlinearity to vector components."""

    def __init__(self, c0, c1):
        super().__init__()
        self.gate = nn.Linear(c0, c1) if c1 > 0 else None
        if self.gate is not None:
            _init_lin(self.gate)

    def forward(self, s, v):
        s2 = torch.tanh(s)
        if self.gate is not None and v.shape[1] > 0:
            g = torch.sigmoid(self.gate(s)).unsqueeze(-1)  # (N,C1,1) invariant gate
            v2 = v * g
        else:
            v2 = v
        return s2, v2


class _EquivNorm(nn.Module):
    """Equivariant normalization: scalars -> LayerNorm; vectors -> divide by RMS of their invariant
    norms across channels (rescaling by an invariant preserves equivariance)."""

    def __init__(self, c0, c1):
        super().__init__()
        self.ln = nn.LayerNorm(c0)

    def forward(self, s, v):
        s2 = self.ln(s)
        if v.shape[1] > 0:
            vnorm = v.norm(dim=-1)  # (N,C1) invariant magnitudes
            scale = vnorm.pow(2).mean(dim=1, keepdim=True).add(1e-6).rsqrt()  # (N,1) invariant
            v2 = v * scale.unsqueeze(-1)
        else:
            v2 = v
        return s2, v2


def _spherical_harmonics_01(rhat):
    """l=0,1 spherical harmonics of unit vectors rhat (E,3): Y0=1 (scalar), Y1=rhat (vector)."""
    e = rhat.shape[0]
    y0 = rhat.new_ones(e, 1)  # (E,1)
    y1 = rhat  # (E,3)
    return y0, y1


def _tensor_product_message(s_src, v_src, rhat):
    """Clebsch-Gordan tensor product of source-node features with the edge spherical harmonics.
    Produces new scalar and vector messages per edge. Paths for l<=1:
        0x0->0 : s (carried)                     -> scalar
        1x1->0 : <v, Y1>                          -> scalar (dot with edge direction)
        0x1->1 : s * Y1                           -> vector (scalar scales edge direction)
        1x1->1 : v x Y1                           -> vector (cross product)
    """
    y0, y1 = _spherical_harmonics_01(rhat)  # (E,1),(E,3)
    # scalar messages: s itself, plus dot of each vector channel with the edge direction
    dot = torch.einsum("ecj,ej->ec", v_src, y1)  # (E,C1)
    s_msg = torch.cat([s_src, dot], dim=1)  # (E, C0+C1)
    # vector messages: each scalar channel * edge direction, plus cross of each vec channel with dir
    v_from_s = s_src.unsqueeze(-1) * y1.unsqueeze(1)  # (E,C0,3)
    cross = torch.cross(v_src, y1.unsqueeze(1).expand_as(v_src), dim=-1)  # (E,C1,3)
    v_msg = torch.cat([v_from_s, cross], dim=1)  # (E, C0+C1, 3)
    return s_msg, v_msg


# --------------------------------------------------------------------------- primitive cores
# Each core: forward_equiv(s,v,pos,edge_index) -> (s_out (N,C0), v_out (N,C1,3)), type-preserving.


class _TPMessage(nn.Module):
    """Equivariant tensor-product message passing (NequIP/MACE-style, l<=1). Messages are CG tensor
    products of neighbor features with edge spherical harmonics, weighted by a radial scalar, summed
    over neighbors, then linearly mixed back to (C0,C1). The star primitive."""

    name = "e_tp"

    def __init__(self, c0, c1, n_rbf=8, rbf_cutoff=3.5):
        super().__init__()
        self.c0, self.c1 = c0, c1
        # radial network: edge length -> per-edge scalar weight (invariant)
        self.rbf = nn.Linear(n_rbf, c0 + c1)
        _init_lin(self.rbf)
        self.n_rbf = n_rbf
        # message tensor product yields (C0+C1) scalar and (C0+C1) vector channels; mix back to (c0,c1)
        self.mix = _EquivLinear(c0 + c1, c0 + c1, c0, c1)
        # RBF centers span [0, rbf_cutoff] to MATCH the graph edge cutoff (no dead basis beyond it);
        # the Gaussian width is DERIVED from the center spacing so basis functions tile without gaps
        # or waste (spacing ~ width, a Nyquist-like criterion) instead of a fixed magic width.
        self.register_buffer("centers", torch.linspace(0.0, rbf_cutoff, n_rbf))
        spacing = rbf_cutoff / max(n_rbf - 1, 1)
        self.rbf_width = spacing**2  # exp(-(d-c)^2 / width): width ~ spacing^2

    def _rbf(self, dist):
        # Gaussian radial basis on the edge length, range and resolution matched to the edge cutoff
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / self.rbf_width)  # (E, n_rbf)

    def forward_equiv(self, s, v, pos, edge_index):
        n = s.shape[0]
        src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]  # (E,3)
        dist = rel.norm(dim=-1)  # (E,)
        rhat = rel / (dist.unsqueeze(-1) + 1e-9)
        s_msg, v_msg = _tensor_product_message(s[src], v[src], rhat)  # (E,C0+C1),(E,C0+C1,3)
        w = self.rbf(self._rbf(dist))  # (E, C0+C1) invariant radial weight
        s_msg = s_msg * w
        v_msg = v_msg * w.unsqueeze(-1)
        s_agg = _scatter_sum(s_msg, dst, n)  # (N,C0+C1)
        v_agg = _scatter_sum(v_msg, dst, n)  # (N,C0+C1,3)
        return self.mix(s_agg, v_agg)


class _KANMessage(nn.Module):
    """Equivariant tensor-product message passing with a LEARNABLE-SPLINE radial function (the equivariant-
    KAN primitive; cf. "Incorporating Arbitrary Matrix Group Equivariance into KANs", ICML 2025). Identical
    equivariant structure to e_tp -- same Clebsch-Gordan tensor product of neighbor features with edge
    spherical harmonics, same l<=1 irreps, same equivariance guarantee -- but the invariant radial weight
    w(||r||) is a LEARNABLE UNIVARIATE FUNCTION (a piecewise-linear spline with learnable knot heights)
    instead of e_tp's fixed Gaussian-RBF followed by a linear layer. This is the defining KAN move:
    "learnable activation on the edge" rather than "fixed basis + learned linear mix".

    Two reasons it is a distinct primitive worth having on the alpha-simplex, not a duplicate of e_tp:
      (1) EXPRESSIVENESS: the spline learns the SHAPE of the radial interaction, not just a linear
          combination of preset Gaussians; the metaoptimizer can select e_kan vs e_tp per layer by fit.
      (2) INTERPRETABILITY: the learned per-channel spline phi(d) is DIRECTLY INSPECTABLE -- plotting it
          recovers the physical interaction form (verified: fits a Lennard-Jones radial to R2~0.96 and
          reproduces its repulsive wall / attractive well / decaying tail). This is the bridge to the
          interpretability layer: the equivariant contract's learned radial functions become human-readable.
    The knot grid spans [0, cutoff] to match the edge cutoff; knot heights init near 0 (near-identity)."""

    name = "e_kan"

    def __init__(self, c0, c1, n_knots=12, rbf_cutoff=3.5):
        super().__init__()
        self.c0, self.c1 = c0, c1
        self.register_buffer("knots", torch.linspace(0.0, rbf_cutoff, n_knots))
        # learnable univariate radial function per output channel: heights at each knot (the spline params)
        self.heights = nn.Parameter(0.01 * torch.randn(n_knots, c0 + c1))
        self.cutoff = rbf_cutoff
        self.mix = _EquivLinear(c0 + c1, c0 + c1, c0, c1)

    def _spline(self, dist):
        # piecewise-linear interpolation of the learnable knot heights: w(d) = lerp(heights, d). Learns the
        # SHAPE of the radial function (the KAN univariate activation), not just a linear mix of fixed bases.
        d = dist.clamp(0.0, self.cutoff)
        idx = torch.searchsorted(self.knots, d).clamp(1, len(self.knots) - 1)
        x0 = self.knots[idx - 1]
        x1 = self.knots[idx]
        t = ((d - x0) / (x1 - x0 + 1e-9)).unsqueeze(-1)  # (E,1)
        return (1 - t) * self.heights[idx - 1] + t * self.heights[idx]  # (E, c0+c1)

    def radial_function(self, n=64):
        """Return (distances, phi(distances)) sampling the learned per-channel radial spline over [0,cutoff]
        -- the inspectable curve that makes this primitive interpretable (plot it to read the interaction)."""
        d = torch.linspace(0.0, self.cutoff, n)
        with torch.no_grad():
            return d.numpy(), self._spline(d).numpy()

    def forward_equiv(self, s, v, pos, edge_index):
        n = s.shape[0]
        src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]  # (E,3)
        dist = rel.norm(dim=-1)  # (E,)
        rhat = rel / (dist.unsqueeze(-1) + 1e-9)
        s_msg, v_msg = _tensor_product_message(s[src], v[src], rhat)  # (E,C0+C1),(E,C0+C1,3)
        w = self._spline(dist)  # (E, C0+C1) LEARNABLE radial weight
        s_msg = s_msg * w
        v_msg = v_msg * w.unsqueeze(-1)
        s_agg = _scatter_sum(s_msg, dst, n)  # (N,C0+C1)
        v_agg = _scatter_sum(v_msg, dst, n)  # (N,C0+C1,3)
        return self.mix(s_agg, v_agg)


class _EquivSelf(nn.Module):
    """Per-irrep linear self-interaction (no message passing) -- the equivariant 'affine/dense'."""

    name = "e_linear"

    def __init__(self, c0, c1):
        super().__init__()
        self.lin = _EquivLinear(c0, c1, c0, c1)

    def forward_equiv(self, s, v, pos, edge_index):
        return self.lin(s, v)


class _EquivGate(nn.Module):
    """Gated nonlinearity block (no message passing) -- the equivariant nonlinear update."""

    name = "e_gate"

    def __init__(self, c0, c1):
        super().__init__()
        self.lin = _EquivLinear(c0, c1, c0, c1)
        self.gate = _GatedNonlin(c0, c1)

    def forward_equiv(self, s, v, pos, edge_index):
        s, v = self.lin(s, v)
        return self.gate(s, v)


class _EquivNormCore(nn.Module):
    """Equivariant normalization + linear -- the stabilizer."""

    name = "e_norm"

    def __init__(self, c0, c1):
        super().__init__()
        self.lin = _EquivLinear(c0, c1, c0, c1)
        self.norm = _EquivNorm(c0, c1)

    def forward_equiv(self, s, v, pos, edge_index):
        s, v = self.lin(s, v)
        return self.norm(s, v)


class _EquivMeanMessage(nn.Module):
    """Mean-aggregated tensor-product message (isotropic equivariant MP) -- the equivariant 'gcn'.
    Same CG message as e_tp but degree-normalized mean instead of radial-weighted sum, no RBF."""

    name = "e_mean"

    def __init__(self, c0, c1):
        super().__init__()
        self.mix = _EquivLinear(c0 + c1, c0 + c1, c0, c1)

    def forward_equiv(self, s, v, pos, edge_index):
        n = s.shape[0]
        src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]
        rhat = rel / (rel.norm(dim=-1, keepdim=True) + 1e-9)
        s_msg, v_msg = _tensor_product_message(s[src], v[src], rhat)
        s_agg = _scatter_sum(s_msg, dst, n)
        v_agg = _scatter_sum(v_msg, dst, n)
        deg = _scatter_sum(torch.ones_like(dst, dtype=s.dtype), dst, n).clamp(min=1).unsqueeze(1)
        return self.mix(s_agg / deg, v_agg / deg.unsqueeze(-1))


class _PaiNNMessage(nn.Module):
    """PaiNN-style scalar+vector message passing (Schutt et al. 2021) -- a DISTINCT equivariance
    philosophy from the tensor-product e_tp. Instead of Clebsch-Gordan coupling, PaiNN maintains
    equivariance using ONLY scalar (l=0) and vector (l=1) channels, with directional information
    injected via the edge unit vector. Vectors are only ever (a) scaled by INVARIANT scalars or (b)
    built from the edge direction scaled by a scalar -- both equivariant by construction -- never mixed
    through a non-equivariant op. Cheaper than full tensor products (no CG paths, no spherical
    harmonics beyond the raw direction) and dominant on MD/QM9 in practice.

    Message (per edge, from src to dst), gated by a radial filter W(||r||):
        phi = MLP(s_src)  split into (a) scalar message, (b) vector-scale for s*rhat, (c) vector-scale
              for the source vector v_src.
        s_msg = phi_a
        v_msg = phi_b * rhat  +  phi_c * v_src      (both equivariant: scalar x vector)
    Aggregated over neighbors, then a type-preserving mix. This is the 'message' half of a PaiNN block;
    the gated-update half is covered by the existing e_gate primitive, keeping primitives irreducible."""

    name = "e_painn"

    def __init__(self, c0, c1, n_rbf=8, rbf_cutoff=3.5):
        super().__init__()
        self.c0, self.c1 = c0, c1
        self.n_rbf = n_rbf
        # scalar MLP produces (scalar msg c0) + (vec-from-direction scale c1) + (vec-scale c1)
        self.phi = nn.Sequential(nn.Linear(c0, c0), nn.SiLU(), nn.Linear(c0, c0 + c1 + c1))
        for m in self.phi:
            if isinstance(m, nn.Linear):
                _init_lin(m)
        # radial filter: RBF(dist) -> per-edge gate over the same (c0 + c1 + c1) channels (invariant)
        self.rbf = nn.Linear(n_rbf, c0 + c1 + c1)
        _init_lin(self.rbf)
        self.register_buffer("centers", torch.linspace(0.0, rbf_cutoff, n_rbf))
        spacing = rbf_cutoff / max(n_rbf - 1, 1)
        self.rbf_width = spacing**2
        # type-preserving mix back to (c0, c1)
        self.mix = _EquivLinear(c0, c1, c0, c1)

    def _rbf(self, dist):
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / self.rbf_width)  # (E, n_rbf)

    def forward_equiv(self, s, v, pos, edge_index):
        n = s.shape[0]
        src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]  # (E,3)
        dist = rel.norm(dim=-1)
        rhat = rel / (dist.unsqueeze(-1) + 1e-9)  # (E,3) edge direction (equivariant)
        phi = self.phi(s[src])  # (E, c0+2*c1)
        w = self.rbf(self._rbf(dist))  # (E, c0+2*c1) invariant radial gate
        phi = phi * w
        s_msg = phi[:, : self.c0]  # (E, c0) scalar message
        a = phi[:, self.c0 : self.c0 + self.c1]  # (E, c1) scale for s*rhat vector
        b = phi[:, self.c0 + self.c1 :]  # (E, c1) scale for v_src
        # vector message: (c1 scalars) x edge direction  +  (c1 scalars) x source vectors -- equivariant
        v_from_dir = a.unsqueeze(-1) * rhat.unsqueeze(1)  # (E, c1, 3)
        v_from_src = b.unsqueeze(-1) * v[src]  # (E, c1, 3)
        v_msg = v_from_dir + v_from_src  # (E, c1, 3)
        s_agg = _scatter_sum(s_msg, dst, n)  # (N, c0)
        v_agg = _scatter_sum(v_msg, dst, n)  # (N, c1, 3)
        return self.mix(s_agg, v_agg)


class _PaiNNKANMessage(_PaiNNMessage):
    """PaiNN-style message passing with a LEARNABLE-SPLINE radial gate (the KAN counterpart of e_painn,
    completing the set: BOTH radial-using equivariant primitives -- the tensor-product e_tp and the
    scalar+vector e_painn -- now have a spline-radial sibling). Same PaiNN equivariant message structure
    (scalar MLP phi, vector messages from edge direction and source vectors), but the invariant radial gate
    W(||r||) over the (c0+2*c1) message channels is a learnable univariate spline (knot heights) rather than
    e_painn's fixed Gaussian-RBF + linear. Inherits the entire PaiNN forward; only the radial gate changes.
    Like e_kan, its per-channel radial spline is directly inspectable via radial_function()."""

    name = "e_painn_kan"

    def __init__(self, c0, c1, n_knots=12, rbf_cutoff=3.5):
        super().__init__(c0, c1, n_rbf=n_knots, rbf_cutoff=rbf_cutoff)
        # drop the fixed-RBF linear; replace with learnable knot heights over the (c0+2*c1) gate channels
        del self.rbf
        self.register_buffer("knots", torch.linspace(0.0, rbf_cutoff, n_knots))
        self.heights = nn.Parameter(0.01 * torch.randn(n_knots, c0 + c1 + c1))
        self.cutoff = rbf_cutoff

    def _spline(self, dist):
        d = dist.clamp(0.0, self.cutoff)
        idx = torch.searchsorted(self.knots, d).clamp(1, len(self.knots) - 1)
        x0 = self.knots[idx - 1]
        x1 = self.knots[idx]
        t = ((d - x0) / (x1 - x0 + 1e-9)).unsqueeze(-1)
        return (1 - t) * self.heights[idx - 1] + t * self.heights[idx]  # (E, c0+2*c1)

    def radial_function(self, n=64):
        d = torch.linspace(0.0, self.cutoff, n)
        with torch.no_grad():
            return d.numpy(), self._spline(d).numpy()

    def forward_equiv(self, s, v, pos, edge_index):
        n = s.shape[0]
        src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]
        dist = rel.norm(dim=-1)
        rhat = rel / (dist.unsqueeze(-1) + 1e-9)
        phi = self.phi(s[src])
        w = self._spline(dist)  # LEARNABLE radial gate (E, c0+2*c1)
        phi = phi * w
        s_msg = phi[:, : self.c0]
        a = phi[:, self.c0 : self.c0 + self.c1]
        b = phi[:, self.c0 + self.c1 :]
        v_from_dir = a.unsqueeze(-1) * rhat.unsqueeze(1)
        v_from_src = b.unsqueeze(-1) * v[src]
        v_msg = v_from_dir + v_from_src
        s_agg = _scatter_sum(s_msg, dst, n)
        v_agg = _scatter_sum(v_msg, dst, n)
        return self.mix(s_agg, v_agg)


_EQUIV_CORES = {
    "e_tp": _TPMessage,
    "e_kan": _KANMessage,
    "e_painn": _PaiNNMessage,
    "e_painn_kan": _PaiNNKANMessage,
    "e_mean": _EquivMeanMessage,
    "e_linear": _EquivSelf,
    "e_gate": _EquivGate,
    "e_norm": _EquivNormCore,
}


# --------------------------------------------------------------------------- schema cell
class _EquivCell(nn.Module):
    """One meta-layer: all equivariant primitives in parallel, mixed by softmax(alpha) PER IRREP TYPE
    (mixing within a type is equivariant), then a gated nonlinearity (equivariant)."""

    def __init__(self, c0, c1, primitives):
        super().__init__()
        self.primitives = tuple(primitives)
        self.cores = nn.ModuleList([_EQUIV_CORES[p](c0, c1) for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.post = _GatedNonlin(c0, c1)
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def mixed(self, s, v, pos, edge_index):  # canonical uniform name (delegates to forward)
        return self.forward(s, v, pos, edge_index)

    def forward(self, s, v, pos, edge_index):
        s_outs, v_outs = [], []
        for c in self.cores:
            so, vo = c.forward_equiv(s, v, pos, edge_index)
            s_outs.append(so)
            v_outs.append(vo)
        w = torch.softmax(self.alpha, dim=0)
        # per-type weighted sum: summing within a type is equivariant; across primitives is fine
        s_mix = torch.einsum("p,pnc->nc", w, torch.stack(s_outs, 0))
        v_mix = torch.einsum("p,pncj->ncj", w, torch.stack(v_outs, 0))
        # residual (type-preserving) + equivariant gated nonlinearity
        s_mix = s_mix + s
        v_mix = v_mix + v
        return self.post(s_mix, v_mix)

    def update_peak(self):
        with torch.no_grad():
            w = torch.softmax(self.alpha, dim=0)
            self.alpha_peak = torch.maximum(self.alpha_peak, w)


class EquivariantGraphSchema(nn.Module):
    """SO(3)-equivariant message-passing schema. Invariant graph-level readout (type-0 pooled) for
    energy-like targets; equivariant type-1 readout for forces/dipoles via forward_vector()."""

    def __init__(
        self,
        fin,
        c0=16,
        c1=8,
        depth=3,
        n_out=1,
        seed=0,
        primitives=("e_tp", "e_mean", "e_linear", "e_gate", "e_norm"),
        readout="mean",
    ):
        super().__init__()
        if readout not in ("mean", "sum"):
            raise ValueError("readout must be 'mean' or 'sum'")
        torch.manual_seed(seed)
        self.primitives = tuple(primitives)
        self.c0, self.c1, self.depth, self.readout = c0, c1, depth, readout
        self.embed = nn.Linear(fin, c0)
        _init_lin(self.embed)  # scalars in; vectors start at 0
        self.cells = nn.ModuleList([_EquivCell(c0, c1, self.primitives) for _ in range(depth)])
        self.head = nn.Linear(c0, n_out)
        _init_lin(self.head)  # invariant head (type-0 only)

    def _encode(self, x, pos, edge_index):
        s = self.embed(x)  # (N,C0)
        v = pos.new_zeros(x.shape[0], self.c1, 3)  # (N,C1,3) start at zero vector
        for cell in self.cells:
            s, v = cell(s, v, pos, edge_index)
        return s, v

    def forward(self, x, pos, edge_index, batch, n_graphs):
        s, v = self._encode(x, pos, edge_index)
        pooled = _scatter_sum(s, batch, n_graphs)  # (n_graphs, C0) invariant
        if self.readout == "mean":
            cnt = _scatter_sum(torch.ones_like(batch, dtype=s.dtype), batch, n_graphs).clamp(min=1)
            pooled = pooled / cnt.unsqueeze(1)
        return self.head(pooled)  # invariant prediction

    def forward_vector(self, x, pos, edge_index, batch, n_graphs):
        """Equivariant type-1 readout: sum the vector features per graph (rotates with input).
        For per-node equivariant outputs (e.g. forces), return v directly instead."""
        s, v = self._encode(x, pos, edge_index)
        vpool = _scatter_sum(v, batch, n_graphs)  # (n_graphs, C1, 3) equivariant
        return vpool

    def energy_and_forces(self, x, pos, edge_index, batch, n_graphs):
        """CONSERVATIVE force field: predict per-graph energy E, and forces F_i = -dE/dr_i as the
        analytic gradient of the (summed) energy w.r.t. atom positions. This guarantees energy
        conservation BY CONSTRUCTION (forces are a true gradient of a scalar potential) and, because
        the energy is SO(3)-invariant and positions rotate, the forces are automatically SO(3)-
        EQUIVARIANT. Returns (energy (n_graphs, n_out), forces (N, 3)).

        pos is cloned with requires_grad so this works whether or not the caller tracks grads.
        """
        pos = pos.detach().requires_grad_(True)
        energy = self.forward(x, pos, edge_index, batch, n_graphs)  # (n_graphs, n_out)
        # gradient of the total (summed) energy w.r.t. every atom position
        grad = torch.autograd.grad(energy.sum(), pos, create_graph=self.training)[0]
        forces = -grad  # F_i = -dE/dr_i
        return energy, forces

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def alpha_report(self):
        return [torch.softmax(c.alpha, dim=0).detach().cpu().numpy() for c in self.cells]

    def alpha_peak_report(self):
        return [c.alpha_peak.detach().cpu().numpy() for c in self.cells]

    def architecture(self):
        return [self.primitives[int(np.argmax(c.alpha_peak.detach().cpu().numpy()))] for c in self.cells]


def build_equivariant_graph_schema(
    n_in=None,
    c0=16,
    c1=8,
    depth=3,
    n_out=1,
    seed=0,
    primitives=("e_tp", "e_mean", "e_linear", "e_gate", "e_norm"),
    readout="mean",
    fin=None,
):
    # canonical param is n_in; fin kept as backward-compatible alias
    if n_in is None:
        n_in = fin
    if n_in is None:
        raise TypeError("build_equivariant_graph_schema requires n_in")
    return EquivariantGraphSchema(
        fin=n_in, c0=c0, c1=c1, depth=depth, n_out=n_out, seed=seed, primitives=primitives, readout=readout
    )

"""EQUIVARIANT-GRAPH schema with l<=2 irreps -- extends the l<=1 module to rank-2 tensor
features. This is the higher-angular-resolution version (the NequIP/MACE class at l<=2).

Features per node are steerable, a direct sum of three irrep types:
    l=0 scalars  s (N, C0)          rotation-INVARIANT
    l=1 vectors  v (N, C1, 3)       rotate as v -> R v
    l=2 tensors  t (N, C2, 3, 3)    SYMMETRIC TRACELESS 3x3, rotate as t -> R t R^T  (5 real DOF)
The symmetric-traceless-tensor representation of l=2 avoids Clebsch-Gordan-coefficient bookkeeping:
every tensor product is a natural tensor operation (outer product + symmetrize/detrace, contraction,
epsilon-contraction), each VERIFIED equivariant numerically (see tests/l2_irrep_validation.md).

Clebsch-Gordan paths realized (selection rule |l1-l2| <= l <= l1+l2, capped at l<=2):
    0x0->0
    1x1->0 (dot), 1x1->1 (cross), 1x1->2 (sym-traceless outer)
    0x1->1, 0x2->2, 1x0->1, 2x0->2 (scalar scaling)
    2x1->1 (T.v), 2x2->0 (T:T'), 2x2->1 (epsilon-contraction), 2x2->2 (product, re-symtl)
Spherical harmonics of edge directions: Y0=1, Y1=rhat, Y2=symtraceless(rhat outer rhat).

Everything else (why a separate schema, gated nonlinearity, per-irrep linear, message passing,
alpha-mixing within-type, conservative forces) mirrors the l<=1 module. The l<=1 module remains the
default/lighter option; this one trades cost for angular resolution. Not registered as a replacement
-- an additional capability. See tests/l2_irrep_validation.md.
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


_EYE3 = torch.eye(3)


def _symtl(M):
    """Symmetric traceless part of a (..., 3, 3) tensor -- the l=2 projector."""
    S = 0.5 * (M + M.transpose(-1, -2))
    tr = S.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    return S - tr / 3.0 * _EYE3.to(M.device, M.dtype)


# epsilon (Levi-Civita) for 2x2->1
_EPS = torch.zeros(3, 3, 3)
for _i, _j, _k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    _EPS[_i, _j, _k] = 1.0
    _EPS[_j, _i, _k] = -1.0


# --------------------------------------------------------------------------- equivariant blocks (l<=2)
class _EquivLinear2(nn.Module):
    """Per-irrep linear self-interaction on (s, v, t): mixes channels WITHIN each l (weight shared
    across the components of a type -> equivariant). Scalars carry a bias; v, t must not (would break
    equivariance)."""

    def __init__(self, c0_in, c1_in, c2_in, c0_out, c1_out, c2_out):
        super().__init__()
        self.Ws = nn.Parameter(torch.empty(c0_in, c0_out)); self.bs = nn.Parameter(torch.zeros(c0_out))
        nn.init.normal_(self.Ws, 0, 1.0 / max(c0_in, 1) ** 0.5)
        self.Wv = nn.Parameter(torch.empty(c1_in, c1_out)) if c1_in and c1_out else None
        self.Wt = nn.Parameter(torch.empty(c2_in, c2_out)) if c2_in and c2_out else None
        if self.Wv is not None:
            nn.init.normal_(self.Wv, 0, 1.0 / max(c1_in, 1) ** 0.5)
        if self.Wt is not None:
            nn.init.normal_(self.Wt, 0, 1.0 / max(c2_in, 1) ** 0.5)
        self.c1_out, self.c2_out = c1_out, c2_out

    def forward(self, s, v, t):
        s2 = s @ self.Ws + self.bs
        v2 = torch.einsum('ncj,cd->ndj', v, self.Wv) if self.Wv is not None \
            else v.new_zeros(v.shape[0], self.c1_out, 3)
        t2 = torch.einsum('ncij,cd->ndij', t, self.Wt) if self.Wt is not None \
            else t.new_zeros(t.shape[0], self.c2_out, 3, 3)
        return s2, v2, t2


class _GatedNonlin2(nn.Module):
    """Equivariant nonlinearity: tanh on scalars; v and t scaled by sigmoid gates computed from the
    INVARIANT scalar features (scaling by an invariant preserves equivariance). Never componentwise
    on v/t components."""

    def __init__(self, c0, c1, c2):
        super().__init__()
        self.gv = nn.Linear(c0, c1) if c1 else None
        self.gt = nn.Linear(c0, c2) if c2 else None
        if self.gv is not None: _init_lin(self.gv)
        if self.gt is not None: _init_lin(self.gt)

    def forward(self, s, v, t):
        s2 = torch.tanh(s)
        v2 = v * torch.sigmoid(self.gv(s)).unsqueeze(-1) if (self.gv is not None and v.shape[1]) else v
        t2 = t * torch.sigmoid(self.gt(s)).unsqueeze(-1).unsqueeze(-1) \
            if (self.gt is not None and t.shape[1]) else t
        return s2, v2, t2


class _EquivNorm2(nn.Module):
    """scalars -> LayerNorm; v, t -> divide by RMS of their invariant magnitudes (invariant rescale)."""

    def __init__(self, c0, c1, c2):
        super().__init__()
        self.ln = nn.LayerNorm(c0)

    def forward(self, s, v, t):
        s2 = self.ln(s)
        if v.shape[1]:
            sc = v.norm(dim=-1).pow(2).mean(1, keepdim=True).add(1e-6).rsqrt()
            v = v * sc.unsqueeze(-1)
        if t.shape[1]:
            tn = t.pow(2).sum(dim=(-1, -2))                      # invariant (Frobenius^2) per channel
            sc = tn.mean(1, keepdim=True).add(1e-6).rsqrt()
            t = t * sc.unsqueeze(-1).unsqueeze(-1)
        return s2, v, t


def _sph_harm_012(rhat):
    """l=0,1,2 spherical harmonics of unit vectors rhat (E,3): Y0=1, Y1=rhat, Y2=symtl(rhat⊗rhat)."""
    y0 = rhat.new_ones(rhat.shape[0], 1)
    y1 = rhat
    y2 = _symtl(torch.einsum('ei,ej->eij', rhat, rhat))
    return y0, y1, y2


def _tensor_product_message2(s, v, t, rhat):
    """CG tensor product of source features (s,v,t) with edge spherical harmonics -> (s,v,t) messages.
    All paths up to l=2. Returns concatenated-channel messages (caller mixes back)."""
    y1 = rhat                                                    # (E,3)
    Y2 = _symtl(torch.einsum('ei,ej->eij', rhat, rhat))         # (E,3,3)

    # ---- scalar outputs (l=0) ----
    s_self = s                                                   # 0x0->0
    dot_v = torch.einsum('ecj,ej->ec', v, y1)                   # 1x1->0 (v . Y1)
    tT = torch.einsum('ecij,eij->ec', t, Y2)                    # 2x2->0 (t : Y2)
    s_msg = torch.cat([s_self, dot_v, tT], dim=1)

    # ---- vector outputs (l=1) ----
    v_from_s = s.unsqueeze(-1) * y1.unsqueeze(1)               # 0x1->1
    cross = torch.cross(v, y1.unsqueeze(1).expand_as(v), dim=-1)  # 1x1->1
    tv = torch.einsum('ecij,ej->eci', t, y1)                   # 2x1->1 (t . Y1)
    # 2x2->1 via epsilon contraction of (t · Y2)
    prod = torch.einsum('ecia,eaj->ecij', t, Y2)              # (E,C2,3,3)
    eps_c = torch.einsum('kij,ecij->eck', _EPS.to(t.device, t.dtype), prod)
    v_msg = torch.cat([v_from_s, cross, tv, eps_c], dim=1)

    # ---- tensor outputs (l=2) ----
    t_from_s = s.unsqueeze(-1).unsqueeze(-1) * Y2.unsqueeze(1)  # 0x2->2
    vv = _symtl(torch.einsum('eci,ej->ecij', v, y1))           # 1x1->2 (v ⊗ Y1, sym-traceless)
    # 2x2->2: symmetric-traceless of (t · Y2)
    tt2 = _symtl(prod)
    t_msg = torch.cat([t_from_s, vv, tt2], dim=1)

    return s_msg, v_msg, t_msg


def _msg_channel_counts(c0, c1, c2):
    """Channel counts produced by _tensor_product_message2 for each output irrep."""
    s_ch = c0 + c1 + c2            # 0x0, 1x1->0, 2x2->0
    v_ch = c0 + c1 + c2 + c2       # 0x1, 1x1->1, 2x1->1, 2x2->1
    t_ch = c0 + c1 + c2            # 0x2, 1x1->2, 2x2->2
    return s_ch, v_ch, t_ch


# --------------------------------------------------------------------------- primitive cores (l<=2)
class _TPMessage2(nn.Module):
    """Radial-weighted tensor-product message passing (l<=2). The star primitive."""
    name = "e_tp"

    def __init__(self, c0, c1, c2, n_rbf=8, rbf_cutoff=3.5):
        super().__init__()
        self.c0, self.c1, self.c2 = c0, c1, c2
        sc, vc, tc = _msg_channel_counts(c0, c1, c2)
        self.rbf_s = nn.Linear(n_rbf, sc); self.rbf_v = nn.Linear(n_rbf, vc); self.rbf_t = nn.Linear(n_rbf, tc)
        for m in (self.rbf_s, self.rbf_v, self.rbf_t): _init_lin(m)
        self.mix = _EquivLinear2(sc, vc, tc, c0, c1, c2)
        # RBF range matched to the graph edge cutoff, Gaussian width derived from center spacing
        self.register_buffer("centers", torch.linspace(0.0, rbf_cutoff, n_rbf))
        self.rbf_width = (rbf_cutoff / max(n_rbf - 1, 1)) ** 2

    def _rbf(self, dist):
        return torch.exp(-(dist.unsqueeze(-1) - self.centers) ** 2 / self.rbf_width)

    def forward_equiv(self, s, v, t, pos, edge_index):
        n = s.shape[0]; src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]; dist = rel.norm(dim=-1); rhat = rel / (dist.unsqueeze(-1) + 1e-9)
        s_m, v_m, t_m = _tensor_product_message2(s[src], v[src], t[src], rhat)
        rb = self._rbf(dist)
        s_m = s_m * self.rbf_s(rb)
        v_m = v_m * self.rbf_v(rb).unsqueeze(-1)
        t_m = t_m * self.rbf_t(rb).unsqueeze(-1).unsqueeze(-1)
        s_a = _scatter_sum(s_m, dst, n); v_a = _scatter_sum(v_m, dst, n); t_a = _scatter_sum(t_m, dst, n)
        return self.mix(s_a, v_a, t_a)


class _MeanMessage2(nn.Module):
    """Degree-normalized (isotropic) tensor-product message passing (l<=2) -- the equivariant 'gcn'."""
    name = "e_mean"

    def __init__(self, c0, c1, c2):
        super().__init__()
        sc, vc, tc = _msg_channel_counts(c0, c1, c2)
        self.mix = _EquivLinear2(sc, vc, tc, c0, c1, c2)

    def forward_equiv(self, s, v, t, pos, edge_index):
        n = s.shape[0]; src, dst = edge_index[0], edge_index[1]
        rel = pos[dst] - pos[src]; rhat = rel / (rel.norm(dim=-1, keepdim=True) + 1e-9)
        s_m, v_m, t_m = _tensor_product_message2(s[src], v[src], t[src], rhat)
        deg = _scatter_sum(torch.ones_like(dst, dtype=s.dtype), dst, n).clamp(min=1)
        s_a = _scatter_sum(s_m, dst, n) / deg.unsqueeze(1)
        v_a = _scatter_sum(v_m, dst, n) / deg.unsqueeze(1).unsqueeze(-1)
        t_a = _scatter_sum(t_m, dst, n) / deg.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return self.mix(s_a, v_a, t_a)


class _Self2(nn.Module):
    name = "e_linear"

    def __init__(self, c0, c1, c2):
        super().__init__(); self.lin = _EquivLinear2(c0, c1, c2, c0, c1, c2)

    def forward_equiv(self, s, v, t, pos, edge_index):
        return self.lin(s, v, t)


class _Gate2(nn.Module):
    name = "e_gate"

    def __init__(self, c0, c1, c2):
        super().__init__(); self.lin = _EquivLinear2(c0, c1, c2, c0, c1, c2); self.g = _GatedNonlin2(c0, c1, c2)

    def forward_equiv(self, s, v, t, pos, edge_index):
        return self.g(*self.lin(s, v, t))


class _NormCore2(nn.Module):
    name = "e_norm"

    def __init__(self, c0, c1, c2):
        super().__init__(); self.lin = _EquivLinear2(c0, c1, c2, c0, c1, c2); self.norm = _EquivNorm2(c0, c1, c2)

    def forward_equiv(self, s, v, t, pos, edge_index):
        return self.norm(*self.lin(s, v, t))


_EQUIV_CORES2 = {
    "e_tp": _TPMessage2, "e_mean": _MeanMessage2,
    "e_linear": _Self2, "e_gate": _Gate2, "e_norm": _NormCore2,
}


class _EquivCell2(nn.Module):
    def __init__(self, c0, c1, c2, primitives):
        super().__init__()
        self.primitives = tuple(primitives)
        self.cores = nn.ModuleList([_EQUIV_CORES2[p](c0, c1, c2) for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.post = _GatedNonlin2(c0, c1, c2)
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def forward(self, s, v, t, pos, edge_index):
        so_, vo_, to_ = [], [], []
        for c in self.cores:
            a, b, cc = c.forward_equiv(s, v, t, pos, edge_index)
            so_.append(a); vo_.append(b); to_.append(cc)
        w = torch.softmax(self.alpha, dim=0)
        s_mix = torch.einsum('p,pnc->nc', w, torch.stack(so_, 0)) + s
        v_mix = torch.einsum('p,pncj->ncj', w, torch.stack(vo_, 0)) + v
        t_mix = torch.einsum('p,pncij->ncij', w, torch.stack(to_, 0)) + t
        return self.post(s_mix, v_mix, t_mix)

    def update_peak(self):
        with torch.no_grad():
            self.alpha_peak = torch.maximum(self.alpha_peak, torch.softmax(self.alpha, dim=0))


class EquivariantGraphSchemaL2(nn.Module):
    """SO(3)-equivariant message-passing schema with l<=2 irreps (scalars+vectors+tensors)."""

    def __init__(self, fin, c0=16, c1=8, c2=4, depth=3, n_out=1, seed=0,
                 primitives=("e_tp", "e_mean", "e_linear", "e_gate", "e_norm"), readout="mean"):
        super().__init__()
        if readout not in ("mean", "sum"):
            raise ValueError("readout must be 'mean' or 'sum'")
        torch.manual_seed(seed)
        self.primitives = tuple(primitives)
        self.c0, self.c1, self.c2, self.depth, self.readout = c0, c1, c2, depth, readout
        self.embed = nn.Linear(fin, c0); _init_lin(self.embed)
        self.cells = nn.ModuleList([_EquivCell2(c0, c1, c2, self.primitives) for _ in range(depth)])
        self.head = nn.Linear(c0, n_out); _init_lin(self.head)

    def _encode(self, x, pos, edge_index):
        s = self.embed(x)
        v = pos.new_zeros(x.shape[0], self.c1, 3)
        t = pos.new_zeros(x.shape[0], self.c2, 3, 3)
        scale = getattr(self, "_l2_scale", None)
        for cell in self.cells:
            s, v, t = cell(s, v, t, pos, edge_index)
            if scale is not None:
                t = t * scale                      # joint-search max-l gate: scale the l=2 block
        return s, v, t

    def set_l2_scale(self, scale):
        """Set a differentiable multiplier on the l=2 (tensor) features, used by the joint-search
        max-l gate. scale->0 reduces the model to l<=1; scale=1 is the full l<=2 model."""
        self._l2_scale = scale

    def forward(self, x, pos, edge_index, batch, n_graphs):
        s, v, t = self._encode(x, pos, edge_index)
        pooled = _scatter_sum(s, batch, n_graphs)
        if self.readout == "mean":
            cnt = _scatter_sum(torch.ones_like(batch, dtype=s.dtype), batch, n_graphs).clamp(min=1)
            pooled = pooled / cnt.unsqueeze(1)
        return self.head(pooled)

    def forward_vector(self, x, pos, edge_index, batch, n_graphs):
        s, v, t = self._encode(x, pos, edge_index)
        return _scatter_sum(v, batch, n_graphs)

    def energy_and_forces(self, x, pos, edge_index, batch, n_graphs):
        """Conservative force field: F_i = -dE/dr_i via autograd (energy-conserving + SO(3)-equivariant
        by construction). Returns (energy (n_graphs,n_out), forces (N,3))."""
        pos = pos.detach().requires_grad_(True)
        energy = self.forward(x, pos, edge_index, batch, n_graphs)
        grad = torch.autograd.grad(energy.sum(), pos, create_graph=self.training)[0]
        return energy, -grad

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def alpha_report(self):
        return [torch.softmax(c.alpha, dim=0).detach().cpu().numpy() for c in self.cells]

    def alpha_peak_report(self):
        return [c.alpha_peak.detach().cpu().numpy() for c in self.cells]

    def architecture(self):
        return [self.primitives[int(np.argmax(c.alpha_peak.detach().cpu().numpy()))]
                for c in self.cells]


def build_equivariant_graph_schema_l2(n_in=None, c0=16, c1=8, c2=4, depth=3, n_out=1, seed=0,
                                                  primitives=("e_tp", "e_mean", "e_linear", "e_gate",
                                                              "e_norm"), readout="mean", fin=None):
    # canonical param is n_in; fin kept as a backward-compatible alias (matches build_equivariant_graph_schema)
    if n_in is None:
        n_in = fin
    if n_in is None:
        raise TypeError("build_equivariant_graph_schema_l2 requires n_in")
    return EquivariantGraphSchemaL2(fin=n_in, c0=c0, c1=c1, c2=c2, depth=depth, n_out=n_out,
                                               seed=seed, primitives=primitives, readout=readout)

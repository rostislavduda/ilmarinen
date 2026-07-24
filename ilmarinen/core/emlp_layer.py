"""emlp_layer.py -- Phase 2 of symmetry-driven contract generation: BUILD a genuinely new G-equivariant
layer from discovered group generators, via the EMLP nullspace construction (Finzi, Welling, Wilson 2021).

Given the infinitesimal generators {A_k} of a matrix group G (which the symmetry front-end discovers) and
representations rho_in, rho_out on the layer's in/out feature spaces, the equivariant linear maps
W: V_in -> V_out are exactly those satisfying, for every generator,
        drho_out(A_k) W - W drho_in(A_k) = 0.
Vectorising (column-major) turns this into a homogeneous linear system C vec(W) = 0 with
        C = stack_k [ I_in (x) drho_out(A_k)  -  drho_in(A_k)^T (x) I_out ].
The EQUIVARIANT BASIS is null(C) (via SVD); a learnable layer is a trainable combination
W(theta) = sum_j theta_j Q_j of that basis. By construction the layer commutes with the group action to
machine precision -- for any g = exp(sum_k t_k A_k), W(theta) rho_in(g) = rho_out(g) W(theta).

This is the SAME nullspace-of-a-linear-operator computation the symmetry front-end runs (the Lie-derivative
nullspace), run in the other direction: from generators to layers. It is fully general -- the identical
code builds SO(3), O(1,3) (Lorentz), O(5), Sp(n), ... equivariant layers. That is what lets ilmarinen
GENERATE a new contract for a discovered group that is not among the eight built-in ones (e.g. the Lorentz
contract for particle-physics 4-vectors), rather than forcing the data into an existing box.

Validated (tests/emlp_contract.md): exact equivariance to machine precision for SO(3) and Lorentz; an
expressive basis under direct-sum hidden reps (2 vectors + 3 scalars -> 13 learnable equivariant params);
and, wrapped as a network, it exploits the symmetry -- beating a plain MLP on an equivariant target, with
the gain growing as data shrinks (a LOSSLESS inductive bias, unlike Phase-1 canonicalization).
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- representation algebra
def direct_sum(gens, mult):
    """`mult` copies of a representation: block-diagonal generators (dim d -> d*mult)."""
    out = []
    for A in gens:
        d = A.shape[0]
        M = np.zeros((d * mult, d * mult))
        for m in range(mult):
            M[m * d:(m + 1) * d, m * d:(m + 1) * d] = A
        out.append(M)
    return out


def scalar_rep(gens, n_scalar):
    """Trivial (scalar) representation: every generator acts as zero (dim n_scalar)."""
    return [np.zeros((n_scalar, n_scalar)) for _ in gens]


def stack_reps(reps):
    """Block-diagonal concatenation of several representations (each a list of generator matrices, one per
    group generator, in the SAME order) into one feature space."""
    ngen = len(reps[0])
    dims = [r[0].shape[0] for r in reps]
    D = sum(dims)
    out = []
    for k in range(ngen):
        M = np.zeros((D, D))
        off = 0
        for r in reps:
            d = r[k].shape[0]
            M[off:off + d, off:off + d] = r[k]
            off += d
        out.append(M)
    return out


def hidden_rep(gens, n_vector, n_scalar):
    """A standard hidden representation: n_vector copies of the base (vector) rep + n_scalar scalars."""
    parts = []
    if n_vector > 0:
        parts.append(direct_sum(gens, n_vector))
    if n_scalar > 0:
        parts.append(scalar_rep(gens, n_scalar))
    return stack_reps(parts) if len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------- equivariant basis (nullspace)
def equivariant_basis(gens_in, gens_out, tol=1e-6):
    """The basis of equivariant linear maps V_in -> V_out for a group given by its generators on each
    space. Returns a list of (d_out x d_in) matrices spanning the nullspace of the equivariance
    constraint. Empty list if the only equivariant map is zero."""
    d_in = gens_in[0].shape[0]
    d_out = gens_out[0].shape[0]
    I_in = np.eye(d_in)
    I_out = np.eye(d_out)
    C = np.vstack([np.kron(I_in, Ao) - np.kron(Ai.T, I_out) for Ai, Ao in zip(gens_in, gens_out)])
    _, S, Vt = np.linalg.svd(C)
    smax = max(S.max(), 1.0) if len(S) else 1.0
    rank = int((S > tol * smax).sum())
    null = Vt[rank:]
    return [v.reshape(d_out, d_in, order="F") for v in null]


def equivariance_residual(W, gens_in, gens_out):
    """Max |drho_out(A) W - W drho_in(A)| over generators -- 0 iff W is exactly equivariant."""
    return max(np.abs(Ao @ W - W @ Ai).max() for Ai, Ao in zip(gens_in, gens_out))


# --------------------------------------------------------------------------- torch layer + net
def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class EquivariantLinear:
    """A G-equivariant linear layer W(theta) = sum_j theta_j Q_j, where {Q_j} is the equivariant basis for
    the given in/out representations. Realised as a torch module (built lazily to avoid a hard torch import
    at module load). If the basis is empty the layer is the zero map (caller should widen the reps)."""

    def __init__(self, gens_in, gens_out, bias_rep_gens=None):
        self.basis = equivariant_basis(gens_in, gens_out)
        self.d_in = gens_in[0].shape[0]
        self.d_out = gens_out[0].shape[0]
        self.gens_in = gens_in
        self.gens_out = gens_out
        # an equivariant bias is only allowed along invariant (scalar) output directions; we detect them
        # as the equivariant maps from the trivial rep to V_out.
        triv = [np.zeros((1, 1)) for _ in gens_out]
        self.bias_basis = equivariant_basis(triv, gens_out)

    def torch_module(self, dtype=None):
        torch, nn = _import_torch()

        class _EqLin(nn.Module):
            def __init__(m, basis, bias_basis, d_in, d_out):
                super().__init__()
                if basis:
                    Q = np.stack(basis, 0)                       # (nbasis, d_out, d_in)
                    m.register_buffer("Q", torch.tensor(Q, dtype=torch.float32))
                    m.theta = nn.Parameter(torch.randn(len(basis)) / np.sqrt(len(basis)))
                else:
                    m.register_buffer("Q", torch.zeros(1, d_out, d_in))
                    m.theta = nn.Parameter(torch.zeros(1))
                if bias_basis:
                    Bb = np.stack(bias_basis, 0)[:, :, 0]        # (nbias, d_out)
                    m.register_buffer("Bb", torch.tensor(Bb, dtype=torch.float32))
                    m.beta = nn.Parameter(torch.zeros(len(bias_basis)))
                else:
                    m.register_buffer("Bb", torch.zeros(1, d_out))
                    m.beta = nn.Parameter(torch.zeros(1))
                m.has_bias = bool(bias_basis)

            def weight(m):
                return torch.einsum("j,joi->oi", m.theta, m.Q)   # (d_out, d_in)

            def forward(m, x):                                    # x: (..., d_in)
                W = m.weight()
                out = x @ W.T
                if m.has_bias:
                    out = out + torch.einsum("j,jo->o", m.beta, m.Bb)
                return out

        return _EqLin(self.basis, self.bias_basis, self.d_in, self.d_out)


def group_element(gens, coeffs):
    """g = exp(sum_k coeffs_k gens_k) -- a finite group element from Lie-algebra coefficients."""
    from scipy.linalg import expm
    A = sum(c * g for c, g in zip(coeffs, gens))
    return expm(A)


# --------------------------------------------------------------------------- equivariant bilinear (invariants)
def equivariant_bilinear_invariants(v_flat, n_vec, vec_dim=3, metric=None):
    """Form invariant scalars from the vector channels of a hidden feature by taking pairwise inner
    products v_i . v_j (with an optional metric for pseudo-orthogonal groups like the Lorentz group). For
    a group preserving the (metric) inner product, these are exact invariants -- the standard mechanism by
    which equivariant networks gain INVARIANT expressivity (a pure equivariant-linear + gated net cannot
    form v (x) v -> scalar). Returns the (..., n_vec*(n_vec+1)/2) tensor of upper-triangular inner
    products. Torch-friendly (operates on tensors)."""
    torch, _ = _import_torch()
    V = v_flat.reshape(*v_flat.shape[:-1], n_vec, vec_dim)          # (..., n_vec, vec_dim)
    if metric is not None:
        g = torch.tensor(metric, dtype=V.dtype, device=V.device)
        Vg = V @ g
    else:
        Vg = V
    G = torch.einsum("...id,...jd->...ij", Vg, V)                   # (..., n_vec, n_vec) Gram matrix
    iu = torch.triu_indices(n_vec, n_vec)
    return G[..., iu[0], iu[1]]                                     # upper triangle (incl diagonal = norms)


class EquivariantMLP:
    """A small G-equivariant network: alternating equivariant-linear layers and an invariant-forming
    bilinear step, ending in an invariant scalar readout. Built from generators alone, so the SAME class
    yields an SO(3) net, a Lorentz net, etc. `metric` (e.g. diag(1,-1,-1,-1) for Lorentz) makes the inner
    products the group-correct invariants. This is the network form of a GENERATED contract."""

    def __init__(self, gens, n_in_vec, vec_dim, hidden_vec=4, hidden_scalar=8, depth=2,
                 n_out=1, metric=None):
        self.gens = gens
        self.vec_dim = vec_dim
        self.metric = metric
        self.in_rep = direct_sum(gens, n_in_vec)
        self.hidden_vec = hidden_vec
        self.hidden_scalar = hidden_scalar
        self.hid_rep = hidden_rep(gens, n_vector=hidden_vec, n_scalar=hidden_scalar)
        self.depth = depth
        self.n_out = n_out
        self.n_in_vec = n_in_vec

    def torch_module(self):
        torch, nn = _import_torch()
        outer = self

        class _Net(nn.Module):
            def __init__(m):
                super().__init__()
                m.lin_in = EquivariantLinear(outer.in_rep, outer.hid_rep).torch_module()
                m.lins = nn.ModuleList([EquivariantLinear(outer.hid_rep, outer.hid_rep).torch_module()
                                        for _ in range(outer.depth)])
                vdim = outer.hidden_vec * outer.vec_dim
                n_inv = outer.hidden_vec * (outer.hidden_vec + 1) // 2   # pairwise inner products
                # scalar MLP consumes [hidden scalars + invariants] -> hidden scalars (free, invariant)
                m.scalar_mlp = nn.ModuleList([
                    nn.Sequential(nn.Linear(outer.hidden_scalar + n_inv, outer.hidden_scalar), nn.Tanh())
                    for _ in range(outer.depth)])
                m.readout = nn.Sequential(nn.Linear(outer.hidden_scalar + n_inv, 32), nn.Tanh(),
                                          nn.Linear(32, outer.n_out))
                m.vdim = vdim
                m.n_vec = outer.hidden_vec
                m.hs = outer.hidden_scalar

            def _split(m, h):
                return h[..., :m.vdim], h[..., m.vdim:]

            def _invariants(m, vpart):
                return equivariant_bilinear_invariants(vpart, m.n_vec, outer.vec_dim, outer.metric)

            def forward(m, x):
                h = m.lin_in(x)
                for lin, smlp in zip(m.lins, m.scalar_mlp):
                    h = lin(h)
                    v, s = m._split(h)
                    inv = m._invariants(v)
                    vn = v.reshape(*v.shape[:-1], m.n_vec, outer.vec_dim)
                    vn = vn * torch.tanh(vn.norm(dim=-1, keepdim=True))     # equivariant gate
                    s = smlp(torch.cat([s, inv], dim=-1))                   # invariant scalar update
                    h = torch.cat([vn.reshape(*v.shape[:-1], m.vdim), s], dim=-1)
                v, s = m._split(h)
                inv = m._invariants(v)
                return m.readout(torch.cat([s, inv], dim=-1))              # invariant readout

        return _Net()


# --------------------------------------------------------------------------- conformal group (null-cone lift)
def conformal_generators(d):
    """Generators of the conformal group Conf(d) realised as the pseudo-orthogonal algebra so(d+1,1) acting
    LINEARLY on the (d+2)-dimensional null-cone lift of R^d. Conformal transformations (translations,
    rotations, dilations, special conformal transformations) are NONLINEAR on R^d but become linear on the
    lift X = (1, x, |x|^2) restricted to the null cone of the light-cone metric (middle block identity,
    with an off-diagonal +/- pairing of the first and last coordinates so that <X,X> = |x|^2 - 1*|x|^2 = 0).
    Returns (generators, metric): the so(d+1,1) basis and the (d+2)x(d+2) light-cone metric. Because these
    are just another matrix group with a metric, the SAME equivariant_basis / EquivariantMLP machinery
    builds conformal-equivariant layers (validated: <X,X>=0 exactly, translations linear to ~1e-16,
    equivariant basis residuals ~1e-15)."""
    import numpy as _np
    D = d + 2
    eta = _np.zeros((D, D))
    for k in range(1, 1 + d):
        eta[k, k] = 1.0
    eta[0, D - 1] = -0.5
    eta[D - 1, 0] = -0.5                              # light-cone pairing of coords 0 and D-1
    gens = []
    inv_eta = _np.linalg.inv(eta)
    for i in range(D):
        for j in range(i + 1, D):
            S = _np.zeros((D, D)); S[i, j] = 1; S[j, i] = -1
            gens.append(inv_eta @ S)                  # so(d+1,1) wrt the light-cone metric
    return gens, eta


def null_cone_lift(x):
    """Lift a point x in R^d to the null cone of R^{d+1,1}: X = (1, x, |x|^2). On this lift the conformal
    group acts linearly (see conformal_generators). Torch- and numpy-friendly."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            xsq = (x ** 2).sum(-1, keepdim=True)
            ones = torch.ones_like(xsq)
            return torch.cat([ones, x, xsq], dim=-1)
    except ImportError:
        pass
    import numpy as _np
    x = _np.asarray(x, float)
    xsq = (x ** 2).sum(-1, keepdims=True)
    return _np.concatenate([_np.ones_like(xsq), x, xsq], axis=-1)


# --------------------------------------------------------------------------- symplectic / special-linear
def symplectic_generators(n):
    """Generators of the symplectic Lie algebra sp(2n): matrices A with A^T Omega + Omega A = 0, where
    Omega = [[0, I],[-I, 0]] is the canonical symplectic form. These are A = Omega^{-1} S for symmetric S,
    giving n(2n+1) generators. Returns (generators, Omega). The same equivariant_basis / EquivariantLinear
    machinery builds Sp(2n)-equivariant layers; only the INVARIANT-forming step differs (the skew pairing
    omega(u,v) = u^T Omega v between distinct vectors, since the self-pairing vanishes)."""
    import numpy as _np
    D = 2 * n
    Om = _np.zeros((D, D)); Om[:n, n:] = _np.eye(n); Om[n:, :n] = -_np.eye(n)
    inv = _np.linalg.inv(Om)
    gens = []
    for i in range(D):
        for j in range(i, D):
            S = _np.zeros((D, D)); S[i, j] = 1.0; S[j, i] = 1.0
            gens.append(inv @ S)
    return gens, Om


def special_linear_generators(n):
    """Generators of the special-linear Lie algebra sl(n): the traceless matrices (off-diagonal E_ij for
    i != j, plus the Cartan diagonal generators diag(...,1,-1,...)), giving n^2-1 generators. The
    SL(n)-invariant of n vectors is their determinant det[v_1,...,v_n] (an n-linear alternating form,
    invariant because det(exp(traceless)) = 1), which replaces the bilinear metric invariant."""
    import numpy as _np
    gens = []
    for i in range(n):
        for j in range(n):
            if i != j:
                A = _np.zeros((n, n)); A[i, j] = 1.0; gens.append(A)
    for i in range(n - 1):
        A = _np.zeros((n, n)); A[i, i] = 1.0; A[i + 1, i + 1] = -1.0; gens.append(A)
    return gens


def symplectic_invariants(V, hv, vec_dim, omega):
    """Skew (symplectic) invariants of hv hidden vectors packed in V (shape (..., hv*vec_dim)): the
    antisymmetric pairings omega(v_a, v_b) = v_a^T Omega v_b for a < b. Torch tensor in, torch tensor out.
    Self-pairings vanish, so only the a<b upper triangle is informative (hv(hv-1)/2 invariants)."""
    import torch
    Om = torch.as_tensor(omega, dtype=V.dtype, device=V.device)
    lead = V.shape[:-1]
    Vr = V.reshape(*lead, hv, vec_dim)
    # pairings: <v_a, Omega v_b>
    OmV = torch.einsum('ij,...bj->...bi', Om, Vr)                 # Omega v_b
    pair = torch.einsum('...ai,...bi->...ab', Vr, OmV)           # v_a^T Omega v_b
    iu = torch.triu_indices(hv, hv, offset=1)
    return pair[..., iu[0], iu[1]]                                # (..., hv(hv-1)/2)


def determinant_invariants(V, hv, vec_dim):
    """SL(vec_dim)-invariant determinants of hidden vectors packed in V (shape (..., hv*vec_dim)): the
    determinant of each consecutive vec_dim-frame of the hv hidden vectors. Requires hv >= vec_dim.
    Torch tensor in, torch tensor out (one determinant per frame; frames overlap by stride 1)."""
    import torch
    lead = V.shape[:-1]
    Vr = V.reshape(*lead, hv, vec_dim)
    n = vec_dim
    dets = []
    for start in range(0, hv - n + 1):
        frame = Vr[..., start:start + n, :]                       # (..., n, vec_dim)
        dets.append(torch.linalg.det(frame))
    if not dets:
        return torch.zeros(*lead, 1, dtype=V.dtype, device=V.device)
    return torch.stack(dets, dim=-1)                              # (..., hv-n+1)

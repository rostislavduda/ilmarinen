"""Family-2 symmetry discovery: recover the continuous (Lie-group) symmetries of a trained
network by solving the Lie-derivative nullspace, following the SVD-based approach (Yang et al.
LieGAN 2023; the linear-algebraic variant, ScienceDirect 2025).

Principle. A generator L in gl(n) is an infinitesimal symmetry of a function f iff the Lie
derivative of f along the flow exp(theta*L) vanishes:

    d/dtheta f(exp(theta L) x)|_0 = grad_x f(x)^T (L x) = 0   for all data x.

For a scalar output this is ONE linear equation in the n^2 entries of L per data point; for a
vector output (n_out>1) it is n_out equations. Stacking over data gives a linear system
M vec(L) = 0 whose NULLSPACE is the discovered Lie algebra. We obtain it by SVD of M: the number
of near-zero singular values is the dimension of the symmetry algebra, and the corresponding
right-singular vectors are (an orthogonal basis of) its generators.

This is the physics-native counterpart of the discrete architecture search: instead of relaxing
a discrete group choice (non-differentiable), it works directly with the Lie ALGEBRA (continuous
matrices), which is gradient-able / linear-algebraic. It reaches continuous LINEAR symmetries
(rotations SO(n), scalings, boosts/Lorentz, translations in an augmented coordinate) -- NOT
discrete groups (permutations, parity), which have no generator. That boundary is intrinsic:
continuity is exactly what supplies the algebra.

The discovered generators connect back to primitive (4) (weight-sharing under a group): imposing
a discovered symmetry L on a layer W means constraining W to the COMMUTANT [W, L] = 0 -- the
quantum-mechanical statement that W is conserved under the symmetry.
"""

from __future__ import annotations

import numpy as np
import torch


def lie_derivative_matrix(net, X, output_index=None, basis=None):
    """Build the Lie-derivative operator M such that M @ vec(L) = 0 encodes infinitesimal
    invariance of `net` at the points X.

    net    : a torch module mapping (b, n_in) -> (b, n_out)  (n_out may be 1)
    X      : (n_samples, n_in) tensor of evaluation points
    output_index : if net has vector output, which output coordinate to test invariance of;
                   None means sum over outputs (invariance of every coordinate jointly).
    basis  : optional (K, n, n) array of generator basis matrices to project onto; if None,
             the full gl(n) basis (n^2 entries of L) is used and M has n^2 columns.

    Returns M as a numpy array of shape (n_rows, n^2) [or (n_rows, K) if a basis is given].
    """
    X = X.clone().detach().requires_grad_(True)
    n = X.shape[1]
    out = net(X)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    rows = []
    n_out = out.shape[1]
    idxs = [output_index] if output_index is not None else list(range(n_out))
    for oi in idxs:
        g = torch.autograd.grad(out[:, oi].sum(), X, retain_graph=True)[0]  # (b, n)
        # condition grad^T (L x) = sum_{i,j} g_i L_ij x_j = 0
        # row entries for vec(L) (row-major L_ij): coefficient of L_ij is g_i * x_j
        gi = g.unsqueeze(2)  # (b, n, 1)
        xj = X.unsqueeze(1)  # (b, 1, n)
        coeff = (gi * xj).reshape(X.shape[0], n * n)  # (b, n^2)
        rows.append(coeff.detach().cpu().numpy())
    M = np.concatenate(rows, axis=0)
    if basis is not None:
        B = np.stack([b.reshape(-1) for b in basis], axis=1)  # (n^2, K)
        M = M @ B
    return M


def discover_symmetries(net, X, output_index=None, basis=None, tol_ratio=1.8):
    """Return the discovered Lie-algebra generators of `net` as the SVD nullspace of the
    Lie-derivative operator.

    tol_ratio : a singular value sigma_k is treated as 'null' (a symmetry direction) if the
                gap sigma_{k-1}/sigma_k exceeds tol_ratio, scanning from the smallest upward.
                This locates the spectral gap that separates symmetry directions (near-zero)
                from non-symmetry directions.

    Returns dict with:
      singular_values : all singular values (ascending)
      n_symmetries    : estimated dimension of the symmetry algebra (from the gap)
      generators      : list of (n,n) generator matrices (normalized), smallest-sigma first
      gap_ratio       : sigma[second-smallest]/sigma[smallest] (clear symmetry => large)
    """
    n = X.shape[1]
    M = lie_derivative_matrix(net, X, output_index=output_index, basis=basis)
    # row-normalize so the singular scale is comparable across problems
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S_asc = S[::-1]
    V_asc = Vt[::-1]
    # locate spectral gap from the bottom: how many near-null directions?
    n_sym = 1
    for k in range(1, len(S_asc)):
        if S_asc[k] / (S_asc[k - 1] + 1e-12) > tol_ratio:
            n_sym = k
            break
    else:
        n_sym = 0 if S_asc[0] > 1e-3 else len(S_asc)
    gens = []
    K = basis.shape[0] if basis is not None else n * n
    for k in range(max(n_sym, 1)):
        v = V_asc[k]
        if basis is not None:
            L = sum(v[i] * basis[i] for i in range(K))
        else:
            L = v.reshape(n, n)
        L = L / (np.max(np.abs(L)) + 1e-12)
        gens.append(L)
    gap_ratio = float(S_asc[1] / (S_asc[0] + 1e-12)) if len(S_asc) > 1 else float("inf")
    return {
        "singular_values": S_asc,
        "n_symmetries": n_sym,
        "generators": gens,
        "gap_ratio": gap_ratio,
        "smallest_sv": float(S_asc[0]),
    }


def affine_lie_matrix(net, X, output_index=None):
    """Lie-derivative operator for AFFINE vector fields v(x) = L x + c (extends the linear case).

    Invariance condition: grad_f(x)^T (L x + c) = 0 for all x. The unknowns are vec(L) (n^2
    entries) followed by c (n entries), so M has n^2 + n columns. The linear part's coefficient of
    L_ij is g_i x_j (as in lie_derivative_matrix); the translation part's coefficient of c_i is g_i.

    Translation is the pure-c generator (L = 0, c = shift direction), which the linear-only detector
    cannot see because it has no matrix form; the affine ansatz recovers it.
    """
    X = X.clone().detach().requires_grad_(True)
    n = X.shape[1]
    out = net(X)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    rows = []
    idxs = [output_index] if output_index is not None else list(range(out.shape[1]))
    for oi in idxs:
        g = torch.autograd.grad(out[:, oi].sum(), X, retain_graph=True)[0]  # (b, n)
        coeffL = (g.unsqueeze(2) * X.unsqueeze(1)).reshape(X.shape[0], n * n)  # L_ij -> g_i x_j
        coeffc = g  # c_i  -> g_i
        rows.append(torch.cat([coeffL, coeffc], dim=1).detach().cpu().numpy())
    return np.concatenate(rows, axis=0), n


def discover_affine_symmetries(net, X, output_index=None, tol_ratio=1.8):
    """Discover affine symmetries v(x)=Lx+c (linear generators AND translations) via the nullspace
    of the affine Lie-derivative operator. Each nullspace vector splits into an (n,n) matrix L and
    a length-n translation c; we classify each discovered generator as 'translation' (c dominates),
    'linear' (L dominates), or 'mixed'.

    Returns dict with:
      singular_values, n_symmetries, gap_ratio  (as in discover_symmetries)
      generators : list of dicts {L, c, kind, c_dir} for each discovered affine generator
    """
    M, n = affine_lie_matrix(net, X, output_index=output_index)
    # NOTE: unlike the linear detector we do NOT row-normalize here -- row-normalization compresses
    # the spectral gap for translation generators (whose rows have a different scale than linear
    # ones), hiding the nullspace. The raw operator gives a clean gap.
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S_asc, V_asc = S[::-1], Vt[::-1]
    n_sym = 1
    for k in range(1, len(S_asc)):
        if S_asc[k] / (S_asc[k - 1] + 1e-12) > tol_ratio:
            n_sym = k
            break
    else:
        n_sym = 0 if S_asc[0] > 1e-3 else len(S_asc)
    gens = []
    for k in range(max(n_sym, 1)):
        v = V_asc[k]
        L = v[: n * n].reshape(n, n)
        c = v[n * n :]
        lnorm, cnorm = np.linalg.norm(L), np.linalg.norm(c)
        if cnorm > 3 * lnorm:
            kind = "translation"
        elif lnorm > 3 * cnorm:
            kind = "linear"
        else:
            kind = "mixed"
        scale = max(np.max(np.abs(L)), np.max(np.abs(c))) + 1e-12
        gens.append(
            {"L": L / scale, "c": c / scale, "kind": kind, "c_dir": (c / (cnorm + 1e-12)) if cnorm > 1e-6 else c}
        )
    return {
        "singular_values": S_asc,
        "n_symmetries": n_sym,
        "gap_ratio": float(S_asc[1] / (S_asc[0] + 1e-12)) if len(S_asc) > 1 else float("inf"),
        "generators": gens,
    }


# ---- named generator bank for identification (2D and 3D common Lie generators) -------------
def generator_bank(n):
    """A dict of named standard generators in gl(n) for interpreting discovered algebras."""
    bank = {}
    if n == 2:
        bank["rotation"] = np.array([[0, -1], [1, 0]], float)
        bank["scaling"] = np.eye(2)
        bank["squeeze"] = np.array([[1, 0], [0, -1]], float)  # area-preserving squeeze
        bank["shear_x"] = np.array([[0, 1], [0, 0]], float)
    elif n == 3:
        bank["rot_xy"] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)
        bank["rot_yz"] = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        bank["rot_xz"] = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], float)
        bank["scaling"] = np.eye(3)
    return bank


def identify_generator(L, n):
    """Return the named generator (from the bank) most aligned with L, and the |cosine|."""
    bank = generator_bank(n)
    best, best_c = None, 0.0
    a = L.flatten()
    for name, B in bank.items():
        b = B.flatten()
        c = abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best_c:
            best, best_c = name, c
    return best, best_c

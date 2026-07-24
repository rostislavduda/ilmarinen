"""Discrete symmetry discovery -- the DISCRETE stage of the physicist pipeline.

The continuous (Lie) stage (core/symmetry_discovery.py) uses the Lie-derivative NULLSPACE, which
requires a generator. Discrete groups (Z_2 reflections/parity, cyclic C_n, dihedral D_n,
permutations S_n) have NO generator, so that method is blind to them. Instead we use direct
EQUIVARIANCE TESTING: a candidate discrete transformation g (a matrix with g^k = I) is a symmetry
of a trained function f iff f(g x) = f(x) on the data. We measure the normalized equivariance
error E(g) = mean_x |f(g x) - f(x)| / mean_x |f(x)|; small E(g) => g is a symmetry.

Pipeline placement (why discrete runs AFTER continuous): any Lie group G decomposes into a
connected part G^0 (continuous, has an algebra) and a discrete quotient pi_0(G) = G/G^0 (e.g.
O(n) = SO(n) semidirect Z_2). Continuous discovery is tractable (nullspace) and, by quotienting it
out (forming invariant features), it SHRINKS the residual discrete search to pi_0(G) rather than
searching the raw high-dimensional space. So the default order is continuous-first; if the
continuous pass returns an empty algebra, the symmetry is purely discrete and we come straight
here. The operations do not commute -- continuous-first is a filter that simplifies the residual.

This module handles Z_2 (involutions: g^2 = I) first -- reflections and parity, the simplest and
second-most-common discrete symmetry after permutation. S_n (permutation) is a separate, larger
search built on the same equivariance-testing primitive.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch


def equivariance_error(net, g, X, output_index=None):
    """Normalized equivariance error E(g) = mean|f(gX) - f(X)| / mean|f(X)| for a linear g."""
    gt = torch.tensor(np.asarray(g), dtype=torch.float32)
    with torch.no_grad():
        fX = net(X)
        fgX = net(X @ gt.T)
        if fX.dim() > 1 and output_index is not None:
            fX, fgX = fX[:, output_index], fgX[:, output_index]
        num = (fgX - fX).abs().mean()
        den = fX.abs().mean() + 1e-6
    return float(num / den)


def grad_weighted_equivariance_error(net, g, X, output_index=None, eps=1e-6):
    """Gradient-sensitivity-weighted equivariance error E_grad(g). Like scale_aware_equivariance_error, it
    distinguishes a genuine symmetry from FLATNESS -- but calibrates against the function's OWN gradient in
    the directions g acts, DIRECTLY, rather than against random matched-size transformations. This is the
    report's proposed hardening: only test a transformation along directions in which f genuinely varies, so
    a coordinate f barely depends on cannot contribute a spurious symmetry.

        E_grad(g) = mean_x |f(gx) - f(x)|  /  ( ||g - I||_F * rms_x || d f/dx |_{coords(g)} || )

    The denominator is the function's typical sensitivity RESTRICTED to the coordinates g moves, scaled by
    g's displacement size. Interpretation matches the scale-aware ratio:
      - genuine symmetry: numerator ~ 0, denominator LARGE      -> E_grad ~ 0     (accept).
      - flatness:         numerator ~ 0, denominator ALSO ~ 0   -> E_grad ~ O(1)  (reject; f ignores coords).
      - non-symmetry:     numerator LARGE, denominator LARGE    -> E_grad ~ O(1)  (reject).
    Cost: ONE backward pass (df/dx) plus two forward passes, versus the scale-aware ratio's n_rand matrix
    exponentials and forward passes -- and it needs no scipy. A threshold ~0.1 cleanly separates genuine
    symmetries (~0) from both false-positive modes (~0.3-0.6), matching the scale-aware separation at lower
    cost. If g moves nothing, returns 1.0 (vacuous, not a symmetry)."""
    g = np.asarray(g)
    coords = _affected_coords(g)
    if not coords:
        return 1.0
    gt = torch.tensor(g, dtype=torch.float32)
    Xr = X.clone().detach().requires_grad_(True)
    fXr = net(Xr)
    fXo = fXr[:, output_index] if (fXr.dim() > 1 and output_index is not None) else fXr.squeeze(-1)
    grad, = torch.autograd.grad(fXo.sum(), Xr)                 # (n, d) = df/dx per sample
    with torch.no_grad():
        fX = net(X)
        fgX = net(X @ gt.T)
        if fX.dim() > 1 and output_index is not None:
            fX, fgX = fX[:, output_index], fgX[:, output_index]
        num = float((fgX - fX).abs().mean())
        gsize = float(np.linalg.norm(g - np.eye(g.shape[0])))
        gsub = grad[:, coords]                                 # sensitivity on the acted-on subspace only
        sens = float(gsub.pow(2).sum(1).sqrt().mean())         # rms ||grad|_coords|| over samples
        den = gsize * sens + eps
    return num / den


def _affected_coords(g, thresh=0.1):
    """Coordinates that g actually moves (rows/cols differing from identity)."""
    d = g.shape[0]
    D = np.abs(g - np.eye(d))
    return [i for i in range(d) if D[i].max() > thresh or D[:, i].max() > thresh]


def scale_aware_equivariance_error(net, g, X, n_rand=8, output_index=None, seed=None):
    """Scale-aware equivariance ratio R(g) that distinguishes a genuine symmetry from FLATNESS.

    The standard error E(g) is small in TWO different situations it cannot tell apart:
      (i) genuine symmetry: f varies a lot in general, but g leaves it unchanged;
      (ii) flatness: f barely depends on the coordinates g moves, so f(gx)~f(x) VACUOUSLY.
    This conflation is the root cause of false-positive symmetries on smooth/correlated data
    (e.g. adjacent time-points look exchangeable). We calibrate against the function's OWN
    sensitivity in exactly the directions g acts:

        R(g) = mean|f(gx) - f(x)|  /  mean_over_random-g'|f(g' x) - f(x)|

    where each g' is a RANDOM transformation of the same size (matched Frobenius norm of g - I),
    acting on the SAME coordinates g touches. Then:
      - genuine symmetry: numerator ~ 0, denominator LARGE  ->  R ~ 0.
      - flatness:         numerator ~ 0, denominator ALSO ~ 0 -> R ~ 1 (correctly NOT a symmetry).
      - non-symmetry:     numerator LARGE, denominator LARGE -> R ~ 1.
    Empirically genuine symmetries give R ~ 0.01 while flatness/non-symmetry give R >~ 0.2, a ~15x
    separation, so a threshold ~0.05-0.10 cleanly rejects flatness false-positives WITHOUT any prior
    about the data's coordinate structure. This is the robust, data-driven absence test.

    Requires scipy for the matrix exponential; falls back to the standard error if unavailable.
    """
    if seed is not None:
        np.random.seed(seed)
    gt = torch.tensor(np.asarray(g), dtype=torch.float32)
    d = X.shape[1]
    with torch.no_grad():
        fX = net(X)
        if fX.dim() > 1 and output_index is not None:
            fX = fX[:, output_index]
        fgX = net(X @ gt.T)
        if fgX.dim() > 1 and output_index is not None:
            fgX = fgX[:, output_index]
        num = float((fgX - fX).abs().mean())
    coords = _affected_coords(np.asarray(g))
    if not coords:
        return 1.0                                    # g moves nothing -> vacuous, not a symmetry
    try:
        from scipy.linalg import expm
    except Exception:
        return float(num / (fX.abs().mean() + 1e-6))  # fallback to standard error
    gsize = np.linalg.norm(np.asarray(g) - np.eye(d))
    denoms = []
    for _ in range(n_rand):
        P = np.eye(d)
        A = np.random.randn(len(coords), len(coords))
        A = A - A.T                                    # antisymmetric -> generic rotation on subspace
        Rsub = expm((gsize / (np.linalg.norm(A) + 1e-9)) * A)
        for a, i in enumerate(coords):
            for b, j in enumerate(coords):
                P[i, j] = Rsub[a, b]
        with torch.no_grad():
            fPX = net(X @ torch.tensor(P, dtype=torch.float32).T)
            if fPX.dim() > 1 and output_index is not None:
                fPX = fPX[:, output_index]
            denoms.append(float((fPX - fX).abs().mean()))
    return num / (float(np.mean(denoms)) + 1e-6)


def axis_reflections(n):
    """The n single-axis reflections R_i (flip coordinate i), each an involution R_i^2 = I."""
    refs = {}
    for i in range(n):
        R = np.eye(n); R[i, i] = -1.0
        refs[f"reflect_x{i}"] = R
    return refs


def parity(n):
    """Full parity P = -I (x -> -x), the canonical Z_2. Involution."""
    return {"parity": -np.eye(n)}


def swap_reflections(n):
    """Coordinate-swap involutions (i j): exchange axes i and j. Each is an involution.
    These are the transpositions -- the Z_2 building blocks of permutation symmetry S_n."""
    swaps = {}
    for i, j in itertools.combinations(range(n), 2):
        S = np.eye(n); S[i, i] = S[j, j] = 0.0; S[i, j] = S[j, i] = 1.0
        swaps[f"swap_{i}{j}"] = S
    return swaps


def _select_err_fn(scale_aware, error_mode):
    """Resolve the equivariance-error function. error_mode (if given) wins: 'standard' | 'scale_aware' |
    'grad_weighted'. Otherwise falls back to the legacy scale_aware boolean. grad_weighted is the
    gradient-sensitivity-weighted hardening: cheaper than scale_aware (one backward pass, no scipy) and
    equally rejects flatness false-positives by calibrating against f's own gradient in the acted-on
    directions."""
    mode = error_mode if error_mode is not None else ("scale_aware" if scale_aware else "standard")
    return {"standard": equivariance_error,
            "scale_aware": scale_aware_equivariance_error,
            "grad_weighted": grad_weighted_equivariance_error}[mode]


def discover_z2(net, X, candidates=None, tol=0.15, output_index=None, scale_aware=False, error_mode=None):
    """Test candidate Z_2 involutions and return which the trained net supports as symmetries.

    net        : trained torch module (b, n) -> (b, n_out)
    X          : (n_samples, n) evaluation points
    candidates : dict {name: matrix}; default = axis reflections + parity + coordinate swaps
    tol        : an involution g is accepted as a symmetry if E(g) < tol (fraction of output scale)

    Returns dict with:
      errors     : {name: E(g)} for every candidate
      symmetries : sorted list of (name, E(g)) with E(g) < tol  (the discovered Z_2 symmetries)
      n_symmetries : count accepted
    """
    n = X.shape[1]
    if candidates is None:
        candidates = {}
        candidates.update(axis_reflections(n))
        candidates.update(parity(n))
        candidates.update(swap_reflections(n))
    err_fn = _select_err_fn(scale_aware, error_mode)
    errors = {name: err_fn(net, g, X, output_index=output_index) for name, g in candidates.items()}
    syms = sorted([(k, v) for k, v in errors.items() if v < tol], key=lambda kv: kv[1])
    return {"errors": errors, "symmetries": syms, "n_symmetries": len(syms),
            "candidates": candidates}


def transposition(n, i, j):
    """The transposition (i j): a permutation matrix swapping coordinates i and j. Involution."""
    S = np.eye(n); S[i, i] = S[j, j] = 0.0; S[i, j] = S[j, i] = 1.0
    return S


def discover_permutation_subgroup(net, X, tol=0.15, output_index=None, scale_aware=False, error_mode=None):
    """Discover the permutation symmetry of a trained net efficiently.

    S_n is generated by TRANSPOSITIONS, so rather than test all n! permutations we test the
    O(n^2) transpositions (i j) for equivariance, then recover the permutable BLOCKS as the
    connected components of the graph whose edges are the discovered transpositions (union-find).
    Coordinates in the same block are freely interchangeable => a symmetric group S_k acts on that
    block; the full discovered symmetry is the Young subgroup prod_k S_{|block_k|}.

    Returns dict with:
      transpositions : list of (i, j, E) accepted as symmetries (E < tol)
      blocks         : sorted list of permutable coordinate blocks (size > 1)
      young_subgroup : human-readable, e.g. "S_3 x S_2" (the discovered permutation symmetry)
    """
    n = X.shape[1]
    err_fn = _select_err_fn(scale_aware, error_mode)
    trans = []
    for i, j in itertools.combinations(range(n), 2):
        e = err_fn(net, transposition(n, i, j), X, output_index=output_index)
        if e < tol:
            trans.append((i, j, round(e, 4)))
    # union-find over coordinates joined by a discovered transposition
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _ in trans:
        parent[find(i)] = find(j)
    comp = {}
    for a in range(n):
        comp.setdefault(find(a), []).append(a)
    blocks = sorted([sorted(v) for v in comp.values() if len(v) > 1])
    sizes = sorted([len(b) for b in blocks], reverse=True)
    young = " x ".join(f"S_{s}" for s in sizes) if sizes else "trivial (no permutation symmetry)"
    return {"transpositions": trans, "blocks": blocks, "young_subgroup": young}


def build_permutation_invariant_features(X, blocks, moments=(1, 2, 3)):
    """Given discovered permutable blocks, build permutation-invariant features per block.

    For each block of coordinates, pooling by symmetric functions (power-sum moments
    sum_i x_i^p over the block) is invariant to permuting within the block -- the S_n analogue of
    the SO(3) Gram-matrix quotient. Coordinates outside any block are passed through unchanged.
    Returns the concatenated invariant feature vector.
    """
    n = X.shape[1]
    in_block = set()
    feats = []
    for blk in blocks:
        idx = torch.tensor(blk)
        Xb = X[:, idx]                                   # (b, |block|)
        for p in moments:
            feats.append((Xb ** p).sum(dim=1, keepdim=True))   # power-sum: permutation-invariant
        in_block.update(blk)
    passthrough = [c for c in range(n) if c not in in_block]
    if passthrough:
        feats.append(X[:, torch.tensor(passthrough)])
    return torch.cat(feats, dim=-1)


def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def discover_cyclic_dihedral(net, X, max_order=8, plane=(0, 1), tol=0.06, output_index=None):
    """Discover cyclic (C_n) and dihedral (D_n) point-group symmetry in a 2D plane of the input.

    C_n = rotations by 2*pi*k/n about the origin in the (i,j) plane; D_n = C_n plus a reflection.
    Uses the same equivariance-testing primitive as Z_2 (finite groups have no generator). We find
    the LARGEST n (up to max_order) such that the full C_n is a symmetry -- i.e. the fundamental
    2*pi/n rotation and all its powers have equivariance error < tol -- then test whether a
    reflection is also present (promoting C_n to D_n).

    plane : the pair of coordinate axes (i,j) the rotation acts in (default the first two).
    Returns dict: {cyclic_order, dihedral (bool), group, errors_by_order}.
    """
    n_dim = X.shape[1]
    i, j = plane

    def embed(R2):
        R = np.eye(n_dim)
        R[i, i], R[i, j], R[j, i], R[j, j] = R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]
        return R

    errors_by_order = {}
    passes = {}
    for order in range(2, max_order + 1):
        # C_order present iff EVERY nontrivial rotation 2pi*k/order is a symmetry.
        errs = [equivariance_error(net, embed(_rot2d(2 * np.pi * k / order)), X, output_index)
                for k in range(1, order)]
        errors_by_order[order] = float(max(errs))
        passes[order] = max(errs) < tol
    # The genuine cyclic order is the largest n that passes AND whose every proper divisor > 1 also
    # passes (a real C_n contains all C_d for d | n). This rejects isolated threshold-noise
    # false-positives (e.g. a spurious C_7 pass) that lack the divisor lattice a true group has.
    def divisors_consistent(n):
        for d in range(2, n):
            if n % d == 0 and not passes.get(d, False):
                return False
        return passes.get(n, False)
    best = 1
    for order in range(2, max_order + 1):
        if divisors_consistent(order):
            best = max(best, order)
    # test a reflection in the same plane (flip axis i) -> dihedral if present with the rotation
    refl = np.eye(n_dim); refl[i, i] = -1.0
    refl_err = equivariance_error(net, refl, X, output_index)
    dihedral = (best >= 2) and (refl_err < tol)
    if best <= 1:
        group = "trivial (no cyclic symmetry)"
    elif dihedral:
        group = f"D_{best}"
    else:
        group = f"C_{best}"
    return {"cyclic_order": best, "dihedral": dihedral, "reflection_error": float(refl_err),
            "group": group, "errors_by_order": errors_by_order}


def build_z2_invariant_features(X, g):
    """Given a discovered involution g, build the Z_2-invariant symmetrization of the coordinates:
    the g-symmetric part  (x + g x)/2  concatenated with the |g-antisymmetric| part |x - g x|/2.
    Both are invariant under x -> g x (the symmetric part is fixed; the antisymmetric flips sign,
    so its absolute value is invariant). This is the discrete analogue of the Gram-matrix quotient.
    """
    gt = torch.tensor(np.asarray(g), dtype=torch.float32)
    Xg = X @ gt.T
    sym = 0.5 * (X + Xg)
    anti = 0.5 * (X - Xg).abs()
    return torch.cat([sym, anti], dim=-1)

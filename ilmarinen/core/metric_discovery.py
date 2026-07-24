"""metric_discovery.py -- autonomous discovery of the invariant METRIC (and hence the O(p,q) signature) of
a symmetry group, generalising detection beyond the hardcoded {Euclidean, Lorentz} menu to ARBITRARY
pseudo-orthogonal groups.

Detection so far chooses between a fixed set of metrics (identity for SO(d)/E(d), Minkowski for O(1,d-1)).
But the metric itself is discoverable from data (LieGAN, Yang et al. 2023, recovers the Minkowski metric at
cosine 0.9998; AtlasD 2025 at 0.9996; Clifford metric-learning, Ali et al. 2024, learns the signature).
This module implements the metric solve and reads off the signature, so the group O(p,q) is discovered
rather than assumed.

THE MATH. A generator A lies in the Lie algebra of the group preserving a bilinear form g iff
        A^T g + g A = 0
(A is a g-isometry generator / g-antisymmetric). Given generators {A_k} -- from the symmetry front-end's
Lie-derivative nullspace -- the invariant metric is the symmetric g solving A_k^T g + g A_k = 0 for all k.
Vectorising, this is a nullspace problem (the SAME machinery emlp_layer uses for equivariant layers, with g
as the unknown instead of the layer W):
        g = null( stack_k [ I (x) A_k^T + A_k^T (x) I ] , restricted to symmetric g ).
The eigenvalue SIGNATURE of g -- (#positive, #negative, #zero), taken up to an overall sign gauge -- names
the group: (d,0) -> O(d) Euclidean; (1,d-1) -> O(1,d-1) Lorentz; general (p,q) -> O(p,q); a zero eigenvalue
signals a degenerate/scaling direction.

Validated (tests/metric_discovery.md): exact metrics from known SO(3)/Lorentz generators; correct
signatures from DATA-discovered noisy generators (Lorentz metric recovered to ~3%); and a NOVEL O(2,2)
split-signature group recovered from data -- a group neither the Euclidean nor Lorentz menu entry can
reach.
"""
from __future__ import annotations

import numpy as np


def invariant_metric(gens, tol=1e-6, allow_approx=True):
    """Solve A_k^T g + g A_k = 0 for a symmetric bilinear form g invariant under all generators A_k.

    Returns (g, nullity, residual): the (normalised) recovered metric, the dimension of the exact solution
    space, and the constraint residual |A^T g + g A| for the returned g. When the generators are exact the
    nullspace is clean; when they are noisy (discovered from data) and allow_approx is True, the smallest
    singular direction is returned as the approximate metric (this is what recovers Minkowski from noisy
    Lorentz generators, matching the LieGAN result).
    """
    gens = [np.asarray(A, float) for A in gens]
    d = gens[0].shape[0]
    rows = [np.kron(np.eye(d), A.T) + np.kron(A.T, np.eye(d)) for A in gens]
    # symmetry constraint g - g^T = 0 (column-major vec indexing)
    S = np.zeros((d * d, d * d))
    for i in range(d):
        for j in range(d):
            S[j * d + i, j * d + i] += 1.0
            S[j * d + i, i * d + j] -= 1.0
    M = np.vstack(rows + [S])
    _, s, Vt = np.linalg.svd(M)
    smax = max(s.max(), 1.0) if len(s) else 1.0
    null_mask = s < tol * smax
    nullity = int(null_mask.sum()) + (M.shape[1] - len(s))
    if nullity >= 1:
        # exact solution(s) exist; take the one with the largest spread (most informative metric)
        idxs = list(np.where(null_mask)[0]) + list(range(len(s), M.shape[1]))
        cand = [Vt[i].reshape(d, d, order="F") for i in idxs]
        g = max(cand, key=lambda G: np.abs((G + G.T) / 2).max())
    elif allow_approx:
        g = Vt[-1].reshape(d, d, order="F")           # smallest singular direction (approx metric)
    else:
        return None, 0, float("inf")
    g = (g + g.T) / 2
    g = g / (np.abs(g).max() + 1e-12)
    residual = max(np.abs(A.T @ g + g @ A).max() for A in gens)
    return g, nullity, float(residual)


def metric_signature(g, tol=0.15, rel_tol=0.25):
    """Eigenvalue signature (#positive, #negative, #zero) of a (normalised) metric g, up to an overall sign
    gauge -- so the count with FEWER entries is reported as the 'negative' part. A direction is counted as
    degenerate (zero) only if its |eigenvalue| is below BOTH an absolute floor `tol` and a fraction
    `rel_tol` of the MEDIAN |eigenvalue| (magnitude-robust: uneven but same-sign eigenvalues from a noisy
    data-discovered metric are not spuriously called degenerate). Returns the canonical signature (p,q)
    with p >= q, the number of genuinely degenerate directions, and the group name."""
    ev = np.linalg.eigvalsh((g + g.T) / 2)
    ev = ev / (np.abs(ev).max() + 1e-12)
    med = np.median(np.abs(ev)) + 1e-12
    zero_thr = min(tol, rel_tol * med)
    npos = int((ev > zero_thr).sum())
    nneg = int((ev < -zero_thr).sum())
    nzero = int((np.abs(ev) <= zero_thr).sum())
    p, q = max(npos, nneg), min(npos, nneg)          # overall sign is gauge
    d = len(ev)
    if q == 0 and nzero == 0:
        name = "O(%d)" % d                            # definite -> Euclidean orthogonal
    elif nzero > 0:
        name = "O(%d,%d)+deg%d" % (p, q, nzero)       # degenerate directions (scaling/conformal-like)
    elif q == 1:
        name = "O(1,%d)" % (d - 1)                    # Lorentz
    else:
        name = "O(%d,%d)" % (p, q)                    # general split signature
    return {"signature": (p, q), "n_zero": nzero, "eigenvalues": np.round(ev, 3).tolist(), "name": name}


def discover_metric_group(gens, tol=1e-6):
    """End-to-end: from generators, discover the invariant metric and name the pseudo-orthogonal group
    O(p,q) it defines. Returns (spec, detail) where spec = {"gens", "vec_dim", "metric", "name",
    "scale_norm"} ready for AllGraph(generated_equivariant_group=spec), or None if no metric is recovered.
    The metric is diagonalised to a canonical diag(+1.../-1...) form for the emlp contract."""
    if not gens:
        return None, {"reason": "no generators supplied"}
    d = np.asarray(gens[0]).shape[0]
    g, nullity, residual = invariant_metric(gens, tol=tol)
    if g is None:
        return None, {"reason": "no invariant metric found"}
    sig = metric_signature(g)
    # canonical diagonal metric with the discovered signature (up to sign gauge): p entries +1, q entries -1
    p, q = sig["signature"]
    nz = sig["n_zero"]
    diag = [1.0] * p + [-1.0] * q + [0.0] * nz
    # pad/trim to d (numerical safety)
    diag = (diag + [1.0] * d)[:d]
    metric = np.diag(diag)
    spec = {"gens": [np.asarray(A, float) for A in gens], "vec_dim": d, "metric": metric,
            "name": sig["name"], "scale_norm": False}
    detail = {"recovered_metric_diag": np.round(np.diag(g), 3).tolist(), "nullity": nullity,
              "residual": round(residual, 4), **sig}
    return spec, detail


def fit_metric_regression(vectors, y, standardize=True):
    """DIRECT metric recovery by regression: when a target is a metric norm of a (pooled) vector,
    y_i = s_i^T g s_i = <g, s_i s_i^T>, which is LINEAR in the symmetric matrix g. Solve least-squares for
    g over the data and return (g, r2). This bypasses generator discovery entirely -- no trained net, no
    Lie-derivative nullspace, no ill-conditioning -- and recovers the metric EXACTLY when the target is a
    quadratic form (validated: exact Minkowski from real JetNet at R^2 = 1.0000). The r2 is the honest
    gate: a target that is NOT a metric norm (a cubic, a linear function, noise) fits poorly, so the caller
    abstains rather than hallucinating a metric.
    """
    S = np.asarray(vectors, float)
    y = np.asarray(y, float).ravel()
    n, d = S.shape
    if standardize:
        sc = np.abs(S).std() + 1e-12
        S = S / sc
        y = y / (sc ** 2)
    cols, idx = [], []
    for a in range(d):
        for b in range(a, d):
            coef = 1.0 if a == b else 2.0            # off-diagonal entries appear twice in s^T g s
            cols.append(coef * S[:, a] * S[:, b])
            idx.append((a, b))
    cols.append(np.ones(n))                          # intercept: makes the fit invariant to an affine
    Phi = np.stack(cols, 1)                          # shift of the target (y -> a*y + b); the recovered g
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)      # is the quadratic part, unaffected by standardisation
    g = np.zeros((d, d))
    for (a, b), wi in zip(idx, w[:-1]):
        g[a, b] = wi
        g[b, a] = wi
    pred = Phi @ w
    r2 = float(1 - ((pred - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    return g, r2


def discover_metric_by_regression(data, min_fit=0.6, pool=True):
    """AUTONOMOUS metric discovery via direct regression (the robust data route). Builds a per-datum vector
    (the pooled sum of each cloud's vectors when pool=True, else the first vector), fits y ~ <s,s>_g for a
    symmetric g, and reads the O(p,q) signature -- but ONLY if the quadratic-form fit is good (r2 >=
    min_fit), otherwise abstains (the target is not a metric norm). Returns (spec, detail). Exact and cheap
    where generator discovery from pooled data is ill-conditioned: recovers the exact Minkowski metric
    (O(1,3)) from real JetNet, and arbitrary O(p,q) from data whose target is the corresponding norm."""
    if getattr(data, "positions", None) is None:
        return None, {"reason": "no coordinate vectors"}
    clouds = [np.asarray(P, float) for P in data.positions if np.asarray(P).ndim == 2 and len(P) >= 1]
    y = np.asarray(data.y).ravel() if getattr(data, "y", None) is not None else None
    if not clouds or y is None or np.std(y) < 1e-9:
        return None, {"reason": "no usable target for metric regression"}
    m = min(len(clouds), len(y))
    S = np.array([clouds[i].sum(0) if pool else clouds[i][0] for i in range(m)])
    g, r2 = fit_metric_regression(S, y[:m])
    gn = g / (np.abs(g).max() + 1e-12)
    detail = {"regression_r2": round(r2, 4), "recovered_metric_diag": np.round(np.diag(gn), 3).tolist()}
    if r2 < min_fit:
        return None, {**detail, "reason": "target is not a metric norm (quadratic-form fit too low)"}
    sig = metric_signature(gn)
    d = S.shape[1]
    # Build the deployed metric in DATA-AXIS ORDER by rounding the recovered diagonal to its sign, rather
    # than reordering to a canonical diag(+...,-...). The data's coordinates carry physical meaning (e.g.
    # energy at index 0 for 4-vectors), so a canonical reorder would break the correspondence between metric
    # axes and data axes and the generated equivariant contract would see the wrong invariants.
    dg = np.diag(gn)
    med = np.median(np.abs(dg)) + 1e-12
    thr = min(0.15, 0.25 * med)
    metric_diag = np.where(np.abs(dg) <= thr, 0.0, np.sign(dg))
    # gauge: fix overall sign so the majority sign is +1 (matches the p >= q convention)
    if (metric_diag > 0).sum() < (metric_diag < 0).sum():
        metric_diag = -metric_diag
    spec = {"gens": None, "vec_dim": d, "metric": np.diag(metric_diag), "name": sig["name"], "scale_norm": False}
    detail.update(sig)
    return spec, detail


def _train_vector_reference(vectors, y, seed=0, epochs=300, width=64):
    """Train a small MLP y ~ f(vector) on single group-vectors, so the Lie-derivative front-end can read
    the continuous symmetry (and hence the invariant metric) the target respects on the vector space."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(vectors), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    yt = (yt - yt.mean()) / (yt.std() + 1e-9)
    net = nn.Sequential(nn.Linear(X.shape[1], width), nn.Tanh(),
                        nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(X) - yt) ** 2).mean().backward()
        opt.step()
    return net


def discover_metric_from_data(data, min_samples=200, epochs=300, tol_ratio=1.5, max_vec=2000):
    """AUTONOMOUS metric discovery from a dataset: build a per-vector target by pooling each cloud's
    invariant, train a reference net on single vectors, discover the symmetry generators (Lie-derivative
    nullspace), solve for the invariant metric, and name the O(p,q) group. Returns (spec, detail) or None.

    This generalises the fixed {Euclidean, Lorentz} menu to ANY pseudo-orthogonal signature the data
    respects (e.g. O(2,2)), by DISCOVERING the metric rather than choosing from a menu. Intended as the
    open-ended fallback when the menu-based detector is unsure or when a non-standard signature is present.
    """
    import torch

    from .symmetry_discovery import discover_symmetries
    if getattr(data, "positions", None) is None:
        return None, {"reason": "no coordinate vectors"}
    clouds = [np.asarray(P, float) for P in data.positions if np.asarray(P).ndim == 2 and len(P) >= 1]
    if not clouds:
        return None, {"reason": "no usable clouds"}
    y = np.asarray(data.y).ravel() if getattr(data, "y", None) is not None else None
    # per-vector samples with a per-vector target that VARIES within a cloud, so the reference net learns a
    # function whose symmetry mirrors the group's on the vector space. We use each vector's projection onto
    # its cloud's pooled direction (a G-covariant scalar that varies vector-to-vector), scaled by the
    # datum-level target when available. Falls back to the vector's own contribution.
    vecs, tgt = [], []
    for i, P in enumerate(clouds):
        s = P.sum(0)
        proj = P @ s                                  # per-vector projection onto the pooled direction
        base = float(y[i]) if (y is not None and i < len(y)) else 1.0
        for k, v in enumerate(P):
            vecs.append(v)
            tgt.append(base * proj[k])
    vecs = np.array(vecs)
    tgt = np.array(tgt)
    if len(vecs) < min_samples:
        return None, {"reason": "too few vectors for metric discovery"}
    if len(vecs) > max_vec:
        idx = np.random.RandomState(0).choice(len(vecs), max_vec, replace=False)
        vecs, tgt = vecs[idx], tgt[idx]
    net = _train_vector_reference(vecs, tgt, epochs=epochs)
    res = discover_symmetries(net, torch.tensor(vecs, dtype=torch.float32), tol_ratio=tol_ratio)
    gens = res.get("generators") or []
    if not gens:
        return None, {"reason": "no symmetry generators discovered", "n_symmetries": res.get("n_symmetries", 0)}
    spec, detail = discover_metric_group(gens)
    if spec is not None:
        detail["n_generators"] = len(gens)
        detail["gap_ratio"] = round(float(res.get("gap_ratio", 0.0)), 2)
    return spec, detail


def generators_for_metric(metric):
    """Return a basis of the Lie algebra so(g) preserving the (diagonal) metric g: the matrices A with
    A^T g + g A = 0. For a nondegenerate g these are A = g^{-1} S with S antisymmetric, giving dim
    d(d-1)/2 generators. This is the inverse of invariant_metric: it lets a discovered signature (p,q) --
    e.g. from the metric-regression route, which yields a metric but no explicit generators -- be turned
    into the generator spec the EMLP equivariant contract needs."""
    g = np.asarray(metric, float)
    d = g.shape[0]
    # drop any zero (degenerate) directions to keep g invertible; act on the nondegenerate block
    diag = np.diag(g)
    nondeg = np.abs(diag) > 1e-9
    if not nondeg.all():
        # restrict to the nondegenerate subspace, build generators there, embed back
        idx = np.where(nondeg)[0]
        gb = g[np.ix_(idx, idx)]
        subgens = generators_for_metric(gb)
        gens = []
        for A in subgens:
            B = np.zeros((d, d))
            B[np.ix_(idx, idx)] = A
            gens.append(B)
        return gens
    ginv = np.linalg.inv(g)
    gens = []
    for i in range(d):
        for j in range(i + 1, d):
            S = np.zeros((d, d)); S[i, j] = 1.0; S[j, i] = -1.0
            gens.append(ginv @ S)
    return gens

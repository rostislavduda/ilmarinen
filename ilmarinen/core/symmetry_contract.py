"""symmetry_contract.py -- contract discovery from discovered symmetry (the physics-motivated router).

The eight schema contracts are not arbitrary: each is the natural equivariant architecture for a
symmetry group G. A sequence conv is equivariant to 1-D translation; a spatial conv to Z^2 translation; a
Deep-Set to the symmetric group S_n; a GNN to graph automorphisms; an E(3)-GNN to the Euclidean group
SO(3) |x R^3 on top of S_n. So the contract is a FUNCTION of the data's symmetry group: contract = arch(G).

The container-type router picks the contract by whether edges/positions are present -- a crude PROXY for G.
Two datasets with the same container can have different symmetries: a rotation-invariant point cloud wants
the equivariant contract, a rotation-BREAKING one does not, yet both are "positions, no edges". This module
reads G directly from the data and maps it to the contract -- the Geometric-Deep-Learning blueprint
(symmetry -> architecture) as an automatic DISCOVERY pipeline rather than a hand-choice.

THE ROTATION TEST (validated in tests/symmetry_contract.md). The physically meaningful question for the
equivariant-vs-set choice is: is the target invariant under rotating the coordinate frame? We answer it
with a fit-quality-controlled comparison that avoids the pitfalls of an absolute equivariance error:

    fit the target from ROTATION-INVARIANT features (a pairwise-distance histogram, manifestly SO(3)-
    invariant) and from ORIENTATION-SENSITIVE features (per-axis coordinate histograms, which use the
    absolute frame). If the invariant features fit as well as (or better than) the orientation-sensitive
    ones, the target needs only invariant information -> it is rotation-invariant -> EQUIVARIANT contract.
    If orientation-sensitive features do materially better, the target depends on absolute orientation ->
    it is NOT rotation-invariant -> the geometry is a feature, not a symmetry to respect -> SET/GRAPH.

This is exactly the invariance the equivariant schema is built to exploit, tested directly on the real
target, and it is robust: on real QM7 atomization energy the invariant features win decisively (R^2 0.70
vs -0.16 for orientation features) -> equivariant; on an orientation-dependent target (z-extent) the
orientation features win (0.70 vs 0.11) -> set. It cleanly SPLITS two datasets with the same container.

The result composes with the existing routers: it SUBSUMES the container-type proxy (container is used only
to split set-vs-graph by adjacency once rotation-invariance is settled), and it is a cleaner, more physical
label source for the learned contract router than a from-scratch bake-off.
"""

from __future__ import annotations

import numpy as np


def _pairwise_distance_hist(P, nbins=8, rmax=6.0):
    """Rotation- and translation-invariant descriptor of a point cloud: histogram of pairwise distances."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 2:
        return np.zeros(nbins)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    du = D[np.triu_indices(n, 1)]
    h, _ = np.histogram(du, bins=nbins, range=(0.0, rmax))
    return h / (len(du) + 1e-9)


def _axis_coordinate_hist(P, nbins=8, rng=4.0):
    """Orientation-SENSITIVE descriptor: per-axis coordinate histograms (use the absolute frame)."""
    P = np.asarray(P, float)
    P = P - P.mean(0)
    feats = []
    for ax in range(P.shape[1]):
        h, _ = np.histogram(P[:, ax], bins=nbins, range=(-rng, rng))
        feats.append(h / (len(P) + 1e-9))
    return np.concatenate(feats)


def _fit_r2(feature_fn, clouds, y, seed=0, epochs=200, width=64):
    """Fit a small MLP y ~ f(feature_fn(cloud)) and return held-out R^2."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    F = np.array([feature_fn(P) for P in clouds], float)
    X = torch.tensor(F, dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    yt = (yt - yt.mean()) / (yt.std() + 1e-9)
    n = len(X)
    ntr = int(0.75 * n)
    net = nn.Sequential(
        nn.Linear(X.shape[1], width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(X[:ntr]) - yt[:ntr]) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        pred = net(X[ntr:])
        resid = ((pred - yt[ntr:]) ** 2).sum()
        total = ((yt[ntr:] - yt[ntr:].mean()) ** 2).sum() + 1e-9
        return float(1.0 - resid / total)


def rotation_invariance_score(clouds, y, seed=0, epochs=200, tol=0.05, min_invariant_r2=0.05):
    """Fit-quality-controlled test of whether the target y is rotation-invariant. Returns a dict with the
    invariant-feature R^2, the orientation-feature R^2, their gap, and a boolean `rotation_invariant`
    (True iff invariant features fit at least as well, within tol, AND the invariant fit is genuinely
    predictive -- r2_inv above min_invariant_r2). The absolute floor matters: when BOTH fits are near-zero
    or negative (geometry does not predict the target at this budget), the relative comparison r2_inv >=
    r2_ori is a meaningless comparison of two failures, and declaring "rotation-invariant" would wrongly
    trigger canonicalization -- discarding the equivariant contract for the set contract on noise. A True
    verdict means the target genuinely needs rotation-invariant information -> the equivariant contract (or
    canonicalization) is appropriate."""
    r2_inv = _fit_r2(_pairwise_distance_hist, clouds, y, seed=seed, epochs=epochs)
    r2_ori = _fit_r2(_axis_coordinate_hist, clouds, y, seed=seed, epochs=epochs)
    gap = r2_ori - r2_inv  # how much orientation helps beyond invariants
    invariant = bool(r2_inv >= r2_ori - tol and r2_inv >= min_invariant_r2)
    return {
        "r2_invariant": r2_inv,
        "r2_orientation": r2_ori,
        "orientation_gain": gap,
        "rotation_invariant": invariant,
        "uninformative": bool(r2_inv < min_invariant_r2 and r2_ori < min_invariant_r2),
    }


def contract_from_symmetry(data, epochs=200, tol=0.05, min_clouds=30):
    """Discover the contract from the data's rotational symmetry. Returns (contract, confidence, detail).

    Rule: if the target is rotation-invariant (invariant features fit as well as orientation features) the
    coordinates carry a symmetry the model should respect -> EQUIVARIANT. Otherwise the coordinates are
    orientation-dependent features, not a symmetry -> route by container (adjacency -> GRAPH, else SET). If
    there are no positions, the rotation test does not apply and we defer to the container split.

    This is the symmetry-first analogue of the container router: the invariance the target actually
    respects drives the equivariant-vs-set choice, not the presence of coordinates alone.
    """
    has_pos = getattr(data, "positions", None) is not None
    has_edges = getattr(data, "edges", None) is not None
    if not has_pos:
        return ("graph" if has_edges else "set"), 0.0, {"reason": "no positions; container split"}
    # assemble centered clouds + a datum-level target
    clouds = []
    yy = []
    y = np.asarray(data.y).ravel() if getattr(data, "y", None) is not None else None
    for i, P in enumerate(data.positions):
        P = np.asarray(P, float)
        if P.ndim != 2 or len(P) < 3:
            continue
        clouds.append(P - P.mean(0))
        yy.append(float(y[i]) if (y is not None and i < len(y)) else np.linalg.norm(P.std(0)))
    if len(clouds) < min_clouds:
        return ("graph" if has_edges else "set"), 0.0, {"reason": "too few clouds for the rotation test"}
    score = rotation_invariance_score(clouds, np.array(yy), epochs=epochs, tol=tol)
    if score["rotation_invariant"]:
        conf = float(np.clip(0.5 - score["orientation_gain"], 0.0, 1.0))  # more negative gain -> more confident
        return "equivariant", conf, {"reason": "target is rotation-invariant -> equivariant", **score}
    if score.get("uninformative"):
        # neither invariant nor orientation features predict the target at this budget -- the rotation test
        # is uninformative. Do NOT canonicalize on noise (that would discard the equivariant contract for
        # the set contract). Defer with zero confidence so the learned/structural router decides; with
        # positions+edges present this correctly leaves the equivariant contract in play.
        return (
            ("equivariant" if has_edges else "set"),
            0.0,
            {"reason": "rotation test uninformative (geometry non-predictive at budget); defer", **score},
        )
    contract = "graph" if has_edges else "set"
    conf = float(np.clip(score["orientation_gain"], 0.0, 1.0))  # larger gain -> more confident it is NOT equivariant
    return contract, conf, {"reason": "target depends on orientation -> container split", **score}


# --------------------------------------------------------------------------- autonomous group detection
def _dilation_generator(d):
    """The isotropic-scaling (Euler/dilation) generator: the identity matrix. exp(t D) = e^t * I."""
    import numpy as _np

    return _np.eye(d)


def _candidate_groups(d):
    """Menu of candidate symmetry groups on a d-dimensional coordinate vector. Each entry is a dict with
    name, generators (Lie-algebra basis on the coordinate space), metric (inner product the group
    preserves), n_gen, and scale_norm (whether the group's invariants require scale-normalization, i.e. the
    group includes isotropic dilation). Physically motivated: Euclidean SO(d) rotations always apply; the
    Lorentz group O(1,d-1) for pseudo-Euclidean data; and the SIMILARITY group Sim(d) = R^+ x SO(d)
    (rotations + isotropic scaling) whose invariants are the SCALE-NORMALIZED inner products (shape without
    size). Orthogonal/Lorentz groups preserve an inner product (invariants = inner products); scaling
    rescales it (invariants = ratios/angles), which is why Sim(d) uses scale-normalized features."""
    import numpy as _np

    cands = []
    # SO(d): antisymmetric generators, identity metric, no scale-normalization
    so = []
    for i in range(d):
        for j in range(i + 1, d):
            A = _np.zeros((d, d))
            A[i, j] = -1
            A[j, i] = 1
            so.append(A)
    if so:
        cands.append({"name": "SO(%d)" % d, "gens": so, "metric": _np.eye(d), "n_gen": len(so), "scale_norm": False})
        # Similarity group Sim(d) = R^+ x SO(d): rotations + isotropic dilation. Same generators PLUS the
        # dilation generator; invariants are scale-normalized (shape, not size).
        cands.append(
            {
                "name": "Sim(%d)" % d,
                "gens": so + [_dilation_generator(d)],
                "metric": _np.eye(d),
                "n_gen": len(so) + 1,
                "scale_norm": True,
            }
        )
    # Lorentz O(1,d-1): spatial rotations + boosts, Minkowski metric
    if d >= 2:
        lor = []
        for i in range(1, d):
            for j in range(i + 1, d):
                A = _np.zeros((d, d))
                A[i, j] = -1
                A[j, i] = 1
                lor.append(A)  # spatial rotations
        for i in range(1, d):
            A = _np.zeros((d, d))
            A[0, i] = 1
            A[i, 0] = 1
            lor.append(A)  # boosts
        metric = _np.diag([1.0] + [-1.0] * (d - 1))
        cands.append(
            {"name": "O(1,%d)" % (d - 1), "gens": lor, "metric": metric, "n_gen": len(lor), "scale_norm": False}
        )
    return cands


def _invariant_features(clouds, metric, scale_norm=False):
    """G-invariant per-cloud features from (metric) inner products of the pooled and individual vectors --
    invariant to any group preserving that metric inner product. This is the SAME invariant construction
    emlp_layer uses to BUILD the contract, so detection and generation share one mechanism. When
    scale_norm is True (similarity/conformal groups, which include isotropic dilation), each cloud is
    normalized by its RMS radius first, so the inner products become SCALE-invariant (shape, not size)."""
    import numpy as _np

    feats = []
    for P in clouds:
        P = _np.asarray(P, float)
        if scale_norm:
            rms = _np.sqrt((P**2).sum(1).mean()) + 1e-9
            P = P / rms
        s = P.sum(0)
        spp = float(s @ metric @ s)
        selfnorm = float(_np.mean([p @ metric @ p for p in P]))
        # a second-moment invariant: mean pairwise inner product
        G = P @ metric @ P.T
        offmean = float((G.sum() - _np.trace(G)) / max(len(P) * (len(P) - 1), 1))
        # shape descriptor: sorted eigenvalues of the centered second-moment (rotation-invariant), which
        # (after scale_norm) captures anisotropy/shape ratios that pooled inner products alone miss.
        Pc = P - P.mean(0)
        cov = (Pc.T @ Pc) / max(len(P), 1)  # (d,d) second-moment
        eig = _np.sort(_np.linalg.eigvalsh((cov + cov.T) / 2))[::-1]
        feats.append(_np.concatenate([[spp, selfnorm, offmean], eig]))
    return _np.array(feats)


def _highres_invariants(clouds, metric, scale_norm=False, k_eig=4, n_quant=7):
    """HIGHER-RESOLUTION G-invariant, permutation-invariant descriptor. The pooled descriptor sums all
    vectors and loses subset/distributional structure; this one retains it while staying invariant:
      * the pooled invariant <sum p, sum p>_g (cheap global signal, kept);
      * the top-k EIGENVALUES of the Gram matrix G_ij = <p_i,p_j>_g (permutation- and G-invariant, the
        shape/rank structure), padded to a fixed length;
      * QUANTILES of the off-diagonal G_ij (the pairwise-invariant DISTRIBUTION, which the mean discards --
        this is what captures subset/extreme-value targets like a two-hardest-particle mass);
      * QUANTILES of the diagonal <p_i,p_i>_g (self-norm distribution).
    Every G_ij = p_i^T g p_j is EXACTLY invariant under any group preserving g (L^T g L = g), so all these
    features are exact group invariants; being spectra / order-statistics they are permutation-invariant
    too. scale_norm normalizes each cloud by its RMS radius first (for similarity/conformal groups)."""
    import numpy as _np

    g = _np.asarray(metric, float)
    qs = _np.linspace(0.0, 1.0, n_quant)
    feats = []
    for P in clouds:
        P = _np.asarray(P, float)
        if scale_norm:
            P = P / (_np.sqrt((P**2).sum(1).mean()) + 1e-9)
        s = P.sum(0)
        G = P @ g @ P.T
        ev = _np.sort(_np.linalg.eigvalsh((G + G.T) / 2))[::-1]
        ev = _np.concatenate([ev[:k_eig], _np.zeros(max(0, k_eig - len(ev)))])
        iu = _np.triu_indices(len(P), 1)
        off = G[iu] if len(iu[0]) > 0 else _np.zeros(1)
        feats.append(_np.concatenate([[float(s @ g @ s)], ev, _np.quantile(off, qs), _np.quantile(_np.diag(G), qs)]))
    return _np.array(feats)


def _apply_group(clouds, gens, scale=0.6, seed=1):
    """Apply a random group element g = exp(sum_k t_k A_k) to every cloud (each row is a group vector).
    Used by the stability selector: a truly G-invariant descriptor is unchanged in DISTRIBUTION under this,
    so a target explained by G-invariants must fit equally well on the transformed clouds."""
    import numpy as _np
    from scipy.linalg import expm

    rng = _np.random.RandomState(seed)
    t = rng.randn(len(gens)) * scale
    L = expm(sum(t[k] * _np.asarray(gens[k], float) for k in range(len(gens))))
    return [(L @ _np.asarray(P, float).T).T for P in clouds]


def _raw_features(clouds, d):
    """Frame-sensitive (non-invariant) features: per-axis moments of the pooled vector and coordinates."""
    import numpy as _np

    feats = []
    for P in clouds:
        P = _np.asarray(P, float)
        s = P.sum(0)
        feats.append(_np.concatenate([s, P.mean(0), P.std(0)]))
    return _np.array(feats)


def _fit_r2_features(feats, y, seed=0, epochs=200, width=48):
    import numpy as _np
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X = torch.tensor(_np.asarray(feats), dtype=torch.float32)
    yt = torch.tensor(_np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    yt = (yt - yt.mean()) / (yt.std() + 1e-9)
    n = len(X)
    ntr = int(0.75 * n)
    net = nn.Sequential(
        nn.Linear(X.shape[1], width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(X[:ntr]) - yt[:ntr]) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        p = net(X[ntr:])
        return float(1 - ((p - yt[ntr:]) ** 2).sum() / (((yt[ntr:] - yt[ntr:].mean()) ** 2).sum() + 1e-9))


def _fit_r2_on(Ftr, ytr, Fte, yte, seed=0, epochs=200, width=64):
    """Fit an MLP on (Ftr, ytr) and return R^2 on a SEPARATE (Fte, yte). Standardization is learned on
    train and applied to test, so this is a genuine transfer score -- used by the stability selector, where
    Fte may be the descriptor of GROUP-TRANSFORMED clouds while yte (an invariant target) is unchanged."""
    import numpy as _np
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X = torch.tensor(_np.asarray(Ftr), dtype=torch.float32)
    yt = torch.tensor(_np.asarray(ytr), dtype=torch.float32).reshape(-1, 1)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ym, ysd = yt.mean(), yt.std() + 1e-9
    Xn = (X - mu) / sd
    ytn = (yt - ym) / ysd
    net = nn.Sequential(
        nn.Linear(X.shape[1], width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(Xn) - ytn) ** 2).mean().backward()
        opt.step()
    Xt = (torch.tensor(_np.asarray(Fte), dtype=torch.float32) - mu) / sd
    ytt = (torch.tensor(_np.asarray(yte), dtype=torch.float32).reshape(-1, 1) - ym) / ysd
    with torch.no_grad():
        p = net(Xt)
        return float(1 - ((p - ytt) ** 2).sum() / (((ytt - ytt.mean()) ** 2).sum() + 1e-9))


def detect_symmetry_group(data, tol=0.05, min_clouds=30, epochs=200, min_fit=0.3, hi_res=True):
    """AUTONOMOUS group detection: identify which candidate matrix group the target respects, and emit its
    generator spec. For each candidate group G (menu by coordinate dim) with metric g:

      * build a G-invariant, permutation-invariant descriptor of each cloud (the HIGH-RESOLUTION
        Gram-spectrum + pairwise-invariant-quantile descriptor when hi_res, else the pooled one);
      * fit the target from it on a train split and score on a held-out test split;
      * STABILITY TEST: also score on the test clouds after applying a random element of G. Because every
        <p_i,p_j>_g is exactly invariant under G, a target genuinely explained by G-invariants fits EQUALLY
        well on the transformed clouds; a WRONG metric (whose descriptor is not actually G-invariant, e.g.
        the Euclidean descriptor under a Lorentz boost) COLLAPSES. The group's `stable` score is the worse
        of (plain, transformed) -- this is the honest selector that in-sample R^2 alone cannot provide,
        since a more expressive descriptor can otherwise overfit frame-dependent signal.

    Selection: the group with the best STABLE fit, subject to a min_fit floor and a raw-coordinate margin
    (to reject non-invariant targets), with an Occam prior preferring the largest group within tol and,
    among those, the canonical (Euclidean, non-scale) one unless a larger/pseudo-Euclidean group is clearly
    better. Returns (spec, detail); spec is None (no symmetry) or {"gens","vec_dim","metric","name",
    "scale_norm"} ready for AllGraph(generated_equivariant_group=spec).
    """
    import numpy as _np

    if getattr(data, "positions", None) is None:
        return None, {"reason": "no coordinate vectors; group detection N/A"}
    clouds = [_np.asarray(P, float) for P in data.positions if _np.asarray(P).ndim == 2 and len(P) >= 2]
    if len(clouds) < min_clouds:
        return None, {"reason": "too few clouds for group detection"}
    y = _np.asarray(data.y).ravel()[: len(clouds)] if getattr(data, "y", None) is not None else None
    if y is None or _np.std(y) < 1e-9:
        return None, {"reason": "no usable target for the invariance test"}
    d = clouds[0].shape[1]
    n = len(clouds)
    ntr = int(0.7 * n)
    desc = _highres_invariants if hi_res else _invariant_features

    def stable_score(metric, gens, scale_norm):
        Ftr = desc(clouds[:ntr], metric, scale_norm)
        Fte = desc(clouds[ntr:], metric, scale_norm)
        Fte_t = desc(_apply_group(clouds[ntr:], gens), metric, scale_norm)
        plain = _fit_r2_on(Ftr, y[:ntr], Fte, y[ntr:], epochs=epochs)
        trans = _fit_r2_on(Ftr, y[:ntr], Fte_t, y[ntr:], epochs=epochs)
        return plain, trans

    raw_r2 = _fit_r2_features(_raw_features(clouds, d), y, epochs=epochs)
    results = []
    for c in _candidate_groups(d):
        plain, trans = stable_score(c["metric"], c["gens"], c.get("scale_norm", False))
        results.append(
            {
                "name": c["name"],
                "plain": plain,
                "trans": trans,
                "stable": min(plain, trans),
                "gens": c["gens"],
                "metric": c["metric"],
                "n_gen": c["n_gen"],
                "scale_norm": c.get("scale_norm", False),
            }
        )
    detail = {
        "raw_r2": round(raw_r2, 3),
        "candidates": [
            {
                "name": r["name"],
                "plain": round(r["plain"], 3),
                "trans": round(r["trans"], 3),
                "stable": round(r["stable"], 3),
                "n_gen": r["n_gen"],
            }
            for r in results
        ],
    }
    # qualify on the STABLE fit: it must be good (>= min_fit) and not be beaten by non-invariant raw
    # features by more than raw_margin (rejecting non-invariant targets).
    raw_margin = 0.10
    qualified = [r for r in results if r["stable"] >= max(min_fit, 0.3) and r["stable"] >= raw_r2 - raw_margin]
    if not qualified:
        return None, {**detail, "reason": "no candidate group's invariants stably explain the target"}
    best_stable = max(r["stable"] for r in qualified)
    # among candidates in a TRUE tie for the best stable fit (within a small epsilon), prefer the CANONICAL
    # group: Euclidean (identity metric) and non-scale. A larger/pseudo-Euclidean/scaling group is chosen
    # only when it fits STRICTLY BETTER -- it must earn the extra structure by explaining the target better,
    # not merely by "not hurting". This keeps SO(3) for a Euclidean target that SO(3) already fits best, and
    # keeps O(1,3) for a Lorentz target that O(1,3) fits decisively better.
    eps = 0.02
    tied = [r for r in qualified if r["stable"] >= best_stable - eps]

    def canonical_rank(r):
        is_pseudo = float(not _np.allclose(r["metric"], _np.eye(d)))
        return float(r["scale_norm"]) + is_pseudo  # 0 = canonical Euclidean, higher = less canonical

    best = min(tied, key=lambda r: (canonical_rank(r), -r["stable"], r["n_gen"]))
    spec = {
        "gens": best["gens"],
        "vec_dim": d,
        "metric": best["metric"],
        "name": best["name"],
        "scale_norm": best["scale_norm"],
    }
    detail["selected"] = best["name"]
    return spec, detail

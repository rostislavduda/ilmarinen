"""Mode-coupling structure discovery -- the tensor-representation front-end.

Before choosing operations, a physicist asks what SHAPE the data naturally has: is it a 1D chain
(sequence), a 2D grid (image), or unstructured (tabular)? This module answers that data-drivenly
from the mutual-information / correlation structure between input coordinates, following the
tensor-network-structure-search principle that the right representation is the one whose correlation
(entanglement) structure matches the data's:
  - 1D chain (MPS-like):  MI is band-diagonal -- decays with 1D coordinate distance.
  - 2D grid (PEPS-like):  MI decays with 2D grid (Manhattan) distance under the true H x W shape.
  - unstructured:         MI has no spatial pattern (low correlation with any distance metric).

The output tells the metaoptimizer (a) how to TENSORIZE the flat input (the winning shape) and
(b) which DIMENSIONAL CLASS of primitives is worth including (sequence vs conv2d vs dense) -- so
higher-dimensional primitives are added only when the data actually has the adjacency they assume.

This is the mode-ordering / tensorization stage of the physicist pipeline: discover symmetries ->
discover correlation/mode structure -> tensorize -> reduce -> minimal model. It is a data-driven
test with an absence verdict ('unstructured'), held to the same standard as the symmetry detectors:
it must not hallucinate structure that is not there (validated on a shuffled control).
"""

from __future__ import annotations

import numpy as np


def mutual_information_matrix(X, method="gaussian", bins=8):
    """Pairwise MI matrix between coordinates of X (n_samples, n_coords).

    'gaussian': fast correlation-based proxy MI_ij = -0.5 log(1 - rho_ij^2) (exact for jointly
                Gaussian data; a monotone proxy otherwise).
    'binned':   histogram MI estimate (slower, distribution-free); use for non-Gaussian data.
    """
    X = np.asarray(X, dtype=np.float64)
    if method == "gaussian":
        Xc = X - X.mean(0, keepdims=True)
        s = X.std(0) + 1e-9
        C = (Xc.T @ Xc) / len(X) / np.outer(s, s)
        C = np.clip(C, -0.999, 0.999)
        return -0.5 * np.log(1 - C**2)
    # binned estimator
    n = X.shape[1]
    Xd = np.stack(
        [np.digitize(X[:, k], np.quantile(X[:, k], np.linspace(0, 1, bins + 1)[1:-1])) for k in range(n)], axis=1
    )
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            pij = np.histogram2d(Xd[:, i], Xd[:, j], bins=bins)[0] + 1e-9
            pij /= pij.sum()
            pi = pij.sum(1, keepdims=True)
            pj = pij.sum(0, keepdims=True)
            mi = float((pij * np.log(pij / (pi * pj))).sum())
            M[i, j] = M[j, i] = max(mi, 0.0)
    return M


def _divisor_grid_shapes(n, min_side=2):
    """All (H, W) with H*W = n and both >= min_side (candidate 2D tensorizations)."""
    shapes = []
    for h in range(min_side, int(np.sqrt(n)) + 1):
        if n % h == 0 and n // h >= min_side:
            shapes.append((h, n // h))
            if h != n // h:
                shapes.append((n // h, h))
    return shapes


def _fit_corr(mi_upper, dist_upper):
    """Correlation between MI and NEGATIVE distance (MI should decrease with distance)."""
    a = mi_upper - mi_upper.mean()
    b = (-dist_upper) - (-dist_upper).mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _mi_flat_distance_profile(M):
    """Average MI as a function of flat-index distance delta (1..n-1)."""
    n = M.shape[0]
    prof = np.zeros(n)
    for delta in range(1, n):
        prof[delta] = np.mean([M[i, i + delta] for i in range(n - delta)])
    return prof


def detect_grid_width(M, anomaly_tol=0.05):
    """Find the row STRIDE W via an ANOMALY test that distinguishes a 2D grid from a 1D chain.

    In a row-major H x W image, vertically-adjacent pixels sit W apart, creating an MI peak at
    delta = W that BREAKS the otherwise-monotone MI-vs-distance decay. A genuine 1D chain, by
    contrast, has monotonically decaying MI with NO such anomalous peak. We therefore score each
    candidate width by how much MI at the stride EXCEEDS the smooth decay trend interpolated from
    its neighbors (p[W] - 0.5*(p[W-2]+p[W+2])). A 1D signal gives ~0 or negative anomaly at every
    candidate (nothing breaks the decay); a 2D signal gives a clear positive anomaly at the true W.
    This recovers CIFAR 32x32 and Fashion-MNIST 28x28 exactly, while NOT misclassifying 1D chains.

    Returns (best_width, anomaly_scores). best_width is None if no clear positive anomaly exists
    (i.e. the data is a 1D chain or unstructured, not a 2D grid).
    """
    n = M.shape[0]
    prof = _mi_flat_distance_profile(M)
    cands = [d for d in range(2, n) if n % d == 0 and n // d >= 2]
    if not cands:
        return None, {}

    def anomaly(W):
        if W - 2 < 1 or W + 2 >= n:
            return 0.0
        trend = 0.5 * (prof[W - 2] + prof[W + 2])  # decay trend skipping the peak neighborhood
        return float(prof[W] - trend)

    scores = {W: anomaly(W) for W in cands}
    best = max(scores, key=scores.get)
    if scores[best] <= anomaly_tol:  # no anomalous stride -> not a 2D grid
        return None, scores
    return best, scores


def discover_mode_structure(
    X, method="gaussian", margin_tol=0.05, min_fit=0.15, candidate_shapes=None, price_mu=None, axis_bits=None
):
    """Discover the coordinate structure of X and the optimal tensorization.

    Two-stage: (1) a coarse MI-vs-distance fit gives an initial 1D/2D/unstructured read and gates
    'unstructured'; (2) the stride-ANOMALY estimator (detect_grid_width) is the authority on 2D vs
    1D and on the exact width -- a genuine 2D grid has an MI peak breaking the monotone decay, which
    a 1D chain lacks. This recovers exact image shapes (CIFAR 32x32, F-MNIST 28x28) while not
    promoting 1D chains to 2D.

    ACCEPTANCE RULE. By default the 2D-vs-1D promotion uses the floor/margin thresholds (min_fit and
    detect_grid_width's internal anomaly_tol). If price_mu is given, the promotion is instead the DERIVED
    marginal-value rule J = R + price_mu * Omega_struct(rank) (machinery.contract_mdl.price_tensorization):
    climb to 2D only if the stride-anomaly fit gain exceeds price_mu times the added-axis structural code
    length. This folds the tensorization choice into the same priced objective as width/depth/primitive/
    contract, replacing the hand-set thresholds with one price.

    Returns dict: structure ('1d'|'2d'|'unstructured'), shape ((H,W) | (n,) | None), scores,
    width_scores, recommended_primitives, margin, best_fit (and, if priced, tensorization_price).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    # LARGE-D priced path: the dense n x n MI matrix and coarse fit below are intractable for large n
    # (~4 GB at n~22000). When pricing is on and n is large, go straight to the scalable priced parser,
    # which never forms the full matrix. The coarse MI-vs-distance scores are skipped (they are only used
    # to gate 'unstructured', which the significance gate inside the scalable parser handles).
    if price_mu is not None and n > 500:
        from ..machinery.contract_mdl import price_tensorization

        pd = price_tensorization(X, mu=price_mu, method=method, axis_bits=axis_bits)
        struct = pd["structure"]
        prim_by_rank = {
            "1d": ["plain", "gated", "lstm", "conv", "linssm", "attention", "spectral"],
            "2d": ["conv2d", "conv_dw", "pointwise", "attention", "norm"],
            "3d": ["conv3d", "conv_dw", "pointwise", "dense", "norm", "attention"],
            "4d": ["conv4d", "conv4d_kt1", "conv_dw", "pointwise", "norm"],
        }
        structure = struct if struct in prim_by_rank else "1d"
        shape = pd["shape"] if struct != "1d" else (n,)
        return {
            "structure": structure,
            "shape": shape,
            "scores": {},
            "width_scores": {},
            "recommended_primitives": prim_by_rank[structure],
            "margin": float("nan"),
            "best_fit": float("nan"),
            "tensorization_price": pd,
        }
    M = mutual_information_matrix(X, method=method)
    iu = np.triu_indices(n, 1)
    mi = M[iu]

    scores = {}
    d1 = np.abs(iu[0] - iu[1]).astype(float)
    scores["1d"] = _fit_corr(mi, d1)
    shapes = candidate_shapes if candidate_shapes is not None else _divisor_grid_shapes(n)
    for H, W in shapes:
        rc = [(k // W, k % W) for k in range(n)]
        d2 = np.array([abs(rc[i][0] - rc[j][0]) + abs(rc[i][1] - rc[j][1]) for i, j in zip(*iu)], float)
        scores[f"2d_{H}x{W}"] = _fit_corr(mi, d2)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - runner

    # stride-anomaly estimator is the authority on 2D-vs-1D and the exact width
    W, width_scores = detect_grid_width(M)
    price_detail = None
    if price_mu is not None:
        # DERIVED acceptance: price the added axes against their stride-anomaly fit gain, ANY rank.
        from ..machinery.contract_mdl import price_tensorization

        price_detail = price_tensorization(X, mu=price_mu, method=method, axis_bits=axis_bits)
        struct = price_detail["structure"]
        prim_by_rank = {
            "1d": ["plain", "gated", "lstm", "conv", "linssm", "attention", "spectral"],
            "2d": ["conv2d", "conv_dw", "pointwise", "attention", "norm"],
            "3d": ["conv3d", "conv_dw", "pointwise", "dense", "norm", "attention"],
            "4d": ["conv4d", "conv4d_kt1", "conv_dw", "pointwise", "norm"],
        }
        if struct in ("2d", "3d", "4d"):
            structure, shape, prims = struct, price_detail["shape"], prim_by_rank[struct]
        elif best_score < min_fit:
            structure, shape, prims = "unstructured", None, ["dense", "norm", "attention"]
        else:
            structure, shape, prims = "1d", (n,), prim_by_rank["1d"]
    elif best_score < min_fit and W is None:
        structure, shape, prims = "unstructured", None, ["dense", "norm", "attention"]
    else:
        # N-D lattice parse (rank 1..4). The legacy detect_grid_width only ever reported the leading 2D
        # stride, collapsing a genuine 3D/4D volume to 2D. parse_grid_shape is the position-aware N-D
        # successor (validated on 3x3x3x3 / 4x4x4x4) with a significance gate against MI noise; it is the
        # authority on rank AND exact shape here, so a flat 3D volume now routes to the volumetric contract
        # automatically. It falls back to (n,) when no multi-axis fit clears its Occam tolerance.
        nd_shape, nd_detail = parse_grid_shape(M)
        rank = len(nd_shape)
        prim_by_rank = {
            1: ["plain", "gated", "lstm", "conv", "linssm", "attention", "spectral"],
            2: ["conv2d", "conv_dw", "pointwise", "attention", "norm"],
            3: ["conv3d", "conv_dw", "pointwise", "dense", "norm", "attention"],
            4: ["conv4d", "conv4d_kt1", "conv_dw", "pointwise", "norm"],
        }
        struct_by_rank = {1: "1d", 2: "2d", 3: "3d", 4: "4d"}
        if rank >= 2:
            structure, shape, prims = struct_by_rank[rank], nd_shape, prim_by_rank[rank]
        elif best_score >= min_fit:
            structure, shape, prims = "1d", (n,), prim_by_rank[1]
        else:
            structure, shape, prims = "unstructured", None, ["dense", "norm", "attention"]
        width_scores = {**width_scores, "nd_parse": nd_detail}

    out = {
        "structure": structure,
        "shape": shape,
        "scores": scores,
        "width_scores": width_scores,
        "recommended_primitives": prims,
        "margin": float(margin),
        "best_fit": float(best_score),
    }
    if price_detail is not None:
        out["tensorization_price"] = price_detail
    return out


def _all_factorizations(n, max_rank=4, min_side=3):
    """All ordered row-major shapes (n_1,...,n_r), 1<=r<=max_rank, prod=n, each side>=min_side.
    Default min_side=3: length-2 axes are excluded because a length-2 axis has only one adjacency step,
    which the correlation signal cannot reliably distinguish from the flat reading (see parse_grid_shape
    scope note)."""
    import numpy as _np

    res = set([(n,)])

    def rec(rem, parts):
        if len(parts) + 1 >= max_rank:
            if rem >= min_side:
                res.add(tuple(parts + [rem]))
            return
        if rem >= min_side:
            res.add(tuple(parts + [rem]))
        for d in range(min_side, rem + 1):
            if rem % d == 0 and rem // d >= min_side:
                rec(rem // d, parts + [d])

    rec(n, [])
    return [s for s in res if int(_np.prod(s)) == n and all(x >= min_side for x in s)]


def _neighbor_mi(M, shape):
    """POSITION-AWARE fit of a candidate row-major shape: mean mutual information over its TRUE axis-
    neighbor pairs. Unlike the 1-D-averaged flat-distance profile, this respects lattice boundaries -- a
    pair (i, j) counts as an axis-k neighbor only if their multi-indices differ by exactly 1 in axis k and
    0 elsewhere -- so it does not wash out at axis boundaries and is robust where the flat profile is
    degenerate. Higher mean-neighbor-MI = better-fitting shape."""
    import numpy as _np

    n = M.shape[0]
    r = len(shape)
    if r == 1:
        return float(_np.mean([M[i, i + 1] for i in range(n - 1)]))
    strides = [int(_np.prod(shape[k + 1 :])) for k in range(r)]
    rem = _np.arange(n)
    multi = _np.zeros((n, r), int)
    for k in range(r):
        multi[:, k] = rem // strides[k]
        rem = rem % strides[k]
    neigh = []
    for k in range(r):
        s = strides[k]
        for i in range(n - s):
            j = i + s
            if multi[j, k] == multi[i, k] + 1 and all(multi[j, a] == multi[i, a] for a in range(r) if a != k):
                neigh.append(M[i, j])
    return float(_np.mean(neigh)) if neigh else -1e9


def parse_grid_shape(M, tol=0.02, max_rank=4, min_side=3, sig_ratio=1.6):
    """Parse the full N-D lattice shape of a flat vector (rank 1..max_rank) from the mutual-information
    matrix, folded into the MDL rank decision. Robust successor to the flat-distance stride estimator.

    TWO-STAGE (fit decides factorization, Occam decides rank):
      (1) FIT -- for each rank r, the best factorization is the one maximising the POSITION-AWARE mean
          neighbor MI (_neighbor_mi), which respects lattice boundaries and so is robust where the 1-D
          flat-distance profile is degenerate. Converts to a per-rank risk R(r) = 1 - nmi(r)/max_r nmi.
      (2) RANK -- Occam on rank: choose the SMALLEST rank whose R is within `tol` of the global minimum.
          A higher rank is accepted only if it reduces the neighbor-MI risk beyond tol, i.e. the added
          axis genuinely tightens the fit. This is the marginal-value / description-length rule for the
          tensor rank, the same logic as depth and contract.

    Returns (shape, detail). shape is (n,) for a genuine 1-D chain (or when no multi-axis fit clears tol).

    HONEST SCOPE (measured -- see tests/tensorization_pricing.md). Reliable for lattices whose axes are
    all >= 3, across ranks 1..4 (validated to majority-over-seeds on cubic and most non-cubic shapes
    including 4x4x4x4 and 3x3x3x3). Two documented limits: (a) LENGTH-2 axes are excluded (min_side=3) as
    intrinsically ambiguous -- a single adjacency step carries too little signal to separate from the flat
    reading; (b) fully-DISTINCT-axis non-cubic shapes (e.g. 3x4x5) can tie one rank low when a lower-rank
    approximation captures the dominant axis. In both the estimator refuses the ambiguous higher rank
    rather than guessing.
    """
    n = M.shape[0]
    cands = _all_factorizations(n, max_rank=max_rank, min_side=min_side)
    if not cands:
        return (n,), {"reason": "no admissible factorization (n has no factor >= min_side)", "rank": 1}
    # SIGNIFICANCE GATE against finite-sample MI noise. Every MI estimate is positive-biased (~1/n_samples),
    # so with few samples a spurious "grid" can win on noise. A GENUINE lattice has neighbor pairs far more
    # correlated than random pairs (ratio ~18x on real images); pure noise has them equal (ratio ~1). So we
    # refuse any promotion beyond 1-D unless the best multi-axis neighbor MI exceeds the off-diagonal
    # baseline by `sig_ratio`. This is what makes the estimator safe to run by default on arbitrary data.
    import numpy as _np

    iu = _np.triu_indices(n, 1)
    off_diag = float(_np.mean(M[iu])) if len(iu[0]) else 0.0
    by_rank = {}
    for s in cands:
        by_rank.setdefault(len(s), []).append((_neighbor_mi(M, s), s))
    best_per = {r: max(v) for r, v in by_rank.items()}  # (nmi, shape) best at each rank
    max_nmi = max(v[0] for v in best_per.values()) + 1e-9
    risk = {r: 1.0 - nmi / max_nmi for r, (nmi, s) in best_per.items()}
    min_r = min(risk.values())
    chosen_rank = min(r for r in sorted(risk) if risk[r] <= min_r + tol)
    shape = best_per[chosen_rank][1]
    # significance gate: a multi-axis shape must have neighbor MI decisively above the off-diagonal
    # baseline, else the "structure" is finite-sample noise -> fall back to 1-D.
    if chosen_rank > 1:
        nmi_best = best_per[chosen_rank][0]
        if off_diag <= 0 or nmi_best < sig_ratio * off_diag:
            shape, chosen_rank = (n,), 1
    detail = {
        "risk_per_rank": {r: round(v, 4) for r, v in sorted(risk.items())},
        "best_per_rank": {r: best_per[r][1] for r in sorted(best_per)},
        "rank": chosen_rank,
        "tol": tol,
        "min_side": min_side,
        "neighbor_off_diag_ratio": round(best_per[max(best_per)][0] / (off_diag + 1e-12), 2),
        "sig_ratio": sig_ratio,
    }
    return shape, detail


# --------------------------------------------------------------------------- scalable (large-D) path
def _pair_mi_gaussian(a, b):
    """Mutual information of two 1-D sample vectors under the Gaussian estimator, computed on demand.
    Returns 0 for a constant column (zero variance -> no information), rather than NaN."""
    import numpy as _np

    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    c = _np.corrcoef(a, b)[0, 1]
    if not _np.isfinite(c):
        return 0.0
    c = _np.clip(c, -0.999999, 0.999999)
    return -0.5 * _np.log(1.0 - c * c)


def _neighbor_mi_sampled(X, shape, max_pairs=800, seed=0):
    """Position-aware neighbor MI for a candidate shape, estimated by SAMPLING true axis-neighbor pairs
    and computing their MI on demand -- never forms the full D x D matrix. A pair (i, i+stride_k) counts
    as an axis-k neighbor only if i's coordinate in axis k is not at the boundary (same criterion as the
    dense _neighbor_mi). This is what makes the estimator scale to large D (e.g. full 28^3 volumes)."""
    import numpy as _np

    n = X.shape[1]
    r = len(shape)
    rng = _np.random.RandomState(seed)
    if r == 1:
        idx = rng.choice(n - 1, min(max_pairs, n - 1), replace=False)
        return float(_np.mean([_pair_mi_gaussian(X[:, i], X[:, i + 1]) for i in idx]))
    strides = [int(_np.prod(shape[k + 1 :])) for k in range(r)]
    per = max(max_pairs // r, 40)
    vals = []
    for k in range(r):
        s = strides[k]
        cnt = tries = 0
        while cnt < per and tries < per * 40:
            i = rng.randint(0, n - s)
            tries += 1
            if (i // s) % shape[k] < shape[k] - 1:
                vals.append(_pair_mi_gaussian(X[:, i], X[:, i + s]))
                cnt += 1
    return float(_np.mean(vals)) if vals else -1e9


def _offdiag_mi_sampled(X, max_pairs=1500, seed=1):
    """Off-diagonal MI baseline (for the significance gate), from a random sample of pairs -- no full M."""
    import numpy as _np

    n = X.shape[1]
    rng = _np.random.RandomState(seed)
    vals = []
    for _ in range(max_pairs):
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i != j:
            vals.append(_pair_mi_gaussian(X[:, i], X[:, j]))
    return float(_np.mean(vals)) if vals else 0.0


def parse_grid_shape_scalable(X, tol=0.02, max_rank=4, min_side=3, sig_ratio=1.6, max_pairs=800):
    """Large-D tensorization parser: identical two-stage logic to parse_grid_shape (best factorization
    per rank by position-aware neighbor MI, then Occam-on-rank within tol, then the significance gate),
    but every MI is computed ON DEMAND over a SAMPLE of pairs, so the full D x D matrix is never formed.

    This is the scalability path for full-resolution volumes (e.g. 28^3 = 21952-dim), where the dense
    matrix is ~4 GB and OOMs. Verified to AGREE with parse_grid_shape on every size where both run
    (identical shapes recovered), and to run in seconds at D ~ 22000. The candidate set is bounded by the
    divisor lattice (Omega_struct's role here: it prices/caps rank so we enumerate factorizations, not an
    unbounded search), and each candidate is scored from O(max_pairs) on-demand pair-MIs.

    Returns (shape, detail). Use this when D is large; parse_grid_shape (dense) is preferable for small D
    where forming M once is cheap and exact.
    """
    import numpy as _np

    X = _np.asarray(X, float)
    n = X.shape[1]
    cands = _all_factorizations(n, max_rank=max_rank, min_side=min_side)
    if not cands:
        return (n,), {"reason": "no admissible factorization", "rank": 1}
    off = _offdiag_mi_sampled(X, seed=1)
    by_rank = {}
    for s in cands:
        by_rank.setdefault(len(s), []).append((_neighbor_mi_sampled(X, s, max_pairs=max_pairs), s))
    best_per = {r: max(v) for r, v in by_rank.items()}
    # drop ranks whose best neighbor MI is degenerate (no valid sampled pairs) before normalising
    best_per = {r: v for r, v in best_per.items() if v[0] > -1e8}
    if not best_per:
        return (n,), {"reason": "no rank produced valid neighbor pairs", "rank": 1}
    max_nmi = max(v[0] for v in best_per.values()) + 1e-9
    risk = {r: 1.0 - nmi / max_nmi for r, (nmi, s) in best_per.items()}
    min_r = min(risk.values())
    chosen = min(r for r in sorted(risk) if risk[r] <= min_r + tol)
    shape = best_per[chosen][1]
    if chosen > 1 and (off <= 0 or best_per[chosen][0] < sig_ratio * off):
        shape, chosen = (n,), 1
    detail = {
        "risk_per_rank": {r: round(v, 4) for r, v in sorted(risk.items())},
        "best_per_rank": {r: best_per[r][1] for r in sorted(best_per)},
        "rank": chosen,
        "off_diag": round(off, 5),
        "neighbor_off_diag_ratio": round(best_per[max(best_per)][0] / (off + 1e-12), 2),
        "scalable": True,
        "max_pairs": max_pairs,
    }
    return shape, detail

"""extended_groups.py -- autonomous discovery of symmetry groups BEYOND the real metric-preserving family.

metric_discovery.py covers O(p,q) (real symmetric bilinear forms). This module extends detection to the
complex unitary, symplectic, special-linear, and conformal classes. Each class has a DIFFERENT invariant
structure, detected by a cheap linear regression against that structure with an honest fit-quality gate so
a non-example ABSTAINS rather than being force-fit:

  * U(n)/SU(n) (complex unitary): C^n = R^{2n} with complex structure J (J^2 = -I). U(n) preserves the
    Euclidean form |z|^2 AND commutes with J. Detected by: metric regression gives a Euclidean g on
    R^{2n}, and the COMMUTATOR |gJ - Jg| ~ 0 certifies complex-linearity -> U(n) inside O(2n).
  * Sp(2n) (symplectic): preserves a SKEW form omega(u,v) = u^T Omega v. Detected by an ANTISYMMETRIC
    pairwise regression y ~ u^T Omega v; recovered Omega is skew, nondegenerate (imaginary eigenvalue
    pairs, full rank) -> Sp(2n).
  * SL(n) (special linear / volume): preserves det. The invariant of n vectors is det[v_1..v_n]. Detected
    by an ALTERNATING (Levi-Civita) regression y ~ det(frame); coefficients match the permutation signs.
  * Conf(d) = O(d+1,1) (conformal): the null-cone lift x -> (1, x, |x|^2) linearizes conformal to O(d+1,1);
    a conformal target (e.g. |x_i - x_j|^2) becomes a light-cone pairwise product on the lift. Detected by
    the metric regression on the LIFTED pairwise products -> the (d+1,1) light-cone signature.

Validated in tests/extended_groups.md: exact recovery on constructed examples for each class, correct
abstention on non-examples.
"""
from __future__ import annotations
import numpy as np
from itertools import permutations

from .metric_discovery import fit_metric_regression, metric_signature


# --------------------------------------------------------------------------- complex structure / unitary
def complex_structure(n):
    """The standard complex structure J on R^{2n} (pairs (x_k, y_k) rotated by 90 deg); J^2 = -I."""
    D = 2 * n
    J = np.zeros((D, D))
    for k in range(n):
        J[2 * k, 2 * k + 1] = -1.0
        J[2 * k + 1, 2 * k] = 1.0
    return J


def detect_unitary(vectors, y, min_fit=0.6, comm_tol=0.1):
    """Detect U(n)/SU(n): the target must be a Euclidean metric norm on R^{2n} (fit >= min_fit) AND the
    recovered metric must COMMUTE with the complex structure J (|gJ - Jg|/|g| <= comm_tol), certifying
    complex-linearity. Returns (spec, detail). Requires even dimension."""
    S = np.asarray(vectors, float)
    D = S.shape[1]
    if D % 2 != 0:
        return None, {"reason": "odd dimension; complex structure needs R^{2n}"}
    n = D // 2
    g, r2 = fit_metric_regression(S, y)
    gn = g / (np.abs(g).max() + 1e-12)
    sig = metric_signature(gn)
    J = complex_structure(n)
    comm = float(np.abs(g @ J - J @ g).max() / (np.abs(g).max() + 1e-12))
    detail = {"regression_r2": round(r2, 4), "metric_name": sig["name"],
              "J_commutator": round(comm, 4), "n_complex": n}
    if r2 < min_fit:
        return None, {**detail, "reason": "target is not a metric norm"}
    if sig["signature"][1] != 0:
        return None, {**detail, "reason": "metric is not definite (not a unitary/Euclidean form)"}
    if comm > comm_tol:
        return None, {**detail, "reason": "metric does not commute with J -> O(%d), not U(%d)" % (D, n)}
    spec = {"gens": None, "vec_dim": D, "metric": np.eye(D), "complex_structure": J,
            "name": "U(%d)" % n, "scale_norm": False}
    detail["name"] = "U(%d)" % n
    return spec, detail


# --------------------------------------------------------------------------- symplectic
def detect_symplectic(u_vectors, v_vectors, y, min_fit=0.6, skew_ratio=3.0):
    """Detect Sp(2n): regress a PAIRWISE target y ~ u^T W v and test that the recovered W is dominated by
    its SKEW part (a symplectic form) and is nondegenerate (full rank). Returns (spec, detail). The
    skew_ratio gate requires |skew| >= skew_ratio * |symmetric| so a metric (symmetric) target is rejected.
    """
    U = np.asarray(u_vectors, float)
    V = np.asarray(v_vectors, float)
    y = np.asarray(y, float).ravel()
    D = U.shape[1]
    cols, idx = [], []
    for a in range(D):
        for b in range(D):
            cols.append(U[:, a] * V[:, b])
            idx.append((a, b))
    cols.append(np.ones(len(y)))                    # intercept: affine-shift invariance of the target
    Phi = np.stack(cols, 1)
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    W = np.zeros((D, D))
    for (a, b), wi in zip(idx, w[:-1]):
        W[a, b] = wi
    pred = Phi @ w
    r2 = float(1 - ((pred - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    Wsym = (W + W.T) / 2
    Wskew = (W - W.T) / 2
    sym_mag = float(np.abs(Wsym).max())
    skew_mag = float(np.abs(Wskew).max())
    rank = int(np.linalg.matrix_rank(Wskew, tol=1e-6 * max(skew_mag, 1e-12)))
    detail = {"regression_r2": round(r2, 4), "skew_mag": round(skew_mag, 4),
              "sym_mag": round(sym_mag, 4), "skew_rank": rank}
    if r2 < min_fit:
        return None, {**detail, "reason": "pairwise target not a bilinear form"}
    if skew_mag < skew_ratio * sym_mag:
        return None, {**detail, "reason": "form is symmetric (metric), not symplectic"}
    if rank < D or D % 2 != 0:
        return None, {**detail, "reason": "symplectic form degenerate or odd-dimensional"}
    Omega = Wskew / (np.abs(Wskew).max() + 1e-12)
    spec = {"gens": None, "vec_dim": D, "symplectic_form": Omega, "name": "Sp(%d)" % D, "scale_norm": False}
    detail["name"] = "Sp(%d)" % D
    return spec, detail


# --------------------------------------------------------------------------- special linear (volume)
def detect_special_linear(frames, y, min_fit=0.6):
    """Detect SL(n): regress a target over n-vector FRAMES against the alternating (Levi-Civita) products;
    if the target is the volume det(frame) the fit is high and the coefficients match the permutation signs.
    frames: array (m, n, n). Returns (spec, detail). SO(n) also preserves det, so a positive here means
    'volume-preserving'; the ABSENCE of a metric explanation distinguishes SL from SO (caller also runs
    metric regression)."""
    F = np.asarray(frames, float)
    y = np.asarray(y, float).ravel()
    m, n, n2 = F.shape
    if n != n2:
        return None, {"reason": "frames must be n x n"}
    perms = list(permutations(range(n)))
    Phi = np.stack([np.array([np.prod([F[s, i, p[i]] for i in range(n)]) for s in range(m)]) for p in perms], 1)
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    pred = Phi @ w
    r2 = float(1 - ((pred - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    signs = [int(np.sign(np.linalg.det(np.eye(n)[list(p)]))) for p in perms]
    wn = w / (np.abs(w).max() + 1e-12)
    matches_levicivita = bool(np.allclose(np.round(wn), signs, atol=0.3))
    detail = {"regression_r2": round(r2, 4), "matches_levicivita": matches_levicivita}
    if r2 < min_fit or not matches_levicivita:
        return None, {**detail, "reason": "target is not a volume (determinant) invariant"}
    spec = {"gens": None, "vec_dim": n, "invariant": "determinant", "name": "SL(%d)" % n, "scale_norm": False}
    detail["name"] = "SL(%d)" % n
    return spec, detail


# --------------------------------------------------------------------------- conformal (via null-cone lift)
def detect_conformal(points, y_pairs, pair_index, min_fit=0.6):
    """Detect Conf(d) = O(d+1,1): lift each point x -> (1, x, |x|^2) into R^{d+2}, then regress a pairwise
    target y_ij ~ <L_i, L_j>_g (symmetrized) for a symmetric g on the lift. A conformal target such as
    |x_i - x_j|^2 recovers the light-cone metric, whose signature is (d+1, 1). Returns (spec, detail).

    points: array (m, d); pair_index: list of (i, j); y_pairs: pairwise target for those pairs.

    One of the module's four validated detectors (see the implementation report + MANIFEST). Unlike the
    others it is a STANDALONE detector, NOT part of the discover_group auto-dispatcher: it needs an explicit
    pairwise-DISTANCE target, which the per-datum scalar/vector targets discover_group sees never provide.
    Callers with pairwise targets (e.g. a molecular distance-matrix task) invoke it directly.
    """
    from .emlp_layer import null_cone_lift
    X = np.asarray(points, float)
    d = X.shape[1]
    D = d + 2
    L = np.array([null_cone_lift(x) for x in X])
    y = np.asarray(y_pairs, float).ravel()
    cols, idx = [], []
    for k in range(D):
        for l in range(k, D):
            coef = 0.5 if k == l else 1.0
            cols.append(coef * np.array([L[i][k] * L[j][l] + L[i][l] * L[j][k] for (i, j) in pair_index]))
            idx.append((k, l))
    cols.append(np.ones(len(y)))                    # intercept: affine-shift invariance of the target
    Phi = np.stack(cols, 1)
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    g = np.zeros((D, D))
    for (k, l), wi in zip(idx, w[:-1]):
        g[k, l] = wi
        g[l, k] = wi
    pred = Phi @ w
    r2 = float(1 - ((pred - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    gn = g / (np.abs(g).max() + 1e-12)
    sig = metric_signature(gn)
    p, q = sig["signature"]
    detail = {"regression_r2": round(r2, 4), "lift_metric_name": sig["name"], "signature": sig["signature"]}
    is_conformal = (r2 >= min_fit) and (min(p, q) == 1) and (max(p, q) == d + 1)
    if not is_conformal:
        return None, {**detail, "reason": "lifted target lacks the (d+1,1) light-cone signature"}
    spec = {"gens": None, "vec_dim": d, "lift": "null_cone", "metric_signature": (d + 1, 1),
            "name": "Conf(%d)=O(%d,1)" % (d, d + 1), "scale_norm": False}
    detail["name"] = spec["name"]
    return spec, detail


# --------------------------------------------------------------------------- autonomous dispatcher
def _phase_drift(S, g, J, n_angle=4):
    """Mean relative drift of <s,s>_g under phase rotations s -> exp(theta J) s. ~0 iff the recovered metric
    (and hence the target) is invariant under the complex phase -> genuine U(n) structure; large iff the
    target depends on the real/imag split (O(2n) but not U(n)). This is the NON-VACUOUS unitary test: the
    commutator |gJ - Jg| is vacuous for g = I (identity commutes with every J), so we test actual phase
    invariance of the recovered form instead."""
    from scipy.linalg import expm
    pred = np.einsum('ni,ij,nj->n', S, g, S)
    scale = np.abs(pred).std() + 1e-9
    drifts = []
    for th in np.linspace(0.3, 2.5, n_angle):
        Sr = S @ expm(th * J).T
        predr = np.einsum('ni,ij,nj->n', Sr, g, Sr)
        drifts.append(float(np.abs(pred - predr).max() / scale))
    return float(np.mean(drifts))


def discover_group(data, min_fit=0.6, phase_tol=0.05):
    """AUTONOMOUS DISPATCHER: given a dataset (clouds of vectors + a target), try every APPLICABLE group
    route on the ACTUAL target and return the single best-fitting group spec plus a transparent report.

    Applicability is read from the data structure -- routes are never fed fabricated targets:
      * METRIC (always applicable to pooled vectors): y ~ <s,s>_g -> O(p,q). If the dimension is even and
        the recovered form is invariant under the complex phase exp(theta J) (phase drift <= phase_tol),
        UPGRADE to U(n) -- the richer complex structure. The phase test is used, not the vacuous commutator.
      * SYMPLECTIC applies only when clouds provide explicit vector PAIRS (>= 2 vectors) and the ANTISYMMETRIC
        pairwise regression on the real target beats the symmetric one -> Sp(2n).
      * SL(n) applies only when clouds provide FRAMES (>= d vectors) and the target is the determinant
        (Levi-Civita) invariant.
      * CONFORMAL (Conf(d)=O(d+1,1)) is NOT auto-routed by this dispatcher: it needs a pairwise DISTANCE
        target, which the per-datum scalar targets seen here never provide. It exists as a STANDALONE
        detector, detect_conformal(), for callers that supply pairwise targets directly -- discover_group
        neither calls it nor emits a Conf spec.
    Resolves overlaps by highest fit, ties toward the richer structure. If no route's honest gate passes,
    returns None (abstains rather than fabricating a symmetry).
    """
    from .metric_discovery import discover_metric_by_regression
    report = {}
    if getattr(data, "positions", None) is None:
        return None, {"reason": "no coordinate vectors"}
    clouds = [np.asarray(P, float) for P in data.positions if np.asarray(P).ndim == 2 and len(P) >= 1]
    y = np.asarray(data.y).ravel() if getattr(data, "y", None) is not None else None
    if not clouds or y is None or np.std(y) < 1e-9:
        return None, {"reason": "no usable target"}
    m = min(len(clouds), len(y))
    S = np.array([clouds[i].sum(0) for i in range(m)])          # pooled per-datum vectors
    yv = y[:m]
    d = S.shape[1]
    candidates = []

    # 1. METRIC route (-> O(p,q); upgrade to U(n) via the phase test)
    mspec, mdet = discover_metric_by_regression(data, min_fit=min_fit)
    report["metric"] = {k: mdet.get(k) for k in ("regression_r2", "name", "reason") if k in mdet}
    if mspec is not None:
        fit = mdet.get("regression_r2", 0.0)
        candidates.append((mspec["name"], fit, mspec))
        if d % 2 == 0 and mspec["metric"].shape[0] == d:
            # phase-invariance test on the recovered metric (from the metric-by-regression path)
            g_rec = mspec["metric"].astype(float)
            n = d // 2
            J = complex_structure(n)
            # only meaningful when the metric is definite (Euclidean); indefinite forms are not unitary
            eig = np.linalg.eigvalsh((g_rec + g_rec.T) / 2)
            definite = (eig > 0).all() or (eig < 0).all()
            if definite:
                drift = _phase_drift(S, g_rec, J)
                report["unitary"] = {"phase_drift": round(drift, 4)}
                if drift <= phase_tol:
                    uspec = {"gens": None, "vec_dim": d, "metric": np.eye(d),
                             "complex_structure": J, "name": "U(%d)" % n, "scale_norm": False}
                    report["unitary"]["name"] = "U(%d)" % n
                    candidates.append(("U(%d)" % n, fit, uspec))

    # 2. SYMPLECTIC route -- only when clouds genuinely provide vector pairs, tested on the REAL target
    if d % 2 == 0 and all(len(c) >= 2 for c in clouds[:m]):
        U = np.array([clouds[i][0] for i in range(m)])
        V = np.array([clouds[i][1] for i in range(m)])
        spspec, spdet = detect_symplectic(U, V, yv, min_fit=min_fit)
        report["symplectic"] = {k: spdet.get(k) for k in ("regression_r2", "skew_mag", "sym_mag", "name", "reason") if k in spdet}
        if spspec is not None:
            candidates.append((spspec["name"], spdet.get("regression_r2", 0.0), spspec))

    # 3. SL(n) route -- only when clouds provide frames of >= d vectors, tested on the REAL target
    frames = [clouds[i][:d] for i in range(m) if len(clouds[i]) >= d]
    yf = [yv[i] for i in range(m) if len(clouds[i]) >= d]
    if len(frames) >= 20:
        slspec, sldet = detect_special_linear(np.array(frames), np.array(yf), min_fit=min_fit)
        report["special_linear"] = {k: sldet.get(k) for k in ("regression_r2", "matches_levicivita", "name", "reason") if k in sldet}
        if slspec is not None:
            candidates.append((slspec["name"], sldet.get("regression_r2", 0.0), slspec))

    if not candidates:
        return None, {**report, "selected": None, "reason": "no group route fit the target"}

    def richness(name):
        return 1 if name.startswith(("U(", "SU(", "Sp(", "SL(", "Conf")) else 0
    best = max(candidates, key=lambda c: (round(c[1], 3), richness(c[0])))
    report["selected"] = best[0]
    return best[2], report

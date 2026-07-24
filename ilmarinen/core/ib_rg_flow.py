"""Information-bottleneck-as-RG flow: a quantitative coarse-graining scale for the redundancy reducer (B8).

The package already has a redundancy reducer (core/redundancy_reduction.py) that measures an effective
dimension d_eff as the participation ratio of the covariance spectrum and projects onto that many principal
components. Its docstring calls this "the RG keep-the-relevant-modes move" -- but that is a METAPHOR: a single
static PCA snapshot, with no scale and no flow. Renormalization is fundamentally about how quantities CHANGE as
one integrates out degrees of freedom scale by scale. This module supplies the missing quantitative object: a
genuine flow of an effective dimension along a coarse-graining scale, grounded in an EXACT correspondence
rather than an analogy.

The anchor is the Gaussian Information Bottleneck (GIB; Chechik, Globerson, Tishby & Weiss, JMLR 2005) and its
formal equivalence to a non-perturbative RG coarsening (Kline & Palmer, New J. Phys. 2022, arXiv:2107.13700).
For jointly Gaussian (X, Y) the IB-optimal compressed representation T is a noisy linear projection onto the
eigenvectors of the normalized regression matrix

    M = Sigma_{X|Y} Sigma_X^{-1},     eigenvalues  lambda_i in (0, 1),

which is the CANONICAL-CORRELATION basis between X and Y (lambda_i = 1 - rho_i^2 for canonical correlations
rho_i). A smaller lambda_i means mode i carries MORE information about the relevance variable Y. The IB
tradeoff parameter beta plays the role of an inverse RG scale: each mode i switches on only once beta exceeds a
critical value

    beta_c(lambda_i) = 1 / (1 - lambda_i),

so the effective dimension

    d_IB(beta) = #{ i : beta > beta_c(lambda_i) } = #{ i : lambda_i < 1 - 1/beta }

is a STAIRCASE that grows from 0 (maximal compression) to the full rank (no compression) through a cascade of
structural phase transitions -- exactly the RG picture in which relevant modes (small lambda, large canonical
correlation with Y) persist to the coarsest scales and irrelevant modes are integrated out first. This is the
quantitative content the metaphor lacked: a flow d_IB(beta) with EXACTLY LOCATED transitions, not a single
number. The map is exact for Gaussian statistics and carries a semigroup structure (successive coarsenings
compose), which is the defining property of an RG flow.

Two effective dimensions, contrasted:
  * d_eff (redundancy_reduction): participation ratio of the covariance of X alone -- UNSUPERVISED, static, no
    relevance variable, no scale. "How many directions carry the variance."
  * d_IB(beta) (here): SUPERVISED (uses the X-Y relation), a flow in the scale beta with located transitions.
    "How many directions carry information about Y, resolved by coarse-graining scale."
The first is a snapshot; the second is the flow the RG framing actually refers to. This module reports both so
the relationship is explicit and measurable, not asserted.
"""

from __future__ import annotations

import numpy as np


def gib_spectrum(X, Y, ridge=1e-6):
    """Canonical GIB eigenvalues lambda_i of M = Sigma_{X|Y} Sigma_X^{-1} for data X (n,dx), Y (n,dy) or (n,).

    Returns (lam, rho2) with lam sorted ASCENDING (smallest = most informative about Y) and rho2 = 1 - lam the
    squared canonical correlations. lam in (0,1] up to sampling/regularization. A ridge stabilizes the inverse
    for near-degenerate covariances (irrelevant modes with lambda ~ 1)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    n = len(X)
    Sxx = (X.T @ X) / (n - 1) + ridge * np.eye(X.shape[1])
    Syy = (Y.T @ Y) / (n - 1) + ridge * np.eye(Y.shape[1])
    Sxy = (X.T @ Y) / (n - 1)
    Sx_given_y = Sxx - Sxy @ np.linalg.inv(Syy) @ Sxy.T
    M = Sx_given_y @ np.linalg.inv(Sxx)
    lam = np.linalg.eigvals(M).real
    lam = np.clip(lam, 1e-12, 1.0)
    lam = np.sort(lam)
    rho2 = np.clip(1.0 - lam, 0.0, 1.0)
    return lam, rho2


def critical_betas(lam):
    """The switch-on scale of each mode: beta_c(lambda) = 1/(1-lambda). Sorted ascending with lam (so the most
    informative mode, smallest lambda, has the smallest beta_c and is the first to survive coarse-graining)."""
    lam = np.asarray(lam, dtype=np.float64)
    return 1.0 / np.clip(1.0 - lam, 1e-12, None)


def ib_effective_dimension(lam, beta):
    """The IB effective dimension at scale beta: number of modes already switched on, #{i: beta > beta_c}.
    Equivalently #{i: lambda_i < 1 - 1/beta}. beta may be a scalar or array; returns matching int(s)."""
    lam = np.asarray(lam, dtype=np.float64)
    bc = critical_betas(lam)
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim == 0:
        return int((beta > bc).sum())
    return np.array([(b > bc).sum() for b in beta], dtype=int)


def layer_rg_flow(layer_activations, Y, betas=None, ridge=1e-6):
    """Measure the GIB flow ACROSS a trained network's layers -- the RG-flow-in-neural-nets reading of depth
    (Mehta & Schwab 2014; Tishby & Zaslavsky 2015): each layer coarse-grains, so the per-layer representation
    should carry progressively more of the RELEVANT (target) information in fewer collective modes.

    layer_activations : list of (n, d_l) arrays, the hidden representation at each layer (in order).
    Y                 : (n,) or (n, dy) target / relevance variable.

    For each layer computes the GIB spectrum of (activation, Y) and reports:
      top_canonical_corr : sqrt(1 - min lambda), how linearly decodable Y is from that layer (rises toward 1
                           as the net coarse-grains the target into a single collective coordinate),
      relevant_information: sum_i -0.5 log lambda_i, the GIB relevant information at beta->inf (grows as the
                           representation aligns with Y),
      n_informative      : #{ lambda_i < 0.9 }, an effective count of target-informative modes,
      d_IB_at_beta       : d_IB(beta_ref) if a single reference scale is given via betas as a scalar.
    A monotone rise in top_canonical_corr / relevant_information across layers is the measurable RG flow: the
    same GIB object that gives the beta-flow (ib_rg_flow) applied along depth instead of along the scale.
    """
    beta_ref = None
    if betas is not None and np.ndim(betas) == 0:
        beta_ref = float(betas)
    rows = []
    for li, H in enumerate(layer_activations):
        H = np.asarray(H, dtype=np.float64)
        lam, rho2 = gib_spectrum(H, Y, ridge=ridge)
        row = {"layer": li, "top_canonical_corr": float(np.sqrt(np.clip(1.0 - lam.min(), 0.0, 1.0))),
               "relevant_information": float(-0.5 * np.sum(np.log(np.clip(lam, 1e-6, 1.0)))),
               "n_informative": int((lam < 0.9).sum()), "min_lambda": float(lam.min())}
        if beta_ref is not None:
            row["d_IB_at_beta"] = int(ib_effective_dimension(lam, beta_ref))
        rows.append(row)
    ccs = [r["top_canonical_corr"] for r in rows]
    monotone = all(ccs[i] <= ccs[i + 1] + 1e-6 for i in range(len(ccs) - 1))
    return {"layers": rows, "top_canonical_corr_monotone": bool(monotone),
            "note": "GIB flow across layers: a monotone rise in top_canonical_corr/relevant_information is the "
                    "measurable RG coarse-graining of depth toward the target, quantified by the same Gaussian "
                    "IB spectrum used for the beta-scale flow."}


def ib_rg_flow(X, Y, betas=None, ridge=1e-6):
    """Measure the full IB-as-RG flow of the effective dimension along the coarse-graining scale beta.

    Returns a dict:
      lam                : GIB eigenvalues (ascending)
      canonical_corr     : sqrt(1 - lam), the canonical correlations with Y (descending)
      critical_betas     : beta_c(lambda_i) = 1/(1-lambda_i), the located mode-switch-on transitions
      betas              : the scan of scales
      d_IB               : d_IB(beta), the staircase effective dimension along the flow
      transitions        : list of (beta_c, mode_index) where d_IB jumps, i.e. the RG phase transitions
      d_eff_static       : the UNSUPERVISED participation-ratio d_eff of Cov(X) (the metaphor's single number),
                           for contrast with the flow
    The flow is the quantitative object; d_eff_static is the pre-existing snapshot. Reporting both makes the
    metaphor-to-mechanism upgrade explicit and measurable.
    """
    lam, rho2 = gib_spectrum(X, Y, ridge=ridge)
    bc = critical_betas(lam)
    if betas is None:
        lo = max(1.0001, bc.min() * 0.5)
        hi = bc.max() * 2.0 if np.isfinite(bc.max()) else 1e4
        betas = np.logspace(np.log10(lo), np.log10(hi), 200)
    betas = np.asarray(betas, dtype=np.float64)
    d = ib_effective_dimension(lam, betas)
    transitions = [(float(b), int(i)) for i, b in enumerate(np.sort(bc)) if np.isfinite(b)]
    # Separate the GENUINELY informative modes from the finite-sample noise floor. With d features on n
    # samples, near-degenerate modes (lambda -> 1, canonical corr -> 0) carry no real information about Y and
    # produce spuriously huge beta_c; they are not part of the RG cascade. Flag informative modes by a
    # canonical-correlation floor scaled to the sampling noise ~ sqrt(dy / n).
    n = len(np.asarray(X))
    dy = 1 if np.asarray(Y).ndim == 1 else np.asarray(Y).shape[1]
    corr_floor = max(0.1, np.sqrt(dy / max(n, 1)) * 1.5)
    informative = rho2 > corr_floor ** 2                    # modes with real canonical correlation to Y
    n_informative = int(informative.sum())
    bc_informative = np.sort(bc[informative]) if n_informative else np.array([])
    # static unsupervised participation ratio of Cov(X) (the metaphor)
    Xc = np.asarray(X, dtype=np.float64); Xc = Xc - Xc.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    ev = s ** 2
    p = ev / ev.sum() if ev.sum() > 0 else np.ones_like(ev) / len(ev)
    d_eff_static = float(1.0 / np.sum(p ** 2))
    return {"lam": lam, "canonical_corr": np.sort(np.sqrt(rho2))[::-1], "critical_betas": np.sort(bc),
            "betas": betas, "d_IB": d, "transitions": transitions, "d_eff_static": d_eff_static,
            "n_informative": n_informative, "critical_betas_informative": bc_informative,
            "corr_floor": float(corr_floor)}

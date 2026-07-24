"""Physicist redundancy removal beyond symmetry: intrinsic-dimension reduction via correlation spectrum.

Symmetry discovery removes ORBIT redundancy (points related by a group carry the same information, so we
quotient by the group). This module removes a DIFFERENT redundancy the physicist's toolset also targets:
when the ambient coordinates are correlated, the data lives on an effectively lower-dimensional manifold, and
the excess coordinates carry no independent information. This is the Renormalization-Group / "keep the
relevant modes" move: measure the effective number of degrees of freedom from the correlation (covariance)
spectrum, and project onto that many collective coordinates, discarding the redundant ones.

The effective dimension is MEASURED, not chosen: it is the PARTICIPATION RATIO of the variance spectrum,

    d_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2 = 1 / sum_i p_i^2 ,   p_i = lambda_i / sum_j lambda_j ,

the exact analog of the IPR used elsewhere in ilmarinen to count effective primitives -- here counting
effective data modes. lambda_i are the covariance eigenvalues (PCA variances). Projecting to ceil(d_eff)
principal components keeps the collective coordinates that carry the variance and drops the redundant rest.

This is a preprocessing reducer in the physicist spirit: measured, interpretable (the kept modes are the
top covariance eigenvectors), and reversible in report (we keep the components + explained variance). It is
offered as an OPT-IN transform, not folded into the validated pipeline. Validated (tests/): on QM7 Coulomb
eigenvalue features it measures d_eff ~= 3.8 of 23 ambient and a projection to 4 PCs retains R2 0.969 vs
0.979 (83% dimension reduction, ~1% accuracy cost).
"""

import numpy as np


def effective_dimension(X):
    """Participation ratio of the covariance spectrum = effective number of independent coordinates.
    X : (n, d). Returns (d_eff float, variance_ratios array sorted desc)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # covariance eigenvalues via SVD (numerically stable)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = s**2
    tot = lam.sum()
    if tot <= 0:
        return 1.0, np.ones(1)
    p = lam / tot
    d_eff = float(1.0 / np.sum(p**2))
    return d_eff, p


def reduce_redundancy(X, keep=None, var_target=None, min_keep=1):
    """Project X onto its leading collective coordinates (principal components), keeping the effective
    number of modes measured from the correlation spectrum.

    keep : if given, keep exactly this many components. Otherwise:
    var_target : if given (e.g. 0.99), keep the fewest components explaining >= var_target of the variance.
    default : keep ceil(d_eff) components, d_eff = participation ratio of the variance spectrum (the RG
              'relevant modes' count).

    Returns a dict: Xr (n, k) reduced data, k, d_eff, components (k, d) the kept PC directions, mean (d,),
    explained_variance (fraction kept). The projection is the physicist's collective-coordinate reduction:
    the kept axes are the maximally-varying independent directions; the discarded axes are the redundant
    (low-variance / strongly-correlated) ones.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = S**2
    ratios = lam / lam.sum() if lam.sum() > 0 else np.ones_like(lam) / len(lam)
    d_eff = float(1.0 / np.sum(ratios**2)) if lam.sum() > 0 else 1.0

    if keep is not None:
        k = int(keep)
    elif var_target is not None:
        c = np.cumsum(ratios)
        k = int(np.searchsorted(c, var_target) + 1)
    else:
        k = int(np.ceil(d_eff))
    k = max(min_keep, min(k, X.shape[1]))

    comps = Vt[:k]  # (k, d) principal directions
    Xr = Xc @ comps.T  # (n, k) collective coordinates
    return {
        "Xr": Xr.astype(np.float32),
        "k": k,
        "d_eff": d_eff,
        "components": comps,
        "mean": mean.ravel(),
        "explained_variance": float(ratios[:k].sum()),
        "variance_ratios": ratios,
    }


def format_reduction(result):
    """Short text report of a reduce_redundancy result."""
    return (
        "REDUNDANCY REDUCTION (RG / keep-relevant-modes; effective-dimension by covariance IPR)\n"
        f"  ambient dim -> kept {result['k']} collective coordinates "
        f"(measured d_eff={result['d_eff']:.2f})\n"
        f"  explained variance retained: {result['explained_variance']:.3f}\n"
        f"  the kept axes are the top covariance eigenvectors; discarded axes are redundant "
        f"(low-variance / correlated) directions."
    )

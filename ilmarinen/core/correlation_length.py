"""Derive the conv RECEPTIVE FIELD (kernel size) from the data's spatial CORRELATION LENGTH, instead
of hardcoding kernel_size=3. Addresses fixed-hyperparameter audit gap B4.

Principle. A conv kernel of size k captures locality up to radius k//2. The right k is set by the
spatial correlation length xi -- the displacement over which pixel/voxel values stay correlated. If
correlations decay as C(r) ~ exp(-r/xi), a kernel needs radius ~xi to capture the local structure;
larger wastes parameters, smaller misses it. This is the 2D/3D analogue of the mode_structure
correlation analysis already used to pick the tensor contract -- the same "measure the correlation
scale, match the resource to it" principle used for width (dual certificate) and depth (priced rule).

Method. Compute the radial autocorrelation of the (channel-averaged) field:
    C(r) = < x(p) . x(p+r) >_p , averaged over displacements of magnitude r,
normalized so C(0)=1. Find xi as the smallest r where C(r) drops below 1/e (or fit exp(-r/xi)).
Recommend kernel size k = 2*ceil(xi) + 1 (odd, covers +-xi), clamped to a small candidate set.

This does NOT itself train; it returns a recommended kernel size (and the measured xi) that a caller
can pass to the spatial/volumetric schema, or use to price a kernel-size selection.
"""
from __future__ import annotations
import numpy as np


def _radial_autocorr_2d(field, max_r):
    """field: (H, W) real. Returns C(r) for r=0..max_r (normalized, C(0)=1)."""
    f = field - field.mean()
    var = (f * f).mean()
    if var < 1e-12:
        return np.ones(max_r + 1)  # constant field: trivially "correlated" everywhere
    H, W = f.shape
    C = np.zeros(max_r + 1)
    C[0] = 1.0
    for r in range(1, max_r + 1):
        # average correlation over the 4 axis-aligned displacements at distance r
        vals = []
        if r < W:
            vals.append((f[:, :-r] * f[:, r:]).mean())
        if r < H:
            vals.append((f[:-r, :] * f[r:, :]).mean())
        C[r] = (np.mean(vals) / var) if vals else 0.0
    return C


def _radial_autocorr_3d(field, max_r):
    """field: (D, H, W) real. Returns C(r) over the 3 axes."""
    f = field - field.mean()
    var = (f * f).mean()
    if var < 1e-12:
        return np.ones(max_r + 1)
    D, H, W = f.shape
    C = np.zeros(max_r + 1); C[0] = 1.0
    for r in range(1, max_r + 1):
        vals = []
        if r < D: vals.append((f[:-r] * f[r:]).mean())
        if r < H: vals.append((f[:, :-r] * f[:, r:]).mean())
        if r < W: vals.append((f[:, :, :-r] * f[:, :, r:]).mean())
        C[r] = (np.mean(vals) / var) if vals else 0.0
    return C


def _xi_from_curve(C):
    """Correlation length: first r where C(r) < 1/e; interpolate for a fractional estimate."""
    thr = 1.0 / np.e
    for r in range(1, len(C)):
        if C[r] < thr:
            # linear interpolation between r-1 and r
            c0, c1 = C[r - 1], C[r]
            frac = (c0 - thr) / max(c0 - c1, 1e-9)
            return (r - 1) + frac
    return float(len(C) - 1)  # never decayed within window -> long-range; cap at window


def recommend_kernel_size(images, ndim=2, candidates=(3, 5, 7, 9), max_r=None, sample=64):
    """Recommend a conv kernel size from the spatial correlation length of a batch of fields.

    images : (N, H, W) or (N, C, H, W) for 2D; (N, D, H, W) or (N, C, D, H, W) for 3D. Channels are
             averaged out (correlation of the scalar intensity field).
    candidates : allowed odd kernel sizes; the recommendation is snapped to the nearest.
    Returns dict(xi, kernel_size, C_curve).
    """
    arr = np.asarray(images, dtype=np.float32)
    # collapse channel axis if present
    if ndim == 2:
        if arr.ndim == 4: arr = arr.mean(1)          # (N,C,H,W)->(N,H,W)
        assert arr.ndim == 3, "2D expects (N,H,W) or (N,C,H,W)"
        H, W = arr.shape[1:]
        mr = max_r or min(max(H, W) // 2, 12)
        auto = _radial_autocorr_2d
    else:
        if arr.ndim == 5: arr = arr.mean(1)
        assert arr.ndim == 4, "3D expects (N,D,H,W) or (N,C,D,H,W)"
        Dd, H, W = arr.shape[1:]
        mr = max_r or min(max(Dd, H, W) // 2, 8)
        auto = _radial_autocorr_3d

    n = min(sample, arr.shape[0])
    curves = np.stack([auto(arr[i], mr) for i in range(n)], 0)
    C = curves.mean(0)
    xi = _xi_from_curve(C)
    # kernel must cover +- xi -> k ~ 2*ceil(xi)+1, snapped to the candidate set
    want = 2 * int(np.ceil(xi)) + 1
    k = min(candidates, key=lambda c: abs(c - want))
    return {"xi": float(xi), "kernel_size": int(k), "want": int(want), "C_curve": C.tolist()}


# map a recommended kernel size to the matching spatial-schema primitive name
_KERNEL_TO_PRIMITIVE = {3: "conv2d", 5: "conv2d_k5", 7: "conv2d_k7", 9: "conv2d_k7"}


def recommend_conv_primitive(images, ndim=2, **kw):
    """Return the spatial-schema conv primitive name whose receptive field matches the data's
    spatial correlation length (plus the measured xi). E.g. smooth data -> 'conv2d_k7'."""
    r = recommend_kernel_size(images, ndim=ndim, **kw)
    prim = _KERNEL_TO_PRIMITIVE.get(r["kernel_size"], "conv2d")
    return {"primitive": prim, "kernel_size": r["kernel_size"], "xi": r["xi"]}


def recommend_n_rbf(edge_distances, cutoff, min_rbf=4, max_rbf=32):
    """Derive the number of radial basis functions from the data's bond-length distribution, instead
    of a fixed n_rbf. Addresses the last radial ad-hoc choice (the RBF COUNT; the range and Gaussian
    width are already derived from the cutoff and spacing).

    Principle (Nyquist-like, the radial analogue of recommend_kernel_size). The RBF centers span
    [0, cutoff] with spacing = cutoff/(n_rbf-1). To resolve the radial structure without gaps or waste,
    the spacing should be no coarser than the finest radial feature scale -- the spread (std) of the
    nearest-neighbor / bond-length peak:
        spacing <= sigma_bond  =>  n_rbf >= cutoff / sigma_bond + 1.
    edge_distances: 1D array of edge lengths (e.g. all bonded distances in the dataset). Returns an
    integer n_rbf clamped to [min_rbf, max_rbf]. For QM7 (sigma_bond ~ 0.55 Bohr, cutoff 3.5) this
    yields ~8, matching the previous hand-set default -- now derived rather than guessed."""
    d = np.asarray(edge_distances, dtype=np.float64)
    d = d[d <= cutoff]
    sigma = float(d.std()) if d.size > 1 else cutoff / 8.0
    sigma = max(sigma, 1e-3)
    n = int(np.ceil(cutoff / sigma)) + 1
    return int(np.clip(n, min_rbf, max_rbf))

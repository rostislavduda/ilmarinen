"""T-RG: redundancy reduction and IB-as-RG flow.

Locks the effective-dimension bounds and the RG-flow monotonicity (more modes switch on as the
inverse-RG-scale beta increases) that B8 rests on -- and that the D3 unification would build on.
"""

import numpy as np

from ilmarinen import effective_dimension, ib_rg_flow, reduce_redundancy


def test_effective_dimension_bounds_lowrank():
    """T-RG-1: with r strong directions + noise, 1 <= d_eff <= ambient dim."""
    rng = np.random.RandomState(0)
    n, amb, r = 200, 10, 3
    Z = rng.randn(n, r)
    W = rng.randn(r, amb)
    X = (Z @ W + 0.01 * rng.randn(n, amb)).astype(np.float32)  # ~rank-3 signal
    d_eff, spectrum = effective_dimension(X)
    assert 1.0 <= d_eff <= amb
    assert d_eff < amb  # genuinely low-rank -> below ambient
    assert spectrum.shape[0] == amb


def test_effective_dimension_isotropic_near_full():
    """T-RG-1b: isotropic data uses ~all directions (d_eff close to ambient)."""
    rng = np.random.RandomState(1)
    amb = 8
    X = rng.randn(400, amb).astype(np.float32)
    d_eff, _ = effective_dimension(X)
    assert d_eff > 0.6 * amb  # participation ratio high for isotropic data


def test_reduce_redundancy_projects_to_k():
    """T-RG-2: reduce_redundancy returns k components and a reduced matrix with k columns."""
    rng = np.random.RandomState(2)
    Z = rng.randn(150, 2)
    W = rng.randn(2, 12)
    X = (Z @ W + 0.01 * rng.randn(150, 12)).astype(np.float32)  # rank-2
    out = reduce_redundancy(X)
    assert isinstance(out, dict)
    k = out["k"]
    assert 1 <= k <= 12
    assert out["Xr"].shape == (150, k)
    assert k < 12  # low-rank input should be compressed


def test_ib_rg_flow_monotone_in_beta():
    """T-RG-3: d_IB(beta) is non-decreasing in beta (modes switch on with finer scale)."""
    rng = np.random.RandomState(3)
    X = rng.randn(200, 6).astype(np.float32)
    # a target that depends on the first two directions
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    r = ib_rg_flow(X, y)
    d_IB = np.asarray(r["d_IB"])
    betas = np.asarray(r["betas"])
    assert betas[0] < betas[-1]
    # allow tiny numerical non-monotone wiggle but require overall non-decreasing trend
    assert np.all(np.diff(d_IB) >= -1e-6), "d_IB should be non-decreasing in beta"
    assert d_IB[-1] >= d_IB[0]


def test_ib_rg_flow_reports_static_deff():
    """T-RG-3b: the flow result also reports a finite static d_eff for comparison."""
    rng = np.random.RandomState(4)
    X = rng.randn(120, 5).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    r = ib_rg_flow(X, y)
    assert np.isfinite(r["d_eff_static"]) and r["d_eff_static"] > 0

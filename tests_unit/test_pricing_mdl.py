"""T-PR: the J = R + mu*Omega pricing pieces.

D1-critical. The planned work replaces the structural Omega with a singular (LLC-based) code length.
These tests lock the exact identities and the lattice ordering that must survive that change.
"""

import math

import numpy as np
import pytest

from ilmarinen import (certificate_lambda_scale, free_energy, score_to_nll,
                     spectral_code_length)
from ilmarinen.machinery.contract_mdl import omega_struct


def test_free_energy_identity():
    """T-PR-1: free_energy(L, lam, n) == n*L + lam*log n exactly. D1 generalizes this object."""
    for L, lam, n in [(0.5, 2.0, 1000), (0.1, 0.0, 50), (1.3, 5.5, 10_000)]:
        assert free_energy(L, lam, n) == pytest.approx(n * L + lam * math.log(n), rel=1e-12)


def test_score_to_nll_monotone_classification():
    """T-PR-2: higher accuracy -> strictly lower NLL (classification)."""
    accs = [0.5, 0.6, 0.75, 0.9, 0.99]
    nlls = [score_to_nll(a, "classification", n_classes=2) for a in accs]
    assert all(nlls[i] > nlls[i + 1] for i in range(len(nlls) - 1)), nlls
    assert all(math.isfinite(v) for v in nlls)


def test_score_to_nll_chance_baseline():
    """T-PR-2b: chance accuracy maps near the log(n_classes) baseline (order of magnitude)."""
    n_classes = 4
    nll_chance = score_to_nll(1.0 / n_classes, "classification", n_classes=n_classes)
    # should be close to -log(1/k) = log k; allow generous tolerance for the estimator's shaping
    assert 0.4 * math.log(n_classes) < nll_chance < 2.5 * math.log(n_classes)


def test_spectral_code_length_nonneg_and_monotone():
    """T-PR-3: mode code length is non-negative and non-decreasing in the number of modes."""
    vals = [spectral_code_length(m, spatial_dims=1, channels=1) for m in [1, 4, 8, 16, 32]]
    assert all(v >= 0 for v in vals)
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)), vals


def test_omega_struct_lattice_ordering():
    """T-PR-4: set <= graph <= equivariant structural code length (same N). D1 must preserve this."""
    N, E, d = 20, 40, 3
    o_set = omega_struct("set", N)
    o_graph = omega_struct("graph", N, E=E)
    o_equiv = omega_struct("equivariant", N, E=E, d=d)
    assert o_set <= o_graph <= o_equiv, (o_set, o_graph, o_equiv)
    assert o_set == pytest.approx(0.0, abs=1e-9)  # set is the reference (unordered multiset)


def test_omega_struct_rank_monotone():
    """T-PR-4b: higher tensor rank costs at least as much structural code length."""
    o_r1 = omega_struct("tensor", N=10, shape=(10,), rank=1)
    o_r2 = omega_struct("tensor", N=10, shape=(10, 4), rank=2)
    o_r3 = omega_struct("tensor", N=10, shape=(10, 4, 4), rank=3)
    assert o_r1 <= o_r2 <= o_r3, (o_r1, o_r2, o_r3)


def test_certificate_lambda_scale_positive_finite():
    """T-PR-5: certificate scale is a positive finite float on a simple regression array."""
    rng = np.random.RandomState(0)
    X = rng.randn(40, 6).astype(np.float32)
    y = (X[:, 0] * 1.0).astype(np.float32)
    out = certificate_lambda_scale(X, y)
    scale = out[0] if isinstance(out, tuple) else out
    assert math.isfinite(scale) and scale > 0

"""T-SM1 (singular MDL / D1): the functional code length lambda*log n and its fusion into Omega.

Locks the identities and, critically, the guard: a non-converged optimum must never be priced. The
value/estimator noise lives in the @smoke tests; the identity and guard tests are fast and exact.
"""

import math

import numpy as np
import pytest
import torch

from ilmarinen.machinery.contract_mdl import omega_struct
from ilmarinen.machinery.singular_mdl import (omega_func, singular_complexity_of,
                                            singular_free_energy, total_code_length)


# --------------------------------------------------------------------------- exact identities (fast)
def test_omega_func_identity():
    """omega_func(lambda, n) == lambda * log n for non-negative lambda."""
    assert omega_func(2.0, 1000) == pytest.approx(2.0 * math.log(1000), rel=1e-12)
    assert omega_func(0.0, 500) == 0.0


def test_omega_func_clamps_negative():
    """A (mildly) negative lambda is clamped to zero code length: RLCT >= 0, so no negative Omega."""
    assert omega_func(-0.3, 1000) == 0.0
    assert omega_func(-5.0, 1000) == 0.0


def test_omega_func_small_n_guard():
    """n <= 1 yields zero code length (log n undefined/zero), not a crash."""
    assert omega_func(3.0, 1) == 0.0
    assert omega_func(3.0, 0) == 0.0


def test_total_code_length_is_struct_plus_func():
    """Omega_total = omega_struct(contract) + omega_func(lambda, n). The two terms are complementary."""
    lam, n = 2.0, 1000
    tot = total_code_length(lam, n, "graph", N=20, E=40)
    expect = omega_struct("graph", N=20, E=40) + omega_func(lam, n)
    assert tot == pytest.approx(expect, rel=1e-12)


def test_total_reduces_to_struct_when_lambda_zero():
    """With lambda=0 (a regular, zero-effective-dimension fit) the total is just the structural term."""
    assert total_code_length(0.0, 1000, "equivariant", N=20, E=40, d=3) == pytest.approx(
        omega_struct("equivariant", N=20, E=40, d=3), rel=1e-12
    )


def test_singular_free_energy_identity():
    """F_n = R + lambda*log n (R the residual NLL)."""
    assert singular_free_energy(50.0, 2.0, 1000) == pytest.approx(50.0 + omega_func(2.0, 1000), rel=1e-12)


def test_functional_term_preserves_structural_ordering():
    """Adding the SAME functional term to each contract preserves set <= graph <= equivariant.

    This is the D1 safety property: the functional price is an additive offset per fitted model; it
    must not reorder the structural lattice when the functional complexity is held fixed.
    """
    lam, n = 1.5, 800
    o_set = total_code_length(lam, n, "set", N=20)
    o_graph = total_code_length(lam, n, "graph", N=20, E=40)
    o_equiv = total_code_length(lam, n, "equivariant", N=20, E=40, d=3)
    assert o_set <= o_graph <= o_equiv


# --------------------------------------------------------------------------- estimator + guard
def _fit_linear(n=200, d=4, seed=0, steps=400):
    rng = np.random.RandomState(seed)
    X = torch.tensor(rng.randn(n, d), dtype=torch.float32)
    w = np.zeros(d, np.float32)
    w[0] = 1.0
    y = torch.tensor(X.numpy() @ w, dtype=torch.float32)
    model = torch.nn.Linear(d, 1, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()

    def closure():
        return ((model(X).squeeze(-1) - y) ** 2).mean()

    return model, closure, n


def test_guard_rejects_nonconverged():
    """D1-CRITICAL: a barely-trained optimum must be flagged invalid and NOT priced.

    The whole point of fusing the LLC into Omega safely is that a garbage lambda (from a non-minimum)
    never enters the objective. This locks that contract.
    """
    model, closure, n = _fit_linear(steps=1)  # deliberately NOT converged
    res = singular_complexity_of(model, closure, n, chains=3, steps=120, burn=40, seed=0)
    assert res["valid"] is False
    assert math.isnan(res["omega_func"]), "non-converged optimum must not yield a finite functional price"


@pytest.mark.smoke
def test_converged_gives_valid_positive_price():
    """At a clean minimum the functional price is valid, finite, and non-negative (SGLD-noisy -> @smoke)."""
    model, closure, n = _fit_linear(steps=400)
    res = singular_complexity_of(model, closure, n, chains=4, steps=250, burn=80, seed=0)
    assert res["valid"] is True
    assert math.isfinite(res["omega_func"])
    assert res["omega_func"] >= 0.0
    # omega_func == max(lambda,0) * log n
    assert res["omega_func"] == pytest.approx(max(res["lambda"], 0.0) * math.log(n), rel=1e-9)


# --------------------------------------------------------------------------- live wiring into fit (D1)
def _graph_data_with_positions(seed=0, ng=24):
    """Graph data carrying BOTH edges and positions, so the full set/graph/equivariant bake-off runs."""
    from ilmarinen import AllData

    rng = np.random.RandomState(seed)
    nf = [rng.randn(6, 3).astype(np.float32) for _ in range(ng)]
    pos = [rng.randn(6, 3).astype(np.float32) for _ in range(ng)]
    edges = [np.array([(i, (i + 1) % 6) for i in range(6)] + [(0, 3)], np.int64).T for _ in range(ng)]
    y = np.array([int(f[:, 0].sum() > 0) for f in nf])
    return AllData.graphs(np.stack(nf), edges, y=y, positions=np.stack(pos))


@pytest.mark.smoke
def test_price_singular_wires_into_fit():
    """D1 end-to-end: price_singular=True runs, records per-contract functional pricing, all omega_func >= 0."""
    from ilmarinen import AllGraph

    data = _graph_data_with_positions()
    mg = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0,
                   price_singular=True, contract_router=None)
    r = mg.fit(data, task="classification", n_out=2, tiebreak=True)
    assert np.isfinite(r["value"])
    sp = (mg.route_detail or {}).get("tiebreak", {}).get("singular_pricing", {})
    assert sp, "singular_pricing detail missing -- D1 did not fire in the bake-off"
    for _c, d in sp.items():
        if d.get("applied"):
            assert d["omega_func"] >= 0.0  # RLCT-clamped functional code length


@pytest.mark.smoke
def test_price_singular_off_is_default():
    """Default (price_singular off) records no singular pricing -- the feature is strictly opt-in."""
    from ilmarinen import AllGraph

    data = _graph_data_with_positions()
    mg = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0, contract_router=None)
    mg.fit(data, task="classification", n_out=2, tiebreak=True)
    sp = (mg.route_detail or {}).get("tiebreak", {}).get("singular_pricing", None)
    assert sp is None, "singular pricing must not appear unless price_singular=True"

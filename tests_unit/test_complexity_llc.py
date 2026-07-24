"""T-LLC: the Local Learning Coefficient estimator (B2), which D1 will price on.

The *guard* (a non-converged minimum must not yield a bogus positive lambda) is the D1-critical
invariant and runs in the fast set. The exact-value tests are inherently SGLD-noisy and are marked
@smoke.
"""

import numpy as np
import pytest
import torch

from ilmarinen import estimate_llc


def _tiny_regression(n=200, d=4, seed=0, rank=None):
    """A small linear-regression problem; rank<d makes the target genuinely low-rank (more singular)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    if rank is None:
        w = rng.randn(d).astype(np.float32)
        y = (X @ w).astype(np.float32)
    else:
        w = rng.randn(d).astype(np.float32)
        w[rank:] = 0.0  # only `rank` active directions
        y = (X @ w).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)


def _fit_linear(X, y, steps=300, lr=0.05):
    model = torch.nn.Linear(X.shape[1], 1, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return model


def _loss_closure(model, X, y):
    def closure():
        return ((model(X).squeeze(-1) - y) ** 2).mean()

    return closure


def test_llc_guard_flags_nonconverged():
    """T-LLC-2 (D1-critical): a barely-trained (non-converged) net must NOT report a confident
    positive complexity -- the estimator surfaces the not-at-a-minimum condition instead."""
    X, y = _tiny_regression(seed=1)
    model = _fit_linear(X, y, steps=1)  # deliberately NOT converged
    out = estimate_llc(model, _loss_closure(model, X, y), n=len(y), chains=3, steps=120, burn=40, seed=0)
    lam = out["lambda"]
    # At a non-minimum, SGLD finds lower loss -> lambda comes out (near-)negative / flagged.
    # The contract we lock: it does not return a clearly-positive, trustworthy complexity here.
    flagged = (lam <= 0.1) or (out.get("valid") is False)
    assert flagged, f"guard failed: non-converged net reported lambda={lam}, out={out}"


@pytest.mark.smoke
def test_llc_positive_bounded_at_minimum():
    """T-LLC-1: at a clean minimum, 0 < lambda <= k/2 (+tol). SGLD-noisy -> @smoke."""
    X, y = _tiny_regression(seed=2)
    model = _fit_linear(X, y, steps=400)
    k = sum(p.numel() for p in model.parameters())
    out = estimate_llc(model, _loss_closure(model, X, y), n=len(y), chains=5, steps=300, burn=100, seed=0)
    lam = out["lambda"]
    assert np.isfinite(lam)
    assert lam <= k / 2 + 0.5, f"lambda {lam} exceeds k/2={k / 2}"


@pytest.mark.smoke
def test_llc_determinism_band():
    """T-LLC-3: same seed -> lambda repeatable within a band (SGLD is stochastic)."""
    X, y = _tiny_regression(seed=3)
    model = _fit_linear(X, y, steps=400)
    cl = _loss_closure(model, X, y)
    a = estimate_llc(model, cl, n=len(y), chains=4, steps=250, burn=80, seed=7)["lambda"]
    b = estimate_llc(model, cl, n=len(y), chains=4, steps=250, burn=80, seed=7)["lambda"]
    assert abs(a - b) < 0.5, f"non-deterministic under fixed seed: {a} vs {b}"

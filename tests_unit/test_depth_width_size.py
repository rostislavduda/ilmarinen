"""T-SZ: priced depth/size selection.

Uses fabricated depth curves (no training) to lock the pricing monotonicity and significance
gating: complexity that costs more (larger mu) must never select a larger model, and depth only
extends when the marginal gain clears the seed-noise threshold.
"""

import numpy as np

from ilmarinen import measure_depth_curve, select_depth


def _curve_from_losses(losses, seeds=(0, 1, 2)):
    """losses: dict depth->val_loss. Build a DepthCurve via the real measurement helper."""
    depths = sorted(losses)

    def train_eval(L, sd):
        rng = np.random.RandomState(sd)
        vl = losses[L] + 1e-4 * rng.randn()  # tiny seed noise
        return vl, 1.0 - vl  # (val_loss, val_acc)

    return measure_depth_curve(train_eval, depths, list(seeds))


def test_select_depth_monotone_in_mu():
    """T-SZ-1: raising mu (pricier complexity) never selects a deeper model."""
    curve = _curve_from_losses({1: 0.5, 2: 0.35, 3: 0.30, 4: 0.29})  # diminishing returns
    depths_by_mu = [select_depth(curve, mu=m) for m in [1e-4, 1e-2, 1e-1, 1.0, 10.0]]
    assert all(depths_by_mu[i] >= depths_by_mu[i + 1] for i in range(len(depths_by_mu) - 1)), depths_by_mu


def test_select_depth_cheap_complexity_goes_deep():
    """T-SZ-1b: near-free complexity selects (close to) the deepest useful depth."""
    curve = _curve_from_losses({1: 0.5, 2: 0.35, 3: 0.30, 4: 0.29})
    assert select_depth(curve, mu=1e-4) >= 3


def test_select_depth_flat_gains_stay_shallow():
    """T-SZ-2: if deeper models don't help (flat loss), don't extend depth."""
    curve = _curve_from_losses({1: 0.40, 2: 0.40, 3: 0.40, 4: 0.40})  # no marginal gain
    # with any non-trivial price, the shallowest depth is selected
    assert select_depth(curve, mu=1e-2) == 1


def test_measure_depth_curve_shape():
    """T-SZ-3: the curve has one loss per depth and finite marginals."""
    curve = _curve_from_losses({1: 0.5, 2: 0.4, 3: 0.35})
    # DepthCurve is a namedtuple(depths, S_mean, S_se, acc_mean, marginals)
    S = np.asarray(curve.S_mean)
    assert np.all(np.isfinite(S))
    assert len(S) == len(curve.depths) == 3

"""Priced spectral mode selection for the operator contract (direction B7).

Every other contract runs a degrees-of-freedom stage: width/depth are selected by a priced marginal-value rule
(machinery.priced_depth), so the model's capacity is an OUTPUT of a measurement, not a fixed hyperparameter.
The operator (neural-operator / FNO) contract was the one exception -- its "size" is the number of retained
Fourier modes, not a cell width, so the width/depth probe did not apply and the mode budget was set by a fixed
heuristic (min(12, grid/2)) and, worse, hardcoded per primitive. This module supplies the missing analogue:
the mode budget is a d.o.f. knob priced by the SAME marginal-value rule as depth.

The physics. A spectral conv keeps the lowest M Fourier modes with a learnable complex weight per mode, so its
parameter count grows LINEARLY in M per axis (M^d in d dimensions) -- the mode budget is the operator's
degrees of freedom. Retaining a mode is worth its price only while the validation-loss reduction it buys
exceeds mu times its added spectral code length. A band-limited target (most PDE solution operators are smooth
-> low-k) needs only modes up to its spectral bandwidth; modes beyond that add parameters and fit noise,
raising validation loss. So the marginal value of modes decays (and can go negative past the knee), exactly the
structure the depth rule already exploits. Verified on Burgers1D: the marginal reduction per mode is +5.4e-3
(1->2), +4.5e-5 (2->3), +1.4e-5 (3->4), then NEGATIVE (overfitting) at 6-8 modes -- optimal ~4, versus the
hardcoded 12.

This module measures S(M) = validation loss vs mode budget and selects M* by the marginal-value certificate,
with a spectral code length Omega(M) that grows with the retained-mode parameter count. It makes the operator
contract consistent with the physicist's ordering (kinematics -> d.o.f. -> dynamics): the mode budget joins width
and depth as a priced, measured quantity rather than a fixed number.
"""

from __future__ import annotations

import numpy as np


def spectral_code_length(modes, spatial_dims=1, channels=1):
    """Description-length proxy for a mode budget, in the SAME per-step units as the depth code length so a
    single mu governs all degrees of freedom. A spectral conv stores a complex weight per (in-ch, out-ch, kept
    mode), i.e. ~ channels^2 * modes^spatial_dims real parameters; the honest code length for MODEL SELECTION
    is the log of that parameter count (a description length is ~ log of the number of parameters, and it is
    the DIFFERENCE across mode budgets that prices a step). Using the raw parameter count would put the mode
    price on a channels^2-inflated scale incomparable to the depth price (whose per-layer code length is O(1));
    the log keeps the per-mode marginal code length O(1/M), matching depth's O(1)-per-layer footing. The
    channels factor cancels in the mode-to-mode DIFFERENCE, so it does not distort selection."""
    per_axis = float(max(modes, 1)) ** int(spatial_dims)
    n_params = float(channels * channels * per_axis)
    return float(np.log(max(n_params, 1.0)))


def measure_mode_curve(train_eval_fn, mode_grid, seeds):
    """train_eval_fn(modes, seed) -> val_loss. Averages over seeds. Returns (mode_grid, S_mean, S_se,
    marginals) where each marginal is (midpoint_modes, per-mode loss reduction, se), analogous to the depth
    curve. The per-mode reduction divides by the mode step so grids need not be uniform."""
    mode_grid = list(mode_grid)
    S_mean, S_se = [], []
    for M in mode_grid:
        vls = []
        for sd in seeds:
            vl = train_eval_fn(M, sd)
            if np.isfinite(vl):
                vls.append(vl)
        vls = np.array(vls)
        S_mean.append(vls.mean() if len(vls) else np.inf)
        S_se.append(vls.std(ddof=1) / np.sqrt(len(vls)) if len(vls) > 1 else 0.0)
    S_mean, S_se = np.array(S_mean), np.array(S_se)
    marginals = []
    for i in range(1, len(mode_grid)):
        dM = mode_grid[i] - mode_grid[i - 1]
        m = (S_mean[i - 1] - S_mean[i]) / dM
        me = np.sqrt(S_se[i - 1] ** 2 + S_se[i] ** 2) / dM
        marginals.append(((mode_grid[i - 1] + mode_grid[i]) / 2.0, m, me))
    return mode_grid, S_mean, S_se, marginals


def select_modes(mode_grid, S_mean, marginals, mu, spatial_dims=1, channels=1):
    """M* = the largest mode budget still worth its price: walk the grid upward and stop adding modes when the
    per-mode validation-loss reduction drops below mu times the per-mode ADDED spectral code length. This is
    the operator analogue of select_depth, with the price scaled by the (dimension-aware) spectral d.o.f. cost
    so a mode in 2D (which costs ~channels^2 * 2M per step) is held to a proportionally higher bar than in 1D.

    Returns (M*, detail). If every step clears the bar, M* is the deepest measured budget; if none do (even the
    first mode is not worth it, which should not happen for a nontrivial target), M* is the smallest budget.
    """
    if not marginals:
        return mode_grid[0], {"reason": "single mode budget measured", "selected_modes": mode_grid[0]}
    Mstar = mode_grid[0]
    ladder = []
    for (mid, m, _me) in marginals:
        lo = int(np.floor(mid - 0.5)); hi = int(np.ceil(mid + 0.5))
        # per-mode added code length going lo -> hi (average step cost)
        dOmega = (spectral_code_length(hi, spatial_dims, channels)
                  - spectral_code_length(lo, spatial_dims, channels)) / max(hi - lo, 1)
        bar = mu * dOmega
        pays = m >= bar
        ladder.append({"from_modes": lo, "to_modes": hi, "marginal": float(m),
                       "price_bar": float(bar), "pays": bool(pays)})
        if pays:
            Mstar = hi
        else:
            break                                  # stop before the first unprofitable mode
    detail = {"selected_modes": int(Mstar), "mu": float(mu), "spatial_dims": int(spatial_dims),
              "channels": int(channels), "ladder": ladder,
              "note": "mode budget via the marginal-value rule: add modes while the per-mode val-loss "
                      "reduction exceeds mu * (added spectral code length). The operator-contract analogue of "
                      "width/depth selection."}
    return int(Mstar), detail

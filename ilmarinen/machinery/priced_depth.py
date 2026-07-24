"""Priced-depth machinery (Route 2): marginal-value stopping rule.

Selects depth L* by the rule:  add layers while the per-layer marginal loss
reduction -dS*/dL exceeds a price mu; stop when it drops below.

Because S*(L) is a difference of independent training outcomes it is inherently
high-variance (unlike the width certificate). This module therefore measures
S*(L) as a multi-seed mean with standard error, and the stopping rule reads the
*denoised* marginal curve. Significance flags mark rungs whose marginal exceeds
2 standard errors (genuine signal vs. noise).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DepthCurve:
    depths: list
    S_mean: np.ndarray          # mean validation loss per depth
    S_se: np.ndarray            # standard error
    acc_mean: np.ndarray
    marginals: list = field(default_factory=list)  # (mid_depth, per_layer_marginal, marginal_se)


def measure_depth_curve(train_eval_fn, depths, seeds) -> DepthCurve:
    """train_eval_fn(depth, seed) -> (val_loss, val_acc).  Averages over seeds."""
    S_mean, S_se, acc_mean = [], [], []
    for L in depths:
        vls, vas = [], []
        for sd in seeds:
            vl, va = train_eval_fn(L, sd)
            if np.isfinite(vl):
                vls.append(vl); vas.append(va)
        vls, vas = np.array(vls), np.array(vas)
        S_mean.append(vls.mean())
        S_se.append(vls.std(ddof=1) / np.sqrt(len(vls)) if len(vls) > 1 else 0.0)
        acc_mean.append(vas.mean())
    S_mean, S_se, acc_mean = map(np.array, (S_mean, S_se, acc_mean))

    marginals = []
    for i in range(1, len(depths)):
        dL = depths[i] - depths[i - 1]
        m = (S_mean[i - 1] - S_mean[i]) / dL
        me = np.sqrt(S_se[i - 1] ** 2 + S_se[i] ** 2) / dL
        marginals.append(((depths[i - 1] + depths[i]) / 2, m, me))
    return DepthCurve(list(depths), S_mean, S_se, acc_mean, marginals)


def select_depth(curve: DepthCurve, mu: float) -> int:
    """L* = the depth AFTER which the next layer is not worth its price mu.

    Each marginal m at midpoint mid = (L, L+1)/2 is the per-layer loss reduction from going L -> L+1.
    We add layer L+1 only while its marginal exceeds mu. So L* is the SHALLOWER endpoint (floor(mid))
    of the first marginal that drops below mu -- i.e. we stop BEFORE paying for the unprofitable layer.
    If every marginal beats mu, take the deepest measured depth.
    """
    for (mid, m, _me) in curve.marginals:
        if m < mu:
            return int(np.floor(mid))
    return curve.depths[-1]


def predict_depth_scaling(alpha: float, mu_ref: float, Lstar_ref: int):
    """RG-theory prediction  L*(mu) ~ mu^{-1/(alpha+1)}.

    Given the critical exponent alpha (from core.exponent) and ONE measured
    (mu_ref, Lstar_ref) anchor point, return a function mu -> predicted L*.
    This wires the depth-RG theory into the depth machinery: instead of
    searching the whole mu grid empirically, the exponent predicts how L*
    scales with the price, so a single measured point extrapolates the curve.

    The exponent enters as the power -1/(alpha+1):
        smooth (alpha=1): L* ~ mu^{-1/2}
        ReLU   (alpha=2): L* ~ mu^{-1/3}
    """
    power = -1.0 / (alpha + 1.0)
    C = Lstar_ref / (mu_ref ** power)      # calibrate the constant from the anchor

    def L_of_mu(mu: float) -> float:
        return C * (mu ** power)

    return L_of_mu, power


def compare_predicted_vs_measured(curve: DepthCurve, alpha: float, prices):
    """Measure L*(mu) empirically and compare to the exponent prediction.

    Returns a list of (mu, measured_Lstar, predicted_Lstar). The anchor for the
    prediction is the median-price measured point, so the test is whether the
    exponent correctly captures the SHAPE of L*(mu), not just one point.
    """
    prices = sorted(prices)
    measured = [(mu, select_depth(curve, mu)) for mu in prices]
    # anchor at the median price
    mid = len(measured) // 2
    mu_ref, L_ref = measured[mid]
    L_of_mu, power = predict_depth_scaling(alpha, mu_ref, L_ref)
    rows = [(mu, Lm, L_of_mu(mu)) for (mu, Lm) in measured]
    return rows, power


def significant_elbow(curve: DepthCurve, n_se: float = 2.0) -> int:
    """Depth at which the marginal value stops being significantly > 0.

    A model-selection answer independent of any chosen price: the last depth
    whose marginal exceeds n_se standard errors is where added depth stops
    demonstrably paying.
    """
    elbow = curve.depths[0]
    for (mid, m, me) in curve.marginals:
        if m - n_se * me > 0:
            elbow = int(np.ceil(mid))
    return elbow

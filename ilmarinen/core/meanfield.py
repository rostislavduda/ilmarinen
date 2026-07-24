"""Mean-field signal-propagation theory (the depth-RG core).

Implements the layer-to-layer correlation map for a pointwise activation with
Gaussian weights, computed by Gauss-Hermite quadrature over the fixed-point
preactivation distribution. Provides:

  - chi_1 = R'(1) = sigma_w^2 * E[sigma'(z)^2]   (the Lyapunov multiplier)
  - the length fixed point q*
  - the correlation fixed point c*
  - phase classification (ordered / critical / chaotic)
  - critical-point location by bisection of chi_1 = 1

This is the corrected diagnostic: phase is decided by chi_1 and c* directly,
never by a decay-rate fit (which conflates 'approaching 1' with 'fleeing 1').
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Gauss-Hermite nodes/weights: E_{z~N(0,q)}[f] = (1/sqrt(pi)) sum_i w_i f(sqrt(2q) x_i)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(80)


def _E_gauss(f, q: float) -> float:
    z = np.sqrt(2.0 * q) * _GH_X
    return float(np.sum(_GH_W * f(z)) / np.sqrt(np.pi))


# ---- built-in activations with their derivatives (extend as needed) ----
ACTIVATIONS = {
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "erf":  (lambda z: np.vectorize(__import__("math").erf)(z),
             lambda z: 2.0 / np.sqrt(np.pi) * np.exp(-z ** 2)),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(float)),
}


@dataclass
class PhaseResult:
    sigma_w2: float
    sigma_b2: float
    chi1: float
    q_star: float
    c_star: float
    xi: float
    phase: str


class MeanFieldTheory:
    """Signal-propagation analysis for a given pointwise activation."""

    def __init__(self, activation: str = "tanh"):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; have {list(ACTIVATIONS)}")
        self.name = activation
        self.sigma, self.sigma_prime = ACTIVATIONS[activation]

    def length_fixed_point(self, sigma_w2: float, sigma_b2: float, iters: int = 200) -> float:
        q = 1.0
        for _ in range(iters):
            q = sigma_w2 * _E_gauss(lambda z: self.sigma(z) ** 2, q) + sigma_b2
        return q

    def chi1(self, sigma_w2: float, sigma_b2: float) -> tuple[float, float]:
        q = self.length_fixed_point(sigma_w2, sigma_b2)
        val = sigma_w2 * _E_gauss(lambda z: self.sigma_prime(z) ** 2, q)
        return val, q

    def correlation_step(self, c: float, sigma_w2: float, sigma_b2: float, q: float) -> float:
        x, w = _GH_X, _GH_W
        Z1 = np.sqrt(2 * q) * x[:, None]
        Z2 = np.sqrt(2 * q) * (c * x[:, None] + np.sqrt(max(1 - c * c, 0.0)) * x[None, :])
        W2 = w[:, None] * w[None, :]
        integ = np.sum(W2 * self.sigma(Z1) * self.sigma(Z2)) / np.pi
        q12 = sigma_w2 * integ + sigma_b2
        return q12 / q

    def correlation_fixed_point(self, sigma_w2, sigma_b2, q, c0=0.6, iters=300) -> float:
        c = c0
        for _ in range(iters):
            c = float(np.clip(self.correlation_step(c, sigma_w2, sigma_b2, q), -0.999, 0.999))
        return c

    def classify(self, sigma_w2: float, sigma_b2: float = 0.05, tol: float = 0.03) -> PhaseResult:
        x1, q = self.chi1(sigma_w2, sigma_b2)
        cstar = self.correlation_fixed_point(sigma_w2, sigma_b2, q)
        if abs(x1 - 1.0) < 1e-6:
            xi = np.inf
        elif x1 < 1.0:
            xi = -1.0 / np.log(x1)
        else:
            xi = np.nan
        if abs(x1 - 1.0) < tol:
            phase = "critical"
        elif x1 < 1.0:
            phase = "ordered"
        else:
            phase = "chaotic"
        return PhaseResult(sigma_w2, sigma_b2, x1, q, cstar, xi, phase)

    def critical_sigma_w2(self, sigma_b2: float = 0.05, lo: float = 0.5, hi: float = 5.0,
                          iters: int = 40) -> float:
        """Locate sigma_w^2 such that chi_1 = 1 by bisection."""
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            x1, _ = self.chi1(mid, sigma_b2)
            if x1 < 1.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

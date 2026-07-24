"""Critical exponent alpha from the mean-field correlation map.

At the critical fixed point (chi_1 = 1) the correlation map expands as
    R(1 - eps) = 1 - eps + kappa * eps^p + O(eps^{p+1}),
and the depth decay of (1 - c) is  eps^(l) ~ l^{-alpha}  with

    alpha = 1 / (p - 1).

  smooth activations: p = 2   -> alpha = 1   (eps ~ l^{-1})
  ReLU (kink):        p = 3/2 -> alpha = 2   (eps ~ l^{-2})

We extract p WITHOUT differentiating R twice (which fails for ReLU, whose
eps^{3/2} term has no finite second derivative). Instead we isolate the leading
nonlinear part by subtracting the marginal linear term:

    g(eps) := (1 - eps) - R(1 - eps)  =  -kappa * eps^p + ...

and read p as the log-log slope of |g(eps)| vs eps as eps -> 0. This is
well-conditioned and works for fractional p.

Cost: a handful of Gauss-Hermite quadrature evaluations of R -- milliseconds,
no training, no data. alpha is a pure property of the activation + init.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.meanfield import MeanFieldTheory


@dataclass
class ExponentResult:
    activation: str
    sigma_w2: float           # the (critical) init variance used
    chi1: float               # should be ~1 at the critical point
    p: float                  # leading nonlinear power of R near c=1
    kappa: float              # its coefficient (sign/magnitude)
    alpha: float              # alpha = 1/(p-1): depth decay (1-c) ~ l^{-alpha}
    r2: float                 # log-log fit quality (closeness to a clean power law)
    reliable: bool            # True only if the power-law fit is clean (R2 high)
    method: str               # which extraction path produced the result
    eps_grid: np.ndarray
    g_vals: np.ndarray


def critical_exponent(theory: MeanFieldTheory, sigma_b2: float = 0.05,
                      sigma_w2: float | None = None,
                      eps_lo: float = 1e-4, eps_hi: float = 1e-1,
                      n_pts: int = 25) -> ExponentResult:
    """Compute alpha from the leading nonlinearity of R near c=1.

    If sigma_w2 is None, use the critical value (chi_1 = 1) located by bisection.
    """
    if sigma_w2 is None:
        sigma_w2 = theory.critical_sigma_w2(sigma_b2)
    chi1, q = theory.chi1(sigma_w2, sigma_b2)

    # At criticality R'(1) = chi_1 = 1 exactly, so subtracting the marginal
    # linear part is simply  g(eps) = (1 - eps) - R(1-eps) ~ -kappa eps^p.
    # This is exact and clean for SMOOTH activations (Gauss-Hermite resolves R
    # to machine precision). It is UNRELIABLE for ReLU, whose eps^{3/2} term sits
    # on a sqrt(1-c^2) integrand that plain Gauss-Hermite resolves poorly near
    # c=1; we detect that via the fit quality (low R2) and flag it, routing the
    # caller to empirical_exponent() instead of returning a confident wrong p.
    eps = np.geomspace(eps_lo, eps_hi, n_pts)
    g = np.empty_like(eps)
    for i, e in enumerate(eps):
        g[i] = (1.0 - e) - theory.correlation_step(1.0 - e, sigma_w2, sigma_b2, q)

    mask = eps <= np.sqrt(eps_lo * eps_hi)
    if mask.sum() < 5:
        mask = eps <= eps[len(eps) // 2]
    absg = np.abs(g[mask])
    good = absg > 1e-14
    if good.sum() < 5:
        return ExponentResult(theory.name, sigma_w2, chi1, np.nan, np.nan,
                              np.nan, 0.0, False, "degenerate", eps, g)
    x = np.log(eps[mask][good])
    ly = np.log(absg[good])
    slope, intercept = np.polyfit(x, ly, 1)
    p = float(slope)
    pred = slope * x + intercept
    r2 = float(1 - np.sum((ly - pred) ** 2) / (np.sum((ly - ly.mean()) ** 2) + 1e-30))
    kappa = float(-np.exp(intercept) * np.sign(g[mask][good][0]))

    reliable = (r2 > 0.99) and (p > 1.0 + 1e-3)
    alpha = float(1.0 / (p - 1.0)) if (p - 1.0) > 1e-6 else np.inf
    method = "map-expansion" if reliable else "map-expansion (UNRELIABLE: use empirical)"
    return ExponentResult(theory.name, sigma_w2, chi1, p, kappa, alpha, r2,
                          reliable, method, eps, g)


def empirical_exponent(theory_name: str, sigma_w2: float, sigma_b2: float = 0.05,
                       depth: int = 200, width: int = 2000, c0: float = 0.7,
                       seed: int = 0):
    """Cross-check: measure alpha from an actual deep random-net trajectory.

    Regress log(1 - c^(l)) on log(l) over the algebraic-decay window.
    One forward pass; ~seconds.
    """
    from ..core.meanfield import ACTIVATIONS
    rng = np.random.default_rng(seed)
    sigma, _ = ACTIVATIONS[theory_name]
    d = 200
    x1 = rng.standard_normal(d)
    x2 = c0 * x1 + np.sqrt(1 - c0 ** 2) * rng.standard_normal(d)
    x1 /= np.linalg.norm(x1) / np.sqrt(d)
    x2 /= np.linalg.norm(x2) / np.sqrt(d)
    h1, h2 = x1, x2
    nin = d
    eps_traj = []
    for _ in range(depth):
        W = rng.standard_normal((width, nin)) * np.sqrt(sigma_w2 / nin)
        b = rng.standard_normal(width) * np.sqrt(sigma_b2)
        h1 = sigma(W @ h1 + b)
        h2 = sigma(W @ h2 + b)
        nin = width
        c = np.mean(h1 * h2) / np.sqrt(np.mean(h1 ** 2) * np.mean(h2 ** 2) + 1e-12)
        eps_traj.append(1.0 - c)
    eps_traj = np.array(eps_traj)
    # fit on a mid-depth window where algebraic decay is clean and eps>0
    ls = np.arange(10, min(depth, 120))
    e = eps_traj[ls]
    good = e > 1e-9
    if good.sum() < 5:
        return np.nan
    slope = np.polyfit(np.log(ls[good]), np.log(e[good]), 1)[0]
    return float(-slope)   # eps ~ l^{-alpha}  ->  slope = -alpha

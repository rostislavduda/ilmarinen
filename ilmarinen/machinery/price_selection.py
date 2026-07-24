"""Automating mu, the complexity price in J = R + mu*Omega.

Is mu a fundamental limitation? Analytically: mu is the exchange rate between accuracy and complexity
in a SCALARIZED multi-objective problem (minimize R AND Omega). Sweeping mu traces the PARETO FRONTIER;
no single point is "the" answer without an external preference. So in its BARE form -- "trade accuracy
against complexity, unspecified how much" -- mu genuinely CANNOT be derived from data: it encodes a
user value, not a fact. This matches the analytical notes' "there is no single correct architecture
size, only the frontier."

BUT the moment the user states WHAT they actually want, mu becomes DERIVABLE by a 1-D determination:

  (B) CONSTRAINT-DRIVEN (dual variable). Goal: best accuracy SUBJECT TO a budget (<= B on Omega, e.g.
      params/FLOPs/latency). Then mu is the LAGRANGE MULTIPLIER of the budget: raise mu until the
      constraint is just active. mu*(B) = the value where Omega(mu) = B. Automated by bisection on the
      frontier. -> select_mu_for_budget.

  (C) CRITERION-DRIVEN (elbow / tolerance). Goal: "the most compact model that loses no meaningful
      accuracy." Pick a distinguished frontier point with NO external price:
        - KNEE/ELBOW: the max-curvature point of accuracy-vs-Omega (point of diminishing returns).
        - TOLERANCE: cheapest model within eps accuracy of the best.
      Both encode a weak, near-universal preference ("don't waste capacity"); eps/knee replaces mu
      with a far more interpretable knob. -> select_mu_by_elbow, select_by_tolerance.

  (D) STATISTICAL (generalization-optimal). Goal: best HELD-OUT accuracy. Then complexity control is
      regularization and mu is set to MINIMIZE VALIDATION LOSS (1-D search) -- the bias-variance
      optimum; MDL gives the price of one bit. -> select_mu_by_validation.

CONCLUSION: mu is "fundamental" only while the objective is an unanchored accuracy-vs-complexity
tradeoff. Given a budget (B), a no-waste tolerance (C), or a generalization goal (D) -- each a concrete,
interpretable spec -- mu is automated. The irreducible residue is only WHICH objective the user wants,
which is genuinely a preference, not derivable from data. So mu is not a fundamental limitation; it is a
preference that standard 1-D determinations resolve once made concrete.
"""
from __future__ import annotations

import numpy as np


def select_mu_for_budget(fit_fn, budget, mu_lo=1e-4, mu_hi=10.0, iters=12):
    """(B) Dual-variable mu for a complexity BUDGET. fit_fn(mu) -> (accuracy, omega). Bisection on mu
    to make Omega just <= budget (Omega is monotone decreasing in mu). Returns (mu, accuracy, omega).
    mu is the Lagrange multiplier of the budget constraint -- DETERMINED by the budget, not free."""
    lo, hi = mu_lo, mu_hi
    best = None
    for _ in range(iters):
        mu = np.sqrt(lo * hi)                      # geometric bisection (mu spans orders of magnitude)
        acc, omega = fit_fn(mu)
        best = (mu, acc, omega)
        if omega > budget:
            lo = mu                                # over budget -> need higher price
        else:
            hi = mu                                # within budget -> can afford lower price
    return best


def select_by_tolerance(frontier, acc_tol=0.01):
    """(C) Tolerance rule: cheapest model within acc_tol of the best accuracy. frontier is a list of
    dicts with 'accuracy' and 'omega' (or 'cost'). Returns the chosen entry. mu-free frontier selector
    -- the mu->0+ limit restricted to a no-harm accuracy band."""
    accs = np.array([e["accuracy"] for e in frontier])
    cost = np.array([e.get("omega", e.get("cost")) for e in frontier])
    best = accs.max()
    ok = np.where(accs >= best - acc_tol)[0]
    return frontier[int(ok[np.argmin(cost[ok])])]


def select_mu_by_elbow(frontier):
    """(C) Knee/elbow of the accuracy-vs-Omega frontier: the point of maximum curvature (diminishing
    returns). Uses the standard 'largest distance to the chord' (Kneedle-style) criterion on the
    (omega, accuracy) curve sorted by omega. Returns the chosen entry -- a distinguished frontier point
    requiring NO external price, just the shape of the frontier."""
    pts = sorted(frontier, key=lambda e: e.get("omega", e.get("cost")))
    x = np.array([e.get("omega", e.get("cost")) for e in pts], dtype=float)
    y = np.array([e["accuracy"] for e in pts], dtype=float)
    if len(pts) < 3:
        return pts[-1]
    # normalize to [0,1], measure perpendicular distance from each point to the chord (first->last)
    xn = (x - x.min()) / (np.ptp(x) + 1e-12)
    yn = (y - y.min()) / (np.ptp(y) + 1e-12)
    x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
    num = np.abs((y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0) + 1e-12
    dist = num / den
    return pts[int(np.argmax(dist))]


def select_mu_by_validation(fit_fn, mus):
    """(D) Generalization-optimal mu: the mu minimizing held-out VALIDATION loss (not train). fit_fn(mu)
    -> (val_loss, accuracy, omega). Returns (best_mu, entry). Complexity control AS regularization: the
    bias-variance optimum, fully automated by a 1-D search over mus."""
    results = []
    for mu in mus:
        vloss, acc, omega = fit_fn(mu)
        results.append({"mu": mu, "val_loss": vloss, "accuracy": acc, "omega": omega})
    best = min(results, key=lambda r: r["val_loss"])
    return best["mu"], best

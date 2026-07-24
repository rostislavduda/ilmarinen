"""Deriving alpha: the architecture parameter as the Gibbs stationary measure of a free-energy flow on
the primitive simplex (closes the standing theoretical seam -- Future Direction, "the standing
theoretical seam").

THE DERIVATION (established analytically + verified numerically to 1e-15; see tests/gibbs_alpha_*.md).
Put on the primitive simplex the SAME free energy the analytical report uses for the neuron measure,
    F_alpha(alpha) = <alpha, Psi> - (1/beta) H(alpha),      H = Shannon entropy,
with Psi_i the MARGINAL (solo) risk / description-length cost of primitive i and beta the SAME MDL
inverse temperature as F = R + (1/beta) KL. Then:

  (1) FORM.      The interior stationary point of F_alpha is the Gibbs law
                    alpha_i  ∝  exp(-beta Psi_i),
                 exactly the outer analogue of the report's neuron Gibbs law drho/dpi ∝ exp(-beta Psi).
  (2) DYNAMICS.  The Fisher-Rao / Shahshahani natural-gradient flow of F_alpha is the REPLICATOR
                 equation  dalpha_i/dt = -[J grad F_alpha]_i,  J = diag(alpha) - alpha alpha^T,
                 i.e. selection dynamics with fitness f_i = -Psi_i. It converges to the Gibbs law (1).
  (3) DARTS.     Softmax-logit gradient descent is dalpha/dt = -J^2 grad F_alpha = the replicator field
                 pushed once more through the PSD map J. Since ker J = normal-to-simplex, DARTS shares
                 the replicator's ENTIRE critical set and interior Gibbs equilibrium -- it is the same
                 flow in a mildly preconditioned metric (for N=2, exactly a positive time-rescale).
  (4) ONE KNOB.  beta -> infinity concentrates alpha on argmin_i Psi_i (the hard selection, the
                 primitive analogue of the width dual certificate); finite beta is the smooth mixture.

So alpha is NOT a bolt-on: it is the Gibbs stationary measure / replicator equilibrium of the same
free energy that governs weights and width, and DARTS is recovered as its (preconditioned) gradient
flow. The three views -- Gibbs form, replicator dynamics, hard certificate -- are one object at three
temperatures.

THE ENERGY IS A SOLO QUANTITY. The one physical choice the derivation MAKES is Psi_i: the free energy
forces it to be the MARGINAL risk of primitive i (its solo description length), NOT the mixture
gradient dR/dalpha_i that plain DARTS uses. That mixture gradient is exactly what carries the
co-adaptation contamination. So the derived selection is intrinsically solo -- and is predicted to fix
the co-adaptation failure that argmax-DARTS gets wrong. This module therefore consumes the same
clean-solo scores that robust_select already computes.

This module leaves DARTS and every validated schema UNTOUCHED; it is a separate, principled
selector that reads clean-solo energies and returns the derived Gibbs-alpha.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- core objects
def gibbs_alpha(energies, beta):
    """The derived architecture distribution: alpha_i ∝ exp(-beta * Psi_i).

    energies : 1-D array of marginal (solo) risks Psi_i (LOWER = better primitive).
    beta     : MDL inverse temperature. beta->0 uniform; beta->inf one-hot on argmin Psi.
    Returns the Gibbs probability vector over primitives.
    """
    Psi = np.asarray(energies, float)
    z = -beta * Psi
    z -= z.max()  # stable softmax
    w = np.exp(z)
    return w / w.sum()


def replicator_flow(energies, beta, dt=0.01, steps=20000, alpha0=None, tol=1e-12):
    """Integrate the Fisher-Rao (replicator) gradient flow of F_alpha = <alpha,Psi> - (1/beta)H(alpha)
    to its Gibbs fixed point. Provided to demonstrate the DYNAMICS (2) converge to the FORM (1); the
    closed form gibbs_alpha() is what callers normally use. Returns the converged alpha.
    """
    Psi = np.asarray(energies, float)
    n = len(Psi)
    alpha = np.ones(n) / n if alpha0 is None else np.asarray(alpha0, float).copy()
    for _ in range(steps):
        grad = Psi + (1.0 / beta) * (np.log(np.clip(alpha, tol, None)) + 1.0)  # grad_alpha F_alpha
        J = np.diag(alpha) - np.outer(alpha, alpha)  # Fisher-Rao map
        alpha = alpha - dt * (J @ grad)
        alpha = np.clip(alpha, tol, None)
        alpha /= alpha.sum()
    return alpha


# --------------------------------------------------------------------------- the selector
def gibbs_alpha_select(build_and_train_solo, primitives, beta=8.0, return_flow=False):
    """Derived primitive selection: the Gibbs-alpha of the free energy on the primitive simplex, using
    clean-solo marginal risks as the energy Psi_i (the quantity the derivation forces).

    build_and_train_solo(prim) -> float SCORE (higher = better, e.g. R^2 or accuracy) -- the SAME
        interface as clean_solo_select, so callers reuse their existing solo trainer verbatim.
    primitives : iterable of primitive names.
    beta       : MDL inverse temperature (the one knob). Larger beta = sharper selection; beta->inf is
                 the hard argmin-energy limit (== clean_solo_select's argmax).
    return_flow: if True, also return the replicator-flow alpha (to confirm dynamics==closed form).

    Returns dict:
        best        : argmax primitive of the Gibbs measure (== clean-solo argmax for any beta>0)
        alpha       : {primitive: Gibbs probability}
        energies    : {primitive: Psi_i}  (Psi_i = -score_i, the marginal risk)
        scores      : {primitive: solo score}  (higher=better, as returned by the trainer)
        beta        : the temperature used
        [flow_alpha]: {primitive: replicator-flow probability}  if return_flow

    The derivation guarantees best == argmin_i Psi_i == argmax_i score_i, i.e. this AGREES with
    clean_solo_select's winner while additionally producing the full principled distribution and its
    temperature dependence. The value over argmax-DARTS is that the energy is a SOLO quantity, so the
    winner is the co-adaptation-robust one (validated separately on QM7/pna).
    """
    prims = list(primitives)
    scores = {p: float(build_and_train_solo(p)) for p in prims}
    # energy = marginal risk = -score (lower risk <=> higher score <=> better primitive)
    Psi = np.array([-scores[p] for p in prims], float)
    a = gibbs_alpha(Psi, beta)
    alpha = {p: float(a[i]) for i, p in enumerate(prims)}
    best = prims[int(np.argmin(Psi))]  # == argmax score == argmax Gibbs prob
    out = {
        "best": best,
        "alpha": alpha,
        "energies": {p: float(Psi[i]) for i, p in enumerate(prims)},
        "scores": scores,
        "beta": beta,
    }
    if return_flow:
        af = replicator_flow(Psi, beta)
        out["flow_alpha"] = {p: float(af[i]) for i, p in enumerate(prims)}
    return out


def gibbs_frontier(energies, betas):
    """Sweep the MDL inverse temperature beta and return the Gibbs-alpha at each -- the primitive-simplex
    analogue of the width/depth fit-complexity frontier. As beta grows the measure sharpens from uniform
    (all primitives, maximum entropy) to one-hot (the single least-energy primitive, the certificate
    limit). energies is a 1-D array of Psi_i; returns array [len(betas), n_primitives].

    THE FRONTIER (Step 4, beta as one MDL knob). Rewrite the free energy in the SAME priced form as the
    width/depth objective J = R + mu*Omega:
        F_alpha = <alpha, Psi>  +  (1/beta) * (-H(alpha)),
        FIT        = <alpha, Psi>     (expected risk of the primitive mixture),
        COMPLEXITY = -H(alpha)        (negative entropy: how CONCENTRATED / committed the choice is),
        PRICE      = 1/beta           (the exchange rate).
    So 1/beta is the price of committing to few primitives, exactly as mu is the price of capacity:
    high beta (cheap to commit) => sharp alpha / simple committed architecture (== high mu, compact);
    low beta (expensive to commit) => broad alpha / rich hedged mixture (== low mu, buy accuracy with
    capacity). Sweeping beta traces the fit-vs-commitment Pareto frontier; each beta is one point.
    """
    Psi = np.asarray(energies, float)
    return np.stack([gibbs_alpha(Psi, b) for b in betas], axis=0)


def frontier_curve(energies, betas):
    """The (complexity, fit) points of the beta-frontier: for each beta, return (-H(alpha), <alpha,Psi>).
    Analogue of the width/depth (Omega, R) frontier. Returns (neg_entropy[betas], fit[betas])."""
    Psi = np.asarray(energies, float)
    A = gibbs_frontier(Psi, betas)
    fit = A @ Psi
    negH = np.array([(a * np.log(a + 1e-12)).sum() for a in A])  # = -H
    return negH, fit


def select_beta_by_elbow(energies, betas=None):
    """Choose beta at the KNEE of the fit-vs-commitment frontier -- the beta past which committing
    further (lower entropy) stops buying meaningful fit improvement. The primitive-simplex analogue of
    select_mu_by_elbow: no external price, just "don't hedge more than the fit gain justifies." Returns
    (beta_star, alpha_star). Knee = max-distance point from the chord of the (fit vs -H) curve.
    """
    Psi = np.asarray(energies, float)
    if betas is None:
        betas = np.geomspace(0.3, 120.0, 40)
    betas = np.asarray(betas, float)
    negH, fit = frontier_curve(Psi, betas)
    # normalize both axes to [0,1], measure perpendicular distance to the chord end-to-end
    x = (negH - negH.min()) / (np.ptp(negH) + 1e-12)
    yv = (fit - fit.min()) / (np.ptp(fit) + 1e-12)
    x0, y0, x1, y1 = x[0], yv[0], x[-1], yv[-1]
    num = np.abs((y1 - y0) * x - (x1 - x0) * yv + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0) + 1e-12
    k = int(np.argmax(num / den))
    return float(betas[k]), gibbs_alpha(Psi, betas[k])

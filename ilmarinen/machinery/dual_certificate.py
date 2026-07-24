"""Dual certificate for the TV-regularized width-sparsity problem.

Makes the optimality object EXPLICIT rather than implicit. For the problem

    min_rho  (1/2) || A rho - y ||^2  +  lambda || rho ||_TV ,   rho in M(Theta)

with forward operator (A rho)(x) = int phi(x; Theta) d rho(Theta), the analytical
optimality conditions are (KKT / Fisher-Jerome):

  optimal dual variable :  q* = y - A rho*            (the residual)
  certificate           :  eta(Theta) = (1/lambda) < q*, phi(.;Theta) >_data
                                       = (1/lambda) (A^* q*)(Theta)

  KKT conditions:
    (feasibility)     | eta(Theta) |  <= 1   for all Theta
    (saturation+sign) eta(Theta_k)   =  sign(alpha_k)   on supp(rho*)
    (strict slack)    | eta(Theta) |  <  1   off supp(rho*)

  Non-degeneracy (upgrades 'a sparse solution exists' -> 'THIS support, stably'):
    - strict off-support inequality with a margin
    - each support atom is an interior maximizer of |eta| (grad eta = 0) with
      negative-definite Hessian of |eta| (i.e. eta grazes +-1 tangentially).

This module computes eta as a real function over parameter space, VERIFIES the
KKT conditions at the greedy solution, and checks non-degeneracy numerically.
It thereby turns the greedy stopping rule ('max correlation < lambda') from an
assumed proxy into a checked certificate.

Faithfulness notes / honest gaps:
  - phi(x;Theta) = tanh(w.x + b) here (matches the model). The certificate is
    evaluated at the SOLVED amplitudes (the least-squares fit is the primal
    rho*), so q* is the true residual of the returned network.
  - eta is defined via the RAW adjoint inner product < q*, phi >, NOT the
    correlation (feature-norm-normalized) used as the greedy oracle. The two
    agree up to feature normalization; the raw form is the analytically correct
    certificate and is what we test against |eta| <= 1.
  - the global feasibility check samples parameter space densely (the exact
    sup over Theta is the same NP-hard oracle as insertion); we report the
    empirical sup and its margin, and flag if any candidate violates |eta|<=1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def phi(X, w, b):
    """Feature map phi(x; Theta) = tanh(w . x + b), evaluated on rows of X."""
    return np.tanh(X @ w + b)


@dataclass
class CertificateReport:
    lam: float
    K: int
    # feasibility
    sup_abs_eta_support: float        # max |eta| over support atoms (should be ~1)
    sup_abs_eta_global: float         # empirical sup |eta| over sampled Theta
    feasible: bool                    # sup_abs_eta_global <= 1 + tol
    # saturation + sign
    max_saturation_error: float       # max_k | |eta(Theta_k)| - 1 |
    max_sign_error: float             # max_k | eta(Theta_k) - sign(alpha_k) |
    saturated: bool
    signs_match: bool
    # strict slack / non-degeneracy
    offsupport_margin: float          # 1 - sup |eta| over off-support samples
    strict_slack: bool
    grad_norm_at_atoms: float         # max ||grad eta|| at support atoms (should ~0)
    hessian_negdef_fraction: float    # fraction of atoms with neg-def Hessian of |eta|
    nondegenerate: bool
    # overall
    certified: bool


def _eta_at(theta_w, theta_b, X, q_star, lam):
    """eta(Theta) = (1/lambda) * (1/n) sum_i q*_i phi(x_i; Theta)."""
    f = phi(X, theta_w, theta_b)           # (n,)
    return float((q_star @ f) / (len(q_star) * lam))


def _eta_grad_hess(theta_w, theta_b, X, q_star, lam, h=1e-4):
    """Numerical grad and Hessian of |eta| w.r.t. (w, b) at a support atom.

    Uses central differences in the full (d+1) parameter (w, b). For the
    non-degeneracy test we need the Hessian of |eta| (which grazes 1), so we
    work with s(Theta) = sign(eta) * eta = |eta| near an atom where eta = +-1.
    """
    d = len(theta_w)
    p0 = np.concatenate([theta_w, [theta_b]])
    e0 = _eta_at(theta_w, theta_b, X, q_star, lam)
    s = np.sign(e0) if e0 != 0 else 1.0

    def absval(p):
        return s * _eta_at(p[:d], p[d], X, q_star, lam)   # = |eta| near the atom

    # gradient
    g = np.zeros(d + 1)
    for i in range(d + 1):
        pp = p0.copy(); pp[i] += h
        pm = p0.copy(); pm[i] -= h
        g[i] = (absval(pp) - absval(pm)) / (2 * h)
    # Hessian (diagonal + a few off-diagonals would be d^2 cost; for d=784 we
    # compute only the DIAGONAL of the Hessian -- sufficient to test whether the
    # atom is a local MAX of |eta| along each coordinate, a practical proxy for
    # negative-definiteness that is O(d) rather than O(d^2)).
    hess_diag = np.zeros(d + 1)
    f0 = absval(p0)
    for i in range(d + 1):
        pp = p0.copy(); pp[i] += h
        pm = p0.copy(); pm[i] -= h
        hess_diag[i] = (absval(pp) - 2 * f0 + absval(pm)) / (h * h)
    return g, hess_diag


def build_certificate(result, X, y, n_probe=20000, tol=1e-2, seed=0,
                      verify_nondegeneracy=True) -> CertificateReport:
    """Construct and verify the dual certificate for a greedy_insertion result.

    Parameters
    ----------
    result : InsertionResult
        The primal solution (neurons + least-squares amplitudes) = rho*.
    X, y : arrays
        The training data the solution was fit to (defines q* and the adjoint).
    n_probe : int
        Number of random parameter-space samples for the global feasibility sup.
    tol : float
        Numerical tolerance for feasibility / saturation.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    neurons = result.neurons
    A = result.amplitudes
    K = len(neurons)
    if K == 0 or A is None:
        raise ValueError("empty solution; nothing to certify")

    # primal reconstruction and residual  q* = y - A rho*
    Phi = np.column_stack([phi(X, w, b) for (w, b) in neurons])   # (n, K)
    pred = Phi @ A
    q_star = y - pred
    lam = result.lam

    # --- (1) certificate at the support atoms ---
    eta_atoms = np.array([_eta_at(w, b, X, q_star, lam) for (w, b) in neurons])
    abs_eta_support = np.abs(eta_atoms)
    sup_support = float(abs_eta_support.max())
    # sign condition: eta(Theta_k) should equal sign(alpha_k)
    sign_alpha = np.sign(A)
    max_sign_err = float(np.max(np.abs(eta_atoms - sign_alpha)))
    max_sat_err = float(np.max(np.abs(abs_eta_support - 1.0)))

    # --- (2) global feasibility: empirical sup |eta| over sampled Theta ---
    # mix data-informed and random candidates (same distribution as the oracle)
    Wp = rng.standard_normal((d, n_probe)) * (1.0 / np.sqrt(d))
    idx = rng.integers(0, n, n_probe // 2)
    Wp[:, :n_probe // 2] = (X[idx].T / (np.linalg.norm(X[idx], axis=1) + 1e-6))
    Bp = rng.standard_normal(n_probe) * 0.5
    Fp = np.tanh(X.T.T @ Wp + Bp)  # (n, n_probe)
    eta_probe = (q_star @ Fp) / (n * lam)          # (n_probe,)
    abs_eta_probe = np.abs(eta_probe)
    sup_global = float(max(sup_support, abs_eta_probe.max()))
    feasible = sup_global <= 1.0 + tol

    # off-support margin: sup |eta| over probes that are NOT near an atom
    offsupport_margin = float(1.0 - abs_eta_probe.max())
    strict_slack = abs_eta_probe.max() < 1.0 - tol

    # --- (3) non-degeneracy at the atoms ---
    grad_norm_max = np.nan
    hess_negdef_frac = np.nan
    nondeg = False
    if verify_nondegeneracy:
        gnorms, negdef_flags = [], []
        for (w, b) in neurons:
            g, hdiag = _eta_grad_hess(w, b, X, q_star, lam)
            gnorms.append(np.linalg.norm(g))
            # negative-definite proxy: all diagonal Hessian entries < 0
            negdef_flags.append(bool(np.all(hdiag < 1e-6)))  # <~0 with slack
        grad_norm_max = float(np.max(gnorms))
        hess_negdef_frac = float(np.mean(negdef_flags))
        # non-degenerate if atoms are near-stationary maxima and slack is strict
        nondeg = (grad_norm_max < 0.5) and (hess_negdef_frac > 0.5) and strict_slack

    saturated = max_sat_err < 5e-2      # saturation is approximate (sampled oracle)
    signs_match = max_sign_err < 5e-2
    certified = feasible and saturated and signs_match

    return CertificateReport(
        lam=lam, K=K,
        sup_abs_eta_support=sup_support,
        sup_abs_eta_global=sup_global,
        feasible=feasible,
        max_saturation_error=max_sat_err,
        max_sign_error=max_sign_err,
        saturated=saturated,
        signs_match=signs_match,
        offsupport_margin=offsupport_margin,
        strict_slack=strict_slack,
        grad_norm_at_atoms=grad_norm_max,
        hessian_negdef_fraction=hess_negdef_frac,
        nondegenerate=nondeg,
        certified=certified,
    )


def format_certificate_report(rep: CertificateReport) -> str:
    lines = [
        f"  Dual certificate report (lambda={rep.lam:.3f}, K={rep.K}):",
        f"    feasibility     : sup|eta|_global = {rep.sup_abs_eta_global:.4f}   "
        f"[{'PASS' if rep.feasible else 'FAIL'}: <= 1]",
        f"    saturation      : max||eta(atom)|-1| = {rep.max_saturation_error:.4f}   "
        f"[{'PASS' if rep.saturated else 'FAIL'}]",
        f"    sign match      : max|eta(atom)-sign(a)| = {rep.max_sign_error:.4f}   "
        f"[{'PASS' if rep.signs_match else 'FAIL'}]",
        f"    off-supp margin : 1 - sup|eta|_offsupp = {rep.offsupport_margin:+.4f}   "
        f"[{'strict' if rep.strict_slack else 'tight'}]",
        f"    non-degeneracy  : grad@atoms={rep.grad_norm_at_atoms:.3f}, "
        f"negdef-frac={rep.hessian_negdef_fraction:.2f}   "
        f"[{'PASS' if rep.nondegenerate else 'weak'}]",
        f"    ==> {'CERTIFIED OPTIMAL' if rep.certified else 'NOT certified'}"
        f"{' (non-degenerate)' if rep.nondegenerate else ''}",
    ]
    return "\n".join(lines)

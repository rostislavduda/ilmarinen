"""Certify the TV/L1 solution.

For the L1-fitted amplitudes, the KKT conditions of
   min_a 1/2||Phi a - y||^2 + lam1||a||_1
guarantee, by construction:
   eta_k = (1/lam1) Phi_k^T (y - Phi a)  satisfies
       eta_k = sign(a_k)        on the active support (|eta_k| = 1)
       |eta_k| <= 1             on inactive selected atoms
This module verifies those hold numerically (they should, to solver tolerance)
and checks GLOBAL feasibility |eta(Theta)| <= 1 over freshly sampled Theta --
the one condition that is NOT automatic (it tests whether the selected
dictionary was rich enough that no unselected feature violates the certificate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TVCertReport:
    lam1: float
    K_active: int
    sat_error_active: float      # max_k on support | |eta_k| - 1 |
    sign_error_active: float     # max_k on support | eta_k - sign(a_k) |
    max_abs_eta_inactive: float  # max over inactive SELECTED atoms (should be <=1)
    sup_abs_eta_global: float    # empirical sup over fresh Theta samples
    feasible_global: bool
    kkt_satisfied: bool
    certified: bool


def certify_tv(result, X, y, n_probe=20000, tol=2e-2, seed=1) -> TVCertReport:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    neurons = result.neurons
    a = result.amplitudes
    lam1 = result.lam1
    Phi = np.column_stack([np.tanh(X @ w + b) for (w, b) in neurons])   # (n, K)
    resid = y - Phi @ a                                                  # y - Phi a

    # certificate on the SELECTED atoms:  eta_k = (1/lam1) Phi_k^T resid
    eta_sel = (Phi.T @ resid) / lam1
    active = result.active_mask
    if active.sum() > 0:
        sat_err = float(np.max(np.abs(np.abs(eta_sel[active]) - 1.0)))
        sign_err = float(np.max(np.abs(eta_sel[active] - np.sign(a[active]))))
    else:
        sat_err = sign_err = float("nan")
    max_inactive = float(np.max(np.abs(eta_sel[~active]))) if (~active).any() else 0.0

    # GLOBAL feasibility over fresh Theta:  eta(Theta) = (1/lam1) <resid, phi>
    Wp = rng.standard_normal((d, n_probe)) * (1.0 / np.sqrt(d))
    idx = rng.integers(0, n, n_probe // 2)
    Wp[:, :n_probe // 2] = (X[idx].T / (np.linalg.norm(X[idx], axis=1) + 1e-6))
    Bp = rng.standard_normal(n_probe) * 0.5
    Fp = np.tanh(X @ Wp + Bp)                       # (n, n_probe)
    eta_probe = (resid @ Fp) / lam1
    sup_global = float(max(np.abs(eta_sel).max(), np.abs(eta_probe).max()))
    feasible = sup_global <= 1.0 + tol

    kkt = (not np.isnan(sat_err)) and sat_err < tol and sign_err < tol and max_inactive <= 1 + tol
    certified = kkt and feasible
    return TVCertReport(
        lam1=lam1, K_active=int(active.sum()),
        sat_error_active=sat_err, sign_error_active=sign_err,
        max_abs_eta_inactive=max_inactive,
        sup_abs_eta_global=sup_global, feasible_global=feasible,
        kkt_satisfied=kkt, certified=certified,
    )


def format_tv_report(rep: TVCertReport) -> str:
    return "\n".join([
        f"  TV/L1 certificate (lam1={rep.lam1:.2f}, active K={rep.K_active}):",
        f"    saturation on support : max||eta|-1| = {rep.sat_error_active:.4f}  "
        f"[{'PASS' if rep.sat_error_active < 2e-2 else 'FAIL'}]",
        f"    sign on support       : max|eta-sign(a)| = {rep.sign_error_active:.4f}  "
        f"[{'PASS' if rep.sign_error_active < 2e-2 else 'FAIL'}]",
        f"    inactive |eta| <= 1   : max = {rep.max_abs_eta_inactive:.4f}  "
        f"[{'PASS' if rep.max_abs_eta_inactive <= 1.02 else 'FAIL'}]",
        f"    global feasibility    : sup|eta| = {rep.sup_abs_eta_global:.4f}  "
        f"[{'PASS' if rep.feasible_global else 'FAIL'}]",
        f"    ==> {'CERTIFIED TV-OPTIMAL' if rep.certified else 'NOT certified'}",
    ])

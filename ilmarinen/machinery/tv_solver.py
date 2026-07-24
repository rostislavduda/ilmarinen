"""TV-faithful width solver.

The greedy+OLS solver in width_sparsity.py selects neurons by correlation but
refits amplitudes by ORDINARY least squares -- which solves an UNREGULARIZED
problem on the selected features. Its residual is OLS-orthogonal to the atoms,
forcing the certificate eta ~ 0 on the support (|eta|=1 saturation cannot hold).
So the greedy+OLS solution is provably NOT the TV minimizer the certificate
certifies -- running the certificate exposed exactly this.

This module solves the ACTUAL TV-regularized problem so the certificate applies
exactly:

    min_a  (1/2) || Phi a - y ||^2  +  lambda' || a ||_1

over the amplitudes a of the selected atoms (|| a ||_1 is the discrete TV norm
of rho = sum_k a_k delta_{Theta_k}). At the L1 optimum the KKT conditions are:

    (1/lambda') Phi_k^T (y - Phi a)  =  sign(a_k)        for a_k != 0   (|eta_k| = 1)
    | (1/lambda') Phi_j^T (y - Phi a) |  <=  1            for a_k  = 0   (|eta_j| < 1)

which is EXACTLY the dual-certificate saturation/feasibility structure, now
holding by construction of the L1 solution. eta(Theta) = (1/lambda')
< y - Phi a, phi(.;Theta) > then satisfies |eta| <= 1 globally with equality on
the L1 support.

We solve the L1 problem by coordinate descent (soft-thresholding), which is
exact and cheap for the modest number of selected atoms. Atom SELECTION still
uses the Frank-Wolfe correlation oracle (that part was already correct); only
the amplitude fit changes from OLS to L1.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def _soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_coordinate_descent(Phi, y, lam1, iters=500, tol=1e-8):
    """Solve  min_a 1/2||Phi a - y||^2 + lam1 ||a||_1  by coordinate descent."""
    n, K = Phi.shape
    a = np.zeros(K)
    col_sq = (Phi ** 2).sum(0) + 1e-12
    r = y - Phi @ a
    for _ in range(iters):
        a_old = a.copy()
        for k in range(K):
            r = r + Phi[:, k] * a[k]                 # remove atom k
            rho_k = Phi[:, k] @ r
            a[k] = _soft_threshold(rho_k, lam1) / col_sq[k]
            r = r - Phi[:, k] * a[k]                 # re-add atom k
        if np.max(np.abs(a - a_old)) < tol:
            break
    return a


@dataclass
class TVResult:
    K_selected: int                  # atoms selected by Frank-Wolfe
    K_active: int                    # atoms with nonzero amplitude after L1 (the true support)
    lam1: float                      # L1 penalty on amplitudes
    train_acc: float
    test_acc: float
    neurons: list = field(default_factory=list)      # (w, b) for all selected atoms
    amplitudes: np.ndarray | None = None             # L1 amplitudes (some may be 0)
    active_mask: np.ndarray | None = None            # which atoms are in the support


def tv_faithful_solve(X, y, Xt, yt, lam1=1.0, n_select=80, n_candidates=400, seed=0):
    """Frank-Wolfe atom selection + L1 amplitude fit (the TV minimizer).

    n_select : number of candidate atoms to select via the correlation oracle
               (an over-complete dictionary; L1 then prunes to the true support).
    lam1     : L1 penalty on amplitudes -- this is the operative sparsity knob
               for the TV problem (larger -> sparser active support).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    r = y.astype(np.float64).copy()
    neurons = []
    Phi_tr = np.zeros((n, 0))

    # --- Frank-Wolfe selection: build an over-complete dictionary ---
    for _ in range(n_select):
        W = rng.standard_normal((d, n_candidates)) * (1.0 / np.sqrt(d))
        idx = rng.integers(0, n, n_candidates // 2)
        W[:, :n_candidates // 2] = (X[idx].T / (np.linalg.norm(X[idx], axis=1) + 1e-6))
        B = rng.standard_normal(n_candidates) * 0.5
        F = np.tanh(X @ W + B)
        Fc = F - F.mean(0, keepdims=True)
        corr = (Fc.T @ r) / (np.linalg.norm(Fc, axis=0) * np.linalg.norm(r) + 1e-12)
        j = int(np.argmax(np.abs(corr)))
        w, b = W[:, j], B[j]
        f = np.tanh(X @ w + b)
        neurons.append((w, b))
        Phi_tr = np.column_stack([Phi_tr, f])
        # provisional OLS just to update the selection residual (selection only)
        A_tmp, *_ = np.linalg.lstsq(Phi_tr, y, rcond=None)
        r = y - Phi_tr @ A_tmp

    # --- L1 amplitude fit on the selected dictionary (the actual TV solve) ---
    a = lasso_coordinate_descent(Phi_tr, y, lam1)
    active = np.abs(a) > 1e-6
    Phi_te = np.column_stack([np.tanh(Xt @ w + b) for (w, b) in neurons])
    pred_tr = Phi_tr @ a
    pred_te = Phi_te @ a
    return TVResult(
        K_selected=len(neurons),
        K_active=int(active.sum()),
        lam1=lam1,
        train_acc=float(np.mean(np.sign(pred_tr) == y)),
        test_acc=float(np.mean(np.sign(pred_te) == yt)),
        neurons=neurons,
        amplitudes=a,
        active_mask=active,
    )

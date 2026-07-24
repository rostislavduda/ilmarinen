"""Width-sparsity machinery: greedy / conditional-gradient neuron insertion.

Fits a two-layer network f(x) = sum_k a_k tanh(w_k . x + b_k) to a scalar
target by inserting, one at a time, the hidden unit whose feature most
correlates (in absolute value) with the current residual -- the certificate's
argmax oracle, approximated by sampling data-informed candidates. Stops when
the max correlation falls below lambda (certificate feasibility).

This is the best-conditioned piece of the framework: exact for the two-layer
scalar problem, single-run, monotone. Returns the integer neuron count K and
the full certificate trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InsertionResult:
    K: int
    lam: float
    train_acc: float
    test_acc: float
    final_max_corr: float
    trajectory: list = field(default_factory=list)  # (step, max_corr, train_acc, test_acc)
    neurons: list = field(default_factory=list)  # (w, b)
    amplitudes: np.ndarray | None = None


def greedy_insertion(
    X, y, Xt, yt, lam: float = 0.05, max_neurons: int = 200, n_candidates: int = 400, seed: int = 0
) -> InsertionResult:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    r = y.astype(np.float64).copy()
    neurons, traj = [], []
    Phi_tr = np.zeros((n, 0))
    Phi_te = np.zeros((Xt.shape[0], 0))
    A = None

    for step in range(max_neurons):
        # data-informed candidate neurons (better argmax oracle than pure random)
        W = rng.standard_normal((d, n_candidates)) * (1.0 / np.sqrt(d))
        idx = rng.integers(0, n, n_candidates // 2)
        W[:, : n_candidates // 2] = X[idx].T / (np.linalg.norm(X[idx], axis=1) + 1e-6)
        B = rng.standard_normal(n_candidates) * 0.5
        F = np.tanh(X @ W + B)
        Fc = F - F.mean(0, keepdims=True)
        corr = (Fc.T @ r) / (np.linalg.norm(Fc, axis=0) * np.linalg.norm(r) + 1e-12)
        j = int(np.argmax(np.abs(corr)))
        max_corr = float(np.abs(corr[j]))

        if max_corr < lam:  # certificate feasible -> stop
            traj.append((step, max_corr, np.nan, np.nan))
            break

        w, b = W[:, j], B[j]
        Phi_tr = np.column_stack([Phi_tr, np.tanh(X @ w + b)])
        Phi_te = np.column_stack([Phi_te, np.tanh(Xt @ w + b)])
        neurons.append((w, b))
        A, *_ = np.linalg.lstsq(Phi_tr, y, rcond=None)  # fully-corrective refit
        pred_tr = Phi_tr @ A
        r = y - pred_tr
        atr = float(np.mean(np.sign(pred_tr) == y))
        ate = float(np.mean(np.sign(Phi_te @ A) == yt))
        traj.append((step, max_corr, atr, ate))

    last = [t for t in traj if not np.isnan(t[2])]
    return InsertionResult(
        K=len(neurons),
        lam=lam,
        train_acc=last[-1][2] if last else float("nan"),
        test_acc=last[-1][3] if last else float("nan"),
        final_max_corr=traj[-1][1] if traj else float("nan"),
        trajectory=traj,
        neurons=neurons,
        amplitudes=A,
    )

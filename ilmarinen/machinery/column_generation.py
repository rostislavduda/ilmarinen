"""Certificate-driven column generation (cutting-plane Frank-Wolfe).

The TV/L1 solver over a fixed sampled dictionary is only optimal WITHIN that
dictionary; the global certificate typically reports sup|eta| > 1, meaning
neurons exist outside the dictionary that the certificate says belong in the
support. This module closes that gap the analytically correct way:

    repeat:
      1. solve  min_a 1/2||Phi a - y||^2 + lam1||a||_1   over current dictionary
      2. compute residual  q = y - Phi a  and certificate eta(.) = <q,phi>/lam1
      3. find the WORST VIOLATORS: argmax_Theta |eta(Theta)| over fresh samples
      4. if max|eta| <= 1 + tol : CERTIFIED, stop
         else add the top-m violating neurons to the dictionary and repeat

The violating neurons ARE the cutting planes / the atoms Frank-Wolfe should add
next: the certificate condition |eta|<=1 is the dual constraint, and each
violation is a dual constraint the current primal fails -- adding that atom to
the primal restores it. At convergence sup|eta| <= 1 everywhere sampled, so the
KKT conditions hold globally (to sampling resolution) and the support is the
certified TV-optimal support -- however large the continuous problem demands.

This is the honest realization: instead of assuming a small K, we let the
certificate tell us the true support size, and we stop only when the solution
is genuinely certified rather than merely dictionary-optimal.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .tv_solver import lasso_coordinate_descent


@dataclass
class ColGenResult:
    lam1: float
    n_rounds: int
    K_active: int                    # certified support size
    dictionary_size: int             # total atoms generated
    sup_eta_history: list            # sup|eta| after each round (should -> ~1)
    Kactive_history: list            # active support size after each round
    train_acc: float
    test_acc: float
    certified: bool                  # sup|eta| <= 1 + tol at termination
    neurons: list = field(default_factory=list)
    amplitudes: np.ndarray | None = None
    active_mask: np.ndarray | None = None
    final_sup_eta: float = float("nan")


def _sample_candidates(X, n_candidates, rng):
    n, d = X.shape
    W = rng.standard_normal((d, n_candidates)) * (1.0 / np.sqrt(d))
    idx = rng.integers(0, n, n_candidates // 2)
    W[:, :n_candidates // 2] = (X[idx].T / (np.linalg.norm(X[idx], axis=1) + 1e-6))
    B = rng.standard_normal(n_candidates) * 0.5
    return W, B


def column_generation_solve(
    X, y, Xt, yt,
    lam1: float = 20.0,
    init_atoms: int = 40,
    add_per_round: int = 20,
    max_rounds: int = 40,
    n_probe: int = 20000,
    tol: float = 2e-2,
    seed: int = 0,
    verbose: bool = False,
):
    """Solve the TV problem by certificate-driven column generation.

    Parameters
    ----------
    lam1 : L1 penalty on amplitudes (the sparsity knob).
    init_atoms : size of the initial sampled dictionary.
    add_per_round : number of top violating neurons to add each round.
    max_rounds : cap on outer iterations.
    n_probe : fresh parameter-space samples for the global certificate each round.
    tol : feasibility tolerance; certified when sup|eta| <= 1 + tol.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # --- initial dictionary: sampled atoms selected by correlation with y ---
    W0, B0 = _sample_candidates(X, max(init_atoms * 8, 400), rng)
    F0 = np.tanh(X @ W0 + B0)
    Fc = F0 - F0.mean(0, keepdims=True)
    corr = np.abs((Fc.T @ y) / (np.linalg.norm(Fc, axis=0) * np.linalg.norm(y) + 1e-12))
    top = np.argsort(corr)[::-1][:init_atoms]
    neurons = [(W0[:, j], B0[j]) for j in top]

    sup_hist, kact_hist = [], []
    a = None
    active = None
    certified = False
    sup_eta = np.inf

    for rnd in range(max_rounds):
        Phi = np.column_stack([np.tanh(X @ w + b) for (w, b) in neurons])   # (n, K)
        a = lasso_coordinate_descent(Phi, y, lam1)
        active = np.abs(a) > 1e-6
        resid = y - Phi @ a

        # global certificate: sample fresh Theta, find worst violators
        Wp, Bp = _sample_candidates(X, n_probe, rng)
        Fp = np.tanh(X @ Wp + Bp)                        # (n, n_probe)
        eta_probe = (resid @ Fp) / lam1
        abs_eta = np.abs(eta_probe)
        # include current selected atoms in the sup
        eta_sel = (Phi.T @ resid) / lam1
        sup_eta = float(max(abs_eta.max(), np.abs(eta_sel).max()))
        sup_hist.append(sup_eta)
        kact_hist.append(int(active.sum()))

        if verbose:
            print(f"    round {rnd:2d}: dict={len(neurons):3d} active={int(active.sum()):3d} "
                  f"sup|eta|={sup_eta:6.3f}")

        if sup_eta <= 1.0 + tol:
            certified = True
            break

        # add the top violating neurons as new columns (the cutting planes)
        viol = np.argsort(abs_eta)[::-1][:add_per_round]
        for j in viol:
            neurons.append((Wp[:, j], Bp[j]))

    # final metrics on the last solve
    Phi = np.column_stack([np.tanh(X @ w + b) for (w, b) in neurons])
    a = lasso_coordinate_descent(Phi, y, lam1)
    active = np.abs(a) > 1e-6
    Phi_te = np.column_stack([np.tanh(Xt @ w + b) for (w, b) in neurons])
    tr_acc = float(np.mean(np.sign(Phi @ a) == y))
    te_acc = float(np.mean(np.sign(Phi_te @ a) == yt))

    return ColGenResult(
        lam1=lam1, n_rounds=len(sup_hist), K_active=int(active.sum()),
        dictionary_size=len(neurons), sup_eta_history=sup_hist,
        Kactive_history=kact_hist, train_acc=tr_acc, test_acc=te_acc,
        certified=certified, neurons=neurons, amplitudes=a, active_mask=active,
        final_sup_eta=sup_eta,
    )


def format_colgen(res: ColGenResult) -> str:
    hist = "  ".join(f"{s:.2f}" for s in res.sup_eta_history)
    status = "CERTIFIED" if res.certified else f"not certified (sup|eta|={res.final_sup_eta:.2f})"
    return "\n".join([
        f"  Column generation (lam1={res.lam1:.1f}):",
        f"    rounds={res.n_rounds}  final dictionary={res.dictionary_size}  "
        f"certified support K={res.K_active}",
        f"    sup|eta| per round: {hist}",
        f"    test_acc={res.test_acc:.3f}   ==> {status}",
    ])

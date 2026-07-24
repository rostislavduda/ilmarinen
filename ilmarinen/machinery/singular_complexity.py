"""Singularity-aware complexity via the Local Learning Coefficient (direction B2).

The analytical report's depth/width objectives and the contract evidence (B1) use a parameter-count
complexity -- effectively the classical BIC term (k/2) log n. But neural networks are SINGULAR: their loss
minima are degenerate and non-isolated (permutation/scaling symmetries, redundant units), so the raw
parameter count massively OVER-states their complexity. Watanabe's singular learning theory replaces (k/2)
with the Real Log Canonical Threshold (RLCT) lambda, a geometric invariant of the loss-singular set, in the
free-energy / WBIC expansion

    F_n  =  n * L_n(w*)  +  lambda * log n  +  O_P(log log n),   lambda <= k/2   (equality only for regular models).

This module estimates lambda empirically by the Local Learning Coefficient (LLC) of Lau, Furman, Wang,
Murfet & Wei (2024-2025), the canonical estimator: it samples the tempered, localized Gibbs posterior around
the fitted optimum w* by stochastic-gradient Langevin dynamics (SGLD) and reads

    lambda_hat(w*)  =  n * beta * ( E_posterior[L_n(w)] - L_n(w*) ),     beta = 1 / log n,

with the posterior  p(w) ∝ exp( -n*beta*L_n(w) - (gamma/2) ||w - w*||^2 )  (tempered by beta, localized by
gamma). Intuition: poke the loss basin around w* and see how fast the loss rises -- a broad/degenerate
(singular) basin gives a small lambda; a sharp (regular) basin gives lambda ~ k/2.

Why this fits the package. The report itself flags that its sharpest depth-scaling claims are heuristic and
that the parameter-count complexity is the wrong one for singular models; this is the principled replacement.
It also refines B1: the parameter-Occam term that CANCELLED across contracts (at a shared budget) can be
reintroduced as lambda_c * log n, which does NOT cancel when contracts have different effective dimension at
the same nominal budget -- a strictly more faithful evidence.

Validation of the estimator (see tests/b2_singular_complexity.md): on genuinely singular over-wide tanh nets
fitting a rank-1 target, lambda grows SUBLINEARLY in width while k/2 grows linearly, so lambda/(k/2) collapses
(0.25 -> 0.08 -> 0.04 for H=1,4,16) -- the correct SLT signature. On a regular linear-Gaussian model
lambda_hat ~ k/2. Cost is ~one SGD step per SGLD step, so it is cheap on the package's small nets.

Honest scope. lambda_hat is an SGLD estimate (minibatch Langevin, not exact MCMC); it is sensitive to the
localization gamma and step size eps (documented knobs), and formal unbiasedness of stochastic-gradient MCMC
is not guaranteed. It is a singularity-aware COMPLEXITY reading, reported with a per-chain std, not a certified
invariant. Defaults are tuned for the package's small models; the estimator exposes the knobs and a
calibration helper.
"""

from __future__ import annotations

import numpy as np
import torch


def estimate_llc(
    model,
    loss_closure,
    n,
    *,
    chains=5,
    steps=400,
    burn=150,
    eps=3e-5,
    gamma=100.0,
    nbeta=None,
    device=None,
    seed=0,
    return_traces=False,
):
    """Estimate the Local Learning Coefficient (approx. RLCT) of a FITTED model at its current parameters.

    model         : an nn.Module whose current parameters are the fitted optimum w* (LLC is LOCAL to w*).
    loss_closure  : callable() -> scalar tensor, the MEAN training loss L_n(w) at the model's current params.
                    (It should recompute the loss on the training data / a fixed minibatch each call.)
    n             : number of training points (sets beta = 1/log n and the n*beta scale).
    chains        : independent SGLD chains from w* (variance reduction).
    steps, burn   : SGLD steps per chain and burn-in discarded before averaging.
    eps           : SGLD step size (the sensitive knob; too large diverges, too small under-explores).
    gamma         : localization strength (Gaussian tether to w*); too large drowns the geometry, too small
                    lets the chain drift. Defaults suit small nets; use calibrate_llc to check the plateau.
    nbeta         : if given, use beta = nbeta / n instead of 1/log n (for the standard n*beta parameterization).

    Returns dict: lambda (LLC estimate), lambda_std (per-chain std), n_params (k), half_params (k/2),
    ratio = lambda/(k/2) (singularity index: 1 ~ regular, ->0 ~ increasingly singular), L_star, beta.
    """
    if device is None:
        device = next(model.parameters()).device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    beta = (1.0 / float(np.log(max(n, 3)))) if nbeta is None else float(nbeta) / n

    params = [p for p in model.parameters() if p.requires_grad]
    w_star = [p.detach().clone() for p in params]
    k = int(sum(p.numel() for p in params))

    with torch.no_grad():
        L_star = float(loss_closure().item())

    chain_means, traces = [], []
    for _ in range(chains):
        for p, w0 in zip(params, w_star):
            p.data.copy_(w0)
        acc, tr = [], []
        for t in range(steps):
            loss = loss_closure()
            for p in params:
                if p.grad is not None:
                    p.grad = None
            loss.backward()
            with torch.no_grad():
                for p, w0 in zip(params, w_star):
                    grad = p.grad if p.grad is not None else torch.zeros_like(p)
                    noise = torch.randn(p.shape, generator=g).to(p.device) * float(np.sqrt(eps))
                    p.data.add_(-(eps / 2.0) * (n * beta * grad + gamma * (p.data - w0)) + noise)
            lv = float(loss.item())
            if t >= burn:
                acc.append(lv)
            if return_traces:
                tr.append(lv)
        chain_means.append(float(np.mean(acc)) if acc else float("nan"))
        if return_traces:
            traces.append(tr)

    # restore w*
    with torch.no_grad():
        for p, w0 in zip(params, w_star):
            p.data.copy_(w0)

    per_chain_lambda = [n * beta * (cm - L_star) for cm in chain_means]
    lam = float(np.mean(per_chain_lambda))
    # VALIDITY: the LLC is only meaningful at a genuine local MINIMUM of the training loss. If w* is not
    # converged (still on a downward slope), SGLD immediately finds lower loss, E[L] < L*, and lambda comes
    # out NEGATIVE -- which is unphysical (RLCT >= 0). A clearly-negative lambda therefore flags a
    # non-converged w*, not a real complexity; we surface that rather than report nonsense.
    valid = lam > -abs(0.5)  # small negative tolerance for SGLD noise; strongly negative => not at a min
    out = {
        "lambda": lam,
        "lambda_std": float(np.std(per_chain_lambda)),
        "n_params": k,
        "half_params": k / 2.0,
        "ratio": float(lam / (k / 2.0)) if k > 0 else float("nan"),
        "L_star": L_star,
        "beta": beta,
        "n": int(n),
        "valid": bool(valid),
        "note": (
            "SGLD local learning coefficient (Lau et al.): lambda ~ RLCT <= k/2. ratio->0 signals a more "
            "singular (degenerate) solution; the free energy uses lambda*log n in place of (k/2)*log n. "
            "VALID ONLY at a converged local minimum -- a strongly negative lambda means w* is not "
            "converged (train the net longer), not a real complexity."
        ),
    }
    if return_traces:
        out["traces"] = traces
    return out


def free_energy(L_star, lam, n):
    """Singular free energy / WBIC-style stochastic complexity:  F_n = n*L_star + lambda*log n.

    This is the singularity-aware replacement for the BIC score n*L_star + (k/2)*log n. Lower is better; the
    lambda*log n term is the correct Occam factor for singular models (lambda <= k/2)."""
    return float(n) * float(L_star) + float(lam) * float(np.log(max(n, 3)))


def calibrate_llc(
    model,
    loss_closure,
    n,
    *,
    gammas=(1.0, 10.0, 100.0, 300.0),
    eps_list=(1e-5, 3e-5, 1e-4),
    chains=3,
    steps=300,
    burn=120,
    seed=0,
):
    """Sweep (gamma, eps) and report lambda_hat for each, to find the hyperparameter PLATEAU where the
    estimate is insensitive to the knobs (the standard LLC calibration -- Lau et al. look for the range where
    lambda_hat is flat in eps). Returns a list of {gamma, eps, lambda, lambda_std}; pick a (gamma, eps) in
    the flat region for estimate_llc. Diverged chains show up as large/NaN lambda and should be excluded."""
    rows = []
    for gm in gammas:
        for ep in eps_list:
            r = estimate_llc(model, loss_closure, n, chains=chains, steps=steps, burn=burn, eps=ep, gamma=gm, seed=seed)
            rows.append({"gamma": gm, "eps": ep, "lambda": r["lambda"], "lambda_std": r["lambda_std"]})
    return rows

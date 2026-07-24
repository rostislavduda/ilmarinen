"""Bayesian evidence reading of the priced contract selection (direction B1).

The tie-break selects a computational contract (set / graph / equivariant) by the MDL action
J(c) = R(c) + mu_c * Omega_struct(c), argmin over c. This module shows that -- once the fit term is put on
a log-likelihood scale -- that same objective IS an (approximate) Bayesian log-joint over contracts, and
turns the argmin into a calibrated POSTERIOR.

Derivation (why J is already a log-joint, made explicit)
--------------------------------------------------------
For a contract c with a fitted model, n training points, and mean per-datum negative log-likelihood Lhat_c,
the standard Laplace/BIC expansion of the model evidence is

    -log p(data | c)  ~  n * Lhat_c  +  (k_c / 2) * log n  +  O(1),

and with a prior p(c) over contracts the log-joint is  -log p(c, data) = -log p(data|c) - log p(c).

Two observations specialize this to the contract choice:

  (1) The STRUCTURE PRIOR. Omega_struct(c) (nats) is precisely a description length for the contract's
      structure (set: 0; graph: the adjacency code E*log(C(N,2)/E); equivariant: adjacency + the geometry
      code (N*d - dim SO(d))*log(1/delta)). A description-length prior IS a -log prior: p(c) ∝ exp(-Omega).
      So  -log p(c) = Omega_struct(c) + const.

  (2) The PARAMETER-OCCAM TERM CANCELS. The (k_c/2) log n term is the parameter-count Occam factor from
      integrating out the weights. Neural nets are heavily over-parameterized and singular, so this raw
      count is both the WRONG complexity (classical BIC over-penalizes -- the singular-learning-theory point)
      and, crucially, approximately COMMON across contracts here: the bake-off trains every candidate contract
      at the SAME width/depth budget, so k_c is comparable across c and the term cancels in the posterior
      (a softmax over c). What remains that genuinely differs by contract is exactly Omega_struct.

Hence the honest per-contract log-joint for the CONTRACT posterior (at a fixed parameter budget) is

    -log p(c | data)  =  n * Lhat_c  +  Omega_struct(c)  + const,

i.e. the package's J with R replaced by an NLL-scale fit n*Lhat and the exchange rate mu_c fixed to 1
(nats vs nats -- no free knob). The posterior is  P(c | data) = softmax_c( -[n*Lhat_c + Omega_struct(c)] ).

This does two things the argmin did not: (a) it puts the fit-vs-structure tradeoff on a calibrated (nats)
scale instead of the tuned mu_c, and (b) it reports a DISTRIBUTION over contracts (with an entropy /
confidence), so a near-tie is visible as a broad posterior rather than an arbitrary argmin.

Scope / honesty
---------------
* This is an APPROXIMATE evidence: Lhat is estimated from a held-out fit score (below), not a marginal
  likelihood, and the parameter-Occam cancellation assumes a shared budget (which the bake-off enforces).
  It is a Laplace/BIC-flavored reading, not an exact model evidence -- stated as such.
* mu_c is retained as an OPTIONAL structure-prior temperature (a genuine prior choice: how strongly to
  prefer simpler structure). mu_c=1 is the calibrated nats-vs-nats default; mu_c!=1 tempers the prior.
* The MAP of this posterior and the argmin of J = n*Lhat + Omega coincide by construction; the value add is
  the calibrated scale and the reported uncertainty, not a different winner.
"""

from __future__ import annotations

import math

import numpy as np


def score_to_nll(score, task, n_classes=2, y_std=1.0):
    """Convert a held-out fit score to a per-datum negative log-likelihood (nats), a log-likelihood-scale
    fit term. This is the bridge from the bake-off's score (R2 for regression, accuracy for classification)
    to the Lhat that the evidence needs.

    Regression (score = R2 on standardized targets): the residual variance is (1 - R2) * Var(y); the NLL of
    a Gaussian with that variance is 0.5*log(2*pi*e*sigma^2). With standardized y (Var=1), sigma^2 = 1 - R2.
    Classification (score = accuracy a): a calibrated model that is right with prob a over K classes has
    per-datum NLL bounded below by the binary-style cross-entropy -[a log a + (1-a) log((1-a)/(K-1))]
    (correct-class mass a spread over the rest); this is the standard accuracy->cross-entropy surrogate.

    Returns Lhat (nats per datum). Clipped for numerical safety.
    """
    if task == "regression":
        r2 = float(np.clip(score, -10.0, 1.0 - 1e-4))
        sigma2 = max((1.0 - r2) * (y_std ** 2), 1e-6)
        return 0.5 * math.log(2.0 * math.pi * math.e * sigma2)
    # classification
    a = float(np.clip(score, 1e-4, 1.0 - 1e-4))
    K = max(int(n_classes), 2)
    # cross-entropy of a model that puts mass a on the true class and spreads (1-a) over the other K-1
    ce = -(a * math.log(a) + (1.0 - a) * math.log((1.0 - a) / (K - 1)))
    return float(ce)


def contract_evidence(scores, omegas, n, task, n_classes=2, y_std=1.0, mu_c=1.0, temperature=1.0):
    """Approximate Bayesian posterior over contracts from held-out scores and structural code lengths.

    scores    : {contract: held-out score} (R2 for regression, accuracy for classification; higher=better).
    omegas    : {contract: Omega_struct (nats)} from machinery.dataset_omega_struct.
    n         : number of training points (sets the evidence scale; the fit term is n*Lhat).
    task      : "regression" | "classification".
    n_classes : number of classes (classification only), for the accuracy->NLL surrogate.
    y_std     : std of the targets (regression); pass 1.0 if scores are on standardized y (the default).
    mu_c      : structure-prior temperature. 1.0 = calibrated nats-vs-nats (the honest default); >1 prefers
                simpler structure more strongly, <1 less. This is the one retained prior choice.
    temperature: optional softmax temperature on the posterior (1.0 = the evidence-implied sharpness). Raise
                it only to REPORT a less saturated distribution; it does not change the MAP.

    Returns a dict: per-contract Lhat, neg_log_joint = n*Lhat + mu_c*Omega, the posterior probabilities,
    the MAP contract, and the posterior entropy (nats) as a confidence summary.
    """
    cs = list(scores)
    Lhat = {c: score_to_nll(scores[c], task, n_classes=n_classes, y_std=y_std) for c in cs}
    # -log joint per contract (up to a shared constant): fit (n*Lhat) + structure prior (mu_c*Omega)
    nlj = {c: float(n) * Lhat[c] + float(mu_c) * float(omegas.get(c, 0.0)) for c in cs}
    v = np.array([nlj[c] for c in cs], dtype=float)
    # posterior = softmax(-nlj / temperature), stabilized
    z = -(v - v.min()) / max(temperature, 1e-6)
    w = np.exp(z)
    p = w / w.sum()
    post = {c: float(pi) for c, pi in zip(cs, p)}
    kmap = int(np.argmin(v))
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "posterior": post,
        "map": cs[kmap],
        "neg_log_joint": {c: float(x) for c, x in zip(cs, v)},
        "nll": {c: float(Lhat[c]) for c in cs},
        "omega_struct": {c: float(omegas.get(c, 0.0)) for c in cs},
        "n": int(n), "mu_c": float(mu_c), "temperature": float(temperature),
        "posterior_entropy": ent,
        "max_entropy": float(math.log(len(cs))) if len(cs) > 1 else 0.0,
        "note": ("approximate contract evidence: -log p(c|data) ~ n*Lhat_c + mu_c*Omega_struct(c) "
                 "(parameter-Occam term cancels at the shared bake-off budget). MAP == argmin J."),
    }


def format_evidence(ev):
    """One-line-per-contract human-readable summary of the contract posterior."""
    lines = [f"contract posterior (n={ev['n']}, mu_c={ev['mu_c']}, "
             f"entropy={ev['posterior_entropy']:.3f}/{ev['max_entropy']:.3f} nats):"]
    order = sorted(ev["posterior"], key=lambda c: ev["neg_log_joint"][c])
    for c in order:
        star = "  <- MAP" if c == ev["map"] else ""
        lines.append(f"  {c:12} P={ev['posterior'][c]:.3f}  n*NLL+Omega={ev['neg_log_joint'][c]:8.2f}  "
                     f"(NLL={ev['nll'][c]:.3f}, Omega={ev['omega_struct'][c]:.2f}){star}")
    return "\n".join(lines)

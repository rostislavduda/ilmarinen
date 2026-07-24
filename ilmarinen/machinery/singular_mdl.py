"""Singular MDL: pricing the functional code length lambda*log n (direction D1).

This module welds two pieces the package already had but kept separate:

  * ``machinery/contract_mdl.omega_struct`` -- the STRUCTURAL code length of a contract (the
    group/interface scaffolding it commits to), a function of the contract TYPE, computed without
    training; and
  * ``machinery/singular_complexity.estimate_llc`` -- the local learning coefficient (LLC) lambda,
    a geometric invariant of the fitted loss basin, estimated by SGLD at the converged optimum.

The bridge is the singular-model minimum description length, which the recent MDL-meets-SLT result
(Urdshals, Lau, Hoogland, van Wingerden & Murfet, arXiv:2510.12077, 2025) states as

    MDL_sing(D)  =  -log p(D | w*)  +  lambda * log n  +  const,

with lambda the LLC. The first term is the package's residual R; the second is a FUNCTIONAL code
length -- the description length of the *fitted function's* effective degrees of freedom, which the
structural term cannot see (two models of the same contract type but different realized complexity
have identical omega_struct yet different lambda). This module exposes that functional term and a
total code length that ADDS it to the structural one:

    Omega_total  =  omega_struct(contract)  +  omega_func(lambda, n),
    omega_func(lambda, n)  =  lambda * log n     (nats; the singular free-energy complexity).

Design stance (why ADD, not REPLACE). ``contract_mdl`` deliberately prices STRUCTURAL richness and
argues (correctly) that the parameter count 1/2 k log n MISORDERS contracts. The LLC does not
contradict that: lambda prices the *functional* effective dimension of the realized fit, a different
axis from structural scaffolding. Singular learning theory replaces the *parametric* 1/2 k log n by
lambda * log n (lambda <= k/2, equality only for regular models) -- so omega_func is the principled
form of the parametric-Occam term that B1 found cancels across contracts at a shared budget but does
NOT cancel when contracts realize different effective dimension. The structural and functional terms
are complementary rungs of the same J = R + mu*Omega ladder.

Everything here is OPT-IN. Nothing in this module changes the default objective; a caller must ask
for the functional term explicitly (``price_singular=True`` on the selector, or by calling these
functions directly). The LLC is valid ONLY at a converged local minimum -- a strongly negative
lambda_hat flags a non-converged w* (see ``singular_complexity``), and we surface that rather than
price with a meaningless value.
"""

from __future__ import annotations

import math

from .contract_mdl import omega_struct
from .singular_complexity import estimate_llc


# --------------------------------------------------------------------------- functional code length
def omega_func(lam: float, n: int) -> float:
    """Functional (singular) code length in nats: max(lambda, 0) * log n.

    This is the complexity term of the singular free energy F_n = n*L(w*) + lambda*log n. It prices
    the effective degrees of freedom of the *fitted function*, complementary to the structural code
    length of the contract type. An RLCT is non-negative, so a slightly-negative lambda estimate (SGLD
    noise near a shallow/near-converged basin) is clamped to zero -- it denotes negligible functional
    complexity, never a negative code length. A STRONGLY negative lambda (a non-converged optimum) is a
    separate condition handled by the validity guard in ``singular_complexity_of``, which declines to
    price at all rather than clamping.
    """
    if n <= 1:
        return 0.0
    return max(float(lam), 0.0) * math.log(n)


def total_code_length(lam: float, n: int, contract: str, **omega_struct_kwargs) -> float:
    """Omega_total = omega_struct(contract) + omega_func(lambda, n).

    The structural term prices the contract's scaffolding (group/interface richness); the functional
    term prices the fitted function's effective dimension. Extra keyword args are forwarded to
    ``omega_struct`` (N, E, d, rank, shape, ...).
    """
    return omega_struct(contract, **omega_struct_kwargs) + omega_func(lam, n)


# --------------------------------------------------------------------------- estimate + price in one call
def singular_complexity_of(model, loss_closure, n, *, negative_tol=0.5, **llc_kwargs):
    """Estimate lambda at the model's current (assumed converged) parameters and return the priced
    functional code length, with an explicit validity flag.

    Returns a dict:
      lambda        : the LLC estimate lambda_hat(w*)
      omega_func    : lambda_hat * log n         (the functional code length, nats)
      valid         : False if lambda_hat is strongly negative (w* not at a minimum -> do not price)
      n, log_n      : bookkeeping

    llc_kwargs are forwarded to ``estimate_llc`` (chains, steps, burn, eps, gamma, seed, ...).
    The caller decides what to do when ``valid`` is False; this function never silently prices a
    non-converged point.
    """
    out = estimate_llc(model, loss_closure, n, **llc_kwargs)
    lam = out["lambda"]
    # mirror singular_complexity's guard: a clearly-negative lambda is non-physical (RLCT >= 0) and
    # signals a non-converged w*, so the functional price is not meaningful there.
    valid = bool(out.get("valid", lam > -abs(negative_tol)))
    of = omega_func(lam, n) if valid else float("nan")
    return {
        "lambda": lam,
        "omega_func": of,
        "valid": valid,
        "n": n,
        "log_n": math.log(n) if n > 1 else 0.0,
        "llc_raw": out,
    }


# --------------------------------------------------------------------------- objective with functional term
def singular_free_energy(residual_nll: float, lam: float, n: int) -> float:
    """The singular free energy F_n = residual + lambda*log n, i.e. R + omega_func.

    ``residual_nll`` is the fit term R = -log p(D | w*) (a total negative log likelihood in nats).
    This is the singular-MDL objective the D1 pricing minimizes when the functional term is enabled;
    with lambda*log n in place of a parameter-count penalty it is the SLT-correct code length.
    """
    return float(residual_nll) + omega_func(lam, n)

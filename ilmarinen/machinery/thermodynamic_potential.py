"""One free energy, one form, three temperatures (direction D2).

The package reads out three different Gibbs/Boltzmann measures, each governing a different architectural
decision, and each with its own inverse temperature. A careful reader (or referee) will notice that the
symbol beta is used in more than one place and ask whether these are the same temperature. This module is
the single documented answer: they are instances of ONE free-energy FORM at three levels of coarse-graining,
with THREE DELIBERATELY-DECOUPLED temperatures -- not one temperature, and (importantly) not temperatures
that may be naively equated.

--------------------------------------------------------------------------------------------------------------
THE ONE FORM

Every selection in the package minimizes a free energy of the identical shape

    F(rho)  =  <rho, E>  -  (1 / beta) * H(rho),                                (*)

a fit/energy term traded against an entropy term at inverse temperature beta, whose interior stationary
measure is the Gibbs law  rho ∝ exp(-beta * E).  (This is exactly the F_alpha of gibbs_alpha.py, the F =
R + (1/beta) KL of the analytical report, and the WBIC free energy of singular_complexity.py, written once.)
What changes between the three readouts is only (i) the SPACE the measure rho lives on, (ii) the ENERGY E on
that space, and (iii) hence the natural VALUE and UNITS of beta. The unification D2 makes explicit is that
these are the SAME functional (*) at successive coarse-grainings of the model -- weights, then the primitive
mixture, then the contract -- NOT three unrelated uses of a letter.

--------------------------------------------------------------------------------------------------------------
THE THREE LEVELS (coarsest energy first is easiest to read as fine->coarse)

  LEVEL W  (weights).  Space: weight space w. Energy: the total training loss n*L_n(w) (nats). Measure: the
    tempered, localized Gibbs posterior  p(w) ∝ exp(-n*beta_W*L_n(w) - (gamma/2)||w-w*||^2). The WBIC/SLT
    inverse temperature is

        beta_W = 1 / log n            (Watanabe's beta* -- the ONLY value at which
                                       n*beta_W*(E_post[L] - L*) estimates the RLCT lambda).

    Derivative read-out: the local learning coefficient lambda = d F_n / d log n  (singular_complexity.py, B2;
    fused into the pricing in singular_mdl.py, D1; tracked over training in developmental_llc.py, D4).

  LEVEL A  (primitives).  Space: the primitive simplex (a mixture weight alpha_i per candidate primitive).
    Energy: the solo cost Psi_i of primitive i (a held-out fit score put on an energy scale, LOWER=better).
    Measure: alpha_i ∝ exp(-beta_A * Psi_i). The inverse temperature is a SELECTION sharpness

        beta_A = gibbs_beta           (a free knob: default 8.0, or elbow-calibrated to the actual
                                       energy spread via select_beta_by_elbow; beta_A->inf is the hard
                                       argmin, finite beta_A the smooth mixture).

    Derivative read-out: the derived architecture distribution alpha, i.e. which primitive is selected and how
    confidently (gibbs_alpha.py). beta_A is the replicator/Gibbs equilibrium temperature of F_alpha.

  LEVEL C  (contracts).  Space: the contract set {set, graph, equivariant, ...}. Energy: the per-contract
    log-joint n*Lhat_c + Omega_struct(c) (nats). Measure: P(c|data) = softmax_c(-[n*Lhat_c + Omega_c]).
    The inverse temperature is fixed to unity,

        beta_C = 1                    (nats vs nats -- Lhat_c is an NLL and Omega_c a description length,
                                       both already in nats, so there is NO free exchange rate; the
                                       optional mu_c only TEMPERS the structure prior, mu_c=1 = calibrated).

    Derivative read-out: the contract posterior / its MAP == the priced-J winner (contract_evidence.py, B1).

--------------------------------------------------------------------------------------------------------------
WHY THE TEMPERATURES ARE DECOUPLED (the honest, load-bearing point)

It is tempting -- and the round-2 directions note phrased it this way -- to say "these are the same beta in
one Z" and to seek to set them consistently EQUAL. The premise-check (tests/d2_thermodynamic_potential.md)
shows that would be WRONG:

  * beta_W = 1/log n is ~0.14 at n=1200; gibbs_beta defaults to 8.0. Equating them (gibbs_beta := 1/log n)
    would drop the primitive-readout temperature ~57x and collapse alpha to nearly uniform -- destroying the
    selection. They are simply not the same number, by design.
  * The energies are on different spaces AND different units: E_W = n*L is O(n) NATS on weight space; Psi_A is
    an O(1) dimensionless fit metric on the simplex; E_C is O(n) nats on the contract set. There is no single
    Z_n(beta) whose marginals are all three; there is one FUNCTIONAL FORM (*) instantiated three times.
  * beta_W is THEORY-FIXED (only 1/log n recovers the RLCT); beta_C is UNIT-FIXED (nats:nats forces 1);
    beta_A is a genuine free SHARPNESS knob (how decisively to read off the mixture). Three different kinds of
    quantity: a physical constant of the estimator, a bookkeeping identity, and a tunable readout sharpness.

So the reconciliation is: ONE free-energy form, THREE levels, THREE temperatures that are each pinned by a
DIFFERENT principle (WBIC theory / nats-bookkeeping / readout sharpness) and must NOT be identified. This
module records that, and `assert_temperature_consistency` checks each beta sits at its principled value and
raises if code has accidentally coupled them.
--------------------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import math

# The three levels of the one free-energy form, as data (single source of truth for the report + the check).
POTENTIAL_LEVELS = (
    {
        "level": "W",
        "name": "weights",
        "space": "weight space w",
        "energy": "n * L_n(w)  (total training loss, nats)",
        "measure": "p(w) ∝ exp(-n*beta_W*L_n(w) - (gamma/2)||w-w*||^2)",
        "beta_symbol": "beta_W = 1/log n",
        "beta_kind": "theory-fixed (WBIC: only 1/log n recovers the RLCT)",
        "readout": "lambda = LLC = dF_n/d log n  (singular_complexity / singular_mdl / developmental_llc)",
    },
    {
        "level": "A",
        "name": "primitives",
        "space": "primitive simplex (alpha)",
        "energy": "Psi_i  (solo held-out cost of primitive i, energy scale, lower=better)",
        "measure": "alpha_i ∝ exp(-beta_A * Psi_i)",
        "beta_symbol": "beta_A = gibbs_beta",
        "beta_kind": "free readout-sharpness knob (default 8.0 or elbow-calibrated)",
        "readout": "alpha = derived architecture distribution  (gibbs_alpha)",
    },
    {
        "level": "C",
        "name": "contracts",
        "space": "contract set {set, graph, equivariant, ...}",
        "energy": "n*Lhat_c + Omega_struct(c)  (per-contract log-joint, nats)",
        "measure": "P(c|data) = softmax_c(-[n*Lhat_c + Omega_c])",
        "beta_symbol": "beta_C = 1",
        "beta_kind": "unit-fixed (nats vs nats -> no free exchange rate; mu_c only tempers the prior)",
        "readout": "contract posterior; MAP == priced-J winner  (contract_evidence, B1)",
    },
)


def wbic_beta(n):
    """The Level-W (weights) inverse temperature beta_W = 1/log n -- Watanabe's WBIC beta*, the unique value
    at which the tempered posterior's expected loss yields the RLCT. This is the SAME expression used inside
    singular_complexity.estimate_llc; exposed here so the potential's temperatures have one definition."""
    return 1.0 / math.log(max(int(n), 3))


def free_energy_form(mean_energy, entropy, beta):
    """The one free-energy FORM  F = <rho,E> - (1/beta) H(rho)  evaluated from its two scalar ingredients
    (the mean energy <rho,E> and the entropy H(rho) of the measure rho). Every level's free energy is this
    shape; provided so the identity is executable, not only prose. Returns F (same units as the energy)."""
    return float(mean_energy) - (1.0 / float(beta)) * float(entropy)


def assert_temperature_consistency(n, gibbs_beta, *, contract_beta=1.0, tol=1e-9, raise_on_fail=True):
    """Check that the three temperatures of the one thermodynamic potential are each set at their PRINCIPLED
    value, and -- the load-bearing check -- that they have NOT been accidentally coupled/equated.

    n            : training-set size (fixes beta_W = 1/log n).
    gibbs_beta   : the Level-A readout temperature actually in use (a float; 'auto'/elbow resolves to a float
                   upstream, so pass the resolved value). Must be > 0 and finite; it is a FREE knob, so any
                   positive value is 'consistent' EXCEPT one that betrays coupling (see below).
    contract_beta: the Level-C temperature (must be exactly 1 -- nats vs nats; a value != 1 means the
                   nats-bookkeeping was broken).

    The specific error this guards against (from the D2 premise-check): naively "unifying" the temperatures by
    setting gibbs_beta := 1/log n. That is WRONG (it collapses the primitive readout to ~uniform) and is
    flagged as a coupling error, because beta_A and beta_W are different kinds of quantity on different spaces.

    Returns a dict {ok, beta_W, beta_A, beta_C, issues:[...]}. Raises ValueError on failure when
    raise_on_fail (default), else returns the dict with ok=False.
    """
    beta_W = wbic_beta(n)
    issues = []

    # Level C: unit-fixed to 1 (nats vs nats).
    if abs(float(contract_beta) - 1.0) > tol:
        issues.append(f"beta_C = {contract_beta} != 1: the contract posterior is nats-vs-nats, so its inverse "
                      f"temperature must be exactly 1 (mu_c tempers the prior but does not move beta_C).")

    # Level A: a free positive knob -- but must be finite/positive, and must NOT have been coupled to beta_W.
    ga = float(gibbs_beta)
    if not math.isfinite(ga) or ga <= 0:
        issues.append(f"beta_A = gibbs_beta = {gibbs_beta} is not a positive finite number.")
    elif abs(ga - beta_W) <= max(tol, 1e-6):
        issues.append(f"beta_A == beta_W (= 1/log n = {beta_W:.4f}): this is the coupling error the D2 check "
                      f"guards against. The primitive-readout temperature and the WBIC weight temperature are "
                      f"different kinds of quantity on different spaces (a dimensionless fit metric on the "
                      f"simplex vs n*L nats on weight space); equating them collapses the primitive readout to "
                      f"~uniform. Set gibbs_beta on its own scale (default 8.0 or elbow-calibrated).")

    ok = not issues
    result = {"ok": ok, "beta_W": beta_W, "beta_A": ga, "beta_C": float(contract_beta), "issues": issues,
              "note": ("One free-energy form F = <rho,E> - (1/beta)H at three levels (weights/primitives/"
                       "contracts). beta_W=1/log n is theory-fixed (WBIC->RLCT), beta_C=1 is unit-fixed "
                       "(nats:nats), beta_A=gibbs_beta is a free readout-sharpness knob. They are deliberately "
                       "DECOUPLED; do not equate them.")}
    if not ok and raise_on_fail:
        raise ValueError("temperature consistency check failed: " + " | ".join(issues))
    return result

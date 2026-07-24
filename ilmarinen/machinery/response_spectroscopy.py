"""D5 -- Response / susceptibility spectroscopy of the selection.

The package prices selections by an objective (the Gibbs free energy of the primitive readout at
Level A; the priced contract objective J = R + mu_c * Omega at Level C) but reports only the
*argmin* -- never the CURVATURE of that objective at the chosen point. This module adds the
physicist's second half of the measurement: perturb the priced control and report how sharply the
argmin is preferred (its susceptibility) and how far it is to the nearest selection boundary.

It reuses quantities the fit ALREADY computed (the solo energies behind the Gibbs-alpha, and the
{scores, omegas, mu_c} behind the contract selection); no retraining, just finite differences and a
couple of closed forms. It changes NO selection -- it is a pure OBSERVABLE (interpretability tier).

Two channels, deliberately distinct because the two objectives have DIFFERENT analytic character
(this distinction is the main honest finding of the premise checks, and it is reported, not hidden):

  * Level A -- primitive readout (SMOOTH).  alpha_i = e^{-beta Psi_i}/Z is a softmax over the solo
    energies Psi_i = -score_i.  The free energy is F(beta) = -(1/beta) log Z(beta), and the
    susceptibility to the control (beta, equivalently the inverse energy scale) is the textbook
    specific-heat identity

        chi_A  ==  d^2(-log Z)/d beta^2  ==  Var_alpha(Psi)  >= 0.

    Large variance  <=>  a soft / ambiguous readout (several primitives near-tied); chi_A -> 0
    <=>  one primitive dominates.  This is a genuine, smooth, non-negative second derivative.

  * Level C -- contract selection (FIRST ORDER).  J_c(mu) = R_c + mu * Omega_tilde_c is
    piecewise-LINEAR in the price mu, so the envelope J*(mu) = min_c J_c(mu) has
    d^2 J*/d mu^2 = 0 almost everywhere, with the curvature concentrated at the KINKS where the
    lower envelope switches winners.  A naive chi = d^2 J*/d mu^2 is therefore ~0 and useless.
    The correct observable is the location of the nearest first-order transition -- the critical
    price mu* at which the winner changes -- the SLOPE JUMP (Delta of the normalized Omega) across
    that transition (the finite susceptibility of a first-order line), and the J-gap to the
    runner-up at the operating price.  Selection in mu is a sequence of first-order transitions,
    not a smooth response; the module says so.

References (premise-checked, not asserted): the devinterp "spectroscopy = response to perturbation"
picture (Gordon et al. 2026) and thermodynamic susceptibility of the posterior (2025); pairs with
the single thermodynamic potential of D2 (this module measures second derivatives of the very same
Level-A and Level-C free energies that D2 documents).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

__all__ = [
    "gibbs_susceptibility",
    "contract_transition",
    "response_spectrum",
]


# --------------------------------------------------------------------------------------------------
# Level A -- smooth readout susceptibility (specific heat of the primitive Gibbs measure)
# --------------------------------------------------------------------------------------------------
def gibbs_susceptibility(energies: Dict[str, float], beta: float) -> dict:
    """Specific-heat-like susceptibility of the Gibbs-alpha primitive readout.

    energies : {primitive: Psi_i}  (the SOLO energies behind gibbs_alpha_select; Psi_i = -score_i,
               so LOWER energy = better primitive). Exactly `result["gibbs_energies"]`.
    beta     : the readout inverse temperature actually used (result echoes it; gibbs_beta or its
               elbow-resolved value).

    Returns a dict with:
        chi            : Var_alpha(Psi) = d^2(-log Z)/d beta^2, the genuine thermodynamic
                         susceptibility (>= 0). NOTE this is the specific heat and is NON-MONOTONE in
                         beta: it PEAKS at the "melting" temperature where alpha crosses over from
                         uniform to concentrated, and -> 0 at BOTH extremes (beta->0 uniform, and
                         beta->inf frozen). It is therefore reported as the true physical response,
                         but it is NOT itself a monotone decisiveness score -- use `entropy` and
                         `energy_gap` for that.
        specific_heat  : alias for chi (C = Var_alpha(Psi)); named for the identity.
        entropy        : H(alpha) in nats -- the MONOTONE sharpness of the readout. 0 = one primitive
                         has all the weight (decisive); log(#prims) = uniform (maximally ambiguous).
        entropy_frac   : H(alpha)/log(#prims) in [0,1]; 0 decisive, 1 fully ambiguous. This is the
                         scale-free readout-ambiguity number the verdict keys off.
        alpha          : the Gibbs weights {primitive: prob} at this beta.
        winner         : argmin-energy primitive (== the deployed one).
        runner_up      : second-lowest-energy primitive (or None if <2 primitives).
        energy_gap     : Psi_runnerup - Psi_winner  (nats; larger = more decisive).
        beta           : echoed.
        note           : scope statement.

    chi is a closed form (no finite differences): for a Boltzmann measure the second derivative of
    -log Z w.r.t. beta is exactly the energy variance.
    """
    prims = list(energies)
    if len(prims) == 0:
        return {"chi": 0.0, "specific_heat": 0.0, "entropy": 0.0, "entropy_frac": 0.0,
                "alpha": {}, "winner": None, "runner_up": None, "energy_gap": float("inf"),
                "beta": float(beta), "note": "no primitives"}
    psi = np.array([float(energies[p]) for p in prims], float)
    b = float(beta)
    # numerically-stable softmax of -beta*psi
    z = -b * psi
    z -= z.max()
    w = np.exp(z)
    w_sum = w.sum()
    alpha = w / w_sum if w_sum > 0 else np.full(len(prims), 1.0 / len(prims))
    mean_psi = float(np.sum(alpha * psi))
    var_psi = float(np.sum(alpha * (psi - mean_psi) ** 2))
    H = float(-np.sum(alpha * np.log(alpha + 1e-12)))
    Hmax = math.log(len(prims)) if len(prims) > 1 else 1.0
    order = np.argsort(psi)  # ascending energy: best first
    winner = prims[int(order[0])]
    runner_up = prims[int(order[1])] if len(prims) > 1 else None
    gap = float(psi[int(order[1])] - psi[int(order[0])]) if len(prims) > 1 else float("inf")
    return {
        "chi": var_psi,
        "specific_heat": var_psi,
        "entropy": H,
        "entropy_frac": float(H / Hmax) if Hmax > 0 else 0.0,
        "alpha": {p: float(a) for p, a in zip(prims, alpha)},
        "winner": winner,
        "runner_up": runner_up,
        "energy_gap": gap,
        "beta": b,
        "note": "chi = Var_alpha(Psi) = d^2(-log Z)/d beta^2 is the specific heat (a real "
                "susceptibility, but NON-MONOTONE in beta -- peaks at the melting point). Use "
                "entropy_frac (0 decisive, 1 ambiguous) and energy_gap for the decisiveness read.",
    }


# --------------------------------------------------------------------------------------------------
# Level C -- first-order contract transition spectroscopy
# --------------------------------------------------------------------------------------------------
def _normalized_omega(omegas: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize Omega across the admissible set, matching select_contract_mdl exactly."""
    cs = list(omegas)
    om = np.array([omegas[c] for c in cs], float)
    omn = (om - om.min()) / (np.ptp(om) + 1e-12)
    return {c: float(v) for c, v in zip(cs, omn)}


def contract_transition(scores: Dict[str, float], omegas: Dict[str, float],
                        mu_c: float) -> dict:
    """First-order transition spectroscopy of the priced contract selection.

    Replicates select_contract_mdl's objective  J_c(mu) = (1 - score_c) + mu * Omega_tilde_c  (with
    Omega_tilde the min-max normalized structural code length), then measures -- at the operating
    price mu_c -- the sharpness of the argmin and the distance to the nearest winner-switch.

    scores : {contract: validation score}   (higher = better; same input as the selector).
    omegas : {contract: Omega_struct nats}   (RAW, pre-normalization -- normalized internally).
    mu_c   : the operating contract price.

    Returns a dict with:
        winner         : argmin-J contract at mu_c (== the selected one).
        runner_up      : second-best contract at mu_c (or None).
        J_gap          : J_runnerup - J_winner at mu_c  (>=0; the sharpness of the choice).
        mu_star_up     : the smallest price ABOVE mu_c at which the winner changes (or None if the
                         winner is stable for all larger mu -- i.e. it is the cheapest contract).
        mu_star_down   : the largest price BELOW mu_c at which the winner changes (or None if stable
                         down to mu=0 -- i.e. it is the best-fit contract).
        margin_up      : mu_star_up - mu_c   (how much the price must RISE to flip the choice; None if
                         no upward transition).
        margin_down    : mu_c - mu_star_down (how much the price must FALL to flip; None if none).
        slope_jump     : |Omega_tilde(winner) - Omega_tilde(next_winner)| across the nearest
                         transition -- the change in dJ/dmu, i.e. the finite susceptibility of the
                         first-order line (or None if no transition on either side).
        robust         : True iff BOTH the operating point is not within `tol` of a transition in mu
                         AND the J_gap exceeds a small floor (a calibrated, not ordinal, verdict).
        J              : {contract: J_c(mu_c)}.
        note           : the piecewise-linear / first-order scope statement.
    """
    cs = list(scores)
    R = {c: 1.0 - float(scores[c]) for c in cs}
    omn = _normalized_omega(omegas)

    def J_at(mu: float) -> Dict[str, float]:
        return {c: R[c] + mu * omn[c] for c in cs}

    def winner_at(mu: float) -> str:
        j = J_at(mu)
        return min(cs, key=lambda c: j[c])

    # operating point
    Jc = J_at(mu_c)
    order = sorted(cs, key=lambda c: Jc[c])
    winner = order[0]
    runner_up = order[1] if len(order) > 1 else None
    J_gap = float(Jc[order[1]] - Jc[order[0]]) if len(order) > 1 else float("inf")

    # Pairwise crossing prices: J_a(mu) = J_b(mu)  =>  mu = (R_a - R_b) / (Omega_b - Omega_a).
    # A crossing is a genuine winner-switch only if the crossing pair are the two lowest lines there.
    crossings = []  # (mu_cross, contract_losing, contract_gaining)
    for i in range(len(cs)):
        for jx in range(i + 1, len(cs)):
            a, b = cs[i], cs[jx]
            denom = omn[b] - omn[a]
            if abs(denom) < 1e-12:
                continue  # parallel lines: never cross
            mu_x = (R[a] - R[b]) / denom
            if mu_x < 0:
                continue  # only physical (non-negative) prices
            # confirm this crossing is on the lower envelope (a and b are tied for the min there)
            j = J_at(mu_x)
            jmin = min(j.values())
            if j[a] <= jmin + 1e-9 and j[b] <= jmin + 1e-9:
                crossings.append(float(mu_x))

    crossings = sorted(set(round(c, 12) for c in crossings))
    tol = 1e-6
    ups = [c for c in crossings if c > mu_c + tol]
    downs = [c for c in crossings if c < mu_c - tol]
    mu_star_up = min(ups) if ups else None
    mu_star_down = max(downs) if downs else None
    margin_up = (mu_star_up - mu_c) if mu_star_up is not None else None
    margin_down = (mu_c - mu_star_down) if mu_star_down is not None else None

    # slope jump across the NEAREST transition (in |mu - mu_c|)
    slope_jump = None
    nearest = None
    cand = []
    if mu_star_up is not None:
        cand.append(mu_star_up)
    if mu_star_down is not None:
        cand.append(mu_star_down)
    if cand:
        nearest = min(cand, key=lambda m: abs(m - mu_c))
        # winners just below and just above the nearest crossing
        wlo = winner_at(nearest - max(tol, 1e-6))
        whi = winner_at(nearest + max(tol, 1e-6))
        slope_jump = float(abs(omn[wlo] - omn[whi]))

    # "near a transition" means the price margin is small on the natural scale of the control, which
    # is the operating price mu_c itself (a margin of 0.005 is negligible if mu_c=0.2 but decisive if
    # mu_c=0.05). Use a relative band of 25% of mu_c, with a tiny absolute floor for mu_c -> 0.
    band = max(0.25 * abs(mu_c), 1e-4)
    near_transition = (margin_up is not None and margin_up < band) or \
                      (margin_down is not None and margin_down < band)
    robust = (not near_transition) and (J_gap > 1e-3)

    return {
        "winner": winner,
        "runner_up": runner_up,
        "J_gap": J_gap,
        "mu_star_up": mu_star_up,
        "mu_star_down": mu_star_down,
        "margin_up": margin_up,
        "margin_down": margin_down,
        "slope_jump": slope_jump,
        "robust": bool(robust),
        "J": {c: float(Jc[c]) for c in cs},
        "note": "J_c(mu) is piecewise-LINEAR in mu, so d^2J*/dmu^2 = 0 in the interior; the "
                "curvature lives at the winner-switch kinks. Reported: nearest critical price mu*, "
                "the price margins to flip, and the slope jump (finite susceptibility of the "
                "first-order line). Selection in mu is a sequence of first-order transitions.",
    }


# --------------------------------------------------------------------------------------------------
# Bundle -- both channels + a plain-language robustness read
# --------------------------------------------------------------------------------------------------
def response_spectrum(*, energies: Optional[Dict[str, float]] = None,
                      beta: Optional[float] = None,
                      scores: Optional[Dict[str, float]] = None,
                      omegas: Optional[Dict[str, float]] = None,
                      mu_c: Optional[float] = None) -> dict:
    """Assemble whichever channels are available into one response read-out.

    Provide (energies, beta) for the Level-A readout channel and/or (scores, omegas, mu_c) for the
    Level-C contract channel. Missing a channel's inputs simply omits that channel (a run may have
    only a primitive readout, or only a contract selection, or both).

    Returns {readout, contract, summary} where absent channels are None and `summary` is a short
    human-readable robustness verdict combining both.
    """
    out: Dict[str, object] = {"readout": None, "contract": None, "summary": ""}

    if energies is not None and beta is not None and len(energies) > 0:
        out["readout"] = gibbs_susceptibility(energies, beta)

    if scores is not None and omegas is not None and mu_c is not None and len(scores) > 0:
        out["contract"] = contract_transition(scores, omegas, mu_c)

    bits = []
    r = out["readout"]
    if r is not None and r["runner_up"] is not None:
        amb = r["entropy_frac"]  # 0 decisive, 1 ambiguous (monotone)
        verdict = "decisive" if amb < 0.4 else ("moderate" if amb < 0.8 else "soft")
        bits.append(f"primitive readout {verdict} (H(alpha)/log k={amb:.2f}, specific heat "
                    f"chi={r['chi']:.4f}, energy gap {r['energy_gap']:.3f} nats over '{r['runner_up']}')")
    c = out["contract"]
    if c is not None and c["runner_up"] is not None:
        mtxt = []
        if c["margin_up"] is not None:
            mtxt.append(f"+{c['margin_up']:.3f} to prefer a cheaper contract")
        if c["margin_down"] is not None:
            mtxt.append(f"-{c['margin_down']:.3f} to prefer a richer one")
        mstr = ("; price margins " + ", ".join(mtxt)) if mtxt else "; winner stable for all mu>=0" \
            if (c["margin_up"] is None and c["margin_down"] is None) else ""
        bits.append(f"contract choice '{c['winner']}' {'robust' if c['robust'] else 'near a transition'} "
                    f"(J-gap {c['J_gap']:.4f} over '{c['runner_up']}'{mstr})")
    out["summary"] = "; ".join(bits) if bits else "no perturbable selection to report"
    return out

"""T-RESPONSE (susceptibility spectroscopy / D5): the curvature read-out of the selection objective.

Fast, exact tests -- no training. Two channels with DIFFERENT analytic character, each checked:

  * Level A (smooth readout): chi = Var_alpha(Psi) is the specific-heat identity (a real second
    derivative of -log Z), and the MONOTONE decisiveness read is the entropy fraction (0 decisive,
    1 ambiguous). We check chi matches an independent variance, that entropy_frac orders decisive
    below soft, and the two extreme betas.
  * Level C (first-order contract selection): J_c(mu) = R_c + mu*Omega_tilde_c is piecewise-linear,
    so the observable is the critical price mu* at the winner-switch, the margins to flip, and the
    slope jump. We check mu* against a hand-computed crossing, the relative-scale robustness verdict,
    and the cheapest-contract / best-fit boundary cases (no upward / no downward transition).
"""
import numpy as np
import pytest

from ilmarinen.machinery.response_spectroscopy import (gibbs_susceptibility, contract_transition,
                                                     response_spectrum)


# =============================================================== Level A: readout susceptibility
def test_chi_is_energy_variance():
    """chi == Var_alpha(Psi) computed independently (the specific-heat identity d^2(-log Z)/dbeta^2)."""
    energies = {"a": -0.9, "b": -0.55, "c": -0.5}
    beta = 8.0
    r = gibbs_susceptibility(energies, beta)
    # independent softmax + variance
    psi = np.array([-0.9, -0.55, -0.5])
    w = np.exp(-beta * psi - np.max(-beta * psi))
    a = w / w.sum()
    mean = np.sum(a * psi)
    var = np.sum(a * (psi - mean) ** 2)
    assert r["chi"] == pytest.approx(var, rel=1e-12)
    assert r["specific_heat"] == pytest.approx(r["chi"], rel=1e-12)
    assert r["chi"] >= 0.0


def test_entropy_frac_monotone_decisive_below_soft():
    """The MONOTONE decisiveness read: a clearly-separated (decisive) readout has LOWER entropy_frac
    than a near-tied (soft) one. (chi itself is non-monotone in beta, so it must NOT be used for this.)"""
    decisive = gibbs_susceptibility({"a": -0.9, "b": -0.55, "c": -0.5}, beta=8.0)
    soft = gibbs_susceptibility({"a": -0.72, "b": -0.71, "c": -0.70}, beta=8.0)
    assert decisive["entropy_frac"] < soft["entropy_frac"]
    assert 0.0 <= decisive["entropy_frac"] <= 1.0
    assert 0.0 <= soft["entropy_frac"] <= 1.0
    # soft (near-tied at low beta) is close to fully ambiguous
    assert soft["entropy_frac"] > 0.9


def test_entropy_extremes():
    """entropy_frac -> 0 as beta -> inf (frozen onto the winner); -> 1 as beta -> 0 (uniform)."""
    e = {"a": -0.9, "b": -0.5, "c": -0.4}
    hot = gibbs_susceptibility(e, beta=1e-6)   # ~uniform
    cold = gibbs_susceptibility(e, beta=1e6)   # ~frozen
    assert hot["entropy_frac"] == pytest.approx(1.0, abs=1e-4)
    assert cold["entropy_frac"] == pytest.approx(0.0, abs=1e-6)
    # winner is the min-energy primitive regardless of beta
    assert hot["winner"] == "a" and cold["winner"] == "a"


def test_energy_gap_and_winner():
    """energy_gap = Psi_runnerup - Psi_winner; winner is argmin energy."""
    r = gibbs_susceptibility({"x": -1.0, "y": -0.3, "z": -0.7}, beta=5.0)
    assert r["winner"] == "x"
    assert r["runner_up"] == "z"           # second lowest energy
    assert r["energy_gap"] == pytest.approx(-0.7 - (-1.0), rel=1e-12)  # 0.3


# ============================================================ Level C: first-order contract transition
def test_contract_transition_locates_crossing():
    """mu* matches the hand-computed line crossing. With risk from scores and normalized Omega:
    lines J_c(mu) = (1-score_c) + mu*Omega_tilde_c cross where the winner switches."""
    scores = {"dense": 0.740, "graph": 0.759, "equivariant": 0.756}
    omegas = {"dense": 0.0, "graph": 32.0, "equivariant": 127.0}
    c = contract_transition(scores, omegas, mu_c=0.05)
    assert c["winner"] == "graph"          # at mu=0.05 graph wins on J
    assert c["runner_up"] == "dense"
    # graph (Omega_tilde=32/127) vs dense (Omega_tilde=0): crossing where
    # (1-0.759)+mu*32/127 == (1-0.740)  => mu = (0.260-0.241)*127/32 = 0.0754...
    assert c["mu_star_up"] == pytest.approx(0.0754, abs=1e-3)
    assert c["margin_up"] == pytest.approx(0.0754 - 0.05, abs=1e-3)
    # dense is the cheapest contract -> no upward transition beyond it; graph is not best-fit here so
    # a downward transition does not exist below mu=0.05 either
    assert c["mu_star_down"] is None
    assert c["J_gap"] > 0.0
    assert c["slope_jump"] == pytest.approx(32.0 / 127.0, abs=1e-6)


def test_contract_piecewise_linear_note_and_robust_relative_scale():
    """Robustness is judged on the natural scale of mu (relative band ~25% of mu_c), so a fixed
    absolute margin flips from robust to not-robust as mu_c shrinks."""
    scores = {"dense": 0.740, "graph": 0.759, "equivariant": 0.756}
    omegas = {"dense": 0.0, "graph": 32.0, "equivariant": 127.0}
    # comfortably away from the ~0.075 crossing
    assert contract_transition(scores, omegas, mu_c=0.05)["robust"] is True
    # just past the crossing (winner=dense), margin_down ~0.0045 << 0.25*0.08 -> near-transition
    c_near = contract_transition(scores, omegas, mu_c=0.0799)
    assert c_near["winner"] == "dense"
    assert c_near["robust"] is False
    # deep in the dense region -> robust again
    assert contract_transition(scores, omegas, mu_c=0.20)["robust"] is True


def test_contract_cheapest_and_bestfit_boundaries():
    """The cheapest contract has no UPWARD transition; the best-fit contract has no DOWNWARD one."""
    scores = {"cheap": 0.70, "rich": 0.80}
    omegas = {"cheap": 0.0, "rich": 100.0}
    # at large mu the cheap contract wins and stays winner for all larger mu (no upward switch)
    hi = contract_transition(scores, omegas, mu_c=0.50)
    assert hi["winner"] == "cheap"
    assert hi["mu_star_up"] is None and hi["margin_up"] is None
    # at mu=0 the best-fit (rich) wins and stays winner for all smaller mu (no downward switch)
    lo = contract_transition(scores, omegas, mu_c=0.0)
    assert lo["winner"] == "rich"
    assert lo["mu_star_down"] is None and lo["margin_down"] is None


# ===================================================================== bundle
def test_response_spectrum_bundles_available_channels():
    """response_spectrum includes whichever channels have inputs; omits the rest; summary is non-empty."""
    energies = {"a": -0.9, "b": -0.5}
    scores = {"dense": 0.74, "graph": 0.76}
    omegas = {"dense": 0.0, "graph": 30.0}
    # both channels
    both = response_spectrum(energies=energies, beta=8.0, scores=scores, omegas=omegas, mu_c=0.05)
    assert both["readout"] is not None and both["contract"] is not None
    assert both["summary"]
    # readout only
    ro = response_spectrum(energies=energies, beta=8.0)
    assert ro["readout"] is not None and ro["contract"] is None
    # contract only
    co = response_spectrum(scores=scores, omegas=omegas, mu_c=0.05)
    assert co["readout"] is None and co["contract"] is not None
    # neither
    none = response_spectrum()
    assert none["readout"] is None and none["contract"] is None
    assert none["summary"] == "no perturbable selection to report"


def test_empty_energies_safe():
    """Degenerate inputs do not crash."""
    r = gibbs_susceptibility({}, beta=8.0)
    assert r["winner"] is None and r["chi"] == 0.0

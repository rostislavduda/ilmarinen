"""T-THERMO (one free energy / D2): the single thermodynamic potential and its temperature-consistency check.

Fast, exact tests: the free-energy form identity, the WBIC temperature, the three-level hierarchy, and --
the load-bearing part -- that the consistency check PASSES on principled temperatures and CATCHES the
coupling error (gibbs_beta := 1/log n) and broken nats-bookkeeping (beta_C != 1). No training needed.
"""

import math

import pytest

from ilmarinen.machinery.thermodynamic_potential import (POTENTIAL_LEVELS, wbic_beta, free_energy_form,
                                                       assert_temperature_consistency)


# --------------------------------------------------------------------------- the one form (exact)
def test_wbic_beta_is_inverse_log_n():
    """beta_W = 1/log n (Watanabe's WBIC temperature), matching singular_complexity's convention."""
    assert wbic_beta(1200) == pytest.approx(1.0 / math.log(1200), rel=1e-12)
    # small-n guard: log(max(n,3)), never divides by log of 0/1
    assert wbic_beta(1) == pytest.approx(1.0 / math.log(3), rel=1e-12)


def test_free_energy_form_identity():
    """F = <rho,E> - (1/beta) H(rho): the single functional shape, executable."""
    # uniform 2-measure over energies {1,3}: mean 2, H=log2
    assert free_energy_form(2.0, math.log(2), 1.0) == pytest.approx(2.0 - math.log(2), rel=1e-12)
    # higher beta -> less entropy weight -> F closer to the mean energy
    assert free_energy_form(2.0, math.log(2), 100.0) == pytest.approx(2.0 - math.log(2) / 100.0, rel=1e-12)


def test_three_levels_present():
    """The potential is documented at exactly three coarse-graining levels: weights, primitives, contracts."""
    levels = [L["level"] for L in POTENTIAL_LEVELS]
    assert levels == ["W", "A", "C"]
    names = {L["level"]: L["name"] for L in POTENTIAL_LEVELS}
    assert names["W"] == "weights" and names["A"] == "primitives" and names["C"] == "contracts"


# --------------------------------------------------------------------------- consistency check (exact)
def test_consistency_passes_on_principled_temperatures():
    """beta_W=1/log n (implicit), beta_A=gibbs_beta free knob (8.0), beta_C=1: all principled -> ok."""
    r = assert_temperature_consistency(1200, 8.0, contract_beta=1.0, raise_on_fail=False)
    assert r["ok"]
    assert r["beta_W"] == pytest.approx(1.0 / math.log(1200), rel=1e-12)
    assert r["beta_A"] == 8.0 and r["beta_C"] == 1.0


def test_consistency_catches_coupling_error():
    """The load-bearing guard: setting gibbs_beta := 1/log n (naive 'unification') is flagged, because
    beta_A and beta_W are different kinds of quantity on different spaces and equating them collapses alpha."""
    r = assert_temperature_consistency(1200, wbic_beta(1200), contract_beta=1.0, raise_on_fail=False)
    assert not r["ok"]
    assert any("coupling" in s for s in r["issues"])


def test_consistency_catches_broken_contract_beta():
    """beta_C must be exactly 1 (nats vs nats); a value != 1 means broken nats-bookkeeping."""
    r = assert_temperature_consistency(1200, 8.0, contract_beta=2.0, raise_on_fail=False)
    assert not r["ok"]
    assert any("beta_C" in s for s in r["issues"])


def test_consistency_rejects_nonpositive_gibbs_beta():
    """beta_A must be a positive finite number."""
    r = assert_temperature_consistency(1200, 0.0, contract_beta=1.0, raise_on_fail=False)
    assert not r["ok"]
    r2 = assert_temperature_consistency(1200, -3.0, contract_beta=1.0, raise_on_fail=False)
    assert not r2["ok"]


def test_consistency_raises_when_asked():
    """raise_on_fail=True turns a failed check into a ValueError (default behavior for callers that want it)."""
    with pytest.raises(ValueError):
        assert_temperature_consistency(1200, wbic_beta(1200))  # coupling error, raise_on_fail default True

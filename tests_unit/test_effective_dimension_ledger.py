"""T-LEDGER (effective-dimension ledger / D3): one participation-ratio functional read at ascending
coarse-graining levels, plus lambda at the model level.

Fast, exact tests -- no training. The load-bearing claims: (1) the shared functional matches the two
existing implementations (redundancy_reduction.effective_dimension on a covariance spectrum, and
sparsity_priced_alpha.participation on alpha) EXACTLY -- that identity is the whole point of the
unification; (2) the ledger assembles the available legs onto one axis with distinct, non-summed units;
(3) partial and degenerate inputs are handled.
"""
import numpy as np
import pytest

from ilmarinen.core.redundancy_reduction import effective_dimension
from ilmarinen.machinery.effective_dimension_ledger import (
    LEDGER_LEVELS,
    effective_dimension_ledger,
    participation_ratio,
)
from ilmarinen.machinery.sparsity_priced_alpha import participation


# ============================================================ the shared functional
def test_participation_ratio_extremes():
    """PR = 1 on a one-mode spectrum, = m on a uniform m-mode spectrum."""
    assert participation_ratio([1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-9)
    assert participation_ratio([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0, abs=1e-9)
    assert participation_ratio([5.0]) == pytest.approx(1.0, abs=1e-9)
    # scale-free: multiplying the spectrum by a constant does not change PR
    assert participation_ratio([2.0, 2.0]) == pytest.approx(participation_ratio([7.0, 7.0]), rel=1e-12)


def test_participation_ratio_matches_effective_dimension():
    """The ledger's functional == redundancy_reduction.effective_dimension on the SAME covariance
    spectrum. This identity is the unification: not a re-implementation, the same 1/sum(p^2)."""
    rng = np.random.RandomState(0)
    Z = rng.randn(300, 4)
    X = Z @ rng.randn(4, 15) + 0.05 * rng.randn(300, 15)
    d_eff_ref, var_ratios = effective_dimension(X)
    assert participation_ratio(var_ratios) == pytest.approx(d_eff_ref, rel=1e-9)


def test_participation_ratio_matches_alpha_participation():
    """The ledger's functional == sparsity_priced_alpha.participation on alpha (effective #primitives)."""
    alpha = np.array([0.7, 0.2, 0.05, 0.03, 0.02])
    assert participation_ratio(alpha) == pytest.approx(participation(alpha), rel=1e-6)


def test_participation_ratio_degenerate():
    """Empty / all-zero spectra return 0.0 rather than dividing by zero."""
    assert participation_ratio([]) == 0.0
    assert participation_ratio([0.0, 0.0]) == 0.0


# ============================================================ the ledger assembly
def test_ledger_assembles_available_levels():
    """Each supplied input contributes its level; the axis and units are recorded and NOT summed."""
    rng = np.random.RandomState(1)
    X = rng.randn(200, 4) @ rng.randn(4, 12)
    _, var_ratios = effective_dimension(X)
    alpha = np.array([0.6, 0.3, 0.1])
    llc = {"lambda": 2.5, "half_params": 100.0, "ratio": 0.025, "valid": True}
    led = effective_dimension_ledger(cov_spectrum=var_ratios, alpha=alpha, llc=llc)
    lvls = {l["level"]: l for l in led["levels"]}
    assert set(lvls) == {"data_modes", "primitive_mixture", "model"}
    # values match the functionals
    assert lvls["data_modes"]["value"] == pytest.approx(participation_ratio(var_ratios), rel=1e-9)
    assert lvls["primitive_mixture"]["value"] == pytest.approx(participation_ratio(alpha), rel=1e-9)
    assert lvls["model"]["value"] == pytest.approx(2.5, rel=1e-9)
    # distinct units (the ledger does not conflate them)
    units = {l["unit"] for l in led["levels"]}
    assert len(units) == 3
    # model level is flagged as a DISTINCT functional, not a participation ratio
    assert "RLCT" in lvls["model"]["note"] or "not a participation ratio" in lvls["model"]["note"]


def test_ledger_partial_inputs():
    """A level is omitted when its input is missing; empty when nothing is supplied."""
    only_alpha = effective_dimension_ledger(alpha=np.array([0.5, 0.5]))
    assert [l["level"] for l in only_alpha["levels"]] == ["primitive_mixture"]
    only_model = effective_dimension_ledger(llc={"lambda": 1.0, "valid": True})
    assert [l["level"] for l in only_model["levels"]] == ["model"]
    assert effective_dimension_ledger()["levels"] == []


def test_ledger_ib_flow_level():
    """An IB-RG flow dict contributes the supervised scale-resolved data-modes level."""
    flow = {"d_IB": np.array([1, 2, 3, 3]), "transitions": [(0.5, 0), (0.7, 1)], "d_eff_static": 3.1,
            "betas": np.array([1, 2, 3, 4])}
    led = effective_dimension_ledger(ib_flow=flow)
    lv = led["levels"][0]
    assert lv["level"] == "data_modes_flow"
    assert lv["supervised"] is True
    assert lv["n_transitions"] == 2
    assert lv["d_eff_static"] == pytest.approx(3.1, rel=1e-9)


def test_ledger_invalid_lambda_not_certified():
    """A non-converged lambda is reported but flagged uncertified (the B2 guard, surfaced in the ledger)."""
    led = effective_dimension_ledger(llc={"lambda": -3.0, "valid": False})
    assert led["levels"][0]["level"] == "model"
    assert "invalid" in led["levels"][0]["note"] or "not certified" in led["levels"][0]["note"]


def test_ledger_levels_metadata():
    """LEDGER_LEVELS documents the four levels along the coarse-graining axis."""
    levels = [l["level"] for l in LEDGER_LEVELS]
    assert levels == ["data_modes", "data_modes_flow", "primitive_mixture", "model"]
    # exactly one leg is a non-participation-ratio functional (lambda)
    non_pr = [l for l in LEDGER_LEVELS if "participation ratio" not in l["functional"]]
    assert len(non_pr) == 2  # d_IB staircase and lambda are not the PR functional

"""T-GA: the derived Gibbs/Boltzmann readout over primitives.

alpha_p ∝ exp(-beta * energy_p). These lock the Boltzmann invariants the readout stage depends on.
"""

import pytest

from ilmarinen.machinery import gibbs_alpha_select

PRIMS = ["a", "b", "c", "d"]
SCORES = {"a": 1.0, "b": 0.5, "c": 0.2, "d": -0.3}


def _alpha(beta):
    return gibbs_alpha_select(lambda p: SCORES[p], PRIMS, beta=beta)["alpha"]


def test_alpha_is_simplex_point():
    """T-GA-1: alpha sums to 1 and is strictly positive."""
    a = _alpha(8.0)
    assert sum(a.values()) == pytest.approx(1.0, rel=1e-9)
    assert all(v > 0 for v in a.values())


def test_temperature_limits():
    """T-GA-2: beta->inf concentrates on the argmax; beta->0 approaches uniform."""
    hot = _alpha(1e6)
    assert hot["a"] > 0.999  # 'a' has the highest score -> lowest energy
    cold = _alpha(1e-6)
    for p in PRIMS:
        assert cold[p] == pytest.approx(1.0 / len(PRIMS), abs=1e-2)


def test_best_is_argmax():
    """T-GA-3: 'best' matches the argmax of the scores."""
    sel = gibbs_alpha_select(lambda p: SCORES[p], PRIMS, beta=8.0)
    assert sel["best"] == max(SCORES, key=SCORES.get)


def test_determinism():
    """T-GA-4: identical inputs -> identical alpha (no hidden RNG in the readout)."""
    a1 = _alpha(8.0)
    a2 = _alpha(8.0)
    assert a1 == a2


def test_monotone_in_score():
    """T-GA-5: raising one primitive's score never decreases its alpha."""
    base = _alpha(8.0)
    bumped_scores = dict(SCORES, c=SCORES["c"] + 1.0)
    bumped = gibbs_alpha_select(lambda p: bumped_scores[p], PRIMS, beta=8.0)["alpha"]
    assert bumped["c"] >= base["c"]

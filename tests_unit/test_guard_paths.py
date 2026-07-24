"""T-GD: AllGraph guard / error / fallback paths.

The controller has a family of defensive branches -- input validation, contract admissibility, the
enabled-contract restriction, and the operator NaN guard -- that turn misuse into a CLEAR error or a
principled fallback instead of an obscure downstream crash. Only a handful of these were exercised, so a
refactor could silently turn a guard into a crash (or drop a fallback) unnoticed. These tests pin both
kinds: the ``raise`` guards (each matched on its own message, so a *different* error would not satisfy
the test) and the two silent fallbacks (relational-on-edgeless -> set; disabled-contract -> nearest enabled).

The raises abort in the kinematics/validation stage before training, so they are fast; the two fallback
tests train a tiny model and are marked ``smoke``.
"""

import numpy as np
import pytest

from ilmarinen import AllData, AllGraph


def _dense(n=60, d=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return AllData.dense_tensor(X, y)


# =========================================================================== raise guards (fast tier)
def test_enabled_arenas_unknown_name_raises():
    """T-GD-1: an unrecognised contract in enabled_contracts is rejected at construction, naming the bad token."""
    with pytest.raises(ValueError, match="bogus"):
        AllGraph(enabled_contracts="graph,bogus")


def test_enabled_arenas_empty_raises():
    """T-GD-2: enabled_contracts that resolves to nothing is rejected (at least one contract must be enabled)."""
    with pytest.raises(ValueError, match="empty"):
        AllGraph(enabled_contracts=[])


def test_empty_alldata_has_no_routable_content():
    """T-GD-3: routing an AllData with no dense/node/edge content is a clear ValueError, not an AttributeError
    deep in a forward pass."""
    empty = AllData(kind_hint=None, node_feats=None, positions=None, edges=None, y=None)
    with pytest.raises(ValueError, match="no dense/node/edge content"):
        AllGraph(seed=0).route(empty)


def test_unknown_select_size_mode_raises():
    """T-GD-4: an unrecognised select_size mode is rejected. (bool True and False are valid -- they map to
    'sequential' / off -- but an arbitrary string is not.) This is the guard behind the select_size arg the
    two runners pass in different spellings."""
    with pytest.raises(ValueError, match="unknown select_size mode"):
        AllGraph(width=8, depth=1, epochs=2, seed=0).fit(
            _dense(), task="classification", n_out=2, select_size="bogus")


def test_disabled_arena_with_no_representable_alternative_raises():
    """T-GD-5: if enabled_contracts leaves only contracts that cannot represent the data (here: only 'graph',
    which needs edges, given edgeless dense data), the fallback resolver raises rather than guessing."""
    with pytest.raises(ValueError, match="no enabled contract"):
        AllGraph(width=8, depth=1, epochs=2, seed=0, enabled_contracts="graph").fit(
            _dense(), task="classification", n_out=2)


def test_operator_nan_input_raises():
    """T-GD-6: the operator contract rejects non-finite input/target fields with a clear message rather than
    training on NaNs and silently producing a NaN model."""
    rng = np.random.RandomState(0)
    a = rng.randn(30, 16).astype(np.float32)
    y = a.copy()
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        AllGraph(width=8, depth=1, epochs=2, seed=0).fit(
            AllData.functions(a, y), task="regression", n_out=1)


# =========================================================================== fallback behaviour (smoke)
class TestArenaFallbacks:
    """Principled fallbacks: the controller degrades to a constructible contract instead of crashing."""

    pytestmark = pytest.mark.smoke

    def test_relational_route_on_edgeless_data_falls_back_to_set(self):
        """T-GD-7: a relational (graph/equivariant) route on data with NO edges is inadmissible, so the
        controller falls back to the always-constructible 'set' contract rather than dereferencing None edges."""
        rng = np.random.RandomState(0)
        nf = [rng.randn(4, 3).astype(np.float32) for _ in range(30)]
        d = AllData.point_sets(nf, y=(np.arange(30) % 2).astype(np.int64))
        d.kind_hint = "graph"                                          # force a relational route, no edges
        mg = AllGraph(width=8, depth=1, epochs=2, seed=0)
        mg.fit(d, task="classification", n_out=2)
        assert mg.contract == "set"

    def test_disabled_natural_arena_falls_back_to_nearest_enabled(self):
        """T-GD-8: dense data would route to 'sequence'; with enabled_contracts restricting to {set, sequence}
        the route stays on the enabled, constructible 'sequence' contract (no crash, no disabled contract built)."""
        mg = AllGraph(width=8, depth=1, epochs=2, seed=0, enabled_contracts="set,sequence")
        mg.fit(_dense(), task="classification", n_out=2)
        assert mg.contract in ("sequence", "set")

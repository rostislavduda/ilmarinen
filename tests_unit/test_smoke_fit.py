"""T-SM: end-to-end smoke tests (marked @smoke; ~1 min total on CPU).

These run a minimal AllGraph.fit per contract and assert the *result contract* (well-formed output,
correct contract, deterministic selection) -- not accuracy, except one wide-margin better-than-chance
guard on a clean synthetic signal. Real-performance baselines (ESOL, Burgers) stay in the runners.
"""

import numpy as np
import pytest

from ilmarinen import AllData, AllGraph

pytestmark = pytest.mark.smoke

# keys present in every contract's result (probed across dense/set/graph/operator). Note 'readout'
# is dense/sequence-only; 'selected_primitive' + 'metric' are the universal selection fields.
RESULT_KEYS = {"contract", "value", "architecture", "n_params", "route", "selected_primitive", "metric"}


def _fit(data, task="classification", n_out=2, **kw):
    mg = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0, **kw)
    return mg.fit(data, task=task, n_out=n_out)


def test_dense_tensor_fit_contract(linsep_tabular):
    """T-SM-1: dense/tabular fit returns a well-formed result with expected keys."""
    X, y = linsep_tabular()
    r = _fit(AllData.dense_tensor(X, y), n_out=2)
    assert RESULT_KEYS.issubset(r.keys()), r.keys()
    assert np.isfinite(r["value"])
    assert len(r["architecture"]) == 1  # depth == 1


def test_point_sets_fit_geometric(point_cloud_sets):
    """T-SM-2: point-set input routes to a geometric contract and fits."""
    node_feats, pts, y = point_cloud_sets()
    r = _fit(AllData.point_sets(node_feats, y=y, positions=pts), n_out=2)
    assert r["contract"] in ("set", "equivariant"), r["contract"]
    assert RESULT_KEYS.issubset(r.keys())


def test_graph_fit(small_graphs):
    """T-SM-3: graph input fits and reports the graph contract."""
    node_feats, edges, y = small_graphs()
    r = _fit(AllData.graphs(node_feats, edges, y=y), n_out=2)
    assert r["contract"] == "graph", r["contract"]


def test_operator_fit(function_samples):
    """T-SM-4: function/operator input fits and reports the operator contract.

    grid coordinates default to a uniform meshgrid when omitted (the `grid` arg expects coordinate
    arrays, not a size), which is the intended usage for a regular 1D grid.
    """
    a, y, _grid = function_samples()
    r = _fit(AllData.functions(a, y), task="regression", n_out=1)
    assert r["contract"] == "operator", r["contract"]


def test_fit_determinism(linsep_tabular):
    """T-SM-5: same seed -> same contract and same architecture (selection pipeline is deterministic)."""
    X, y = linsep_tabular()
    r1 = _fit(AllData.dense_tensor(X, y), n_out=2)
    r2 = _fit(AllData.dense_tensor(X, y), n_out=2)
    assert r1["contract"] == r2["contract"]
    assert r1["architecture"] == r2["architecture"]


def test_fit_better_than_chance(linsep_tabular):
    """T-SM-6: on a clean linearly-separable signal, accuracy beats chance by a wide margin.
    Guards a catastrophic training regression without being brittle."""
    X, y = linsep_tabular(n=120)
    mg = AllGraph(width=16, depth=1, epochs=25, verbose=False, seed=0)
    r = mg.fit(AllData.dense_tensor(X, y), task="classification", n_out=2)
    # value is accuracy for classification; chance is 0.5 -> require a clear margin
    assert r["value"] > 0.65, f"suspiciously low accuracy on separable data: {r['value']}"

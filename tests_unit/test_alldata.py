"""T-MD: AllData constructor contracts.

AllData signals the contract through which attributes are populated (positions / edges / grid /
kind_hint), not a single 'contract' field. These tests lock that input contract so routing
downstream stays well-defined.
"""

import numpy as np

from ilmarinen import AllData


def test_dense_tensor_basic():
    """T-MD-1: dense_tensor stores the tensor; y optional (unsupervised path allowed)."""
    X = np.random.RandomState(0).randn(20, 8).astype(np.float32)
    md = AllData.dense_tensor(X)
    assert md.dense is not None
    assert md.y is None
    # supervised variant
    y = (X[:, 0] > 0).astype(np.int64)
    md2 = AllData.dense_tensor(X, y)
    assert md2.y is not None and len(md2.y) == len(X)
    # a dense tensor has no relational/geometric/grid structure by default
    assert md.edges is None and md.positions is None and md.grid is None


def test_point_sets_marks_positions(point_cloud_sets):
    """T-MD-3: point_sets records positions (enables the geometric contract)."""
    node_feats, pts, y = point_cloud_sets()
    md = AllData.point_sets(node_feats, y=y, positions=pts)
    assert md.positions is not None
    assert np.asarray(md.positions).shape == pts.shape


def test_graphs_stores_edges(small_graphs):
    """T-MD-2: graphs records edges; positions default absent."""
    node_feats, edges, y = small_graphs()
    md = AllData.graphs(node_feats, edges, y=y)
    assert md.edges is not None
    assert md.positions is None  # no geometry unless supplied


def test_functions_sets_grid(function_samples):
    """T-MD-4: functions builds grid coordinates (operator contract).

    With grid omitted, a uniform meshgrid of shape (n, *field_shape, spatial_dims) is constructed.
    """
    a, y, _grid = function_samples()
    md = AllData.functions(a, y)
    assert md.grid is not None
    assert md.grid.shape[0] == a.shape[0]  # one coordinate grid per sample


def test_constructors_return_metadata():
    """T-MD-5: all four constructors return a AllData instance (stable factory contract)."""
    X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    assert isinstance(AllData.dense_tensor(X, y), AllData)

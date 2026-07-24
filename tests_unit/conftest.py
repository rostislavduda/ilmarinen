"""Shared fixtures and helpers for the ilmarinen unit suite.

Everything here is tiny and in-process: no network, no dataset downloads, no large tensors.
The suite locks down behaviour and the physics invariants so later changes (notably the planned
LLC-into-Omega work) cannot silently break essentials. Slow end-to-end ``fit`` tests are marked
``@pytest.mark.smoke`` and can be deselected with ``pytest -m "not smoke"``.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- determinism
@pytest.fixture(autouse=True)
def _fixed_seeds():
    """Seed numpy and torch before every test for reproducibility."""
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(False)  # some ops lack deterministic kernels; we assert bands
    except Exception:
        pass
    yield


# --------------------------------------------------------------------------- tiny synthetic data
@pytest.fixture
def linsep_tabular():
    """A small, linearly separable 2-class tabular problem (fast, has real signal)."""

    def _make(n=80, d=8, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d).astype(np.float32)
        # label depends on a clean linear combination of two features -> better-than-chance is easy
        y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(np.int64)
        return X, y

    return _make


@pytest.fixture
def tabular_multiclass():
    def _make(n=60, d=8, k=3, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d).astype(np.float32)
        y = (rng.rand(n) * k).astype(np.int64)
        return X, y

    return _make


@pytest.fixture
def point_cloud_sets():
    """Small batch of 3D point-set data (geometric path)."""

    def _make(n=40, m=6, seed=0):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, m, 3).astype(np.float32)
        node_feats = np.zeros((n, m, 1), np.float32)
        y = (pts.sum(axis=(1, 2)) > 0).astype(np.int64)
        return node_feats, pts, y

    return _make


@pytest.fixture
def small_graphs():
    """A few tiny graphs (node feats + edge list), no positions."""

    def _make(n_graphs=12, n_nodes=6, seed=0):
        rng = np.random.RandomState(seed)
        node_feats, edges, ys = [], [], []
        for g in range(n_graphs):
            nf = rng.randn(n_nodes, 4).astype(np.float32)
            # a simple ring + a couple random chords
            e = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
            e += [(0, n_nodes // 2)]
            node_feats.append(nf)
            edges.append(np.array(e, dtype=np.int64).T)  # (2, E)
            ys.append(int(nf[:, 0].sum() > 0))
        return node_feats, edges, np.array(ys, dtype=np.int64)

    return _make


@pytest.fixture
def function_samples():
    """1D function->function data for the operator contract.

    The neural-operator contract maps an input field a(x) to a target field u(x) on the SAME grid,
    so both a and y have shape (n, grid). Here u is a simple smoothing of a (a well-posed operator).
    """

    def _make(n=40, grid=16, seed=0):
        rng = np.random.RandomState(seed)
        a = rng.randn(n, grid).astype(np.float32)  # input functions on a 1D grid
        # target field: a mild local average (a genuine function-to-function map on the grid)
        kernel = np.array([0.25, 0.5, 0.25], np.float32)
        y = np.stack([np.convolve(row, kernel, mode="same") for row in a]).astype(np.float32)
        return a, y, grid

    return _make


# --------------------------------------------------------------------------- so(3) generators
@pytest.fixture
def so3_generators():
    """The three standard so(3) generators as raw 3x3 matrices."""
    Gx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    Gy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float)
    Gz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)
    return [Gx, Gy, Gz]

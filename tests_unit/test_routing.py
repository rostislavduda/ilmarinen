"""T-RT: structural routing (route_by_structure).

Locks that routing is deterministic, respects an explicit override, and returns the documented
dict contract. Routing is the 'kinematics' step -- it must be fixed before anything is trained.
"""

import numpy as np

from ilmarinen import route_by_structure

RKEYS = {"kind", "structure", "shape", "tensorize", "build_hint", "detection"}


def test_returns_expected_keys():
    """T-RT: the routing result carries the documented fields."""
    X = np.random.RandomState(0).randn(50, 8).astype(np.float32)
    r = route_by_structure(X)
    assert isinstance(r, dict)
    assert RKEYS.issubset(r.keys()), r.keys()


def test_determinism():
    """T-RT-1: same input -> same routing decision every call."""
    X = np.random.RandomState(1).randn(40, 12).astype(np.float32)
    r1 = route_by_structure(X)
    r2 = route_by_structure(X)
    assert r1["kind"] == r2["kind"] and r1["shape"] == r2["shape"]


def test_force_override():
    """T-RT-4: force= overrides the structural decision (explicit user contract respected)."""
    X = np.random.RandomState(2).randn(30, 9).astype(np.float32)
    auto = route_by_structure(X)["kind"]
    # pick a target different from the automatic one where possible
    target = "sequence" if auto != "sequence" else "set"
    forced = route_by_structure(X, force=target)
    assert forced["kind"] == target


def test_flat_table_routes_consistently():
    """T-RT-2: a plain flat table routes to a stable non-geometric contract."""
    X = np.random.RandomState(3).randn(60, 10).astype(np.float32)
    r = route_by_structure(X)
    # no positions/edges were given, so the routed kind should be a non-relational, non-geometric one
    assert r["kind"] not in ("graph", "equivariant")

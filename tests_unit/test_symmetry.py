"""T-SY: symmetry machinery invariants.

Covers the equivariance-contract building blocks: continuous-invariant features actually absorb
the group action, and canonicalization is idempotent (it is reused at train and test, so double
application must be a no-op).
"""

import numpy as np
from scipy.linalg import expm

from ilmarinen import continuous_invariant_features
from ilmarinen.core.canonicalization import canonicalize_positions


def _as_np(x):
    return x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)


def test_continuous_invariant_features_rotation_invariant(so3_generators):
    """T-SY-1: features are invariant under a rotation in the generated plane."""
    Gz = so3_generators[2]  # rotation in the xy-plane
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3).astype(np.float32)
    R = expm(0.7 * Gz)

    feat, _axes = continuous_invariant_features(X, [Gz])
    feat_rot, _ = continuous_invariant_features((X @ R.T).astype(np.float32), [Gz])
    drift = np.abs(_as_np(feat) - _as_np(feat_rot)).max()
    assert drift < 1e-4, f"features not rotation-invariant, drift={drift}"


def test_continuous_invariant_features_reduces_dimension(so3_generators):
    """T-SY-1b: an xy-rotation generator collapses (x,y) -> radius, so a 3D cloud yields 2 features."""
    Gz = so3_generators[2]
    X = np.random.RandomState(1).randn(15, 3).astype(np.float32)
    feat, axes = continuous_invariant_features(X, [Gz])
    assert _as_np(feat).shape[1] == 2  # (radius, z)
    assert any(tag == "radius" for tag, _ in axes)


def test_canonicalize_positions_idempotent():
    """T-SY-3: canonicalizing already-canonical clouds is a no-op (consistency invariant)."""
    rng = np.random.RandomState(0)
    positions = [rng.randn(15, 3).astype(np.float32) for _ in range(5)]
    c1, _frac1 = canonicalize_positions(positions)
    c2, _frac2 = canonicalize_positions(c1)
    drift = max(np.abs(_as_np(a) - _as_np(b)).max() for a, b in zip(c1, c2))
    assert drift < 1e-4, f"canonicalization not idempotent, drift={drift}"


def test_canonicalize_positions_centers_clouds():
    """T-SY-3b: canonical clouds are mean-centered."""
    rng = np.random.RandomState(2)
    positions = [rng.randn(20, 3).astype(np.float32) + 5.0 for _ in range(3)]  # off-center
    canon, _frac = canonicalize_positions(positions)
    for cloud in canon:
        assert np.abs(_as_np(cloud).mean(axis=0)).max() < 1e-4

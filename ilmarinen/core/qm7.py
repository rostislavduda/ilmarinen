"""QM7 molecular dataset loader (raw 3D coordinates).

QM7 is 7165 small organic molecules (up to 23 atoms; elements H, C, N, O, S) with atomization
energies. Distributed as a .mat file with arrays:
  X (n,23,23) Coulomb matrices   -- ALREADY rotation-invariant by construction; NOT used here
  R (n,23,3)  raw 3D coordinates -- what we load (invariance NOT pre-baked, so it can be tested)
  Z (n,23)    atomic charges (0 = padding)
  T (1,n)     atomization energies (kcal/mol)
  P (5,1433)  cross-validation splits

We deliberately load R + Z, not X: the whole point of the symmetry-discovery pipeline is to
detect/exploit SO(3) rotation invariance, which the Coulomb matrix X would pre-bake away.

The .mat file is not redistributed with the package; point `path` at a local copy (e.g. the
canonical quantum-machine.org / deepchem qm7.mat).
"""

from __future__ import annotations

import numpy as np

QM7_ELEMENTS = [1, 6, 7, 8, 16]  # H, C, N, O, S


def load_qm7(path, max_atoms=23, center=True):
    """Load QM7 from a .mat file. Returns (coords, types, mask, energies):

    coords   : (n, max_atoms, 3) float32, per-molecule centered 3D coordinates
    types    : (n, max_atoms, 5) float32, one-hot atom type over [H,C,N,O,S]
    mask     : (n, max_atoms)    float32, 1 for real atoms, 0 for padding
    energies : (n,)              float32, atomization energy (kcal/mol)
    """
    from scipy.io import loadmat

    d = loadmat(path)
    R = d["R"].astype(np.float32)
    Z = d["Z"].astype(np.int64)
    y = d["T"].astype(np.float32).ravel()
    n, A, _ = R.shape
    mask = (Z > 0).astype(np.float32)
    types = np.zeros((n, A, len(QM7_ELEMENTS)), np.float32)
    for zi, z in enumerate(QM7_ELEMENTS):
        types[..., zi] = (Z == z).astype(np.float32)
    if center:
        for i in range(n):
            m = mask[i].astype(bool)
            R[i, m] -= R[i, m].mean(0, keepdims=True)
            R[i, ~m] = 0.0
    return R, types, mask, y


def random_rotation(seed=None):
    """A uniformly-random 3x3 rotation matrix (det +1), for the rotated-test invariance check."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, Rm = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(Rm)))  # fix signs
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.astype(np.float32)

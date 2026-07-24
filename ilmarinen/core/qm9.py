"""QM9 loader. QM9 (Ramakrishnan et al. 2014) is QM7's superset from the same GDB series: ~134k stable
organic molecules with up to 9 heavy atoms, element set CHONF (vs QM7's CHONS with <=7 heavy atoms),
each with DFT-computed properties including U0 (atomization/internal energy at 0K). The SO(3) rotational
symmetry of molecular energy is identical to QM7's -- only the molecule set differs.

QM9 ships as one extended-XYZ file per molecule (133,885 files, the 'dsgdb9nsd' set). This loader parses
that native format. NOTE on access: the canonical QM9 hosts (figshare, deepchem S3, quantum-machine.org,
Kaggle) may be network-restricted; obtain the dsgdb9nsd.xyz.tar.bz2 and extract to a directory, then
point load_qm9_dir at it.
"""
from __future__ import annotations

import os

import numpy as np

QM9_ELEMENTS = ['H', 'C', 'N', 'O', 'F']
_ZMAP = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
# QM9 property-line columns (index into whitespace-split line 2), the U0 internal energy at 0K is idx 12:
# tag gdb_idx A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv
_U0_COL = 12


def parse_qm9_xyz(text):
    """Parse a single QM9 extended-XYZ file. Returns (coords (N,3) float32, atomic_numbers (N,) int64,
    props list[str]). QM9 encodes exponents as '*^' (e.g. '1.2*^-3'); we normalize to 'e'."""
    lines = text.strip().split('\n')
    na = int(lines[0])
    props = lines[1].split()
    coords, Z = [], []
    for i in range(2, 2 + na):
        p = lines[i].split()
        Z.append(_ZMAP[p[0]])
        coords.append([float(x.replace('*^', 'e')) for x in p[1:4]])
    return np.asarray(coords, np.float32), np.asarray(Z, np.int64), props


def load_qm9_dir(path, max_files=None, center=True):
    """Load a directory of QM9 .xyz files. Returns (coords_list, Z_list, U0 array). coords are per-
    molecule (variable N_atoms, 3); Z_list per-molecule atomic numbers; U0 the 0K internal energy.
    Variable atom count is intentional (QM9 molecules differ in size); pad/mask downstream as needed."""
    files = sorted(f for f in os.listdir(path) if f.endswith('.xyz'))
    if max_files:
        files = files[:max_files]
    C, Zs, U0 = [], [], []
    for f in files:
        c, z, props = parse_qm9_xyz(open(os.path.join(path, f)).read())
        if center:
            c = c - c.mean(0, keepdims=True)
        C.append(c)
        Zs.append(z)
        U0.append(float(props[_U0_COL]) if len(props) > _U0_COL else 0.0)
    return C, Zs, np.asarray(U0, np.float32)


def pad_qm9(coords_list, Z_list, max_atoms=29):
    """Pad variable-size QM9 molecules to a fixed (n, max_atoms, *) tensor form matching the QM7 loader
    contract: returns (coords (n,max_atoms,3), types_onehot (n,max_atoms,5 over CHONF), mask
    (n,max_atoms)). max_atoms=29 covers QM9's largest molecules."""
    n = len(coords_list)
    R = np.zeros((n, max_atoms, 3), np.float32)
    T = np.zeros((n, max_atoms, len(QM9_ELEMENTS)), np.float32)
    M = np.zeros((n, max_atoms), np.float32)
    znum = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    for i, (c, z) in enumerate(zip(coords_list, Z_list)):
        na = len(z)
        R[i, :na] = c
        M[i, :na] = 1.0
        for j, zz in enumerate(z):
            T[i, j, znum[int(zz)]] = 1.0
    return R, T, M

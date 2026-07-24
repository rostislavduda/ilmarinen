"""Shared real-dataset registry for the validation pipelines. Every real dataset investigated in the
project, each as a uniform entry that yields (AllData_train, AllData_test, meta) so both the quick and
the standard runners draw from ONE source of truth.

Each entry is a callable loader(reduced: bool, device) -> dict with keys:
  train   : AllData for training
  test    : AllData for held-out test (or None to split from train)
  task    : 'classification' | 'regression'
  chance  : trivial-baseline metric (for the skill normalization)
  field   : natural-science / medical field
  sota    : short SOTA reference string
  rotated : (equivariant only) bool -- apply a random rotation at test (true SO(3) test)

'reduced=True' returns small subsets / caps for the quick runner; 'reduced=False' returns full data.
Loaders raise FileNotFoundError / ImportError if their data isn't present; the runner skips those.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- helpers
def _split(n, frac=0.8, seed=0):
    idx = np.random.RandomState(seed).permutation(n)
    k = int(frac * n)
    return idx[:k], idx[k:]


# --------------------------------------------------------------------------- quick-suite data scaling
# A single global multiplier applied to every loader's REDUCED (quick) subset size. The quick runner sets it
# from --data_scale before loading, so one flag dials the whole smoke suite's diagnostic up (larger subsets ->
# slower, more accurate / closer to full-data skill) or down (smaller -> faster, rougher). Full (reduced=False)
# sizes are never touched. Because dialing UP must load MORE data, the scale is applied INSIDE each loader (via
# qscale), not by post-subsampling the returned data. Default 1.0 preserves the current tuned sizes exactly.
_QUICK_SCALE = 1.0


def set_quick_scale(s):
    """Set the global reduced-size multiplier (>0). Called once by the quick runner from --data_scale."""
    global _QUICK_SCALE
    _QUICK_SCALE = max(1e-3, float(s))


def qscale(n, lo=8):
    """Scale a reduced-mode size `n` by the global quick-scale, rounding to an int and flooring at `lo` so a
    small --data_scale can't collapse a dataset below a usable minimum. No-op (returns int(n)) at scale 1.0.
    Use inside loaders on the reduced branch: e.g. `ntr = qscale(2000) if reduced else len(...)`."""
    return max(int(lo), int(round(n * _QUICK_SCALE)))


def _mol_graphs_from_qm7(n_max, cutoff=3.5):
    """QM7 molecules -> (node_feats, edges, positions, y) for graph/equivariant/set use."""
    from .qm7_graph import build_qm7_equivariant
    graphs, ys = build_qm7_equivariant(n_max=n_max, cutoff_bohr=cutoff)
    nf = [g["x"].numpy() for g in graphs]
    ed = [g["edge_index"].numpy() for g in graphs]
    pos = [g["pos"].numpy() for g in graphs]
    return nf, ed, pos, ys.astype(np.float32)


# --------------------------------------------------------------------------- SEQUENCE
def _seq_ucr(name, field, sota, reduced, max_T=400, max_test=1000):
    """Load a UCR/aeon time-series classification dataset. In reduced (quick) mode two GPU-time caps keep a
    long-series or large-test dataset from dominating the smoke run: the series is linspace-subsampled to at
    most `max_T` timesteps (a strided view preserving the global shape -- e.g. ACSF1 1460 -> 400, the single
    biggest quick-suite cost), and the test split is linspace-subsampled to at most `max_test` samples (e.g.
    ECG5000 4500 -> 1000). Both are evenly-spaced so class proportions and the accuracy diagnostic stay
    representative while the dominant cost (per-sample length and per-epoch eval size) drops. Full mode is
    unchanged (whole UCR train/test)."""
    import torch
    from aeon.datasets import load_classification

    from .allgraph import AllData
    Xtr, ytr = load_classification(name, split="train")
    Xte, yte = load_classification(name, split="test")
    cls = sorted(set(ytr)); m = {c: i for i, c in enumerate(cls)}
    def prep(X):
        X = np.transpose(X, (0, 2, 1)).astype(np.float32)
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        return X
    Xtr, Xte = prep(Xtr), prep(Xte)
    ytr = np.array([m[c] for c in ytr]); yte = np.array([m[c] for c in yte])
    if reduced:
        mT, mTest = qscale(max_T, lo=64), qscale(max_test, lo=100)   # --data_scale dials the length + test caps
        T = Xtr.shape[1]
        if T > mT:                                       # long series -> strided (linspace) subsample of time axis
            ti = np.linspace(0, T - 1, mT).astype(int)
            Xtr, Xte = Xtr[:, ti], Xte[:, ti]
        if len(Xte) > mTest:                             # large test split -> cap per-epoch eval cost
            ki = np.linspace(0, len(Xte) - 1, mTest).astype(int)
            Xte, yte = Xte[ki], yte[ki]
    return {"train": AllData.dense_tensor(torch.tensor(Xtr), y=ytr),
            "test": AllData.dense_tensor(torch.tensor(Xte), y=yte),
            "task": "classification", "chance": 1.0 / len(cls), "field": field, "sota": sota}


def load_ecg5000(reduced, device):
    # UCR cardiac-ECG beats via aeon (same python-native path as the other UCR sequence datasets); the old
    # timeseriesclassification.com zip mirror now returns HTTP 403.
    return _seq_ucr("ECG5000", "cardiology (ECG)", "acc ~0.94-0.95 (ResNet/InceptionTime/TCN)", reduced)


def load_gunpoint(reduced, device):
    return _seq_ucr("GunPoint", "motion capture", "acc ~0.99 (HIVE-COTE/ROCKET)", reduced)


def load_osuleaf(reduced, device):
    return _seq_ucr("OSULeaf", "botany", "acc ~0.97 (HIVE-COTE 2.0), ~0.96 (MultiRocket), ~0.94 (ROCKET)", reduced)


def load_italypower(reduced, device):
    return _seq_ucr("ItalyPowerDemand", "energy sensor", "acc ~0.97 (ROCKET/InceptionTime)", reduced)


def load_acsf1(reduced, device):
    return _seq_ucr("ACSF1", "appliance power", "acc ~0.88 (ROCKET/MultiRocket)", reduced)


def load_basicmotions(reduced, device):
    return _seq_ucr("BasicMotions", "accelerometry (6ch)", "acc ~1.0 (ROCKET/multivariate)", reduced)


# --------------------------------------------------------------------------- SPATIAL
def load_bloodmnist(reduced, device):
    import torch

    from .allgraph import AllData
    from .benchmark_datasets import load_bloodmnist as _l
    d = _l()
    def zc(x):
        m = x.reshape(-1, 3, 784).mean((0, 2))[None, :, None, None]
        s = x.reshape(-1, 3, 784).std((0, 2))[None, :, None, None]
        # .contiguous(): the images arrive channels-last (from the loader's (0,3,1,2) transpose), which
        # yields a non-contiguous NCHW tensor. CPU conv/BatchNorm tolerate it, but MPS' batchnorm backward
        # rejects non-contiguous input ("view size is not compatible ..."), so make it contiguous here.
        return torch.tensor((x - m) / (s + 1e-6)).contiguous()
    ntr = qscale(2000) if reduced else len(d["train_y"])
    return {"train": AllData.dense_tensor(zc(d["train_x"])[:ntr], y=d["train_y"][:ntr]),
            "test": AllData.dense_tensor(zc(d["test_x"]), y=d["test_y"]),
            "task": "classification", "chance": 0.125, "field": "hematology", "report_auc": True,
            "sota": "acc ~0.958 (ResNet-18@28), ~0.966 (AutoML); macro-OvR AUC ~0.998"}


def load_mnist(reduced, device):
    import torch
    import torchvision

    from .allgraph import AllData
    from .paths import data_dir
    # python-native fetch via torchvision (auto-downloads + caches); uses MNIST's canonical train/test split
    root = data_dir()
    tr = torchvision.datasets.MNIST(root=root, train=True, download=True)
    te = torchvision.datasets.MNIST(root=root, train=False, download=True)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0; ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().astype(np.float32) / 255.0; yte = te.targets.numpy().astype(np.int64)
    mu, sd = Xtr.mean(), Xtr.std(); Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    if reduced:
        rtr = np.random.RandomState(0).permutation(len(Xtr))[:qscale(2500)]
        rte = np.random.RandomState(1).permutation(len(Xte))[:qscale(1200)]
        Xtr, ytr, Xte, yte = Xtr[rtr], ytr[rtr], Xte[rte], yte[rte]
    Xtr = torch.tensor(Xtr).unsqueeze(1); Xte = torch.tensor(Xte).unsqueeze(1)
    return {"train": AllData.dense_tensor(Xtr, y=ytr),
            "test": AllData.dense_tensor(Xte, y=yte),
            "task": "classification", "chance": 0.1, "field": "digits (vision)",
            "sota": "acc ~0.99+ (any CNN)"}


# --------------------------------------------------------------------------- VOLUMETRIC
def load_organmnist3d(reduced, device):
    import torch

    from .allgraph import AllData
    from .benchmark_datasets import load_organmnist3d as _l
    d = _l()
    def zn(x): return torch.tensor((x - x.mean()) / x.std()).unsqueeze(1)
    ntr = qscale(400) if reduced else len(d["train_y"])
    return {"train": AllData.dense_tensor(zn(d["train_x"])[:ntr], y=d["train_y"][:ntr]),
            "test": AllData.dense_tensor(zn(d["test_x"]), y=d["test_y"]),
            "task": "classification", "chance": 1.0 / 11, "field": "radiology", "report_auc": True,
            "sota": "acc ~0.907 (ResNet-18+3D; MedMNIST v2 benchmark best); macro-OvR AUC ~0.996"}


# --------------------------------------------------------------------------- GRAPH
def load_esol(reduced, device):
    from .allgraph import AllData
    from .moleculenet import load_esol as _l
    graphs, y = _l(n_max=qscale(500) if reduced else None)
    ym, ys = y.mean(), y.std()
    nf = [g["x"].numpy() for g in graphs]; ed = [g["edge_index"].numpy() for g in graphs]
    tr, te = _split(len(graphs))
    return {"train": AllData.graphs([nf[i] for i in tr], [ed[i] for i in tr], y=(y[tr] - ym) / ys),
            "test": AllData.graphs([nf[i] for i in te], [ed[i] for i in te], y=(y[te] - ym) / ys),
            "task": "regression", "chance": 0.0, "field": "chemistry (solubility)",
            "target_scale": float(ys), "target_units": "log mol/L",   # de-normalize z-scored logS -> physical-unit MAE
            "sota": "MAE ~0.40-0.45 log mol/L (D-MPNN/best GNN, random split); R2 ~0.90-0.93"}


def load_tox21(reduced, device):
    from .allgraph import AllData
    from .moleculenet import load_tox21 as _l
    graphs, y = _l(task="NR-AR", n_max=qscale(1600) if reduced else None)
    nf = [g["x"].numpy() for g in graphs]; ed = [g["edge_index"].numpy() for g in graphs]
    tr, te = _split(len(graphs))
    return {"train": AllData.graphs([nf[i] for i in tr], [ed[i] for i in tr], y=y[tr]),
            "test": AllData.graphs([nf[i] for i in te], [ed[i] for i in te], y=y[te]),
            "task": "classification", "chance": float(max(y.mean(), 1 - y.mean())),
            "field": "toxicology", "sota": "ROC-AUC ~0.75-0.83 (GNN)", "auc": True}


# --------------------------------------------------------------------------- EQUIVARIANT
def load_rmd17(reduced, device):
    import tempfile

    from .allgraph import AllData
    from .data_sources import rmd17_npz
    from .rmd17 import load_rmd17 as _l
    # fetch-with-fallback: rmd17_npz resolves the file (upload/RMD17_DIR); _l parses it
    npz = rmd17_npz("ethanol")
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(tmp.name, **{k: npz[k] for k in npz.files}); tmp.close()
    data = _l(tmp.name, max_conf=qscale(800) if reduced else 5000)
    Eraw = data["energies"].astype("f"); ys = float(Eraw.std())   # retain the kcal/mol scale for physical MAE
    E = (Eraw - Eraw.mean()) / ys
    Z = data["Z"]; elems = sorted(set(Z.tolist())); emap = {e: i for i, e in enumerate(elems)}
    feat0 = np.zeros((len(Z), len(elems)), "f")
    for i, z in enumerate(Z):
        feat0[i, emap[z]] = 1.0
    n = len(E)
    nf, ed, pos = [], [], []
    for i in range(n):
        p = data["coords"][i]; D = np.linalg.norm(p[:, None] - p[None], axis=-1)
        src, dst = np.where((D < 3.0) & (D > 0))
        nf.append(feat0); ed.append(np.stack([src, dst])); pos.append(p.astype("f"))
    tr, te = _split(n)
    return {"train": AllData.graphs([nf[i] for i in tr], [ed[i] for i in tr], y=E[tr], positions=[pos[i] for i in tr]),
            "test": AllData.graphs([nf[i] for i in te], [ed[i] for i in te], y=E[te], positions=[pos[i] for i in te]),
            "task": "regression", "chance": 0.0, "field": "molecular dynamics",
            "target_scale": ys, "target_units": "kcal/mol",   # energy MAE in physical units (forces: follow-up)
            "sota": "energy MAE ~0.009 kcal/mol (~0.4 meV); force MAE ~2-3 meV/A (MACE/NequIP, rMD17 1000 configs); R2 ~0.999+",
            "rotated": True}


def load_qm7(reduced, device):
    """Full QM7 (7165 molecules) with the RICHEST representation -- one-hot atom types, bond edges, AND 3D
    positions -- so the AllGraph router picks the contract itself (equivariant by default; set under
    canonicalizing symmetry routing). One QM7 dataset, replacing the former graph/equivariant/set
    feature-subset split."""
    from .allgraph import AllData
    nf, ed, pos, y = _mol_graphs_from_qm7(n_max=qscale(1500) if reduced else None)   # standard: all 7165; quick: 1500 x scale
    ym, ys = y.mean(), y.std()
    tr, te = _split(len(nf))
    return {"train": AllData.graphs([nf[i] for i in tr], [ed[i] for i in tr], y=(y[tr] - ym) / ys, positions=[pos[i] for i in tr]),
            "test": AllData.graphs([nf[i] for i in te], [ed[i] for i in te], y=(y[te] - ym) / ys, positions=[pos[i] for i in te]),
            "task": "regression", "chance": 0.0, "field": "quantum chemistry",
            "target_scale": float(ys), "target_units": "kcal/mol",   # de-normalize z-scored MAE -> physical
            "sota": "MAE < 1 kcal/mol = chemical accuracy (SchNet/PaiNN); R2 ~0.999+", "rotated": True}


def load_qm9(reduced, device):
    from .allgraph import AllData
    from .data_sources import qm9_xyz_dir
    from .qm9 import load_qm9_dir
    # fetch-with-fallback: qm9_xyz_dir extracts a bounded xyz subset from the uploaded zip or figshare tarball
    n_files = qscale(1500) if reduced else 15000
    inner = qm9_xyz_dir(n_files)
    R, Z, y = load_qm9_dir(inner, max_files=n_files)
    ym, ys = y.mean(), y.std()
    nf, ed, pos = [], [], []
    for i in range(len(R)):
        P = R[i]; nz = (Z[i] > 0)
        P = P[nz]; zt = Z[i][nz]
        feat = np.zeros((len(zt), 5), "f")
        common = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        for a, zz in enumerate(zt):
            feat[a, common.get(int(zz), 4)] = 1.0
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        src, dst = np.where((D < 3.0) & (D > 0))
        if len(src) == 0:
            src, dst = np.arange(len(P)), np.arange(len(P))
        nf.append(feat); ed.append(np.stack([src, dst])); pos.append(P.astype("f"))
    tr, te = _split(len(nf))
    return {"train": AllData.graphs([nf[i] for i in tr], [ed[i] for i in tr], y=(y[tr] - ym) / ys, positions=[pos[i] for i in tr]),
            "test": AllData.graphs([nf[i] for i in te], [ed[i] for i in te], y=(y[te] - ym) / ys, positions=[pos[i] for i in te]),
            "task": "regression", "chance": 0.0, "field": "quantum chemistry",
            "target_scale": float(ys * 27211.386), "target_units": "meV",   # U0 std (Hartree) -> meV MAE
            "sota": "U0 MAE ~5-15 meV (SchNet/PaiNN/DimeNet); R2 ~0.999+", "rotated": True}


# --------------------------------------------------------------------------- SET
def load_jetnet(reduced, device):
    from .allgraph import AllData
    from .jetnet import load_jetnet as _l
    particles, mask, labels, names = _l(n_per_class=qscale(700, lo=40) if reduced else None)
    nf = [particles[i][mask[i]] for i in range(len(labels))]
    tr, te = _split(len(labels))
    return {"train": AllData.point_sets([nf[i] for i in tr], y=labels[tr]),
            "test": AllData.point_sets([nf[i] for i in te], y=labels[te]),
            "task": "classification", "chance": 0.2, "field": "particle physics", "report_auc": True,
            "sota": "acc ~0.78-0.82 (JEDI-net/PELICAN, 5-class); macro-OvR AUC ~0.95"}


# --------------------------------------------------------------------------- the registry
# name -> (loader, expected_modality, in_quick_suite)
REGISTRY = {
    # sequence
    "ECG5000":          (load_ecg5000,     "sequence",    True),
    "GunPoint":         (load_gunpoint,    "sequence",    True),
    "OSULeaf":          (load_osuleaf,     "sequence",    True),
    "ItalyPowerDemand": (load_italypower,  "sequence",    False),
    "ACSF1":            (load_acsf1,       "sequence",    False),
    "BasicMotions":     (load_basicmotions,"sequence",    False),
    # spatial
    "BloodMNIST":       (load_bloodmnist,  "spatial",     True),
    "MNIST":            (load_mnist,       "spatial",     True),
    # volumetric
    "OrganMNIST3D":     (load_organmnist3d,"volumetric",  True),
    # graph
    "ESOL":             (load_esol,        "graph",       True),
    "Tox21":            (load_tox21,       "graph",       True),
    # equivariant / quantum chemistry (QM7 carries full features; the allgraph routes the contract)
    "rMD17-ethanol":    (load_rmd17,       "equivariant", True),
    "QM7":              (load_qm7,         "equivariant", True),
    "QM9":              (load_qm9,         "equivariant", False),
    # set
    "JetNet":           (load_jetnet,      "set",         True),
    # operator (neural-operator / FNO contract): function -> function on a grid
    "Burgers1D":        (lambda reduced, device: load_burgers1d(reduced, device), "operator", True),
    "Darcy2D":          (lambda reduced, device: load_darcy2d(reduced, device),   "operator", True),
}


def _burgers_solve(u0, nu=0.02, T=1.0, dt=2.5e-4):
    """Spectral (Fourier) Burgers solver, 2/3 de-aliased and semi-implicit -- numerically stable for the
    moderate-amplitude ICs used here (an unstable solver would yield NaN targets; see the NaN guard)."""
    N = len(u0); k = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    freq = np.fft.fftfreq(N, d=1.0 / N)
    dealias = (np.abs(freq) < (N / 2) * (2 / 3)).astype(float)
    uh = np.fft.fft(u0.astype(np.complex128))
    for _ in range(int(T / dt)):
        u = np.fft.ifft(uh); ux = np.fft.ifft(1j * k * uh)
        nl = np.fft.fft(u * ux) * dealias
        uh = (uh - dt * nl) / (1 + dt * nu * k ** 2)
    return np.real(np.fft.ifft(uh)).astype(np.float32)


def load_burgers1d(reduced, device):
    """1D Burgers operator: initial condition u0(x) -> solution u(x, T=1). Nonlinear (shock-forming), the
    canonical 1D FNO benchmark (Li et al. 2021)."""
    from .allgraph import AllData
    N = 64; n = 300 if not reduced else qscale(160)
    rng = np.random.RandomState(0); x = np.linspace(0, 1, N, endpoint=False).astype(np.float32)
    a0, aT = [], []
    for _ in range(n):
        a = rng.randn(5) * 0.5; ph = rng.rand(5) * 2 * np.pi
        u0 = sum(a[k] / (k + 1) * np.sin(2 * np.pi * (k + 1) * x + ph[k]) for k in range(5)).astype(np.float32)
        a0.append(u0); aT.append(_burgers_solve(u0))
    a0 = np.array(a0, np.float32); aT = np.array(aT, np.float32)
    tr, te = _split(n)
    return {"train": AllData.functions(a=a0[tr], y=aT[tr]),
            "test": AllData.functions(a=a0[te], y=aT[te]),
            "task": "regression", "chance": 0.0, "field": "PDE (1D Burgers, nonlinear)",
            "sota": "field R2 ~0.999 (FNO)"}


def load_darcy2d(reduced, device):
    """2D Darcy flow operator: permeability field a(x) -> pressure field u(x), -div(a grad u)=1, u=0 on the
    boundary (elliptic, non-periodic). The canonical 2D FNO benchmark (Li et al. 2021)."""
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve

    from .allgraph import AllData
    N = 20; n = 130 if not reduced else qscale(90)
    def solve(a, f=1.0):
        h = 1.0 / (N - 1); Ni = N - 2; idx = lambda i, j: i * Ni + j
        A = lil_matrix((Ni * Ni, Ni * Ni)); b = np.full(Ni * Ni, f * h * h)
        for i in range(Ni):
            for j in range(Ni):
                I, J = i + 1, j + 1; c = idx(i, j)
                aE = 0.5 * (a[I, J] + a[I, J + 1]); aW = 0.5 * (a[I, J] + a[I, J - 1])
                aN = 0.5 * (a[I, J] + a[I + 1, J]); aS = 0.5 * (a[I, J] + a[I - 1, J])
                A[c, c] = aE + aW + aN + aS
                if j + 1 < Ni: A[c, idx(i, j + 1)] = -aE
                if j - 1 >= 0: A[c, idx(i, j - 1)] = -aW
                if i + 1 < Ni: A[c, idx(i + 1, j)] = -aN
                if i - 1 >= 0: A[c, idx(i - 1, j)] = -aS
        u = np.zeros((N, N)); u[1:-1, 1:-1] = spsolve(csr_matrix(A), b).reshape(Ni, Ni)
        return u.astype(np.float32)
    rng = np.random.RandomState(0); xs = np.linspace(0, 1, N); X, Y = np.meshgrid(xs, xs, indexing="ij")
    aa, uu = [], []
    for _ in range(n):
        c = rng.randn(4, 4) * 0.6; field = np.zeros((N, N))
        for i in range(4):
            for j in range(4): field += c[i, j] * np.cos(i * np.pi * X) * np.cos(j * np.pi * Y)
        a = np.exp(0.5 * field).astype(np.float32); aa.append(a); uu.append(solve(a))
    aa = np.array(aa, np.float32); uu = np.array(uu, np.float32)
    tr, te = _split(n)
    return {"train": AllData.functions(a=aa[tr], y=uu[tr]),
            "test": AllData.functions(a=aa[te], y=uu[te]),
            "task": "regression", "chance": 0.0, "field": "PDE (2D Darcy flow, elliptic)",
            "sota": "field R2 ~0.999 (FNO)"}


def quick_suite():
    return {k: v for k, v in REGISTRY.items() if v[2]}


def full_suite():
    return REGISTRY

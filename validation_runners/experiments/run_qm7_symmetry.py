#!/usr/bin/env python
"""
run_qm7_symmetry.py -- Family-2 end-to-end on QM7: does the metaoptimizer select the
rotation-invariant primitive for molecular atomization-energy regression from RAW 3D coordinates?

Pipeline (the integrated Family-2 loop):
  1. Load QM7 raw coordinates (invariance NOT pre-baked, unlike the Coulomb-matrix representation).
  2. Build a schema mixing an INVARIANT branch (SO(3)-invariant pairwise-distance features) and
     a DENSE branch (raw coordinates, must learn invariance).
  3. Select between them, judged on a RANDOMLY ROTATED validation set (the true invariance test),
     by two protocols:
       --protocol solo   : train branches separately, pick lower rotated-val loss (robust)
       --protocol mixing : DARTS softmax-alpha over the two branches (fails at small data, works at
                           scale)
  4. Report the selection and the rotated-test MAE.

Findings (see tests/qm7_so3_selection.md): the invariant bias helps at scarce data; solo-comparison
selects it robustly; DARTS-mixing selects it once data is sufficient (n>=800 on full QM7).

Usage:
  python run_qm7_symmetry.py --qm7 /path/to/qm7.mat --n_train 800 --protocol solo
  python run_qm7_symmetry.py --qm7 /path/to/qm7.mat --n_train 800 --protocol mixing
"""
import argparse, os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ilmarinen.core.qm7 import load_qm7, random_rotation


def rotate(X, seed):
    Q = torch.tensor(random_rotation(seed))
    return torch.einsum("nad,de->nae", X, Q)


class InvariantBranch(nn.Module):
    """SO(3)-invariant: pairwise distances + atom-type context, invariant pooling over pairs."""

    def __init__(self, n_atoms, n_elem, h=80):
        super().__init__()
        self.nA, self.nElem = n_atoms, n_elem
        self.pair = nn.Sequential(nn.Linear(1 + 2 * n_elem, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, Xc, Tt, Mk):
        n = Xc.shape[0]
        D = torch.cdist(Xc, Xc)                                     # invariant pairwise distances
        pm = Mk[:, :, None] * Mk[:, None, :]
        ti = Tt[:, :, None, :].expand(n, self.nA, self.nA, self.nElem)
        tj = Tt[:, None, :, :].expand(n, self.nA, self.nA, self.nElem)
        f = torch.cat([D[..., None], ti, tj], dim=-1)
        h = self.pair(f) * pm[..., None]
        pooled = h.sum((1, 2)) / (pm.sum((1, 2), keepdim=True) + 1e-6)
        return self.out(pooled).squeeze(-1)


class DenseBranch(nn.Module):
    """No symmetry: MLP on raw flattened coordinates + types (must learn invariance)."""

    def __init__(self, n_atoms, n_elem, h=160):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_atoms * (3 + n_elem), h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, Xc, Tt, Mk):
        n = Xc.shape[0]
        return self.net((torch.cat([Xc, Tt], dim=-1) * Mk[..., None]).reshape(n, -1)).squeeze(-1)


class MixSuperGraph(nn.Module):
    def __init__(self, nA, nElem):
        super().__init__()
        self.inv = InvariantBranch(nA, nElem)
        self.dense = DenseBranch(nA, nElem)
        self.alpha = nn.Parameter(torch.zeros(2))
        self.primitives = ("invariant", "dense")

    def forward(self, Xc, Tt, Mk):
        w = torch.softmax(self.alpha, 0)
        return w[0] * self.inv(Xc, Tt, Mk) + w[1] * self.dense(Xc, Tt, Mk)


def train_branch(branch, X, T, M, yz, tr, epochs, bs=128, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    op = torch.optim.Adam(branch.parameters(), lr=lr); lf = nn.MSELoss()
    for ep in range(epochs):
        pm = tr[torch.randperm(len(tr))]
        for i in range(0, len(pm), bs):
            bi = pm[i:i + bs]
            op.zero_grad(); l = lf(branch(X[bi], T[bi], M[bi]), yz[bi])
            if torch.isfinite(l):
                l.backward(); op.step()


def run(args):
    R, Tp, Mk, y = load_qm7(args.qm7)
    X = torch.tensor(R); T = torch.tensor(Tp); M = torch.tensor(Mk); y = torch.tensor(y)
    ymean, ystd = y.mean(), y.std(); yz = (y - ymean) / ystd
    nA, nElem = X.shape[1], T.shape[2]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(X), generator=g)
    tr = perm[:args.n_train]
    va = perm[args.n_train:args.n_train + 300]
    te = perm[args.n_train + 300:args.n_train + 300 + 800]
    Xv, Xte = rotate(X[va], args.seed + 1), rotate(X[te], args.seed + 2)

    if args.protocol == "solo":
        inv = InvariantBranch(nA, nElem); dense = DenseBranch(nA, nElem)
        train_branch(inv, X, T, M, yz, tr, args.epochs, seed=args.seed)
        train_branch(dense, X, T, M, yz, tr, args.epochs, seed=args.seed)
        lf = nn.MSELoss()
        with torch.no_grad():
            vi = float(lf(inv(Xv, T[va], M[va]), yz[va]))
            vd = float(lf(dense(Xv, T[va], M[va]), yz[va]))
            sel = "invariant" if vi < vd else "dense"
            best = inv if vi < vd else dense
            mae = float((best(Xte, T[te], M[te]) - yz[te]).abs().mean() * ystd)
        print(f"=== QM7 symmetry selection (solo-comparison, n_train={args.n_train}) ===")
        print(f"  rotated-val loss: invariant={vi:.3f}  dense={vd:.3f}  -> select {sel.upper()}")
        print(f"  selected model rotated-test MAE: {mae:.1f} kcal/mol")
    else:  # mixing
        net = MixSuperGraph(nA, nElem)
        ap = [net.alpha]; wp = [p for n_, p in net.named_parameters() if not n_.endswith("alpha")]
        ow = torch.optim.Adam(wp, lr=3e-3); oa = torch.optim.Adam(ap, lr=1e-2); lf = nn.MSELoss()
        warmup = args.epochs // 3
        for ep in range(args.epochs):
            pm = tr[torch.randperm(len(tr))]
            for i in range(0, len(pm), 128):
                bi = pm[i:i + 128]
                ow.zero_grad(); lf(net(X[bi], T[bi], M[bi]), yz[bi]).backward(); ow.step()
            if ep >= warmup:
                oa.zero_grad(); lf(net(Xv, T[va], M[va]), yz[va]).backward(); oa.step()
        a = torch.softmax(net.alpha, 0).detach().numpy()
        sel = net.primitives[int(a.argmax())]
        with torch.no_grad():
            mae = float((net(Xte, T[te], M[te]) - yz[te]).abs().mean() * ystd)
        print(f"=== QM7 symmetry selection (DARTS mixing, n_train={args.n_train}) ===")
        print(f"  alpha = (invariant {a[0]:.2f}, dense {a[1]:.2f})  -> select {sel.upper()}")
        print(f"  mixed-model rotated-test MAE: {mae:.1f} kcal/mol")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qm7", required=True, help="path to qm7.mat")
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--protocol", choices=["solo", "mixing"], default="solo")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())

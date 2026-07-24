#!/usr/bin/env python
"""
run_penalized_selection.py -- compact-AND-accurate metaoptimization via a DIFFERENTIABLE
complexity penalty folded directly into the architecture objective (not a post-hoc stopping
rule). This implements the analytical action J = R + mu * complexity as a single differentiable
objective, following the resource-aware differentiable-NAS literature (FBNet's latency-aware
loss term; SA-DARTS' FLOPs term L = sum_i beta_i c_i), adapted to our variational framing where
mu is the chemical-potential / description-length price.

Mechanism:
  - Each primitive i has a real parameter cost c_i (its core's parameter count), normalized to
    [0,1] by the max over primitives. The architecture (alpha) loss gains a term
        mu * sum_i softmax(alpha)_i * c_i,
    so gradient descent on alpha is pulled toward CHEAP primitives in proportion to the price
    mu, DURING search -- simultaneously optimizing fit (cross-entropy) and complexity (cost).
  - An entropy-sharpening term (-gamma * H(alpha)) is added to reduce the DARTS discretization
    gap (documented failure: the continuous mixture is accurate but the argmax-discretized
    architecture collapses). It sharpens alpha so the selected primitive ~ the mixture.
  - Width is also priced: a width sweep is run and the smallest width within a tolerance of the
    best validation accuracy is chosen (accuracy-first compaction), rather than a marginal
    threshold -- this keeps the high-accuracy end of the frontier (the earlier method's weakness).

Protocol: bilevel (weights on train-part, alpha on held-out val); TEST accuracy on official
test split.

Usage:
  python run_penalized_selection.py --dataset GunPoint --mu 0.3
  python run_penalized_selection.py --dataset ItalyPowerDemand --mu 0.5 --widths 8,16,32,64 --acc_tol 0.02
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.datasets import load_classification

from ilmarinen.models.schema import build_schema

DEFAULT_PRIMS = ("plain", "gated", "lstm", "conv", "attention", "dense", "norm", "spectral")


def load(name, max_t, val_frac, seed):
    Xtr, ytr = load_classification(name, split="train")
    Xte, yte = load_classification(name, split="test")
    cls = sorted(set(ytr)); m = {c: i for i, c in enumerate(cls)}
    Xtr = np.transpose(Xtr, (0, 2, 1)).astype(np.float32)
    Xte = np.transpose(Xte, (0, 2, 1)).astype(np.float32)
    T = Xtr.shape[1]
    if max_t and T > max_t:
        step = T // max_t
        def pool(X):
            k = (X.shape[1] // step) * step
            return X[:, :k].reshape(X.shape[0], k // step, step, X.shape[2]).mean(axis=2)
        Xtr, Xte = pool(Xtr), pool(Xte)
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = np.array([m[c] for c in ytr]); yte = np.array([m[c] for c in yte])
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xte, yte = torch.tensor(Xte), torch.tensor(yte)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Xtr), generator=g)
    nval = max(len(cls), int(round(val_frac * len(Xtr))))
    vi, wi = perm[:nval], perm[nval:]
    return (Xtr[wi], ytr[wi], Xtr[vi], ytr[vi], Xte, yte, len(cls), Xtr.shape[2])


def primitive_costs(net):
    """Normalized parameter cost per primitive (from the first cell's cores)."""
    cell = net.cells[0]
    raw = torch.tensor([sum(x.numel() for x in core.parameters()) for core in cell.cores],
                       dtype=torch.float32)
    return raw / raw.max()


def fit(Xw, yw, Xv, yv, n_in, n_out, prims, width, depth, seed, epochs, readout,
        mu, gamma, lr=0.003, alpha_lr=0.02, bs=32):
    torch.manual_seed(seed)
    net = build_schema(depth=depth, width=width, n_in=n_in, n_out=n_out,
                                   seed=seed, primitives=prims, readout=readout)
    costs = primitive_costs(net)                      # (n_prim,), normalized
    ap = [c.alpha for c in net.cells]
    wp = [p for n, p in net.named_parameters() if not n.endswith("alpha")]
    ow = torch.optim.Adam(wp, lr=lr); oa = torch.optim.Adam(ap, lr=alpha_lr)
    lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Xw))
        for i in range(0, len(Xw), bs):
            bi = perm[i:i + bs]
            ow.zero_grad(); l = lf(net(Xw[bi]), yw[bi])
            if torch.isfinite(l):
                l.backward(); torch.nn.utils.clip_grad_norm_(wp, 5.0); ow.step()
        perm = torch.randperm(len(Xv))
        for i in range(0, len(Xv), bs):
            bi = perm[i:i + bs]
            oa.zero_grad()
            la = lf(net(Xv[bi]), yv[bi])
            # differentiable complexity penalty + entropy sharpening, summed over layers
            comp = net.cells[0].alpha.new_zeros(())
            ent = net.cells[0].alpha.new_zeros(())
            for cell in net.cells:
                w = torch.softmax(cell.alpha, dim=0)
                comp = comp + (w * costs).sum()
                ent = ent - (w * torch.log(w + 1e-9)).sum()
            obj = la + mu * comp + gamma * ent        # gamma>0 sharpens (subtracts entropy)
            if torch.isfinite(obj):
                obj.backward(); torch.nn.utils.clip_grad_norm_(ap, 5.0); oa.step()
        net.update_peak()
    with torch.no_grad():
        va = float((net(Xv).argmax(-1) == yv).float().mean())
    return va, net


def run(args):
    Xw, yw, Xv, yv, Xte, yte, n_out, n_in = load(args.dataset, args.max_t, args.val_frac, args.seed)
    prims = tuple(args.primitives.split(","))
    widths = [int(x) for x in args.widths.split(",")]

    # width selection: accuracy-first compaction. Sweep widths WITH the complexity penalty
    # active; pick the SMALLEST width whose val accuracy is within acc_tol of the best.
    rows = []
    for w in widths:
        va, net = fit(Xw, yw, Xv, yv, n_in, n_out, prims, w, args.depth, args.seed,
                      args.epochs, args.readout, args.mu, args.gamma)
        rows.append((w, va, net))
    best_va = max(va for _, va, _ in rows)
    Kstar, netK = next((w, net) for (w, va, net) in rows if va >= best_va - args.acc_tol)

    with torch.no_grad():
        test_acc = float((netK(Xte).argmax(-1) == yte).float().mean())
    maj = float(np.bincount(yte.numpy(), minlength=n_out).max() / len(yte))
    arch = netK.architecture()
    params = sum(p.numel() for p in netK.parameters())
    peak = netK.alpha_peak_report()[0]

    print(f"=== {args.dataset}: penalized (compact-and-accurate) selection, mu={args.mu} ===")
    print("  width sweep (val acc): " + ", ".join(f"{w}:{va:.3f}" for w, va, _ in rows))
    print(f"  -> WIDTH K* = {Kstar} (smallest within {args.acc_tol} of best val acc {best_va:.3f})")
    print(f"  -> architecture: {arch}   ({params} schema params)")
    print("  peak-alpha: " + ", ".join(f"{p}={v:.2f}" for p, v in zip(prims, peak)))
    print(f"  TEST accuracy: {test_acc:.3f}   (majority baseline {maj:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--primitives", default=",".join(DEFAULT_PRIMS))
    ap.add_argument("--widths", default="8,16,32,64")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--readout", default="mean", choices=["mean", "last"])
    ap.add_argument("--mu", type=float, default=0.3, help="complexity price (differentiable penalty weight)")
    ap.add_argument("--gamma", type=float, default=0.01, help="entropy-sharpening weight (reduces discretization gap)")
    ap.add_argument("--acc_tol", type=float, default=0.02, help="width compaction tolerance in val acc")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_t", type=int, default=120)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())

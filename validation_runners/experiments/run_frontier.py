#!/usr/bin/env python
"""
run_frontier.py -- trace the fit-vs-complexity PARETO FRONTIER for a dataset by sweeping the
capacity price mu. For each price, the marginal-value criterion selects a (width, depth,
primitive) architecture; we report accuracy and size vs price. This is the honest
"minimal representation" answer: not one architecture at an arbitrary price, but the frontier
of metaoptimal architectures as the description-length cost of capacity varies.

The width sweep is measured ONCE (val loss per width); each price then reads off the smallest
width whose marginal per-neuron gain exceeds it. Depth is selected per (width) at each price
via the same marginal rule. Primitive is the alpha argmax of the final model.

Protocol: bilevel (weights on train-part, alpha + marginal curves on held-out val); TEST
accuracy on the official test split. Depths restricted to {1,2} (no task we can pose justifies
deeper; see tests/depth_necessity_probe.md).

Usage:
  python run_frontier.py --dataset GunPoint
  python run_frontier.py --dataset ItalyPowerDemand --widths 8,16,32,64,128 --mus 0.0005,0.001,0.002,0.005,0.01
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
    cls = sorted(set(ytr))
    m = {c: i for i, c in enumerate(cls)}
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
    ytr = np.array([m[c] for c in ytr])
    yte = np.array([m[c] for c in yte])
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xte, yte = torch.tensor(Xte), torch.tensor(yte)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Xtr), generator=g)
    nval = max(len(cls), int(round(val_frac * len(Xtr))))
    vi, wi = perm[:nval], perm[nval:]
    return (Xtr[wi], ytr[wi], Xtr[vi], ytr[vi], Xte, yte, len(cls), Xtr.shape[2])


def fit(Xw, yw, Xv, yv, n_in, n_out, prims, width, depth, seed, epochs, readout):
    torch.manual_seed(seed)
    net = build_schema(depth=depth, width=width, n_in=n_in, n_out=n_out, seed=seed, primitives=prims, readout=readout)
    ap = [c.alpha for c in net.cells]
    wp = [p for n, p in net.named_parameters() if not n.endswith("alpha")]
    ow = torch.optim.Adam(wp, lr=0.003)
    oa = torch.optim.Adam(ap, lr=0.02)
    lf = nn.CrossEntropyLoss()
    bs = 32
    for ep in range(epochs):
        perm = torch.randperm(len(Xw))
        for i in range(0, len(Xw), bs):
            bi = perm[i : i + bs]
            ow.zero_grad()
            l = lf(net(Xw[bi]), yw[bi])
            if torch.isfinite(l):
                l.backward()
                torch.nn.utils.clip_grad_norm_(wp, 5.0)
                ow.step()
        perm = torch.randperm(len(Xv))
        for i in range(0, len(Xv), bs):
            bi = perm[i : i + bs]
            oa.zero_grad()
            la = lf(net(Xv[bi]), yv[bi])
            if torch.isfinite(la):
                la.backward()
                torch.nn.utils.clip_grad_norm_(ap, 5.0)
                oa.step()
        net.update_peak()
    with torch.no_grad():
        vl = float(lf(net(Xv), yv))
    return vl, net


def run(args):
    Xw, yw, Xv, yv, Xte, yte, n_out, n_in = load(args.dataset, args.max_t, args.val_frac, args.seed)
    prims = tuple(args.primitives.split(","))
    widths = [int(x) for x in args.widths.split(",")]
    mus = [float(x) for x in args.mus.split(",")]
    maj = float(np.bincount(yte.numpy(), minlength=n_out).max() / len(yte))

    # measure width curve (val loss per width) at depth 1, ONCE
    wloss = {}
    wnet = {}
    for w in widths:
        vl, net = fit(Xw, yw, Xv, yv, n_in, n_out, prims, w, 1, args.seed, args.epochs, args.readout)
        wloss[w] = vl
        wnet[w] = net
    # per-neuron marginals between consecutive widths
    wmarg = [
        (widths[i], (wloss[widths[i - 1]] - wloss[widths[i]]) / (widths[i] - widths[i - 1]))
        for i in range(1, len(widths))
    ]
    # measure depth-2 val loss at each width (for the depth marginal at that width)
    d2loss = {}
    for w in widths:
        vl, net = fit(Xw, yw, Xv, yv, n_in, n_out, prims, w, 2, args.seed, args.epochs, args.readout)
        d2loss[w] = vl

    print(f"=== {args.dataset}: fit-vs-complexity frontier (majority {maj:.3f}) ===")
    print("  width val-loss @depth1: " + ", ".join(f"{w}:{wloss[w]:.3f}" for w in widths))
    print(f"  {'price mu':>10} | {'K*':>4} | {'L*':>3} | {'primitive(s)':>18} | {'params':>7} | test acc")
    print("  " + "-" * 68)
    for mu in mus:
        # width: smallest width whose NEXT marginal falls below mu (i.e. stop paying)
        Kstar = widths[0]
        for w, m in wmarg:
            if m >= mu:
                Kstar = w  # this step still pays -> keep the larger width
            else:
                break
        # depth: justify layer 2 at Kstar iff per-layer marginal (depth1->2) exceeds mu
        d_marg = wloss[Kstar] - d2loss[Kstar]  # per-added-layer (one layer step)
        Lstar = 2 if d_marg > mu else 1
        # final model at (K*, L*)
        vl, net = fit(Xw, yw, Xv, yv, n_in, n_out, prims, Kstar, Lstar, args.seed, args.epochs, args.readout)
        with torch.no_grad():
            acc = float((net(Xte).argmax(-1) == yte).float().mean())
        params = sum(p.numel() for p in net.parameters())
        arch = net.architecture()
        print(f"  {mu:>10.4f} | {Kstar:>4} | {Lstar:>3} | {str(arch):>18} | {params:>7} | {acc:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--primitives", default=",".join(DEFAULT_PRIMS))
    ap.add_argument("--widths", default="8,16,32,64,128")
    ap.add_argument("--mus", default="0.0005,0.001,0.002,0.005,0.01,0.02")
    ap.add_argument("--readout", default="mean", choices=["mean", "last"])
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--max_t", type=int, default=120)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())

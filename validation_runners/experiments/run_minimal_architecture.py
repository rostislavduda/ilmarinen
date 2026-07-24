#!/usr/bin/env python
"""
run_minimal_architecture.py -- the FULL metaoptimization: select not only the primitive
(via alpha) but also the WIDTH (neuron count) and DEPTH (layer count) by the marginal-value
criterion, so the architecture size is an OUTPUT of metaoptimality, not a fixed hyperparameter.

This wires the width/depth machinery (ilmarinen.machinery.priced_depth) into the unified
schema pipeline. The minimal-representation idea: add capacity (neurons, then layers)
only while the marginal reduction in validation loss per unit of added capacity exceeds a
price. Stop when it falls below -- that stopping point is the metaoptimal size.

Protocol (honest / bilevel):
  - Hold out a validation split from TRAIN. Weights train on the train part; the marginal-
    value curves for width and depth are measured on the held-out validation part; alpha
    (primitive) is selected on validation too. TEST accuracy is reported on the official
    held-out test split, never used for any selection.

Selection:
  - WIDTH: sweep widths; for each, train + measure val loss. Pick the smallest width whose
    marginal val-loss reduction per added neuron-block falls below price --width_mu (or the
    significant-elbow if --width_mu is 0).
  - DEPTH: at the selected width, sweep depths; use priced_depth.select_depth / significant_elbow.
  - PRIMITIVE: alpha argmax at the selected (width, depth).

Usage:
  python run_minimal_architecture.py --dataset GunPoint
  python run_minimal_architecture.py --dataset ItalyPowerDemand --widths 8,16,32,64,128 --depths 1,2,3
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.datasets import load_classification

from ilmarinen.machinery.priced_depth import measure_depth_curve
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
    # bilevel split of train -> (weight-train, validation)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Xtr), generator=g)
    nval = max(len(cls), int(round(val_frac * len(Xtr))))
    vi, wi = perm[:nval], perm[nval:]
    return (Xtr[wi], ytr[wi], Xtr[vi], ytr[vi], Xte, yte, len(cls), Xtr.shape[2])


def train_eval(Xw, yw, Xv, yv, n_in, n_out, prims, width, depth, seed, epochs, readout, lr=0.003, alpha_lr=0.02, bs=32):
    """Train weights on (Xw,yw) and alpha on (Xv,yv); return (val_loss, val_acc, net)."""
    torch.manual_seed(seed)
    net = build_schema(depth=depth, width=width, n_in=n_in, n_out=n_out, seed=seed, primitives=prims, readout=readout)
    ap = [c.alpha for c in net.cells]
    wp = [p for n, p in net.named_parameters() if not n.endswith("alpha")]
    ow = torch.optim.Adam(wp, lr=lr)
    oa = torch.optim.Adam(ap, lr=alpha_lr)
    lf = nn.CrossEntropyLoss()
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
        va = float((net(Xv).argmax(-1) == yv).float().mean())
    return vl, va, net


def select_width(Xw, yw, Xv, yv, n_in, n_out, prims, widths, depth, seed, epochs, readout, width_mu, n_se):
    """Marginal-value width selection. Returns (K*, curve rows)."""
    losses, accs = [], []
    for w in widths:
        vl, va, _ = train_eval(Xw, yw, Xv, yv, n_in, n_out, prims, w, depth, seed, epochs, readout)
        losses.append(vl)
        accs.append(va)
    losses = np.array(losses)
    # per-added-neuron marginal reduction in val loss between consecutive widths
    marg = []
    for i in range(1, len(widths)):
        dW = widths[i] - widths[i - 1]
        marg.append((widths[i], (losses[i - 1] - losses[i]) / dW))
    # priced stopping: smallest width whose next-step marginal (per neuron) < width_mu
    Kstar = widths[0]
    if width_mu > 0:
        Kstar = widths[-1]
        for w, m in marg:
            if m < width_mu:
                Kstar = w
                break
    else:
        # price-free: last width whose marginal is still positive beyond noise floor
        Kstar = widths[0]
        for w, m in marg:
            if m > n_se * 1e-4:  # tiny positive floor
                Kstar = w
    return Kstar, list(zip(widths, losses, accs))


def run(args):
    Xw, yw, Xv, yv, Xte, yte, n_out, n_in = load(args.dataset, args.max_t, args.val_frac, args.seed)
    prims = tuple(args.primitives.split(","))
    widths = [int(x) for x in args.widths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    maj = float(np.bincount(yte.numpy(), minlength=n_out).max() / len(yte))

    # 1. WIDTH selection at depth=1
    Kstar, wcurve = select_width(
        Xw, yw, Xv, yv, n_in, n_out, prims, widths, 1, args.seed, args.epochs, args.readout, args.width_mu, args.n_se
    )

    # 2. DEPTH selection at the selected width, via the priced_depth machinery
    def depth_eval(L, sd):
        vl, va, _ = train_eval(Xw, yw, Xv, yv, n_in, n_out, prims, Kstar, L, sd, args.epochs, args.readout)
        return vl, va

    curve = measure_depth_curve(depth_eval, depths, seeds=[args.seed])
    # L* = largest depth still justified: step up only while the marginal per-layer val-loss
    # reduction exceeds the price (depth_mu) and is positive. The first non-justified step
    # stops us at the depth BEFORE it.
    Lstar = depths[0]
    for mid, m, me in curve.marginals:
        justified = (m > args.depth_mu) if args.depth_mu > 0 else (m > args.n_se * me and m > 0)
        if justified:
            Lstar = int(np.ceil(mid))  # the deeper endpoint of this justified step
        else:
            break

    # 3. final model at (K*, L*): read primitive + report TEST accuracy
    vl, va, net = train_eval(Xw, yw, Xv, yv, n_in, n_out, prims, Kstar, Lstar, args.seed, args.epochs, args.readout)
    with torch.no_grad():
        test_acc = float((net(Xte).argmax(-1) == yte).float().mean())
    arch = net.architecture()
    n_params = sum(p.numel() for p in net.parameters())

    print(f"=== {args.dataset}: metaoptimized architecture ===")
    print("  width sweep (val loss): " + ", ".join(f"{w}:{l:.3f}" for w, l, a in wcurve))
    print(f"  -> selected WIDTH K* = {Kstar}")
    print("  depth sweep (val loss): " + ", ".join(f"L{d}:{s:.3f}" for d, s in zip(curve.depths, curve.S_mean)))
    print(f"  -> selected DEPTH L* = {Lstar}")
    print(f"  -> selected PRIMITIVE(s) per layer: {arch}")
    print(f"  DEPLOYED: {arch} x {Lstar} layer(s), width {Kstar}, {n_params} schema params")
    print(f"  TEST accuracy: {test_acc:.3f}   (majority baseline {maj:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--primitives", default=",".join(DEFAULT_PRIMS))
    ap.add_argument("--widths", default="8,16,32,64,128")
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--readout", default="mean", choices=["mean", "last"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_t", type=int, default=120)
    ap.add_argument("--width_mu", type=float, default=0.0, help="price per neuron (0=significant-elbow)")
    ap.add_argument("--depth_mu", type=float, default=0.0, help="price per layer (0=significant-elbow)")
    ap.add_argument("--n_se", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())

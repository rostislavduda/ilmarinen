#!/usr/bin/env python
"""
run_unified_ucr.py -- run the unified sequence schema on a UCR/UEA time-series dataset
and report the selected architecture (per-layer primitive) + test accuracy.

Reproducible locally. Requires: torch, aeon, numpy.

Usage:
    python run_unified_ucr.py --dataset GunPoint
    python run_unified_ucr.py --dataset ACSF1 --max_t 120 --epochs 25 --bilevel
    python run_unified_ucr.py --dataset BasicMotions --primitives plain,gated,lstm,conv,attention,dense,norm,spectral

Key flag:
    --bilevel   Hold out a validation split from TRAIN for the alpha (architecture) selection,
                so alpha is chosen on data the weights did NOT train on. This is the honest
                architecture-selection protocol (training-loss alpha-selection can chase
                capacity). Without it, weights and alpha share the train split (faster, but
                the architecture choice is not held-out). Test accuracy is ALWAYS on the
                official held-out test split regardless of this flag.

Notes:
- Long series are average-pooled on the time axis to --max_t for tractable recurrence
  (the pure-Python per-timestep recurrent scan is the bottleneck at large T).
- Reports TEST accuracy (official split), never train accuracy.
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


def load(name, max_t):
    Xtr, ytr = load_classification(name, split="train")
    Xte, yte = load_classification(name, split="test")
    cls = sorted(set(ytr)); m = {c: i for i, c in enumerate(cls)}
    Xtr = np.transpose(Xtr, (0, 2, 1)).astype(np.float32)   # (n, T, chan)
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
    return (torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte),
            len(cls), Xtr.shape[2], Xtr.shape[1])


def run(args):
    Xtr, ytr, Xte, yte, n_out, chan, Tds = load(args.dataset, args.max_t)
    prims = tuple(args.primitives.split(","))
    torch.manual_seed(args.seed)

    # optional bilevel split: hold out a val fraction of TRAIN for alpha
    if args.bilevel:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(Xtr), generator=g)
        nval = max(n_out, int(round(args.val_frac * len(Xtr))))
        vi, wi = perm[:nval], perm[nval:]
        Xw, yw, Xv, yv = Xtr[wi], ytr[wi], Xtr[vi], ytr[vi]
    else:
        Xw, yw, Xv, yv = Xtr, ytr, Xtr, ytr   # alpha sees the same data as weights

    net = build_schema(depth=args.depth, width=args.width, n_in=chan,
                                   n_out=n_out, seed=args.seed, primitives=prims,
                                   readout=args.readout, chrono_tmax=args.chrono_tmax)
    ap = [c.alpha for c in net.cells]
    wp = [p for n, p in net.named_parameters() if not n.endswith("alpha")]
    ow = torch.optim.Adam(wp, lr=args.lr)
    oa = torch.optim.Adam(ap, lr=args.alpha_lr)
    lf = nn.CrossEntropyLoss(); bs = args.batch_size

    for ep in range(args.epochs):
        # weight step on the weight split
        perm = torch.randperm(len(Xw))
        for i in range(0, len(Xw), bs):
            bi = perm[i:i + bs]
            ow.zero_grad(); l = lf(net(Xw[bi]), yw[bi])
            if torch.isfinite(l):
                l.backward(); torch.nn.utils.clip_grad_norm_(wp, 5.0); ow.step()
        # alpha step on the val split (== train split if not bilevel)
        perm = torch.randperm(len(Xv))
        for i in range(0, len(Xv), bs):
            bi = perm[i:i + bs]
            oa.zero_grad(); la = lf(net(Xv[bi]), yv[bi])
            if torch.isfinite(la):
                la.backward(); torch.nn.utils.clip_grad_norm_(ap, 5.0); oa.step()
        net.update_peak()

    with torch.no_grad():
        test_acc = float((net(Xte).argmax(-1) == yte).float().mean())
    maj = float(np.bincount(yte.numpy(), minlength=n_out).max() / len(yte))
    peak = net.alpha_peak_report()
    print(f"dataset={args.dataset}  T->{Tds}  classes={n_out}  bilevel={args.bilevel}")
    print(f"  TEST accuracy: {test_acc:.3f}   (majority baseline {maj:.3f})")
    print(f"  architecture (per layer): {net.architecture()}")
    for l, pk in enumerate(peak):
        print(f"    layer {l} peak-alpha: " +
              ", ".join(f"{p}={v:.2f}" for p, v in zip(prims, pk)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="aeon UCR/UEA dataset name, e.g. GunPoint")
    ap.add_argument("--primitives", default=",".join(DEFAULT_PRIMS),
                    help="comma-separated subset of: " + ",".join(DEFAULT_PRIMS))
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--readout", default="mean", choices=["mean", "last"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--alpha_lr", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_t", type=int, default=120, help="pool time axis to at most this many steps (0=off)")
    ap.add_argument("--chrono_tmax", type=int, default=None, help="LSTM chrono-init T_max (long-range)")
    ap.add_argument("--bilevel", action="store_true", help="hold out val split from train for alpha selection")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())

#!/usr/bin/env python
"""
run_full_pipeline.py -- end-to-end: scale-aware symmetry detection (NO coordinate-structure prior)
+ schema + autonomous width/depth selection, on all real datasets.

For each dataset we run TWO metaoptimizer passes and report both, so the effect of the symmetry
front-end is visible:
  (A) RAW: the metaoptimizer on the native sequence representation (primitive + width + depth
      chosen autonomously via the penalized/compact-and-accurate criterion).
  (B) SYM: symmetry detection (scale_aware=True, coordinate_structure='unknown' -- no prior) runs on
      the flattened training features; the discovered quotient reduce_fn transforms the features;
      the metaoptimizer then runs on the reduced representation (as a length-1 sequence of the
      reduced feature vector, so the same schema machinery selects primitive+width+depth).

We record: detected symmetries, the chosen architecture (primitive, width, depth, params), and test
accuracy, for every dataset. This is an honest end-to-end measurement -- including any accuracy loss
from false-positive symmetry detection, per the experiment's intent.

CAVEAT recorded in the results: the [RAW] vs [SYM] delta confounds the symmetry effect with a
representation change (sequence vs flattened length-1). When no symmetry is found (the UCR case),
the delta is purely the representation artifact, not a symmetry benefit.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.datasets import load_classification

from ilmarinen.core.symmetry_pipeline import discover_and_reduce
from ilmarinen.models.schema import build_schema

DEFAULT_PRIMS = ("plain", "gated", "lstm", "conv", "spectral", "attention", "dense", "linssm", "norm")


def load_seq(name, max_t, seed, val_frac=0.3):
    Xtr, ytr = load_classification(name, split="train")
    Xte, yte = load_classification(name, split="test")
    cls = sorted(set(ytr)); m = {c: i for i, c in enumerate(cls)}
    Xtr = np.transpose(Xtr, (0, 2, 1)).astype(np.float32)      # (n, T, channels)
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
    return (Xtr[wi], ytr[wi], Xtr[vi], ytr[vi], Xte, yte, len(cls))


def primitive_costs(net):
    cell = net.cells[0]
    raw = torch.tensor([sum(x.numel() for x in core.parameters()) for core in cell.cores],
                       dtype=torch.float32)
    return raw / raw.max()


def fit(Xw, yw, Xv, yv, n_in, n_out, prims, width, depth, seed, epochs, readout,
        mu, gamma, lr=0.003, alpha_lr=0.02, bs=32):
    torch.manual_seed(seed)
    net = build_schema(depth=depth, width=width, n_in=n_in, n_out=n_out,
                                   seed=seed, primitives=prims, readout=readout)
    costs = primitive_costs(net)
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
            oa.zero_grad(); la = lf(net(Xv[bi]), yv[bi])
            comp = net.cells[0].alpha.new_zeros(()); ent = net.cells[0].alpha.new_zeros(())
            for cell in net.cells:
                w = torch.softmax(cell.alpha, dim=0)
                comp = comp + (w * costs).sum()
                ent = ent - (w * torch.log(w + 1e-9)).sum()
            obj = la + mu * comp + gamma * ent
            if torch.isfinite(obj):
                obj.backward(); torch.nn.utils.clip_grad_norm_(ap, 5.0); oa.step()
        net.update_peak()
    with torch.no_grad():
        va = float((net(Xv).argmax(-1) == yv).float().mean())
    return va, net


def metaoptimize(Xw, yw, Xv, yv, Xte, yte, n_in, n_out, prims, widths, depths,
                 seed, epochs, readout, mu, gamma, acc_tol):
    rows = []
    for depth in depths:
        for w in widths:
            va, net = fit(Xw, yw, Xv, yv, n_in, n_out, prims, w, depth, seed, epochs,
                          readout, mu, gamma)
            rows.append((depth, w, va, net))
    best_va = max(va for _, _, va, _ in rows)
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
    depth_s, w_s, va_s, net_s = next((d, w, va, net) for (d, w, va, net) in rows_sorted
                                     if va >= best_va - acc_tol)
    with torch.no_grad():
        test_acc = float((net_s(Xte).argmax(-1) == yte).float().mean())
    return {"depth": depth_s, "width": w_s, "val_acc": va_s, "test_acc": test_acc,
            "arch": net_s.architecture(), "params": sum(p.numel() for p in net_s.parameters()),
            "peak": net_s.alpha_peak_report()[0]}


def run(args):
    Xw, yw, Xv, yv, Xte, yte, n_out = load_seq(args.dataset, args.max_t, args.seed)
    prims = DEFAULT_PRIMS
    widths = [int(x) for x in args.widths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    n_in_seq = Xw.shape[2]
    maj = float(np.bincount(yte.numpy(), minlength=n_out).max() / len(yte))
    print(f"\n########## {args.dataset}  (majority baseline {maj:.3f}) ##########", flush=True)

    A = metaoptimize(Xw, yw, Xv, yv, Xte, yte, n_in_seq, n_out, prims, widths, depths,
                     args.seed, args.epochs, args.readout, args.mu, args.gamma, args.acc_tol)
    print(f"[RAW] arch={A['arch']} depth={A['depth']} width={A['width']} "
          f"params={A['params']} val={A['val_acc']:.3f} TEST={A['test_acc']:.3f}", flush=True)

    Xw_flat = Xw.reshape(len(Xw), -1)
    sym = discover_and_reduce(Xw_flat, yw.float(), n_refits=args.n_refits, epochs=args.sym_epochs,
                              coordinate_structure="unknown", scale_aware=True, verbose=False)
    syms = (f"continuous={[k for k, _ in sym['continuous']]}, cyclic={sym['cyclic']}, "
            f"z2={len(sym['z2'])} {sym['z2'][:6]}, permutation={sym['permutation']['young_subgroup']}")
    print(f"[SYM] detected: {syms}", flush=True)

    def to_reduced_seq(Xseq):
        flat = Xseq.reshape(len(Xseq), -1)
        red = sym["reduce_fn"](flat)
        return red.unsqueeze(1)
    Xw_r, Xv_r, Xte_r = to_reduced_seq(Xw), to_reduced_seq(Xv), to_reduced_seq(Xte)
    n_in_r = Xw_r.shape[2]
    B = metaoptimize(Xw_r, yw, Xv_r, yv, Xte_r, yte, n_in_r, n_out, prims, widths, depths,
                     args.seed, args.epochs, "mean", args.mu, args.gamma, args.acc_tol)
    print(f"[SYM] arch={B['arch']} depth={B['depth']} width={B['width']} "
          f"params={B['params']} val={B['val_acc']:.3f} TEST={B['test_acc']:.3f} "
          f"(feature dim {Xw_flat.shape[1]}->{n_in_r})", flush=True)
    print(f"[DELTA] raw TEST {A['test_acc']:.3f} vs sym TEST {B['test_acc']:.3f} "
          f"({B['test_acc']-A['test_acc']:+.3f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--widths", default="8,16,32")
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--readout", default="mean", choices=["mean", "last"])
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.03)
    ap.add_argument("--acc_tol", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--sym_epochs", type=int, default=120)
    ap.add_argument("--n_refits", type=int, default=2)
    ap.add_argument("--max_t", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())

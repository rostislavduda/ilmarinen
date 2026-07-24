#!/usr/bin/env python
"""
run_unified_synthetic.py -- run the unified sequence schema on the synthetic tasks with
known ground truth, and report the architecture the metaoptimizer selects for each.

Tasks:
    copy    -- remember a sequence across a delay   (ground truth: recurrent memory)
    adding  -- sum two marked values across T       (long-range; attention or gated)
    recall  -- retrieve value for a query key       (ground truth: attention routing)

Usage:
    python run_unified_synthetic.py --task recall
    python run_unified_synthetic.py --task adding --T 80 --epochs 15
    python run_unified_synthetic.py --task copy --delay 15
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ilmarinen.models.schema import build_schema

ALL = ("plain", "gated", "lstm", "conv", "attention", "dense", "norm", "spectral")


def make_copy(n, S, delay, V, seed):
    rng = np.random.default_rng(seed)
    T = S + delay + S
    X = np.zeros((n, T, V + 2), np.float32)
    Y = np.zeros((n, S), np.int64)
    for i in range(n):
        seq = rng.integers(0, V, size=S)
        for t in range(S):
            X[i, t, seq[t]] = 1.0
        X[i, S : S + delay, V] = 1.0
        X[i, S + delay :, V + 1] = 1.0
        Y[i] = seq
    return torch.tensor(X), torch.tensor(Y), S


def make_adding(n, T, seed):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0, 1, (n, T)).astype(np.float32)
    mk = np.zeros((n, T), np.float32)
    Y = np.zeros(n, np.float32)
    for i in range(n):
        idx = rng.choice(T, 2, replace=False)
        mk[i, idx] = 1.0
        Y[i] = vals[i, idx].sum()
    return torch.tensor(np.stack([vals, mk], -1)), torch.tensor(Y)


def make_recall(n, N, Vk, Vv, seed):
    rng = np.random.default_rng(seed)
    T = N + 1
    X = np.zeros((n, T, Vk + Vv), np.float32)
    Y = np.zeros(n, np.int64)
    for i in range(n):
        keys = rng.permutation(Vk)[:N]
        vals = rng.integers(0, Vv, N)
        for s in range(N):
            X[i, s, keys[s]] = 1.0
            X[i, s, Vk + vals[s]] = 1.0
        qi = rng.integers(N)
        X[i, N, keys[qi]] = 1.0
        Y[i] = vals[qi]
    return torch.tensor(X), torch.tensor(Y)


def train_select(net, X, Y, loss_fn, epochs, seqk=None):
    ap = [c.alpha for c in net.cells]
    wp = [p for n, p in net.named_parameters() if not n.endswith("alpha")]
    ow = torch.optim.Adam(wp, lr=0.003)
    oa = torch.optim.Adam(ap, lr=0.02)
    n = len(X)
    bs = 64
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bi = perm[i : i + bs]
            ow.zero_grad()
            l = loss_fn(net, X[bi], Y[bi], seqk)
            if torch.isfinite(l):
                l.backward()
                torch.nn.utils.clip_grad_norm_(wp, 5.0)
                ow.step()
            oa.zero_grad()
            la = loss_fn(net, X[bi], Y[bi], seqk)
            if torch.isfinite(la):
                la.backward()
                torch.nn.utils.clip_grad_norm_(ap, 5.0)
                oa.step()
        net.update_peak()


def run(args):
    torch.manual_seed(args.seed)
    if args.task == "copy":
        X, Y, S = make_copy(args.n, args.S, args.delay, args.V, args.seed)
        net = build_schema(
            depth=1, width=64, n_in=args.V + 2, n_out=args.V, seed=args.seed, primitives=ALL, readout="last"
        )

        def loss(net, x, y, k):
            o = net.forward_seq_readout(x, k)
            return nn.functional.cross_entropy(o.reshape(-1, o.shape[-1]), y.reshape(-1))

        train_select(net, X, Y, loss, args.epochs, seqk=S)
        with torch.no_grad():
            o = net.forward_seq_readout(X[:500], S)
            perf = f"acc {float((o.argmax(-1) == Y[:500]).float().mean()):.3f}"
    elif args.task == "adding":
        X, Y = make_adding(args.n, args.T, args.seed)
        net = build_schema(
            depth=1, width=64, n_in=2, n_out=1, seed=args.seed, primitives=ALL, readout="last", chrono_tmax=args.T
        )

        def loss(net, x, y, k):
            return nn.functional.mse_loss(net(x).squeeze(-1), y)

        train_select(net, X, Y, loss, args.epochs)
        with torch.no_grad():
            perf = f"mse {float(((net(X[:500]).squeeze(-1) - Y[:500]) ** 2).mean()):.4f} (baseline 0.17)"
    else:  # recall
        X, Y = make_recall(args.n, args.N, args.Vk, args.Vv, args.seed)
        net = build_schema(
            depth=1, width=64, n_in=args.Vk + args.Vv, n_out=args.Vv, seed=args.seed, primitives=ALL, readout="last"
        )

        def loss(net, x, y, k):
            return nn.functional.cross_entropy(net(x), y)

        train_select(net, X, Y, loss, args.epochs)
        with torch.no_grad():
            perf = f"acc {float((net(X[:500]).argmax(-1) == Y[:500]).float().mean()):.3f}"

    pk = net.alpha_peak_report()[0]
    print(f"task={args.task}  {perf}")
    print(f"  architecture: {net.architecture()}")
    print("  peak-alpha: " + ", ".join(f"{p}={v:.2f}" for p, v in zip(ALL, pk)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["copy", "adding", "recall"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    # copy
    ap.add_argument("--S", type=int, default=6)
    ap.add_argument("--delay", type=int, default=15)
    ap.add_argument("--V", type=int, default=6)
    # adding
    ap.add_argument("--T", type=int, default=80)
    # recall
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--Vk", type=int, default=8)
    ap.add_argument("--Vv", type=int, default=6)
    run(ap.parse_args())

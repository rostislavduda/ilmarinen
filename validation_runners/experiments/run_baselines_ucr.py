#!/usr/bin/env python
"""
run_baselines_ucr.py -- matched-complexity fixed-architecture baselines (GRU, 1-D CNN, MLP)
on a UCR/UEA dataset, for honest comparison against the schema's selected
architecture (run_unified_ucr.py). Same data preprocessing, split, and training budget.

Usage:
    python run_baselines_ucr.py --dataset GunPoint
    python run_baselines_ucr.py --dataset ACSF1 --width 64 --epochs 25 --max_t 120

Reports TEST accuracy (official split) and parameter count for each baseline.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.datasets import load_classification


def load(name, max_t):
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
    return (torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte),
            len(cls), Xtr.shape[2], Xtr.shape[1])


class GRUNet(nn.Module):
    def __init__(s, chan, n_out, w):
        super().__init__(); s.rnn = nn.GRU(chan, w, batch_first=True); s.head = nn.Linear(w, n_out)
    def forward(s, x):
        o, _ = s.rnn(x); return s.head(o.mean(1))


class CNNNet(nn.Module):
    def __init__(s, chan, n_out, w):
        super().__init__(); s.c1 = nn.Conv1d(chan, w, 5, padding=2)
        s.c2 = nn.Conv1d(w, w, 5, padding=2); s.head = nn.Linear(w, n_out)
    def forward(s, x):
        h = torch.relu(s.c1(x.transpose(1, 2))); h = torch.relu(s.c2(h)); return s.head(h.mean(2))


class MLPNet(nn.Module):
    def __init__(s, chan, T, n_out, w):
        super().__init__(); s.f = nn.Linear(chan * T, w); s.h = nn.Linear(w, n_out)
    def forward(s, x):
        return s.h(torch.relu(s.f(x.reshape(x.shape[0], -1))))


def train(net, Xtr, ytr, Xte, yte, epochs, lr, bs):
    opt = torch.optim.Adam(net.parameters(), lr=lr); lf = nn.CrossEntropyLoss(); n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bi = perm[i:i + bs]; opt.zero_grad(); l = lf(net(Xtr[bi]), ytr[bi])
            if torch.isfinite(l):
                l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    with torch.no_grad():
        return float((net(Xte).argmax(-1) == yte).float().mean())


def pc(net):
    return sum(p.numel() for p in net.parameters())


def run(args):
    Xtr, ytr, Xte, yte, n_out, chan, T = load(args.dataset, args.max_t)
    maj = float(np.bincount(yte.numpy(), minlength=n_out).max() / len(yte))
    results = {}
    for name, ctor in [("GRU", lambda: GRUNet(chan, n_out, args.width)),
                       ("CNN", lambda: CNNNet(chan, n_out, args.width)),
                       ("MLP", lambda: MLPNet(chan, T, n_out, args.width))]:
        torch.manual_seed(args.seed)
        net = ctor()
        acc = train(net, Xtr, ytr, Xte, yte, args.epochs, args.lr, args.batch_size)
        results[name] = (acc, pc(net))
    print(f"dataset={args.dataset}  T->{T}  classes={n_out}  majority baseline {maj:.3f}")
    for name, (acc, params) in results.items():
        print(f"  {name:4s} TEST acc {acc:.3f}  ({params/1000:.1f}k params)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_t", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())

"""Bilevel vs joint architecture-selection: a systematic cross-dataset study.

A STUDY (not a package feature): closes the "bilevel selection at scale" open question by
measuring, across a panel of datasets, whether the honest bilevel protocol (train the
weights on one split, select the architecture parameter alpha on a disjoint held-out split)
selects a DIFFERENT architecture than joint selection (weights and alpha trained together on
the same data), and whether that difference improves generalization. See
tests/bilevel_vs_joint_study.md for the full write-up and verdict.

Why this matters. Joint selection minimizes alpha on the training loss, which -- by the DARTS
capacity-chasing failure -- can reward the highest-CAPACITY primitive because capacity lowers
training loss regardless of generalization. Bilevel selects alpha on data the weights never saw,
so it should reward the primitive that GENERALIZES. The bilevel machinery
(ilmarinen/machinery/bilevel.py) enforces the split discipline; this study measures whether the
theoretical benefit is visible at the compact scales these schemas operate at.

The measurement subtlety this study addresses. A single alpha-selection per seed is dominated by
seed noise at these scales (a recurring lesson in this project). So the study reads the
SELECTION SIGNAL as the seed-AVERAGED mixture weight per primitive -- which separates a
systematic joint-vs-bilevel tendency from per-seed noise -- and reports test accuracy as
mean +/- std over the same seeds. The honest quantity is not "which primitive won once" but
"where does each protocol systematically place its mass, and how stable is the result".

Two parts:
  1. controlled_capacity_trap(): a task built so a high-capacity primitive CAN memorize the
     train set (train acc -> 1.0) while a local primitive generalizes. Demonstrates the
     mechanism the open question is about, with the seed-averaged measurement.
  2. real_panel(): the same joint-vs-bilevel comparison on real UCR datasets that load here.

Run: python studies/bilevel_vs_joint_study.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from ilmarinen.machinery.bilevel import three_way_split  # noqa: E402
from ilmarinen.models.schema import Schema  # noqa: E402

PRIMS = ("plain", "conv", "dense", "attention", "lstm")


def _train(model, Xtr, ytr, Xte, yte, mode, epochs=35, lr=0.01, bs=16, seed=0):
    """Train a schema under 'joint' (alpha+weights on the same train pool) or 'bilevel'
    (weights on the w-split, alpha on a disjoint alpha-split). Returns (test_acc, train_acc,
    seed-averaged alpha vector). The test set is disjoint from all training data in both modes.
    """
    torch.manual_seed(seed)
    if mode == "joint":
        Xw, yw, Xa, ya = Xtr, ytr, Xtr, ytr
    elif mode == "bilevel":
        Xw, yw, Xa, ya = three_way_split(Xtr, ytr, alpha_frac=0.5, seed=seed)
    else:
        raise ValueError(mode)
    Xw_t, yw_t, Xa_t, ya_t = map(torch.tensor, (Xw, yw, Xa, ya))
    ap = [c.alpha for c in model.cells]
    wp = [p for nm, p in model.named_parameters() if not nm.endswith("alpha")]
    oa = torch.optim.Adam(ap, lr=0.02)
    ow = torch.optim.Adam(wp, lr=lr)
    lf = nn.CrossEntropyLoss()
    nw, na = len(Xw), len(Xa)
    for _ in range(epochs):
        pw, pa = torch.randperm(nw), torch.randperm(na)
        q = 0
        for i in range(0, nw, bs):
            a_bi = pa[q:q + bs]
            q = (q + bs) % max(na - bs, 1)
            if len(a_bi) > 0:
                oa.zero_grad()
                la = lf(model(Xa_t[a_bi]), ya_t[a_bi])
                if torch.isfinite(la):
                    la.backward()
                    torch.nn.utils.clip_grad_norm_(ap, 5.0)
                    oa.step()
            w_bi = pw[i:i + bs]
            ow.zero_grad()
            lw = lf(model(Xw_t[w_bi]), yw_t[w_bi])
            if torch.isfinite(lw):
                lw.backward()
                torch.nn.utils.clip_grad_norm_(wp, 5.0)
                ow.step()
    with torch.no_grad():
        te = float((model(torch.tensor(Xte)).argmax(1) == torch.tensor(yte)).float().mean())
        tr = float((model(torch.tensor(Xtr)).argmax(1) == torch.tensor(ytr)).float().mean())
    return te, tr, np.asarray(model.alpha_report()).ravel()


def _compare(Xtr, ytr, Xte, yte, n_out, n_seeds=8, width=24, readout="flatten", epochs=35):
    """Run joint and bilevel over n_seeds; return per-mode dict of seed-averaged alpha,
    selection histogram, and test acc mean/std/train acc mean."""
    out = {}
    for mode in ("joint", "bilevel"):
        alphas, tes, trs, sels = [], [], [], []
        for seed in range(n_seeds):
            m = Schema(depth=1, width=width, n_in=Xtr.shape[2], n_out=n_out,
                                  seed=seed, primitives=PRIMS, readout=readout)
            te, tr, a = _train(m, Xtr, ytr, Xte, yte, mode, epochs=epochs, seed=seed)
            alphas.append(a); tes.append(te); trs.append(tr)
            sels.append(PRIMS[int(np.argmax(a))])
        A = np.array(alphas)
        out[mode] = dict(
            mean_alpha=A.mean(0),
            sel_hist={p: sels.count(p) for p in PRIMS if sels.count(p) > 0},
            te_mean=float(np.mean(tes)), te_std=float(np.std(tes)),
            tr_mean=float(np.mean(trs)),
        )
    return out


def _print_block(title, res):
    print(title)
    for mode in ("joint", "bilevel"):
        r = res[mode]
        amask = "  ".join(f"{p}={r['mean_alpha'][i]:.2f}" for i, p in enumerate(PRIMS))
        print(f"  {mode.upper():7s} train={r['tr_mean']:.2f}  test={r['te_mean']:.3f} +/- {r['te_std']:.3f}")
        print(f"          mean-alpha: {amask}")
        print(f"          selections: {r['sel_hist']}")
    # capacity-chasing readout: mass joint places on the highest-capacity primitives vs bilevel
    hi = [PRIMS.index("lstm"), PRIMS.index("attention"), PRIMS.index("dense")]
    jm = res["joint"]["mean_alpha"][hi].sum()
    bm = res["bilevel"]["mean_alpha"][hi].sum()
    print(f"  high-capacity mass (lstm+attention+dense):  joint={jm:.2f}  bilevel={bm:.2f}")
    print(f"  test-accuracy std:  joint={res['joint']['te_std']:.3f}  bilevel={res['bilevel']['te_std']:.3f}\n")


def _make_trap(n, T=60, seed=0):
    """A local rule (sign of the mean of a short window) is the generalizing solution;
    long noisy sequences + a flatten readout give high-capacity primitives room to memorize."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, 1)).astype(np.float32)
    y = (X[:, 10:15, 0].mean(1) > 0).astype(np.int64)
    return X, y


def controlled_capacity_trap(n_seeds=8):
    print("=" * 78)
    print("PART 1 -- controlled capacity trap (local rule, flatten readout, moderate train)")
    print("=" * 78)
    Xte, yte = _make_trap(800, seed=99)
    for ntr in (60, 120):
        Xtr, ytr = _make_trap(ntr, seed=1)
        res = _compare(Xtr, ytr, Xte, yte, n_out=2, n_seeds=n_seeds)
        _print_block(f"\ntrain={ntr}, test=800, T=60:", res)


def _load_ucr(name, max_t=80):
    from aeon.datasets import load_classification
    Xtr, ytr = load_classification(name, split="train")
    Xte, yte = load_classification(name, split="test")
    Xtr = np.transpose(Xtr, (0, 2, 1)).astype(np.float32)
    Xte = np.transpose(Xte, (0, 2, 1)).astype(np.float32)
    if Xtr.shape[1] > max_t:
        idx = np.linspace(0, Xtr.shape[1] - 1, max_t).astype(int)
        Xtr, Xte = Xtr[:, idx], Xte[:, idx]
    classes = sorted(set(ytr))
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cmap[c] for c in ytr], dtype=np.int64)
    yte = np.array([cmap[c] for c in yte], dtype=np.int64)
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte, len(classes)


def real_panel(names=("ItalyPowerDemand", "GunPoint"), n_seeds=6):
    print("=" * 78)
    print("PART 2 -- real UCR datasets")
    print("=" * 78)
    for name in names:
        try:
            Xtr, ytr, Xte, yte, nc = _load_ucr(name)
        except Exception as e:
            print(f"\n{name}: SKIP (load failed: {str(e)[:50]})")
            continue
        res = _compare(Xtr, ytr, Xte, yte, n_out=nc, n_seeds=n_seeds, readout="mean")
        _print_block(f"\n{name}: train={len(Xtr)} test={len(Xte)} T={Xtr.shape[1]} classes={nc}:", res)


if __name__ == "__main__":
    controlled_capacity_trap()
    real_panel()
    print("(see tests/bilevel_vs_joint_study.md for the full write-up and verdict)")

"""Bilevel architecture-selection training with an enforced three-way split.

Reproduces + fixes the canonical DARTS failure: single-level alpha training (on
the training loss) selects the highest-CAPACITY primitive because capacity
minimizes training loss regardless of generalization (e.g. a 4.2M-param dense
primitive memorizes and wins on train loss). Training alpha on a HELD-OUT split
makes it select the primitive that GENERALIZES instead.

CRITICAL SPLIT DISCIPLINE (the subtlety this module enforces):
  In bilevel training BOTH the w-split and the alpha-split have gradient descent
  run against them -- the alpha-split is a SECOND training set, not a clean
  validation set. So together they are the full training data, and any accuracy
  measured on either is optimistically biased. A genuinely independent test set
  must be disjoint from BOTH.

  This module takes the TRAIN pool and the TEST pool as SEPARATE inputs,
  partitions ONLY the train pool into (w-split, alpha-split), and never touches
  the test pool. The test accuracy it returns is therefore honest: measured on
  data that neither the weights nor alpha ever saw.

Applies to any model exposing a `.block.alpha` (spatial) or per-cell `.alpha`
(recurrent) parameter; the alpha params are detected by name.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def three_way_split(X_train, y_train, alpha_frac=0.5, seed=0):
    """Partition the TRAIN pool into (w-split, alpha-split). Test stays separate.

    Returns (Xw, yw, Xa, ya). The caller supplies the independent test set
    separately -- this function never sees it, so it cannot leak.
    """
    n = len(X_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_a = int(round(n * alpha_frac))
    a_idx, w_idx = perm[:n_a], perm[n_a:]
    return X_train[w_idx], y_train[w_idx], X_train[a_idx], y_train[a_idx]


def _split_alpha_params(model):
    """Separate architecture (alpha) parameters from ordinary weights by name."""
    alpha_params, w_params = [], []
    for name, p in model.named_parameters():
        if name.endswith("alpha") or ".alpha" in name:
            alpha_params.append(p)
        else:
            w_params.append(p)
    return alpha_params, w_params


def bilevel_train(model, Xw, yw, Xa, ya, Xte, yte,
                  epochs=12, lr_w=0.01, lr_a=0.02, bs=64,
                  grad_clip=None, track_alpha=False):
    """Bilevel training. Weights optimized on (Xw,yw); alpha on (Xa,ya).

    Xte/yte is the INDEPENDENT test set (disjoint from both training splits);
    used only for the final honest accuracy. Returns (test_acc, alpha_report,
    train_acc_on_w). alpha_report is the model's per-block softmax(alpha).

    grad_clip: set (e.g. 5.0) for recurrent models to prevent BPTT blow-up.
    """
    alpha_params, w_params = _split_alpha_params(model)
    if not alpha_params:
        raise ValueError("model exposes no alpha parameters; not a schema")
    opt_w = torch.optim.Adam(w_params, lr=lr_w)
    opt_a = torch.optim.Adam(alpha_params, lr=lr_a)
    lossf = nn.CrossEntropyLoss()
    nw, na = len(Xw), len(Xa)
    reports = []

    for ep in range(epochs):
        pw = torch.randperm(nw)
        pa = torch.randperm(na)
        a_ptr = 0
        for i in range(0, nw, bs):
            # (1) alpha step on the held-out alpha-split
            a_bi = pa[a_ptr:a_ptr + bs]
            a_ptr = (a_ptr + bs) % max(na - bs, 1)
            if len(a_bi) > 0:
                opt_a.zero_grad()
                la = lossf(model(Xa[a_bi]), ya[a_bi])
                if torch.isfinite(la):
                    la.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(alpha_params, grad_clip)
                    opt_a.step()
            # (2) weight step on the w-split
            w_bi = pw[i:i + bs]
            opt_w.zero_grad()
            lw = lossf(model(Xw[w_bi]), yw[w_bi])
            if not torch.isfinite(lw):
                return float("nan"), model.alpha_report(), float("nan")
            lw.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(w_params, grad_clip)
            opt_w.step()
        if track_alpha:
            reports.append(model.alpha_report())
    with torch.no_grad():
        test_acc = float((model(Xte).argmax(1) == yte).float().mean())
        train_acc = float((model(Xw).argmax(1) == yw).float().mean())
    return test_acc, model.alpha_report(), train_acc


def discretize_and_finetune(model, select_fn, Xw, yw, Xte, yte,
                            epochs=8, lr=0.01, bs=64, grad_clip=None):
    """Terminal discretization: freeze alpha to the argmax primitive, fine-tune.

    select_fn(model) should hard-set each block's alpha to one-hot at its argmax
    (implemented per-model). After discretization, alpha is frozen and only the
    weights of the SELECTED primitive matter; we fine-tune on the full train pool
    (Xw here may be the recombined train pool) and report honest test accuracy.

    This converts a correct SELECTION into full PERFORMANCE (the soft mixture
    dilutes the winning primitive; discretization removes the loser entirely).
    """
    select_fn(model)
    for p_name, p in model.named_parameters():
        if p_name.endswith("alpha") or ".alpha" in p_name:
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xw)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bi = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xw[bi]), yw[bi])
            if not torch.isfinite(loss):
                return float("nan")
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
    with torch.no_grad():
        return float((model(Xte).argmax(1) == yte).float().mean())

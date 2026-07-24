"""Training and evaluation utilities shared across validation pipelines."""

from __future__ import annotations

import torch
import torch.nn as nn


def gradient_norms_at_init(model, X, y, batch: int = 256):
    """First-layer and last-layer gradient norms at initialization.

    The direct signal-propagation probe: ordered init -> vanishing first-layer
    gradient; chaotic init -> exploding. Well-behaved (O(1)) at criticality or
    with normalization present.
    """
    model.zero_grad()
    out = model(X[:batch])
    loss = nn.CrossEntropyLoss()(out, y[:batch])
    loss.backward()
    w_first, w_last = model.first_last_weight()
    return float(w_first.grad.norm()), float(w_last.grad.norm())


def train_and_eval(
    model,
    Xtr,
    ytr,
    Xval,
    yval,
    epochs: int = 15,
    lr: float = 0.05,
    bs: int = 128,
    momentum: float = 0.9,
    cosine: bool = True,
    grad_clip: float = 1e4,
):
    """Train with SGD(+momentum), return (val_loss, val_acc, train_acc).

    Plain SGD (not Adam) and optional cosine schedule; grad_clip only catches
    true blow-up so chaotic-init divergence is still visible as NaN.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bi = perm[i : i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[bi]), ytr[bi])
            if not torch.isfinite(loss):
                return float("nan"), float("nan"), float("nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        if sched:
            sched.step()
    with torch.no_grad():
        val_loss = float(lossf(model(Xval), yval))
        val_acc = float((model(Xval).argmax(1) == yval).float().mean())
        train_acc = float((model(Xtr).argmax(1) == ytr).float().mean())
    return val_loss, val_acc, train_acc


def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_and_eval_rnn(
    model, Xtr, ytr, Xval, yval, epochs: int = 15, lr: float = 0.005, bs: int = 64, grad_clip: float = 5.0
):
    """Train a recurrent model with Adam + gradient clipping (RNN-appropriate).

    Separate from train_and_eval so the validated MLP/SGD path is untouched.
    RNNs need (a) Adam not plain SGD, (b) real gradient clipping (~5) to prevent
    BPTT exploding-gradient divergence -- without clipping, a long-unroll RNN
    diverges to NaN and masquerades as a signal-propagation failure.
    Returns (val_loss, val_acc, train_acc); NaN if it diverged despite clipping.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bi = perm[i : i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[bi]), ytr[bi])
            if not torch.isfinite(loss):
                return float("nan"), float("nan"), float("nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
    with torch.no_grad():
        val_loss = float(lossf(model(Xval), yval))
        val_acc = float((model(Xval).argmax(1) == yval).float().mean())
        train_acc = float((model(Xtr).argmax(1) == ytr).float().mean())
    return val_loss, val_acc, train_acc

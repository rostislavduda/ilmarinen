"""Model architectures with criticality-aware initialization.

Two families, both parameterized by the initialization variance sigma_w^2 so
the mean-field theory in core.meanfield can drive their init:

  - PlainMLP:  deep tanh MLP, no normalization (isolates signal propagation;
               untrainable at large depth by design -- used to *demonstrate*
               the criticality effect on gradients, not for production).
  - ResNetMLP: pre-norm residual blocks with LayerNorm (primitive #6). The
               trainable regime; depth-robust.

Both expose `.init_report()` hooks so validation code can read gradient norms
at initialization.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..models.init_utils import _init_linear   # canonical home is init_utils; re-exported here for legacy importers


class PlainMLP(nn.Module):
    """Deep tanh MLP with critical-style init and no normalization."""

    def __init__(self, depth: int, width: int, sigma_w2: float,
                 sigma_b2: float = 0.05, n_in: int = 784, n_out: int = 10, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        din = n_in
        for _ in range(depth):
            lin = nn.Linear(din, width)
            _init_linear(lin, sigma_w2, sigma_b2)
            layers += [lin, nn.Tanh()]
            din = width
        head = nn.Linear(din, n_out)
        _init_linear(head, 1.0, 0.0)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def first_last_weight(self):
        return self.net[0].weight, self.net[-1].weight


class _ResBlock(nn.Module):
    """Pre-norm residual block: x -> x + W2 * tanh(W1 * LN(x))."""

    def __init__(self, width: int, sigma_w2: float, sigma_b2: float = 0.05):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.l1 = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)
        _init_linear(self.l1, sigma_w2, sigma_b2)
        _init_linear(self.l2, sigma_w2, 0.0, scale=0.5)  # near-identity residual branch

    def forward(self, x):
        return x + self.l2(torch.tanh(self.l1(self.ln(x))))


class ResNetMLP(nn.Module):
    """Pre-norm residual MLP (primitive #6 present): trainable at large depth."""

    def __init__(self, depth: int, width: int, sigma_w2: float,
                 sigma_b2: float = 0.05, n_in: int = 784, n_out: int = 10, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.inp = nn.Linear(n_in, width)
        _init_linear(self.inp, sigma_w2, 0.0)
        self.blocks = nn.ModuleList([_ResBlock(width, sigma_w2, sigma_b2) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_out)
        _init_linear(self.head, 1.0, 0.0)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_out(x))

    def first_last_weight(self):
        return self.inp.weight, self.head.weight


MODEL_REGISTRY = {"plain_mlp": PlainMLP, "resnet_mlp": ResNetMLP}


def build_model(kind: str, **kwargs) -> nn.Module:
    if kind not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {kind!r}; have {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[kind](**kwargs)

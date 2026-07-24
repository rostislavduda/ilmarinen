"""Recurrent models for sequential tasks (Step 1: model class, added in isolation).

Adds recurrent architectures with the SAME criticality-aware init interface as
the sibling networks.py, so the existing mean-field / priced-depth / width machinery
drives them unchanged. Two "depth" axes exist here and are handled separately:

  - UNROLL LENGTH T: the number of timesteps. This is a TASK property (you must
    feed all pixels), not an optimizer choice. The relevant quantity is the
    TEMPORAL correlation length: does the recurrent map propagate signal across
    all T steps at its critical init? The recurrence h_{t+1} = tanh(W_h h_t +
    W_x x_t) is a repeated map, so the SAME chi_1 = sigma_w^2 E[tanh'^2] governs
    whether gradients survive T steps -- criticality on the time axis.

  - STACKED LAYERS L: how many recurrent layers to stack. This IS an optimizer
    choice; priced-depth selects L* exactly as for the MLP, but on a task where
    the answer should be > 1.

PlainRNN uses a tanh cell with sigma_w^2-scaled recurrent + input weights, no
gating (the plain cell that FAILS the long unroll -- the baseline against which
gating will later be shown necessary).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .networks import _init_linear


class _RNNCell(nn.Module):
    """Single tanh recurrent cell: h' = tanh(W_h h + W_x x + b).

    Recurrent weight init variance is the criticality knob: chi_1 for the
    time axis is sigma_w2_h * E[tanh'(z)^2] at the length fixed point, exactly
    as in the mean-field theory (the recurrent map is the layer map).
    """

    def __init__(
        self, n_in: int, width: int, sigma_w2_h: float, sigma_w2_x: float | None = None, sigma_b2: float = 0.05
    ):
        super().__init__()
        self.width = width
        self.Wx = nn.Linear(n_in, width, bias=False)
        self.Wh = nn.Linear(width, width, bias=True)
        # input weights: standard scaling; recurrent weights: criticality-scaled
        _init_linear(self.Wx, sigma_w2_x if sigma_w2_x is not None else 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2_h, sigma_b2)

    def forward(self, x, h):
        return torch.tanh(self.Wx(x) + self.Wh(h))


class PlainRNN(nn.Module):
    """Stacked plain-tanh RNN for sequence classification.

    Parameters
    ----------
    depth : number of stacked recurrent layers (the priced-depth axis).
    width : hidden width per layer.
    sigma_w2 : recurrent-weight init variance (the criticality knob; drives the
               TIME-axis correlation length).
    n_in : input dim per timestep (1 for sequential-MNIST pixel stream).
    n_out : number of classes.
    """

    def __init__(
        self,
        depth: int,
        width: int,
        sigma_w2: float,
        sigma_b2: float = 0.05,
        n_in: int = 1,
        n_out: int = 10,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.depth = depth
        self.width = width
        cells = []
        din = n_in
        for _ in range(depth):
            cells.append(_RNNCell(din, width, sigma_w2, sigma_w2_x=None, sigma_b2=sigma_b2))
            din = width
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(width, n_out)
        _init_linear(self.head, 1.0, 0.0)

    def forward(self, x):
        """x: (batch, T, n_in). Returns logits (batch, n_out)."""
        b, T, _ = x.shape
        hs = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hs[l] = cell(inp, hs[l])
                inp = hs[l]
        return self.head(hs[-1])

    def first_last_weight(self):
        # first recurrent weight and the head weight -- for the gradient probe
        return self.cells[0].Wh.weight, self.head.weight


RNN_REGISTRY = {"plain_rnn": PlainRNN}


def build_rnn(kind: str, **kwargs) -> nn.Module:
    if kind not in RNN_REGISTRY:
        raise ValueError(f"unknown rnn {kind!r}; have {list(RNN_REGISTRY)}")
    return RNN_REGISTRY[kind](**kwargs)

"""Spatial 2-primitive supergraph (convolution vs dense) for CIFAR-10.

Two spatial primitives in parallel, mixed by a learned softmax architecture
weight alpha:

  primitive 0: CONVOLUTION   weight-sharing under translation (the structurally
               hardest primitive -- a symmetry constraint on the parameter space,
               not a point in it). 3x3 conv, same padding.
  primitive 1: DENSE         unconstrained affine over the flattened feature map,
               reshaped back -- the no-inductive-bias baseline.

Both map a (C, H, W) feature map to a (C', H, W) feature map so they are mixable:

  out = softmax(alpha)_0 * conv(x) + softmax(alpha)_1 * dense(x)

alpha init uniform -> 0.5/0.5, unbiased. Ground-truth: on CIFAR-10 (natural
images) convolution decisively beats dense; a correct supergraph should drive
alpha toward conv. The margin is large enough that even slow soft-selection
should show clear preference -- unlike the modest gating margin on seq-MNIST.

The dense primitive over a full 32x32x3 map is huge, so we operate the supergraph
on a REDUCED spatial resolution after an initial shared stem (a fixed conv+pool
that both primitives sit on top of), keeping the dense primitive tractable while
preserving the conv-vs-dense contrast at the mixing layer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(conv, sigma_w2, scale=1.0):
    with torch.no_grad():
        fan_in = conv.weight.shape[1] * conv.weight.shape[2] * conv.weight.shape[3]
        conv.weight.normal_(0, np.sqrt(sigma_w2 / fan_in) * scale)
        if conv.bias is not None:
            conv.bias.zero_()


def _init_lin(lin, sigma_w2, scale=1.0):
    with torch.no_grad():
        lin.weight.normal_(0, np.sqrt(sigma_w2 / lin.weight.shape[1]) * scale)
        if lin.bias is not None:
            lin.bias.zero_()


class _ConvPrimitive(nn.Module):
    """3x3 convolution, same padding: weight-sharing under translation."""

    def __init__(self, cin, cout, sigma_w2):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        _init_conv(self.conv, sigma_w2)

    def forward(self, x):
        return self.conv(x)


class _DensePrimitive(nn.Module):
    """Unconstrained dense map over the flattened feature map, reshaped back.

    Same input/output spatial shape as the conv primitive so they are mixable,
    but with NO weight sharing -- every spatial-channel location has its own
    weights. This is the no-inductive-bias baseline conv should beat.
    """

    def __init__(self, cin, cout, hw, sigma_w2):
        super().__init__()
        self.cin, self.cout, self.hw = cin, cout, hw
        self.lin = nn.Linear(cin * hw * hw, cout * hw * hw)
        _init_lin(self.lin, sigma_w2)

    def forward(self, x):
        b = x.shape[0]
        out = self.lin(x.reshape(b, -1))
        return out.reshape(b, self.cout, self.hw, self.hw)


class SpatialSuperBlock(nn.Module):
    """Mix conv and dense primitives by softmax(alpha), then nonlinearity."""

    def __init__(self, cin, cout, hw, sigma_w2):
        super().__init__()
        self.conv = _ConvPrimitive(cin, cout, sigma_w2)
        self.dense = _DensePrimitive(cin, cout, hw, sigma_w2)
        self.alpha = nn.Parameter(torch.zeros(2))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        w = torch.softmax(self.alpha, dim=0)
        out = w[0] * self.conv(x) + w[1] * self.dense(x)
        return F.relu(self.bn(out))

    def alpha_weights(self):
        with torch.no_grad():
            return torch.softmax(self.alpha, dim=0).cpu().numpy()  # [conv, dense]


class SpatialSuperGraph(nn.Module):
    """CIFAR-10 spatial supergraph.

    A fixed shared stem (conv + pool) reduces 32x32 -> hw x hw so the dense
    primitive is tractable; the supergraph block (conv|dense mix) sits on top,
    followed by global pooling and a linear head.
    """

    def __init__(self, width=32, hw=8, sigma_w2=2.0, n_classes=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        # shared stem: 3->width, downsample 32->hw via strided conv + pool
        stem_stride = 32 // hw
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=stem_stride, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(),
        )
        for m in self.stem:
            if isinstance(m, nn.Conv2d):
                _init_conv(m, sigma_w2)
        self.block = SpatialSuperBlock(width, width, hw, sigma_w2)
        self.head = nn.Linear(width, n_classes)
        _init_lin(self.head, 1.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = x.mean(dim=(2, 3))            # global average pool
        return self.head(x)

    def alpha_report(self):
        return [self.block.alpha_weights()]     # [conv_weight, dense_weight]

    def first_last_weight(self):
        return self.block.conv.conv.weight, self.head.weight


def build_spatial_supergraph(**kwargs):
    return SpatialSuperGraph(**kwargs)

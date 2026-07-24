"""SPATIAL schema -- the image-interface analogue of the unified sequence
schema. All spatially-meaningful primitives compete on (b, C, H, W) feature maps under a
single per-layer architecture parameter alpha, discretized by argmax.

Motivation. The unified sequence schema (schema.py) resolved the interface
split for sequence data, but 2D convolution -- weight-sharing under the 2D translation group --
has no meaningful realization on a 1D sequence: time series carry no 2D spatial structure. 2D
conv earns its place only on genuine image data, so it lives here, on the (b, C, H, W)
interface, competing against the other primitives adapted to a spatial grid.

Common spatial contract: forward_spatial(x: (b, cin, H, W)) -> (b, width, H, W).

Primitive vocabulary (spatial forms of the irreducibles + useful non-irreducible conv variants):
  conv2d     : 3x3 convolution              -- weight-sharing under 2D translation (the star)
  conv_dw    : 3x3 depthwise + 1x1 pointwise -- separable conv, a cheaper conv variant
  pointwise  : 1x1 convolution              -- per-location channel mixing, NO spatial sharing
  dense      : flatten + linear + reshape   -- unconstrained, no spatial bias (conv should beat)
  norm       : BatchNorm + 1x1              -- the spatial stabilizer
  attention  : spatial self-attention        -- content routing over grid locations

conv2d is the structurally hardest primitive: a symmetry constraint on the parameter space, not
a point in it. On natural images it should decisively beat dense/pointwise; a correct
metaoptimizer drives alpha toward conv where 2D structure is real, and toward the others where it
is not (e.g. permuted-pixel images, where translation equivariance is destroyed).

All prior modules (including spatial_supergraph.py, the 2-way conv/dense version) are left
UNTOUCHED; this is a new capability in a new module.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(conv, sigma_w2=2.0):
    fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    with torch.no_grad():
        conv.weight.normal_(0, np.sqrt(sigma_w2 / max(fan_in, 1)))
        if conv.bias is not None:
            conv.bias.zero_()


def _init_lin(lin, sigma_w2=1.0):
    with torch.no_grad():
        lin.weight.normal_(0, np.sqrt(sigma_w2 / lin.in_features))
        if lin.bias is not None:
            lin.bias.zero_()


# --------------------------------------------------------------------------- primitives
class _Conv2dSpatial(nn.Module):
    """k x k convolution, same padding: weight-sharing under the 2D translation group. The kernel
    size k sets the RECEPTIVE FIELD; rather than hardcoding k=3, it can be chosen to match the data's
    spatial correlation length (core/correlation_length.py) -- see fixed_hyperparameter_audit.md B4."""
    name = "conv2d"

    def __init__(self, cin, width, hw, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, width, kernel_size=kernel_size, padding=kernel_size // 2)
        _init_conv(self.conv)

    def forward_spatial(self, x):
        return self.conv(x)


class _ConvDWSpatial(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 (separable convolution): a cheaper conv variant.
    Depthwise operates per-channel (channels must match), so we first lift to `width`
    with a 1x1, then depthwise-mix spatially. Fewer params than full conv at large width."""
    name = "conv_dw"

    def __init__(self, cin, width, hw):
        super().__init__()
        self.lift = nn.Conv2d(cin, width, kernel_size=1)        # channel lift (pointwise)
        self.dw = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)  # depthwise
        _init_conv(self.lift); _init_conv(self.dw)

    def forward_spatial(self, x):
        return self.dw(self.lift(x))


class _PointwiseSpatial(nn.Module):
    """1x1 convolution: per-location channel mixing, NO spatial weight-sharing across a
    neighborhood -- the 'affine + pointwise' primitive with no 2D receptive field."""
    name = "pointwise"

    def __init__(self, cin, width, hw):
        super().__init__()
        self.conv = nn.Conv2d(cin, width, kernel_size=1)
        _init_conv(self.conv)

    def forward_spatial(self, x):
        return self.conv(x)


class _DenseSpatial(nn.Module):
    """Flatten + linear + reshape: unconstrained affine over the whole feature map -- the
    no-inductive-bias baseline that convolution should beat on natural images."""
    name = "dense"

    def __init__(self, cin, width, hw):
        super().__init__()
        self.cin, self.width, self.hw = cin, width, hw
        self.lin = nn.Linear(cin * hw * hw, width * hw * hw)
        _init_lin(self.lin)

    def forward_spatial(self, x):
        b = x.shape[0]
        out = self.lin(x.reshape(b, -1))
        return out.reshape(b, self.width, self.hw, self.hw)


class _NormSpatial(nn.Module):
    """BatchNorm over channels + a 1x1 mix: the spatial stabilizer (normalization primitive)."""
    name = "norm"

    def __init__(self, cin, width, hw):
        super().__init__()
        self.proj = nn.Conv2d(cin, width, kernel_size=1)
        self.bn = nn.BatchNorm2d(width)
        _init_conv(self.proj)

    def forward_spatial(self, x):
        return self.bn(self.proj(x))


class _AttentionSpatial(nn.Module):
    """Single-head spatial self-attention: content routing over the H*W grid locations.
    Q,K,V are 1x1 projections; attention mixes locations by softmax(QK^T/sqrt(d)). This is
    the routing primitive on a spatial grid (permutation-equivariant over locations)."""
    name = "attention"

    def __init__(self, cin, width, hw):
        super().__init__()
        self.width = width
        self.q = nn.Conv2d(cin, width, kernel_size=1)
        self.k = nn.Conv2d(cin, width, kernel_size=1)
        self.v = nn.Conv2d(cin, width, kernel_size=1)
        for m in (self.q, self.k, self.v):
            _init_conv(m, 1.0)

    def forward_spatial(self, x):
        b, _, H, W = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)      # (b, HW, width)
        k = self.k(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(1, 2) / (self.width ** 0.5), dim=-1)
        out = att @ v                                  # (b, HW, width)
        return out.transpose(1, 2).reshape(b, self.width, H, W)


class _Conv2dK5(_Conv2dSpatial):
    """5x5 conv variant -- larger receptive field for data with longer spatial correlation length."""
    name = "conv2d_k5"

    def __init__(self, cin, width, hw):
        super().__init__(cin, width, hw, kernel_size=5)


class _Conv2dK7(_Conv2dSpatial):
    """7x7 conv variant -- largest receptive field, for smooth/long-correlation data."""
    name = "conv2d_k7"

    def __init__(self, cin, width, hw):
        super().__init__(cin, width, hw, kernel_size=7)


class _DilatedConv2dSpatial(nn.Module):
    """Multi-scale atrous (dilated) convolution -- ASPP / DeepLab style. Runs several k x k convolutions
    in PARALLEL at different dilation rates, so a single layer captures spatial context at multiple
    scales (fine and coarse) without downsampling and without inflating the kernel. This is the
    multi-scale spatial receptive field that a single-rate conv2d cannot express -- the primitive the
    segmentation / modern-CNN literature (DeepLab ASPP, dilated/atrous nets) is built on. Weight-sharing
    under 2D translation, 'same' spatial size preserved."""
    name = "atrous"

    def __init__(self, cin, width, hw, kernel_size=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        # use at most `width` dilation branches so no branch gets 0 channels (a variable-width d.o.f. stage
        # can pick width < len(dilations)); each active branch then holds >= 1 channel.
        self.dilations = tuple(dilations)[:max(1, width)]
        nb = len(self.dilations)
        base = width // nb
        self.branch_widths = [base] * (nb - 1) + [width - base * (nb - 1)]
        self.branches = nn.ModuleList([
            nn.Conv2d(cin, bw, kernel_size=kernel_size, dilation=d,
                      padding=((kernel_size - 1) // 2) * d)                 # 'same' padding per dilation
            for bw, d in zip(self.branch_widths, self.dilations)])
        self.proj = nn.Conv2d(width, width, kernel_size=1)                  # mix multi-scale features
        for b in self.branches:
            _init_conv(b)
        _init_conv(self.proj)

    def forward_spatial(self, x):
        outs = [conv(x) for conv in self.branches]                         # each (b, bw, H, W)
        c = torch.cat(outs, dim=1)                                         # (b, width, H, W)
        return self.proj(c)


_SPATIAL_CORES = {
    "conv2d": _Conv2dSpatial,
    "conv2d_k5": _Conv2dK5,
    "conv2d_k7": _Conv2dK7,
    "atrous": _DilatedConv2dSpatial,
    "conv_dw": _ConvDWSpatial,
    "pointwise": _PointwiseSpatial,
    "dense": _DenseSpatial,
    "norm": _NormSpatial,
    "attention": _AttentionSpatial,
}


# --------------------------------------------------------------------------- schema cell
class _SpatialCell(nn.Module):
    """One meta-layer: all primitives in parallel, mixed by softmax(alpha), then BN + ReLU."""

    def __init__(self, cin, width, hw, primitives):
        super().__init__()
        self.primitives = tuple(primitives)
        self.cores = nn.ModuleList([_SPATIAL_CORES[p](cin, width, hw) for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.bn = nn.BatchNorm2d(width)
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def mixed(self, x):  # canonical uniform name (delegates to forward)
        return self.forward(x)

    def forward(self, x):
        # weighted sum over primitives without stacking (avoids an extra full (P,b,w,H,W) copy of all P
        # primitives' feature maps; sum_p w_p*out_p == the einsum over the stack). Matches the 4d/operator cells.
        w = torch.softmax(self.alpha, dim=0)
        outs = [c.forward_spatial(x) for c in self.cores]                      # P x (b,w,H,W)
        mixed = sum(wi * o for wi, o in zip(w, outs))
        return F.relu(self.bn(mixed))

    def update_peak(self):
        with torch.no_grad():
            w = torch.softmax(self.alpha, dim=0)
            self.alpha_peak = torch.maximum(self.alpha_peak, w)


class SpatialSchema(nn.Module):
    """Stack of spatial meta-layers over a shared stem, then global pool + linear head.

    A shared stem (conv + optional downsample) maps the raw image (b, 3, 32, 32) to
    (b, width, hw, hw), keeping the dense primitive tractable; the schema meta-layers
    (each mixing the full spatial vocabulary) sit on top; global average pool + linear head
    produce the class logits.
    """

    def __init__(self, width=32, hw=8, depth=1, n_in=3, n_classes=10, seed=0,
                 primitives=("conv2d", "conv_dw", "pointwise", "dense", "norm", "attention"),
                 img_size=32):
        super().__init__()
        torch.manual_seed(seed)
        self.primitives = tuple(primitives)
        self.width, self.hw, self.depth = width, hw, depth
        stem_stride = max(1, img_size // hw)
        self.stem = nn.Sequential(
            nn.Conv2d(n_in, width, kernel_size=3, stride=stem_stride, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(),
        )
        for m in self.stem:
            if isinstance(m, nn.Conv2d):
                _init_conv(m)
        self.cells = nn.ModuleList([
            _SpatialCell(width, width, hw, self.primitives) for _ in range(depth)
        ])
        self.head = nn.Linear(width, n_classes)
        _init_lin(self.head, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = x.mean(dim=(2, 3))               # global average pool
        return self.head(x)

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def alpha_report(self):
        return [torch.softmax(c.alpha, dim=0).detach().cpu().numpy() for c in self.cells]

    def alpha_peak_report(self):
        return [c.alpha_peak.detach().cpu().numpy() for c in self.cells]

    def architecture(self):
        return [self.primitives[int(np.argmax(c.alpha_peak.detach().cpu().numpy()))]
                for c in self.cells]


def build_spatial_schema(width=32, hw=8, depth=1, n_in=3, n_out=None, seed=0,
                                     primitives=("conv2d", "conv_dw", "pointwise", "dense",
                                                 "norm", "attention"), img_size=32, n_classes=None):
    # canonical param is n_out; n_classes kept as backward-compatible alias
    if n_out is None:
        n_out = n_classes if n_classes is not None else 10
    return SpatialSchema(width=width, hw=hw, depth=depth, n_in=n_in,
                                    n_classes=n_out, seed=seed, primitives=primitives,
                                    img_size=img_size)

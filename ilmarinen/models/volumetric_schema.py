"""VOLUMETRIC schema -- the 3D (volume) analogue of the unified spatial schema.
All volumetrically-meaningful primitives compete on (b, C, D, H, W) feature volumes under a single
per-layer architecture parameter alpha, discretized by argmax.

Motivation. The metamodel taxonomy is a short, bounded ladder of mutually-exclusive tensor
contracts, one per input rank (see tests/supergraph_taxonomy_audit.md and
tests/tensor_order_literature.md):

    schema          : (b, T, n_in)        sequence   [1 ordered axis]
    spatial_schema  : (b, C, H, W)        grid       [2 ordered axes]
    volumetric_schema (HERE)    : (b, C, D, H, W)     volume     [3 ordered axes]

These are mutually exclusive because their tensor contracts differ in RANK: a Conv3d (5D I/O)
cannot be alpha-mixed with a Conv2d (4D I/O) in one cell -- the operations act on different-rank
tensors. So 3D convolution needs its own schema, exactly as 2D conv needed one separate from
the 1D sequence schema. This module provides it.

The literature bounds this ladder: dense grid-structured input tops out around order 4-5 in
practice (order-3 volumes: MRI, CT, voxel grids, grayscale video; order-4: color video). Order 3
is the most common higher-order dense case, so the volumetric schema is the clear next contract
once volumetric data enters scope. Higher orders are either sparse-relational (a graph contract, not
a grid one) or scientific solution fields (tensor-network-decomposed to low-order cores), so this is
not an open-ended ladder.

Common volumetric contract: forward_volumetric(x: (b, cin, D, H, W)) -> (b, width, D, H, W).

Primitive vocabulary (volumetric forms of the irreducibles + useful conv variants):
  conv3d     : 3x3x3 convolution                -- weight-sharing under the 3D translation group
  conv_dw    : 3x3x3 depthwise + 1x1x1 pointwise -- separable 3D conv, a cheaper conv variant
  pointwise  : 1x1x1 convolution               -- per-voxel channel mixing, NO spatial sharing
  dense      : flatten + linear + reshape      -- unconstrained, no volumetric bias (conv beats it)
  norm       : BatchNorm3d + 1x1x1             -- the volumetric stabilizer
  attention  : volumetric self-attention        -- content routing over the D*H*W voxel locations

conv3d is the structurally hardest primitive here: a symmetry constraint (3D translation
equivariance) on the parameter space. On genuine volumetric data it should decisively beat
dense/pointwise; a correct metaoptimizer drives alpha toward conv3d where 3D structure is real and
toward the others where it is not (e.g. permuted voxels, where translation equivariance is broken).

All prior modules are left UNTOUCHED; this is a new capability in a new module, mirroring
spatial_schema.py one tensor rank higher.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv3d(conv, sigma_w2=2.0):
    k = conv.kernel_size
    fan_in = conv.in_channels * k[0] * k[1] * k[2]
    with torch.no_grad():
        conv.weight.normal_(0, np.sqrt(sigma_w2 / max(fan_in, 1)))
        if conv.bias is not None:
            conv.bias.zero_()


def _init_lin(lin, sigma_w2=1.0):
    with torch.no_grad():
        lin.weight.normal_(0, np.sqrt(sigma_w2 / lin.in_features))
        if lin.bias is not None:
            lin.bias.zero_()


# --------------------------------------------------------------------------- primitive cores
class _Conv3dVolumetric(nn.Module):
    """3x3x3 convolution, same padding: weight-sharing under the 3D translation group. kernel_size is
    parameterized so larger-receptive-field variants (k5) can subclass this, mirroring the 2D spatial
    schema's conv2d_k5/k7; padding is kept 'same' as kernel_size//2."""
    name = "conv3d"

    def __init__(self, cin, width, dhw, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(cin, width, kernel_size=kernel_size, padding=kernel_size // 2)
        _init_conv3d(self.conv)

    def forward_volumetric(self, x):
        return self.conv(x)


class _Conv3dK5(_Conv3dVolumetric):
    """5x5x5 conv variant -- larger 3D receptive field for volumetric data with longer spatial correlation
    length (the 3D analogue of conv2d_k5). Lets kernel_from_xi extend to the volumetric contract."""
    name = "conv3d_k5"

    def __init__(self, cin, width, dhw):
        super().__init__(cin, width, dhw, kernel_size=5)


class _ConvDWVolumetric(nn.Module):
    """Depthwise 3x3x3 + pointwise 1x1x1 (separable 3D conv): a cheaper conv variant.
    Depthwise operates per-channel, so we first lift to `width` with a 1x1x1, then depthwise-mix
    spatially. Far fewer params than full 3D conv at large width (3D kernels are k^3)."""
    name = "conv_dw"

    def __init__(self, cin, width, dhw):
        super().__init__()
        self.lift = nn.Conv3d(cin, width, kernel_size=1)                              # pointwise lift
        self.dw = nn.Conv3d(width, width, kernel_size=3, padding=1, groups=width)     # depthwise
        _init_conv3d(self.lift); _init_conv3d(self.dw)

    def forward_volumetric(self, x):
        return self.dw(self.lift(x))


class _PointwiseVolumetric(nn.Module):
    """1x1x1 convolution: per-voxel channel mixing, NO spatial weight-sharing across a
    neighborhood -- the 'affine + pointwise' primitive with no 3D receptive field."""
    name = "pointwise"

    def __init__(self, cin, width, dhw):
        super().__init__()
        self.conv = nn.Conv3d(cin, width, kernel_size=1)
        _init_conv3d(self.conv)

    def forward_volumetric(self, x):
        return self.conv(x)


class _DenseVolumetric(nn.Module):
    """Flatten + linear + reshape: unconstrained affine over the whole volume -- the
    no-inductive-bias baseline that convolution should beat on genuine volumetric data.
    Note: at a D*H*W volume this is very large; kept small by the stem's downsampling."""
    name = "dense"

    def __init__(self, cin, width, dhw):
        super().__init__()
        self.cin, self.width, self.dhw = cin, width, dhw
        vol = dhw * dhw * dhw
        self.lin = nn.Linear(cin * vol, width * vol)
        _init_lin(self.lin)

    def forward_volumetric(self, x):
        b = x.shape[0]
        out = self.lin(x.reshape(b, -1))
        return out.reshape(b, self.width, self.dhw, self.dhw, self.dhw)


class _NormVolumetric(nn.Module):
    """BatchNorm3d over channels + a 1x1x1 mix: the volumetric stabilizer (normalization)."""
    name = "norm"

    def __init__(self, cin, width, dhw):
        super().__init__()
        self.proj = nn.Conv3d(cin, width, kernel_size=1)
        self.bn = nn.BatchNorm3d(width)
        _init_conv3d(self.proj)

    def forward_volumetric(self, x):
        return self.bn(self.proj(x))


class _AttentionVolumetric(nn.Module):
    """Single-head volumetric self-attention: content routing over the D*H*W voxel locations.
    Q,K,V are 1x1x1 projections; attention mixes voxels by softmax(QK^T/sqrt(d)). This is the
    routing primitive on a 3D grid (permutation-equivariant over voxel locations). Cost is
    O((DHW)^2) so it is only tractable on small/pooled volumes -- the stem downsamples for this."""
    name = "attention"

    def __init__(self, cin, width, dhw):
        super().__init__()
        self.width = width
        self.q = nn.Conv3d(cin, width, kernel_size=1)
        self.k = nn.Conv3d(cin, width, kernel_size=1)
        self.v = nn.Conv3d(cin, width, kernel_size=1)
        for m in (self.q, self.k, self.v):
            _init_conv3d(m, 1.0)

    def forward_volumetric(self, x):
        b, _, D, H, W = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)      # (b, DHW, width)
        k = self.k(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(1, 2) / (self.width ** 0.5), dim=-1)
        out = att @ v                                  # (b, DHW, width)
        return out.transpose(1, 2).reshape(b, self.width, D, H, W)


_VOLUMETRIC_CORES = {
    "conv3d": _Conv3dVolumetric,
    "conv3d_k5": _Conv3dK5,
    "conv_dw": _ConvDWVolumetric,
    "pointwise": _PointwiseVolumetric,
    "dense": _DenseVolumetric,
    "norm": _NormVolumetric,
    "attention": _AttentionVolumetric,
}


# --------------------------------------------------------------------------- schema cell
class _VolumetricCell(nn.Module):
    """One meta-layer: all primitives in parallel, mixed by softmax(alpha), then BN3d + ReLU."""

    def __init__(self, cin, width, dhw, primitives):
        super().__init__()
        self.primitives = tuple(primitives)
        self.cores = nn.ModuleList([_VOLUMETRIC_CORES[p](cin, width, dhw) for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.bn = nn.BatchNorm3d(width)
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def mixed(self, x):  # canonical uniform name (delegates to forward)
        return self.forward(x)

    def forward(self, x):
        # weighted sum over primitives without stacking (avoids an extra full (P,b,w,D,H,W) copy of all P
        # primitives' volumes; sum_p w_p*out_p == the einsum over the stack). Matches the 4d/operator cells.
        w = torch.softmax(self.alpha, dim=0)
        outs = [c.forward_volumetric(x) for c in self.cores]                      # P x (b,w,D,H,W)
        mixed = sum(wi * o for wi, o in zip(w, outs))
        return F.relu(self.bn(mixed))

    def update_peak(self):
        with torch.no_grad():
            w = torch.softmax(self.alpha, dim=0)
            self.alpha_peak = torch.maximum(self.alpha_peak, w)


class VolumetricSchema(nn.Module):
    """Stack of volumetric meta-layers over a shared stem, then global pool + linear head.

    A shared stem (conv3d + optional downsample) maps the raw volume (b, cin, D, H, W) to
    (b, width, dhw, dhw, dhw), keeping the dense/attention primitives tractable; the schema
    meta-layers (each mixing the full volumetric vocabulary) sit on top; global average pool over
    the three spatial axes + a linear head produce the class logits.
    """

    def __init__(self, width=16, dhw=8, depth=1, n_in=1, n_classes=10, seed=0,
                 primitives=("conv3d", "conv_dw", "pointwise", "dense", "norm", "attention"),
                 vol_size=16):
        super().__init__()
        torch.manual_seed(seed)
        self.primitives = tuple(primitives)
        self.width, self.dhw, self.depth = width, dhw, depth
        stem_stride = max(1, vol_size // dhw)
        self.stem = nn.Sequential(
            nn.Conv3d(n_in, width, kernel_size=3, stride=stem_stride, padding=1),
            nn.BatchNorm3d(width), nn.ReLU(),
        )
        for m in self.stem:
            if isinstance(m, nn.Conv3d):
                _init_conv3d(m)
        self.cells = nn.ModuleList([
            _VolumetricCell(width, width, dhw, self.primitives) for _ in range(depth)
        ])
        self.head = nn.Linear(width, n_classes)
        _init_lin(self.head, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = x.mean(dim=(2, 3, 4))            # global average pool over D,H,W
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


def build_volumetric_schema(width=16, dhw=8, depth=1, n_in=1, n_out=None, seed=0,
                                        primitives=("conv3d", "conv_dw", "pointwise", "dense",
                                                    "norm", "attention"), vol_size=16, n_classes=None):
    # canonical param is n_out; n_classes kept as backward-compatible alias
    if n_out is None:
        n_out = n_classes if n_classes is not None else 10
    return VolumetricSchema(width=width, dhw=dhw, depth=depth, n_in=n_in,
                                       n_classes=n_out, seed=seed, primitives=primitives,
                                       vol_size=vol_size)

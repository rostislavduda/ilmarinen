"""4D (spatiotemporal) schema -- the rank-4 grid-locality contract.

Completes the grid-locality ladder: 1D sequence -> 2D spatial -> 3D volumetric -> 4D spatiotemporal.
Tensor rank is a HARD structural boundary: the N-D translation group is not a subgroup of the (N-1)-D
translation group, so an N-D grid cannot be unrolled to (N-1)-D while preserving locality. Hence 4D
needs its OWN contract, exactly as 3D did.

CONTRACT: input (b, C, T, D, H, W) -- batch of C-channel 4D grids over a 4th axis T (e.g. time) and 3
spatial axes (D,H,W). Symmetry: 4D translation equivariance (weight-sharing over all four grid axes).

conv4d: torch has conv1d/2d/3d but NO conv4d. We implement it EXACTLY (not approximately) by decomposing
the 4D convolution over the temporal axis: a (K_t, K_d, K_h, K_w) kernel = sum over the K_t temporal taps
of a conv3d applied to the correspondingly time-shifted volume. K_t conv3d calls, using cuDNN's conv3d.

Primitive vocabulary (rank-4 compatible):
  conv4d      : full 4D local convolution (defining primitive; space-time coupling).
  conv4d_kt1  : per-time 3D spatial conv (K_t=1, no temporal mixing) -- the space-time SEPARABLE option,
                the genuinely-new axis at rank 4.
  conv_dw     : depthwise 4D conv (per-channel local) -- cheap.
  pointwise   : 1^4 conv = per-voxel channel mixing (pointwise irreducible).
  dense       : full affine over the flattened 4D volume -- INHERENTLY huge ((C*T*D*H*W)^2); small grids
                only (flagged).
  norm        : normalization stabilizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv4d(x, weight, bias=None, padding=(1, 1, 1, 1)):
    """Exact 4D convolution via temporal decomposition into conv3d. x: (B,Cin,T,D,H,W); weight:
    (Cout,Cin,Kt,Kd,Kh,Kw). Returns (B,Cout,T',D',H',W'). 'same' T length via temporal padding."""
    B, Cin, T, D, H, W = x.shape
    Cout, _, Kt, Kd, Kh, Kw = weight.shape
    pt, pd, ph, pw = padding
    # pad along time
    xp = F.pad(x, (0, 0, 0, 0, 0, 0, pt, pt))                      # pad T on both sides
    Tp = xp.shape[2]
    out_T = Tp - Kt + 1
    out = None
    for kt in range(Kt):
        # slice the temporal window for this tap: frames [kt : kt+out_T]
        xt = xp[:, :, kt:kt + out_T]                                # (B,Cin,out_T,D,H,W)
        # merge time into batch, apply conv3d with the kt-th temporal slice of the kernel
        w3 = weight[:, :, kt]                                       # (Cout,Cin,Kd,Kh,Kw)
        xt_ = xt.permute(0, 2, 1, 3, 4, 5).reshape(B * out_T, Cin, D, H, W)
        y = F.conv3d(xt_, w3, bias=bias if kt == 0 else None, padding=(pd, ph, pw))
        y = y.reshape(B, out_T, Cout, y.shape[-3], y.shape[-2], y.shape[-1]).permute(0, 2, 1, 3, 4, 5)
        out = y if out is None else out + y
    return out


class _Conv4dCore(nn.Module):
    def __init__(self, cin, width, kt=3, ks=3, depthwise=False, kt1=False):
        super().__init__()
        self.kt = 1 if kt1 else kt
        self.ks = ks
        groups = cin if depthwise else 1
        cout = cin if depthwise else width
        self.weight = nn.Parameter(torch.randn(cout, cin // groups, self.kt, ks, ks, ks) *
                                   (2.0 / (cin * self.kt * ks ** 3)) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(cout))
        self.depthwise = depthwise; self.groups = groups
        self.proj = None if (depthwise or cout == width) else nn.Conv3d(cout, width, 1)
        self.pt = self.kt // 2; self.ps = ks // 2

    def forward_grid(self, x):
        if self.depthwise:
            # depthwise 4D: per-channel conv4d (grouped) -- do it as grouped conv3d in the temporal loop
            B, C, T, D, H, W = x.shape
            xp = F.pad(x, (0, 0, 0, 0, 0, 0, self.pt, self.pt))
            out_T = xp.shape[2] - self.kt + 1
            out = None
            for kt in range(self.kt):
                xt = xp[:, :, kt:kt + out_T].permute(0, 2, 1, 3, 4, 5).reshape(B * out_T, C, D, H, W)
                w3 = self.weight[:, :, kt]
                y = F.conv3d(xt, w3, padding=(self.ps, self.ps, self.ps), groups=C)
                y = y.reshape(B, out_T, C, y.shape[-3], y.shape[-2], y.shape[-1]).permute(0, 2, 1, 3, 4, 5)
                out = y if out is None else out + y
            return out
        y = conv4d(x, self.weight, self.bias, padding=(self.pt, self.ps, self.ps, self.ps))
        return self.proj(y) if self.proj is not None else y


class _Pointwise4d(nn.Module):
    def __init__(self, cin, width):
        super().__init__(); self.lin = nn.Conv3d(cin, width, 1)   # 1x1x1 over space applied per-time

    def forward_grid(self, x):
        B, C, T, D, H, W = x.shape
        x_ = x.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, D, H, W)
        y = self.lin(x_)
        return y.reshape(B, T, -1, D, H, W).permute(0, 2, 1, 3, 4, 5)


class _Dense4d(nn.Module):
    """Full affine over the flattened 4D volume. INHERENTLY expensive ((C*T*D*H*W)^2); small grids only."""
    def __init__(self, cin, width, shape):
        super().__init__()
        T, D, H, W = shape
        self.vol = T * D * H * W
        self.width = width; self.C = cin; self.shape = shape
        self.lin = nn.Linear(cin * self.vol, width * self.vol)

    def forward_grid(self, x):
        B = x.shape[0]
        y = self.lin(x.reshape(B, -1)).reshape(B, self.width, *self.shape)
        return y


class _Norm4d(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        self.gn = nn.GroupNorm(1, cin)
        self.proj = None if cin == width else nn.Conv3d(cin, width, 1)
        self.out_ch = width if cin != width else cin

    def forward_grid(self, x):
        B, C, T, D, H, W = x.shape
        y = self.gn(x)
        if self.proj is not None:
            y = self.proj(y.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, D, H, W))
            y = y.reshape(B, T, -1, D, H, W).permute(0, 2, 1, 3, 4, 5)
        return y


class _Grid4dCell(nn.Module):
    def __init__(self, cin, width, primitives, grid_shape):
        super().__init__()
        self.primitives = tuple(primitives)
        cores = []
        for p in self.primitives:
            if p == "conv4d":       cores.append(_Conv4dCore(cin, width, kt=3, ks=3))
            elif p == "conv4d_kt1": cores.append(_Conv4dCore(cin, width, kt=3, ks=3, kt1=True))
            elif p == "conv_dw":    cores.append(_Conv4dCore(cin, width, kt=3, ks=3, depthwise=True))
            elif p == "pointwise":  cores.append(_Pointwise4d(cin, width))
            elif p == "dense":      cores.append(_Dense4d(cin, width, grid_shape))
            elif p == "norm":       cores.append(_Norm4d(cin, width))
            else: raise ValueError(f"unknown 4D primitive {p}")
        self.cores = nn.ModuleList(cores)
        # conv_dw keeps cin channels (depthwise) and needs a channel-fixer to reach `width`.
        # norm self-projects internally (_Norm4d.proj), so it must NOT get an external fixer.
        self.fixers = nn.ModuleList([
            (nn.Conv3d(cin, width, 1) if p == "conv_dw" and cin != width else None)
            for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def _fix(self, y, fixer):
        if fixer is None: return y
        B, C, T, D, H, W = y.shape
        y = fixer(y.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, D, H, W))
        return y.reshape(B, T, -1, D, H, W).permute(0, 2, 1, 3, 4, 5)

    def mixed(self, x):
        w = torch.softmax(self.alpha, dim=0)
        outs = [self._fix(c.forward_grid(x), f) for c, f in zip(self.cores, self.fixers)]
        return sum(wi * o for wi, o in zip(w, outs))

    def update_peak(self):
        with torch.no_grad():
            self.alpha_peak = torch.maximum(self.alpha_peak, torch.softmax(self.alpha, dim=0))


class Grid4dSchema(nn.Module):
    def __init__(self, cin, grid_shape, width=8, depth=2, n_out=1,
                 primitives=("conv4d", "conv4d_kt1", "conv_dw", "pointwise", "norm"), seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.grid_shape = grid_shape; self.width = width; self.depth = depth
        self.cells = nn.ModuleList()
        c = cin
        for _ in range(depth):
            self.cells.append(_Grid4dCell(c, width, primitives, grid_shape)); c = width
        self.head = nn.Linear(width, n_out)

    def forward(self, x):
        h = None
        for l, cell in enumerate(self.cells):
            out = cell.mixed(x if l == 0 else h)
            h = out if l == 0 else h + out
        pooled = h.mean(dim=(2, 3, 4, 5))            # global average pool over all 4 grid axes -> (B,width)
        return self.head(pooled)

    def update_peak(self):
        for cell in self.cells: cell.update_peak()

    def alpha_report(self):
        return [torch.softmax(c.alpha, dim=0).detach().cpu().numpy() for c in self.cells]

    def alpha_peak_report(self):
        return [c.alpha_peak.detach().cpu().numpy() for c in self.cells]

    def architecture(self):
        with torch.no_grad():
            return {"primitives": [c.primitives[int(torch.argmax(c.alpha))] for c in self.cells]}


def build_grid4d_schema(n_in=None, grid_shape=None, width=8, depth=2, n_out=1,
                                primitives=("conv4d", "conv4d_kt1", "conv_dw", "pointwise", "norm"),
                                seed=0, cin=None):
    """grid_shape = (T, D, H, W). Vocabulary defaults exclude 'dense'/'attention' (huge at 4D); add them
    explicitly for small grids."""
    # canonical param is n_in; cin kept as backward-compatible alias
    if n_in is None:
        n_in = cin
    if n_in is None:
        raise TypeError("build_grid4d_schema requires n_in (channel count)")
    if grid_shape is None:
        raise TypeError("build_grid4d_schema requires grid_shape=(T,D,H,W)")
    return Grid4dSchema(n_in, grid_shape, width, depth, n_out, primitives, seed)

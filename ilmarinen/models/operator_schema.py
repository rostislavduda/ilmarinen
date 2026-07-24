"""Neural-operator schema (the 8th contract): maps FUNCTIONS to FUNCTIONS on a grid.

This is a genuinely different contract from the seven existing contracts. Sequence/spatial/volumetric/4d map
fixed-rank TENSORS; graph/equivariant/set map irregular structures; all produce a FIXED-SIZE output. A
NEURAL OPERATOR maps a function a(x) (sampled on a grid) to a function u(x) (on a grid) and its defining
property is DISCRETIZATION INVARIANCE: trained at one resolution it evaluates at another, because the
learnable weights live in a resolution-independent space (Fourier modes), not on grid points. The canonical
instance is the Fourier Neural Operator (Li et al. 2021): lift -> [spectral conv + pointwise skip] blocks
-> project, where the spectral conv keeps the low-k Fourier modes, applies a learnable COMPLEX weight per
mode, and inverse-transforms back to a function.

The existing `spectral` sequence primitive is NOT this: it rFFTs then COLLAPSES the spectrum to fixed
statistics (mean+max power) -- it discards the function. Here the function is preserved end to end.

Primitives on the alpha-simplex are operator layers that differ in their spectral treatment (how many modes,
whether a pointwise/local skip is included), so the metaoptimizer selects the operator's spectral capacity
the same way it selects primitives elsewhere. All are discretization-invariant by construction.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _fft_safe(fn, *tensors):
    """Run an FFT-based closure, transparently falling back to CPU if the backend lacks the kernel.

    torch.fft / complex ops have incomplete coverage on some MPS (Apple-Silicon) PyTorch/macOS versions;
    when that is the case the call raises NotImplementedError (or a RuntimeError mentioning the missing op).
    Rather than force the whole model onto the CPU, this runs just the spectral computation on the CPU and
    moves the result back to the input device. If PYTORCH_ENABLE_MPS_FALLBACK=1 is set (recommended), the
    op runs on device with PyTorch's own fallback and this wrapper is a no-op. On CUDA/CPU it is always a
    no-op. The first tensor's device is the target device for the result.
    """
    try:
        return fn(*tensors)
    except (NotImplementedError, RuntimeError) as e:
        msg = str(e).lower()
        if "mps" not in msg and "not implemented" not in msg and "fft" not in msg:
            raise
        dev = tensors[0].device
        cpu_tensors = [t.to("cpu") for t in tensors]
        return fn(*cpu_tensors).to(dev)


def _init_linear(m, gain=1.0, bias=0.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias)


class _SpectralConv1d(nn.Module):
    """The core operator layer: keep the lowest `modes` Fourier modes, apply a learnable complex weight per
    mode (a per-mode linear mix across channels), inverse-transform back to a function at the input
    resolution. Resolution-independent: the parameter tensor is indexed by MODE, not by grid point, so the
    same weights act at any grid size (discretization invariance)."""

    def __init__(self, cin, cout, modes):
        super().__init__()
        self.cin, self.cout, self.modes = cin, cout, modes
        scale = 1.0 / (cin * cout)
        # complex weight per (in-channel, out-channel, mode), stored as (...,2) real/imag
        self.weight = nn.Parameter(scale * torch.randn(cin, cout, modes, 2))

    def forward(self, x):  # x: (b, cin, N)
        def _spec(x):
            b, cin, N = x.shape
            xf = torch.fft.rfft(x, dim=-1)                          # (b, cin, N//2+1) complex
            m = min(self.modes, xf.shape[-1])
            wc = torch.view_as_complex(self.weight)[:, :, :m]       # (cin, cout, m)
            out = torch.zeros(b, self.cout, xf.shape[-1], dtype=torch.cfloat, device=x.device)
            out[..., :m] = torch.einsum("bim,iom->bom", xf[..., :m], wc)
            # MPS rfft/irfft can silently emit non-finite values on some PyTorch/macOS versions (CPU stays
            # finite); keep the returned field finite so training does not diverge on a NaN. Mirrors the
            # sequence spectral primitive's guard (models/schema.py). No-op on CPU/CUDA (already finite).
            return torch.nan_to_num(torch.fft.irfft(out, n=N, dim=-1))  # (b, cout, N) -- a FUNCTION at res N
        return _fft_safe(_spec, x)


class _SpectralConv2d(nn.Module):
    """2D spectral conv (FNO2d): rfft2, keep a low-frequency block in each axis, learnable complex weight
    per kept (kx,ky), irfft2. Two blocks are kept -- low-kx x low-ky AND high-kx(negative) x low-ky -- so
    both signs of the kx frequency are represented (the last axis is the rfft half-axis). Resolution-
    independent: weights indexed by MODE, so the same layer acts at any grid size."""

    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.cin, self.cout, self.m1, self.m2 = cin, cout, m1, m2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, 2))

    def forward(self, x):  # (b, cin, H, W)
        def _spec(x):
            b, cin, H, W = x.shape
            xf = torch.fft.rfft2(x, dim=(-2, -1))                  # (b, cin, H, W//2+1)
            out = torch.zeros(b, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1 = min(self.m1, H // 2); m2 = min(self.m2, W // 2 + 1)
            w1 = torch.view_as_complex(self.w1)[:, :, :m1, :m2]
            w2 = torch.view_as_complex(self.w2)[:, :, :m1, :m2]
            out[:, :, :m1, :m2] = torch.einsum("bihw,iohw->bohw", xf[:, :, :m1, :m2], w1)
            out[:, :, -m1:, :m2] = torch.einsum("bihw,iohw->bohw", xf[:, :, -m1:, :m2], w2)
            return torch.nan_to_num(torch.fft.irfft2(out, s=(H, W), dim=(-2, -1)))  # keep field finite (MPS FFT guard)
        return _fft_safe(_spec, x)


class _SpectralConv3d(nn.Module):
    """3D spectral conv (FNO3d): rfftn over the 3 spatial axes, keep a low-frequency block per axis with
    all four sign combinations of the two full-FFT axes (the last axis is the rfft half-axis), learnable
    complex weight per kept mode, irfftn. Resolution-independent in all three axes."""

    def __init__(self, cin, cout, m1, m2, m3):
        super().__init__()
        self.cin, self.cout, self.m1, self.m2, self.m3 = cin, cout, m1, m2, m3
        scale = 1.0 / (cin * cout)
        # four blocks for the (+/- kx, +/- ky) sign combinations; kz is the rfft half-axis (low only)
        self.ws = nn.ParameterList([nn.Parameter(scale * torch.randn(cin, cout, m1, m2, m3, 2))
                                    for _ in range(4)])

    def forward(self, x):  # (b, cin, D, H, W)
        def _spec(x):
            b, cin, D, H, W = x.shape
            xf = torch.fft.rfftn(x, dim=(-3, -2, -1))             # (b, cin, D, H, W//2+1)
            out = torch.zeros(b, self.cout, D, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1 = min(self.m1, D // 2); m2 = min(self.m2, H // 2); m3 = min(self.m3, W // 2 + 1)
            slices = [(slice(None, m1), slice(None, m2)),
                      (slice(None, m1), slice(-m2, None)),
                      (slice(-m1, None), slice(None, m2)),
                      (slice(-m1, None), slice(-m2, None))]
            for k, (sx, sy) in enumerate(slices):
                w = torch.view_as_complex(self.ws[k])[:, :, :m1, :m2, :m3]
                out[:, :, sx, sy, :m3] = torch.einsum("bidhw,iodhw->bodhw", xf[:, :, sx, sy, :m3], w)
            return torch.nan_to_num(torch.fft.irfftn(out, s=(D, H, W), dim=(-3, -2, -1)))  # keep field finite (MPS FFT guard)
        return _fft_safe(_spec, x)




def _make_spectral(width, modes, sdims):
    """Dimension-generic spectral conv factory: picks the 1D/2D/3D FNO spectral conv, using `modes` per
    spatial axis (same mode budget on each axis)."""
    if sdims == 1:
        return _SpectralConv1d(width, width, modes)
    if sdims == 2:
        return _SpectralConv2d(width, width, modes, modes)
    if sdims == 3:
        return _SpectralConv3d(width, width, modes, modes, modes)
    raise ValueError(f"operator contract supports spatial_dims in {{1,2,3}}, got {sdims}")


def _make_pointwise(width, sdims):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[sdims](width, width, 1)


class _FourierOp(nn.Module):
    """Standard FNO layer (any spatial rank): spectral conv (global, low-mode) + pointwise 1x1 conv (local
    skip), summed then activated. Both parts are resolution-independent."""
    name = "fourier"

    def __init__(self, width, modes=12, sdims=1):
        super().__init__()
        self.spectral = _make_spectral(width, modes, sdims)
        self.pointwise = _make_pointwise(width, sdims)

    def forward(self, h, coords=None):
        return torch.relu(self.spectral(h) + self.pointwise(h))


class _FourierWideOp(nn.Module):
    """FNO layer with MORE spectral modes (higher spectral capacity) for less-smooth kernels. A distinct
    point on the alpha-simplex from `fourier`."""
    name = "fourier_wide"

    def __init__(self, width, modes=24, sdims=1):
        super().__init__()
        self.spectral = _make_spectral(width, modes, sdims)
        self.pointwise = _make_pointwise(width, sdims)

    def forward(self, h, coords=None):
        return torch.relu(self.spectral(h) + self.pointwise(h))


class _LocalOp(nn.Module):
    """Pointwise-only operator layer (no spectral mixing): a resolution-independent local (Nemytskii)
    operator u(x)=sigma(W a(x)). The 'no global coupling' baseline; lets the metaoptimizer discover when a
    task needs global spectral coupling vs only a local nonlinearity."""
    name = "local"

    def __init__(self, width, modes=None, sdims=1):
        super().__init__()
        self.pointwise = _make_pointwise(width, sdims)

    def forward(self, h, coords=None):
        return torch.relu(self.pointwise(h))


class _DeepONetOp(nn.Module):
    """DeepONet-style (branch-trunk) operator layer -- the mesh-free, GLOBAL-low-rank complement to the
    spectral FNO primitives (Lu et al. 2021). Where `fourier` couples via a fixed low-mode Fourier basis and
    `local` is purely pointwise, this layer forms a rank-p GLOBAL operator:
        branch: globally pool the field (mean over the grid) -> per-channel coefficients b_k  (b, width, p)
        trunk:  map the query coordinates -> a spatial basis t_k(x)                             (b, *grid, p)
        out:    u(x) = sum_k b_k t_k(x)                                                          (b, width, *grid)
    This is the DeepONet inner-product u(y)=sum_k branch_k(a) trunk_k(y) lifted to a per-channel operator
    layer, so it (a) keeps the field->field interface (mixes on the alpha-simplex with fourier/local and
    chains across depth) and (b) carries DeepONet's defining strength: the trunk is a continuous function of
    the coordinate, so it is MESH-FREE (evaluable at query points off the training grid), unlike the FFT-
    bound spectral primitives. The metaoptimizer selects it when the operator is better captured by a few
    global modes + a learned coordinate basis than by low Fourier modes."""
    name = "deeponet"

    def __init__(self, width, modes=None, sdims=1, p=24):
        super().__init__()
        self.width, self.sdims, self.p = width, sdims, p
        # branch: per-channel global-coefficient generator from the pooled field (invariant to grid size)
        self.branch = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width * p))
        # trunk: coordinate -> p-dim basis (a continuous, mesh-free function of position)
        self.trunk = nn.Sequential(nn.Linear(sdims, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, p))
        self.mix = nn.Conv1d(width, width, 1) if sdims == 1 else (
            nn.Conv2d(width, width, 1) if sdims == 2 else nn.Conv3d(width, width, 1))

    def forward(self, h, coords):
        # h: (b, width, *grid); coords: (b, *grid, sdims)
        b = h.shape[0]; grid = h.shape[2:]
        # BRANCH: global mean-pool over the grid -> (b, width) -> per-channel coeffs (b, width, p)
        pooled = h.flatten(2).mean(-1)                               # (b, width)
        coeff = self.branch(pooled).view(b, self.width, self.p)     # (b, width, p)
        # TRUNK: coordinate basis (b, *grid, p) -- continuous in the coordinate (mesh-free)
        basis = self.trunk(coords)                                   # (b, *grid, p)
        # combine: u[b,c,*grid] = sum_p coeff[b,c,p] * basis[b,*grid,p]
        basis_flat = basis.flatten(1, len(grid))                     # (b, G, p)
        out = torch.einsum("bcp,bgp->bcg", coeff, basis_flat)       # (b, width, G)
        out = out.view(b, self.width, *grid)                        # (b, width, *grid)
        return torch.relu(self.mix(out))


_OP_CORES = {
    "fourier": _FourierOp,
    "fourier_wide": _FourierWideOp,
    "local": _LocalOp,
    "deeponet": _DeepONetOp,
}
# per-primitive default mode budget (fourier_wide gets more modes); local/deeponet ignore modes
_OP_MODES = {"fourier": 12, "fourier_wide": 24, "local": None, "deeponet": None}


class _OperatorCell(nn.Module):
    """Alpha-mixed operator cell, matching the schema pattern used by the other contracts: a tuple of
    operator primitives, a learnable alpha simplex over them (the contract-agnostic Gibbs hook), forward is
    the softmax-weighted mixture. Dimension-generic via sdims."""

    def __init__(self, width, primitives, modes=12, sdims=1, mode_override=None):
        super().__init__()
        self.primitives = tuple(primitives)
        # mode_override (B7): when set, all mode-using primitives (fourier, fourier_wide) use THIS budget
        # instead of their hardcoded _OP_MODES defaults -- so a priced mode selection actually controls the
        # spectral d.o.f. local/deeponet ignore modes regardless.
        def _pmodes(p):
            if p == "local":
                return None
            if mode_override is not None and _OP_MODES[p] is not None:
                return int(mode_override)
            return _OP_MODES[p] if _OP_MODES[p] is not None else modes
        self.cores = nn.ModuleList([
            _OP_CORES[p](width, _pmodes(p), sdims)
            for p in self.primitives])
        self.alpha = nn.Parameter(torch.zeros(len(self.primitives)))
        self.register_buffer("alpha_peak", torch.zeros(len(self.primitives)))

    def forward(self, h, coords=None):
        w = torch.softmax(self.alpha, dim=0)
        out = sum(w[i] * core(h, coords) for i, core in enumerate(self.cores))
        if self.training:
            self.alpha_peak = torch.maximum(self.alpha_peak, torch.softmax(self.alpha, dim=0))
        return out


class OperatorSchema(nn.Module):
    """Fourier Neural Operator schema for 1D/2D/3D function-valued data. Input: a field a(x) sampled on
    a grid, plus the grid coordinates (so the lift sees position). Output: a field u(x) on the SAME grid.
    Discretization-invariant: train at one resolution, evaluate at any other, in every spatial axis. Depth =
    number of operator cells; width = channel lifting dim; spatial_dims in {1,2,3}."""

    def __init__(self, width=32, depth=2, n_out=1, primitives=("fourier", "fourier_wide", "local", "deeponet"),
                 modes=12, in_channels=1, spatial_dims=1, mode_override=None):
        super().__init__()
        self.width = width
        self.spatial_dims = spatial_dims
        self.lift = nn.Linear(in_channels + spatial_dims, width)   # (a(x), coords) -> width
        self.cells = nn.ModuleList([_OperatorCell(width, primitives, modes, spatial_dims, mode_override=mode_override)
                                    for _ in range(depth)])
        self.proj1 = nn.Linear(width, 64)
        self.proj2 = nn.Linear(64, n_out)
        _init_linear(self.lift); _init_linear(self.proj1); _init_linear(self.proj2)

    def forward(self, a, coords):
        """a: (b, *grid, in_channels) or (b, *grid); coords: (b, *grid, spatial_dims). grid is N (1D),
        (H,W) (2D), or (D,H,W) (3D)."""
        sd = self.spatial_dims
        if a.dim() == 1 + sd:                                    # no channel axis -> add one
            a = a.unsqueeze(-1)
        h = torch.cat([a, coords], dim=-1)                      # (b, *grid, in_ch+sd)
        # move channels to position 1: (b, C, *grid)
        perm = [0, h.dim() - 1] + list(range(1, h.dim() - 1))
        h = self.lift(h).permute(*perm).contiguous()
        for cell in self.cells:
            h = cell(h, coords)
        # move channels back to last: (b, *grid, width)
        inv = [0] + list(range(2, h.dim())) + [1]
        h = h.permute(*inv).contiguous()
        u = self.proj2(torch.relu(self.proj1(h)))               # (b, *grid, n_out)
        return u.squeeze(-1)

    def alpha_weights(self):
        return [torch.softmax(c.alpha, dim=0).detach().cpu().numpy() for c in self.cells]

    def alpha_report(self):
        # alias matching the other contracts' interface (softmax of the live alpha per cell)
        return self.alpha_weights()

    def architecture(self):
        # per-cell selected primitive by peak alpha over training, matching the other schemas
        import numpy as _np
        return [c.primitives[int(_np.argmax(c.alpha_peak.detach().cpu().numpy()))]
                for c in self.cells]


def build_operator_schema(width=32, depth=2, n_out=1,
                                      primitives=("fourier", "fourier_wide", "local", "deeponet"),
                                      modes=12, in_channels=1, spatial_dims=1, mode_override=None, n_in=None):
    # the operator contract's input width is its channel count; accept the family-canonical n_in as an alias
    if n_in is not None:
        in_channels = n_in
    return OperatorSchema(width=width, depth=depth, n_out=n_out,
                                     primitives=primitives, modes=modes, in_channels=in_channels,
                                     spatial_dims=spatial_dims, mode_override=mode_override)


# --------------------------------------------------------------------------- standalone mesh-free DeepONet
class StandaloneDeepONet(nn.Module):
    """TRUE (mesh-free) DeepONet, kept as a distinct architecture rather than a cell primitive, because its
    defining property -- evaluating the output at ARBITRARY query coordinates decoupled from the input
    sensor grid -- cannot survive the grid-locked schema pipeline (where lift concatenates the field
    with its own coordinates and every cell preserves the grid). Here the branch reads the input function at
    m fixed sensors and the trunk reads ANY query coordinate, so u(y)=sum_k b_k(f) t_k(y) can be evaluated
    off the training grid (query at higher resolution or scattered points). This is DeepONet's unique value
    over the FFT-bound FNO primitives; the metaoptimizer's operator schema offers a cell-integrated
    global 'deeponet' primitive for GRID tasks, and this standalone builder for genuinely MESH-FREE tasks."""

    def __init__(self, n_sensors, sdims=1, p=40, branch_hidden=128, trunk_hidden=128):
        super().__init__()
        self.p = p
        self.branch = nn.Sequential(nn.Linear(n_sensors, branch_hidden), nn.ReLU(),
                                    nn.Linear(branch_hidden, branch_hidden), nn.ReLU(),
                                    nn.Linear(branch_hidden, p))
        self.trunk = nn.Sequential(nn.Linear(sdims, trunk_hidden), nn.ReLU(),
                                   nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
                                   nn.Linear(trunk_hidden, p))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, f, y):
        """f: (b, m) input function at the m sensors; y: (Q, sdims) query coordinates (ANY points).
        Returns (b, Q): the output function evaluated at each query coordinate for each input."""
        b = self.branch(f)                       # (b, p)
        t = self.trunk(y)                        # (Q, p)
        return b @ t.T + self.b0                 # (b, Q) -- mesh-free evaluation


def build_standalone_deeponet(n_sensors, sdims=1, p=40):
    return StandaloneDeepONet(n_sensors=n_sensors, sdims=sdims, p=p)

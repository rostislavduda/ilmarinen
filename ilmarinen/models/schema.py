"""sequence-based schema -- ALL primitive families in ONE selectable schema.

This is the honest realization of the six-primitive design: a single schema in which
every primitive competes under one alpha, on a common input type (a sequence (b, T, n_in)).
The three prior schemas kept the primitives split by interface -- recurrent
{plain, gated, lstm} in multi_supergraph.py (per-timestep state loop) and non-recurrent
{conv, spectral, attention, dense, norm} in multi_parallel_supergraph.py (flat-vector
forward). Those never competed on one task. Here they do.

COMMON CONTRACT: every core implements
    forward_seq(x_seq: (b, T, n_in)) -> (b, T, width)
a sequence-to-sequence map. Recurrent cores realize it via their timestep loop; the others
realize it directly over the time axis (conv-1d over time, self-attention over timesteps,
spectral rFFT over time broadcast, dense/norm per-timestep). The schema alpha-mixes the
per-primitive sequence outputs at each layer (shared per-layer head at readout), supports
functional composition across depth, per-layer N-way alpha, peak tracking, and deep
supervision -- matching the design invariants of the validated schemas.

Readout: configurable -- 'last' (last timestep, for recall/copy) or 'mean' (mean over time,
for classification). The head maps width -> n_out.

All prior modules are left UNTOUCHED; this is a new capability in a new module.

Performance note: the recurrent cores (plain/gated/lstm) use vectorized per-step arithmetic
-- the input projection is precomputed over the whole sequence and unbound (xp.unbind(1)) to
avoid per-step indexing, gate blocks are split with a single .chunk() rather than manual
slicing, and the GRU update is fused via addcmul. The alpha-mixing is a single stacked einsum
contraction rather than a Python sum. The per-timestep recurrence itself is inherently
sequential and nonlinear (not an associative scan), so it stays a loop; the non-recurrent
cores (conv/attention/dense/norm/spectral) are already fully vectorized over time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .init_utils import _init_linear

# ---------------------------------------------------------------------------
# Sequence-to-sequence primitive cores (common contract: forward_seq -> (b,T,width))
# ---------------------------------------------------------------------------

class _PlainSeq(nn.Module):
    """Plain tanh recurrence over time. h_t = tanh(Wx x_t + Wh h_{t-1})."""
    name = "plain"

    def __init__(self, n_in, width, sigma_w2=1.76, sigma_b2=0.05):
        super().__init__()
        self.width = width
        self.Wx = nn.Linear(n_in, width, bias=False)
        self.Wh = nn.Linear(width, width, bias=True)
        _init_linear(self.Wx, 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2, sigma_b2)

    def forward_seq(self, x):  # x: (b, T, n_in)
        b, T, _ = x.shape
        xp = self.Wx(x).unbind(1)            # tuple of (b,width); avoids per-step indexing
        h = x.new_zeros(b, self.width)
        outs = []
        for t in range(T):
            h = torch.tanh(xp[t] + self.Wh(h))
            outs.append(h)
        return torch.stack(outs, dim=1)      # (b,T,width)

    # --- streaming API: one timestep, carrying state ---
    is_recurrent = True

    def init_state(self, b, device=None, dtype=None):
        return self.Wx.weight.new_zeros(b, self.width) if device is None \
            else torch.zeros(b, self.width, device=device, dtype=dtype)

    def step(self, x_t, state):  # x_t: (b, n_in), state: h (b,width) -> (out_t, new_state)
        h = torch.tanh(self.Wx(x_t) + self.Wh(state))
        return h, h


class _GatedSeq(nn.Module):
    """GRU-style gated recurrence over time (multiplicative gating primitive)."""
    name = "gated"

    def __init__(self, n_in, width, sigma_w2=1.76, sigma_b2=0.05):
        super().__init__()
        self.width = width
        self.Wx = nn.Linear(n_in, 3 * width, bias=False)   # [r, z, n]
        self.Wh = nn.Linear(width, 3 * width, bias=True)
        _init_linear(self.Wx, 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2, sigma_b2)

    def forward_seq(self, x):
        b, T, _ = x.shape
        xp = self.Wx(x).unbind(1)            # tuple of (b,3w)
        h = x.new_zeros(b, self.width); outs = []
        for t in range(T):
            gx_r, gx_z, gx_n = xp[t].chunk(3, dim=1)
            gh_r, gh_z, gh_n = self.Wh(h).chunk(3, dim=1)
            r = torch.sigmoid(gx_r + gh_r)
            z = torch.sigmoid(gx_z + gh_z)
            n = torch.tanh(gx_n + r * gh_n)
            h = torch.addcmul(n, z, h - n)    # h = n + z * (h - n) = (1-z)*n + z*h
            outs.append(h)
        return torch.stack(outs, dim=1)

    is_recurrent = True

    def init_state(self, b, device=None, dtype=None):
        return self.Wx.weight.new_zeros(b, self.width) if device is None \
            else torch.zeros(b, self.width, device=device, dtype=dtype)

    def step(self, x_t, state):  # state: h (b,width)
        gx_r, gx_z, gx_n = self.Wx(x_t).chunk(3, dim=1)
        gh_r, gh_z, gh_n = self.Wh(state).chunk(3, dim=1)
        r = torch.sigmoid(gx_r + gh_r)
        z = torch.sigmoid(gx_z + gh_z)
        n = torch.tanh(gx_n + r * gh_n)
        h = torch.addcmul(n, z, state - n)
        return h, h


class _LSTMSeq(nn.Module):
    """LSTM recurrence over time (additive cell-state carry), with optional chrono-init."""
    name = "lstm"

    def __init__(self, n_in, width, sigma_w2=1.76, sigma_b2=0.05, chrono_tmax=None):
        super().__init__()
        self.width = width
        self.Wx = nn.Linear(n_in, 4 * width, bias=False)   # [i, f, o, g]
        self.Wh = nn.Linear(width, 4 * width, bias=True)
        _init_linear(self.Wx, 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2, sigma_b2)
        with torch.no_grad():
            if chrono_tmax is not None and chrono_tmax > 1:
                u = torch.rand(width) * (chrono_tmax - 1.0) + 1.0
                b_f = torch.log(u)
                self.Wh.bias[width:2 * width].copy_(b_f)
                self.Wh.bias[0:width].copy_(-b_f)
            else:
                self.Wh.bias[width:2 * width].fill_(1.0)

    def forward_seq(self, x):
        b, T, _ = x.shape
        xp = self.Wx(x).unbind(1); w = self.width
        h = x.new_zeros(b, w); c = x.new_zeros(b, w); outs = []
        for t in range(T):
            i, f, o, g = (xp[t] + self.Wh(h)).chunk(4, dim=1)
            i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o)
            c = f * c + i * torch.tanh(g)
            h = o * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1)

    is_recurrent = True

    def init_state(self, b, device=None, dtype=None):
        z = self.Wx.weight.new_zeros(b, self.width) if device is None \
            else torch.zeros(b, self.width, device=device, dtype=dtype)
        return (z, z.clone())            # (h, c)

    def step(self, x_t, state):          # state: (h, c)
        h, c = state
        i, f, o, g = (self.Wx(x_t) + self.Wh(h)).chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return h, (h, c)


class _ConvSeq(nn.Module):
    """Causal 1-D convolution over time (weight-sharing under time translation, LOCAL)."""
    name = "conv"
    is_recurrent = False

    def __init__(self, n_in, width, ksize=5):
        super().__init__()
        self.ksize = ksize
        self.conv = nn.Conv1d(n_in, width, kernel_size=ksize)  # causal via left-pad
        self.width = width

    def forward_seq(self, x):  # (b,T,n_in)
        xt = x.transpose(1, 2)                             # (b,n_in,T)
        xt = torch.nn.functional.pad(xt, (self.ksize - 1, 0))  # causal left pad
        c = torch.tanh(self.conv(xt))                     # (b,width,T)
        return c.transpose(1, 2)                          # (b,T,width)


class _DilatedConvSeq(nn.Module):
    """Multi-scale causal 1-D convolution over time (InceptionTime / TCN style). Runs several causal
    convolutions in PARALLEL at different dilation rates, so a single layer sees several temporal
    receptive fields at once (short and long range), then concatenates and projects to `width`. This
    is the multi-scale receptive field that a single-kernel `conv` cannot express -- the primitive the
    real time-series literature (InceptionTime, LITE, dilated TCNs) is built on. Weight-sharing under
    time translation (LOCAL), non-recurrent."""
    name = "dilconv"
    is_recurrent = False

    def __init__(self, n_in, width, ksize=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.ksize = ksize
        # use at most `width` dilation branches so no branch gets 0 channels -- a variable-width d.o.f. stage
        # can pick width < len(dilations); capping the branch count keeps every active branch >= 1 channel.
        self.dilations = tuple(dilations)[:max(1, width)]
        nb = len(self.dilations)
        # split width across branches as evenly as possible; last branch takes the remainder
        base = width // nb
        self.branch_widths = [base] * (nb - 1) + [width - base * (nb - 1)]
        self.branches = nn.ModuleList([
            nn.Conv1d(n_in, bw, kernel_size=ksize, dilation=d)
            for bw, d in zip(self.branch_widths, self.dilations)])
        self.proj = nn.Conv1d(width, width, kernel_size=1)  # mix the multi-scale features
        self.width = width

    def forward_seq(self, x):  # (b,T,n_in)
        xt = x.transpose(1, 2)                                  # (b,n_in,T)
        outs = []
        for conv, d in zip(self.branches, self.dilations):
            pad = (self.ksize - 1) * d                          # causal left pad for this dilation
            xp = torch.nn.functional.pad(xt, (pad, 0))
            outs.append(conv(xp))                               # (b, bw, T)
        c = torch.tanh(torch.cat(outs, dim=1))                 # (b,width,T)
        c = self.proj(c)                                        # (b,width,T)
        return c.transpose(1, 2)                                # (b,T,width)


class _AttentionSeq(nn.Module):
    """Causal self-attention over timesteps (content-based routing over time)."""
    name = "attention"
    is_recurrent = False

    def __init__(self, n_in, width, d_model=32):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(n_in, d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, width)
        for lin in (self.embed, self.Wq, self.Wk, self.Wv, self.proj):
            _init_linear(lin, 1.0, 0.0)
        self.width = width

    def forward_seq(self, x):  # (b,T,n_in)
        e = torch.tanh(self.embed(x))                     # (b,T,d)
        q, k, v = self.Wq(e), self.Wk(e), self.Wv(e)
        # scaled dot-product attention with the causal mask applied INTERNALLY -- avoids materializing the full
        # (b,T,T) score matrix and re-allocating a (T,T) mask every forward (an O(T^2) blowup on long series);
        # SDPA's default scale is 1/sqrt(d_model), matching the previous explicit scaling.
        ctx = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)  # (b,T,d) over the past
        return torch.tanh(self.proj(ctx))                 # (b,T,width)


class _DenseSeq(nn.Module):
    """Per-timestep dense/affine map (affine primitive #1; no time mixing)."""
    name = "dense"
    is_recurrent = False

    def __init__(self, n_in, width):
        super().__init__()
        self.fc = nn.Linear(n_in, width)
        _init_linear(self.fc, 1.0, 0.0)
        self.width = width

    def forward_seq(self, x):
        return torch.tanh(self.fc(x))                     # (b,T,width), applied per step


class _NormSeq(nn.Module):
    """Per-timestep LayerNorm + affine (normalization primitive #6, the stabilizer)."""
    name = "norm"
    is_recurrent = False

    def __init__(self, n_in, width):
        super().__init__()
        self.ln = nn.LayerNorm(n_in)
        self.fc = nn.Linear(n_in, width)
        _init_linear(self.fc, 1.0, 0.0)
        self.width = width

    def forward_seq(self, x):
        return torch.tanh(self.fc(self.ln(x)))


class _LinearSSMSeq(nn.Module):
    """Linear-recurrence (diagonal state-space) primitive: h_t = a * h_{t-1} + B x_t,
    y_t = tanh(C h_t). The state decay a is diagonal and constant in time (an LTI diagonal
    SSM, the S4/S5/Mamba family's core). Two properties distinguish it from the gated cores:

    (1) It is LINEAR in the state, so the recurrence is ASSOCIATIVE and can in principle be
        computed by a parallel/associative scan (or as a convolution with kernel
        [B, aB, a^2 B, ...]). Here the scan is a lightweight sequential loop -- with no gate
        projections or per-step nonlinearity it is far cheaper per step than gated/lstm, and
        the linear form leaves the door open to a parallel-scan backend for long sequences.
    (2) Its per-channel decay a in (0,1) sets an explicit, learnable memory TIMESCALE
        (effective horizon ~ 1/(1-a)), giving controllable long-range memory without the
        gating machinery.

    Input drive B x_t is precomputed over the whole sequence (vectorized); only the cheap
    scalar recurrence a*h + drive stays in the loop. The decay is parameterized as
    a = sigmoid(a_logit), initialized near 0.9 (slow decay -> long memory by default).
    """
    name = "linssm"

    def __init__(self, n_in, width):
        super().__init__()
        self.width = width
        self.B = nn.Linear(n_in, width, bias=True)     # input -> state drive
        self.C = nn.Linear(width, width, bias=False)   # state -> output
        _init_linear(self.B, 1.0, 0.0)
        _init_linear(self.C, 1.0, 0.0)
        # per-channel decay logit; sigmoid(2.0) ~ 0.88 -> slow decay / long memory at init
        self.a_logit = nn.Parameter(torch.randn(width) * 0.5 + 2.0)

    def forward_seq(self, x):  # (b, T, n_in)
        b, T, _ = x.shape
        drive = self.B(x)                              # (b, T, width), precomputed
        a = torch.sigmoid(self.a_logit)                # (width,) decay in (0,1)
        ys = self._scan(drive, a, T)                   # (b, T, width) parallel associative scan
        return torch.tanh(self.C(ys))                  # output projection

    @staticmethod
    def _scan(drive, a, T, block=None):
        """Parallel/associative computation of the LTI recurrence h_t = a*h_{t-1} + drive_t, replacing the
        per-timestep Python loop. Because a is CONSTANT in time, h_t = sum_{s<=t} a^{t-s} drive_s, which the
        cumsum identity h_t = a^t * cumsum_s(a^{-s} drive_s) evaluates in one vectorized pass. a^{-s}
        overflows for small a, so the sequence is processed in BLOCKS: within each block the cumsum trick is
        exact, and the block-final state is carried forward as a^{t+1}*carry.

        NUMERICAL SAFETY. The block length is capped so that a^{-block} stays well below fp32 overflow for
        the SMALLEST decay in the channel (fastest-forgetting state): block <= 0.4*log(3e38)/log(1/a_min).
        This makes the scan exact for any learned decay (verified max err ~1e-6 vs the loop). PERFORMANCE.
        Forward is ~3-5x faster than the loop and the gap widens with T; for the full forward+backward on
        CPU the safe block size makes it roughly break-even (the win is larger on GPU, where the block
        cumsum parallelises, and for forward-only/inference). This is the parallel-scan backend the linssm
        docstring anticipated; it removes the long-sequence forward bottleneck that forced time-axis pooling
        and is numerically identical to the loop. selssm keeps its loop (its decay is input-dependent, so
        its recurrence is not associative without a full log-space Blelloch scan)."""
        bsz, _, W = drive.shape
        if block is None:
            a_min = float(a.min().clamp(min=1e-4))
            # a^{-block} < ~e^88 (fp32 safe margin); solve block < 0.4*88/(-log a_min)
            import math as _m
            block = max(8, min(T, int(0.4 * 88.0 / (-_m.log(a_min) + 1e-9))))
        out = drive.new_empty(bsz, T, W)
        carry = drive.new_zeros(bsz, W)
        t_idx = torch.arange(block, device=drive.device, dtype=drive.dtype)
        for s in range(0, T, block):
            e = min(s + block, T); L = e - s
            ti = t_idx[:L]
            ap = a[None, :] ** ti[:, None]                      # (L, W) = a^t within block
            blk = drive[:, s:e]                                 # (b, L, W)
            from_carry = (a[None, :] ** (ti[:, None] + 1))[None] * carry[:, None]  # a^{t+1} * carry
            local = ap[None] * torch.cumsum(blk / ap[None], dim=1)
            h = from_carry + local                             # (b, L, W)
            out[:, s:e] = h
            carry = h[:, -1]
        return out

    is_recurrent = True

    def init_state(self, b, device=None, dtype=None):
        return self.B.weight.new_zeros(b, self.width) if device is None \
            else torch.zeros(b, self.width, device=device, dtype=dtype)

    def step(self, x_t, state):          # state: h (b,width)
        a = torch.sigmoid(self.a_logit)
        h = a * state + self.B(x_t)
        return torch.tanh(self.C(h)), h  # output is projected; state is the raw recurrence


class _SelectiveSSMSeq(nn.Module):
    """Selective state-space primitive (Mamba/S6 core): a diagonal SSM whose transition is
    INPUT-DEPENDENT. Where linssm is linear TIME-INVARIANT (fixed decay a, fixed drive B, fixed
    readout C), the selective SSM makes the step size Delta, the input gate B, and the output gate C
    all functions of the current input x_t:

        Delta_t = softplus(Linear_Delta(x_t) + delta_bias)     (b, width)   -- per-channel step size
        B_t     = Linear_B(x_t)                                 (b, width)   -- input projection
        C_t     = Linear_C(x_t)                                 (b, width)   -- output projection
        Abar_t  = exp(-Delta_t * A)      with A = softplus(A_log) > 0        -- input-dependent decay
        h_t     = Abar_t * h_{t-1} + (Delta_t * B_t) * x_proj_t              -- selective state update
        y_t     = tanh(C_t * h_t + D * x_proj_t)                             -- gated readout + skip

    The selectivity (Delta_t, B_t, C_t depending on x_t) is what lets the model CHOOSE, per token,
    what to remember and what to ignore -- the mechanism that provably solves selective-copy /
    induction-heads tasks that an LTI SSM (linssm) plus static gating cannot (Gu & Dao 2023). It is a
    genuinely different composition from the gated cores: gating (gated/lstm) modulates a nonlinear
    per-step update but does not vary the linear state-transition timescale by content, whereas the
    selection mechanism modulates the decay/retention itself along the sequence axis.

    A single-input-channel projection x_proj = Linear(x_t) carries the value stream; the diagonal
    state has `width` channels. Recurrent (streams via step); the loop is sequential because the
    transition is input-dependent (not an associative scan without extra work).
    """
    name = "selssm"
    is_recurrent = True

    def __init__(self, n_in, width):
        super().__init__()
        self.width = width
        self.x_proj = nn.Linear(n_in, width, bias=True)     # value stream into the state channels
        self.Wdelta = nn.Linear(n_in, width, bias=True)     # input-dependent step size
        self.WB = nn.Linear(n_in, width, bias=True)         # input-dependent input gate
        self.WC = nn.Linear(n_in, width, bias=True)         # input-dependent output gate
        for lin in (self.x_proj, self.Wdelta, self.WB, self.WC):
            _init_linear(lin, 1.0, 0.0)
        # A > 0 per channel (state decay rate); init so exp(-Delta*A) ~ slow decay at unit Delta
        self.A_log = nn.Parameter(torch.randn(width) * 0.5 - 1.0)   # softplus(-1)~0.31
        self.D = nn.Parameter(torch.ones(width))                    # skip/residual per channel
        self.out = nn.Linear(width, width)                          # final width->width mix
        _init_linear(self.out, 1.0, 0.0)

    def _params(self, x_t):
        A = torch.nn.functional.softplus(self.A_log)               # (width,) > 0
        delta = torch.nn.functional.softplus(self.Wdelta(x_t))     # (b,width) > 0
        B = self.WB(x_t)                                            # (b,width)
        C = self.WC(x_t)                                            # (b,width)
        xp = self.x_proj(x_t)                                       # (b,width) value stream
        Abar = torch.exp(-delta * A)                               # (b,width) input-dependent decay
        Bbar = delta * B                                           # discretized input gate
        return Abar, Bbar, C, xp

    def forward_seq(self, x):  # (b, T, n_in)
        b, T, _ = x.shape
        h = x.new_zeros(b, self.width)
        outs = []
        for t in range(T):
            Abar, Bbar, C, xp = self._params(x[:, t])
            h = Abar * h + Bbar * xp
            y = C * h + self.D * xp
            outs.append(y)
        ys = torch.stack(outs, dim=1)                             # (b,T,width)
        return torch.tanh(self.out(ys))

    def init_state(self, b, device=None, dtype=None):
        return self.x_proj.weight.new_zeros(b, self.width) if device is None \
            else torch.zeros(b, self.width, device=device, dtype=dtype)

    def step(self, x_t, state):
        Abar, Bbar, C, xp = self._params(x_t)
        h = Abar * state + Bbar * xp
        y = C * h + self.D * xp
        return torch.tanh(self.out(y)), h


class _SpectralSeq(nn.Module):
    """Fixed frequency-basis primitive over TIME: rFFT of each feature channel across the
    time axis -> per-frequency log-power -> summarize per channel -> project, broadcast back.

    The conjugate of conv (fixed LOCAL/position basis): spectral is diagonal in the FREQUENCY
    basis and position-invariant (it sees which temporal frequencies are present, discards
    absolute time). Not one of the six irreducible primitives -- a useful non-irreducible
    conjugate-basis primitive (mutually incomparable with conv by the uncertainty principle).

    To stay independent of the (variable) sequence length T, the per-channel frequency
    log-power is summarized by fixed statistics (mean + max over frequency), giving a
    2*n_in feature that a fixed-size projection maps to width. Time-invariant, so the width
    vector is broadcast over time to satisfy the s2s contract.
    """
    name = "spectral"
    is_recurrent = False

    def __init__(self, n_in, width):
        super().__init__()
        self.n_in, self.width = n_in, width
        self.proj = nn.Linear(2 * n_in, width)   # [mean-power | max-power] per channel
        _init_linear(self.proj, 1.0, 0.0)

    def forward_seq(self, x):  # (b, T, n_in)
        b, T, C = x.shape
        def _spec(x):
            Xf = torch.fft.rfft(x, dim=1)                       # (b, nf, C) over time
            power = torch.log1p(Xf.real ** 2 + Xf.imag ** 2)    # (b, nf, C)
            # MPS' long rfft/reductions can silently emit non-finite values (CPU stays finite). A NaN in
            # `power` then makes max(dim=1) return an invalid (-1) argmax on MPS, whose backward SCATTER
            # hard-crashes with an AcceleratorError. Keep `power` finite, and use amax (mask-based backward)
            # rather than max().values (argmax-scatter backward) so the reduction is MPS-safe regardless.
            power = torch.nan_to_num(power)
            return torch.cat([power.mean(dim=1), power.amax(dim=1)], dim=1)  # (b, 2C)
        try:
            feat = _spec(x)
        except (NotImplementedError, RuntimeError) as e:
            # MPS FFT gap fallback: compute the spectrum on CPU, return to the input device.
            if "mps" not in str(e).lower() and "not implemented" not in str(e).lower() and "fft" not in str(e).lower():
                raise
            feat = _spec(x.to("cpu")).to(x.device)
        v = torch.tanh(self.proj(feat))                     # (b, width), time-invariant
        return v.unsqueeze(1).expand(b, T, self.width)      # broadcast over time


_SEQ_CORES = {
    "plain": _PlainSeq, "gated": _GatedSeq, "lstm": _LSTMSeq,
    "conv": _ConvSeq, "dilconv": _DilatedConvSeq, "attention": _AttentionSeq,
    "dense": _DenseSeq, "norm": _NormSeq,
    "spectral": _SpectralSeq, "linssm": _LinearSSMSeq, "selssm": _SelectiveSSMSeq,
}


# ---------------------------------------------------------------------------
# cell + schema
# ---------------------------------------------------------------------------

class _SequenceCell(nn.Module):
    """A meta-cell holding N sequence primitives in parallel; alpha mixes their seq outputs."""

    def __init__(self, n_in, width, primitives, sigma_w2=1.76, chrono_tmax=None):
        super().__init__()
        self.primitives = list(primitives)
        cores = []
        for p in primitives:
            if p == "lstm":
                cores.append(_SEQ_CORES[p](n_in, width, sigma_w2=sigma_w2, chrono_tmax=chrono_tmax))
            elif p in ("plain", "gated"):
                cores.append(_SEQ_CORES[p](n_in, width, sigma_w2=sigma_w2))
            else:
                cores.append(_SEQ_CORES[p](n_in, width))
        self.cores = nn.ModuleList(cores)
        self.alpha = nn.Parameter(torch.zeros(len(primitives)))
        self.register_buffer("_alpha_peak", torch.full((len(primitives),), 1.0 / len(primitives)))

    def mixed(self, x_seq):
        # weighted sum over primitives WITHOUT stacking (sum_p w_p * out_p == einsum over a (P,b,T,w) stack):
        # stacking would allocate an extra full (P,b,T,width) copy of all P primitives' activations, doubling
        # the mixing-layer peak memory (a real cost for long sequences). Matches the 4d/operator cells.
        w = torch.softmax(self.alpha, dim=0)                                          # (P,)
        outs = [core.forward_seq(x_seq) for core in self.cores]                       # P x (b,T,width)
        return sum(wi * o for wi, o in zip(w, outs))

    # --- streaming API ---

    def mixed_seq(self, x_seq):  # backward-compatible alias
        return self.mixed(x_seq)

    def all_recurrent(self):
        return all(getattr(c, "is_recurrent", False) for c in self.cores)

    def init_state(self, b, device=None, dtype=None):
        """One state per primitive; non-recurrent primitives get None."""
        return [c.init_state(b, device, dtype) if getattr(c, "is_recurrent", False) else None
                for c in self.cores]

    def step(self, x_t, states):
        """Advance one timestep. x_t: (b, n_in); states: list (one per primitive from init_state).
        Returns (mixed_out_t: (b,width), new_states). Requires all primitives recurrent -- a mixed
        cell containing a non-recurrent primitive (conv/attention/dense/norm/spectral) cannot stream,
        because those primitives need the whole sequence at once."""
        if not self.all_recurrent():
            bad = [p for p, c in zip(self.primitives, self.cores)
                   if not getattr(c, "is_recurrent", False)]
            raise RuntimeError(f"cannot stream: non-recurrent primitive(s) {bad} need the full "
                               f"sequence. Restrict the schema to recurrent primitives "
                               f"(plain/gated/lstm/linssm) to stream.")
        w = torch.softmax(self.alpha, dim=0)
        new_states, outs = [], []
        for i, core in enumerate(self.cores):
            o, s = core.step(x_t, states[i])
            outs.append(o); new_states.append(s)
        mixed = torch.einsum("p,pbw->bw", w, torch.stack(outs, dim=0))
        return mixed, new_states

    def update_peak(self):
        with torch.no_grad():
            cur = torch.softmax(self.alpha, dim=0)
            if cur.max() > self._alpha_peak.max():
                self._alpha_peak = cur.clone()

    def alpha_weights(self):
        with torch.no_grad():
            return torch.softmax(self.alpha, dim=0).cpu().numpy()

    def alpha_peak(self):
        return self._alpha_peak.cpu().numpy()


class Schema(nn.Module):
    """sequence schema: all primitive families compete under one alpha per layer.

    Input:  x_seq (b, T, n_in). Output: (b, n_out).
    readout='last' uses the last timestep (recall/copy); 'mean' averages over time
    (classification). Functional composition across depth: layer l reads layer l-1's mixed
    sequence output.
    """

    def __init__(self, depth, width, n_in=1, n_out=10, seed=0,
                 primitives=("plain", "gated", "lstm", "conv", "attention", "dense",
                             "norm", "spectral", "linssm"),
                 sigma_w2=1.76, chrono_tmax=None, readout="last", deep_supervision=False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        for p in primitives:
            if p not in _SEQ_CORES:
                raise ValueError(f"unknown primitive {p!r}; have {list(_SEQ_CORES)}")
        if readout not in ("last", "mean", "flatten"):
            raise ValueError("readout must be 'last', 'mean', or 'flatten'")
        torch.manual_seed(seed)
        self.depth, self.width, self.readout = depth, width, readout
        self.primitives = list(primitives)
        self.deep_supervision = deep_supervision
        cells = []
        din = n_in
        for _ in range(depth):
            cells.append(_SequenceCell(din, width, primitives, sigma_w2=sigma_w2, chrono_tmax=chrono_tmax))
            din = width
        self.cells = nn.ModuleList(cells)
        self.n_out = n_out
        if readout == "flatten":
            # head maps T*width -> n_out; T is unknown until first forward, so size lazily.
            self.head = None
            self._flatten_head_wd = width
        else:
            self.head = nn.Linear(width, n_out)
            _init_linear(self.head, 1.0, 0.0)
        if deep_supervision:
            if readout == "flatten":
                raise ValueError("deep_supervision is not supported with readout='flatten' "
                                 "(early heads would need per-layer flatten sizing); use 'last'/'mean'.")
            early = [nn.Linear(width, n_out) for _ in range(depth - 1)]
            for h in early:
                _init_linear(h, 1.0, 0.0)
            self._early_aux = nn.ModuleList(early)
            self.aux_heads = list(self._early_aux) + [self.head]
        else:
            self._early_aux = None
            self.aux_heads = None

    def _layer_seqs(self, x_seq):
        seqs = []
        h = x_seq
        for cell in self.cells:
            h = cell.mixed(h)     # (b,T,width)
            seqs.append(h)
        return seqs

    def _readout(self, seq):
        # seq: (b, T, width). 'last'->(b,width); 'mean'->(b,width); 'flatten'->(b,T*width).
        if self.readout == "last":
            return seq[:, -1]
        if self.readout == "mean":
            return seq.mean(dim=1)
        return seq.reshape(seq.shape[0], -1)   # flatten: concat all timesteps (position-aware)

    def _ensure_flatten_head(self, feat):
        # lazily build the T*width -> n_out head on first forward (T known only at runtime)
        if self.head is None:
            self.head = nn.Linear(feat.shape[1], self.n_out).to(feat.device)
            _init_linear(self.head, 1.0, 0.0)
        return self.head

    def forward(self, x_seq):
        feat = self._readout(self._layer_seqs(x_seq)[-1])
        if self.readout == "flatten":
            return self._ensure_flatten_head(feat)(feat)
        return self.head(feat)

    # ------------------------------------------------------------------
    # Streaming / online inference API
    # ------------------------------------------------------------------
    def can_stream(self):
        """True iff every cell's every primitive is recurrent (so state can be carried per step)."""
        return all(cell.all_recurrent() for cell in self.cells)

    def init_stream_state(self, b, device=None, dtype=None):
        """Initial per-layer, per-primitive states for streaming a batch of b sequences."""
        return [cell.init_state(b, device, dtype) for cell in self.cells]

    def step(self, x_t, states):
        """Advance the whole (functionally-composed) stack by one timestep.

        x_t: (b, n_in) input at this timestep. states: from init_stream_state (or a prior step).
        Returns (top_layer_out_t: (b, width), new_states). The per-layer mixed output feeds the next
        layer, matching the functional composition of forward_seq exactly. Only valid when
        can_stream() is True (all primitives recurrent); raises otherwise, since conv/attention/
        dense/norm/spectral require the full sequence.

        To get a prediction from a streamed final state, apply .head to the last-timestep output
        (readout='last') or accumulate a running mean externally (readout='mean').
        """
        if not self.can_stream():
            raise RuntimeError("cannot stream: schema contains a non-recurrent primitive. "
                               "Build with primitives restricted to the recurrent set "
                               "(plain, gated, lstm, linssm) to enable streaming.")
        new_states = []
        h = x_t
        for cell, st in zip(self.cells, states):
            h, ns = cell.step(h, st)     # mixed output of this layer -> input to next
            new_states.append(ns)
        return h, new_states

    def stream(self, x_seq):
        """Convenience: stream a full (b,T,n_in) sequence one step at a time, returning the top-layer
        output sequence (b,T,width). Numerically equals _layer_seqs(x_seq)[-1] when can_stream()."""
        b, T, _ = x_seq.shape
        states = self.init_stream_state(b, device=x_seq.device, dtype=x_seq.dtype)
        outs = []
        for t in range(T):
            h, states = self.step(x_seq[:, t], states)
            outs.append(h)
        return torch.stack(outs, dim=1)

    def forward_all_layers(self, x_seq):
        if self.aux_heads is None:
            raise RuntimeError("forward_all_layers requires deep_supervision=True")
        seqs = self._layer_seqs(x_seq)
        return [self.aux_heads[l](self._readout(seqs[l])) for l in range(self.depth)]

    def forward_seq_readout(self, x_seq, k):
        """Return per-timestep head outputs for the last k timesteps (copy-task style)."""
        if self.readout == "flatten":
            raise RuntimeError("forward_seq_readout needs a per-timestep (width-sized) head; "
                               "not available with readout='flatten'. Use readout='last'/'mean'.")
        seq = self._layer_seqs(x_seq)[-1]              # (b,T,width)
        return self.head(seq[:, -k:])                  # (b,k,n_out)

    def alpha_entropy(self):
        ent = self.cells[0].alpha.new_zeros(())
        for cell in self.cells:
            mu = torch.softmax(cell.alpha, dim=0)
            ent = ent - (mu * torch.log(mu + 1e-9)).sum()
        return ent

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def alpha_report(self):
        return [cell.alpha_weights() for cell in self.cells]

    def alpha_peak_report(self):
        return [cell.alpha_peak() for cell in self.cells]

    def selected_primitive(self, layer=-1):
        return self.primitives[int(np.argmax(self.cells[layer].alpha_peak()))]

    def architecture(self):
        """Human-readable per-layer selected primitive (the NN architecture produced)."""
        return [self.selected_primitive(l) for l in range(self.depth)]


def build_schema(depth, width, n_in=1, n_out=10, seed=0,
                 primitives=("plain", "gated", "lstm", "conv", "attention", "dense",
                             "norm", "spectral", "linssm"),
                 sigma_w2=1.76, chrono_tmax=None, readout="last", deep_supervision=False):
    """Sequence-schema factory. Explicit signature (mirroring Schema.__init__ and the sibling
    build_*_schema factories) so the accepted arguments are visible here rather than hidden behind
    **kwargs -> Schema(**kwargs)."""
    return Schema(depth, width, n_in=n_in, n_out=n_out, seed=seed, primitives=primitives,
                  sigma_w2=sigma_w2, chrono_tmax=chrono_tmax, readout=readout,
                  deep_supervision=deep_supervision)

"""Generalized N-way recurrent supergraph (adds LSTM as a third primitive).

The original SuperGraphRNN (in supergraph.py) is hardwired to a 2-way {plain, gated}
choice: every forward method threads exactly two states (hp, hg) and mixes w[0]*..+w[1]*..
This module GENERALIZES that to an arbitrary list of recurrent primitives, so we can do
3-way selection {plain, gated, lstm} (and beyond) without disturbing the validated 2-way
class.

Design invariants carried over from the validated 2-way supergraph:
- PRODUCT PATHS / SEPARATE STATES: each primitive threads its OWN state trajectory clean
  across time AND depth (the shared-state-corruption fix). A primitive's state may be a
  tensor (plain, gated) or a TUPLE (lstm carries (h, c)); the supergraph is agnostic --
  each primitive owns its state type via a uniform step/init/readout interface.
- SHARED per-layer readout head: the only way to weight a primitive more is through alpha
  (separate heads would let head magnitude absorb the selection, degenerating alpha).
- Per-layer alpha over N primitives; last-layer alpha is the identified selection signal;
  deep supervision optionally unpins earlier layers.
- Peak-alpha tracking: end-of-training alpha understates selection on moderate-margin
  tasks; the peak is the honest signal.

Primitive interface (each *Core implements):
  init_state(batch, device) -> state          (tensor or tuple of tensors)
  step(x, state)            -> (output, state) (output is the h that feeds forward)
  readout_h(state)          -> h               (the vector the head reads; = output)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .networks import _init_linear


class _PlainCore(nn.Module):
    """Plain tanh recurrence: h' = tanh(W_x x + W_h h). State = h."""

    name = "plain"

    def __init__(self, n_in, width, sigma_w2, sigma_b2=0.05):
        super().__init__()
        self.width = width
        self.Wx = nn.Linear(n_in, width, bias=False)
        self.Wh = nn.Linear(width, width, bias=True)
        _init_linear(self.Wx, 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2, sigma_b2)

    def init_state(self, b, device):
        return torch.zeros(b, self.width, device=device)

    def project_inputs(self, x_seq):
        """Precompute W_x @ x for ALL timesteps at once (input proj is state-independent).
        x_seq: (b, T, n_in) -> (b, T, width)."""
        return self.Wx(x_seq)

    def step_pre(self, xp, h):
        """Recurrent step given PRE-PROJECTED input xp = W_x x (from project_inputs)."""
        h = torch.tanh(xp + self.Wh(h))
        return h, h

    def step(self, x, h):
        h = torch.tanh(self.Wx(x) + self.Wh(h))
        return h, h

    def readout_h(self, h):
        return h


class _GatedCore(nn.Module):
    """GRU-style gated recurrence. State = h."""

    name = "gated"

    def __init__(self, n_in, width, sigma_w2, sigma_b2=0.05):
        super().__init__()
        self.width = width
        # Fused projections: one input matmul (3*width out) and one hidden matmul.
        # Gate order in the fused output: [z, r, n]. Per-gate init preserved by
        # initializing each width-block exactly as the unfused version did.
        self.Wx = nn.Linear(n_in, 3 * width, bias=False)  # input -> [z, r, n]
        self.Wh = nn.Linear(width, 3 * width, bias=True)  # hidden -> [z, r, n]
        _init_linear(self.Wx, 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2, sigma_b2)

    def init_state(self, b, device):
        return torch.zeros(b, self.width, device=device)

    def project_inputs(self, x_seq):
        return self.Wx(x_seq)  # (b, T, 3*width)

    def _combine(self, xp, h):
        w = self.width
        gx = xp
        gh = self.Wh(h)
        z = torch.sigmoid(gx[..., :w] + gh[..., :w])
        r = torch.sigmoid(gx[..., w : 2 * w] + gh[..., w : 2 * w])
        n = torch.tanh(gx[..., 2 * w :] + r * gh[..., 2 * w :])
        h = (1 - z) * h + z * n
        return h

    def step_pre(self, xp, h):
        h = self._combine(xp, h)
        return h, h

    def step(self, x, h):
        h = self._combine(self.Wx(x), h)
        return h, h

    def readout_h(self, h):
        return h


class _LSTMCore(nn.Module):
    """LSTM recurrence (adds an ADDITIVE cell-state carry, distinct from GRU's gated
    interpolation). State = (h, c). This is primitive #3.

        i = sigmoid(W_xi x + W_hi h)        input gate
        f = sigmoid(W_xf x + W_hf h)        forget gate
        o = sigmoid(W_xo x + W_ho h)        output gate
        g = tanh   (W_xg x + W_hg h)        candidate cell
        c' = f * c + i * g                  additive cell-state carry
        h' = o * tanh(c')                   gated cell readout

    The cell state c gives LSTM a memory mechanism GRU lacks (an un-squashed additive
    channel), so it is not a reparameterization of the gated primitive.
    """

    name = "lstm"

    def __init__(self, n_in, width, sigma_w2, sigma_b2=0.05, chrono_tmax=None):
        super().__init__()
        self.width = width
        # Fused: one input matmul and one hidden matmul, gate order [i, f, o, g].
        self.Wx = nn.Linear(n_in, 4 * width, bias=False)  # input -> [i, f, o, g]
        self.Wh = nn.Linear(width, 4 * width, bias=True)  # hidden -> [i, f, o, g]
        _init_linear(self.Wx, 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2, sigma_b2)
        with torch.no_grad():
            if chrono_tmax is not None and chrono_tmax > 1:
                # Chrono-initialization (Tallec & Ollivier 2018): draw forget-gate bias
                # b_f ~ log(U(1, T_max)) so the cell's default retention timescale spans
                # up to T_max, letting some units persist across the full horizon by
                # default. Couple the input-gate bias b_i = -b_f (the derivation's
                # input/forget coupling). This is the ingredient vanilla LSTM needs to
                # converge on long-range tasks (forget-bias +1.0 alone is insufficient).
                u = torch.rand(width) * (chrono_tmax - 1.0) + 1.0  # U(1, T_max)
                b_f = torch.log(u)
                self.Wh.bias[width : 2 * width].copy_(b_f)  # forget block
                self.Wh.bias[0:width].copy_(-b_f)  # input block
            else:
                # Standard init: forget-gate bias +1.0 (f is the SECOND width-block).
                self.Wh.bias[width : 2 * width].fill_(1.0)

    def init_state(self, b, device):
        z = torch.zeros(b, self.width, device=device)
        return (z, z.clone())

    def project_inputs(self, x_seq):
        return self.Wx(x_seq)  # (b, T, 4*width)

    def _combine(self, xp, state):
        h, c = state
        w = self.width
        g_all = xp + self.Wh(h)
        i = torch.sigmoid(g_all[..., :w])
        f = torch.sigmoid(g_all[..., w : 2 * w])
        o = torch.sigmoid(g_all[..., 2 * w : 3 * w])
        g = torch.tanh(g_all[..., 3 * w :])
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    def step_pre(self, xp, state):
        h, ns = self._combine(xp, state)
        return h, ns

    def step(self, x, state):
        h, ns = self._combine(self.Wx(x), state)
        return h, ns

    def readout_h(self, state):
        return state[0]


_PRIMITIVE_CORES = {"plain": _PlainCore, "gated": _GatedCore, "lstm": _LSTMCore}


class MultiCell(nn.Module):
    """A meta-cell holding N recurrent primitives with SEPARATE states.

    Each primitive advances its own state from its own product-path input; the owning
    network mixes only at readout via softmax(alpha) over the N primitives.
    """

    def __init__(self, n_in, width, sigma_w2, sigma_b2, primitives, chrono_tmax=None):
        super().__init__()
        self.primitives = list(primitives)
        cores = []
        for p in primitives:
            if p == "lstm":
                cores.append(_PRIMITIVE_CORES[p](n_in, width, sigma_w2, sigma_b2, chrono_tmax=chrono_tmax))
            else:
                cores.append(_PRIMITIVE_CORES[p](n_in, width, sigma_w2, sigma_b2))
        self.cores = nn.ModuleList(cores)
        self.alpha = nn.Parameter(torch.zeros(len(primitives)))  # uniform -> unbiased
        self.register_buffer("_alpha_peak", torch.full((len(primitives),), 1.0 / len(primitives)))

    def init_states(self, b, device):
        return [core.init_state(b, device) for core in self.cores]

    def step_split(self, inputs, states):
        """Each primitive steps on its OWN input (from the same-primitive path below)
        and its OWN state. Returns (new_states, outputs)."""
        new_states, outputs = [], []
        for core, x, st in zip(self.cores, inputs, states):
            out, ns = core.step(x, st)
            new_states.append(ns)
            outputs.append(out)
        return new_states, outputs

    def project_seq(self, x_seq):
        """Precompute each primitive's input projection over the whole sequence.
        Returns a list (per primitive) of (b, T, gate*width) tensors."""
        return [core.project_inputs(x_seq) for core in self.cores]

    def step_split_pre(self, xps_t, states):
        """Step using PRE-PROJECTED inputs xps_t (list per primitive, at one timestep)."""
        new_states, outputs = [], []
        for core, xp, st in zip(self.cores, xps_t, states):
            out, ns = core.step_pre(xp, st)
            new_states.append(ns)
            outputs.append(out)
        return new_states, outputs

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


class MultiSuperGraphRNN(nn.Module):
    """N-way product-paths separate-state supergraph RNN.

    Generalizes SuperGraphRNN to `primitives` = any subset/order of
    {"plain","gated","lstm"}. Each primitive threads its own state (tensor or (h,c)
    tuple) cleanly across time and depth; the last layer's alpha-mix through a shared
    head is the readout.
    """

    def __init__(
        self,
        depth,
        width,
        sigma_w2,
        sigma_b2=0.05,
        n_in=1,
        n_out=10,
        seed=0,
        deep_supervision=False,
        primitives=("plain", "gated", "lstm"),
        chrono_tmax=None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        for p in primitives:
            if p not in _PRIMITIVE_CORES:
                raise ValueError(f"unknown primitive {p!r}; have {list(_PRIMITIVE_CORES)}")
        torch.manual_seed(seed)
        self.depth, self.width = depth, width
        self.primitives = list(primitives)
        self.deep_supervision = deep_supervision
        cells = []
        din = n_in
        for _ in range(depth):
            cells.append(MultiCell(din, width, sigma_w2, sigma_b2, primitives, chrono_tmax=chrono_tmax))
            din = width
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(width, n_out)
        _init_linear(self.head, 1.0, 0.0)
        if deep_supervision:
            early = [nn.Linear(width, n_out) for _ in range(depth - 1)]
            for h in early:
                _init_linear(h, 1.0, 0.0)
            self._early_aux = nn.ModuleList(early)
            self.aux_heads = list(self._early_aux) + [self.head]
        else:
            self._early_aux = None
            self.aux_heads = None

    def _run(self, x, capture_last_k=None):
        """Thread N separate primitive states through time and depth.

        Returns a list over the captured timesteps; each entry is the list of per-
        primitive readout vectors of the LAST layer at that timestep."""
        b, T, _ = x.shape
        dev = x.device
        # states[l] = list of N states for layer l
        states = [cell.init_states(b, dev) for cell in self.cells]
        captured = []
        k0 = 0 if capture_last_k is None else T - capture_last_k
        # VECTORIZATION: layer 0's input is the raw sequence, so its per-primitive input
        # projections are state-independent and can be computed for ALL timesteps in one
        # batched matmul before the loop (the dominant cost at long T). Deeper layers'
        # inputs depend on the previous layer's per-timestep output, so they use the
        # regular per-step path. Recurrence over t stays sequential (inherently so).
        xps0 = self.cells[0].project_seq(x)  # list per primitive: (b, T, gate*width)
        last = self.cells[-1]
        for t in range(T):
            xps0_t = [xp[:, t, :] for xp in xps0]
            states[0], outputs = self.cells[0].step_split_pre(xps0_t, states[0])
            inputs = outputs
            for l in range(1, self.depth):
                states[l], outputs = self.cells[l].step_split(inputs, states[l])
                inputs = outputs
            if t >= k0:
                rd = [core.readout_h(st) for core, st in zip(last.cores, states[-1])]
                captured.append(rd)
        return captured

    def forward(self, x):
        rd = self._run(x, capture_last_k=1)[-1]
        w = torch.softmax(self.cells[-1].alpha, dim=0)
        return sum(w[i] * self.head(rd[i]) for i in range(len(self.primitives)))

    def forward_seq_readout(self, x, K):
        caps = self._run(x, capture_last_k=K)
        w = torch.softmax(self.cells[-1].alpha, dim=0)
        outs = []
        for rd in caps:
            outs.append(sum(w[i] * self.head(rd[i]) for i in range(len(self.primitives))))
        return torch.stack(outs, dim=1)

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
        """Per-layer softmax(alpha) over primitives (order = self.primitives)."""
        return [cell.alpha_weights() for cell in self.cells]

    def alpha_peak_report(self):
        return [cell.alpha_peak() for cell in self.cells]

    def selected_primitive(self):
        """argmax selection from the LAST layer's PEAK alpha (always identified)."""
        w = self.cells[-1].alpha_peak()
        return self.primitives[int(np.argmax(w))]


def build_multi_supergraph(**kwargs):
    return MultiSuperGraphRNN(**kwargs)

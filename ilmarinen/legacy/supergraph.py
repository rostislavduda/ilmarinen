"""Minimal 2-primitive supergraph (Step 3).

Two recurrent primitives in parallel, mixed by a learned softmax architecture
weight alpha, trained jointly with the cell weights:

  primitive 0: PLAIN recurrence   h' = tanh(W_x x + W_h h)          (no gating)
  primitive 1: GATED recurrence   GRU-style cell with update/reset gates
                                   (adds primitive #3: multiplicative interaction)

  meta-cell output:  h' = softmax(alpha)_0 * h'_plain + softmax(alpha)_1 * h'_gated

alpha is a single 2-vector shared across timesteps and layers, initialized
uniform (0,0) -> softmax 0.5/0.5, so selection is driven ENTIRELY by which
primitive reduces the loss -- no init bias toward either.

Ground-truth test: on sequential Fashion-MNIST (T=784) the plain cell caps
~0.30; gating is known to reach ~0.90. A correct supergraph should drive
softmax(alpha) toward the gated primitive and break the plain ceiling. If it
selects plain or fails to improve, the supergraph is broken -- and the
quantified baseline tells us which.

Design notes (fairness of the test):
  - both primitives share the SAME hidden state h (a common mixed state), so
    the mix is meaningful timestep-to-timestep rather than two independent RNNs
    read out at the end.
  - alpha init is exactly uniform; neither primitive is favored.
  - both cells use criticality-aware init (sigma_w2) so neither is handicapped
    by bad initialization.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .networks import _init_linear


class _PlainCellCore(nn.Module):
    """Plain tanh recurrence: h' = tanh(W_x x + W_h h)."""

    def __init__(self, n_in, width, sigma_w2, sigma_b2=0.05):
        super().__init__()
        self.Wx = nn.Linear(n_in, width, bias=False)
        self.Wh = nn.Linear(width, width, bias=True)
        _init_linear(self.Wx, 1.0, 0.0)
        _init_linear(self.Wh, sigma_w2, sigma_b2)

    def forward(self, x, h):
        return torch.tanh(self.Wx(x) + self.Wh(h))


class _GatedCellCore(nn.Module):
    """GRU-style gated recurrence (adds multiplicative gating, primitive #3).

        z = sigmoid(W_xz x + W_hz h)          update gate
        r = sigmoid(W_xr x + W_hr h)          reset gate
        n = tanh(W_xn x + r * (W_hn h))       candidate  (multiplicative: r * ...)
        h' = (1 - z) * h + z * n              (multiplicative: z * n, (1-z)*h)
    """

    def __init__(self, n_in, width, sigma_w2, sigma_b2=0.05):
        super().__init__()
        self.Wxz = nn.Linear(n_in, width, bias=False)
        self.Whz = nn.Linear(width, width, bias=True)
        self.Wxr = nn.Linear(n_in, width, bias=False)
        self.Whr = nn.Linear(width, width, bias=True)
        self.Wxn = nn.Linear(n_in, width, bias=False)
        self.Whn = nn.Linear(width, width, bias=True)
        for lin_x in (self.Wxz, self.Wxr, self.Wxn):
            _init_linear(lin_x, 1.0, 0.0)
        for lin_h in (self.Whz, self.Whr, self.Whn):
            _init_linear(lin_h, sigma_w2, sigma_b2)

    def forward(self, x, h):
        z = torch.sigmoid(self.Wxz(x) + self.Whz(h))
        r = torch.sigmoid(self.Wxr(x) + self.Whr(h))
        n = torch.tanh(self.Wxn(x) + r * self.Whn(h))
        return (1 - z) * h + z * n


class SuperCell(nn.Module):
    """A meta-cell holding plain and gated primitives with SEPARATE states.

    CORRECTED DESIGN (v0.3): recurrent primitives must NOT share a mixed hidden
    state. Sharing a mixed state prevents each cell from learning its own
    recurrent dynamics -- a memory-preserving cell (gating) needs to control its
    OWN state trajectory across time to learn preservation, but a shared mixed
    state corrupts that trajectory every step. (Confirmed empirically: shared-
    state supergraph stuck at chance 0.167 on the copy task; separate-state
    recovered 0.979.)

    So this cell advances plain and gated on SEPARATE states and returns BOTH
    next-states; the owning network threads each state through time independently
    and mixes only at readout (softmax(alpha) over the two heads' logits).

    This differs from the STATELESS case (conv|dense on CIFAR), where sharing the
    input and mixing outputs is fine because there is no persistent state to
    corrupt. The mixing rule depends on whether the primitive is stateful.
    """

    def __init__(self, n_in, width, sigma_w2, sigma_b2=0.05):
        super().__init__()
        self.plain = _PlainCellCore(n_in, width, sigma_w2, sigma_b2)
        self.gated = _GatedCellCore(n_in, width, sigma_w2, sigma_b2)
        self.alpha = nn.Parameter(torch.zeros(2))  # uniform -> 0.5/0.5, unbiased
        # peak-alpha tracking: on MODERATE-margin tasks the gated advantage
        # peaks mid-training then decays as the weaker primitive catches up
        # (empirically: seq-MNIST alpha_gated rises to ~0.89 then falls to ~0.56
        # while accuracy keeps improving). End-of-training alpha UNDERSTATES the
        # selection; the peak is the meaningful signal. This buffer records it.
        self.register_buffer("_alpha_peak", torch.tensor([0.5, 0.5]))

    def step(self, x, h_plain, h_gated):
        """Advance both primitives on their OWN states from a SHARED input x.
        (Used at the first layer, where both primitives see the sequence input.)"""
        return self.plain(x, h_plain), self.gated(x, h_gated)

    def step_split(self, x_plain, x_gated, h_plain, h_gated):
        """Product-paths step: each primitive receives its OWN-primitive input
        from the layer below (plain<-plain, gated<-gated), keeping each path's
        state trajectory clean across depth as well as time."""
        return self.plain(x_plain, h_plain), self.gated(x_gated, h_gated)

    def update_peak(self):
        """Call once per epoch (or step) to track the max-selection alpha.

        Tracks the alpha whose max component is largest -- i.e. the most
        DECISIVE selection seen so far, regardless of which primitive it favors.
        """
        with torch.no_grad():
            cur = torch.softmax(self.alpha, dim=0)
            if cur.max() > self._alpha_peak.max():
                self._alpha_peak = cur.clone()

    def alpha_weights(self):
        with torch.no_grad():
            return torch.softmax(self.alpha, dim=0).cpu().numpy()

    def alpha_peak(self):
        return self._alpha_peak.cpu().numpy()


class SuperGraphRNN(nn.Module):
    """Product-paths separate-state supergraph RNN (v0.3, depth>=1).

    Holds `depth` SuperCells. Each PRIMITIVE keeps its own state trajectory clean
    across BOTH time and depth: the plain stack feeds plain->plain between layers,
    the gated stack feeds gated->gated. Only the FINAL-layer readout mixes the two
    primitives' heads by softmax(alpha). This is the validated 'product-paths'
    design (depth-2 copy task -> 0.999); it avoids the shared-state corruption that
    stalled the naive mixed-state design at chance.

    Per-layer alpha exists for interface symmetry, but note the LAYER-
    IDENTIFIABILITY CAVEAT: only layers that influence the readout get a selection
    signal. With readout from the last layer, earlier-layer alpha may stay pinned
    at 0.5 (no gradient to select) unless the task genuinely needs those layers'
    primitive choices. Selection/discretization therefore uses the LAST layer's
    alpha, which is always identified.

    Peak-alpha: each cell tracks its most-decisive alpha via update_peak(); on
    moderate-margin tasks the end-of-training alpha understates selection, so
    alpha_peak_report() is the honest selection signal.
    """

    def __init__(self, depth, width, sigma_w2, sigma_b2=0.05,
                 n_in=1, n_out=10, seed=0, deep_supervision=False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        torch.manual_seed(seed)
        self.depth = depth
        self.width = width
        self.deep_supervision = deep_supervision
        cells = []
        din = n_in
        for _ in range(depth):
            cells.append(SuperCell(din, width, sigma_w2, sigma_b2))
            din = width
        self.cells = nn.ModuleList(cells)
        # DEEP SUPERVISION (coupled-lattice Route 1): give each layer its OWN
        # bare field h_l>0 by attaching a per-layer readout head. Without this,
        # an earlier layer with no direct readout path has its field SCREENED by
        # the layers above (measured: raw h0=+0.456 collapses to conditional
        # +0.028 once the last layer locks its primitive), so alpha_l pins at 0.5
        # (the disordered phase of the depth-Ising model). An unscreened per-layer
        # field unpins it -- validated: first-layer alpha_gated 0.500 -> 0.982 as
        # the aux weight (bare-field magnitude) rises. Each aux head is SHARED
        # across the two primitives of its layer (same alpha-faithfulness reason
        # as the main head). Default OFF: only the last-layer alpha is identified
        # without it, matching the documented layer-identifiability caveat.
        if deep_supervision:
            # The LAST layer's aux head IS the shared readout head (self.head,
            # created just below), so the main readout path (forward /
            # forward_seq_readout) and the deep-supervision path share one trained
            # final head -- no eval-head mismatch. Only the EARLIER layers get
            # fresh aux heads to supply their bare fields.
            aux = [nn.Linear(width, n_out) for _ in range(depth - 1)]
            for h in aux:
                _init_linear(h, 1.0, 0.0)
            self._early_aux = nn.ModuleList(aux)  # depth-1 heads for layers 0..depth-2
        else:
            self._early_aux = None
        self.aux_heads = None  # assembled after self.head exists (see below)
        # SHARED readout head applied to each primitive's final state, then
        # alpha-mixed at the logits. A shared head is ESSENTIAL: separate per-
        # primitive heads let head magnitude absorb the selection (the gated head
        # grows to dominate the logits while alpha_gated stays < 0.5), making
        # alpha a DEGENERATE selection signal -- empirically alpha_gated=0.475
        # despite acc=0.991 on the copy task. With a shared head the only way to
        # weight a primitive more is through alpha itself, so alpha faithfully
        # tracks the selection (alpha_gated=0.943, acc=0.989).
        self.head = nn.Linear(width, n_out)
        _init_linear(self.head, 1.0, 0.0)
        # assemble the per-layer readout list used by deep supervision: early
        # layers use their fresh aux heads, the LAST layer reuses self.head so
        # the deep-sup readout and the main readout are the same trained head.
        if self.deep_supervision:
            self.aux_heads = list(self._early_aux) + [self.head]

    def _run(self, x):
        """Thread separate plain/gated states through time AND depth.
        Returns final-layer (h_plain, h_gated) per timestep as two lists."""
        b, T, _ = x.shape
        hp = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        hg = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        finals = []
        for t in range(T):
            inp_p = x[:, t, :]
            inp_g = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hp[l], hg[l] = cell.step_split(inp_p, inp_g, hp[l], hg[l])
                inp_p = hp[l]
                inp_g = hg[l]
            finals.append((hp[-1], hg[-1]))
        return finals

    def forward(self, x):
        finals = self._run(x)
        h_plain, h_gated = finals[-1]
        w = torch.softmax(self.cells[-1].alpha, dim=0)
        return w[0] * self.head(h_plain) + w[1] * self.head(h_gated)

    def forward_seq_readout(self, x, K):
        """Emit K predictions over the last K steps (copy-style tasks)."""
        finals = self._run(x)
        w = torch.softmax(self.cells[-1].alpha, dim=0)
        outs = []
        for (h_plain, h_gated) in finals[-K:]:
            outs.append(w[0] * self.head(h_plain) + w[1] * self.head(h_gated))
        return torch.stack(outs, dim=1)

    def alpha_entropy(self):
        """Sum of per-layer softmax(alpha) entropies.

        ENTROPY REGULARIZATION (makes the Gibbs picture hold): the derived
        Scenario-2 theory says alpha* is the Gibbs measure softmax(-beta*Delta) --
        but DEFAULT soft-DARTS does NOT equilibrate to it. It arrests at a
        gradient-starvation fixed point (the plain path's gradient vanishes once
        gated dominates the mixed logit), so no well-defined selection temperature
        exists (three equilibrium estimators of beta all failed / sign-flipped).

        Adding an explicit -(1/beta)*entropy term to the alpha objective makes the
        stationary alpha the actual Gibbs measure at temperature 1/beta, so beta
        becomes a controllable selection temperature (validated: s* monotonic in
        beta_set, 0.04->0.26 as beta 0.5->4). The training loop should add
        `-coef * net.alpha_entropy()` to the alpha loss, with coef = 1/beta.

        Returns a scalar tensor (grad flows to alpha). Sums over ALL layers so it
        composes with deep supervision (each layer's alpha then equilibrates)."""
        ent = self.cells[0].alpha.new_zeros(())
        for cell in self.cells:
            mu = torch.softmax(cell.alpha, dim=0)
            ent = ent - (mu * torch.log(mu + 1e-9)).sum()
        return ent

    def forward_all_layers(self, x):
        """Per-layer alpha-mixed readout logits (requires deep_supervision=True).

        Returns a list of `depth` logit tensors, one per layer, each mixed by that
        layer's own alpha via its own aux head. The main-loss target is applied to
        the LAST layer's logits (== forward()); the aux losses on earlier layers
        supply their bare fields. This is the mechanism that unpins earlier-layer
        alpha (0.500 -> 0.982 validated)."""
        if self.aux_heads is None:
            raise RuntimeError("forward_all_layers requires deep_supervision=True")
        b, T, _ = x.shape
        hp = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        hg = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        # thread states to the final timestep, capturing each layer's final state
        for t in range(T):
            inp_p = x[:, t, :]
            inp_g = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hp[l], hg[l] = cell.step_split(inp_p, inp_g, hp[l], hg[l])
                inp_p = hp[l]
                inp_g = hg[l]
        outs = []
        for l in range(self.depth):
            w = torch.softmax(self.cells[l].alpha, dim=0)
            outs.append(w[0] * self.aux_heads[l](hp[l]) + w[1] * self.aux_heads[l](hg[l]))
        return outs

    def forward_all_layers_seq(self, x, K):
        """Deep-supervision analogue of forward_seq_readout: per-layer readout over
        the last K steps. Returns a list of `depth` tensors of shape (b, K, n_out)."""
        if self.aux_heads is None:
            raise RuntimeError("forward_all_layers_seq requires deep_supervision=True")
        b, T, _ = x.shape
        hp = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        hg = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        per_layer_steps = [[] for _ in range(self.depth)]
        for t in range(T):
            inp_p = x[:, t, :]
            inp_g = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hp[l], hg[l] = cell.step_split(inp_p, inp_g, hp[l], hg[l])
                inp_p = hp[l]
                inp_g = hg[l]
            if t >= T - K:
                for l in range(self.depth):
                    per_layer_steps[l].append((hp[l], hg[l]))
        outs = []
        for l in range(self.depth):
            w = torch.softmax(self.cells[l].alpha, dim=0)
            seq = [w[0] * self.aux_heads[l](a) + w[1] * self.aux_heads[l](g)
                   for (a, g) in per_layer_steps[l]]
            outs.append(torch.stack(seq, dim=1))
        return outs

    def update_peak(self):
        for cell in self.cells:
            cell.update_peak()

    def first_last_weight(self):
        return self.cells[0].plain.Wh.weight, self.head.weight

    def alpha_report(self):
        """Per-layer softmax(alpha) = [plain_weight, gated_weight]."""
        return [cell.alpha_weights() for cell in self.cells]

    def alpha_peak_report(self):
        """Per-layer PEAK alpha -- the honest selection signal on moderate-margin
        tasks where end-of-training alpha decays after the mid-training peak."""
        return [cell.alpha_peak() for cell in self.cells]

    def selected_primitive(self):
        """argmax selection from the LAST layer's PEAK alpha (always identified)."""
        w = self.cells[-1].alpha_peak()
        return "gated" if w[1] >= w[0] else "plain"


class DiscreteRNN(nn.Module):
    """A discretized network: one chosen primitive per layer (no mixture).

    Built from a trained SuperGraphRNN by taking the alpha-argmax primitive at
    each layer and INHERITING that primitive's trained weights (warm start).
    This is the mandatory discretization the analytical picture prescribes: soft
    architecture search must terminate by dropping the loser and keeping the
    winner. Fine-tuning from the inherited weights then recovers the pure-
    primitive performance that the soft mixture diluted.
    """

    def __init__(self, super_net: "SuperGraphRNN"):
        super().__init__()
        self.width = super_net.width
        self.depth = super_net.depth
        self.choice = super_net.selected_primitive()   # last-layer peak-alpha argmax
        # inherit the winning primitive's FULL stack (product-paths: the chosen
        # primitive already ran as a clean stack across all layers)
        cells = []
        for cell in super_net.cells:
            cells.append(cell.gated if self.choice == "gated" else cell.plain)
        self.cells = nn.ModuleList(cells)
        self.head = super_net.head    # shared head (inherited)

    def forward(self, x):
        b, T, _ = x.shape
        hs = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hs[l] = cell(inp, hs[l])
                inp = hs[l]
        return self.head(hs[-1])

    def forward_seq_readout(self, x, K):
        b, T, _ = x.shape
        hs = [x.new_zeros(b, self.width) for _ in range(self.depth)]
        outs = []
        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hs[l] = cell(inp, hs[l])
                inp = hs[l]
            if t >= T - K:
                outs.append(self.head(hs[-1]))
        return torch.stack(outs, dim=1)

    def first_last_weight(self):
        c0 = self.cells[0]
        w0 = c0.Wh.weight if hasattr(c0, "Wh") else c0.Whz.weight
        return w0, self.head.weight


def discretize(super_net: "SuperGraphRNN") -> DiscreteRNN:
    """Extract the discrete architecture from a trained supergraph."""
    return DiscreteRNN(super_net)


SUPERGRAPH_REGISTRY = {"supergraph_rnn": SuperGraphRNN}


def build_supergraph(kind="supergraph_rnn", **kwargs):
    if kind not in SUPERGRAPH_REGISTRY:
        raise ValueError(f"unknown supergraph {kind!r}")
    return SUPERGRAPH_REGISTRY[kind](**kwargs)

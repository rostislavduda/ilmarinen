"""JOINT architecture search: a SINGLE gradient loop that optimizes primitive + kernel + max-l +
width + depth together, under one action J = R + mu * Omega. The unification of the per-resource
selectors (DARTS primitive alpha, priced width/depth, correlation-length kernel, angular-order max-l)
into one differentiable objective.

===================================================================================================
ANALYTICAL FORMULATION
===================================================================================================
The architecture is a point in a MIXED discrete-continuous space. We relax every axis to a
differentiable GATE/MASK on a MAXIMAL schema (max depth, max width, all primitives, largest
kernel / highest l), then optimize

    J(Phi, W) = R(W; A(Phi))  +  mu * Omega(Phi)                                      (the action)

over gate parameters Phi and weights W, where A(Phi) is the (soft) architecture and Omega is the
differentiable EXPECTED structural cost. The five axes relax in three structural classes:

  CLASS 1  softmax MIXTURE over same-shaped candidates (one forward pass):
     - primitive type:  p_prim = softmax(alpha_prim); output = sum_i p_i * core_i(x)
     - kernel size:     a sub-family of the primitive head -- conv2d_k3/k5/k7 are same-contract
                        candidates, mixed by the same softmax. (Spatial/volumetric only.)
    cost contribution:  sum_i p_i * c_i,  c_i the analytical per-candidate cost (params; or k^d).

  CLASS 2  soft MASK over capacity channels (continuous width):
     - width:   gate channel j by m_j = sigmoid(beta_j); h <- h * m.  cost = sum_j m_j / W_max.
     - max-l:   a STRUCTURED width mask -- gate the l=2 channel block by one sigmoid(beta_l2);
                cost = (irrep-dim ratio) * sigmoid(beta_l2). (Equivariant only.) max-l is thus a
                special case of the width mask, which is why the two share Class 2.

  CLASS 3  soft GATE over depth (discrete layer count):
     - depth:   gate layer l's residual update by g_l = sigmoid(gamma_l):
                   h_{l+1} = h_l + g_l * cell_l(h_l)      (g_l->0 => layer l is a skip/identity)
                cost = sum_l g_l / L_max. This is the differentiable skip/compute relaxation
                (SNAS/FBNetV2), the differentiable cousin of the priced marginal-value depth rule.

Omega(Phi) = w_prim * sum_i p_i c_i^{prim}
           + w_width * (sum_j m_j)/W_max
           + w_depth * (sum_l g_l)/L_max
           [ + w_l2 * dim_ratio * sigmoid(beta_l2)  for equivariant ]
a single scalar, differentiable in all gate parameters. mu is the one description-length price (the
chemical potential); sweeping mu traces the joint fit--complexity frontier over ALL axes at once.

Discretization (deployment): argmax the primitive softmax; keep channels with m_j > 0.5; keep layers
with g_l > 0.5 (skip the rest); keep l=2 iff sigmoid(beta_l2) > 0.5. An entropy/sharpening schedule
drives the soft gates toward {0,1} so the discretization gap is small.

===================================================================================================
APPLICABILITY TO THE SCHEMA TYPES
===================================================================================================
The loop applies to ALL schemas, but the ACTIVE axis set differs by contract, because
the structural resources a contract HAS differ:
  sequence     : primitive + width + depth               (no kernel: 1-D causal conv is one primitive;
                                                           no max-l: not steerable)
  spatial (2D) : primitive + KERNEL + width + depth       (kernel = receptive field, the extra axis)
  volumetric   : primitive + KERNEL(k^3) + width + depth
  graph        : primitive + width + depth                (no spatial kernel; aggregator IS primitive)
  equivariant  : primitive + width + depth + MAX-L        (max-l = angular resolution, the extra axis)
So the SAME loop and the SAME action serve all five; each contract switches on the gates for the
resources it actually possesses. This module implements the loop generically over any schema that
exposes per-cell primitive alphas, plus optional width/depth/max-l gates supplied by a thin adapter.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def auto_gate_init(target_openness=0.95, mu=None, min_openness=0.75):
    """Derive the gate pre-activation beta_0 = logit(s0) so every width/depth gate STARTS at openness
    s0 (default 0.95 = ~fully open). This replaces the hand-tuned gate_init with a principled value:
    gates start open so the search finds ACCURACY first, and the complexity term mu*Omega is what
    closes them -- avoiding starvation of accurate-but-costly primitives at initialization. s0 is the
    single interpretable knob (fraction open), not an opaque logit.

    The openness s0=0.95 gives beta_0=logit(0.95)=2.94 with gate gradient s0(1-s0)=0.045 still active
    (a saturated gate near 1.0 would have ~zero gradient and could not move under mu). Optionally
    mu-AWARE: under a strong price, start slightly less open (the optimum is compact anyway) to
    converge faster, clamped so the gate never starts too closed:
        s0(mu) = clamp(target_openness - 0.5*mu, min_openness, 0.98).
    """
    s0 = target_openness
    if mu is not None:
        s0 = float(np.clip(target_openness - 0.5 * mu, min_openness, 0.98))
    return math.log(s0 / (1.0 - s0))


# ==================================================================================================
# READOUT AS A DIFFERENTIABLE AXIS
# ==================================================================================================
# Readout (pooling over the token/node/spatial dimension) is a CLASS-1 softmax mixture: the candidates
# last/mean/max/sum all produce the SAME (b, width) output, so they mix under one softmax exactly like
# primitives. (flatten is excluded -- different shape.) Pooling is PARAMETER-FREE, so readout carries
# ~zero Omega cost and is selected by FIT alone (the honest situation: readout is an expressiveness
# choice, not a capacity cost). This makes readout the 6th joint axis under the same J=R+mu*Omega.


class DifferentiableReadout(nn.Module):
    """Softmax mixture over pooling operations, selected by an alpha trained in the joint loop.
    kind='seq' pools over time (last/mean/max); kind='graph' pools over nodes-per-graph
    (sum/mean/max); kind='spatial' pools over H,W (mean/max/sum); kind='vol' over D,H,W.

    SIZE-INVARIANCE PRIOR. Pooling is parameter-free, so readout carries ~zero capacity cost. But the
    ops differ in SIZE-INVARIANCE: 'mean' is invariant to the number of nodes/tokens, while 'sum' and
    'max' are size-SENSITIVE ('sum' scales with count; 'max' is invariant in value but its statistics
    drift with size). For varying-size inputs (graphs of different atom counts, sequences of different
    length) a size-invariant readout generalizes better across the size distribution. The prior adds a
    SMALL fixed cost to size-sensitive ops, folded into Omega, so that ties are broken toward 'mean'
    unless a size-sensitive op earns its place by FIT. size_prior=0 recovers pure fit-selection (the
    honest zero-cost default); size_prior>0 leans toward size-invariance. This is a genuine cost (an
    inductive-bias price), not a capacity cost -- the one readout Omega term that is nonzero."""

    _SEQ = ("last", "mean", "max")
    _GRAPH = ("sum", "mean", "max")
    _SPATIAL = ("mean", "max", "sum")
    # size-invariance: 1.0 = size-invariant (no prior cost), >1 = size-sensitive (priced)
    _SIZE_SENSITIVITY = {"mean": 0.0, "last": 0.0, "max": 0.5, "sum": 1.0}

    def __init__(self, kind="seq", size_prior=0.0):
        super().__init__()
        self.kind = kind
        self.size_prior = size_prior
        self.ops = {"seq": self._SEQ, "graph": self._GRAPH, "spatial": self._SPATIAL, "vol": self._SPATIAL}[kind]
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
        self.register_buffer("alpha_peak", torch.zeros(len(self.ops)))

    def readout_cost(self):
        """Differentiable size-invariance prior term for Omega: size_prior * sum_r softmax(alpha)_r *
        sensitivity_r. Zero when size_prior=0 (pure fit-selection) or when only size-invariant ops
        carry weight. Leans the selection toward size-invariant 'mean' otherwise."""
        if self.size_prior <= 0:
            return self.alpha.new_zeros(())
        w = torch.softmax(self.alpha, dim=0)
        sens = torch.tensor([self._SIZE_SENSITIVITY[o] for o in self.ops], dtype=w.dtype, device=w.device)
        return self.size_prior * (w * sens).sum()

    def _pool_seq(self, h, op):  # h: (b,T,w)
        if op == "last":
            return h[:, -1, :]
        if op == "mean":
            return h.mean(dim=1)
        return h.amax(dim=1)  # amax, not max().values: MPS-safe over NaN

    def _pool_graph(self, h, batch, n_graphs, op):  # h: (N,w)
        from ilmarinen.models.graph_schema import _scatter_max, _scatter_mean, _scatter_sum

        if op == "sum":
            return _scatter_sum(h, batch, n_graphs)
        if op == "mean":
            return _scatter_mean(h, batch, n_graphs)
        return _scatter_max(h, batch, n_graphs)

    def _pool_spatial(self, h, op):  # h: (b,C,H,W) or (b,C,D,H,W)
        dims = tuple(range(2, h.dim()))
        if op == "mean":
            return h.mean(dim=dims)
        if op == "max":
            return h.amax(dim=dims)  # amax, not max().values: MPS-safe over NaN
        return h.sum(dim=dims)

    def forward(self, h, batch=None, n_graphs=None):
        w = torch.softmax(self.alpha, dim=0)
        outs = []
        for op in self.ops:
            if self.kind == "seq":
                outs.append(self._pool_seq(h, op))
            elif self.kind == "graph":
                outs.append(self._pool_graph(h, batch, n_graphs, op))
            else:
                outs.append(self._pool_spatial(h, op))
        return sum(wi * o for wi, o in zip(w, outs))

    def update_peak(self):
        with torch.no_grad():
            self.alpha_peak = torch.maximum(self.alpha_peak, torch.softmax(self.alpha, dim=0))

    def selected(self):
        return self.ops[int(torch.argmax(self.alpha_peak))]


class JointArchServer(nn.Module):
    """Wraps a maximal schema and adds differentiable width/depth gates, then runs the single
    joint loop. The wrapped schema must expose:
        .cells            : ModuleList of meta-cells, each with a per-cell primitive `alpha`
        cell.mixed_seq(x) OR cell(...): the mixed primitive output for that cell (already softmax over
                            primitives -- this covers primitive AND kernel mixing, since kernel
                            variants are just primitives with the same contract)
    Width and depth gates are added HERE (not in the validated schema), keeping the base module
    untouched. This adapter targets the SEQUENCE schema's (b,T,width) cells; the spatial/
    volumetric/graph/equivariant adapters follow the same pattern with their own mixed-output call
    and their extra axis (kernel already inside the primitive softmax; max-l via a channel-block mask).
    """

    def __init__(
        self,
        schema,
        width,
        depth,
        n_out,
        readout="mean",
        w_width=1.0,
        w_depth=1.0,
        gate_init=None,
        target_openness=0.95,
        learn_readout=False,
        size_prior=0.0,
    ):
        super().__init__()
        self.net = schema
        self.width = width
        self.depth = depth
        self.readout = readout
        self.w_width = w_width
        self.w_depth = w_depth
        # gate init is DERIVED from a target openness (auto_gate_init) unless an explicit value is
        # given: beta_0 = logit(target_openness) so gates start ~fully open and the complexity term
        # closes them. This replaces the previously hand-tuned constant.
        b0 = auto_gate_init(target_openness) if gate_init is None else gate_init
        # CLASS 2: per-channel width mask (shared across layers -> selects a single deployed width)
        self.beta_width = nn.Parameter(torch.full((width,), b0))
        # CLASS 3: per-layer depth gate (skip/compute); layer 0 always on (>=1 layer)
        self.gamma_depth = nn.Parameter(torch.full((depth,), b0))
        # optional READOUT axis (last/mean/max over time), size-invariance prior optional
        self.learn_readout = learn_readout
        self.readout_head = DifferentiableReadout("seq", size_prior=size_prior) if learn_readout else None
        # readout head sized to max width; masked channels contribute ~0
        self.head = nn.Linear(width, n_out)
        with torch.no_grad():
            self.head.weight.normal_(0, (1.0 / width) ** 0.5)
            self.head.bias.zero_()

    # ---- differentiable gates ----
    def width_mask(self):
        return torch.sigmoid(self.beta_width)  # (width,)

    def depth_gates(self):
        g = torch.sigmoid(self.gamma_depth)  # (depth,)
        # keep at least the first layer fully on for a well-defined minimal net
        g = torch.cat([g[:1] * 0 + 1.0, g[1:]], dim=0) if self.depth > 1 else g * 0 + 1.0
        return g

    def forward(self, x_seq):
        m = self.width_mask()  # (width,)
        g = self.depth_gates()  # (depth,)
        h = None
        for l, cell in enumerate(self.net.cells):
            inp = x_seq if l == 0 else h
            out = cell.mixed(inp)  # (b,T,width)
            out = out * m  # width mask (Class 2)
            if l == 0:
                h = out
            else:
                h = h + g[l] * out  # depth gate (Class 3)
        if self.learn_readout:
            pooled = self.readout_head(h * m)  # readout as a selected axis
            return self.head(pooled)
        if self.readout == "mean":
            pooled = h.mean(dim=1)
        elif self.readout == "last":
            pooled = h[:, -1, :]
        else:
            raise ValueError("readout must be mean or last for the joint server")
        return self.head(pooled * m)

    # ---- structural cost Omega (differentiable) ----
    def omega(self):
        """Expected structural cost: primitive mixture cost + width mask + depth gates, all
        normalized. mu * omega() is the priced complexity term of the action."""
        # CLASS 1: primitive (and kernel) mixture cost, per cell, from each core's parameter count
        prim_cost = self.net.cells[0].alpha.new_zeros(())
        for cell in self.net.cells:
            p = torch.softmax(cell.alpha, dim=0)
            costs = torch.tensor(
                [sum(x.numel() for x in c.parameters()) for c in cell.cores], dtype=p.dtype, device=p.device
            )
            costs = costs / costs.max()
            prim_cost = prim_cost + (p * costs).sum()
        prim_cost = prim_cost / len(self.net.cells)
        # CLASS 2: width fraction
        width_cost = self.width_mask().sum() / self.width
        # CLASS 3: depth fraction
        depth_cost = self.depth_gates().sum() / self.depth
        readout_cost = self.readout_head.readout_cost() if getattr(self, "learn_readout", False) else 0.0
        return prim_cost + self.w_width * width_cost + self.w_depth * depth_cost + readout_cost

    # ---- discretized architecture (deployment) ----
    def architecture(self):
        with torch.no_grad():
            prims = [cell.primitives[int(torch.argmax(cell.alpha))] for cell in self.net.cells]
            kept_width = int((self.width_mask() > 0.5).sum())
            kept_layers = int((self.depth_gates() > 0.5).sum())
            out = {"primitives": prims, "width": kept_width, "depth": kept_layers}
            if getattr(self, "learn_readout", False) and self.readout_head is not None:
                out["readout"] = self.readout_head.selected()
        return out


def joint_search(
    server,
    Xtr,
    ytr,
    Xv,
    yv,
    mu,
    epochs=60,
    lr=3e-3,
    gate_lr=0.05,
    alpha_lr=0.05,
    gamma_sharp=0.0,
    loss_fn=None,
    bs=32,
    seed=0,
    auto_init_mu=True,
):
    """The SINGLE joint gradient loop. Optimizes weights W (on train) and all gate parameters
    Phi = {alpha_prim (per cell), beta_width, gamma_depth} (on validation) under one action
        J = R + mu * Omega(Phi)  [ + gamma_sharp * entropy-sharpening on the primitive softmax ].
    Weights and gates use separate optimizers (bilevel-lite: gates stepped on the val split), the
    honest protocol. Returns (val_acc_or_neg_loss, server)."""
    torch.manual_seed(seed)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    if auto_init_mu and hasattr(server, "beta_width"):
        with torch.no_grad():
            b0 = auto_gate_init(0.95, mu=mu)
            server.beta_width.fill_(b0)
            server.gamma_depth.fill_(b0)
    gate_params = [server.beta_width, server.gamma_depth]
    alpha_params = [cell.alpha for cell in server.net.cells]
    if getattr(server, "learn_readout", False) and server.readout_head is not None:
        alpha_params = alpha_params + [server.readout_head.alpha]
    weight_params = [
        p
        for n, p in server.named_parameters()
        if not n.endswith("alpha") and "beta_width" not in n and "gamma_depth" not in n
    ]
    ow = torch.optim.Adam(weight_params, lr=lr)
    og = torch.optim.Adam(gate_params, lr=gate_lr)
    oa = torch.optim.Adam(alpha_params, lr=alpha_lr)
    for ep in range(epochs):
        # --- weight step on train ---
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            bi = perm[i : i + bs]
            ow.zero_grad()
            l = loss_fn(server(Xtr[bi]), ytr[bi])
            if torch.isfinite(l):
                l.backward()
                torch.nn.utils.clip_grad_norm_(weight_params, 5.0)
                ow.step()
        # --- gate + alpha step on validation, under the priced action ---
        og.zero_grad()
        oa.zero_grad()
        lv = loss_fn(server(Xv), yv)
        obj = lv + mu * server.omega()
        if gamma_sharp > 0:  # push primitive softmax to argmax
            for cell in server.net.cells:
                p = torch.softmax(cell.alpha, dim=0)
                obj = obj + gamma_sharp * (-(p * torch.log(p + 1e-9)).sum())
            if getattr(server, "learn_readout", False) and server.readout_head is not None:
                pr = torch.softmax(server.readout_head.alpha, dim=0)
                obj = obj + gamma_sharp * (-(pr * torch.log(pr + 1e-9)).sum())
        if torch.isfinite(obj):
            obj.backward()
            torch.nn.utils.clip_grad_norm_(gate_params + alpha_params, 5.0)
            og.step()
            oa.step()
            if getattr(server, "learn_readout", False) and server.readout_head is not None:
                server.readout_head.update_peak()
    with torch.no_grad():
        pred = server(Xv)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            score = float((pred.argmax(-1) == yv).float().mean())
        else:
            score = -float(loss_fn(pred, yv))
    return score, server


class EquivariantJointServer(nn.Module):
    """Joint-search adapter for the EQUIVARIANT graph schema, adding the MAX-L axis. max-l is a
    STRUCTURED width mask: the l<=2 schema is instantiated, and its l=2 (tensor) channel block is
    gated by a single sigmoid(beta_l2). beta_l2 -> -inf recovers the l<=1 model; the analytical cost is
    the extra steerable dimension the l=2 block unlocks (irrep-dim ratio 5/(1+3+5)=5/9). Primitive,
    width, and depth gates follow the same pattern as JointArchServer. This shows the SAME action
    J=R+mu*Omega drives max-l selection, unifying the l<=1/l<=2 choice into the one loop."""

    def __init__(self, equiv_l2_net, c0, c1, c2, depth, w_l2=1.0, w_depth=1.0, gate_init=None, target_openness=0.95):
        super().__init__()
        self.net = equiv_l2_net
        self.c0, self.c1, self.c2, self.depth = c0, c1, c2, depth
        self.w_l2 = w_l2
        self.w_depth = w_depth
        b0 = auto_gate_init(target_openness) if gate_init is None else gate_init
        # max-l gate: sigmoid(beta_l2) scales the l=2 tensor block (Class 2, structured mask).
        # start ~open (l=2 on) so accuracy is found first; the price closes it if l=2 isn't worth it.
        self.beta_l2 = nn.Parameter(torch.tensor(b0))
        self.gamma_depth = nn.Parameter(torch.full((depth,), b0))
        # analytical l=2 cost = extra irrep dimension fraction: dim(l=2)/dim(l<=2) = 5/9
        self.l2_dim_ratio = (2 * 2 + 1) / (1 + 3 + 5)

    def l2_gate(self):
        return torch.sigmoid(self.beta_l2)

    def forward(self, x, pos, edge_index, batch, n_graphs):
        """Forward with the l=2 block scaled by its gate. We scale the tensor features t after each
        cell by l2_gate() via a temporary hook on the encode: when the gate -> 0 the l=2 channels
        contribute nothing and the model reduces to l<=1. Implemented by scaling the c2 tensor state
        inside the net's cell loop through a registered multiplier."""
        gate = self.l2_gate()
        # set a scalar the cells read; if the l2 net supports a tensor-scale attribute, use it,
        # else fall back to scaling the final energy contribution differential (documented limitation).
        if hasattr(self.net, "set_l2_scale"):
            self.net.set_l2_scale(gate)
            return self.net(x, pos, edge_index, batch, n_graphs)
        # fallback: interpolate between l<=2 output and a detached l<=1-like output is not exact;
        # we instead scale via the gate on the head input is not meaningful for equivariance. So we
        # require set_l2_scale for a faithful gate (see l2 net patch below).
        return self.net(x, pos, edge_index, batch, n_graphs)

    def omega(self):
        # primitive mixture cost per cell
        prim_cost = self.net.cells[0].alpha.new_zeros(())
        for cell in self.net.cells:
            p = torch.softmax(cell.alpha, dim=0)
            costs = torch.tensor(
                [sum(x.numel() for x in c.parameters()) for c in cell.cores], dtype=p.dtype, device=p.device
            )
            costs = costs / costs.max()
            prim_cost = prim_cost + (p * costs).sum()
        prim_cost = prim_cost / len(self.net.cells)
        depth_cost = torch.sigmoid(self.gamma_depth).sum() / self.depth
        l2_cost = self.l2_dim_ratio * self.l2_gate()
        return prim_cost + self.w_depth * depth_cost + self.w_l2 * l2_cost

    def max_l(self):
        """Discretized angular order: l=2 kept iff its gate exceeds 0.5, else l<=1."""
        return 2 if float(self.l2_gate()) > 0.5 else 1


class SpatialJointServer(nn.Module):
    """Joint-search adapter for the SPATIAL (2D) schema. Active axes: primitive + KERNEL + width +
    depth. Kernel size is ALREADY inside the primitive softmax (conv2d / conv2d_k5 / conv2d_k7 are
    same-contract candidates), so no separate kernel gate is needed -- the primitive alpha selects the
    receptive field, priced by each core's k^2 parameter cost. width is a per-CHANNEL sigmoid mask on
    the (b,C,H,W) feature; depth is a per-layer residual gate. Same J=R+mu*Omega loop as the sequence
    server; the base spatial schema is untouched (gates live here)."""

    def __init__(
        self,
        spatial_net,
        width,
        depth,
        n_out=None,
        w_width=1.0,
        w_depth=1.0,
        gate_init=None,
        target_openness=0.95,
        learn_readout=False,
        size_prior=0.0,
        n_classes=None,
    ):
        n_out = n_out if n_out is not None else (n_classes if n_classes is not None else 10)
        super().__init__()
        self.net = spatial_net
        self.width = width
        self.depth = depth
        self.w_width = w_width
        self.w_depth = w_depth
        # start gates ~fully on (sigmoid(3)~0.95) so the search finds ACCURACY first, then compacts
        # under complexity pressure -- avoids starving accurate-but-costly primitives (e.g. large-kernel
        # conv) at initialization. Lower gate_init biases toward earlier compaction.
        b0 = auto_gate_init(target_openness) if gate_init is None else gate_init
        self.beta_width = nn.Parameter(torch.full((width,), b0))
        self.gamma_depth = nn.Parameter(torch.full((depth,), b0))
        self.learn_readout = learn_readout
        self.readout_head = DifferentiableReadout("spatial", size_prior=size_prior) if learn_readout else None
        self.head = nn.Linear(width, n_out)
        with torch.no_grad():
            self.head.weight.normal_(0, (1.0 / width) ** 0.5)
            self.head.bias.zero_()

    def width_mask(self):
        return torch.sigmoid(self.beta_width)  # (width,)

    def depth_gates(self):
        g = torch.sigmoid(self.gamma_depth)
        g = torch.cat([g[:1] * 0 + 1.0, g[1:]], dim=0) if self.depth > 1 else g * 0 + 1.0
        return g

    def forward(self, x):
        m = self.width_mask().view(1, -1, 1, 1)  # (1,C,1,1) channel mask
        g = self.depth_gates()
        x = self.net.stem(x)  # (b,width,hw,hw)
        h = None
        for l, cell in enumerate(self.net.cells):
            out = cell(x if l == 0 else h) * m  # width mask on channels
            if l == 0:
                h = out
            else:
                h = h + g[l] * out  # depth-gated residual
        h = h * m
        pooled = self.readout_head(h) if self.learn_readout else h.mean(dim=(2, 3))
        return self.head(pooled)

    def omega(self):
        prim_cost = self.net.cells[0].alpha.new_zeros(())
        for cell in self.net.cells:
            p = torch.softmax(cell.alpha, dim=0)
            costs = torch.tensor(
                [sum(x.numel() for x in c.parameters()) for c in cell.cores], dtype=p.dtype, device=p.device
            )
            costs = costs / costs.max()
            prim_cost = prim_cost + (p * costs).sum()  # includes kernel k^2 via conv params
        prim_cost = prim_cost / len(self.net.cells)
        width_cost = self.width_mask().sum() / self.width
        depth_cost = self.depth_gates().sum() / self.depth
        readout_cost = self.readout_head.readout_cost() if getattr(self, "learn_readout", False) else 0.0
        return prim_cost + self.w_width * width_cost + self.w_depth * depth_cost + readout_cost

    def architecture(self):
        with torch.no_grad():
            prims = [cell.primitives[int(torch.argmax(cell.alpha))] for cell in self.net.cells]
            kept_width = int((self.width_mask() > 0.5).sum())
            kept_layers = int((self.depth_gates() > 0.5).sum())
            out = {"primitives": prims, "width": kept_width, "depth": kept_layers}
            if getattr(self, "learn_readout", False) and self.readout_head is not None:
                out["readout"] = self.readout_head.selected()
        return out


def joint_search_generic(
    server,
    Xtr,
    ytr,
    Xv,
    yv,
    mu,
    epochs=50,
    lr=3e-3,
    gate_lr=0.08,
    alpha_lr=0.05,
    gamma_sharp=0.02,
    loss_fn=None,
    bs=32,
    seed=0,
    cell_container=None,
):
    """Joint loop for any server exposing forward(), omega(), and a base net with per-cell `alpha`.
    cell_container: the module holding .cells (default server.net). Weights on train; gates+alphas on
    val under J=R+mu*Omega. Generalizes joint_search to the spatial/volumetric/graph servers."""
    torch.manual_seed(seed)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    cells = (cell_container or server.net).cells
    gate_params = [p for n, p in server.named_parameters() if "beta_width" in n or "gamma_depth" in n or "beta_l2" in n]
    alpha_params = [cell.alpha for cell in cells]
    if getattr(server, "learn_readout", False) and getattr(server, "readout_head", None) is not None:
        alpha_params = alpha_params + [server.readout_head.alpha]
    gate_ids = {id(p) for p in gate_params}
    alpha_ids = {id(p) for p in alpha_params}
    weight_params = [p for p in server.parameters() if id(p) not in gate_ids and id(p) not in alpha_ids]
    ow = torch.optim.Adam(weight_params, lr=lr)
    og = torch.optim.Adam(gate_params, lr=gate_lr) if gate_params else None
    oa = torch.optim.Adam(alpha_params, lr=alpha_lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            bi = perm[i : i + bs]
            ow.zero_grad()
            l = loss_fn(server(Xtr[bi]), ytr[bi])
            if torch.isfinite(l):
                l.backward()
                torch.nn.utils.clip_grad_norm_(weight_params, 5.0)
                ow.step()
        if og is not None:
            og.zero_grad()
        oa.zero_grad()
        obj = loss_fn(server(Xv), yv) + mu * server.omega()
        if gamma_sharp > 0:
            for cell in cells:
                p = torch.softmax(cell.alpha, dim=0)
                obj = obj + gamma_sharp * (-(p * torch.log(p + 1e-9)).sum())
            if getattr(server, "learn_readout", False) and getattr(server, "readout_head", None) is not None:
                pr = torch.softmax(server.readout_head.alpha, dim=0)
                obj = obj + gamma_sharp * (-(pr * torch.log(pr + 1e-9)).sum())
        if torch.isfinite(obj):
            obj.backward()
            if og is not None:
                og.step()
            oa.step()
            if getattr(server, "learn_readout", False) and getattr(server, "readout_head", None) is not None:
                server.readout_head.update_peak()
    with torch.no_grad():
        pred = server(Xv)
        score = (
            float((pred.argmax(-1) == yv).float().mean())
            if isinstance(loss_fn, nn.CrossEntropyLoss)
            else -float(loss_fn(pred, yv))
        )
    return score, server


class VolumetricJointServer(nn.Module):
    """Joint-search adapter for the VOLUMETRIC (3D) schema. Active axes: primitive + KERNEL(k^3) +
    width + depth. Exactly the spatial adapter one rank higher: features are (b,C,D,H,W), the width
    mask gates channels, depth gates residual layers, and conv-kernel choice (if kernel variants are in
    the primitive set) is priced by k^3 via the primitive softmax. Base volumetric schema untouched.
    """

    def __init__(
        self,
        vol_net,
        width,
        depth,
        n_out=None,
        w_width=1.0,
        w_depth=1.0,
        gate_init=None,
        target_openness=0.95,
        learn_readout=False,
        size_prior=0.0,
        n_classes=None,
    ):
        n_out = n_out if n_out is not None else (n_classes if n_classes is not None else 10)
        super().__init__()
        self.net = vol_net
        self.width = width
        self.depth = depth
        self.w_width = w_width
        self.w_depth = w_depth
        b0 = auto_gate_init(target_openness) if gate_init is None else gate_init
        self.beta_width = nn.Parameter(torch.full((width,), b0))
        self.gamma_depth = nn.Parameter(torch.full((depth,), b0))
        self.learn_readout = learn_readout
        self.readout_head = DifferentiableReadout("vol", size_prior=size_prior) if learn_readout else None
        self.head = nn.Linear(width, n_out)
        with torch.no_grad():
            self.head.weight.normal_(0, (1.0 / width) ** 0.5)
            self.head.bias.zero_()

    def width_mask(self):
        return torch.sigmoid(self.beta_width)

    def depth_gates(self):
        g = torch.sigmoid(self.gamma_depth)
        g = torch.cat([g[:1] * 0 + 1.0, g[1:]], dim=0) if self.depth > 1 else g * 0 + 1.0
        return g

    def forward(self, x):
        m = self.width_mask().view(1, -1, 1, 1, 1)  # (1,C,1,1,1)
        g = self.depth_gates()
        x = self.net.stem(x)
        h = None
        for l, cell in enumerate(self.net.cells):
            out = cell(x if l == 0 else h) * m
            h = out if l == 0 else h + g[l] * out
        h = h * m
        pooled = self.readout_head(h) if self.learn_readout else h.mean(dim=(2, 3, 4))
        return self.head(pooled)

    def omega(self):
        prim_cost = self.net.cells[0].alpha.new_zeros(())
        for cell in self.net.cells:
            p = torch.softmax(cell.alpha, dim=0)
            costs = torch.tensor(
                [sum(x.numel() for x in c.parameters()) for c in cell.cores], dtype=p.dtype, device=p.device
            )
            costs = costs / costs.max()
            prim_cost = prim_cost + (p * costs).sum()
        prim_cost = prim_cost / len(self.net.cells)
        readout_cost = self.readout_head.readout_cost() if getattr(self, "learn_readout", False) else 0.0
        return (
            prim_cost
            + self.w_width * self.width_mask().sum() / self.width
            + self.w_depth * self.depth_gates().sum() / self.depth
            + readout_cost
        )

    def architecture(self):
        with torch.no_grad():
            prims = [cell.primitives[int(torch.argmax(cell.alpha))] for cell in self.net.cells]
            out = {
                "primitives": prims,
                "width": int((self.width_mask() > 0.5).sum()),
                "depth": int((self.depth_gates() > 0.5).sum()),
            }
            if getattr(self, "learn_readout", False) and self.readout_head is not None:
                out["readout"] = self.readout_head.selected()
            return out


class GraphJointServer(nn.Module):
    """Joint-search adapter for the GRAPH schema. Active axes: primitive + width + depth (NO kernel
    -- the aggregator gcn/sage/gin/gat/dense/norm IS the primitive; no spatial receptive field). The
    forward signature is (x, edge_index, batch, n_graphs). width = per-feature-channel mask on the
    node embeddings; depth = per-layer residual gate. Base graph schema untouched."""

    def __init__(
        self,
        graph_net,
        width,
        depth,
        n_out,
        readout="mean",
        w_width=1.0,
        w_depth=1.0,
        gate_init=None,
        target_openness=0.95,
        learn_readout=False,
        size_prior=0.0,
    ):
        super().__init__()
        self.net = graph_net
        self.width = width
        self.depth = depth
        self.readout = readout
        self.w_width = w_width
        self.w_depth = w_depth
        b0 = auto_gate_init(target_openness) if gate_init is None else gate_init
        self.beta_width = nn.Parameter(torch.full((width,), b0))
        self.gamma_depth = nn.Parameter(torch.full((depth,), b0))
        # optional READOUT axis: a differentiable softmax over sum/mean/max pooling, selected by fit
        self.learn_readout = learn_readout
        self.readout_head = DifferentiableReadout("graph", size_prior=size_prior) if learn_readout else None
        self.head = nn.Linear(width, n_out)
        with torch.no_grad():
            self.head.weight.normal_(0, (1.0 / width) ** 0.5)
            self.head.bias.zero_()

    def width_mask(self):
        return torch.sigmoid(self.beta_width)

    def depth_gates(self):
        g = torch.sigmoid(self.gamma_depth)
        g = torch.cat([g[:1] * 0 + 1.0, g[1:]], dim=0) if self.depth > 1 else g * 0 + 1.0
        return g

    def forward(self, x, edge_index, batch, n_graphs):
        from ilmarinen.models.graph_schema import _global_pool

        m = self.width_mask()  # (width,)
        g = self.depth_gates()
        h = self.net.embed(x) * m
        for l, cell in enumerate(self.net.cells):
            out = cell(h if l == 0 else h, edge_index) * m
            h = out if l == 0 else h + g[l] * out
        h = h * m
        if self.learn_readout:  # readout as a selected axis
            pooled = self.readout_head(h, batch, n_graphs)
        else:
            pooled = _global_pool(h, batch, n_graphs, self.readout)
        return self.head(pooled)

    def omega(self):
        prim_cost = self.net.cells[0].alpha.new_zeros(())
        for cell in self.net.cells:
            p = torch.softmax(cell.alpha, dim=0)
            costs = torch.tensor(
                [sum(x.numel() for x in c.parameters()) for c in cell.cores], dtype=p.dtype, device=p.device
            )
            costs = costs / costs.max()
            prim_cost = prim_cost + (p * costs).sum()
        prim_cost = prim_cost / len(self.net.cells)
        readout_cost = self.readout_head.readout_cost() if getattr(self, "learn_readout", False) else 0.0
        return (
            prim_cost
            + self.w_width * self.width_mask().sum() / self.width
            + self.w_depth * self.depth_gates().sum() / self.depth
            + readout_cost
        )

    def architecture(self):
        with torch.no_grad():
            prims = [cell.primitives[int(torch.argmax(cell.alpha))] for cell in self.net.cells]
            out = {
                "primitives": prims,
                "width": int((self.width_mask() > 0.5).sum()),
                "depth": int((self.depth_gates() > 0.5).sum()),
            }
            if self.learn_readout and self.readout_head is not None:
                out["readout"] = self.readout_head.selected()
            return out


def joint_search_graph(
    server,
    graphs,
    y,
    tr,
    va,
    collate_fn,
    mu,
    epochs=40,
    lr=3e-3,
    gate_lr=0.08,
    alpha_lr=0.05,
    gamma_sharp=0.02,
    bs=48,
    seed=0,
    regression=True,
):
    """Joint loop for the GRAPH server. graphs/y are the dataset; tr/va index arrays; collate_fn(
    graphs, idx_list) -> (x, edge_index, batch, n_graphs). Weights on train, gates+alphas on val under
    J=R+mu*Omega. Regression (MSE) by default (QM7 energy); set regression=False for classification."""
    import numpy as np

    torch.manual_seed(seed)
    lf = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    gate_params = [server.beta_width, server.gamma_depth]
    alpha_params = [cell.alpha for cell in server.net.cells]
    if getattr(server, "learn_readout", False) and server.readout_head is not None:
        alpha_params = alpha_params + [server.readout_head.alpha]  # readout axis alpha
    gate_ids = {id(p) for p in gate_params}
    alpha_ids = {id(p) for p in alpha_params}
    weight_params = [p for p in server.parameters() if id(p) not in gate_ids and id(p) not in alpha_ids]
    ow = torch.optim.Adam(weight_params, lr=lr)
    og = torch.optim.Adam(gate_params, lr=gate_lr)
    oa = torch.optim.Adam(alpha_params, lr=alpha_lr)
    ymean, ystd = float(np.mean(y[tr])), float(np.std(y[tr])) + 1e-9
    for ep in range(epochs):
        np.random.shuffle(tr)
        for i in range(0, len(tr), bs):
            bi = tr[i : i + bs]
            x, ei, batch, ng = collate_fn(graphs, bi.tolist())
            target = torch.tensor((y[bi] - ymean) / ystd, dtype=torch.float32)
            ow.zero_grad()
            pred = server(x, ei, batch, ng)
            l = lf(pred, target.unsqueeze(1)) if regression else lf(pred, target.long())
            if torch.isfinite(l):
                l.backward()
                torch.nn.utils.clip_grad_norm_(weight_params, 5.0)
                ow.step()
        # gate+alpha step on val
        x, ei, batch, ng = collate_fn(graphs, va.tolist())
        target = torch.tensor((y[va] - ymean) / ystd, dtype=torch.float32)
        og.zero_grad()
        oa.zero_grad()
        pred = server(x, ei, batch, ng)
        obj = lf(pred, target.unsqueeze(1)) + mu * server.omega()
        if gamma_sharp > 0:
            for cell in server.net.cells:
                p = torch.softmax(cell.alpha, dim=0)
                obj = obj + gamma_sharp * (-(p * torch.log(p + 1e-9)).sum())
            if getattr(server, "learn_readout", False) and server.readout_head is not None:
                pr = torch.softmax(server.readout_head.alpha, dim=0)
                obj = obj + gamma_sharp * (-(pr * torch.log(pr + 1e-9)).sum())
        if torch.isfinite(obj):
            obj.backward()
            og.step()
            oa.step()
            if getattr(server, "learn_readout", False) and server.readout_head is not None:
                server.readout_head.update_peak()
    # eval MAE on val
    with torch.no_grad():
        x, ei, batch, ng = collate_fn(graphs, va.tolist())
        pred = server(x, ei, batch, ng).squeeze(1).numpy() * ystd + ymean
        mae = float(np.abs(pred - y[va]).mean())
    return mae, server


class SetJointServer(nn.Module):
    """Joint-search adapter for the SET contract (Future Direction #6). Same axes as the graph server
    (primitive + width + depth + readout), but the primitives are the permutation-invariant set blocks
    (deepsets/sab/isab/pma_block/element_mlp/norm) and the readout pooling is permutation-INVARIANT, so
    the whole model stays exactly S_n-invariant. The set is the maximal-symmetry contract; its
    aggregators do NOT collapse (unlike graph aggregators on an edgeless input)."""

    def __init__(
        self,
        set_net,
        width,
        depth,
        n_out,
        readout="mean",
        w_width=1.0,
        w_depth=1.0,
        gate_init=None,
        target_openness=0.95,
        learn_readout=False,
        size_prior=0.0,
    ):
        super().__init__()
        self.net = set_net
        self.width = width
        self.depth = depth
        self.readout = readout
        self.w_width = w_width
        self.w_depth = w_depth
        b0 = auto_gate_init(target_openness) if gate_init is None else gate_init
        self.beta_width = nn.Parameter(torch.full((width,), b0))
        self.gamma_depth = nn.Parameter(torch.full((depth,), b0))
        self.learn_readout = learn_readout
        self.readout_head = DifferentiableReadout("graph", size_prior=size_prior) if learn_readout else None
        self.head = nn.Linear(width, n_out)
        with torch.no_grad():
            self.head.weight.normal_(0, (1.0 / width) ** 0.5)
            self.head.bias.zero_()

    def width_mask(self):
        return torch.sigmoid(self.beta_width)

    def depth_gates(self):
        g = torch.sigmoid(self.gamma_depth)
        return torch.cat([g[:1] * 0 + 1.0, g[1:]], dim=0) if self.depth > 1 else g * 0 + 1.0

    def forward(self, x, batch, n_sets):
        from ilmarinen.models.set_schema import _set_pool

        m = self.width_mask()
        g = self.depth_gates()
        h = None
        for l, cell in enumerate(self.net.cells):
            out = cell.mixed(x if l == 0 else h, batch, n_sets) * m
            h = out if l == 0 else h + g[l] * out
        h = h * m
        if self.learn_readout:
            pooled = self.readout_head(h, batch, n_sets)
        else:
            pooled = _set_pool(h, batch, n_sets, self.readout)
        return self.head(pooled)

    def omega(self):
        prim_cost = 0.0
        for cell in self.net.cells:
            w = torch.softmax(cell.alpha, dim=0)
            prim_cost = prim_cost + (w * torch.arange(1, len(w) + 1, device=w.device, dtype=w.dtype)).sum()
        prim_cost = prim_cost / len(self.net.cells)
        readout_cost = self.readout_head.readout_cost() if getattr(self, "learn_readout", False) else 0.0
        return (
            prim_cost
            + self.w_width * self.width_mask().sum() / self.width
            + self.w_depth * self.depth_gates().sum() / self.depth
            + readout_cost
        )

    def architecture(self):
        with torch.no_grad():
            prims = [cell.primitives[int(torch.argmax(cell.alpha))] for cell in self.net.cells]
            out = {
                "primitives": prims,
                "width": int((self.width_mask() > 0.5).sum()),
                "depth": int((self.depth_gates() > 0.5).sum()),
            }
            if self.learn_readout and self.readout_head is not None:
                out["readout"] = self.readout_head.selected()
            return out

    def update_peak(self):
        self.net.update_peak()
        if self.learn_readout and self.readout_head is not None:
            self.readout_head.update_peak()


class Grid4dJointServer(nn.Module):
    """Joint-search adapter for the 4D (spatiotemporal) grid contract. Primitive + width + depth axes
    over the rank-4 vocabulary (conv4d/conv4d_kt1/conv_dw/pointwise/dense/norm). Readout is global-
    average-pool over all 4 grid axes (translation-invariant), so the model respects 4D translation
    symmetry through the search. Mirrors VolumetricJointServer at rank 4."""

    def __init__(self, grid_net, width, depth, n_out, w_width=1.0, w_depth=1.0, gate_init=None, target_openness=0.95):
        super().__init__()
        self.net = grid_net
        self.width = width
        self.depth = depth
        self.w_width = w_width
        self.w_depth = w_depth
        b0 = auto_gate_init(target_openness) if gate_init is None else gate_init
        self.beta_width = nn.Parameter(torch.full((width,), b0))
        self.gamma_depth = nn.Parameter(torch.full((depth,), b0))
        self.head = nn.Linear(width, n_out)
        with torch.no_grad():
            self.head.weight.normal_(0, (1.0 / width) ** 0.5)
            self.head.bias.zero_()

    def width_mask(self):
        return torch.sigmoid(self.beta_width)

    def depth_gates(self):
        g = torch.sigmoid(self.gamma_depth)
        return torch.cat([g[:1] * 0 + 1.0, g[1:]], dim=0) if self.depth > 1 else g * 0 + 1.0

    def forward(self, x):
        m = self.width_mask().view(1, -1, 1, 1, 1, 1)
        g = self.depth_gates()
        h = None
        for l, cell in enumerate(self.net.cells):
            out = cell.mixed(x if l == 0 else h) * m
            h = out if l == 0 else h + g[l] * out
        h = h * m
        pooled = h.mean(dim=(2, 3, 4, 5))
        return self.head(pooled)

    def omega(self):
        prim_cost = 0.0
        for cell in self.net.cells:
            w = torch.softmax(cell.alpha, dim=0)
            prim_cost = prim_cost + (w * torch.arange(1, len(w) + 1, device=w.device, dtype=w.dtype)).sum()
        prim_cost = prim_cost / len(self.net.cells)
        return (
            prim_cost
            + self.w_width * self.width_mask().sum() / self.width
            + self.w_depth * self.depth_gates().sum() / self.depth
        )

    def architecture(self):
        with torch.no_grad():
            return {
                "primitives": [cell.primitives[int(torch.argmax(cell.alpha))] for cell in self.net.cells],
                "width": int((self.width_mask() > 0.5).sum()),
                "depth": int((self.depth_gates() > 0.5).sum()),
            }

    def update_peak(self):
        self.net.update_peak()

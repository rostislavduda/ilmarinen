"""Robust discretization of the metaoptimizer's selections under tied / identical architecture weights.

Raw argmax(alpha) silently breaks ties by FIRST INDEX -- arbitrary (depends on vocabulary ordering),
unstable (a tiny perturbation flips it), and silent (no signal it was a coin-flip). Two distinct
edge cases need principled handling:

CASE 1 -- TIED alpha WITHIN a layer (top-1 ~ top-2). Fix: a tie-aware discretization that
  (a) DETECTS the tie (top1 - top2 < eps),
  (b) BREAKS it principledly instead of by list order: prefer the CHEAPER primitive (the priced-Omega
      / Occam tiebreak, consistent with J = R + mu*Omega); if cost-tied, fall back to the ablation
      solo-loss (which op is genuinely better ALONE, defeating co-adaptation ties from #5),
  (c) FLAGS it (returns the tie + margin so #7's confidence reflects the ambiguity).

CASE 2 -- IDENTICAL selection ACROSS adjacent layers. Valid (two conv layers is fine) but can signal
  DEPTH REDUNDANCY: if adjacent layers select the same primitive AND collapsing them (dropping the
  second) barely changes val loss, the extra depth is not doing independent work. Fix: surface it as a
  diagnostic (recommend the shallower net) -- catching what the priced-depth marginal-value rule should
  have caught.
"""

from __future__ import annotations

import numpy as np
import torch


def _primitive_costs(primitives, cost_map=None):
    """Relative cost per primitive for the Occam tiebreak. Default: parameter-free ops (norm) cheapest,
    then pointwise/plain, conv, attention/dense most expensive -- a coarse capacity ordering."""
    default = {
        "norm": 0.5,
        "pointwise": 0.8,
        "plain": 1.0,
        "gated": 1.2,
        "conv": 1.5,
        "conv2d": 1.5,
        "spectral": 1.6,
        "lstm": 1.8,
        "linssm": 1.8,
        "selssm": 2.0,
        "attention": 2.5,
        "dense": 3.0,
        "gcn": 1.3,
        "sage": 1.4,
        "gin": 1.6,
        "gat": 2.2,
    }
    cm = cost_map or default
    return np.array([cm.get(p, 1.5) for p in primitives])


def robust_discretize(server, eval_fn=None, tie_eps=0.05, cost_map=None):
    """Tie-aware per-layer discretization. For each cell, return the selected primitive with a
    principled tiebreak and a flag. eval_fn (optional) enables the ablation solo-loss tiebreak when
    cost is also tied. Returns list of dicts: selected, tied (bool), margin, tiebreak, alpha_top2."""
    from ilmarinen.machinery.coadapt import solo_importances

    out = []
    for cell in server.net.cells:
        w = torch.softmax(cell.alpha, dim=0).detach().cpu().numpy()
        order = np.argsort(-w)
        top1, top2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        margin = float(w[top1] - w[top2])
        prims = cell.primitives
        if margin >= tie_eps:
            out.append(
                {
                    "selected": prims[top1],
                    "tied": False,
                    "margin": round(margin, 4),
                    "tiebreak": "clear_argmax",
                    "alpha_top2": [prims[top1], prims[top2]],
                }
            )
            continue
        # TIE: gather the tied set (all within tie_eps of the top)
        tied_idx = [int(i) for i in order if w[order[0]] - w[i] < tie_eps]
        costs = _primitive_costs([prims[i] for i in tied_idx], cost_map)
        cmin = costs.min()
        cheap = [tied_idx[j] for j in range(len(tied_idx)) if costs[j] <= cmin + 1e-9]
        if len(cheap) == 1:
            sel = cheap[0]
            reason = "cheapest_among_tied (Occam/Omega)"
        elif eval_fn is not None:
            # cost-tied too: ablation solo-loss among the cost-cheapest tied ops
            solo = solo_importances(server, eval_fn, cell)
            sel = min(cheap, key=lambda i: solo[i])
            reason = "ablation_solo_best (cost-tied)"
        else:
            sel = cheap[0]
            reason = "cheapest_then_first (no eval_fn for ablation)"
        out.append(
            {
                "selected": prims[sel],
                "tied": True,
                "margin": round(margin, 4),
                "tiebreak": reason,
                "tied_set": [prims[i] for i in tied_idx],
            }
        )
    return out


def depth_redundancy_report(server, collate_and_eval, tol=0.02):
    """CASE 2: detect adjacent layers with IDENTICAL selection whose collapse barely changes val loss.
    collate_and_eval(drop_layer) -> val loss with that layer's depth gate forced to 0 (dropped);
    collate_and_eval(None) -> full-net val loss. Returns per-adjacent-pair dicts: layers, same_primitive,
    drop_delta (val-loss increase from dropping the 2nd), redundant (bool). A redundant pair means the
    extra depth does no independent work -> recommend the shallower net."""
    prims = [cell.primitives[int(torch.argmax(cell.alpha))] for cell in server.net.cells]
    base = collate_and_eval(None)
    report = []
    for l in range(1, len(prims)):
        same = prims[l] == prims[l - 1]
        delta = collate_and_eval(l) - base  # val-loss increase from dropping layer l
        redundant = same and (delta <= tol)
        report.append(
            {
                "layers": (l - 1, l),
                "same_primitive": same,
                "primitive": prims[l],
                "drop_delta": round(float(delta), 4),
                "redundant": redundant,
            }
        )
    n_red = sum(r["redundant"] for r in report)
    return {
        "pairs": report,
        "n_redundant": n_red,
        "recommendation": (
            f"{n_red} redundant layer(s): the depth does no independent work; recommend the shallower net"
            if n_red
            else "no depth redundancy: each layer contributes"
        ),
    }

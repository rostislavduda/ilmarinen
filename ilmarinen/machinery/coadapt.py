"""Treatment of the DARTS co-adaptation pathology (Future Direction #5).

THE PATHOLOGY. In a differentiable-search supernet, all candidate operations run together and their
outputs are mixed by softmax(alpha). Two coupled failure modes result:
  (A) PARAMETER-FREE DOMINANCE: parameter-free ops (skip, norm) train faster and get an unfair early
      advantage, so the softmax over-weights them (Zela 2020; Chu 2020, DARTS-).
  (B) CO-ADAPTATION / INFORMATION BYPASS: ops co-adapt in the shared supernet -- a high-capacity branch
      can dominate the MIXTURE objective by fitting the residual the others leave, so alpha measures
      "who helps the mixture", not "who is best ALONE". The argmax-alpha op can then be the WRONG one.
We observed (B) directly: on a 450-molecule equivariant selection the dense branch dominated via
co-adaptation and argmax-alpha selection failed, while solo comparison recovered the right branch.

THE TREATMENTS (literature-grounded, indicator-free, wired to work with our joint servers):
  T1  alpha PERTURBATION (SDARTS; Chen & Hsieh 2020): perturb alpha with random noise each step during
      search -> implicitly regularizes the collapse indicator (Hessian) and flattens the sharp alpha
      landscape, reducing the unfair advantage.
  T2  alpha REGULARIZATION (beta-DARTS; Ye 2022): decay on the alpha logits (a smooth L2 on the logits)
      -> prevents any single op's logit running away, a generic explicit regularizer.
  T3  ABLATION SELECTION (DARTS-PT; Wang 2021): DISCRETIZE not by argmax alpha but by each op's SOLO
      importance = how much removing it (routing all weight to it, or measuring it alone) changes the
      validation loss. This directly defeats co-adaptation because it scores each op's standalone
      contribution to the trained supernet, not its mixture weight.

CHARACTERIZATION. measure_coadaptation compares the argmax-alpha choice to the solo-best choice on a
held-out split and returns the disagreement plus each op's mixture-weight vs solo-loss -- a direct
quantification of the pathology.
"""
from __future__ import annotations
import numpy as np
import torch


def perturb_alpha(alpha, scale=0.1):
    """T1 (SDARTS random smoothing): return alpha + Gaussian noise for a perturbed forward during
    search. Applied to the architecture logits, not weights. scale ~ 0.1 is a mild smoothing."""
    return alpha + scale * torch.randn_like(alpha)


def alpha_reg_loss(alphas, weight=1e-3):
    """T2 (beta-DARTS): smooth L2 decay on the architecture logits to prevent logit runaway. alphas is
    a list of alpha parameters. Returns a scalar to add to the objective."""
    return weight * sum((a ** 2).sum() for a in alphas)


@torch.no_grad()
def solo_importances(server, eval_fn, cell, restore=True):
    """T3 core: for one cell, score each candidate op by its SOLO validation loss -- set alpha to a
    one-hot on op i, evaluate the whole net's val loss, for every i. Lower solo loss = better op. This
    is the ablation/perturbation importance (DARTS-PT flavour): it measures each op ALONE, defeating
    co-adaptation. eval_fn() must return a scalar val loss with the server in its current alpha state.
    Returns np.array of per-op solo losses (len = n ops in the cell)."""
    saved = cell.alpha.detach().clone()
    n = cell.alpha.shape[0]
    losses = np.zeros(n)
    for i in range(n):
        cell.alpha.zero_(); cell.alpha[i] = 20.0            # ~one-hot on op i
        losses[i] = float(eval_fn())
    if restore:
        cell.alpha.copy_(saved)
    return losses


def ablation_select(server, eval_fn):
    """T3 (DARTS-PT selection): pick, per cell, the op with the LOWEST solo validation loss (the op that
    alone gives the best val performance), instead of argmax(alpha). Returns a list of selected op names
    per cell. Defeats co-adaptation: a co-adapted high-capacity op that wins the mixture but is poor
    alone will NOT be selected.

    IMPORTANT BOUNDARY (validated on real data, QM7 graph pna): this one-hot ablation uses the
    MIXTURE-TRAINED weights. It reliably exposes PARAMETER-FREE / LOW-capacity co-adapters (norm, skip)
    that fit residuals and collapse when isolated. But it can UNDER-rate a HIGH-capacity op (e.g. PNA
    with a large internal projection) whose weights were co-adapted to the mixture context and never
    learned to solve the task alone -- forcing those weights solo gives a FALSE NEGATIVE. For the true
    expressiveness ranking of high-capacity ops, use clean_solo_select (train each op from scratch)."""
    selected = []
    for cell in server.net.cells:
        solo = solo_importances(server, eval_fn, cell)
        selected.append(cell.primitives[int(np.argmin(solo))])
    return selected


def clean_solo_select(build_and_train_solo, primitives):
    """Robust expressiveness ranking that avoids the ablation false-negative for high-capacity ops: train
    EACH primitive ALONE from scratch (single-primitive vocabulary) and rank by held-out performance.
    This is slower than ablation_select (one full train per primitive) but is the reliable readout when
    candidates differ in capacity, because every op's weights are trained to solve the task on their own
    -- no mixture co-adaptation to distort the comparison.

    build_and_train_solo(prim) -> float score (higher = better, e.g. R^2 or accuracy). Returns
    (best_primitive, {primitive: score}). Use ablation_select as the fast proxy; use this when the fast
    proxy disagrees with intuition or when a high-capacity op is a candidate (validated: on QM7 graphs
    ablation under-rates pna, clean-solo correctly ranks pna best)."""
    scores = {p: float(build_and_train_solo(p)) for p in primitives}
    best = max(scores, key=scores.get)
    return best, scores


@torch.no_grad()
def measure_coadaptation(server, eval_fn):
    """CHARACTERIZE the pathology: for each cell, compare the argmax-alpha op to the solo-best op and
    report the disagreement, plus the mixture weight and solo loss of each. Returns dict per cell with
    argmax_op, solo_best_op, disagree (bool), and the arrays. A high disagreement rate across cells is
    the co-adaptation signature (alpha says one thing, standalone performance says another).

    NOTE: solo_best here is the ablation (mixture-trained one-hot) best; see ablation_select's boundary
    note -- for high-capacity candidates cross-check with clean_solo_select."""
    report = []
    n_disagree = 0
    for ci, cell in enumerate(server.net.cells):
        w = torch.softmax(cell.alpha, dim=0).cpu().numpy()
        solo = solo_importances(server, eval_fn, cell)
        argmax_op = cell.primitives[int(np.argmax(w))]
        solo_best = cell.primitives[int(np.argmin(solo))]
        disagree = argmax_op != solo_best
        n_disagree += int(disagree)
        report.append({"cell": ci, "argmax_alpha_op": argmax_op, "solo_best_op": solo_best,
                       "disagree": disagree, "alpha_weights": w.round(3).tolist(),
                       "solo_losses": solo.round(4).tolist()})
    return {"per_cell": report, "n_cells": len(report), "n_disagree": n_disagree,
            "disagreement_rate": n_disagree / max(1, len(report)),
            "interpretation": "high disagreement_rate = co-adaptation (argmax-alpha != solo-best); "
                              "use ablation_select instead of argmax for a robust discretization."}


def robust_select(server, eval_fn, clean_solo_fn=None, disagreement_threshold=0.5, verbose=False):
    """UNIFIED robust primitive selection across ALL contracts. The co-adaptation pathology is
    data/architecture-dependent (empirically it fired on the graph/PNA case but NOT on spatial/atrous or
    sequence/dilconv, despite all three being high-capacity), so robustness cannot be a per-case manual
    judgment -- it must be a single protocol. This function:

      1. Reads argmax(alpha) per cell (the naive selection).
      2. Runs measure_coadaptation -> per-cell ablation solo-best + disagreement_rate.
      3. If disagreement_rate <= threshold, returns the ablation/argmax verdict (fast path, trustworthy).
      4. If disagreement_rate > threshold AND a clean_solo_fn is provided, ARBITRATES with clean_solo
         (train each primitive from scratch) -- the authoritative readout that avoids the ablation
         false-negative for high-capacity co-adapted ops (validated: PNA on QM7). The clean-solo verdict
         wins conflicts. If no clean_solo_fn is given under high disagreement, returns ablation but flags
         'clean_solo_recommended'.

    clean_solo_fn(primitive) -> higher-is-better score (e.g. val accuracy / R^2), trains that primitive
    ALONE. Only called when disagreement is high, so its cost (one train per primitive) is paid only when
    needed. Returns dict: selected (per cell), method used, and the diagnostic.

    This is the single entrypoint the joint loop / meta-router should call for defensible selection."""
    diag = measure_coadaptation(server, eval_fn)
    argmax_sel = [c["argmax_alpha_op"] for c in diag["per_cell"]]
    ablation_sel = [c["solo_best_op"] for c in diag["per_cell"]]
    rate = diag["disagreement_rate"]
    if rate <= disagreement_threshold:
        return {"selected": ablation_sel, "method": "ablation", "disagreement_rate": rate,
                "argmax": argmax_sel, "diagnostic": diag}
    # high disagreement: co-adaptation suspected. Arbitrate with clean-solo if available.
    if clean_solo_fn is None:
        return {"selected": ablation_sel, "method": "ablation_flagged", "disagreement_rate": rate,
                "argmax": argmax_sel, "clean_solo_recommended": True, "diagnostic": diag,
                "note": "high co-adaptation; provide clean_solo_fn to arbitrate high-capacity ops."}
    # clean-solo is per-primitive (not per-cell); apply its verdict to every cell that disagrees.
    primitives = server.net.cells[0].primitives
    best, scores = clean_solo_select(clean_solo_fn, primitives)
    selected = []
    for c in diag["per_cell"]:
        # if this cell's ablation and argmax disagree, trust clean-solo; else keep ablation.
        selected.append(best if c["disagree"] else c["solo_best_op"])
    if verbose:
        print(f"robust_select: disagreement {rate:.2f} > {disagreement_threshold} -> clean-solo arbitration")
        print(f"  clean-solo scores: { {k: round(v,4) for k,v in scores.items()} }")
    return {"selected": selected, "method": "clean_solo_arbitrated", "disagreement_rate": rate,
            "argmax": argmax_sel, "ablation": ablation_sel, "clean_solo_best": best,
            "clean_solo_scores": scores, "diagnostic": diag}

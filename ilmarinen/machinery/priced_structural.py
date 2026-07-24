"""Fold KERNEL SIZE (conv receptive field) and ANGULAR ORDER (max-l) into the differentiable priced
objective, so they trade against accuracy on the same frontier as width and depth. Addresses the
audit follow-ups to B1/B4.

Analytical grounding. Hardware-/complexity-aware differentiable NAS optimizes
        J(alpha, w) = R(w, alpha)  +  mu * sum_i softmax(alpha)_i * cost_i
(Wu et al. FBNet; You-Only-Search-Once, arXiv:2208.14446), i.e. accuracy plus a mu-weighted EXPECTED
cost over the softmax mixture of candidate operations. ilmarinen already uses exactly this action
(run_penalized_selection: J = R + mu * sum softmax(alpha)*c). The only missing piece was cost models
that reflect the SPECIFIC structural resource rather than a lumped parameter count. This module
supplies analytically-grounded per-candidate costs:

  KERNEL SIZE k (conv):   cost ~ k^d  (d = spatial dims). A k-kernel conv has k^d weights per
      channel pair and k^d MACs per output element -- the receptive-field resource scales as the
      kernel VOLUME. So among {conv2d(k=3), conv2d_k5, conv2d_k7} the costs are 3^2:5^2:7^2 = 9:25:49
      (2D) -- the price of a larger receptive field is quadratic (d=2) / cubic (d=3) in k.

  ANGULAR ORDER l (equivariant):  cost ~ sum_{l'=0}^{l} (2l'+1) * C_{l'}  (the total dimension of the
      steerable feature through order l, times channels). Adding order l grows the per-node feature by
      (2l+1) components and the tensor-product message by the corresponding CG paths. For l<=0,1,2 the
      cumulative irrep dimension is 1, 1+3=4, 1+3+5=9 (per channel) -- so max-l is priced by the
      steerable-dimension it unlocks, quadratic-ish in l.

Both are the "measure the resource, price it, let mu select" pattern the project uses for width
(dual certificate) and depth (priced rule). Given costs and a fit routine, select_by_priced_rule() returns
the cheapest candidate whose accuracy is within tol of the best, i.e. the analytical
description-length-optimal receptive field / angular order.
"""

from __future__ import annotations

import numpy as np


def kernel_costs(kernel_sizes, ndim=2):
    """Analytical conv receptive-field cost ~ k^ndim (kernel volume = params & MACs per channel pair).
    Returns costs normalized to the smallest kernel."""
    k = np.asarray(kernel_sizes, dtype=np.float64)
    c = k**ndim
    return c / c.min()


def angular_order_costs(max_ls, channels_per_l=None):
    """Analytical max-l cost ~ cumulative steerable dimension sum_{l'<=l}(2l'+1)*C_l'.
    channels_per_l: optional per-order channel counts (default 1). Normalized to l=0."""
    costs = []
    for L in max_ls:
        dim = 0.0
        for lp in range(L + 1):
            c_lp = 1.0 if channels_per_l is None else channels_per_l.get(lp, 1.0)
            dim += (2 * lp + 1) * c_lp
        costs.append(dim)
    costs = np.asarray(costs, dtype=np.float64)
    return costs / costs.min()


def priced_objective(accuracies, costs, mu):
    """The priced action per candidate: J = -accuracy + mu*cost (lower is better). Accuracies and
    costs are arrays over the candidate set. Returns J and the argmin (selected candidate index)."""
    acc = np.asarray(accuracies, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64)
    J = -acc + mu * cost
    return J, int(np.argmin(J))


def priced_frontier(accuracies, costs, mus):
    """Sweep mu and return the selected candidate index at each price -- the description-length
    frontier. As mu rises, the selection moves from the most accurate to the cheapest candidate."""
    return {float(mu): priced_objective(accuracies, costs, mu)[1] for mu in mus}


def select_by_priced_rule(accuracies, costs, acc_tol=0.01):
    """Non-differentiable companion (accuracy-first compaction, like priced_depth): pick the CHEAPEST
    candidate whose accuracy is within acc_tol of the best. This is the mu->0+ limit of the priced
    objective restricted to a no-harm accuracy band -- the analytical minimal-resource choice."""
    acc = np.asarray(accuracies, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64)
    best = acc.max()
    ok = np.where(acc >= best - acc_tol)[0]
    return int(ok[np.argmin(cost[ok])])

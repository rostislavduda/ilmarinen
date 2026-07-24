"""Sparsity-priced alpha: charge the in-layer primitive mixture its description length so it keeps extra
primitives only when they pay for themselves. The deployment analogue of the width (effective #neurons)
and depth (#layers) charges, one more term in the same J = R + mu*Omega.

CONTEXT. The dense in-layer mixture cannot be DERIVED beyond DARTS (tests/mixture_derivation.md: for a
live net alpha and theta minimize the same non-convex loss, so the "derived" alpha is just mirror
descent = DARTS). But it CAN be PRICED. This keeps the project's minimal-architecture spirit -- one
deployed net, priced like every other rung -- unlike ensembling (a bag of separate models).

THE PRINCIPLED CHARGE. The MDL two-part code for a mixture over P primitives is ~ k*log P + k*(weight
bits), with k = #active primitives -- LINEAR in the support size ||alpha||_0. L0 is combinatorial; the
smooth surrogate is the INVERSE PARTICIPATION RATIO
    IPR(alpha) = 1 / sum_p alpha_p^2   = "effective number of active primitives"
(physics participation ratio; = 1 for one-hot, = P for uniform). Pricing it means rewarding
concentration, i.e. Omega(alpha) = -sum_p alpha_p^2 = -(collision/Renyi-2 mass), so that adding
mu*Omega to the loss drives IPR down. Verified (tests/sparsity_priced_alpha.md): IPR falls monotonically
in mu toward 1, useless primitives are suppressed at every mu, and the mu-sweep traces an
accuracy-vs-effective-#-primitives frontier with a knee where the mixture stops paying. Entropy is NOT
used here: H prices spread not count (and is the -1/beta soft-selection term); IPR literally counts
effective branches.

INTEGRATION. sparsity_price(alpha) returns Omega=-sum alpha^2 for adding to a live training loss; the
frontier builder + price_selection selectors (select_mu_by_elbow / select_by_tolerance /
select_mu_for_budget) choose mu by the SAME rules as width/depth. Torch-friendly (accepts a tensor and
returns a differentiable scalar) and numpy-friendly (for the frontier/analysis paths).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- core quantities
def participation(alpha):
    """Inverse participation ratio IPR = 1/sum_p alpha_p^2 = effective number of active primitives.
    1 for a single primitive (one-hot), P for a uniform mixture over P primitives. Accepts a numpy
    array or anything with .detach().cpu().numpy(); returns a float."""
    a = _to_np(alpha)
    return float(1.0 / (np.square(a).sum() + 1e-12))


def sparsity_omega(alpha):
    """The priced complexity Omega(alpha) = -sum_p alpha_p^2 (concentration reward). Adding mu*Omega to
    the loss drives the effective support size (IPR) down. Returns a float for numpy input; for a torch
    tensor use sparsity_price (differentiable)."""
    a = _to_np(alpha)
    return float(-np.square(a).sum())


def sparsity_price(alpha_tensor):
    """Differentiable priced term for a LIVE training loss: returns -sum_p alpha_p^2 as a torch scalar,
    to be added as `loss = task_loss + mu * sparsity_price(alpha)`. alpha_tensor is the softmax mixture
    weights (per cell); for a multi-cell net sum this over cells. Rewarding concentration (minimizing
    -sum a^2) pushes each cell toward few primitives, keeping a mixture only where it pays."""
    import torch  # local import so the module is usable without torch for the numpy paths
    a = alpha_tensor
    if not torch.is_tensor(a):
        a = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return -(a.square().sum())


def _to_np(alpha):
    if hasattr(alpha, "detach"):
        return alpha.detach().cpu().numpy().astype(float)
    return np.asarray(alpha, float)


# --------------------------------------------------------------------------- the mu-frontier
def sparsity_frontier(fit_at_mu, mus):
    """Build the accuracy-vs-effective-#-primitives frontier by fitting at each price mu.

    fit_at_mu(mu) -> dict with at least {'accuracy': float, 'alpha': array-or-tensor}  (the fit of the
        live net trained with loss = task_loss + mu * sparsity_price(alpha)). May also carry any extra
        keys (e.g. 'architecture'); they are preserved.
    mus : iterable of prices (ascending recommended).

    Returns a list of frontier entries, each augmented with:
        'omega'  = IPR(alpha)  (effective # primitives; the cost axis price_selection reads)
        'ipr'    = same, explicit name
        'mu'     = the price used
    so the standard selectors (select_mu_by_elbow, select_by_tolerance) apply verbatim -- 'omega' here
    is the effective primitive count, exactly as it is the neuron/layer count for width/depth.
    """
    frontier = []
    for mu in mus:
        e = dict(fit_at_mu(mu))
        ipr = participation(e["alpha"])
        e["omega"] = ipr
        e["ipr"] = ipr
        e["mu"] = float(mu)
        frontier.append(e)
    return frontier


def select_sparsity_by_elbow(fit_at_mu, mus):
    """Convenience: build the sparsity frontier and pick its knee via price_selection.select_mu_by_elbow
    -- the effective-#-primitives at which extra branches stop paying (the deployment analogue of the
    width/depth elbow). Returns (chosen_entry, frontier)."""
    from .price_selection import select_mu_by_elbow
    frontier = sparsity_frontier(fit_at_mu, mus)
    return select_mu_by_elbow(frontier), frontier


def effective_num_primitives(alpha, thresh=None):
    """A reportable integer count of active primitives. If thresh is given, counts weights > thresh;
    otherwise rounds the inverse participation ratio. Useful for logging the deployed mixture size."""
    a = _to_np(alpha)
    if thresh is not None:
        return int((a > thresh).sum())
    return int(round(participation(a)))

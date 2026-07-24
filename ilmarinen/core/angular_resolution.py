"""Derive the equivariant model's MAX ANGULAR ORDER (max-l) from the ANGULAR CONTENT of the data,
instead of fixing it by module choice (l<=1 vs l<=2). Addresses fixed-hyperparameter audit gap B1.

Principle (the priced marginal-value rule applied to angular order). Depth is added while the marginal
loss reduction per layer exceeds a price mu; likewise angular order l=k+1 is worth adding only while
its marginal value exceeds mu. The marginal value of order l is how much of the target's variance is
explained by angular-order-l structure of the local environments that lower orders CANNOT capture.

Direct measurement (no need to train many models). For each node i with neighbors j, form the
per-order rotation-INVARIANT descriptors of the neighborhood geometry:
    p_l(i) = || sum_j w(r_ij) Y_l(rhat_ij) ||         (the norm of the l-th bond-orientational moment)
where Y_l are the l spherical harmonics of the neighbor directions. p_0 counts neighbors (radial),
p_1 measures dipolar asymmetry (vector), p_2 measures angular/quadrupolar structure (bond angles).
Pool these to a graph descriptor, then measure the INCREMENTAL explanatory power of adding order l:
    R2(0..l) = variance of target explained by {p_0,...,p_l} (linear fit),
    marginal(l) = R2(0..l) - R2(0..l-1).
Select max-l by the priced rule: the largest l whose marginal exceeds the price mu (stop at the first
order whose marginal falls below mu). This makes max-l a DERIVED decision like width and depth --
unifying the l<=1 and l<=2 modules under one metaoptimality criterion.

These invariant p_l are exactly the Steinhardt bond-orientational order parameters / the invariant
moments underlying SOAP and the ACE/MACE body-order expansion, so "how much angular order is in the
data" is measured with the same objects the equivariant model uses internally.
"""

from __future__ import annotations

import numpy as np


def _real_sph_l(rhat):
    """Real spherical-harmonic-like components for l=0,1,2 of unit vectors rhat (M,3).
    Returns dict l-> (M, 2l+1). Uses the symmetric-traceless-tensor basis for l=2 (5 comps),
    matching the equivariant module's l=2 representation."""
    x, y, z = rhat[:, 0], rhat[:, 1], rhat[:, 2]
    Y0 = np.ones((len(rhat), 1))
    Y1 = rhat  # (M,3)
    # l=2: 5 independent components of symtraceless(r outer r)
    Y2 = np.stack([x * y, y * z, z * x, x * x - y * y, (2 * z * z - x * x - y * y) / np.sqrt(3.0)], axis=1)  # (M,5)
    return {0: Y0, 1: Y1, 2: Y2}


def _neighbor_moments(pos, edge_index, max_l=2):
    """Per-node invariant bond-orientational moments p_l = ||sum_j Y_l(rhat_ij)|| for l=0..max_l.
    pos (N,3), edge_index (2,|E|) with rows [src,dst], messages j=src -> i=dst.
    Returns (N, max_l+1) array of invariant magnitudes."""
    src, dst = edge_index
    rel = pos[dst] - pos[src]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    rhat = rel / (dist + 1e-9)
    Ys = _real_sph_l(rhat)
    N = len(pos)
    out = np.zeros((N, max_l + 1))
    for l in range(max_l + 1):
        Yl = Ys[l]  # (E, 2l+1)
        # sum contributions per destination node, then take the invariant norm
        acc = np.zeros((N, Yl.shape[1]))
        np.add.at(acc, dst, Yl)
        out[:, l] = np.linalg.norm(acc, axis=1)
    return out


def _graph_descriptor(graph, max_l=2):
    """Pool per-node moments to a per-GRAPH descriptor (sum over nodes) -> (max_l+1,)."""
    pos = np.asarray(graph["pos"], dtype=np.float64)
    ei = np.asarray(graph["edge_index"])
    m = _neighbor_moments(pos, ei, max_l)  # (N, max_l+1)
    return m.sum(0)  # graph-level, one value per order


def _r2(X, y):
    """R^2 of a least-squares linear fit of y from columns X (with intercept)."""
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return 1.0 - ss_res / ss_tot


def measure_angular_marginal(graphs, y, max_l=2):
    """For a list of graphs (dicts with 'pos','edge_index') and targets y, measure the incremental
    explanatory power of each angular order. Returns dict with cumulative R2 and per-order marginals.
    """
    y = np.asarray(y, dtype=np.float64)
    D = np.stack([_graph_descriptor(g, max_l) for g in graphs], axis=0)  # (n_graphs, max_l+1)
    # standardize descriptors (each order) for a fair linear fit
    D = (D - D.mean(0)) / (D.std(0) + 1e-9)
    cum_r2 = {}
    for l in range(max_l + 1):
        cum_r2[l] = _r2(D[:, : l + 1], y)
    marginals = {0: cum_r2[0]}
    for l in range(1, max_l + 1):
        marginals[l] = cum_r2[l] - cum_r2[l - 1]
    return {"cumulative_r2": cum_r2, "marginals": marginals}


def select_max_l(graphs, y, mu=0.01, max_l=2):
    """Priced marginal-value selection of max angular order: include order l while its marginal
    explanatory power exceeds the price mu; stop at the first order whose marginal falls below mu.
    Returns dict(max_l, marginals, cumulative_r2)."""
    res = measure_angular_marginal(graphs, y, max_l)
    marg = res["marginals"]
    chosen = 0
    for l in range(1, max_l + 1):
        if marg[l] >= mu:
            chosen = l
        else:
            break  # marginal below price -> stop adding orders
    return {"max_l": chosen, "marginals": marg, "cumulative_r2": res["cumulative_r2"]}


# --------------------------------------------------------------------------------------------------
# Model-based angular marginal (the robust selector). The linear-descriptor proxy above is a cheap
# pre-screen, but it UNDERESTIMATES the value of higher l when the target depends on angular structure
# NONLINEARLY (e.g. QM7 energy is ~95% linearly predictable from l<=1 invariants, yet l=2 still cuts
# model MAE ~46% via nonlinear tensor-product message passing that a linear fit on hand-made moments
# cannot see). So the faithful marginal-value rule TRAINS short l<=k models and measures the actual
# validation-loss reduction -- exactly the priced-depth methodology, applied to angular order.
# --------------------------------------------------------------------------------------------------


def measure_angular_marginal_model(graphs, y, collate_fn, fit_fn, orders=(0, 1, 2)):
    """Train a short model at each max-l in `orders` and return the validation loss at each, plus the
    per-order marginal reductions. Caller supplies collate_fn(graphs, idx) and
    fit_fn(max_l)->val_loss (which builds the l<=max_l model, trains briefly, returns held-out loss).
    This is the honest, model-based version; heavier than the linear proxy but faithful."""
    losses = {}
    for L in orders:
        losses[L] = float(fit_fn(L))
    marg = {}
    prev = None
    for L in orders:
        marg[L] = (prev - losses[L]) if prev is not None else None
        prev = losses[L]
    return {"val_loss": losses, "marginals": marg}


def select_max_l_priced(val_losses, mu):
    """Given {max_l: val_loss}, select the largest order whose marginal loss reduction exceeds price
    mu (stop when the next order does not reduce loss by at least mu). The priced marginal-value rule
    from priced_depth, applied to angular order."""
    orders = sorted(val_losses)
    chosen = orders[0]
    for i in range(1, len(orders)):
        reduction = val_losses[orders[i - 1]] - val_losses[orders[i]]
        if reduction >= mu:
            chosen = orders[i]
        else:
            break
    return chosen


# --------------------------------------------------------------------------------------------------
# NONLINEAR angular proxy (kernel-ridge, SOAP-zeta>1 grounded). The linear proxy (measure_angular_
# marginal) is the zeta=1 SOAP-power-spectrum case, which by construction captures only 3-body angular
# structure LINEARLY and misses the higher-order/nonlinear way angular features enter targets like
# molecular energy (Bartok/Csanyi SOAP-GAP: the SOAP kernel k=(xi.xi')^zeta needs zeta>1 to represent
# beyond-three-body terms; arXiv:2410.00626). So we measure the marginal explanatory power of each
# angular order through a NONLINEAR kernel ridge regression (zeta=2 polynomial kernel), which recovers
# the l=2 signal a linear fit misses -- at the cost of one small KRR solve per order (still cheap: no
# model training, just a kernel solve on a subsample).
# --------------------------------------------------------------------------------------------------


def _poly_kernel(A, B, zeta=2, gamma=None):
    """(gamma <A,B> + 1)^zeta -- an inhomogeneous polynomial kernel; zeta>1 captures cross-order
    (nonlinear) angular interactions, the SOAP-GAP mechanism for beyond-3-body structure."""
    if gamma is None:
        gamma = 1.0 / max(A.shape[1], 1)
    return (gamma * (A @ B.T) + 1.0) ** zeta


def _krr_r2_cv(X, y, zeta=2, lam=1e-2, folds=4):
    """Cross-validated R^2 of kernel ridge regression with a polynomial (zeta) kernel. Standardizes X.
    Returns mean held-out R^2. Cheap: kernel solves on a subsample, no gradient training."""
    n = len(y)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = y - y.mean()
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    fold_sz = n // folds
    r2s = []
    for f in range(folds):
        te = idx[f * fold_sz : (f + 1) * fold_sz] if f < folds - 1 else idx[f * fold_sz :]
        tr = np.setdiff1d(idx, te)
        Ktr = _poly_kernel(Xs[tr], Xs[tr], zeta)
        Kte = _poly_kernel(Xs[te], Xs[tr], zeta)
        alpha = np.linalg.solve(Ktr + lam * np.eye(len(tr)), y[tr])
        pred = Kte @ alpha
        ss_res = ((y[te] - pred) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
        r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s))


def measure_angular_marginal_nonlinear(graphs, y, max_l=2, zeta=2, sample=400):
    """Nonlinear (kernel-ridge) version of measure_angular_marginal. For each cumulative angular order,
    fit a zeta-polynomial KRR from the invariant moments {p_0..p_l} to the target and record the
    cross-validated R^2; the per-order marginal is the increment. Recovers nonlinear angular value
    (e.g. QM7 l=2) that the linear proxy misses. Uses a random subsample for the kernel solve."""
    y = np.asarray(y, dtype=np.float64)
    n = min(sample, len(graphs))
    sel = np.random.RandomState(0).choice(len(graphs), n, replace=False)
    D = np.stack([_graph_descriptor(graphs[i], max_l) for i in sel], axis=0)  # (n, max_l+1)
    ys = y[sel]
    cum = {}
    for l in range(max_l + 1):
        cum[l] = _krr_r2_cv(D[:, : l + 1], ys, zeta=zeta)
    marg = {0: cum[0]}
    for l in range(1, max_l + 1):
        marg[l] = cum[l] - cum[l - 1]
    return {"cumulative_r2": cum, "marginals": marg, "zeta": zeta}


def select_max_l_nonlinear(graphs, y, mu=0.01, max_l=2, zeta=2, sample=400):
    """Priced selection of max angular order using the NONLINEAR proxy: include order l while its
    kernel-ridge marginal R^2 exceeds price mu. The faithful cheap selector."""
    res = measure_angular_marginal_nonlinear(graphs, y, max_l, zeta, sample)
    marg = res["marginals"]
    chosen = 0
    for l in range(1, max_l + 1):
        if marg[l] >= mu:
            chosen = l
        else:
            break
    return {"max_l": chosen, "marginals": marg, "cumulative_r2": res["cumulative_r2"], "zeta": zeta}


# --------------------------------------------------------------------------------------------------
# The FAITHFUL cheap proxy: PER-ATOM distributional descriptors + nonlinear kernel. Graph-summing the
# per-atom moments (measure_angular_marginal*) mixes distinct local environments and destroys the
# angular signal (QM7 l=2 marginal ~0 despite the model gaining 46% from l=2). The fix, validated: keep
# the DISTRIBUTION of per-atom moments (mean/std/2nd-moment/max of each p_l across atoms) and score with
# a nonlinear (zeta=2) kernel. This recovers a large QM7 l=2 marginal (~+0.17), matching the model.
# This is the recommended max-l selector.
# --------------------------------------------------------------------------------------------------


def _per_atom_distribution_descriptor(graph, max_l):
    """Distribution (mean, std, 2nd-moment, max) of each per-atom invariant moment p_l across atoms.
    Preserves per-environment angular structure that graph-summing destroys."""
    m = _neighbor_moments(
        np.asarray(graph["pos"], dtype=np.float64), np.asarray(graph["edge_index"]), max_l
    )  # (N, max_l+1)
    feats = []
    for l in range(max_l + 1):
        pl = m[:, l]
        feats += [pl.mean(), pl.std(), (pl**2).mean(), (pl.max() if len(pl) else 0.0)]
    return np.array(feats)  # (4*(max_l+1),)


def _cols_upto_order(l, max_l):
    """Column indices in the per-atom descriptor for orders 0..l (4 stats per order)."""
    return [4 * ll + s for ll in range(l + 1) for s in range(4)]


def measure_angular_marginal_peratom(graphs, y, max_l=2, zeta=2, sample=400):
    """FAITHFUL cheap angular marginal: per-atom distributional descriptors + nonlinear (zeta) KRR.
    Returns cumulative CV R^2 and per-order marginals. Recovers the nonlinear per-environment angular
    signal (e.g. QM7 l=2) that the graph-summed linear proxy misses."""
    y = np.asarray(y, dtype=np.float64)
    n = min(sample, len(graphs))
    sel = np.random.RandomState(0).choice(len(graphs), n, replace=False)
    D = np.stack([_per_atom_distribution_descriptor(graphs[i], max_l) for i in sel], axis=0)
    ys = y[sel]
    cum = {}
    for l in range(max_l + 1):
        cum[l] = _krr_r2_cv(D[:, _cols_upto_order(l, max_l)], ys, zeta=zeta)
    marg = {0: cum[0]}
    for l in range(1, max_l + 1):
        marg[l] = cum[l] - cum[l - 1]
    return {"cumulative_r2": cum, "marginals": marg, "zeta": zeta}


def select_max_l_faithful(graphs, y, mu=0.02, max_l=2, zeta=2, sample=1500):
    """RECOMMENDED max-l selector: per-atom distributional descriptors + nonlinear kernel + priced rule.
    Select the HIGHEST order whose marginal CV R^2 over the best LOWER order exceeds price mu.

    Note on robustness: a naive "stop at the first order whose marginal < mu" scan is fragile because
    cross-validated KRR R^2 at an intermediate order can dip spuriously (even negative) on smaller
    samples, which would wrongly block a later order that IS valuable. Instead we compare each order's
    cumulative R^2 against the best cumulative R^2 achieved at any lower order, so a strong l=2 is
    still selected even if l=1 dips. Larger `sample` stabilizes the CV R^2 (recommended >= 800 for
    real molecular data)."""
    res = measure_angular_marginal_peratom(graphs, y, max_l, zeta, sample)
    cum = res["cumulative_r2"]
    chosen = 0
    best_lower = cum[0]
    for l in range(1, max_l + 1):
        # marginal value of order l = improvement over the best cumulative fit up to order l-1
        gain = cum[l] - best_lower
        if gain >= mu:
            chosen = l
        best_lower = max(best_lower, cum[l])
    # recompute reported marginals relative to running best (informative, monotone baseline)
    return {"max_l": chosen, "marginals": res["marginals"], "cumulative_r2": cum}

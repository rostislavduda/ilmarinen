"""Tier-2 interpretability: in-model feature attribution via a priced feature gate.

Tier 1 reports WHICH PRIMITIVE each layer uses. Tier 2 reports WHICH INPUT FEATURES the model uses, by
inserting a per-feature multiplicative gate g in [0,1]^d at the input and pricing it with the SAME sparsity
price the framework already uses one level up (on the primitive simplex). Because the gate is literally part
of the forward pass and is trained jointly with the model, the attribution is FAITHFUL BY CONSTRUCTION (the
Section-2 argument of the interpretability foundation): it is not a post-hoc saliency estimate that could
diverge from the model -- the reported features are exactly the coordinates the model is allowed to see.

The foundation specifies that this gate lives on the INVARIANT COORDINATES (after canonicalization / symmetry
reduction), so the reported features are the physically meaningful invariants, not raw coordinates. This
module provides the gate and the fit/report machinery; the caller supplies features already in the desired
(e.g. invariant) basis.

Design (validated in tests/interpretability_tier2.md):
  * g = sigmoid(gate_logit), initialized open; a sparsity price mu * ||g||_1 drives unused gates toward 0.
  * the surviving gates (g > keep_thresh) are the attributed feature set; the gate magnitudes rank them.
  * a small sweep over mu traces the feature-selection path (which features survive as the price rises),
    exactly analogous to the priced width/primitive sweeps elsewhere in the framework.
Nothing here is post-hoc: no separate explainer model is fit.
"""

import numpy as np
import torch
import torch.nn as nn


class FeatureGate(nn.Module):
    """A per-feature stochastic gate g in [0,1]^d applied at the input: x -> x * g, using the HARD-CONCRETE
    (L0) relaxation (Louizos et al. 2018). Unlike an L1-on-sigmoid gate -- which pulls every gate down
    together so a kept feature cannot stay near 1 while noise goes to 0 -- the hard-concrete gate genuinely
    SATURATES: it stretches a concrete distribution past [0,1] and hard-clamps, so at convergence kept
    features sit near 1 and dropped features near 0. The L0 price is the expected number of OPEN gates, the
    feature-level analog of the framework's IPR/sparsity price on the primitive simplex. Part of the forward
    pass and trained jointly, so the attribution is faithful."""

    def __init__(self, n_features, init_open=True, beta=0.66, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.beta, self.gamma, self.zeta = beta, gamma, zeta  # hard-concrete stretch parameters
        init = 2.0 if init_open else 0.0  # start mostly OPEN
        self.log_alpha = nn.Parameter(torch.full((n_features,), float(init)))

    def _sample_z(self):
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma  # stretch beyond [0,1]
        return s_bar.clamp(0.0, 1.0)  # hard-clamp -> genuine 0/1 saturation

    def gate(self):
        # deterministic gate value (for reporting / eval): clamped stretched sigmoid at the mode.
        s = torch.sigmoid(self.log_alpha / self.beta)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def forward(self, x):
        return x * self._sample_z()

    def _p_open(self):
        # probability a gate is non-zero = 1 - CDF_stretched(0): the L0 expected-open summand.
        return torch.sigmoid(self.log_alpha - self.beta * float(np.log(-self.gamma / self.zeta)))

    def price(self, mu):
        """L0 price to ADD to the loss: mu * (expected number of open gates). Drives the expected feature-
        support size down, so at convergence only needed features stay open (saturated near 1)."""
        return mu * self._p_open().sum()

    def attribution(self, feature_names=None, keep_thresh=0.5):
        """The faithful feature-attribution report read from the trained gate.
        Returns a dict: gates (per feature), active set (g > keep_thresh), ranking (by gate magnitude),
        and the effective number of features (an IPR on the normalized gate vector)."""
        g = self.gate().detach().cpu().numpy()
        d = len(g)
        names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(d)]
        order = np.argsort(g)[::-1]
        # effective number of features: IPR on the gate vector, but computed on g^2 (energy) so that a few
        # dominant gates are not swamped by many small-but-nonzero noise gates. This tracks the ACTIVE-set
        # size closely when the gate is well-separated, and is the honest 'how many features carry weight'.
        g2 = g**2
        s2 = g2.sum()
        gnorm = g2 / s2 if s2 > 0 else g2
        eff = float(1.0 / np.sum(gnorm**2)) if np.sum(gnorm**2) > 0 else float(d)
        return {
            "gates": {names[i]: float(g[i]) for i in range(d)},
            "active": [names[i] for i in order if g[i] > keep_thresh],
            "ranking": [names[i] for i in order],
            "effective_num_features": eff,
            "n_features": d,
        }


class GatedMLP(nn.Module):
    """A minimal gated predictor used by fit_feature_attribution: FeatureGate -> small MLP. The gate is the
    attribution device; the MLP is a generic head so the gate has something to be pruned against. This is a
    STANDALONE attribution probe over a fixed feature basis (e.g. discovered invariants), not a replacement
    for the metaoptimized schema -- it answers 'which of these features carry the signal', faithfully,
    because the gate it reports is the gate the head sees."""

    def __init__(self, n_features, hidden=32, n_out=1):
        super().__init__()
        self.fgate = FeatureGate(n_features)
        self.head = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.n_out = n_out

    def forward(self, x):
        h = self.head(self.fgate(x))
        return h.squeeze(-1) if self.n_out == 1 else h


def fit_feature_attribution(
    X,
    y,
    task="regression",
    mu=0.03,
    feature_names=None,
    hidden=32,
    epochs=400,
    lr=5e-3,
    keep_thresh=0.5,
    seed=0,
    val_frac=0.25,
):
    """Fit a priced feature gate over the features X to attribute which ones the signal needs.

    X : (n, d) feature matrix (ideally already in the invariant basis).
    y : (n,) targets. task in {regression, classification}.
    mu : the feature-sparsity price; larger -> fewer surviving features. (Sweep it with
         feature_selection_path for the full picture.)
    Returns a dict: the attribution (active set, ranking, gates, effective count), the held-out fit, and mu.
    Faithful: the reported gate is exactly the gate the head was trained through.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n, d = X.shape
    # COLLINEARITY GUARD: the L0 feature gate has no unique sparse solution when features are strongly
    # correlated (redundant), so the attribution becomes unreliable (validated: on correlated features it
    # fails to isolate the truly-relevant ones, and the ranking can even be worse than random). Measure the
    # mean absolute off-diagonal correlation; flag when high so the caller does not trust a misleading pick.
    collinearity = 0.0
    if d > 1 and n > 2:
        Xc = X - X.mean(0)
        sd = X.std(0) + 1e-8
        C = (Xc / sd).T @ (Xc / sd) / n
        off = C - np.eye(d)
        collinearity = float(np.abs(off).sum() / (d * (d - 1)))
    idx = rng.permutation(n)
    ntr = int(n * (1 - val_frac))
    tr, va = idx[:ntr], idx[ntr:]
    Xt, Xv = torch.tensor(X[tr]), torch.tensor(X[va])
    n_out = 1 if task == "regression" else int(np.max(y) + 1)
    if task == "regression":
        yt = torch.tensor(y[tr].astype(np.float32))
        yv = torch.tensor(y[va].astype(np.float32))
        lossf = lambda p, t: ((p - t) ** 2).mean()
    else:
        yt = torch.tensor(y[tr].astype(np.int64))
        yv = torch.tensor(y[va].astype(np.int64))
        lossf = lambda p, t: nn.functional.cross_entropy(p, t)

    m = GatedMLP(d, hidden=hidden, n_out=n_out)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()  # stochastic hard-concrete gate active during fitting
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(m(Xt), yt) + m.fgate.price(mu)
        loss.backward()
        opt.step()

    m.eval()  # deterministic gate for scoring + attribution
    with torch.no_grad():
        if task == "regression":
            pv = m(Xv)
            denom = ((yv - yv.mean()) ** 2).sum().item()
            fit = 1 - ((pv - yv) ** 2).sum().item() / (denom + 1e-12)
            metric = "R2"
        else:
            fit = float((m(Xv).argmax(1) == yv).float().mean().item())
            metric = "acc"
    attr = m.fgate.attribution(feature_names=feature_names, keep_thresh=keep_thresh)
    # RELIABILITY: two independent checks. (1) collinearity -- a correlated basis has no unique sparse
    # solution. (2) saturation -- if the L0 gates did not decisively open/close (few gates near 0 or 1, most
    # stuck mid-range), the gate failed to concentrate and the attribution is not trustworthy. This second
    # check catches locally-correlated bases (e.g. adjacent time-series timesteps) that a global mean
    # correlation under-measures.
    gv = np.array(list(attr["gates"].values()))
    frac_saturated = float(np.mean((gv < 0.1) | (gv > 0.9)))  # fraction decisively closed or open
    reliable = (collinearity < 0.35) and (frac_saturated > 0.5)
    return {
        "attribution": attr,
        "metric": metric,
        "value": float(fit),
        "mu": mu,
        "collinearity": collinearity,
        "frac_saturated": frac_saturated,
        "reliable": bool(reliable),
    }


def feature_selection_path(X, y, task="regression", mus=(0.0, 0.05, 0.1, 0.2, 0.4), feature_names=None, **kwargs):
    """Trace which features survive as the sparsity price mu rises -- the feature-level analog of the priced
    width/primitive sweeps. Returns a list of (mu, active_set, effective_num_features, fit) rows. The stable
    active set across a range of mu is the robust attributed feature set."""
    rows = []
    for mu in mus:
        r = fit_feature_attribution(X, y, task=task, mu=mu, feature_names=feature_names, **kwargs)
        rows.append(
            {
                "mu": mu,
                "active": r["attribution"]["active"],
                "effective_num_features": r["attribution"]["effective_num_features"],
                "value": r["value"],
                "metric": r["metric"],
            }
        )
    return rows


def format_attribution(result):
    """Render a fit_feature_attribution result as a short text block."""
    a = result["attribution"]
    L = [
        "FEATURE ATTRIBUTION (Tier-2: priced in-model gate, faithful by construction)",
        f"  fit {result['metric']}={result['value']:.3f}   price mu={result['mu']}   "
        f"effective #features={a['effective_num_features']:.2f} of {a['n_features']}",
        f"  active (g>0.5): {a['active'] if a['active'] else '(none survived)'}",
        "  gates (top): " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(a["gates"].items(), key=lambda kv: -kv[1])[:6]),
    ]
    if not result.get("reliable", True):
        reasons = []
        if result.get("collinearity", 0) >= 0.35:
            reasons.append(f"mean|corr|={result['collinearity']:.2f} (collinear/redundant features)")
        if result.get("frac_saturated", 1.0) <= 0.5:
            reasons.append(
                f"only {result.get('frac_saturated', 0) * 100:.0f}% of gates saturated (the gate did not concentrate)"
            )
        L.append(
            "  ** UNRELIABLE: " + "; ".join(reasons) + " -- no unique sparse attribution exists here; "
            "treat the active set and ranking with caution."
        )
    return "\n".join(L)

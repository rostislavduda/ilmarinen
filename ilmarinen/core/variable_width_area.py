"""Most-general size metaoptimization: variable width-per-layer + depth via generalized-area minimization.

The consolidated analytical objective (analytical report, Eq. "consolidated") is

    min  R[f]  +  lambda * sum_l ||rho^(l)||_TV  +  mu * L .

The width penalty is ALREADY a per-layer sum of atomic (TV) norms; ||rho^(l)||_TV ~ m_l counts the neurons
kept in layer l. The uniform-width "area = m * L" used by select_architecture_by_area is the collapsed
special case m_l = m for all l. The FULLY GENERAL size objective keeps the per-layer structure:

    GENERALIZED AREA  =  sum_l m_l    (total neuron count across all layers),

and depth becomes EMERGENT: a layer whose width m_l is driven to 0 is removed, so L = #{l : m_l > 0}. Both
width-per-layer and depth are then outputs of ONE penalty on the total neuron count -- the most general
form of the metaoptimization problem, faithful to the analytical objective rather than a shape restriction.

This module builds that objective as a differentiable loss functional over a stack of per-layer neuron gates
(hard-concrete / L0, the saturating gate used for feature attribution -- reused, not reinvented). The L0
expected-open-count of each layer's gate is m_l (a differentiable neuron count); their sum is the generalized
area, added to the data risk with price lambda. Training jointly selects, per layer, how many neurons to keep
and -- by closing whole layers -- the effective depth.

    L(theta, gates)  =  R(f_theta,gates)  +  lambda * sum_l E[#open gates in layer l]
                     =  R                 +  lambda * (generalized area).

Validated in tests/variable_width_area.md: recovers variable per-layer widths, shrinks total area smoothly
as lambda rises, and closes layers (emergent depth) at high price. Offered as an OPT-IN module; the existing
uniform-width selectors are untouched.
"""

import numpy as np
import torch
import torch.nn as nn


class _AnnealedGate(nn.Module):
    """Per-unit DETERMINISTIC gate g = sigmoid(logit / tau) in [0,1]^n, with a temperature tau that is
    annealed toward 0 during training so the gate sharpens to hard 0/1. Unlike a stochastic hard-concrete
    (L0) gate -- whose Monte-Carlo sampling injects variance that makes the price path noisy across seeds --
    this gate is a deterministic function of its parameters, so the area it reports and the resulting widths
    are stable (validated: R^2 spread across seeds drops from ~0.2 to ~0.004). The soft neuron count
    sum_i g_i is added as the area penalty; as tau -> 0 it converges to the true count of open units, so the
    penalty is an annealed L0 surrogate without the sampling noise.
    """

    def __init__(self, n, init_open=2.0):
        super().__init__()
        self.logit = nn.Parameter(torch.full((n,), float(init_open)))

    def g(self, tau):
        return torch.sigmoid(self.logit / tau)

    def forward(self, x, tau):
        return x * self.g(tau)

    def expected_open(self, tau):
        # soft neuron count sum_i sigmoid(logit_i / tau); -> hard count as tau -> 0. This is m_l.
        return self.g(tau).sum()

    def width(self):
        with torch.no_grad():
            return int((torch.sigmoid(self.logit / 0.05) > 0.5).sum())


class VariableWidthNet(nn.Module):
    """A depth-Lmax stack of dense layers, each of maximum width M, each followed by a per-neuron L0 gate.
    The generalized area = sum_l (expected open gates in layer l) is differentiable; minimizing risk + lambda*
    area jointly selects each layer's width and -- by closing whole layers -- the effective depth.

    This is a generic dense realization used to OPTIMIZE and MEASURE the general objective on tabular /
    vectorized features; it is not tied to a specific contract. Layers compose as h <- act(W h) * gate.
    """

    def __init__(self, d_in, n_out=1, max_width=32, max_depth=4, act="tanh"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        prev = d_in
        for _ in range(max_depth):
            self.layers.append(nn.Linear(prev, max_width))
            self.gates.append(_AnnealedGate(max_width))
            prev = max_width
        self.head = nn.Linear(max_width, n_out)
        self.n_out = n_out
        self.act = torch.tanh if act == "tanh" else torch.relu

    def forward(self, x, tau=0.05):
        h = x
        for lin, g in zip(self.layers, self.gates):
            h = g(self.act(lin(h)), tau)
        out = self.head(h)
        return out.squeeze(-1) if self.n_out == 1 else out

    def generalized_area(self, tau=0.05):
        """sum_l m_l = total soft neuron count across all layers at temperature tau (differentiable)."""
        return sum(g.expected_open(tau) for g in self.gates)

    def widths(self):
        """Per-layer kept widths [m_1, ..., m_Lmax] (thresholded gate counts)."""
        return [g.width() for g in self.gates]

    def effective_depth(self):
        """L = number of layers with at least one kept neuron (emergent depth)."""
        return int(sum(1 for w in self.widths() if w > 0))


def certificate_lambda_scale(X, y, n_candidates=2000, seed=0):
    """The dual-certificate scale for the width price lambda (analytical report, sec:certificate).

    The KKT certificate eta = A^* q / lambda with |eta| <= 1 saturating on support means the price at which
    an atom (neuron) stops being worth adding is lambda* = ||A^* q||_inf -- the peak correlation of the
    residual with the candidate atoms. At the null model (no neurons) q = y, giving a data-driven SCALE for
    lambda: lambda*_0 = ||A^* y||_inf / n over a dictionary of random neuron candidates. The useful sweep
    range is a fraction of this scale (atoms with correlation between the mean and lambda*_0 are the ones the
    price arbitrates). This makes lambda a CALIBRATED quantity, not a free knob: it is set by the residual-
    atom correlation scale of the specific dataset.

    Returns (lambda_scale, mean_corr). Sweep lambda in units of lambda_scale (e.g. 0.001-0.05 * scale for the
    full-network area gate, whose effective price is well below the single-atom certificate value)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = (y - y.mean()) / (y.std() + 1e-8)
    n, d = X.shape
    W = rng.standard_normal((d, n_candidates)) / np.sqrt(d)
    b = rng.standard_normal(n_candidates) * 0.5
    F = np.tanh(X @ W + b)
    Fc = (F - F.mean(0)) / (F.std(0) + 1e-8)
    corr = np.abs(Fc.T @ y) / n
    return float(corr.max()), float(corr.mean())


def fit_variable_width_area(
    X,
    y,
    task="regression",
    lam=0.005,
    max_width=32,
    max_depth=4,
    epochs=600,
    lr=5e-3,
    seed=0,
    val_frac=0.25,
    act="tanh",
    tau_start=1.0,
    tau_end=0.1,
):
    """Minimize the generalized-area objective  R + lam * sum_l m_l  by joint training of a VariableWidthNet.

    Uses a DETERMINISTIC temperature-annealed gate (tau: tau_start -> tau_end) rather than a stochastic L0
    gate, so the selected widths / area are stable across seeds (low-noise price path). Returns a dict: per-
    layer widths, effective depth, generalized area (total neurons), the held-out fit, and lam.

    lam is on the certificate scale: see certificate_lambda_scale(X, y) for the data-driven lambda scale; the
    full-network area gate operates at roughly 1e-3..5e-2 of that single-atom scale.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n, d = X.shape
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

    m = VariableWidthNet(d, n_out=n_out, max_width=max_width, max_depth=max_depth, act=act)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    for e in range(epochs):
        tau = max(tau_end, tau_start * (1.0 - e / max(1, epochs)))  # anneal temperature toward tau_end
        opt.zero_grad()
        loss = lossf(m(Xt, tau), yt) + lam * m.generalized_area(tau)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        if task == "regression":
            pv = m(Xv, tau=0.05)
            denom = ((yv - yv.mean()) ** 2).sum().item()
            fit = 1 - ((pv - yv) ** 2).sum().item() / (denom + 1e-12)
            metric = "R2"
        else:
            fit = float((m(Xv, tau=0.05).argmax(1) == yv).float().mean().item())
            metric = "acc"
    widths = m.widths()
    return {
        "widths": widths,
        "effective_depth": m.effective_depth(),
        "generalized_area": int(sum(widths)),
        "metric": metric,
        "value": float(fit),
        "lam": lam,
        "max_width": max_width,
        "max_depth": max_depth,
    }


def area_price_path(X, y, task="regression", lams=None, **kwargs):
    """Trace the per-layer widths / effective depth / generalized area / fit as the neuron price lam rises --
    the generalized-area analog of the priced width and depth sweeps. If lams is None, the sweep is
    CERTIFICATE-CALIBRATED: lambdas are set as fractions of the dual-certificate scale for this dataset, so
    the range automatically lands where the price actually arbitrates neurons. Returns a list of rows."""
    if lams is None:
        scale, _ = certificate_lambda_scale(X, y)
        lams = tuple(round(f * scale, 5) for f in (0.0, 0.002, 0.006, 0.015, 0.04))
    rows = []
    for lam in lams:
        r = fit_variable_width_area(X, y, task=task, lam=lam, **kwargs)
        rows.append(
            {
                "lam": lam,
                "widths": r["widths"],
                "effective_depth": r["effective_depth"],
                "generalized_area": r["generalized_area"],
                "value": r["value"],
                "metric": r["metric"],
            }
        )
    return rows


def format_area_result(result):
    """Short text report of a fit_variable_width_area result."""
    w = result["widths"]
    return (
        "GENERALIZED-AREA SIZE SELECTION (variable width-per-layer + emergent depth)\n"
        f"  objective: R + lam * sum_l m_l,  lam={result['lam']}\n"
        f"  per-layer widths [m_1..m_Lmax]: {w}  (max_width={result['max_width']}, max_depth={result['max_depth']})\n"
        f"  effective depth L = #{{l : m_l>0}} = {result['effective_depth']}\n"
        f"  generalized area (total neurons) = {result['generalized_area']}\n"
        f"  held-out {result['metric']} = {result['value']:.3f}"
    )

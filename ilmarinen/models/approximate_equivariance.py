"""Approximate equivariance as a priced quantity (direction B5).

Real data is often only APPROXIMATELY symmetric, yet the package's geometric contract is a hard choice:
strictly equivariant, or not. This module lets the amount of equivariance be a SELECTED quantity, priced by
the same MDL objective J = R + mu*Omega that governs every other structural decision, so "how much symmetry"
is chosen on the same description-length footing as "which contract" and "how wide".

Mechanism: residual-pathway approximate equivariance (Finzi et al. 2021). An approximately-equivariant model
is a strictly equivariant network plus a scaled free (non-equivariant) residual,

    f(x)  =  f_equiv(x)  +  relax * f_free(x),

with relax >= 0 the relaxation strength: relax = 0 recovers exact equivariance; relax > 0 admits controlled
symmetry breaking. f_equiv is built by the package's equivariance machinery (scalable or EMLP), f_free is a
plain MLP on the same features.

Why the price is essential (and what it prices). Symmetry is a CONSTRAINT, so it is never encouraged by a
training loss that measures fit -- relaxing always lowers training loss (verified: strict vs relaxed train R2
on broken data). Selecting relax by fit alone therefore always breaks the symmetry. The relaxation must be
paid for. The symmetry-breaking pathway is ADDED STRUCTURE, and the honest code-length charge for it is the
FRACTION of the output that actually flows through it,

    Omega(relax)  =  E|| relax * f_free(x) ||^2  /  E|| f(x) ||^2   in [0, 1),

the relative power of the breaking pathway: 0 when the free path is unused (exact equivariance), growing as
breaking is used. Then relax is selected by argmin over a small ladder of

    J(relax)  =  R_val(relax)  +  mu_c * Omega(relax),

evaluated on a held-out split. On exactly-symmetric data the strict model already minimizes R_val and Omega=0,
so relax=0 is selected (Occam toward symmetry); on symmetry-broken data the strict model's R_val is large and
the fit gain from relaxing overrides the Omega charge, so a positive relax is selected -- the amount matched to
how broken the data is. Validated across breaking levels in tests/b5_approximate_equivariance.md.

Scope / honesty. This selects a scalar relaxation on a residual-pathway model; it does not implement per-layer
or per-group relaxation (the mixed-symmetry literature). It reuses the package's exact equivariant heads for
f_equiv, so at relax=0 the model is exactly equivariant by construction. The price Omega is a fit-measured
code-length proxy (relative breaking power), not a certified evidence term; it is the same kind of priced,
held-out-validated selection the package uses for contract and size.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ApproxEquivariantModel(nn.Module):
    """Residual-pathway approximately-equivariant model: f(x) = f_equiv(x) + relax * f_free(x).

    equiv_module : a strictly equivariant nn.Module (from the scalable or EMLP realization) consuming the
                   feature form it expects (this wrapper feeds it `x` directly, so the caller supplies an
                   equiv_module whose forward matches `x`).
    free_in_dim  : input dimension for the free (non-equivariant) MLP pathway.
    relax        : relaxation strength (0 = exactly equivariant). Fixed at construction; the SELECTION of relax
                   is done by select_relaxation (which builds models at several relax values and prices them).
    """

    def __init__(self, equiv_module, free_in_dim, n_out=1, relax=0.0, free_hidden=32):
        super().__init__()
        self.equiv = equiv_module
        # The free (symmetry-breaking) pathway operates on the raw input features, whose scale is arbitrary on
        # real data (e.g. atomic coordinates with std ~2-3, range +/-10) while the equivariant head outputs an
        # O(1) invariant. Left unnormalized, a full-strength residual on large uncentered inputs destabilizes
        # optimization -- so we standardize the free-pathway input with buffers set from the training data
        # (set_free_normalization), keeping the free MLP well-conditioned so that `relax` (not an optimization
        # blowup) controls the breaking contribution.
        self.register_buffer("_free_mean", torch.zeros(free_in_dim))
        self.register_buffer("_free_std", torch.ones(free_in_dim))
        self.free = nn.Sequential(nn.Linear(free_in_dim, free_hidden), nn.Tanh(), nn.Linear(free_hidden, n_out))
        self.relax = float(relax)

    def set_free_normalization(self, X_free):
        """Set the free-pathway input standardization from a batch of raw free-pathway features (b, free_in)."""
        X = X_free if isinstance(X_free, torch.Tensor) else torch.as_tensor(np.asarray(X_free), dtype=torch.float32)
        with torch.no_grad():
            self._free_mean.copy_(X.mean(0))
            self._free_std.copy_(X.std(0).clamp_min(1e-6))

    def _free_forward(self, x_free):
        return self.free((x_free - self._free_mean) / self._free_std)

    def forward(self, x, x_equiv=None):
        """x feeds the free pathway; x_equiv (default x) feeds the equivariant pathway (they may differ in
        shape, e.g. the equiv head wants (b, n_vec, d) while the free head wants (b, features))."""
        xe = x if x_equiv is None else x_equiv
        out = self.equiv(xe)
        if self.relax > 0:
            out = out + self.relax * self._free_forward(x)
        return out

    def breaking_power(self, x, x_equiv=None):
        """Relative power in the symmetry-breaking pathway, Omega in [0,1): 0 if the free path is unused."""
        with torch.no_grad():
            xe = x if x_equiv is None else x_equiv
            equiv = self.equiv(xe)
            full = equiv + (self.relax * self._free_forward(x) if self.relax > 0 else 0.0)
            bp = float(((full - equiv) ** 2).mean().item())
            tp = float((full**2).mean().item()) + 1e-9
            return bp / tp


def price_relaxation(risk_by_relax, omega_by_relax, mu_c=0.3):
    """Select the relaxation strength by J = R_val + mu_c * Omega over a ladder of candidates.

    risk_by_relax  : {relax: validation risk (1 - score)}.
    omega_by_relax : {relax: breaking power Omega in [0,1)}.
    Returns (best_relax, detail). Occam toward exact equivariance: on a near-tie the smaller relax (smaller
    Omega) wins, so exactly-symmetric data selects relax=0.
    """
    relaxes = sorted(risk_by_relax)
    J = {r: float(risk_by_relax[r]) + float(mu_c) * float(omega_by_relax.get(r, 0.0)) for r in relaxes}
    # argmin J; ties broken toward smaller relax (already sorted ascending)
    best = min(relaxes, key=lambda r: (J[r], r))
    detail = {
        "J": J,
        "risk": {r: float(risk_by_relax[r]) for r in relaxes},
        "omega": {r: float(omega_by_relax.get(r, 0.0)) for r in relaxes},
        "mu_c": float(mu_c),
        "selected_relax": float(best),
        "note": "relaxation selected by J = R_val + mu_c * Omega(relax); Omega = relative power of the "
        "symmetry-breaking pathway. relax=0 is exact equivariance (Occam default on a tie).",
    }
    return best, detail


def select_relaxation(build_and_train, relax_ladder=(0.0, 0.1, 0.3, 1.0), mu_c=0.3):
    """Full priced selection of the relaxation strength.

    build_and_train(relax) -> (val_risk, omega): a callback that builds an ApproxEquivariantModel at the given
    relax, trains it (equiv + free pathways) on the training split, and returns its held-out validation risk
    (1 - score) and its breaking power Omega on the validation split. This module stays agnostic to the data
    shape by delegating construction/training to the callback.
    Returns (best_relax, detail) from price_relaxation over the ladder.
    """
    risk_by_relax, omega_by_relax = {}, {}
    for r in relax_ladder:
        vr, om = build_and_train(r)
        risk_by_relax[r] = vr
        omega_by_relax[r] = om
    return price_relaxation(risk_by_relax, omega_by_relax, mu_c=mu_c)

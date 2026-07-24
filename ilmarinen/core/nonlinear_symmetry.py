"""Nonlinear continuous symmetry discovery via latent linearization (LaLiGAN principle;
Yang, Dehmamy, Walters & Yu, "Latent Space Symmetry Discovery", ICML 2024, arXiv:2310.00105).

The linear Lie-derivative detector (core/symmetry_discovery.py) finds a generator L with
grad_f(x) . (L x) = 0 -- it sees only symmetries that act LINEARLY on the input coordinates. Many real
symmetries act NONLINEARLY in the given coordinates (they become linear only after a coordinate change:
e.g. a scaling that is a rotation in log-polar coordinates, or a symmetry of a dynamical system that is
linear only in a learned latent space).

LaLiGAN's insight: factor a nonlinear group action as
    x --phi--> z (encoder),   LINEAR group acts on z,   z --psi--> x (decoder, ~ phi^{-1}),
so the symmetry is LINEAR in the latent space z. Then discover it there with the existing linear
detector. We adapt this to our setting (we already have a trained task model f and the linear
Lie-derivative detector), avoiding the full GAN:

  1. Train an autoencoder phi/psi on X (reconstruction) -> latent z = phi(x).
  2. Form the latent task model g = f . psi  (decoder then task model), g : z -> y.
  3. Run the EXISTING linear discover_symmetries on g in LATENT space. A latent linear generator
     L_z with grad_g(z) . (L_z z) = 0 is a latent symmetry -- it pulls back to a NONLINEAR symmetry
     of f in x-space through phi.
  4. Classify/route the latent group with the same (linear) classifier -- it is linear in z.

This reuses the entire linear stack in the autoencoder's latent coordinates. Honest caveat: the
discovered group is only as good as the autoencoder's linearization -- a poor latent space yields
spurious or missed symmetry. This is a genuine architectural addition (it needs the AE), not merely a
call into the existing detector. It is the nonlinear counterpart of the linear Family-2 detector.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Small MLP autoencoder x <-> z. latent_dim defaults to the input dim (an invertible-ish
    reparameterization, the LaLiGAN setting where the latent has the same dimension and the nonlinear
    map is a learned coordinate change)."""

    def __init__(self, d_in, latent_dim=None, hidden=64):
        super().__init__()
        latent_dim = latent_dim or d_in
        self.enc = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d_in)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.dec(self.enc(x))


def train_autoencoder(X, latent_dim=None, epochs=400, lr=3e-3, hidden=64, seed=0):
    """Train the autoencoder on X (reconstruction). Returns the trained AE."""
    torch.manual_seed(seed)
    X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    ae = Autoencoder(X.shape[1], latent_dim=latent_dim, hidden=hidden)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lf(ae(X), X)
        loss.backward()
        opt.step()
    return ae, float(lf(ae(X), X))


class _LatentTaskModel(nn.Module):
    """g = f . psi : z -> y (decode latent, then apply the task model). Its Lie-derivative in z-space
    exposes symmetries that are LINEAR in z -- i.e. nonlinear in x."""

    def __init__(self, task_model, ae):
        super().__init__()
        self.f = task_model
        self.dec = ae.dec

    def forward(self, z):
        return self.f(self.dec(z))


def discover_nonlinear_symmetries(
    task_model, X, latent_dim=None, ae_epochs=400, tol_ratio=1.8, ae=None, return_ae=False
):
    """Discover NONLINEAR continuous symmetries of `task_model` on data X, via latent linearization.

    Trains (or accepts) an autoencoder, forms the latent task model g = f . decoder, and runs the
    linear Lie-derivative detector in LATENT space. Returns the discovery dict (as discover_symmetries)
    with the generators living in latent coordinates, plus the classified latent group. If return_ae,
    also returns the autoencoder (so the caller can pull back / route).
    """
    from ilmarinen.core.equivariance_discovery import identify_group
    from ilmarinen.core.symmetry_discovery import discover_symmetries

    X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    if ae is None:
        ae, recon = train_autoencoder(X, latent_dim=latent_dim, epochs=ae_epochs)
    else:
        recon = None
    g = _LatentTaskModel(task_model, ae)
    with torch.no_grad():
        Z = ae.enc(X)
    disc = discover_symmetries(g, Z, tol_ratio=tol_ratio)
    gens = disc["generators"] if disc["n_symmetries"] > 0 else []
    group = identify_group(gens)
    out = {
        "n_symmetries": disc["n_symmetries"],
        "gap_ratio": disc["gap_ratio"],
        "latent_group": group["group"],
        "latent_labels": group["labels"],
        "generators_latent": gens,
        "ae_recon": recon,
        "note": "generators are LINEAR in the autoencoder latent space; they correspond to a "
        "NONLINEAR symmetry of the task model in the original coordinates.",
    }
    if return_ae:
        return out, ae
    return out


class _AntisymGenerator(nn.Module):
    """A learnable antisymmetric generator L_z = A - A^T (a rotation-family candidate) in latent space,
    used to REGULARIZE the autoencoder toward a latent where a linear symmetry exists (LaLiGAN's joint
    objective). Antisymmetric = the so(n) rotation family, the most common continuous symmetry."""

    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(0.01 * torch.randn(d, d))

    def L(self):
        return self.A - self.A.T

    def forward(self, z, eps=0.1):
        return z + eps * z @ self.L().T  # infinitesimal action z -> (I + eps L) z


def discover_nonlinear_symmetries_joint(
    task_model,
    X,
    latent_dim=None,
    epochs=600,
    lr=3e-3,
    recon_weight=1.0,
    sym_weight=20.0,
    eps=0.1,
    hidden=64,
    seed=0,
    tol_ratio=1.8,
    null_ratio=1.5,
):
    """Nonlinear symmetry discovery with a JOINTLY-TRAINED symmetry-regularized autoencoder (the actual
    LaLiGAN mechanism). A plain reconstruction AE has no incentive to LINEARIZE the symmetry; here the
    AE is trained with a symmetry-consistency loss that rewards a latent in which an infinitesimal
    linear (antisymmetric / rotation-family) action leaves g = f . decoder invariant:

        L_total = recon_weight * ||psi(phi(x)) - x||^2
                + sym_weight  * mean| g((I + eps L_z) phi(x)) - g(phi(x)) |,   L_z antisymmetric.

    After joint training, the standard linear detector confirms and classifies the latent symmetry.
    Returns dict with the discovered latent group and the learned generator; honest about the
    candidate family (antisymmetric/rotation) it searches -- the fully general version sweeps families.
    """
    from ilmarinen.core.equivariance_discovery import identify_group
    from ilmarinen.core.symmetry_discovery import discover_symmetries

    torch.manual_seed(seed)
    X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    d = X.shape[1]
    ld = latent_dim or d
    ae = Autoencoder(d, latent_dim=ld, hidden=hidden)
    gen = _AntisymGenerator(ld)
    for p in task_model.parameters():
        p.requires_grad_(False)  # freeze the task model; adapt only the AE + generator
    opt = torch.optim.Adam(list(ae.parameters()) + list(gen.parameters()), lr=lr)
    mse = nn.MSELoss()

    def g_of_z(z):
        return task_model(ae.dec(z))

    for _ in range(epochs):
        opt.zero_grad()
        z = ae.enc(X)
        recon = mse(ae.dec(z), X)
        z_t = gen(z, eps=eps)
        sym = (g_of_z(z_t) - g_of_z(z)).abs().mean()
        (recon_weight * recon + sym_weight * sym).backward()
        opt.step()

    with torch.no_grad():
        z = ae.enc(X)
        recon = float(mse(ae.dec(z), X))
        sym_viol = float((g_of_z(gen(z, eps=eps)) - g_of_z(z)).abs().mean())
        # NULL-BASELINE GUARD against a spurious "symmetry". The joint objective drives sym_viol toward 0
        # BY CONSTRUCTION, so a small sym_viol alone does not confirm a real symmetry -- an AE can hide a
        # non-symmetry by collapsing the acted-on latent direction. Calibrate against RANDOM antisymmetric
        # generators of the same size in the same latent: a genuine symmetry has sym_viol far below the
        # typical random-direction violation (scale-aware logic, matching the discrete detector). If the
        # learned generator is not decisively quieter than random (by null_ratio), the "symmetry" is an AE
        # artefact and is rejected.
        d_lat = z.shape[1]
        rand_viols = []
        for r in range(8):
            torch.manual_seed(seed + 100 + r)
            # GENERAL random generator (NOT restricted to antisymmetric): a genuine symmetry is quiet under
            # its own family but LOUD under a generic linear perturbation. Using antisymmetric-only randoms
            # would be ill-posed in low dim (in 2D the only antisymmetric generator IS the rotation, so the
            # baseline would coincide with the symmetry). A generic generator spans scaling+shear+rotation,
            # so its typical violation measures how much g actually varies in the acted-on latent.
            Lr = torch.randn(d_lat, d_lat)
            Lr = Lr / (Lr.norm() + 1e-9) * (gen.L().norm() + 1e-9)  # match the learned generator's size
            zt = z + eps * z @ Lr.T
            rand_viols.append(float((g_of_z(zt) - g_of_z(z)).abs().mean()))
        null_viol = float(np.median(rand_viols)) if rand_viols else 0.0
        confirmed_by_null = (sym_viol * null_ratio) < null_viol  # learned must be >=null_ratio quieter

    # confirm + classify with the standard linear detector in the learned latent
    g = _LatentTaskModel(task_model, ae)
    disc = discover_symmetries(g, z.detach(), tol_ratio=tol_ratio)
    gens = disc["generators"] if disc["n_symmetries"] > 0 else []
    group = identify_group(gens) if confirmed_by_null else {"group": "none", "labels": []}
    return {
        "n_symmetries": disc["n_symmetries"] if confirmed_by_null else 0,
        "gap_ratio": disc["gap_ratio"],
        "latent_group": group["group"],
        "latent_labels": group["labels"],
        "generators_latent": gens if confirmed_by_null else [],
        "learned_generator": gen.L().detach().numpy(),
        "ae": ae,
        "latent_dim": int(z.shape[1]),
        "ae_recon": recon,
        "sym_violation": sym_viol,
        "null_violation": null_viol,
        "confirmed_by_null": bool(confirmed_by_null),
        "note": "AE jointly trained with a symmetry-consistency loss (LaLiGAN); the antisymmetric "
        "candidate biases toward the rotation family. A discovered symmetry is confirmed only "
        "if its sym_violation is >=null_ratio quieter than random generators of matched size in "
        "the same latent (guards against the AE hiding a non-symmetry by collapsing a latent axis).",
    }


def discover_symmetries_with_nonlinear_fallback(task_model, X, tol_ratio=1.8, nonlinear=True, joint=True, **nl_kwargs):
    """Convenience escalation: try LINEAR symmetry discovery first (cheap, exact for linearly-acting
    groups); only if it finds nothing AND nonlinear=True, escalate to latent-linearization discovery (the
    LaLiGAN route) to catch a symmetry that acts nonlinearly in the given coordinates. This is the intended
    top-level entry when the coordinate system is unknown: linear-first keeps the common case fast and
    exact, and the nonlinear escalation is attempted only when it could add something.

    Returns a dict with 'route' ('linear' | 'nonlinear' | 'none'), the discovered group, and the raw
    detail from whichever detector fired. Honest scope: the nonlinear route's success is bounded by the
    autoencoder's ability to linearize the action -- it reliably recovers moderate warps (tanh, mild
    polynomial) of a rotation symmetry and is guarded against false positives by the null-baseline check,
    but a severe coordinate warp can still defeat the AE (recorded limitation, not a silent failure)."""
    from ilmarinen.core.equivariance_discovery import identify_group
    from ilmarinen.core.symmetry_discovery import discover_symmetries

    X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    lin = discover_symmetries(task_model, X, tol_ratio=tol_ratio)
    if lin.get("n_symmetries", 0) > 0:
        grp = identify_group(lin["generators"])
        return {
            "route": "linear",
            "group": grp["group"],
            "labels": grp["labels"],
            "n_symmetries": lin["n_symmetries"],
            "detail": lin,
        }
    if not nonlinear:
        return {"route": "none", "group": "none", "labels": [], "n_symmetries": 0, "detail": lin}
    nl = (discover_nonlinear_symmetries_joint if joint else discover_nonlinear_symmetries)(
        task_model, X, tol_ratio=tol_ratio, **nl_kwargs
    )
    found = nl.get("n_symmetries", 0) > 0 and nl.get("latent_group", "none") != "none"
    return {
        "route": "nonlinear" if found else "none",
        "group": nl.get("latent_group", "none"),
        "labels": nl.get("latent_labels", []),
        "n_symmetries": nl.get("n_symmetries", 0) if found else 0,
        "detail": nl,
    }

"""Latent-equivariant contract: deploy a discovered NONLINEAR symmetry (direction B3).

The linear symmetry path already closes the loop discover -> GENERATE an equivariant contract (EMLP) ->
deploy. The nonlinear path stopped one step short: nonlinear_symmetry only REPORTED a discovered latent
symmetry (LaLiGAN-style) as a diagnostic, because there was no validated way to BUILD a contract for it.
This module supplies that missing realization.

Construction. LaLiGAN (Yang et al., ICML 2024) discovers a nonlinear symmetry by learning an autoencoder
phi: x -> z (encoder) and psi: z -> x (decoder) to a latent space in which the symmetry acts LINEARLY, with
generators L_z of that linear latent action. Once the symmetry is linear in z, the package's own EMLP
machinery (emlp_layer.EquivariantMLP, built from generators alone) constructs a network equivariant to it.
The deployed contract is therefore

    x  --phi-->  z  --EquivariantMLP(L_z)-->  y,

an end-to-end predictor whose invariance/equivariance to the discovered NONLINEAR symmetry of x is exact by
construction (the EMLP is exactly equivariant to the linear latent action, and phi carries the nonlinear
symmetry to that linear action). The encoder phi is taken from the joint symmetry-regularized autoencoder
that discovered the symmetry, so detection and deployment share one latent chart.

Why this is the right realization (validated in tests/b3_latent_equivariant_contract.md). On a controlled
task with a genuine nonlinear symmetry (SO(2) acting on a latent z, observed through a nonlinear embedding,
with a rotation-invariant target), the latent-equivariant contract beats a matched plain MLP, and the gain
GROWS as data shrinks (n=500: +0.03; n=200: +0.11; n=80: +0.17) -- exactly the low-data inductive-bias
advantage equivariance is supposed to provide. The EMLP built from the latent SO(2) generator is exactly
invariant to the latent rotation (residual 0).

Honest scope. The latent chart phi is learned (approximate); the contract's equivariance is exact in the
LATENT coordinates but only as faithful as phi is to the true nonlinear symmetry. Discovery uses the
package's antisymmetric/rotation-family latent search (the common continuous case), not an exhaustive family
sweep. This is a genuine deployable contract for the discovered latent symmetry, gated -- like the linear
generated contract -- behind signal-quality checks; it is not a claim to have solved general nonlinear
equivariance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LatentEquivariantContract(nn.Module):
    """x -> encoder(phi) -> latent z -> EquivariantMLP(latent generators) -> y.

    encoder      : nn.Module mapping (b, d_in) -> (b, latent_dim), the LaLiGAN latent chart phi.
    latent_gens  : list of (latent_dim x latent_dim) generator matrices of the linear latent action.
    latent_dim   : dimension of the latent vector (a single vector in the latent group's representation).
    The EquivariantMLP is exactly equivariant to the linear latent action, so the whole map is equivariant
    to the corresponding nonlinear symmetry of x (to the fidelity of phi).
    """

    def __init__(self, encoder, latent_gens, latent_dim, n_out=1, hidden_vec=4, hidden_scalar=8,
                 depth=2, metric=None, freeze_encoder=True, realization="emlp"):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        gens = [g if isinstance(g, torch.Tensor) else torch.as_tensor(np.asarray(g), dtype=torch.float32)
                for g in latent_gens]
        self.latent_dim = int(latent_dim)
        self.realization = realization
        # the latent vector is a single vector in the group rep (n_in_vec=1, vec_dim=latent_dim). The head can
        # be realized two ways with the SAME invariance contract: "emlp" (exact basis-solve; general but the
        # O(D^3) solve does not scale) or "scalable" (G-RepsNet/Vector-Neurons vector mixing; equivariant by
        # construction, no basis solve, linear in channels -- see models/scalable_equivariant.py).
        if realization == "scalable":
            from .scalable_equivariant import ScalableEquivariantMLP
            self._emlp = ScalableEquivariantMLP(gens, n_in_vec=1, vec_dim=self.latent_dim,
                                                hidden_vec=max(hidden_vec, 8), hidden_scalar=max(hidden_scalar, 16),
                                                depth=depth, n_out=n_out, metric=metric)
        else:
            from ..core.emlp_layer import EquivariantMLP
            self._emlp = EquivariantMLP(gens, n_in_vec=1, vec_dim=self.latent_dim, hidden_vec=hidden_vec,
                                        hidden_scalar=hidden_scalar, depth=depth, n_out=n_out, metric=metric)
        self.head = self._emlp.torch_module()

    def forward(self, x):
        z = self.encoder(x)                      # (b, latent_dim) : the latent chart
        if self.realization == "scalable":
            z = z.unsqueeze(1)                   # (b, 1, latent_dim) : one vector in the group rep
        return self.head(z)                      # equivariant/invariant head on the latent vector

    def latent(self, x):
        with torch.no_grad():
            return self.encoder(x)


def build_latent_equivariant_contract(encoder, latent_gens, latent_dim, n_out=1, hidden_vec=4,
                                      hidden_scalar=8, depth=2, metric=None, freeze_encoder=True,
                                      realization="emlp"):
    """Factory mirroring the other schema builders: returns a LatentEquivariantContract module. Set
    realization='scalable' to use the G-RepsNet/Vector-Neurons head (equivariant by construction, scalable)
    instead of the exact-but-cubic EMLP head."""
    return LatentEquivariantContract(encoder, latent_gens, latent_dim, n_out=n_out, hidden_vec=hidden_vec,
                                     hidden_scalar=hidden_scalar, depth=depth, metric=metric,
                                     freeze_encoder=freeze_encoder, realization=realization)

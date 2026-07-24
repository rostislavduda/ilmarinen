"""Route 1: continuous-depth (neural ODE) model with a criticality regularizer.

This is the GEOMETRIC depth route from the Scenario-1 analysis: depth is integration
time T of a flow dz/dt = f(z), not a discrete layer count. The forward pass integrates
the vector field; the backward pass uses the adjoint method (Pontryagin), so
backpropagation is continuous-depth costate integration.

Requires torchdiffeq (odeint_adjoint).

=== VALIDATED (what works) ===
- The neural ODE trains end-to-end via the adjoint on ground-truth tasks
  (concentric rings: tanh 1.000, ReLU 0.930).
- The CRITICALITY REGULARIZER (`criticality_penalty`) drives the flow to marginal
  stability (leading Jacobian eigenvalue -> 0) WITHOUT destroying task accuracy.
  Validated: a scheduled penalty reached matched strict criticality across seeds
  (tanh eig +0.086, ReLU +0.014) while accuracy stayed 0.96-1.00. (This overturned
  an initial hypothesis that criticality and task-solving fundamentally conflict.)

=== REFUTED (what does not hold) ===
- The derived "depth-freedom" prediction -- that arc length (complexity) is CONVERGENT
  for ReLU (alpha=2) but LOGARITHMIC for tanh (alpha=1), i.e. higher alpha makes depth
  asymptotically free -- DOES NOT HOLD. At matched strict criticality, BOTH activations
  give LINEAR arc-length growth (R^2 ~ 1.0), and ReLU is NOT more bounded than tanh
  (7.97 vs 6.90 at T=8, 3 seeds each). An earlier "directionally consistent" signal was
  an artifact of UNMATCHED criticality; the matched test exposed it.
  INTERPRETATION: at criticality the flow settles to ~constant speed, so arc length ~ T
  (linear) independent of activation -- the heuristic marginal-mode-dominance derivation
  does not survive a real trained neural ODE. (Consistent with the novelty audit's own
  low-confidence hedge on this result.)

So this module is a validated continuous-depth MODEL + a validated criticality control,
NOT a validated instance of the depth-freedom scaling law. `arc_length` and
`leading_jacobian_re` are retained as measurement tools (they produced the refutation).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint_adjoint as _odeint
    _HAVE_TORCHDIFFEQ = True
except Exception:  # pragma: no cover
    _HAVE_TORCHDIFFEQ = False


class ODEFunc(nn.Module):
    """Vector field f(z) for the flow dz/dt = f(z). Stateless in t (autonomous)."""

    def __init__(self, dim, hidden=48, act="tanh", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        A = nn.Tanh if act == "tanh" else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), A(), nn.Linear(hidden, dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        return self.net(z)


def criticality_penalty(func, z, iters=4):
    """Differentiable proxy for the leading eigenvalue of J = df/dz along the flow.

    Power iteration on J via JVPs, returning the per-example Rayleigh quotient
    (leading-eigenvalue estimate). Penalizing its SQUARE toward 0 drives the flow to
    criticality (marginal stability). Returns a (batch,) tensor; caller squares+means.

    Validated: a schedule ramping this penalty to a large weight reaches strict
    criticality (leading Re-eig ~ 0) without breaking task accuracy.
    """
    zc = z.detach().requires_grad_(True)
    f = func(0.0, zc)
    v = torch.randn_like(zc)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
    lam = None
    for _ in range(iters):
        Jv = torch.autograd.grad(f, zc, grad_outputs=v, create_graph=True,
                                 retain_graph=True)[0]
        lam = (v * Jv).sum(1)
        v = (Jv / (Jv.norm(dim=1, keepdim=True) + 1e-9)).detach()
    return lam


class NeuralODE(nn.Module):
    """Continuous-depth classifier: augment -> integrate f -> readout.

    depth == integration time T (steps just set the RK4 grid). Augmentation dims give
    the planar flow room to untangle non-separable classes.
    """

    def __init__(self, n_in=2, n_out=2, act="tanh", aug=4, T=1.0, steps=30,
                 hidden=48, seed=0):
        super().__init__()
        if not _HAVE_TORCHDIFFEQ:
            raise ImportError("NeuralODE requires torchdiffeq (pip install torchdiffeq)")
        self.n_in = n_in
        self.aug = aug
        self.dim = n_in + aug
        self.T = T
        self.func = ODEFunc(self.dim, hidden=hidden, act=act, seed=seed)
        torch.manual_seed(seed + 100)
        self.head = nn.Linear(self.dim, n_out)
        self.register_buffer("t", torch.linspace(0, T, steps))

    def embed(self, x):
        b = x.shape[0]
        return torch.cat([x, x.new_zeros(b, self.aug)], 1)

    def forward(self, x):
        zT = _odeint(self.func, self.embed(x), self.t, method="rk4")[-1]
        return self.head(zT)

    def arc_length(self, x, T_eval, steps=200):
        """Measure integral_0^{T_eval} mean_batch ||f(z(t))|| dt along the flow.

        This is the complexity functional of the geometric route. NOTE: the derived
        alpha-dependent scaling of this quantity was REFUTED (linear regardless of
        activation at criticality); retained as a measurement tool.
        """
        with torch.no_grad():
            tt = torch.linspace(0, T_eval, steps)
            traj = _odeint(self.func, self.embed(x), tt, method="rk4")
            sp = [self.func(tt[k], traj[k]).norm(dim=1).mean().item()
                  for k in range(steps)]
            return float(np.trapz(np.array(sp), dx=T_eval / (steps - 1)))

    def leading_jacobian_re(self, x, n_pts=6):
        """Mean leading real-part eigenvalue of J=df/dz sampled along the trajectory.

        The criticality order parameter: >0 chaotic (expanding), <0 ordered
        (contracting), ~0 critical (marginal). Uses functional jacobian.
        """
        from torch.func import jacrev
        with torch.no_grad():
            tt = torch.linspace(0, self.T, 4)
            traj = _odeint(self.func, self.embed(x), tt, method="rk4")

        def f_single(z):
            return self.func(0.0, z.unsqueeze(0)).squeeze(0)

        vals = []
        for k in range(4):
            for zi in traj[k][:n_pts]:
                J = jacrev(f_single)(zi.detach())
                # eigvals (non-Hermitian) has no MPS kernel -> raises on Apple Silicon with
                # PYTORCH_ENABLE_MPS_FALLBACK unset. J is tiny (latent x latent), so compute the spectrum on
                # CPU: robust across backends and faster for a small matrix. No-op effect on CUDA/CPU.
                vals.append(torch.linalg.eigvals(J.cpu()).real.max().item())
        return float(np.mean(vals))


def build_neural_ode(**kwargs):
    return NeuralODE(**kwargs)

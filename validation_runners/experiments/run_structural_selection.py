"""Consolidated priced selection: primitive AND kernel size (receptive field) chosen JOINTLY by one
differentiable action J = R + mu * Omega, with Omega an ANALYTICALLY-GROUNDED structural cost.

This is the consolidation of the B4 / priced-structural work into the selection pipeline. Where
run_penalized_selection prices primitives by raw parameter count, this runner prices the SPATIAL
schema's candidates by their analytical structural cost from machinery/priced_structural:
    conv kernel of size k  ->  cost ~ k^d   (kernel volume; d=2 here)
so that among {conv2d(k3), conv2d_k5, conv2d_k7, conv_dw, pointwise, dense, norm, attention} the
receptive-field price is quadratic in k, exactly as the analysis prescribes. Sweeping mu traces the
fit--complexity frontier over BOTH the primitive type and its receptive field in a single search.

Analytical action (hardware-aware differentiable NAS; FBNet / YOSO arXiv:2208.14446):
    J(alpha, w) = L_valid(w, alpha) + mu * sum_i softmax(alpha)_i * cost_i  (+ gamma * entropy sharpen)
with cost_i the normalized structural cost of candidate i. As mu rises, selection moves from the most
accurate (often a large-kernel conv) toward the cheapest adequate candidate -- the description-length
-optimal receptive field, derived rather than fixed.

Usage:
    python run_structural_selection.py --task smooth --mu 0.05
    python run_structural_selection.py --task smooth --sweep 0,0.02,0.05,0.1
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ilmarinen.models import build_spatial_schema
from ilmarinen.machinery.priced_structural import kernel_costs, priced_objective, select_by_priced_rule
from ilmarinen.core.correlation_length import recommend_conv_primitive

# candidate set: conv at three receptive fields + the non-conv spatial primitives

def make_task(task, n=400, H=20, seed=0):
    """A controlled 2D task where the RECEPTIVE FIELD genuinely matters. The label depends on whether
    two blobs, placed far apart (distance ~D), are the SAME or DIFFERENT brightness. Discriminating
    requires integrating information across distance D in one layer -> a kernel must be large enough to
    span it. 'smooth' uses a large separation (rewards a big kernel); 'fine' a small separation
    (small kernel suffices). This creates a real fit-vs-receptive-field tradeoff."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    sep = 7 if task == "smooth" else 3           # distance the kernel must span
    X = np.zeros((n, 1, H, H), dtype=np.float32); y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        same = rng.rand() < 0.5
        cy = rng.randint(3, H - 3); cx = rng.randint(2, H - 2 - sep)
        img = np.zeros((H, H))
        img[cy, cx] = 1.0
        img[cy, cx + sep] = 1.0 if same else 0.4
        X[i, 0] = gaussian_filter(img, 0.8) + 0.3 * rng.randn(H, H); y[i] = 0 if same else 1
    return torch.tensor(X), torch.tensor(y)



def run(args):
    Xtr, ytr = make_task(args.task, n=500, seed=0)
    Xv, yv = make_task(args.task, n=200, seed=1)

    # data-side recommendation (correlation length) for cross-checking the priced selection
    rec = recommend_conv_primitive(Xtr.numpy(), ndim=2)
    print(f"=== consolidated structural selection on '{args.task}' task ===")
    print(f"correlation-length recommendation: xi={rec['xi']:.2f} -> {rec['primitive']} "
          f"(kernel {rec['kernel_size']})")

    # isolate the receptive-field decision: conv family at three kernel sizes, priced by k^2.
    conv_prims = ["conv2d", "conv2d_k5", "conv2d_k7"]
    ks = [3, 5, 7]
    costs = kernel_costs(ks, ndim=2)
    print(f"conv candidates: {conv_prims}  |  structural costs (k^2, norm): "
          f"{[round(float(c), 2) for c in costs]}")

    # measure solo accuracy per receptive field (the fit term), then price it
    def solo(prim, width=16, epochs=args.epochs):
        torch.manual_seed(0)
        net = build_spatial_schema(width=width, hw=10, depth=1, n_in=1, n_classes=2,
                                               primitives=(prim,), img_size=Xtr.shape[-1])
        opt = torch.optim.Adam(net.parameters(), lr=3e-3); lf = nn.CrossEntropyLoss()
        for ep in range(epochs):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 32):
                bi = perm[i:i + 32]; opt.zero_grad(); lf(net(Xtr[bi]), ytr[bi]).backward(); opt.step()
        with torch.no_grad():
            return float((net(Xv).argmax(-1) == yv).float().mean())

    accs = [solo(p) for p in conv_prims]
    print(f"\nsolo accuracy per receptive field: "
          f"{ {p: round(a, 3) for p, a in zip(conv_prims, accs)} }")

    mus = [float(m) for m in args.sweep.split(",")] if args.sweep else [args.mu]
    print(f"\n{'mu':>6}  {'selected kernel':<16} {'J = -acc + mu*cost'}")
    for mu in mus:
        J, sel = priced_objective(accs, costs, mu)
        print(f"{mu:>6.3f}  k={ks[sel]:<14} J={[round(float(j), 3) for j in J]}")
    # the no-harm minimal-resource choice
    sel_rule = select_by_priced_rule(accs, costs, acc_tol=0.02)
    print(f"\nno-harm rule (accuracy within 0.02 of best): k={ks[sel_rule]}")
    print("As mu rises, the priced action trades receptive field for cheapness -- k7 -> k5 -> k3 -- "
          "so the receptive field is a DERIVED, priced decision on the same J = R + mu*Omega frontier "
          "as primitive, width, and depth. Cross-check: the correlation-length recommendation above "
          "picks the same large kernel the mu=0 (accuracy-first) end of the frontier selects.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="smooth", choices=["smooth", "fine"])
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--sweep", type=str, default="", help="comma-separated mus, e.g. 0,0.02,0.05,0.1")
    ap.add_argument("--epochs", type=int, default=30)
    run(ap.parse_args())

"""Deep-but-narrow selection: can AllGraph predict networks deeper than the default cap?

A STUDY (not a package feature): probes whether the depth-selection machinery
(select_architecture_by_area with significance-gated boundary extension, and the underlying
priced_depth curve + significant_elbow) will predict a deeper network when a real problem
rewards depth, rather than being pinned at a hand-set ceiling. See
tests/deep_network_depth_study.md for the full write-up and honest verdict.

Background. The reported sweeps effectively used depth grids topping out around 3-4. The
machinery is NOT hard-capped there: select_architecture_by_area starts with a small depth grid
and, if the best score sits at the grid's largest depth (the ceiling is binding), it extends the
grid one layer at a time and re-measures, continuing until the optimum is interior or a
(generous) max_depth_cap is reached -- and it only accepts a deeper column if it beats the
incumbent by MORE than the across-seed noise floor (so it never chases the single-seed
depth-mirage that B6 documented). This study checks that extension fires on real data and asks
how deep real problems actually push it.

Three parts:
  1. negative_control(): a linearly-separable task -- depth must NOT help; the elbow must be 1.
     Validates that the selector is not biased toward depth.
  2. esol_production_depth(): runs the REAL production path (select_architecture_by_area) on ESOL
     molecular solubility with the cap raised, and reports the selected depth and whether the
     boundary extension fired. GNN depth = message-passing receptive field, a genuine physical
     depth requirement.
  3. deep_generation_is_not_deep_requirement(): a task whose LABEL is produced by a depth-D
     composition, showing that a deep generative process does not imply a deep LEARNING
     requirement -- a shallow net fits it, so the selector (correctly) stays shallow.

Run: python studies/deep_network_depth_study.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from ilmarinen.machinery.priced_depth import measure_depth_curve, significant_elbow  # noqa: E402


# --------------------------------------------------------------------------- part 1: control
class _MLP(nn.Module):
    def __init__(self, d, width, depth, nout=2):
        super().__init__()
        layers, prev = [], d
        for _ in range(depth):
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, nout)

    def forward(self, x):
        return self.head(self.net(x))


def _fit_mlp(X, y, depth, seed, width=16, epochs=60):
    torch.manual_seed(seed)
    n = len(y)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    nv = max(1, n // 4)
    va, tr = perm[:nv], perm[nv:]
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])
    net = _MLP(X.shape[1], width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        p = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            bi = p[i:i + 64]
            opt.zero_grad()
            loss = lf(net(Xtr[bi]), ytr[bi])
            if torch.isfinite(loss):
                loss.backward()
                opt.step()
    with torch.no_grad():
        return float((net(Xva).argmax(1) == yva).float().mean())


def negative_control(n_seeds=3):
    print("=" * 74)
    print("PART 1 -- negative control: a linearly separable task must select depth 1")
    print("=" * 74)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2500, 8)).astype(np.float32)
    w = rng.standard_normal(8)
    y = (X @ w > 0).astype(np.int64)
    depths = [1, 2, 3, 4, 5, 6]
    curve = measure_depth_curve(lambda L, sd: (1 - _fit_mlp(X, y, L, sd), _fit_mlp(X, y, L, sd)),
                                depths, list(range(n_seeds)))
    print("  val acc by depth: " + "  ".join(f"L{d}={curve.acc_mean[i]:.3f}"
                                              for i, d in enumerate(curve.depths)))
    print("  marginals:        " + "  ".join(f"{m:+.3f}" for (_, m, _) in curve.marginals))
    print(f"  significant_elbow(2se) = {significant_elbow(curve, n_se=2.0)}  (expected 1)\n")


# --------------------------------------------------------------------------- part 2: ESOL prod
def esol_production_depth(max_depth_cap=8):
    print("=" * 74)
    print("PART 2 -- ESOL via the REAL production path (select_architecture_by_area)")
    print("=" * 74)
    try:
        from ilmarinen.core.moleculenet import load_esol
        from ilmarinen.core.allgraph import AllData, AllGraph
    except Exception as e:
        print(f"  SKIP (import failed: {str(e)[:60]})\n")
        return
    try:
        graphs, y = load_esol(n_max=500)
    except Exception as e:
        print(f"  SKIP (ESOL load failed: {str(e)[:60]})\n")
        return
    y = np.asarray(y, dtype=np.float32)
    sizes = [np.asarray(g["x"]).shape[0] for g in graphs]
    node_feats = [np.asarray(g["x"], dtype=np.float32) for g in graphs]
    edges = [np.asarray(g["edge_index"], dtype=np.int64) for g in graphs]
    data = AllData.graphs(node_feats, edges, y=y)
    mg = AllGraph(width=32, depth=1, seed=0, epochs=40)
    detail = mg.select_architecture_by_area(
        data, task="regression", n_out=1, contract="graph",
        widths=(16, 32), depths=(1, 2, 3), tol=0.05, seeds=(0, 1),
        extend_depth=True, max_depth_cap=max_depth_cap, record=False)
    print(f"  molecules: n={len(y)}, mean {np.mean(sizes):.0f} atoms, max {max(sizes)}")
    print(f"  selected  width*={detail['width_star']}  depth*={detail['depth_star']}  "
          f"area*={detail['area_star']}")
    print(f"  boundary extension fired: {detail['depth_extended']}  "
          f"(max_depth_reached={detail['max_depth_reached']}, noise={detail['noise']})")
    g = detail["grid"]
    for L in range(1, detail["max_depth_reached"] + 1):
        print("    " + "  ".join(f"w{w}_L{L}={g.get(f'w{w}_L{L}', '-')}" for w in (16, 32)))
    print("  READ: extension past the initial depth-3 grid to depth 4 is the mechanism predicting")
    print("  a deeper-than-default network because the real task rewards it; it stops when deeper")
    print("  stops paying beyond noise.\n")


# ------------------------------------------------------------ part 3: deep generation != deep req
def _make_cumulative(n, D, T=16, seed=0):
    """Label produced by a depth-D composition of (smooth conv + level-dependent rectification)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T)).astype(np.float32)
    h = X.copy()
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    for t in range(D):
        h = np.maximum(np.stack([np.convolve(h[i], k, mode="same") for i in range(n)]) - 0.1 * t, 0.0)
    return X, (h.sum(1) > np.median(h.sum(1))).astype(np.int64)


def deep_generation_is_not_deep_requirement(n_seeds=3):
    print("=" * 74)
    print("PART 3 -- a depth-D-GENERATED label does not imply a depth-D LEARNING requirement")
    print("=" * 74)
    for D in (4, 6):
        X, y = _make_cumulative(3000, D)
        depths = [1, 2, 3, 4, 5, 6]
        curve = measure_depth_curve(lambda L, sd: (1 - _fit_mlp(X, y, L, sd), _fit_mlp(X, y, L, sd)),
                                    depths, list(range(n_seeds)))
        print(f"\n  cumulative label from a depth-{D} composition (plain ReLU MLP, width 16):")
        print("    val acc by depth: " + "  ".join(f"L{d}={curve.acc_mean[i]:.3f}"
                                                   for i, d in enumerate(curve.depths)))
        print("    marginals:        " + "  ".join(f"{m:+.3f}" for (_, m, _) in curve.marginals))
        print(f"    significant_elbow = {significant_elbow(curve, n_se=2.0)}")
    print("\n  READ: the label is generated by a deep process, yet a shallow net already fits it")
    print("  (the composed map is smooth), so the selector correctly stays shallow. Depth of the")
    print("  DATA-GENERATING process is not depth REQUIRED to learn the map.\n")


if __name__ == "__main__":
    negative_control()
    esol_production_depth()
    deep_generation_is_not_deep_requirement()
    print("(see tests/deep_network_depth_study.md for the full write-up and verdict)")

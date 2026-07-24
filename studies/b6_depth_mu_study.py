"""B6: empirical test of the depth-scaling prediction L* ~ mu^{-1/(alpha+1)}.

A STUDY (not a package feature): measures the depth-vs-mu behavior of trained networks across the validation
suite using the existing priced_depth machinery, and checks whether the RG-predicted exponent -1/(alpha+1) is
recovered. See tests/b6_depth_mu_study.md for the full write-up and verdict (a first-class negative result:
the exponent is seed-unstable in the small-network regime, so the prediction is not empirically confirmed).

Two parts:
  1. synthetic_recovery(): validates the measurement pipeline on curves with a KNOWN marginal ~ l^{-(alpha+1)}
     -- it recovers the exponent to a few percent, so any real-data failure is a property of the nets, not the
     measurement.
  2. measure_real_curve(name): measures S(L) = val loss vs depth on an AllGraph net, fits the marginal decay,
     and reports the apparent exponent + power-law R^2 (which is seed-dependent -- run with 2 vs 3 seeds).

Run: python studies/b6_depth_mu_study.py [dataset_name]
"""
import warnings; warnings.filterwarnings("ignore")
import sys

import numpy as np

sys.path.insert(0, ".")
from ilmarinen.machinery.priced_depth import DepthCurve, measure_depth_curve, select_depth  # noqa: E402


def synthetic_recovery(alphas=(1.0, 2.0, 3.0), Lmax=60):
    """Verify select_depth + log-log fit recover -1/(alpha+1) on curves with marginal ~ l^{-(alpha+1)}."""
    print("Synthetic recovery (marginal ~ l^{-(alpha+1)} by construction):")
    for alpha in alphas:
        depths = list(range(1, Lmax + 1))
        S, acc = np.zeros(len(depths)), 0.0
        for i, L in enumerate(depths):
            acc += L ** (-(alpha + 1.0))
            S[i] = 5.0 - acc
        marg = [((depths[i - 1] + depths[i]) / 2, (S[i - 1] - S[i]) / (depths[i] - depths[i - 1]), 0.0)
                for i in range(1, len(depths))]
        curve = DepthCurve(depths, S, np.zeros(len(depths)), np.zeros(len(depths)), marg)
        mus = np.logspace(-3.5, -1.0, 12)
        Ls = np.array([select_depth(curve, mu) for mu in mus]); ok = Ls > 1
        slope = np.polyfit(np.log(mus[ok]), np.log(Ls[ok]), 1)[0]
        print(f"  alpha={alpha}: predicted {-1/(alpha+1):.3f}, measured {slope:.3f}")


def measure_real_curve(name="ESOL", task="regression", width=48, depths=(1, 2, 3, 4, 5), seeds=(0, 1, 2)):
    """Measure a real depth curve vian AllGraph and fit the marginal-value power law."""
    import torch

    from ilmarinen.core.allgraph import AllGraph
    from ilmarinen.core.dataset_registry import quick_suite
    d = quick_suite()[name][0](reduced=True, device="cpu"); tr, te = d["train"], d["test"]

    def train_eval(depth, seed):
        np.random.seed(seed); torch.manual_seed(seed)
        mg = AllGraph(width=width, depth=depth, epochs=40, verbose=False, seed=seed)
        mg.fit(tr, task=task); y = np.asarray(te.y)
        with torch.no_grad():
            ids = np.arange(len(te.node_feats))
            wp = (mg.contract == "equivariant"); ue = mg.contract in ("graph", "equivariant")
            out = mg._forward_contract(mg.net, te, ids, mg.contract, wp, ue, 3.0).cpu().numpy()
        p = out.squeeze(-1) if out.ndim > 1 else out
        return float(((p - y[:len(p)]) ** 2).mean()), 0.0

    curve = measure_depth_curve(train_eval, list(depths), list(seeds))
    mids = np.array([m[0] for m in curve.marginals]); ms = np.array([m[1] for m in curve.marginals])
    pos = ms > 1e-5
    print(f"\n{name} ({len(seeds)} seeds): S={[round(float(s), 4) for s in curve.S_mean]}")
    print(f"  marginals={[(round(m[0], 1), round(float(m[1]), 4)) for m in curve.marginals]}")
    if pos.sum() >= 3:
        slope, inter = np.polyfit(np.log(mids[pos]), np.log(ms[pos]), 1)
        pred = slope * np.log(mids[pos]) + inter
        denom = ((np.log(ms[pos]) - np.log(ms[pos]).mean()) ** 2).sum() + 1e-9
        r2 = 1 - ((np.log(ms[pos]) - pred) ** 2).sum() / denom
        print(f"  apparent (alpha+1)={-slope:.2f} (alpha={-slope-1:.2f}), power-law R2={r2:.3f}")
        print("  NOTE: this exponent is seed-unstable -- rerun with a different number of seeds to see it move.")
    else:
        print(f"  only {int(pos.sum())} positive marginals -- depth saturates/does not pay (no power law).")


if __name__ == "__main__":
    synthetic_recovery()
    if len(sys.argv) > 1:
        measure_real_curve(sys.argv[1])
    else:
        print("\n(pass a dataset name, e.g. `python studies/b6_depth_mu_study.py ESOL`, for a real curve)")

"""D6: depth scaling re-measured under the SINGULAR (functional) complexity lambda*log n.

A STUDY (not a package feature), following B6. B6 tested the depth-selection prediction
L* ~ mu^{-1/(alpha+1)} with the DEFAULT depth price -- each added layer costs one unit, so
select_depth compares each per-layer marginal loss reduction against mu directly. B6's verdict was a
first-class negative: the exponent is seed-unstable in the small-net regime.

D1 gave the package a FUNCTIONAL complexity: the singular code length omega_func = lambda*log n
(machinery.singular_mdl), where lambda is the local learning coefficient of the FITTED net -- its
effective degrees of freedom, which for singular models grow far slower than the parameter count.

D6 asks the falsification question: does depth selection behave differently when a layer is priced by
the FUNCTIONAL complexity it actually adds -- Delta(lambda*log n) from depth L to L+1 -- instead of a
flat unit cost? The premise check (see below / the write-up) found lambda*log n SATURATES with depth
on a rank-1 target (roughly 8.2 -> 12.9 -> 12.0 -> 12.1 for depths 1..4): the first hidden layer adds
real functional complexity, later layers add almost none, even as the parameter count grows linearly.
So the functional per-layer cost is front-loaded and then collapses -- structurally different from the
flat unit cost, which is exactly what could move (or break) the scaling exponent.

What this study does:
  1. measure_lambda_vs_depth(): fit nets of increasing depth to a controlled target, estimate lambda
     at each converged depth, and report lambda, lambda*log n, and the per-layer functional marginal
     Delta(lambda*log n). This is the raw ingredient D6 adds.
  2. compare_depth_selection(): using a measured loss curve S(L), select depth two ways as mu sweeps --
     (a) UNIT cost (B6: marginal vs mu) and (b) FUNCTIONAL cost (marginal-loss vs mu * Delta(omega_func))
     -- and fit each L*(mu) on log-log axes. If the two exponents differ, the complexity choice
     materially changes depth selection; if the functional curve is flat/degenerate, that is the
     honest finding.

Run: python studies/d6_depth_singular_scaling.py [dataset_name]

Data note: the real-data leg reads ESOL through the package loader, which looks for
delaney-processed.csv (columns 'smiles' and 'measured log solubility in mols per litre') in the cache
dir ($ILMARINEN_DATA_DIR else <os-temp>/ilmarinen_data) or $ILMARINEN_UPLOADS_DIR before trying the network.
To run offline from a local ESOL file with columns logS,canonical_smiles, convert it once with a short
csv.DictReader -> csv.DictWriter into that schema and drop it in the cache dir (see MANIFEST v1.5.1).

Verdict is recorded in the write-up (studies-style honest reporting): whether pricing depth by the
singular complexity firms up, changes, or leaves unchanged B6's seed-unstable exponent.
"""

import warnings

warnings.filterwarnings("ignore")

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, ".")
from ilmarinen import estimate_llc  # noqa: E402
from ilmarinen.machinery.priced_depth import DepthCurve, measure_depth_curve, select_depth  # noqa: E402
from ilmarinen.machinery.singular_mdl import omega_func  # noqa: E402


# --------------------------------------------------------------------------- 1. lambda vs depth
def _rank1_regression(n=250, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = torch.tensor(rng.randn(n, d), dtype=torch.float32)
    w = np.zeros(d, np.float32)
    w[0] = 1.2
    y = torch.tensor(X.numpy() @ w, dtype=torch.float32)
    return X, y, n, d


def _mlp(depth, d, H=16):
    layers = [torch.nn.Linear(d, H), torch.nn.Tanh()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(H, H), torch.nn.Tanh()]
    layers += [torch.nn.Linear(H, 1)]
    return torch.nn.Sequential(*layers)


def measure_lambda_vs_depth(
    depths=(1, 2, 3, 4, 5), H=16, steps=1500, seed=0, n_seeds=1, lr=0.02, report_convergence=True
):
    """Fit each depth to convergence and estimate lambda; return per-depth lambda, omega_func, and the
    per-layer functional marginal Delta(omega_func).

    steps defaults to 1500 (raised from the exploratory 600) so the optimum is cleaner and lambda_hat
    is less noisy -- the LLC is only valid at a converged minimum, so under-training inflates its
    variance. With n_seeds>1 the lambda estimate is averaged over that many init seeds and a standard
    deviation is reported, so the noisiness is quantified rather than hidden."""
    X, y, n, d = _rank1_regression(seed=seed)
    rows = []
    for L in depths:
        lam_samples, final_losses = [], []
        for s in range(n_seeds):
            torch.manual_seed(seed + s)
            net = _mlp(L, d, H)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                loss = ((net(X).squeeze(-1) - y) ** 2).mean()
                loss.backward()
                opt.step()

            def closure(net=net):
                return ((net(X).squeeze(-1) - y) ** 2).mean()

            out = estimate_llc(net, closure, n=n, chains=4, steps=200, burn=60, eps=2e-5, seed=seed)
            if out.get("valid", True):
                lam_samples.append(out["lambda"])
            final_losses.append(float(loss.item()))
        k = sum(p.numel() for p in _mlp(L, d, H).parameters())
        lam_mean = float(np.mean(lam_samples)) if lam_samples else float("nan")
        lam_std = float(np.std(lam_samples)) if len(lam_samples) > 1 else 0.0
        of = omega_func(lam_mean, n) if lam_samples else float("nan")
        rows.append(
            {
                "depth": L,
                "k": k,
                "lambda": lam_mean,
                "lambda_std": lam_std,
                "omega_func": of,
                "n_valid": len(lam_samples),
                "loss": float(np.mean(final_losses)),
            }
        )
    # per-layer functional marginal
    for i in range(1, len(rows)):
        rows[i]["d_omega_func"] = rows[i]["omega_func"] - rows[i - 1]["omega_func"]
    rows[0]["d_omega_func"] = rows[0]["omega_func"]  # cost of the first layer vs a depth-0 baseline (0)

    print(
        f"lambda / functional complexity vs depth (rank-1 target, {steps} steps"
        f"{f', {n_seeds} seeds' if n_seeds > 1 else ''}):"
    )
    hdr = f"  {'depth':>5} {'k':>6} {'lambda':>8}"
    if n_seeds > 1:
        hdr += f" {'+/-std':>7}"
    hdr += f" {'omega_func':>11} {'d(omega_func)':>13} {'k/2':>7} {'loss':>10}"
    print(hdr)
    for r in rows:
        line = f"  {r['depth']:>5} {r['k']:>6} {r['lambda']:>8.3f}"
        if n_seeds > 1:
            line += f" {r['lambda_std']:>7.3f}"
        line += f" {r['omega_func']:>11.2f} {r['d_omega_func']:>13.2f} {r['k'] / 2:>7.1f} {r['loss']:>10.2e}"
        print(line)
    return rows


# --------------------------------------------------------------------------- 2. depth selection: unit vs functional
def select_depth_functional(curve: DepthCurve, mu: float, d_omega_by_mid):
    """Depth selection where layer L->L+1 is priced by mu * Delta(omega_func) at that step, instead of
    the flat unit cost of the default select_depth. Stops before the first layer whose marginal loss
    reduction fails to beat its FUNCTIONAL price.

    d_omega_by_mid: dict midpoint -> Delta(omega_func) for that layer transition (>= 0; a tiny floor is
    used so a zero functional cost does not make every layer free)."""
    floor = 1e-3
    for mid, m, _me in curve.marginals:
        price = mu * max(d_omega_by_mid.get(mid, 1.0), floor)
        if m < price:
            return int(np.floor(mid))
    return curve.depths[-1]


def compare_depth_selection(
    name="ESOL",
    task="regression",
    width=32,
    depths=(1, 2, 3, 4, 5),
    seeds=(0, 1, 2),
    H=16,
    fit_epochs=60,
    lam_steps=1500,
    lam_seeds=3,
):
    """Measure a real loss curve S(L), and select depth under UNIT vs FUNCTIONAL cost as mu sweeps.
    Fits both L*(mu) on log-log axes and reports the apparent exponents side by side.

    fit_epochs (raised to 60) trains the real AllGraph nets longer so the depth curve is less
    seed-noisy; lam_steps/lam_seeds control the functional-cost measurement (1500 steps, averaged over
    3 seeds) so the per-layer functional cost is estimated at a cleaner minimum with quantified spread."""
    # (a) real loss-vs-depth curve on an AllGraph net (same machinery + loader B6 used)
    from ilmarinen.core.allgraph import AllGraph

    try:
        from ilmarinen.core.dataset_registry import quick_suite

        d = quick_suite()[name][0](reduced=True, device="cpu")
        tr, te = d["train"], d["test"]
    except Exception as e:
        print(f"[D6] could not load {name} ({type(e).__name__}); using synthetic curve instead.")
        tr = te = None

    if tr is not None:

        def train_eval(L, sd):
            np.random.seed(sd)
            torch.manual_seed(sd)
            mg = AllGraph(width=width, depth=L, epochs=fit_epochs, verbose=False, seed=sd)
            mg.fit(tr, task=task)
            y = np.asarray(te.y)
            with torch.no_grad():
                ids = np.arange(len(te.node_feats))
                wp = mg.contract == "equivariant"
                ue = mg.contract in ("graph", "equivariant")
                out = mg._forward_contract(mg.net, te, ids, mg.contract, wp, ue, 3.0).cpu().numpy()
            p = out.squeeze(-1) if out.ndim > 1 else out
            mse = float(((p - y[: len(p)]) ** 2).mean())
            return mse, -mse  # (val_loss, val_acc-proxy)

        curve = measure_depth_curve(train_eval, list(depths), list(seeds))
    else:
        # synthetic saturating curve (loss improves then plateaus)
        S = np.array([0.5, 0.34, 0.30, 0.29, 0.288])[: len(depths)]
        marg = [
            ((depths[i - 1] + depths[i]) / 2, (S[i - 1] - S[i]) / (depths[i] - depths[i - 1]), 0.0)
            for i in range(1, len(depths))
        ]
        curve = DepthCurve(list(depths), S, np.zeros(len(depths)), np.zeros(len(depths)), marg)

    # (b) functional per-layer costs from the lambda-vs-depth measurement
    lam_rows = measure_lambda_vs_depth(depths=depths, H=H, seed=seeds[0], steps=lam_steps, n_seeds=lam_seeds)
    # map each curve midpoint (L+0.5) to Delta(omega_func) for the L->L+1 step (row index i corresponds
    # to depth depths[i]; the transition into depth depths[i] is d_omega_func of that row)
    d_omega_by_mid = {}
    for i in range(1, len(depths)):
        mid = (depths[i - 1] + depths[i]) / 2
        d_omega_by_mid[mid] = max(lam_rows[i].get("d_omega_func", 1.0), 0.0)

    mus = np.logspace(-3.5, -1.0, 12)
    L_unit = np.array([select_depth(curve, mu) for mu in mus])
    L_func = np.array([select_depth_functional(curve, mu, d_omega_by_mid) for mu in mus])

    def _fit(Ls):
        ok = Ls > 1
        if ok.sum() < 3:
            return float("nan"), int(ok.sum())
        slope = np.polyfit(np.log(mus[ok]), np.log(Ls[ok]), 1)[0]
        return slope, int(ok.sum())

    s_unit, n_unit = _fit(L_unit)
    s_func, n_func = _fit(L_func)

    print(f"\nDepth selection L*(mu) exponents on {name}:")
    print(f"  UNIT cost (B6):        slope = {s_unit:.3f}  (over {n_unit} points where L*>1)")
    print(f"  FUNCTIONAL cost (D6):  slope = {s_func:.3f}  (over {n_func} points where L*>1)")
    print(f"  L* range: unit {L_unit.min()}-{L_unit.max()}, functional {L_func.min()}-{L_func.max()}")
    print(
        "\nReading: if the functional-cost exponent differs from the unit-cost one, pricing depth by "
        "the singular complexity materially changes depth selection. If the functional per-layer cost "
        "saturates (Delta(omega_func) -> 0 after the first layer), functional pricing tends to select "
        "SHALLOWER nets at fixed mu -- deeper layers are 'free' in loss but add no functional dimension, "
        "so they are not worth even a tiny price."
    )
    return {"mus": mus, "L_unit": L_unit, "L_func": L_func, "slope_unit": s_unit, "slope_func": s_func}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ESOL"
    print("=" * 70)
    print("D6: depth scaling under the singular (functional) complexity")
    print("=" * 70)
    measure_lambda_vs_depth()
    print()
    compare_depth_selection(name=name)

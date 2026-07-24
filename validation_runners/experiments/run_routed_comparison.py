#!/usr/bin/env python
"""
run_routed_comparison.py -- does automatic tensorization (mode-structure routing) improve the
metaoptimizer's accuracy and change the architecture it selects?

Two pipelines, same datasets, same autonomous width/depth/primitive selection:
  [FIXED]  the metaoptimizer on the native SEQUENCE representation only (the status quo: every
           dataset is a (b, T, channels) sequence, sequence-primitive vocabulary).
  [ROUTED] mode-structure detection routes each dataset to the matching schema with the correct
           tensorization: 2D grids -> spatial (conv2d) schema on the discovered H x W; 1D ->
           sequence schema; unstructured -> sequence schema as a length-1 dense vector.

We report, per dataset: detected structure, chosen architecture (primitive/depth/width/params), and
test accuracy, for BOTH pipelines -- so the effect of automatic tensorization on accuracy AND on the
selected network structure is visible.

Datasets: image datasets (CIFAR-10, Fashion-MNIST) where routing should send data to conv2d, and
UCR time series where routing should keep them as sequences (control that routing does no harm).
"""
import argparse, os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ilmarinen.models.schema import build_schema
from ilmarinen.models.spatial_schema import build_spatial_schema
from ilmarinen.core.route import route_by_structure

SEQ_PRIMS = ("plain", "gated", "lstm", "conv", "spectral", "attention", "dense", "linssm", "norm")


# ---------------- data loaders (flat + labels) ----------------
def load_flat(name, seed=0, per_class=120):
    """Return (Xtr_flat, ytr, Xte_flat, yte, n_classes) as flat feature vectors."""
    if name in ("cifar10", "fmnist"):
        if name == "cifar10":
            from ilmarinen.core.cifar import CIFAR10
            Xtr, ytr = CIFAR10().balanced_subset(per_class=per_class, split="train")
            Xte, yte = CIFAR10().balanced_subset(per_class=max(40, per_class // 3), split="test")
            Xtr = np.asarray(Xtr).mean(1).reshape(len(Xtr), -1)     # grayscale
            Xte = np.asarray(Xte).mean(1).reshape(len(Xte), -1)
            ytr, yte = np.asarray(ytr), np.asarray(yte)
        else:
            from ilmarinen.core.data import FashionMNIST
            fm = FashionMNIST()
            Xtr, ytr = fm.balanced_subset(per_class=per_class, split="train")
            Xte, yte = fm.balanced_subset(per_class=max(40, per_class // 3), split="test")
            Xtr = np.asarray(Xtr).reshape(len(Xtr), -1)
            Xte = np.asarray(Xte).reshape(len(Xte), -1)
            ytr, yte = np.asarray(ytr), np.asarray(yte)
    else:
        from aeon.datasets import load_classification
        Xtr, ytr = load_classification(name, split="train")
        Xte, yte = load_classification(name, split="test")
        Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32)
        Xte = Xte.reshape(len(Xte), -1).astype(np.float32)
        cls = sorted(set(ytr)); m = {c: i for i, c in enumerate(cls)}
        ytr = np.array([m[c] for c in ytr]); yte = np.array([m[c] for c in yte])
    # per-pixel standardization with a ROBUST variance floor: near-constant pixels (e.g. image
    # borders) have tiny per-pixel std, and dividing test data by ~1e-6 explodes it (train/test
    # scale mismatch -> test collapse). Floor sd at a fraction of the global std so constant pixels
    # are effectively zeroed rather than amplified.
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    floor = 0.1 * float(Xtr.std()) + 1e-6
    sd = np.maximum(sd, floor)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr),
            torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte), len(set(ytr.tolist())))


def split_val(Xtr, ytr, val_frac, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Xtr), generator=g)
    nval = max(int(ytr.max()) + 1, int(round(val_frac * len(Xtr))))
    vi, wi = perm[:nval], perm[nval:]
    return Xtr[wi], ytr[wi], Xtr[vi], ytr[vi]


# ---------------- generic metaoptimize over a schema builder ----------------
def primitive_costs(net):
    cell = net.cells[0]
    raw = torch.tensor([sum(x.numel() for x in core.parameters()) for core in cell.cores],
                       dtype=torch.float32)
    return raw / raw.max()


def fit_once(build_fn, Xw, yw, Xv, yv, seed, epochs, mu, gamma, lr=3e-3, alpha_lr=0.02, bs=32):
    torch.manual_seed(seed)
    net = build_fn(seed)
    costs = primitive_costs(net)
    ap = [c.alpha for c in net.cells]
    wp = [p for n, p in net.named_parameters() if not n.endswith("alpha")]
    ow = torch.optim.Adam(wp, lr=lr); oa = torch.optim.Adam(ap, lr=alpha_lr)
    lf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Xw))
        for i in range(0, len(Xw), bs):
            bi = perm[i:i + bs]
            ow.zero_grad(); l = lf(net(Xw[bi]), yw[bi])
            if torch.isfinite(l):
                l.backward(); torch.nn.utils.clip_grad_norm_(wp, 5.0); ow.step()
        perm = torch.randperm(len(Xv))
        for i in range(0, len(Xv), bs):
            bi = perm[i:i + bs]
            oa.zero_grad(); la = lf(net(Xv[bi]), yv[bi])
            comp = net.cells[0].alpha.new_zeros(()); ent = net.cells[0].alpha.new_zeros(())
            for cell in net.cells:
                w = torch.softmax(cell.alpha, dim=0)
                comp = comp + (w * costs).sum(); ent = ent - (w * torch.log(w + 1e-9)).sum()
            obj = la + mu * comp + gamma * ent
            if torch.isfinite(obj):
                obj.backward(); torch.nn.utils.clip_grad_norm_(ap, 5.0); oa.step()
        net.update_peak()
    with torch.no_grad():
        va = float((net(Xv).argmax(-1) == yv).float().mean())
        vl = float(nn.CrossEntropyLoss()(net(Xv), yv))
    return va, vl, net


def metaoptimize(build_for, Xw, yw, Xv, yv, Xte, yte, widths, depths, seed, epochs, mu, gamma,
                 acc_tol, selection="priced", depth_mu=0.02, width_mu=0.02):
    """Select width and depth. Two mechanisms:

    selection='grid'   : hyperparameter grid search -- smallest (depth,width) within acc_tol of the
                         best VALIDATION ACCURACY. (Ordinary tuning; kept for ablation.)
    selection='priced' : the THEORETICAL marginal-value rule (ilmarinen.machinery.priced_depth). Builds
                         the val-LOSS curve over depths (at a reference width) and over widths (at a
                         reference depth), then applies the priced stopping rule:
                           L* = smallest depth whose per-layer marginal loss reduction < depth_mu;
                           K* = smallest width whose per-neuron marginal loss reduction < width_mu.
                         Selection is a FUNCTION OF PRICE mu, not the accuracy argmin -- it stops when
                         the next layer/neuron block is not worth its price. This is the Route-2
                         grand-canonical depth condition -dS*/dL = mu from the analytical report.
    """
    from ilmarinen.machinery.priced_depth import measure_depth_curve, select_depth, significant_elbow

    def fit(w, d):
        va, vl, net = fit_once(lambda s: build_for(w, d, s), Xw, yw, Xv, yv, seed, epochs, mu, gamma)
        return va, vl, net

    if selection == "grid":
        rows = []
        for depth in depths:
            for w in widths:
                va, vl, net = fit(w, depth)
                rows.append((depth, w, va, net))
        best = max(va for _, _, va, _ in rows)
        for d, w, va, net in sorted(rows, key=lambda r: (r[0], r[1])):
            if va >= best - acc_tol:
                with torch.no_grad():
                    ta = float((net(Xte).argmax(-1) == yte).float().mean())
                return {"depth": d, "width": w, "val": va, "test": ta, "arch": net.architecture(),
                        "params": sum(p.numel() for p in net.parameters()),
                        "selection": "grid"}

    # ---- priced marginal-value selection ----
    # 1. WIDTH at reference depth = min(depths): build the val-loss-vs-width curve, priced stop.
    ref_depth = min(depths)
    wloss, wnets = [], {}
    for w in widths:
        _, vl, net = fit(w, ref_depth); wloss.append(vl); wnets[w] = net
    Kstar = widths[-1]
    for i in range(1, len(widths)):
        dW = widths[i] - widths[i - 1]
        marg = (wloss[i - 1] - wloss[i]) / dW           # per-neuron marginal loss reduction
        if marg < width_mu:
            Kstar = widths[i - 1]; break
    # 2. DEPTH at K* via the priced_depth machinery (marginal-value loss curve + priced stop).
    depth_nets = {}
    def train_eval_depth(L, sd):
        va, vl, net = fit_once(lambda s: build_for(Kstar, L, s), Xw, yw, Xv, yv, sd, epochs, mu, gamma)
        depth_nets[L] = net
        return vl, va
    curve = measure_depth_curve(train_eval_depth, depths, seeds=[seed])
    Lstar = select_depth(curve, depth_mu)
    if Lstar not in depth_nets:                          # ensure we have the selected net
        train_eval_depth(Lstar, seed)
    net = depth_nets.get(Lstar, wnets[Kstar])
    with torch.no_grad():
        ta = float((net(Xte).argmax(-1) == yte).float().mean())
    return {"depth": Lstar, "width": Kstar, "val": float(curve.acc_mean[list(depths).index(Lstar)])
            if Lstar in depths else 0.0, "test": ta, "arch": net.architecture(),
            "params": sum(p.numel() for p in net.parameters()),
            "selection": "priced", "depth_marginals": [(round(m[0], 1), round(m[1], 4))
                                                        for m in curve.marginals]}



def run(args):
    widths = [int(x) for x in args.widths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    Xtr, ytr, Xte, yte, n_out = load_flat(args.dataset, args.seed, args.per_class)
    Xw, yw, Xv, yv = split_val(Xtr, ytr, args.val_frac, args.seed)
    maj = float(np.bincount(yte.numpy(), minlength=n_out).max() / len(yte))
    d = Xtr.shape[1]
    sel = args.selection
    print(f"\n########## {args.dataset} (n_train={len(Xtr)}, dim={d}, maj={maj:.3f}) "
          f"[selection={sel}] ##########", flush=True)

    # ---- OPTIONAL symmetry preprocessing (NOT default; --symmetry to enable) ----
    # Runs scale-aware symmetry detection on the flattened TRAIN features; if a genuine symmetry is
    # found, the discovered quotient reduce_fn is applied to all splits before routing. Off by default
    # per project policy (symmetry detection is an opt-in preprocessing stage, not part of the default
    # pipeline).
    if args.symmetry:
        from ilmarinen.core.symmetry_pipeline import discover_and_reduce
        sym = discover_and_reduce(Xw, yw.float(), n_refits=2, epochs=args.sym_epochs,
                                  coordinate_structure="unknown", scale_aware=True, verbose=False)
        found = (f"cont={[k for k,_ in sym['continuous']]}, z2={len(sym['z2'])}, "
                 f"perm={sym['permutation']['young_subgroup']}")
        red = sym["reduce_fn"]
        d0 = Xw.shape[1]
        Xw, Xv, Xte = red(Xw), red(Xv), red(Xte)
        print(f"[SYM   ] detected: {found}; feature dim {d0}->{Xw.shape[1]}", flush=True)

    def mopt(build_for, Xw_, Xv_, Xte_):
        return metaoptimize(build_for, Xw_, yw, Xv_, yv, Xte_, yte, widths, depths, args.seed,
                            args.epochs, args.mu, args.gamma, args.acc_tol, selection=sel,
                            depth_mu=args.depth_mu, width_mu=args.width_mu)

    # [FIXED] sequence representation, long inputs pooled to max_seq steps
    def to_seq(Z):
        T = Z.shape[1]
        if args.max_seq and T > args.max_seq:
            step = T // args.max_seq; k = (T // step) * step
            Z = Z[:, :k].reshape(len(Z), k // step, step).mean(-1)
        return Z.reshape(len(Z), Z.shape[1], 1)
    def build_seq(width, depth, seed):
        return build_schema(depth=depth, width=width, n_in=1, n_out=n_out,
                                        seed=seed, primitives=SEQ_PRIMS, readout=args.readout)
    Xw_s, Xv_s, Xte_s = to_seq(Xw), to_seq(Xv), to_seq(Xte)
    F = mopt(build_seq, Xw_s, Xv_s, Xte_s)
    mtag = f" marg={F.get('depth_marginals')}" if sel == "priced" else ""
    print(f"[FIXED ] seq repr: arch={F['arch']} d{F['depth']} w{F['width']} "
          f"params={F['params']} TEST={F['test']:.3f}{mtag}", flush=True)

    # [ROUTED] auto-tensorization
    route = route_by_structure(Xw, yw, verbose=False)
    print(f"[ROUTED] detected structure={route['structure']} shape={route['shape']} "
          f"-> {route['kind']} schema", flush=True)
    Xw_r, Xv_r, Xte_r = route["tensorize"](Xw), route["tensorize"](Xv), route["tensorize"](Xte)
    hint = route["build_hint"]
    if route["kind"] == "spatial":
        H, W = route["shape"]
        if args.pool_hw and max(H, W) > args.pool_hw:
            import torch.nn.functional as _F
            def _pool(Z):
                return _F.adaptive_avg_pool2d(Z, (args.pool_hw, args.pool_hw))
            Xw_r, Xv_r, Xte_r = _pool(Xw_r), _pool(Xv_r), _pool(Xte_r)
            H = W = args.pool_hw
        def build_routed(width, depth, seed):
            return build_spatial_schema(width=width, hw=max(H, W), depth=depth,
                                                    n_in=1, n_classes=n_out, seed=seed,
                                                    primitives=hint["primitives"], img_size=max(H, W))
    else:
        n_in_r = Xw_r.shape[2]
        seq_prims = tuple(p for p in hint["primitives"] if p in SEQ_PRIMS) or SEQ_PRIMS
        def build_routed(width, depth, seed):
            return build_schema(depth=depth, width=width, n_in=n_in_r, n_out=n_out,
                                            seed=seed, primitives=seq_prims, readout=args.readout)
    R = mopt(build_routed, Xw_r, Xv_r, Xte_r)
    mtag = f" marg={R.get('depth_marginals')}" if sel == "priced" else ""
    print(f"[ROUTED] arch={R['arch']} d{R['depth']} w{R['width']} "
          f"params={R['params']} TEST={R['test']:.3f}{mtag}", flush=True)
    print(f"[DELTA ] FIXED {F['test']:.3f} -> ROUTED {R['test']:.3f} "
          f"({R['test']-F['test']:+.3f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.03)
    ap.add_argument("--acc_tol", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per_class", type=int, default=120)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--max_seq", type=int, default=64)
    ap.add_argument("--pool_hw", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", default="priced", choices=["priced", "grid"],
                    help="priced=theoretical marginal-value rule; grid=hyperparameter search")
    ap.add_argument("--depth_mu", type=float, default=0.02, help="depth price (marginal-value stop)")
    ap.add_argument("--width_mu", type=float, default=0.02, help="width price (marginal-value stop)")
    ap.add_argument("--symmetry", action="store_true",
                    help="OPT-IN: run scale-aware symmetry preprocessing before routing (off by default)")
    ap.add_argument("--sym_epochs", type=int, default=100)
    ap.add_argument("--readout", default="mean", choices=["mean","last","flatten"],
                    help="sequence readout: mean/last collapse time; flatten is position-aware "
                         "(good for short effectively-tabular series like ItalyPowerDemand)")
    run(ap.parse_args())

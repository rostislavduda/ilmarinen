"""Validation pipelines: reproducible experiments over the three machinery pieces.

Each pipeline returns a structured result dict and can print a report. These
are the consolidated, de-noised versions of the proof-of-concept experiments.
"""
from __future__ import annotations

import numpy as np

from ..core.data import FashionMNIST
from ..core.meanfield import MeanFieldTheory
from ..machinery.priced_depth import measure_depth_curve, select_depth, significant_elbow
from ..machinery.width_sparsity import greedy_insertion
from .networks import build_model
from .training import gradient_norms_at_init, to_tensor, train_and_eval


# ----------------------------------------------------------------------
# Pipeline 1: width-sparsity certificate (exact regime)
# ----------------------------------------------------------------------
def validate_width_sparsity(data: FashionMNIST, cls_a=0, cls_b=6,
                            lambdas=(0.10, 0.06, 0.03), verbose=True):
    Xtr, ytr, Xte, yte = data.binary_task(cls_a, cls_b)
    results = {}
    for lam in lambdas:
        res = greedy_insertion(Xtr, ytr, Xte, yte, lam=lam, max_neurons=150)
        results[lam] = res
        if verbose:
            print(f"  lambda={lam:4.2f}: K={res.K:3d} neurons  "
                  f"train_acc={res.train_acc:.3f}  test_acc={res.test_acc:.3f}  "
                  f"final_max_corr={res.final_max_corr:.3f}")
    return results


# ----------------------------------------------------------------------
# Pipeline 2: criticality diagnostic + init-time gradient propagation
# ----------------------------------------------------------------------
def validate_criticality(data: FashionMNIST, activation="tanh", sigma_b2=0.05,
                         model_kind="resnet_mlp", depths=(10, 30, 50, 80),
                         width=128, verbose=True):
    theory = MeanFieldTheory(activation)
    sw2_crit = theory.critical_sigma_w2(sigma_b2)
    inits = {"ordered": 1.0, "critical": round(sw2_crit, 3), "chaotic": 2.5}

    Xsub, ysub = data.balanced_subset(per_class=800)
    Xt, yt = to_tensor(Xsub, ysub)

    grad_table, phases = {}, {}
    for name, sw2 in inits.items():
        phases[name] = theory.classify(sw2, sigma_b2)
        grad_table[name] = {}
        for d in depths:
            model = build_model(model_kind, depth=d, width=width, sigma_w2=sw2)
            g0, gL = gradient_norms_at_init(model, Xt, yt)
            grad_table[name][d] = (g0, gL)

    if verbose:
        print(f"  critical sigma_w^2 (chi_1=1) = {sw2_crit:.4f}")
        for name, pr in phases.items():
            xi = f"{pr.xi:.1f}" if np.isfinite(pr.xi) else "inf/na"
            print(f"  {name:8s} sw2={inits[name]:<5} chi1={pr.chi1:.3f} "
                  f"c*={pr.c_star:.3f} xi={xi}")
        print("  first-layer grad norm at init (signal propagation probe):")
        for name in inits:
            row = "  ".join(f"d{d}:{grad_table[name][d][0]:.1e}" for d in depths)
            print(f"    {name:8s} {row}")
    return {"sw2_crit": sw2_crit, "inits": inits, "phases": phases, "grad_table": grad_table}


# ----------------------------------------------------------------------
# Pipeline 3: priced depth (multi-seed, denoised)
# ----------------------------------------------------------------------
def validate_priced_depth(data: FashionMNIST, model_kind="resnet_mlp",
                          depths=(1, 2, 4, 8, 16, 32), seeds=(0, 1, 2),
                          width=128, sigma_w2=1.76, epochs=15,
                          prices=(0.05, 0.02, 0.01, 0.005, 0.002), verbose=True):
    Xsub, ysub = data.balanced_subset(per_class=800, split="train")
    Xval_np, yval_np = data.balanced_subset(per_class=400, split="test")
    Xtr, ytr = to_tensor(Xsub, ysub)
    Xval, yval = to_tensor(Xval_np, yval_np)

    def train_eval_fn(depth, seed):
        model = build_model(model_kind, depth=depth, width=width,
                            sigma_w2=sigma_w2, seed=seed)
        vl, va, _ = train_and_eval(model, Xtr, ytr, Xval, yval, epochs=epochs)
        return vl, va

    curve = measure_depth_curve(train_eval_fn, list(depths), list(seeds))
    selections = {mu: select_depth(curve, mu) for mu in prices}
    elbow = significant_elbow(curve)

    if verbose:
        print("  S*(L) mean +- SE  (val_acc):")
        for i, L in enumerate(curve.depths):
            print(f"    L={L:3d}: {curve.S_mean[i]:.4f} +- {curve.S_se[i]:.4f}  "
                  f"acc={curve.acc_mean[i]:.3f}")
        print(f"  significant-elbow depth (marginal > 2 SE): L={elbow}")
        for mu, Lstar in selections.items():
            print(f"    price mu={mu:.3f} -> L*={Lstar}")
    return {"curve": curve, "selections": selections, "elbow": elbow}


# ----------------------------------------------------------------------
# Pipeline 4: sequential (recurrent) baseline -- the depth-limited task
# ----------------------------------------------------------------------
def validate_sequential_baseline(data: FashionMNIST, per_class_tr=300, per_class_te=100,
                                 width=128, epochs=8, lr=0.003,
                                 inits=(("ordered", 1.2), ("critical", None), ("chaotic", 2.5)),
                                 verbose=True):
    """Plain RNN on sequential Fashion-MNIST (T=784): the depth-limited baseline.

    Establishes the quantified ceiling that gating must break. Runs the plain
    tanh RNN at ordered/critical/chaotic init and reports val accuracy. Expected
    finding: plain recurrence caps ~0.30 (well below the ~0.90 the task allows),
    criticality gives only a weak edge -- the ~60-point gap is 'gating-shaped',
    motivating the supergraph's first primitive-selection decision.

    inits: list of (name, sigma_w2); sigma_w2=None means use the critical value.
    """
    import numpy as np
    import torch

    from ..core.meanfield import MeanFieldTheory
    from .recurrent import build_rnn
    from .training import train_and_eval_rnn

    th = MeanFieldTheory("tanh")
    sw2_crit = th.critical_sigma_w2(0.05)

    Xtr_np, ytr_np = data.sequential_subset(per_class=per_class_tr, split="train")
    Xte_np, yte_np = data.sequential_subset(per_class=per_class_te, split="test")
    Xtr = torch.tensor(Xtr_np); ytr = torch.tensor(ytr_np)
    Xte = torch.tensor(Xte_np); yte = torch.tensor(yte_np)

    results = {}
    if verbose:
        print(f"  sequential Fashion-MNIST T=784: train {tuple(Xtr.shape)}, "
              f"chance=0.100, achievable(gated)~0.90")
    for name, sw2 in inits:
        sw2 = sw2_crit if sw2 is None else sw2
        pr = th.classify(sw2, 0.05)
        net = build_rnn("plain_rnn", depth=1, width=width, sigma_w2=sw2, n_in=1, n_out=10)
        vl, va, ta = train_and_eval_rnn(net, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr, bs=64)
        results[name] = {"sigma_w2": sw2, "chi1": pr.chi1, "xi_t": pr.xi,
                         "val_acc": va, "val_loss": vl}
        if verbose:
            xi = f"{pr.xi:.1f}" if np.isfinite(pr.xi) else ("inf" if pr.phase == "critical" else "n/a")
            va_s = "nan" if va != va else f"{va:.3f}"
            print(f"    {name:8s} sw2={sw2:5.2f} chi1={pr.chi1:.3f} xi_t={xi:>5} -> val_acc={va_s}")
    return results


# ----------------------------------------------------------------------
# Pipeline 5: supergraph primitive selection + discretization (Step 3)
# ----------------------------------------------------------------------
def validate_supergraph(data: FashionMNIST, per_class_tr=250, per_class_te=100,
                        width=128, search_epochs=5, finetune_epochs=5,
                        lr=0.003, verbose=True):
    """2-primitive supergraph (plain|gated recurrence) on sequential Fashion-MNIST.

    The full analytically-faithful flow:
      1. soft search: train supergraph, alpha selects a primitive per layer
      2. discretize: keep alpha-argmax primitive, inherit its trained weights
      3. fine-tune the discrete net

    Ground-truth check: on this depth-limited task gating is correct. Expected:
    alpha -> gated; discretized+finetuned recovers pure-gated performance,
    beating the diluted soft mixture (confirming 'soft search selects,
    discretization delivers').
    """
    import torch
    import torch.nn as nn

    from ..core.meanfield import MeanFieldTheory
    from .supergraph import build_supergraph, discretize

    sw2 = MeanFieldTheory("tanh").critical_sigma_w2(0.05)
    Xtr_np, ytr_np = data.sequential_subset(per_class=per_class_tr, split="train")
    Xte_np, yte_np = data.sequential_subset(per_class=per_class_te, split="test")
    Xtr = torch.tensor(Xtr_np); ytr = torch.tensor(ytr_np)
    Xte = torch.tensor(Xte_np); yte = torch.tensor(yte_np)

    def train(net, epochs, lr=lr, bs=64, clip=5.0):
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss(); n = len(Xtr)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                bi = perm[i:i + bs]; opt.zero_grad()
                l = lossf(net(Xtr[bi]), ytr[bi])
                if not torch.isfinite(l): return float("nan")
                l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), clip); opt.step()
        with torch.no_grad():
            return float((net(Xte).argmax(1) == yte).float().mean())

    sg = build_supergraph(depth=1, width=width, sigma_w2=sw2, n_in=1, n_out=10)
    acc_soft = train(sg, search_epochs)
    alpha = sg.alpha_report()[0]
    choice = sg.selected_primitive()

    disc = discretize(sg)
    with torch.no_grad():
        acc_warm = float((disc(Xte).argmax(1) == yte).float().mean())
    acc_disc = train(disc, finetune_epochs)

    if verbose:
        print(f"  soft search: acc={acc_soft:.3f}, alpha=[plain {alpha[0]:.3f}, gated {alpha[1]:.3f}], "
              f"selected={choice}")
        print(f"  discretized: warm-start acc={acc_warm:.3f} -> fine-tuned acc={acc_disc:.3f}")
        print(f"  discretization gain: {acc_disc - acc_soft:+.3f}")
    return {"acc_soft": acc_soft, "acc_warm": acc_warm, "acc_disc": acc_disc,
            "alpha": alpha, "selected": choice}


# ----------------------------------------------------------------------
# Pipeline 5: supergraph primitive selection (the architecture-search loop)
# ----------------------------------------------------------------------
def _make_copy(n, K=4, delay=30, S=6, seed=0):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    L = K + delay + K; V = S + 2
    X = np.zeros((n, L, V), np.float32); Y = np.zeros((n, K), np.int64)
    for i in range(n):
        s = rng.integers(1, S + 1, size=K); Y[i] = s - 1
        for t in range(K): X[i, t, s[t]] = 1.0
        for t in range(K, K + delay): X[i, t, 0] = 1.0
        for t in range(K + delay, L): X[i, t, S + 1] = 1.0
    return torch.tensor(X), torch.tensor(Y), V


def validate_supergraph_copy(depth=1, width=96, K=4, delay=30, S=6,
                             epochs=22, verbose=True):
    """The recurrent architecture-search loop on the copy task (large gating margin).

    Runs the full validated pipeline through the framework classes:
      bilevel select (w on train-split, alpha on held-out) -> peak-alpha ->
      discretize to the winner -> fine-tune -> honest independent-test accuracy.

    Expected (ground truth: copy needs gating): alpha selects gated decisively,
    soft ~0.98-0.99, discretize+finetune recovers ~0.99. Validates:
      - separate-state per primitive (shared mixed state stalls at chance)
      - shared readout head (separate heads make alpha degenerate)
      - product-paths for depth>1 (clean state across time AND depth)
      - peak-alpha (moderate-margin alpha decays after a mid-training peak)

    ROBUSTNESS CAVEAT: depth=1 selection is robust across seeds. depth=2 is
    seed-sensitive: across 4 seeds it selected gated 3/4 times (reaching ~0.999),
    but one seed's last-layer alpha landed on plain despite the soft model using
    gating (soft ~0.998). The first layer's alpha is never identified (stays 0.5)
    -- the documented layer-identifiability caveat (only readout-adjacent layers
    get selection signal). So depth>1 selection is NOT yet reliable and is
    reported here as a known limitation, not a validated-robust result.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    from ..core.meanfield import MeanFieldTheory
    from ..machinery import three_way_split
    from .supergraph import build_supergraph, discretize

    sw2 = MeanFieldTheory("tanh").critical_sigma_w2(0.05)
    torch.manual_seed(0); np.random.seed(0)
    Xtr, Ytr, V = _make_copy(2000, K, delay, S, seed=0)
    Xte, Yte, _ = _make_copy(400, K, delay, S, seed=1)
    Xw, yw, Xa, ya = three_way_split(Xtr, Ytr, 0.5, 0)
    Xfull = torch.cat([Xw, Xa]); yfull = torch.cat([yw, ya])

    net = build_supergraph(depth=depth, width=width, sigma_w2=sw2, n_in=V, n_out=S)
    ap = [c.alpha for c in net.cells]
    wp = [p for n, p in net.named_parameters() if not n.endswith("alpha")]
    ow = torch.optim.Adam(wp, lr=0.005); oa = torch.optim.Adam(ap, lr=0.02)
    lf = nn.CrossEntropyLoss(); nw, na = len(Xw), len(Xa); bs = 64

    def ev(m):
        with torch.no_grad():
            return float((m.forward_seq_readout(Xte, K).argmax(-1) == Yte).float().mean())

    for _ in range(epochs):
        pw = torch.randperm(nw); pa = torch.randperm(na); aptr = 0
        for i in range(0, nw, bs):
            abi = pa[aptr:aptr + bs]; aptr = (aptr + bs) % max(na - bs, 1)
            if len(abi) > 0:
                oa.zero_grad(); o = net.forward_seq_readout(Xa[abi], K)
                la = lf(o.reshape(-1, S), ya[abi].reshape(-1))
                if torch.isfinite(la):
                    la.backward(); torch.nn.utils.clip_grad_norm_(ap, 5.0); oa.step()
            wbi = pw[i:i + bs]; ow.zero_grad(); o = net.forward_seq_readout(Xw[wbi], K)
            lw = lf(o.reshape(-1, S), yw[wbi].reshape(-1))
            lw.backward(); torch.nn.utils.clip_grad_norm_(wp, 5.0); ow.step()
        net.update_peak()

    acc_soft = ev(net)
    peak = net.alpha_peak_report()[-1]
    d = discretize(net); acc_raw = ev(d)
    # fine-tune the discretized stack
    opt = torch.optim.Adam(d.parameters(), lr=0.003); n = len(Xfull)
    for _ in range(12):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bi = perm[i:i + bs]; opt.zero_grad()
            o = d.forward_seq_readout(Xfull[bi], K); l = lf(o.reshape(-1, S), yfull[bi].reshape(-1))
            if not torch.isfinite(l): break
            l.backward(); torch.nn.utils.clip_grad_norm_(d.parameters(), 5.0); opt.step()
    acc_ft = ev(d)

    if verbose:
        print(f"  copy task depth={depth}: soft={acc_soft:.3f}, choice={d.choice}, "
              f"peak_alpha_gated={peak[1]:.3f}")
        print(f"    discretize {acc_raw:.3f} -> finetune {acc_ft:.3f}")
    return {"soft": acc_soft, "choice": d.choice, "peak_alpha_gated": float(peak[1]),
            "discretized_raw": acc_raw, "discretized_finetuned": acc_ft}

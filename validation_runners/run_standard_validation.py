#!/usr/bin/env python
"""
run_standard_validation.py -- FULL benchmark of the ilmarinen AllGraph across EVERY real dataset
investigated in the project, on complete data, for your own machine. Uses the shared dataset registry
(core/dataset_registry) and the official dataset packages where available (medmnist, deepchem+rdkit,
jetnet, aeon) with local-file fallbacks.

DEVICE: auto-detects Apple Silicon (M1/M2/M3) Metal GPU via MPS, else CUDA, else CPU. Override with
--device {mps,cuda,cpu}. On M1, MPS gives a large speedup for the conv (spatial/volumetric) and dense
paths; some ops may fall back to CPU (PYTORCH_ENABLE_MPS_FALLBACK=1 is set automatically).

Every dataset is auto-routed by AllGraph, trained on full data, evaluated on the held-out test split,
and reported on the consistent skill axis (acc-normalized / R2), with the SOTA reference alongside.

DEPENDENCIES (install what you need):
    pip install torch numpy
    pip install medmnist              # BloodMNIST, OrganMNIST3D
    pip install deepchem rdkit         # ESOL, Tox21 (+ other MoleculeNet)
    pip install jetnet                 # JetNet (or use local .hdf5)
    pip install aeon                   # UCR sequence datasets
    # QM7/QM9/rMD17: place the files where the registry expects, or pass --data_dir

USAGE:
    python run_standard_validation.py                      # every dataset, auto device
    python run_standard_validation.py --device mps         # force Apple Silicon GPU
    python run_standard_validation.py --only ESOL,rMD17-ethanol,JetNet
    python run_standard_validation.py --contracts graph,equivariant
    python run_standard_validation.py --epochs_scale 2.0   # longer training
"""
import argparse, gc, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # let unsupported MPS ops fall back to CPU
from ilmarinen.core.allgraph import AllGraph
from ilmarinen.core.dataset_registry import full_suite

# full-data budgets (width, depth, epochs) per contract. Epoch budget is a uniform 100 for every contract
# (a common cap); pair with --auto_epoch to stop early once a model's training plateaus.
BUDGET = {
    "sequence":    dict(width=64, depth=1, epochs=100),   # depth auto-deepened for LONG series (_seq_depth_for)
    "spatial":     dict(width=32, depth=2, epochs=100),
    "volumetric":  dict(width=16, depth=2, epochs=100),
    "4d":          dict(width=16, depth=2, epochs=100),
    "graph":       dict(width=64, depth=3, epochs=100),
    "equivariant": dict(width=32, depth=3, epochs=100),
    "set":         dict(width=64, depth=2, epochs=100),
    "operator":    dict(width=24, depth=3, epochs=100),
}


def resolve_device(pref):
    if pref and pref != "auto":
        return pref
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _auc(scores, labels):
    """Binary ROC-AUC as the Mann-Whitney U statistic: P(score(pos) > score(neg)). `labels` is a 0/1
    array (positives are label==1); `scores` is the positive-class score. NaN when either class is absent."""
    order = np.argsort(scores); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(scores))
    pos = labels == 1; npos = int(pos.sum()); nneg = len(labels) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return (ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg)


def _binary_scores(out):
    """Positive-class (signal) score for a binary head: the raw logit for a 1-column head, else the softmax
    P(class 1) for a 2-column head. Monotone in P(signal), so it is a valid ranking score for the threshold/
    rank metrics (ROC-AUC, background rejection); the sigmoid is omitted since those metrics are rank-based,
    not probability-calibrated. Higher score = more signal-like, consistent with label==1 being the positive."""
    return out[:, 0].numpy() if out.shape[1] == 1 else out.softmax(1)[:, 1].numpy()


def _roc_auc(out, y):
    """ROC-AUC for a classification head, dispatching on the number of output columns. A binary head
    (<=2 columns) reduces to the Mann-Whitney `_auc` of the positive-class score. A multiclass head (K>2)
    uses MACRO one-vs-rest: the unweighted mean of the K per-class binary AUCs, where class c's AUC ranks its
    softmax column against the (label==c) indicator. This is the standard multiclass ROC-AUC (sklearn
    `multi_class="ovr"`, average="macro"; the MedMNIST leaderboard's AUC). Classes absent from the test split
    are skipped so a missing label can't poison the mean. `out` is the (n, K) logit tensor, `y` int labels."""
    y = np.asarray(y)
    if out.shape[1] <= 2:
        return float(_auc(_binary_scores(out), y))
    probs = out.softmax(1).numpy()
    per_class = [_auc(probs[:, c], (y == c).astype(int))
                 for c in range(out.shape[1]) if 0 < int((y == c).sum()) < len(y)]
    return float(np.mean(per_class)) if per_class else float("nan")


def _bkg_rejection(scores, labels, eps_s=0.3):
    """Background rejection 1/eps_B at a fixed signal efficiency eps_S -- the standard top-tagging figure of
    merit (Kasieczka et al. 2019, arXiv:1902.09914). `scores` is the signal-class score (higher = more
    signal-like), `labels` a 0/1 array (1=signal/top, 0=background/QCD). The threshold tau is the k-th largest
    signal score with k=ceil(eps_S * N_S), so a fraction ~eps_S of SIGNAL passes -- a COUNT-EXACT working point
    that pins the signal efficiency to k/N_S without quantile-interpolation or tie bias on the signal side.
    eps_B is then the fraction of BACKGROUND with score >= tau, and the rejection is 1/eps_B. It is reported as
    N_B / max(n_bkg_pass, 1): this equals 1/eps_B whenever any background survives, and floors the zero-survivor
    case to N_B -- the largest rejection a background sample of size N_B can resolve (one surviving mistag) --
    so the value stays finite/JSON-safe and honestly sample-capped rather than a spurious infinity. (On the
    reduced quick run N_B ~ a few hundred, so the number saturates well below literature 1/eps_B; it is a smoke
    metric there, not a benchmark.) Non-finite scores are dropped first so a diverged logit can't mask itself."""
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels)
    finite = np.isfinite(scores); scores, labels = scores[finite], labels[finite]
    sig = scores[labels == 1]; bkg = scores[labels == 0]
    if len(sig) == 0 or len(bkg) == 0:
        return float("nan")
    k = min(max(int(np.ceil(eps_s * len(sig))), 1), len(sig))   # number of top signal jets to admit
    tau = np.partition(sig, len(sig) - k)[len(sig) - k]         # k-th largest signal score = the threshold
    n_pass = int((bkg >= tau).sum())                            # background mistags at that threshold
    return float(len(bkg)) / max(n_pass, 1)


def _eval_test(mg, test, task, rotated=False, auc=False, report_auc=False, bg_rejection=None, dense_bs=256,
               relational_bs=128, target_scale=None, target_units=None):
    """Evaluate a fitted model on the held-out test split, returning (metric_name, value, extra) where
    `extra` is a list of (name, value) SECONDARY metrics (empty for most datasets). Shared by the standard and
    quick runners (which import this one copy); the two batch sizes are the only knob that varies by runner.
    (The cellpainting runners keep their own eval paths and do not call this.) The multi-metric return lets a dataset report the field-standard headline plus a
    secondary column -- e.g. QM/MLIP energy regression reports physical-unit MAE (headline) with R2 as a
    scale-free secondary, since R2 ~0.999+ carries no ranking information there. `target_scale`/`target_units`
    (from the loader) supply the de-normalization factor when the loader z-scored the target."""
    net = mg.net; mod = mg.contract; dev = mg.device
    if getattr(mg, "_canonicalization_applied", False) and mod not in ("sequence", "spatial", "volumetric", "4d"):
        test = mg.apply_canonicalization(test)
    y = np.asarray(test.y)
    if mod == "operator":
        # neural-operator contract: function-on-a-grid -> function; test data carries (dense, grid), not
        # node_feats, so it needs its own eval path (mirrors run_quick_validation._eval_test).
        a = test.dense if isinstance(test.dense, torch.Tensor) else torch.tensor(np.asarray(test.dense), dtype=torch.float32)
        xg = test.grid if isinstance(test.grid, torch.Tensor) else torch.tensor(np.asarray(test.grid), dtype=torch.float32)
        with torch.no_grad():
            pred = net(a.to(dev), xg.to(dev)).cpu().numpy()
        return "field_R2", float(1 - ((pred - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)), []
    if mod == "generated_equivariant":
        # A DISCOVERED-GROUP contract (e.g. Lorentz O(1,3) on JetMassLorentz under --discover extended, i.e.
        # --preset med|max) is per-datum: each sample's own point set -> group invariants, so there is no
        # batched relational forward to call. Ask the model to replay the forward its fit published.
        out = mg.forward_generated_equivariant(test)
    elif mod == "latent_equivariant":
        # A nonlinear latent contract (--deploy_nonlinear_contract) encodes each sample's flattened cloud;
        # replay that transform rather than the relational batched forward.
        out = mg.forward_latent_equivariant(test)
    elif mod in ("sequence", "spatial", "volumetric", "4d"):
        X = test.dense
        if mod == "sequence" and X.dim() == 2: X = X.unsqueeze(-1)
        if mod == "spatial" and X.dim() == 3: X = X.unsqueeze(1)
        if mod == "volumetric" and X.dim() == 4: X = X.unsqueeze(1)
        bs = 64 if mod == "volumetric" else dense_bs
        outs = []
        for j in range(0, len(X), bs):
            with torch.no_grad():
                if mod == "sequence":
                    outs.append(net.forward_seq_readout(X[j:j+bs].to(dev), 1).squeeze(1).cpu())
                else:
                    outs.append(net(X[j:j+bs].to(dev)).cpu())
        out = torch.cat(outs)
    else:
        R = None
        if rotated:
            A = np.random.RandomState(1).randn(3, 3); Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0: Q[:, 0] = -Q[:, 0]
            R = torch.tensor(Q, dtype=torch.float32)
        outs = []; n = len(test.node_feats)
        for j in range(0, n, relational_bs):
            ids = np.arange(j, min(j+relational_bs, n)); xs, eis, ps, batch = [], [], [], []; off = 0
            for gi, i in enumerate(ids):
                nf = test.node_feats[i]; k = nf.shape[0]
                xs.append(torch.as_tensor(nf, dtype=torch.float32)); batch.append(torch.full((k,), gi, dtype=torch.long))
                if mod == "equivariant":
                    p = torch.as_tensor(test.positions[i], dtype=torch.float32); ps.append(p @ R.T if R is not None else p)
                if mod in ("graph", "equivariant"):
                    eis.append(torch.as_tensor(test.edges[i], dtype=torch.long) + off)
                off += k
            x = torch.cat(xs).to(dev); b = torch.cat(batch).to(dev); ng = len(ids)
            with torch.no_grad():
                if mod == "set": outs.append(net(x, b, ng).cpu())
                elif mod == "graph": outs.append(net(x, torch.cat(eis, 1).to(dev), b, ng).cpu())
                else: outs.append(net(x, torch.cat(ps).to(dev), torch.cat(eis, 1).to(dev), b, ng).cpu())
        out = torch.cat(outs)
    if task == "classification":
        acc = float((out.argmax(1).numpy() == y).mean())
        secondary = []
        if bg_rejection and out.shape[1] <= 2:
            # binary-only fixed-working-point background rejection 1/eps_B (top-tagging figure of merit); the
            # loader supplies a list of signal-efficiency operating points, e.g. [0.3, 0.5]. Guarded to binary
            # heads: a single signal-vs-background threshold is undefined for a multiclass head.
            sc = _binary_scores(out)
            secondary += [(f"1/eB@eS{eff:g}", _bkg_rejection(sc, y, eff)) for eff in bg_rejection]
        if auc:
            # ROC-AUC is the field-standard HEADLINE (Tox21/MoleculeNet, where the imbalanced positives make
            # bare accuracy misleading); accuracy still rides along as a secondary column. _roc_auc dispatches
            # binary-vs-macro-OvR on the head width, so a multiclass auc=True dataset reports proper macro-OvR.
            return "ROC-AUC", _roc_auc(out, y), [("acc", acc)] + secondary
        if report_auc:
            # accuracy stays the headline (these datasets' SOTA is quoted as acc), with macro one-vs-rest
            # ROC-AUC as a companion column -- the leaderboard-standard second number for MedMNIST/JetNet.
            return "acc", acc, [("ROC-AUC", _roc_auc(out, y))] + secondary
        return "acc", acc, secondary
    pred = out.squeeze(-1).numpy()
    r2 = float(1 - ((pred - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))
    if target_scale is not None:
        # physical-unit MAE is the field-standard headline for QM/MLIP energy regression (R2 ~0.999+ is
        # non-discriminative there); the loader z-scored the target, so multiply the standardized MAE by the
        # retained scale to recover physical units. R2 stays as the scale-free secondary + the skill axis.
        mae = float(np.abs(pred - y).mean() * target_scale)
        return f"MAE[{target_units}]", mae, [("R2", r2)]
    return "R2", r2, []


def _apply_preset(args):
    """Apply an evidence-based processing-layer preset, unless the user set the flag explicitly on the CLI.
    Shared by all three runners. Presets encode the 9-dataset ablation conclusions:
      min -> everything off (plain baseline).
      med -> the stable + cheap set that only ever helped: sparse readout (safe on all 9 datasets; cuts
             variance on noisy sets at ~1x cost) + the routing/contract insurance layers (symmetry_routing,
             canonicalize, discover) which are correct no-ops when unneeded and high-value on mis-routed
             geometry. Deliberately EXCLUDES gibbs (hurts on clean data), tiebreak (can backfire on
             geometric data), and select_size (10x-to-prohibitive cost).
      max -> every processing layer engaged (gibbs + tiebreak + symmetry_routing + canonicalize + discover +
             select_size) -- the most expensive, most-engaged config.
      opt -> PER-SCHEMA optima (see _OPT_FLAGS / apply_opt_preset). NOT a global flag set -- the per-contract
             optima conflict (select_size helps graph/equivariant but hurts set; gibbs helps operator but hurts
             4d/set), so 'opt' is applied per-dataset by expected contract in the train loop, NOT here.
    Explicit CLI flags always win: a preset only fills options the user did not set. select_size is a mode
    string ('sequential' | 'area' | 'variable') or None (off) -- 'sequential' is the priced width-then-depth
    selector."""
    if not args.preset or args.preset == "opt":
        return                                            # 'opt' is contract-specific -> applied per-dataset, not globally
    import sys
    given = {tok[2:].split("=")[0] for tok in sys.argv[1:] if tok.startswith("--")}
    presets = {
        "min": dict(select="argmax", sparsity_mu=0.0, tiebreak=False, symmetry_routing=False,
                    canonicalize=False, discover=None, select_size=None),
        "med": dict(select="sparse", sparsity_mu=0.3, tiebreak=False, symmetry_routing=True,
                    canonicalize=True, discover="extended", select_size=None),
        "max": dict(select="gibbs", sparsity_mu=0.3, tiebreak=True, symmetry_routing=True,
                    canonicalize=True, discover="extended", select_size="sequential"),
    }
    for k, v in presets[args.preset].items():
        if k not in given:
            setattr(args, k, v)


def _seq_depth_for(train, base_depth):
    """Deepen the SEQUENCE budget for LONG series. A depth-1 conv's receptive field spans only a few steps,
    so a length-~1460 dataset like ACSF1 underfits at the base depth (observed: zero skill at depth 1, but
    skill +0.44 at depth 3). Scale depth with sequence length; short series keep the cheaper base depth (no
    over-parameterization). Sequence length is the middle axis of the (n, T, features) dense tensor."""
    dn = getattr(train, "dense", None)
    seqlen = int(dn.shape[1]) if (dn is not None and dn.ndim >= 2) else 0
    if seqlen >= 512:
        return max(base_depth, 3)
    if seqlen >= 200:
        return max(base_depth, 2)
    return base_depth


def _train_size(d):
    """Number of training elements, used to order datasets smallest-first (so quick results land before the
    large datasets). Total elements = n_samples x per-sample size, which reflects data volume across
    contracts: dense (n x features) and relational (summed node-feature elements over all graphs/sets)."""
    tr = d["train"]
    nf = getattr(tr, "node_feats", None)
    if nf is not None:
        return sum(int(np.asarray(x).size) for x in nf)
    dn = getattr(tr, "dense", None)
    if dn is not None:
        return int(dn.numel()) if hasattr(dn, "numel") else int(np.asarray(dn).size)
    return 0


def add_pipeline_args(ap):
    """Add the shared AllGraph pipeline flags -- device, budget scaling, primitive readout, processing
    layers, diagnostics, and presets. Shared by the standard and Cell Painting validation runners so their
    model configuration is identical; only the DATA-selection flags differ per runner."""
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu (auto prefers MPS on Apple Silicon)")
    ap.add_argument("--epochs_scale", type=float, default=1.0)
    ap.add_argument("--select", default="argmax", choices=["argmax", "gibbs", "sparse"],
                    help="primitive readout: argmax (DARTS), gibbs (derived, co-adaptation robust), or sparse (sparsity-priced mixture)")
    ap.add_argument("--sparsity_mu", type=float, default=0.3,
                    help="price mu for select=sparse (charges the mixture its effective #primitives)")
    ap.add_argument("--tensorize_mu", type=float, default=0.05,
                    help="price for auto-tensorization of FLAT vectors (priced grid discovery); <0 disables")
    ap.add_argument("--tiebreak", action="store_true",
                    help="enable the contract MDL tie-break (J=R+mu_c*Omega_struct) for geometric data")
    ap.add_argument("--no_learned_router", action="store_true",
                    help="disable the learned contract router (always bake off); default uses the warm cross-domain "
                         "router. Only matters with --tiebreak -- the router lives inside the bake-off.")
    ap.add_argument("--symmetry_routing", action="store_true",
                    help="symmetry-driven routing (contract=arch(G)) for geometric data: route equivariant-vs-set "
                         "by the data's discovered symmetry. Effective STANDALONE (default route path) and also "
                         "ahead of the --tiebreak bake-off -- it no longer requires --tiebreak.")
    ap.add_argument("--canonicalize", action="store_true",
                    help="enable Phase-1 canonicalization reuse (requires --symmetry_routing)")
    ap.add_argument("--discover", default=None, choices=["menu", "extended"],
                    help="autonomous group discovery -> generated contract: 'menu' (SO/Sim/Lorentz) or 'extended' (O(p,q)/U/Sp/SL)")
    ap.add_argument("--kernel_from_xi", action="store_true",
                    help="spatial/volumetric: measure correlation length xi and add larger-kernel primitives (conv2d_k5/k7) when the receptive field warrants it")
    ap.add_argument("--angular_from_data", action="store_true",
                    help="equivariant: select max angular order by the priced marginal-value rule (drop l=1 vectors when the target is radial)")
    ap.add_argument("--nonlinear_symmetry_fallback", action="store_true",
                    help="opt-in LaLiGAN nonlinear symmetry diagnostic when linear group discovery finds nothing (requires --discover)")
    ap.add_argument("--flatten_grids", action="store_true",
                    help="feed spatial/volumetric/4d data FLAT so the allgraph must rediscover the grid (tests tensorization end to end)")
    ap.add_argument("--enabled_contracts", dest="enabled_contracts", default=None,
                    help="restrict the model to a comma-separated SUBSET of the eight peer contracts "
                         "(sequence,spatial,volumetric,4d,graph,equivariant,set,operator); routing/tiebreak "
                         "may only land on an enabled contract. Default: all. Exclusive with --disable_contracts.")
    ap.add_argument("--disable_contracts", dest="disable_contracts", default=None,
                    help="complement of --enabled_contracts: comma-separated contracts to DISABLE (all others enabled).")
    ap.add_argument("--contract_posterior", action="store_true",
                    help="tie-break: report an approximate Bayesian posterior over contracts "
                         "(-log p(c|data) ~ n*NLL + Omega_struct); MAP == the J winner (reported uncertainty only).")
    ap.add_argument("--report_llc", action="store_true",
                    help="after each fit, estimate the deployed net's Local Learning Coefficient (approx. RLCT, "
                         "singularity-aware complexity <= #params/2) via SGLD. Diagnostic; valid only at a converged minimum.")
    ap.add_argument("--developmental_llc", action="store_true",
                    help="after each fit, re-train a fresh copy of the selected architecture and probe the LLC at "
                         "checkpoints, reporting the developmental curve lambda_hat(t) (D4): locates the convergence "
                         "onset and staged-learning structure. Diagnostic; adds a retrain + short SGLD probes.")
    ap.add_argument("--report_thermo", action="store_true",
                    help="after each fit, record the single thermodynamic potential's three-level temperature "
                         "hierarchy (beta_W=1/log n, beta_A=gibbs_beta, beta_C=1) and assert the temperatures are "
                         "principled and not coupled (D2). Diagnostic / conceptual-hygiene; changes nothing.")
    ap.add_argument("--report_response", action="store_true",
                    help="after each fit, report the CURVATURE of the selection objective at the chosen "
                         "point -- the specific heat of the primitive readout and the first-order transition "
                         "structure of the contract choice (critical price mu*, margins, slope jump) (D5). "
                         "Reuses already-computed quantities; diagnostic, changes nothing. Needs --select gibbs "
                         "(readout channel) and/or --tiebreak (contract channel) to have anything to report.")
    ap.add_argument("--report_ledger", action="store_true",
                    help="after each fit, assemble the effective-dimension ledger on one coarse-graining "
                         "axis (D3): participation ratio at the data-covariance and alpha-simplex levels, "
                         "plus lambda (RLCT) at the model level if --report_llc ran. Reuses already-computed "
                         "pieces; diagnostic, changes nothing.")
    ap.add_argument("--deploy_nonlinear_contract", action="store_true",
                    help="geometric data with no linear group: joint LaLiGAN discovery -> deploy the latent-equivariant "
                         "contract if a latent symmetry is confirmed (opt-in; usually falls back on real data).")
    ap.add_argument("--equivariant_realization", default="emlp", choices=["emlp", "scalable"],
                    help="realization of the deployed LATENT-equivariant head (effective only with "
                         "--deploy_nonlinear_contract): 'emlp' (exact, O(D^3)) or 'scalable' (G-RepsNet/"
                         "Vector-Neurons, equivariant by construction, linear in channels). The discovered-group "
                         "generated_equivariant contract always uses the exact EMLP regardless of this flag.")
    ap.add_argument("--price_equivariance", action="store_true",
                    help="select how strictly to enforce a discovered symmetry: relaxation priced by "
                         "J=R_val+mu*Omega (relax=0 exact; broken data admits matched breaking). The priced "
                         "DEPLOYMENT requires --deploy_nonlinear_contract; otherwise this only emits the "
                         "breaking-probe diagnostic.")
    ap.add_argument("--price_modes", action="store_true",
                    help="operator contract: select the Fourier-mode budget by the priced marginal-value rule "
                         "(operator analogue of width/depth) instead of a fixed heuristic; saves parameters.")
    ap.add_argument("--price_singular", action="store_true",
                    help="D1: in the geometric contract bake-off, augment each contract's Omega_struct with "
                         "the functional singular code length omega_func = max(lambda,0)*log n (LLC of the "
                         "fitted function); non-converged candidates (guard lambda_hat < -0.5) keep "
                         "structural-only pricing. Requires --tiebreak. Refines J and reports effective DoF; "
                         "on realistic converged data it reprices but does not change the chosen contract.")
    ap.add_argument("--singular_mu", type=float, default=None,
                    help="D1: exchange rate s_mu for the functional term (extra Omega = (s_mu/mu_c)*"
                         "omega_func); default None -> uses mu_c. Only with --price_singular.")
    ap.add_argument("--singular_llc_steps", type=int, default=150,
                    help="D1: SGLD steps for the per-contract LLC (higher -> cleaner lambda, slower); "
                         "default 150. Only with --price_singular.")
    ap.add_argument("--singular_llc_chains", type=int, default=3,
                    help="D1: SGLD chains for the per-contract LLC (higher -> lower variance); default 3. "
                         "Only with --price_singular.")
    ap.add_argument("--select_size", nargs="?", const="sequential", default=None,
                    choices=["sequential", "area", "variable"],
                    help="in-line degrees-of-freedom stage: select width/depth after routing, before training. "
                         "Modes: 'sequential' (priced width-then-depth), 'area' (joint width x depth), "
                         "'variable' (variable-width-per-layer + emergent depth). Bare flag == 'sequential'. "
                         "'sequential'/'area' take effect only on RELATIONAL contracts (set/graph/equivariant); "
                         "on dense (sequence/spatial/volumetric/4d) and operator/generated they are a no-op.")
    ap.add_argument("--preset", default=None, choices=["min", "med", "max", "opt"],
                    help="processing-layer preset (from the 9-dataset ablation): 'min' all off (plain), "
                         "'med' stable+cheap set that only helped (sparse + symmetry_routing + canonicalize + "
                         "discover), 'max' every layer on (gibbs + tiebreak + symmetry_routing + canonicalize "
                         "+ discover + select_size), 'opt' the data-size-ROBUST per-schema optimal processing "
                         "flags from the quick-validation flag search -- applied PER-DATASET by contract "
                         "(kernel_from_xi for spatial/volumetric, gibbs readout for operator, discover extended "
                         "for set; graph/equivariant/sequence/4d keep defaults -- their only quick win, "
                         "select_size, did not transfer to full data). Explicit flags override the preset.")
    ap.add_argument("--auto_epoch", nargs="?", const="train", default=None, choices=["train", "val"],
                    help="early-stop each deployed model's training on a plateau. Bare flag or '=train' "
                         "monitors the epoch mean TRAIN loss; '=val' holds out ~15%% of the training data and "
                         "monitors VALIDATION loss (more overfitting-robust). Stops when the relative "
                         "reduction stays < --auto_epoch_min_delta for --auto_epoch_patience epochs, after "
                         "--auto_epoch_min_epochs. The contract's epoch budget (x epochs_scale) is the CAP; "
                         "internal search sub-fits keep their fixed budgets. NOTE: '=val' needs enough data "
                         "for a reliable held-out monitor (>= ~50 val samples); smaller datasets fall back to "
                         "train-loss monitoring automatically.")
    ap.add_argument("--auto_epoch_patience", type=int, default=4,
                    help="--auto_epoch: consecutive non-improving epochs before stopping (default 4)")
    ap.add_argument("--auto_epoch_min_delta", type=float, default=0.01,
                    help="--auto_epoch: minimum RELATIVE train-loss reduction that counts as improvement "
                         "(default 0.01 = 1%%)")
    ap.add_argument("--auto_epoch_min_epochs", type=int, default=5,
                    help="--auto_epoch: minimum epochs to train before early stopping is allowed (default 5)")
    ap.add_argument("--save_models", action="store_true",
                    help="after each dataset's fit, save the trained model to the package out/ folder as "
                         "<dataset>_<UTC-timestamp>.pt (loadable with AllGraph.load for on-demand inference). "
                         "Off by default.")
    # (shared pipeline flags end here)


def resolve_pipeline(args, ap):
    """Apply the preset and resolve device / learned router / tensorize price / enabled-contract set from parsed
    args. Returns (device, router, tzmu, enabled_sg). Shared by both validation runners."""
    _apply_preset(args)
    device = resolve_device(args.device)
    router = None if args.no_learned_router else "default"
    tzmu = None if (isinstance(args.tensorize_mu, float) and args.tensorize_mu < 0) else args.tensorize_mu
    _ALL_CONTRACTS = ("sequence", "spatial", "volumetric", "4d", "graph", "equivariant", "set", "operator")
    if args.enabled_contracts and args.disable_contracts:
        ap.error("use only one of --enabled_contracts / --disable_contracts")
    enabled_sg = None
    if args.enabled_contracts:
        enabled_sg = args.enabled_contracts
    elif args.disable_contracts:
        drop = {s.strip() for s in args.disable_contracts.split(",") if s.strip()}
        enabled_sg = ",".join(a for a in _ALL_CONTRACTS if a not in drop)
    return device, router, tzmu, enabled_sg


def make_allgraph(args, bud, device, router, tzmu, enabled_sg):
    """Construct an AllGraph from parsed pipeline args + a (width, depth, epochs) budget. Shared by both
    validation runners so the model is configured identically from the same flags."""
    return AllGraph(width=bud["width"], depth=bud["depth"],
                     epochs=max(1, int(bud["epochs"] * args.epochs_scale)), device=device, verbose=False, seed=0,
                     sparsity_mu=args.sparsity_mu, tensorize_mu=tzmu, contract_router=router,
                     symmetry_routing=args.symmetry_routing, canonicalize_reuse=args.canonicalize,
                     discover_equivariant_contract=(True if args.discover == "menu" else
                                                    "extended" if args.discover == "extended" else False),
                     kernel_from_xi=args.kernel_from_xi, angular_from_data=args.angular_from_data,
                     nonlinear_symmetry_fallback=args.nonlinear_symmetry_fallback,
                     enabled_contracts=enabled_sg, contract_posterior=args.contract_posterior,
                     report_llc=args.report_llc, deploy_nonlinear_contract=args.deploy_nonlinear_contract,
                     equivariant_realization=args.equivariant_realization,
                     price_equivariance=args.price_equivariance, price_modes=args.price_modes,
                     developmental_llc=args.developmental_llc, report_thermo=args.report_thermo,
                     report_response=args.report_response, report_ledger=args.report_ledger,
                     price_singular=args.price_singular, singular_mu=args.singular_mu,
                     singular_llc_steps=args.singular_llc_steps, singular_llc_chains=args.singular_llc_chains,
                     progress=True, auto_epoch=args.auto_epoch, auto_epoch_patience=args.auto_epoch_patience,
                     auto_epoch_min_delta=args.auto_epoch_min_delta, auto_epoch_min_epochs=args.auto_epoch_min_epochs)


# Per-SCHEMA optimal PROCESSING flags from the quick-validation flag search (run_quick_validation --contracts
# <c>, epochs held fixed, every winner confirmed at full data_scale). Only flags that HELP a contract are set;
# the optima CONFLICT across contracts (select_size sequential helps graph/equivariant but hurts set; gibbs
# helps operator but hurts 4d/set; kernel_from_xi helps grids only) -- which is exactly why --preset opt is
# applied PER-DATASET by expected contract, not as one global flag set. sequence, 4d, and the discovered
# contracts (generated_equivariant/latent_equivariant) keep defaults (no flag beat them). Values are (attr, value)
# on the parsed args: kernel_from_xi/select map to the AllGraph build + fit; discover -> discover_equivariant_
# contract; select_size -> the fit size stage.
_OPT_FLAGS = {
    "spatial":     {"kernel_from_xi": True},      # larger conv kernels (k5/k7) -- receptive-field win, data-size-independent
    "volumetric":  {"kernel_from_xi": True},      # same, 3D
    "operator":    {"select": "gibbs"},           # co-adaptation-robust readout -- transfers to full data (Darcy2D +0.087)
    "set":         {"discover": "extended"},      # group discovery: no-op without a symmetry, reroute win with one (Lorentz jets)
    # graph & equivariant DELIBERATELY keep defaults: their only quick-suite win, select_size=sequential, is a
    # REDUCED-DATA ARTIFACT -- it helps a capacity-limited reduced model pick a better width/depth, but at FULL data
    # those models are near-ceiling and it did NOT transfer (verified: ESOL 0.764->0.739, rMD17-ethanol saturated at
    # 0.965). Only the data-size-ROBUST flags survive here: kernel_from_xi (receptive field), gibbs (readout),
    # discover (routing/reroute). sequence & 4d also keep defaults -- no flag beat them.
}


def apply_opt_preset(args, expected_mod):
    """For --preset opt: return a shallow COPY of `args` with the expected contract's optimal processing flags
    applied (from _OPT_FLAGS), EXCEPT any flag the user set explicitly on the CLI (explicit always wins). A no-op
    (returns args unchanged) when preset != 'opt' or the contract's default is already optimal (sequence, 4d,
    discovered contracts). Called per-dataset in the train loop so make_allgraph()/fit() see the contract's flags."""
    if args.preset != "opt":
        return args
    overrides = _OPT_FLAGS.get(expected_mod)
    if not overrides:
        return args
    import copy, sys
    given = {tok[2:].split("=")[0] for tok in sys.argv[1:] if tok.startswith("--")}
    run = copy.copy(args)
    for k, v in overrides.items():
        if k not in given:                                # explicit CLI flag wins over the preset
            setattr(run, k, v)
    return run


def maybe_flatten_grids(args, d, expected_mod):
    """If --flatten_grids is set, feed spatial/volumetric/4d grid data FLAT so the router must REDISCOVER the
    tensor shape (an end-to-end test of auto-tensorization). Only SINGLE-CHANNEL grids are flattened: a
    multi-channel image (e.g. BloodMNIST 3x28x28) would lose the channel/spatial distinction as a bare vector
    (an ill-posed tensorization -- the channel axis is not a spatial lattice axis), so those keep their tensor
    shape and exercise the normal grid route. Mutates d in place. Shared by both runners so the flag behaves
    identically (previously only the quick runner honored it)."""
    if not args.flatten_grids or expected_mod not in ("spatial", "volumetric", "4d"):
        return
    dd0 = d["train"].dense
    single_channel = (dd0.dim() >= 2 and dd0.shape[1] == 1) or dd0.dim() == 3
    if single_channel:
        for split in ("train", "test"):
            dd = d[split].dense
            d[split].dense = dd.reshape(dd.shape[0], -1)


def main():
    ap = argparse.ArgumentParser(description="Full standard validation across every ilmarinen dataset.")
    ap.add_argument("--only", default=None); ap.add_argument("--skip", default=None)
    ap.add_argument("--contracts", dest="contracts", default=None, help="comma-separated contracts (a.k.a. contracts) to include")
    add_pipeline_args(ap)
    args = ap.parse_args()
    device, router, tzmu, enabled_sg = resolve_pipeline(args, ap)

    # every dataset (core registry + extended scientific datasets) is included by default; use --only/--skip
    # to run a subset.
    from ilmarinen.core.extended_datasets import register_extended_datasets
    suite = register_extended_datasets(full_suite())
    names = list(suite)
    if args.only: names = [n for n in names if n in args.only.split(",")]
    if args.skip: names = [n for n in names if n not in args.skip.split(",")]
    if args.contracts:
        mods = set(args.contracts.split(",")); names = [n for n in names if suite[n][1] in mods]

    print("=" * 100)
    print(f"FULL STANDARD VALIDATION of AllGraph  |  device={device}  epochs_scale={args.epochs_scale}")
    print(f"select={args.select}  sparsity_mu={args.sparsity_mu}  tensorize_mu={tzmu}  tiebreak={args.tiebreak}  learned_router={not args.no_learned_router}")
    if enabled_sg is not None:
        print(f"enabled_contracts={enabled_sg}")
    print("skill in [0,1] = (acc-chance)/(1-chance) [clf] or R2 [reg]; Tox21 = ROC-AUC")
    print("=" * 100)
    # LOAD PHASE: standard validation loads every dataset at FULL size (reduced=False -> all train/test data;
    # the model is trained on the entire train split and evaluated on the entire test split). Load all first
    # so the run can be ordered by size.
    loaded = []
    for name in names:
        loader, expected_mod, _ = suite[name]
        try:
            d = loader(reduced=False, device=device)
            loaded.append((name, expected_mod, d))
        except ImportError as e:
            print(f"[{expected_mod:11}] {name:16} SKIP -- missing package: {str(e)[:40]}")
        except FileNotFoundError as e:
            print(f"[{expected_mod:11}] {name:16} SKIP -- data not found: {str(e)[:40]}")
        except Exception as e:
            print(f"[{expected_mod:11}] {name:16} ERROR (load) -- {type(e).__name__}: {str(e)[:45]}")
    # smallest datasets first, so the fastest results appear before the large ones
    loaded.sort(key=lambda t: _train_size(t[2]))

    # TRAIN PHASE (ascending dataset size)
    for _i in range(len(loaded)):
        name, expected_mod, d = loaded[_i]
        loaded[_i] = None                               # drop the suite's ref so this dataset frees after its fit
        t0 = time.time()
        mg = None
        try:
            run_args = apply_opt_preset(args, expected_mod)  # --preset opt: contract-specific optimal flags
            bud = dict(BUDGET[expected_mod])            # copy so the per-dataset depth tweak doesn't leak
            if expected_mod == "sequence":              # deepen for long series (ACSF1 etc.); see _seq_depth_for
                bud["depth"] = _seq_depth_for(d["train"], bud["depth"])
            maybe_flatten_grids(run_args, d, expected_mod)  # honor --flatten_grids (tensorization rediscovery test)
            mg = make_allgraph(run_args, bud, device, router, tzmu, enabled_sg)
            mg.progress_desc = name                     # label this model's training bar with the dataset
            res = mg.fit(d["train"], task=d["task"], select=run_args.select, tiebreak=run_args.tiebreak,
                         select_size=run_args.select_size)
            metric, value, extra = _eval_test(mg, d["test"], d["task"], rotated=d.get("rotated", False),
                                               auc=d.get("auc", False), report_auc=d.get("report_auc", False),
                                               bg_rejection=d.get("bg_rejection"),
                                               target_scale=d.get("target_scale"),
                                               target_units=d.get("target_units"))
            chance = d["chance"]
            # skill is the cross-dataset comparable axis: R2 for regression (from `extra` when a physical-unit
            # MAE is the headline, else the headline itself), AUC for auc datasets, else acc normalized by chance.
            extra_d = dict(extra)
            if d.get("auc"):
                skill = value
            elif d["task"] == "regression":
                skill = extra_d.get("R2", value)
            else:
                skill = (value - chance) / (1 - chance)
            dt = time.time() - t0
            arch = "→".join(res.get("architecture") or [c.primitives[int(c.alpha.argmax())] for c in mg.net.cells]) if hasattr(mg.net, "cells") else "?"
            params = sum(p.numel() for p in mg.net.parameters())
            tag = f" IPR={res['ipr']:.2f}" if "ipr" in res else ""
            # rejection figures are O(100-3000): render with :.4g (compact, no meaningless decimals); the
            # scale-bound metrics (R2/acc/AUC) keep the fixed-decimal :.4f.
            extra_str = "".join((f" {n}={v:.4g}" if n.startswith("1/eB") else f" {n}={v:.4f}") for n, v in extra)
            print(f"[{mg.contract:11}] {name:16} {metric:9}={value:.4f}{extra_str} skill={skill:+.3f} "
                  f"arch=[{arch}]{tag} params={params:>8} {dt:.0f}s")
            print(f"{'':13} {'':16} field={d['field']:24} SOTA: {d['sota']}")
            if args.save_models:
                saved = mg.save(stem=name)             # <dataset>_<timestamp>.pt in the package out/ folder
                print(f"{'':13} {'':16} saved model -> {saved}")
        except Exception as e:
            print(f"[{expected_mod:11}] {name:16} ERROR -- {type(e).__name__}: {str(e)[:55]}")
        finally:
            # RELEASE MEMORY BETWEEN DATASETS. Apple Silicon uses UNIFIED memory (CPU+GPU share one pool), so a
            # prior fit's retained tensors starve the next model -- observed as a large slowdown across the
            # rMD17->ACSF1 handover and, under the full epoch budget, Metal allocation failures
            # (AcceleratorError). Drop the model + this dataset, run GC, and return MPS allocations to the pool.
            mg = None; d = None
            gc.collect()
            if str(device).startswith("mps") and hasattr(torch, "mps"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
    print("=" * 100)


if __name__ == "__main__":
    main()

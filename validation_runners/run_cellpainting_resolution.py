#!/usr/bin/env python
"""run_cellpainting_resolution.py -- choose the DEFAULT --hw (downsampling) for the Cell Painting suite.

The question this answers: how far can JUMP Cell Painting fields be downsampled before the morphological
signal that separates CRISPR perturbations is destroyed? The answer ships as the default --hw of
run_cellpainting_validation.py, so it must be the value that stays right when the gene set changes -- not
the argmax of a noisy curve.

DECISION RULE (1-SE knee, not argmax). With ~12 wells/class x 2 sites, the held-out split is small and
adjacent hw values differ by less than their own noise; argmax over such a curve does not replicate. So we
resample the BY-WELL split several times per cell, take mean +- standard error of retrieval mAP across
those splits, and report the SMALLEST hw whose mean is within 1 SE of the best cell's mean. Error bars come
from re-splitting WELLS because a well's sites are near-replicates -- fields are not independent samples.

THREE MEASUREMENTS, which answer different questions:

  1. PIXEL CEILING (--ceiling, model-free). Retrieval mAP computed directly on the downsampled pixels
     (PCA-reduced, no training). This separates "the resampling destroyed the information" from "the model
     failed to exploit it" -- two failure modes a trained sweep alone cannot tell apart. Where this curve
     breaks is a hard floor: no hw below it can be right for ANY architecture. Costs seconds per point.

  2. TRAINED hw x DEPTH GRID (the main result). A plain hw sweep at fixed depth is CONFOUNDED: the conv
     stack's receptive field is a fixed number of PIXELS, so as hw grows it covers a shrinking FRACTION of
     the field of view, and the sweep conflates "more detail" with "less context" -- yielding a falsely
     early optimum. We therefore grid hw x depth and take the best depth per hw, so each resolution is
     judged at the capacity it needs.

  3. PER-CHANNEL CURVES (--per_channel, optional). The same hw curve one stain at a time. The expected
     pattern is DNA nearly flat (nuclei are coarse) while Mito/AGP fall off early (granularity is fine).
     If that holds, the default is being set by the finest-texture channel, which is worth saying out loud
     rather than shipping a bare number.

COST. One S3 fetch, not N. The cache key in core/cellpainting includes hw, so a naive sweep re-downloads
the whole gallery per resolution. Here we build ONE cache at --build_hw and derive every lower hw locally
via resample_stack(). Set ILMARINEN_DATA_DIR to a PERSISTENT path first -- it defaults to the OS temp dir,
which is reaped, and this fetch is the expensive part of the whole study.

    export ILMARINEN_DATA_DIR=~/ilmarinen_data

Two-stage resampling (native -> build_hw -> hw) is not bit-identical to a direct native -> hw resize. That
is the controlled choice for a sweep (identical fields and normalization; resolution the only variable),
but VERIFY the chosen hw against a direct build before changing the default -- see --verify.

USAGE:
    python validation_runners/run_cellpainting_resolution.py                       # full study
    python validation_runners/run_cellpainting_resolution.py --ceiling_only        # cheap, no training
    python validation_runners/run_cellpainting_resolution.py --hw_grid 32,48,64 --depth_grid 2,3
    python validation_runners/run_cellpainting_resolution.py --per_channel
    python validation_runners/run_cellpainting_resolution.py --verify 64           # direct-build check
    python validation_runners/run_cellpainting_resolution.py --classes MYC,RHOA,BRD4,CTNNB1  # transfer
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ilmarinen.core.cellpainting import (CLASSES, CHANNELS, NATIVE_SHAPE, _cached_arrays,
                                         resample_stack, split_by_well)
from ilmarinen.core.paths import data_dir
# reuse the maintained runner's embedding hook + retrieval metric, so the numbers here are the SAME
# quantity the Cell Painting suite reports (no reimplementation drift).
from run_cellpainting_validation import _embeddings, retrieval_map
from run_standard_validation import add_pipeline_args, resolve_pipeline, make_allgraph, BUDGET


def _fmt(v, nd=4):
    return "  n/a " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def pixel_ceiling_map(X, y, wells, classes, split_seed, n_pca=50):
    """MODEL-FREE retrieval mAP on the downsampled pixels of the held-out fields.

    No training: flatten each field, PCA to n_pca dims (fit on TRAIN fields only, so the held-out
    embedding is not built with knowledge of the test set), and score the same retrieval metric. This is
    an information-survival probe, not a model -- it upper-bounds nothing formally, but a collapse here is
    unambiguous evidence that the resampling itself removed the class signal."""
    d = split_by_well(X, y, wells, classes, split_seed=split_seed)
    Xtr = d["train"].dense.numpy().reshape(len(d["train"].dense), -1)
    Xte = d["test"].dense.numpy().reshape(len(d["test"].dense), -1)
    yte = np.asarray(d["test"].y)
    if len(Xte) < 3 or len(Xtr) < 2:
        return float("nan")
    mu = Xtr.mean(0, keepdims=True)
    k = int(min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1]))
    if k < 2:
        return float("nan")
    # right singular vectors of the centred TRAIN matrix = PCA basis; project the test fields onto it
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return retrieval_map((Xte - mu) @ Vt[:k].T, yte)


def trained_map(args, X, y, wells, classes, split_seed, depth, device, router, tzmu, enabled_sg):
    """Fit the AllGraph at one (resolution, depth, split) cell and return (held-out retrieval mAP, acc,
    seconds). Mirrors run_cellpainting_validation's train->embed->mAP path exactly."""
    d = split_by_well(X, y, wells, classes, split_seed=split_seed)
    Xte = d["test"].dense
    yte = np.asarray(d["test"].y)
    if len(Xte) < 3:
        return float("nan"), float("nan"), 0.0
    bud = dict(BUDGET["spatial"]); bud["depth"] = depth
    t0 = time.time()
    mg = make_allgraph(args, bud, device, router, tzmu, enabled_sg)
    # make_allgraph hardwires progress=True (right for a single headline fit, wrong here): a grid runs
    # hw x depth x splits fits, and their bars would shred the results table.
    mg.progress = False
    mg.fit(d["train"], task=d["task"], select=args.select, tiebreak=args.tiebreak,
           select_size=args.select_size)
    emb = _embeddings(mg, Xte)
    mapv = retrieval_map(emb, yte)
    with torch.no_grad():
        logits = torch.cat([mg.net(Xte[j:j + 128].to(mg.device)).cpu() for j in range(0, len(Xte), 128)])
    acc = float((logits.argmax(1).numpy() == yte).mean())
    return mapv, acc, time.time() - t0


def mean_se(vals):
    """Mean and STANDARD ERROR over split seeds (SE = sd/sqrt(n), the spread of the mean). Returns
    (mean, se, n) over the non-NaN entries."""
    v = np.asarray([x for x in vals if not np.isnan(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return float(v[0]), float("nan"), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), len(v)


def pick_knee(rows):
    """1-SE rule: among cells, find the best mean mAP; return the SMALLEST hw whose mean is within one SE
    of that best (SE taken from the best cell -- the standard conservative choice). `rows` is a list of
    dicts with 'hw', 'mean', 'se'. Returns (chosen_hw, best_hw, threshold) or (None, None, None)."""
    ok = [r for r in rows if not np.isnan(r["mean"])]
    if not ok:
        return None, None, None
    best = max(ok, key=lambda r: r["mean"])
    se = 0.0 if np.isnan(best["se"]) else best["se"]
    thr = best["mean"] - se
    within = sorted([r["hw"] for r in ok if r["mean"] >= thr])
    return (within[0] if within else best["hw"]), best["hw"], thr


def main():
    ap = argparse.ArgumentParser(description="Cell Painting resolution study: pick the default --hw.")
    ap.add_argument("--per_class", type=int, default=12, help="distinct WELLS per class (data-amount knob)")
    ap.add_argument("--sites", type=int, default=2, help="fields (sites) per well")
    ap.add_argument("--classes", default=None, help="comma-separated perturbation genes")
    ap.add_argument("--build_hw", type=int, default=224,
                    help="resolution of the ONE cached build; every hw in the grid is derived from it "
                         "locally (default 224). Must be >= max(--hw_grid).")
    ap.add_argument("--hw_grid", default="16,24,32,48,64,96,128", help="comma-separated hw values to test")
    ap.add_argument("--depth_grid", default="2,3,4",
                    help="comma-separated conv depths; the best depth per hw is reported, so each "
                         "resolution is judged at the capacity it needs (see the receptive-field note)")
    ap.add_argument("--splits", type=int, default=5,
                    help="by-well split seeds per cell -> the error bars (default 5)")
    ap.add_argument("--ceiling", action="store_true", default=True,
                    help="also run the model-free pixel-ceiling curve (default on)")
    ap.add_argument("--no_ceiling", dest="ceiling", action="store_false")
    ap.add_argument("--ceiling_only", action="store_true",
                    help="only the model-free curve -- no training at all (seconds, not hours)")
    ap.add_argument("--per_channel", action="store_true",
                    help="also run the model-free curve one STAIN at a time (which channel sets the floor)")
    ap.add_argument("--verify", type=int, default=None,
                    help="after the sweep, rebuild this hw DIRECTLY from S3 (not derived) and compare -- "
                         "confirms the two-stage resampling did not bias the choice")
    add_pipeline_args(ap)
    args = ap.parse_args()
    device, router, tzmu, enabled_sg = resolve_pipeline(args, ap)
    classes = tuple(s.strip() for s in args.classes.split(",")) if args.classes else CLASSES
    hw_grid = sorted({int(s) for s in args.hw_grid.split(",") if s.strip()})
    depth_grid = sorted({int(s) for s in args.depth_grid.split(",") if s.strip()})
    if max(hw_grid) > args.build_hw:
        ap.error(f"--build_hw {args.build_hw} < max(--hw_grid) {max(hw_grid)}: cannot UPSAMPLE a derived "
                 f"resolution above the build. Raise --build_hw.")

    print("=" * 100)
    print(f"CELL PAINTING RESOLUTION STUDY  |  device={device}  epochs_scale={args.epochs_scale}")
    print(f"per_class(wells)={args.per_class}  sites/well={args.sites}  classes={list(classes)}")
    print(f"hw_grid={hw_grid}  depth_grid={depth_grid}  splits={args.splits}  build_hw={args.build_hw}")
    print(f"cache dir={data_dir()}"
          f"{'   [TEMP -- set ILMARINEN_DATA_DIR to keep the fetch]' if not os.environ.get('ILMARINEN_DATA_DIR') else ''}")
    print("metric = held-out retrieval mAP (mean +- SE over by-well split seeds); random ~ %.3f" % (1.0 / len(classes)))
    print("=" * 100)

    # ---- ONE fetch: build (or reuse) the high-resolution cache, then derive every hw locally ----
    t0 = time.time()
    try:
        Xb, y, wells, classes = _cached_arrays(args.build_hw, args.per_class, args.sites, classes)
    except ImportError as e:
        print(f"SKIP -- jump-portrait not installed ({str(e)[:60]}); pip install jump-portrait")
        return
    except Exception as e:
        print(f"ERROR (data) -- {type(e).__name__}: {str(e)[:70]}")
        return
    native = NATIVE_SHAPE.get("shape")
    print(f"built/loaded {len(Xb)} fields at hw={args.build_hw} in {time.time()-t0:.0f}s "
          f"({len(set(wells.tolist()))} wells, {len(classes)} classes)")
    if native:
        print(f"NATIVE field = {native[0]}x{native[1]} px -> a given hw samples at "
              f"{native[0]}/hw px per output px; multiply by the source's um/px for the physical rate "
              f"(nuclei ~10-20um survive coarse sampling; Mito/AGP granularity ~1-2um does not)")
    else:
        print("NATIVE field size unknown (cache was reused, so no image passed through the resizer); "
              "delete the npz and rebuild to record it")

    # ---- 1. model-free pixel ceiling ----
    ceiling = {}
    if args.ceiling or args.ceiling_only:
        print("\n--- model-free pixel ceiling (PCA of downsampled pixels, NO training) ---")
        print(f"{'hw':>5}  {'mAP':>8}  {'SE':>7}   px/field")
        for hw in hw_grid:
            Xh = resample_stack(Xb, hw)
            vals = [pixel_ceiling_map(Xh, y, wells, classes, s) for s in range(args.splits)]
            m, se, _ = mean_se(vals)
            ceiling[hw] = (m, se)
            print(f"{hw:>5}  {_fmt(m):>8}  {_fmt(se, 4):>7}   {hw*hw*len(CHANNELS):>8}")
            del Xh

    # ---- 1b. per-channel curves ----
    if args.per_channel:
        print("\n--- model-free ceiling PER STAIN (which channel sets the resolution floor) ---")
        header = "   hw  " + "".join(f"{c:>9}" for c in CHANNELS)
        print(header)
        for hw in hw_grid:
            Xh = resample_stack(Xb, hw)
            cells = []
            for ci in range(len(CHANNELS)):
                vals = [pixel_ceiling_map(Xh[:, ci:ci + 1], y, wells, classes, s) for s in range(args.splits)]
                cells.append(mean_se(vals)[0])
            print(f"{hw:>5}  " + "".join(f"{_fmt(v, 3):>9}" for v in cells))
            del Xh

    if args.ceiling_only:
        print("\n(--ceiling_only: no training run. The break in the curve above is a hard FLOOR -- no hw "
              "below it can be right for any architecture. Re-run without the flag for the trained grid.)")
        print("=" * 100)
        return

    # ---- 2. trained hw x depth grid ----
    print(f"\n--- trained hw x depth grid ({len(hw_grid)}x{len(depth_grid)} cells x {args.splits} splits) ---")
    print(f"{'hw':>5} {'depth':>6}  {'mAP':>8} {'SE':>7}  {'acc':>7}  {'s/fit':>7}")
    grid = {}
    for hw in hw_grid:
        Xh = resample_stack(Xb, hw)
        for depth in depth_grid:
            maps, accs, secs = [], [], []
            for s in range(args.splits):
                try:
                    mv, av, dt = trained_map(args, Xh, y, wells, classes, s, depth,
                                             device, router, tzmu, enabled_sg)
                except Exception as e:
                    print(f"{hw:>5} {depth:>6}  ERROR -- {type(e).__name__}: {str(e)[:45]}")
                    mv, av, dt = float("nan"), float("nan"), 0.0
                maps.append(mv); accs.append(av); secs.append(dt)
            m, se, n = mean_se(maps)
            grid[(hw, depth)] = {"mean": m, "se": se, "n": n, "acc": mean_se(accs)[0],
                                 "sec": float(np.mean(secs)) if secs else 0.0}
            print(f"{hw:>5} {depth:>6}  {_fmt(m):>8} {_fmt(se, 4):>7}  {_fmt(mean_se(accs)[0], 3):>7}  "
                  f"{np.mean(secs):>7.0f}")
        del Xh

    # ---- 3. best depth per hw, then the 1-SE knee ----
    print("\n--- best depth per hw (each resolution judged at the capacity it needs) ---")
    print(f"{'hw':>5}  {'best_depth':>10}  {'mAP':>8} {'SE':>7}  {'ceiling':>8}  {'s/fit':>7}")
    rows = []
    for hw in hw_grid:
        cells = [(d, grid[(hw, d)]) for d in depth_grid if not np.isnan(grid[(hw, d)]["mean"])]
        if not cells:
            print(f"{hw:>5}  {'-':>10}  {'n/a':>8}")
            continue
        bd, cell = max(cells, key=lambda t: t[1]["mean"])
        rows.append({"hw": hw, "depth": bd, "mean": cell["mean"], "se": cell["se"], "sec": cell["sec"]})
        cm = ceiling.get(hw, (None, None))[0]
        print(f"{hw:>5}  {bd:>10}  {_fmt(cell['mean']):>8} {_fmt(cell['se'], 4):>7}  {_fmt(cm, 3):>8}  "
              f"{cell['sec']:>7.0f}")

    chosen, best_hw, thr = pick_knee(rows)
    print("\n" + "=" * 100)
    if chosen is None:
        print("no usable cells -- every fit failed or the held-out split was too small to score")
        print("=" * 100)
        return
    best = next(r for r in rows if r["hw"] == best_hw)
    pick = next(r for r in rows if r["hw"] == chosen)
    print(f"BEST cell        : hw={best_hw} depth={best['depth']}  mAP={best['mean']:.4f} "
          f"(+-{_fmt(best['se'], 4).strip()} SE)")
    print(f"1-SE threshold   : {thr:.4f}")
    print(f"RECOMMENDED --hw : {chosen}  (depth {pick['depth']}, mAP={pick['mean']:.4f}, "
          f"{pick['sec']:.0f}s/fit)  <- smallest hw within 1 SE of best")
    if chosen != best_hw and best["sec"] > 0:
        print(f"                   {best['sec']/max(pick['sec'],1e-9):.1f}x cheaper than the best cell for "
              f"a difference inside the noise")
    print(f"current default  : hw=48 in run_cellpainting_validation.py")
    if args.splits < 3:
        print("CAUTION: --splits < 3, so the SE is not meaningful; the knee is not trustworthy here.")
    print("Before changing the default, confirm on a DIFFERENT --classes set and --per_class "
          "(a resolution tuned to one gene set need not transfer), and with --verify.")
    print("=" * 100)

    # ---- 4. optional direct-build verification of the chosen hw ----
    if args.verify:
        hw = args.verify
        print(f"\n--- verify hw={hw}: DIRECT build from S3 vs derived from hw={args.build_hw} ---")
        try:
            Xd, yd, wd, cd = _cached_arrays(hw, args.per_class, args.sites, classes)
        except Exception as e:
            print(f"  ERROR -- {type(e).__name__}: {str(e)[:60]}")
            return
        dvals = [pixel_ceiling_map(Xd, yd, wd, cd, s) for s in range(args.splits)]
        rvals = [pixel_ceiling_map(resample_stack(Xb, hw), y, wells, classes, s) for s in range(args.splits)]
        dm, dse, _ = mean_se(dvals); rm, rse, _ = mean_se(rvals)
        print(f"  direct  ceiling mAP = {_fmt(dm)} +- {_fmt(dse, 4).strip()}")
        print(f"  derived ceiling mAP = {_fmt(rm)} +- {_fmt(rse, 4).strip()}")
        gap = abs(dm - rm)
        tol = 2 * max(dse if not np.isnan(dse) else 0.0, rse if not np.isnan(rse) else 0.0)
        print(f"  |gap| = {gap:.4f}  vs 2*SE = {tol:.4f}  -> "
              f"{'consistent (two-stage resampling did not bias the choice)' if gap <= tol else 'DIVERGENT -- re-run the trained grid on direct builds before shipping'}")
        print("=" * 100)


if __name__ == "__main__":
    main()

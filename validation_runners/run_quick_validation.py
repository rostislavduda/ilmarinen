#!/usr/bin/env python
"""
run_quick_validation.py -- FAST smoke of the standard validation suite: a SAMPLE-DOWNSCALED MIRROR of
run_standard_validation.py. It runs the SAME datasets (the full registry) and takes the SAME flags, but
loads a representative SUBSET of each dataset (reduced=True) with smaller per-contract budgets, so a run
finishes in a few minutes. Use it to get a rough read on which flags help BEFORE committing to a full
run_standard_validation.py pass; it is NOT a benchmark.

Alignment: every pipeline flag (routing, presets, --device, --auto_epoch, discovery, pricing, diagnostics)
comes from the shared `add_pipeline_args` / `resolve_pipeline` / `make_allgraph` in run_standard_validation,
so this runner is configured identically to the standard one. The ONLY differences are scale: reduced=True
subsets, the smaller BUDGET below, and the smaller eval batch sizes (dense_bs=128, relational_bs=64).

Every dataset comes from the shared registry (core/dataset_registry). Each is auto-routed by AllGraph from
its data container, trained on the subset, evaluated on the held-out test split, and reported on a CONSISTENT
skill axis:  skill = (acc-chance)/(1-chance) for classification, R2 for regression (0 = trivial baseline,
1 = perfect). AUC datasets (Tox21, MedMNIST3D binaries) report ROC-AUC as their skill.

Usage:
    python run_quick_validation.py                       # all datasets, subset each
    python run_quick_validation.py --device mps          # force Apple Silicon GPU (same as standard)
    python run_quick_validation.py --only ESOL,JetNet
    python run_quick_validation.py --skip OrganMNIST3D   # 3D conv is the slowest
    python run_quick_validation.py --contracts graph,set
    python run_quick_validation.py --preset med          # preview a preset before the full run
"""

import argparse
import gc
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> import ilmarinen
# ...and THIS directory, so the sibling-module import below also resolves under
# `python -m validation_runners.run_quick_validation`: under -m, sys.path[0] is the CWD rather than the
# script's own directory, so the repo-root insert alone is not enough.
sys.path.insert(0, _HERE)
# The pipeline (flags, device/preset resolution, model construction, eval, grid-flatten) is shared with the
# standard runner so the two stay in lockstep -- this runner only changes the DATA SCALE.
from run_standard_validation import (
    _eval_test,
    _seq_depth_for,
    _train_size,
    add_pipeline_args,
    apply_opt_preset,
    make_allgraph,
    maybe_flatten_grids,
    resolve_pipeline,
)

from ilmarinen.core.dataset_registry import full_suite, set_quick_scale

# Down-scaled per-contract budgets (width, depth, epochs): the standard runner's budgets shrunk so a full
# quick pass over the subsets finishes in minutes. This is the intended difference from standard.
BUDGET = {
    "sequence": dict(width=32, depth=1, epochs=12),
    "spatial": dict(width=16, depth=1, epochs=6),
    "volumetric": dict(width=10, depth=1, epochs=5),
    "4d": dict(width=16, depth=2, epochs=40),
    "graph": dict(width=48, depth=3, epochs=25),
    "equivariant": dict(width=24, depth=3, epochs=15),
    "set": dict(width=48, depth=2, epochs=12),
    "operator": dict(width=24, depth=3, epochs=40),
}


def main():
    ap = argparse.ArgumentParser(description="Fast sample-downscaled mirror of the standard validation suite.")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--contracts", dest="contracts", default=None, help="comma-separated contracts to include")
    ap.add_argument(
        "--data_scale",
        type=float,
        default=1.0,
        help="multiply every dataset's REDUCED (quick) subset size by this factor (default 1.0). "
        ">1 = larger subsets -> slower but a more accurate/full-data-representative diagnostic; "
        "<1 = smaller -> faster, rougher. Dials the whole smoke suite's runtime<->accuracy from one "
        "knob; full (standard-runner) sizes are unaffected. Per-sample dims (grid resolution) are "
        "not scaled -- this scales sample counts and the sequence length/test caps.",
    )
    add_pipeline_args(ap)  # identical flag set to run_standard_validation
    args = ap.parse_args()
    device, router, tzmu, enabled_sg = resolve_pipeline(args, ap)
    set_quick_scale(args.data_scale)  # applied inside every loader's reduced branch (via qscale)

    # SAME dataset list as the standard runner (full registry + extended datasets); quick just subsets each.
    from ilmarinen.core.extended_datasets import register_extended_datasets

    suite = register_extended_datasets(full_suite())
    names = list(suite)
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    if args.skip:
        names = [n for n in names if n not in args.skip.split(",")]
    if args.contracts:
        mods = set(args.contracts.split(","))
        names = [n for n in names if suite[n][1] in mods]

    print("=" * 100)
    print(
        f"QUICK VALIDATION of AllGraph  |  device={device}  data_scale={args.data_scale}  (sample-downscaled "
        f"MIRROR of run_standard_validation -- reduced subsets + small budgets; rough smoke, NOT a benchmark)"
    )
    print(
        f"select={args.select}  sparsity_mu={args.sparsity_mu}  tensorize_mu={tzmu}  tiebreak={args.tiebreak}  "
        f"learned_router={not args.no_learned_router}  flatten_grids={args.flatten_grids}"
    )
    print(
        f"symmetry_routing={args.symmetry_routing}  canonicalize={args.canonicalize}  discover={args.discover}  "
        f"select_size={args.select_size}  auto_epoch={args.auto_epoch}"
    )
    if enabled_sg is not None:
        print(f"enabled_contracts={enabled_sg}")
    print("consistent axis: skill in [0,1] = (acc-chance)/(1-chance) [clf] or R2 [reg]; auc datasets = ROC-AUC")
    print("=" * 100)
    rows = []
    # LOAD PHASE: quick validation loads a SUBSET of each dataset (reduced=True) on the resolved device; the
    # standard runner is identical here except reduced=False. Load all first so the run can be ordered by size.
    loaded = []
    for name in names:
        loader, expected_mod, _ = suite[name]
        try:
            d = loader(reduced=True, device=device)
            loaded.append((name, expected_mod, d))
        except ImportError as e:
            print(f"[{expected_mod:11}] {name:16} SKIP -- missing package: {str(e)[:40]}")
        except FileNotFoundError as e:
            print(f"[{expected_mod:11}] {name:16} SKIP -- data not found: {str(e)[:40]}")
        except Exception as e:
            print(f"[{expected_mod:11}] {name:16} ERROR (load) -- {type(e).__name__}: {str(e)[:50]}")
    loaded.sort(key=lambda t: _train_size(t[2]))  # smallest datasets first

    # TRAIN PHASE (ascending dataset size) -- mirrors the standard runner's loop, at the reduced scale.
    for _i in range(len(loaded)):
        name, expected_mod, d = loaded[_i]
        loaded[_i] = None  # drop the suite's ref so this dataset frees after its fit
        t0 = time.time()
        mg = None
        try:
            run_args = apply_opt_preset(args, expected_mod)  # --preset opt: contract-specific optimal flags
            bud = dict(BUDGET[expected_mod])  # copy so the per-dataset depth tweak doesn't leak
            if expected_mod == "sequence":  # deepen for long series (as standard does)
                bud["depth"] = _seq_depth_for(d["train"], bud["depth"])
            maybe_flatten_grids(run_args, d, expected_mod)  # honor --flatten_grids (tensorization rediscovery test)
            mg = make_allgraph(run_args, bud, device, router, tzmu, enabled_sg)
            mg.progress_desc = name  # label this model's training bar with the dataset
            res = mg.fit(
                d["train"],
                task=d["task"],
                select=run_args.select,
                tiebreak=run_args.tiebreak,
                select_size=run_args.select_size,
            )
            metric, value, extra = _eval_test(
                mg,
                d["test"],
                d["task"],
                rotated=d.get("rotated", False),
                auc=d.get("auc", False),
                report_auc=d.get("report_auc", False),
                bg_rejection=d.get("bg_rejection"),
                dense_bs=128,
                relational_bs=64,
                target_scale=d.get("target_scale"),
                target_units=d.get("target_units"),
            )
            chance = d["chance"]
            extra_d = dict(extra)
            if d.get("auc"):
                skill = value
            elif d["task"] == "classification":
                skill = (value - chance) / (1 - chance)
            else:
                skill = extra_d.get("R2", value)  # R2 stays the skill axis even when MAE is the headline
            dt = time.time() - t0
            arch = (
                "→".join(res.get("architecture") or [c.primitives[int(c.alpha.argmax())] for c in mg.net.cells])
                if hasattr(mg.net, "cells")
                else "?"
            )
            params = sum(p.numel() for p in mg.net.parameters())
            # size annotation: report the chosen width x depth, and the variable-width per-layer profile if used
            size_note = f" [{mg.width}w×{mg.depth}L]"
            szd = (mg.route_detail or {}).get("select_size") if mg.route_detail else None
            if isinstance(szd, dict) and szd.get("mode") == "variable":
                size_note += f" profile={szd.get('width_profile_mean')}"
            sota = d.get("sota")
            tag = f" IPR={res['ipr']:.2f}" if "ipr" in res else ""
            # note if tensorization changed the routed contract vs expected
            reroute = "" if mg.contract == expected_mod else f" [routed {mg.contract}]"
            # rejection figures are O(100-3000): render with :.4g; scale-bound metrics keep :.3f.
            extra_str = "".join((f" {n}={v:.4g}" if n.startswith("1/eB") else f" {n}={v:.3f}") for n, v in extra)
            print(
                f"[{mg.contract:11}] {name:16} {metric:9}={value:.3f}{extra_str} skill={skill:+.3f} "
                f"arch=[{arch}]{size_note}{tag} params={params:>7}{reroute} vs SOTA={sota} {dt:.0f}s ({d['field']})"
            )
            rows.append((name, mg.contract, metric, value, skill, arch, params, d["field"], d.get("sota")))
            if args.save_models:
                saved = mg.save(stem=name)  # <dataset>_<timestamp>.pt in the package out/ folder
                print(f"{'':13} {'':16} saved model -> {saved}")
        except Exception as e:
            print(f"[{expected_mod:11}] {name:16} ERROR -- {type(e).__name__}: {str(e)[:60]}")
        finally:
            # RELEASE MEMORY BETWEEN DATASETS (mirrors standard): Apple Silicon shares one unified CPU+GPU pool,
            # so a prior fit's retained tensors starve the next model. Drop refs, GC, and return MPS allocations.
            mg = None
            d = None
            gc.collect()
            if str(device).startswith("mps") and hasattr(torch, "mps"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
    print("=" * 100)
    print(f"Ran {len(rows)}/{len(names)} datasets. Full-data benchmark: run_standard_validation.py")
    import json

    from ilmarinen.core.paths import cache_path

    _out = cache_path("quick_val_rows.json")
    json.dump(
        [
            {
                "name": r[0],
                "contract": r[1],
                "metric": r[2],
                "value": r[3],
                "skill": r[4],
                "arch": r[5],
                "params": r[6],
                "field": r[7],
                "sota": r[8],
            }
            for r in rows
        ],
        open(_out, "w"),
        default=float,
    )
    print(f"Wrote per-dataset rows to {_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""run_cellpainting_validation.py -- a SEPARATE validation suite for the JUMP Cell Painting Gallery.

Trains the AllGraph on multi-channel (5-stain: DNA/RNA/ER/AGP/Mito) Cell Painting microscopy of human cells
under CRISPR perturbation -- routed to the spatial schema -- and evaluates RETRIEVAL mAP, the standard
Cell Painting metric: for each held-out microscopy field, how well same-perturbation fields rank above
different-perturbation fields in the learned embedding space (cosine similarity of the pre-head features).

DATA AMOUNT: --per_class chooses how many distinct wells per perturbation class to train on (with --sites
fields per well). Images are fetched once from the public Cell Painting Gallery S3 (no AWS account; requires
`pip install jump-portrait`) and cached.

INHERITED PIPELINE FLAGS: every model/processing flag from the standard validation suite is available and
behaves identically -- --select, --sparsity_mu, --tiebreak, --symmetry_routing, --discover, --select_size,
--preset, --device, --epochs_scale, --enabled_contracts, the diagnostic read-outs, etc.

USAGE:
    python run_cellpainting_validation.py                          # default 12 wells/class, auto device
    python run_cellpainting_validation.py --per_class 24 --sites 3 # more training data
    python run_cellpainting_validation.py --preset max --select sparse
    python run_cellpainting_validation.py --classes CDK1,AURKB,PLK1,KIF11   # custom perturbation set
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> import ilmarinen
# ...and THIS directory, so the sibling-module import below also resolves under
# `python -m validation_runners.run_cellpainting_validation`: under -m, sys.path[0] is the CWD rather than
# the script's own directory, so the repo-root insert alone is not enough.
sys.path.insert(0, _HERE)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# inherit the standard suite's shared pipeline machinery (identical flags + model construction)
from run_standard_validation import BUDGET, add_pipeline_args, make_allgraph, resolve_pipeline

from ilmarinen.core.cellpainting import CLASSES, load_cellpainting


def _embeddings(mg, X, batch=128):
    """Pre-head embeddings (the global-pooled feature vector) of the trained spatial net, captured via a hook
    on the final Linear head's INPUT. Falls back to the logits if the net exposes no single Linear head."""
    net = mg.net
    dev = mg.device
    cap = {}
    head = getattr(net, "head", None)
    handle = None
    if isinstance(head, torch.nn.Linear):
        handle = head.register_forward_hook(lambda m, inp, out: cap.__setitem__("z", inp[0].detach()))
    outs = []
    for j in range(0, len(X), batch):
        with torch.no_grad():
            o = net(X[j : j + batch].to(dev))
        outs.append((cap["z"] if handle is not None else o).cpu())
    if handle is not None:
        handle.remove()
    return torch.cat(outs).numpy()


def retrieval_map(emb, labels):
    """Mean Average Precision for label-based retrieval in embedding space (cosine similarity). For each query
    field, rank every other field by similarity; relevant = same perturbation label. This is the standard
    Cell Painting replicate-retrieval metric. Queries with no same-label partner are skipped."""
    labels = np.asarray(labels)
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    n = len(labels)
    aps = []
    for i in range(n):
        order = np.argsort(-S[i])
        rel = (labels[order] == labels[i]).astype(np.float64)
        if rel.sum() == 0:
            continue
        prec = np.cumsum(rel) / (np.arange(n) + 1.0)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return float(np.mean(aps)) if aps else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Cell Painting validation: train the AllGraph, report retrieval mAP.")
    # --- Cell-Painting DATA flags (the amount-of-data knob) ---
    ap.add_argument(
        "--per_class",
        type=int,
        default=12,
        help="distinct WELLS per perturbation class to fetch/train on (the training-data amount knob)",
    )
    ap.add_argument("--sites", type=int, default=2, help="fields (sites) per well")
    ap.add_argument("--hw", type=int, default=48, help="image resize (hw x hw)")
    ap.add_argument(
        "--classes",
        default=None,
        help="comma-separated perturbation genes (default: the built-in %d-gene set)" % len(CLASSES),
    )
    # --- inherited pipeline flags (device, budget scaling, readout, processing layers, diagnostics, preset) ---
    add_pipeline_args(ap)
    args = ap.parse_args()
    device, router, tzmu, enabled_sg = resolve_pipeline(args, ap)
    classes = tuple(s.strip() for s in args.classes.split(",")) if args.classes else CLASSES

    print("=" * 100)
    print(f"CELL PAINTING VALIDATION via AllGraph  |  device={device}  epochs_scale={args.epochs_scale}")
    print(f"per_class(wells)={args.per_class}  sites/well={args.sites}  hw={args.hw}  classes={list(classes)}")
    print(f"select={args.select}  select_size={args.select_size}  tiebreak={args.tiebreak}  preset={args.preset}")
    print("metric = retrieval mAP on held-out fields (same-perturbation fields rank above others); random ~ 1/K")
    print("=" * 100)

    t0 = time.time()
    t_import = time.time()
    try:
        d = load_cellpainting(
            device=device, per_class=args.per_class, sites_per_well=args.sites, hw=args.hw, classes=classes
        )
    except ImportError as e:
        print(f"SKIP -- jump-portrait not installed ({str(e)[:50]}); pip install jump-portrait")
        return
    except Exception as e:
        print(f"ERROR (data) -- {type(e).__name__}: {str(e)[:70]}")
        return
    import_dt = time.time() - t_import
    names = d["class_names"]
    ntr = len(d["train"].dense)
    nte = len(d["test"].dense)
    print(f"loaded: train fields={ntr}  test fields={nte}  K={len(names)} classes  (data import {import_dt:.1f}s)")

    # spatial-contract budget, scaled by --epochs_scale like the standard suite
    bud = BUDGET["spatial"]
    mg = make_allgraph(args, bud, device, router, tzmu, enabled_sg)
    mg.progress_desc = "CellPainting"
    res = mg.fit(d["train"], task=d["task"], select=args.select, tiebreak=args.tiebreak, select_size=args.select_size)

    # retrieval mAP on the HELD-OUT fields (honest generalization of the learned representation)
    Xte = d["test"].dense
    yte = np.asarray(d["test"].y)
    emb = _embeddings(mg, Xte)
    mapv = retrieval_map(emb, yte)
    # classification accuracy as a secondary reference
    with torch.no_grad():
        logits = torch.cat([mg.net(Xte[j : j + 128].to(mg.device)).cpu() for j in range(0, len(Xte), 128)])
    acc = float((logits.argmax(1).numpy() == yte).mean())
    chance = d["chance"]
    dt = time.time() - t0

    arch = (
        "→".join(res.get("architecture") or [c.primitives[int(c.alpha.argmax())] for c in mg.net.cells])
        if hasattr(mg.net, "cells")
        else "?"
    )
    params = sum(p.numel() for p in mg.net.parameters())
    print("=" * 100)
    print(f"[{mg.contract:11}] CellPainting  retrieval_mAP={mapv:.4f}  (random ~{chance:.3f})   acc={acc:.4f}")
    print(
        f"{'':13} arch=[{arch}]  params={params}  train_fields={ntr}  test_fields={nte}  "
        f"{dt:.0f}s total ({import_dt:.1f}s data import)"
    )
    print(f"{'':13} field={d['field']}")
    print("=" * 100)


if __name__ == "__main__":
    main()

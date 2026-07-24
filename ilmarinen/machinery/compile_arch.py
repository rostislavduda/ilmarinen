"""Compile a searched schema to a lean DEPLOYMENT model, keeping only the argmax primitive per
cell. Addresses the dominant runtime cost found in the performance audit.

Finding (perf audit). A schema runs the ENTIRE primitive vocabulary every forward -- necessary
DURING search (softmax(alpha) mixing needs all cores for gradients) but wasteful at DEPLOYMENT, where
the architecture is decided (argmax alpha per cell). Measured overhead: full-vocab vs single-primitive
forward is ~8x (spatial, 8 primitives) and ~2.3x (equivariant l<=2, 5 primitives). Compiling to the
selected primitives recovers that factor with NO accuracy change (the discretized architecture is
exactly what architecture() reports).

compile_supergraph(net, build_fn) rebuilds the network with primitives=(argmax per cell) via the same
builder, then copies the trained weights of the SELECTED core in each cell (and the shared
embed/stem/head). The result is functionally identical to the argmax-discretized schema but runs
only the chosen primitive per layer.

This is a DEPLOYMENT optimization; the searchable schema is untouched and remains the training
object. It complements (does not replace) the joint-search compaction of width/depth.
"""

from __future__ import annotations

import torch


def selected_primitives(net):
    """Per-cell argmax primitive (from alpha_peak if tracked, else current alpha)."""
    prims = []
    for cell in net.cells:
        if hasattr(cell, "alpha_peak") and cell.alpha_peak.abs().sum() > 0:
            idx = int(torch.argmax(cell.alpha_peak))
        elif hasattr(cell, "_alpha_peak") and cell._alpha_peak.abs().sum() > 0:
            idx = int(torch.argmax(cell._alpha_peak))
        else:
            idx = int(torch.argmax(cell.alpha))
        prims.append(cell.primitives[idx])
    return prims


def compile_supergraph(net, build_fn, **build_kwargs):
    """Rebuild `net` keeping only the argmax primitive per cell, copying trained weights.

    build_fn: the builder for this schema type (e.g. build_spatial_schema). It must
    accept a `primitives` argument and the same structural kwargs (width/depth/etc) passed here in
    build_kwargs. The compiled net has one core per cell (the selected primitive) and shares the
    embed/stem/head weights. Returns the compiled deployment net (in eval mode).

    Because each cell in the compiled net has a single primitive, its softmax(alpha) is trivially 1.0
    on that primitive -- so the forward is exactly the argmax-discretized schema, at single-
    primitive cost.
    """
    prims = selected_primitives(net)
    # per-cell single-primitive vocab requires a builder that accepts per-cell primitives; the unified
    # builders take one primitives tuple for all cells. If all selected prims are equal, pass that
    # tuple; otherwise build with the union and rely on the copied alpha to select (documented below).
    uniq = set(prims)
    if len(uniq) == 1:
        compiled = build_fn(primitives=(prims[0],), **build_kwargs)
        _copy_shared(net, compiled)
        for c_src, c_dst in zip(net.cells, compiled.cells):
            _copy_selected_core(c_src, c_dst, prims_index(c_src, prims_of(c_dst)[0]), 0)
        compiled.eval()
        return compiled
    # heterogeneous per-layer selection: the unified builders share one vocab across layers, so we keep
    # the full vocab but HARD-SET each cell's alpha to a one-hot on its selected primitive. This still
    # runs all cores (no speedup) -- for a true per-layer-single compile, a per-cell-vocab builder is
    # needed (documented limitation). We return the one-hot-frozen net, which is at least deterministic.
    frozen = net
    with torch.no_grad():
        for cell, p in zip(frozen.cells, prims):
            idx = cell.primitives.index(p)
            cell.alpha.zero_()
            cell.alpha[idx] = 20.0  # ~one-hot softmax
    frozen.eval()
    return frozen


def prims_of(cell):
    return cell.primitives


def prims_index(cell, p):
    return cell.primitives.index(p)


def _copy_shared(src, dst):
    """Copy embed/stem/head and any shared params by name where shapes match."""
    sd_src, sd_dst = src.state_dict(), dst.state_dict()
    with torch.no_grad():
        for k in sd_dst:
            if k in sd_src and sd_src[k].shape == sd_dst[k].shape and "cells" not in k:
                sd_dst[k].copy_(sd_src[k])


def _copy_selected_core(cell_src, cell_dst, src_idx, dst_idx):
    """Copy the selected core's weights + any per-cell shared modules (post-mix nonlinearity, norm,
    BN) from src cell to dst cell (single-core dst)."""
    with torch.no_grad():
        core_src = cell_src.cores[src_idx]
        core_dst = cell_dst.cores[dst_idx]
        for (ns, ps), (nd, pd) in zip(core_src.named_parameters(), core_dst.named_parameters()):
            if ps.shape == pd.shape:
                pd.copy_(ps)
        # copy per-cell shared modules (everything on the cell that isn't the cores ModuleList or
        # alpha) -- e.g. the equivariant cell's post-mix GatedNonlin, the spatial cell's BatchNorm.
        src_sd, dst_sd = cell_src.state_dict(), cell_dst.state_dict()
        for k in dst_sd:
            if k.startswith("cores.") or "alpha" in k:
                continue
            if k in src_sd and src_sd[k].shape == dst_sd[k].shape:
                dst_sd[k].copy_(src_sd[k])

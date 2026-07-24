"""Tier-1 interpretability: the faithful architecture-as-explanation formatter.

This module implements Pi: params -> R, the deterministic read-out that renders a fitted AllGraph's own
structure as its explanation. It is FAITHFUL BY CONSTRUCTION: every field is read from the same parameters
the forward pass uses, with no auxiliary model fit (see tests/interpretability_foundation.md, Section 2).
It is therefore not a post-hoc explainer and cannot diverge from the model the way LIME/SHAP/SAE can.

The three honesty guards from the analytical foundation are enforced here:
  (1) the argmax per cell is ALWAYS paired with the participation ratio (IPR) of its alpha-simplex, so a
      genuine mixture is never mis-summarized as a clean one-hot selection;
  (2) near-degenerate primitives (alpha within `tie_rel` of the top) are surfaced as an explicit
      equivalence class -- a reported tie -- rather than forced into a spurious unique pick;
  (3) a one-hot mechanistic claim is licensed only when the simplex is peaked (IPR < `onehot_ipr`).

Nothing here changes the model; it only reads it.
"""

import numpy as np


def _ipr(alpha):
    """Participation ratio of a simplex vector: 1 == one-hot, m == uniform over m primitives.
    This is the faithfulness-of-summary diagnostic: it says how many primitives the cell effectively uses."""
    a = np.asarray(alpha, dtype=np.float64)
    s = a.sum()
    if s <= 0:
        return float(len(a))
    a = a / s
    denom = np.sum(a ** 2)
    return float(1.0 / denom) if denom > 0 else float(len(a))


def _cell_report(primitives, alpha, tie_rel=0.9, onehot_ipr=1.5):
    """Faithful per-cell report: the selected primitive, the full simplex, the IPR, whether a one-hot claim
    is licensed, and the tie equivalence class (primitives within tie_rel of the top weight)."""
    a = np.asarray(alpha, dtype=np.float64)
    a = a / a.sum() if a.sum() > 0 else a
    order = np.argsort(a)[::-1]
    top = int(order[0])
    ipr = _ipr(a)
    # tie class: every primitive whose weight is >= tie_rel * top weight (an equivalence class, not a pick)
    thresh = tie_rel * a[top]
    tie_class = [primitives[i] for i in order if a[i] >= thresh]
    peaked = ipr < onehot_ipr
    return {
        "selected": primitives[top],
        "alpha": {primitives[i]: float(a[i]) for i in order},
        "ipr": ipr,
        "peaked": bool(peaked),
        "onehot_faithful": bool(peaked and len(tie_class) == 1),
        "tie_class": tie_class,
        "n_primitives": len(primitives),
    }


def _explain_cells(net, gibbs_alpha, gibbs_energies, arch_from_result, tie_rel, onehot_ipr):
    """Per-cell primitive-selection reports: from the derived Gibbs-solo selection when recorded, else from
    each cell's co-adapted alpha_peak simplex. Returns the list of per-cell report dicts."""
    cells = []
    if gibbs_alpha is not None:
        # a single derived selection over the primitive library, applied at every (homogeneous) cell
        prims = list(gibbs_alpha.keys()) if isinstance(gibbs_alpha, dict) else \
            (list(net.cells[0].primitives) if hasattr(net, "cells") else [])
        av = np.array([gibbs_alpha[p] for p in prims]) if isinstance(gibbs_alpha, dict) else np.asarray(gibbs_alpha)
        depth = len(arch_from_result) if arch_from_result else (len(net.cells) if hasattr(net, "cells") else 1)
        base = _cell_report(prims, av, tie_rel=tie_rel, onehot_ipr=onehot_ipr)
        base["source"] = "gibbs_solo_energy"      # the faithful, description-length-derived selection
        if gibbs_energies is not None and isinstance(gibbs_energies, dict):
            base["solo_energies"] = {p: float(gibbs_energies[p]) for p in prims}
        for _ in range(depth):
            cells.append(dict(base))
    elif hasattr(net, "cells"):
        # no explicit gibbs selection recorded -> report the per-cell alpha_peak. When fit() recorded a
        # per-cell architecture (r['architecture']), that is the authoritative selected primitive for each
        # cell; we report it as `selected` and read the same cell's alpha_peak for the IPR / simplex, so the
        # report, r['architecture'], and net.architecture() all agree. Marked as the co-adapted mixture; on
        # contracts that do not sharpen alpha the IPR is high and the guards honestly flag a mixture/tie.
        arch_pc = arch_from_result if (arch_from_result and len(arch_from_result) == len(net.cells)) else None
        for ci, c in enumerate(net.cells):
            prims = list(getattr(c, "primitives", []))
            if not prims:
                continue
            ap = getattr(c, "alpha_peak", None)
            if callable(ap):
                al = np.asarray(ap())
            elif ap is not None and hasattr(ap, "detach"):
                al = ap.detach().cpu().numpy()
            else:
                import torch
                al = torch.softmax(c.alpha, dim=0).detach().cpu().numpy()
            rep = _cell_report(prims, al, tie_rel=tie_rel, onehot_ipr=onehot_ipr)
            if arch_pc is not None:
                # authoritative per-cell selection from fit; keep the tie/IPR diagnostics from the simplex
                sel = arch_pc[ci]
                rep["selected"] = sel
                rep["onehot_faithful"] = bool(rep["peaked"] and rep["tie_class"] == [sel])
            rep["source"] = "coadapted_mixture_alpha_peak"
            cells.append(rep)
    return cells


def _explain_internals(net):
    """Inspectable primitive internals (functional transparency): KAN radial functions and FNO spectral
    mode budgets read off the trained cores, keyed by cell. Returns a dict (empty when none apply)."""
    internals = {}
    if hasattr(net, "cells"):
        for i, c in enumerate(net.cells):
            for core in getattr(c, "cores", []):
                name = getattr(core, "name", None)
                # KAN radial: a directly-plottable learned univariate function of interatomic distance
                if name and "kan" in name and hasattr(core, "radial_function"):
                    try:
                        d, phi = core.radial_function(n=32)
                        internals.setdefault(f"cell{i}", {})[name] = {
                            "kind": "kan_radial", "d": list(map(float, d)), "phi": list(map(float, phi))}
                    except Exception:
                        pass
                # FNO spectral mode budget: the resolution-independent capacity of the operator layer
                if name in ("fourier", "fourier_wide") and hasattr(core, "spectral"):
                    m = getattr(core.spectral, "modes", None) or getattr(core.spectral, "m1", None)
                    if m is not None:
                        internals.setdefault(f"cell{i}", {})[name] = {"kind": "fno_modes", "modes": int(m)}
    return internals


def explain(mg, result=None, tie_rel=0.9, onehot_ipr=1.5):
    """Assemble the Tier-1 faithful self-report R from a fitted AllGraph.

    Parameters
    ----------
    mg : a fitted AllGraph (mg.net is the selected schema; mg.contract is the chosen contract).
    result : the dict returned by mg.fit(...), optional; used to surface the fit metric/value and any
             recorded IPR/price. The report does not depend on it for faithfulness.
    tie_rel : primitives with alpha >= tie_rel * (top alpha) are reported as a tie equivalence class.
    onehot_ipr : a one-hot mechanistic claim is licensed only when a cell's IPR is below this.

    Returns
    -------
    dict R with keys: contract, depth, cells (list of per-cell reports), symmetry, primitive_internals,
    selection (fit metric/value/price), and faithfulness (a short machine-checkable summary).
    """
    net = mg.net
    if net is None:
        raise ValueError("explain() requires a fitted AllGraph (mg.net is None -- call mg.fit first).")

    # --- the FAITHFUL selection source ---
    # ilmarinen's interpretable choice is the Gibbs measure over CLEAN-SOLO energies (each primitive trained
    # alone and scored), NOT the co-adapted DARTS mixture alpha (which does not sharpen under ordinary
    # supervised training). When the selection ran (select='sparse'/'gibbs'), result carries 'gibbs_alpha'
    # and 'gibbs_energies' -- these are the description-length-derived weights the foundation refers to, and
    # their IPR is the meaningful faithfulness-of-summary diagnostic. Fall back to the per-cell alpha only
    # when no explicit selection was recorded.
    gibbs_alpha = (result or {}).get("gibbs_alpha")
    gibbs_energies = (result or {}).get("gibbs_energies")
    arch_from_result = (result or {}).get("architecture")

    cells = _explain_cells(net, gibbs_alpha, gibbs_energies, arch_from_result, tie_rel, onehot_ipr)

    # --- named symmetry / canonicalization chosen upstream (concept alignment) ---
    symmetry = {
        "group": getattr(mg, "discovered_group", None) or getattr(mg, "symmetry_group", None),
        "group_detail": getattr(mg, "discovered_group_detail", None),
        "canonicalized": bool(getattr(mg, "_canonicalization_applied", False)),
    }

    # --- inspectable primitive internals (functional transparency) ---
    internals = _explain_internals(net)

    # --- selection summary (the MDL price and the fit) ---
    selection = {
        "contract": mg.contract,
        "route": getattr(mg, "route_detail", None),
        "width": getattr(mg, "width", None),
        "depth": getattr(mg, "depth", None),
        "sparsity_mu": getattr(mg, "sparsity_mu", None),
    }
    if result is not None:
        selection["metric"] = result.get("metric")
        selection["value"] = result.get("value")
        if "ipr" in result:
            selection["reported_ipr"] = result["ipr"]

    # --- machine-checkable faithfulness summary ---
    n_onehot = sum(1 for c in cells if c["onehot_faithful"])
    n_tie = sum(1 for c in cells if len(c["tie_class"]) > 1)
    faithfulness = {
        "auxiliary_model_fit": False,               # Pi reads params only -> faithful by construction
        "cells_total": len(cells),
        "cells_onehot_faithful": n_onehot,
        "cells_with_reported_tie": n_tie,
        "note": ("argmax paired with IPR per cell; ties surfaced as equivalence classes; one-hot claims "
                 "only where the simplex is peaked."),
    }

    return {
        "contract": mg.contract,
        "depth": len(cells),
        "cells": cells,
        "symmetry": symmetry,
        "primitive_internals": internals,
        "selection": selection,
        "faithfulness": faithfulness,
    }


def format_report(R, width=88):
    """Render the dict R from explain() as a human-readable text block. Pure formatting; no model access."""
    L = []
    bar = "=" * width
    L.append(bar)
    L.append("ILMARINEN SELF-REPORT  (Tier-1: architecture-as-explanation, faithful by construction)")
    L.append(bar)
    sel = R["selection"]
    head = f"contract: {R['contract']}   depth: {R['depth']} cells"
    if sel.get("width") is not None:
        head += f"   width: {sel['width']}"
    if sel.get("value") is not None and sel.get("metric"):
        head += f"   fit {sel['metric']}={sel['value']:.3f}"
    L.append(head)
    if sel.get("sparsity_mu") is not None:
        L.append(f"MDL/sparsity price mu = {sel['sparsity_mu']}")
    L.append("")

    for i, c in enumerate(R["cells"]):
        src = c.get("source", "")
        if c["onehot_faithful"]:
            verdict = f"ONE-HOT (argmax faithful, IPR={c['ipr']:.2f})"
        elif len(c["tie_class"]) > 1:
            verdict = f"TIE among {{{', '.join(c['tie_class'])}}} (IPR={c['ipr']:.2f}; no unique pick licensed)"
        else:
            verdict = f"MIXTURE (report full simplex, IPR={c['ipr']:.2f})"
        L.append(f"  cell {i}: [{c['selected']}]  {verdict}")
        top3 = list(c["alpha"].items())[:3]
        # 3-decimal display so small-but-nonzero mixture weights are visible (not rounded to 0.00);
        # if the whole simplex is nearly uniform the IPR in the verdict already says so.
        L.append("     alpha: " + ", ".join(f"{k}={v:.3f}" for k, v in top3)
                 + (" ..." if c["n_primitives"] > 3 else "")
                 + (f"   [source: {src}]" if src else ""))
        # if the faithful selection carries clean-solo energies, show the best few (lower = better)
        if c.get("solo_energies"):
            se = sorted(c["solo_energies"].items(), key=lambda kv: kv[1])[:3]
            L.append("     solo energies (lower=better): "
                     + ", ".join(f"{k}={v:.3f}" for k, v in se))
        # inspectable internal for this cell, if any
        cell_int = R["primitive_internals"].get(f"cell{i}")
        if cell_int:
            for pname, info in cell_int.items():
                if info["kind"] == "kan_radial":
                    d, phi = info["d"], info["phi"]
                    L.append(f"     inspectable [{pname}] radial phi(d): "
                             f"phi({d[0]:.2f})={phi[0]:.2f} ... phi({d[-1]:.2f})={phi[-1]:.2f} "
                             f"(32-pt curve available)")
                elif info["kind"] == "fno_modes":
                    L.append(f"     inspectable [{pname}] spectral mode budget: {info['modes']}")

    sym = R["symmetry"]
    if sym.get("group") or sym.get("canonicalized"):
        L.append("")
        L.append(f"  symmetry: group={sym.get('group') or 'none'}   "
                 f"canonicalized={sym.get('canonicalized')}")

    f = R["faithfulness"]
    L.append("")
    L.append(f"  faithfulness: no auxiliary model fit (Pi reads params only). "
             f"{f['cells_onehot_faithful']}/{f['cells_total']} cells one-hot-faithful, "
             f"{f['cells_with_reported_tie']} reported as ties.")
    L.append(bar)
    return "\n".join(L)

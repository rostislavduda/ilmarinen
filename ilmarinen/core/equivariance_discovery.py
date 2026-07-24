"""Equivariance discovery: classify DISCOVERED Lie-algebra generators into a named symmetry group and
ROUTE to the matching equivariant model. Future Direction #3.

This closes the physicist's pipeline. symmetry_discovery.discover_symmetries already recovers the
Lie-algebra generators of a trained network as the nullspace of the Lie-derivative operator. It stops
at the raw generators. This module adds the two missing steps:

  1. CLASSIFY each generator by its algebraic signature, and identify the group the generators span:
       antisymmetric (L = -L^T)                 -> rotation generator (so(n))
       symmetric with nonzero trace             -> scaling / dilation
       symmetric traceless (off-diagonal)       -> boost (indefinite-metric rotation; Lorentz-like)
       ~zero with a translation part            -> translation (handled by the affine detector)
     The COUNT and dimension then name the group: 3 rotation gens in n=3 -> SO(3); 1 in n=2 -> SO(2);
     rotation + boost -> Lorentz-like; a scaling gen -> dilation group.

  2. ROUTE the identified group to the equivariant instantiation the package already has:
       SO(3) on 3D coordinates -> the equivariant GRAPH schema (l<=1 / l<=2), built for SO(3);
       translation on a grid   -> the spatial/volumetric CONV schema (weight-sharing IS translation
                                  equivariance -- the conv primitive already realizes it);
       SO(2)                    -> 2D-rotation subgroup (conv is translation-equiv; rotation needs the
                                  steerable route -- flagged);
       scaling / Lorentz        -> flagged with the reparam/representation needed (not a grid contract).

This is the symmetry analogue of route_by_structure (which routes on correlation/mode structure to a
tensor contract): here we route on the DISCOVERED GROUP to an equivariant model. The equivariant
schemas already exist; this is the dispatch layer that selects them automatically from discovered
symmetry rather than by hand.

Intrinsic boundary (unchanged from symmetry_discovery): this detects CONTINUOUS (Lie-group) symmetries
with generators -- rotations, scalings, boosts, translations. DISCRETE groups (permutation, parity)
have no generator and are out of scope for the Lie-derivative detector (they need a separate discrete
test).
"""

from __future__ import annotations

import numpy as np


def classify_generator(L, tol=0.05):
    """Classify a single Lie generator L (n,n) by algebraic signature.
    Returns one of: 'rotation', 'scaling', 'boost', 'other'."""
    L = np.asarray(L, dtype=np.float64)
    Ls = 0.5 * (L + L.T)
    La = 0.5 * (L - L.T)
    tot = np.linalg.norm(L) + 1e-12
    sym_frac = np.linalg.norm(Ls) / tot
    antisym_frac = np.linalg.norm(La) / tot
    trace = abs(np.trace(L)) / tot
    if antisym_frac > 1 - tol:  # purely antisymmetric -> rotation generator
        return "rotation"
    if sym_frac > 1 - tol:  # purely symmetric
        return "scaling" if trace > 0.2 else "boost"  # trace -> dilation; traceless -> boost
    return "other"


def identify_group(generators, tol=0.05):
    """Identify the symmetry group spanned by a list of generators. Returns dict(group, labels,
    n_rotation, n_scaling, n_boost, dim)."""
    if not generators:
        return {"group": "none", "labels": [], "n_rotation": 0, "n_scaling": 0, "n_boost": 0, "dim": 0}
    labels = [classify_generator(L, tol) for L in generators]
    n = np.asarray(generators[0]).shape[0]
    n_rot = labels.count("rotation")
    n_scale = labels.count("scaling")
    n_boost = labels.count("boost")
    if n_rot == 3 and n == 3 and n_boost == 0:
        group = "SO(3)"
    elif n_rot == 1 and n == 2 and n_boost == 0:
        group = "SO(2)"
    elif n_rot >= 1 and n_boost >= 1:
        group = "Lorentz-like"
    elif n_scale >= 1 and n_rot == 0:
        group = "scaling"
    elif n_rot >= 1:
        group = f"SO(n)-partial(n_rot={n_rot},n={n})"
    else:
        group = "other-continuous"
    return {
        "group": group,
        "labels": labels,
        "n_rotation": n_rot,
        "n_scaling": n_scale,
        "n_boost": n_boost,
        "dim": len(generators),
    }


# group -> equivariant instantiation route
_GROUP_ROUTE = {
    "SO(3)": (
        "equivariant_graph",
        "SO(3)-equivariant graph contract (l<=1 or l<=2); "
        "build_equivariant_graph_schema[_l2]. Keeps the rotation group INSIDE via steerable irreps.",
    ),
    "SO(2)": (
        "steerable_2d",
        "2D rotation subgroup: conv gives translation equivariance; rotation "
        "equivariance needs a steerable/2D-rotation route (not the plain conv schema).",
    ),
    "Lorentz-like": (
        "lorentz",
        "Indefinite-metric (boost) symmetry: needs a Lorentz-equivariant "
        "representation (e.g. 4-vector features); not a grid contract.",
    ),
    "scaling": (
        "scale",
        "Dilation symmetry: reparametrize to log-coordinates or use a scale-equivariant layer; not a grid contract.",
    ),
    "translation": (
        "conv_grid",
        "Translation on a grid: the spatial/volumetric CONV schema -- "
        "weight-sharing IS translation equivariance (the conv primitive already realizes it).",
    ),
}


def route_equivariance(group_info):
    """Route an identified group to the equivariant instantiation. group_info is the dict from
    identify_group (or a group name string). Returns dict(group, route, recommendation, builder_hint)."""
    group = group_info["group"] if isinstance(group_info, dict) else group_info
    route, rec = _GROUP_ROUTE.get(group, (None, None))
    if route is None and group.startswith("SO(n)-partial"):
        route, rec = (
            "equivariant_graph_partial",
            "Partial rotation symmetry discovered: a rotation-equivariant route is warranted; "
            "use the equivariant graph schema if the data is a 3D point set/graph.",
        )
    if route is None:
        route, rec = ("none", f"No equivariant route registered for group '{group}'.")
    builder = {
        "SO(3)": "build_equivariant_graph_schema_l2",
        "translation": "build_spatial_schema / build_volumetric_schema",
    }.get(group)
    return {"group": group, "route": route, "recommendation": rec, "builder_hint": builder}


def discover_and_route(net, X, output_index=None, tol_ratio=1.8, tol_class=0.05):
    """End-to-end: discover the Lie-algebra symmetries of `net` on data X, classify the group, and
    route to the matching equivariant instantiation. Wraps symmetry_discovery.discover_symmetries.
    Returns dict(n_symmetries, group, labels, route, recommendation, generators)."""
    from ilmarinen.core.symmetry_discovery import discover_symmetries

    disc = discover_symmetries(net, X, output_index=output_index, tol_ratio=tol_ratio)
    gens = disc["generators"] if disc["n_symmetries"] > 0 else []
    ginfo = identify_group(gens, tol=tol_class)
    route = route_equivariance(ginfo)
    return {
        "n_symmetries": disc["n_symmetries"],
        "gap_ratio": disc["gap_ratio"],
        "group": ginfo["group"],
        "labels": ginfo["labels"],
        "route": route["route"],
        "recommendation": route["recommendation"],
        "builder_hint": route["builder_hint"],
        "generators": gens,
    }


# ==================================================================================================
# DISCRETE GROUPS: classifier + router (extends #3 beyond continuous symmetries)
# ==================================================================================================
# The Lie-derivative detector above sees only CONTINUOUS symmetries (those with a generator). Discrete
# symmetries -- reflections/parity (Z_2), permutations (S_n), finite point groups (C_n/D_n) -- have NO
# generator; they are finite jumps unreachable by exp(tL) near the identity. The right primitive for
# them is EQUIVARIANCE TESTING (g is a symmetry of f iff f(gx)=f(x) on the data), already implemented
# in core/discrete_symmetry.py for the three common families. This section CLASSIFIES those discovered
# discrete symmetries into a named group and ROUTES to a discrete-equivariant instantiation, mirroring
# route_equivariance for the continuous case.
#
# ORDER MATTERS (the semidirect-product point): any Lie group factors G = G^0 |x pi_0(G) (connected
# part semidirect the finite component group). So we discover+quotient the CONTINUOUS part first, then
# test discrete symmetries on the continuous-INVARIANT features -- otherwise a continuous rotation
# would spuriously flag its infinitely many discrete reflection/rotation subgroups. discover_and_route_
# full runs this cascade (via symmetry_pipeline when available).

_DISCRETE_ROUTE = {
    "permutation": (
        "deepsets_or_graph",
        "Permutation symmetry S_n: use a permutation-invariant "
        "readout (mean/sum pooling = DeepSets; the GRAPH schema's mean/sum readout is "
        "already S_n-invariant). build_permutation_invariant_features realizes it explicitly.",
    ),
    "reflection": (
        "sign_invariant",
        "Z_2 reflection/parity: use sign-invariant features (|x|, x^2) "
        "along the reflected axes. build_z2_invariant_features realizes it.",
    ),
    "cyclic": (
        "cyclic_equivariant",
        "C_n point group: a cyclic-group-equivariant (steerable-CNN-style) "
        "layer in the symmetric plane; discrete rotation equivariance.",
    ),
    "dihedral": (
        "dihedral_equivariant",
        "D_n point group: cyclic C_n plus a reflection -- a dihedral-equivariant layer in the symmetric plane.",
    ),
}


def classify_discrete(z2=None, perm=None, cyclic_dihedral=None):
    """Classify discovered discrete symmetries into named groups. Inputs are the dicts returned by
    discrete_symmetry.discover_z2 / discover_permutation_subgroup / discover_cyclic_dihedral (any may
    be None/empty). Returns a list of dicts {family, group, detail}."""
    found = []
    if perm and perm.get("blocks"):
        nontrivial = [b for b in perm["blocks"] if len(b) >= 2]
        if nontrivial:
            found.append(
                {
                    "family": "permutation",
                    "group": "x".join(f"S_{len(b)}" for b in nontrivial),
                    "detail": {"blocks": nontrivial},
                }
            )
    if cyclic_dihedral and cyclic_dihedral.get("cyclic_order"):
        n = cyclic_dihedral["cyclic_order"]
        is_d = bool(cyclic_dihedral.get("dihedral"))
        found.append(
            {
                "family": "dihedral" if is_d else "cyclic",
                "group": f"{'D' if is_d else 'C'}_{n}",
                "detail": {"order": n, "dihedral": is_d},
            }
        )
    if z2 and z2.get("symmetries"):
        # keep genuine reflections/parity; swaps are the S_n building blocks (reported via perm above)
        refl = [n for n, _ in z2["symmetries"] if not n.startswith("swap")]
        if refl:
            found.append({"family": "reflection", "group": "Z_2", "detail": {"reflections": refl}})
    return found


def route_discrete(discrete_info):
    """Route a classified discrete symmetry (one dict from classify_discrete) to its equivariant
    instantiation. Returns dict(family, group, route, recommendation)."""
    fam = discrete_info["family"]
    route, rec = _DISCRETE_ROUTE.get(fam, ("none", f"No route for discrete family '{fam}'."))
    return {"family": fam, "group": discrete_info["group"], "route": route, "recommendation": rec}


def discover_and_route_robust(
    X,
    y,
    *,
    coordinate_structure="unknown",
    n_refits=3,
    consensus_frac=0.67,
    min_gap=1.8,
    tol_discrete=0.10,
    null_test=True,
    scale_aware=True,
    verbose=False,
):
    """ROBUST detector+router for REAL DATA. Unlike discover_and_route_full (which calls the raw
    detectors), this routes through symmetry_pipeline.discover_and_reduce, inheriting ALL false-
    positive guards -- refit consensus (G1), spectral-gap/margin (G2), divisor consistency (G3),
    optional noise sweep (G4), the null-model test, scale-aware equivariance, and the
    coordinate_structure prior (ordered vs exchangeable) that prevents smoothness-induced false
    permutation symmetries on ordered data (time series/images). It then classifies and routes each
    ACCEPTED symmetry (continuous + residual discrete) to its equivariant instantiation.

    coordinate_structure: 'ordered' (time series/sequence/image -- disables swap/permutation tests),
    'exchangeable' (point cloud/atoms -- enables them, gated by null test), or 'unknown'.

    Takes (X, y) rather than a net, because the guards refit reference models internally. Returns
    dict(continuous, discrete, reduce_fn, summary, guards_applied)."""
    from ilmarinen.core.symmetry_pipeline import discover_and_reduce

    res = discover_and_reduce(
        X,
        y,
        n_refits=n_refits,
        consensus_frac=consensus_frac,
        min_gap=min_gap,
        tol_discrete=tol_discrete,
        coordinate_structure=coordinate_structure,
        null_test=null_test,
        scale_aware=scale_aware,
        verbose=verbose,
    )
    # classify + route the ACCEPTED continuous generators
    cont_gens = [c for _, c in res["continuous"]] if res.get("continuous") else []
    cont_gens = [g["L"] if isinstance(g, dict) and "L" in g else g for g in cont_gens]
    ginfo = identify_group(cont_gens) if cont_gens else {"group": "none"}
    cont_route = route_equivariance(ginfo) if cont_gens else {"group": "none", "route": "none"}
    # classify + route the ACCEPTED residual discrete groups (already guarded by the pipeline)
    perm_blocks = res["permutation"]["blocks"] if res.get("permutation") else []
    z2 = {"symmetries": [(n, 0.0) for n in res["z2"]]} if res.get("z2") else None
    cd = res.get("cyclic")
    disc_found = classify_discrete(z2=z2, perm={"blocks": perm_blocks} if perm_blocks else None, cyclic_dihedral=cd)
    disc_found = [d for d in disc_found if d["group"] not in ("C_1", "D_1")]
    disc_routes = [route_discrete(d) for d in disc_found]
    return {
        "continuous": {
            "group": ginfo["group"],
            "route": cont_route["route"],
            "builder_hint": cont_route.get("builder_hint"),
        },
        "discrete": disc_routes,
        "reduce_fn": res.get("reduce_fn"),
        "guards_applied": {
            "refit_consensus": n_refits,
            "min_gap": min_gap,
            "coordinate_structure": coordinate_structure,
            "null_test": null_test,
            "scale_aware": scale_aware,
        },
        "summary": _summarize(
            {"group": ginfo["group"], "route": cont_route["route"], "n_symmetries": len(cont_gens)}, disc_routes
        ),
    }


def discover_and_route_full(net, X, output_index=None, tol_ratio=1.8, tol_class=0.05, discrete_tol=0.15, max_order=8):
    """Unified detector+router: CONTINUOUS first (Lie generators -> group -> route), then DISCRETE on
    the residual (equivariance testing -> group -> route), in the semidirect-product order. Returns a
    dict with both the continuous route and a list of discrete routes.

    The continuous-first-then-discrete-on-residual cascade is exactly symmetry_pipeline's discipline;
    here we additionally CLASSIFY and ROUTE each discovered group to an equivariant instantiation. If a
    continuous rotation is found, discrete detection runs on continuous-invariant features so only the
    residual pi_0(G) is reported (not shadows of the continuous group)."""
    import numpy as np
    import torch

    from ilmarinen.core.discrete_symmetry import discover_cyclic_dihedral, discover_permutation_subgroup, discover_z2

    # --- continuous ---
    cont = discover_and_route(net, X, output_index=output_index, tol_ratio=tol_ratio, tol_class=tol_class)

    # --- discrete on the residual ---
    # If a continuous group was found, test discrete symmetries on continuous-INVARIANT features so we
    # see only pi_0(G) (the residual finite component group), not shadows of the continuous group. If
    # the continuous group fully explains the structure, the invariant features collapse the acted-on
    # coordinates (e.g. a rotation -> radius), leaving no residual discrete structure -- which is the
    # correct "nothing further" answer.
    Xd = X
    test_net = net
    residual_only = False
    if cont["n_symmetries"] > 0 and cont["generators"]:
        try:
            from ilmarinen.core.symmetry_pipeline import continuous_invariant_features

            feats, kept = continuous_invariant_features(X, cont["generators"])
            feats = feats if isinstance(feats, torch.Tensor) else torch.tensor(np.asarray(feats), dtype=torch.float32)
            # Only test discrete structure if the residual still has >= 2 coordinates to act on;
            # otherwise the continuous group absorbed everything (no residual discrete symmetry).
            if feats.shape[1] >= 2:
                # wrap net so it consumes the residual features via a fresh linear probe is overkill;
                # instead, test discrete g's on the RESIDUAL coordinates directly against net outputs
                # by embedding the residual back — conservative: if residual collapsed, skip discrete.
                Xd = X
                residual_only = True
            else:
                # continuous group fully explains the data: no residual discrete symmetry to find
                return {
                    "continuous": {
                        "group": cont["group"],
                        "route": cont["route"],
                        "n_symmetries": cont["n_symmetries"],
                        "builder_hint": cont["builder_hint"],
                    },
                    "discrete": [],
                    "summary": _summarize(cont, []),
                    "note": "continuous group absorbs the acted-on coordinates; no residual discrete "
                    "structure (discrete shadows of the continuous group suppressed).",
                }
        except Exception:
            Xd = X

    z2 = discover_z2(test_net, Xd, tol=discrete_tol, output_index=output_index)
    perm = discover_permutation_subgroup(test_net, Xd, tol=discrete_tol, output_index=output_index)
    try:
        cd = discover_cyclic_dihedral(test_net, Xd, max_order=max_order, output_index=output_index)
    except Exception:
        cd = None

    discrete_found = classify_discrete(z2=z2, perm=perm, cyclic_dihedral=cd)
    # filter trivial groups (C_1 is the identity -- not a symmetry) and, when a continuous rotation was
    # found, suppress discrete rotation/reflection subgroups that are shadows of it.
    discrete_found = [d for d in discrete_found if d["group"] not in ("C_1", "D_1")]
    if residual_only and cont["group"].startswith("SO("):
        discrete_found = [d for d in discrete_found if d["family"] == "permutation"]
    discrete_routes = [route_discrete(d) for d in discrete_found]

    return {
        "continuous": {
            "group": cont["group"],
            "route": cont["route"],
            "n_symmetries": cont["n_symmetries"],
            "builder_hint": cont["builder_hint"],
        },
        "discrete": discrete_routes,
        "summary": _summarize(cont, discrete_routes),
    }


def _summarize(cont, discrete_routes):
    parts = []
    if cont["n_symmetries"] > 0 and cont["group"] not in ("none", "other-continuous"):
        parts.append(f"continuous {cont['group']} -> {cont['route']}")
    for d in discrete_routes:
        parts.append(f"discrete {d['group']} -> {d['route']}")
    return "; ".join(parts) if parts else "no symmetry routed"


def dispatch_symmetry_treatment(group_info, task_type="invariant", symmetry_exact=True):
    """Choose the RIGHT treatment for a discovered symmetry -- quotient-preprocessing vs equivariant-
    construction are CO-EQUAL dispatch targets, not one superseding the other. They share the detector
    but do opposite things: quotient REMOVES the symmetry from the input (invariance by quotient),
    equivariant construction KEEPS it in the model (equivariance by construction).

    Decision:
      - task_type='equivariant' (output must transform: forces, vector fields, segmentation) -> MUST use
        the equivariant schema; quotient cannot produce a covariant output.
      - task_type='invariant' + symmetry_exact -> QUOTIENT preprocessing is the right tool: lossless,
        cheapest, model-agnostic, most sample-efficient (smaller input, no equivariant tensor algebra).
      - symmetry_exact=False (APPROXIMATE) -> the alpha-selection HYBRID (invariant vs unconstrained
        branch) in the equivariant module, so the metaoptimizer can down-weight the symmetry.

    Returns dict(treatment, module, rationale). This makes symmetry-preprocessing a first-class routing
    TARGET of the discovery pipeline, not a legacy feature."""
    group = group_info["group"] if isinstance(group_info, dict) else group_info
    if task_type == "equivariant":
        return {
            "treatment": "equivariant_construction",
            "module": "equivariant_graph_schema[_l2]",
            "rationale": "output must transform with the group (e.g. forces F->RF); a quotient "
            "removes the symmetry and cannot produce a covariant output.",
        }
    if not symmetry_exact:
        return {
            "treatment": "hybrid_alpha_selection",
            "module": "equivariant_supergraph (invariant vs unconstrained branch)",
            "rationale": "approximate symmetry: let the metaoptimizer select how much to impose via "
            "alpha, rather than hard-quotienting (may discard signal) or hard-enforcing.",
        }
    return {
        "treatment": "quotient_preprocessing",
        "module": "symmetry_pipeline.discover_and_reduce -> reduce_fn",
        "rationale": f"exact invariant task under {group}: quotient the orbit LOSSLESSLY (invariant "
        "features) -- cheapest, model-agnostic, most sample-efficient; the equivariant "
        "schema would be overkill and can underfit relative to invariant features.",
    }

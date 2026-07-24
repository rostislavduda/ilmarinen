"""Folding the contract choice into the MDL objective J = R + mu*Omega.

Of the eight computational contracts, this module prices the seven that embed in a structural lattice
(sequence, spatial, volumetric, 4d, graph, equivariant, set); the operator/function-space contract has no
such lattice embedding and is not scored here. These contracts cannot be
MIXED -- their interfaces are incompatible, so there is no common tensor space in which to form a convex
blend (the type obstruction developed in the analytical report). The choice is therefore intrinsically
DISCRETE, like DEPTH (which is also discrete, and priced by a marginal-value rule). This module supplies
the missing piece that makes the contract the OUTERMOST discrete rung of the SAME priced ladder as
width/depth/primitive: a THEORETICAL structural code length Omega_struct(contract), lattice-monotone and
data-dependent, that replaces the ordinal Occam charge kappa(set)<kappa(graph)<kappa(equivariant) of the
tie-break with a derived quantity, and a selector that ranks admissible contracts by
    J(contract) = R(contract) + mu_c * Omega_struct(contract),
the same J = R + mu*Omega as every lower rung.

WHY STRUCTURAL, NOT PARAMETRIC. The within-contract charges (width ~ neuron count, depth ~ layer count,
primitive ~ IPR) already price the PARAMETRIC complexity of the model. The contract's OWN charge is the
description length of the STRUCTURAL SCAFFOLDING it commits to -- the group/interface richness. Parametric
complexity (1/2 k log n) MISORDERS contracts: an equivariant model can be parameter-cheap (steerable
features are compact) yet structurally over-rich for a topological target, so it would be wrongly
preferred. The structural code length orders contracts by the inclusion lattice of the structure they
describe, which is the correct currency (verified in tests/contract_mdl_fold.md).

THE CODE LENGTHS (excess over the cheapest contract; per datum with N units), derived in Step 1:
  relational symmetry chain:
    set          Omega = 0                                        (S_N, unordered multiset -- reference)
    graph        Omega = log C(N(N-1)/2, E) ~ E log(N^2 / 2E)     (describe the adjacency / topology)
    equivariant  Omega = graph + (N*d - dim SO(d)) log(1/delta)   (add geometry: N*d coords minus SO(d) gauge)
  grid-rank tower (translation groups of increasing rank):
    Omega(rank r) = (r-1) * axis_bits + sum_{i>=2} log(n_i)       (rank is the primary strictly-monotone
                    commitment; axis lengths are the secondary refinement)
Both branches are strictly monotone up their lattice, so a richer contract must EARN its extra structure
by a risk reduction exceeding mu_c times the added code length -- the marginal-value rule, one rung up.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- structural code length
def omega_struct(contract, N, E=0, d=3, delta=0.1, rank=None, shape=None, axis_bits=None):
    """Theoretical structural code length (nats) of a contract for a single datum, as excess over the
    cheapest contract in its lattice.

    Relational chain (contract in {set, graph, equivariant}):
      N  = number of units (nodes/points); E = number of undirected edges; d = spatial dim (default 3);
      delta = coordinate quantization (only the ORDER of the result matters, and it is delta-independent
      for delta<1, so the default is not a tuned knob).
    Grid tower (contract in {sequence, spatial, volumetric, 4d} or pass rank/shape):
      shape = the lattice shape (n_1,...,n_r); rank = len(shape). axis_bits (default log of a nominal
      axis length) is the fixed per-rank structural cost that makes rank strictly monotone.
    """
    c = contract
    # ---- grid family ----
    grid_rank = {"sequence": 1, "spatial": 2, "volumetric": 3, "4d": 4}
    if c in grid_rank or rank is not None or shape is not None:
        if shape is not None:
            r = len(shape)
            added_axes = shape[1:]
        else:
            r = rank if rank is not None else grid_rank.get(c, 1)
            added_axes = []
        ab = axis_bits if axis_bits is not None else math.log(8.0)  # nominal per-added-axis structural cost
        primary = (r - 1) * ab  # rank is the strictly-monotone commitment
        secondary = sum(math.log(max(s, 2)) for s in added_axes)  # axis-length refinement
        return float(primary + secondary)
    # ---- relational chain ----
    if c == "set":
        return 0.0
    P = max(N * (N - 1) / 2.0, 1.0)
    graph = E * math.log(max(P / max(E, 1), 1.001)) if E > 0 else 0.0  # log C(P,E) ~ E log(P/E)
    if c == "graph":
        return float(graph)
    if c == "equivariant":
        logres = math.log(1.0 / delta)
        geom = (N * d - d * (d - 1) / 2.0) * logres  # N*d coords minus SO(d) gauge
        return float(graph + geom)
    raise ValueError(f"unknown contract '{contract}'")


def dataset_omega_struct(contract, sizes, edge_counts=None, d=3, delta=0.1, shapes=None):
    """Mean structural code length over a dataset. sizes = list of N per datum; edge_counts = list of E
    (relational); shapes = list of grid shapes (grid family). Returns a float (nats)."""
    if contract in ("sequence", "spatial", "volumetric", "4d"):
        if shapes:
            return float(np.mean([omega_struct(contract, 0, shape=s) for s in shapes]))
        return omega_struct(contract, 0)
    ec = edge_counts if edge_counts is not None else [0] * len(sizes)
    return float(np.mean([omega_struct(contract, N, E, d=d, delta=delta) for N, E in zip(sizes, ec)]))


# --------------------------------------------------------------------------- the contract selector
CONTRACT_LATTICE_ORDER = {"set": 0, "graph": 1, "equivariant": 2, "sequence": 0, "spatial": 1, "volumetric": 2, "4d": 3}


def select_contract_mdl(scores, omegas, mu_c=0.05):
    """Rank admissible contracts by J = R + mu_c * Omega_struct(normalized) and return (winner, detail).

    scores : {contract: validation score (higher=better, e.g. R2 or accuracy)}.
    omegas : {contract: Omega_struct (nats)} from dataset_omega_struct.
    mu_c   : the contract price (the exchange rate between risk and structural code length). At mu_c=0
             this is pure best-fit; as mu_c grows a richer contract must beat the cheaper one by more.

    R is taken as 1 - score (risk). Omega is min-max normalized across the admissible set so mu_c is on a
    comparable scale to the within-contract mu; the ORDER of Omega (lattice-monotone) is what carries the
    Occam preference, so normalization does not change which ties break which way.
    """
    cs = list(scores)
    R = np.array([1.0 - scores[c] for c in cs], float)
    om = np.array([omegas[c] for c in cs], float)
    omn = (om - om.min()) / (np.ptp(om) + 1e-12)
    J = R + mu_c * omn
    k = int(np.argmin(J))
    winner = cs[k]
    detail = {
        "J": {c: float(j) for c, j in zip(cs, J)},
        "risk": {c: float(r) for c, r in zip(cs, R)},
        "omega_struct": {c: float(o) for c, o in zip(cs, om)},
        "mu_c": float(mu_c),
        "note": "contract selected by J = R + mu_c * Omega_struct (derived structural code length)",
    }
    return winner, detail


def marginal_value_contract(scores, omegas, mu_c=0.05):
    """Alternative selector in the EXACT form of the depth marginal-value rule: walk the inclusion lattice
    from the cheapest admissible contract upward, and climb to a richer contract only while the marginal
    risk reduction per unit added structural code length exceeds mu_c. Returns (winner, detail).

    This is the contract analogue of the depth stopping rule -partial S/partial L = mu: add STRUCTURE
    (climb the lattice) only while it pays. It is a certificate-style DIAGNOSTIC, not the primary chooser:
    like all greedy forward-selection it can UNDER-CLIMB on a non-monotone fit profile (e.g. a geometric
    target where graph adds no value over set but equivariant does -- the zero-gain set->graph step stops
    the climb before the payoff two steps up). Use select_contract_mdl (global J over all admissible
    contracts) as the primary selector -- the correct min-total-description-length choice, matching depth's
    FRONTIER logic (pick the best J on the swept frontier) rather than the greedy stop -- and read this
    marginal-value form as the 'does climbing pay?' certificate.
    """
    order = sorted(scores, key=lambda c: (CONTRACT_LATTICE_ORDER.get(c, 99), omegas.get(c, 0.0)))
    winner = order[0]
    steps = []
    for prev, nxt in zip(order[:-1], order[1:]):
        dR = (1.0 - scores[winner]) - (1.0 - scores[nxt])  # risk reduction from climbing to nxt
        dOmega = max(omegas[nxt] - omegas[winner], 1e-9)  # added structural code length
        marginal = dR / dOmega
        climb = marginal > mu_c
        steps.append(
            {
                "from": winner,
                "to": nxt,
                "dR": float(dR),
                "dOmega": float(dOmega),
                "marginal": float(marginal),
                "climb": bool(climb),
            }
        )
        if climb:
            winner = nxt
    detail = {
        "order": order,
        "steps": steps,
        "mu_c": float(mu_c),
        "note": "contract via marginal-value rule: climb the lattice while dR/dOmega > mu_c",
    }
    return winner, detail


def price_tensorization(X, mu=0.05, method="gaussian", axis_bits=None, max_rank=4):
    """Fold the TENSORIZATION choice (which lattice shape to impose on a flat vector) into the same
    J = R + mu*Omega objective as every other rung, for ANY rank 1..max_rank. This replaces the
    representation front-end's hand-set floor+margin acceptance thresholds with a derived marginal-value
    rule: impose an axis only if its stride-anomaly fit gain exceeds mu times the added-axis structural
    code length.

    THE SPLIT (see tests/tensorization_pricing.md). Across ranks (1-D vs 2-D vs 3-D ...) the choice is
    priced by Omega_struct(rank); within a rank the exact factorization is a fit decision, made by the
    multi-peak stride signature (a rank-r row-major lattice produces anomaly peaks at its r-1 partial-
    product strides). R(shape) is the STRIDE-ANOMALY mass explained, NOT the coarse MI-vs-distance
    correlation (which is fooled by a 1-D sequence reshaped to H x W). Delegates to
    mode_structure.parse_grid_shape, which enumerates factorizations and minimises J = R + mu*Omega_struct.

    Returns {structure ('1d'|'2d'|'3d'|'4d'), shape, rank, detail (J per candidate, anomalies), mu}.

    HONEST SCOPE. Reliable for well-conditioned lattices with axes >= 4 (volumes 4x4x4, 2x4x8; all clean
    2-D grids incl. 28x28, 32x32) and for genuine 1-D chains. Factorizations with a length-2 axis are a
    degeneracy of the flat-distance profile and may be mis-ranked; refused rather than guessed.
    """
    import numpy as _np

    from ..core.mode_structure import mutual_information_matrix, parse_grid_shape, parse_grid_shape_scalable

    X = _np.asarray(X, float)
    D = X.shape[1]
    # For large D the dense D x D MI matrix is intractable (~4 GB at D=22000), so use the scalable
    # on-demand path, which agrees with the dense parser on every size where both run. Threshold ~500
    # keeps small problems on the exact dense path.
    if D > 500:
        shape, detail = parse_grid_shape_scalable(X, max_rank=max_rank)
    else:
        M = mutual_information_matrix(X, method=method)
        shape, detail = parse_grid_shape(M, max_rank=max_rank)
    rank = len(shape)
    struct = {1: "1d", 2: "2d", 3: "3d", 4: "4d"}[rank]
    return {"structure": struct, "shape": shape, "rank": rank, "detail": detail, "mu": float(mu)}

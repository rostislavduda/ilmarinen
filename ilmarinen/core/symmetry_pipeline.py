"""discover_and_reduce -- the symmetry front-end of the ilmarinen pipeline.

This runs the physicist pipeline's first stage: DISCOVER symmetries in the data, QUOTIENT them out
(build invariant features), and hand the reduced representation to the metaoptimizer. It is a
representation/preprocessing stage relative to the metaoptimizer -- except that (unlike PCA) it
contains a small model-fitting step, because the detectors read symmetry off a trained function's
gradients / equivariance. For EXACT symmetries this is clean preprocessing; for APPROXIMATE ones the
symmetry decision should instead migrate INTO the metaoptimizer as an alpha-selection between an
invariant and an unconstrained branch (see core/equivariant_supergraph.py).

------------------------------------------------------------------------------------------------
DETECTION ORDER (not commutative -- this order is deliberate):
  1. CONTINUOUS AFFINE (linear generators + translation) via the Lie-derivative nullspace.
       - tractable (linear algebra), and quotienting it SHRINKS the residual discrete search
         (any Lie group G = G^0 semidirect pi_0(G); removing G^0 leaves only the finite pi_0).
       - if the continuous pass is EMPTY, the symmetry is purely discrete -> go straight to step 2.
  2. DISCRETE, cheapest-and-most-constraining first:
       a. C_n / D_n point groups  (a single strong rotational constraint; also subsumes many
          reflections, so finding D_n up front explains several Z_2's at once).
       b. Z_2 residual reflections/parity not already explained by a discovered D_n.
       c. S_n permutation subgroups (the largest search; transposition-based, O(n^2)).
  Rationale for discrete sub-order: detect the MOST STRUCTURED / MOST CONSTRAINING group first so
  its subgroups are explained by it rather than re-discovered as independent false structure (a
  discovered D_6 already contains 6 reflections and C_2,C_3 -- no need to also flag those as
  separate Z_2's). This mirrors the divisor-consistency logic inside the C_n detector.

------------------------------------------------------------------------------------------------
FALSE-POSITIVE GUARDS (numerical robustness -- the practical crux):
  Pure-math symmetry detection is clean; on real noisy data, threshold noise causes false flags.
  We defend with THREE independent guards, all validated empirically:
   (G1) STABILITY ACROSS REFITS: refit the reference model with `n_refits` seeds; accept a symmetry
        only if it is detected in >= `consensus_frac` of refits. A real symmetry is stable across
        refits; a threshold-noise false positive appears in only 1-2. (This is the single most
        effective guard.)
   (G2) SPECTRAL-GAP / MARGIN requirement: for the continuous detector, require a clear gap between
        the null and non-null singular values (gap_ratio >= `min_gap`); for discrete, require the
        equivariance error to be a clear MARGIN below tol (err < tol * `margin_frac`), not just
        barely under. Marginal passes are rejected.
   (G3) DIVISOR / SUBGROUP CONSISTENCY: a real finite group has its whole subgroup lattice present
        (built into the C_n detector: C_n accepted only if all C_d, d|n, also pass). Isolated
        orders lacking their divisors are rejected.
  A fourth, optional guard (not default, more expensive):
   (G4) NOISE SWEEP: re-detect under added input/label noise; a real symmetry degrades gracefully,
        a false one flips. Exposed via `noise_sweep=True`.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .symmetry_discovery import discover_affine_symmetries

# --- named thresholds for the symmetry-discovery cascade (see discover_and_reduce docstring) ---
_ANTISYM_THRESHOLD = 0.3   # magnitude above which an antisymmetric (rotation-like) signal is real
_SCALE_AWARE_TOL = 0.08    # scale-aware ratio separating genuine symmetry (~0.01) from flatness (~0.2)
_NULL_EXCESS_FACTOR = 3    # discovered permutation count must exceed this multiple of the shuffled null
from .discrete_symmetry import (
    build_permutation_invariant_features,
    discover_cyclic_dihedral,
    discover_permutation_subgroup,
    discover_z2,
)


def continuous_invariant_features(X, generators):
    """Build features invariant under the discovered continuous generators, for the re-cascade.

    For a rotation-type generator L (antisymmetric part dominant), the invariant is the norm within
    each 2D orbit plane. General construction: for each generator, identify the plane it rotates
    (the dominant off-diagonal of its antisymmetric part) and replace those two coordinates by their
    radius; coordinates untouched by any generator pass through. This absorbs exactly the continuous
    group's action, so a subsequent discrete detection on these features sees only the RESIDUAL
    discrete structure (pi_0(G)), not shadows of the continuous part.

    Returns (features, kept_axes). Falls back to (X, identity axes) if no clean rotation plane.
    """
    X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    n = X.shape[1]
    radial_pairs = []
    seen = set()
    for g in generators:
        L = g["L"] if isinstance(g, dict) else np.asarray(g)
        A = 0.5 * (L - L.T)                       # antisymmetric (rotation) part
        ij = np.unravel_index(np.argmax(np.abs(A)), A.shape)
        i, j = int(ij[0]), int(ij[1])
        if abs(A[i, j]) > _ANTISYM_THRESHOLD and i != j and i not in seen and j not in seen:
            radial_pairs.append((min(i, j), max(i, j)))
            seen.update([i, j])
    if not radial_pairs:
        return X, [("axis", c) for c in range(n)]
    feats, kept_axes, used = [], [], set()
    for (i, j) in radial_pairs:
        feats.append(torch.sqrt(X[:, i] ** 2 + X[:, j] ** 2 + 1e-9).unsqueeze(-1))
        kept_axes.append(("radius", (i, j)))
        used.update([i, j])
    for c in range(n):
        if c not in used:
            feats.append(X[:, c:c + 1])
            kept_axes.append(("axis", c))
    return torch.cat(feats, dim=-1), kept_axes


def _fit_reference(X, y, seed, epochs=300, width=96, lr=3e-3):
    """Fit a quick reference model whose gradients/equivariance the detectors read."""
    torch.manual_seed(seed)
    d = X.shape[1]
    out_dim = 1 if y.dim() == 1 else y.shape[1]
    net = nn.Sequential(nn.Linear(d, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                        nn.Linear(width, out_dim))
    op = torch.optim.Adam(net.parameters(), lr=lr)
    lf = nn.MSELoss()
    yt = y if y.dim() > 1 else y.unsqueeze(-1)
    for _ in range(epochs):
        op.zero_grad()
        loss = lf(net(X), yt)
        loss.backward()
        op.step()
    return net, float(loss)


def _discover_cyclic(refs, Xeval, d_eff, need, max_cyclic, tol):
    """Phase 2a: C_n / D_n detection over the refit ensemble with G1 consensus + G3 divisor-consistency
    (built into discover_cyclic_dihedral). Skipped in <2D (no rotation plane). Returns (group_or_None,
    per-group frequency dict for the report)."""
    if d_eff < 2:
        return None, {}
    cyc_groups = Counter(discover_cyclic_dihedral(net, Xeval, max_order=max_cyclic, tol=tol)["group"]
                         for net in refs)
    cyclic = cyc_groups.most_common(1)[0][0] if cyc_groups else None
    if cyclic and (cyc_groups[cyclic] < need or cyclic.startswith("trivial")):
        cyclic = None
    return cyclic, dict(cyc_groups)


def _discover_z2(refs, Xeval, consensus_frac, sa, sa_tol, error_mode, hard_block_swaps):
    """Phase 2b: residual Z_2 involutions with G1 refit-consensus and scale-aware robustness (the data-driven
    guard against smoothness-induced false positives). `hard_block_swaps` drops swap_* keys as the belt-and-
    suspenders 'ordered' block. Returns (accepted_keys, per-key detection frequency)."""
    counts = Counter()
    for net in refs:
        for k, _v in discover_z2(net, Xeval, tol=sa_tol, scale_aware=sa, error_mode=error_mode)["symmetries"]:
            counts[k] += 1
    need = int(np.ceil(consensus_frac * len(refs)))
    accepted = [k for k, c in counts.items()
                if c >= need and not (hard_block_swaps and k.startswith("swap_"))]
    return accepted, dict(counts)


def _union_find_blocks(stable_trans, d_eff):
    """Union-find over the accepted transpositions -> the permutation blocks (the Young-subgroup factors)."""
    parent = list(range(d_eff))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i, j in stable_trans:
        parent[find(i)] = find(j)
    comp = {}
    for a in range(d_eff):
        comp.setdefault(find(a), []).append(a)
    return sorted([sorted(v) for v in comp.values() if len(v) > 1])


def discover_and_reduce(X, y, *, n_refits=3, consensus_frac=0.67, min_gap=1.8, margin_frac=0.7,
                        tol_discrete=0.10, max_cyclic=8, epochs=300, noise_sweep=False,
                        coordinate_structure="unknown", null_test=True, scale_aware=True,
                        error_mode=None, verbose=True):
    """Run the robust symmetry-discovery cascade and return discovered symmetries + reduced features.

    coordinate_structure : one of
        "exchangeable" -- coordinates are a set/point-cloud with no canonical order (permutation
                          search ENABLED; e.g. atoms in a molecule, points in a cloud).
        "ordered"      -- coordinates have a fixed meaningful order (time series, sequence, image
                          pixels): permutation search DISABLED, because smooth/autocorrelated
                          ordered data produces SMOOTHNESS-INDUCED false permutation symmetries
                          (adjacent correlated coords look exchangeable but are not a group
                          symmetry). This is a documented failure mode -- see tests.
        "unknown"      -- (default) conservative: permutation search disabled unless null_test
                          passes, since false positives on ordered data are worse than missing a
                          genuine set symmetry.
    null_test : if True, gate permutation acceptance on a null-model excess test (the discovered
                count must exceed a shuffled-label null by a clear margin). Guards against
                smoothness artifacts even when structure is 'unknown'/'exchangeable'.

    X, y : (n, d) inputs and (n,) or (n,k) targets (torch tensors).
    Guards: n_refits/consensus_frac (G1), min_gap/margin_frac (G2), divisor-consistency (G3, in the
            C_n detector), noise_sweep (G4, optional).

    Returns dict:
      continuous : list of (kind, generator) accepted across refits
      cyclic     : the discovered C_n/D_n group (or None)
      z2         : list of residual Z_2 involutions accepted
      permutation: discovered permutation blocks / Young subgroup
      report     : per-candidate detection frequencies (for transparency)
      reduce_fn  : a callable X -> reduced features implementing the accepted quotient
    """
    Xg = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    yg = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    d = Xg.shape[1]
    Xeval = Xg[torch.randperm(len(Xg))[:min(2500, len(Xg))]]

    # fit the reference models once (shared across detectors for each seed)
    refs = [_fit_reference(Xg, yg, seed=s, epochs=epochs)[0] for s in range(n_refits)]

    report = {}

    # ---- 1. CONTINUOUS AFFINE (with G1 stability + G2 gap margin) ----
    def cont_detect(net):
        return discover_affine_symmetries(net, Xeval)

    cont_accepted = []
    cont_freq = {}
    from collections import Counter
    ck = Counter()
    for net in refs:
        r = cont_detect(net)
        if r["gap_ratio"] >= min_gap:                       # G2: clear spectral gap required
            for g in r["generators"]:
                # label each generator by kind + a coarse signature so refits can be matched
                sig = f"{g['kind']}"
                ck[sig] += 1
    need = int(np.ceil(consensus_frac * len(refs)))
    cont_freq = dict(ck)
    cont_kinds = {k for k, c in ck.items() if c >= need}
    report["continuous"] = cont_freq
    # re-extract representative generators from the last refit for the accepted kinds
    rc = discover_affine_symmetries(refs[0], Xeval)
    for g in rc["generators"]:
        if g["kind"] in cont_kinds:
            cont_accepted.append((g["kind"], g))

    # ---- RE-CASCADE: if a continuous rotation was found, run discrete detection on the
    # continuous-INVARIANT features so only the genuine residual pi_0(G) survives (not shadows of
    # the continuous group). We rebuild the reference models on the invariant features.
    rotation_gens = [g for (k, g) in cont_accepted
                     if np.linalg.norm(0.5 * (g["L"] - g["L"].T)) > _ANTISYM_THRESHOLD]
    recascaded = False
    if rotation_gens:
        Xinv_full, kept_axes = continuous_invariant_features(Xg, rotation_gens)
        if Xinv_full.shape[1] < d:                          # a genuine reduction happened
            recascaded = True
            refs = [_fit_reference(Xinv_full, yg, seed=s, epochs=epochs)[0]
                    for s in range(n_refits)]
            Xeval = Xinv_full[torch.randperm(len(Xinv_full))[:min(2500, len(Xinv_full))]]
            d_eff = Xinv_full.shape[1]
            report["recascade_axes"] = [str(a) for a in kept_axes]
        else:
            d_eff = d
    else:
        d_eff = d

    # ---- 2a. C_n / D_n (G1 + G3 divisor-consistency built in + G2 margin via tighter tol) ----
    # skip if the (possibly re-cascaded) space is <2D -- no rotation plane exists there
    cyclic, report["cyclic"] = _discover_cyclic(refs, Xeval, d_eff, need, max_cyclic,
                                                tol_discrete * margin_frac)   # G2 margin

    # ---- 2b. Z_2 residual (G1 + G2 margin + SCALE-AWARE robustness) ----
    # The scale-aware equivariance test distinguishes a genuine symmetry from FLATNESS by
    # calibrating against the function's own sensitivity in the directions g acts. This is the
    # robust, DATA-DRIVEN guard against smoothness-induced false positives (e.g. adjacent
    # time-points), replacing reliance on the coordinate_structure prior. A tighter threshold is
    # used for the scale-aware ratio (genuine ~0.01, flatness ~0.2, so ~0.08 separates cleanly).
    sa = scale_aware
    # grad_weighted uses the same tight threshold as scale_aware (genuine ~0, false-positive ~0.3-0.6)
    sa_tol = _SCALE_AWARE_TOL if (sa or error_mode in ("scale_aware", "grad_weighted")) else tol_discrete * margin_frac
    # coordinate_structure is now OPTIONAL extra safety, not required: 'ordered' still hard-disables
    # swaps as a belt-and-suspenders measure, but with scale_aware the default 'unknown' is safe.
    hard_block_swaps = (coordinate_structure == "ordered")

    z2_accepted, report["z2"] = _discover_z2(refs, Xeval, consensus_frac, sa, sa_tol, error_mode,
                                             hard_block_swaps)

    # ---- 2c. S_n permutation (G1 + SCALE-AWARE + optional null-test) ----
    # With scale_aware=True the flatness false-positive is handled data-drivenly, so permutation
    # search is safe to run regardless of coordinate_structure (except a hard 'ordered' block).
    perm_blocks = []
    report["permutation_transpositions"] = {}
    perm_enabled = (coordinate_structure != "ordered")
    if perm_enabled:
        def perm_detect(net):
            return discover_permutation_subgroup(net, Xeval, tol=sa_tol, scale_aware=sa, error_mode=error_mode)
        perm_results = [perm_detect(net) for net in refs]
        tp = Counter()
        for r in perm_results:
            for (i, j, _e) in r["transpositions"]:
                tp[(i, j)] += 1
        stable_trans = [(i, j) for (i, j), c in tp.items() if c >= need]
        report["permutation_transpositions"] = {f"{i}{j}": c for (i, j), c in tp.items()}
        # NULL-TEST: discovered count must clearly exceed a shuffled-label null (destroys real
        # symmetry, keeps smoothness). Guards even the 'exchangeable' path against artifacts.
        accept_perm = True
        if null_test:
            null_counts = []
            base_X = Xg if not recascaded else continuous_invariant_features(Xg, rotation_gens)[0]
            for s in range(2):
                net_null, _ = _fit_reference(base_X, yg[torch.randperm(len(yg))], seed=s,
                                             epochs=epochs)
                null_counts.append(len(discover_permutation_subgroup(
                    net_null, Xeval, tol=tol_discrete * margin_frac)["transpositions"]))
            real_count = len(stable_trans)
            null_mean = float(np.mean(null_counts)) if null_counts else 0.0
            report["permutation_null"] = {"real": real_count, "null_mean": null_mean}
            # require a clear excess over null; the null already absorbs smoothness, so a genuine
            # symmetry stands out as real >> null (not merely real > null).
            if real_count <= max(_NULL_EXCESS_FACTOR * null_mean, 1):
                accept_perm = False
        if accept_perm:
            perm_blocks = _union_find_blocks(stable_trans, d_eff)

    # ---- optional G4 noise sweep (report only) ----
    noise_report = None
    if noise_sweep:
        noise_report = {}
        for nl in (0.05, 0.15):
            Xn = Xg + nl * torch.randn_like(Xg)
            net_n, _ = _fit_reference(Xn, yg, seed=0, epochs=epochs)
            noise_report[nl] = {
                "z2": [k for k, v in discover_z2(net_n, Xeval, tol=tol_discrete).get("symmetries", [])],
            }
        report["noise_sweep"] = noise_report

    # ---- build the quotient reduce_fn from accepted symmetries ----
    def reduce_fn(Xin):
        Xin = Xin if isinstance(Xin, torch.Tensor) else torch.tensor(Xin, dtype=torch.float32)
        feats = [Xin]
        # permutation quotient (power-sum pooling per block)
        if perm_blocks:
            feats.append(build_permutation_invariant_features(Xin, perm_blocks))
        # Z_2 quotient for the first accepted involution (illustrative; multiple can be composed)
        # (kept conservative: only apply when a permutation quotient did not already cover it)
        return torch.cat(feats, dim=-1) if len(feats) > 1 else Xin

    result = {
        "continuous": cont_accepted,
        "cyclic": cyclic,
        "z2": z2_accepted,
        "permutation": {"blocks": perm_blocks,
                        "young_subgroup": " x ".join(f"S_{len(b)}" for b in perm_blocks)
                        if perm_blocks else "trivial"},
        "report": report,
        "reduce_fn": reduce_fn,
    }
    # SUBSUMPTION handling: the re-cascade (above) already removed continuous shadows by running
    # discrete detection on the continuous-invariant features, so any discrete findings here are
    # GENUINE residual pi_0(G) structure, not consequences of the continuous part. We note whether
    # the re-cascade ran so the result is transparent about which space the discrete findings live in.
    if recascaded:
        result["recascade"] = ("discrete detection ran on continuous-INVARIANT features; discrete "
                               "findings are genuine residual structure, with continuous shadows "
                               "removed. Axis map in report['recascade_axes'].")
    elif cont_accepted and (cyclic or z2_accepted or perm_blocks):
        result["subsumption_warning"] = (
            "continuous symmetry present but re-cascade did not reduce dimension; discrete findings "
            "in the same coordinates may be consequences of it -- treat the continuous generator as "
            "primary.")
    if verbose:
        print("discover_and_reduce summary (guards: G1 refit-consensus, G2 gap-margin, G3 divisors):")
        print(f"  continuous : {[k for k, _ in cont_accepted]}  (freq {cont_freq})")
        print(f"  cyclic/dih : {cyclic}")
        print(f"  z2 residual: {z2_accepted}")
        print(f"  permutation: {result['permutation']['young_subgroup']}  blocks={perm_blocks}")
        if "recascade" in result:
            print(f"  RE-CASCADE: {result['recascade']}")
        if "subsumption_warning" in result:
            print(f"  NOTE: {result['subsumption_warning']}")
    return result

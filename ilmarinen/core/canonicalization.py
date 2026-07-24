"""canonicalization.py -- Phase 1 of symmetry-driven contract generation: exploit a discovered symmetry
by CANONICALIZING inputs to the group's canonical frame, then reusing an EXISTING contract.

When the symmetry front-end discovers that a target is invariant under a group G acting on the coordinates
(e.g. SO(3) rotations of a point cloud), we have two ways to exploit it. Phase 2 will BUILD a G-equivariant
layer. Phase 1 -- here -- takes the cheaper canonicalization route (LieLAC / frame-averaging idea): map
every input to a canonical representative of its G-orbit, so that G-equivalent inputs become identical.
A plain, non-equivariant downstream net (an EXISTING schema, e.g. the set contract) is then EFFECTIVELY
G-invariant without being built equivariant -- the symmetry is "used up" by the alignment. This
demonstrates the full loop "discover a symmetry -> exploit it by reusing an existing contract" with almost
no new machinery, and is the low-risk precursor to generating a genuinely new equivariant contract.

SO(3) CANONICALIZATION (the case we discover reliably). Align each cloud's principal axes (inertia-tensor
eigenvectors, sorted by eigenvalue) to the coordinate axes, with a deterministic sign convention. This
maps every rotation of a cloud to the SAME canonical pose, exactly (to machine precision) when the inertia
spectrum is non-degenerate.

VALIDATED (tests/canonicalization.md):
  * invariance: canon(R x) = canon(x) to ~1e-15 for anisotropic clouds (distinct inertia eigenvalues);
  * it HELPS a rotation-invariant target (plain DeepSets R^2 0.93 -> 0.98, invariance for free) and
    correctly HURTS a rotation-breaking one (0.93 -> -0.42, it destroys the orientation signal) -- so it
    must be applied ONLY when the target is rotation-invariant (which symmetry_contract decides);
  * a degeneracy guard flags clouds with a near-degenerate inertia spectrum (spheres, symmetric tops,
    tetrahedra/cubes), where the frame is ambiguous, so the caller can fall back (skip canonicalization).
"""

from __future__ import annotations

import numpy as np


def canonicalize_cloud(P, degen_tol=0.12):
    """Rotate a single point cloud into its canonical (principal-axis) frame.

    Returns (P_canon, degenerate). `degenerate` is True when the inertia spectrum has near-equal
    eigenvalues (the canonical frame is ambiguous); the caller should then skip canonicalization.
    """
    P = np.asarray(P, float)
    P = P - P.mean(0)
    if len(P) < 2:
        return P, True
    cov = P.T @ P / len(P)
    w, V = np.linalg.eigh(cov)  # ascending
    order = np.argsort(w)[::-1]  # largest inertia eigenvalue first
    w = w[order]
    V = V[:, order]
    # degeneracy guard: any adjacent pair of eigenvalues within degen_tol (relative) -> ambiguous frame
    degenerate = False
    for i in range(len(w) - 1):
        denom = abs(w[i]) + 1e-12
        if (w[i] - w[i + 1]) / denom < degen_tol:
            degenerate = True
            break
    Pc = P @ V  # rotate into the principal frame
    # deterministic sign convention: along each axis, make the largest-magnitude coordinate positive.
    for k in range(Pc.shape[1]):
        j = int(np.argmax(np.abs(Pc[:, k])))
        if Pc[j, k] < 0:
            Pc[:, k] *= -1
    return Pc, degenerate


def canonicalize_positions(positions, degen_tol=0.12):
    """Canonicalize a list of clouds. Returns (list_of_canonical_clouds, degenerate_fraction). Clouds with
    an ambiguous (degenerate) frame are left centered-but-unrotated and counted in the fraction, so the
    caller can decide to fall back if too many are ambiguous."""
    out = []
    ndeg = 0
    for P in positions:
        Pc, deg = canonicalize_cloud(P, degen_tol=degen_tol)
        if deg:
            ndeg += 1
            P0 = np.asarray(P, float)
            out.append(P0 - P0.mean(0))  # centered but not rotated (frame ambiguous)
        else:
            out.append(Pc)
    frac = ndeg / max(len(positions), 1)
    return out, frac


def canonicalize_data(data, max_degenerate_frac=0.5, degen_tol=0.12):
    """Return a shallow copy of a AllData-like object with its positions canonicalized in place, plus a
    detail dict. If too large a fraction of clouds are degenerate (frame ambiguous), canonicalization is
    SKIPPED (returns the data unchanged with applied=False) -- the fail-safe: never impose an ambiguous
    frame. This is the Phase-1 preprocessing that lets a plain contract exploit a discovered rotational
    symmetry.

    Intended use: call ONLY when symmetry_contract has judged the target rotation-invariant. Applying it
    to an orientation-dependent target would (correctly) destroy signal, so the caller gates on that.
    """
    import copy

    if getattr(data, "positions", None) is None:
        return data, {"applied": False, "reason": "no positions"}
    canon, frac = canonicalize_positions(data.positions, degen_tol=degen_tol)
    if frac > max_degenerate_frac:
        return data, {"applied": False, "reason": f"too many degenerate frames ({frac:.0%})", "degenerate_frac": frac}
    new = copy.copy(data)
    new.positions = canon
    return new, {
        "applied": True,
        "degenerate_frac": frac,
        "note": "positions canonicalized to principal-axis frame (SO(3) symmetry used up)",
    }

"""D3 -- the effective-dimension ledger: one coarse-graining axis for "how many degrees of freedom
does this representation really use."

The package measures effective dimension in FOUR places that grew up as separate modules with separate
vocabularies:

  * `redundancy_reduction.effective_dimension`  -- d_eff of the DATA, the participation ratio of the
    covariance spectrum (unsupervised, static).
  * `ib_rg_flow.ib_rg_flow`                     -- d_IB(beta) of the SUPERVISED representation, a
    scale-resolved staircase with located RG transitions (B8), plus its own unsupervised d_eff_static.
  * `sparsity_priced_alpha.participation`       -- the effective number of primitives, the inverse
    participation ratio of the alpha mixture.
  * `singular_complexity` / `singular_mdl`      -- lambda, the RLCT: the functional effective dimension
    of the whole fitted MODEL (lambda <= k/2).

The premise check for this unification found (as for the thermodynamic potential of D2) that the honest
statement is more precise than "one number." Three of the four are LITERALLY THE SAME FUNCTIONAL --

    PR(spectrum)  =  (sum_i s_i)^2 / sum_i s_i^2  =  1 / sum_i p_i^2 ,   p_i = s_i / sum_j s_j

the physics participation ratio (= 1 when the spectrum is concentrated on one mode, = m when it is
uniform over m modes) -- applied to three different spectra: the data covariance eigenvalues (data
modes), the supervised Gib spectrum via the IB flow (data modes, scale-resolved), and the alpha weights
(primitive mixture). They differ only in WHICH spectrum PR eats. They are therefore ONE FUNCTIONAL read
at ascending coarse-graining LEVELS -- but NOT one commensurable scalar: a count of data modes and a
count of effective primitives are in different units and must not be summed. The fourth leg, lambda, is
a genuinely DIFFERENT functional (the RLCT, not a participation ratio); it answers the same QUESTION --
effective degrees of freedom -- at the model level, and is the natural cap of the ledger (lambda <= k/2).

So the ledger is: one participation-ratio functional read at successive kinematic levels (data modes,
then the primitive mixture), plus lambda as the model-level functional dimension. This is the
quantitative completion of the package's "kinematics -> degrees of freedom" ordering: the same
"count the effective modes" measurement, stated once and read at each level, rather than three metaphors
in three modules. It is a pure REPORTING object -- it computes nothing new that changes a selection; it
reuses the existing estimators and states them on one axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "participation_ratio",
    "effective_dimension_ledger",
    "LEDGER_LEVELS",
]


# --------------------------------------------------------------------------------------------------
# The one shared functional (stated once)
# --------------------------------------------------------------------------------------------------
def participation_ratio(spectrum: Sequence[float]) -> float:
    """The participation ratio PR(s) = (sum s_i)^2 / sum s_i^2 = 1 / sum p_i^2 of a NON-NEGATIVE
    spectrum. This is the single functional the ledger reads at every kinematic level: the covariance
    eigenvalues (effective data modes), the supervised Gib spectrum (scale-resolved data modes), and
    the alpha weights (effective primitives). Returns 1.0 when the spectrum is concentrated on one
    mode and m when it is uniform over m modes.

    Matches `redundancy_reduction.effective_dimension` (covariance) and `sparsity_priced_alpha.
    participation` (alpha) exactly -- it is the same 1/sum(p^2), factored out so the ledger states it
    once rather than reimplementing it per level.
    """
    s = np.asarray(spectrum, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    return float(1.0 / np.sum(p**2))


# --------------------------------------------------------------------------------------------------
# The ledger levels, as data (the coarse-graining axis, coarsest input -> whole model)
# --------------------------------------------------------------------------------------------------
LEDGER_LEVELS = (
    {
        "level": "data_modes",
        "functional": "participation ratio 1/sum(p^2)",
        "spectrum": "covariance eigenvalues of X",
        "unit": "effective data coordinates",
        "supervised": False,
        "source": "redundancy_reduction.effective_dimension",
    },
    {
        "level": "data_modes_flow",
        "functional": "d_IB(beta) = #{i: lambda_i < 1 - 1/beta}",
        "spectrum": "supervised Gib spectrum of (X,Y)",
        "unit": "effective data modes at scale beta",
        "supervised": True,
        "source": "ib_rg_flow.ib_rg_flow",
    },
    {
        "level": "primitive_mixture",
        "functional": "participation ratio 1/sum(alpha^2)",
        "spectrum": "alpha mixture weights on the primitive simplex",
        "unit": "effective primitives",
        "supervised": True,
        "source": "sparsity_priced_alpha.participation",
    },
    {
        "level": "model",
        "functional": "lambda (RLCT / local learning coefficient)",
        "spectrum": "loss-singular geometry at w*",
        "unit": "functional dimensions (nats-scale, <= k/2)",
        "supervised": True,
        "source": "singular_complexity.estimate_llc",
    },
)


# --------------------------------------------------------------------------------------------------
# Assemble the ledger from whatever pieces are available (no recomputation of selections)
# --------------------------------------------------------------------------------------------------
def effective_dimension_ledger(
    *,
    cov_spectrum: Sequence[float] | None = None,
    ib_flow: dict | None = None,
    alpha: Sequence[float] | None = None,
    llc: dict | None = None,
) -> dict:
    """Assemble the effective-dimension ledger from already-computed pieces. Every argument is
    optional; a level is included only when its input is supplied (a run may have data modes but no
    fitted mixture, or a model lambda but no IB flow, etc.).

    cov_spectrum : covariance eigenvalues / variance ratios of the data (from
                   `effective_dimension(X)`'s second return, or raw eigenvalues) -> data-modes level.
    ib_flow      : the dict returned by `ib_rg_flow(X, Y)` (keys d_IB, transitions, d_eff_static,
                   betas) -> the supervised scale-resolved data-modes level.
    alpha        : the primitive mixture weights (the deployed alpha) -> primitive-mixture level.
    llc          : the dict from `estimate_llc`/`singular_complexity_of` (keys lambda, n_params,
                   half_params, ratio, valid) -> model level.

    Returns {levels: [...], axis, note} where each level records its value, functional, unit, and
    provenance. The levels are ordered along the coarse-graining axis (data -> mixture -> model). The
    values are NOT summed: data modes, effective primitives, and lambda are in different units. The
    ledger states one functional (the participation ratio) read at the kinematic levels, plus lambda
    as the model-level functional dimension.
    """
    levels = []

    # --- data modes (unsupervised participation ratio) ---
    if cov_spectrum is not None and len(cov_spectrum) > 0:
        d_eff = participation_ratio(cov_spectrum)
        levels.append(
            {
                "level": "data_modes",
                "value": d_eff,
                "ambient": int(len(cov_spectrum)),
                "functional": "participation ratio 1/sum(p^2)",
                "unit": "effective data coordinates",
                "supervised": False,
                "source": "redundancy_reduction.effective_dimension",
                "note": f"{d_eff:.2f} effective of {len(cov_spectrum)} ambient covariance modes",
            }
        )

    # --- data modes, supervised scale-resolved (the IB-RG flow) ---
    if ib_flow is not None and isinstance(ib_flow, dict):
        d_ib = ib_flow.get("d_IB")
        trans = ib_flow.get("transitions", [])
        d_static = ib_flow.get("d_eff_static")
        # d_IB may be a staircase (array over betas) or a scalar; summarize both ends
        d_ib_arr = np.asarray(d_ib).ravel() if d_ib is not None else np.array([])
        entry = {
            "level": "data_modes_flow",
            "value": (float(d_ib_arr[-1]) if d_ib_arr.size else None),
            "d_eff_static": (float(d_static) if d_static is not None else None),
            "n_transitions": int(len(trans)),
            "functional": "d_IB(beta) = #{i: lambda_i < 1 - 1/beta}",
            "unit": "effective data modes at scale beta",
            "supervised": True,
            "source": "ib_rg_flow.ib_rg_flow",
            "note": (
                f"supervised staircase; d_eff_static={d_static:.2f}, {len(trans)} located RG transitions"
                if d_static is not None
                else f"supervised staircase; {len(trans)} located RG transitions"
            ),
        }
        levels.append(entry)

    # --- primitive mixture (participation ratio of alpha) ---
    if alpha is not None and len(alpha) > 0:
        a = np.asarray(alpha, dtype=np.float64)
        ipr = participation_ratio(a)  # alpha already normalized; PR is scale-free anyway
        levels.append(
            {
                "level": "primitive_mixture",
                "value": ipr,
                "n_primitives": int(len(a)),
                "functional": "participation ratio 1/sum(alpha^2)",
                "unit": "effective primitives",
                "supervised": True,
                "source": "sparsity_priced_alpha.participation",
                "note": f"{ipr:.2f} effective of {len(a)} primitives in the mixture",
            }
        )

    # --- model (lambda / RLCT) ---
    if llc is not None and isinstance(llc, dict) and "lambda" in llc:
        lam = llc.get("lambda")
        half = llc.get("half_params")
        ratio = llc.get("ratio")
        valid = llc.get("valid", True)
        levels.append(
            {
                "level": "model",
                "value": (float(lam) if lam is not None else None),
                "half_params": (float(half) if half is not None else None),
                "ratio_to_half_params": (float(ratio) if ratio is not None else None),
                "valid": bool(valid),
                "functional": "lambda (RLCT / local learning coefficient)",
                "unit": "functional dimensions (<= k/2)",
                "supervised": True,
                "source": "singular_complexity.estimate_llc",
                "note": (
                    f"lambda={lam:.2f} vs k/2={half:.0f} (ratio {ratio:.3f}); a DISTINCT functional "
                    f"(RLCT, not a participation ratio) at the model level"
                    if (lam is not None and half is not None and ratio is not None)
                    else "model-level functional dimension (RLCT)"
                )
                if valid
                else "lambda invalid (non-converged optimum); model level not certified",
            }
        )

    return {
        "levels": levels,
        "axis": "coarse-graining: data modes -> supervised data modes(beta) -> primitive mixture -> model",
        "note": "One participation-ratio functional (1/sum p^2) read at the kinematic levels (data "
        "covariance, alpha simplex) plus lambda (RLCT) at the model level. Values are in "
        "DIFFERENT units (data modes vs effective primitives vs functional dimensions) and are "
        "NOT summed -- the ledger unifies the QUESTION and (for the kinematic levels) the "
        "FUNCTIONAL, not the scalar. This is the 'kinematics -> degrees of freedom' axis stated "
        "once rather than as separate metaphors.",
    }

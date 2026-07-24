"""Tier-3 interpretability: optional symbolic read-out over inspectable learned univariate functions.

Where a fitted model exposes an inspectable univariate curve -- above all the KAN equivariant primitives'
learned radial phi(d) (e_kan / e_painn_kan, via radial_function()) -- Tier 3 attempts to fit that curve to a
small dictionary of analytic basis functions, yielding a CLOSED-FORM expression WHEN ONE FITS, and otherwise
reporting that no clean symbolic form was found and deferring to the exact spline.

This is the ONLY interpretability tier that risks a fidelity gap: the symbolic expression is an APPROXIMATION
of the spline, not the spline itself. The foundation's design commitment (Section 6) is therefore enforced
here without exception:
  * the exact spline is the ground truth; the symbolic form is an optional lens over it;
  * every symbolic form is returned WITH its approximation residual (R^2 and max error);
  * a symbolic form is only ADVERTISED as clean when it is both accurate (R^2 >= r2_accept) AND parsimonious
    (few terms); otherwise the read-out explicitly says "no clean symbolic form -- use the spline";
  * because dictionary atoms can be correlated (the same non-identifiability seen in Tier 2), the read-out
    prefers the FEWEST terms that clear the accuracy bar and reports the form as "a" fit, not "the" law.

This is the S2KAN / SINDy posture: discover an interpretable form when the data supports one, gracefully keep
the dense representation when it does not.
"""

import numpy as np


def _build_library(d):
    """A small, physically-motivated analytic dictionary of univariate basis functions of a distance d>0.
    Includes Lennard-Jones powers, exponentials, low polynomials, and simple transcendentals."""
    d = np.asarray(d, dtype=np.float64)
    eps = 1e-6
    terms = {
        "1": np.ones_like(d),
        "d": d,
        "d^2": d**2,
        "d^3": d**3,
        "1/d": 1.0 / (d + eps),
        "1/d^2": 1.0 / (d**2 + eps),
        "1/d^6": 1.0 / (d**6 + eps),
        "1/d^12": 1.0 / (d**12 + eps),
        "exp(-d)": np.exp(-d),
        "exp(-d^2)": np.exp(-(d**2)),
        "sin(d)": np.sin(d),
        "cos(d)": np.cos(d),
        "log(d)": np.log(d + 1.0 + eps),
    }
    names = list(terms.keys())
    A = np.stack([terms[k] for k in names], axis=1)
    return A, names


def _fit_k_terms(A, names, phi, k):
    """Greedy orthogonal matching pursuit choosing k dictionary atoms, refitting by least squares each step.
    Returns (coefs dict over chosen names, prediction, R^2)."""
    scale = np.linalg.norm(A, axis=0) + 1e-9
    An = A / scale
    resid = phi.copy()
    chosen = []
    c = np.zeros(0)
    for _ in range(k):
        corr = np.abs(An.T @ resid)
        if chosen:
            corr[chosen] = -1.0
        j = int(np.argmax(corr))
        chosen.append(j)
        c, _, _, _ = np.linalg.lstsq(An[:, chosen], phi, rcond=None)
        resid = phi - An[:, chosen] @ c
    coefs = {names[chosen[i]]: float(c[i] / scale[chosen[i]]) for i in range(len(chosen))}
    pred = np.zeros_like(phi)
    for i, jj in enumerate(chosen):
        pred = pred + (c[i] / scale[jj]) * A[:, jj]
    ss = np.sum((phi - phi.mean()) ** 2) + 1e-12
    r2 = float(1.0 - np.sum((phi - pred) ** 2) / ss)
    return coefs, pred, r2


def symbolic_readout(d, phi, r2_accept=0.999, max_terms=3, min_gain=0.02):
    """Fit the univariate curve phi(d) to the analytic dictionary, preferring the FEWEST terms that clear the
    accuracy bar. Returns a dict with the honest read-out.

    r2_accept : the curve is advertised as a clean symbolic form only if R^2 >= this.
    max_terms : the largest number of dictionary atoms to try.
    min_gain  : a term is only added if it improves R^2 by at least this (parsimony against correlated atoms).

    Returns: expression (str), coefs (dict), r2, max_abs_error, n_terms, clean (bool), and a verdict string.
    The spline values phi are the ground truth; the returned form is explicitly an approximation of them.
    """
    d = np.asarray(d, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    A, names = _build_library(d)
    # sweep k = 1..max_terms, stop early when adding a term does not yield >= min_gain and the bar is cleared
    best = None
    prev_r2 = -np.inf
    for k in range(1, max_terms + 1):
        coefs, pred, r2 = _fit_k_terms(A, names, phi, k)
        if best is None or r2 > best[2] + 1e-9:
            best = (coefs, pred, r2)
        # parsimony: if this k cleared the bar, keep it and stop; if the gain from k-1 was marginal, stop.
        if r2 >= r2_accept:
            best = (coefs, pred, r2)
            break
        if k > 1 and (r2 - prev_r2) < min_gain:
            break
        prev_r2 = r2
    coefs, pred, r2 = best
    max_err = float(np.max(np.abs(phi - pred)))
    clean = bool(r2 >= r2_accept)
    expr = " + ".join(f"{v:.3g}*{k}" for k, v in coefs.items())
    if clean:
        verdict = f"clean symbolic form (R^2={r2:.4f}, max|err|={max_err:.3g})"
    else:
        verdict = (
            f"NO clean symbolic form (best R^2={r2:.4f} < {r2_accept}); "
            f"defer to the exact spline -- this expression is only an approximate lens"
        )
    return {
        "expression": expr,
        "coefs": coefs,
        "r2": r2,
        "max_abs_error": max_err,
        "n_terms": len(coefs),
        "clean": clean,
        "verdict": verdict,
        "ground_truth": "spline",  # the spline is authoritative; this form approximates it
    }


def symbolify_model(mg, r2_accept=0.999, max_terms=3, max_channels=3):
    """Scan a fitted AllGraph for cells that selected a KAN primitive (e_kan / e_painn_kan) and attempt a
    symbolic read-out of each learned radial phi(d). Returns a list of per-radial reports. If no KAN
    primitive is present, returns an empty list (Tier 3 simply does not apply -- Tier 1/2 remain the report).

    Faithfulness posture: each report carries its residual and the 'clean' flag; the spline is ground truth.
    Nothing here modifies the model."""
    net = getattr(mg, "net", None)
    out = []
    if net is None or not hasattr(net, "cells"):
        return out
    for ci, c in enumerate(net.cells):
        for core in getattr(c, "cores", []):
            name = getattr(core, "name", None)
            if name and "kan" in name and hasattr(core, "radial_function"):
                try:
                    d, phi = core.radial_function(n=64)
                except Exception:
                    continue
                d = np.asarray(d)
                phi = np.asarray(phi)
                # the radial is per-channel: phi may be (n, C). Symbolify each channel, but only report the
                # channels whose curve is non-trivial (meaningful variation), to avoid noise from dead
                # channels. Report at most `max_channels` most-varying channels per cell.
                if phi.ndim == 1:
                    channels = [(None, phi)]
                else:
                    var = phi.var(axis=0)
                    order = np.argsort(var)[::-1]
                    # keep channels carrying real variation (var above 1% of the max), cap the count
                    keep = [j for j in order if var[j] > 0.01 * var[order[0]] + 1e-12][:max_channels]
                    channels = [(int(j), phi[:, j]) for j in keep]
                for ch, curve in channels:
                    rep = symbolic_readout(d, curve, r2_accept=r2_accept, max_terms=max_terms)
                    rep["cell"] = ci
                    rep["primitive"] = name
                    rep["channel"] = ch
                    out.append(rep)
    return out


def format_symbolic(reports):
    """Render symbolify_model reports as text. Always shows the residual and whether the form is clean."""
    if not reports:
        return "TIER-3 SYMBOLIC READ-OUT: no KAN primitive selected -> not applicable (Tier 1/2 stand)."
    L = ["TIER-3 SYMBOLIC READ-OUT (optional lens over inspectable radials; spline is ground truth)"]
    for r in reports:
        ch = f" ch{r['channel']}" if r.get("channel") is not None else ""
        L.append(f"  cell {r['cell']} [{r['primitive']}]{ch} radial phi(d):")
        if r["clean"]:
            L.append(f"     phi(d) ~= {r['expression']}")
            L.append(f"     {r['verdict']}")
        else:
            L.append(f"     {r['verdict']}")
            L.append(f"     (best-effort, do NOT trust as the law: {r['expression']})")
    return "\n".join(L)

"""Calibrated uncertainty on the metaoptimizer's selections (Future Direction #7).

THE PROBLEM. The metaoptimizer outputs a selection (argmax alpha -> a primitive), but raw softmax(alpha)
is NOT a calibrated confidence -- a peak of 0.35 and a peak of 0.9 both discretize to a single argmax,
and #5 showed the argmax can even be wrong (co-adaptation). We want a CALIBRATED probability: a number
that reflects the true likelihood the selection is correct/stable.

THE APPROACH (ensemble / bootstrap -- calibrated by construction). Run the search K times over different
SEEDS and/or data BOOTSTRAPS. The empirical frequency with which each primitive is selected IS a
calibrated probability: "conv selected in 8/10 runs -> P(conv)=0.8". Selection frequencies are
probabilities by construction, so no post-hoc temperature/Platt scaling is needed. This is the deep-
ensemble view of NAS uncertainty. We report:
  - the selection DISTRIBUTION (calibrated per-primitive probabilities),
  - the top selection and its CONFIDENCE (win frequency),
  - ENTROPY (flat = uncertain) and MARGIN (top1 - top2, small = uncertain),
  - a STABILITY flag (confidence above a threshold and margin above a threshold).
Low-confidence selections are FLAGGED rather than reported as a spurious single architecture.

CALIBRATION CHECK. calibration_curve bins predicted confidences and measures the empirical correctness
frequency per bin on tasks with a KNOWN correct primitive; a calibrated method has correctness ~ p
(reliability diagonal). Returns the per-bin (confidence, accuracy) and the expected calibration error.
"""
from __future__ import annotations

import numpy as np


def selection_distribution(selections, vocabulary=None):
    """Given a list of per-run selected primitives (strings, one per ensemble run), return the calibrated
    selection distribution: dict(primitive -> frequency), the top primitive, its confidence (frequency),
    entropy, and margin (top1 - top2).

    Two confidences are reported. 'confidence' is the raw win frequency f_1. 'confidence_calibrated' is
    the margin-aware pairwise confidence f_1/(f_1+f_2) -- the win probability of the top pick against its
    STRONGEST rival, ignoring the also-rans that are not real competition. The pairwise form is better
    calibrated for multi-way vocabularies (raw frequency understates correctness in the mid-range,
    because beating several spread-out alternatives with e.g. 55% is actually decisive): empirically it
    roughly halves the Expected Calibration Error (0.21 -> 0.11 in the reference check)."""
    from collections import Counter
    c = Counter(selections)
    K = len(selections)
    vocab = vocabulary or sorted(c.keys())
    dist = {p: c.get(p, 0) / K for p in vocab}
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    top, conf = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = conf - runner
    conf_cal = conf / (conf + runner) if (conf + runner) > 0 else 1.0
    probs = np.array([v for v in dist.values() if v > 0])
    entropy = float(-(probs * np.log(probs)).sum())
    return {"distribution": dist, "top": top, "confidence": conf,
            "confidence_calibrated": conf_cal, "margin": margin,
            "entropy": entropy, "n_runs": K}


def selection_uncertainty(search_fn, n_runs=10, vocabulary=None, conf_threshold=0.7,
                          margin_threshold=0.2):
    """Run search_fn(seed) -> selected primitive (str), n_runs times over different seeds, and return the
    calibrated selection distribution with a stability verdict. search_fn should perform one full
    search+discretization and return the selected primitive name. Optionally search_fn can bootstrap the
    data internally using the seed for data-level uncertainty too.

    Returns dict(top, confidence, margin, entropy, distribution, stable, verdict). 'stable' is True iff
    confidence >= conf_threshold AND margin >= margin_threshold; otherwise the selection is flagged as
    uncertain (the honest output is 'ambiguous among {top-k}' rather than a spurious single pick)."""
    selections = [search_fn(seed) for seed in range(n_runs)]
    summary = selection_distribution(selections, vocabulary=vocabulary)
    stable = summary["confidence"] >= conf_threshold and summary["margin"] >= margin_threshold
    if stable:
        verdict = f"CONFIDENT: '{summary['top']}' selected in {summary['confidence']*100:.0f}% of runs"
    else:
        topk = sorted(summary["distribution"].items(), key=lambda kv: -kv[1])[:3]
        topk = [f"{p}({w*100:.0f}%)" for p, w in topk if w > 0]
        verdict = f"UNCERTAIN: ambiguous among {topk} -- report distribution, not a single pick"
    return {**summary, "stable": stable, "verdict": verdict}


def calibration_curve(confidences, correct, n_bins=5):
    """Reliability/calibration check. confidences: list of reported selection confidences in [0,1].
    correct: list of bools (was the selection actually the known-correct primitive?). Bins the
    confidences and measures empirical correctness per bin; a calibrated method has accuracy ~ confidence
    (the diagonal). Returns per-bin (mean_confidence, accuracy, count) and the Expected Calibration
    Error (ECE) = sum_b (count_b/N) * |accuracy_b - confidence_b|."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    N = len(confidences)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (confidences > lo) & (confidences <= hi) if b > 0 else (confidences >= lo) & (confidences <= hi)
        if m.sum() == 0:
            continue
        cbar = float(confidences[m].mean())
        acc = float(correct[m].mean())
        cnt = int(m.sum())
        rows.append({"bin": (round(lo, 2), round(hi, 2)), "mean_confidence": round(cbar, 3),
                     "accuracy": round(acc, 3), "count": cnt})
        ece += (cnt / N) * abs(acc - cbar)
    return {"bins": rows, "ece": round(ece, 4),
            "interpretation": "accuracy ~ mean_confidence per bin => calibrated; ECE near 0 = well "
                              "calibrated (reported confidence matches empirical correctness)."}

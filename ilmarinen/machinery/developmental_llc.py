"""Developmental read-out: track the Local Learning Coefficient lambda_hat(t) OVER training (direction D4).

B2 (`singular_complexity.py`) estimates lambda_hat ONCE, at the converged optimum w* -- correctly, because the
LLC is only meaningful at a genuine local minimum. But the developmental-interpretability program (Hoogland,
Wang, Farrugia-Roberts, Murfet, Wei 2024, "The Developmental Landscape of In-Context Learning"; Lehalleur,
Hoogland et al. 2025; the Timaeus/devinterp line; and the Dec-2025 study arXiv:2512.00686 testing an
Arrhenius-style rate law for the SLT free energy and lambda-vs-difficulty scaling) has moved to tracking
lambda_hat(t) as a TRAJECTORY, whose plateaus and jumps mark STAGED learning -- Bayesian phase transitions in
training. Metaopt's DYNAMICS stage trains weights (+ alpha) and reads out at the end; it never looks at the
trajectory. This module adds that read-out: record lambda_hat at checkpoints DURING a fit and report the
developmental curve -- where complexity turns on and (when the data induces it) where it jumps.

The estimator is REUSED UNCHANGED. At each checkpoint we simply call `singular_complexity.estimate_llc` on the
net at its current (partially-trained) parameters. The whole content of D4 is the SCHEDULE and the READING of
the resulting curve; no change is made to the LLC estimator, to selection, or to any validated fit path.

--------------------------------------------------------------------------------------------------------------
What the curve MEANS (and the honest scope -- both established by premise-check before this module was wired)

  1. CONVERGENCE ONSET (robust, first-class signal). Before w* reaches a genuine minimum it is still on a
     downward slope, so SGLD immediately escapes to lower loss, E_post[L] < L*, and lambda_hat comes out
     strongly NEGATIVE (unphysical: RLCT >= 0) -- exactly the invalid regime B2 documents. As training
     converges, lambda_hat CROSSES FROM NEGATIVE (invalid) TO A STABLE POSITIVE value. That located crossing
     is the checkpoint where the architecture's usable capacity turns on. Premise-checked: on a singular
     over-wide tanh net the flip is sharp and coincides with the training loss flattening, and the final
     lambda_hat << k/2 (the singular signature). This signal is reliable.

  2. STAGED SUB-STRUCTURE (conditional). Distinct plateaus/jumps WITHIN the positive regime mark separate
     developmental stages (successive sub-circuits turning on) -- the clean lambda(t) staircases of the
     literature. Premise-check found these appear ONLY when the data induces temporally-separated learning
     phases: with a small MLP + Adam, an easy dominant mode is fit fast and swamps a subtle mode, giving a
     SINGLE onset rather than two. So this module REPORTS the curve shape and lets the caller SEE whether
     staging occurred; it does not assert multiple stages. The field's clean staircases come from settings
     (larger models, curricula, toy algorithmic tasks) engineered to separate the stages.

  3. VALUES ARE NOISY PER CHECKPOINT. A short per-checkpoint SGLD probe is deliberately cheap, so absolute
     lambda_hat(t) values -- especially in the negative regime, where the per-chain std is large -- are NOT to
     be over-read. The deliverable is the SHAPE and the LOCATED TRANSITIONS, which is precisely how the
     developmental-interpretability literature uses lambda(t). This is stated in every returned record.

--------------------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

from .singular_complexity import estimate_llc, free_energy


def default_checkpoints(total_epochs, n=12):
    """A checkpoint schedule DENSER EARLY (where the convergence onset lives) and sparser late (where the
    curve plateaus). Geometric-ish spacing on [0, total_epochs]. Returns a sorted list of unique ints
    including 0 and total_epochs."""
    total = int(max(total_epochs, 1))
    if total <= n:
        return list(range(total + 1))
    # geometric spacing gives more early points; anchor 0 and total
    raw = [0] + [int(round(total * (i / (n - 1)) ** 1.7)) for i in range(1, n)]
    cps = sorted(set(min(c, total) for c in raw))
    if cps[-1] != total:
        cps.append(total)
    return sorted(set(cps))


def _locate_transitions(curve, onset_tol=0.5):
    """Read the developmental curve for (a) the convergence-onset checkpoint -- the first checkpoint where
    lambda crosses from invalid (strongly negative) to valid (positive within noise) and STAYS valid, and
    (b) candidate staged jumps within the positive regime -- consecutive valid checkpoints where lambda rises
    by more than the local per-chain noise. Returns a dict; entries are None/empty when not present."""
    eps = [c["epoch"] for c in curve]
    lam = [c["lambda"] for c in curve]
    val = [c["valid"] for c in curve]

    # (a) convergence onset: first index i that is valid AND all subsequent are valid (stable flip)
    onset = None
    for i in range(len(curve)):
        if val[i] and all(val[j] for j in range(i, len(curve))):
            onset = eps[i]
            break
    # fallback: first valid at all (if it never becomes permanently valid)
    first_valid = next((eps[i] for i in range(len(curve)) if val[i]), None)

    # (b) staged jumps: among the tail of consecutive valid checkpoints, flag a rise exceeding the summed std
    jumps = []
    valid_idx = [i for i in range(len(curve)) if val[i]]
    for a, b in zip(valid_idx, valid_idx[1:]):
        if b != a + 1:
            continue  # only compare adjacent checkpoints
        rise = lam[b] - lam[a]
        noise = curve[a]["lambda_std"] + curve[b]["lambda_std"]
        if rise > max(noise, onset_tol) and rise > 0:
            jumps.append(
                {
                    "from_epoch": eps[a],
                    "to_epoch": eps[b],
                    "delta_lambda": round(float(rise), 3),
                    "noise_band": round(float(noise), 3),
                }
            )

    return {
        "convergence_onset_epoch": onset,
        "first_valid_epoch": first_valid,
        "candidate_staged_jumps": jumps,
        "n_valid_checkpoints": len(valid_idx),
    }


def developmental_llc(
    build_net,
    make_closure,
    train_step,
    n,
    *,
    total_epochs=None,
    checkpoints=None,
    chains=3,
    steps=120,
    burn=45,
    eps=1e-4,
    gamma=100.0,
    seed=0,
    k_params=None,
    verbose=False,
    log=None,
):
    """Track lambda_hat over ONE training trajectory of a freshly built net, probing at checkpoints.

    This drives its own explicit training loop (so it touches no validated fit path) using caller-supplied
    hooks that describe the deployed architecture and contract:

      build_net()          -> a fresh nn.Module (the SELECTED architecture), on the right device, untrained.
                              Called once. Its parameters are the trajectory that gets probed.
      make_closure(net)    -> a callable()->scalar mean-loss tensor at net's CURRENT params on a fixed
                              training minibatch, matching the contract's forward (exactly the closure B2's
                              _llc_report builds). Used BOTH to take training steps and to feed estimate_llc.
      train_step(net, opt) -> take ONE optimizer step on net (compute loss via its own closure, backward,
                              opt.step()). Returns the scalar training-loss value (float) for logging.
      n                    -> number of training points (sets the SGLD beta = 1/log n scale), as in B2.

    total_epochs / checkpoints: the trajectory length and where to probe. If checkpoints is None, a
      denser-early default schedule over total_epochs is used. total_epochs defaults to max(checkpoints).
    chains/steps/burn/eps/gamma/seed: the per-checkpoint SGLD probe budget -- kept SMALL by default (the
      curve is probed many times); raise steps/chains for a less noisy curve at higher cost.
    k_params: optional param count for the ratio column (else read from the built net).

    Returns a dict:
      {"curve": [ {epoch, train_loss, lambda, lambda_std, ratio, valid, free_energy_singular}, ... ],
       "transitions": {convergence_onset_epoch, candidate_staged_jumps, ...},
       "final": <the last checkpoint record>,
       "k_params": k, "half_params": k/2, "n": n,
       "note": <how to read this: shape + located onset are the deliverable; values are noisy>}
    The trajectory is trained ONCE; the returned net trajectory is not persisted (this is a read-out, not a
    replacement for the deployed fit).
    """
    import torch  # local import: keep module import light and torch-optional at import time

    def _log(msg):
        if verbose:
            (log or print)(msg)

    net = build_net()
    if k_params is None:
        k_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    half = k_params / 2.0

    closure = make_closure(net)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)

    if checkpoints is None:
        te = int(total_epochs if total_epochs is not None else 200)
        checkpoints = default_checkpoints(te, n=12)
    else:
        checkpoints = sorted(set(int(c) for c in checkpoints))
    if total_epochs is None:
        total_epochs = checkpoints[-1]

    curve = []
    ep = 0
    last_train_loss = float("nan")
    for target in checkpoints:
        while ep < target:
            last_train_loss = float(train_step(net, opt))
            ep += 1
        # probe lambda at THIS checkpoint (net params are the current w*)
        r = estimate_llc(net, closure, n, chains=chains, steps=steps, burn=burn, eps=eps, gamma=gamma, seed=seed)
        rec = {
            "epoch": ep,
            "train_loss": (last_train_loss if ep > 0 else float(r["L_star"])),
            "lambda": round(float(r["lambda"]), 4),
            "lambda_std": round(float(r["lambda_std"]), 4),
            "ratio": (round(float(r["lambda"]) / half, 6) if half > 0 else float("nan")),
            "valid": bool(r["valid"]),
            "free_energy_singular": round(free_energy(r["L_star"], r["lambda"], n), 4),
        }
        curve.append(rec)
        _log(
            f"[developmental_llc] epoch {ep:>4}: L={rec['train_loss']:.4f} "
            f"lambda={rec['lambda']:.3f}+/-{rec['lambda_std']:.3f} "
            f"ratio={rec['ratio']:.4f} valid={rec['valid']}"
        )

    transitions = _locate_transitions(curve)
    out = {
        "curve": curve,
        "transitions": transitions,
        "final": curve[-1] if curve else None,
        "k_params": int(k_params),
        "half_params": half,
        "n": int(n),
        "checkpoints": checkpoints,
        "note": (
            "Developmental LLC lambda_hat(t) over training (D4). READ THE SHAPE, NOT THE VALUES: the "
            "robust signal is the CONVERGENCE ONSET -- the located checkpoint where lambda flips from "
            "strongly negative (invalid: w* not yet at a minimum, SGLD escapes downhill) to a stable "
            "positive value (usable capacity turns on). Distinct plateaus/jumps WITHIN the positive "
            "regime mark staged learning, but appear only when the data induces temporally-separated "
            "phases; candidate_staged_jumps flags rises exceeding the per-checkpoint SGLD noise and is "
            "advisory. Per-checkpoint lambda is a short, noisy SGLD estimate (especially in the negative "
            "regime); the final lambda << k/2 is the singular-complexity signature, consistent with B2."
        ),
    }
    return out

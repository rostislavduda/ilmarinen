"""T-DEV (developmental LLC / D4): the checkpointed lambda_hat(t) tracker.

Fast tests lock the schedule and the transition-location logic (pure, no training). The end-to-end
tracker run over a real trajectory is @smoke (SGLD + training loop).
"""

import math

import pytest
import torch

from ilmarinen.machinery.developmental_llc import (default_checkpoints, developmental_llc,
                                                 _locate_transitions)


# --------------------------------------------------------------------------- schedule (fast, exact)
def test_default_checkpoints_anchors_and_monotone():
    """Schedule includes 0 and total_epochs, is strictly increasing, and denser early."""
    cps = default_checkpoints(200, n=12)
    assert cps[0] == 0 and cps[-1] == 200
    assert all(b > a for a, b in zip(cps, cps[1:])), "checkpoints must be strictly increasing"
    # denser early: first gap < last gap (geometric spacing)
    assert (cps[1] - cps[0]) < (cps[-1] - cps[-2])


def test_default_checkpoints_small_total():
    """When total <= n, every epoch is a checkpoint (0..total)."""
    assert default_checkpoints(5, n=12) == [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- transition location (fast)
def _rec(epoch, lam, std=0.1, valid=True):
    return {"epoch": epoch, "lambda": lam, "lambda_std": std, "valid": valid}


def test_locate_convergence_onset():
    """The onset is the first checkpoint that is valid AND stays valid afterwards (the neg->pos flip)."""
    curve = [
        _rec(0, -50.0, 5.0, False),
        _rec(10, -8.0, 2.0, False),
        _rec(25, 0.5, 0.4, True),      # <- first stable-valid checkpoint
        _rec(50, 1.2, 0.3, True),
        _rec(90, 1.5, 0.3, True),
    ]
    tr = _locate_transitions(curve)
    assert tr["convergence_onset_epoch"] == 25
    assert tr["first_valid_epoch"] == 25
    assert tr["n_valid_checkpoints"] == 3


def test_onset_requires_stable_validity():
    """A lone valid blip that then goes invalid again is NOT the onset (must stay valid)."""
    curve = [
        _rec(0, -20.0, 3.0, False),
        _rec(10, 0.3, 2.0, True),      # transient blip
        _rec(20, -5.0, 2.0, False),    # back to invalid
        _rec(40, 0.8, 0.3, True),      # <- real, stable onset
        _rec(70, 1.0, 0.3, True),
    ]
    tr = _locate_transitions(curve)
    assert tr["convergence_onset_epoch"] == 40


def test_staged_jump_detected_above_noise():
    """A rise between adjacent valid checkpoints exceeding the summed per-chain std is flagged."""
    curve = [
        _rec(0, 0.5, 0.05, True),
        _rec(10, 0.55, 0.05, True),    # small rise within noise -> not a jump
        _rec(20, 2.5, 0.10, True),     # big rise (1.95 >> 0.15+0.10) -> jump
    ]
    tr = _locate_transitions(curve)
    jumps = tr["candidate_staged_jumps"]
    assert len(jumps) == 1
    assert jumps[0]["from_epoch"] == 10 and jumps[0]["to_epoch"] == 20


def test_no_jump_when_all_within_noise():
    """A gently monotone valid curve with large noise bands yields no staged jumps."""
    curve = [_rec(e, 0.5 + 0.02 * i, 0.5, True) for i, e in enumerate([0, 10, 20, 40])]
    tr = _locate_transitions(curve)
    assert tr["candidate_staged_jumps"] == []


# --------------------------------------------------------------------------- end-to-end tracker (@smoke)
@pytest.mark.smoke
def test_developmental_llc_runs_and_is_wellformed():
    """The tracker trains one trajectory, probes lambda at checkpoints, and returns a well-formed dict
    whose final lambda << k/2 (singular signature). SGLD-noisy -> @smoke."""
    torch.manual_seed(0)
    n, d = 256, 6
    X = torch.randn(n, d)
    w = torch.randn(d)
    y = torch.tanh(3.0 * (X @ w) / math.sqrt(d)).unsqueeze(1)

    def build_net():
        return torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.Tanh(),
                                   torch.nn.Linear(16, 1))

    def make_closure(net):
        lf = torch.nn.MSELoss()
        return lambda: lf(net(X), y)

    def train_step(net, opt):
        lf = torch.nn.MSELoss()
        opt.zero_grad()
        loss = lf(net(X), y)
        loss.backward()
        opt.step()
        return float(loss.item())

    out = developmental_llc(build_net, make_closure, train_step, n,
                            checkpoints=[0, 10, 30, 70, 130], chains=2, steps=60, burn=20, seed=0)
    # well-formed
    assert len(out["curve"]) == 5
    for rec in out["curve"]:
        assert set(rec) >= {"epoch", "train_loss", "lambda", "lambda_std", "valid", "free_energy_singular"}
    assert out["curve"][0]["epoch"] == 0 and out["curve"][-1]["epoch"] == 130
    # singular signature: converged lambda is far below k/2
    k = out["k_params"]
    assert out["final"]["lambda"] < k / 2.0
    assert "convergence_onset_epoch" in out["transitions"]

"""T-DG: the autonomously-DISCOVERED-GROUP contract (the Lorentz / generated-equivariant path).

Under ``discover_equivariant_contract="extended"`` AllGraph recovers the data's symmetry METRIC from
the point positions, identifies the group, synthesises its Lie generators, and deploys a bespoke EMLP
contract (``contract == "generated_equivariant"``) instead of one of the eight built-in contracts. This is
AllGraph's headline capability and the most complex path in the controller -- and it was the ONE fit
path with effectively no unit coverage (the discovery/EMLP modules ran at ~0%), even though the project
docs advertise it as "verified end-to-end (metric R2=1.0, O(1,3), 6 generators)".

These tests pin that claim as a regression guard. They also lock the two eval/deploy fixes the path
depends on: the per-datum forward is deployed as a real ``nn.Module`` (so param-counting and persistence
work), and ``forward_generated_equivariant`` gives held-out data a batched forward -- the operation that
previously raised ``ValueError: torch.cat(): expected a non-empty list of Tensors`` at test-eval time.

The data is a self-contained synthetic Lorentz set (no JetNet dependency): each sample is a set of
massive 4-momenta [E, px, py, pz] with E = sqrt(m^2 + |p|^2) (timelike), and the target is the Minkowski
norm^2 of the POOLED 4-vector, (sum E)^2 - |sum p|^2 -- a genuine O(1,3)-invariant quadratic form, so
metric discovery recovers the indefinite Minkowski metric and identifies O(1,3). Fit-based, so ``smoke``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from ilmarinen import AllData, AllGraph
from ilmarinen.core.emlp_layer import special_linear_generators, symplectic_generators


def _lorentz_sets(n=160, kmin=3, kmax=8, seed=0):
    """n samples of variable-size massive-4-momentum sets, target = Minkowski norm^2 of the pooled
    4-vector (standardized). Positions carry the 4-vectors so metric discovery can read O(1,3) off them."""
    rng = np.random.RandomState(seed)
    sets, y = [], []
    for _ in range(n):
        k = rng.randint(kmin, kmax + 1)
        p = rng.randn(k, 3).astype(np.float32)                          # 3-momenta
        m = rng.uniform(0.5, 2.0, size=k).astype(np.float32)            # masses
        E = np.sqrt(m ** 2 + (p ** 2).sum(1)).astype(np.float32)        # energy (timelike: E > |p|)
        four = np.concatenate([E[:, None], p], axis=1).astype(np.float32)  # [E, px, py, pz]
        sets.append(four)
        S = four.sum(0)
        y.append(float(S[0] ** 2 - (S[1] ** 2 + S[2] ** 2 + S[3] ** 2)))   # Minkowski norm^2 of pooled vec
    y = np.array(y, np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return AllData.point_sets(sets, y=y, positions=sets)


def test_forward_generated_raises_without_a_discovered_fit():
    """T-DG-8: forward_generated_equivariant on an AllGraph that never deployed a discovered-group contract
    raises a clear RuntimeError, not an obscure crash. `_geq_forward` is published only by the generated-
    group fits and is reset to None at the start of every fit, so an unfitted (or normally-fitted) model has
    none. Fast tier -- no fit needed, the guard is checked before anything else."""
    dummy = AllData.point_sets([np.zeros((2, 4), np.float32)], y=np.zeros(1, np.float32),
                               positions=[np.zeros((2, 4), np.float32)])
    mg = AllGraph(width=8, depth=1, epochs=1, device="cpu", seed=0)
    with pytest.raises(RuntimeError):
        mg.forward_generated_equivariant(dummy)


class TestDiscoveredLorentzContract:
    """discover_equivariant_contract='extended' on a Lorentz set -> the generated_equivariant EMLP path."""

    pytestmark = pytest.mark.smoke

    @pytest.fixture(scope="class")
    def fitted(self):
        """Fit ONCE and share (all assertions below are read-only). Seed-fixed and CPU, ~1s."""
        mg = AllGraph(width=8, depth=1, epochs=3, device="cpu", verbose=False, seed=0,
                      discover_equivariant_contract="extended")
        res = mg.fit(_lorentz_sets(n=160, seed=0), task="regression")
        test = _lorentz_sets(n=40, seed=1)                              # unseen sets for the eval path
        return mg, res, test

    def test_routes_to_generated_equivariant_lorentz(self, fitted):
        """T-DG-1: discovery routes the Lorentz set to a generated O(1,3) contract, not a built-in contract."""
        mg, res, _ = fitted
        assert mg.contract == "generated_equivariant"
        assert res["contract"] == "generated_equivariant"
        spec = mg.generated_equivariant_group
        assert spec is not None and spec["name"] == "O(1,3)"
        # so(1,3) has dim 6 = 3 rotations + 3 boosts; the result echoes the generator count
        assert len(spec["gens"]) == 6
        assert res["group_generators"] == 6

    def test_discovered_metric_is_lorentzian(self, fitted):
        """T-DG-1b: the recovered metric is INDEFINITE (Minkowski signature), not Euclidean -- exactly one
        eigenvalue differs in sign from the other three, which is what makes the group O(1,3) not O(4)."""
        mg, _, _ = fitted
        g = np.asarray(mg.generated_equivariant_group["metric"], dtype=float)
        assert g.shape == (4, 4)
        signs = np.sign(np.round(np.linalg.eigvalsh(g), 6))
        assert abs(int(signs.sum())) == 2, f"signature not Lorentzian (eig signs {signs})"

    def test_deployed_net_is_a_module_with_params(self, fitted):
        """T-DG-2: the deployed contract is a real nn.Module (not a bare dict), so .parameters() works --
        guards the param-count and persistence paths (a plain dict has no .parameters()/.eval())."""
        mg, _, _ = fitted
        assert isinstance(mg.net, torch.nn.Module)
        assert sum(p.numel() for p in mg.net.parameters()) > 0

    def test_forward_on_heldout_is_finite(self, fitted):
        """T-DG-3: forward_generated_equivariant gives UNSEEN sets a batched forward returning finite
        (n, 1) output -- the exact op that raised 'torch.cat(): expected a non-empty list' before the fix."""
        mg, _, test = fitted
        out = mg.forward_generated_equivariant(test)
        assert tuple(out.shape) == (len(np.asarray(test.y)), 1)
        assert np.isfinite(out.detach().numpy()).all()

    def test_predict_roundtrips_on_generated_contract(self, fitted):
        """T-DG-4: predict() routes the generated_equivariant contract through its per-datum forward and
        returns a finite (n,) vector -- guards the persistence/_forward_new branch for this contract."""
        mg, _, test = fitted
        pred = mg.predict(test)
        assert pred.shape == (len(np.asarray(test.y)),)
        assert np.isfinite(pred).all()

    def test_forward_generated_requires_positions(self, fitted):
        """T-DG-9: the per-datum contract reads each sample's positions, so scoring a positions-less AllData
        raises a clear ValueError rather than dereferencing None. Uses the FITTED model (so it is past the
        RuntimeError guard) and hands it edgeless point sets built without positions."""
        mg, _, test = fitted
        no_pos = AllData.point_sets(list(test.node_feats), y=np.asarray(test.y))   # positions omitted -> None
        assert no_pos.positions is None
        with pytest.raises(ValueError):
            mg.forward_generated_equivariant(no_pos)


def _phase_space_sets(n, vec_dim, seed=0):
    """n samples of variable-size vec_dim-dimensional point sets for the Sp/SL contract. node_feats carry a
    CANONICAL one-hot labeling (which axis each point belongs to) -- the skew/volume attention gate keys off
    fixed labels, NOT the coordinates (coordinates transform under the group). Target is a smooth scalar of
    the pooled vector; these tests exercise the deploy path, not fit accuracy, so the target need only be
    well-posed and finite."""
    rng = np.random.RandomState(seed)
    sets, pos, y = [], [], []
    for _ in range(n):
        k = rng.randint(4, 9)
        P = rng.randn(k, vec_dim).astype(np.float32)
        F = np.eye(vec_dim, dtype=np.float32)[rng.randint(0, vec_dim, k)]   # canonical per-point labels
        pos.append(P); sets.append(F)
        S = P.sum(0)
        if vec_dim == 4:
            y.append(float(S[0] * S[2] - S[1] * S[3]))                      # a symplectic pairing omega(q,p)
        else:
            y.append(float(np.prod(S[:3])))                                 # a smooth SL-flavoured scalar
    y = np.array(y, np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return AllData.point_sets(sets, y=y, positions=pos)


def _skew_group_spec(kind):
    """Hand-specified group spec for the Sp/SL branch: Sp(4) (2 dof phase space) or SL(3). Passed via the
    supported generated_equivariant_group constructor arg to deploy the contract directly."""
    if kind == "Sp":
        return {"name": "Sp(4)", "vec_dim": 4, "gens": symplectic_generators(2)[0]}
    return {"name": "SL(3)", "vec_dim": 3, "gens": special_linear_generators(3)}


class TestSkewVolumeContract:
    """The Sp/SL branch of the generated-group contract (_fit_skew_or_volume_contract). Unlike the metric
    (O(p,q)) path, these groups have a 1-D vector->vector commutant, so the contract forms its invariants
    from the INPUT points via learned attention (skew pairings for Sp, determinant frames for SL). This is
    the branch whose pre-fix net was a PLAIN DICT, so the nn.Module assertion below is the load-bearing
    regression guard. Parametrized over Sp(4) and SL(3) to cover both invariant families."""

    pytestmark = pytest.mark.smoke

    @pytest.fixture(scope="class", params=["Sp", "SL"])
    def fitted(self, request):
        spec = _skew_group_spec(request.param)
        vec_dim = spec["vec_dim"]
        mg = AllGraph(width=16, depth=1, epochs=3, device="cpu", verbose=False, seed=0,
                      generated_equivariant_group=spec)
        res = mg.fit(_phase_space_sets(200, vec_dim, seed=0), task="regression")
        test = _phase_space_sets(40, vec_dim, seed=1)
        return spec, mg, res, test

    def test_deploys_generated_contract_as_module(self, fitted):
        """T-DG-5: a forced Sp/SL group deploys the generated_equivariant contract, reporting the group name,
        as a real nn.Module -- guards the plain-dict regression (a dict has no .parameters(), which crashed
        the runners' param count and predict on this branch)."""
        spec, mg, res, _ = fitted
        assert mg.contract == "generated_equivariant"
        assert res["contract"] == "generated_equivariant"
        assert res["group"] == spec["name"]
        assert isinstance(mg.net, torch.nn.Module)
        assert sum(p.numel() for p in mg.net.parameters()) > 0

    def test_forward_on_heldout_is_finite(self, fitted):
        """T-DG-6: the deployed Sp/SL contract gives unseen sets a finite (n, 1) batched forward."""
        _, mg, _, test = fitted
        out = mg.forward_generated_equivariant(test)
        assert tuple(out.shape) == (len(np.asarray(test.y)), 1)
        assert np.isfinite(out.detach().numpy()).all()

    def test_predict_roundtrips(self, fitted):
        """T-DG-7: predict() round-trips the Sp/SL contract to a finite (n,) vector."""
        _, mg, _, test = fitted
        pred = mg.predict(test)
        assert pred.shape == (len(np.asarray(test.y)),)
        assert np.isfinite(pred).all()


# --------------------------------------------------------------------------- nonlinear latent contract (B3)
def _geom_regression(n=40, m=4, seed=0, task="regression"):
    """>=30 3D point sets with positions and a rotation-invariant target (|P|^2). The nonlinear-symmetry
    deploy path needs positions and at least 30 clouds. task='classification' thresholds |P|^2 at its median
    into two balanced classes (so the classification predict/argmax path can be exercised)."""
    rng = np.random.RandomState(seed)
    nf, po, y = [], [], []
    for _ in range(n):
        P = rng.randn(m, 3).astype(np.float32)
        nf.append(np.ones((m, 1), np.float32))
        po.append(P)
        y.append(float((P ** 2).sum()))
    y = np.array(y, np.float32)
    if task == "classification":
        y = (y > np.median(y)).astype(np.int64)
    else:
        y = (y - y.mean()) / (y.std() + 1e-8)
    return AllData.point_sets(nf, y=y, positions=po)


def _fake_confirmed(task_model, X, latent_dim=None, **kw):
    """Stand-in for the LaLiGAN discovery returning a CONFIRMED latent symmetry: a trivial linear encoder
    (d -> latent) and a skew (SO(2)-like) latent generator, so _fit_nonlinear_contract deploys the contract
    without the real ~600-epoch autoencoder search (whose own correctness is nonlinear_symmetry's concern)."""
    d = X.shape[1]
    ld = latent_dim or 2
    enc = nn.Sequential(nn.Linear(d, ld))
    g = np.zeros((ld, ld), np.float32); g[0, 1] = -1.0; g[1, 0] = 1.0
    return {"ae": SimpleNamespace(enc=enc), "generators_latent": [torch.tensor(g)],
            "learned_generator": g, "confirmed_by_null": True, "sym_violation": 1e-3,
            "null_violation": 1.0, "_null_ratio": 1.5, "latent_dim": ld, "n_symmetries": 1}


def _fake_rejected(task_model, X, latent_dim=None, **kw):
    """Stand-in returning a NON-confirmed result (null guard fails), so the confirmation gate rejects it."""
    return {"confirmed_by_null": False, "sym_violation": 0.5, "null_violation": 0.5, "_null_ratio": 1.5,
            "generators_latent": [], "learned_generator": None, "n_symmetries": 0}


class TestNonlinearLatentContract:
    """The nonlinear (B3) sibling of the discovered-group path: deploy_nonlinear_contract discovers a latent
    symmetry (LaLiGAN) and, if confirmed, deploys x -> encoder -> EquivariantMLP(latent gens) -> y. The
    expensive autoencoder search is stubbed (its math lives in nonlinear_symmetry); these lock the
    confirmation GATE and the deploy/fallback branches of _fit_nonlinear_contract."""

    pytestmark = pytest.mark.smoke

    def test_no_positions_returns_none(self):
        """T-DG-10: _fit_nonlinear_contract is a no-op (returns None) when the data carries no positions,
        so a positions-less fit falls through to the normal route. Fast: guarded before any discovery."""
        d = AllData.dense_tensor(np.random.RandomState(0).randn(40, 6).astype(np.float32),
                                 y=np.zeros(40, np.float32))
        mg = AllGraph(width=8, depth=1, epochs=2, device="cpu", seed=0, deploy_nonlinear_contract=True)
        assert mg._fit_nonlinear_contract(d, "regression", 1) is None

    def test_unconfirmed_symmetry_falls_back(self, monkeypatch):
        """T-DG-11: when discovery does NOT confirm a deployable symmetry, the contract is NOT deployed --
        the fit falls back to the normal route and records why in route_detail."""
        monkeypatch.setattr("ilmarinen.core.nonlinear_symmetry.discover_nonlinear_symmetries_joint",
                            _fake_rejected)
        mg = AllGraph(width=8, depth=1, epochs=3, device="cpu", verbose=False, seed=0,
                      deploy_nonlinear_contract=True)
        mg.fit(_geom_regression(), task="regression", n_out=1)
        assert mg.contract != "latent_equivariant"
        assert "nonlinear_contract" in (mg.route_detail or {})

    def test_confirmed_symmetry_deploys_latent_contract(self, monkeypatch):
        """T-DG-12: a confirmed latent symmetry deploys the latent-equivariant contract -- a real nn.Module
        (encoder + latent EMLP head) with a finite trained fit value."""
        monkeypatch.setattr("ilmarinen.core.nonlinear_symmetry.discover_nonlinear_symmetries_joint",
                            _fake_confirmed)
        mg = AllGraph(width=8, depth=1, epochs=3, device="cpu", verbose=False, seed=0,
                      deploy_nonlinear_contract=True)
        res = mg.fit(_geom_regression(), task="regression", n_out=1)
        assert mg.contract == "latent_equivariant"
        assert res["contract"] == "latent_equivariant"
        assert isinstance(mg.net, nn.Module)
        assert sum(p.numel() for p in mg.net.parameters()) > 0
        assert np.isfinite(res["value"])

    def test_predict_roundtrips_on_latent_contract(self, monkeypatch):
        """T-DG-13: predict() scores NEW data through the latent contract -- returning a finite (n,) vector,
        including clouds LARGER and SMALLER than training (truncate / zero-pad to the fit-time input dim).
        Regression guard for the fixed inference routing: this path previously raised
        'LatentEquivariantContract.forward() takes 2 positional arguments' because _forward_new had no
        latent_equivariant branch and called the net with the 5 relational args."""
        monkeypatch.setattr("ilmarinen.core.nonlinear_symmetry.discover_nonlinear_symmetries_joint",
                            _fake_confirmed)
        mg = AllGraph(width=8, depth=1, epochs=3, device="cpu", verbose=False, seed=0,
                      deploy_nonlinear_contract=True)
        mg.fit(_geom_regression(n=40, m=4, seed=0), task="regression", n_out=1)
        for label, te in (("same", _geom_regression(n=12, m=4, seed=1)),
                          ("bigger", _geom_regression(n=8, m=6, seed=2)),     # longer clouds -> truncated
                          ("smaller", _geom_regression(n=6, m=2, seed=3))):   # shorter clouds -> zero-padded
            pred = mg.predict(te)
            assert pred.shape == (len(np.asarray(te.y)),), label
            assert np.isfinite(pred).all(), label

    def test_predict_classification_returns_labels(self, monkeypatch):
        """T-DG-14: a CLASSIFICATION latent contract's predict() returns integer class labels (argmax), and
        predict_proba a per-row simplex. Guards the second half of the bug: the nonlinear deploy early-returns
        before fit()'s STAGE-4 tail, so _infer_task was never set and predict silently treated every latent
        model as regression."""
        monkeypatch.setattr("ilmarinen.core.nonlinear_symmetry.discover_nonlinear_symmetries_joint",
                            _fake_confirmed)
        mg = AllGraph(width=8, depth=1, epochs=3, device="cpu", verbose=False, seed=0,
                      deploy_nonlinear_contract=True)
        mg.fit(_geom_regression(task="classification"), task="classification", n_out=2)
        assert mg._infer_task == "classification"
        te = _geom_regression(n=12, seed=1, task="classification")
        pred = mg.predict(te)
        assert pred.shape == (12,)
        assert set(np.unique(pred).tolist()) <= {0, 1}
        proba = mg.predict_proba(te)
        assert proba.shape == (12, 2)
        assert np.allclose(proba.sum(1), 1.0, atol=1e-5)

"""T-MG: AllGraph orchestration paths not covered by the smoke/contract suites.

The existing suite locks the *machinery* (Gibbs readout, pricing/MDL, LLC, RG, symmetry
primitives, routing, AllData) and one default-settings smoke fit per contract. This module
exercises AllGraph's own orchestration surface that was otherwise untested:

  Phase 1  selection modes         -- select = gibbs / sparse (argmax is the smoke default)
  Phase 2  remaining contracts        -- spatial / volumetric / 4d dense fits
  Phase 3  auto-epoch stopping     -- auto_epoch = train / val early stop
  Phase 4  cross-fit instance reuse -- device + canonicalization state reset between fits
  Phase 5  public methods          -- route / explain / select_architecture[_by_area] /
                                       apply_canonicalization / tiebreak

These assert the *result contract* and documented invariants, not accuracy -- fast, in-process,
seed-fixed. Fit-based tests are marked ``smoke`` (deselect with ``-m "not smoke"``).
"""

import numpy as np
import pytest

from ilmarinen import AllData, AllGraph
from ilmarinen.core.allgraph import _EarlyStopper

# Fit-based classes carry a class-level ``pytestmark = pytest.mark.smoke`` (deselect with -m "not smoke");
# the pure-logic tests (early-stop machinery, routing) stay in the fast tier.

# universal selection/result fields present in every contract's result dict (mirrors test_smoke_fit)
RESULT_KEYS = {"contract", "value", "architecture", "n_params", "route", "selected_primitive", "metric"}


def _mg(**kw):
    kw.setdefault("width", 8)
    kw.setdefault("depth", 1)
    kw.setdefault("epochs", 3)
    kw.setdefault("verbose", False)
    kw.setdefault("seed", 0)
    return AllGraph(**kw)


def _linsep(n=80, d=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(np.int64)
    return AllData.dense_tensor(X, y)


# =========================================================================== Phase 1: selection modes
class TestSelectionModes:
    """select = gibbs / sparse -- the two non-default selection pipelines (argmax is the smoke default)."""

    pytestmark = pytest.mark.smoke

    def test_gibbs_selection_contract(self):
        """T-MG-1: gibbs fit returns the universal keys plus its Boltzmann-readout fields."""
        r = _mg().fit(_linsep(), task="classification", n_out=2, select="gibbs")
        assert RESULT_KEYS.issubset(r.keys()), r.keys()
        assert {"architecture_gibbs", "gibbs_weights", "gibbs_energies"}.issubset(r.keys())
        assert r["gibbs_alpha"] == r["gibbs_weights"]        # deprecated alias still present
        assert np.isfinite(r["value"])

    def test_gibbs_weights_is_simplex(self):
        """T-MG-1b: the deployed gibbs weights are a probability vector over the primitives."""
        r = _mg().fit(_linsep(), task="classification", n_out=2, select="gibbs")
        w = r["gibbs_weights"]
        assert isinstance(w, dict) and w
        assert all(v >= 0 for v in w.values())
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_gibbs_selected_primitive_is_weights_argmax(self):
        """T-MG-1c: the reported selected primitive is the lowest-energy (largest-weight) one."""
        r = _mg().fit(_linsep(), task="classification", n_out=2, select="gibbs")
        w = r["gibbs_weights"]
        assert r["selected_primitive"] == max(w, key=w.get)

    def test_sparse_selection_contract(self):
        """T-MG-2: sparse fit reports the effective-primitive count (ipr) and the price it used."""
        r = _mg(sparsity_mu=1.0).fit(_linsep(), task="classification", n_out=2, select="sparse")
        assert RESULT_KEYS.issubset(r.keys()), r.keys()
        assert "ipr" in r and "sparsity_mu" in r
        n_prim = len(_mg().fit(_linsep(), n_out=2, select="gibbs")["gibbs_weights"])
        assert 1.0 <= r["ipr"] <= n_prim + 1e-6, r["ipr"]
        assert r["sparsity_mu"] == pytest.approx(1.0)

    def test_sparse_pricing_reduces_ipr(self):
        """T-MG-2b: a sparsity price yields a MORE compact mixture than the unpriced fit.

        Regression guard for the vanishing-gradient-at-uniform bug: the mixture alpha init to torch.zeros
        (uniform) is a symmetric critical point of the symmetric -sum(alpha^2) price, so without a
        symmetry break the price could not sparsify and ipr was ~#primitives regardless of mu. With the
        seeded break in place, mu>0 drives ipr well below the mu=0 baseline at a realistic budget.
        """
        data = _linsep(n=120)
        ipr_free = _mg(width=16, epochs=60, sparsity_mu=0.0).fit(data, n_out=2, select="sparse")["ipr"]
        ipr_priced = _mg(width=16, epochs=60, sparsity_mu=10.0).fit(data, n_out=2, select="sparse")["ipr"]
        assert ipr_priced < ipr_free, (ipr_free, ipr_priced)
        assert ipr_priced <= ipr_free - 1.0, (ipr_free, ipr_priced)   # a clear, not marginal, reduction

    def test_selection_is_deterministic(self):
        """T-MG-3: each selection mode is reproducible under a fixed seed."""
        for mode in ("gibbs", "sparse"):
            a = _mg().fit(_linsep(), n_out=2, select=mode)["architecture"]
            b = _mg().fit(_linsep(), n_out=2, select=mode)["architecture"]
            assert a == b, (mode, a, b)


# =========================================================================== Phase 1b: variable size
class TestSelectSize:
    """select_size = variable -- per-layer width/primitive selection (sequential is the default)."""

    pytestmark = pytest.mark.smoke

    def test_variable_size_contract(self):
        """T-MG-4: a variable-size fit returns a well-formed per-layer architecture."""
        r = _mg(depth=2).fit(_linsep(), task="classification", n_out=2, select_size="variable")
        assert RESULT_KEYS.issubset(r.keys()), r.keys()
        arch = r["architecture"]
        assert isinstance(arch, list) and len(arch) >= 1
        assert all(isinstance(p, str) for p in arch)


# =========================================================================== Phase 2: remaining contracts
def _grid_data(shape, seed=0):
    """Dense tensor of shape (n, *shape) with a 2-class label; routes by rank to spatial/volumetric/4d."""
    rng = np.random.RandomState(seed)
    X = rng.randn(*shape).astype(np.float32)
    y = (np.arange(shape[0]) % 2).astype(np.int64)
    return AllData.dense_tensor(X, y)


class TestArenas:
    """Smoke fits for the contracts the existing suite never reaches (it covers sequence/set/graph/operator).

    Routing is by tensor rank: (n,C,H,W) -> spatial, (n,C,D,H,W) -> volumetric, (n,C,X,Y,Z,T) -> 4d.
    """

    pytestmark = pytest.mark.smoke

    def test_spatial_arena(self):
        """T-MG-5: a 2D image tensor routes to the spatial contract and fits."""
        r = _mg().fit(_grid_data((30, 1, 8, 8)), task="classification", n_out=2)
        assert r["contract"] == "spatial", r["contract"]
        assert RESULT_KEYS.issubset(r.keys())
        assert np.isfinite(r["value"])

    def test_volumetric_arena(self):
        """T-MG-6: a 3D volume tensor routes to the volumetric contract and fits."""
        r = _mg().fit(_grid_data((24, 1, 6, 6, 6)), task="classification", n_out=2)
        assert r["contract"] == "volumetric", r["contract"]
        assert RESULT_KEYS.issubset(r.keys())

    def test_4d_arena(self):
        """T-MG-7: a 4D (spatiotemporal) tensor routes to the 4d contract and fits."""
        r = _mg().fit(_grid_data((20, 1, 4, 4, 4, 4)), task="classification", n_out=2)
        assert r["contract"] == "4d", r["contract"]
        assert RESULT_KEYS.issubset(r.keys())

    def test_spatial_multichannel(self):
        """T-MG-5b: multi-channel images also route to spatial (channel dim is not spatial)."""
        r = _mg().fit(_grid_data((30, 3, 8, 8)), task="classification", n_out=2)
        assert r["contract"] == "spatial", r["contract"]


# =========================================================================== Phase 3: auto-epoch stopping
class TestEarlyStopperLogic:
    """The _EarlyStopper plateau detector -- pure, deterministic, no training (fast tier)."""

    def test_stops_after_patience_on_plateau(self):
        """T-MG-8: after min_epochs, patience consecutive non-improving epochs trigger a stop."""
        s = _EarlyStopper(min_delta=0.01, patience=3, min_epochs=2)
        # two clear improvements (past min_epochs floor), then a flat plateau
        assert s.step(1.0) is False      # epoch 1 (improve, but < min_epochs)
        assert s.step(0.5) is False      # epoch 2 (improve)
        assert s.step(0.5) is False      # epoch 3 bad=1
        assert s.step(0.5) is False      # epoch 4 bad=2
        assert s.step(0.5) is True       # epoch 5 bad=3 == patience -> stop

    def test_min_epochs_floor_respected(self):
        """T-MG-8b: never stops before min_epochs even with immediate stagnation."""
        s = _EarlyStopper(min_delta=0.01, patience=1, min_epochs=5)
        stops = [s.step(1.0) for _ in range(4)]   # epochs 1..4, all stagnant, bad>=patience
        assert not any(stops), stops               # but below the min_epochs floor -> no stop

    def test_continuous_improvement_never_stops(self):
        """T-MG-8c: a strictly-improving metric resets patience and never stops."""
        s = _EarlyStopper(min_delta=0.01, patience=2, min_epochs=2)
        assert not any(s.step(1.0 - 0.05 * i) for i in range(10))

    def test_min_delta_is_relative(self):
        """T-MG-8d: a reduction below the RELATIVE min_delta counts as no-improvement."""
        s = _EarlyStopper(min_delta=0.10, patience=2, min_epochs=1)
        s.step(1.0)                        # best = 1.0
        # 0.95 is only a 5% cut, below the 10% min_delta -> counts as bad
        assert s.step(0.95) is False       # bad=1
        assert s.step(0.94) is True        # bad=2 == patience -> stop


class TestAutoValSplit:
    """_auto_val_split monitor-set policy -- pure index logic, no training (fast tier)."""

    def test_off_mode_uses_all_for_train(self):
        """T-MG-9: with auto_epoch off (or 'train'), no val is held out."""
        tr, va = _mg(auto_epoch=None)._auto_val_split(500)
        assert va is None and len(tr) == 500

    def test_val_mode_holds_out_monitor(self):
        """T-MG-9b: 'val' on ample data holds out max(15%, 50) as the monitor set."""
        mg = _mg(auto_epoch="val")
        tr, va = mg._auto_val_split(1000)
        assert va is not None
        assert len(va) == max(int(0.15 * 1000), mg._AUTO_VAL_MIN) == 150
        assert len(tr) + len(va) == 1000
        assert set(tr).isdisjoint(set(va))     # disjoint split

    def test_val_mode_falls_back_on_small_data(self):
        """T-MG-9c: too-small data can't spare a reliable monitor -> fall back to train-loss monitoring."""
        # n=100: needs max(15, 50)=50 val, but 50 > 0.35*100=35 -> fallback
        tr, va = _mg(auto_epoch="val")._auto_val_split(100)
        assert va is None and len(tr) == 100


class TestMakeStopper:
    """_make_stopper wires the constructor's auto_epoch config into an _EarlyStopper (fast tier)."""

    def test_none_when_auto_epoch_off(self):
        """T-MG-10: no stopper unless auto_epoch is set (full fixed budget)."""
        assert _mg(auto_epoch=None)._make_stopper() is None

    def test_configured_from_ctor(self):
        """T-MG-10b: the stopper inherits patience/min_delta and caps min_epochs at the epoch budget."""
        mg = _mg(epochs=3, auto_epoch="train", auto_epoch_patience=6,
                 auto_epoch_min_delta=0.02, auto_epoch_min_epochs=10)
        s = mg._make_stopper()
        assert isinstance(s, _EarlyStopper)
        assert s.patience == 6 and s.min_delta == pytest.approx(0.02)
        assert s.min_epochs == 3      # capped at self.epochs, not the requested 10


class TestAutoEpochFit:
    """End-to-end that both auto_epoch modes drive a real fit to a sane result (smoke)."""

    pytestmark = pytest.mark.smoke

    def test_auto_epoch_train_fit(self):
        """T-MG-11: auto_epoch='train' completes and beats chance on separable data."""
        r = _mg(width=16, epochs=40, auto_epoch="train",
                auto_epoch_patience=3).fit(_linsep(n=160), n_out=2)
        assert np.isfinite(r["value"]) and r["value"] > 0.6

    def test_auto_epoch_val_fit(self):
        """T-MG-11b: auto_epoch='val' (ample data for a real monitor) completes and beats chance."""
        r = _mg(width=16, epochs=40, auto_epoch="val",
                auto_epoch_patience=3).fit(_linsep(n=400), n_out=2)
        assert np.isfinite(r["value"]) and r["value"] > 0.6


# =========================================================================== relational data helpers
def _graphs(n_graphs=16, n_nodes=6, seed=0):
    rng = np.random.RandomState(seed)
    nf, ed, ys = [], [], []
    for _ in range(n_graphs):
        f = rng.randn(n_nodes, 4).astype(np.float32)
        e = np.array([(i, (i + 1) % n_nodes) for i in range(n_nodes)], dtype=np.int64).T
        nf.append(f); ed.append(e); ys.append(int(f[:, 0].sum() > 0))
    return AllData.graphs(nf, ed, y=np.array(ys, dtype=np.int64))


def _point_sets(n=24, m=6, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, m, 3).astype(np.float32)
    nf = np.zeros((n, m, 1), np.float32)
    y = (pts.sum(axis=(1, 2)) > 0).astype(np.int64)
    return AllData.point_sets(nf, y=y, positions=pts)


# =========================================================================== Phase 4: cross-fit reuse
class TestCrossFitReuse:
    """A reused AllGraph must reset per-fit state so an earlier fit cannot leak into a later one
    (guards the device / canonicalization reset at the top of fit())."""

    pytestmark = pytest.mark.smoke

    def test_fit_restores_base_device_and_clears_canon(self):
        """T-MG-12: stale device + canonicalization state from a prior fit are reset at fit-start."""
        import torch

        mg = _mg()
        assert mg._base_device == mg.device
        # simulate what a prior relational(->CPU) + canonicalizing fit would have left behind
        mg.device = torch.device("meta")
        mg._canonicalized_positions = "STALE"
        mg._canonicalization_applied = True
        mg.fit(_linsep(), n_out=2)     # a fresh, non-relational fit must scrub all three
        assert mg.device == mg._base_device
        assert mg._canonicalized_positions is None
        assert mg._canonicalization_applied is False

    def test_instance_is_reusable_across_arenas(self):
        """T-MG-12b: the same instance fits two different contracts back-to-back with well-formed results."""
        mg = _mg()
        r1 = mg.fit(_linsep(), n_out=2)
        r2 = mg.fit(_grid_data((30, 1, 8, 8)), n_out=2)
        assert r1["contract"] == "sequence" and r2["contract"] == "spatial"
        assert RESULT_KEYS.issubset(r1.keys()) and RESULT_KEYS.issubset(r2.keys())


# =========================================================================== Phase 5: public methods
class TestPublicMethods:
    """The public methods beyond fit() -- previously never called on an instance."""

    _BUILTIN_CONTRACTS = {"sequence", "spatial", "volumetric", "4d", "graph", "equivariant", "set", "operator"}

    def test_route_returns_contract_and_detail(self):
        """T-MG-13: route(data) -> (contract, detail-dict); no training involved (fast tier)."""
        contract, detail = _mg().route(_linsep())
        assert contract in self._BUILTIN_CONTRACTS, contract
        assert isinstance(detail, dict)

    def test_route_is_deterministic(self):
        """T-MG-13b: routing the same data twice yields the same contract."""
        d = _linsep()
        assert _mg().route(d)[0] == _mg().route(d)[0]

    def test_apply_canonicalization_returns_metadata(self):
        """T-MG-14: apply_canonicalization(point-set data) returns AllData with the positions preserved."""
        d = _point_sets()
        out = _mg().apply_canonicalization(d)
        assert isinstance(out, AllData)
        assert out.positions is not None
        assert np.asarray(out.positions).shape[0] == np.asarray(d.positions).shape[0]

    def test_explain_returns_dict_and_text(self):
        """T-MG-15: explain(result) -> dict; explain(result, as_text=True) -> non-empty string (smoke)."""
        mg = _mg()
        r = mg.fit(_linsep(), n_out=2)
        assert isinstance(mg.explain(r), dict)
        txt = mg.explain(r, as_text=True)
        assert isinstance(txt, str) and len(txt) > 0

    test_explain_returns_dict_and_text = pytest.mark.smoke(test_explain_returns_dict_and_text)

    def test_select_architecture_contract(self):
        """T-MG-16: select_architecture (relational) returns the chosen width*/depth* and its curve (smoke)."""
        out = _mg().select_architecture(_graphs(), task="classification", n_out=2,
                                        widths=(8, 16), depths=(1,), seeds=(0,), sweep_epochs=2)
        assert isinstance(out, dict)
        assert {"width_star", "depth_star", "contract"}.issubset(out.keys())
        assert out["width_star"] in (8, 16)

    test_select_architecture_contract = pytest.mark.smoke(test_select_architecture_contract)

    def test_select_architecture_by_area_contract(self):
        """T-MG-16b: select_architecture_by_area returns a joint width*/depth*/area* selection (smoke)."""
        out = _mg().select_architecture_by_area(_graphs(), task="classification", n_out=2,
                                                widths=(16, 32), depths=(1,), seeds=(0,), sweep_epochs=2)
        assert isinstance(out, dict)
        assert {"width_star", "depth_star", "area_star"}.issubset(out.keys())

    test_select_architecture_by_area_contract = pytest.mark.smoke(test_select_architecture_by_area_contract)

    def test_tiebreak_returns_contract(self):
        """T-MG-17: tiebreak(data) -> (contract, ...); the chosen contract is an contract-name string (smoke)."""
        out = _mg().tiebreak(_linsep(), task="classification", n_out=2, tiebreak_epochs=2)
        assert isinstance(out, tuple) and len(out) >= 1
        assert isinstance(out[0], str) and out[0]

    test_tiebreak_returns_contract = pytest.mark.smoke(test_tiebreak_returns_contract)


# =========================================================================== Phase 5b: perf-path invariants
class TestPerfPaths:
    """Guards for the performance refactor (device pre-move / set cache / capped readout bake-off): the
    optimized paths must produce the SAME collation/results as the originals."""

    def test_set_cache_matches_uncached_collation(self):
        """T-MG-21: the cached set collation is identical to the per-batch (uncached) path."""
        import torch

        mg = _mg()                                   # device cpu -> cache tensors land on cpu
        d = _point_sets(n=12, m=5)
        cache = mg._prepare_batch_cache(d, to_device=True)
        assert len(cache["node"]) == 12 and cache["counts"][0] == 5
        ids = np.array([0, 3, 7, 11])
        Xa, ba, na = mg._subbatch_sets(d, ids)
        Xb, bb, nb = mg._subbatch_sets(d, ids, cache=cache)
        assert na == nb == 4
        assert torch.equal(Xa, Xb) and torch.equal(ba, bb)

    def test_readout_bakeoff_runs(self):
        """T-MG-22: the sequence readout bake-off (readout_select=True) fits and picks a valid readout."""
        r = _mg(width=16, epochs=6, readout_select=True).fit(
            _linsep(n=80), task="classification", n_out=2)
        assert r["contract"] == "sequence"
        assert r["readout"] in ("mean", "flatten")

    test_readout_bakeoff_runs = pytest.mark.smoke(test_readout_bakeoff_runs)


# =========================================================================== Phase 6: save / load / predict
class TestSaveLoadPredict:
    """Persist a trained model and run it on new data on-demand (fit -> save -> load -> predict)."""

    pytestmark = pytest.mark.smoke

    def test_predict_on_new_data(self):
        """T-MG-18: a fitted classifier predicts labels for unseen samples (no labels needed)."""
        mg = _mg(width=16, epochs=8)
        mg.fit(_linsep(n=120), task="classification", n_out=2)
        rng = np.random.RandomState(9)
        new = AllData.dense_tensor(rng.randn(15, 8).astype(np.float32))
        pred = mg.predict(new)
        assert pred.shape == (15,)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_predict_proba_contract(self):
        """T-MG-18b: predict_proba returns a per-class simplex; errors on a regression model."""
        mg = _mg(width=16, epochs=6)
        mg.fit(_linsep(n=100), task="classification", n_out=2)
        proba = mg.predict_proba(AllData.dense_tensor(np.random.RandomState(1).randn(10, 8).astype(np.float32)))
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (proba >= 0).all()

    def test_save_load_round_trip_matches(self, tmp_path):
        """T-MG-19: a loaded model reproduces the original's predictions bit-for-bit on new data."""
        mg = _mg(width=16, epochs=8)
        mg.fit(_linsep(n=120), task="classification", n_out=2)
        new = AllData.dense_tensor(np.random.RandomState(7).randn(20, 8).astype(np.float32))
        before = mg.predict(new)
        path = mg.save(dirpath=str(tmp_path))
        loaded = AllGraph.load(path, device="cpu")
        assert loaded.contract == mg.contract
        assert loaded._infer_task == "classification"
        assert np.array_equal(before, loaded.predict(new))

    def test_saved_filename_has_contract_and_timestamp(self, tmp_path):
        """T-MG-19b: the default filename embeds the contract and a UTC timestamp (collision-avoiding)."""
        import os
        import re

        mg = _mg(epochs=3)
        mg.fit(_linsep(), task="classification", n_out=2)
        path = mg.save(dirpath=str(tmp_path))
        name = os.path.basename(path)
        assert re.fullmatch(rf"allgraph_{mg.contract}_\d{{8}}T\d{{6}}_\d+Z\.pt", name), name
        assert os.path.isfile(path)

    def test_save_stem_sets_filename_prefix(self, tmp_path):
        """T-MG-19d: an explicit stem (e.g. a dataset name) becomes the filename prefix (runner behaviour)."""
        import os
        import re

        mg = _mg(epochs=3)
        mg.fit(_linsep(), task="classification", n_out=2)
        path = mg.save(dirpath=str(tmp_path), stem="GunPoint")
        name = os.path.basename(path)
        assert re.fullmatch(r"GunPoint_\d{8}T\d{6}_\d+Z\.pt", name), name

    def test_default_dir_is_out_not_models(self):
        """T-MG-19e: the default save location is the package out/ folder, not models/ (avoids collision)."""
        import os

        d = AllGraph._default_model_dir()
        assert os.path.basename(d) == "out", d
        assert os.path.join("models", "saved") not in d

    def test_regression_round_trip(self, tmp_path):
        """T-MG-19c: a regression (operator) model round-trips and predicts fields on new data."""
        rng = np.random.RandomState(0)
        a = rng.randn(40, 16).astype(np.float32)
        y = np.stack([np.convolve(r, [0.25, 0.5, 0.25], mode="same") for r in a]).astype(np.float32)
        mg = _mg(epochs=4)
        mg.fit(AllData.functions(a, y), task="regression", n_out=1)
        anew = rng.randn(6, 16).astype(np.float32)
        ynew = np.stack([np.convolve(r, [0.25, 0.5, 0.25], mode="same") for r in anew]).astype(np.float32)
        before = mg.predict(AllData.functions(anew, ynew))
        loaded = AllGraph.load(mg.save(dirpath=str(tmp_path)), device="cpu")
        assert loaded._infer_task == "regression"
        with pytest.raises(ValueError):
            loaded.predict_proba(AllData.functions(anew, ynew))
        assert np.allclose(before, loaded.predict(AllData.functions(anew, ynew)), atol=1e-5)

    def test_save_before_fit_errors(self):
        """T-MG-20: saving an unfitted AllGraph is a clear error, not a corrupt file."""
        with pytest.raises(RuntimeError):
            _mg().save()

    def test_predict_before_fit_errors(self):
        """T-MG-20b: predicting with no trained net raises rather than crashing in the forward path."""
        with pytest.raises(RuntimeError):
            _mg().predict(_linsep())


# =========================================================================== Phase 7: Apple-Silicon device routing
class TestDeviceRoutingMPS:
    """The CPU-vs-MPS routing policy (ilmarinen.device.prefer_cpu_on_mps wired into fit()/load()).

    Measured CPU-vs-MPS at this package's per-contract budgets: the launch/scatter-bound contracts are
    FASTER on CPU -- relational graph/equivariant/set (~4x), sequence (~1.7x), 4d (~2.1x), volumetric
    (~1.4x) -- so they are pinned to CPU when the requested device is MPS; dense conv2d (spatial, MPS ~5x)
    and the matmul-dominated operator (MPS 1.2-5.2x across 1D/2D/3D) stay on the GPU. The routing is
    MPS-gated: on CUDA these ops are fast and must NEVER be forced to CPU.

    Portability: faking ``_base_device='mps'`` routes the launch-bound contracts to CPU BEFORE any MPS op
    runs, so those assertions hold on any machine; the 'stays on MPS' case is guarded on real hardware.
    """

    def test_routed_set_and_cuda_safety(self):
        """T-MG-23: the routed contract set is exact; CUDA/CPU are never forced to CPU; spatial/operator excluded."""
        from ilmarinen.device import MPS_CPU_FASTER_CONTRACTS, prefer_cpu_on_mps
        assert MPS_CPU_FASTER_CONTRACTS == {"graph", "equivariant", "set", "sequence", "volumetric", "4d"}
        for c in MPS_CPU_FASTER_CONTRACTS:
            assert prefer_cpu_on_mps(c, "mps") and prefer_cpu_on_mps(c, __import__("torch").device("mps"))
            assert not prefer_cpu_on_mps(c, "cuda")      # never force CPU on CUDA (Apple-Silicon phenomenon only)
            assert not prefer_cpu_on_mps(c, "cpu")
        for c in ("spatial", "operator"):
            assert not prefer_cpu_on_mps(c, "mps")

    @pytest.mark.smoke
    def test_fit_pins_launch_and_scatter_bound_to_cpu_on_mps(self):
        """T-MG-24: a sequence (dense, launch-bound) and a graph (relational, scatter-bound) fit both pin to
        CPU when the requested device is MPS -- covering both the post-routing dense pin and the pre-routing
        relational pin."""
        import torch
        for data_fn, expect in [(_linsep, "sequence"), (_graphs, "graph")]:
            mg = _mg()
            mg._base_device = torch.device("mps")        # simulate device='auto'/'mps' on Apple Silicon
            r = mg.fit(data_fn(), task="classification", n_out=2)
            assert r["contract"] == expect
            assert str(mg.device) == "cpu", (expect, str(mg.device))

    @pytest.mark.smoke
    def test_fit_dense_grid_contracts_pin_to_cpu_on_mps(self):
        """T-MG-24b: the launch-bound dense grid contracts volumetric and 4d pin to CPU on MPS (spatial does
        NOT -- covered by the real-hardware test below)."""
        import torch
        for shape, expect in [((24, 1, 6, 6, 6), "volumetric"), ((20, 1, 4, 4, 4, 4), "4d")]:
            mg = _mg()
            mg._base_device = torch.device("mps")
            r = mg.fit(_grid_data(shape), task="classification", n_out=2)
            assert r["contract"] == expect and str(mg.device) == "cpu", (expect, str(mg.device))

    @pytest.mark.smoke
    def test_load_pins_relational_to_cpu_when_mps_requested(self, tmp_path):
        """T-MG-25: a reloaded relational model never lands on MPS. On MPS hardware the load() guard forces
        CPU; on non-MPS resolve_device('mps') already falls back to CPU -- either way predict() runs on CPU
        (the fast, crash-free path), mirroring fit()."""
        mg = _mg()
        mg.fit(_graphs(), task="classification", n_out=2)
        path = mg.save(dirpath=str(tmp_path))
        loaded = AllGraph.load(path, device="mps")
        assert str(loaded.device) == "cpu"

    @pytest.mark.smoke
    def test_spatial_stays_on_mps_on_real_hardware(self):
        """T-MG-26: on genuine Apple-Silicon MPS, the dense-conv spatial contract is NOT routed to CPU (it is
        ~5x faster on the GPU). Skipped where MPS is unavailable."""
        from ilmarinen.device import mps_available
        if not mps_available():
            pytest.skip("requires Apple-Silicon MPS")
        mg = AllGraph(width=8, depth=1, epochs=2, device="auto", verbose=False, seed=0, contract_router=None)
        r = mg.fit(_grid_data((30, 1, 8, 8)), task="classification", n_out=2)
        assert r["contract"] == "spatial" and str(mg.device).startswith("mps")

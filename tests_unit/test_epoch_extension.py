"""T-EX: epoch EXTENSION, best-weights restore, and the convergence telemetry they report.

Before this, ``self.epochs`` was a hard ceiling: ``--auto_epoch`` could only shorten a deployed fit
(``_EarlyStopper.step`` -> ``break``), never lengthen it, and nothing recorded whether a model stopped
because it converged or because it ran out of budget. These lock the three pieces that changed:

  Phase 1  _EarlyStopper state      -- `fired`, and the raw-argmin channel used for checkpointing
  Phase 2  _deploy_epoch_blocks     -- the budget -> block generator (the extension policy itself)
  Phase 3  telemetry + restore-best -- what a fit reports, and that the best weights come back
  Phase 4  results store            -- the runner's merge-on-write JSON (batch accumulation)

The default path must stay byte-identical: with none of the new flags set, a fit trains exactly
``self.epochs`` epochs, which `test_default_path_unchanged` pins. Fit-based tests are marked ``smoke``.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

from ilmarinen import AllData, AllGraph
from ilmarinen.core.allgraph import _EarlyStopper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation_runners"))


def _mg(**kw):
    kw.setdefault("width", 8)
    kw.setdefault("depth", 1)
    kw.setdefault("epochs", 5)
    kw.setdefault("verbose", False)
    kw.setdefault("seed", 0)
    return AllGraph(**kw)


def _seq(n=400, t=16, c=2, seed=0):
    """A separable sequence task big enough (n >= ~334) for auto_epoch='val' to keep a real monitor."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, t, c).astype("float32")
    y = (X[:, :, 0].sum(1) > 0).astype("int64")
    return AllData.dense_tensor(X, y)


class TestEarlyStopperState:
    """Phase 1: the stopper exposes what extension and checkpointing need (fast tier)."""

    def test_fired_mirrors_return(self):
        """T-EX-1: `fired` records the plateau verdict so a caller can ask after the loop ends."""
        s = _EarlyStopper(min_delta=0.01, patience=2, min_epochs=2)
        assert s.fired is False
        for m in (1.0, 0.5, 0.4):  # still improving -> not fired
            assert s.step(m) is False and s.fired is False
        s.step(0.3999)  # flat
        assert s.step(0.3998) is True and s.fired is True  # two flat epochs -> patience exhausted

    def test_raw_argmin_is_separate_from_gated_best(self):
        """T-EX-2: `best_raw` tracks the plain argmin; `best` stays min_delta-gated.

        Checkpointing needs the true minimum -- an epoch that improves by less than min_delta is still the
        best weights seen, even though it must NOT reset the plateau counter."""
        s = _EarlyStopper(min_delta=0.5, patience=99, min_epochs=99)  # 50% gate: tiny gains don't count
        s.step(1.0)
        s.step(0.99)  # a 1% gain: below the gate, but a genuine new minimum
        assert s.improved_raw is True
        assert s.best_raw == pytest.approx(0.99)
        assert s.best_epoch == 2
        assert s.best == pytest.approx(1.0)  # the GATED best did not move
        assert s.bad == 1  # ...and the epoch counted as non-improving

    def test_improved_raw_false_when_worse(self):
        """T-EX-3: a worse epoch is not a checkpoint candidate."""
        s = _EarlyStopper()
        s.step(1.0)
        s.step(2.0)
        assert s.improved_raw is False and s.best_epoch == 1


class TestMetricCriterion:
    """The held-out SCORE criterion: stop when validation accuracy/R2 stops moving (fast tier).

    Monitoring val LOSS is a poor proxy for this on classification -- it rises with model confidence while
    accuracy is still improving, which this package documents on ItalyPowerDemand. 'max' mode watches the
    score directly, with an ABSOLUTE min_delta in the score's own units."""

    def _s(self, **kw):
        kw.setdefault("min_delta", 0.005)  # half an accuracy point
        kw.setdefault("patience", 3)
        kw.setdefault("min_epochs", 2)
        return _EarlyStopper(mode="max", absolute=True, **kw)

    def _fires_at(self, accs, **kw):
        s, at = self._s(**kw), None
        for i, a in enumerate(accs, 1):
            if s.step(a) and at is None:
                at = i
        return at

    def test_fires_once_the_score_stops_moving(self):
        """T-EX-40: a score flat within min_delta for `patience` epochs is a plateau."""
        assert self._fires_at([0.40, 0.55, 0.70, 0.701, 0.702, 0.703, 0.704]) == 6

    def test_sub_threshold_drift_still_counts_as_flat(self):
        """T-EX-41: improvements SMALLER than min_delta do not reset the patience counter -- that is what
        makes the criterion 'stops changing by more than 0.5%' rather than 'stops changing at all'."""
        assert self._fires_at([0.40, 0.55, 0.70, 0.7005, 0.7010, 0.7015]) == 6

    def test_a_still_improving_model_is_never_stopped(self):
        """T-EX-42: steady gains above min_delta keep training alive."""
        assert self._fires_at([0.5, 0.52, 0.54, 0.56, 0.58, 0.60]) is None

    def test_absolute_not_relative_delta(self):
        """T-EX-43: accuracy and R2 are already on a fixed 0-1 scale, so the threshold is absolute -- a
        relative one would mean something different at 0.4 than at 0.95."""
        s = self._s(min_delta=0.005)
        s.step(0.90)
        s.step(0.906)  # +0.6pp absolute: counts
        assert s.bad == 0
        s.step(0.909)  # +0.3pp: does not
        assert s.bad == 1

    def test_warmup_guard_allows_a_high_starting_score(self):
        """T-EX-44: the guard must not be an absolute offset above the opening score -- a task starting at
        0.95 could never clear 0.95+warmup and would train forever."""
        assert self._fires_at([0.95, 0.96, 0.961, 0.962, 0.963]) == 5

    def test_warmup_guard_never_stops_a_model_that_never_improved(self):
        """T-EX-45: a model pinned at its opening score has not begun to fit; it trains the full budget,
        the same safe fallback the loss path has."""
        assert self._fires_at([0.10] * 8) is None

    def test_min_mode_is_untouched(self):
        """T-EX-46: the default loss criterion keeps its relative-delta, lower-is-better semantics."""
        s = _EarlyStopper(min_delta=0.01, patience=2, min_epochs=2)
        assert s.mode == "min" and s.absolute is False
        assert [s.step(m) for m in (1.0, 0.5, 0.4, 0.3999, 0.3998)] == [False, False, False, False, True]

    def test_rejects_an_unknown_mode(self):
        """T-EX-47: a typo'd mode fails loudly rather than silently monitoring the wrong direction."""
        with pytest.raises(ValueError):
            _EarlyStopper(mode="maximise")

    def test_criterion_requires_a_val_split(self):
        """T-EX-48: 'metric' needs auto_epoch='val' -- there is no held-out score to read under 'train',
        so it degrades to the loss criterion rather than silently scoring on training data."""
        assert _mg(auto_epoch="val", auto_epoch_criterion="metric")._use_metric_criterion() is True
        assert _mg(auto_epoch="train", auto_epoch_criterion="metric")._use_metric_criterion() is False
        assert _mg(auto_epoch="val", auto_epoch_criterion="loss")._use_metric_criterion() is False

    def test_stopper_is_built_in_max_mode_for_the_metric_criterion(self):
        """T-EX-49: the criterion selects the stopper's direction and delta semantics."""
        s = _mg(auto_epoch="val", auto_epoch_criterion="metric")._make_stopper()
        assert s.mode == "max" and s.absolute is True
        s = _mg(auto_epoch="val")._make_stopper()
        assert s.mode == "min" and s.absolute is False

    def test_rejects_an_unknown_criterion(self):
        with pytest.raises(ValueError):
            _mg(auto_epoch_criterion="accuracy")


class TestMetricCriterionFit:
    """The metric criterion drives a real fit and reports the monitor it used (smoke)."""

    pytestmark = pytest.mark.smoke

    def test_fit_reports_the_val_metric_monitor(self):
        """T-EX-50: a fit stopped on the held-out score records `val_metric`, so the published protocol
        says what was actually monitored."""
        mg = _mg(
            epochs=10,
            auto_epoch="val",
            auto_epoch_criterion="metric",
            auto_epoch_min_delta=0.005,
            auto_epoch_patience=4,
            auto_epoch_min_epochs=4,
            auto_epoch_extend=10,
            auto_epoch_max=60,
        )
        r = mg.fit(_seq(n=600), task="classification")
        assert r["auto_epoch_monitor"] == "val_metric"
        assert r["epochs_trained"] >= 4

    def test_loss_criterion_still_reports_val(self):
        """T-EX-51: the default path is unchanged and still reports the loss monitor."""
        mg = _mg(epochs=8, auto_epoch="val", auto_epoch_patience=3, auto_epoch_min_epochs=3)
        assert mg.fit(_seq(n=600), task="classification")["auto_epoch_monitor"] == "val"


class TestDeployEpochBlocks:
    """Phase 2: the extension policy -- how a budget becomes one or more blocks (fast tier)."""

    def test_single_block_without_stopper(self):
        """T-EX-4: no stopper (auto_epoch off) -> exactly the old fixed range(self.epochs)."""
        assert list(_mg(epochs=100)._deploy_epoch_blocks(None)) == [(100, 0)]

    def test_single_block_when_extension_off(self):
        """T-EX-5: a stopper alone does not extend -- auto_epoch_extend must be set."""
        mg = _mg(epochs=100, auto_epoch="train")
        assert list(mg._deploy_epoch_blocks(_EarlyStopper())) == [(100, 0)]

    def test_extends_to_ceiling_while_unconverged(self):
        """T-EX-6: an unconverged stopper keeps drawing blocks until auto_epoch_max."""
        mg = _mg(epochs=100, auto_epoch="train", auto_epoch_extend=100, auto_epoch_max=400)
        blocks = list(mg._deploy_epoch_blocks(_EarlyStopper()))
        assert blocks == [(100, 0), (100, 100), (100, 200), (100, 300)]
        assert sum(c for c, _ in blocks) == 400  # never exceeds the ceiling

    def test_final_block_is_truncated_to_the_ceiling(self):
        """T-EX-7: a ceiling that is not a multiple of the block size yields a short last block."""
        mg = _mg(epochs=100, auto_epoch="train", auto_epoch_extend=100, auto_epoch_max=250)
        assert list(mg._deploy_epoch_blocks(_EarlyStopper())) == [(100, 0), (100, 100), (50, 200)]

    def test_no_extension_once_fired(self):
        """T-EX-8: a converged fit is not extended -- convergence, not budget, ended it."""
        s = _EarlyStopper()
        s.fired = True
        mg = _mg(epochs=100, auto_epoch="train", auto_epoch_extend=100, auto_epoch_max=1000)
        assert list(mg._deploy_epoch_blocks(s)) == [(100, 0)]

    def test_offsets_are_contiguous_and_absolute(self):
        """T-EX-9: offsets continue the epoch count across blocks (the streaming shuffle seeds on them,
        so a repeat would replay an epoch's batch order)."""
        mg = _mg(epochs=10, auto_epoch="train", auto_epoch_extend=10, auto_epoch_max=50)
        blocks = list(mg._deploy_epoch_blocks(_EarlyStopper()))
        assert [o for _, o in blocks] == [0, 10, 20, 30, 40]

    def test_epoch_cap_reports_the_ceiling(self):
        """T-EX-10: the reported cap is the extension ceiling when extending, else the plain budget."""
        assert _mg(epochs=100)._epoch_cap() == 100
        mg = _mg(epochs=100, auto_epoch="train", auto_epoch_extend=100, auto_epoch_max=1000)
        assert mg._epoch_cap() == 1000


class TestEpochTelemetry:
    """Phase 3: what a real fit reports about its own training (smoke)."""

    pytestmark = pytest.mark.smoke

    def test_default_path_unchanged(self):
        """T-EX-11: with no auto_epoch flags a fit trains exactly self.epochs and claims no convergence.

        This is the regression guard for every pre-existing run: extension must be strictly opt-in."""
        r = _mg(epochs=4).fit(_seq(), task="classification")
        assert r["epochs_trained"] == 4
        assert r["converged"] is None  # nothing measured it
        assert r["epoch_cap"] == 4

    def test_extension_runs_past_the_base_budget(self):
        """T-EX-12: a model that has not plateaued by the base budget keeps training."""
        mg = _mg(
            epochs=3,
            auto_epoch="train",
            auto_epoch_patience=50,  # never satisfied -> always unconverged
            auto_epoch_min_epochs=3,
            auto_epoch_extend=3,
            auto_epoch_max=15,
        )
        r = mg.fit(_seq(), task="classification")
        assert r["epochs_trained"] == 15  # ran to the ceiling, well past the base 3
        assert r["converged"] is False  # ...and says so: budget-limited, not converged
        assert r["epoch_cap"] == 15

    def test_converged_fit_stops_short_of_the_ceiling(self):
        """T-EX-13: when the plateau test fires, training stops and converged=True."""
        mg = _mg(
            epochs=5,
            auto_epoch="train",
            auto_epoch_patience=2,
            auto_epoch_min_epochs=2,
            auto_epoch_extend=5,
            auto_epoch_max=200,
        )
        r = mg.fit(_seq(), task="classification")
        assert r["converged"] is True
        assert r["epochs_trained"] < 200

    def test_monitor_records_val_when_data_allows(self):
        """T-EX-14: auto_epoch='val' on ample data reports that a held-out monitor was really used."""
        mg = _mg(epochs=4, auto_epoch="val", auto_epoch_min_epochs=2)
        assert mg.fit(_seq(n=400), task="classification")["auto_epoch_monitor"] == "val"

    def test_monitor_records_the_silent_train_fallback(self):
        """T-EX-15: on data too small for a reliable monitor, 'val' degrades to train-loss -- and the
        result says 'train', not 'val'. Without this the reported protocol would be wrong for every
        small dataset in the suite."""
        mg = _mg(epochs=4, auto_epoch="val", auto_epoch_min_epochs=2)
        r = mg.fit(_seq(n=60), task="classification")  # 60 << the ~334 needed to spare a val split
        assert r["auto_epoch_monitor"] == "train"

    def test_telemetry_resets_between_fits(self):
        """T-EX-16: a reused AllGraph does not report the previous fit's epoch count."""
        mg = _mg(epochs=3, auto_epoch="train", auto_epoch_patience=50, auto_epoch_extend=3, auto_epoch_max=12)
        first = mg.fit(_seq(), task="classification")
        assert first["epochs_trained"] == 12
        mg.auto_epoch = None  # second fit measures nothing
        assert mg.fit(_seq(), task="classification")["converged"] is None


class TestRestoreBest:
    """Phase 3b: best-weights restore (smoke)."""

    pytestmark = pytest.mark.smoke

    def test_restores_the_best_epoch_weights(self):
        """T-EX-17: with restore_best on, the deployed net is the best-monitored epoch's, not the last.

        Driven directly through _run_epochs with a scripted loss whose minimum is early and which then
        gets worse, so 'last' and 'best' are unambiguously different nets."""
        mg = _mg(epochs=6, auto_epoch="train", auto_epoch_restore_best=True)
        net = torch.nn.Linear(4, 1)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)  # frozen: only our marker moves the weights
        losses = iter([1.0, 0.1, 0.5, 0.9, 1.5, 2.0])  # minimum at epoch 2
        marks = {}

        def batch_loss(ids):
            return net.weight.sum() * 0.0 + float(next(losses))

        def permute(idx):
            with torch.no_grad():  # stamp the epoch index into the weights so we can identify the restore
                net.weight.fill_(len(marks))
            marks[len(marks)] = float(net.weight[0, 0])
            return np.arange(1)

        stopper = mg._make_stopper()
        mg._run_epochs(net, opt, np.arange(1), None, stopper, batch_loss, 1, permute, show_progress=True)
        # epoch 2 had the lowest loss (0.1); its weights were stamped with 1 (0-based epoch index)
        assert stopper.best_epoch == 2
        assert float(net.weight[0, 0]) == pytest.approx(1.0)

    def test_keeps_last_weights_when_disabled(self):
        """T-EX-18: default (restore off) keeps the final epoch's weights -- prior behaviour."""
        mg = _mg(epochs=6, auto_epoch="train", auto_epoch_restore_best=False)
        net = torch.nn.Linear(4, 1)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
        losses = iter([1.0, 0.1, 0.5, 0.9, 1.5, 2.0])
        seen = []

        def batch_loss(ids):
            return net.weight.sum() * 0.0 + float(next(losses))

        def permute(idx):
            with torch.no_grad():
                net.weight.fill_(len(seen))
            seen.append(1)
            return np.arange(1)

        mg._run_epochs(net, opt, np.arange(1), None, mg._make_stopper(), batch_loss, 1, permute, show_progress=True)
        assert float(net.weight[0, 0]) == pytest.approx(5.0)  # the LAST epoch's stamp


class TestResultsStore:
    """Phase 4: the runner's merge-on-write results document (fast tier)."""

    def _mod(self):
        import run_standard_validation as rsv

        return rsv

    def test_upsert_preserves_other_batches(self, tmp_path):
        """T-EX-19: a second batch's write keeps the first batch's rows.

        This is the property the whole batching workflow rests on -- the suite is run as several
        --contracts invocations and they must accumulate into one table."""
        rsv = self._mod()
        path = str(tmp_path / "rows.json")
        doc = rsv.load_results(path)  # absent file -> empty document
        rsv.record_row(doc, path, "Burgers1D", {"name": "Burgers1D", "status": "ok", "value": 0.99})

        doc2 = rsv.load_results(path)  # a fresh process, as a second batch would be
        rsv.record_row(doc2, path, "ESOL", {"name": "ESOL", "status": "ok", "value": 0.76})

        rows = rsv.load_results(path)["rows"]
        assert sorted(rows) == ["Burgers1D", "ESOL"]
        assert rows["Burgers1D"]["value"] == 0.99

    def test_rerunning_a_dataset_replaces_its_row(self, tmp_path):
        """T-EX-20: re-running one dataset overwrites just that row."""
        rsv = self._mod()
        path = str(tmp_path / "rows.json")
        doc = rsv.load_results(path)
        rsv.record_row(doc, path, "ESOL", {"name": "ESOL", "status": "ok", "value": 0.70})
        doc = rsv.load_results(path)
        rsv.record_row(doc, path, "ESOL", {"name": "ESOL", "status": "ok", "value": 0.76})
        rows = rsv.load_results(path)["rows"]
        assert len(rows) == 1 and rows["ESOL"]["value"] == 0.76

    def test_rows_carry_a_timestamp(self, tmp_path):
        """T-EX-21: every recorded row is stamped, so a stale batch is identifiable."""
        rsv = self._mod()
        path = str(tmp_path / "rows.json")
        doc = rsv.load_results(path)
        rsv.record_row(doc, path, "ESOL", {"name": "ESOL", "status": "ok"})
        assert rsv.load_results(path)["rows"]["ESOL"]["timestamp"].endswith("Z")

    def test_corrupt_file_does_not_lose_the_run(self, tmp_path):
        """T-EX-22: an unreadable results file yields a fresh document rather than raising mid-suite."""
        rsv = self._mod()
        path = str(tmp_path / "rows.json")
        with open(path, "w") as fh:
            fh.write("{not json")
        assert rsv.load_results(path) == {"meta": {"runs": []}, "rows": {}}

    def test_stub_row_records_a_skip(self, tmp_path):
        """T-EX-23: datasets that never produced a number are recorded, not silently dropped."""
        rsv = self._mod()
        row = rsv._stub_row("QM9", "equivariant", "skip", "missing package: no module named x")
        assert row["status"] == "skip" and row["name"] == "QM9" and row["contract"] is None


class TestEmlpPicklability:
    """The discovered-group EMLP layer must survive torch.save/load (fast tier).

    Regression for a real loss: JetMassLorentz routed to the generated-equivariant contract, trained to
    R2=1.0 over ~4.8 h, and then --save_models raised `AttributeError: Can't get local object
    'EquivariantLinear.torch_module.<locals>._EqLin'` -- the class was defined inside the method, so pickle
    could not name it."""

    def test_layer_class_is_module_level(self):
        """T-EX-32: the layer's class resolves to a real module path, not a <locals> qualname."""
        from ilmarinen.core.emlp_layer import EquivariantLinear

        g = [np.array([[0.0, -1.0], [1.0, 0.0]])]
        m = EquivariantLinear(g, g).torch_module()
        assert type(m).__qualname__ == "_EqLin"
        assert type(m).__module__ == "ilmarinen.core.emlp_layer"
        assert "<locals>" not in type(m).__qualname__

    def test_save_load_round_trip_is_identical(self, tmp_path):
        """T-EX-33: a saved layer reloads and computes the same function."""
        from ilmarinen.core.emlp_layer import EquivariantLinear

        g = [np.array([[0.0, -1.0], [1.0, 0.0]])]
        m = EquivariantLinear(g, g).torch_module()
        p = str(tmp_path / "eqlin.pt")
        torch.save(m, p)
        m2 = torch.load(p, weights_only=False)
        x = torch.randn(4, 2)
        assert torch.allclose(m(x), m2(x))

    def test_class_is_resolvable_by_attribute_lookup(self):
        """T-EX-34: the PEP-562 hook exposes _EqLin without anyone having built a layer first -- this is
        what torch.load needs in a cold process that never called torch_module()."""
        import ilmarinen.core.emlp_layer as el

        assert el._EqLin is not None
        with pytest.raises(AttributeError):
            getattr(el, "definitely_not_a_real_attribute")  # noqa: B009 - the lookup IS the assertion

    def test_class_is_cached(self):
        """T-EX-35: repeated builds reuse ONE class object; two distinct classes would make an old
        checkpoint unloadable against a new one."""
        from ilmarinen.core.emlp_layer import EquivariantLinear

        g = [np.array([[0.0, -1.0], [1.0, 0.0]])]
        a = type(EquivariantLinear(g, g).torch_module())
        b = type(EquivariantLinear(g, g).torch_module())
        assert a is b


class TestTableRendering:
    """Phase 4b: the markdown generator (fast tier)."""

    def _mod(self):
        import make_results_table as mrt

        return mrt

    def test_sota_trim_respects_parentheses(self):
        """T-EX-24: several registry SOTA strings carry a ';' INSIDE their parenthetical; splitting on the
        first ';' regardless of depth would emit a severed, unbalanced cell."""
        mrt = self._mod()
        got = mrt.trim_sota("R2 n/a (synthetic-from-solver; -> 1 for a sufficient 4d model)")
        assert got == "n/a (synthetic-from-solver; -> 1 for a sufficient 4d model)"
        assert got.count("(") == got.count(")")

    def test_sota_trim_drops_the_redundant_metric_label(self):
        """T-EX-25: the metric name is already its own column."""
        mrt = self._mod()
        assert mrt.trim_sota("acc ~0.99+ (any CNN)") == "~0.99+ (any CNN)"
        assert mrt.trim_sota("MAE ~0.40-0.45 log mol/L (D-MPNN, random split); R2 ~0.90-0.93") == (
            "~0.40-0.45 log mol/L (D-MPNN, random split)"
        )

    def test_sota_trim_handles_missing(self):
        """T-EX-26: a dataset with no SOTA string still renders a cell."""
        assert self._mod().trim_sota(None) == "n/a"
        assert self._mod().trim_sota("") == "n/a"

    def test_render_groups_by_contract_and_flags_unconverged(self):
        """T-EX-27: rows group under their routed contract, and a budget-limited row is daggered."""
        mrt = self._mod()
        doc = {
            "rows": {
                "ESOL": {
                    "name": "ESOL",
                    "status": "ok",
                    "contract": "graph",
                    "metric": "R2",
                    "value": 0.76,
                    "skill": 0.76,
                    "arch": "pna→gin",
                    "params": 126790,
                    "sota": "R2 ~0.90-0.93 (D-MPNN)",
                    "epochs_trained": 140,
                    "converged": True,
                    "extra": {},
                },
                "Burgers1D": {
                    "name": "Burgers1D",
                    "status": "ok",
                    "contract": "operator",
                    "metric": "field_R2",
                    "value": 0.995,
                    "skill": 0.995,
                    "arch": "deeponet",
                    "params": 195909,
                    "sota": "field R2 ~0.999 (FNO)",
                    "epochs_trained": 1000,
                    "converged": False,
                    "extra": {},
                },
            }
        }
        md = mrt.render(doc)
        assert md.index("**`graph`**") < md.index("**`operator`**")  # canonical contract order
        assert "| 140 |" in md and "| 1000† |" in md
        assert "budget-limited" in md  # the dagger is explained

    def test_caveated_dataset_is_footnoted(self):
        """T-EX-36: a dataset whose SOTA string measures a DIFFERENT quantity than the loader's target is
        marked and explained, so the table cannot be read as a like-for-like comparison it isn't.

        QM9 is the live case: the loader regresses raw U0 total energy, the literature figure is for
        atomization energy, and the two differ by orders of magnitude."""
        mrt = self._mod()
        doc = {
            "rows": {
                "QM9": {
                    "name": "QM9",
                    "status": "ok",
                    "contract": "equivariant",
                    "metric": "MAE[meV]",
                    "value": 101331.5,
                    "skill": 0.98,
                    "arch": "e_norm",
                    "params": 24985,
                    "sota": "U0 MAE ~5-15 meV (SchNet)",
                    "epochs_trained": 51,
                    "converged": True,
                    "extra": {"R2": 0.9835},
                }
            }
        }
        md = mrt.render(doc)
        assert "QM9 ‡" in md
        assert "‡ **QM9**" in md and "atomization" in md.lower()

    def test_uncaveated_dataset_has_no_footnote(self):
        """T-EX-37: the marker appears only where a caveat is actually registered."""
        mrt = self._mod()
        doc = {
            "rows": {
                "MNIST": {
                    "name": "MNIST",
                    "status": "ok",
                    "contract": "spatial",
                    "metric": "acc",
                    "value": 0.99,
                    "skill": 0.99,
                    "arch": "atrous",
                    "params": 100,
                    "sota": "acc ~0.99+ (any CNN)",
                    "epochs_trained": 21,
                    "converged": True,
                    "extra": {},
                }
            }
        }
        assert "‡" not in mrt.render(doc)

    def test_arch_names_the_group_for_discovered_contracts(self):
        """T-EX-38: a discovered-group contract has no alpha cells, so the per-layer readout is empty and
        the runner records '?'. The table names the GROUP instead -- on that path the group is the
        architecture, and a bare '?' would read as missing data."""
        mrt = self._mod()
        base = {"contract": "generated_equivariant", "arch": "?"}
        assert mrt.fmt_arch({**base, "group": "O(1,3)", "group_generators": 6}) == "EMLP `O(1,3)` (6 generators)"
        assert "6 generators" in mrt.fmt_arch({**base, "group_generators": 6})
        assert mrt.fmt_arch(base) == "EMLP (discovered group)"

    def test_arch_passes_through_normal_contracts(self):
        """T-EX-39: an ordinary per-layer architecture is rendered verbatim in code ticks."""
        mrt = self._mod()
        assert mrt.fmt_arch({"contract": "graph", "arch": "gat→gcn"}) == "`gat→gcn`"
        assert mrt.fmt_arch({"contract": "graph", "arch": "?"}) == "`?`"

    def test_render_lists_failed_datasets(self):
        """T-EX-28: skipped/errored datasets appear as a note so the table shows its own gaps."""
        mrt = self._mod()
        doc = {
            "rows": {
                "QM9": {"name": "QM9", "status": "skip", "expected_contract": "equivariant", "note": "data not found"}
            }
        }
        md = mrt.render(doc)
        assert "Not measured" in md and "QM9" in md

    def test_insert_readme_requires_markers(self, tmp_path):
        """T-EX-29: splicing refuses a file without the marker pair rather than appending blindly."""
        mrt = self._mod()
        p = tmp_path / "README.md"
        p.write_text("# no markers here\n")
        with pytest.raises(SystemExit):
            mrt.insert_readme(str(p), "table")

    def test_insert_readme_replaces_only_the_marked_region(self, tmp_path):
        """T-EX-30: prose either side of the markers survives regeneration."""
        mrt = self._mod()
        p = tmp_path / "README.md"
        p.write_text(f"before\n\n{mrt.BEGIN}\nOLD\n{mrt.END}\n\nafter\n")
        mrt.insert_readme(str(p), "NEW TABLE\n")
        got = p.read_text()
        assert "before" in got and "after" in got
        assert "OLD" not in got and "NEW TABLE" in got


def test_results_json_is_serialisable(tmp_path):
    """T-EX-31: numpy scalars (skill/value come from numpy) must not break the JSON write."""
    import run_standard_validation as rsv

    path = str(tmp_path / "rows.json")
    doc = rsv.load_results(path)
    rsv.record_row(doc, path, "X", {"name": "X", "status": "ok", "value": np.float32(0.5), "skill": np.float64(0.4)})
    with open(path) as fh:
        assert json.load(fh)["rows"]["X"]["value"] == pytest.approx(0.5)

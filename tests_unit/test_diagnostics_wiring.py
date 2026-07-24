"""T-DX: AllGraph diagnostic-report WIRING.

AllGraph exposes opt-in observables (LLC, developmental LLC, thermodynamic potential, response
spectroscopy, effective-dimension ledger) through constructor flags. Each flag gates one key that
``_attach_diagnostics`` adds to the fit-result dict after training (allgraph_reports.py). The underlying
COMPUTATIONS have their own unit suites (test_complexity_llc, test_developmental_llc,
test_thermodynamic_potential, test_response_spectroscopy, test_effective_dimension_ledger); what was
untested is AllGraph's WIRING -- that ``report_x=True`` actually lands ``result["x"]`` and that the keys
stay absent by default. A refactor that silently stops gating a report would otherwise pass unnoticed
(the reports module ran at ~9%).

The cheap reports (thermo, ledger, gibbs-response) are asserted through a REAL fit. The two expensive
ones -- LLC (~20s of SGLD) and developmental LLC (~200s of retraining) -- are asserted at the wiring
level by stubbing their report method to a sentinel, so this file stays fast; their real output is
locked by the dedicated suites above. Fit-based, so ``smoke``.
"""

import numpy as np
import pytest

from ilmarinen import AllData, AllGraph

# every diagnostic key _attach_diagnostics can add, keyed by the flag that gates it
_DIAG_KEYS = (
    "llc",
    "developmental_llc",
    "thermodynamic_potential",
    "response_spectroscopy",
    "effective_dimension_ledger",
)


def _tabular(n=80, d=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(np.int64)
    return AllData.dense_tensor(X, y)


def _fit(**flags):
    select = flags.pop("select", "argmax")
    mg = AllGraph(width=8, depth=1, epochs=3, device="cpu", verbose=False, seed=0, **flags)
    return mg.fit(_tabular(), task="classification", n_out=2, select=select)


class TestDiagnosticsWiring:
    """report_* flags attach their result keys; the default result carries none of them."""

    pytestmark = pytest.mark.smoke

    def test_thermo_flag_attaches_key(self):
        """T-DX-1: report_thermo=True lands result['thermodynamic_potential'] (real fit)."""
        res = _fit(report_thermo=True)
        assert isinstance(res.get("thermodynamic_potential"), dict)

    def test_ledger_flag_attaches_key(self):
        """T-DX-2: report_ledger=True lands result['effective_dimension_ledger'] (real fit)."""
        res = _fit(report_ledger=True)
        assert isinstance(res.get("effective_dimension_ledger"), dict)

    def test_response_flag_attaches_key_under_gibbs(self):
        """T-DX-3: report_response=True lands result['response_spectroscopy'] when a gibbs readout ran (the
        response channel reads the gibbs energies; under the default argmax select there is nothing
        perturbable, so gibbs is the meaningful configuration)."""
        res = _fit(report_response=True, select="gibbs")
        assert isinstance(res.get("response_spectroscopy"), dict)

    def test_llc_flag_wires_report_into_result(self, monkeypatch):
        """T-DX-4: report_llc=True routes _llc_report's output into result['llc']. Stubbed to keep the test
        fast (the real SGLD LLC is ~20s and is covered by test_complexity_llc); this locks the WIRING."""
        sentinel = {"lambda": 0.5, "_stub": True}
        monkeypatch.setattr(AllGraph, "_llc_report", lambda self, *a, **k: sentinel)
        res = _fit(report_llc=True)
        assert res.get("llc") == sentinel

    def test_developmental_flag_wires_report_into_result(self, monkeypatch):
        """T-DX-5: developmental_llc=True routes _developmental_report into result['developmental_llc'].
        Stubbed (the real retrained trajectory is ~200s, covered by test_developmental_llc)."""
        sentinel = {"final": 0.1, "_stub": True}
        monkeypatch.setattr(AllGraph, "_developmental_report", lambda self, *a, **k: sentinel)
        res = _fit(developmental_llc=True)
        assert res.get("developmental_llc") == sentinel

    def test_no_diagnostics_by_default(self):
        """T-DX-6: a fit with no diagnostic flags carries NONE of the report keys -- the reports are strictly
        opt-in, so an accidental always-on report (or a lost gate) is caught here."""
        res = _fit()
        present = [k for k in _DIAG_KEYS if k in res]
        assert present == [], f"unexpected diagnostic keys with no flags set: {present}"


def _equivariant_regression(n=40, m=5, seed=0):
    """Small geometric regression: point clouds with positions + a ring edge set + a rotation-invariant
    target (|P|^2), so AllGraph routes to the strict-equivariant contract."""
    rng = np.random.RandomState(seed)
    nf, ed, po, y = [], [], [], []
    for _ in range(n):
        P = rng.randn(m, 3).astype(np.float32)
        nf.append(np.ones((m, 1), np.float32))
        ed.append(np.array([(i, (i + 1) % m) for i in range(m)], np.int64).T)
        po.append(P)
        y.append(float((P**2).sum()))
    y = np.array(y, np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return AllData.graphs(nf, ed, y=y, positions=po)


class TestDiagnosticReportBodies:
    """The REPORT COMPUTATIONS behind the flags (test_diagnostics_wiring above locks only the wiring).
    These call the report methods with reduced MCMC/trajectory budgets so the SGLD-heavy read-outs (LLC
    ~20s, developmental ~200s at default budget) run in a few seconds; correctness of the estimator at
    scale is a machinery concern (test_complexity_llc / test_developmental_llc). Fit-based -> smoke."""

    pytestmark = pytest.mark.smoke

    @pytest.fixture(scope="class")
    def dense_fit(self):
        # short developmental trajectory/checkpoints so _developmental_report is fast; harmless to _llc_report
        mg = AllGraph(
            width=8,
            depth=1,
            epochs=5,
            device="cpu",
            verbose=False,
            seed=0,
            developmental_llc_epochs=8,
            developmental_llc_checkpoints=[2, 4, 8],
        )
        data = _tabular()
        mg.fit(data, task="classification", n_out=2)
        return mg, data

    def test_llc_report_produces_estimator(self, dense_fit):
        """T-DX-7: _llc_report returns the SGLD LLC estimator dict (lambda, its spread, and the k/2 bound)
        with finite values. Reduced SGLD budget for speed; lambda may be flagged invalid (negative) at a
        short epoch budget -- we assert structure and finiteness, not the sign."""
        mg, data = dense_fit
        r = mg._llc_report(mg.net, data, "classification", chains=2, steps=25, burn=8)
        assert r is not None
        for k in ("lambda", "lambda_std", "half_params", "valid"):
            assert k in r, f"missing LLC key {k!r}"
        assert np.isfinite(r["lambda"]) and r["half_params"] > 0

    def test_developmental_report_produces_curve(self, dense_fit):
        """T-DX-8: _developmental_report retrains one trajectory and returns lambda(t) over checkpoints plus
        the located transitions. Reduced trajectory + SGLD for speed."""
        mg, data = dense_fit
        dev = mg._developmental_report(data, "classification", chains=2, steps=15, burn=6)
        assert dev is not None
        for k in ("curve", "final", "checkpoints"):
            assert k in dev, f"missing developmental key {k!r}"
        assert len(dev["checkpoints"]) >= 1

    def test_equivariance_breaking_probe_wires_and_reports(self):
        """T-DX-9: price_equivariance on a strict-equivariant regression fit runs the (cheap) breaking probe
        and lands result['equivariance_breaking'] with its full report -- covers both the probe body and the
        price_equivariance branch of _attach_diagnostics (the equivariant-only wiring the dense tests miss)."""
        mg = AllGraph(width=8, depth=1, epochs=5, device="cpu", verbose=False, seed=0, price_equivariance=True)
        res = mg.fit(_equivariant_regression(), task="regression", n_out=1)
        assert mg.contract == "equivariant"
        probe = res.get("equivariance_breaking")
        assert isinstance(probe, dict)
        for k in ("breaking_signal", "symmetry_broken", "resid_explained_invariant", "resid_explained_noninvariant"):
            assert k in probe, f"missing probe key {k!r}"
        assert isinstance(probe["symmetry_broken"], bool)

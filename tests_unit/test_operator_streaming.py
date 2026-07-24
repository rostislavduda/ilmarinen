"""T-OSTREAM: opt-in dataset streaming for the neural-operator contract (function -> function).

The operator contract is the hard streaming case: unlike every other contract, the TARGET u is field-valued
and large, so the resident field-R2 (one global mean over the whole target, whole prediction held in memory)
cannot be reused. Streaming computes the field-R2 in a two-pass accumulation (global field mean, then
residual/total sums of squares). Training is bit-for-bit equivalent to the resident fit; the streamed R2
matches the resident R2 to fp tolerance.
"""

import numpy as np
import pytest
import torch

from ilmarinen import AllData, AllGraph, InMemoryOperatorSource, MemmapOperatorSource, OperatorSource
from ilmarinen.core.allgraph_streaming import _default_operator_grid, _infer_operator_sdims

_SMOOTH = np.array([0.25, 0.5, 0.25], np.float32)


def _weights_identical(net_a, net_b):
    sa, sb = net_a.state_dict(), net_b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _op_1d(n=40, grid=16, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n, grid).astype(np.float32)
    u = np.stack([np.convolve(r, _SMOOTH, mode="same") for r in a]).astype(np.float32)
    return a, u


def _op_2d(n=30, hw=8, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n, hw, hw).astype(np.float32)
    u = np.stack([np.stack([np.convolve(row, _SMOOTH, mode="same") for row in img]) for img in a]).astype(np.float32)
    return a, u


# --------------------------------------------------------------------------- equivalence
@pytest.mark.parametrize("maker", [_op_1d, _op_2d])
def test_operator_streaming_bit_identical(maker):
    """T-OSTREAM-1: 1D and 2D operator fits -> bit-identical weights; streamed field-R2 matches resident."""
    a, u = maker()
    mg_r = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_r = mg_r.fit(AllData.functions(a, u), task="regression", n_out=1)
    mg_s = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_s = mg_s.fit(AllData.functions_stream(InMemoryOperatorSource(a, u)), task="regression", n_out=1)
    assert r_r["contract"] == r_s["contract"] == "operator"
    assert r_r["metric"] == r_s["metric"] == "field_R2"
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-5)


def test_operator_streaming_auto_epoch_val():
    """T-OSTREAM-2: auto_epoch='val' (held-out monitor + early stop) stays bit-identical under streaming."""
    a, u = _op_1d(n=400)
    mg_r = AllGraph(width=8, depth=1, epochs=25, verbose=False, seed=0, auto_epoch="val")
    r_r = mg_r.fit(AllData.functions(a, u), task="regression", n_out=1)
    mg_s = AllGraph(width=8, depth=1, epochs=25, verbose=False, seed=0, auto_epoch="val")
    r_s = mg_s.fit(AllData.functions_stream(InMemoryOperatorSource(a, u)), task="regression", n_out=1)
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-5)


def test_operator_memmap_source_equivalent(tmp_path):
    """T-OSTREAM-3: a MemmapOperatorSource over real .npy field files is bit-identical to the resident fit."""
    a, u = _op_1d(n=48)
    pa, pu = tmp_path / "a.npy", tmp_path / "u.npy"
    np.save(pa, a)
    np.save(pu, u)
    mg_r = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_r = mg_r.fit(AllData.functions(a, u), task="regression", n_out=1)
    mg_s = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_s = mg_s.fit(AllData.functions_stream(MemmapOperatorSource(str(pa), str(pu))), task="regression", n_out=1)
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-5)


def test_operator_explicit_grid_equivalent():
    """T-OSTREAM-4: an explicit per-sample grid streams bit-identically (source indexes the same grid rows)."""
    a, u = _op_1d(n=40, grid=16)
    x = np.linspace(0, 1, 16, dtype=np.float32)[None, :, None].repeat(40, 0)  # (n, grid, sdims=1)
    mg_r = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    r_r = mg_r.fit(AllData.functions(a, u, grid=x), task="regression", n_out=1)
    mg_s = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    r_s = mg_s.fit(AllData.functions_stream(InMemoryOperatorSource(a, u, grid=x)), task="regression", n_out=1)
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-5)


def test_operator_streaming_repeatability_and_data_sensitivity():
    """T-OSTREAM-5: reproducible over the same source; different fields -> different weights."""
    a, u = _op_1d(seed=0)
    a2, u2 = _op_1d(seed=9)
    x = AllData.functions_stream
    A = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    B = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    C = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    A.fit(x(InMemoryOperatorSource(a, u)), task="regression", n_out=1)
    B.fit(x(InMemoryOperatorSource(a, u)), task="regression", n_out=1)
    C.fit(x(InMemoryOperatorSource(a2, u2)), task="regression", n_out=1)
    assert _weights_identical(A.net, B.net)
    assert not _weights_identical(A.net, C.net)


# --------------------------------------------------------------------------- streamed field-R2 correctness
def test_streamed_field_r2_matches_resident_formula():
    """T-OSTREAM-6: the two-pass streamed field-R2 (via _stream_operator_eval) equals the resident whole-array
    field-R2 formula on the same predictions, to fp tolerance -- verified directly against a fixed net output."""
    a, u = _op_1d(n=32, grid=16)
    mg = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0)
    mg.fit(AllData.functions_stream(InMemoryOperatorSource(a, u)), task="regression", n_out=1)
    src = InMemoryOperatorSource(a, u)
    # streamed field-R2
    streamed = mg._stream_operator_eval(mg.net, src, bs=7)
    # resident field-R2 over the whole dataset with the same (train-mode) net
    with torch.no_grad():
        pred = (
            mg.net(src.a(np.arange(len(src))).to(mg.device), src.grid(np.arange(len(src))).to(mg.device)).cpu().numpy()
        )
    uy = u
    resident = 1.0 - ((pred - uy) ** 2).sum() / (((uy - uy.mean()) ** 2).sum() + 1e-12)
    assert streamed == pytest.approx(float(resident), abs=1e-5)


# --------------------------------------------------------------------------- no full materialization
class _CountingOperatorSource(OperatorSource):
    """Counts field fetches and the largest single request; asserts no request exceeds a cap."""

    def __init__(self, a, u, row_cap):
        self._a = np.ascontiguousarray(a)
        self._u = np.ascontiguousarray(u)
        self.a_shape = self._a.shape
        self.spatial_dims = _infer_operator_sdims(self._a.shape, self._u.shape, None)
        self._setup_grid(None)
        self.row_cap = row_cap
        self.a_calls = self.u_calls = 0
        self.max_rows = 0

    def __len__(self):
        return int(self._a.shape[0])

    def _a_raw(self, ids):
        ids = np.asarray(ids)
        self.a_calls += 1
        self.max_rows = max(self.max_rows, len(ids))
        assert len(ids) <= self.row_cap, f"a() requested {len(ids)} > cap {self.row_cap}"
        return self._a[ids]

    def _u_raw(self, ids):
        ids = np.asarray(ids)
        self.u_calls += 1
        self.max_rows = max(self.max_rows, len(ids))
        assert len(ids) <= self.row_cap, f"u() requested {len(ids)} > cap {self.row_cap}"
        return self._u[ids]


def test_operator_no_full_materialization():
    """T-OSTREAM-7: training + the two-pass eval only ever fetch <= train_batch rows of fields per call (never
    the whole dataset). Eval is exactly two passes (mean, then ss); training is `epochs` passes."""
    a, u = _op_1d(n=200, grid=16)
    train_bs, epochs = 32, 2
    cs = _CountingOperatorSource(a, u, row_cap=train_bs)
    mg = AllGraph(width=8, depth=1, epochs=epochs, verbose=False, seed=0)  # auto_epoch off, train_batch=32
    mg.fit(AllData.functions_stream(cs), task="regression", n_out=1)
    n = len(a)
    per_pass = -(-n // train_bs)  # ceil
    assert cs.max_rows <= train_bs < n
    # training: epochs passes fetch a+grid+u (a_calls and u_calls each get `epochs*per_pass`); eval pass 1
    # fetches u only (+per_pass u_calls), eval pass 2 fetches a+u (+per_pass each).
    assert cs.a_calls == epochs * per_pass + per_pass  # train + eval pass 2
    assert cs.u_calls == epochs * per_pass + 2 * per_pass  # train + eval pass 1 + eval pass 2


# --------------------------------------------------------------------------- predict / inertness
def test_operator_predict_streamed_and_resident():
    """T-OSTREAM-8: predict on a streamed OR resident operator test set after a streaming train matches the
    resident-trained baseline (bit-identical weights)."""
    a, u = _op_1d(n=40)
    mg_r = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    mg_r.fit(AllData.functions(a, u), task="regression", n_out=1)
    mg_s = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    mg_s.fit(AllData.functions_stream(InMemoryOperatorSource(a, u)), task="regression", n_out=1)
    base = mg_r.predict(AllData.functions(a, u))
    ps_res = mg_s.predict(AllData.functions(a, u))
    ps_str = mg_s.predict(AllData.functions_stream(InMemoryOperatorSource(a, u)))
    assert np.allclose(base, ps_res, atol=1e-6)
    assert np.allclose(base, ps_str, atol=1e-6)


def test_operator_constant_target_r2_parity():
    """T-OSTREAM-12: on a (near-)constant target field, field-R2 is DEGENERATE (undefined -- ss_tot -> 0, so both
    paths report a large-negative garbage value). The invariant that matters is that streamed and resident do
    NOT diverge by orders of magnitude: both accumulate the field mean in float64 so ss_tot collapses to exactly
    0 and hits the 1e-12 guard identically on any backend (a float32 mean drifts under e.g. MKL, keeping the
    streamed ss_tot spuriously non-zero -> a ~100x divergence). The residual gap is only float32 ss_res
    summation order, so a loose relative tolerance is used (this is not a meaningful-R2 comparison)."""
    a = np.random.RandomState(0).randn(50, 40).astype(np.float32)
    u = np.full((50, 40), 3.14159, np.float32)
    r_r = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0).fit(
        AllData.functions(a, u), task="regression", n_out=1
    )["value"]
    r_s = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0).fit(
        AllData.functions_stream(InMemoryOperatorSource(a, u)), task="regression", n_out=1
    )["value"]
    assert abs(r_r - r_s) / max(abs(r_r), abs(r_s), 1e-12) < 1e-3  # was ~1.0 (100x divergence) before the fix


def test_operator_predict_empty_streamed_test_set():
    """T-OSTREAM-13: predict() on an EMPTY streamed operator test set returns a well-formed (0, *grid) tensor
    WITHOUT forwarding an empty batch through the net (the operator FFT rejects a 0-size batch on some backends
    such as MKL, so a forward -- resident or streamed -- would crash; we return the empty field directly)."""
    a, u = _op_1d(n=30, grid=16)
    mg = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0)
    mg.fit(AllData.functions(a, u), task="regression", n_out=1)
    empty = InMemoryOperatorSource(np.zeros((0, 16), np.float32), np.zeros((0, 16), np.float32))
    out = mg.predict(AllData.functions_stream(empty))
    assert out.shape[0] == 0 and tuple(out.shape[1:]) == (16,)  # (0, grid)


def test_operator_resident_not_streaming():
    """T-OSTREAM-9: resident operator data is never seen as streaming."""
    a, u = _op_1d(n=8)
    mg = AllGraph(verbose=False)
    assert mg._is_streaming_operator(AllData.functions(a, u)) is False
    assert mg._is_streaming_operator(AllData.functions_stream(InMemoryOperatorSource(a, u))) is True


# --------------------------------------------------------------------------- source metadata / guards
def test_operator_source_metadata():
    """T-OSTREAM-10: an OperatorSource exposes spatial_dims / a_shape / grid as metadata; the default grid
    matches AllData.functions' torch meshgrid, and a shared grid is broadcast per batch."""
    a1, u1 = _op_1d(n=10, grid=16)
    s1 = InMemoryOperatorSource(a1, u1)
    assert s1.spatial_dims == 1 and s1.a_shape == (10, 16)
    g = s1.grid(np.arange(3))
    assert g.shape == (3, 16, 1)
    assert torch.equal(g[0], _default_operator_grid((10, 16), 1))  # matches the resident default grid
    a2, u2 = _op_2d(n=6, hw=8)
    s2 = InMemoryOperatorSource(a2, u2)
    assert s2.spatial_dims == 2 and s2.grid(np.arange(2)).shape == (2, 8, 8, 2)


def test_operator_stream_constructor_and_fit_guards():
    """T-OSTREAM-11: functions_stream and fit guards for the operator source."""
    a, u = _op_1d(n=30)
    src = InMemoryOperatorSource(a, u)
    with pytest.raises(TypeError):
        AllData.functions_stream(a)  # not an OperatorSource
    with pytest.raises(ValueError):
        AllData.functions_stream(src, kind_hint="spatial")  # operator only
    with pytest.raises(ValueError):
        InMemoryOperatorSource(a, u[:10])  # mismatched sample counts
    data = AllData.functions_stream(src)
    with pytest.raises(NotImplementedError):  # price_modes re-reads the fields
        AllGraph(verbose=False, seed=0, price_modes=True).fit(data, task="regression", n_out=1)
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=2)
    with pytest.raises(ValueError):
        mg.fit(AllData.functions(a, u), task="regression", n_out=1, stream=True)  # resident, demanded stream
    r = mg.fit(data, task="regression", n_out=1, stream=True)
    assert r["contract"] == "operator"


# --------------------------------------------------------------------------- multi-channel input fields
def _vec_in_1d(n=40, grid=16, cin=3, seed=0):
    """A multi-channel INPUT field a=(n, N, cin) with a SCALAR target u=(n, N) (the operator maps a vector
    input field to a scalar output field; requires an explicit spatial_dims since a.ndim != u.ndim)."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n, grid, cin).astype(np.float32)
    u = np.stack([np.convolve(a[i, :, 0], _SMOOTH, mode="same") for i in range(n)]).astype(np.float32)
    return a, u


def _vec_in_2d(n=24, hw=8, cin=2, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n, hw, hw, cin).astype(np.float32)
    u = np.stack([np.stack([np.convolve(row, _SMOOTH, mode="same") for row in a[i, :, :, 0]]) for i in range(n)])
    return a, u.astype(np.float32)


@pytest.mark.parametrize("maker,sdims", [(_vec_in_1d, 1), (_vec_in_2d, 2)])
def test_operator_vector_input_streaming_bit_identical(maker, sdims):
    """T-OSTREAM-14: multi-channel INPUT field -> scalar output, streamed bit-identically (the source reports
    in_ch from a_shape[-1] and the explicit spatial_dims)."""
    a, u = maker()
    mg_r = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_r = mg_r.fit(AllData.functions(a, u, spatial_dims=sdims), task="regression", n_out=1)
    src = InMemoryOperatorSource(a, u, spatial_dims=sdims)
    assert src.spatial_dims == sdims and src.a_shape[-1] == a.shape[-1]
    mg_s = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_s = mg_s.fit(AllData.functions_stream(src), task="regression", n_out=1)
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-5)


def test_operator_vector_input_memmap_and_prefetch(tmp_path):
    """T-OSTREAM-15: multi-channel input over a memmap source, and prefetch-bit-identity, both hold."""
    a, u = _vec_in_1d()
    pa, pu = tmp_path / "a.npy", tmp_path / "u.npy"
    np.save(pa, a)
    np.save(pu, u)
    mg_r = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_r = mg_r.fit(AllData.functions(a, u, spatial_dims=1), task="regression", n_out=1)
    mg_m = AllGraph(width=8, depth=1, epochs=6, verbose=False, seed=0)
    r_m = mg_m.fit(
        AllData.functions_stream(MemmapOperatorSource(str(pa), str(pu), spatial_dims=1)), task="regression", n_out=1
    )
    assert _weights_identical(mg_r.net, mg_m.net) and r_r["value"] == pytest.approx(r_m["value"], abs=1e-5)

    def pf(depth):
        mg = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0, stream_prefetch=depth)
        mg.fit(AllData.functions_stream(InMemoryOperatorSource(a, u, spatial_dims=1)), task="regression", n_out=1)
        return mg.net.state_dict()

    s0 = pf(0)
    assert all(torch.equal(s0[k], pf(3)[k]) for k in s0)

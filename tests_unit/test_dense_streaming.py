"""T-STREAM: opt-in dataset streaming for the dense contracts (core/allgraph_streaming.py).

The load-bearing guarantee is EQUIVALENCE: a deployed dense fit driven from a DenseSource trains to the
bit-for-bit SAME weights as the equivalent resident fit over identical bytes (the only intended change is
dropping a device move on the shuffle index, which consumes no RNG). These tests lock that down, plus the
source-gather contract, the incremental scorer, the no-full-materialization property, and the first-cut
guards. They are small and in-process (tiny nets, few epochs), so they run in the default suite.
"""

import numpy as np
import pytest
import torch

from ilmarinen import AllGraph, AllData, DenseSource, InMemoryDenseSource, MemmapDenseSource
from ilmarinen.core.allgraph_streaming import _GridView, _StreamMetric, _reservoir_ids


# --------------------------------------------------------------------------- helpers
def _weights_identical(net_a, net_b):
    sa, sb = net_a.state_dict(), net_b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _fit_pair(Xarr, y, *, kind, task, n_out, source_cls=InMemoryDenseSource, source_arg=None, **cfg):
    """Fit the SAME data resident and streaming; return (resident_mg, resident_res, stream_mg, stream_res)."""
    mg_r = AllGraph(verbose=False, seed=0, **cfg)
    res_r = mg_r.fit(AllData.dense_tensor(Xarr, y, kind_hint=kind), task=task, n_out=n_out)
    src = source_cls(source_arg if source_arg is not None else Xarr)
    mg_s = AllGraph(verbose=False, seed=0, **cfg)
    res_s = mg_s.fit(AllData.dense_stream(src, y=y, kind_hint=kind), task=task, n_out=n_out)
    return mg_r, res_r, mg_s, res_s


def _spatial_cls(n=48, hw=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, hw, hw).astype(np.float32)
    y = (X.sum(axis=(1, 2)) > 0).astype(np.int64)
    return X, y


def _sequence_cls(n=50, T=12, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, T, 1).astype(np.float32)          # rank-3 samples (T, features) required under streaming
    y = (X[:, :, 0].mean(1) > 0).astype(np.int64)
    return X, y


# --------------------------------------------------------------------------- equivalence (the core guarantee)
@pytest.mark.parametrize("kind,task,n_out,maker", [
    ("spatial", "classification", 2, _spatial_cls),
    ("sequence", "classification", 2, _sequence_cls),
])
def test_streaming_weights_bit_identical(kind, task, n_out, maker):
    """T-STREAM-1: streaming vs resident deploy fit -> bit-identical trained weights AND equal reported value."""
    X, y = maker()
    mg_r, r_r, mg_s, r_s = _fit_pair(X, y, kind=kind, task=task, n_out=n_out, width=8, depth=1, epochs=4)
    assert r_r["contract"] == r_s["contract"] == kind
    assert _weights_identical(mg_r.net, mg_s.net), "streaming diverged from the resident weights"
    assert r_r["value"] == r_s["value"]
    assert r_r["n_params"] == r_s["n_params"]


def test_streaming_regression_metric_parity():
    """T-STREAM-2: regression R2 matches to fp tolerance (chunked ss_res vs whole-array), weights bit-identical."""
    rng = np.random.RandomState(0)
    X = rng.randn(45, 6, 6).astype(np.float32)
    y = (X.sum((1, 2)) + 0.1 * rng.randn(45)).astype(np.float32)
    mg_r, r_r, mg_s, r_s = _fit_pair(X, y, kind="spatial", task="regression", n_out=1, width=8, depth=1, epochs=5)
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-6)


def test_streaming_auto_epoch_val_equivalent():
    """T-STREAM-3: with auto_epoch='val' (held-out monitor + early stop) streaming stays bit-identical -- the
    seeded val split references the identical samples under map-style random access."""
    rng = np.random.RandomState(0)
    X = rng.randn(400, 6, 6).astype(np.float32)        # large enough for a reliable held-out monitor (>=50)
    y = (X.sum((1, 2)) > 0).astype(np.int64)
    mg_r, r_r, mg_s, r_s = _fit_pair(X, y, kind="spatial", task="classification", n_out=2,
                                     width=8, depth=1, epochs=30, auto_epoch="val")
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == r_s["value"]


def test_streaming_memmap_source_equivalent(tmp_path):
    """T-STREAM-4: a MemmapDenseSource over a real .npy file is bit-identical to the resident fit (exercises the
    sorted-gather-then-unpermute path end to end)."""
    X, _ = _spatial_cls(n=60)
    y = (X.sum((1, 2)) + 0.1 * np.random.RandomState(1).randn(60)).astype(np.float32)
    p = tmp_path / "X.npy"
    np.save(p, X)
    mg_r, r_r, mg_s, r_s = _fit_pair(X, y, kind="spatial", task="regression", n_out=1,
                                     source_cls=MemmapDenseSource, source_arg=str(p),
                                     width=8, depth=1, epochs=5)
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-6)


def test_streaming_repeatability_and_data_sensitivity():
    """T-STREAM-5: streaming is reproducible (two fits over the same source -> identical weights) and a genuine
    function of the data (a different source -> different weights). Note the schema builders seed their init
    deterministically, so this package's fits are reproducible-by-construction; seed-sensitivity is therefore
    intentionally NOT asserted here."""
    X, y = _spatial_cls()
    X2, y2 = _spatial_cls(seed=9)
    a = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=3)
    b = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=3)
    c = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=3)
    a.fit(AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial"), n_out=2)
    b.fit(AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial"), n_out=2)
    c.fit(AllData.dense_stream(InMemoryDenseSource(X2), y=y2, kind_hint="spatial"), n_out=2)
    assert _weights_identical(a.net, b.net)              # reproducible
    assert not _weights_identical(a.net, c.net)          # actually depends on the streamed data


# --------------------------------------------------------------------------- the source-gather contract
def test_unsorted_id_gather_matches_resident():
    """T-STREAM-6: get() with an unsorted id array returns rows in the REQUESTED order, matching X[ids], for
    both the in-memory and the memmap (sorted-then-unpermute) backings."""
    rng = np.random.RandomState(0)
    A = rng.randn(20, 3, 3).astype(np.float32)
    ids = np.array([7, 2, 19, 5, 0, 13], dtype=np.int64)
    exp = A[ids]
    assert np.array_equal(InMemoryDenseSource(A).get(ids).numpy(), exp)
    assert np.array_equal(InMemoryDenseSource(A).get(torch.as_tensor(ids)).numpy(), exp)   # torch selector too
    mm = MemmapDenseSource(A)                                  # ndarray backing exercises the same gather path
    assert np.array_equal(mm.get(ids).numpy(), exp)


def test_dense_source_shape_metadata_no_materialization():
    """T-STREAM-7: shape/dim are metadata (no row fetch), and _GridView exposes the channel-inserted shape so a
    builder can read n_in / extent without touching data."""
    class NoGet(DenseSource):
        def __init__(self, n, sample_shape):
            self._n = n
            self._sample_shape = tuple(sample_shape)
        def __len__(self):
            return self._n
        def get(self, ids):
            raise AssertionError("get() must not be called for shape metadata")

    src = NoGet(10_000_000, (6, 6))                            # 'huge' n, never materialized
    assert src.shape == (10_000_000, 6, 6) and src.dim() == 3
    gv = _GridView(src, rank=4)                                # rank-1 sample -> channel inserted
    assert gv.shape == (10_000_000, 1, 6, 6) and gv.dim() == 4


def test_gridview_rejects_flat_reshape():
    """T-STREAM-8: a source too low-rank for the grid contract (would need a latent-lattice reshape) raises a
    clear error rather than silently mis-shaping."""
    class Flat(DenseSource):
        _sample_shape = (36,)
        def __len__(self):
            return 8
        def get(self, ids):
            raise AssertionError
    with pytest.raises(NotImplementedError):
        _GridView(Flat(), rank=4)                              # dim()==2, needs rank 3 or 4


# --------------------------------------------------------------------------- the incremental scorer
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("chunk", [4, 7, 1000])
def test_stream_metric_matches_whole_array(task, chunk):
    """T-STREAM-9: _StreamMetric fed chunked predictions equals AllGraph._metric on the whole array (integer-
    exact for accuracy; fp-tolerant for R2), including the single-chunk (chunk >= n) degenerate case."""
    rng = np.random.RandomState(0)
    n = 37
    if task == "classification":
        out = torch.as_tensor(rng.randn(n, 3).astype(np.float32))
        y = (rng.rand(n) * 3).astype(np.int64)
    else:
        out = torch.as_tensor(rng.randn(n, 1).astype(np.float32))
        y = rng.randn(n).astype(np.float32)
    ref = AllGraph(verbose=False)._metric(out, y, task)
    acc = _StreamMetric(task, y)
    for j in range(0, n, chunk):
        acc.update(out[j:j + chunk], y[j:j + chunk])
    name, val = acc.result()
    assert name == ref[0]
    assert val == pytest.approx(ref[1], abs=1e-6)


# --------------------------------------------------------------------------- no full materialization / passes
class _CountingSource(DenseSource):
    """Records get() calls and the largest single request; asserts no request exceeds a cap."""

    def __init__(self, arr, row_cap):
        self._a = np.ascontiguousarray(arr)
        self._sample_shape = tuple(arr.shape[1:])
        self.dtype = arr.dtype
        self.row_cap = row_cap
        self.calls = 0
        self.max_rows = 0

    def __len__(self):
        return len(self._a)

    def get(self, ids):
        ids = ids.numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
        self.calls += 1
        self.max_rows = max(self.max_rows, len(ids))
        assert len(ids) <= self.row_cap, f"get() requested {len(ids)} rows > cap {self.row_cap}"
        return torch.as_tensor(self._a[ids.astype(np.int64)], dtype=torch.float32)


def test_no_full_materialization_and_pass_accounting():
    """T-STREAM-10: the deploy fit + eval only ever request <= max(train_bs, eval_bs) rows per get (never the
    full n), and the number of get() calls equals exactly epochs*train_batches + eval_batches (no hidden pass)."""
    X, y = _spatial_cls(n=200)
    train_bs, eval_bs, epochs = 32, 128, 3               # spatial eval_bs is 128; default train batch is 32
    cs = _CountingSource(X, row_cap=max(train_bs, eval_bs))
    mg = AllGraph(width=8, depth=1, epochs=epochs, verbose=False, seed=0)   # auto_epoch off -> full n train, no val
    mg.fit(AllData.dense_stream(cs, y=y, kind_hint="spatial"), task="classification", n_out=2)
    n = len(X)
    train_batches = -(-n // train_bs)                    # ceil
    eval_batches = -(-n // eval_bs)
    assert cs.max_rows <= eval_bs < n
    assert cs.calls == epochs * train_batches + eval_batches


# --------------------------------------------------------------------------- val-split identity / subsample RNG
def test_auto_val_split_identical_under_streaming():
    """T-STREAM-11: _auto_val_split depends only on n (and the isolated seed+7 RandomState), so streaming holds
    out exactly the same samples the resident path does."""
    mg = AllGraph(verbose=False, seed=0, auto_epoch="val")
    n = 400
    tr, va = mg._auto_val_split(n)
    k = max(int(0.15 * n), mg._AUTO_VAL_MIN)
    ref = np.random.RandomState(mg.seed + 7).permutation(n)
    assert np.array_equal(va, ref[:k]) and np.array_equal(tr, ref[k:])


def test_reservoir_ids_isolated_and_bounded():
    """T-STREAM-12: the search subsample is bounded, sorted, unique, and drawn from an ISOLATED RandomState so
    it perturbs neither the global numpy nor the global torch RNG stream (deploy fit stays reproducible)."""
    np.random.seed(123)
    torch.manual_seed(123)
    np_state = np.random.get_state()[1].copy()
    torch_state = torch.get_rng_state().clone()
    ids = _reservoir_ids(1000, cap=64, seed=7)
    assert len(ids) == 64 and len(np.unique(ids)) == 64 and list(ids) == sorted(ids) and ids.max() < 1000
    assert np.array_equal(np.random.get_state()[1], np_state)          # global numpy RNG untouched
    assert torch.equal(torch.get_rng_state(), torch_state)             # global torch RNG untouched
    assert np.array_equal(_reservoir_ids(1000, 64, 7), ids)            # deterministic
    assert np.array_equal(_reservoir_ids(30, 64, 7), np.arange(30))    # n <= cap -> everything


# --------------------------------------------------------------------------- inertness (resident path unchanged)
def test_resident_input_is_not_streaming():
    """T-STREAM-13: a resident tensor / ndarray is never seen as streaming, so the resident code path is taken
    verbatim (the single predicate gating every streaming branch)."""
    mg = AllGraph(verbose=False)
    assert mg._is_streaming(torch.randn(3, 3)) is False
    assert mg._is_streaming(np.zeros((3, 3))) is False
    assert mg._is_streaming(InMemoryDenseSource(np.zeros((3, 3), np.float32))) is True
    assert mg._is_streaming(_GridView(InMemoryDenseSource(np.zeros((3, 3), np.float32)), rank=3)) is True


# --------------------------------------------------------------------------- first-cut guards
def test_dense_stream_constructor_guards():
    """T-STREAM-14: dense_stream requires a DenseSource and a dense kind_hint."""
    src = InMemoryDenseSource(np.zeros((4, 6, 6), np.float32))
    with pytest.raises(TypeError):
        AllData.dense_stream(np.zeros((4, 6, 6), np.float32), kind_hint="spatial")   # not a DenseSource
    with pytest.raises(ValueError):
        AllData.dense_stream(src)                                                    # missing kind_hint
    with pytest.raises(ValueError):
        AllData.dense_stream(src, kind_hint="graph")                                 # non-dense contract


@pytest.mark.parametrize("flag", ["kernel_from_xi", "price_singular", "report_llc", "symmetry_routing"])
def test_fit_guards_still_blocked_options(flag):
    """T-STREAM-15: options that still re-read the FULL dataset (or read data.positions, absent under a source)
    raise a clear NotImplementedError under streaming, BEFORE any materialization."""
    X, y = _spatial_cls(n=30)
    data = AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial")
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=2, **{flag: True})
    with pytest.raises(NotImplementedError):
        mg.fit(data, task="classification", n_out=2)


@pytest.mark.parametrize("kw", [
    {"select": "gibbs"},
    {"select_size": "variable"},
])
def test_fit_selection_options_supported_under_streaming(kw):
    """T-STREAM-15b: select='gibbs' and select_size run under streaming (on a bounded resident subsample; the
    winner deploy-trains on the full stream), producing a finite result rather than raising."""
    X, y = _spatial_cls(n=60)
    data = AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial")
    mg = AllGraph(verbose=False, seed=0, width=16, depth=2, epochs=3)
    r = mg.fit(data, task="classification", n_out=2, **kw)
    assert r["contract"] == "spatial" and np.isfinite(r["value"])


def test_fit_stream_assertion_kwarg():
    """T-STREAM-16: fit(stream=...) is an assertion of caller intent -- it never flips streaming on/off, only
    errors on a mismatch with the container type."""
    X, y = _spatial_cls(n=30)
    src = InMemoryDenseSource(X)
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=2)
    with pytest.raises(ValueError):
        mg.fit(AllData.dense_tensor(X, y, kind_hint="spatial"), n_out=2, stream=True)   # resident, demanded stream
    with pytest.raises(ValueError):
        mg.fit(AllData.dense_stream(src, y=y, kind_hint="spatial"), n_out=2, stream=False)  # stream, forbade it
    # stream=True on a genuine stream is accepted
    r = mg.fit(AllData.dense_stream(src, y=y, kind_hint="spatial"), n_out=2, stream=True)
    assert r["contract"] == "spatial"


def test_raw_constructor_streaming_without_kind_hint_errors_clearly():
    """T-STREAM-19: a DenseSource attached via the raw AllData(dense=...) constructor (bypassing dense_stream,
    so kind_hint is None) raises a clear ValueError in routing -- NOT an opaque AttributeError from a flat-vector
    materialization attempt. (dense_stream itself forbids kind_hint=None; this guards the back door.)"""
    src = InMemoryDenseSource(np.zeros((40, 36), np.float32))     # rank-2 flat -> would hit route_grid_rank
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=2)
    with pytest.raises(ValueError, match="kind_hint"):
        mg.fit(AllData(dense=src, y=np.zeros(40, np.int64)), task="classification", n_out=2)


def test_streaming_sequence_requires_feature_axis():
    """T-STREAM-17: a streaming sequence source must present rank-3 samples (n, T, features); a rank-2 source
    raises rather than silently mis-shaping (the flat unsqueeze has no per-batch analogue)."""
    rng = np.random.RandomState(0)
    X = rng.randn(20, 12).astype(np.float32)             # (n, T) -- missing the feature axis
    y = (X.mean(1) > 0).astype(np.int64)
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=2)
    with pytest.raises(NotImplementedError):
        mg.fit(AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="sequence"), n_out=2)


# --------------------------------------------------------------------------- predict round-trip
def test_predict_after_streaming_train_matches_resident():
    """T-STREAM-18: predicting on resident data after a STREAMING train equals the resident-trained baseline
    (the trained weights are bit-identical, and predict() always runs on resident data)."""
    X, y = _spatial_cls(n=48)
    Xnew, _ = _spatial_cls(n=16, seed=5)
    mg_r = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=4)
    mg_r.fit(AllData.dense_tensor(X, y, kind_hint="spatial"), n_out=2)
    mg_s = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=4)
    mg_s.fit(AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial"), n_out=2)
    pr = mg_r.predict(AllData.dense_tensor(Xnew, kind_hint="spatial"))
    ps = mg_s.predict(AllData.dense_tensor(Xnew, kind_hint="spatial"))
    assert np.array_equal(pr, ps)

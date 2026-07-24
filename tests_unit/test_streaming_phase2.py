"""T-P2: the phase-2 streaming additions -- bounded LRU cache, selection-on-subsample, async prefetch, and
iterable sources. Each preserves the sacred constraints (streaming-off unchanged; map-style deploy bit-identical
except the iterable regime, which is deterministic-given-seed).
"""

import numpy as np
import pytest
import torch

from ilmarinen import (AllGraph, AllData,
                       InMemoryDenseSource, MemmapDenseSource,
                       InMemoryGraphSource, LazyGraphSource,
                       InMemoryOperatorSource, MemmapOperatorSource)
from ilmarinen import IterableDenseSource, InMemoryIterableDenseSource
from ilmarinen.core.allgraph_streaming import _LRUCache, _IterMetric, _iter_val_key


def _wsd(net):
    return {k: v.clone() for k, v in net.state_dict().items()}


def _ident(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _spatial(n=60, hw=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, hw, hw).astype(np.float32)
    return X, (X.sum((1, 2)) > 0).astype(np.int64)


def _graphs(n=24, nn=6, seed=0):
    rng = np.random.RandomState(seed)
    nf, ed, ys = [], [], []
    for _ in range(n):
        f = rng.randn(nn, 4).astype(np.float32)
        e = [(i, (i + 1) % nn) for i in range(nn)] + [(0, nn // 2)]
        nf.append(f); ed.append(np.array(e, dtype=np.int64).T); ys.append(int(f[:, 0].sum() > 0))
    return nf, ed, np.array(ys, dtype=np.int64)


def _op(n=40, grid=16, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(n, grid).astype(np.float32)
    k = np.array([0.25, 0.5, 0.25], np.float32)
    u = np.stack([np.convolve(r, k, mode="same") for r in a]).astype(np.float32)
    return a, u


# ========================================================================= item 2: bounded LRU
def test_lru_cache_semantics():
    c = _LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)          # evicts 'a'
    assert c.get("a") is None and c.get("b") == 2 and c.get("c") == 3
    c.get("b"); c.put("d", 4)                            # 'c' is now oldest -> evicted
    assert c.get("c") is None and c.get("b") == 2 and c.get("d") == 4
    d = _LRUCache(0); d.put("x", 1); assert d.get("x") is None   # capacity 0 disables
    # byte budget: an item larger than the whole budget is never stored
    e = _LRUCache(10); e.put("big", "v", cost=100); assert e.get("big") is None


@pytest.mark.parametrize("cap", [0, 4, 100000])
def test_memmap_dense_cache_parity(tmp_path, cap):
    rng = np.random.RandomState(0)
    A = rng.randn(50, 3, 4).astype(np.float32)
    p = tmp_path / "A.npy"; np.save(p, A)
    src = MemmapDenseSource(str(p), cache_size=cap)
    for _ in range(20):
        ids = rng.randint(0, 50, size=rng.randint(1, 10))
        assert np.array_equal(src.get(ids).numpy(), A[ids])   # cache-state-independent, unsorted+repeat


def test_memmap_dense_cache_deploy_bit_identical(tmp_path):
    X, y = _spatial(n=200)
    p = tmp_path / "X.npy"; np.save(p, X)

    def fit(cs):
        mg = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
        mg.fit(AllData.dense_stream(MemmapDenseSource(str(p), cache_size=cs), y=y, kind_hint="spatial"), n_out=2)
        return _wsd(mg.net)
    assert _ident(fit(0), fit(500))                       # cache on vs off -> identical weights


def test_memmap_operator_cache_independent_and_bit_identical(tmp_path):
    a, u = _op(n=48)
    pa, pu = tmp_path / "a.npy", tmp_path / "u.npy"; np.save(pa, a); np.save(pu, u)
    src = MemmapOperatorSource(str(pa), str(pu), cache_bytes=1_000_000)
    ids = np.array([5, 1, 5, 9, 0])
    assert np.array_equal(src.a(ids).numpy(), a[ids])    # a-cache
    assert np.array_equal(src.u(ids).numpy(), u[ids])    # u-cache, independent of a-cache

    def fit(cb):
        mg = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
        mg.fit(AllData.functions_stream(MemmapOperatorSource(str(pa), str(pu), cache_bytes=cb)),
               task="regression", n_out=1)
        return _wsd(mg.net)
    assert _ident(fit(0), fit(2_000_000))


def test_lazy_graph_cache_default_is_memo():
    nf, ed, _ = _graphs(n=8)
    calls = {"n": 0}

    def loader(i):
        calls["n"] += 1
        return {"node": nf[i], "edge": ed[i]}
    s = LazyGraphSource(loader, n=8, n_in=4, has_edges=True)     # default cache_size=1
    s.node(3); s.edge(3)                                 # same i, consecutive -> 1 loader call
    assert calls["n"] == 1
    s.node(3)                                            # still cached
    assert calls["n"] == 1
    s.node(4); s.node(3)                                 # cache_size=1: 4 evicts 3 -> reload
    assert calls["n"] == 3


def test_lazy_graph_cache_keeps_k_recent():
    nf, ed, _ = _graphs(n=8)
    calls = {"n": 0}

    def loader(i):
        calls["n"] += 1
        return {"node": nf[i]}
    s = LazyGraphSource(loader, n=8, n_in=4, cache_size=4)
    for i in [0, 1, 2, 3]:
        s.node(i)
    assert calls["n"] == 4
    for i in [0, 1, 2, 3]:                               # all still cached
        s.node(i)
    assert calls["n"] == 4


def test_cache_threadsafe_smoke():
    import threading
    c = _LRUCache(64, threadsafe=True)
    errors = []

    def worker(base):
        try:
            for i in range(200):
                c.put((base, i % 80), i)
                c.get((base, i % 80))
        except Exception as e:                           # pragma: no cover
            errors.append(e)
    ts = [threading.Thread(target=worker, args=(b,)) for b in range(6)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert not errors
    assert _LRUCache(4, threadsafe=False)._lock is None


# ========================================================================= item 1: selection on subsample
def test_streaming_deploy_bit_identical_preserved():
    """Sacred: a plain map-style deploy fit (no selection) is still bit-identical to resident -- the subsample
    plumbing is never drawn and never touches the deploy RNG."""
    X, y = _spatial(n=60)
    m1 = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    m1.fit(AllData.dense_tensor(X, y, kind_hint="spatial"), n_out=2)
    m2 = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    m2.fit(AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial"), n_out=2)
    assert _ident(_wsd(m1.net), _wsd(m2.net))


@pytest.mark.parametrize("select_size", ["variable"])
def test_dense_select_size_streaming_deterministic(select_size):
    X, y = _spatial(n=120, hw=8)

    def run():
        mg = AllGraph(width=32, depth=2, epochs=3, verbose=False, seed=0)
        mg.fit(AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial"),
               n_out=2, select_size=select_size)
        return mg.width, mg.depth, _wsd(mg.net)
    a, b = run(), run()
    assert (a[0], a[1]) == (b[0], b[1]) and _ident(a[2], b[2])   # deterministic-given-seed


def test_graph_gibbs_streaming_deterministic():
    nf, ed, y = _graphs()

    def run():
        mg = AllGraph(width=8, depth=2, epochs=4, verbose=False, seed=0)
        r = mg.fit(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph"),
                   n_out=2, select="gibbs")
        return r, _wsd(mg.net)
    (r1, w1), (r2, w2) = run(), run()
    assert r1.get("architecture_gibbs") == r2.get("architecture_gibbs")
    assert _ident(w1, w2)


def test_operator_gibbs_streaming_runs():
    a, u = _op()
    mg = AllGraph(width=8, depth=2, epochs=4, verbose=False, seed=0)
    r = mg.fit(AllData.functions_stream(InMemoryOperatorSource(a, u)), task="regression", n_out=1, select="gibbs")
    assert np.isfinite(r["value"]) and r["contract"] == "operator"
    assert "architecture_gibbs" in r


def test_reservoir_subsample_rng_isolation():
    """Drawing the selection subsample uses an isolated RandomState(seed+23) and never perturbs the global
    numpy/torch RNG -> the deploy fit's shuffle (and weights) are unaffected."""
    from ilmarinen.core.allgraph_streaming import _reservoir_ids
    np.random.seed(123); torch.manual_seed(123)
    np_state = np.random.get_state()[1].copy(); torch_state = torch.get_rng_state().clone()
    ids = _reservoir_ids(1000, 64, seed=23)
    assert len(ids) == 64 and list(ids) == sorted(ids)
    assert np.array_equal(np.random.get_state()[1], np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)


# ========================================================================= item 3: async prefetch
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_prefetch_dense_bit_identical(depth):
    X, y = _spatial(n=120, hw=8)

    def fit(pf):
        mg = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0, stream_prefetch=pf)
        mg.fit(AllData.dense_stream(InMemoryDenseSource(X), y=y, kind_hint="spatial"), n_out=2)
        return _wsd(mg.net)
    assert _ident(fit(0), fit(depth))


def test_prefetch_all_families_bit_identical():
    nf, ed, yg = _graphs(n=40)
    a, u = _op(n=60)
    X, y = _spatial(n=80)
    cases = [
        lambda pf: (AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0, stream_prefetch=pf),
                    AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=yg, kind_hint="graph"), dict(n_out=2)),
        lambda pf: (AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0, stream_prefetch=pf),
                    AllData.graph_stream(InMemoryGraphSource(nf), y=yg, kind_hint="set"), dict(n_out=2)),
        lambda pf: (AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0, stream_prefetch=pf),
                    AllData.functions_stream(InMemoryOperatorSource(a, u)), dict(task="regression", n_out=1)),
    ]
    for case in cases:
        def fit(pf):
            mg, data, kw = case(pf)
            mg.fit(data, **kw)
            return _wsd(mg.net)
        assert _ident(fit(0), fit(3))


def test_prefetch_resident_inert():
    """Sacred: stream_prefetch on a RESIDENT fit is silently ignored -> byte-identical to prefetch off."""
    X, y = _spatial(n=60)

    def fit(pf):
        mg = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0, stream_prefetch=pf)
        mg.fit(AllData.dense_tensor(X, y, kind_hint="spatial"), n_out=2)
        return _wsd(mg.net)
    assert _ident(fit(0), fit(4))


def test_prefetch_lazy_graph_threadsafe():
    """A LazyGraphSource (with a threadsafe cache) driven under prefetch trains bit-identically to no prefetch --
    the background producer is the only in-loop source consumer and the cache lock is the defensive belt."""
    import time
    nf, ed, yg = _graphs(n=40)

    def make(threadsafe):
        def loader(i):
            time.sleep(0.0005)                           # simulate I/O latency to actually overlap
            return {"node": nf[i], "edge": ed[i]}
        return LazyGraphSource(loader, n=40, n_in=4, has_edges=True, cache_size=8, cache_threadsafe=threadsafe)

    def fit(pf, ts):
        mg = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0, stream_prefetch=pf)
        mg.fit(AllData.graph_stream(make(ts), y=yg, kind_hint="graph"), n_out=2)
        return _wsd(mg.net)
    assert _ident(fit(0, False), fit(4, True))


def test_prefetch_worker_exception_propagates():
    """A fetch failure on the worker surfaces on the main thread (no hang, no swallowed error)."""
    X, y = _spatial(n=60)

    class _Boom(InMemoryDenseSource):
        def get(self, ids):
            raise RuntimeError("fetch boom")
    mg = AllGraph(width=8, depth=1, epochs=2, verbose=False, seed=0, stream_prefetch=2)
    with pytest.raises(RuntimeError, match="boom"):
        mg.fit(AllData.dense_stream(_Boom(X), y=y, kind_hint="spatial"), n_out=2)


def test_prefetch_validation():
    with pytest.raises(ValueError):
        AllGraph(stream_prefetch=-1)
    with pytest.raises(ValueError):
        AllGraph(stream_prefetch="x")
    assert AllGraph(stream_prefetch=True)._prefetch_depth() == 1
    assert AllGraph(stream_prefetch=3)._prefetch_depth() == 3
    assert AllGraph(stream_prefetch=False)._prefetch_depth() == 0


def test_assemble_batch_refactor_parity():
    """_batch_to_device(_collate_cpu(...)) == the old single-pass _assemble_batch, with/without edges+positions."""
    mg = AllGraph(verbose=False)
    nf = [torch.randn(5, 4), torch.randn(7, 4)]
    ed = [torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 2], [2, 0]])]
    pos = [torch.randn(5, 3), torch.randn(7, 3)]
    node_t = lambda i: nf[i]; edge_t = lambda i: ed[i]; pos_t = lambda i: pos[i]
    ids = [0, 1]
    x, ei, p, b, ng = mg._assemble_batch(ids, node_t, edge_t, pos_t)
    cpu = mg._collate_cpu(ids, node_t, edge_t, pos_t)
    x2, ei2, p2, b2, ng2 = mg._batch_to_device(cpu)
    assert torch.equal(x, x2) and torch.equal(ei, ei2) and torch.equal(p, p2) and torch.equal(b, b2) and ng == ng2


def test_generalized_subsample_builders_correct():
    """_resident_subsample for graph/operator materializes the reservoir ids' fields correctly (operator y comes
    from src.u since data.y is None)."""
    mg = AllGraph(verbose=False, seed=0)
    # graph: N<=cap -> subsample is all graphs (reservoir returns arange)
    nf, ed, y = _graphs(n=10)
    gdata = AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph")
    sub = mg._resident_subsample(gdata, cap=10000)
    assert len(sub.node_feats) == 10 and torch.equal(sub.node_feats[0], torch.as_tensor(nf[0]))
    assert np.array_equal(np.asarray(sub.y), y)
    # operator: y is the target field src.u
    a, u = _op(n=12)
    odata = AllData.functions_stream(InMemoryOperatorSource(a, u))
    mg.contract = "operator"
    osub = mg._resident_subsample(odata, cap=10000)
    assert np.allclose(np.asarray(osub.y), u) and osub.dense.shape[0] == 12


# ========================================================================= item 4: iterable regime
def _iter_spatial(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 1, 8, 8).astype(np.float32)         # channeled (rank-4) samples: no _as_grid fix-up
    y = (X.sum((1, 2, 3)) > 0).astype(np.int64)
    return X, y


def test_iterable_deterministic_given_seed_and_seed_sensitive():
    X, y = _iter_spatial()

    def fit(seed):
        mg = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=seed)
        mg.fit(AllData.dense_iter(InMemoryIterableDenseSource(X, y, n_out=2), kind_hint="spatial"),
               task="classification", n_out=2)
        return _wsd(mg.net)
    assert _ident(fit(0), fit(0))                        # deterministic given seed
    assert not _ident(fit(0), fit(1))                    # seed-sensitive (shuffle + hash split both reseed)


def test_iterable_not_bit_identical_to_resident_but_valid():
    X, y = _iter_spatial()
    mg_i = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    r_i = mg_i.fit(AllData.dense_iter(InMemoryIterableDenseSource(X, y, n_out=2), kind_hint="spatial"),
                   task="classification", n_out=2)
    mg_r = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    mg_r.fit(AllData.dense_tensor(X, y, kind_hint="spatial"), n_out=2)
    assert not _ident(_wsd(mg_i.net), _wsd(mg_r.net))    # different shuffle/split by construction
    assert np.isfinite(r_i["value"])


def test_iterable_sequence_and_auto_epoch_val():
    rng = np.random.RandomState(0)
    Xs = rng.randn(150, 12, 1).astype(np.float32)
    ys = (Xs[:, :, 0].mean(1) > 0).astype(np.int64)
    mg = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    r = mg.fit(AllData.dense_iter(InMemoryIterableDenseSource(Xs, ys, n_out=2), kind_hint="sequence"),
               task="classification", n_out=2)
    assert r["readout"] == "mean" and np.isfinite(r["value"])
    X, y = _iter_spatial()
    mgv = AllGraph(width=8, depth=1, epochs=20, verbose=False, seed=0, auto_epoch="val")
    rv = mgv.fit(AllData.dense_iter(InMemoryIterableDenseSource(X, y, n_out=2), kind_hint="spatial"),
                 task="classification", n_out=2)
    assert np.isfinite(rv["value"])


def test_iterable_hash_split_stable_and_balanced():
    buckets = [_iter_val_key(i, 0) % 1000 < 150 for i in range(10000)]
    assert 0.12 < sum(buckets) / 10000 < 0.18            # ~15% val
    assert _iter_val_key(5, 0) != _iter_val_key(5, 1)    # seed reshuffles the split
    assert _iter_val_key(5, 0) == _iter_val_key(5, 0)    # stable (process-independent blake2b, not salted hash)


def test_iter_metric_matches_metric():
    rng = np.random.RandomState(0)
    for task, out, y in [
        ("classification", torch.as_tensor(rng.randn(37, 3).astype(np.float32)), (rng.rand(37) * 3).astype(np.int64)),
        ("regression", torch.as_tensor(rng.randn(37, 1).astype(np.float32)), rng.randn(37).astype(np.float32)),
    ]:
        ref = AllGraph(verbose=False)._metric(out, y, task)
        acc = _IterMetric(task)
        for j in range(0, 37, 7):
            acc.update(out[j:j + 7], y[j:j + 7])
        assert acc.result()[0] == ref[0]
        assert acc.result()[1] == pytest.approx(ref[1], abs=1e-4)


def test_iterable_restartable_multiple_epochs():
    """A restartable source is re-iterated each epoch (multi-epoch training sees data every epoch)."""
    X, y = _iter_spatial(n=80)

    class _CountingIter(InMemoryIterableDenseSource):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.iters = 0

        def __iter__(self):
            self.iters += 1                              # __iter__ is resolved on the TYPE, so subclass it
            return super().__iter__()

    src = _CountingIter(X, y, n_out=2)
    mg = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0)
    mg.fit(AllData.dense_iter(src, kind_hint="spatial"), task="classification", n_out=2)
    assert src.iters >= 3                                # at least one restart per epoch (3 train epochs)


@pytest.mark.parametrize("bad", [
    {"select_size": "variable"},
    {"select": "gibbs"},
    {"tiebreak": True},
])
def test_iterable_selection_blocked(bad):
    X, y = _iter_spatial(n=40)
    data = AllData.dense_iter(InMemoryIterableDenseSource(X, y, n_out=2), kind_hint="spatial")
    mg = AllGraph(width=8, depth=1, epochs=2, verbose=False, seed=0)
    with pytest.raises(NotImplementedError):
        mg.fit(data, task="classification", n_out=2, **bad)


def test_iterable_classification_needs_n_out():
    X, y = _iter_spatial(n=40)
    mg = AllGraph(width=8, depth=1, epochs=2, verbose=False, seed=0)
    with pytest.raises(ValueError, match="n_out"):
        mg.fit(AllData.dense_iter(InMemoryIterableDenseSource(X, y), kind_hint="spatial"), task="classification")


def test_iterable_constructor_guards():
    X, y = _iter_spatial(n=20)
    with pytest.raises(TypeError):
        AllData.dense_iter(X, kind_hint="spatial")                     # not an IterableDenseSource
    with pytest.raises(ValueError):
        AllData.dense_iter(InMemoryIterableDenseSource(X, y))          # missing kind_hint
    with pytest.raises(ValueError):
        AllData.dense_iter(InMemoryIterableDenseSource(X, y), kind_hint="graph")   # non-dense


# ========================================================================= review fixes (phase-2)
def test_lru_cached_row_is_standalone_copy(tmp_path):
    """Review fix: a cached row must be an independent COPY, not a view into the whole minibatch read buffer
    (a view would pin the whole buffer and blow the cache's memory budget)."""
    A = np.random.RandomState(0).randn(40, 30, 30).astype(np.float32)
    p = tmp_path / "A.npy"; np.save(p, A)
    src = MemmapDenseSource(str(p), cache_size=8)
    src.get(np.array([0, 1, 2, 3]))                      # populate the cache with a 4-row read
    row = src._cache.get(3)                              # still-cached, most-recent
    assert row is not None and row.nbytes == 30 * 30 * 4  # one sample, not the whole 4-row buffer
    assert not np.shares_memory(row, src.get(np.array([0, 1, 2, 3, 4, 5])).numpy())


def test_prefetch_consumer_exception_no_deadlock():
    """Review fix: a consumer (compute) exception mid-epoch, with the producer parked on a full queue, must not
    deadlock -- the generator signals stop, drains, and joins."""
    import threading
    import time
    mg = AllGraph(verbose=False, seed=0, stream_prefetch=2)

    def fetch(ids):
        time.sleep(0.005)                                # slow enough that the producer fills the queue
        return ("payload", ids)

    def run():
        for n, (ids, payload) in enumerate(mg._prefetch_batches(np.arange(200), 10, fetch, depth=2)):
            if n == 1:
                raise RuntimeError("compute boom")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=8)
    assert not t.is_alive(), "prefetch generator deadlocked on a consumer exception"


def test_iter_metric_empty_is_nan():
    """Review fix: an empty / exhausted scoring stream reports nan, not a fabricated perfect R2=1.0 / acc=0.0."""
    import math
    assert math.isnan(_IterMetric("regression").result()[1])
    assert math.isnan(_IterMetric("classification").result()[1])


def test_gibbs_deploy_respects_train_batch():
    """Review fix: the gibbs deploy trainer slices idx[j:j+self._tb()] (was a hardcoded 32), so a non-default
    train_batch covers all data instead of skipping it. Isolated via the operator deploy (its score chunks are
    32, so a deploy-train batch of 50 with train_batch=64 is distinguishable and would be capped at 32 before)."""
    a, u = _op(n=50)

    class _SpyOp(InMemoryOperatorSource):
        def __init__(self, *ar, **kw):
            super().__init__(*ar, **kw)
            self.lens = []

        def a(self, ids):
            self.lens.append(len(np.asarray(ids)))
            return super().a(ids)

    src = _SpyOp(a, u)
    mg = AllGraph(width=8, depth=1, epochs=2, verbose=False, seed=0, train_batch=64, stream_subsample_cap=16)
    mg.fit(AllData.functions_stream(src), task="regression", n_out=1, select="gibbs")
    # deploy-train batch = idx[0:64] over n=50 -> 50 samples in one batch; before the fix it capped at 32.
    assert max(src.lens) == 50


def test_gibbs_nondefault_train_batch_deterministic():
    nf, ed, y = _graphs(n=48)

    def run():
        mg = AllGraph(width=8, depth=1, epochs=2, verbose=False, seed=0, train_batch=7, stream_subsample_cap=16)
        r = mg.fit(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph"),
                   n_out=2, select="gibbs")
        return r["value"], _wsd(mg.net)
    (v1, w1), (v2, w2) = run(), run()
    assert np.isfinite(v1) and _ident(w1, w2)

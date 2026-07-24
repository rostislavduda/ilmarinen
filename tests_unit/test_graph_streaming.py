"""T-GSTREAM: opt-in dataset streaming for the relational contracts (graph / equivariant / set).

Same guarantee as the dense case: a deployed relational fit driven from a GraphSource trains to bit-for-bit
identical weights as the equivalent resident fit over the same graphs. The relational path already shuffles
with np.random.permutation and collates per-sample on CPU, so streaming changes only the tensor SOURCE (the
GraphSource vs the pre-built _prepare_batch_cache) -- the shuffle, batch membership, and forward are untouched.
"""

import numpy as np
import pytest
import torch

from ilmarinen import AllData, AllGraph, GraphSource, InMemoryGraphSource, LazyGraphSource


# --------------------------------------------------------------------------- helpers / data
def _weights_identical(net_a, net_b):
    sa, sb = net_a.state_dict(), net_b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _graphs(n=24, nn=6, seed=0):
    """Tiny graphs: node feats + a ring-plus-chord edge list, binary label on the node-feature sum."""
    rng = np.random.RandomState(seed)
    nf, ed, ys = [], [], []
    for _ in range(n):
        f = rng.randn(nn, 4).astype(np.float32)
        e = [(i, (i + 1) % nn) for i in range(nn)] + [(0, nn // 2)]
        nf.append(f)
        ed.append(np.array(e, dtype=np.int64).T)
        ys.append(int(f[:, 0].sum() > 0))
    return nf, ed, np.array(ys, dtype=np.int64)


def _positions(n=24, nn=6, seed=1):
    return [np.asarray(np.random.RandomState(seed + g).randn(nn, 3), np.float32) for g in range(n)]


def _point_sets(n=24, nn=7, seed=0):
    rng = np.random.RandomState(seed)
    nf = [np.asarray(rng.randn(nn, 4), np.float32) for _ in range(n)]
    y = np.array([int(x[:, 0].sum() > 0) for x in nf], dtype=np.int64)
    return nf, y


# --------------------------------------------------------------------------- equivalence
def test_graph_streaming_bit_identical():
    """T-GSTREAM-1: graph contract -> bit-identical weights + equal value vs resident."""
    nf, ed, y = _graphs()
    mg_r = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    r_r = mg_r.fit(AllData.graphs(nf, ed, y=y), task="classification", n_out=2)
    mg_s = AllGraph(width=8, depth=1, epochs=5, verbose=False, seed=0)
    r_s = mg_s.fit(
        AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph"), task="classification", n_out=2
    )
    assert r_r["contract"] == r_s["contract"] == "graph"
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == r_s["value"]


def test_equivariant_streaming_bit_identical():
    """T-GSTREAM-2: equivariant contract (edges + positions) -> bit-identical vs resident."""
    nf, ed, y = _graphs()
    pos = _positions()
    mg_r = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    r_r = mg_r.fit(AllData.graphs(nf, ed, y=y, positions=pos), task="classification", n_out=2)
    mg_s = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    r_s = mg_s.fit(
        AllData.graph_stream(InMemoryGraphSource(nf, edges=ed, positions=pos), y=y, kind_hint="equivariant"),
        task="classification",
        n_out=2,
    )
    assert r_r["contract"] == r_s["contract"] == "equivariant"
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == r_s["value"]


@pytest.mark.parametrize(
    "task,n_out,auto_epoch",
    [
        ("classification", 2, None),
        ("regression", 1, "val"),
    ],
)
def test_set_streaming_bit_identical(task, n_out, auto_epoch):
    """T-GSTREAM-3: set contract (edgeless point sets), classification and regression+auto_epoch='val'."""
    n = 300 if auto_epoch == "val" else 24  # 'val' needs enough for a held-out monitor
    nf, ycls = _point_sets(n=n)
    y = ycls if task == "classification" else np.array([float(x[:, 0].sum()) for x in nf], np.float32)
    epochs = 20 if auto_epoch else 5
    mg_r = AllGraph(width=8, depth=1, epochs=epochs, verbose=False, seed=0, auto_epoch=auto_epoch)
    r_r = mg_r.fit(AllData.point_sets(nf, y=y), task=task, n_out=n_out)
    mg_s = AllGraph(width=8, depth=1, epochs=epochs, verbose=False, seed=0, auto_epoch=auto_epoch)
    r_s = mg_s.fit(AllData.graph_stream(InMemoryGraphSource(nf), y=y, kind_hint="set"), task=task, n_out=n_out)
    assert r_r["contract"] == r_s["contract"] == "set"
    assert _weights_identical(mg_r.net, mg_s.net)
    assert r_r["value"] == pytest.approx(r_s["value"], abs=1e-6)


def test_lazy_graph_source_matches_inmemory():
    """T-GSTREAM-4: a LazyGraphSource(loader) trains bit-identically to the in-memory source over the same data
    (the 1-element memo makes node/edge/pos for one graph share a single loader call)."""
    nf, ed, y = _graphs()
    loads = {"n": 0}

    def loader(i):
        loads["n"] += 1
        return {"node": nf[i], "edge": ed[i]}

    lazy = LazyGraphSource(loader, n=len(nf), n_in=4, has_edges=True)
    mg_a = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    mg_a.fit(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph"), n_out=2)
    mg_b = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    mg_b.fit(AllData.graph_stream(lazy, y=y, kind_hint="graph"), n_out=2)
    assert _weights_identical(mg_a.net, mg_b.net)
    assert loads["n"] > 0  # the loader was actually driven


def test_streaming_repeatability_and_data_sensitivity():
    """T-GSTREAM-5: reproducible over the same source; a different source -> different weights."""
    nf, ed, y = _graphs(seed=0)
    nf2, ed2, y2 = _graphs(seed=9)
    a = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    b = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    c = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    a.fit(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph"), n_out=2)
    b.fit(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph"), n_out=2)
    c.fit(AllData.graph_stream(InMemoryGraphSource(nf2, edges=ed2), y=y2, kind_hint="graph"), n_out=2)
    assert _weights_identical(a.net, b.net)
    assert not _weights_identical(a.net, c.net)


# --------------------------------------------------------------------------- streaming, not caching
class _CountingGraphSource(GraphSource):
    """Counts per-graph fetches; asserts a graph is only ever fetched one at a time (never a bulk list)."""

    def __init__(self, node_feats, edges):
        self._nf = node_feats
        self._ed = edges
        self.n_in = int(node_feats[0].shape[1])
        self.has_edges = True
        self.has_pos = False
        self.node_calls = 0

    def __len__(self):
        return len(self._nf)

    def node(self, i):
        assert np.isscalar(i) or np.ndim(i) == 0, "node() must be fetched per-graph, not in bulk"
        self.node_calls += 1
        return torch.as_tensor(self._nf[i], dtype=torch.float32)

    def edge(self, i):
        return torch.as_tensor(self._ed[i], dtype=torch.long)


def test_graphs_fetched_per_sample_not_cached():
    """T-GSTREAM-6: each graph is fetched from the source ON DEMAND every epoch (no full pre-cache): node() is
    called exactly epochs*n (training) + n (eval) times for auto_epoch off (full-n train, no val)."""
    nf, ed, y = _graphs(n=20)
    epochs = 3
    cs = _CountingGraphSource(nf, ed)
    mg = AllGraph(width=8, depth=1, epochs=epochs, verbose=False, seed=0)  # auto_epoch off -> train on all n
    mg.fit(AllData.graph_stream(cs, y=y, kind_hint="graph"), task="classification", n_out=2)
    n = len(nf)
    assert cs.node_calls == epochs * n + n  # re-fetched each epoch + once for eval; never cached


# --------------------------------------------------------------------------- predict / inertness
def test_predict_after_streaming_train_matches_resident():
    """T-GSTREAM-7: predict on a resident (or streamed) graph set after a streaming train equals the resident
    baseline (bit-identical weights)."""
    nf, ed, y = _graphs()
    mg_r = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    mg_r.fit(AllData.graphs(nf, ed, y=y), n_out=2)
    mg_s = AllGraph(width=8, depth=1, epochs=4, verbose=False, seed=0)
    mg_s.fit(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph"), n_out=2)
    pr = mg_r.predict(AllData.graphs(nf, ed))
    ps_resident = mg_s.predict(AllData.graphs(nf, ed))
    ps_stream = mg_s.predict(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), kind_hint="graph"))
    assert np.array_equal(pr, ps_resident)
    assert np.array_equal(pr, ps_stream)  # predict accepts a streamed test set too


def test_streamed_predict_with_canonicalization_errors_clearly():
    """T-GSTREAM-13: predict() on a STREAMED test set for a model that applied a canonicalization quotient at
    fit time raises a clear NotImplementedError (not an opaque TypeError from indexing the source as a list)."""
    nf, ed, y = _graphs(n=12)
    mg = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0)
    mg.fit(AllData.graphs(nf, ed, y=y), n_out=2)
    mg._canonicalization_applied = True  # simulate a canonicalized fit
    with pytest.raises(NotImplementedError, match="canonicalization"):
        mg.predict(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), kind_hint="graph"))
    # resident predict on the same (canonicalization-flagged) model still takes the normal path
    # (apply_canonicalization on resident lists) -- not asserted here beyond "no streaming crash".


def test_resident_relational_not_streaming():
    """T-GSTREAM-8: resident lists are never seen as a graph stream (the single predicate gating the branch)."""
    nf, ed, y = _graphs(n=4)
    mg = AllGraph(verbose=False)
    assert mg._is_streaming_graph(AllData.graphs(nf, ed, y=y)) is False
    assert mg._is_streaming_graph(AllData.point_sets(nf, y=y)) is False
    assert (
        mg._is_streaming_graph(AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph")) is True
    )


# --------------------------------------------------------------------------- guards
def test_graph_stream_constructor_guards():
    """T-GSTREAM-9: graph_stream requires a GraphSource, a relational kind_hint, and matching source capabilities."""
    nf, ed, y = _graphs(n=6)
    src_no_edges = InMemoryGraphSource(nf)
    src_edges = InMemoryGraphSource(nf, edges=ed)
    with pytest.raises(TypeError):
        AllData.graph_stream(nf, kind_hint="graph")  # not a GraphSource
    with pytest.raises(ValueError):
        AllData.graph_stream(src_edges)  # missing kind_hint
    with pytest.raises(ValueError):
        AllData.graph_stream(src_edges, kind_hint="spatial")  # dense contract
    with pytest.raises(ValueError):
        AllData.graph_stream(src_no_edges, kind_hint="graph")  # graph needs edges
    with pytest.raises(ValueError):
        AllData.graph_stream(src_edges, kind_hint="equivariant")  # equivariant needs positions


@pytest.mark.parametrize("flag", ["price_singular", "report_llc", "symmetry_routing"])
def test_graph_stream_still_blocked_options(flag):
    """T-GSTREAM-10: options that still re-read the FULL dataset (or read data.positions, absent under a source)
    raise NotImplementedError under graph streaming."""
    nf, ed, y = _graphs(n=20)
    src = InMemoryGraphSource(nf, edges=ed)
    data = AllData.graph_stream(src, y=y, kind_hint="graph")
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=2, **{flag: True})
    with pytest.raises(NotImplementedError):
        mg.fit(data, task="classification", n_out=2)


@pytest.mark.parametrize(
    "kw,setup,kind",
    [
        ({"select": "gibbs"}, {}, "graph"),
        ({"select_size": "sequential"}, {}, "graph"),
        ({"tiebreak": True}, {}, "equivariant"),
        ({}, {"angular_from_data": True}, "equivariant"),
    ],
)
def test_graph_stream_selection_supported(kw, setup, kind):
    """T-GSTREAM-10b: select='gibbs' / select_size / tiebreak / angular_from_data run under graph streaming (on a
    bounded resident subsample; the winner deploy-trains on the full stream), producing a finite result."""
    nf, ed, y = _graphs(n=24)
    pos = _positions(n=24)
    src = InMemoryGraphSource(nf, edges=ed, positions=pos if kind == "equivariant" else None)
    data = AllData.graph_stream(src, y=y, kind_hint=kind)
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=3, **setup)
    r = mg.fit(data, task="classification", n_out=2, **kw)
    assert np.isfinite(r["value"]) and r["contract"] in ("graph", "equivariant", "set")


def test_lazy_source_retries_after_transient_loader_error():
    """T-GSTREAM-12: LazyGraphSource commits its memo only AFTER the loader succeeds -- a retry of the same index
    after a transient loader failure re-loads (never silently returns the previously loaded graph)."""
    nf, ed, _ = _graphs(n=4)
    state = {"fail_next": False}

    def loader(i):
        if state["fail_next"]:
            state["fail_next"] = False
            raise OSError("transient backend hiccup")
        return {"node": nf[i], "edge": ed[i]}

    src = LazyGraphSource(loader, n=4, n_in=4, has_edges=True)
    assert torch.equal(src.node(0), torch.as_tensor(nf[0]))  # load graph 0 -> memo = graph 0
    state["fail_next"] = True
    with pytest.raises(IOError):
        src.node(1)  # loader raises: memo must stay on graph 0
    assert torch.equal(src.node(1), torch.as_tensor(nf[1]))  # retry re-loads graph 1 (not stale graph 0)


def test_graph_stream_assertion_kwarg():
    """T-GSTREAM-11: fit(stream=...) asserts caller intent against the container type."""
    nf, ed, y = _graphs(n=12)
    mg = AllGraph(verbose=False, seed=0, width=8, depth=1, epochs=2)
    with pytest.raises(ValueError):
        mg.fit(AllData.graphs(nf, ed, y=y), n_out=2, stream=True)  # resident, demanded stream
    d = AllData.graph_stream(InMemoryGraphSource(nf, edges=ed), y=y, kind_hint="graph")
    with pytest.raises(ValueError):
        mg.fit(d, n_out=2, stream=False)  # stream, forbade it
    r = mg.fit(d, n_out=2, stream=True)
    assert r["contract"] == "graph"

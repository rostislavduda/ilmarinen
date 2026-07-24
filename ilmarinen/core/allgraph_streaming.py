"""Opt-in dataset streaming for AllGraph's dense contracts (sequence / spatial / volumetric / 4d).

The training loops in ``allgraph.py`` minibatch over a dataset that is fully resident in host RAM and, for
the dense contracts, moved onto the compute device ONCE (``_train_dense``: ``Xd = X.to(self.device)``). That
bounds the trainable dataset size by device memory. This module adds an OPT-IN way to train on data that is
larger than RAM / VRAM by feeding one minibatch at a time from a lazy, out-of-core source.

DESIGN (see also the class docstrings):
  * Streaming is activated purely by CONTAINER TYPE: it is on iff ``data.dense`` is a :class:`DenseSource`.
    Build the input with :meth:`AllData.dense_stream` to opt in; every existing constructor keeps the exact
    in-memory code path, so resident behaviour and performance are unchanged (the entire overhead is one
    ``isinstance`` per fit).
  * A :class:`DenseSource` is MAP-STYLE (random-access, index-addressable), NOT an iterable / shuffle-buffer.
    Random access is required precisely so the existing index-driven training loop, the seeded
    ``torch.randperm`` shuffle order, and the seeded validation split reproduce the resident path bit-for-bit
    (the streaming deploy fit trains to the SAME weights as the equivalent resident fit).
  * This is a FIRST-CUT capability. It covers the deployed dense fit with ``select in {'argmax','sparse'}``,
    ``auto_epoch`` and ``readout_select``. The dataset-re-reading options (``select_size``, ``select='gibbs'``,
    ``tiebreak``, the priced-* / report_* diagnostics) are guarded off under streaming in ``fit`` with a clear
    error; relational / operator streaming is deferred. See the module note in ``allgraph.py``.

This is DISTINCT from two other "stream" usages in the package: the recurrent per-timestep streaming API in
``models/schema.py`` (online sequence inference) and the URL download-streaming in ``core/data_sources.py``.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
import torch


# --------------------------------------------------------------------------- bounded LRU cache
class _LRUCache:
    """A reusable bounded LRU keyed on the pure sample id, used by the disk-backed sources to avoid re-reading
    a sample fetched again within a fit. Its defining invariant: the value stored for a key is a function of the
    key ALONE (the read-only backing bytes), so a returned value is byte-identical whether it came from a hit or
    a miss, regardless of capacity or eviction order. Because outputs are independent of cache state, the cache
    can NEVER change a source's returned tensor -- hence never the shuffle order, batch membership, or trained
    weights (bit-identity to a resident/uncached fit is preserved at any capacity).

    `capacity` is a total-cost budget: pass cost=1 to bound by COUNT (dense rows / graphs), or cost=nbytes to
    bound by BYTES (operator fields, which are individually large). `capacity <= 0` disables the cache (every
    get misses, every put is a no-op). `threadsafe=True` guards get/put with a lock for the async-prefetch
    composition (item 3); off by default so the single-threaded deploy loop pays no lock cost."""

    __slots__ = ("_cap", "_od", "_cost", "_total", "_lock")

    def __init__(self, capacity, threadsafe=False):
        self._cap = max(0, int(capacity))
        self._od = OrderedDict()
        self._cost = {}
        self._total = 0
        self._lock = threading.Lock() if threadsafe else None

    def get(self, key):
        if self._cap <= 0:
            return None
        if self._lock is not None:
            with self._lock:
                return self._get(key)
        return self._get(key)

    def _get(self, key):
        v = self._od.get(key)
        if v is not None:
            self._od.move_to_end(key)
        return v

    def put(self, key, value, cost=1):
        if self._cap <= 0 or cost > self._cap:
            return                                       # a single item larger than the whole budget is not stored
        if self._lock is not None:
            with self._lock:
                self._put(key, value, cost)
        else:
            self._put(key, value, cost)

    def _put(self, key, value, cost):
        if key in self._od:
            self._total -= self._cost[key]
        self._od[key] = value
        self._od.move_to_end(key)
        self._cost[key] = cost
        self._total += cost
        while self._total > self._cap and len(self._od) > 1:
            old, _ = self._od.popitem(last=False)
            self._total -= self._cost.pop(old)


# --------------------------------------------------------------------------- id coercion
def _as_id_array(ids):
    """Coerce a batch-index selector (numpy array, list, or torch tensor) to a contiguous int64 numpy array.
    The training loop hands a torch permutation slice; the eval / val paths hand numpy ``arange`` slices."""
    if isinstance(ids, torch.Tensor):
        return ids.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.ascontiguousarray(np.asarray(ids, dtype=np.int64))


# --------------------------------------------------------------------------- the source protocol
class DenseSource:
    """Map-style (random-access, index-addressable) out-of-core backing for ``AllData.dense``.

    A ``DenseSource`` stands in for the full ``(n, *sample_shape)`` dense tensor WITHOUT ever materializing it:
    the training loop pulls one minibatch at a time via :meth:`get`. Random access -- not an iterable or a
    shuffle-buffer -- is required so the existing index-driven ``_run_epochs`` loop, the ``torch.randperm``
    shuffle order, and the seeded ``_auto_val_split`` all reproduce the resident path bit-for-bit.

    Subclasses implement ``__len__`` and :meth:`get`, and set ``_sample_shape`` (the per-sample shape, WITHOUT
    the leading n) and ``dtype`` in their constructor. :attr:`shape` / :meth:`dim` are then derived so the
    contract builders (which read ``X.shape[1]`` / ``X.shape[-1]`` etc.) see the full shape as metadata,
    never materializing n rows.

    The single hot-path contract for :meth:`get`:
      * takes an integer id selector (numpy / list / torch), possibly UNSORTED and of arbitrary length;
      * returns one ``torch.FloatTensor`` of shape ``(len(ids), *sample_shape)`` on CPU, in float32, with the
        rows IN THE REQUESTED ORDER (so ``source.get(ids)`` equals a resident ``X[ids]``);
      * is RNG-FREE (pure indexing) -- it must not touch the global numpy / torch RNG, or it would desync the
        deploy fit's shuffle order and break bit-for-bit reproducibility.
    """

    _sample_shape: tuple = ()
    dtype = np.float32

    def __len__(self):                                   # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def shape(self):
        """Full ``(n, *sample_shape)`` shape, derived from ``__len__`` and ``_sample_shape``. Metadata only --
        reading it never materializes rows."""
        return (len(self),) + tuple(self._sample_shape)

    def dim(self):
        """Rank of the full tensor (``1 + len(sample_shape)``), mirroring ``torch.Tensor.dim()``."""
        return 1 + len(self._sample_shape)

    def get(self, ids):                                  # pragma: no cover - abstract
        raise NotImplementedError


class InMemoryDenseSource(DenseSource):
    """A :class:`DenseSource` backed by an in-RAM ndarray / tensor.

    This does NOT save memory -- it exists so the full streaming code path (a ``get`` per minibatch, the
    incremental eval, the guards) can be exercised in tests and small demos, and so streaming can be checked
    for bit-for-bit equivalence against the resident fit over identical bytes."""

    def __init__(self, array):
        arr = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else np.asarray(array)
        self._arr = np.ascontiguousarray(arr)
        self._sample_shape = tuple(self._arr.shape[1:])
        self.dtype = self._arr.dtype

    def __len__(self):
        return int(self._arr.shape[0])

    def get(self, ids):
        idx = _as_id_array(ids)
        return torch.as_tensor(np.ascontiguousarray(self._arr[idx]), dtype=torch.float32)


class MemmapDenseSource(DenseSource):
    """A :class:`DenseSource` over a ``.npy`` file (or an existing ``np.memmap``), read lazily so the array is
    never fully resident -- the OS page cache serves the minibatches actually touched.

    Fancy indexing over a memmap is fastest with ascending indices (locality), but the per-epoch shuffle hands
    UNSORTED ids. :meth:`get` therefore gathers with SORTED ids (one near-sequential read) and then unpermutes
    back to the requested order, so the returned rows still match a resident ``X[ids]`` exactly."""

    def __init__(self, path_or_memmap, mmap_mode="r", cache_size=0, cache_threadsafe=False):
        if isinstance(path_or_memmap, np.memmap):
            self._mm = path_or_memmap
        elif isinstance(path_or_memmap, np.ndarray):
            self._mm = path_or_memmap                    # already an array-like; indexed lazily if it is a memmap
        else:
            self._mm = np.load(path_or_memmap, mmap_mode=mmap_mode)
        if self._mm.ndim < 1:
            raise ValueError("MemmapDenseSource expects at least a 1-D array (n, ...); got a scalar.")
        self._sample_shape = tuple(self._mm.shape[1:])
        self.dtype = self._mm.dtype
        # optional bounded LRU of up to `cache_size` native-dtype rows (count budget); off by default. Cached
        # rows are byte-identical to a fresh read, so the cache never changes get()'s output (bit-identity holds).
        self._cache = _LRUCache(cache_size, cache_threadsafe)

    def __len__(self):
        return int(self._mm.shape[0])

    def get(self, ids):
        idx = _as_id_array(ids)
        if self._cache._cap <= 0:
            order = np.argsort(idx, kind="stable")       # ascending read order for memmap locality
            gathered = np.asarray(self._mm[idx[order]])  # contiguous-ish read
            out = np.empty_like(gathered)
            out[order] = gathered                        # unpermute: out[k] is the row for idx[k]
            return torch.as_tensor(np.ascontiguousarray(out), dtype=torch.float32)
        # cached path: serve hits from the LRU, read only the missing ids from the memmap (sorted for locality)
        out = np.empty((len(idx),) + tuple(self._sample_shape), dtype=self.dtype)
        miss = []
        for k, i in enumerate(idx):
            row = self._cache.get(int(i))
            if row is None:
                miss.append(k)
            else:
                out[k] = row
        if miss:
            miss = np.asarray(miss)
            mids = idx[miss]
            order = np.argsort(mids, kind="stable")
            gathered = np.asarray(self._mm[mids[order]])
            for j, k in enumerate(miss[order]):
                row = np.array(gathered[j])              # COPY (not a view into `gathered`): a view would pin
                out[k] = row                             # the whole minibatch buffer, blowing the cache budget
                self._cache.put(int(idx[k]), row, cost=1)
        return torch.as_tensor(np.ascontiguousarray(out), dtype=torch.float32)


# --------------------------------------------------------------------------- grid rank fix-up
class _GridView(DenseSource):
    """Wraps a :class:`DenseSource` with the deterministic per-sample rank fix-up that ``_as_grid`` applies to
    a resident tensor -- inserting a channel axis when the sample lacks one -- applied inside :meth:`get` per
    batch. Exposes the TRANSFORMED :attr:`shape` / :meth:`dim` so ``build_fn(X, ...)`` reads channel count and
    grid extent as metadata without materializing n rows.

    Mirrors the resident ``_as_grid``: a source presenting rank-``rank`` samples (channel included) passes
    through; a source presenting rank-``rank-1`` samples gets a size-1 channel inserted at axis 0 of the
    sample (axis 1 of the batch). The flat-vector -> latent-lattice reshape branch of ``_as_grid`` is NOT
    supported under streaming (it needs a discovered lattice shape, which the required ``kind_hint`` bypasses);
    such a source raises with an actionable message."""

    def __init__(self, source, rank):
        self._src = source
        self._rank = rank
        sdim = source.dim()
        if sdim == rank:
            self._insert_channel = False
            self._sample_shape = tuple(source.shape[1:])
        elif sdim == rank - 1:
            self._insert_channel = True
            self._sample_shape = (1,) + tuple(source.shape[1:])
        else:
            raise NotImplementedError(
                f"streaming DenseSource presents rank-{sdim} samples but the rank-{rank} grid contract expects "
                f"samples of rank {rank - 1} (grid only) or {rank} (channel + grid). Reshape the source so each "
                f"sample is (C, *grid) or (*grid); flat-vector reshape-to-latent-lattice is not supported under "
                f"streaming (pass a kind_hint and pre-shaped samples).")
        self.dtype = getattr(source, "dtype", np.float32)

    def __len__(self):
        return len(self._src)

    def get(self, ids):
        t = self._src.get(ids)
        if self._insert_channel:
            t = t.unsqueeze(1)
        return t.contiguous()


# --------------------------------------------------------------------------- incremental scorer
class _StreamMetric:
    """Incremental scorer matching :meth:`AllGraph._metric` exactly, but fed prediction CHUNKS so the full
    output is never held in RAM.

    Classification: a running correct / total (integer-exact, chunking-invariant). Regression: a running
    residual sum of squares, with the total sum of squares computed ONCE from the RESIDENT labels (labels are
    small even when the features are out-of-core). Chunks must partition the full sample set in order, which
    the sequential ``arange`` eval guarantees; then ``sum_chunks == whole-array`` up to fp associativity, so a
    single-chunk eval (eval_bs >= n) is bit-identical to the resident metric and a multi-chunk eval matches to
    fp tolerance."""

    def __init__(self, task, y_resident):
        self.task = task
        y = np.asarray(y_resident)
        self._n = 0
        if task == "classification":
            self._correct = 0
        else:
            ybar = float(y.mean()) if y.size else 0.0
            self._ss_tot = float(((y - ybar) ** 2).sum())
            self._ss_res = 0.0

    def update(self, out_chunk, y_chunk):
        """Accumulate one chunk. ``out_chunk`` is a CPU tensor of this chunk's raw net outputs (logits or
        values); ``y_chunk`` the matching resident labels."""
        y = np.asarray(y_chunk)
        if self.task == "classification":
            pred = out_chunk.argmax(1).numpy()
            self._correct += int((pred == y).sum())
        else:
            pred = out_chunk.squeeze(-1).numpy()
            self._ss_res += float(((pred - y) ** 2).sum())
        self._n += int(len(y))

    def result(self):
        """Return ``(metric_name, value)`` identical to :meth:`AllGraph._metric`."""
        if self.task == "classification":
            return "acc", float(self._correct / max(self._n, 1))
        return "R2", float(1 - self._ss_res / (self._ss_tot + 1e-12))


# --------------------------------------------------------------------------- selection subsample
def _reservoir_ids(n, cap, seed):
    """Return up to ``cap`` sorted, unique sample ids drawn WITHOUT replacement from ``range(n)`` using a
    DEDICATED ``np.random.RandomState(seed)`` that touches neither the global numpy nor the torch RNG stream
    (so drawing the search subsample cannot desync the deploy fit's shuffle order). Ids are sorted so the
    backing :meth:`DenseSource.get` reads in ascending, cache-friendly order; a search subsample is a random
    SUBSET, so its internal order does not matter."""
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    ids = rng.choice(n, size=cap, replace=False)
    ids.sort()
    return ids.astype(np.int64, copy=False)


# =========================================================================== relational streaming
# The dense contracts hold one big tensor; the relational contracts (graph / equivariant / set) hold a LIST of
# variable-size samples -- per graph a node-feature matrix, optional edge_index, optional 3D positions -- which
# the resident path pre-converts to tensors ONCE (_prepare_batch_cache) and keeps for the whole fit. A
# GraphSource replaces that full in-RAM materialization: it yields ONE sample's tensors on demand, so the
# collation loop (_assemble_batch) fetches per minibatch instead of indexing a resident cache. Because the
# relational path already shuffles with np.random.permutation (numpy global RNG) and collates per-sample on
# CPU before moving to device, streaming here changes ONLY the tensor SOURCE (source vs cache): the shuffle,
# the batch membership, and the forward are untouched, so a streaming relational fit trains to the SAME weights
# as the resident fit. Per-graph outputs are tiny (one row per graph), so the eval needs no incremental scorer.
class GraphSource:
    """Map-style (random-access) out-of-core backing for a batch of variable-size graphs / point sets.

    Stands in for the resident ``node_feats`` / ``edges`` / ``positions`` lists WITHOUT materializing them: the
    relational collation pulls one sample at a time via :meth:`node` / :meth:`edge` / :meth:`pos`. Random access
    is required so the seeded ``np.random.permutation`` shuffle and the seeded ``_auto_val_split`` reproduce the
    resident path bit-for-bit.

    Subclasses set the metadata attributes ``n_in`` (node-feature dimension), ``has_edges``, ``has_pos`` in
    their constructor and implement ``__len__`` plus the three accessors. The accessor contract (mirrors what
    ``_prepare_batch_cache`` would have built, so streaming == resident):
      * ``node(i)`` -> ``FloatTensor`` of shape ``(n_i, n_in)`` on CPU;
      * ``edge(i)`` -> ``LongTensor`` of shape ``(2, |E_i|)`` on CPU, or ``None`` when ``has_edges`` is False;
      * ``pos(i)``  -> ``FloatTensor`` of shape ``(n_i, 3)`` on CPU, or ``None`` when ``has_pos`` is False;
      * all accessors are RNG-FREE (pure fetch) and deterministic in ``i``.
    """

    n_in: int = 0
    has_edges: bool = False
    has_pos: bool = False

    def __len__(self):                                   # pragma: no cover - abstract
        raise NotImplementedError

    def node(self, i):                                   # pragma: no cover - abstract
        raise NotImplementedError

    def edge(self, i):
        return None

    def pos(self, i):
        return None


class InMemoryGraphSource(GraphSource):
    """A :class:`GraphSource` backed by in-RAM lists / arrays (the same objects ``AllData.graphs`` /
    ``point_sets`` take). Does not save memory -- it exercises the full relational streaming path (a fetch per
    minibatch) in tests and small demos, and lets streaming be checked for bit-for-bit equivalence against the
    resident relational fit over identical data."""

    def __init__(self, node_feats, edges=None, positions=None):
        self._nf = node_feats
        self._ed = edges
        self._pos = positions
        self.n_in = int(np.asarray(node_feats[0]).shape[1])
        self.has_edges = edges is not None
        self.has_pos = positions is not None

    def __len__(self):
        return len(self._nf)

    def node(self, i):
        return torch.as_tensor(np.asarray(self._nf[i]), dtype=torch.float32)

    def edge(self, i):
        return torch.as_tensor(np.asarray(self._ed[i]), dtype=torch.long) if self.has_edges else None

    def pos(self, i):
        return torch.as_tensor(np.asarray(self._pos[i]), dtype=torch.float32) if self.has_pos else None


class LazyGraphSource(GraphSource):
    """A general-purpose lazy :class:`GraphSource`: ``loader(i)`` returns a dict for graph ``i`` with keys
    ``'node'`` (required) and optionally ``'edge'`` / ``'pos'`` (array-likes). Back it with anything
    random-access -- a directory of per-graph ``.npy`` files, an HDF5 group per graph, a database row, an object
    store -- and the collation streams one graph at a time.

    ``n_in`` / ``has_edges`` / ``has_pos`` are declared up front (metadata) so the builder never fetches a
    sample to learn the shapes. A bounded LRU of ``cache_size`` graphs makes the three accessors for the SAME
    ``i`` (called consecutively by the collation loop) share a single ``loader(i)`` call; the default
    ``cache_size=1`` is exactly the original 1-element memo (behaviour + perf), ``cache_size=0`` disables
    caching, and ``cache_size > 1`` keeps the K most-recently-loaded graphs to skip re-reads across epochs.
    Cached dicts are byte-identical to a fresh load, so the cache never changes an accessor's output.
    ``cache_threadsafe=True`` guards the cache for the async-prefetch composition (item 3)."""

    def __init__(self, loader, n, n_in, has_edges=False, has_pos=False, cache_size=1, cache_threadsafe=False):
        self._loader = loader
        self._n = int(n)
        self.n_in = int(n_in)
        self.has_edges = bool(has_edges)
        self.has_pos = bool(has_pos)
        self._cache = _LRUCache(cache_size, cache_threadsafe)

    def __len__(self):
        return self._n

    def _load(self, i):
        v = self._cache.get(i)
        if v is not None:
            return v
        # load THEN commit: if loader(i) raises (a transient I/O / object-store / DB failure on the out-of-core
        # backends this targets), nothing is cached, so a retry of the same i re-attempts the load instead of
        # silently returning a previously loaded graph's arrays.
        loaded = self._loader(i)
        self._cache.put(i, loaded, cost=1)
        return loaded

    def node(self, i):
        return torch.as_tensor(np.asarray(self._load(i)["node"]), dtype=torch.float32)

    def edge(self, i):
        return torch.as_tensor(np.asarray(self._load(i)["edge"]), dtype=torch.long) if self.has_edges else None

    def pos(self, i):
        return torch.as_tensor(np.asarray(self._load(i)["pos"]), dtype=torch.float32) if self.has_pos else None


# =========================================================================== operator streaming
# The neural-operator contract maps an input field a(x) to a target field u(x) on a grid; the loss is a
# per-grid-point field MSE, so BOTH the input a and the TARGET u are field-valued and large (unlike every other
# contract, whose target y is small). The resident path moves the whole a/x/u onto the device at once and, for
# the field-R2, holds the whole prediction AND the whole target in memory with a single global mean. An
# OperatorSource streams a/x/u per minibatch; the eval field-R2 is computed in a STREAMED TWO-PASS (pass 1: the
# global field mean; pass 2: the residual and total sums of squares), since the resident-y ss_tot shortcut the
# dense/relational scorers rely on does not apply when the target is out-of-core. Determinism matches the
# relational path: np.random.permutation shuffle, per-batch collate, so a streamed operator fit trains to the
# same weights as the resident fit.
def _memmap_gather(mm, ids):
    """Gather rows `ids` from a memmap/ndarray with a sorted read then unpermute back to requested order (see
    MemmapDenseSource.get). Returns an ndarray in the requested id order."""
    idx = np.asarray(ids, dtype=np.int64)
    order = np.argsort(idx, kind="stable")
    gathered = np.asarray(mm[idx[order]])
    out = np.empty_like(gathered)
    out[order] = gathered
    return out


def _infer_operator_sdims(a_shape, u_shape, spatial_dims):
    """Spatial rank of an operator dataset, mirroring AllData.functions: total dims minus the batch axis, minus
    a trailing channel axis when a carries one (a.ndim==u.ndim, >=3, small last axis != the preceding one)."""
    if spatial_dims is not None:
        return max(1, min(3, int(spatial_dims)))
    a_ndim, u_ndim = len(a_shape), len(u_shape)
    sdims = a_ndim - 1
    if a_ndim == u_ndim and a_ndim >= 3 and a_shape[-1] <= 4 and a_shape[-1] != a_shape[-2]:
        sdims = a_ndim - 2
    return max(1, min(3, sdims))


def _default_operator_grid(a_shape, spatial_dims):
    """The default uniform [0,1]^d meshgrid (*grid, sdims), built with torch exactly as AllData.functions does,
    so a grid=None OperatorSource is bit-identical to the resident default grid."""
    grid_shape = a_shape[1:1 + spatial_dims]
    axes = [torch.linspace(0.0, 1.0, int(s)) for s in grid_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1)                     # (*grid, sdims) torch float32


class OperatorSource:
    """Map-style (random-access) out-of-core backing for a neural-operator dataset: input fields a, target
    fields u, and grid coordinates x. Stands in for AllData.functions' resident a / grid / y WITHOUT
    materializing them: training and eval pull one minibatch of fields at a time via :meth:`a` / :meth:`grid` /
    :meth:`u`.

    Metadata attributes (set in the constructor, read as shape without materializing n samples): ``spatial_dims``
    (1/2/3) and ``a_shape`` (full ``(n, *grid[, c])``). Accessor contract (batch fetch, since operator samples
    are fixed-size on a common grid), returning CPU float32 tensors in the requested id order, RNG-free:
      * ``a(ids)``    -> ``(len(ids), *grid[, c])`` input fields;
      * ``grid(ids)`` -> ``(len(ids), *grid, spatial_dims)`` coordinates (a shared grid is broadcast per batch);
      * ``u(ids)``    -> ``(len(ids), *grid[, c])`` target fields.
    """

    spatial_dims: int = 1
    a_shape: tuple = ()
    _grid_single = None                                  # (*grid, sdims) torch tensor, broadcast per batch
    _grid_full = None                                    # (n, *grid, sdims) ndarray, indexed per batch

    def __len__(self):                                   # pragma: no cover - abstract
        raise NotImplementedError

    def _a_raw(self, ids):                               # pragma: no cover - abstract
        raise NotImplementedError

    def _u_raw(self, ids):                               # pragma: no cover - abstract
        raise NotImplementedError

    def _setup_grid(self, grid):
        """Store either a single shared grid (grid=None -> the default meshgrid, broadcast per batch, so n copies
        are never resident) or an explicit per-sample grid of shape (n, *grid, sdims)."""
        if grid is None:
            self._grid_single = _default_operator_grid(self.a_shape, self.spatial_dims)
            self._grid_full = None
        else:
            g = grid.detach().cpu().numpy() if isinstance(grid, torch.Tensor) else np.asarray(grid, dtype=np.float32)
            if g.shape[0] != self.a_shape[0]:
                raise ValueError(
                    f"explicit operator grid must have shape (n, *grid, spatial_dims) with n={self.a_shape[0]}; "
                    f"got leading dim {g.shape[0]}. Pass grid=None to broadcast one shared grid.")
            self._grid_full = np.ascontiguousarray(g)
            self._grid_single = None

    def a(self, ids):
        return torch.as_tensor(np.ascontiguousarray(self._a_raw(ids)), dtype=torch.float32)

    def u(self, ids):
        return torch.as_tensor(np.ascontiguousarray(self._u_raw(ids)), dtype=torch.float32)

    def grid(self, ids):
        if self._grid_single is not None:
            g = self._grid_single
            return g.unsqueeze(0).expand(len(ids), *g.shape).contiguous()
        return torch.as_tensor(np.ascontiguousarray(self._grid_full[np.asarray(ids)]), dtype=torch.float32)


class InMemoryOperatorSource(OperatorSource):
    """An :class:`OperatorSource` backed by in-RAM ndarrays / tensors (same a / u / grid as AllData.functions).
    Exercises the full operator streaming path (per-minibatch fetch + streamed field-R2) in tests and demos."""

    def __init__(self, a, u, grid=None, spatial_dims=None):
        self._a = np.ascontiguousarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a))
        self._u = np.ascontiguousarray(u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else np.asarray(u))
        if self._a.shape[0] != self._u.shape[0]:
            raise ValueError(f"a and u must have the same number of samples; got {self._a.shape[0]} and "
                             f"{self._u.shape[0]}.")
        self.a_shape = self._a.shape
        self.spatial_dims = _infer_operator_sdims(self._a.shape, self._u.shape, spatial_dims)
        self._setup_grid(grid)

    def __len__(self):
        return int(self._a.shape[0])

    def _a_raw(self, ids):
        return self._a[np.asarray(ids)]

    def _u_raw(self, ids):
        return self._u[np.asarray(ids)]


class MemmapOperatorSource(OperatorSource):
    """An :class:`OperatorSource` over ``.npy`` files (or existing memmaps) for the input and target fields, read
    lazily via ``np.memmap`` so neither the a nor the u array is ever fully resident. Fields are gathered with a
    sorted read then unpermuted to the requested order. ``grid`` is a shared array (or None for the default).

    An optional bounded LRU per field (``cache_bytes`` -- a BYTE budget, since operator fields are individually
    large) skips re-reading a field fetched again within a fit. The a and u caches are INDEPENDENT (different
    arrays, same ids -> separate keys). Cached rows are byte-identical to a fresh read, so caching never changes
    a returned field; ``cache_threadsafe=True`` guards them for async prefetch (item 3)."""

    def __init__(self, a_path, u_path, grid=None, spatial_dims=None, mmap_mode="r",
                 cache_bytes=0, cache_threadsafe=False):
        self._a = a_path if isinstance(a_path, (np.ndarray, np.memmap)) else np.load(a_path, mmap_mode=mmap_mode)
        self._u = u_path if isinstance(u_path, (np.ndarray, np.memmap)) else np.load(u_path, mmap_mode=mmap_mode)
        if self._a.shape[0] != self._u.shape[0]:
            raise ValueError(f"a and u must have the same number of samples; got {self._a.shape[0]} and "
                             f"{self._u.shape[0]}.")
        self.a_shape = tuple(self._a.shape)
        self.spatial_dims = _infer_operator_sdims(tuple(self._a.shape), tuple(self._u.shape), spatial_dims)
        self._setup_grid(grid)
        self._a_cache = _LRUCache(cache_bytes, cache_threadsafe)
        self._u_cache = _LRUCache(cache_bytes, cache_threadsafe)

    def __len__(self):
        return int(self._a.shape[0])

    @staticmethod
    def _gather_cached(mm, ids, cache):
        idx = _as_id_array(ids)
        if cache._cap <= 0:
            return _memmap_gather(mm, idx)
        out = np.empty((len(idx),) + tuple(mm.shape[1:]), dtype=mm.dtype)
        miss = []
        for k, i in enumerate(idx):
            row = cache.get(int(i))
            if row is None:
                miss.append(k)
            else:
                out[k] = row
        if miss:
            miss = np.asarray(miss)
            mids = idx[miss]
            order = np.argsort(mids, kind="stable")
            gathered = np.asarray(mm[mids[order]])
            for j, k in enumerate(miss[order]):
                row = np.array(gathered[j])              # COPY (a view would pin the whole field minibatch buffer)
                out[k] = row
                cache.put(int(idx[k]), row, cost=int(row.nbytes))   # BYTE budget: fields are large
        return out

    def _a_raw(self, ids):
        return self._gather_cached(self._a, ids, self._a_cache)

    def _u_raw(self, ids):
        return self._gather_cached(self._u, ids, self._u_cache)


# =========================================================================== forward-only iterable sources
# Everything above is MAP-STYLE (random-access): the training loop shuffles by index (torch.randperm /
# np.random.permutation) and holds out a seeded index split, which reproduces the resident fit bit-for-bit. A
# forward-only source has NO random access and no known length, so it cannot be index-shuffled or seed-split;
# it trains via a SEPARATE regime (a seeded windowed shuffle buffer + a hash-of-id train/val split) whose
# guarantee is DETERMINISTIC-GIVEN-SEED, explicitly NOT bit-identical to the map-style / resident fit. This is
# scoped to the dense contract family.
import hashlib                                            # noqa: E402 (kept next to its only user)

_ITER_SHUFFLE_SEED = 47                                   # offset for the per-epoch windowed-shuffle RandomState
_ITER_VAL_PERMILLE = 150                                  # ~15% of ids fall in the val bucket


def _iter_val_key(sample_id, seed):
    """A stable, process-independent, seed-dependent hash of a sample id for the iterable train/val split. Uses
    blake2b keyed on the seed (NOT Python's salted hash), so the split is identical across runs and processes."""
    h = hashlib.blake2b(str(sample_id).encode(), key=str(int(seed)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


class IterableDenseSource:
    """A FORWARD-ONLY (non-random-access) dense source: no ``__len__``, no ``get(ids)``. It exposes only
    per-sample shape metadata and a restartable, RNG-free ``__iter__`` yielding ``(sample_id, x_tensor, y)`` per
    sample. Use it for data that cannot be indexed (a network / one-pass reader).

    Opt in via :meth:`AllData.dense_iter`. GUARANTEE: deterministic given the seed (same seed + source ->
    same weights), but -- unlike the map-style DenseSource -- NOT bit-identical to the resident fit, because the
    windowed-shuffle order and the hash-based train/val partition differ by construction.

    ``__iter__`` contract: return a FRESH iterator over the SAME sequence every call (RESTARTABLE -- each epoch
    re-iterates), yielding ``(sample_id, x, y)`` where ``sample_id`` is a stable hashable id, ``x`` a fixed-shape
    sample tensor (float32-able), ``y`` the target. It must be RNG-FREE. Set ``_sample_shape`` / ``dtype`` /
    ``n_out`` (classification class count, since targets stream by and cannot be scanned) in the constructor."""

    _sample_shape: tuple = ()
    dtype = np.float32
    n_out = None

    @property
    def shape(self):
        return (None,) + tuple(self._sample_shape)

    def dim(self):
        return 1 + len(self._sample_shape)

    def __iter__(self):                                  # pragma: no cover - abstract
        raise NotImplementedError


class InMemoryIterableDenseSource(IterableDenseSource):
    """A parity vehicle for the iterable regime: yields ``(id, row_tensor, y[k])`` over an in-RAM array in index
    order, restartable. Real forward-only sources subclass IterableDenseSource with a streaming ``__iter__``."""

    def __init__(self, array, y, ids=None, n_out=None):
        arr = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else np.asarray(array)
        self._arr = np.ascontiguousarray(arr)
        self._y = np.asarray(y)
        self._ids = np.arange(len(self._arr)) if ids is None else np.asarray(ids)
        self._sample_shape = tuple(self._arr.shape[1:])
        self.dtype = self._arr.dtype
        self.n_out = n_out

    def __iter__(self):
        for k in range(len(self._arr)):
            yield int(self._ids[k]), torch.as_tensor(self._arr[k], dtype=torch.float32), self._y[k]


class _IterMetric:
    """One-pass scorer for the iterable regime, where the target is NOT resident (it streams by). Classification
    accumulates correct/total; regression accumulates ss_res AND the sums needed for a single-pass total sum of
    squares (ss_tot = sum_y2 - sum_y^2 / n), so no second pass over the stream is needed."""

    def __init__(self, task):
        self.task = task
        self._n = 0
        if task == "classification":
            self._correct = 0
        else:
            self._sum_y = 0.0
            self._sum_y2 = 0.0
            self._ss_res = 0.0

    def update(self, out_chunk, y_chunk):
        y = np.asarray(y_chunk)
        if self.task == "classification":
            self._correct += int((out_chunk.argmax(1).numpy() == y).sum())
        else:
            pred = out_chunk.squeeze(-1).numpy()
            self._ss_res += float(((pred - y) ** 2).sum())
            yd = y.astype(np.float64)
            self._sum_y += float(yd.sum())
            self._sum_y2 += float((yd ** 2).sum())
        self._n += int(len(y))

    def result(self):
        if self._n == 0:
            # no samples scored (empty or exhausted/non-restartable stream): report nan rather than a
            # fabricated perfect score (R2 would collapse to 1 - 0/1e-12 = 1.0; acc to 0.0).
            return ("acc" if self.task == "classification" else "R2"), float("nan")
        if self.task == "classification":
            return "acc", float(self._correct / self._n)
        ss_tot = self._sum_y2 - self._sum_y ** 2 / self._n
        return "R2", float(1 - self._ss_res / (ss_tot + 1e-12))

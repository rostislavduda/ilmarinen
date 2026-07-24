"""AllGraph -- the meta-router that unifies all 8 schemas into one entry point.

The 8 schemas each search WITHIN one computational contract (sequence, spatial, volumetric, 4d,
graph, equivariant, set, operator). They deliberately do NOT share an input type -- you cannot softmax-mix
a conv3d output with a graph-conv output the way primitives mix inside a schema, because the
contracts differ. So the meta-router is not a differentiable mixture over schemas; it is a
two-level DISPATCH that discovers which contract the data satisfies, then hands off to that
schema's own (differentiable) primitive/width/depth search.

This mirrors the project's core discipline exactly:
  * WITHIN a schema: primitives share one contract -> differentiable alpha-mix is valid.
  * ACROSS schemas: contracts differ -> hard structural dispatch, then per-schema search.
AllGraph is the outermost, cheapest rung of the same minimum-description-length ladder the analytical
notes develop: discover the coarsest structural fact first (what KIND of object is a sample), then
progressively finer ones (grid rank -> symmetry -> primitive -> width/depth).

TWO LEVELS
----------
Level 1 -- TYPE dispatch (rule-based, certain, overridable). Reads the DATA CONTAINER, not a learned
  signal, because the container already carries the answer:
    - has edge_index         -> 'graph'      (or 'equivariant' if it also carries 3D positions)
    - variable-size point set -> 'set'        (or 'equivariant' if positions + a detected symmetry)
    - fixed-size dense tensor -> dense-grid family -> Level 2
  The user can always override with kind_hint=...  (transparent, no guessing).

Level 2 -- GRID-RANK dispatch (existing machinery). For a dense tensor, pick the grid rank:
    - explicit tensor rank from the array shape (b,C,H,W)->spatial, (b,C,D,H,W)->volumetric,
      (b,C,T,D,H,W)->4d, (b,T,f)->sequence; OR
    - for a FLAT vector (b,d): discover_mode_structure -> 1d (sequence) / 2d (spatial) / unstructured.

The class exposes a single .fit(data, y) that routes, builds, trains, and reports.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager

from .mode_structure import discover_mode_structure
from .allgraph_types import _SweepCtx, _EDGE_NN_FACTOR
from .allgraph_reports import _ReportsMixin
from .allgraph_persistence import _PersistenceMixin
from .allgraph_selection import _SizeSelectionMixin
from .allgraph_contracts import _ContractFitMixin


# --------------------------------------------------------------------------- the input container
class AllData:
    """A uniform container that makes Level-1 type dispatch unambiguous. Callers wrap their data in one
    of the classmethods; the presence/absence of edges and positions IS the type signal.

    Fields (only the relevant ones are set per constructor):
      kind_hint   : optional explicit contract string (overrides detection)
      dense       : (n, ...) dense tensor  (grid/sequence/flat)  -- for the dense-grid family
      node_feats  : list of (n_i, f) per-sample node features    -- graph / equivariant / set
      positions   : list of (n_i, 3) per-sample 3D coords        -- equivariant only
      edges       : list of (2, |E_i|) per-sample edge_index      -- graph / equivariant (None for set)
      y           : targets
    """

    def __init__(self, kind_hint=None, dense=None, node_feats=None, positions=None, edges=None, y=None,
                 grid=None):
        self.kind_hint = kind_hint
        self.dense = dense
        self.node_feats = node_feats
        self.positions = positions
        self.edges = edges
        self.y = y
        self.grid = grid            # (n, N) sample coordinates for function-valued (operator) data

    # ---- ergonomic constructors (these encode the type in how you build the object) ----
    @classmethod
    def dense_tensor(cls, X, y=None, kind_hint=None):
        """A fixed-size dense array. Shape decides grid rank (see Level 2). Flat (n,d) -> mode detect."""
        X = X if isinstance(X, torch.Tensor) else torch.tensor(np.asarray(X), dtype=torch.float32)
        return cls(kind_hint=kind_hint, dense=X, y=y)

    @classmethod
    def dense_stream(cls, source, y=None, kind_hint=None):
        """A STREAMING dense input: `source` is a :class:`~ilmarinen.core.allgraph_streaming.DenseSource` that
        yields minibatches on demand, so the full (n, *sample_shape) tensor is NEVER materialized. Use this to
        train on datasets larger than host RAM / device memory (the dense contracts otherwise move the whole
        tensor onto the compute device at once).

        Opt-in is by THIS constructor alone: building via dense_tensor / graphs / point_sets / functions keeps
        the exact in-memory code path, so resident behaviour and performance are unchanged. `kind_hint` is
        REQUIRED and must name a dense contract ('sequence' | 'spatial' | 'volumetric' | '4d') so routing never
        falls back to the flat-vector mode-discovery pass (which would materialize the whole matrix). `y`
        (targets) stays resident -- labels are small even when the features are out-of-core.

        FIRST CUT: streaming covers the deployed dense fit with select in {'argmax','sparse'}, auto_epoch, and
        readout_select. select_size / select='gibbs' / tiebreak / the priced-* and report_* diagnostics are not
        yet supported under streaming and raise a clear error in :meth:`AllGraph.fit`. Note readout_select is
        supported by drawing ONE bounded resident subsample (up to stream_subsample_cap samples) for the readout
        bake-off, so it relaxes the strict single-minibatch RAM footprint of the pure deploy fit -- lower
        stream_subsample_cap (or leave readout_select off) if that peak allocation is a concern."""
        from .allgraph_streaming import DenseSource
        if not isinstance(source, DenseSource):
            raise TypeError(
                f"dense_stream(source=...) expects a DenseSource (e.g. MemmapDenseSource over a .npy file); "
                f"got {type(source).__name__}. See ilmarinen.core.allgraph_streaming.")
        if kind_hint is None:
            raise ValueError(
                "dense_stream requires an explicit kind_hint in {'sequence','spatial','volumetric','4d'} so "
                "routing never materializes the source; pass e.g. kind_hint='spatial'.")
        if kind_hint not in _DENSE_CONTRACTS:
            raise ValueError(
                f"dense_stream kind_hint must be a dense contract {sorted(_DENSE_CONTRACTS)}; got {kind_hint!r}. "
                f"For relational data use AllData.graph_stream(...).")
        return cls(kind_hint=kind_hint, dense=source, y=y)

    @classmethod
    def graph_stream(cls, source, y=None, kind_hint=None):
        """A STREAMING relational input: `source` is a :class:`~ilmarinen.core.allgraph_streaming.GraphSource`
        that yields one graph's (node features, optional edges, optional 3D positions) on demand, so the whole
        list of variable-size graphs is NEVER materialized. Use this to train the relational contracts on graph
        datasets larger than host RAM (the resident path pre-converts every graph to tensors up front).

        Opt-in is by THIS constructor alone (the source is stored as `node_feats`, the field type dispatch reads,
        so resident lists keep their exact path). `kind_hint` is REQUIRED and must name a relational contract:
          * 'graph'       -- topology only; the source must carry edges (has_edges).
          * 'equivariant' -- geometric graph; the source must carry BOTH edges and positions.
          * 'set'         -- unordered point set; edges are ignored.
        `y` (targets) stays resident. Per-graph outputs are tiny, so the whole relational streaming path adds no
        per-node materialization.

        FIRST CUT: streaming covers the deployed relational fit with select in {'argmax','sparse'} and
        auto_epoch. select_size / select='gibbs' / tiebreak / angular_from_data / the priced-* and report_*
        diagnostics are not yet supported under streaming and raise a clear error in :meth:`AllGraph.fit`."""
        from .allgraph_streaming import GraphSource
        if not isinstance(source, GraphSource):
            raise TypeError(
                f"graph_stream(source=...) expects a GraphSource (e.g. InMemoryGraphSource / LazyGraphSource); "
                f"got {type(source).__name__}. See ilmarinen.core.allgraph_streaming.")
        if kind_hint is None:
            raise ValueError(
                "graph_stream requires an explicit kind_hint in {'graph','equivariant','set'}; pass e.g. "
                "kind_hint='graph'.")
        if kind_hint not in _IRREGULAR_CONTRACTS:
            raise ValueError(
                f"graph_stream kind_hint must be a relational contract {sorted(_IRREGULAR_CONTRACTS)}; got "
                f"{kind_hint!r}. For dense data use AllData.dense_stream(...).")
        if kind_hint in ("graph", "equivariant") and not source.has_edges:
            raise ValueError(
                f"kind_hint={kind_hint!r} needs edges, but the GraphSource reports has_edges=False. Provide a "
                f"source that yields edge_index per graph, or use kind_hint='set'.")
        if kind_hint == "equivariant" and not source.has_pos:
            raise ValueError(
                "kind_hint='equivariant' needs 3D positions, but the GraphSource reports has_pos=False.")
        return cls(kind_hint=kind_hint, node_feats=source, y=y)

    @classmethod
    def functions_stream(cls, source, kind_hint="operator"):
        """A STREAMING neural-operator input: `source` is an
        :class:`~ilmarinen.core.allgraph_streaming.OperatorSource` that yields per-minibatch input fields a(x),
        grid coordinates x, and target fields u(x) on demand, so neither the (large, field-valued) inputs NOR
        the (large, field-valued) targets are ever fully resident. Use this to train the operator contract on
        function-to-function datasets larger than RAM. Unlike every other contract, the operator TARGET is
        field-valued, so its streamed field-R2 is computed in a two-pass accumulation (global field mean, then
        residual/total sums of squares) rather than from a resident target.

        `kind_hint` must be 'operator'. Opt-in is by this constructor alone (the source is stored as `dense`).

        FIRST CUT: streaming covers the deployed operator fit with select in {'argmax','sparse'} and auto_epoch.
        price_modes (mode-budget selection) and the priced-* / report_* diagnostics re-read the fields and are
        not yet supported under streaming -- they raise a clear error in :meth:`AllGraph.fit`."""
        from .allgraph_streaming import OperatorSource
        if not isinstance(source, OperatorSource):
            raise TypeError(
                f"functions_stream(source=...) expects an OperatorSource (e.g. InMemoryOperatorSource / "
                f"MemmapOperatorSource); got {type(source).__name__}. See ilmarinen.core.allgraph_streaming.")
        if kind_hint != "operator":
            raise ValueError(f"functions_stream kind_hint must be 'operator'; got {kind_hint!r}.")
        obj = cls(kind_hint="operator", dense=source)
        obj.spatial_dims = source.spatial_dims
        return obj

    @classmethod
    def dense_iter(cls, source, kind_hint=None, n_out=None):
        """A FORWARD-ONLY (non-random-access) dense input: `source` is an
        :class:`~ilmarinen.core.allgraph_streaming.IterableDenseSource` that only yields ``(id, x, y)`` per
        sample (no indexing, no length). Use it for data that cannot be indexed/shuffled by position.

        Because the map-style loop needs random access + a known length, an iterable source trains via a
        SEPARATE regime (a seeded windowed shuffle buffer + a hash-based train/val split). GUARANTEE:
        deterministic given the seed, but -- unlike AllData.dense_stream -- NOT bit-identical to the resident
        fit (the shuffle order and the val partition differ by construction). `kind_hint` must be a dense
        contract; `n_out` (class count) is REQUIRED for classification since the targets stream by and cannot be
        scanned. Selection (select_size / gibbs / tiebreak / readout_select) is not available here (a forward-only
        source cannot draw a random-access subsample); select must be 'argmax' or 'sparse'."""
        from .allgraph_streaming import IterableDenseSource
        if not isinstance(source, IterableDenseSource):
            raise TypeError(
                f"dense_iter(source=...) expects an IterableDenseSource; got {type(source).__name__}. See "
                f"ilmarinen.core.allgraph_streaming.")
        if kind_hint is None:
            raise ValueError("dense_iter requires an explicit kind_hint in "
                             "{'sequence','spatial','volumetric','4d'}; pass e.g. kind_hint='spatial'.")
        if kind_hint not in _DENSE_CONTRACTS:
            raise ValueError(f"dense_iter kind_hint must be a dense contract {sorted(_DENSE_CONTRACTS)}; got "
                             f"{kind_hint!r}.")
        if n_out is not None:
            source.n_out = int(n_out)
        return cls(kind_hint=kind_hint, dense=source, y=None)

    @classmethod
    def graphs(cls, node_feats, edges, y=None, positions=None):
        """A batch of graphs (nodes + edges). If positions is given, this is an equivariant graph."""
        return cls(kind_hint=None, node_feats=node_feats, edges=edges, positions=positions, y=y)

    @classmethod
    def point_sets(cls, node_feats, y=None, positions=None):
        """A batch of unordered point sets (no edges). positions optional (geometric sets)."""
        return cls(kind_hint=None, node_feats=node_feats, positions=positions, edges=None, y=y)

    @classmethod
    def functions(cls, a, y, grid=None, spatial_dims=None):
        """A batch of FUNCTION-valued samples for the neural-operator contract: input fields a(x) sampled on
        a grid, target fields y=u(x) on the same grid. Supports 1D/2D/3D:
          1D: a,y are (n, N)          [or (n, N, c)]
          2D: a,y are (n, H, W)       [or (n, H, W, c)]
          3D: a,y are (n, D, H, W)    [or (n, D, H, W, c)]
        `grid` is (n, *shape, spatial_dims) coordinates (defaults to a uniform [0,1]^d meshgrid). The
        spatial rank is inferred from the array shape unless given. The operator maps a -> u and is
        discretization-invariant. kind_hint='operator' routes to the operator schema."""
        a = a if isinstance(a, torch.Tensor) else torch.tensor(np.asarray(a), dtype=torch.float32)
        y = y if isinstance(y, torch.Tensor) else torch.tensor(np.asarray(y), dtype=torch.float32)
        n = a.shape[0]
        # infer spatial rank: total dims minus batch, minus a trailing channel axis if the last two match a<->y
        if spatial_dims is None:
            # a scalar field has a.dim() == 1 + sdims; a vector field has 1 + sdims + 1
            # disambiguate by comparing against y (same spatial shape); assume scalar unless a has an extra
            # trailing axis that y lacks or that is small (<=4)
            sdims = a.dim() - 1
            if a.dim() == y.dim() and a.dim() >= 3 and a.shape[-1] <= 4 and a.shape[-1] != a.shape[-2]:
                sdims = a.dim() - 2                       # trailing channel axis
            spatial_dims = max(1, min(3, sdims))
        grid_shape = a.shape[1:1 + spatial_dims]
        if grid is None:
            axes = [torch.linspace(0.0, 1.0, int(s)) for s in grid_shape]
            mesh = torch.meshgrid(*axes, indexing="ij")
            coords = torch.stack(mesh, dim=-1)                       # (*shape, sdims)
            grid = coords.unsqueeze(0).expand(n, *coords.shape).contiguous()
        else:
            grid = grid if isinstance(grid, torch.Tensor) else torch.tensor(np.asarray(grid), dtype=torch.float32)
        obj = cls(kind_hint="operator", dense=a, grid=grid, y=y)
        obj.spatial_dims = spatial_dims
        return obj


# --------------------------------------------------------------------------- Level-1 type dispatch
_DENSE_CONTRACTS = {"sequence", "spatial", "volumetric", "4d"}
_IRREGULAR_CONTRACTS = {"graph", "equivariant", "set"}
_FUNCTIONAL_CONTRACTS = {"operator"}          # neural-operator contract: function -> function on a grid
_ALL_CONTRACTS = _DENSE_CONTRACTS | _IRREGULAR_CONTRACTS | _FUNCTIONAL_CONTRACTS


def route_type(data: AllData, equivariant_if_positions=True):
    """Level-1: decide the contract FAMILY from the data container (rule-based, certain).
    Returns (contract, reason). Honors data.kind_hint as an explicit override."""
    if data.kind_hint is not None:
        if data.kind_hint not in _ALL_CONTRACTS:
            raise ValueError(f"unknown contract override '{data.kind_hint}'; valid: {sorted(_ALL_CONTRACTS)}")
        return data.kind_hint, "explicit override"
    # irregular families first: edges / point-sets are unambiguous structural signals
    if data.edges is not None:
        if data.positions is not None and equivariant_if_positions:
            return "equivariant", "has edges + 3D positions -> equivariant graph"
        return "graph", "has edges (relational) -> graph"
    if data.node_feats is not None:                 # variable-size collection, no edges
        if data.positions is not None and equivariant_if_positions:
            return "equivariant", "point set + 3D positions -> equivariant (geometric set)"
        return "set", "unordered point set (no edges) -> set"
    if data.dense is not None:
        return "_dense", "fixed-size dense tensor -> grid family (Level 2 decides rank)"
    raise ValueError("AllData carries no dense/node/edge content")


# --------------------------------------------------------------------------- Level-2 grid-rank dispatch
def route_grid_rank(dense: torch.Tensor, verbose=False, price_mu=0.05):
    """Level-2: for a dense tensor, pick the grid contract. Explicit tensor rank wins; a flat (n,d)
    vector is sent to mode-structure discovery. Returns (contract, detail).

    price_mu (default 0.05): when a flat vector is seen, the latent lattice shape is discovered by the
    PRICED tensorization (discover_mode_structure(price_mu=...)), which folds the rank choice into
    J = R + mu*Omega_struct with a significance gate against finite-sample MI noise. It is fail-safe --
    it only promotes to a higher rank when a real grid is confidently detected, else leaves the vector
    as a 1D sequence -- so enabling it by default can only refine routing, never mis-route. Set
    price_mu=None to fall back to the legacy floor+margin thresholds."""
    nd = dense.dim()
    # explicit multi-axis tensors: rank tells the contract directly
    if nd == 6:
        return "4d", {"shape": tuple(dense.shape[1:]), "why": "(b,C,T,D,H,W) rank-6 -> 4d"}
    if nd == 5:
        return "volumetric", {"shape": tuple(dense.shape[1:]), "why": "(b,C,D,H,W) rank-5 -> volumetric"}
    if nd == 4:
        return "spatial", {"shape": tuple(dense.shape[1:]), "why": "(b,C,H,W) rank-4 -> spatial"}
    if nd == 3:
        return "sequence", {"shape": tuple(dense.shape[1:]), "why": "(b,T,features) rank-3 -> sequence"}
    if nd == 2:
        # flat vector: discover latent grid structure from mutual information (priced by default)
        det = discover_mode_structure(np.asarray(dense.cpu()), price_mu=price_mu)
        struct = det["structure"]
        rank_to_mod = {"2d": "spatial", "3d": "volumetric", "4d": "4d"}
        if struct in rank_to_mod:
            return rank_to_mod[struct], {"shape": det["shape"],
                                         "why": f"flat vector, MI-detected {struct} grid {det['shape']}"
                                                + (" (priced)" if price_mu is not None else "")}
        if struct == "1d":
            return "sequence", {"shape": det["shape"], "why": "flat vector, MI-detected 1D chain"}
        return "sequence", {"shape": (dense.shape[1],), "why": "flat vector, unstructured -> dense-family sequence",
                            "unstructured": True}
    raise ValueError(f"cannot route a rank-{nd} dense tensor")


class _EarlyStopper:
    """Patience-based plateau detector for the deployed training loop (--auto_epoch). Fed one monitored metric
    per epoch (mean train loss, lower=better); an epoch counts as improvement only if it beats the best-so-far
    by at least a RELATIVE `min_delta` (scale-free, so it works across CE/MSE/field-MSE losses). After a
    `min_epochs` floor, .step() returns True once `patience` consecutive epochs show no such improvement.

    WARMUP GUARD (`warmup_drop`): the loop will NOT stop until the loss has left the INITIAL plateau -- i.e.
    dropped at least `warmup_drop` (relative) below the first epoch's baseline. A hard classification/long-
    sequence task can sit at a flat constant-prediction plateau for many epochs BEFORE it begins to fit; the
    plain patience test reads that flat stretch as convergence and freezes the collapse. Requiring the model
    to first leave the baseline plateau prevents stopping pre-learning; a model that never leaves it simply
    trains the full budget (the safe fallback)."""

    def __init__(self, min_delta=0.01, patience=4, min_epochs=5, warmup_drop=0.05):
        self.min_delta = float(min_delta); self.patience = int(patience); self.min_epochs = int(min_epochs)
        self.warmup_drop = float(warmup_drop)
        self.best = float("inf"); self.bad = 0; self.epoch = 0; self.init = None

    def step(self, metric):
        self.epoch += 1
        metric = float(metric)
        if self.init is None:
            self.init = metric                               # baseline loss (after the first epoch's steps)
        if metric < self.best * (1.0 - self.min_delta):      # a meaningful (>= min_delta relative) reduction
            self.best = metric; self.bad = 0
        else:
            self.bad += 1
        left_plateau = metric < self.init * (1.0 - self.warmup_drop)   # has the model begun to fit at all?
        return self.epoch >= self.min_epochs and self.bad >= self.patience and left_plateau


# --------------------------------------------------------------------------- the AllGraph
class AllGraph(_ContractFitMixin, _ReportsMixin, _PersistenceMixin, _SizeSelectionMixin):
    """Single entry point over all 8 schemas. Route -> build -> train -> report, via .fit(data, y).

    Usage:
        mg = AllGraph(width=32, depth=2, epochs=30)
        result = mg.fit(AllData.dense_tensor(X, y), task="classification")
        # result: dict(contract, selected_primitive, metric, value, detail)

    The router itself is not trained; each contract's OWN alpha/width/depth search runs after dispatch.
    """

    def __init__(self, width=32, depth=2, epochs=30, lr=2e-3, weight_decay=None, train_batch=None, device="cpu", seed=0,
                 equivariant_if_positions=True, verbose=True, gibbs_beta=8.0, sparsity_mu=0.0,
                 tensorize_mu=0.05, contract_router="default", symmetry_routing=False,
                 canonicalize_reuse=False, generated_equivariant_group=None,
                 discover_equivariant_contract=False,
                 readout_select=False, readout_mu=0.05, seq_flatten_max_T=64, kernel_from_xi=False, angular_from_data=False, nonlinear_symmetry_fallback=False,
                 enabled_contracts=None, contract_posterior=False, report_llc=False,
                 deploy_nonlinear_contract=False, equivariant_realization="emlp", price_equivariance=False,
                 price_modes=False, mode_mu=None,
                 price_singular=False, singular_mu=None,
                 singular_llc_steps=150, singular_llc_chains=3,
                 developmental_llc=False, developmental_llc_checkpoints=None,
                 developmental_llc_epochs=None, report_thermo=False, report_response=False,
                 report_ledger=False, progress=False,
                 auto_epoch=None, auto_epoch_patience=4, auto_epoch_min_delta=0.01, auto_epoch_min_epochs=5,
                 stream_subsample_cap=20000, stream_pin_memory=None, stream_prefetch=False,
                 stream_shuffle_buffer=8192):
        self.width = width; self.depth = depth; self.epochs = epochs; self.lr = lr
        self.weight_decay = weight_decay   # None -> per-contract default (_WEIGHT_DECAY / _DENSE_WEIGHT_DECAY); see _wd
        self.train_batch = train_batch     # None -> per-contract default (_TRAIN_BATCH / _SET_TRAIN_BATCH); see _tb
        # device may be "cpu" (default, for reproducible validation), "auto" (pick CUDA>MPS>CPU -- uses the
        # Apple-Silicon GPU when present), "cuda"/"mps", or an explicit torch.device. resolve_device turns
        # any of these into a concrete device and falls back to CPU with a warning if the backend is absent.
        from ..device import resolve_device
        self.device = resolve_device(device); self.seed = seed
        self._base_device = self.device   # requested device; fit() restores from this so the relational->CPU
                                          # fallback (below) never leaks into a later, non-relational reuse of this instance
        self.equivariant_if_positions = equivariant_if_positions; self.verbose = verbose
        # gibbs_beta: MDL inverse-temperature for the derived Gibbs-alpha selection. A fixed float (default
        # 8.0, reproducible) OR "auto" -> derived per-fit at the fit-vs-commitment frontier knee from the
        # actual solo energies (see _resolve_gibbs_beta). "auto" changes only the reported alpha confidence
        # (interpretability read-out), never the deployed primitive (which is argmin energy, beta-independent).
        self.gibbs_beta = gibbs_beta; self.select = "argmax"; self.sparsity_mu = sparsity_mu
        self.tensorize_mu = tensorize_mu
        # sequence readout selection: try 'flatten' (position-aware, for short "effectively tabular" series)
        # against 'mean', priced by parameter cost. Off by default (preserves prior behaviour); enable with
        # readout_select=True. Only bakes off for sequences up to seq_flatten_max_T steps.
        self.readout_select = readout_select
        self.readout_mu = readout_mu
        self.seq_flatten_max_T = seq_flatten_max_T
        self.kernel_from_xi = kernel_from_xi
        self.angular_from_data = angular_from_data
        self.nonlinear_symmetry_fallback = nonlinear_symmetry_fallback
        # learned, transferable contract router (amortizes the geometric-data bake-off). "default" seeds a
        # warm router from the validated archetypes; None disables learned routing (always bake off); or
        # pass a ContractRouter instance to supply/persist a corpus.
        if contract_router == "default":
            from ..machinery import default_router
            self.contract_router = default_router()
        else:
            self.contract_router = contract_router
        # symmetry-first routing: discover the data's rotational symmetry and route the geometric
        # equivariant-vs-set choice by it (the physics-motivated contract = arch(G) blueprint), consulted
        # BEFORE the learned router when enabled. See core/symmetry_contract.py.
        self.symmetry_routing = symmetry_routing
        # Phase-1 canonicalization reuse: when symmetry routing finds a rotation-invariant target, instead
        # of the equivariant contract, CANONICALIZE the coordinates (principal-axis frame) and route to the
        # cheaper SET contract -- which is then effectively E(3)-invariant. Demonstrates "discover a
        # symmetry -> exploit it by reusing an existing contract". Off by default.
        self.canonicalize_reuse = canonicalize_reuse
        self._canonicalized_positions = None
        self._canonicalization_applied = False
        # a GENERATED equivariant contract for a discovered group (Phase 2). A dict spec:
        #   {"gens": [A_k], "vec_dim": d, "metric": M or None, "n_in_vec": k}
        # When set, fit(...) can route to _fit_generated_equivariant, which builds an EMLP net that is
        # equivariant to this group -- a contract the eight built-ins may not cover (e.g. Lorentz).
        self.generated_equivariant_group = generated_equivariant_group
        # AUTONOMOUS loop: when True, fit() discovers WHICH group the data respects (via
        # _discover_equivariant_group -> extended_groups.discover_group / symmetry_contract) and
        # auto-populates generated_equivariant_group, then builds+deploys that equivariant contract -- with
        # no hand-supplied group. data -> discovered group -> generated equivariant net, end to end.
        self.discover_equivariant_contract = discover_equivariant_contract
        self.discovered_group_detail = None
        self._autonomous_forced_set = False
        # RESTRICT THE CONTRACT SET: enabled_contracts=None means all eight peer contracts are available; pass a
        # subset (list/set/tuple or comma-string) to disable the rest, so routing/tiebreak can only land on
        # an enabled contract. The eight peers are sequence, spatial, volumetric, 4d, graph, equivariant, set,
        # operator (generated_equivariant is the discovered-group EMLP path, always allowed when a group is
        # supplied/discovered since it is an explicit request, not an auto-route target). If a resolved
        # contract is disabled, fit() falls back to the nearest enabled contract (see _resolve_enabled_fallback).
        self.enabled_contracts = self._normalize_enabled_arenas(enabled_contracts)
        # contract_posterior: when True, the tie-break additionally reports an approximate Bayesian POSTERIOR
        # over the admissible contracts (machinery.contract_evidence): -log p(c|data) ~ n*Lhat_c + Omega(c),
        # so the argmin becomes a calibrated distribution with an entropy/confidence. The MAP equals the J
        # winner by construction, so this NEVER changes the selection -- it only adds a reported uncertainty.
        self.contract_posterior = contract_posterior
        # report_llc: when True, after the fit, estimate the Local Learning Coefficient (approx. RLCT, a
        # singularity-aware complexity <= #params/2) of the DEPLOYED net via SGLD (machinery.singular_
        # complexity) and record it in the result under "llc". This is a DIAGNOSTIC read-out (the report's
        # flagged parameter-count complexity is the wrong one for singular nets; this is the principled
        # replacement) -- it does not change selection. Off by default (adds SGLD cost).
        self.report_llc = report_llc
        # D4 (developmental read-out): when developmental_llc=True, after the fit, RE-TRAIN a fresh copy of the
        # SELECTED architecture for one trajectory and probe the Local Learning Coefficient lambda_hat at
        # checkpoints, recording the developmental curve under "developmental_llc". Whereas report_llc reads
        # lambda ONCE at the converged optimum, this reads lambda(t) OVER training -- the located negative->
        # positive convergence onset marks where usable capacity turns on, and plateaus/jumps mark staged
        # learning. Diagnostic read-out; does not change selection. Off by default (adds a retrain + repeated
        # short SGLD probes). developmental_llc_checkpoints overrides the (denser-early) default schedule;
        # developmental_llc_epochs sets the trajectory length (defaults to a convergence-scale budget).
        self.developmental_llc = developmental_llc
        self.developmental_llc_checkpoints = developmental_llc_checkpoints
        self.developmental_llc_epochs = developmental_llc_epochs
        # D2 (one free energy, one form, three temperatures): when report_thermo=True, after a fit, verify
        # that the three inverse temperatures of the single thermodynamic potential are each at their
        # principled value (beta_W=1/log n WBIC at the weights, beta_C=1 nats-vs-nats at the contracts,
        # beta_A=gibbs_beta a free readout-sharpness knob at the primitives) and that they have not been
        # accidentally coupled/equated -- and record the temperature hierarchy under "thermodynamic_potential"
        # (machinery.thermodynamic_potential). Diagnostic / conceptual-hygiene read-out; changes nothing.
        self.report_thermo = report_thermo
        # D5 (response / susceptibility spectroscopy): when report_response=True, after a fit, report the
        # CURVATURE of the selection objective at the chosen point -- how sharply the argmin is preferred
        # and how far to the nearest selection boundary. Two channels (machinery.response_spectroscopy):
        # the primitive readout (SMOOTH: specific heat chi=Var_alpha(Psi) + the monotone entropy sharpness)
        # and the contract choice (FIRST ORDER: the critical price mu* to the nearest winner-switch, the
        # price margins to flip, and the slope jump). Reuses quantities the fit already computed (the solo
        # energies behind the Gibbs-alpha, and the {scores, omegas, mu_c} behind the contract selection);
        # no retraining. Stored under "response_spectroscopy". Pure OBSERVABLE -- changes no selection.
        self.report_response = report_response
        # D3 (effective-dimension ledger): when report_ledger=True, after a fit, assemble the package's
        # several effective-dimension measures onto ONE coarse-graining axis (machinery.
        # effective_dimension_ledger): the participation ratio 1/sum(p^2) read at the data-covariance level
        # (effective data modes) and the alpha-simplex level (effective primitives) -- literally one
        # functional -- plus lambda (RLCT) at the model level if a singular estimate is available. Reuses
        # already-computed pieces (the deployed alpha, an optional LLC, a cheap data covariance); values are
        # in different units and are NOT summed. Stored under "effective_dimension_ledger". Pure reporting.
        self.report_ledger = report_ledger
        # D1: when price_singular=True, the contract bake-off augments each contract's structural code
        # length Omega_struct with the FUNCTIONAL singular code length lambda*log n (machinery.singular_mdl),
        # estimated on that contract's briefly-trained candidate net. This makes contract selection price
        # the fitted function's effective dimension, not just the contract's structural scaffolding. The
        # LLC validity guard applies: a candidate whose lambda_hat is non-physical (non-converged) falls
        # back to structural-only pricing for that contract. singular_mu is the price on the functional term
        # (defaults to the contract price mu_c when None). Opt-in; default off leaves selection unchanged.
        self.price_singular = price_singular
        self.singular_mu = singular_mu       # the report's mu_s (follows the code's <term>_mu price convention,
                                             # as sparsity_mu/width_mu/depth_mu do); paired with mu_c as mu_s/mu_c
        # SGLD budget for the per-contract LLC when price_singular is on (defaults kept modest so the
        # feature is tractable on real datasets; raise for a cleaner, less noisy lambda at more cost).
        self.singular_llc_steps = singular_llc_steps
        self.singular_llc_chains = singular_llc_chains
        # deploy_nonlinear_contract: when True (and the data has positions but no linear group was found), run
        # the JOINT LaLiGAN discovery and, IF it confirms a latent symmetry (null-baseline-guarded), BUILD and
        # deploy the latent-equivariant contract x -> encoder -> EMLP(latent generators) -> y (B3), closing
        # the discover->generate->deploy loop for NONLINEAR symmetry (the linear path already closes it).
        # Off by default: discovery is unreliable on quick budgets and the contract's benefit is contingent on
        # the learned latent chart's fidelity, so this is a deliberate opt-in, gated behind confirmation.
        self.deploy_nonlinear_contract = deploy_nonlinear_contract
        # equivariant_realization: how the deployed LATENT-equivariant head (--deploy_nonlinear_contract) is
        # realized -- "emlp" (exact basis-solve, general but the O(D^3) solve does not scale) or "scalable"
        # (G-RepsNet/Vector-Neurons vector mixing, equivariant by construction, linear in channels). The
        # discovered-group generated_equivariant contract always uses the exact EMLP, independent of this.
        self.equivariant_realization = equivariant_realization
        # price_equivariance: when True, a deployed latent-equivariant contract selects HOW STRICTLY to enforce
        # the discovered symmetry -- the relaxation strength of a residual-pathway approximate-equivariant model
        # is chosen by the same priced criterion J = R_val + mu_c*Omega (models/approximate_equivariance.py),
        # so exactly-symmetric data keeps exact equivariance (relax=0) and symmetry-broken data admits the
        # matched amount of breaking. Off by default (adds a small relax ladder of fits).
        self.price_equivariance = price_equivariance
        # price_modes (B7): in the operator contract, select the Fourier-MODE BUDGET by the same priced marginal-
        # value rule as width/depth (machinery.spectral_selection) instead of the fixed heuristic -- the mode
        # count is the operator's degrees of freedom, so this brings the one contract outside the d.o.f. stage
        # into it. Off by default (adds a short mode-ladder sweep before the final fit).
        self.price_modes = price_modes
        # mode_mu: price for the spectral mode d.o.f. (used when price_modes is on). None -> a default
        # calibrated to the operator field-MSE scale. Larger -> fewer modes retained (stricter Occam).
        self.mode_mu = mode_mu
        # progress: when True, each DEPLOYED model's training loop shows a live tqdm epoch bar (labeled with
        # the dataset/contract). Off by default so library use and the quick runner are unchanged; the
        # standard validation runner turns it on. progress_desc is set per-fit by the caller (e.g. the
        # dataset name) to label the bar; it falls back to the routed contract.
        self.progress = progress
        self.progress_desc: "str | None" = None
        # auto_epoch: early-stop the DEPLOYED training loop on a plateau. None/False = off (train the full
        # budget). "train" = monitor the epoch's mean TRAIN loss. "val" = hold out ~15% of the training data
        # and monitor VALIDATION loss (more overfitting-robust). Stop when the relative reduction stays below
        # auto_epoch_min_delta for auto_epoch_patience epochs, after a min-epochs floor. self.epochs stays the
        # CAP; internal search sub-fits keep their fixed budgets so candidate comparisons stay fair.
        self.auto_epoch = auto_epoch
        self.auto_epoch_patience = auto_epoch_patience
        self.auto_epoch_min_delta = auto_epoch_min_delta
        self.auto_epoch_min_epochs = auto_epoch_min_epochs
        # opt-in dataset streaming (see core/allgraph_streaming.py). Both are INERT unless the data is built
        # with AllData.dense_stream: stream_subsample_cap bounds the resident subsample drawn once from the
        # source for any search/selection sub-fit; stream_pin_memory (None -> auto: pin only on CUDA) enables
        # host-pinned + non_blocking H2D copies of each streamed minibatch.
        self.stream_subsample_cap = int(stream_subsample_cap)
        self.stream_pin_memory = stream_pin_memory
        # stream_prefetch (item 3): overlap the next minibatch's RNG-free fetch with the current batch's compute
        # on a background thread; bit-identical to prefetch off. False/0 -> off; True -> depth 1; int k -> depth k.
        if not (stream_prefetch is False or stream_prefetch is None
                or (isinstance(stream_prefetch, bool))
                or (isinstance(stream_prefetch, int) and stream_prefetch >= 0)):
            raise ValueError(f"stream_prefetch must be False/True or a non-negative int; got {stream_prefetch!r}.")
        self.stream_prefetch = stream_prefetch
        # stream_shuffle_buffer (item 4): windowed shuffle-buffer size for the forward-only iterable regime.
        self.stream_shuffle_buffer = int(stream_shuffle_buffer)
        self.net = None; self.contract = None; self.route_detail = None
        self._infer_task = None; self._infer_readout = None   # set by fit(); used by predict()/save()

    # ---- contract restriction (disable an arbitrary subset of the peer schemas) ----
    # canonical peer-contract names (the eight builders _fit_<name>); generated_equivariant is handled
    # separately as an explicit discovered-group path, not an auto-route target.
    _BUILTIN_CONTRACTS = ("sequence", "spatial", "volumetric", "4d", "graph", "equivariant", "set", "operator")
    _WEIGHT_DECAY = 1e-5          # default L2 weight decay (relational/set/operator contracts + sub-fits)
    _DENSE_WEIGHT_DECAY = 1e-4    # the dense-grid contracts (sequence/spatial/volumetric/4d) use a higher decay
    _TRAIN_BATCH = 32            # default SGD minibatch for the training loops (grid/eval chunking is
                                 # separate, via _run_epochs' batch_size and _fit_grid's eval_bs)
    _SET_TRAIN_BATCH = 64        # the set contract deploys with a larger minibatch (variable-size collation)
    _SURROGATE_LR = 5e-3         # lr for the LaLiGAN nonlinear-symmetry surrogate probe net
    _SKEW_LR_FLOOR = 1e-2        # lr floor for the Sp/SL (skew/volume) EMLP contract fit
    _SKEW_MIN_GROUPS = 6         # floor on the number of learned soft-groups/frames (Sp/SL contract)
    _SKEW_ATTN_HIDDEN = 32       # hidden width of the Sp/SL attention-gate MLP
    _SKEW_READOUT_HIDDEN = 64    # hidden width of the Sp/SL invariant-readout MLP
    _VOL_MAX_DHW = 16            # cap on the volumetric WORKING grid (the stem downsamples larger volumes to
                                 # keep the (b,C,D,H,W) conv/dense/attention primitives tractable; a raw 28^3
                                 # volume ran at full res -> ~8x the activation memory and OOM on MPS)

    @staticmethod
    def _vol_work_dhw(raw, cap=None):
        """Working grid the volumetric cells operate on: the raw side length if it already fits under the cap,
        else the stem-downsampled size for stride s=ceil(raw/cap). Returns the EXACT conv3d(k=3,pad=1,stride=s)
        output size so the cells' assumed dhw matches the strided stem output (avoids a size mismatch in the
        dense/attention primitives); paired with vol_size=raw the builder derives the same stride (raw//dhw)."""
        cap = AllGraph._VOL_MAX_DHW if cap is None else cap
        if raw <= cap:
            return raw
        s = -(-raw // cap)                       # ceil(raw / cap)
        return (raw - 1) // s + 1                # floor((raw+2*pad-k)/s)+1 with k=3,pad=1

    def _wd(self, default=None):
        """Optimizer weight decay: the constructor override (self.weight_decay) when set, else the contract
        default (_WEIGHT_DECAY, or `default` for the dense-grid contracts)."""
        base = self._WEIGHT_DECAY if default is None else default
        return base if self.weight_decay is None else self.weight_decay

    def _tb(self, default=None):
        """Training minibatch size: the constructor override (self.train_batch) when set, else the contract
        default (_TRAIN_BATCH, or `default` for the set contract's larger batch)."""
        base = self._TRAIN_BATCH if default is None else default
        return base if self.train_batch is None else self.train_batch

    @staticmethod
    def _grads_finite(net):
        """True iff every parameter gradient is finite -- used to SKIP a diverged optimizer step (a transient
        inf/NaN batch) rather than write non-finite weights that poison all subsequent steps."""
        for p in net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True

    @classmethod
    def _normalize_enabled_arenas(cls, enabled):
        """Validate and normalize the enabled-schema spec to a set (or None = all enabled)."""
        if enabled is None:
            return None
        if isinstance(enabled, str):
            enabled = [s.strip() for s in enabled.split(",") if s.strip()]
        enabled = set(enabled)
        unknown = enabled - set(cls._BUILTIN_CONTRACTS)
        if unknown:
            raise ValueError(f"unknown schema name(s) {sorted(unknown)}; valid contracts are {cls._BUILTIN_CONTRACTS}")
        if not enabled:
            raise ValueError("enabled_contracts is empty; at least one contract must be enabled")
        return enabled

    def _contract_enabled(self, name):
        """True if contract `name` is available under the current restriction (None = all enabled)."""
        return self.enabled_contracts is None or name in self.enabled_contracts


    def _resolve_enabled_fallback(self, contract, data):
        """If `contract` is disabled, return the nearest ENABLED contract to fall back to; else return it
        unchanged. The fallback order is principled by constructibility and generality: relational contracts
        degrade graph -> set (drop edge structure) and equivariant -> graph -> set (drop geometry, then
        edges); dense/grid contracts degrade toward the always-constructible 'sequence' (a rank-1 tensor view)
        and finally 'set'. A ValueError is raised only if NOTHING enabled can represent the data (e.g. the
        only enabled contracts need edges the data lacks)."""
        if self._contract_enabled(contract):
            return contract
        # per-contract degradation preference (most specific first, then progressively more general/constructible)
        pref = {
            "equivariant": ["graph", "set", "sequence"],
            "graph":       ["set", "sequence"],
            "set":         ["sequence"],
            "operator":    ["sequence", "spatial", "volumetric"],
            "4d":          ["volumetric", "spatial", "sequence"],
            "volumetric":  ["spatial", "sequence"],
            "spatial":     ["sequence"],
            "sequence":    ["set"],
        }.get(contract, ["sequence", "set"])
        has_edges = getattr(data, "edges", None) is not None
        has_pos = getattr(data, "positions", None) is not None
        has_dense = getattr(data, "dense", None) is not None
        for cand in pref:
            if not self._contract_enabled(cand):
                continue
            # constructibility guards mirroring the admissibility rule used elsewhere in fit()
            if cand in ("graph", "equivariant") and not has_edges:
                continue
            if cand in ("set", "graph", "equivariant") and not (has_pos or getattr(data, "node_feats", None) is not None):
                continue
            if cand in ("sequence", "spatial", "volumetric", "4d", "operator") and not has_dense:
                continue
            return cand
        # last resort: any enabled contract that is constructible at all
        for cand in self._BUILTIN_CONTRACTS:
            if self._contract_enabled(cand):
                if cand in ("graph", "equivariant") and not has_edges:
                    continue
                if cand in ("sequence", "spatial", "volumetric", "4d", "operator") and not has_dense:
                    continue
                return cand
        raise ValueError(
            f"contract '{contract}' is disabled and no enabled contract in {sorted(self.enabled_contracts)} "
            f"can represent this data (edges={has_edges}, positions={has_pos}, dense={has_dense}).")

    # ---- routing only (no training): useful for inspection / the meta-router's decision ----
    def route(self, data: AllData):
        """Return (contract, detail) without training. Two-level dispatch (rule-based Level 1)."""
        fam, reason = route_type(data, self.equivariant_if_positions)
        if fam != "_dense":
            return fam, {"level1": reason}
        # A streaming DenseSource reaching Level-2 grid-rank dispatch means it was attached WITHOUT a kind_hint
        # (dense_stream forbids that; this is only reachable via the raw AllData(dense=source) constructor).
        # route_grid_rank would then try to materialize a flat source (np.asarray(dense.cpu())); pre-empt that
        # with the same actionable message dense_stream gives, rather than an opaque AttributeError.
        if self._is_streaming(data.dense) or self._is_iterable(data.dense):
            ctor = "dense_iter" if self._is_iterable(data.dense) else "dense_stream"
            raise ValueError(
                "a streaming/iterable source on AllData.dense requires an explicit kind_hint in "
                f"{sorted(_DENSE_CONTRACTS)} so routing never materializes the source; build the input with "
                f"AllData.{ctor}(source, kind_hint=...).")
        contract, gdetail = route_grid_rank(data.dense, verbose=self.verbose, price_mu=self.tensorize_mu)
        return contract, {"level1": reason, "level2": gdetail}

    # ---- learned Level-1 tie-break for the genuinely ambiguous geometric case ----
    def _tiebreak_candidates(self, data: AllData):
        """Which contracts are constructible for this data? Only geometric relational data (positions
        present) is ambiguous; everything else has a single valid contract."""
        if self._is_streaming_graph(data):
            # under a GraphSource, positions/edges live INSIDE the source (data.positions is None), so read the
            # source metadata; tiebreak=True is the explicit opt-in, so the kind_hint short-circuit is bypassed.
            src = data.node_feats
            if not src.has_pos:
                return None
            cands = ["set", "equivariant"] + (["graph"] if src.has_edges else [])
            cands = [c for c in cands if self._contract_enabled(c)]
            return cands if len(cands) > 1 else None
        if data.kind_hint is not None:
            return None                                  # explicit override -> no bake-off
        if data.positions is None:
            return None                                  # no geometry -> rule-based route is decisive
        cands = ["set"]                                  # atoms as a bare point set: always constructible
        if data.node_feats is not None:
            cands.append("equivariant")                  # positions used geometrically (edges from cutoff)
            if data.edges is not None:
                cands.append("graph")                    # topology only (edges, positions folded/ignored)
        # honor the contract restriction: the bake-off may only consider ENABLED contracts
        cands = [c for c in cands if self._contract_enabled(c)]
        return cands if len(cands) > 1 else None

    def _symmetry_route(self, data, cands):
        """SYMMETRY-FIRST routing (physics-motivated: contract = arch(G)). When --symmetry_routing is enabled,
        discover the data's rotational symmetry directly and route the equivariant-vs-set choice by it -- the
        most principled signal (the actual invariance the target respects). A rotation-invariant target with
        --canonicalize enabled is canonicalized to the principal-axis frame and routed to the cheaper SET
        contract (then effectively E(3)-invariant; the canonicalized positions are stashed for the fit);
        otherwise the symmetry-discovered contract is used when it is an admissible candidate. Returns
        (contract, detail) when confident, else None (the caller falls back to the bake-off, if any, or the
        rule-based route). Consulted on BOTH the --tiebreak bake-off path AND the default route path, so
        --symmetry_routing / --canonicalize are honored WITHOUT requiring --tiebreak (and the `med` preset,
        which enables them with tiebreak off, actually engages them)."""
        if not self.symmetry_routing or not cands:
            return None
        from .symmetry_contract import contract_from_symmetry
        try:
            sc, sconf, sdetail = contract_from_symmetry(data)
        except Exception as e:
            self._log(f"[AllGraph] symmetry routing failed ({type(e).__name__}); falling back")
            return None
        if sc == "equivariant" and sconf > 0.0 and self.canonicalize_reuse and "set" in cands:
            # Phase-1 reuse: rotation-invariant target -> canonicalize the coordinates and route to the
            # cheaper SET contract (then effectively E(3)-invariant). Stash the canonicalized positions.
            from .canonicalization import canonicalize_data
            cdata, cdetail = canonicalize_data(data)
            if cdetail.get("applied"):
                self._canonicalized_positions = cdata.positions
                self._log(f"[AllGraph] symmetry+canonicalize -> set  "
                          f"[rotation-invariant target canonicalized; {cdetail.get('note','')}]")
                return "set", {"note": "symmetry canonicalization reuse (rotation-invariant -> canonicalize "
                               "-> set contract)", "predicted": "set", "confidence": sconf,
                               "symmetry": sdetail, "canonicalization": cdetail}
        if sc in cands and sconf > 0.0:
            self._log(f"[AllGraph] symmetry -> {sc}  confidence={sconf:.2f} [{sdetail.get('reason','')}]")
            return sc, {"note": "symmetry-discovered contract (arch(G))", "predicted": sc,
                        "confidence": sconf, "symmetry": sdetail}
        return None

    def tiebreak(self, data: AllData, task="classification", candidates=None, val_frac=0.25,
                 tiebreak_epochs=None, edge_cutoff=None, noise_margin=0.02, n_out=None, mu_c=0.05,
                 record=True, trust_budget=10):
        """Learned Level-1 tie-break (clean-solo, one level up). For geometric data (positions present),
        the best CONTRACT -- equivariant vs graph vs set -- is data-dependent, so we train each candidate
        contract briefly from scratch on a shared held-out split and pick the best by the MDL objective
        J = R + mu_c * Omega_struct. This is the clean_solo_select protocol applied to contracts instead
        of primitives (no mixing -- contracts differ), with the contract choice folded into the SAME
        priced J = R + mu*Omega as width/depth/primitive. Returns (winner, scores, detail).

        The Occam preference among near-tied contracts is now DERIVED, not ordinal: Omega_struct is the
        theoretical structural code length of each contract (set 0; graph = adjacency description
        ~ E log(N^2/2E); equivariant = graph + geometry (N*d - dim SO(d)) log(1/delta)), lattice-monotone
        and data-dependent (machinery.contract_mdl). A richer contract must beat the cheaper one by more
        than mu_c times its added structural code length. mu_c is the contract price (0 -> pure best-fit).
        """
        from ..machinery import clean_solo_select, dataset_omega_struct, select_contract_mdl, marginal_value_contract
        # under streaming, run the whole contract bake-off on the bounded resident subsample (drawn once); the
        # winning contract then deploy-trains on the FULL stream. _tiebreak_candidates reads the source metadata.
        cands = candidates or self._tiebreak_candidates(data)
        sub = self._streaming_subsample(data)
        if sub is not None:
            data = sub
        if cands is None:
            contract, _ = self.route(data)
            return contract, {contract: float("nan")}, {"note": "not ambiguous; rule-based route used"}
        # SYMMETRY-FIRST ROUTING (physics-motivated: contract = arch(G)): route the equivariant-vs-set choice
        # by the discovered symmetry BEFORE the learned router or the bake-off. Shared with the default
        # (non-tiebreak) route path via _symmetry_route, so it takes precedence when confident.
        sym = self._symmetry_route(data, cands)
        if sym is not None:
            c, det = sym
            return c, {c: float("nan")}, det
        # LEARNED, TRANSFERABLE ROUTING (amortizes the bake-off). Ask the learned router first; if it is
        # CONFIDENT its predicted contract is admissible, use it and SKIP the expensive bake-off. If it is
        # unconfident (or untrained), fall through to the bake-off (ground truth) and RECORD the outcome so
        # the router improves. This has a correctness floor: never worse than the bake-off, cheaper when
        # confident. Set contract_router=None to always bake off.
        descriptor = None
        if self.contract_router is not None:
            descriptor = self._dataset_descriptor(data, edge_cutoff)
            pred, conf, pdetail = self.contract_router.predict(descriptor)
            ep_planned = tiebreak_epochs if tiebreak_epochs is not None else self._search_ep(max(5, self.epochs // 3))
            # (a) confident prediction -> use it, skip the bake-off entirely.
            # (b) even if not fully confident, if the bake-off we would run is LOW-BUDGET (untrustworthy:
            #     the equivariant branch cannot converge, so the reduced bake-off spuriously prefers the
            #     cheap contract via the Omega charge), PREFER the router's prediction. The router is
            #     seeded from FULL-BUDGET outcomes, so its label is more reliable than a tiny bake-off.
            if pred in cands and (conf >= self.contract_router.min_confidence or ep_planned < trust_budget):
                why = "confident" if conf >= self.contract_router.min_confidence else \
                      f"bake-off budget {ep_planned} < {trust_budget} is untrustworthy; trusting full-budget-seeded router"
                self._log(f"[AllGraph] tiebreak (learned) -> {pred}  confidence={conf:.2f} ({why})")
                return pred, {pred: float("nan")}, {"note": f"learned contract router ({why}); bake-off skipped",
                                                    "predicted": pred, "confidence": conf, "router": pdetail,
                                                    "descriptor": list(descriptor)}
        n = len(data.node_feats)
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n); nval = max(1, int(val_frac * n))
        va, tr = perm[:nval], perm[nval:]
        ep = tiebreak_epochs if tiebreak_epochs is not None else self._search_ep(max(5, self.epochs // 3))
        n_out = self._infer_nout(data.y, task, n_out)

        # D1: optionally estimate the functional code length lambda*log n per contract, on the candidate
        # net trained in this bake-off. When price_singular is on we capture each contract's trained net
        # (return_net=True) and stash (net, closure, n); the score returned is identical to the plain path.
        singular_nets = {}
        _price_singular = getattr(self, "price_singular", False)

        def build_and_train_solo(contract):
            ctx = _SweepCtx(data, contract, tr, va, task, n_out, ep, edge_cutoff)
            if _price_singular:
                value, net, closure, n_tr = self._train_candidate_contract(ctx, return_net=True)
                singular_nets[contract] = (net, closure, n_tr)
                return value
            return self._train_candidate_contract(ctx)

        winner, scores = clean_solo_select(build_and_train_solo, cands)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
        # DERIVED structural code length per admissible contract, from the actual data (node/edge counts).
        sizes, edge_counts = self._contract_sizes(data, edge_cutoff)
        omegas = {c: dataset_omega_struct(c, sizes, edge_counts) for c in scores}
        # D1: augment with the FUNCTIONAL singular code length lambda*log n (opt-in). Omega_total =
        # Omega_struct + singular_mu/mu_c * lambda*log n. A contract whose candidate net did not converge
        # (guard: non-physical lambda_hat) keeps structural-only pricing, so a garbage lambda never enters J.
        singular_detail = {}
        if getattr(self, "price_singular", False) and singular_nets:
            from ..machinery.singular_mdl import singular_complexity_of
            s_mu = self.singular_mu if self.singular_mu is not None else mu_c
            # SGLD budget for the per-contract LLC: default is modest so price_singular stays tractable
            # on real datasets; raise via singular_llc_steps/singular_llc_chains for a cleaner lambda.
            _llc_steps = getattr(self, "singular_llc_steps", 150)
            _llc_chains = getattr(self, "singular_llc_chains", 3)
            for c, (net, closure, n_tr) in singular_nets.items():
                try:
                    sc = singular_complexity_of(net, closure, n_tr, chains=_llc_chains,
                                                steps=_llc_steps, burn=max(30, _llc_steps // 4),
                                                eps=2e-5, seed=self.seed)
                except Exception as e:
                    singular_detail[c] = {"error": f"{type(e).__name__}: {e}", "applied": False}
                    continue
                if sc["valid"]:
                    # price relative to mu_c so the two Omega terms share one currency: add
                    # (s_mu/mu_c)*omega_func as extra Omega (guarding mu_c==0 -> just add omega_func).
                    extra = (s_mu / mu_c) * sc["omega_func"] if mu_c > 0 else sc["omega_func"]
                    omegas[c] = omegas[c] + extra
                    singular_detail[c] = {"lambda": sc["lambda"], "omega_func": sc["omega_func"],
                                          "priced_extra_omega": extra, "valid": True, "applied": True}
                else:
                    singular_detail[c] = {"lambda": sc["lambda"], "valid": False, "applied": False,
                                          "note": "non-converged candidate; structural-only pricing"}
            self._log(f"[AllGraph] D1 singular pricing applied to contracts: "
                      f"{ {c: round(d.get('omega_func', float('nan')), 2) for c, d in singular_detail.items()} }")
        # PRIMARY selection: global J = R + mu_c * Omega_struct (min total description length). Robust to
        # non-monotone fit profiles (unlike the greedy marginal-value rule); the structural charge breaks
        # only genuine near-ties, exactly where the old ordinal kappa did.
        mdl_winner, mdl_detail = select_contract_mdl(scores, omegas, mu_c=mu_c)
        # marginal-value CERTIFICATE (diagnostic reading: does climbing the lattice pay?)
        _, mv_detail = marginal_value_contract(scores, omegas, mu_c=mu_c)
        detail = {"scores": scores, "margin": margin, "epochs": ep, "val_frac": val_frac,
                  "omega_struct": omegas, "mdl": mdl_detail, "marginal_value": mv_detail,
                  "best_fit": winner}
        if singular_detail:
            detail["singular_pricing"] = singular_detail
        # OPT-IN Bayesian reading (direction B1): report an approximate posterior over contracts. The fit
        # term is put on an NLL scale and combined with Omega_struct as a structure prior; the parameter-
        # Occam term cancels at the shared bake-off budget. The MAP coincides with the J winner, so this is
        # a REPORTED uncertainty layer, not a different selection.
        if getattr(self, "contract_posterior", False):
            from ..machinery.contract_evidence import contract_evidence
            n_ct = len(data.node_feats) if getattr(data, "node_feats", None) is not None else len(sizes)
            n_classes = self._infer_nout(data.y, task, n_out) if task == "classification" else 2
            # scores are R2/accuracy (scale-invariant); use the standardized-y NLL scale (y_std=1). Any
            # log(y_std) offset is common to all contracts and cancels in the posterior, so this is exact
            # for the COMPARISON while keeping the fit term on a clean per-datum nats scale.
            ev = contract_evidence(scores, omegas, n=n_ct, task=task, n_classes=n_classes,
                                   y_std=1.0, mu_c=1.0)
            detail["posterior"] = ev
            self._log(f"[AllGraph] contract posterior -> {ev['map']} "
                      f"(P={ev['posterior'][ev['map']]:.3f}, entropy={ev['posterior_entropy']:.2f} nats)")
        if mdl_winner != winner:
            detail["note"] = (f"'{winner}' best fit ({ranked[0][1]:.3f}) but J = R + mu_c*Omega_struct "
                              f"selects '{mdl_winner}' (derived structural Occam over the contract lattice)")
        winner = mdl_winner
        # RECORD the bake-off outcome to improve the router -- but ONLY if recording is enabled AND the
        # bake-off ran at a trustworthy budget. Reduced-budget bake-offs give budget-artifact labels (the
        # equivariant branch under-converges and the Omega charge then prefers the cheap contract), which
        # would POISON the corpus; the dedicated full-budget recorder is where labels are learned.
        if self.contract_router is not None and descriptor is not None and record and ep >= trust_budget:
            self.contract_router.add(descriptor, winner)
            detail["router_updated"] = True
        self._log(f"[AllGraph] tiebreak -> {winner}  scores={ {k: round(v,4) for k,v in scores.items()} }"
                  f"  Omega_struct={ {k: round(v,1) for k,v in omegas.items()} }  mu_c={mu_c}")
        return winner, scores, detail

    def _dataset_descriptor(self, data, edge_cutoff=None):
        """Build the transfer-friendly dataset descriptor the learned contract router consumes, from a
        AllData container: per-datum positions, node features, dense adjacency (from edges if present,
        else a distance-cutoff graph on positions), and the target y."""
        from ..machinery import dataset_descriptor
        positions = list(data.positions) if data.positions is not None else [None] * len(data.node_feats)
        node_feats = list(data.node_feats)
        adjacencies = []
        for i, nf in enumerate(node_feats):
            N = len(nf)
            A = None
            if data.edges is not None:
                e = np.asarray(data.edges[i])
                A = np.zeros((N, N))
                if e.size:
                    A[e[0], e[1]] = 1.0; A[e[1], e[0]] = 1.0
            elif data.positions is not None:
                P = np.asarray(data.positions[i])
                D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
                A = ((D < self._edge_cut(D, edge_cutoff)) & (D > 0)).astype(float)
            adjacencies.append(A)
        y = None
        if data.y is not None:
            yy = np.asarray(data.y).ravel()
            if len(yy) == len(node_feats):
                y = yy
        return dataset_descriptor(positions, node_feats, adjacencies, y)

    def _contract_sizes(self, data, edge_cutoff=None):
        """Per-datum node counts N and edge counts E for the structural code length. Edges come from the
        data if present, else are built from a distance cutoff on positions (same rule the graph/
        equivariant candidate builders use), so Omega_struct(graph) reflects the adjacency actually used."""
        sizes = [nf.shape[0] for nf in data.node_feats]
        if data.edges is not None:
            edge_counts = [np.asarray(e).shape[1] // 2 for e in data.edges]
        elif data.positions is not None:
            edge_counts = []
            for p in data.positions:
                p = np.asarray(p); D = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
                E = int(((D < self._edge_cut(D, edge_cutoff)).sum() - len(p)) // 2)   # undirected, excluding self
                edge_counts.append(E)
        else:
            edge_counts = [0] * len(sizes)
        return sizes, edge_counts

    def _deploy_approx_equivariant(self, enc, gens, res, latent_dim, X, y, Xt, task, n_out):
        """B5: deploy an APPROXIMATELY-equivariant contract, selecting the relaxation strength by the priced
        criterion J = R_val + mu_c*Omega (models/approximate_equivariance). Builds a residual-pathway model
        (latent-equivariant head + free MLP on the latent) at each relax on a small ladder, trains on a
        train split, prices by held-out validation risk + relative breaking power, and deploys the winner.
        Returns a result dict (sets contract/net) or None to fall back to the strict contract."""
        try:
            from ..models.approximate_equivariance import ApproxEquivariantModel, select_relaxation
            from ..models.latent_equivariant_contract import build_latent_equivariant_contract
            ld = res.get("latent_dim", latent_dim)
            realization = getattr(self, "equivariant_realization", "emlp")
            lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
            # train/val split for the priced selection
            n = len(X); rng = np.random.RandomState(self.seed); perm = rng.permutation(n)
            ntr = int(0.75 * n); tr_idx, va_idx = perm[:ntr], perm[ntr:]
            yt = torch.as_tensor(y)

            def score(out, idx):
                with torch.no_grad():
                    if task == "classification":
                        pred = out.argmax(-1).cpu().numpy()
                        return float((pred == y[idx]).mean())
                    p = out.squeeze(-1).cpu().numpy(); t = y[idx]
                    return float(1 - ((p - t) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9))

            class _Wrap(ApproxEquivariantModel):
                # equiv head consumes the latent chart output; free head consumes the same latent (encoder out)
                def forward(m, x, x_equiv=None):
                    out = m.equiv(x)                      # x is the raw input; equiv is the full contract
                    if m.relax > 0:
                        z = m.equiv.latent(x)             # frozen-encoder latent for the free pathway
                        out = out + m.relax * m.free(z)
                    return out

                def breaking_power(m, x):
                    with torch.no_grad():
                        eq = m.equiv(x)
                        z = m.equiv.latent(x)
                        full = eq + (m.relax * m.free(z) if m.relax > 0 else 0.0)
                        return float(((full - eq) ** 2).mean()) / (float((full ** 2).mean()) + 1e-9)

            built = {}

            def build_and_train(relax):
                torch.manual_seed(self.seed)
                contract = build_latent_equivariant_contract(enc, gens, latent_dim=ld, n_out=n_out,
                                                             depth=self.depth, realization=realization).to(self.device)
                net = _Wrap(contract, free_in_dim=ld, n_out=n_out, relax=relax).to(self.device)
                opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.lr,
                                       weight_decay=self._wd())
                idx = np.array(tr_idx)
                for _ in range(max(self.epochs, 60)):
                    np.random.shuffle(idx)
                    for j in range(0, len(idx), 64):
                        b = idx[j:j + 64]
                        opt.zero_grad()
                        out = net(Xt[b])
                        tgt = (yt[b].long().to(self.device) if task == "classification"
                               else yt[b].float().unsqueeze(1).to(self.device))
                        lf(out, tgt).backward(); opt.step()
                built[relax] = net
                with torch.no_grad():
                    out_va = net(Xt[va_idx])
                vr = 1.0 - score(out_va, va_idx)
                om = net.breaking_power(Xt[va_idx])
                return vr, om

            best, detail = select_relaxation(build_and_train, relax_ladder=(0.0, 0.1, 0.3, 1.0),
                                             mu_c=float(self.mu) if hasattr(self, "mu") else 0.3)
            net = built[best]
            self.contract = "latent_equivariant"
            self.net = net
            with torch.no_grad():
                out = net(Xt).cpu()
            metric, value = self._metric(out, y, task)
            self.route_detail = {**(self.route_detail or {}),
                                 "nonlinear_contract": f"CONFIRMED latent {res.get('latent_group')} symmetry; "
                                 f"priced approximate equivariance selected relax={best} "
                                 f"(exact equivariance if 0)",
                                 "approx_equivariance": detail}
            self._log(f"[AllGraph] approx-equivariant contract DEPLOYED: latent {res.get('latent_group')}, "
                      f"selected relax={best}, {metric}={value:.3f}")
            return {"contract": "latent_equivariant", "architecture": ["encoder", "latent_EMLP", "free_residual"],
                    "metric": metric, "value": float(value), "selected_relax": float(best),
                    "latent_group": res.get("latent_group"), "latent_dim": ld, "route_detail": self.route_detail}
        except Exception as e:
            self._log(f"[AllGraph] approx-equivariant deploy skipped ({str(e)[:70]})")
            return None

    def _fit_nonlinear_contract(self, data, task, n_out):
        """B3: discover a NONLINEAR symmetry (joint LaLiGAN) and, if confirmed, deploy the latent-equivariant
        contract x -> encoder(phi) -> EquivariantMLP(latent generators) -> y. Returns a result dict on success
        (sets self.contract='latent_equivariant', self.net) or None to fall back to the normal route. Gated
        behind the discovery's null-baseline confirmation; honest that chart fidelity bounds the benefit."""
        try:
            import torch
            from .nonlinear_symmetry import discover_nonlinear_symmetries_joint
            from ..models.latent_equivariant_contract import build_latent_equivariant_contract
            if data.positions is None:
                return None
            clouds = [np.asarray(p, dtype=np.float32).ravel() for p in data.positions]
            if len(clouds) < 30:
                return None
            d = min(len(c) for c in clouds)
            self._latent_input_dim = int(d)          # flatten-truncation length; predict() replays it on new data
            X = np.stack([c[:d] for c in clouds]).astype(np.float32)
            y = np.asarray(data.y, dtype=np.float32).ravel()[:len(X)]
            Xt = torch.tensor(X).to(self.device)
            latent_dim = min(max(2, d // 4), 6)
            # surrogate task model so the discovery has an f whose latent invariance to test
            yt = torch.tensor((y - y.mean()) / (y.std() + 1e-8)).to(self.device)
            surrogate = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.Tanh(),
                                            torch.nn.Linear(32, 1)).to(self.device)
            opt = torch.optim.Adam(surrogate.parameters(), lr=self._SURROGATE_LR)
            for _ in range(200):
                opt.zero_grad(); ((surrogate(Xt).squeeze(-1) - yt) ** 2).mean().backward(); opt.step()
            res = discover_nonlinear_symmetries_joint(surrogate, Xt.cpu(), latent_dim=latent_dim,
                                                      epochs=600, sym_weight=30.0, seed=self.seed)
            # CONFIRMATION GATE: require the null-baseline guard AND a non-degenerate, classifiable signal.
            # confirmed_by_null already encodes "learned generator is >=null_ratio quieter than random", the
            # scale-free criterion (absolute violations can be tiny when the surrogate fits well yet the RATIO
            # is decisive). We additionally require that the null violation is not utterly negligible relative
            # to the surrogate's own loss scale (guards the pathological case where g is constant in the whole
            # latent, making every direction vacuously "invariant"), and that a latent generator is available.
            sviol = float(res.get("sym_violation", 0.0)); nviol = float(res.get("null_violation", 0.0))
            confirmed = res.get("confirmed_by_null", False)
            quieter = nviol / (sviol + 1e-12)                 # how many x quieter the learned gen is
            # relative-scale non-degeneracy: null violation must be an appreciable fraction of a reference
            # perturbation of g (here, the surrogate output std ~ O(1) after standardization); use a small
            # relative floor so a genuinely-quiet-but-well-fit symmetry (large `quieter`) still passes.
            nondegenerate = (quieter >= res.get("_null_ratio", 1.5)) or (nviol > 1e-3)
            gens_available = bool(res.get("generators_latent")) or (res.get("learned_generator") is not None)
            if not (confirmed and nondegenerate and gens_available):
                reason = ("learned generator not decisively quieter than random "
                          f"({quieter:.1f}x)" if not nondegenerate
                          else "no latent generator available" if not gens_available
                          else f"null guard failed (sym={sviol:.4f} vs null={nviol:.4f})")
                self.route_detail = {**(self.route_detail or {}),
                                     "nonlinear_contract": f"discovery did NOT confirm a deployable latent "
                                     f"symmetry ({reason}); falling back to the normal route"}
                return None
            # confirmed: build the contract from the AE encoder + the (confirmed) latent generators
            enc = res["ae"].enc.to(self.device)
            gens = res.get("generators_latent") or []
            if not gens and res.get("learned_generator") is not None:
                gens = [torch.tensor(np.asarray(res["learned_generator"]), dtype=torch.float32)]
            if not gens:
                return None
            n_out = self._infer_nout(data.y, task, n_out)
            # B5: optionally select HOW STRICTLY to enforce the discovered symmetry (priced relaxation).
            if getattr(self, "price_equivariance", False):
                relax_res = self._deploy_approx_equivariant(enc, gens, res, latent_dim, X, y, Xt, task, n_out)
                if relax_res is not None:
                    return relax_res
            net = build_latent_equivariant_contract(enc, gens, latent_dim=res.get("latent_dim", latent_dim),
                                                    n_out=n_out, depth=self.depth,
                                                    realization=getattr(self, "equivariant_realization", "emlp")).to(self.device)
            # train the EMLP head (encoder frozen) on the FULL data
            optn = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.lr,
                                    weight_decay=self._wd())
            lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
            yt2 = torch.as_tensor(y)
            idx = np.arange(len(X))
            for _ in range(max(self.epochs, 60)):        # give the head enough to converge
                np.random.shuffle(idx)
                for j in range(0, len(idx), 64):
                    b = idx[j:j + 64]
                    optn.zero_grad()
                    out = net(Xt[b])
                    tgt = (yt2[b].long().to(self.device) if task == "classification"
                           else yt2[b].float().unsqueeze(1).to(self.device))
                    lf(out, tgt).backward(); optn.step()
            self.contract = "latent_equivariant"
            self.net = net
            with torch.no_grad():
                out = net(Xt).cpu()
            metric, value = self._metric(out, y, task)
            self.route_detail = {**(self.route_detail or {}),
                                 "nonlinear_contract": f"CONFIRMED latent {res.get('latent_group')} symmetry "
                                 f"(sym_viol={res.get('sym_violation'):.4f} << null={res.get('null_violation'):.4f}); "
                                 f"deployed latent-equivariant contract (latent_dim={res.get('latent_dim', latent_dim)})"}
            self._log(f"[AllGraph] nonlinear contract DEPLOYED: latent {res.get('latent_group')} symmetry, "
                      f"{metric}={value:.3f}")
            return {"contract": "latent_equivariant", "architecture": ["encoder", "latent_EMLP"],
                    "metric": metric, "value": float(value),
                    "latent_group": res.get("latent_group"), "latent_dim": res.get("latent_dim", latent_dim),
                    "route_detail": self.route_detail}
        except Exception as e:
            self._log(f"[AllGraph] nonlinear contract skipped ({str(e)[:70]})")
            return None

    def _try_nonlinear_symmetry(self, data):
        """Opt-in nonlinear (LaLiGAN) symmetry discovery, run when the linear menu found no group. Builds a
        cheap proxy task-model on the point clouds (a small invariant-features regressor is overkill here; we
        use the flattened clouds directly with a tiny MLP surrogate) and calls the latent-linearization
        detector. Returns a diagnostic dict or None. Conservative + expensive by construction: reports whether
        a nonlinear symmetry exists in a learned latent space, without altering the contract."""
        try:
            import torch
            from .nonlinear_symmetry import discover_nonlinear_symmetries
            # assemble a matrix of flattened clouds as the coordinate space X (n, d)
            clouds = []
            for p in (data.positions if data.positions is not None else []):
                a = np.asarray(p, dtype=np.float32).ravel()
                clouds.append(a)
            if len(clouds) < 30:
                return None
            d = min(len(c) for c in clouds)
            X = np.stack([c[:d] for c in clouds]).astype(np.float32)
            y = np.asarray(data.y, dtype=np.float32).ravel()[:len(X)]
            # a tiny surrogate task model y ~ g(X), so the detector has an f to test invariance of
            Xt = torch.tensor(X); yt = torch.tensor((y - y.mean()) / (y.std() + 1e-8))
            surrogate = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(),
                                            torch.nn.Linear(32, 1))
            opt = torch.optim.Adam(surrogate.parameters(), lr=self._SURROGATE_LR)
            for _ in range(150):
                opt.zero_grad(); ((surrogate(Xt).squeeze(-1) - yt) ** 2).mean().backward(); opt.step()
            res = discover_nonlinear_symmetries(surrogate, Xt, ae_epochs=200)
            return {"n_symmetries": int(res.get("n_symmetries", 0)),
                    "gap_ratio": round(float(res.get("gap_ratio", 0.0)), 3),
                    "latent_group": res.get("latent_group", "none"),
                    "ae_recon": round(float(res.get("ae_recon", 0.0)), 5),
                    "note": res.get("note", "")}
        except Exception as e:
            self._log(f"[AllGraph] nonlinear_symmetry_fallback skipped ({str(e)[:60]})")
            return None

    def _select_angular_order(self, data, task, mu=0.03):
        """Angular-order selection (integrating core/angular_resolution.py). Applies the priced marginal-value
        rule to ANGULAR order: order l is worth including only while the target variance it explains, that
        lower orders cannot, exceeds a price mu. The current equivariant builder carries scalar (l=0, c0) and
        vector (l=1, c1) channels, so the decision here is whether l=1 (directional/vector) features are
        warranted: if the selected max_l is 0 the target is (approximately) radial and vectors are dropped
        (c1=0, a leaner scalar-only net); if >=1 vectors are kept. This makes angular resolution an OUTPUT of
        a measured quantity rather than fixed by module choice. Returns (keep_vectors: bool, detail)."""
        try:
            from .angular_resolution import select_max_l
            graphs = []
            for i in range(len(data.node_feats)):
                pos = np.asarray(data.positions[i], dtype=np.float32)
                ei = np.asarray(data.edges[i])
                graphs.append({"pos": pos, "edge_index": ei})
            y = np.asarray(data.y)
            if task == "classification":
                y = y.astype(np.float64)
            res = select_max_l(graphs, y, mu=mu, max_l=2)
            keep = int(res["max_l"]) >= 1
            detail = {"selected_max_l": int(res["max_l"]),
                      "marginals": {int(k): round(float(v), 4) for k, v in res["marginals"].items()},
                      "keep_vectors": bool(keep)}
            self._log(f"[AllGraph] angular_from_data -> max_l={res['max_l']} "
                      f"({'keep' if keep else 'drop'} l=1 vector channels)")
            return keep, detail
        except Exception as e:
            self._log(f"[AllGraph] angular_from_data skipped ({str(e)[:60]})")
            return True, None

    def _resolve_gibbs_beta(self, scores, prims):
        """Return the Gibbs inverse-temperature for the derived selection. self.gibbs_beta is either a fixed
        float (default 8.0, reproducible) or "auto", in which case beta is DERIVED per-fit at the knee of the
        fit-vs-commitment frontier of the actual solo energies (machinery.select_beta_by_elbow) -- the beta
        past which committing further (lower entropy) stops buying meaningful fit improvement. This makes the
        MDL temperature an OUTPUT of the energy landscape rather than a hand-set constant. Note the DEPLOYED
        primitive (argmin energy) is beta-independent, so "auto" changes only the reported alpha confidence
        distribution (the interpretability read-out), making it calibrated to how separated the energies
        actually are, never the selection itself."""
        if isinstance(self.gibbs_beta, str) and self.gibbs_beta.lower() == "auto":
            from ..machinery.gibbs_alpha import select_beta_by_elbow
            energies = np.array([scores[p] for p in prims], dtype=float)
            if len(energies) >= 2 and np.ptp(energies) > 1e-9:
                bstar, _ = select_beta_by_elbow(energies)
                return float(bstar)
            return 8.0            # degenerate (all-equal) energies -> fall back to the fixed default
        return float(self.gibbs_beta)

    def apply_canonicalization(self, data):
        """Apply the SAME kinematics quotient used at fit time to new (e.g. test) data: canonicalize its
        positions to the principal-axis frame and append them as node features, exactly as the fit path did
        to the training data. This keeps train and inference consistent when canonicalize_reuse fired --
        without it, the deployed net (trained on augmented, higher-dimensional node features) would receive
        the wrong feature dimension at test time. Returns the transformed data (a shallow copy); if
        canonicalization was not applied at fit time, returns the data unchanged."""
        if not getattr(self, "_canonicalization_applied", False):
            return data
        import copy
        from .canonicalization import canonicalize_data
        cdata, _ = canonicalize_data(data)
        out = copy.copy(data)
        cpos = cdata.positions if cdata.positions is not None else data.positions
        out.node_feats = [np.concatenate([np.asarray(nf, np.float32),
                                          np.asarray(cpos[i], np.float32)], axis=1)
                          for i, nf in enumerate(data.node_feats)]
        out.positions = cpos
        return out

    def explain(self, result=None, as_text=False, **kwargs):
        """Tier-1 faithful self-report: render this fitted model's own selected architecture as its
        explanation (see core/interpretability.py). Faithful by construction -- reads the same parameters
        the forward pass uses, with no auxiliary model. Pass the dict returned by fit() as `result` to
        surface the selection scores (gibbs energies) and the fit metric. Returns the report dict, or the
        formatted text block if as_text=True."""
        from .interpretability import explain as _explain, format_report
        R = _explain(self, result=result, **kwargs)
        return format_report(R) if as_text else R

    # ------------------------------------------------------------------- inference on new data / persistence
    def _train_candidate_contract(self, ctx, return_net=False):
        # These candidate/bake-off/sweep sub-fits are scatter-bound in the relational contracts; run them on CPU
        # when deploying on MPS (see _subfit_device). Exclude return_net (D1 price_singular), which hands back
        # a net + closure bound to self.device. `ctx` is a _SweepCtx bundling the sweep-invariant args.
        with self._subfit_device(ctx.contract, active=not return_net):
            return self._train_candidate_contract_impl(ctx, return_net=return_net)

    def _train_candidate_contract_impl(self, ctx, return_net=False):
        """Train ONE candidate contract (described by `ctx`, a _SweepCtx) on the train split, return held-out
        validation score (higher better: acc for classification, R2 for regression). Same budget across
        candidates for fairness.

        If return_net=True, returns (value, net, loss_closure, n_train) instead of just value, where
        loss_closure() is the mean training loss at the current (trained) parameters -- used by the D1
        singular-complexity pricing to estimate the LLC on the converged candidate. Default False keeps
        the plain float-returning contract used by the size/primitive sweeps."""
        data, contract, tr, va, task, n_out, epochs, edge_cutoff = ctx
        from ..models import (build_graph_schema, build_equivariant_graph_schema,
                              build_set_schema)
        torch.manual_seed(self.seed)
        n_in = data.node_feats[0].shape[1]
        y = np.asarray(data.y); yt = torch.as_tensor(y)
        # build a lightweight net for the contract
        if contract == "graph":
            net = build_graph_schema(n_in=n_in, width=self.width, depth=self.depth,
                                                 n_out=n_out, primitives=("gcn", "gin", "pna", "norm"), readout="mean")
            with_pos = False; use_edges = True
        elif contract == "equivariant":
            net = build_equivariant_graph_schema(n_in=n_in, c0=self.width, c1=max(self.width // 2, 2),
                                                             depth=self.depth, n_out=n_out,
                                                             primitives=("e_tp", "e_kan", "e_painn", "e_gate", "e_norm"))
            with_pos = True; use_edges = True
        else:  # set
            net = build_set_schema(n_in=n_in, width=self.width, depth=self.depth, n_out=n_out,
                                               primitives=("deepsets", "element_mlp", "norm"), readout="mean")
            with_pos = False; use_edges = False
        net = net.to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

        def fwd(ids):
            return self._forward_contract(net, data, ids, contract, with_pos, use_edges, edge_cutoff)

        tr = np.asarray(tr)
        for _ in range(epochs):
            np.random.shuffle(tr)
            for j in range(0, len(tr), self._tb()):
                ids = tr[j:j + self._tb()]
                opt.zero_grad()
                out = fwd(ids)
                target = yt[ids].long().to(self.device) if task == "classification" else yt[ids].float().unsqueeze(1).to(self.device)
                lf(out, target).backward(); opt.step()
        # validation score
        outs = []
        va = np.asarray(va)
        for j in range(0, len(va), 64):
            ids = va[j:j + 64]
            with torch.no_grad(): outs.append(fwd(ids).cpu())
        out = torch.cat(outs)
        _, value = self._metric(out, y[va], task)
        if return_net:
            # a mean-training-loss closure at the current (trained) parameters, for LLC estimation (D1).
            tr_arr = np.asarray(tr)

            def loss_closure():
                total, cnt = 0.0, 0
                for j in range(0, len(tr_arr), 64):
                    ids = tr_arr[j:j + 64]
                    o = fwd(ids)
                    tgt = (yt[ids].long().to(self.device) if task == "classification"
                           else yt[ids].float().unsqueeze(1).to(self.device))
                    b = len(ids)
                    total = total + lf(o, tgt) * b
                    cnt += b
                return total / max(cnt, 1)

            return value, net, loss_closure, len(tr_arr)
        return value

    @staticmethod
    def _edge_cut(D, edge_cutoff):
        """Effective distance cutoff for building edges from a pairwise-distance matrix D. An explicit
        `edge_cutoff` is used as an ABSOLUTE distance (e.g. 3.0 for molecular Angstrom data); `edge_cutoff=None`
        is SCALE-ADAPTIVE -- _EDGE_NN_FACTOR times the median nearest-neighbor distance -- so the resulting
        edge density is invariant to the coordinate scale (Angstrom, unit cube, particle-pT, ...)."""
        if edge_cutoff is not None:
            return edge_cutoff
        n = D.shape[0]
        if n < 2:
            return 0.0
        Dm = D + np.eye(n) * (float(D.max()) + 1.0)      # mask self-distances (the diagonal zeros)
        return _EDGE_NN_FACTOR * float(np.median(Dm.min(axis=1)))   # NN-factor x median nearest-neighbor dist

    def _forward_contract(self, net, data, ids, contract, with_pos, use_edges, edge_cutoff):
        """Forward one batch under a candidate contract. Builds edges from a distance cutoff if the
        contract needs edges and none were supplied; the set contract ignores edges entirely.

        The distance-cutoff edge construction depends only on the (static) positions and the fixed cutoff, so
        it is cached per (graph index, cutoff) on the instance. Without this the full pairwise-distance matrix
        and np.where were recomputed for every graph on every minibatch of every epoch of every candidate in
        a size sweep -- a large redundant cost. Node/position tensors are likewise cached once."""
        # lazily-built caches, keyed so they are safe across datasets within one fit
        if not hasattr(self, "_fc_cache"):
            self._fc_cache = {"node": {}, "pos": {}, "edge": {}}
        ncache, pcache, ecache = self._fc_cache["node"], self._fc_cache["pos"], self._fc_cache["edge"]

        def node_t(i):
            t = ncache.get(i)
            if t is None:
                t = torch.as_tensor(data.node_feats[i], dtype=torch.float32); ncache[i] = t
            return t

        def pos_t(i):
            t = pcache.get(i)
            if t is None:
                t = torch.as_tensor(data.positions[i], dtype=torch.float32); pcache[i] = t
            return t

        def edge_t(i):
            key = (i, edge_cutoff)
            t = ecache.get(key)
            if t is None:
                if data.edges is not None:
                    t = torch.as_tensor(data.edges[i], dtype=torch.long)
                else:                                    # build from distance cutoff on positions ONCE
                    P = np.asarray(data.positions[i]); D = np.linalg.norm(P[:, None] - P[None], axis=-1)
                    src, dst = np.where((D < self._edge_cut(D, edge_cutoff)) & (D > 0))
                    if len(src) == 0:
                        src, dst = np.arange(P.shape[0]), np.arange(P.shape[0])
                    t = torch.as_tensor(np.stack([src, dst]), dtype=torch.long)
                ecache[key] = t
            return t

        x, ei, p, b, ng = self._assemble_batch(
            ids, node_t, edge_t if use_edges else None, pos_t if with_pos else None)
        if contract == "set":
            return net(x, b, ng)
        if with_pos:
            return net(x, p, ei, b, ng)
        return net(x, ei, b, ng)

    def _log(self, *a):
        if self.verbose:
            print(*a)

    @contextmanager
    def _subfit_device(self, contract, active=True):
        """Temporarily run a SEARCH sub-fit on CPU when the deploy device is Apple-Silicon MPS and the contract
        is one measured faster on CPU (relational graph/equivariant/set -- scatter/index_add_ bound, ~15x on
        the op; and the launch-bound dense sequence/volumetric/4d -- see ilmarinen.device.prefer_cpu_on_mps).
        The many small candidate trainings in the width/depth sweep, the contract bake-off, and gibbs
        solo-scoring then run far faster on CPU -- the pathology behind ESOL "hanging" under --preset max on
        MPS. In the normal fit() path self.device is already CPU by here (pinned right after routing), so this
        is a no-op there; it still fires for standalone sub-fit entry points (e.g. select_architecture) whose
        self.device is the requested MPS. Safe because the forward-contract cache holds device-independent CPU
        tensors and every forward re-moves to self.device; only the sub-fit's throwaway candidate lives on CPU.
        The DEPLOYED model is untouched."""
        from ..device import prefer_cpu_on_mps
        if active and prefer_cpu_on_mps(contract, getattr(self, "device", "cpu")):
            orig = self.device
            self.device = torch.device("cpu")
            try:
                yield
            finally:
                self.device = orig
        else:
            yield

    #: ceiling on a SEARCH/SELECTION sub-fit's epoch budget. These sub-fits (gibbs primitive solo-scoring,
    #: width/depth sweeps, contract tie-break) only need enough training to RANK candidates, not to converge,
    #: so their budgets have a floor but no ceiling -- at a large deployed budget (e.g. epochs=500) the gibbs
    #: solo-scoring alone becomes 9 primitives x 250 silent epochs, dominating wall-clock and appearing hung.
    #: Cap them here; the DEPLOYED model still trains the full budget (and honors auto_epoch).
    _SEARCH_EPOCH_CAP = 80

    #: gibbs safety guard -- keep the argmax mixture instead of the derived single-primitive deploy when the
    #: latter is worse by more than this (in-sample R2/accuracy) margin. Catches degenerate solo primitives
    #: (which turn a good fit into a negative one) without reverting the small, expected in-sample drop a
    #: healthy gibbs solo shows.
    _GIBBS_REVERT_MARGIN = 0.25

    def _search_ep(self, ep):
        """Cap a search/selection sub-fit's epoch budget at _SEARCH_EPOCH_CAP (see the note above)."""
        return min(int(ep), self._SEARCH_EPOCH_CAP)

    def _make_stopper(self):
        """Return an _EarlyStopper for the DEPLOYED training loop when auto_epoch is on, else None (train the
        full self.epochs). It monitors each epoch's mean train loss and stops when the RELATIVE reduction stays
        below auto_epoch_min_delta for auto_epoch_patience epochs, after a min-epochs floor. Applied only to
        the final per-model loops -- never the fixed-budget search sub-fits."""
        if not getattr(self, "auto_epoch", None):
            return None
        return _EarlyStopper(min_delta=self.auto_epoch_min_delta, patience=self.auto_epoch_patience,
                             min_epochs=min(self.auto_epoch_min_epochs, self.epochs))

    #: auto_epoch=='val' holds out max(15%, _AUTO_VAL_MIN) samples as the monitor, but only if that still
    #: leaves enough to train on (val <= _AUTO_VAL_MAXFRAC of the data). Below that the held-out set is too
    #: small to be a reliable early-stop signal -- val loss on a handful of samples is noisy and, for
    #: classification, RISES with model confidence even while accuracy improves (verified on ItalyPowerDemand:
    #: a 10-sample val loss climbed for ~30 epochs while test accuracy was still improving). Small datasets
    #: therefore fall back to monitoring TRAIN loss (like auto_epoch='train'), which is stable.
    _AUTO_VAL_MIN = 50
    _AUTO_VAL_MAXFRAC = 0.35

    def _auto_val_split(self, n):
        """For auto_epoch=='val', return (train_idx, val_idx) with a held-out monitor of max(15%, _AUTO_VAL_MIN)
        samples. Returns val_idx=None (train on everything, monitor TRAIN loss) for 'train'/off, or when the
        data is too small to spare a reliable val set -- see the class note above."""
        if getattr(self, "auto_epoch", None) != "val":
            return np.arange(n), None
        k = max(int(0.15 * n), self._AUTO_VAL_MIN)
        if k > int(self._AUTO_VAL_MAXFRAC * n):          # can't hold out a reliable monitor without gutting training
            self._log(f"[AllGraph] auto_epoch=val: n={n} too small for a reliable held-out monitor "
                      f"(needs >= {self._AUTO_VAL_MIN} val samples); monitoring TRAIN loss instead")
            return np.arange(n), None
        rng = np.random.RandomState(self.seed + 7)
        perm = rng.permutation(n)
        return perm[k:], perm[:k]

    def _auto_val_loss(self, net, va_idx, batch_loss, batch=64):
        """Mean loss over the held-out val indices in EVAL mode (batchnorm/dropout off). batch_loss(idx_array)
        returns the scalar loss tensor for those indices; the net is restored to train mode afterward."""
        net.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(va_idx), batch):
                tot += float(batch_loss(va_idx[j:j + batch])); nb += 1
        net.train()
        return tot / max(nb, 1)

    def _epoch_iter(self, active=True):
        """Iterate the training epochs (range(self.epochs)), wrapped in a live tqdm progress bar when
        self.progress is on AND this is a deployed-model training loop (active=True). Internal search/
        bake-off sub-fits pass active=False so only the final per-model training shows a bar. Falls back to a
        plain range when progress is off or tqdm is unavailable, so nothing else changes."""
        rng = range(self.epochs)
        if not (active and getattr(self, "progress", False)):
            return rng
        try:
            from tqdm.auto import tqdm
        except Exception:
            return rng
        desc = getattr(self, "progress_desc", None)
        label = f"{desc} [{self.contract}]" if (desc and self.contract) else (desc or self.contract or "model")
        return tqdm(rng, desc=f"  training {label}", leave=False, unit="ep", dynamic_ncols=True)

    def fit(self, data: AllData, task="classification", n_out=None, primitives=None, tiebreak=False,
            select="argmax", select_size=False, stream=None):
        """Route the data to its schema, build it, train, and report. task in {classification,
        regression}. Returns a result dict. The heavy lifting (primitive/width/depth search) is the
        chosen schema's own machinery; AllGraph only dispatches and drives training.

        DATA USAGE / SPLITTING. `data` is the TRAINING set; fit() never sees a test set. The deployed model
        trains on ALL of `data`. Any in-fit DECISION routine (tiebreak, select_size, select='gibbs' solo
        energies, readout_select) holds out an internal validation split FROM `data` -- always the same
        seed-shared 25% (RandomState(self.seed), val_frac=0.25) -- trains candidates on the rest, and scores
        them on that held-out fold; the winning configuration is then retrained on the full `data`. So the
        validation fold informs the hyperparameter CHOICE only, never the deployed weights (standard bilevel
        protocol, no leakage). To measure generalization, evaluate the fitted AllGraph on a SEPARATE test set
        (as the validation runners do) -- do NOT read result["value"] as a generalization number: it is
        re-scored on the full training `data` and is therefore an IN-SAMPLE fit.

        tiebreak=True: for genuinely ambiguous geometric data (3D positions present), run a learned
        Level-1 bake-off (clean-solo over candidate CONTRACTS equivariant/graph/set) to pick the best
        contract on a held-out split, instead of the rule-based default. No-op for unambiguous data.

        select: how the deployed primitive is read off after the mixture trains.
          'argmax' (default) -- the DARTS trained-mixture argmax of alpha (fast, but co-adaptation
             vulnerable: a parameter-free branch can win the flat mixture).
          'gibbs'  -- the DERIVED selection: reselect each layer's primitive as the Gibbs-alpha over
             clean-solo energies (machinery.gibbs_alpha_select). Robust to co-adaptation by construction
             because the energy is a SOLO quantity. Costs one solo train per primitive; the result dict
             gains 'architecture_gibbs' and 'architecture_argmax' so the two readouts are comparable.
          'sparse' -- train the mixture with a SPARSITY PRICE on alpha (machinery.sparsity_price,
             Omega=-sum alpha^2 = pricing the effective #primitives IPR). Uses self.sparsity_mu as the
             price mu. The deployed net is the resulting COMPACT MIXTURE (not collapsed to one primitive):
             the frontier keeps a mixture only where a single primitive cannot do the job. The result
             dict gains 'ipr' (effective #primitives) and 'sparsity_mu'."""
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.select = select
        self._fc_cache = {"node": {}, "pos": {}, "edge": {}}   # fresh per-fit (static-per-graph tensor cache)
        self._subsample_cache = None            # the per-fit resident selection subsample (streaming; drawn once)
        # Restore the requested device and clear canonicalization state so a REUSED AllGraph instance starts
        # each fit clean: the relational->CPU fallback below must not persist onto a later dense fit, and a
        # prior fit's canonicalized-positions / applied-flag must not bleed into evaluation of new data.
        self.device = self._base_device
        self._canonicalized_positions = None
        self._canonicalization_applied = False
        self._geq_forward = None                # a prior discovered-group fit must not score this one's data
        self._latent_input_dim = None           # ditto for a prior nonlinear latent-contract fit
        # Apple-Silicon device policy: the relational contracts (graph/equivariant/set) are dominated by
        # scatter/index_add_ aggregation, which on MPS is markedly slower than CPU (~4x end-to-end on a full
        # ESOL fit, ~15x on the scatter op itself) with no offsetting speedup. When the data is relational
        # (has node_feats) and MPS was requested, transparently run the WHOLE fit on CPU -- deployment AND the
        # subsequent test eval, which reads self.device. This generalizes the per-sub-fit CPU routing to every
        # graph-type dataset, not just the max-preset search sub-fits.
        if getattr(data, "node_feats", None) is not None and str(self.device).startswith("mps"):
            self._log("[AllGraph] relational contract (graph/equivariant/set) requested on MPS -> running on CPU "
                      "(scatter/index_add_ bound; CPU is faster on Apple Silicon)")
            self.device = torch.device("cpu")
        # STAGE 1 -- KINEMATICS: resolve the contract (discovery / tiebreak / route / admissibility) and
        # apply any canonicalization quotient to the data.
        data, tb_detail = self._resolve_contract(data, task, n_out, tiebreak)
        # Opt-in dataset streaming (data.dense is a DenseSource): assert caller intent and guard the first-cut
        # scope BEFORE any full-dataset materialization. A no-op for resident inputs (the common case).
        self._check_streaming_supported(data, task, select, select_size, tiebreak, stream, n_out=n_out)
        # Apple-Silicon device policy (dense contracts): the relational override above catches graph/
        # equivariant/set pre-resolution (keyed on node_feats, so the in-resolution bake-off also runs on
        # CPU). The launch/sync-bound DENSE contracts -- sequence (recurrent per-step loops), 4d (the conv4d
        # temporal loop), volumetric (small/depthwise conv3d) -- are only known AFTER routing, so pin them
        # here: they are measurably faster on CPU than MPS at this package's sizes (sequence ~1.7x, 4d ~2.1x,
        # volumetric ~1.4x end-to-end). spatial (dense conv2d) and operator (matmul-dominated FFT) are FASTER
        # on MPS and excluded. MPS-gated via self._base_device (the requested device) so this never forces
        # CPU on CUDA; sits downstream of the self.device reset (top of fit) so it cannot become sticky, and
        # upstream of size-selection + deploy so every sub-fit and the final train inherit CPU. See
        # ilmarinen.device.prefer_cpu_on_mps.
        from ..device import prefer_cpu_on_mps
        if prefer_cpu_on_mps(self.contract, self._base_device) and not str(self.device).startswith("cpu"):
            self._log(f"[AllGraph] {self.contract} contract on MPS -> running on CPU "
                      f"(launch/sync-bound; CPU is faster on Apple Silicon)")
            self.device = torch.device("cpu")
        # STAGE 2 -- DEGREES OF FREEDOM: priced width/depth selection in-line before training.
        self._apply_size_selection(data, task, n_out, select_size)
        # B3: optionally deploy a latent-equivariant contract for a discovered NONLINEAR symmetry, when
        # enabled and the data carries geometry. Gated behind the joint discovery's confirmation; returns a
        # result (bypassing the normal builder) only when a symmetry is confirmed AND the contract is built.
        if getattr(self, "deploy_nonlinear_contract", False) and getattr(data, "positions", None) is not None:
            nlc = self._fit_nonlinear_contract(data, task, n_out)
            if nlc is not None:
                if tb_detail is not None:
                    nlc["tiebreak"] = tb_detail
                # this path bypasses the normal STAGE-4 tail below, so set the inference state predict()/save()
                # read (task drives classification-argmax vs regression-value formatting) here too.
                self._infer_task = task
                self._infer_readout = nlc.get("readout")
                return nlc
        builder = getattr(self, f"_fit_{self.contract}")
        result = builder(data, task, n_out, primitives)
        if tb_detail is not None:
            result["tiebreak"] = tb_detail
        if select == "gibbs":
            result = self._gibbs_reselect(data, task, n_out, result)
        elif select == "sparse" and hasattr(self.net, "cells"):
            ipr = float(np.mean([self._cell_ipr(c) for c in self.net.cells]))
            result["ipr"] = ipr
            result["sparsity_mu"] = float(self.sparsity_mu)
            result["effective_num_primitives"] = round(ipr, 2)
            self._log(f"[AllGraph] sparse: effective #primitives (IPR)={ipr:.2f} at mu={self.sparsity_mu}")
        # STAGE 4 -- OBSERVABLES: opt-in diagnostic read-outs (reuse computed quantities; change nothing).
        result = self._attach_diagnostics(result, data, task)
        # Retain the minimal inference context so predict()/save() can replay the EXACT deployed forward
        # path on new data: the task (classification vs regression head) and the sequence readout choice.
        self._infer_task = task
        self._infer_readout = result.get("readout")
        return result

    def _discover_equivariant_group(self, data):
        """Autonomous symmetry -> contract (STAGE 1 sub-step): discover the group the target respects and
        populate self.generated_equivariant_group; on no group, optionally run the nonlinear-symmetry
        diagnostic and route edgeless point sets to 'set'. Guard-returns when discovery is disabled or
        inapplicable, so the caller stays flat."""
        # AUTONOMOUS symmetry -> contract: discover WHICH group the target respects and emit its generators,
        # populating generated_equivariant_group automatically (no hand-specified group). Closes the loop:
        # discover the symmetry group -> generate the equivariant contract for it. When
        # discover_equivariant_contract="extended" the FULL dispatcher (metric O(p,q), U(n), Sp, SL,
        # conformal) is used; the default True uses the SO/Sim/Lorentz stability detector.
        if not (self.discover_equivariant_contract and self.generated_equivariant_group is None
                and getattr(data, "positions", None) is not None):
            return
        try:
            if self.discover_equivariant_contract == "extended":
                from .extended_groups import discover_group
                from .metric_discovery import generators_for_metric
                from .emlp_layer import symplectic_generators, special_linear_generators
                spec, ddetail = discover_group(data)
                # the dispatcher's routes emit a group name but often no explicit generators; synthesize
                # them so the EMLP contract can be built. Metric/unitary -> so(g); Sp -> sp(2n); SL ->
                # sl(n). All three now have a generated equivariant contract.
                if spec is not None and spec.get("gens") is None:
                    nm = spec["name"]
                    if nm.startswith(("O(", "U(", "SO(")) and spec.get("metric") is not None:
                        spec = dict(spec); spec["gens"] = generators_for_metric(spec["metric"])
                    elif nm.startswith("Sp("):
                        D = spec["vec_dim"]; spec = dict(spec)
                        spec["gens"], _ = symplectic_generators(D // 2)
                    elif nm.startswith("SL("):
                        spec = dict(spec); spec["gens"] = special_linear_generators(spec["vec_dim"])
                    else:
                        ddetail = {**ddetail, "reason": "group %s has no EMLP contract yet" % nm}
                        spec = None
            else:
                from .symmetry_contract import detect_symmetry_group
                spec, ddetail = detect_symmetry_group(data)
            self.discovered_group_detail = ddetail
            if spec is not None:
                self.generated_equivariant_group = spec
                self._log(f"[AllGraph] autonomous detection -> group {spec['name']} "
                          f"({len(spec['gens'])} generators); generating its equivariant contract")
            else:
                self._log(f"[AllGraph] autonomous detection: no group found "
                          f"({ddetail.get('reason','')}); using standard routing")
                # OPT-IN nonlinear (LaLiGAN) fallback: when the LINEAR menu finds nothing and the user
                # requested it, attempt nonlinear symmetry discovery (a symmetry linear only after a
                # learned coordinate change). This is expensive (trains an autoencoder) and conservative
                # (null-guarded), and a latent-space generator does not map onto an existing EMLP contract,
                # so it is reported as a DIAGNOSTIC (route_detail) rather than switching the contract --
                # it tells the user a nonlinear symmetry exists, without silently acting on an
                # unvalidated transform. Off by default; strictly a hard-case escalation.
                if getattr(self, "nonlinear_symmetry_fallback", False):
                    nldetail = self._try_nonlinear_symmetry(data)
                    if nldetail is not None:
                        self.route_detail = {**(self.route_detail or {}), "nonlinear_symmetry": nldetail}
                # no discovered group + positions but no edges: the equivariant/graph contracts need
                # edges, so route the edgeless point set to the SET contract (avoids an edge lookup on
                # None). This is the honest fallback when no symmetry is found.
                if getattr(data, "edges", None) is None and getattr(data, "node_feats", None) is not None:
                    self.contract = "set"
                    self.route_detail = {"level1": "autonomous: no group found -> set (edgeless points)"}
                    self._autonomous_forced_set = True
        except Exception as e:
            self._log(f"[AllGraph] autonomous detection failed ({type(e).__name__}); standard routing")

    def _resolve_contract(self, data, task, n_out, tiebreak):
        """STAGE 1 (kinematics): resolve the contract -- autonomous symmetry discovery, generated-
        equivariant / learned-tiebreak / rule-based routing, contract admissibility + enabled-contract restriction,
        and any canonicalization quotient applied to the data. Returns (data, tb_detail): `data` may be a
        canonicalized shallow copy, tb_detail carries the tiebreak bake-off scores (None if no tiebreak)."""
        tb_detail = None
        self._discover_equivariant_group(data)
        # a GENERATED equivariant contract (Phase 2) takes precedence when specified: the discovered group
        # defines a bespoke equivariant architecture that the eight built-ins may not cover.
        if self.generated_equivariant_group is not None and getattr(data, "positions", None) is not None:
            self.contract = "generated_equivariant"
            self.route_detail = {"level1": "generated equivariant contract (EMLP from group generators)",
                                 "n_generators": len(self.generated_equivariant_group["gens"])}
        elif getattr(self, "_autonomous_forced_set", False):
            pass   # autonomous detection found no group -> already routed to set above
        elif tiebreak and self._tiebreak_candidates(data) is not None:
            winner, scores, tb_detail = self.tiebreak(data, task=task, n_out=n_out)
            self.contract = winner
            self.route_detail = {"level1": "learned tie-break (clean-solo over contracts)", "tiebreak": tb_detail}
        else:
            # SYMMETRY-FIRST routing applies even WITHOUT the --tiebreak bake-off: when --symmetry_routing is
            # enabled and the geometric contract is ambiguous, route by the discovered symmetry (+ Phase-1
            # canonicalization); else fall back to the rule-based route. This is what makes --symmetry_routing
            # / --canonicalize (and the `med` preset, which sets them with tiebreak off) effective standalone.
            sym = self._symmetry_route(data, self._tiebreak_candidates(data)) if self.symmetry_routing else None
            if sym is not None:
                self.contract, _sdet = sym
                self.route_detail = {"level1": "symmetry routing (arch(G); no bake-off)", **_sdet}
            else:
                self.contract, self.route_detail = self.route(data)
        # CONTRACT ADMISSIBILITY (physicist's consistency condition, applied uniformly across ALL routing
        # paths): a relational contract (graph/equivariant) is well-posed only if the data supplies the field
        # content its dynamics need. The relational forward pass reads data.edges directly, so it requires
        # EXPLICIT edges (it does not build them from coordinates). If any routing path selected a relational
        # contract on edgeless data, the contract is inadmissible -> fall back to the always-constructible 'set'.
        # This generalizes the guard the autonomous path already applies ("no group found -> set"). A streaming
        # GraphSource carries edges INSIDE the source (data.edges is None but source.has_edges is True), so it is
        # admissible for graph/equivariant -- exclude it from the edgeless fallback.
        _stream_has_edges = self._is_streaming_graph(data) and getattr(data.node_feats, "has_edges", False)
        if self.contract in ("graph", "equivariant") and getattr(data, "edges", None) is None and not _stream_has_edges:
            self.route_detail = {**(self.route_detail or {}),
                                 "admissibility": f"'{self.contract}' inadmissible (no edges) -> set"}
            self.contract = "set"
        # CONTRACT RESTRICTION: if the resolved contract is disabled via enabled_contracts, fall back to the
        # nearest enabled, constructible contract. Applied uniformly after every routing path (rule-based,
        # tiebreak, autonomous) so no disabled schema is ever built. generated_equivariant is exempt
        # (it is an explicit discovered-group request, not an auto-route target).
        if self.contract != "generated_equivariant" and not self._contract_enabled(self.contract):
            _restricted_from = self.contract
            self.contract = self._resolve_enabled_fallback(self.contract, data)
            self.route_detail = {**(self.route_detail or {}),
                                 "contract_restriction": f"'{_restricted_from}' disabled -> '{self.contract}' "
                                 f"(enabled={sorted(self.enabled_contracts)})"}
        self._log(f"[AllGraph] routed -> {self.contract}  ({self.route_detail})")
        # if Phase-1 canonicalization was applied in tiebreak, feed the canonicalized coordinates to the
        # (plain, non-equivariant) set contract as extra node features. The set net then sees geometry that
        # is already aligned to the principal-axis frame, so it exploits the rotational invariance for free
        # -- effectively E(3)-invariant while reusing an existing contract. This is part of the KINEMATICS
        # stage (a quotient of the configuration space), so it MUST precede the degrees-of-freedom stage:
        # it changes the node-feature dimension, and the width/depth sweep must size the net for the FINAL
        # feature layout, not the pre-canonicalization one.
        if self._canonicalized_positions is not None:
            import copy
            data = copy.copy(data)
            cpos = self._canonicalized_positions
            data.node_feats = [np.concatenate([np.asarray(nf, np.float32),
                                               np.asarray(cpos[i], np.float32)], axis=1)
                               for i, nf in enumerate(data.node_feats)]
            data.positions = cpos
            self._canonicalization_applied = True    # so evaluation applies the SAME quotient to new data
        return data, tb_detail

    def _apply_size_selection(self, data, task, n_out, select_size):
        """STAGE 2 (degrees of freedom): priced width/depth (size) selection in-line before training.
        Mutates self.width/self.depth and route_detail per the select_size mode; a no-op when select_size
        is falsy or the contract has no cell sweep (operator / generated / Sp / SL)."""
        # STAGE 2 -- DEGREES OF FREEDOM (physicist's ordering): with the contract (contract) AND its final
        # field content (post-canonicalization node features) now fixed by the kinematics stage above,
        # select the mode count (width) and number of scales (depth) IN-LINE before the dynamics stage
        # (training). This makes the four-stage causal chain -- kinematics -> d.o.f. -> dynamics ->
        # observables -- a single flow rather than a detached pre-step. Only meaningful for the cell-based
        # contracts (set/graph/equivariant/dense); the generated/Sp/SL contracts have no width/depth sweep.
        # select_size may be a bool (True == "sequential", the original priced width-then-depth selector) or
        # a mode string: "sequential" | "area" (joint uniform width x depth area minimization) | "variable"
        # (generalized-area variable-width-per-layer selector; sets depth to the emergent effective depth and
        # width to the largest per-layer width it keeps, so the schema is then built at that size).
        size_mode = None
        if select_size:
            size_mode = "sequential" if select_size is True else str(select_size)
        # sequential/area size selection use _train_candidate_contract, which only builds the relational
        # schemas (graph/set/equivariant) -> those modes stay gated to relational contracts. The
        # variable-width selector featurizes to a fixed vector (relational OR dense of any grid rank), so it
        # applies to every DENSE contract too (sequence/spatial/volumetric/4d). The operator contract is the one
        # genuine exception: its size is Fourier-mode truncation (derived from grid resolution), not the
        # per-layer cell width the selector produces, and its objective is a field a(x)->u(x) MSE rather than
        # a scalar/label the featurize-then-probe selector optimizes -- so select_size does not apply there
        # (its width/depth remain settable directly). generated/Sp/SL have no cell sweep and are excluded too.
        _RELATIONAL = ("set", "graph", "equivariant")
        _SIZE_OK = _RELATIONAL + ("sequence", "spatial", "volumetric", "4d")
        # under streaming, the sweep runs on the bounded resident subsample (drawn once, ONLY when a sweep
        # actually runs); the chosen (K*, L*) then deploy-trains on the full stream. sel == data when resident.
        if size_mode == "variable" and self.contract in _SIZE_OK:
            sel = self._streaming_subsample(data) or data
            szdetail = self._select_size_variable(sel, task=task, n_out=n_out)
            self.route_detail = {**(self.route_detail or {}), "select_size": szdetail}
            self._log(f"[AllGraph] d.o.f. stage (variable) -> width={self.width}, depth={self.depth}")
        elif size_mode in ("sequential", "area") and self.contract in _RELATIONAL:
            sel = self._streaming_subsample(data) or data
            if size_mode == "sequential":
                szdetail = self.select_architecture(sel, task=task, n_out=n_out, contract=self.contract)
            else:
                szdetail = self.select_architecture_by_area(sel, task=task, n_out=n_out, contract=self.contract)
            self.route_detail = {**(self.route_detail or {}), "select_size": szdetail}
            self._log(f"[AllGraph] d.o.f. stage ({size_mode}) -> width={self.width}, depth={self.depth}")
        elif size_mode and size_mode not in ("sequential", "area", "variable"):
            raise ValueError(f"unknown select_size mode {size_mode!r}")

    def _alpha_penalty(self, net):
        """Sparsity price on the mixture: mu * sum_cells (-sum_p softmax(alpha)_p^2). Nonzero only when
        select='sparse'. Added to the training loss so gradient descent keeps a primitive only when it
        pays for its IPR cost. Returns a torch scalar (0.0 tensor when inactive)."""
        if self.select != "sparse" or self.sparsity_mu <= 0 or not hasattr(net, "cells"):
            return torch.zeros((), device=self.device)
        from ..machinery import sparsity_price
        return self.sparsity_mu * sum(sparsity_price(torch.softmax(c.alpha, 0)) for c in net.cells)

    def _cell_ipr(self, cell):
        from ..machinery import participation
        return participation(torch.softmax(cell.alpha, 0))

    def _break_alpha_symmetry(self, net):
        """One-time symmetry break for select='sparse' mixture alphas. The sparsity price (-sum alpha^2,
        _alpha_penalty) is permutation-SYMMETRIC, and the cells init alpha to torch.zeros (exactly uniform).
        A symmetric penalty has ZERO effective gradient at a symmetric point (equal gradient on every logit,
        which softmax's shift-invariance cancels), so from the uniform init the price cannot start sparsifying
        -- at realistic epoch budgets alpha stays uniform and ipr ~ #primitives regardless of sparsity_mu
        (the price only bites after ~150 epochs, once the data loss has slowly broken the tie). Seeding a
        small, deterministic perturbation gives the price a descent direction from epoch 1; the data loss and
        price then jointly settle the compact mixture (DARTS-style alpha init noise). No-op unless sparse."""
        if self.select != "sparse":
            return
        g = torch.Generator().manual_seed(int(self.seed) + 101)   # CPU generator -> deterministic, MPS-safe
        with torch.no_grad():
            for c in getattr(net, "cells", []):
                if hasattr(c, "alpha"):
                    c.alpha.add_((torch.randn(c.alpha.shape, generator=g) * 0.1).to(c.alpha.device))

    def _solo_scores(self, data, task, n_out, prims):
        """Train each primitive ALONE from scratch (single-primitive schema) in the current contract
        and score on a held-out split -- the clean-solo energies the derived Gibbs-alpha consumes. Reuses
        the same per-contract builders and forward paths as the mixture fit, so the solo comparison is
        apples-to-apples. Returns {primitive: held-out score (higher=better)}."""
        n_out = self._infer_nout(data.y, task, n_out)
        scores = {}
        # gibbs solo-scoring trains one net per primitive; in the relational contracts these are scatter-bound,
        # so route them to CPU when deploying on MPS (see _subfit_device).
        with self._subfit_device(self.contract):
            for p in prims:
                torch.manual_seed(self.seed + 11); np.random.seed(self.seed + 11)
                scores[p] = float(self._train_score_solo_primitive(data, task, n_out, p))
        return scores

    def _operator_solo_setup(self, data, primitive):
        """Build a single-primitive neural-operator net and move its (a, grid, u) tensors to device. Shared by
        the gibbs solo-score and solo-deploy paths so the operator contract is handled like the others (its config
        -- Fourier-mode budget, input channels, spatial dims -- mirrors _fit_operator)."""
        from ..models import build_operator_schema
        a = data.dense if isinstance(data.dense, torch.Tensor) else torch.tensor(np.asarray(data.dense), dtype=torch.float32)
        xg = data.grid if isinstance(data.grid, torch.Tensor) else torch.tensor(np.asarray(data.grid), dtype=torch.float32)
        u = torch.as_tensor(np.asarray(data.y), dtype=torch.float32)
        sdims = getattr(data, "spatial_dims", 1)
        in_ch = a.shape[-1] if a.dim() == 2 + sdims else 1
        grid_min = min(int(s) for s in a.shape[1:1 + sdims]); modes = max(2, min(12, grid_min // 2))
        net = build_operator_schema(width=self.width, depth=self.depth, n_out=1,
                                                primitives=(primitive,), modes=modes, in_channels=in_ch,
                                                spatial_dims=sdims).to(self.device)
        return net, a.to(self.device), xg.to(self.device), u.to(self.device)

    def _train_score_solo_primitive(self, data, task, n_out, primitive):
        """One primitive, trained alone from scratch on a train split, scored on a held-out split, in the
        current contract. Mirrors _train_candidate_contract but at the primitive granularity."""
        from ..models import (build_schema, build_spatial_schema,
                              build_volumetric_schema, build_grid4d_schema,
                              build_graph_schema, build_equivariant_graph_schema,
                              build_set_schema)
        mod = self.contract
        # split
        n = len(data.node_feats) if data.node_feats is not None else len(data.dense)
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n); nval = max(1, int(0.25 * n)); va, tr = perm[:nval], perm[nval:]
        ep = self._search_ep(max(5, self.epochs // 2))          # ranking budget, capped (see _search_ep)
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        yt = torch.as_tensor(np.asarray(data.y))

        if mod in ("sequence", "spatial", "volumetric", "4d"):
            X = data.dense
            if mod == "sequence" and X.dim() == 2: X = X.unsqueeze(-1)
            # each dense contract's builder takes its OWN size argument (spatial hw, volumetric dhw, 4d
            # grid_shape); build_grid4d_schema has no grid_shape default, so a generic call breaks 4d.
            if mod == "sequence":
                net = build_schema(n_in=X.shape[-1], width=self.width, depth=self.depth,
                                               n_out=n_out, primitives=(primitive,)).to(self.device)
            elif mod == "spatial":
                net = build_spatial_schema(n_in=X.shape[1], width=self.width, hw=X.shape[-1],
                                                       depth=self.depth, n_out=n_out, primitives=(primitive,)).to(self.device)
            elif mod == "volumetric":
                net = build_volumetric_schema(n_in=X.shape[1], width=self.width, dhw=self._vol_work_dhw(X.shape[-1]), vol_size=X.shape[-1],
                                                          depth=self.depth, n_out=n_out, primitives=(primitive,)).to(self.device)
            else:  # 4d
                net = build_grid4d_schema(n_in=X.shape[1], grid_shape=tuple(X.shape[2:]), width=self.width,
                                                  depth=self.depth, n_out=n_out, primitives=(primitive,)).to(self.device)
            fwd = (lambda ids: net.forward_seq_readout(X[ids].to(self.device), 1).squeeze(1)) if mod == "sequence" \
                  else (lambda ids: net(X[ids].to(self.device)))
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd(self._DENSE_WEIGHT_DECAY))
            tr = np.asarray(tr)
            for _ in range(ep):
                np.random.shuffle(tr)
                for j in range(0, len(tr), self._tb()):
                    ids = tr[j:j + self._tb()]; opt.zero_grad()
                    out = fwd(ids)
                    target = yt[ids].long().to(self.device) if task == "classification" else yt[ids].float().unsqueeze(1).to(self.device)
                    lf(out, target).backward(); opt.step()
            outs = []
            for j in range(0, len(va), 64):
                ids = np.asarray(va[j:j + 64])
                with torch.no_grad(): outs.append(fwd(ids).cpu())
            out = torch.cat(outs)
            return self._score(out, np.asarray(data.y)[va], task)

        if mod == "operator":
            # operator contract: train a single-primitive neural-operator net and score the held-out FIELD R2,
            # mirroring _fit_operator (function a(x) -> field u(x), per-grid-point MSE). Higher is better.
            net, a_d, x_d, u_d = self._operator_solo_setup(data, primitive)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
            tr = np.asarray(tr)
            for _ in range(ep):
                np.random.shuffle(tr)
                for j in range(0, len(tr), self._tb()):
                    ids = tr[j:j + self._tb()]; opt.zero_grad()
                    ((net(a_d[ids], x_d[ids]) - u_d[ids]) ** 2).mean().backward(); opt.step()
            va = np.asarray(va)
            with torch.no_grad():
                pred = net(a_d[va], x_d[va]).cpu().numpy()
            uy = np.asarray(data.y, dtype=np.float32)[va]
            return float(1.0 - ((pred - uy) ** 2).sum() / (((uy - uy.mean()) ** 2).sum() + 1e-12))

        # relational (graph / equivariant / set): reuse the candidate-contract machinery at primitive level
        n_in = data.node_feats[0].shape[1]
        if mod == "graph":
            net = build_graph_schema(n_in=n_in, width=self.width, depth=self.depth,
                                                 n_out=n_out, primitives=(primitive,), readout="mean")
            with_pos, use_edges = False, True
        elif mod == "equivariant":
            net = build_equivariant_graph_schema(n_in=n_in, c0=self.width, c1=max(self.width // 2, 2),
                                                             depth=self.depth, n_out=n_out, primitives=(primitive,))
            with_pos, use_edges = True, True
        else:
            net = build_set_schema(n_in=n_in, width=self.width, depth=self.depth, n_out=n_out,
                                               primitives=(primitive,), readout="mean")
            with_pos, use_edges = False, False
        net = net.to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
        tr = np.asarray(tr)
        for _ in range(ep):
            np.random.shuffle(tr)
            for j in range(0, len(tr), self._tb()):
                ids = tr[j:j + self._tb()]; opt.zero_grad()
                out = self._forward_contract(net, data, ids, mod, with_pos, use_edges, 3.0)
                target = yt[ids].long().to(self.device) if task == "classification" else yt[ids].float().unsqueeze(1).to(self.device)
                lf(out, target).backward(); opt.step()
        outs = []
        for j in range(0, len(va), 64):
            ids = np.asarray(va[j:j + 64])
            with torch.no_grad(): outs.append(self._forward_contract(net, data, ids, mod, with_pos, use_edges, 3.0).cpu())
        out = torch.cat(outs)
        return self._score(out, np.asarray(data.y)[va], task)

    def _score(self, out, y, task):
        """Held-out score, higher = better: accuracy for classification, R^2 for regression. The value half
        of _metric (which also returns the metric NAME); kept as a named alias for the scoring call sites."""
        return self._metric(out, y, task)[1]

    def _gibbs_reselect(self, data, task, n_out, result):
        """After the mixture trains, reselect the deployed primitive per layer via the DERIVED Gibbs-alpha
        over clean-solo energies (machinery.gibbs_alpha_select), instead of the co-adaptation-vulnerable
        mixture argmax. Reports both readouts so the difference is visible. Only the graph/equivariant/set
        and dense contracts that expose net.cells are handled; others pass through unchanged.

        Because the mixture weights are co-adapted, the deployed net is NOT the compiled mixture but a
        FRESH single-primitive net of the winning primitive trained on the full data (compile_supergraph
        would copy co-adapted weights; the derived selection's whole point is the solo primitive). Setting
        self.net to this deployment net makes the reported architecture the DERIVED one.

        NOTE ON result["value"]: this is re-scored on the FULL training data the deployed net trained on --
        i.e. an IN-SAMPLE (training-set) fit, consistent with how the main fit paths report result["value"].
        It is NOT a generalization estimate; measure that on held-out data (evaluate the fitted AllGraph on a
        separate test set, as the validation runners do). The clean-solo ENERGIES that drive the gibbs choice
        are held-out (each primitive scored on the internal 25% val split), so the selection is honest even
        though the reported value is in-sample."""
        from ..machinery import gibbs_alpha_select
        net = self.net
        if not hasattr(net, "cells"):
            return result
        prims = list(net.cells[0].primitives)
        if len(prims) < 2:
            return result
        # clean-solo energy per primitive: train each ALONE from scratch, score on a held-out split. Under
        # streaming the SOLO scoring (selection) runs on the bounded resident subsample; the winner then
        # DEPLOY-trains on the full stream (_train_deploy_solo / _score_full below), which have streaming branches.
        scores = self._solo_scores(self._streaming_subsample(data) or data, task, n_out, prims)
        gsel = gibbs_alpha_select(lambda p: scores[p], prims, beta=self._resolve_gibbs_beta(scores, prims))
        best = gsel["best"]
        result["architecture_argmax"] = result.get("architecture")
        result["architecture_gibbs"] = [best] * len(net.cells)
        # the Gibbs mixture WEIGHTS w = softmax(alpha) (report: w_p ~ e^{-beta*Psi_p}); named _weights, not
        # _alpha, because these are the normalized weights, not the logits alpha the docs reserve that symbol for.
        result["gibbs_weights"] = gsel["alpha"]
        result["gibbs_alpha"] = result["gibbs_weights"]   # deprecated alias (holds w, not the logits)
        result["gibbs_energies"] = gsel["energies"]
        result["selected_primitive"] = best
        result["architecture"] = result["architecture_gibbs"]
        # DEPLOY: train the winning single-primitive net on the FULL training data and re-score. SAFETY GUARD:
        # the derived single primitive must not DEGRADE the argmax mixture. Solo-scoring can crown a primitive
        # that cannot actually fit the task on its own -- e.g. on HeatDiffusion3D (4d PDE) it picked 'norm'
        # (a normalization primitive), whose solo net scores R2~0 while the conv4d-bearing argmax mixture
        # reaches R2~0.88 -- so deploying it turns a good fit into a negative-R2 one. Keep whichever net scores
        # better (same in-sample metric, higher=better for both R2 and accuracy).
        argmax_net = self.net                       # the trained argmax mixture, kept as the fallback
        argmax_val = result.get("value")
        deploy = self._train_deploy_solo(data, task, n_out, best)
        if deploy is not None:
            val = self._score_full(deploy, data, task)
            # A normal gibbs solo fits the TRAINING data a little worse than the higher-capacity mixture
            # (expected -- it trades in-sample fit for co-adaptation robustness), so revert ONLY on a
            # CATASTROPHIC drop: a degenerate primitive that cannot fit the task at all (e.g. 'norm' on the 4d
            # PDE -> R2~0 vs the conv4d mixture ~0.88). 0.25 in R2/accuracy units is far larger than any
            # honest in-sample fit-vs-robustness trade.
            catastrophic = (argmax_val is not None and val < argmax_val - self._GIBBS_REVERT_MARGIN)
            if not catastrophic:
                self.net = deploy                   # keep the derived (gibbs) net
                result["value_argmax"] = argmax_val
                result["value"] = float(val)
                result["n_params"] = sum(p.numel() for p in deploy.parameters())
            else:
                self.net = argmax_net               # degenerate solo -> keep the argmax mixture
                result["architecture"] = result.get("architecture_argmax")
                result["value_gibbs_deploy"] = float(val)
                result["gibbs_reverted"] = True
        reverted = result.get("gibbs_reverted", False)
        self._log(f"[AllGraph] gibbs reselect -> {best}  (argmax was {result['architecture_argmax']}); "
                  f"deployed value={result.get('value'):.4f}"
                  + (f"  [REVERTED to argmax: solo {result.get('value_gibbs_deploy'):.4f} < "
                     f"{argmax_val:.4f}]" if reverted else ""))
        return result

    def _train_deploy_solo(self, data, task, n_out, primitive):
        """Train the winning single-primitive net on ALL training data (full epochs) for deployment.
        Mirrors the per-contract fit paths but with a fixed one-primitive vocabulary. Returns the trained
        net, or None if the contract isn't cell-based."""
        from ..models import (build_schema, build_spatial_schema,
                              build_volumetric_schema, build_grid4d_schema,
                              build_graph_schema, build_equivariant_graph_schema,
                              build_set_schema)
        n_out = self._infer_nout(data.y, task, n_out)
        mod = self.contract
        torch.manual_seed(self.seed + 3); np.random.seed(self.seed + 3)
        if mod in ("sequence", "spatial", "volumetric", "4d"):
            X = data.dense
            if self._is_streaming(X):
                # wrap spatial/volumetric via _as_grid (channel fix-up + shape metadata) exactly as _fit_grid;
                # sequence and 4d present full-rank samples already. _train_dense's streaming branch handles X.
                _rank = {"spatial": 4, "volumetric": 5}.get(mod)
                if _rank is not None:
                    X = self._as_grid(X, _rank)
            elif mod == "sequence" and X.dim() == 2:
                X = X.unsqueeze(-1)
            if mod == "sequence":
                net = build_schema(n_in=X.shape[-1], width=self.width, depth=self.depth,
                                               n_out=n_out, primitives=(primitive,)).to(self.device)
                net = self._train_dense(net, X, data.y, task,
                                        forward=lambda xb: net.forward_seq_readout(xb, 1).squeeze(1))
            elif mod == "spatial":
                net = build_spatial_schema(n_in=X.shape[1], width=self.width, hw=X.shape[-1],
                                                       depth=self.depth, n_out=n_out, primitives=(primitive,)).to(self.device)
                net = self._train_dense(net, X, data.y, task)
            elif mod == "volumetric":
                net = build_volumetric_schema(n_in=X.shape[1], width=self.width, dhw=self._vol_work_dhw(X.shape[-1]), vol_size=X.shape[-1],
                                                          depth=self.depth, n_out=n_out, primitives=(primitive,)).to(self.device)
                net = self._train_dense(net, X, data.y, task)
            else:
                net = build_grid4d_schema(n_in=X.shape[1], grid_shape=tuple(X.shape[2:]), width=self.width,
                                                  depth=self.depth, n_out=n_out, primitives=(primitive,)).to(self.device)
                net = self._train_dense(net, X, data.y, task)
            return net
        if mod == "operator":
            # operator contract: deploy the winning single-primitive neural-operator net on ALL data (field MSE).
            if self._is_streaming_operator(data):
                src = data.dense
                net = self._operator_solo_net(src.a_shape, src.spatial_dims, primitive)
                opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
                n = len(src); idx = np.arange(n)
                for _ in range(self.epochs):
                    np.random.shuffle(idx)
                    for j in range(0, n, self._tb()):
                        ids = idx[j:j + self._tb()]; opt.zero_grad()
                        ab, xb, ub = src.a(ids).to(self.device), src.grid(ids).to(self.device), src.u(ids).to(self.device)
                        ((net(ab, xb) - ub) ** 2).mean().backward(); opt.step()
                return net
            net, a_d, x_d, u_d = self._operator_solo_setup(data, primitive)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
            n = a_d.shape[0]; idx = np.arange(n)
            for _ in range(self.epochs):
                np.random.shuffle(idx)
                for j in range(0, n, self._tb()):
                    ids = idx[j:j + self._tb()]; opt.zero_grad()
                    ((net(a_d[ids], x_d[ids]) - u_d[ids]) ** 2).mean().backward(); opt.step()
            return net
        # relational
        streaming_graph = self._is_streaming_graph(data)
        n_in = data.node_feats.n_in if streaming_graph else data.node_feats[0].shape[1]
        if mod == "graph":
            net = build_graph_schema(n_in=n_in, width=self.width, depth=self.depth,
                                                 n_out=n_out, primitives=(primitive,), readout="mean").to(self.device)
            with_pos, use_edges = False, True
        elif mod == "equivariant":
            net = build_equivariant_graph_schema(n_in=n_in, c0=self.width, c1=max(self.width // 2, 2),
                                                             depth=self.depth, n_out=n_out, primitives=(primitive,)).to(self.device)
            with_pos, use_edges = True, True
        else:
            net = build_set_schema(n_in=n_in, width=self.width, depth=self.depth, n_out=n_out,
                                               primitives=(primitive,), readout="mean").to(self.device)
            with_pos, use_edges = False, False
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        yt = torch.as_tensor(np.asarray(data.y))
        n = len(data.node_feats); idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for j in range(0, n, self._tb()):
                ids = idx[j:j + self._tb()]; opt.zero_grad()
                out = (self._stream_relational_out(net, data, ids, mod) if streaming_graph
                       else self._forward_contract(net, data, ids, mod, with_pos, use_edges, 3.0))
                target = yt[ids].long().to(self.device) if task == "classification" else yt[ids].float().unsqueeze(1).to(self.device)
                lf(out, target).backward(); opt.step()
        return net

    def _operator_solo_net(self, a_shape, sdims, primitive):
        """Build a single-primitive neural-operator net sized from the field SHAPE metadata alone (in_channels /
        Fourier-mode budget), so a streaming gibbs solo-deploy net is the SAME size as the resident/subsample
        one (which _operator_solo_setup builds from the materialized tensor)."""
        from ..models import build_operator_schema
        in_ch = a_shape[-1] if len(a_shape) == 2 + sdims else 1
        grid_min = min(int(s) for s in a_shape[1:1 + sdims]); modes = max(2, min(12, grid_min // 2))
        return build_operator_schema(width=self.width, depth=self.depth, n_out=1, primitives=(primitive,),
                                     modes=modes, in_channels=in_ch, spatial_dims=sdims).to(self.device)

    def _stream_relational_out(self, net, data, ids, mod):
        """Streaming relational forward for the gibbs deploy/score path: collate the minibatch from the
        GraphSource (via the streaming-aware _forward_relational / _subbatch_sets, cache=None) instead of the
        resident-index-only _forward_contract."""
        if mod == "set":
            Xb, bb, ng = self._subbatch_sets(data, ids, cache=None)
            return net(Xb, bb, ng)
        return self._forward_relational(net, data, ids, with_pos=(mod == "equivariant"), cache=None)

    def _selected_net_builder(self, data, task):
        """Return a zero-arg callable that rebuilds the SELECTED architecture as a FRESH, UNTRAINED net of
        identical structure, or None if the deployed net is not a single rebuildable nn.Module (e.g. the
        attn-dict readout or the generated-equivariant dict-net paths, which have no single module to clone).

        Strategy: deep-copy the deployed `self.net` (capturing its exact structure -- selected primitives,
        width, depth, readout -- without re-deriving builder config) and FULLY re-initialize every parameter
        so the copy starts untrained. Full reinit (not just modules exposing reset_parameters, which covers
        only a subset here) is required so no trained weight leaks into the developmental trajectory --
        verified by premise-check (the rebuilt net returns to an untrained loss ~ ln(#classes))."""
        import copy, math
        net = self.net
        if net is None or not hasattr(net, "parameters") or isinstance(net, dict):
            return None
        template = net

        def build():
            fresh = copy.deepcopy(template)
            g = torch.Generator().manual_seed(int(self.seed))
            with torch.no_grad():
                for p in fresh.parameters():
                    if p.dim() >= 2:
                        fan_in = p.shape[1] if p.dim() == 2 else max(int(p[0].numel()), 1)
                        bound = 1.0 / math.sqrt(max(fan_in, 1))
                        p.copy_(torch.empty(p.shape).uniform_(-bound, bound, generator=g))
                    else:
                        p.zero_()
            # let modules with a principled reset (norm layers, etc.) override the generic init
            for mod in fresh.modules():
                if hasattr(mod, "reset_parameters"):
                    try:
                        mod.reset_parameters()
                    except Exception:
                        pass
            return fresh.to(self.device)

        return build

    def _score_full(self, net, data, task):
        """Score a deployed net on the training data's own labels (in-sample proxy for the report's
        'value' field; the runner re-evaluates on the held-out TEST split separately)."""
        mod = self.contract
        if mod == "operator":
            if self._is_streaming_operator(data):        # streamed two-pass field-R2 (returns the scalar)
                return float(self._stream_operator_eval(net, data.dense, 32))
            # field a(x) -> u(x): score the in-sample field R2, matching _fit_operator's metric.
            a = data.dense if isinstance(data.dense, torch.Tensor) else torch.tensor(np.asarray(data.dense), dtype=torch.float32)
            xg = data.grid if isinstance(data.grid, torch.Tensor) else torch.tensor(np.asarray(data.grid), dtype=torch.float32)
            with torch.no_grad():
                pred = net(a.to(self.device), xg.to(self.device)).cpu().numpy()
            uy = np.asarray(data.y, dtype=np.float32)
            return float(1.0 - ((pred - uy) ** 2).sum() / (((uy - uy.astype(np.float64).mean()) ** 2).sum() + 1e-12))
        y = np.asarray(data.y)
        if mod in ("sequence", "spatial", "volumetric", "4d"):
            X = data.dense
            fwd = (lambda xb: net.forward_seq_readout(xb, 1).squeeze(1)) if mod == "sequence" else None
            if self._is_streaming(X):
                # chunked accumulated score (no _report side effect on self.net -- this scores a candidate).
                from .allgraph_streaming import _StreamMetric
                _rank = {"spatial": 4, "volumetric": 5}.get(mod)
                src = self._as_grid(X, _rank) if _rank is not None else X
                acc = _StreamMetric(task, y); n = len(src)
                for j in range(0, n, 128):
                    ids = np.arange(j, min(j + 128, n))
                    with torch.no_grad():
                        out = (fwd or (lambda xb: net(xb)))(src.get(ids).to(self.device)).cpu()
                    acc.update(out, y[ids])
                return acc.result()[1]
            if mod == "sequence" and X.dim() == 2: X = X.unsqueeze(-1)
            out = self._deploy_grid_eval(net, X, 128, forward=fwd)
        elif self._is_streaming_graph(data):
            outs = []; n = len(data.node_feats)
            for j in range(0, n, 64):
                ids = np.arange(j, min(j + 64, n))
                with torch.no_grad():
                    outs.append(self._stream_relational_out(net, data, ids, mod).cpu())
            out = torch.cat(outs)
        else:
            with_pos = (mod == "equivariant"); use_edges = mod in ("graph", "equivariant")
            outs = []; n = len(data.node_feats)
            for j in range(0, n, 64):
                ids = np.arange(j, min(j+64, n))
                with torch.no_grad():
                    outs.append(self._forward_contract(net, data, ids, mod, with_pos, use_edges, 3.0).cpu())
            out = torch.cat(outs)
        return self._score(out, y, task)

    # ================================================================= per-contract fit paths
    def _infer_nout(self, y, task, n_out):
        if n_out is not None:
            return n_out
        if task == "regression":
            return 1
        return int(np.asarray(y).max()) + 1

    def _prefetch_depth(self):
        """Prefetch depth from stream_prefetch: None/False/0 -> 0 (off), True -> 1, int k -> k. Single source
        of truth for whether async prefetch is active."""
        p = getattr(self, "stream_prefetch", False)
        if p is True:
            return 1
        if not p:
            return 0
        return int(p)

    def _prefetch_batches(self, pm, batch_size, fetch, depth):
        """Yield (ids, payload) for each minibatch of the fixed permutation `pm`, with `fetch(ids)` -> RNG-free
        CPU payload run up to `depth` batches AHEAD on ONE background daemon thread through a bounded queue. The
        permutation is already materialized on the main thread (RNG consumed), so the batch ORDER is fixed and no
        RNG crosses the thread boundary -> bit-identical to running fetch inline. A worker (fetch) exception is
        re-raised on the main thread after the queue drains.

        Robustness: if the CONSUMER (compute on the main thread) raises mid-epoch while the producer is blocked
        on a full queue, a plain blocking put()/join() would deadlock. So the producer puts with a timeout and
        checks a `stop` Event, and the generator's finally sets `stop` and drains the queue -- the producer then
        unblocks, observes stop, and exits, so join() always completes (no hang, no leaked thread)."""
        import queue as _queue
        import threading as _threading
        slices = [pm[j:j + batch_size] for j in range(0, len(pm), batch_size)]
        q = _queue.Queue(maxsize=max(1, depth))
        box = {}
        stop = _threading.Event()
        _SENTINEL = object()

        def _put(item):                                  # blocking put that yields to a stop request
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.2)
                    return
                except _queue.Full:
                    continue

        def _produce():
            try:
                for ids in slices:
                    if stop.is_set():
                        return
                    _put((ids, fetch(ids)))
            except BaseException as e:                    # surface on the main thread, don't die silently
                box["err"] = e
            finally:
                if not stop.is_set():
                    _put(_SENTINEL)

        t = _threading.Thread(target=_produce, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                yield item
            if "err" in box:
                raise box["err"]
        finally:
            stop.set()                                   # tell the producer to stop
            try:                                         # unblock a producer parked on a full queue
                while True:
                    q.get_nowait()
            except _queue.Empty:
                pass
            t.join()

    def _run_epochs(self, net, opt, tr_idx, va_idx, stopper, batch_loss, batch_size, permute, show_progress=True,
                    prefetch=None):
        """Shared minibatch training loop for every contract's deploy fit. `batch_loss(ids)` -> scalar loss (the
        contract-specific forward + target); `permute(idx)` -> one epoch's permuted index sequence (each contract
        keeps its own RNG -- torch for dense, numpy for the rest -- so determinism is preserved). Handles the
        alpha sparsity penalty, the running-train-loss early-stop mean (accumulated on-device, read only when a
        stopper monitors TRAIN loss), and val-monitored stopping. Callers own net/opt/data-prep, the val split,
        the stopper, and the post-loop eval.

        `prefetch` (optional): a (fetch, compute) pair for async input prefetch. When supplied AND the prefetch
        depth is > 0, each batch's RNG-free fetch runs a few batches ahead on a worker thread and `compute(ids,
        payload)` runs the device-move + forward + loss on the main thread -- bit-identical to the inline path
        (which is `batch_loss = lambda ids: compute(ids, fetch(ids))`). The val path always uses batch_loss."""
        track = stopper is not None and va_idx is None
        depth = self._prefetch_depth() if prefetch is not None else 0
        for _ in self._epoch_iter(show_progress):
            pm = permute(tr_idx)                          # RNG draw on the MAIN thread, BEFORE any fetch
            run, nb = None, 0
            if depth > 0:
                fetch, compute = prefetch
                batches = self._prefetch_batches(pm, batch_size, fetch, depth)
            else:
                batches = ((pm[j:j + batch_size], None) for j in range(0, len(pm), batch_size))
            for ids, payload in batches:
                opt.zero_grad()
                loss = (compute(ids, payload) if depth > 0 else batch_loss(ids)) + self._alpha_penalty(net)
                loss.backward()
                # Skip a non-finite update instead of corrupting the weights: deep stacks over long sequences
                # can transiently blow a batch to inf/NaN; keeping the last finite weights lets training
                # continue on the well-behaved batches rather than poisoning every downstream step.
                if self._grads_finite(net):
                    opt.step()
                if track:
                    run = loss.detach() if run is None else run + loss.detach(); nb += 1
            if stopper is not None:
                m = self._auto_val_loss(net, va_idx, batch_loss) if va_idx is not None else float(run / max(nb, 1))
                if stopper.step(m):
                    break
        return net

    def _train_dense(self, net, X, y, task, forward=None, show_progress=False):
        """Shared training loop for dense-grid schemas (sequence/spatial/volumetric/4d). show_progress
        shows the epoch bar only for the DEPLOYED fit (not the readout bake-off / solo-score sub-fits)."""
        fwd = forward if forward is not None else (lambda xb: net(xb))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd(self._DENSE_WEIGHT_DECAY))
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        tr_idx, va_idx = self._auto_val_split(len(X)) if show_progress else (np.arange(len(X)), None)
        stopper = self._make_stopper() if show_progress else None   # early-stop only the deployed fit
        if show_progress:
            self._break_alpha_symmetry(net)     # sparse: seed the mixture off uniform (deployed fit only)
        yt = torch.as_tensor(y)
        yt_dev = (yt.long() if task == "classification" else yt.float().unsqueeze(1)).to(self.device)
        if not self._is_streaming(X):
            # RESIDENT FAST PATH (unchanged): the inputs/targets are STATIC across epochs -- move them to the
            # compute device ONCE and index on-device each step, instead of re-copying every minibatch every
            # epoch. On unified-memory (MPS) this is free; on CUDA it adds only |X| to VRAM (<< activations).
            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            Xd = X.to(self.device)
            tr_idx_dev = torch.as_tensor(np.asarray(tr_idx), device=self.device)
            def _dloss(b):
                if not torch.is_tensor(b):                   # val path passes numpy index slices
                    b = torch.as_tensor(np.asarray(b), device=self.device)
                return lf(fwd(Xd[b]), yt_dev[b])
            # dense permutes on-device via the CPU torch RNG (identical stream to the pre-refactor loop)
            permute = lambda idx: idx[torch.randperm(len(idx)).to(self.device)]
            return self._run_epochs(net, opt, tr_idx_dev, va_idx, stopper, _dloss, self._tb(), permute, show_progress)
        # STREAMING BRANCH: X is a DenseSource -- fetch + H2D only the current minibatch each step, so the full
        # (n, *sample_shape) tensor is never resident. The labels stay resident (small). This is bit-for-bit
        # equivalent to the resident path: the ONLY change is dropping the trailing `.to(self.device)` on the
        # permutation index (a device move consumes no RNG), so torch.randperm draws from the identical global
        # CPU RNG state at the identical call site -> identical batch membership per step -> identical grads.
        pin = self._resolve_pin()
        tr_idx = torch.as_tensor(np.asarray(tr_idx))         # CPU int64 (NOT moved to device)
        # split fetch (RNG-free CPU read, prefetch-safe) from compute (device move + forward + loss, main thread)
        def _dfetch(b):
            ids = b if torch.is_tensor(b) else torch.as_tensor(np.asarray(b))   # val path passes numpy slices
            Xb = X.get(ids)                                  # (len(ids), *sample_shape) CPU float32; RNG-free
            if pin:
                Xb = Xb.pin_memory()
            return ids, Xb
        def _dcompute(b, payload):
            ids, Xb = payload
            Xb = Xb.to(self.device, non_blocking=pin)
            return lf(fwd(Xb), yt_dev[ids.to(self.device)])
        _dloss = lambda b: _dcompute(b, _dfetch(b))          # fused: val path + non-prefetch (byte-identical)
        permute = lambda idx: idx[torch.randperm(len(idx))]  # SAME draw+site as resident; ONLY .to(device) dropped
        prefetch = (_dfetch, _dcompute) if self._prefetch_depth() > 0 else None
        return self._run_epochs(net, opt, tr_idx, va_idx, stopper, _dloss, self._tb(), permute, show_progress,
                                prefetch=prefetch)

    def _train_dense_iter(self, net, source, task, n_out, forward=None, show_progress=False):
        """Training loop for the FORWARD-ONLY iterable regime (dense contracts). Self-contained -- it does NOT
        use _run_epochs/_auto_val_split, which need random access. Each epoch is one restartable pass; a seeded
        windowed shuffle buffer (RandomState(seed+_ITER_SHUFFLE_SEED+epoch)) replaces torch.randperm, and a
        hash-of-id split (via _iter_val_member) replaces the seeded index val split. Deterministic given the
        seed; NOT bit-identical to the map-style / resident fit."""
        from .allgraph_streaming import _ITER_SHUFFLE_SEED
        fwd = forward if forward is not None else (lambda xb: net(xb))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd(self._DENSE_WEIGHT_DECAY))
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        use_val = show_progress and getattr(self, "auto_epoch", None) == "val"
        stopper = self._make_stopper() if show_progress else None
        if show_progress:
            self._break_alpha_symmetry(net)
        bs, B = self._tb(), max(1, self.stream_shuffle_buffer)
        pin = self._resolve_pin()

        def _step(batch, state):
            xb = torch.stack([b[0] for b in batch])
            if pin:
                xb = xb.pin_memory()
            xb = xb.to(self.device, non_blocking=pin)
            yy = np.asarray([b[1] for b in batch])
            yt = torch.as_tensor(yy).long() if task == "classification" else torch.as_tensor(yy).float().unsqueeze(1)
            opt.zero_grad()
            loss = lf(fwd(xb), yt.to(self.device)) + self._alpha_penalty(net)
            loss.backward()
            if self._grads_finite(net):
                opt.step()
            if stopper is not None and not use_val:
                state["run"] = loss.detach() if state["run"] is None else state["run"] + loss.detach()
                state["nb"] += 1

        for epoch in self._epoch_iter(show_progress):
            rng = np.random.RandomState(self.seed + _ITER_SHUFFLE_SEED + (epoch if isinstance(epoch, int) else 0))
            buf, pending, state = [], [], {"run": None, "nb": 0}
            def _emit(sample):
                pending.append(sample)
                if len(pending) >= bs:
                    _step(pending, state); pending.clear()
            for sid, x, y in source:
                if use_val and self._iter_val_member(sid):
                    continue                             # val samples never train
                if len(buf) < B:
                    buf.append((x, y))
                else:
                    j = int(rng.randint(len(buf)))       # emit a random buffered sample, replace with the new one
                    _emit(buf[j]); buf[j] = (x, y)
            for j in rng.permutation(len(buf)):          # drain the buffer in seeded order
                _emit(buf[int(j)])
            if pending:
                _step(pending, state)                    # flush the last partial batch
            if stopper is not None:
                m = self._iter_val_loss(net, source, lf, fwd, task) if use_val \
                    else float(state["run"] / max(state["nb"], 1))
                if stopper.step(m):
                    break
        return net

    def _iter_val_loss(self, net, source, lf, fwd, task, bs=64):
        """Mean loss over the iterable regime's hash-held-out val samples, in eval mode (no dropout/BN update ->
        consumes NO torch RNG, so it cannot perturb the next epoch's weights or shuffle)."""
        net.eval()
        tot, nb, pending = 0.0, 0, []

        def _flush():
            nonlocal tot, nb
            xb = torch.stack([b[0] for b in pending]).to(self.device)
            yy = np.asarray([b[1] for b in pending])
            yt = torch.as_tensor(yy).long() if task == "classification" else torch.as_tensor(yy).float().unsqueeze(1)
            tot += float(lf(fwd(xb), yt.to(self.device))); nb += 1
            pending.clear()

        with torch.no_grad():
            for sid, x, y in source:
                if not self._iter_val_member(sid):
                    continue
                pending.append((x, y))
                if len(pending) >= bs:
                    _flush()
            if pending:
                _flush()
        net.train()
        return tot / max(nb, 1)

    def _iter_grid_eval(self, net, source, task, bs, forward=None):
        """In-sample score for the iterable regime: one sequential (unshuffled) pass over ALL samples, scored
        incrementally via _IterMetric (target streams by, so ss_tot is single-pass). Net left in TRAIN mode like
        the other grid evals. Returns the result dict via _report."""
        from .allgraph_streaming import _IterMetric
        fwd = forward if forward is not None else (lambda xb: net(xb))
        acc = _IterMetric(task)
        pending = []

        def _flush():
            xb = torch.stack([b[0] for b in pending]).to(self.device)
            with torch.no_grad():
                out = fwd(xb).cpu()
            acc.update(out, [b[1] for b in pending])
            pending.clear()

        for sid, x, y in source:
            pending.append((x, y))
            if len(pending) >= bs:
                _flush()
        if pending:
            _flush()
        metric, value = acc.result()
        return self._report(net, value, metric)

    def _report(self, net, value, metric, extra=None):
        self.net = net                                   # keep the trained net for inspection / test eval
        sel = net.cells[0].primitives[int(net.cells[0].alpha.argmax())] if hasattr(net, "cells") else None
        arch = [c.primitives[int(c.alpha.argmax())] for c in net.cells] if hasattr(net, "cells") else None
        r = {"contract": self.contract, "selected_primitive": sel, "architecture": arch,
             "n_params": sum(p.numel() for p in net.parameters()),
             "metric": metric, "value": float(value), "route": self.route_detail}
        if extra:
            r.update(extra)
        self._log(f"[AllGraph] {self.contract}: {metric}={value:.4f}  arch={arch}")
        return r

    def _deploy_grid_eval(self, net, X, bs, forward=None):
        """Batched in-sample forward for a deployed grid net: cat(net(X[j:j+bs]).cpu()). `forward` defaults to
        net(xb). NOTE: the net is left in TRAIN mode (batchnorm uses batch stats), so `bs` is part of the
        result -- callers pass the same batch size the contract used to keep scoring identical."""
        fwd = forward if forward is not None else (lambda xb: net(xb))
        outs = []
        for j in range(0, len(X), bs):
            with torch.no_grad():
                outs.append(fwd(X[j:j + bs].to(self.device)).cpu())
        return torch.cat(outs)

    # --------------------------------------------------------------------- opt-in dataset streaming
    def _is_streaming(self, X):
        """True iff X is a streaming DenseSource (opt-in via AllData.dense_stream). This ONE duck-typed
        predicate gates every dense streaming branch; when it is False the resident code path is textually
        unchanged, so resident behaviour and performance are provably identical (one isinstance per fit)."""
        from .allgraph_streaming import DenseSource
        return isinstance(X, DenseSource)

    def _is_streaming_graph(self, data):
        """True iff `data` carries a streaming GraphSource (opt-in via AllData.graph_stream, stored as
        node_feats). Gates every relational streaming branch; False leaves the resident relational path
        textually unchanged."""
        from .allgraph_streaming import GraphSource
        return isinstance(getattr(data, "node_feats", None), GraphSource)

    def _is_streaming_operator(self, data):
        """True iff `data` carries a streaming OperatorSource (opt-in via AllData.functions_stream, stored as
        dense). Gates the operator streaming branch; False leaves the resident operator path unchanged."""
        from .allgraph_streaming import OperatorSource
        return isinstance(getattr(data, "dense", None), OperatorSource)

    def _is_iterable(self, X):
        """True iff X is a forward-only IterableDenseSource (opt-in via AllData.dense_iter). Gates the separate
        iterable training regime; False leaves every map-style / resident branch unchanged."""
        from .allgraph_streaming import IterableDenseSource
        return isinstance(X, IterableDenseSource)

    def _iter_val_member(self, sample_id):
        """Whether a sample id falls in the iterable regime's held-out val bucket (a seed-keyed blake2b hash),
        used both to SKIP val samples during training and to KEEP them in the val-loss pass."""
        from .allgraph_streaming import _iter_val_key, _ITER_VAL_PERMILLE
        return _iter_val_key(sample_id, self.seed) % 1000 < _ITER_VAL_PERMILLE

    def _resolve_pin(self):
        """Whether to host-pin streamed minibatches and copy them non_blocking. Pinning only helps (and is only
        well-supported) on CUDA; None -> auto (pin iff CUDA), truthy/falsy -> forced but still CUDA-gated."""
        dev = str(getattr(self, "device", "cpu"))
        if not dev.startswith("cuda"):
            return False
        return True if self.stream_pin_memory is None else bool(self.stream_pin_memory)

    def _stream_grid_eval(self, net, source, bs, y, task, forward=None):
        """Streaming counterpart of `_deploy_grid_eval` + `_eval`: forward the deployed net over the source in
        sequential arange chunks and accumulate the score INCREMENTALLY (via _StreamMetric), so the full output
        is never materialized. The net is left in TRAIN mode (batchnorm batch-stats), so -- exactly as the
        resident eval -- `bs` is part of the result: callers pass the same batch size the contract used.
        Matches `_metric` exactly (ss_tot from the resident labels), so a single-chunk eval (bs >= n) is
        bit-identical to the resident score and a multi-chunk eval matches to fp tolerance. Returns the same
        result dict as `_eval` (via `_report`)."""
        from .allgraph_streaming import _StreamMetric
        fwd = forward if forward is not None else (lambda xb: net(xb))
        yy = np.asarray(y)
        acc = _StreamMetric(task, yy)
        n = len(source)
        for j in range(0, n, bs):
            ids = np.arange(j, min(j + bs, n))
            with torch.no_grad():
                out = fwd(source.get(ids).to(self.device)).cpu()
            acc.update(out, yy[ids])
        metric, value = acc.result()
        return self._report(net, value, metric)

    def _stream_operator_eval(self, net, source, bs):
        """Streamed field-R2 for the operator contract, matching the resident
        `1 - sum((pred-u)^2) / (sum((u-mean)^2) + 1e-12)` to fp tolerance without ever holding the full
        prediction or the full target field resident. TWO passes over the source: pass 1 accumulates the global
        field mean (over every grid point of every sample); pass 2 accumulates the residual sum of squares (from
        the net forward) and the total sum of squares (mean-subtracted, matching the resident formula). The net
        is left in TRAIN mode like the resident eval; operator primitives are per-sample (no batch-coupling), so
        the sequential arange chunking is score-neutral."""
        n = len(source)
        # pass 1: global mean of the target fields, accumulated in FLOAT64 (matches the resident
        # uy.astype(np.float64).mean(); a float32 sum drifts platform-dependently -- e.g. under MKL -- and on a
        # (near-)constant target that drift keeps ss_tot spuriously non-zero instead of collapsing to the guard).
        sum_u, count = 0.0, 0
        for j in range(0, n, bs):
            ub = source.u(np.arange(j, min(j + bs, n)))
            sum_u += float(ub.double().sum().item())
            count += int(ub.numel())
        mean = sum_u / max(count, 1)
        # pass 2: residual (float32, matching the resident float32 (pred-u)^2 sum) and the mean-subtracted total
        # sum of squares (float64, matching the resident float64 (u - float64 mean)^2 sum).
        ss_res, ss_tot = 0.0, 0.0
        for j in range(0, n, bs):
            ids = np.arange(j, min(j + bs, n))
            ab, xb, ub = source.a(ids).to(self.device), source.grid(ids).to(self.device), source.u(ids)
            with torch.no_grad():
                pred = net(ab, xb).cpu()
            ss_res += float(((pred - ub) ** 2).sum().item())
            ss_tot += float(((ub.double() - mean) ** 2).sum().item())
        return 1.0 - ss_res / (ss_tot + 1e-12)

    def _resident_subsample(self, data, cap=None):
        """Draw ONCE a bounded resident subsample (<= stream_subsample_cap samples) from a streaming source and
        materialize it as an ordinary in-RAM AllData. Search / selection sub-fits (which re-iterate the data many
        times) then run UNCHANGED on real tensors, so the out-of-core source is read ONCE for selection rather
        than hundreds of times. Uses an ISOLATED RandomState(seed+23) (via _reservoir_ids) that touches neither
        global RNG stream, so the deploy fit's RNG (and its weights) are unperturbed. Dispatches on source
        family: dense (DenseSource), relational (GraphSource), or operator (OperatorSource)."""
        from .allgraph_streaming import _reservoir_ids
        cap = self.stream_subsample_cap if cap is None else cap
        if self._is_streaming_graph(data):
            return self._resident_subsample_graph(data, cap)
        if self._is_streaming_operator(data):
            return self._resident_subsample_operator(data, cap)
        src = data.dense
        ids = _reservoir_ids(len(src), cap, self.seed + 23)
        # materialize the samples in the SAME shape the deploy fit builds on: spatial/volumetric add a channel
        # axis via _as_grid (so a (S,H,W) source becomes (S,1,H,W)); sequence/4d present full-rank samples
        # already. This keeps the selection sub-fits (which read sub.dense directly) shape-consistent with deploy.
        _rank = {"spatial": 4, "volumetric": 5}.get(self.contract)
        grid_src = self._as_grid(src, _rank) if _rank is not None else src
        X = grid_src.get(ids)                                # (S, *deploy_sample_shape) resident float32 CPU
        y = np.asarray(data.y)[ids] if data.y is not None else None
        return AllData.dense_tensor(X, y=y, kind_hint=data.kind_hint)

    def _resident_subsample_graph(self, data, cap):
        """Bounded resident subsample of a streaming GraphSource -> an in-RAM AllData.graphs / .point_sets with
        the reservoir ids' node/edge/pos materialized and y sliced. kind_hint is left None so the resident
        tiebreak-candidate / forward logic reads the materialized positions/edges directly."""
        from .allgraph_streaming import _reservoir_ids
        src = data.node_feats
        ids = _reservoir_ids(len(src), cap, self.seed + 23)
        nf = [src.node(int(i)) for i in ids]
        y = np.asarray(data.y)[ids] if data.y is not None else None
        pos = [src.pos(int(i)) for i in ids] if src.has_pos else None
        if src.has_edges:
            return AllData.graphs(nf, [src.edge(int(i)) for i in ids], y=y, positions=pos)
        return AllData.point_sets(nf, y=y, positions=pos)

    def _resident_subsample_operator(self, data, cap):
        """Bounded resident subsample of a streaming OperatorSource -> an in-RAM AllData.functions. The operator
        target is the field src.u(ids) (streaming operator data.y is None)."""
        from .allgraph_streaming import _reservoir_ids
        src = data.dense
        ids = _reservoir_ids(len(src), cap, self.seed + 23)
        return AllData.functions(src.a(ids), src.u(ids), grid=src.grid(ids), spatial_dims=src.spatial_dims)

    def _streaming_subsample(self, data):
        """Memoized per-fit accessor for the resident selection subsample: returns None for a non-streaming
        input (so callers fall back to `data`), else draws the subsample at most ONCE per fit (self._subsample_
        cache, reset at the top of fit()). Every SELECTION site does `sel = self._streaming_subsample(data) or
        data` so it runs on the bounded subsample under streaming and on the full data otherwise."""
        streaming = (self._is_streaming(getattr(data, "dense", None)) or self._is_streaming_graph(data)
                     or self._is_streaming_operator(data))
        if not streaming:
            return None
        if getattr(self, "_subsample_cache", None) is None:
            self._subsample_cache = self._resident_subsample(data)
        return self._subsample_cache

    def _check_streaming_supported(self, data, task, select, select_size, tiebreak, stream, n_out=None):
        """Streaming guard, run right after contract resolution. Streaming is active iff data.dense is a
        DenseSource (dense), data.node_feats is a GraphSource (relational), data.dense is an OperatorSource
        (operator), or data.dense is an IterableDenseSource (the forward-only regime); `stream` is an optional
        caller ASSERTION (True demands streaming, False forbids it). Enforces that the resolved contract matches
        the source family and raises a clear, actionable error for options that would re-read the full source
        (or, for the iterable regime, that need random access). Supported under streaming: select in
        {'argmax','sparse','gibbs'}, select_size, tiebreak, angular_from_data (via a bounded subsample), and
        auto_epoch; the forward-only iterable regime supports only select in {'argmax','sparse'} + auto_epoch."""
        dense_stream = self._is_streaming(getattr(data, "dense", None))
        graph_stream = self._is_streaming_graph(data)
        op_stream = self._is_streaming_operator(data)
        iter_stream = self._is_iterable(getattr(data, "dense", None))
        streaming = dense_stream or graph_stream or op_stream or iter_stream
        if stream is True and not streaming:
            raise ValueError("fit(stream=True) but the input is not streaming; build it with "
                             "AllData.dense_stream(...) / .graph_stream(...) / .functions_stream(...) / "
                             ".dense_iter(...) to stream.")
        if stream is False and streaming:
            raise ValueError("fit(stream=False) but the input is a streaming source; build a resident input "
                             "(AllData.dense_tensor / .graphs / .point_sets / .functions) to train in memory.")
        if not streaming:
            return
        if iter_stream:
            # forward-only regime: dense contracts only; no random-access selection (subsample/readout), no
            # full-dataset diagnostics; classification needs an explicit n_out (targets stream by).
            if self.contract not in _DENSE_CONTRACTS:
                raise NotImplementedError(
                    f"an IterableDenseSource resolved to contract {self.contract!r}; the forward-only iterable "
                    f"regime covers the dense contracts {sorted(_DENSE_CONTRACTS)}.")
            blk = []
            if select_size:
                blk.append("select_size")
            if select == "gibbs":
                blk.append("select='gibbs'")
            if tiebreak:
                blk.append("tiebreak=True")
            if select not in ("argmax", "sparse"):
                blk.append(f"select={select!r}")
            for flag in ("readout_select", "kernel_from_xi", "angular_from_data", "price_singular", "price_modes",
                         "price_equivariance", "report_llc", "developmental_llc", "report_thermo",
                         "report_response", "report_ledger", "symmetry_routing", "canonicalize_reuse"):
                if getattr(self, flag, False):
                    blk.append(flag)
            if blk:
                raise NotImplementedError(
                    "the forward-only iterable regime (AllData.dense_iter) cannot random-access the data, so it "
                    f"does not support: {', '.join(blk)}. Use AllData.dense_stream (map-style) for those, or "
                    "disable them. Supported: select in {'argmax','sparse'}, auto_epoch.")
            if task == "classification" and n_out is None and getattr(data.dense, "n_out", None) is None:
                raise ValueError("iterable classification needs an explicit n_out (the targets stream by and "
                                 "cannot be scanned); pass n_out to fit() or AllData.dense_iter(..., n_out=...).")
            return
        if dense_stream and self.contract not in _DENSE_CONTRACTS:
            raise NotImplementedError(
                f"a dense DenseSource resolved to contract {self.contract!r}; dense streaming covers "
                f"{sorted(_DENSE_CONTRACTS)}.")
        if graph_stream and self.contract not in _IRREGULAR_CONTRACTS:
            raise NotImplementedError(
                f"a relational GraphSource resolved to contract {self.contract!r}; relational streaming covers "
                f"{sorted(_IRREGULAR_CONTRACTS)}.")
        if op_stream and self.contract != "operator":
            raise NotImplementedError(
                f"an OperatorSource resolved to contract {self.contract!r}; operator streaming covers 'operator'.")
        # select_size / select='gibbs' / tiebreak / angular_from_data are SUPPORTED under streaming: they run on
        # a bounded resident subsample (drawn once, isolated RNG) while the winner deploy-trains on the full
        # stream. The options below still re-read the FULL dataset and are not yet wired, so they stay blocked.
        # symmetry_routing / canonicalize_reuse read data.positions (None under a GraphSource) and would crash,
        # so they are blocked too until routed through the subsample.
        blocked = []
        for flag in ("kernel_from_xi", "price_singular", "price_modes", "price_equivariance",
                     "report_llc", "developmental_llc", "report_thermo", "report_response", "report_ledger",
                     "symmetry_routing", "canonicalize_reuse"):
            if getattr(self, flag, False):
                blocked.append(flag)
        if blocked:
            fam = "dense_tensor" if dense_stream else ("functions" if op_stream else "graphs / .point_sets")
            extra = ", readout_select" if dense_stream else ""
            raise NotImplementedError(
                "dataset streaming does not support these options because they re-read the full dataset many "
                f"times: {', '.join(blocked)}. Train on a resident AllData.{fam}, or disable them. Supported "
                f"under streaming: select in {{'argmax','sparse','gibbs'}}, select_size, tiebreak, "
                f"angular_from_data, auto_epoch{extra}.")


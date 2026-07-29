"""Per-contract network builders for AllGraph (mixin).

The _fit_<contract> builders (sequence / spatial / volumetric / 4d / graph / equivariant / set / operator,
plus the discovered-group generated_equivariant / Sp-SL and the nonlinear latent_equivariant variants) and
their low-level tensor plumbing (collate / forward / shape / metric helpers), split out of allgraph.py to
keep the controller focused -- the same discipline that already put persistence and the diagnostic reports
in their own mixins. AllGraph mixes this in, so `self` resolves every builder-shared helper (_report,
_run_epochs, _train_dense, _subfit_device, ...) and every class constant exactly as before; the builders
lazily import their model schemas from ..models inside each method, so there is no import cycle."""

import numpy as np
import torch
import torch.nn as nn


class _ContractFitMixin:
    # ------------------------------------------------------------------- per-contract builders + plumbing
    def _fit_grid(
        self, data, task, n_out, primitives, *, default_prims, build_fn, grid_rank=None, xi_ndim=None, eval_bs=128
    ):
        """Shared spatial/volumetric/4d deploy path: prepare the grid, default the primitive menu (with the
        optional kernel-from-xi augmentation), build via `build_fn(X, n_out, prims)`, train, and score. The
        three grid contracts differ only in builder, menu, grid rank, xi dimensionality, and eval batch size."""
        if self._is_iterable(data.dense):
            # forward-only regime: the source presents full-rank samples (no _as_grid fix-up); train via the
            # windowed-shuffle loop and score with a single-pass streamed metric.
            X = data.dense
            if grid_rank is not None and X.dim() != grid_rank:
                raise NotImplementedError(
                    f"iterable {self.contract} source must present rank-{grid_rank} samples (channel + grid); got "
                    f"rank {X.dim()}. The iterable regime applies no _as_grid channel fix-up."
                )
            n_out = self._infer_nout(None, task, n_out if n_out is not None else getattr(X, "n_out", None))
            net = build_fn(X, n_out, primitives or default_prims).to(self.device)
            net = self._train_dense_iter(net, X, task, n_out, show_progress=True)
            return self._iter_grid_eval(net, X, task, eval_bs)
        X = self._as_grid(data.dense, rank=grid_rank) if grid_rank is not None else data.dense
        n_out = self._infer_nout(data.y, task, n_out)
        prims = primitives or default_prims
        if primitives is None and xi_ndim is not None and getattr(self, "kernel_from_xi", False):
            prims, kdetail = self._augment_kernels_by_xi(X, prims, ndim=xi_ndim)
            if kdetail is not None:
                self.route_detail = {**(self.route_detail or {}), "kernel_from_xi": kdetail}
        net = build_fn(X, n_out, prims).to(self.device)
        net = self._train_dense(net, X, data.y, task, show_progress=True)
        if self._is_streaming(X):  # accumulate the in-sample score without materializing n
            return self._stream_grid_eval(net, X, eval_bs, data.y, task)
        out = self._deploy_grid_eval(net, X, eval_bs)
        return self._eval(net, out, data.y, task)

    def _fit_sequence(self, data, task, n_out, primitives):
        from ..models import build_schema

        X = data.dense
        if self._is_iterable(X):
            # forward-only regime: rank-3 samples required (readout_select is blocked, so readout='mean').
            if X.dim() != 3:
                raise NotImplementedError(
                    "iterable sequence source must present rank-3 samples (T, features) per sample; got rank "
                    f"{X.dim()}."
                )
            n_out = self._infer_nout(None, task, n_out if n_out is not None else getattr(X, "n_out", None))
            n_in = X.shape[2]
            prims = primitives or (
                "plain",
                "gated",
                "lstm",
                "conv",
                "dilconv",
                "attention",
                "dense",
                "spectral",
                "norm",
            )
            net = build_schema(
                n_in=n_in, width=self.width, depth=self.depth, n_out=n_out, primitives=prims, readout="mean"
            ).to(self.device)
            fwd = lambda xb: net.forward_seq_readout(xb, 1).squeeze(1)
            net = self._train_dense_iter(net, X, task, n_out, forward=fwd, show_progress=True)
            res = self._iter_grid_eval(net, X, task, 256, forward=fwd)
            res["readout"] = "mean"
            return res
        streaming = self._is_streaming(X)
        if streaming:
            # Streaming sequence sources must present rank-3 samples (n, T, features); the flat (n, T) ->
            # (n, T, 1) unsqueeze is a whole-tensor op with no per-batch analogue here, so we require the
            # feature axis up front rather than silently reshaping.
            if X.dim() != 3:
                raise NotImplementedError(
                    "streaming sequence contract requires rank-3 samples (n, T, features); reshape the source "
                    "so each sample carries its feature axis (e.g. (T, 1) per sample)."
                )
        elif X.dim() == 2:  # flat -> length-T sequence, 1 channel
            X = X.unsqueeze(-1)
        n_out = self._infer_nout(data.y, task, n_out)
        n_in = X.shape[2]
        # default menu = 9 of the 11-core sequence vocabulary (_SEQ_CORES in schema.py); the two
        # SSM cores (linssm, selssm) are opt-in via an explicit `primitives=` list, not the default.
        prims = primitives or ("plain", "gated", "lstm", "conv", "dilconv", "attention", "dense", "spectral", "norm")
        # readout choice is a hyperparameter: under streaming, bake it off on a bounded resident subsample (drawn
        # once), then deploy-train the winner on all N via streaming. Off (returns 'mean') unless readout_select.
        readout = (
            self._select_seq_readout_streaming(data, task, n_out, n_in, prims)
            if streaming
            else self._select_seq_readout(X, data.y, task, n_out, n_in, prims)
        )
        fwd_of = (
            (lambda net: lambda xb: net.forward(xb))
            if readout == "flatten"
            else (lambda net: lambda xb: net.forward_seq_readout(xb, 1).squeeze(1))
        )
        net = build_schema(
            n_in=n_in, width=self.width, depth=self.depth, n_out=n_out, primitives=prims, readout=readout
        ).to(self.device)
        net = self._train_dense(net, X, data.y, task, forward=fwd_of(net), show_progress=True)
        if streaming:
            # Chunked, accumulated eval replacing the resident single-batch forward. Chunking at bs=256 is
            # OUTPUT-SAFE here (unlike the grid contracts, whose BatchNorm uses batch statistics in train mode):
            # the sequence vocabulary is entirely per-sample (recurrent / conv / attention / LayerNorm / spectral),
            # so a per-sequence output does not depend on the batch it is scored in. If a batch-coupling primitive
            # (BatchNorm / batch-Dropout) is ever added to the sequence menu, this must switch to the resident
            # eval batch size to preserve parity.
            res = self._stream_grid_eval(net, X, 256, data.y, task, forward=fwd_of(net))
        else:
            with torch.no_grad():
                out = fwd_of(net)(X.to(self.device)).cpu()
            res = self._eval(net, out, data.y, task)
        res["readout"] = readout
        return res

    def _select_seq_readout_streaming(self, data, task, n_out, n_in, prims):
        """Streaming readout bake-off: run the resident `_select_seq_readout` on a bounded in-RAM subsample of
        the source (drawn once). The bake-off only CHOOSES a hyperparameter; the winner then deploy-trains on
        all N via streaming. Returns 'mean' immediately (no source read) unless readout_select is enabled."""
        if not self.readout_select:
            return "mean"
        sub = self._streaming_subsample(data)  # memoized (shared with any co-occurring size selection)
        return self._select_seq_readout(sub.dense, sub.y, task, n_out, n_in, prims)

    def _select_seq_readout(self, X, y, task, n_out, n_in, prims):
        """Choose the sequence readout by a priced held-out bake-off between 'mean' (pool over time) and
        'flatten' (concatenate all timesteps -> position-aware, the flatten-then-dense operation optimal on
        short 'effectively tabular' series like ItalyPowerDemand, which the mean-over-time readout cannot
        express). flatten multiplies the head input by T, so it is charged its extra-parameter cost in the
        same J = R + mu*Omega spirit as every other selection: flatten wins only if its held-out fit gain
        exceeds readout_mu times the (log) parameter-count increase. Guarded to short sequences (T <=
        seq_flatten_max_T) where flatten is tractable and tabular structure is plausible; longer series
        keep 'mean' without a bake-off. Returns 'mean' or 'flatten'."""
        T = X.shape[1]
        if not self.readout_select or T > self.seq_flatten_max_T:
            return "mean"
        import numpy as _np

        n = len(X)
        rng = _np.random.RandomState(self.seed)
        perm = rng.permutation(n)
        nval = max(1, int(0.25 * n))
        va, tr = perm[:nval], perm[nval:]
        Xtr, Xva = X[tr], X[va]
        ytr = _np.asarray(y)[tr]
        yva = _np.asarray(y)[va]

        def score(readout):
            torch.manual_seed(self.seed)
            net = build_schema(
                n_in=n_in, width=self.width, depth=self.depth, n_out=n_out, primitives=prims, readout=readout
            ).to(self.device)
            fwd = (
                (lambda xb: net.forward(xb))
                if readout == "flatten"
                else (lambda xb: net.forward_seq_readout(xb, 1).squeeze(1))
            )
            # PERF: budget this ranking sub-fit like every other search phase (min(epochs, cap)) rather than
            # the full deployment budget -- the readout bake-off was the only uncapped search sub-fit. For
            # the usual budgets (<= _SEARCH_EPOCH_CAP) this is unchanged; only large budgets are trimmed.
            ep0 = self.epochs
            self.epochs = self._search_ep(self.epochs)
            try:
                self._train_dense(net, Xtr, ytr, task, forward=fwd)
            finally:
                self.epochs = ep0
            with torch.no_grad():
                out = fwd(Xva.to(self.device)).cpu()
            _, val = self._metric(out, yva, task)
            npar = sum(p.numel() for p in net.parameters())
            return float(val), npar

        from ..models import build_schema

        r_mean, p_mean = score("mean")
        r_flat, p_flat = score("flatten")
        # priced acceptance: flatten must beat mean by more than readout_mu * (relative log param increase)
        price = self.readout_mu * float(np.log(max(p_flat, 1) / max(p_mean, 1)))
        chosen = "flatten" if (r_flat - r_mean) > price else "mean"
        self._log(
            f"[AllGraph] readout bake-off: mean={r_mean:.3f}({p_mean}p) "
            f"flatten={r_flat:.3f}({p_flat}p) price={price:.3f} -> {chosen}"
        )
        return chosen

    def _augment_kernels_by_xi(self, X, base_prims, ndim=2):
        """Correlation-length -> kernel-size selection (integrating core/correlation_length.py). The spatial/
        volumetric schemas already carry larger-kernel primitives (conv2d_k5/k7, conv3d_k5) that COMPETE
        via primitive selection, but the default menu omits them. Here we MEASURE the data's spatial
        correlation length xi and, when it is large enough that a k=3 receptive field would miss the local
        structure, ADD the larger-kernel primitives to the menu so the selector can choose them; when xi is
        small we leave the lean default. This makes receptive field an OUTPUT of a measured quantity (the
        marginal-value / mode-structure principle applied to kernel size) rather than a fixed k=3, while
        respecting that the schema selects among the primitives it is given. Only augments when the user
        did not pass an explicit primitive list."""
        try:
            from .correlation_length import recommend_kernel_size

            rec = recommend_kernel_size(np.asarray(X), ndim=ndim)
            k = int(rec.get("kernel_size", 3))
            xi = float(rec.get("xi", 0.0))
        except Exception:
            return base_prims, None
        prims = list(base_prims)
        added = []
        # 2D spatial has conv2d_k5/k7; 3D volumetric now has conv3d_k5. Augment the menu with the larger-
        # kernel variant that exists for this dimensionality when the measured xi warrants it.
        if ndim == 2:
            if k >= 5 and "conv2d_k5" not in prims:
                prims.append("conv2d_k5")
                added.append("conv2d_k5")
            if k >= 7 and "conv2d_k7" not in prims:
                prims.append("conv2d_k7")
                added.append("conv2d_k7")
        elif ndim == 3:
            if k >= 5 and "conv3d_k5" not in prims:
                prims.append("conv3d_k5")
                added.append("conv3d_k5")
        else:
            return base_prims, {
                "xi": round(xi, 3),
                "recommended_kernel": k,
                "added_primitives": [],
                "note": f"no larger-kernel {ndim}D primitive available",
            }
        detail = {"xi": round(xi, 3), "recommended_kernel": k, "added_primitives": added}
        if added:
            self._log(f"[AllGraph] kernel_from_xi -> xi={xi:.2f}, added {added} to the {ndim}D menu")
        return tuple(prims), detail

    def _fit_spatial(self, data, task, n_out, primitives):
        from ..models import build_spatial_schema

        return self._fit_grid(
            data,
            task,
            n_out,
            primitives,
            default_prims=("conv2d", "atrous", "conv_dw", "pointwise", "norm"),
            build_fn=lambda X, no, pr: build_spatial_schema(
                n_in=X.shape[1], width=self.width, hw=X.shape[-1], depth=self.depth, n_out=no, primitives=pr
            ),
            grid_rank=4,
            xi_ndim=2,
            eval_bs=128,
        )

    def _fit_volumetric(self, data, task, n_out, primitives):
        from ..models import build_volumetric_schema

        return self._fit_grid(
            data,
            task,
            n_out,
            primitives,
            default_prims=("conv3d", "conv_dw", "pointwise", "norm"),
            build_fn=lambda X, no, pr: build_volumetric_schema(
                n_in=X.shape[1],
                width=self.width,
                dhw=self._vol_work_dhw(X.shape[-1]),
                vol_size=X.shape[-1],
                depth=self.depth,
                n_out=no,
                primitives=pr,
            ),
            grid_rank=5,
            xi_ndim=3,
            eval_bs=64,
        )

    def _fit_4d(self, data, task, n_out, primitives):
        from ..models import build_grid4d_schema

        return self._fit_grid(
            data,
            task,
            n_out,
            primitives,
            default_prims=("conv4d", "conv4d_kt1", "conv_dw", "pointwise", "norm"),
            build_fn=lambda X, no, pr: build_grid4d_schema(
                n_in=X.shape[1],
                grid_shape=tuple(X.shape[2:]),
                width=self.width,
                depth=self.depth,
                n_out=no,
                primitives=pr,
            ),
            grid_rank=None,
            xi_ndim=None,
            eval_bs=16,
        )

    def _fit_graph(self, data, task, n_out, primitives):
        from ..models import build_graph_schema

        return self._fit_relational(data, task, n_out, primitives, build_graph_schema, with_pos=False, kind="graph")

    def _fit_equivariant(self, data, task, n_out, primitives):
        from ..models import build_equivariant_graph_schema

        return self._fit_relational(
            data, task, n_out, primitives, build_equivariant_graph_schema, with_pos=True, kind="equivariant"
        )

    def _fit_skew_or_volume_contract(self, data, task, n_out, gname, vec_dim, yt, lf, idx):
        """Sp(2n) / SL(n) contract. Unlike metric groups, the vector->vector commutant is 1-dimensional
        (Schur), so an equivariant-linear hidden map collapses to proportional vectors and the skew/det
        invariants vanish. Instead we form the group invariants from the INPUT points directly with LEARNED
        attention over the set: Sp uses the skew pairing omega(sum_A p, sum_B p) between two learned
        soft-subsets A, B (a genuine, generally nonzero Sp-invariant); SL uses determinants of learned
        soft-frames of vec_dim weighted point-combinations (an SL-invariant volume). A learned scalar MLP
        of node features gates the attention; the resulting invariants feed a readout."""
        from .emlp_layer import determinant_invariants, symplectic_generators, symplectic_invariants

        is_sp = gname.startswith("Sp(")
        H = max(2 * vec_dim, self._SKEW_MIN_GROUPS)  # number of learned soft-groups / frame-vectors
        # The soft-grouping must be based on FIXED node features (a canonical labeling: which points are q
        # vs p, or a frame index), NOT on the coordinates -- coordinates are transformed by the group, so
        # grouping by them would break invariance. With a fixed labeling the group-vectors are fixed linear
        # combinations of the (transformed) coordinates, and the skew/det invariants of those combinations
        # ARE exactly group-invariant. This is the honest scope: Sp/SL contracts apply to data carrying a
        # canonical pairing/frame labeling in the node features (e.g. phase-space q/p), which the invariant
        # (antisymmetric / alternating) structure requires -- an unlabeled point set admits no nonzero
        # permutation-symmetric Sp/SL invariant.
        base_nf = data.node_feats[0].shape[1] if getattr(data, "node_feats", None) is not None else 1
        attn = nn.Sequential(
            nn.Linear(base_nf, self._SKEW_ATTN_HIDDEN), nn.Tanh(), nn.Linear(self._SKEW_ATTN_HIDDEN, H)
        ).to(self.device)
        if is_sp:
            _, Om = symplectic_generators(vec_dim // 2)
            n_inv = H * (H - 1) // 2
        else:
            n_inv = max(H - vec_dim + 1, 1)
        readout = nn.Sequential(
            nn.Linear(n_inv, self._SKEW_READOUT_HIDDEN), nn.Tanh(), nn.Linear(self._SKEW_READOUT_HIDDEN, n_out)
        ).to(self.device)
        # Bundle the two trained modules up front (this becomes self.net below) so the deployed loop can carry a
        # stopper and a best-weights snapshot like every other contract -- both need one state_dict to act on.
        bundle = nn.ModuleDict({"attn": attn, "readout": readout})
        params = list(bundle.parameters())
        opt = torch.optim.Adam(params, lr=max(self.lr, self._SKEW_LR_FLOOR), weight_decay=self._wd())

        # takes the datum's ARRAYS (not an index into the training `data`) so the same closure scores NEW
        # data -- test eval, predict(). Published as self._geq_forward below.
        def one_out(pos_arr, feat_arr=None):
            P = torch.as_tensor(np.asarray(pos_arr), dtype=torch.float32, device=self.device)
            F = torch.as_tensor(np.asarray(feat_arr), dtype=torch.float32, device=self.device)
            gate = torch.sigmoid(attn(F))  # (n_nodes, H) per-node per-group gate in [0,1]
            groups = torch.einsum("nh,nd->hd", gate, P)  # (H, vec_dim) weighted-SUM group vectors
            gv = groups.reshape(-1)  # (H*vec_dim)
            if is_sp:
                inv = symplectic_invariants(gv, H, vec_dim, Om)  # skew pairings between group-vectors
            else:
                inv = determinant_invariants(gv, H, vec_dim)  # determinants of frames
            return readout(inv)

        def datum_out(i):
            return one_out(data.positions[i], data.node_feats[i])

        self._geq_forward = one_out

        y_target = yt.to(self.device)
        idx = np.asarray(idx)
        # MINIBATCH (matching _fit_generated_equivariant): stacking datum_out over the WHOLE dataset per epoch
        # builds one autograd graph spanning all n samples, so peak memory scaled with dataset size (OOM on
        # large point clouds); 32-sample batches bound it.
        # This is a DEPLOYED fit, so it honors auto_epoch (plateau stop) and the epoch-extension policy. There
        # is no held-out monitor on this path -- the invariants are computed per-datum from a python list, not a
        # random-access tensor -- so it monitors the epoch-mean TRAIN loss and records that as the monitor used.
        stopper = self._make_stopper()
        if stopper is not None:
            self._last_auto_monitor = "train"
        best_state, done = None, False
        for count, offset in self._deploy_epoch_blocks(stopper):
            for _ in self._epoch_iter(count=count, offset=offset):
                perm = idx[np.random.permutation(len(idx))]
                run, nb = 0.0, 0
                for j in range(0, len(perm), self._tb()):
                    ids = perm[j : j + 32]
                    opt.zero_grad()
                    outs = torch.stack([datum_out(i) for i in ids])
                    tgt = y_target[ids]
                    if task != "classification":
                        outs = outs.squeeze(-1)  # (b,1)->(b,) so MSELoss doesn't broadcast
                        loss = lf(outs, tgt)
                    else:
                        loss = lf(outs, tgt.long())
                    loss.backward()
                    opt.step()
                    run += float(loss.detach())
                    nb += 1
                if stopper is not None:
                    fired = stopper.step(run / max(nb, 1))
                    best_state = self._snapshot_if_best(bundle, stopper, best_state)
                    if fired:
                        done = True
                        break
            if done:
                break
        self._finish_deploy_epochs(bundle, stopper, best_state, self._epoch_cap())
        with torch.no_grad():
            outs = torch.cat(
                [torch.stack([datum_out(i) for i in idx[j : j + 64]]).cpu() for j in range(0, len(idx), 64)]
            )
            if task != "classification":
                outs = outs.squeeze(-1)
            val = self._score(outs, yt.cpu(), task)
        # nn.ModuleDict (not a plain dict): keeps the ["attn"]/["readout"] access while exposing
        # .parameters()/.eval() like every other contract's deployed net -- a plain dict has neither, so the
        # runners' param count and the predict() path raised AttributeError on this contract.
        self.net = bundle
        self._log(
            f"[AllGraph] generated {gname} contract (learned-attention {'skew' if is_sp else 'volume'} "
            f"invariants): score={val:.3f}"
        )
        return self._with_epoch_telemetry({"value": val, "contract": "generated_equivariant", "group": gname})

    def _fit_generated_equivariant(self, data, task, n_out, primitives):
        """Fit a GENERATED G-equivariant contract (Phase 2): build an EMLP net equivariant to the group in
        self.generated_equivariant_group, using each datum's positions as its set of group vectors. A
        per-datum invariant readout is pooled over the set. This realises a contract for a discovered group
        that the eight built-ins may not cover (e.g. Lorentz O(1,3) on particle 4-vectors)."""
        from .emlp_layer import equivariant_bilinear_invariants

        spec = self.generated_equivariant_group
        gens = [np.asarray(A, float) for A in spec["gens"]]
        vec_dim = spec["vec_dim"]
        metric = spec.get("metric")
        n_out = self._infer_nout(data.y, task, n_out)
        # An equivariant per-node map to a hidden VECTOR rep, then POOL the vectors across the set (a sum of
        # equivariant vectors is still equivariant), then form invariants from the POOLED vectors and read
        # out. Pooling BEFORE the invariant step is what captures cross-node invariants (e.g. a jet's
        # invariant mass <sum p, sum p>), which a per-node-then-pool design misses.
        from .emlp_layer import (
            EquivariantLinear,
            direct_sum,
        )

        hv = self.width // 8 + 2
        # invariant family: metric bilinear (O(p,q)/U(n)) uses an equivariant-linear hidden map; symplectic
        # (Sp) and special-linear (SL) do NOT -- their vector->vector commutant is 1-dimensional (Schur),
        # so an equivariant-linear map makes all hidden vectors proportional and the skew/det invariants
        # vanish. For Sp/SL the invariants are formed from the INPUT points directly (skew pairings /
        # determinant frames) with a learned attention over the set.
        gname = str(spec.get("name", ""))
        n_out = self._infer_nout(data.y, task, n_out)
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        y = np.asarray(data.y)
        yt = torch.as_tensor(y)
        n = len(data.positions)
        idx = np.arange(n)
        if gname.startswith(("Sp(", "SL(")):
            return self._fit_skew_or_volume_contract(data, task, n_out, gname, vec_dim, yt, lf, idx)
        in_rep = direct_sum(gens, 1)
        vec_out_rep = direct_sum(gens, hv)  # map each node to hv hidden vectors
        node_map = EquivariantLinear(in_rep, vec_out_rep).torch_module().to(self.device)
        n_inv = hv * (hv + 1) // 2
        readout = nn.Sequential(
            nn.Linear(2 * n_inv, self._SKEW_READOUT_HIDDEN), nn.Tanh(), nn.Linear(self._SKEW_READOUT_HIDDEN, n_out)
        ).to(self.device)
        # Bundle both trained modules up front (this becomes self.net below) so the deployed loop can carry a
        # stopper and a best-weights snapshot like every other contract -- both need one state_dict to act on.
        bundle = nn.ModuleDict({"node_map": node_map, "readout": readout})
        params = list(bundle.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self._wd())
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        y = np.asarray(data.y)
        yt = torch.as_tensor(y)
        n = len(data.positions)
        idx = np.arange(n)

        # center only for Euclidean/orthogonal groups (metric = identity), where the physical group is
        # E(d) = translations x SO(d) and translation-invariance helps. For Lorentz/momentum data the
        # target (e.g. jet invariant mass) depends on the UNCENTERED pooled 4-vector, so do NOT center.
        euclidean = metric is None or bool(np.allclose(metric, np.eye(vec_dim)))
        # for a SIMILARITY group (rotations + isotropic dilation), scale-normalize each cloud by its RMS
        # radius so the contract is dilation-invariant (shape, not size).
        scale_norm = bool(spec.get("scale_norm", False))

        # The per-datum forward takes the datum's ARRAYS (not an index into the training `data`), so the same
        # closure can score NEW data -- test eval, predict(). It is published as self._geq_forward below.
        def one_out(pos_arr, feat_arr=None):
            P = torch.as_tensor(np.asarray(pos_arr), dtype=torch.float32, device=self.device)
            if euclidean:
                P = P - P.mean(0, keepdim=True)
            if scale_norm:
                rms = torch.sqrt((P**2).sum(1).mean()) + 1e-9
                P = P / rms
            V = node_map(P)  # (n_nodes, hv*vec_dim) equivariant vectors
            Vpool = V.sum(0)  # pool vectors across the set -> still equivariant
            inv = equivariant_bilinear_invariants(Vpool, hv, vec_dim, metric)  # cross-node invariants
            inv_nodes = equivariant_bilinear_invariants(V, hv, vec_dim, metric).mean(0)  # per-node spread
            return readout(torch.cat([inv, inv_nodes], dim=-1))

        def datum_out(i):
            return one_out(data.positions[i])

        self._geq_forward = one_out
        self._geq_modules = [node_map, readout]

        # DEPLOYED fit: honors auto_epoch (plateau stop) and the epoch-extension policy. No held-out monitor on
        # this path (per-datum forwards over a python list, not random-access tensors), so it monitors the
        # epoch-mean TRAIN loss and records that as the monitor actually used.
        stopper = self._make_stopper()
        if stopper is not None:
            self._last_auto_monitor = "train"
        best_state, done = None, False
        for count, offset in self._deploy_epoch_blocks(stopper):
            for _ in self._epoch_iter(count=count, offset=offset):
                np.random.shuffle(idx)
                run, nb = 0.0, 0
                for j in range(0, len(idx), self._tb()):
                    ids = idx[j : j + 32]
                    opt.zero_grad()
                    out = torch.stack([datum_out(i) for i in ids], 0)
                    target = (
                        yt[ids].long().to(self.device)
                        if task == "classification"
                        else yt[ids].float().unsqueeze(1).to(self.device)
                    )
                    loss = lf(out, target)
                    loss.backward()
                    opt.step()
                    run += float(loss.detach())
                    nb += 1
                if stopper is not None:
                    fired = stopper.step(run / max(nb, 1))
                    best_state = self._snapshot_if_best(bundle, stopper, best_state)
                    if fired:
                        done = True
                        break
            if done:
                break
        self._finish_deploy_epochs(bundle, stopper, best_state, self._epoch_cap())
        outs = []
        with torch.no_grad():
            for i in range(n):
                outs.append(datum_out(i).unsqueeze(0).cpu())
        pred = torch.cat(outs)
        # deploy BOTH trained modules as self.net (an nn.Module container, so .parameters() /
        # .eval() work as for every other contract and the param count includes the readout head).
        res = self._eval(bundle, pred, y, task)
        res["contract"] = "generated_equivariant"
        res["group_generators"] = len(gens)
        return self._with_epoch_telemetry(res)

    def forward_generated_equivariant(self, data):
        """Forward pass of a fitted 'generated_equivariant' contract on NEW data, returning the raw output
        tensor (n_samples, n_out) on CPU.

        The discovered-group contracts (metric/EMLP and the Sp/SL attention variant) are per-datum: each
        sample's own point set is mapped to group invariants, so they have no batched net(x, ei, pos, batch,
        n) signature like the built-in relational contracts. Both fits publish their per-datum forward as
        self._geq_forward, which this replays over `data`. Test eval and predict() go through here."""
        fn = getattr(self, "_geq_forward", None)
        if fn is None:
            raise RuntimeError(
                "no generated-equivariant forward available -- this AllGraph was not fitted "
                "with a discovered-group contract (or was restored via load(), which does not "
                "yet persist the discovered-group modules)."
            )
        pos = getattr(data, "positions", None)
        if pos is None:
            raise ValueError(
                "the generated_equivariant contract needs per-sample positions; this AllData "
                "carries none (build it with AllData.point_sets(..., positions=...))."
            )
        feats = getattr(data, "node_feats", None)
        net = self.net
        was_training = getattr(net, "training", False)
        if hasattr(net, "eval"):
            net.eval()
        outs = []
        with torch.no_grad():
            for i in range(len(pos)):
                outs.append(fn(pos[i], feats[i] if feats is not None else None).reshape(1, -1).cpu())
        if was_training and hasattr(net, "train"):
            net.train()
        return torch.cat(outs, 0)

    def forward_latent_equivariant(self, data):
        """Forward pass of a fitted 'latent_equivariant' contract (the nonlinear B3 path) on NEW data,
        returning the raw output tensor (n_samples, n_out) on CPU.

        Unlike the per-datum discovered-group contracts, the latent contract encodes each sample's ENTIRE
        position cloud FLATTENED to a fixed vector: at fit time the clouds were flattened and truncated to
        d = min(cloud length) (self._latent_input_dim). This replays that exact transform -- truncating
        longer clouds and zero-padding shorter ones to d -- so the frozen encoder receives the width it was
        built for. Test eval and predict() go through here."""
        d = getattr(self, "_latent_input_dim", None)
        if d is None or self.net is None:
            raise RuntimeError(
                "no latent-equivariant contract deployed -- fit with "
                "deploy_nonlinear_contract=True first (load() does not persist the latent "
                "chart, so a restored model cannot score through this path yet)."
            )
        pos = getattr(data, "positions", None)
        if pos is None:
            raise ValueError(
                "the latent_equivariant contract needs per-sample positions; this AllData "
                "carries none (build it with positions, e.g. AllData.point_sets(..., positions=...))."
            )
        rows = []
        for p in pos:
            c = np.asarray(p, dtype=np.float32).ravel()
            c = c[:d] if len(c) >= d else np.concatenate([c, np.zeros(d - len(c), np.float32)])
            rows.append(c)
        X = torch.as_tensor(np.stack(rows), dtype=torch.float32).to(self.device)
        net = self.net
        was_training = getattr(net, "training", False)
        net.eval()
        with torch.no_grad():
            out = net(X).cpu()
        if was_training:
            net.train()
        return out

    def _fit_set(self, data, task, n_out, primitives):
        from ..models import build_set_schema

        # sets share the graph-style batch but have no edges
        streaming = self._is_streaming_graph(data)
        X, batch, n_sets, y = self._collate_sets(data)
        n_out = self._infer_nout(data.y, task, n_out)
        n_in = data.node_feats.n_in if streaming else data.node_feats[0].shape[1]
        prims = primitives or ("deepsets", "element_mlp", "norm")  # SAB is O(N^2); opt-in via primitives
        net = build_set_schema(
            n_in=n_in, width=self.width, depth=self.depth, n_out=n_out, primitives=prims, readout="mean"
        ).to(self.device)
        self._break_alpha_symmetry(net)  # sparse: seed the mixture off uniform
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        yt = torch.as_tensor(y)
        yt_dev = (yt.long() if task == "classification" else yt.float().unsqueeze(1)).to(self.device)
        # STREAMING: skip the full on-device node cache; _subbatch_sets fetches each set from the GraphSource
        # per minibatch (cache=None) and _assemble_batch moves it to the device. Bit-identical to resident.
        cache = None if streaming else self._prepare_batch_cache(data, to_device=True)
        tr_idx, va_idx = self._auto_val_split(n_sets)
        stopper = self._make_stopper()
        prefetch = None
        if streaming:
            src = data.node_feats
            node_t = lambda i: src.node(i)
            _sfetch = lambda ids: self._collate_cpu(ids, node_t)

            def _scompute(ids, cpu):
                Xb, _ei, _p, bb, ng = self._batch_to_device(cpu)
                return lf(net(Xb, bb, ng), yt_dev[ids])

            _sloss = lambda ids: _scompute(ids, _sfetch(ids))
            prefetch = (_sfetch, _scompute) if self._prefetch_depth() > 0 else None
        else:

            def _sloss(ids):
                Xb, bb, ng = self._subbatch_sets(data, ids, cache=cache)  # already on device
                return lf(net(Xb, bb, ng), yt_dev[ids])

        permute = lambda idx: idx[np.random.permutation(len(idx))]
        self._run_epochs(
            net, opt, tr_idx, va_idx, stopper, _sloss, self._tb(self._SET_TRAIN_BATCH), permute, prefetch=prefetch
        )
        # eval
        outs = []
        for j in range(0, n_sets, 128):
            ids = np.arange(j, min(j + 128, n_sets))
            Xb, bb, ng = self._subbatch_sets(data, ids, cache=cache)
            with torch.no_grad():
                outs.append(net(Xb, bb, ng).cpu())
        return self._eval(net, torch.cat(outs), y, task)

    def _select_mode_budget(self, a, x, u, prims, sdims, in_ch, grid_min, task):
        """B7: select the Fourier-mode budget by the priced marginal-value rule (machinery.spectral_selection).
        Trains short operator models over a mode ladder on a train/val split, measures S(M)=val loss, and picks
        M* where the per-mode reduction stops beating mu * (added spectral code length). Returns the selection
        detail (with selected_modes) or None on failure -- falls back to the heuristic."""
        try:
            import torch

            from ..machinery.spectral_selection import measure_mode_curve, select_modes
            from ..models import build_operator_schema

            n = a.shape[0]
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(n)
            ntr = max(8, int(0.75 * n))
            tr, va = perm[:ntr], perm[ntr:]
            if len(va) < 2:
                return None
            a_d, x_d, u_d = a.to(self.device), x.to(self.device), u.to(self.device)
            cap = max(2, grid_min // 2)
            ladder = [m for m in (1, 2, 3, 4, 6, 8, 12) if m <= cap]
            if len(ladder) < 3:
                ladder = sorted(set([1, 2, max(2, cap)]))

            def train_eval(M, seed):
                torch.manual_seed(seed)
                np.random.seed(seed)
                net = build_operator_schema(
                    width=self.width,
                    depth=self.depth,
                    n_out=1,
                    primitives=prims,
                    modes=M,
                    in_channels=in_ch,
                    spatial_dims=sdims,
                    mode_override=M,
                ).to(self.device)
                opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
                ep = self._search_ep(max(15, self.epochs // 2))  # short sweep fits (capped)
                idx = np.array(tr)
                for _ in range(ep):
                    np.random.shuffle(idx)
                    for j in range(0, len(idx), self._tb()):
                        ids = idx[j : j + 32]
                        opt.zero_grad()
                        (((net(a_d[ids], x_d[ids]) - u_d[ids]) ** 2).mean()).backward()
                        opt.step()
                with torch.no_grad():
                    return float(((net(a_d[va], x_d[va]) - u_d[va]) ** 2).mean().item())

            mode_grid, S_mean, S_se, marginals = measure_mode_curve(train_eval, ladder, seeds=[self.seed])
            # price for the mode d.o.f.: use the explicit mode_mu if set, else a default calibrated to the
            # operator field-MSE scale (per-mode reductions are typically ~1e-5..1e-3 on normalized fields).
            mu = float(getattr(self, "mode_mu", None) or 1e-5)
            Mstar, detail = select_modes(mode_grid, S_mean, marginals, mu, spatial_dims=sdims, channels=self.width)
            detail["mode_grid"] = list(mode_grid)
            detail["val_loss_by_mode"] = [float(s) for s in S_mean]
            detail["heuristic_modes"] = max(2, min(12, grid_min // 2))
            return detail
        except Exception as e:
            self._log(f"[AllGraph] mode-budget selection skipped ({str(e)[:70]})")
            return None

    def _fit_operator(self, data, task, n_out, primitives):
        """Neural-operator contract: learn the function->function map a(x) -> u(x) on a grid. The loss is a
        per-grid-point field MSE (the whole output function is supervised), not a scalar readout. Fully
        discretization-invariant: the learned weights live in Fourier-mode space, so a model trained here at
        one resolution evaluates at any other (verified separately)."""
        from ..models import build_operator_schema

        streaming = self._is_streaming_operator(data)
        if streaming:
            # STREAMING: read shape metadata from the OperatorSource (no field materialized); a/x/u are fetched
            # per minibatch below. The upfront NaN scan is skipped (it would read every field); streaming
            # assumes clean fields, as documented on functions_stream.
            src = data.dense
            sdims = src.spatial_dims
            a_shape = src.a_shape  # full (n, *grid[, c])
            n = int(a_shape[0])
            in_ch = a_shape[-1] if len(a_shape) == 2 + sdims else 1
            grid_min = min(int(s) for s in a_shape[1 : 1 + sdims])
        else:
            a = (
                data.dense
                if isinstance(data.dense, torch.Tensor)
                else torch.tensor(np.asarray(data.dense), dtype=torch.float32)
            )
            x = (
                data.grid
                if isinstance(data.grid, torch.Tensor)
                else torch.tensor(np.asarray(data.grid), dtype=torch.float32)
            )
            u = torch.as_tensor(np.asarray(data.y), dtype=torch.float32)
            if torch.isnan(u).any() or torch.isnan(a).any():
                raise ValueError(
                    "operator contract received NaN in the input/target fields; check the data "
                    "generation (e.g. an unstable PDE solver) before fitting."
                )
            sdims = getattr(data, "spatial_dims", 1)
            # input channels: a has shape (n, *grid) [scalar] or (n, *grid, c) [vector]
            in_ch = a.shape[-1] if a.dim() == 2 + sdims else 1
            n = a.shape[0]
            grid_min = min(int(s) for s in a.shape[1 : 1 + sdims])
        prims = primitives or ("fourier", "fourier_wide", "local", "deeponet")
        # keep the mode budget below the smallest grid axis so the truncation is well-posed at train res
        modes = max(2, min(12, grid_min // 2))
        mode_override = None
        # B7: optionally SELECT the mode budget by the priced marginal-value rule (the operator-contract analogue
        # of width/depth selection), replacing the fixed heuristic. Measures S(M) on a train/val split over a
        # short ladder, then picks M* where the per-mode val-loss reduction stops beating mu * spectral code.
        if getattr(self, "price_modes", False):
            sel = self._select_mode_budget(a, x, u, prims, sdims, in_ch, grid_min, task)
            if sel is not None:
                mode_override = sel["selected_modes"]
                self.route_detail = {**(self.route_detail or {}), "select_modes": sel}
                self._log(
                    f"[AllGraph] d.o.f. stage (spectral) -> mode budget M*={mode_override} "
                    f"(heuristic was {modes}) by priced marginal-value rule"
                )
        net = build_operator_schema(
            width=self.width,
            depth=self.depth,
            n_out=1,
            primitives=prims,
            modes=modes,
            in_channels=in_ch,
            spatial_dims=sdims,
            mode_override=mode_override,
        ).to(self.device)
        self._break_alpha_symmetry(net)  # sparse: seed the mixture off uniform
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
        tr_idx, va_idx = self._auto_val_split(n)
        stopper = self._make_stopper()
        prefetch = None
        if streaming:
            # fetch a/grid/u for the minibatch from the source (per batch), then the same field MSE. The
            # np.random.permutation shuffle and the val split are unchanged, so the fit is bit-for-bit
            # equivalent to the resident path (which indexes whole-dataset device tensors). fetch (worker-safe
            # CPU read) is split from compute (device move + forward + loss) for async prefetch.
            pin = self._resolve_pin()

            def _ofetch(ids):
                ids = ids if isinstance(ids, np.ndarray) else np.asarray(ids)
                ab, xb, ub = src.a(ids), src.grid(ids), src.u(ids)
                if pin:
                    ab, xb, ub = ab.pin_memory(), xb.pin_memory(), ub.pin_memory()
                return ab, xb, ub

            def _ocompute(ids, payload):
                ab, xb, ub = (t.to(self.device, non_blocking=pin) for t in payload)
                return ((net(ab, xb) - ub) ** 2).mean()

            _oloss = lambda ids: _ocompute(ids, _ofetch(ids))
            prefetch = (_ofetch, _ocompute) if self._prefetch_depth() > 0 else None
        else:
            a_d, x_d, u_d = a.to(self.device), x.to(self.device), u.to(self.device)

            def _oloss(ids):
                return ((net(a_d[ids], x_d[ids]) - u_d[ids]) ** 2).mean()

        permute = lambda idx: idx[np.random.permutation(len(idx))]
        self._run_epochs(net, opt, tr_idx, va_idx, stopper, _oloss, self._tb(), permute, prefetch=prefetch)
        if streaming:
            r2 = self._stream_operator_eval(net, src, self._tb())  # streamed two-pass field-R2 (no full pred/u)
        else:
            with torch.no_grad():
                pred = net(a_d, x_d).cpu().numpy()
            uy = u.numpy()
            # accumulate the field mean in float64 (matches the streamed two-pass eval; a float32 mean drifts,
            # spuriously inflating ss_tot on a (near-)constant target and disagreeing with the streamed R2).
            r2 = 1.0 - ((pred - uy) ** 2).sum() / (((uy - uy.astype(np.float64).mean()) ** 2).sum() + 1e-12)
        self.net = net
        # architecture = the peak (argmax) primitive per cell, from the alpha simplex
        arch = [net.cells[i].primitives[int(net.cells[i].alpha_peak.argmax())] for i in range(len(net.cells))]
        return {"contract": "operator", "value": float(r2), "metric": "field_R2", "architecture": arch, "task": task}

    def _fit_relational(self, data, task, n_out, primitives, builder, with_pos, kind):
        streaming = self._is_streaming_graph(data)
        n_out = self._infer_nout(data.y, task, n_out)
        n_in = data.node_feats.n_in if streaming else data.node_feats[0].shape[1]
        if kind == "graph":
            prims = primitives or ("gcn", "gin", "pna", "gat", "norm")
            net = builder(
                n_in=n_in, width=self.width, depth=self.depth, n_out=n_out, primitives=prims, readout="mean"
            ).to(self.device)
        else:  # equivariant
            prims = primitives or ("e_tp", "e_painn", "e_gate", "e_norm")
            c1 = self.width // 2
            if getattr(self, "angular_from_data", False):
                keep_vec, adetail = self._select_angular_order(self._streaming_subsample(data) or data, task)
                if adetail is not None:
                    self.route_detail = {**(self.route_detail or {}), "angular_from_data": adetail}
                if not keep_vec:
                    c1 = 0  # radial target -> scalars only (l=0)
            net = builder(n_in=n_in, c0=self.width, c1=c1, depth=self.depth, n_out=n_out, primitives=prims).to(
                self.device
            )
        self._break_alpha_symmetry(net)  # sparse: seed the mixture off uniform
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self._wd())
        lf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        y = np.asarray(data.y)
        yt = torch.as_tensor(y)
        ng_total = len(data.node_feats)
        # STREAMING: skip the full per-graph tensor cache; _forward_relational fetches each graph from the
        # GraphSource per minibatch instead (cache=None). The np.random.permutation shuffle, the val split, and
        # the forward are otherwise identical, so the fit is bit-for-bit equivalent to the resident path.
        cache = None if streaming else self._prepare_batch_cache(data, with_pos=with_pos, with_edges=True)
        tr_idx, va_idx = self._auto_val_split(ng_total)
        stopper = self._make_stopper()

        def _tgt(ids):
            return (
                yt[ids].long().to(self.device)
                if task == "classification"
                else yt[ids].float().unsqueeze(1).to(self.device)
            )

        prefetch = None
        if streaming:
            # split fetch (worker-safe CPU collate from the GraphSource) from compute (device move + forward + loss)
            src = data.node_feats
            node_t = lambda i: src.node(i)
            edge_t = lambda i: src.edge(i)
            pos_t = (lambda i: src.pos(i)) if with_pos else None
            _rfetch = lambda ids: self._collate_cpu(ids, node_t, edge_t, pos_t)

            def _rcompute(ids, cpu):
                x, ei, p, b, ng = self._batch_to_device(cpu)
                out = net(x, p, ei, b, ng) if with_pos else net(x, ei, b, ng)
                return lf(out, _tgt(ids))

            _rloss = lambda ids: _rcompute(ids, _rfetch(ids))
            prefetch = (_rfetch, _rcompute) if self._prefetch_depth() > 0 else None
        else:

            def _rloss(ids):
                return lf(self._forward_relational(net, data, ids, with_pos, cache=cache), _tgt(ids))

        permute = lambda idx: idx[np.random.permutation(len(idx))]
        self._run_epochs(net, opt, tr_idx, va_idx, stopper, _rloss, self._tb(), permute, prefetch=prefetch)
        outs = []
        for j in range(0, ng_total, 64):
            ids = np.arange(j, min(j + 64, ng_total))
            with torch.no_grad():
                outs.append(self._forward_relational(net, data, ids, with_pos, cache=cache).cpu())
        return self._eval(net, torch.cat(outs), y, task)

    # ---------------------------------------------------------------- collate / forward helpers
    def _collate_cpu(self, ids, node_t, edge_t=None, pos_t=None):
        """The RNG-free CPU half of collation: run the per-sample accessors and stack (NO device move). Returns
        a payload dict (xs/batch/eis/pos lists + ng). Safe to run on an async-prefetch worker thread."""
        xs, batch, eis, pos = [], [], [], []
        off = 0
        for gi, i in enumerate(ids):
            t = node_t(i)
            n = t.shape[0]
            xs.append(t)
            batch.append(torch.full((n,), gi, dtype=torch.long))
            if edge_t is not None:
                eis.append(edge_t(i) + off)
            if pos_t is not None:
                pos.append(pos_t(i))
            off += n
        return {
            "xs": xs,
            "batch": batch,
            "eis": eis if edge_t is not None else None,
            "pos": pos if pos_t is not None else None,
            "ng": len(ids),
        }

    def _batch_to_device(self, cpu):
        """The main-thread half of collation: cat the CPU payload and move to the compute device."""
        dev = self.device
        x = torch.cat(cpu["xs"], 0).to(dev)
        b = torch.cat(cpu["batch"], 0).to(dev)
        ei = torch.cat(cpu["eis"], 1).to(dev) if cpu["eis"] is not None else None
        p = torch.cat(cpu["pos"], 0).to(dev) if cpu["pos"] is not None else None
        return x, ei, p, b, cpu["ng"]

    def _assemble_batch(self, ids, node_t, edge_t=None, pos_t=None):
        """Collate variable-size samples (graphs/sets) selected by `ids` into one batched call's tensors, each
        moved to the compute device; ei/p are None when their accessor is omitted. Defined as the composition of
        _collate_cpu (worker-safe) and _batch_to_device so every existing caller is byte-identical."""
        return self._batch_to_device(self._collate_cpu(ids, node_t, edge_t, pos_t))

    def _prepare_batch_cache(self, data, with_pos=False, with_edges=False, to_device=False):
        """Pre-convert each sample's static arrays (node features, and optionally edges/positions) to tensors
        ONCE, so per-minibatch collation only indexes + concatenates instead of re-running torch.as_tensor
        every epoch. Unifies the former graph (with_edges, CPU) and set (to_device) caches. `to_device` caches
        on the compute device -- used by the set contract, which runs on GPU; the relational contract caches on CPU
        (it runs on CPU via the Apple-Silicon fallback, so per-batch device moves are no-ops)."""

        def conv(arr, dtype):
            t = torch.as_tensor(arr, dtype=dtype)
            return t.to(self.device) if to_device else t

        node = [conv(nf, torch.float32) for nf in data.node_feats]
        edge = [conv(e, torch.long) for e in data.edges] if with_edges else None
        pos = [conv(p, torch.float32) for p in data.positions] if with_pos else None
        return {"node": node, "edge": edge, "pos": pos, "counts": [t.shape[0] for t in node]}

    def _forward_relational(self, net, data, ids, with_pos, cache=None):
        if cache is not None:
            node, edge, pos = cache["node"], cache["edge"], cache["pos"]
            node_t = lambda i: node[i]
            edge_t = lambda i: edge[i]
            pos_t = (lambda i: pos[i]) if with_pos else None
        elif self._is_streaming_graph(data):  # STREAMING: fetch each graph from the GraphSource
            src = data.node_feats
            node_t = lambda i: src.node(i)
            edge_t = lambda i: src.edge(i)
            pos_t = (lambda i: src.pos(i)) if with_pos else None
        else:  # on-the-fly (callers without a cache, e.g. size probes)
            node_t = lambda i: torch.as_tensor(data.node_feats[i], dtype=torch.float32)
            edge_t = lambda i: torch.as_tensor(data.edges[i], dtype=torch.long)
            pos_t = (lambda i: torch.as_tensor(data.positions[i], dtype=torch.float32)) if with_pos else None
        x, ei, p, b, ng = self._assemble_batch(ids, node_t, edge_t, pos_t)
        return net(x, p, ei, b, ng) if with_pos else net(x, ei, b, ng)

    def _collate_sets(self, data):
        return None, None, len(data.node_feats), np.asarray(data.y)

    def _subbatch_sets(self, data, ids, cache=None):
        if cache is not None:
            node = cache["node"]
            node_t = lambda i: node[i]
        elif self._is_streaming_graph(data):  # STREAMING: fetch each set from the GraphSource
            src = data.node_feats
            node_t = lambda i: src.node(i)
        else:
            node_t = lambda i: torch.as_tensor(data.node_feats[i], dtype=torch.float32)
        x, _ei, _p, b, ng = self._assemble_batch(ids, node_t)
        return x, b, ng

    # ---------------------------------------------------------------- shape + metric helpers
    def _as_grid(self, X, rank):
        """Ensure a dense tensor has a channel axis for the conv schemas. (b,H,W)->(b,1,H,W), etc.
        When the data arrives FLAT (b,D) but the router detected a latent grid (priced tensorization),
        reshape it to that discovered shape first -- this is how auto-tensorized flat vectors reach the
        spatial/volumetric/4d builders. The detected shape is read from route_detail['level2']['shape'].

        The result is always made contiguous: a loader may hand us a channels-last / transposed view, and
        MPS conv+BatchNorm backward requires contiguous NCHW input (CPU tolerates non-contiguous)."""
        if self._is_streaming(X):
            # STREAMING: return a _GridView that records the (channel-insert) rank fix-up and applies it inside
            # get() per batch, while exposing the transformed shape/dim so build_fn reads it without
            # materializing n rows. The flat-vector -> latent-lattice reshape branch below is unreachable here
            # (a streaming input requires a dense kind_hint, so mode discovery never runs).
            from .allgraph_streaming import _GridView

            return _GridView(X, rank)
        if X.dim() == rank:
            return X.contiguous()
        if X.dim() == rank - 1:
            return X.unsqueeze(1).contiguous()
        if X.dim() == 2:
            # flat vector + a router-detected latent lattice shape -> reshape into (b, *shape), add channel
            shape = None
            if isinstance(self.route_detail, dict):
                lvl2 = self.route_detail.get("level2", {})
                shape = lvl2.get("shape") if isinstance(lvl2, dict) else None
            if shape is not None and len(shape) == rank - 2 and int(np.prod(shape)) == X.shape[1]:
                return X.reshape(X.shape[0], 1, *shape).contiguous()
        raise ValueError(
            f"expected rank {rank} or {rank - 1}, got {X.dim()}"
            + (" and no matching detected grid shape for reshape" if X.dim() == 2 else "")
        )

    def _metric(self, out, y, task):
        y = np.asarray(y)
        if task == "classification":
            return "acc", float((out.argmax(1).numpy() == y).mean())
        pred = out.squeeze(-1).numpy()
        ss_res = ((pred - y) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return "R2", float(1 - ss_res / (ss_tot + 1e-12))

    def _eval(self, net, out, y, task):
        """Score already-computed outputs against labels and assemble the result dict. Shared by every
        contract's deploy fit -- the dense and relational paths were byte-identical."""
        metric, value = self._metric(out, y, task)
        return self._report(net, value, metric)

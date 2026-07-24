"""Inference-on-new-data and persistence for AllGraph (mixin).

predict / predict_proba run the trained net on fresh data; save / load persist a fitted model. Split out of
allgraph.py to keep the controller focused; AllGraph mixes this in, so `self` and `cls` resolve exactly
as before. NOTE: _default_model_dir uses __file__ -- this module lives in the same package dir as
allgraph.py (ilmarinen/core/), so the default out/ location is unchanged."""
from datetime import UTC

import numpy as np
import torch


class _PersistenceMixin:
    def _forward_new(self, data):
        """Run the deployed net's forward pass on NEW data and return the raw output tensor (logits for
        classification, field/values for regression), on CPU. Replays the SAME per-contract forward path the
        fit used -- including the sequence readout choice and the fit-time canonicalization quotient -- so a
        loaded model scores new data exactly as it was trained to. No labels required."""
        if self.net is None or self.contract is None:
            raise RuntimeError("AllGraph has no trained net -- call fit() or load() before predict().")
        net, mod, dev = self.net, self.contract, self.device
        net.eval()                                    # running BN/dropout stats -> correct on new/small batches
        if getattr(self, "_canonicalization_applied", False) and mod not in ("sequence", "spatial", "volumetric", "4d"):
            if self._is_streaming_graph(data):
                # apply_canonicalization materializes node features (it appends canonicalized coordinates per
                # sample), which a streamed test set cannot supply lazily -- give a clear error, not an opaque
                # TypeError from indexing the GraphSource as a list.
                raise NotImplementedError(
                    "predict() on a streamed test set is not supported for a model that applied a "
                    "canonicalization quotient at fit time; pass resident data (AllData.graphs / .point_sets).")
            data = self.apply_canonicalization(data)
        if mod == "generated_equivariant":
            # discovered-group contracts are per-datum (each sample's own point set -> group invariants), so
            # they have no batched net(x, ei, pos, batch, n) signature; replay their published forward.
            return self.forward_generated_equivariant(data)
        if mod == "latent_equivariant":
            # the nonlinear latent contract encodes each sample's FLATTENED cloud through a frozen chart, so
            # it also has no relational signature; replay the fit-time flatten + encode.
            return self.forward_latent_equivariant(data)
        if mod == "operator":
            if self._is_streaming_operator(data):        # streamed test set: forward the fields in chunks
                src = data.dense; n = len(src); outs = []
                if n == 0:                               # empty test set: return an empty field WITHOUT forwarding
                    # (the operator FFT rejects a 0-size batch on some backends, e.g. MKL, so we never call net).
                    return torch.zeros((0,) + tuple(src.a_shape[1:1 + src.spatial_dims]))
                for j in range(0, n, 64):
                    ids = np.arange(j, min(j + 64, n))
                    with torch.no_grad():
                        outs.append(net(src.a(ids).to(dev), src.grid(ids).to(dev)).cpu())
                return torch.cat(outs)
            a = data.dense if isinstance(data.dense, torch.Tensor) else torch.as_tensor(np.asarray(data.dense), dtype=torch.float32)
            xg = data.grid if isinstance(data.grid, torch.Tensor) else torch.as_tensor(np.asarray(data.grid), dtype=torch.float32)
            with torch.no_grad():
                return net(a.to(dev), xg.to(dev)).cpu()
        if mod in ("sequence", "spatial", "volumetric", "4d"):
            X = data.dense
            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            if mod == "sequence" and X.dim() == 2: X = X.unsqueeze(-1)
            if mod == "spatial" and X.dim() == 3: X = X.unsqueeze(1)
            if mod == "volumetric" and X.dim() == 4: X = X.unsqueeze(1)
            bs = 64 if mod == "volumetric" else 256
            outs = []
            for j in range(0, len(X), bs):
                with torch.no_grad():
                    if mod == "sequence":
                        xb = X[j:j + bs].to(dev)
                        # respect the readout the fit selected: 'flatten' uses net.forward, else pooled readout
                        o = net.forward(xb) if self._infer_readout == "flatten" \
                            else net.forward_seq_readout(xb, 1).squeeze(1)
                        outs.append(o.cpu())
                    else:
                        outs.append(net(X[j:j + bs].to(dev)).cpu())
            return torch.cat(outs)
        # relational contracts: set / graph / equivariant -- collate variable-size samples into one batched call.
        # Accepts either resident lists or a streaming GraphSource (predict on a streamed test set).
        with_pos = (mod == "equivariant"); use_edges = mod in ("graph", "equivariant")
        if self._is_streaming_graph(data):
            src = data.node_feats
            node_t = lambda i: src.node(i)
            edge_t = (lambda i: src.edge(i)) if use_edges else None
            pos_t = (lambda i: src.pos(i)) if with_pos else None
        else:
            node_t = lambda i: torch.as_tensor(data.node_feats[i], dtype=torch.float32)
            edge_t = (lambda i: torch.as_tensor(data.edges[i], dtype=torch.long)) if use_edges else None
            pos_t = (lambda i: torch.as_tensor(data.positions[i], dtype=torch.float32)) if with_pos else None
        outs = []; n = len(data.node_feats)
        for j in range(0, n, 128):
            ids = np.arange(j, min(j + 128, n))
            x, ei, p, b, ng = self._assemble_batch(ids, node_t, edge_t, pos_t)
            with torch.no_grad():
                if mod == "set": outs.append(net(x, b, ng).cpu())
                elif mod == "graph": outs.append(net(x, ei, b, ng).cpu())
                else: outs.append(net(x, p, ei, b, ng).cpu())
        return torch.cat(outs)

    def predict(self, data):
        """Predict on new data with the trained (or loaded) model. Returns a numpy array: class labels
        (argmax) for a classification model, or values/fields for a regression model. `data` is a AllData
        built the SAME way as the training data for this contract (e.g. AllData.dense_tensor / .graphs /
        .point_sets / .functions); labels in it, if any, are ignored."""
        out = self._forward_new(data)
        if self._infer_task == "classification":
            return out.argmax(1).numpy() if out.dim() == 2 and out.shape[1] > 1 else (out.squeeze(-1) > 0).long().numpy()
        return out.squeeze(-1).numpy()

    def predict_proba(self, data):
        """Class probabilities for a classification model, shape (n_samples, n_classes). Errors on a
        regression model."""
        if self._infer_task != "classification":
            raise ValueError("predict_proba is only defined for a classification model "
                             f"(this model's task is {self._infer_task!r}); use predict() for regression.")
        out = self._forward_new(data)
        if out.dim() == 2 and out.shape[1] > 1:
            return out.softmax(1).numpy()
        p1 = torch.sigmoid(out.squeeze(-1))          # single-logit binary head -> [P(0), P(1)]
        return torch.stack([1 - p1, p1], dim=1).numpy()

    @classmethod
    def _default_model_dir(cls):
        # the package "out" directory -- kept separate from models/ (which holds the schema SOURCE)
        import os
        return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "out"))

    def save(self, path=None, dirpath=None, stem=None):
        """Persist the trained model so it can be loaded later and run on new data. With no arguments the
        file lands in the package out/ folder as <stem>_<UTC-timestamp>.pt (the timestamp avoids
        collisions). `stem` sets the filename prefix (defaults to allgraph_<contract>); the validation
        runners pass the dataset name here. Pass `path` for a fully explicit filename, or `dirpath` to change
        only the directory. Returns the absolute path written. Requires a fitted model (call fit() first)."""
        import os
        from datetime import datetime
        if self.net is None or self.contract is None:
            raise RuntimeError("nothing to save -- call fit() before save().")
        payload = {
            "format_version": 1,
            "contract": self.contract,
            "task": self._infer_task,
            "readout": self._infer_readout,
            "route_detail": self.route_detail,
            "canonicalization_applied": bool(getattr(self, "_canonicalization_applied", False)),
            "config": {k: getattr(self, k, None) for k in
                       ("width", "depth", "epochs", "lr", "seed", "select", "sparsity_mu", "gibbs_beta")},
            "net": self.net,                          # full module (architecture + weights), pickled by torch.save
        }
        try:
            from .. import __version__ as _v
            payload["ilmarinen_version"] = _v
        except Exception:
            payload["ilmarinen_version"] = None
        if path is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")   # microseconds -> collision-free
            d = dirpath or self._default_model_dir()
            os.makedirs(d, exist_ok=True)
            prefix = stem or f"allgraph_{self.contract}"
            path = os.path.join(d, f"{prefix}_{stamp}.pt")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # move the net to CPU for a portable, device-agnostic checkpoint, then restore it
        dev = self.device
        try:
            self.net.to("cpu")
            torch.save(payload, path)
        finally:
            self.net.to(dev)
        self._log(f"[AllGraph] saved {self.contract} model -> {path}")
        return os.path.abspath(path)

    @classmethod
    def load(cls, path, device="auto"):
        """Load a model saved with save() and return an AllGraph ready for predict() on new data. `device`
        selects where inference runs (auto|mps|cuda|cpu or a torch.device); the net is moved there. The
        returned instance is inference-only -- its selection/training config is not fully reconstructed."""
        from ..device import prefer_cpu_on_mps, resolve_device
        dev = resolve_device(device)
        # weights_only=False: the checkpoint holds a full nn.Module, not just tensors (torch>=2.6 default is True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        contract = payload.get("contract", payload.get("modality"))      # accept new + legacy checkpoints
        # Apple-Silicon device policy: mirror fit()'s routing so a RELOADED model runs predict() on the device
        # fit() would have chosen. The scatter-bound relational (graph/equivariant/set) and launch-bound dense
        # (sequence/volumetric/4d) contracts are faster on CPU than MPS; additionally, with
        # PYTORCH_ENABLE_MPS_FALLBACK unset some relational ops (index_reduce, spline searchsorted) would RAISE
        # on MPS rather than degrade. Without this guard a model saved+reloaded with the default device='auto'
        # would predict on MPS -- the exact ~15x (and crash-prone) path fit() routes around. MPS-gated only;
        # CUDA is untouched. See ilmarinen.device.prefer_cpu_on_mps.
        if prefer_cpu_on_mps(contract, dev):
            dev = torch.device("cpu")
        mg = cls(device=dev, verbose=False)
        mg.net = payload["net"].to(dev); mg.net.eval()
        mg.device = dev; mg._base_device = dev
        mg.contract = contract
        mg._infer_task = payload.get("task")
        mg._infer_readout = payload.get("readout")
        mg.route_detail = payload.get("route_detail")
        mg._canonicalization_applied = bool(payload.get("canonicalization_applied", False))
        cfg = payload.get("config") or {}
        for k, v in cfg.items():
            if v is not None:
                setattr(mg, k, v)
        return mg

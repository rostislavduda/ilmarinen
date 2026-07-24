"""Automatic tensorization + schema routing.

Ties the mode-structure front-end to the metaoptimizer: discover the data's coordinate structure
(mode_structure.discover_mode_structure), then ROUTE to the schema whose tensor interface matches
  - '2d'          -> the spatial schema (conv2d/conv_dw/pointwise/...) on the discovered H x W grid
  - '1d'          -> the sequence schema (plain/gated/lstm/conv/linssm/attention/spectral)
  - 'unstructured'-> the sequence schema as a length-1 vector with dense-family primitives
and reshape the flat input into the tensor layout that schema expects. This is the physicist
pipeline's representation stage made operational: the data's correlation structure, discovered from
mutual information, selects both the tensorization and the class of operations searched.

The router does NOT itself train; it returns (supergraph_kind, builder_kwargs, tensorize_fn,
detected_structure) so a caller (a runner) can build and metaoptimize on the correctly-shaped input.
"""
from __future__ import annotations

import numpy as np
import torch

from .mode_structure import discover_mode_structure


def route_by_structure(X_flat, y=None, force=None, verbose=False):
    """Decide the schema + tensorization for flat inputs X_flat (n, d).

    force : optionally override detection with '1d' / '2d' / 'unstructured' (for ablation).
    Returns dict:
      kind        : 'spatial' | 'sequence'
      structure   : the detected (or forced) structure label
      shape       : (H, W) for spatial, (d,) for sequence
      tensorize   : callable mapping (n, d) flat tensor -> the layout the schema expects
                    (spatial: (n, 1, H, W); sequence: (n, T, 1) for 1d, (n, 1, d) for unstructured)
      build_hint  : dict of suggested builder kwargs (primitives, hw/n_in, etc.)
    """
    Xf = X_flat if isinstance(X_flat, torch.Tensor) else torch.tensor(X_flat, dtype=torch.float32)
    d = Xf.shape[1]
    if force is not None:
        if force == "2d":
            # pick the most-square exact factorization as a fallback shape
            H = max(h for h in range(1, int(d ** 0.5) + 1) if d % h == 0)
            det = {"structure": "2d", "shape": (H, d // H), "recommended_primitives":
                   ["conv2d", "conv_dw", "pointwise", "attention", "norm"]}
        elif force == "1d":
            det = {"structure": "1d", "shape": (d,), "recommended_primitives":
                   ["plain", "gated", "lstm", "conv", "linssm", "attention", "spectral"]}
        else:
            det = {"structure": "unstructured", "shape": None, "recommended_primitives":
                   ["dense", "norm", "attention"]}
    else:
        det = discover_mode_structure(np.asarray(Xf.cpu()))

    structure = det["structure"]
    if verbose:
        print(f"[route] structure={structure} shape={det.get('shape')} "
              f"prims={det['recommended_primitives'][:4]}")

    if structure == "2d":
        H, W = det["shape"]
        def tensorize(Z):
            Z = Z if isinstance(Z, torch.Tensor) else torch.tensor(Z, dtype=torch.float32)
            return Z.reshape(Z.shape[0], 1, H, W)              # (n, 1, H, W): 1 input channel
        return {"kind": "spatial", "structure": "2d", "shape": (H, W), "tensorize": tensorize,
                "build_hint": {"primitives": tuple(det["recommended_primitives"]),
                               "hw": H if H == W else max(H, W), "n_in": 1, "img_size": max(H, W)},
                "detection": det}

    if structure in ("3d", "4d"):
        dims = tuple(det["shape"])                              # (D,H,W) or (T,D,H,W)
        kind = "volumetric" if structure == "3d" else "4d"
        def tensorize(Z, _dims=dims):
            Z = Z if isinstance(Z, torch.Tensor) else torch.tensor(Z, dtype=torch.float32)
            return Z.reshape(Z.shape[0], 1, *_dims)             # (n, 1, D, H, W) / (n, 1, T, D, H, W)
        return {"kind": kind, "structure": structure, "shape": dims, "tensorize": tensorize,
                "build_hint": {"primitives": tuple(det["recommended_primitives"]), "n_in": 1,
                               "dims": dims},
                "detection": det}

    if structure == "1d":
        def tensorize(Z):
            Z = Z if isinstance(Z, torch.Tensor) else torch.tensor(Z, dtype=torch.float32)
            return Z.reshape(Z.shape[0], Z.shape[1], 1)        # (n, T, 1): length-T, 1 channel
        return {"kind": "sequence", "structure": "1d", "shape": (d,), "tensorize": tensorize,
                "build_hint": {"primitives": tuple(det["recommended_primitives"]), "n_in": 1},
                "detection": det}

    # unstructured -> sequence schema as a length-1 vector, dense-family primitives
    def tensorize(Z):
        Z = Z if isinstance(Z, torch.Tensor) else torch.tensor(Z, dtype=torch.float32)
        return Z.reshape(Z.shape[0], 1, Z.shape[1])            # (n, 1, d): length-1, d channels
    return {"kind": "sequence", "structure": "unstructured", "shape": (d,), "tensorize": tensorize,
            "build_hint": {"primitives": tuple(det["recommended_primitives"]), "n_in": d},
            "detection": det}

"""Runnable example: train an AllGraph dense contract by STREAMING the dataset from disk.

The dense contracts normally move the whole training tensor onto the compute device at once, which caps the
dataset size by device / host memory. `AllData.dense_stream(source, ...)` instead pulls one minibatch at a
time from a `DenseSource`, so you can train on data far larger than RAM. This is opt-in purely by container
type -- building the input any other way keeps the exact in-memory path.

This script:
  1. writes a synthetic spatial dataset to a .npy file (standing in for an on-disk corpus too big to load);
  2. trains once RESIDENT (whole array in memory) and once STREAMING (memmapped, one minibatch at a time);
  3. shows the two fits produce BIT-IDENTICAL weights -- streaming changes memory behaviour, not the result.

Run from the repository root:

    python -m examples.streaming_dataset
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from ilmarinen import AllData, AllGraph, MemmapDenseSource


def make_dataset(n=600, hw=10, seed=0):
    """A synthetic (n, hw, hw) image dataset with a clean binary label (sign of the total intensity)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, hw, hw).astype(np.float32)
    y = (X.sum(axis=(1, 2)) > 0).astype(np.int64)
    return X, y


def weights_identical(net_a, net_b):
    sa, sb = net_a.state_dict(), net_b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    X, y = make_dataset()
    cfg = dict(width=16, depth=2, epochs=10, seed=0, verbose=False)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "images.npy"
        np.save(path, X)  # the "on-disk corpus"
        size_mb = path.stat().st_size / 1e6
        print(f"wrote {X.shape} dataset to {path.name} ({size_mb:.1f} MB on disk)")

        # ---- RESIDENT: the whole array is materialized and moved onto the device ----
        mg_resident = AllGraph(**cfg)
        r_res = mg_resident.fit(
            AllData.dense_tensor(X, y, kind_hint="spatial"),
            task="classification",
            n_out=2,
        )

        # ---- STREAMING: the .npy is memmapped; only the minibatch in flight is ever resident ----
        # MemmapDenseSource reads rows lazily via np.memmap -- the full array is never loaded. kind_hint is
        # required so routing never falls back to a whole-matrix analysis pass. Labels stay resident (small).
        source = MemmapDenseSource(str(path))
        mg_stream = AllGraph(**cfg)
        r_stream = mg_stream.fit(
            AllData.dense_stream(source, y=y, kind_hint="spatial"),
            task="classification",
            n_out=2,
            stream=True,  # optional: assert we really are streaming
        )

    print()
    print(f"resident : contract={r_res['contract']:8s} value={r_res['value']:.4f}  n_params={r_res['n_params']}")
    print(
        f"streaming: contract={r_stream['contract']:8s} value={r_stream['value']:.4f}  n_params={r_stream['n_params']}"
    )
    print()
    identical = weights_identical(mg_resident.net, mg_stream.net)
    print(f"trained weights bit-identical (streaming == resident): {identical}")
    print(f"reported values equal:                                 {r_res['value'] == r_stream['value']}")

    # The streaming model is a normal fitted AllGraph: predict() runs on ordinary resident data.
    Xnew, _ = make_dataset(n=8, seed=99)
    preds = mg_stream.predict(AllData.dense_tensor(Xnew, kind_hint="spatial"))
    print(f"streaming model predict() on 8 new samples -> {preds.tolist()}")

    # For a source too large to fit even on disk as one .npy, back the DenseSource with anything random-access
    # (a directory of shards, an HDF5 dataset, a cloud object store): subclass DenseSource and implement
    # __len__, `_sample_shape`/dtype, and get(ids) -> (len(ids), *sample_shape) float32 CPU tensor in id order.

    if not identical:
        raise SystemExit("streaming diverged from the resident fit -- this should never happen")
    print("\nOK: streaming trained the same model as the resident fit, without ever loading the full array.")


if __name__ == "__main__":
    main()

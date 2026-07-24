"""Runnable example: train the neural-operator contract by STREAMING function fields from disk.

The operator contract learns a function->function map a(x) -> u(x) on a grid, where BOTH the input a and the
target u are field-valued (large). The resident path moves the whole a / grid / u onto the device and computes
the field-R2 with the whole prediction and whole target in memory. `AllData.functions_stream` instead pulls
a / grid / u one minibatch at a time from an `OperatorSource`, and computes the field-R2 in a streamed
two-pass accumulation (global field mean, then residual/total sums of squares) -- so neither the inputs NOR
the targets need to be resident.

This script:
  1. writes the input fields a and target fields u to .npy files on disk;
  2. trains once RESIDENT and once STREAMING (memmapped, one minibatch of fields at a time);
  3. shows bit-identical weights and a matching streamed field-R2.

Run from the repository root:

    python -m examples.streaming_operator
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from ilmarinen import AllData, AllGraph, MemmapOperatorSource


def make_operator_data(n=400, grid=32, seed=0):
    """A well-posed 1D operator: the target field u is a local smoothing of the input field a."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n, grid).astype(np.float32)
    kernel = np.array([0.25, 0.5, 0.25], np.float32)
    u = np.stack([np.convolve(row, kernel, mode="same") for row in a]).astype(np.float32)
    return a, u


def weights_identical(net_a, net_b):
    sa, sb = net_a.state_dict(), net_b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    a, u = make_operator_data()
    cfg = dict(width=12, depth=1, epochs=12, seed=0, verbose=False)

    with tempfile.TemporaryDirectory() as d:
        pa, pu = Path(d) / "a.npy", Path(d) / "u.npy"
        np.save(pa, a)
        np.save(pu, u)
        mb = (pa.stat().st_size + pu.stat().st_size) / 1e6
        print(f"wrote input fields {a.shape} and target fields {u.shape} to disk ({mb:.1f} MB)")

        # ---- RESIDENT: whole a / grid / u moved onto the device; field-R2 over the whole prediction ----
        mg_resident = AllGraph(**cfg)
        r_res = mg_resident.fit(AllData.functions(a, u), task="regression", n_out=1)

        # ---- STREAMING: memmap a and u; a shared grid is broadcast per batch; two-pass streamed field-R2 ----
        # kind_hint is fixed to 'operator'. spatial_dims is inferred from the field shapes (1 here). Neither
        # the input fields nor the (equally large) target fields are ever fully resident.
        source = MemmapOperatorSource(str(pa), str(pu))
        mg_stream = AllGraph(**cfg)
        r_stream = mg_stream.fit(
            AllData.functions_stream(source),
            task="regression",
            n_out=1,
            stream=True,
        )

    print()
    print(f"resident : contract={r_res['contract']} metric={r_res['metric']} value={r_res['value']:.4f}")
    print(f"streaming: contract={r_stream['contract']} metric={r_stream['metric']} value={r_stream['value']:.4f}")
    print()
    identical = weights_identical(mg_resident.net, mg_stream.net)
    print(f"trained weights bit-identical (streaming == resident): {identical}")
    print(f"streamed field-R2 matches resident (|diff| < 1e-5):     {abs(r_res['value'] - r_stream['value']) < 1e-5}")

    # The streaming model predicts field-valued outputs; predict() also accepts a streamed test set.
    preds = mg_stream.predict(AllData.functions(a[:4], u[:4]))
    print(f"streaming model predict() on 4 samples -> field output shape {preds.shape}")

    if not identical:
        raise SystemExit("streaming diverged from the resident fit -- this should never happen")
    print("\nOK: streamed the operator fields from disk and trained the same model as the resident fit.")


if __name__ == "__main__":
    main()

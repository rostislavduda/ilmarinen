"""T-DEV: device selection, fallback, and per-device execution.

Two layers:

* The **resolve / fallback logic** (``ilmarinen.device``) is tested on any host — it needs
  no GPU. Requesting an absent backend must fall back to CPU *with a warning*.
* The **end-to-end fit** is parametrized over the devices actually present: CPU always, plus
  CUDA and/or Apple-Silicon MPS when the running host exposes them (a dev machine, a
  self-hosted runner, or a GPU CI runner). On a CPU-only host the GPU cases simply don't
  appear in the parametrization — so this file is meaningful everywhere and becomes real
  GPU coverage the moment it runs on GPU hardware.
"""

import numpy as np
import pytest
import torch

from ilmarinen import AllData, AllGraph
from ilmarinen.device import best_device, mps_available, resolve_device


def _available_devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if mps_available():
        devs.append("mps")
    return devs


# --------------------------------------------------------------- resolve / fallback (no GPU needed)
def test_resolve_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    # "auto"/None resolve to whatever is best on this host; always a valid backend.
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    assert resolve_device(None).type in ("cpu", "cuda", "mps")


def test_resolve_torch_device_passthrough():
    d = torch.device("cpu")
    assert resolve_device(d) is d


def test_best_device_auto_is_valid():
    assert best_device("auto", verbose=False).type in ("cpu", "cuda", "mps")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present: the fallback path is not exercised")
def test_cuda_request_falls_back_to_cpu_with_warning():
    with pytest.warns(UserWarning):
        assert best_device("cuda", verbose=False).type == "cpu"
    with pytest.warns(UserWarning):
        assert resolve_device("cuda:0").type == "cpu"


@pytest.mark.skipif(mps_available(), reason="MPS present: the fallback path is not exercised")
def test_mps_request_falls_back_to_cpu_with_warning():
    with pytest.warns(UserWarning):
        assert best_device("mps", verbose=False).type == "cpu"


# --------------------------------------------------------------- per-device execution (real HW when present)
@pytest.mark.parametrize("dev", _available_devices())
def test_fit_runs_on_device(dev, linsep_tabular):
    """A minimal fit completes and is well-formed on each available device.

    Asserts the requested (available) device was actually selected — a silent fall-back to CPU
    when the device is present would be a bug — and that the fit produces a finite objective.
    """
    X, y = linsep_tabular()
    mg = AllGraph(width=8, depth=1, epochs=3, verbose=False, seed=0, device=dev)
    assert mg.device.type == dev  # resolve_device honored the (present) request; no silent fallback
    r = mg.fit(AllData.dense_tensor(X, y), task="classification", n_out=2)
    assert np.isfinite(r["value"])
    assert {"contract", "value", "architecture"}.issubset(r.keys())

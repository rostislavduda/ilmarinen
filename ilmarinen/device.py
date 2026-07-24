"""Device selection for ilmarinen — CUDA, Apple-Silicon (MPS), or CPU.

ilmarinen is written to be device-agnostic: models create their internal tensors with ``device=x.device``
and ``dtype=x.dtype`` (deriving both from the input), constants are moved with ``.to(t.device, t.dtype)``,
and every training loop moves the net and each batch onto ``self.device``. There is no hardcoded ``.cuda()``
anywhere. What was missing is a single place to CHOOSE the device, so this module provides ``best_device()``
(auto-detect) and ``resolve_device()`` (used by ``AllGraph`` to turn ``device="auto"`` into a concrete
device).

Apple Silicon (MPS) notes
-------------------------
* MPS is selected automatically when available and CUDA is not.
* A few operations have historically had incomplete MPS coverage. ilmarinen touches two:
  the Fourier operator/spectral primitives (``torch.fft`` + complex ``cfloat``) in
  ``operator_schema`` and the sequence spectral primitive, and an eigenvalue diagnostic
  (``torch.linalg.eigvals``) in ``neural_ode``. For robustness across PyTorch/macOS versions, set the
  standard PyTorch escape hatch ``PYTORCH_ENABLE_MPS_FALLBACK=1`` in the environment so any op MPS does not
  implement transparently runs on the CPU instead of raising. ``best_device()`` will emit a one-time note
  recommending this when it selects MPS and the flag is not set.
* MPS does not support float64 tensors. ilmarinen never puts a float64 torch tensor on a device (its float64
  use is confined to NumPy-side analysis: mode structure, interpretability, the TV/width solvers), so this
  is not a problem in practice; the helper below nonetheless defaults new device tensors to float32.
"""

from __future__ import annotations

import os
import warnings

import torch

_MPS_FALLBACK_NOTED = False


def mps_available() -> bool:
    """True iff a usable Apple-Silicon MPS backend is present."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def best_device(prefer: str = "auto", verbose: bool = True) -> torch.device:
    """Return the best available torch device.

    Priority: explicit ``prefer`` (if a real device string) > CUDA > MPS (Apple Silicon) > CPU.

    prefer : "auto" (default) picks the best available; or one of "cuda", "mps", "cpu" to request a
             specific backend (falls back to CPU with a warning if the requested backend is unavailable).
    verbose: when selecting MPS, emit a one-time note recommending PYTORCH_ENABLE_MPS_FALLBACK=1 unless it
             is already set.
    """
    prefer = (prefer or "auto").lower()

    if prefer in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA requested but not available; falling back to CPU.", stacklevel=2)
        return torch.device("cpu")
    if prefer == "mps":
        if mps_available():
            _note_mps_fallback(verbose)
            return torch.device("mps")
        warnings.warn("MPS requested but not available; falling back to CPU.", stacklevel=2)
        return torch.device("cpu")
    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        _note_mps_fallback(verbose)
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device) -> torch.device:
    """Turn a user-supplied ``device`` into a concrete ``torch.device``.

    Accepts: ``None`` or ``"auto"`` -> auto-detect via :func:`best_device`; a string like ``"cpu"``,
    ``"cuda"``, ``"cuda:0"``, ``"mps"``; or an existing ``torch.device`` (returned unchanged). This is what
    ``AllGraph(device=...)`` calls, so ``device="auto"`` transparently uses the Apple-Silicon GPU when
    present and falls back to CPU otherwise.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return best_device("auto")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        d = device.lower()
        if d in ("cuda", "gpu", "mps", "cpu"):
            return best_device(d)
        # a specific string like "cuda:1" -- honor it, but fall back if the family is unavailable
        if d.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(f"'{device}' requested but CUDA unavailable; using CPU.", stacklevel=2)
            return torch.device("cpu")
        if d.startswith("mps") and not mps_available():
            warnings.warn(f"'{device}' requested but MPS unavailable; using CPU.", stacklevel=2)
            return torch.device("cpu")
        return torch.device(device)
    raise TypeError(f"unrecognized device specification: {device!r}")


#: Contracts measured FASTER on CPU than on Apple-Silicon MPS at the model sizes ilmarinen uses (median of
#: end-to-end fit() runs at the production per-contract budgets):
#:   * relational family -- graph / equivariant / set: scatter/index_add_ aggregation is ~4x slower on MPS;
#:   * sequence: the recurrent per-timestep Python loops are launch/sync-bound (~1.7x);
#:   * 4d: the conv4d temporal-decomposition K_t loop over tiny 3D convs (~2.1x);
#:   * volumetric: small / depthwise conv3d at width 16 (~1.4x).
#: Deliberately EXCLUDED (faster on MPS, keep on GPU): spatial (dense conv2d, MPS ~5x) and operator
#: (FFT+complex but matmul-dominated at the width-24/depth-3 budget -- MPS 1.2-5.2x across 1D/2D/3D).
MPS_CPU_FASTER_CONTRACTS = frozenset({"graph", "equivariant", "set", "sequence", "volumetric", "4d"})


def prefer_cpu_on_mps(contract, device) -> bool:
    """True iff `contract` runs faster on CPU than on the given `device`, so the caller should pin it to CPU.

    Gated on Apple-Silicon **MPS only**: CPU-beats-GPU here is an MPS launch-overhead / missing-kernel
    phenomenon, so on CUDA (or CPU) these same ops are fast and this returns False -- never force CPU on
    CUDA. Used by :meth:`AllGraph.fit` (post-routing) and :meth:`AllGraph.load` to pin the scatter- and
    launch-bound contracts to CPU when the requested device is MPS. See :data:`MPS_CPU_FASTER_CONTRACTS`.
    """
    return contract in MPS_CPU_FASTER_CONTRACTS and str(device).startswith("mps")


def _note_mps_fallback(verbose: bool) -> None:
    global _MPS_FALLBACK_NOTED
    if verbose and not _MPS_FALLBACK_NOTED and not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        _MPS_FALLBACK_NOTED = True
        warnings.warn(
            "Selected Apple-Silicon MPS device. A few ops ilmarinen can use (FFT/complex in the operator and "
            "spectral primitives; an eigenvalue diagnostic) may lack native MPS kernels on some PyTorch/macOS "
            "versions. For robustness set PYTORCH_ENABLE_MPS_FALLBACK=1 in your environment so any such op "
            "runs on CPU transparently instead of raising.",
            stacklevel=3,
        )

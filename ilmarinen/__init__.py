"""Ilmarinen: a physics-principled neural-architecture meta-optimizer.

Ilmarinen turns architecture search into a single description-length minimization run along a physicist's
ordering of decisions -- ``kinematics -> degrees of freedom -> dynamics -> observables`` -- with the MDL
action ``J = R + mu*Omega`` minimized at every stage. The one entry point is :class:`AllGraph`: given a
:class:`AllData` object it fixes the contract (which symmetry / which of the peer schema contracts), counts
the degrees of freedom (width/depth, optionally per-layer width, kernel size, angular order), trains the
weights and the primitive mixture, and reads off a deployed architecture.

Quick start
-----------
>>> from ilmarinen import AllGraph, AllData
>>> mg = AllGraph(device="auto")          # auto-detect: CUDA > Apple-Silicon MPS > CPU
>>> data = AllData.dense_tensor(X, y)      # or AllData.graphs(...) / .point_sets(...) / .functions(...)
>>> result = mg.fit(data, task="classification")
>>> print(mg.explain(result, as_text=True)) # architecture-as-explanation (Tier 1 interpretability)

Package layout (mirrors the pipeline)
-------------------------------------
* ``ilmarinen.core``      -- the pipeline: :class:`AllGraph`/:class:`AllData`, routing, the symmetry-discovery
                           front-end, the size selectors, and the interpretability/redundancy read-outs.
* ``ilmarinen.models``    -- the eight peer schema contracts (sequence, spatial, volumetric, 4d, graph,
                           equivariant, set, operator) plus the discovered-group EMLP path; each is a uniform
                           ``builder(...) -> nn.Module`` whose cells hold a primitive tuple and an alpha simplex.
* ``ilmarinen.machinery`` -- the priced-selection machinery (Gibbs alpha, sparsity pricing, priced depth, the
                           dual certificate / TV solvers, contract MDL, the learned router).
* ``ilmarinen.validation``-- validation pipelines.
* ``ilmarinen.device``    -- device auto-detection (:func:`best_device`, :func:`resolve_device`), CUDA/MPS/CPU.

Device / GPU
------------
Everything is device-agnostic: models build internal tensors on the input's device and dtype, and every
training loop moves the net and each batch onto ``AllGraph.device``. Pass ``device="auto"`` (or ``"mps"`` /
``"cuda"``) to :class:`AllGraph`; on Apple Silicon the MPS GPU is used, and the FFT-based operator and
spectral primitives fall back to CPU transparently on PyTorch/macOS versions lacking those MPS kernels (set
``PYTORCH_ENABLE_MPS_FALLBACK=1`` for the fully native path). See :mod:`ilmarinen.device`.
"""

# ---- the primary entry point (the physicist's pipeline) -----------------------------------------------
from .core.allgraph import AllGraph, AllData

# ---- opt-in dataset streaming (train on data larger than RAM/VRAM) ------------------------------------
from .core.allgraph_streaming import (DenseSource, InMemoryDenseSource, MemmapDenseSource,
                                      GraphSource, InMemoryGraphSource, LazyGraphSource,
                                      OperatorSource, InMemoryOperatorSource, MemmapOperatorSource,
                                      IterableDenseSource, InMemoryIterableDenseSource)

# ---- device selection ----------------------------------------------------------------------------------
from .device import best_device, resolve_device, mps_available

# ---- stage-level building blocks (import from the subpackages for the full surface) --------------------
from .core.route import route_by_structure
from .core.symmetry_pipeline import discover_and_reduce, continuous_invariant_features
from .core.interpretability import explain, format_report
from .core.feature_attribution import fit_feature_attribution, feature_selection_path
from .core.symbolic_readout import symbolic_readout, symbolify_model
from .core.redundancy_reduction import effective_dimension, reduce_redundancy
from .core.ib_rg_flow import ib_rg_flow, layer_rg_flow, gib_spectrum, ib_effective_dimension, critical_betas
from .core.variable_width_area import (fit_variable_width_area, area_price_path,
                                       certificate_lambda_scale, VariableWidthNet)
from .machinery.priced_depth import measure_depth_curve, select_depth, significant_elbow
from .machinery.contract_evidence import contract_evidence, score_to_nll
from .machinery.singular_complexity import estimate_llc, free_energy
from .machinery.singular_mdl import (omega_func, total_code_length,
                                     singular_complexity_of, singular_free_energy)
from .machinery.developmental_llc import developmental_llc, default_checkpoints
from .machinery.thermodynamic_potential import (wbic_beta, free_energy_form,
                                                assert_temperature_consistency, POTENTIAL_LEVELS)
from .machinery.response_spectroscopy import (gibbs_susceptibility, contract_transition,
                                              response_spectrum)
from .machinery.effective_dimension_ledger import (participation_ratio, effective_dimension_ledger,
                                                    LEDGER_LEVELS)
from .models.latent_equivariant_contract import build_latent_equivariant_contract
from .models.scalable_equivariant import build_scalable_equivariant_mlp, ScalableEquivariantMLP
from .models.approximate_equivariance import ApproxEquivariantModel, select_relaxation, price_relaxation
from .machinery.spectral_selection import select_modes, measure_mode_curve, spectral_code_length

# ---- validation subpackage -----------------------------------------------------------------------------
from . import validation

# ---- legacy / early-theory exports (DEPRECATED; removed in a future release) ---------------------------
# Resolved LAZILY with a DeprecationWarning (PEP 562): keeps them working for one release, drops them from
# THIS module's eager imports, and gives callers notice. See ilmarinen/legacy/ for the quarantined home.
# (A few early-theory modules are still eager-loaded by core/__init__/machinery/__init__ -- separate cleanup.)
_LEGACY_EXPORTS = {
    "FashionMNIST": "ilmarinen.core.data",
    "MeanFieldTheory": "ilmarinen.core.meanfield",
    "critical_exponent": "ilmarinen.core.exponent",
    "empirical_exponent": "ilmarinen.core.exponent",
    "build_model": "ilmarinen.legacy.networks",
    "MODEL_REGISTRY": "ilmarinen.legacy.networks",
    "build_rnn": "ilmarinen.legacy.recurrent",
    "RNN_REGISTRY": "ilmarinen.legacy.recurrent",
    "greedy_insertion": "ilmarinen.machinery.width_sparsity",
    "column_generation_solve": "ilmarinen.machinery.column_generation",
}


def __getattr__(name):          # PEP 562: lazily resolve + deprecation-warn the legacy exports
    source = _LEGACY_EXPORTS.get(name)
    if source is not None:
        import importlib
        from ._deprecation import warn_legacy
        warn_legacy(name, source)
        return getattr(importlib.import_module(source), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():                  # keep the deprecated names discoverable (and visible to test_api_surface)
    return sorted(list(globals()) + list(_LEGACY_EXPORTS))

__version__ = "2.2.0"

__all__ = [
    # primary API
    "AllGraph", "AllData",
    # opt-in dataset streaming
    "DenseSource", "InMemoryDenseSource", "MemmapDenseSource",
    "GraphSource", "InMemoryGraphSource", "LazyGraphSource",
    "OperatorSource", "InMemoryOperatorSource", "MemmapOperatorSource",
    "IterableDenseSource", "InMemoryIterableDenseSource",
    # device
    "best_device", "resolve_device", "mps_available",
    # pipeline stages
    "route_by_structure", "discover_and_reduce", "continuous_invariant_features",
    "explain", "format_report",
    "fit_feature_attribution", "feature_selection_path",
    "symbolic_readout", "symbolify_model",
    "effective_dimension", "reduce_redundancy",
    "ib_rg_flow", "layer_rg_flow", "gib_spectrum", "ib_effective_dimension", "critical_betas",
    "fit_variable_width_area", "area_price_path", "certificate_lambda_scale", "VariableWidthNet",
    "measure_depth_curve", "select_depth", "significant_elbow",
    "contract_evidence", "score_to_nll",
    "estimate_llc", "free_energy",
    "omega_func", "total_code_length", "singular_complexity_of", "singular_free_energy",
    "developmental_llc", "default_checkpoints",
    "wbic_beta", "free_energy_form", "assert_temperature_consistency", "POTENTIAL_LEVELS",
    "gibbs_susceptibility", "contract_transition", "response_spectrum",
    "participation_ratio", "effective_dimension_ledger", "LEDGER_LEVELS",
    "build_latent_equivariant_contract",
    "build_scalable_equivariant_mlp", "ScalableEquivariantMLP",
    "ApproxEquivariantModel", "select_relaxation", "price_relaxation",
    "select_modes", "measure_mode_curve", "spectral_code_length",
    "validation",
    # legacy (DEPRECATED -- emit DeprecationWarning on access; removed in a future release)
    "FashionMNIST", "MeanFieldTheory", "critical_exponent", "empirical_exponent",
    "build_model", "MODEL_REGISTRY", "build_rnn", "RNN_REGISTRY",
    "greedy_insertion", "column_generation_solve",
]

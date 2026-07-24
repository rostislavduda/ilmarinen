"""Quarantined legacy subsystem -- the pre-AllGraph pipeline and supergraphs.

Superseded by AllGraph + the per-arena build_*_schema contracts; retained for provenance and reproducibility of earlier
results. NOT on the primary import path -- AllGraph never imports this. Every symbol below is resolved
LAZILY (PEP 562), so touching `ilmarinen.legacy` (or the backward-compat aliases on ilmarinen / ilmarinen.models /
ilmarinen.core / ilmarinen.validation) does not eagerly drag in the whole island.

Contents:
  supergraphs : supergraph, parallel_supergraph, multi_supergraph, multi_parallel_supergraph, spatial_supergraph
  model zoo   : networks (build_model / MODEL_REGISTRY), recurrent (build_rnn / RNN_REGISTRY)
  pipeline    : training (train/eval utils), pipelines (validate_* proof-of-concept experiments)
"""

# legacy symbol -> submodule it lives in (resolved on first access)
_SYMBOL_MODULE = {
    "build_model": "networks", "MODEL_REGISTRY": "networks", "PlainMLP": "networks", "ResNetMLP": "networks",
    "build_rnn": "recurrent", "RNN_REGISTRY": "recurrent", "PlainRNN": "recurrent",
    "train_and_eval": "training", "train_and_eval_rnn": "training",
    "gradient_norms_at_init": "training", "to_tensor": "training",
    "build_supergraph": "supergraph", "SUPERGRAPH_REGISTRY": "supergraph", "SuperGraphRNN": "supergraph",
    "SuperCell": "supergraph", "DiscreteRNN": "supergraph", "discretize": "supergraph",
    "build_multi_supergraph": "multi_supergraph",
    "build_parallel_supergraph": "parallel_supergraph",
    "build_multi_parallel_supergraph": "multi_parallel_supergraph",
    "build_spatial_supergraph": "spatial_supergraph", "SpatialSuperGraph": "spatial_supergraph",
    "validate_width_sparsity": "pipelines", "validate_criticality": "pipelines",
    "validate_priced_depth": "pipelines", "validate_sequential_baseline": "pipelines",
    "validate_supergraph_copy": "pipelines",
}


def __getattr__(name):          # PEP 562: import the owning submodule only when the symbol is first used
    mod = _SYMBOL_MODULE.get(name)
    if mod is not None:
        import importlib
        return getattr(importlib.import_module(f".{mod}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_SYMBOL_MODULE)

# CURRENT schemas -- the eight peer contracts (see AllGraph._BUILTIN_CONTRACTS)
from .schema import build_schema, Schema
from .spatial_schema import build_spatial_schema, SpatialSchema
from .volumetric_schema import (build_volumetric_schema,
                                            VolumetricSchema)
from .graph_schema import build_graph_schema, GraphSchema
from .equivariant_graph_schema import (build_equivariant_graph_schema,
                                                   EquivariantGraphSchema)
from .equivariant_graph_schema_l2 import (build_equivariant_graph_schema_l2,
                                                      EquivariantGraphSchemaL2)
from .neural_ode import *  # noqa
from .set_schema import build_set_schema, SetSchema
from .grid4d_schema import build_grid4d_schema, Grid4dSchema, conv4d
from .operator_schema import (build_operator_schema,
                                          OperatorSchema,
                                          build_standalone_deeponet, StandaloneDeepONet)

# LEGACY re-exports (superseded by AllGraph + the per-contract build_*_schema contracts; consolidated under ilmarinen/legacy).
# DEPRECATED: resolved lazily with a DeprecationWarning, kept for one release for backward compatibility.
_LEGACY_ZOO = {  # the pre-AllGraph model zoo
    "build_model": "ilmarinen.legacy.networks", "MODEL_REGISTRY": "ilmarinen.legacy.networks",
    "PlainMLP": "ilmarinen.legacy.networks", "ResNetMLP": "ilmarinen.legacy.networks",
    "build_rnn": "ilmarinen.legacy.recurrent", "RNN_REGISTRY": "ilmarinen.legacy.recurrent",
    "PlainRNN": "ilmarinen.legacy.recurrent",
}
_LAZY_LEGACY = {"build_supergraph", "SUPERGRAPH_REGISTRY", "SuperGraphRNN", "SuperCell", "DiscreteRNN",
                "discretize", "build_multi_supergraph", "build_parallel_supergraph",
                "build_multi_parallel_supergraph", "build_spatial_supergraph", "SpatialSuperGraph"}


def __getattr__(name):
    from .._deprecation import warn_legacy
    if name in _LEGACY_ZOO:
        import importlib
        warn_legacy(name, _LEGACY_ZOO[name])
        return getattr(importlib.import_module(_LEGACY_ZOO[name]), name)
    if name in _LAZY_LEGACY:
        warn_legacy(name, "ilmarinen.legacy")
        from .. import legacy
        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LEGACY_ZOO) + list(_LAZY_LEGACY))

from . import equivariance_discovery, nonlinear_symmetry, qm9, rmd17
from .cifar import CIFAR10
from .data import FashionMNIST
from .discrete_symmetry import (
    discover_cyclic_dihedral,
    discover_permutation_subgroup,
    discover_z2,
    equivariance_error,
    scale_aware_equivariance_error,
)
from .equivariant_layer import EquivariantLayer
from .equivariant_supergraph import build_equivariant_supergraph
from .exponent import ExponentResult, critical_exponent, empirical_exponent
from .feature_attribution import feature_selection_path, fit_feature_attribution, format_attribution
from .interpretability import explain, format_report
from .meanfield import MeanFieldTheory, PhaseResult
from .mode_structure import discover_mode_structure, mutual_information_matrix
from .redundancy_reduction import effective_dimension, format_reduction, reduce_redundancy
from .route import route_by_structure
from .symbolic_readout import format_symbolic, symbolic_readout, symbolify_model
from .symmetry_discovery import discover_affine_symmetries, identify_generator
from .symmetry_pipeline import continuous_invariant_features, discover_and_reduce
from .variable_width_area import (
    VariableWidthNet,
    area_price_path,
    certificate_lambda_scale,
    fit_variable_width_area,
    format_area_result,
)

# training utils belong to the legacy validation pipeline (not AllGraph, which trains itself). Load them
# LAZILY (PEP 562) so `import ilmarinen` does not pull in legacy.training on the primary path.
_LAZY_TRAINING = {"train_and_eval", "train_and_eval_rnn", "gradient_norms_at_init", "to_tensor"}


def __getattr__(name):
    if name in _LAZY_TRAINING:
        from .._deprecation import warn_legacy
        warn_legacy(name, "ilmarinen.legacy.training")
        from ..legacy import training  # training utils moved to the quarantined legacy island
        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

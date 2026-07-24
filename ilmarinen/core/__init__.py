from .data import FashionMNIST
from .meanfield import MeanFieldTheory, PhaseResult
from .exponent import critical_exponent, empirical_exponent, ExponentResult
from .cifar import CIFAR10
from .mode_structure import discover_mode_structure, mutual_information_matrix
from .route import route_by_structure
from .symmetry_discovery import discover_affine_symmetries, identify_generator
from .discrete_symmetry import (discover_z2, discover_permutation_subgroup,
                                discover_cyclic_dihedral, scale_aware_equivariance_error,
                                equivariance_error)
from .interpretability import explain, format_report
from .feature_attribution import fit_feature_attribution, feature_selection_path, format_attribution
from .symbolic_readout import symbolic_readout, symbolify_model, format_symbolic
from .redundancy_reduction import effective_dimension, reduce_redundancy, format_reduction
from .variable_width_area import fit_variable_width_area, area_price_path, format_area_result, VariableWidthNet, certificate_lambda_scale
from .symmetry_pipeline import discover_and_reduce, continuous_invariant_features
from .equivariant_layer import EquivariantLayer
from .equivariant_supergraph import build_equivariant_supergraph
from . import equivariance_discovery
from . import nonlinear_symmetry
from . import qm9
from . import rmd17

# training utils belong to the legacy validation pipeline (not AllGraph, which trains itself). Load them
# LAZILY (PEP 562) so `import ilmarinen` does not pull in legacy.training on the primary path.
_LAZY_TRAINING = {"train_and_eval", "train_and_eval_rnn", "gradient_norms_at_init", "to_tensor"}


def __getattr__(name):
    if name in _LAZY_TRAINING:
        from .._deprecation import warn_legacy
        warn_legacy(name, "ilmarinen.legacy.training")
        from ..legacy import training      # training utils moved to the quarantined legacy island
        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

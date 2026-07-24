"""T-API: the public export surface exists and is well-formed.

Complements _selfcheck (which checks internal module imports) by pinning the *public* names
downstream users import, so a refactor cannot silently drop or break one.
"""

import re
import types
import warnings

import pytest

import ilmarinen

# Frozen at the current release. If you intentionally add/remove a public symbol, update this list
# in the same commit -- that is the point (the diff makes an API change explicit and reviewable).
EXPECTED_EXPORTS = {
    "ApproxEquivariantModel",
    "FashionMNIST",
    "MODEL_REGISTRY",
    "MeanFieldTheory",
    "AllData",
    "AllGraph",
    "RNN_REGISTRY",
    "ScalableEquivariantMLP",
    "VariableWidthNet",
    "area_price_path",
    "best_device",
    "build_latent_equivariant_contract",
    "build_model",
    "build_rnn",
    "build_scalable_equivariant_mlp",
    "certificate_lambda_scale",
    "column_generation_solve",
    "continuous_invariant_features",
    "contract_evidence",
    "critical_betas",
    "critical_exponent",
    "discover_and_reduce",
    "effective_dimension",
    "empirical_exponent",
    "estimate_llc",
    "explain",
    "feature_selection_path",
    "fit_feature_attribution",
    "fit_variable_width_area",
    "format_report",
    "free_energy",
    "gib_spectrum",
    "greedy_insertion",
    "ib_effective_dimension",
    "ib_rg_flow",
    "layer_rg_flow",
    "measure_depth_curve",
    "measure_mode_curve",
    "mps_available",
    "price_relaxation",
    "reduce_redundancy",
    "resolve_device",
    "route_by_structure",
    "score_to_nll",
    "select_depth",
    "select_modes",
    "select_relaxation",
    "significant_elbow",
    "spectral_code_length",
    "symbolic_readout",
    "symbolify_model",
    # D1 (singular MDL): functional code length lambda*log n fused into the pricing
    "omega_func",
    "total_code_length",
    "singular_complexity_of",
    "singular_free_energy",
}


def test_expected_exports_present():
    """T-API-1: every expected public name is present on the package."""
    missing = EXPECTED_EXPORTS - set(dir(ilmarinen))
    assert not missing, f"public API dropped these names: {sorted(missing)}"


def test_exports_are_callable_or_class():
    """T-API-2: functions are callable; the registry constants are the expected containers."""
    containers = {"MODEL_REGISTRY": dict, "RNN_REGISTRY": dict}
    for name in EXPECTED_EXPORTS:
        obj = getattr(ilmarinen, name)
        if name in containers:
            assert isinstance(obj, containers[name]), f"{name} should be {containers[name]}"
        else:
            assert callable(obj), f"{name} is not callable"


def test_submodules_importable():
    """T-API-2b: the advertised submodules are real modules."""
    for sub in ("core", "device", "machinery", "models", "validation"):
        assert isinstance(getattr(ilmarinen, sub), types.ModuleType), f"ilmarinen.{sub} missing"


def test_version_is_well_formed():
    """T-API-3: __version__ is a proper X.Y.Z string."""
    v = getattr(ilmarinen, "__version__", None)
    assert isinstance(v, str) and re.fullmatch(r"\d+\.\d+\.\d+", v), f"bad version: {v!r}"


# names that are the DEPRECATED legacy aliases (pre-AllGraph); still work but warn, pending removal
_DEPRECATED = [
    "build_model",
    "MODEL_REGISTRY",
    "build_rnn",
    "RNN_REGISTRY",
    "FashionMNIST",
    "MeanFieldTheory",
    "critical_exponent",
    "empirical_exponent",
    "greedy_insertion",
    "column_generation_solve",
]


def test_legacy_exports_warn_but_resolve():
    """T-API-4: each legacy alias emits a DeprecationWarning yet still resolves (one-release cycle)."""
    for name in _DEPRECATED:
        with pytest.warns(DeprecationWarning):
            obj = getattr(ilmarinen, name)
        assert obj is not None
    # aliased re-exports on the subpackages warn too
    for mod, name in [(ilmarinen.models, "build_supergraph"), (ilmarinen.core, "train_and_eval")]:
        with pytest.warns(DeprecationWarning):
            getattr(mod, name)


def test_legacy_home_does_not_warn():
    """T-API-4b: the quarantined ilmarinen.legacy home is the supported access point and does NOT warn."""
    import ilmarinen.legacy as legacy

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # any DeprecationWarning here would fail the test
        assert callable(legacy.build_supergraph)
        assert callable(legacy.build_model)

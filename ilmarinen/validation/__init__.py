"""Validation pipelines -- the pre-AllGraph proof-of-concept experiments.

Imported LAZILY (PEP 562): `import ilmarinen` binds this subpackage but does NOT pull in the legacy pipeline
(now consolidated under ilmarinen/legacy/: legacy.pipelines -> legacy.training -> legacy schemas). The
primary interface is AllGraph; these `validate_*` helpers load on first access, e.g.
`ilmarinen.validation.validate_width_sparsity(...)`.
"""

_LAZY = {
    "validate_width_sparsity", "validate_criticality", "validate_priced_depth",
    "validate_sequential_baseline", "validate_supergraph_copy",
}


def __getattr__(name):          # PEP 562: defer the heavy pipeline import until a helper is actually used
    if name in _LAZY:
        from .._deprecation import warn_legacy
        warn_legacy(name, "ilmarinen.legacy.pipelines")
        from ..legacy import pipelines  # pipelines moved to the quarantined legacy island
        return getattr(pipelines, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_LAZY)

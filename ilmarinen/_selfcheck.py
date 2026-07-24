"""Integrity self-check: verifies all required modules and critical invariants are present.
Run: python -m ilmarinen._selfcheck"""

REQUIRED_CORE = [
    "data",
    "meanfield",
    "exponent",
    "cifar",
    "qm7",
    "qm9",
    "rmd17",
    "jetnet",
    "benchmark_datasets",
    "moleculenet",
    "allgraph",
    "allgraph_types",
    "allgraph_reports",
    "allgraph_persistence",
    "allgraph_selection",
    "allgraph_contracts",
    "dataset_registry",
    "mode_structure",
    "route",
    "symmetry_discovery",
    "discrete_symmetry",
    "symmetry_pipeline",
    "equivariant_layer",
    "equivariant_supergraph",
    "correlation_length",
    "angular_resolution",
    "equivariance_discovery",
    "nonlinear_symmetry",
    "symmetry_contract",
    "canonicalization",
    "emlp_layer",
    "metric_discovery",
    "extended_groups",
    "interpretability",
    "feature_attribution",
    "symbolic_readout",
    "redundancy_reduction",
    "ib_rg_flow",
    "extended_datasets",
    "data_sources",
    "variable_width_area",
]
REQUIRED_MODELS = [
    "init_utils",
    "schema",
    "spatial_schema",
    "volumetric_schema",
    "graph_schema",
    "set_schema",
    "grid4d_schema",
    "equivariant_graph_schema",
    "equivariant_graph_schema_l2",
    "operator_schema",
    "latent_equivariant_contract",
    "scalable_equivariant",
    "approximate_equivariance",
    "neural_ode",
]
# legacy island consolidated under ilmarinen/legacy/ (supergraphs + the pre-AllGraph model zoo + pipeline)
REQUIRED_LEGACY = [
    "supergraph",
    "multi_supergraph",
    "parallel_supergraph",
    "multi_parallel_supergraph",
    "spatial_supergraph",
    "networks",
    "recurrent",
    "training",
    "pipelines",
]
REQUIRED_MACHINERY = [
    "priced_structural",
    "joint_arch_search",
    "price_selection",
    "compile_arch",
    "coadapt",
    "selection_uncertainty",
    "robust_discretize",
    "gibbs_alpha",
    "sparsity_priced_alpha",
    "contract_mdl",
    "contract_evidence",
    "singular_complexity",
    "singular_mdl",
    "developmental_llc",
    "thermodynamic_potential",
    "response_spectroscopy",
    "effective_dimension_ledger",
    "spectral_selection",
    "learned_contract_router",
]
REQUIRED_PRIMITIVES = {
    "plain",
    "gated",
    "lstm",
    "conv",
    "dilconv",
    "attention",
    "dense",
    "norm",
    "spectral",
    "linssm",
    "selssm",
}
# joint-search adapters (one per supergraph contract) + the mu-automation selectors
REQUIRED_JOINT = {
    "JointArchServer",
    "SpatialJointServer",
    "VolumetricJointServer",
    "GraphJointServer",
    "EquivariantJointServer",
    "SetJointServer",
    "Grid4dJointServer",
    "DifferentiableReadout",
    "auto_gate_init",
    "joint_search",
    "joint_search_generic",
    "joint_search_graph",
}
REQUIRED_MU = {"select_mu_for_budget", "select_by_tolerance", "select_mu_by_elbow", "select_mu_by_validation"}


def run():
    import importlib

    ok = True
    for m in REQUIRED_CORE:
        try:
            importlib.import_module(f"ilmarinen.core.{m}")
        except Exception as e:
            print(f"MISSING core/{m}: {e}")
            ok = False
    for m in REQUIRED_MODELS:
        try:
            importlib.import_module(f"ilmarinen.models.{m}")
        except Exception as e:
            print(f"MISSING models/{m}: {e}")
            ok = False
    for m in REQUIRED_LEGACY:
        try:
            importlib.import_module(f"ilmarinen.legacy.{m}")
        except Exception as e:
            print(f"MISSING legacy/{m}: {e}")
            ok = False
    try:
        pass
    except Exception as e:
        print(f"MISSING gibbs_alpha exports: {e}")
        ok = False
    for m in REQUIRED_MACHINERY:
        try:
            importlib.import_module(f"ilmarinen.machinery.{m}")
        except Exception as e:
            print(f"MISSING machinery/{m}: {e}")
            ok = False
    try:
        from ilmarinen.models.schema import _SEQ_CORES

        missing = REQUIRED_PRIMITIVES - set(_SEQ_CORES)
        if missing:
            print(f"schema MISSING primitives: {missing}")
            ok = False
    except Exception as e:
        print(f"cannot check primitives: {e}")
        ok = False
    try:
        import ilmarinen.machinery.joint_arch_search as J

        miss_j = REQUIRED_JOINT - set(dir(J))
        if miss_j:
            print(f"joint_arch_search MISSING: {miss_j}")
            ok = False
    except Exception as e:
        print(f"cannot check joint search: {e}")
        ok = False
    try:
        import ilmarinen.machinery.price_selection as P

        miss_mu = REQUIRED_MU - set(dir(P))
        if miss_mu:
            print(f"price_selection MISSING: {miss_mu}")
            ok = False
    except Exception as e:
        print(f"cannot check mu selection: {e}")
        ok = False
    # device helper + public API surface (deployability): the package must expose AllGraph/AllData and the
    # device auto-detection entry points at the top level.
    try:
        from ilmarinen.device import best_device, mps_available, resolve_device  # noqa: F401
    except Exception as e:
        print(f"MISSING device module: {e}")
        ok = False
    try:
        import ilmarinen as _mo

        for name in ("AllGraph", "AllData", "best_device", "resolve_device"):
            if not hasattr(_mo, name):
                print(f"top-level API MISSING: ilmarinen.{name}")
                ok = False
    except Exception as e:
        print(f"cannot check top-level API: {e}")
        ok = False
    print("INTEGRITY OK: all modules and primitives present" if ok else "INTEGRITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(run())

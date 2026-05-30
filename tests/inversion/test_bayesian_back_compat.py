"""Back-compat smoke for the T1-6 Bayesian decomposition.

Ensures every name previously exported from the legacy
``cflibs.inversion.solve.bayesian`` flat shim resolves identically after the
``cflibs/inversion/solve/bayesian.py`` -> ``bayesian/`` package split.
"""

from __future__ import annotations

import importlib

LEGACY_NAMES = [
    # Forward models
    "BayesianForwardModel",
    "TwoZoneBayesianForwardModel",
    # Samplers
    "MCMCSampler",
    "NestedSampler",
    "TwoZoneMCMCSampler",
    # Result containers
    "MCMCResult",
    "NestedSamplingResult",
    "TwoZoneMCMCResult",
    "ConvergenceStatus",
    # Configs
    "PriorConfig",
    "TwoZonePriorConfig",
    "NoiseParameters",
    # Atomic / utility
    "AtomicDataArrays",
    "load_atomic_data",
    "partition_function",
    "mcwhirter_log_penalty",
    # Likelihood and graph builders
    "log_likelihood",
    "bayesian_model",
    "two_zone_bayesian_model",
    "run_mcmc",
    # Prior factories
    "create_temperature_prior",
    "create_density_prior",
    "create_concentration_prior",
    # Optional-deps flags
    "HAS_JAX",
    "HAS_NUMPYRO",
    "HAS_DYNESTY",
    "HAS_ARVIZ",
]


def test_legacy_names_resolve_through_flat_shim():
    """``from cflibs.inversion.solve.bayesian import X`` works for every legacy name."""
    mod = importlib.import_module("cflibs.inversion.solve.bayesian")
    missing = [name for name in LEGACY_NAMES if not hasattr(mod, name)]
    assert not missing, f"Missing legacy names on flat shim: {missing}"


def test_legacy_names_resolve_through_package():
    """``from cflibs.inversion.solve.bayesian import X`` works for every legacy name."""
    mod = importlib.import_module("cflibs.inversion.solve.bayesian")
    missing = [name for name in LEGACY_NAMES if not hasattr(mod, name)]
    assert not missing, f"Missing legacy names on solve.bayesian package: {missing}"


def test_flat_and_package_paths_agree():
    """Same object identity between flat shim and package re-export."""
    flat = importlib.import_module("cflibs.inversion.solve.bayesian")
    pkg = importlib.import_module("cflibs.inversion.solve.bayesian")
    for name in LEGACY_NAMES:
        assert getattr(flat, name) is getattr(pkg, name), f"Identity mismatch for {name!r}"


def test_star_import_succeeds():
    """``from cflibs.inversion.solve.bayesian import *`` exposes a representative set."""
    ns: dict[str, object] = {}
    exec(  # noqa: S102 - intentional: test star-import surface
        "from cflibs.inversion.solve.bayesian import *", ns
    )
    sample = {"BayesianForwardModel", "MCMCResult", "NoiseParameters"}
    assert sample.issubset(ns.keys()), f"Star import missing: {sample - ns.keys()}"


def test_cli_imports_unchanged():
    """``cflibs.cli.main`` imports the Bayesian symbols at module level."""
    cli_main = importlib.import_module("cflibs.cli.main")
    assert cli_main is not None


def test_exporters_import_unchanged():
    """``cflibs.io.exporters`` imports ``MCMCResult`` / ``NestedSamplingResult`` at top level."""
    exporters = importlib.import_module("cflibs.io.exporters")
    assert hasattr(exporters, "MCMCResult")
    assert hasattr(exporters, "NestedSamplingResult")


def test_new_api_alongside_legacy():
    """The new API surface (Sampler, SamplerResult, Parameter, ...) coexists."""
    pkg = importlib.import_module("cflibs.inversion.solve.bayesian")
    for name in (
        "Sampler",
        "SamplerResult",
        "Parameter",
        "NumPyroNUTSSampler",
        "DynestyNestedSampler",
    ):
        assert hasattr(pkg, name), f"New API name missing: {name}"

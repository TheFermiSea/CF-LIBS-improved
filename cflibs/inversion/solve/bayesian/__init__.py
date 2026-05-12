"""Bayesian inference sub-package (T1-6 decomposition of legacy bayesian.py).

The legacy 3264-LOC ``cflibs/inversion/solve/bayesian.py`` monolith is split
along Bayesian-inference axes:

* :mod:`priors` -- declarative parameter spec (``Parameter``), prior configs,
  noise parameters, and convergence-status enum.
* :mod:`atomic` -- per-line atomic-data carrier (``AtomicDataArrays``),
  SQLite query helpers, partition function, McWhirter penalty.
* :mod:`forward` -- single-zone and two-zone JAX forward models, plus the
  NumPyro graph builders ``bayesian_model`` / ``two_zone_bayesian_model``.
* :mod:`results` -- ``MCMCResult`` / ``NestedSamplingResult`` /
  ``TwoZoneMCMCResult`` dataclasses.
* :mod:`samplers` -- ``Sampler`` Protocol + ``SamplerResult`` envelope plus
  the NumPyro NUTS and dynesty nested-sampling backends. Legacy class names
  ``MCMCSampler`` / ``NestedSampler`` are preserved alongside the new
  ``NumPyroNUTSSampler`` / ``DynestyNestedSampler`` aliases.

This ``__init__`` re-exports every name that used to live in the monolith so
``from cflibs.inversion.bayesian import *`` and every concrete import path
recorded in tests + downstream packages keeps working unchanged.
"""

from .atomic import (
    STANDARD_MASSES,
    AtomicDataArrays,
    _atomic_data_arrays_from_snapshot,
    _compute_instrument_sigma,
    _resolve_total_species_density_cm3,
    load_atomic_data,
    mcwhirter_log_penalty,
    partition_function,
)
from .forward import (
    BayesianForwardModel,
    TwoZoneBayesianForwardModel,
    bayesian_model,
    log_likelihood,
    two_zone_bayesian_model,
)
from .priors import (
    HAS_JAX,
    HAS_NUMPYRO,
    ConvergenceStatus,
    NoiseParameters,
    Parameter,
    PriorConfig,
    PriorKind,
    TwoZonePriorConfig,
    create_concentration_prior,
    create_density_prior,
    create_temperature_prior,
)
from .results import MCMCResult, NestedSamplingResult, TwoZoneMCMCResult
from .samplers import (
    HAS_ARVIZ,
    HAS_DYNESTY,
    DynestyNestedSampler,
    MCMCSampler,
    NestedSampler,
    NumPyroNUTSSampler,
    Sampler,
    SamplerResult,
    run_mcmc,
)
from .two_zone import TwoZoneMCMCSampler

__all__ = [
    # Atomic / utility
    "STANDARD_MASSES",
    "AtomicDataArrays",
    "_atomic_data_arrays_from_snapshot",
    "_compute_instrument_sigma",
    "_resolve_total_species_density_cm3",
    "load_atomic_data",
    "partition_function",
    "mcwhirter_log_penalty",
    # Priors / configs
    "ConvergenceStatus",
    "NoiseParameters",
    "Parameter",
    "PriorConfig",
    "PriorKind",
    "TwoZonePriorConfig",
    "create_concentration_prior",
    "create_density_prior",
    "create_temperature_prior",
    # Forward models + NumPyro graph builders
    "BayesianForwardModel",
    "TwoZoneBayesianForwardModel",
    "bayesian_model",
    "log_likelihood",
    "two_zone_bayesian_model",
    # Results
    "MCMCResult",
    "NestedSamplingResult",
    "TwoZoneMCMCResult",
    # Samplers + Protocol
    "DynestyNestedSampler",
    "MCMCSampler",
    "NestedSampler",
    "NumPyroNUTSSampler",
    "Sampler",
    "SamplerResult",
    "TwoZoneMCMCSampler",
    "run_mcmc",
    # Optional-deps flags
    "HAS_ARVIZ",
    "HAS_DYNESTY",
    "HAS_JAX",
    "HAS_NUMPYRO",
]

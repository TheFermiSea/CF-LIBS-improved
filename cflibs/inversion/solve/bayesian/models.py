"""NumPyro probabilistic-model graph builders for CF-LIBS Bayesian inference (T1-6).

Hosts :func:`bayesian_model` (single-zone) and :func:`two_zone_bayesian_model`
(core+shell self-reversed). Each wires a JAX forward model from :mod:`forward`
together with the priors from :mod:`priors` and a per-pixel likelihood.

Likelihood choice (single-zone):

* ``"gaussian"`` (legacy) folds the model prediction into the per-pixel Gaussian
  variance -- Pearson-style, biased low on peaks.
* ``"poisson"`` uses the exact Poisson likelihood ``dist.Poisson(mu)`` for the
  shot term (``mu = predicted / gain + dark_current``), the unbiased
  source-recommended choice for shot-noise-dominated spectra.

Separated from :mod:`forward` so every ``bayesian/*.py`` file stays under the
800-LOC limit required by ADR-0001 / T1-6 spec section 6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .atomic import mcwhirter_log_penalty
from .priors import (
    HAS_JAX,
    HAS_NUMPYRO,
    NoiseParameters,
    PriorConfig,
    TwoZonePriorConfig,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .forward import BayesianForwardModel, TwoZoneBayesianForwardModel

if HAS_JAX:
    import jax.numpy as jnp
else:  # pragma: no cover - JAX not installed
    jnp = None  # type: ignore[assignment]

if HAS_NUMPYRO:
    import numpyro
    import numpyro.distributions as dist
else:  # pragma: no cover
    numpyro = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]


def bayesian_model(
    forward_model: "BayesianForwardModel",
    observed,
    prior_config: PriorConfig = PriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
    likelihood_kind: str = "gaussian",
):
    """NumPyro probabilistic model for single-zone CF-LIBS Bayesian inference.

    Parameters
    ----------
    likelihood_kind : {"gaussian", "poisson"}, default "gaussian"
        ``"gaussian"`` (legacy) folds the model prediction into the per-pixel
        Gaussian variance (Pearson-style, biased low on peaks). ``"poisson"``
        uses the exact Poisson likelihood ``dist.Poisson(mu)`` for the shot
        term (``mu = predicted / gain + dark_current``), the unbiased
        source-recommended choice for shot-noise-dominated spectra.
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required. Install with: pip install numpyro")
    if likelihood_kind not in ("gaussian", "poisson"):
        raise ValueError(
            f"likelihood_kind must be 'gaussian' or 'poisson', got {likelihood_kind!r}"
        )

    n_elements = len(forward_model.elements)

    T_eV = numpyro.sample(
        "T_eV",
        dist.Uniform(prior_config.T_eV_range[0], prior_config.T_eV_range[1]),
    )
    log_ne = numpyro.sample(
        "log_ne",
        dist.Uniform(prior_config.log_ne_range[0], prior_config.log_ne_range[1]),
    )

    alpha = jnp.ones(n_elements) * prior_config.concentration_alpha
    concentrations = numpyro.sample("concentrations", dist.Dirichlet(alpha))

    predicted = forward_model.forward(T_eV, log_ne, concentrations)

    if prior_config.baseline_degree > 0:
        if prior_config.baseline_degree > forward_model._max_baseline_degree:
            raise ValueError(
                f"baseline_degree={prior_config.baseline_degree} exceeds max "
                f"({forward_model._max_baseline_degree}). Pre-computed Chebyshev "
                f"basis does not cover this degree."
            )
        n_coeffs = prior_config.baseline_degree + 1
        baseline_scale = prior_config.baseline_scale
        if baseline_scale is not None and baseline_scale <= 0:
            raise ValueError(f"baseline_scale must be positive, got {baseline_scale}")
        if baseline_scale is None:
            baseline_scale = 0.1 * jnp.max(observed)
        baseline_coeffs = numpyro.sample(
            "baseline_coeffs",
            dist.Normal(jnp.zeros(n_coeffs), baseline_scale),
        )
        basis = forward_model._baseline_basis_jax[:, :n_coeffs]
        baseline = jnp.dot(basis, baseline_coeffs)
        predicted = predicted + baseline

    pred_safe = jnp.maximum(predicted, 1e-6)
    pred_safe = jnp.where(jnp.isnan(pred_safe), 1e-6, pred_safe)
    pred_safe = jnp.where(jnp.isinf(pred_safe), 1e6, pred_safe)

    if likelihood_kind == "poisson":
        # Exact Poisson shot-noise term (mu = predicted/gain + dark_current).
        # No model prediction in a Gaussian variance denominator => no Pearson
        # peak-amplitude bias.
        mu = jnp.maximum(pred_safe / noise_params.gain + noise_params.dark_current, 1e-6)
        numpyro.sample("obs", dist.Poisson(mu), obs=observed)
        return

    variance = (
        pred_safe / noise_params.gain + noise_params.readout_noise**2 + noise_params.dark_current
    )
    sigma = jnp.sqrt(jnp.maximum(variance, 1e-6))

    numpyro.sample("obs", dist.Normal(pred_safe, sigma), obs=observed)


def two_zone_bayesian_model(
    forward_model: "TwoZoneBayesianForwardModel",
    observed,
    prior_config: TwoZonePriorConfig = TwoZonePriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
):
    """NumPyro probabilistic model for two-zone CF-LIBS Bayesian inference."""
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required. Install with: pip install numpyro")

    n_elements = len(forward_model.elements)

    T_core_eV = numpyro.sample(
        "T_core_eV",
        dist.Uniform(prior_config.T_core_eV_range[0], prior_config.T_core_eV_range[1]),
    )
    T_shell_eV = numpyro.sample(
        "T_shell_eV",
        dist.Uniform(prior_config.T_shell_eV_range[0], prior_config.T_shell_eV_range[1]),
    )

    if prior_config.enforce_T_ordering:
        numpyro.factor("T_ordering", jnp.where(T_core_eV > T_shell_eV, 0.0, -1e6))

    log_ne = numpyro.sample(
        "log_ne",
        dist.Uniform(prior_config.log_ne_range[0], prior_config.log_ne_range[1]),
    )

    if prior_config.mcwhirter_penalty_scale > 0:
        penalty = mcwhirter_log_penalty(
            T_core_eV,
            log_ne,
            max_delta_E_eV=prior_config.max_delta_E_eV,
            scale=prior_config.mcwhirter_penalty_scale,
        )
        numpyro.factor("mcwhirter_lte", penalty)

    alpha = jnp.ones(n_elements) * prior_config.concentration_alpha
    concentrations = numpyro.sample("concentrations", dist.Dirichlet(alpha))

    shell_fraction = numpyro.sample(
        "shell_fraction",
        dist.Uniform(
            prior_config.shell_fraction_range[0],
            prior_config.shell_fraction_range[1],
        ),
    )
    optical_depth_scale = numpyro.sample(
        "optical_depth_scale",
        dist.Uniform(
            prior_config.optical_depth_scale_range[0],
            prior_config.optical_depth_scale_range[1],
        ),
    )

    predicted = forward_model.forward(
        T_core_eV, T_shell_eV, log_ne, concentrations, shell_fraction, optical_depth_scale
    )

    if prior_config.baseline_degree > 0:
        baseline_coeffs = numpyro.sample(
            "baseline_coeffs",
            dist.Normal(jnp.zeros(prior_config.baseline_degree + 1), 100.0),
        )
        wl = forward_model.wavelength
        wl_norm = (wl - wl[0]) / jnp.maximum(wl[-1] - wl[0], 1e-6)
        baseline = jnp.polyval(baseline_coeffs, wl_norm)
        predicted = predicted + baseline

    pred_safe = jnp.maximum(predicted, 1e-6)
    pred_safe = jnp.where(jnp.isnan(pred_safe), 1e-6, pred_safe)
    pred_safe = jnp.where(jnp.isinf(pred_safe), 1e6, pred_safe)

    variance = (
        pred_safe / noise_params.gain + noise_params.readout_noise**2 + noise_params.dark_current
    )
    sigma = jnp.sqrt(jnp.maximum(variance, 1e-6))

    numpyro.sample("obs", dist.Normal(pred_safe, sigma), obs=observed)


__all__ = [
    "bayesian_model",
    "two_zone_bayesian_model",
]

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


#: Floor on each Dirichlet alpha component. A zero nominal fraction would give
#: alpha=0 -- a degenerate Dirichlet that forces that element to exactly 0 and
#: forbids detecting an unexpected element/contaminant (Codex review). The floor
#: keeps such an element weakly sparse but detectable; it is far below the alphas
#: of present elements (c*x ~ O(1-50) for c~60), so it never perturbs them.
_ALPHA_FLOOR = 1e-2


def _concentration_dirichlet_alpha(prior_config, n_elements):
    """Dirichlet concentration ``alpha`` for the composition prior.

    Symmetric (``ones * concentration_alpha``) by default. When
    ``prior_config.nominal_mole_fracs`` is set (DED feedstock prior), the
    Dirichlet is centered on it: ``alpha = concentration_alpha * x_nom`` so the
    prior MEAN is (essentially) ``x_nom`` and the total concentration
    (``sum(alpha) = concentration_alpha``) sets the spread. A weak
    concentration (~50-80) keeps the posterior data-dominated, so the prior aids
    conditioning without ever pinning the estimate to nominal. Components are
    floored at :data:`_ALPHA_FLOOR` so a zero-nominal element is never hard-
    pinned to 0.
    """
    x_nom = getattr(prior_config, "nominal_mole_fracs", None)
    if x_nom is None:
        return jnp.ones(n_elements) * prior_config.concentration_alpha
    x = jnp.asarray(x_nom)
    if int(x.shape[0]) != int(n_elements):
        raise ValueError(
            f"nominal_mole_fracs length {int(x.shape[0])} != n_elements {int(n_elements)}"
        )
    x = x / jnp.sum(x)
    return jnp.maximum(prior_config.concentration_alpha * x, _ALPHA_FLOOR)


def bayesian_model(
    forward_model: "BayesianForwardModel",
    observed,
    prior_config: PriorConfig = PriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
    likelihood_kind: str = "gaussian",
    strict: bool = False,
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
    strict : bool, default False
        No-fallback mode. When ``False`` (production) a NaN/Inf forward-model
        prediction is silently sanitized (NaN->1e-6, Inf->1e6) so the sample
        gets a finite garbage likelihood and NUTS never sees the divergence.
        When ``True`` the NaN/Inf replacement is skipped so the non-finite value
        propagates into the likelihood and NUTS marks the transition diverging
        (surfaced via ``extra_fields=('diverging',)``). Byte-identical default.
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

    alpha = _concentration_dirichlet_alpha(prior_config, n_elements)
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
    if not strict:
        # Production sanitization (masks forward-model divergence). In strict
        # mode the NaN/Inf are left to propagate so NUTS marks the transition.
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


#: Stiffness (per eV^2) of the smooth T_core > T_shell ordering penalty in the
#: two-zone model. Large enough to strongly enforce ordering while keeping the
#: potential C^1 (HMC-differentiable); approaches a hard truncation as it grows.
_T_ORDERING_PENALTY_SCALE = 1.0e4


def two_zone_bayesian_model(
    forward_model: "TwoZoneBayesianForwardModel",
    observed,
    prior_config: TwoZonePriorConfig = TwoZonePriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
    strict: bool = False,
):
    """NumPyro probabilistic model for two-zone CF-LIBS Bayesian inference.

    ``strict`` (default ``False``) mirrors :func:`bayesian_model`: when ``True``
    the NaN/Inf prediction sanitization is skipped so forward-model divergence
    reaches the likelihood and NUTS marks the transition. Byte-identical default.
    """
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
        # Smooth one-sided penalty enforcing T_core > T_shell. The previous
        # ``jnp.where(T_core>T_shell, 0.0, -1e6)`` is a flat cliff: ZERO gradient
        # on both sides plus a discontinuity at the boundary, so NUTS gets no
        # restoring force and diverges when a trajectory crosses it (audit C6).
        # A quadratic hinge ``-k*max(T_shell-T_core,0)^2`` is C^1 with a genuine
        # restoring gradient ``2k*(T_shell-T_core)`` in the violated region and
        # recovers the hard truncation as k grows, while staying differentiable.
        # (Reparameterizing T_core onto (T_shell, hi] is the tuning-free ideal,
        # but it changes the sample sites and the marginal T_shell prior, so it
        # is deferred to the Step-4 MCMC sampling-quality validation.)
        violation = jnp.maximum(T_shell_eV - T_core_eV, 0.0)
        numpyro.factor("T_ordering", -_T_ORDERING_PENALTY_SCALE * jnp.square(violation))

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

    alpha = _concentration_dirichlet_alpha(prior_config, n_elements)
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
    if not strict:
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

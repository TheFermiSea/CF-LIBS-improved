"""Spectrum log-likelihoods for the Bayesian forward model (T1-6).

Hosts the module-level :func:`log_likelihood` used by dynesty (and any caller
that wants a single scalar log-likelihood for an observed vs. predicted
spectrum) plus the exact Poisson (Cash-statistic) variant.

The two noise treatments differ in how they handle shot noise:

* ``"gaussian"`` (legacy / Pearson-style) folds the *model-predicted* intensity
  into a per-pixel Gaussian variance.  Because the model prediction sits in the
  variance denominator against the residual, the fitter is biased toward
  under-estimating peaks (the classic Pearson chi-square bias).
* ``"poisson"`` uses the EXACT Poisson (Cash 1979) shot-noise log-likelihood
  plus a constant-variance Gaussian readout term.  The Poisson term does NOT
  carry the model prediction in a denominator, so peak amplitudes are unbiased.

Separated from :mod:`forward` to keep every ``bayesian/*.py`` file under the
800-LOC limit required by ADR-0001 / T1-6 spec section 6.
"""

from __future__ import annotations

from typing import Any

from .atomic import logger  # noqa: F401  (re-exported for module parity)
from .priors import HAS_JAX, NoiseParameters

if HAS_JAX:
    import jax.numpy as jnp
else:  # pragma: no cover - JAX not installed
    jnp = None  # type: ignore[assignment]


def log_likelihood(
    predicted: Any,
    observed: Any,
    noise_params: NoiseParameters = NoiseParameters(),
    likelihood_kind: str = "gaussian",
) -> float:
    """Compute log-likelihood for an observed spectrum given a predicted one.

    Parameters
    ----------
    predicted, observed : array
        Model-predicted and observed spectra (counts).
    noise_params : NoiseParameters
        Detector-noise hyperparameters.
    likelihood_kind : {"gaussian", "poisson"}, default "gaussian"
        ``"gaussian"`` (legacy / Pearson-style) folds the *model-predicted*
        intensity into a per-pixel Gaussian variance
        ``variance = predicted / gain + readout_noise**2 + dark_current``.
        Putting the model prediction in the denominator biases the fitter
        toward under-estimating peaks (the classic "Pearson" chi-square bias).

        ``"poisson"`` uses the EXACT Poisson (Cash-statistic) log-likelihood for
        the shot-noise term plus a constant Gaussian readout term. The expected
        photon count is ``mu = predicted / gain + dark_current`` and the shot
        term is ``sum(n * ln(mu) - mu - ln Gamma(n + 1))`` (Cash 1979), which is
        unbiased for peak amplitudes; the readout adds the constant-variance
        Gaussian ``-0.5 * residual**2 / readout_noise**2`` so the readout floor
        is still penalised. The Poisson branch is the source-recommended best
        for shot-noise-dominated LIBS spectra.

    Notes
    -----
    The Gaussian branch is retained as the default to avoid silently changing
    existing posteriors; callers that want the unbiased shot-noise treatment
    must opt in with ``likelihood_kind="poisson"``.
    """
    if likelihood_kind == "poisson":
        return _poisson_cash_log_likelihood(predicted, observed, noise_params)
    if likelihood_kind != "gaussian":
        raise ValueError(
            f"likelihood_kind must be 'gaussian' or 'poisson', got {likelihood_kind!r}"
        )
    pred_safe = jnp.maximum(predicted, 1e-10)
    variance = (
        pred_safe / noise_params.gain + noise_params.readout_noise**2 + noise_params.dark_current
    )
    residual = observed - pred_safe
    log_lik = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance) + residual**2 / variance)
    return log_lik


def _poisson_cash_log_likelihood(
    predicted: Any,
    observed: Any,
    noise_params: NoiseParameters,
) -> float:
    """Exact Poisson (Cash) shot-noise log-likelihood + constant Gaussian readout.

    Shot term (Cash 1979): ``sum(n * ln(mu) - mu - ln Gamma(n + 1))`` with the
    expected count ``mu = predicted / gain + dark_current``. Unlike a Gaussian
    whose variance carries the model prediction, the Poisson term does NOT put
    ``mu`` in a denominator against the residual, so it does not bias peak
    amplitudes downward. A constant-variance Gaussian readout term
    ``-0.5 * residual**2 / readout_noise**2`` (variance independent of the
    model) is added so detector readout noise is still accounted for.
    """
    from jax.scipy.special import gammaln  # noqa: PLC0415

    mu = jnp.maximum(predicted / noise_params.gain + noise_params.dark_current, 1e-10)
    n = jnp.maximum(observed, 0.0)
    shot = jnp.sum(n * jnp.log(mu) - mu - gammaln(n + 1.0))
    readout_var = noise_params.readout_noise**2
    if readout_var > 0.0:
        residual = observed - mu
        readout = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * readout_var) + residual**2 / readout_var)
    else:
        readout = 0.0
    return shot + readout


__all__ = [
    "log_likelihood",
    "_poisson_cash_log_likelihood",
]

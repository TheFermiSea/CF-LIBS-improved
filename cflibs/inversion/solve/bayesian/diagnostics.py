"""Plot and posterior-predictive helpers for Bayesian CF-LIBS inference (T1-6).

These were previously methods on :class:`MCMCSampler` but moved out to keep
:mod:`samplers` under the 800-LOC limit imposed by ADR-0001 / T1-6 spec
section 6. They are exposed as module-level functions and the legacy
:class:`MCMCSampler` continues to expose them as thin method shims.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger

from .results import MCMCResult

logger = get_logger("inversion.bayesian.diagnostics")

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:  # pragma: no cover
    HAS_ARVIZ = False
    az = None  # type: ignore[assignment]


def plot_trace(result: MCMCResult, figsize: Tuple[int, int] = (12, 8)) -> Any:
    """Generate trace plot using ArviZ."""
    if not HAS_ARVIZ or result.inference_data is None:
        logger.warning("ArviZ trace plot unavailable")
        return None
    try:
        return az.plot_trace(result.inference_data, var_names=["T_eV", "log_ne"], figsize=figsize)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Trace plot failed: {e}")
        return None


def plot_posterior(result: MCMCResult, figsize: Tuple[int, int] = (12, 6)) -> Any:
    """Generate posterior distribution plot using ArviZ."""
    if not HAS_ARVIZ or result.inference_data is None:
        logger.warning("ArviZ posterior plot unavailable")
        return None
    try:
        return az.plot_posterior(
            result.inference_data,
            var_names=["T_eV", "log_ne"],
            figsize=figsize,
            hdi_prob=0.95,
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"Posterior plot failed: {e}")
        return None


def plot_corner(
    result: MCMCResult,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> Any:
    """Generate corner / pair plot showing parameter correlations."""
    if not HAS_ARVIZ or result.inference_data is None:
        logger.warning("ArviZ corner plot unavailable")
        return None
    if var_names is None:
        var_names = ["T_eV", "log_ne"]
    try:
        return az.plot_pair(
            result.inference_data,
            var_names=var_names,
            kind="kde",
            marginals=True,
            figsize=figsize,
            textsize=10,
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"Corner plot failed: {e}")
        return None


def plot_forest(result: MCMCResult, figsize: Tuple[int, int] = (10, 6)) -> Any:
    """Generate forest plot comparing parameter estimates."""
    if not HAS_ARVIZ or result.inference_data is None:
        logger.warning("ArviZ forest plot unavailable")
        return None
    try:
        return az.plot_forest(
            result.inference_data,
            var_names=["T_eV", "log_ne"],
            combined=True,
            figsize=figsize,
            hdi_prob=0.95,
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"Forest plot failed: {e}")
        return None


def posterior_predictive_check(
    forward_model: Any,
    noise_params: Any,
    elements: List[str],
    result: MCMCResult,
    observed: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform posterior predictive check for model validation.

    Compares posterior-drawn synthetic spectra against the observed spectrum
    using a Bayesian chi-squared p-value (the proportion of simulated
    chi-squared statistics that exceed the observed value).

    Parameters
    ----------
    forward_model : BayesianForwardModel
        Forward model whose ``forward_numpy`` is used to generate predictions.
    noise_params : NoiseParameters
        Detector noise parameters (used for the variance model).
    elements : list of str
        Element ordering (matches ``result.samples['concentrations']`` axis).
    result : MCMCResult
        Posterior samples.
    observed : np.ndarray
        Observed spectrum.
    n_samples : int
        Number of posterior samples to use for predictive draws.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Predictive mean / std, residuals, chi-squared diagnostics, and the
        Bayesian p-value.
    """
    rng = np.random.default_rng(seed)

    T_samples = np.array(result.samples["T_eV"]).flatten()
    log_ne_samples = np.array(result.samples["log_ne"]).flatten()
    conc_samples = np.array(result.samples["concentrations"]).reshape(-1, len(elements))

    n_available = len(T_samples)
    n_use = min(n_samples, n_available)
    indices = rng.choice(n_available, size=n_use, replace=False)

    predictions = []
    for idx in indices:
        T_eV = float(T_samples[idx])
        log_ne = float(log_ne_samples[idx])
        conc = conc_samples[idx]
        pred = forward_model.forward_numpy(T_eV, log_ne, conc)
        predictions.append(pred)

    predictions = np.array(predictions)
    predicted_mean = np.mean(predictions, axis=0)
    predicted_std = np.std(predictions, axis=0)
    residuals = observed - predicted_mean

    variance = (
        np.abs(predicted_mean) / noise_params.gain
        + noise_params.readout_noise**2
        + noise_params.dark_current
    )
    variance = np.maximum(variance, 1e-6)
    chi_sq_obs = np.sum(residuals**2 / variance)

    chi_sq_sim = []
    for pred in predictions:
        noise_std = np.sqrt(variance)
        simulated = pred + rng.normal(0, noise_std)
        chi_sq = np.sum((simulated - pred) ** 2 / variance)
        chi_sq_sim.append(chi_sq)

    chi_sq_sim = np.array(chi_sq_sim)
    p_value = float(np.mean(chi_sq_sim >= chi_sq_obs))

    return {
        "predicted_mean": predicted_mean,
        "predicted_std": predicted_std,
        "residuals": residuals,
        "chi_squared_obs": float(chi_sq_obs),
        "chi_squared_sim": chi_sq_sim,
        "p_value": p_value,
        "model_adequate": 0.05 < p_value < 0.95,
        "n_samples_used": n_use,
    }


__all__ = [
    "plot_trace",
    "plot_posterior",
    "plot_corner",
    "plot_forest",
    "posterior_predictive_check",
    "HAS_ARVIZ",
]


# ---------------------------------------------------------------------------
# Convergence diagnostics (split-R-hat / autocorrelation-corrected ESS)
# moved here from samplers.py to respect the 800-LOC cap (ADR-0001 / T1-6).
# ---------------------------------------------------------------------------


def _chain_diagnostics(arr: np.ndarray) -> Tuple[float, float]:
    """Return ``(split_rhat, ess)`` for a ``(num_chains, num_draws)`` array.

    Prefers ArviZ (``az.rhat`` / ``az.ess``, which use rank-normalised
    split-R-hat and autocorrelation-corrected ESS). When only a single chain
    is present, the chain is split into two contiguous halves so a real
    split-R-hat is still defined. Falls back to a hand-rolled FFT
    autocorrelation ESS and manual split-R-hat when ArviZ is unavailable.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]

    # Single chain -> split into two contiguous halves so between-"chain"
    # variance (split-R-hat) is well defined instead of fabricating 1.0.
    if arr.shape[0] < 2:
        flat = arr.reshape(-1)
        half = flat.shape[0] // 2
        if half < 2:  # too few draws to diagnose anything meaningful
            return float("nan"), float(flat.shape[0])
        arr = np.stack([flat[:half], flat[half : 2 * half]])

    if HAS_ARVIZ:
        try:
            return float(az.rhat(arr)), float(az.ess(arr))
        except Exception as e:  # pragma: no cover
            logger.warning(f"ArviZ diagnostics failed, using numpy fallback: {e}")

    return _split_rhat_numpy(arr), _ess_numpy(arr)


def _split_rhat_numpy(arr: np.ndarray) -> float:
    """Split-R-hat for a ``(num_chains, num_draws)`` array (BDA3 §11.4).

    Each chain is split into two halves before computing between- and
    within-half variances, so even a single long chain yields a meaningful
    diagnostic. Returns ``nan`` when too few draws are available.
    """
    m_chains, n_draws = arr.shape
    half = n_draws // 2
    if half < 2:
        return float("nan")
    # Split each chain into two halves -> 2*m sub-chains of length ``half``.
    splits = np.concatenate([arr[:, :half], arr[:, half : 2 * half]], axis=0)
    n = half

    chain_means = splits.mean(axis=1)
    chain_vars = splits.var(axis=1, ddof=1)

    b = n * np.var(chain_means, ddof=1)  # between-chain variance
    w = chain_vars.mean()  # within-chain variance
    if w <= 0:
        return float("nan")

    var_hat = (n - 1) / n * w + b / n
    rhat = np.sqrt(var_hat / w)
    return float(rhat)


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of a 1D series via FFT (lag 0 = 1.0)."""
    x = x - x.mean()
    n = x.shape[0]
    # Zero-pad to next power of two >= 2n for linear (non-circular) correlation.
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n]
    if acf[0] == 0:
        return np.zeros(n)
    return acf / acf[0]


def _ess_numpy(arr: np.ndarray) -> float:
    """Autocorrelation-corrected ESS for ``(num_chains, num_draws)`` (Geyer IPS).

    Uses Geyer's initial-positive-sequence estimator on the mean
    autocorrelation across chains. Never returns the raw sample count for an
    autocorrelated series.
    """
    m_chains, n_draws = arr.shape
    if n_draws < 4:
        return float(m_chains * n_draws)

    # Mean autocorrelation across chains.
    acfs = np.array([_autocorr_fft(arr[c]) for c in range(m_chains)])
    rho = acfs.mean(axis=0)

    # Geyer initial-positive-sequence: sum adjacent pairs while positive.
    tau = 1.0
    t = 1
    while t + 1 < n_draws:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
        t += 2
    if tau < 1.0:
        tau = 1.0
    ess = m_chains * n_draws / tau
    return float(ess)

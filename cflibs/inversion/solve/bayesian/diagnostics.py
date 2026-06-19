"""Plot and posterior-predictive helpers for Bayesian CF-LIBS inference (T1-6).

These were previously methods on :class:`MCMCSampler` but moved out to keep
:mod:`samplers` under the 800-LOC limit imposed by ADR-0001 / T1-6 spec
section 6. They are exposed as module-level functions and the legacy
:class:`MCMCSampler` continues to expose them as thin method shims.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

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
    "SBCResult",
    "sbc_rank",
    "sbc_ranks",
    "sbc_uniformity_test",
    "run_sbc",
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


# ---------------------------------------------------------------------------
# Simulation-Based Calibration (SBC) rank-uniformity harness.
#
# Talts, Betancourt, Simpson, Vehtari & Gelman, "Validating Bayesian Inference
# Algorithms with Simulation-Based Calibration", arXiv:1804.06788 (2018).
#
# Self-consistency theorem (their Eq. 1-3): for any prior pi(theta) and
# likelihood pi(y|theta), if {theta_1, ..., theta_L} are L exact posterior
# draws given y ~ pi(y | theta_tilde) with theta_tilde ~ pi(theta), then the
# rank statistic
#
#     r = #{ l : f(theta_l) < f(theta_tilde) }   in {0, 1, ..., L}
#
# of the prior draw within the posterior draws (for any one-dimensional test
# quantity f) is *uniformly* distributed on the L+1 integers {0, ..., L}. A
# correct sampler therefore yields flat rank histograms; systematic deviations
# diagnose miscalibration:
#   - U / bathtub shape (mass at 0 and L)  -> posterior too NARROW (overconfident)
#   - inverted-U / dome shape              -> posterior too WIDE (underconfident)
#   - monotone slope                       -> biased posterior (location shift)
# This module is physics-only: numpy + scipy.stats (chi-square GoF) only.
# ---------------------------------------------------------------------------


class SBCResult:
    """Container for the outcome of a Simulation-Based Calibration run.

    Bundles the per-parameter rank statistics with the chi-square uniformity
    test results from :func:`sbc_uniformity_test`, following Talts et al.
    (arXiv:1804.06788).

    Parameters
    ----------
    ranks : dict of str -> np.ndarray
        Per-parameter integer rank statistics, each of shape ``(n_sims,)`` with
        values in ``{0, ..., n_posterior_draws}``.
    n_posterior_draws : int
        Number of posterior draws ``L`` used per simulation. Ranks therefore
        take ``L + 1`` distinct values.
    p_values : dict of str -> float
        Per-parameter chi-square uniformity p-value (large = consistent with a
        uniform / well-calibrated rank distribution).
    chi2 : dict of str -> float
        Per-parameter chi-square goodness-of-fit statistic.
    n_bins : int
        Number of histogram bins used for the chi-square test.

    References
    ----------
    .. [1] Talts, Betancourt, Simpson, Vehtari & Gelman, "Validating Bayesian
       Inference Algorithms with Simulation-Based Calibration",
       arXiv:1804.06788 (2018).
    """

    def __init__(
        self,
        ranks: Dict[str, np.ndarray],
        n_posterior_draws: int,
        p_values: Dict[str, float],
        chi2: Dict[str, float],
        n_bins: int,
    ) -> None:
        self.ranks = ranks
        self.n_posterior_draws = n_posterior_draws
        self.p_values = p_values
        self.chi2 = chi2
        self.n_bins = n_bins

    @property
    def n_sims(self) -> int:
        """Number of SBC simulations (length of any rank array)."""
        if not self.ranks:
            return 0
        return int(len(next(iter(self.ranks.values()))))

    def is_calibrated(self, alpha: float = 0.05) -> Dict[str, bool]:
        """Per-parameter calibration verdict at significance level ``alpha``.

        A parameter is flagged calibrated when its uniformity p-value exceeds
        ``alpha`` (i.e. the rank histogram is *not* significantly non-uniform).

        Parameters
        ----------
        alpha : float, optional
            Significance threshold for the chi-square uniformity test.

        Returns
        -------
        dict of str -> bool
            ``True`` where the rank histogram is consistent with uniform.
        """
        return {name: (p > alpha) for name, p in self.p_values.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SBCResult(n_sims={self.n_sims}, L={self.n_posterior_draws}, "
            f"n_bins={self.n_bins}, params={list(self.ranks)})"
        )


def sbc_rank(
    prior_draw: float,
    posterior_draws: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Rank of a prior draw among posterior draws (Talts et al. Eq. 2).

    Computes ``r = #{ l : theta_l < theta_tilde }`` with ``theta_tilde`` the
    prior draw and ``{theta_l}`` the ``L`` posterior draws, giving an integer in
    ``{0, ..., L}``. Exact ties (which only arise for discrete test quantities
    or repeated draws) are broken by randomisation so that the rank remains
    uniform on ``{0, ..., L}`` under correct calibration, as recommended by
    Talts et al. (arXiv:1804.06788).

    Parameters
    ----------
    prior_draw : float
        The prior draw ``theta_tilde`` (or a test quantity ``f(theta_tilde)``).
    posterior_draws : np.ndarray
        The ``L`` posterior draws ``{theta_l}`` (or ``{f(theta_l)}``), shape
        ``(L,)``.
    rng : numpy.random.Generator, optional
        Generator used only to break exact ties. When ``None``, a default
        generator is created (ties are rare for continuous quantities).

    Returns
    -------
    int
        The rank statistic in ``{0, ..., L}``.

    References
    ----------
    .. [1] Talts et al., arXiv:1804.06788 (2018), Section 2 / Algorithm 1.
    """
    draws = np.asarray(posterior_draws, dtype=float).ravel()
    n_less = int(np.count_nonzero(draws < prior_draw))
    n_equal = int(np.count_nonzero(draws == prior_draw))
    if n_equal == 0:
        return n_less
    # Randomised tie-break: distribute the tied draws uniformly above/below the
    # prior draw so the rank stays uniform under the null (Talts et al. note 2).
    if rng is None:
        rng = np.random.default_rng()
    extra = int(rng.integers(0, n_equal + 1))
    return n_less + extra


def sbc_ranks(
    prior_draws: np.ndarray,
    posterior_draws: np.ndarray,
    param_names: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect per-parameter SBC rank statistics over many simulations.

    Vectorised form of :func:`sbc_rank` for a full SBC run: given ``N`` prior
    draws and the corresponding ``N x L`` posterior draws for each of ``P``
    parameters, return the ``N`` rank statistics per parameter (Talts et al.,
    arXiv:1804.06788).

    Parameters
    ----------
    prior_draws : np.ndarray
        Prior draws ``theta_tilde``, shape ``(n_sims,)`` for a single parameter
        or ``(n_sims, n_params)`` for several. ``theta_tilde[i]`` is the prior
        draw used to simulate the data for simulation ``i``.
    posterior_draws : np.ndarray
        Posterior draws, shape ``(n_sims, n_draws)`` for a single parameter or
        ``(n_sims, n_draws, n_params)`` for several. ``posterior_draws[i]`` are
        the ``L = n_draws`` posterior samples from the data simulated with
        ``prior_draws[i]``.
    param_names : sequence of str, optional
        Names for the parameter axis. Defaults to ``["param_0", ...]`` (or
        ``["param"]`` for the single-parameter case).
    seed : int, optional
        Seed for the (rarely used) tie-break randomisation; kept fixed for
        reproducibility.

    Returns
    -------
    dict of str -> np.ndarray
        Mapping ``param_name -> ranks`` with each array shape ``(n_sims,)`` and
        integer values in ``{0, ..., L}``.

    Raises
    ------
    ValueError
        If the simulation counts of ``prior_draws`` and ``posterior_draws`` do
        not match, or the parameter axes are inconsistent.

    References
    ----------
    .. [1] Talts et al., arXiv:1804.06788 (2018), Algorithm 1.
    """
    prior = np.asarray(prior_draws, dtype=float)
    post = np.asarray(posterior_draws, dtype=float)

    if prior.ndim == 1:
        prior = prior[:, None]
    if post.ndim == 2:
        post = post[:, :, None]
    if prior.ndim != 2 or post.ndim != 3:
        raise ValueError(
            "prior_draws must be (n_sims,) or (n_sims, n_params); "
            "posterior_draws must be (n_sims, n_draws) or "
            "(n_sims, n_draws, n_params)"
        )

    n_sims, n_params = prior.shape
    if post.shape[0] != n_sims:
        raise ValueError(
            f"simulation-count mismatch: prior has {n_sims} sims, " f"posterior has {post.shape[0]}"
        )
    if post.shape[2] != n_params:
        raise ValueError(
            f"parameter-count mismatch: prior has {n_params} params, "
            f"posterior has {post.shape[2]}"
        )

    if param_names is None:
        names: List[str] = ["param"] if n_params == 1 else [f"param_{j}" for j in range(n_params)]
    else:
        names = list(param_names)
        if len(names) != n_params:
            raise ValueError(f"param_names has {len(names)} entries but data has {n_params} params")

    rng = np.random.default_rng(seed)
    ranks: Dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        # Broadcasted comparison: (n_sims, n_draws) < (n_sims, 1).
        post_j = post[:, :, j]
        prior_j = prior[:, j][:, None]
        n_less = np.count_nonzero(post_j < prior_j, axis=1)
        n_equal = np.count_nonzero(post_j == prior_j, axis=1)
        rank = n_less.astype(np.int64)
        tie_mask = n_equal > 0
        if np.any(tie_mask):
            # integers(high) is exclusive, so high = count + 1 gives [0, count].
            extra = rng.integers(0, n_equal[tie_mask] + 1)
            rank[tie_mask] += extra.astype(np.int64)
        ranks[name] = rank
    return ranks


def sbc_uniformity_test(
    ranks: Dict[str, np.ndarray],
    n_posterior_draws: int,
    n_bins: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """Chi-square goodness-of-fit test of SBC rank uniformity.

    Under correct calibration the rank statistic is uniform on the ``L + 1``
    integers ``{0, ..., L}`` (Talts et al., arXiv:1804.06788). This bins the
    ranks into ``n_bins`` equal-width bins and runs Pearson's chi-square test
    against the uniform expectation ``E_b = n_sims / n_bins`` per bin, with
    ``n_bins - 1`` degrees of freedom. A *small* p-value (e.g. ``< 0.01``)
    indicates a non-uniform — hence miscalibrated — histogram.

    Parameters
    ----------
    ranks : dict of str -> np.ndarray
        Per-parameter rank arrays from :func:`sbc_ranks` / :func:`sbc_rank`,
        each with integer values in ``{0, ..., n_posterior_draws}``.
    n_posterior_draws : int
        Number of posterior draws ``L`` (so ranks span ``L + 1`` values).
    n_bins : int, optional
        Number of histogram bins. Defaults to a divisor of ``L + 1`` close to
        ``sqrt(n_sims)`` so the uniform expectation is exact in every bin
        (Talts et al. recommend choosing ``L`` so that ``L + 1`` is divisible by
        the bin count). Falls back to a single contiguous binning of the
        ``L + 1`` rank values when no clean divisor exists.

    Returns
    -------
    p_values : dict of str -> float
        Per-parameter chi-square uniformity p-value.
    chi2 : dict of str -> float
        Per-parameter chi-square statistic.
    n_bins : int
        The bin count actually used.

    Raises
    ------
    ValueError
        If ``ranks`` is empty or ``n_posterior_draws`` is negative.

    References
    ----------
    .. [1] Talts et al., arXiv:1804.06788 (2018), Section 3 (rank histograms).
    """
    if not ranks:
        raise ValueError("ranks must contain at least one parameter")
    if n_posterior_draws < 0:
        raise ValueError("n_posterior_draws must be non-negative")

    n_rank_values = n_posterior_draws + 1
    n_sims = int(len(next(iter(ranks.values()))))

    if n_bins is None:
        n_bins = _choose_sbc_bins(n_rank_values, n_sims)
    n_bins = max(1, min(int(n_bins), n_rank_values))

    # Equal-count partition of the L+1 integer rank values into n_bins bins.
    # np.linspace edges over [0, L+1] give contiguous integer ranges; using
    # np.histogram with these edges keeps the uniform expectation E = N/n_bins.
    edges = np.linspace(0, n_rank_values, n_bins + 1)
    expected = n_sims / n_bins

    p_values: Dict[str, float] = {}
    chi2_stats: Dict[str, float] = {}
    for name, rank_arr in ranks.items():
        observed, _ = np.histogram(np.asarray(rank_arr, dtype=float), bins=edges)
        if n_bins < 2:
            # A single bin cannot test uniformity; report a vacuous pass.
            p_values[name] = 1.0
            chi2_stats[name] = 0.0
            continue
        result = scipy_stats.chisquare(f_obs=observed, f_exp=np.full(n_bins, expected))
        chi2_stats[name] = float(result.statistic)
        p_values[name] = float(result.pvalue)
    return p_values, chi2_stats, n_bins


def _choose_sbc_bins(n_rank_values: int, n_sims: int) -> int:
    """Pick an SBC histogram bin count dividing ``n_rank_values`` (= L + 1).

    Targets roughly ``sqrt(n_sims)`` bins (a standard rank-histogram heuristic)
    while requiring exact divisibility of the ``L + 1`` integer rank values, so
    every bin holds the same number of rank values and the uniform expectation
    is exact. Falls back to the largest sensible divisor when no near-target one
    exists.
    """
    if n_rank_values <= 1:
        return 1
    target = max(2, int(round(np.sqrt(max(n_sims, 1)))))
    divisors = [d for d in range(2, n_rank_values + 1) if n_rank_values % d == 0]
    if not divisors:
        return 1
    # Choose the divisor closest to the target bin count.
    return min(divisors, key=lambda d: (abs(d - target), d))


def run_sbc(
    prior_sample_fn: Callable[[np.random.Generator], np.ndarray],
    simulate_fn: Callable[[np.ndarray, np.random.Generator], Any],
    posterior_sample_fn: Callable[[Any, int, np.random.Generator], np.ndarray],
    n_sims: int,
    n_posterior_draws: int,
    param_names: Optional[Sequence[str]] = None,
    n_bins: Optional[int] = None,
    seed: int = 0,
) -> SBCResult:
    """Run the full Simulation-Based Calibration harness (Talts et al. 2018).

    Implements Algorithm 1 of Talts, Betancourt, Simpson, Vehtari & Gelman
    (arXiv:1804.06788): for each of ``n_sims`` simulations draw a prior sample
    ``theta_tilde``, simulate data ``y ~ pi(y | theta_tilde)``, draw ``L``
    posterior samples ``theta ~ pi(theta | y)``, and record the rank of the
    prior draw among the posterior draws for every parameter. A correctly
    calibrated posterior sampler yields uniform rank histograms; this routine
    returns both the raw ranks and a per-parameter chi-square uniformity test.

    Parameters
    ----------
    prior_sample_fn : callable
        ``prior_sample_fn(rng) -> theta``, returning a prior draw as a scalar or
        1D array of length ``n_params``.
    simulate_fn : callable
        ``simulate_fn(theta, rng) -> data``, simulating data from the parameter
        draw. The returned object is passed verbatim to ``posterior_sample_fn``.
    posterior_sample_fn : callable
        ``posterior_sample_fn(data, n_draws, rng) -> draws``, returning ``L``
        posterior draws of shape ``(n_draws,)`` (single parameter) or
        ``(n_draws, n_params)`` (multi-parameter), with the parameter ordering
        matching ``prior_sample_fn``.
    n_sims : int
        Number of SBC simulations ``N``.
    n_posterior_draws : int
        Number of posterior draws ``L`` per simulation.
    param_names : sequence of str, optional
        Parameter names; inferred as ``param`` / ``param_j`` when omitted.
    n_bins : int, optional
        Histogram bin count for the uniformity test (see
        :func:`sbc_uniformity_test`).
    seed : int, optional
        Master seed; each simulation uses an independent child generator for
        reproducibility.

    Returns
    -------
    SBCResult
        Bundled ranks, chi-square statistics, and uniformity p-values.

    Raises
    ------
    ValueError
        If ``n_sims`` or ``n_posterior_draws`` is not positive.

    References
    ----------
    .. [1] Talts, Betancourt, Simpson, Vehtari & Gelman, "Validating Bayesian
       Inference Algorithms with Simulation-Based Calibration",
       arXiv:1804.06788 (2018).
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if n_posterior_draws <= 0:
        raise ValueError("n_posterior_draws must be positive")

    # Independent child generators per simulation for reproducible parallelism.
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_sims)

    prior_rows: List[np.ndarray] = []
    post_rows: List[np.ndarray] = []
    for i in range(n_sims):
        rng = np.random.default_rng(child_seeds[i])
        theta = np.atleast_1d(np.asarray(prior_sample_fn(rng), dtype=float)).ravel()
        data = simulate_fn(theta if theta.size > 1 else theta[0], rng)
        draws = np.asarray(posterior_sample_fn(data, n_posterior_draws, rng), dtype=float)
        if draws.ndim == 1:
            draws = draws[:, None]
        if draws.shape[0] != n_posterior_draws:
            raise ValueError(
                f"posterior_sample_fn returned {draws.shape[0]} draws, "
                f"expected {n_posterior_draws}"
            )
        prior_rows.append(theta)
        post_rows.append(draws)

    prior_arr = np.vstack(prior_rows)  # (n_sims, n_params)
    post_arr = np.stack(post_rows, axis=0)  # (n_sims, n_draws, n_params)

    ranks = sbc_ranks(prior_arr, post_arr, param_names=param_names, seed=seed)
    p_values, chi2_stats, used_bins = sbc_uniformity_test(ranks, n_posterior_draws, n_bins=n_bins)
    return SBCResult(
        ranks=ranks,
        n_posterior_draws=n_posterior_draws,
        p_values=p_values,
        chi2=chi2_stats,
        n_bins=used_bins,
    )

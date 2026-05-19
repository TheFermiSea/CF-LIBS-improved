"""Posterior-sampling backends for Bayesian CF-LIBS inference (T1-6).

This module hosts:

* :class:`Sampler` -- runtime-checkable Protocol for posterior-sampling
  backends, plus a generic :class:`SamplerResult` envelope.
* :class:`NumPyroNUTSSampler` -- canonical NUTS adapter (NumPyro).
* :class:`DynestyNestedSampler` -- canonical nested-sampling adapter (dynesty).
* :class:`MCMCSampler` / :class:`NestedSampler` -- legacy class names retained
  as aliases for back-compat. Their ``.run(observed, ...)`` signatures and
  return types are unchanged (``MCMCResult`` / ``NestedSamplingResult``).
* :func:`run_mcmc` -- convenience wrapper returning a legacy ``dict``.

The two-zone sampler lives in :mod:`two_zone`; the forward physics live in
:mod:`forward`; the priors and result containers live in :mod:`priors` and
:mod:`results`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy import stats

from cflibs.core.logging_config import get_logger

from .atomic import _as_jax_real
from .forward import (
    BayesianForwardModel,
    bayesian_model,
)
from .priors import (
    ConvergenceStatus,
    HAS_JAX,
    HAS_NUMPYRO,
    NoiseParameters,
    PriorConfig,
)
from .results import MCMCResult, NestedSamplingResult

logger = get_logger("inversion.bayesian.samplers")


# ---------------------------------------------------------------------------
# Optional-deps gateway
# ---------------------------------------------------------------------------

if HAS_JAX:
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = None  # type: ignore[assignment]

if HAS_NUMPYRO:
    from numpyro.infer import MCMC, NUTS, init_to_uniform
else:  # pragma: no cover
    MCMC = None  # type: ignore[assignment]
    NUTS = None  # type: ignore[assignment]
    init_to_uniform = None  # type: ignore[assignment]

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:  # pragma: no cover
    HAS_ARVIZ = False
    az = None  # type: ignore[assignment]

try:
    import dynesty
    from dynesty import NestedSampler as _DynestyNestedSampler

    HAS_DYNESTY = True
except ImportError:  # pragma: no cover
    HAS_DYNESTY = False
    dynesty = None  # type: ignore[assignment]
    _DynestyNestedSampler = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Sampler Protocol + generic SamplerResult envelope (T1-6 spec section 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerResult:
    """Common result envelope across NumPyro and dynesty backends.

    Attributes
    ----------
    posterior_samples : dict[str, np.ndarray]
        ``{name: (n_samples,) or (n_samples, k)}`` posterior draws.
    log_evidence : Optional[float]
        ``None`` for MCMC; finite log marginal-likelihood for nested sampling.
    diagnostics : dict[str, object]
        Per-parameter diagnostics: ``r_hat``, ``ess``, iterations, etc.
    metadata : dict[str, object]
        Sampler name, prior_config snapshot, sample count, ...
    """

    posterior_samples: Dict[str, np.ndarray]
    log_evidence: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Sampler(Protocol):
    """Common interface for posterior-sampling backends.

    Pattern source: petitRADTRANS (D-P1) + exojax NumPyro coupling (C-P14).
    Concrete samplers (:class:`NumPyroNUTSSampler`, :class:`DynestyNestedSampler`)
    expose this :meth:`fit` method in addition to their legacy ``.run`` API.
    """

    def fit(
        self,
        model: BayesianForwardModel,
        data: Any,
        **kwargs: Any,
    ) -> SamplerResult: ...


# ---------------------------------------------------------------------------
# Diagnostics helpers shared by NUTS samplers
# ---------------------------------------------------------------------------


def _diagnostics_from_mcmc(
    mcmc: Any,
    num_chains: int,
    variables: Tuple[str, ...] = ("T_eV", "log_ne"),
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute R-hat / ESS diagnostics with an ArviZ fallback to single-chain estimates."""
    r_hat: Dict[str, float] = {}
    ess: Dict[str, float] = {}

    if HAS_ARVIZ and num_chains > 1:
        try:
            idata = az.from_numpyro(mcmc)
            rhat_data = az.rhat(idata)
            ess_data = az.ess(idata)
            for var in variables:
                if var in rhat_data:
                    r_hat[var] = float(rhat_data[var].values)
                if var in ess_data:
                    ess[var] = float(ess_data[var].values)
        except Exception as e:  # pragma: no cover
            logger.warning(f"ArviZ diagnostics failed: {e}")
            r_hat, ess = _simple_diagnostics_from_mcmc(mcmc, variables)
    else:
        r_hat, ess = _simple_diagnostics_from_mcmc(mcmc, variables)

    return r_hat, ess


def _simple_diagnostics_from_mcmc(
    mcmc: Any, variables: Tuple[str, ...]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Simple single-chain ESS / R-hat fallback."""
    samples = mcmc.get_samples()
    r_hat: Dict[str, float] = {}
    ess: Dict[str, float] = {}
    for var in variables:
        if var in samples:
            s = np.array(samples[var]).flatten()
            ess[var] = float(len(s))  # naive: assume independent samples
            r_hat[var] = 1.0  # single chain: between-chain variance undefined
    return r_hat, ess


def _assess_convergence(
    r_hat: Dict[str, float], ess: Dict[str, float], num_samples: int
) -> ConvergenceStatus:
    """Assess overall convergence from per-parameter R-hat / ESS."""
    if not r_hat or not ess:
        return ConvergenceStatus.UNKNOWN
    max_rhat = max(r_hat.values()) if r_hat else 1.0
    min_ess = min(ess.values()) if ess else float(num_samples)
    if max_rhat < 1.01 and min_ess > 100:
        return ConvergenceStatus.CONVERGED
    if max_rhat < 1.1 and min_ess > 50:
        return ConvergenceStatus.WARNING
    return ConvergenceStatus.NOT_CONVERGED


def _to_arviz(mcmc: Any) -> Any:
    """Convert MCMC results to ArviZ ``InferenceData``."""
    if not HAS_ARVIZ:
        return None
    try:
        return az.from_numpyro(mcmc)
    except Exception as e:  # pragma: no cover
        logger.warning(f"ArviZ conversion failed: {e}")
        return None


# ---------------------------------------------------------------------------
# NUTS sampler (legacy MCMCSampler / new NumPyroNUTSSampler)
# ---------------------------------------------------------------------------


class MCMCSampler:
    """MCMC sampler for Bayesian CF-LIBS inference.

    Wraps NumPyro's NUTS sampler with sensible defaults for CF-LIBS,
    including convergence diagnostics, initialization strategies, and
    ArviZ integration for analysis and visualisation.

    Notes
    -----
    Retained as a class for backward compatibility. New code should prefer
    :class:`NumPyroNUTSSampler` (an alias) or instantiate via the
    :class:`Sampler` protocol's :meth:`fit` adapter.
    """

    def __init__(
        self,
        forward_model: BayesianForwardModel,
        prior_config: PriorConfig = PriorConfig(),
        noise_params: NoiseParameters = NoiseParameters(),
    ):
        if not HAS_NUMPYRO:
            raise ImportError("NumPyro required. Install with: pip install numpyro")

        self.forward_model = forward_model
        self.prior_config = prior_config
        self.noise_params = noise_params
        self.elements = forward_model.elements

        logger.info(
            f"MCMCSampler initialized: {len(self.elements)} elements, "
            f"T range={prior_config.T_eV_range} eV"
        )

    def _get_init_values(self) -> Dict[str, Any]:
        """Get sensible initial values for MCMC."""
        n_elements = len(self.elements)
        T_init = (self.prior_config.T_eV_range[0] + self.prior_config.T_eV_range[1]) / 2
        log_ne_init = (self.prior_config.log_ne_range[0] + self.prior_config.log_ne_range[1]) / 2
        conc_init = jnp.ones(n_elements) / n_elements

        init_values: Dict[str, Any] = {
            "T_eV": T_init,
            "log_ne": log_ne_init,
            "concentrations": conc_init,
        }

        if self.prior_config.baseline_degree > 0:
            n_coeffs = self.prior_config.baseline_degree + 1
            init_values["baseline_coeffs"] = jnp.zeros(n_coeffs)

        return init_values

    def run(
        self,
        observed: np.ndarray,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        seed: int = 0,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 8,
        progress_bar: bool = True,
        chain_method: str = "vectorized",
    ) -> MCMCResult:
        """Run MCMC sampling and return a fully populated :class:`MCMCResult`."""
        import jax.random as random

        observed_jax = _as_jax_real(observed)
        n_elements = len(self.elements)

        def model(obs):
            bayesian_model(self.forward_model, obs, self.prior_config, self.noise_params)

        kernel = NUTS(
            model,
            init_strategy=init_to_uniform(radius=0.5),
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )

        rng_key = random.PRNGKey(seed)
        logger.info(
            f"Starting MCMC: {num_chains} chains, {num_warmup} warmup, {num_samples} samples"
        )
        mcmc.run(rng_key, observed_jax)

        samples = mcmc.get_samples(group_by_chain=(num_chains > 1))
        T_samples = samples["T_eV"]
        log_ne_samples = samples["log_ne"]
        conc_samples = samples["concentrations"]

        T_flat = np.array(T_samples).flatten()
        log_ne_flat = np.array(log_ne_samples).flatten()
        conc_flat = np.array(conc_samples).reshape(-1, n_elements)

        r_hat, ess = _diagnostics_from_mcmc(mcmc, num_chains)
        convergence_status = _assess_convergence(r_hat, ess, num_samples)

        result = MCMCResult(
            samples={k: np.array(v) for k, v in samples.items()},
            T_eV_mean=float(np.mean(T_flat)),
            T_eV_std=float(np.std(T_flat)),
            T_eV_q025=float(np.percentile(T_flat, 2.5)),
            T_eV_q975=float(np.percentile(T_flat, 97.5)),
            log_ne_mean=float(np.mean(log_ne_flat)),
            log_ne_std=float(np.std(log_ne_flat)),
            log_ne_q025=float(np.percentile(log_ne_flat, 2.5)),
            log_ne_q975=float(np.percentile(log_ne_flat, 97.5)),
            concentrations_mean={
                el: float(np.mean(conc_flat[:, i])) for i, el in enumerate(self.elements)
            },
            concentrations_std={
                el: float(np.std(conc_flat[:, i])) for i, el in enumerate(self.elements)
            },
            concentrations_q025={
                el: float(np.percentile(conc_flat[:, i], 2.5)) for i, el in enumerate(self.elements)
            },
            concentrations_q975={
                el: float(np.percentile(conc_flat[:, i], 97.5))
                for i, el in enumerate(self.elements)
            },
            r_hat=r_hat,
            ess=ess,
            convergence_status=convergence_status,
            n_samples=num_samples,
            n_chains=num_chains,
            n_warmup=num_warmup,
            inference_data=_to_arviz(mcmc),
        )

        logger.info(
            f"MCMC complete: T = {result.T_eV_mean:.3f} +/- {result.T_eV_std:.3f} eV, "
            f"n_e = {result.n_e_mean:.2e} cm^-3, "
            f"convergence={convergence_status.value}"
        )
        return result

    # ------------------------------------------------------------------
    # Protocol adapter (new Sampler API)
    # ------------------------------------------------------------------

    def fit(
        self,
        model: BayesianForwardModel,  # noqa: ARG002 - retained for Protocol parity
        data: Any,
        **kwargs: Any,
    ) -> SamplerResult:
        """:class:`Sampler` Protocol adapter wrapping :meth:`run`.

        ``data`` may be a bare array (legacy) or any object exposing a
        ``flux`` attribute (e.g. a ``SpectrumData``-like container).
        ``kwargs`` are forwarded to :meth:`run`.
        """
        observed = getattr(data, "flux", data)
        result = self.run(observed, **kwargs)
        return SamplerResult(
            posterior_samples=result.samples,
            log_evidence=None,
            diagnostics={
                "r_hat": dict(result.r_hat),
                "ess": dict(result.ess),
                "convergence_status": result.convergence_status.value,
                "n_samples": result.n_samples,
                "n_chains": result.n_chains,
            },
            metadata={
                "sampler": "MCMCSampler",
                "elements": list(self.elements),
            },
        )

    # ------------------------------------------------------------------
    # Plot/diagnostic helpers (ArviZ wrappers; preserved API)
    # ------------------------------------------------------------------

    def plot_trace(self, result: MCMCResult, figsize: Tuple[int, int] = (12, 8)) -> Any:
        """Generate trace plot (delegates to :mod:`diagnostics`)."""
        from .diagnostics import plot_trace as _plot_trace

        return _plot_trace(result, figsize=figsize)

    def plot_posterior(self, result: MCMCResult, figsize: Tuple[int, int] = (12, 6)) -> Any:
        """Generate posterior plot (delegates to :mod:`diagnostics`)."""
        from .diagnostics import plot_posterior as _plot_posterior

        return _plot_posterior(result, figsize=figsize)

    def plot_corner(
        self,
        result: MCMCResult,
        var_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 10),
        show_titles: bool = True,  # noqa: ARG002 - preserved arg for back-compat
    ) -> Any:
        """Generate corner / pair plot (delegates to :mod:`diagnostics`)."""
        from .diagnostics import plot_corner as _plot_corner

        return _plot_corner(result, var_names=var_names, figsize=figsize)

    def plot_forest(self, result: MCMCResult, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """Generate forest plot (delegates to :mod:`diagnostics`)."""
        from .diagnostics import plot_forest as _plot_forest

        return _plot_forest(result, figsize=figsize)

    def posterior_predictive_check(
        self,
        result: MCMCResult,
        observed: np.ndarray,
        n_samples: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Perform posterior predictive check (delegates to :mod:`diagnostics`)."""
        from .diagnostics import posterior_predictive_check as _ppc

        return _ppc(
            self.forward_model,
            self.noise_params,
            list(self.elements),
            result,
            observed,
            n_samples=n_samples,
            seed=seed,
        )


# Canonical alias under the new naming convention. New code should prefer
# ``NumPyroNUTSSampler``; ``MCMCSampler`` remains for back-compat.
NumPyroNUTSSampler = MCMCSampler


# ---------------------------------------------------------------------------
# Nested sampler (legacy NestedSampler / new DynestyNestedSampler)
# ---------------------------------------------------------------------------


class NestedSampler:
    """Nested sampler for Bayesian CF-LIBS inference with model comparison.

    Uses dynesty under the hood. Provides direct evidence (marginal likelihood)
    calculation in addition to posterior samples.
    """

    def __init__(
        self,
        forward_model: BayesianForwardModel,
        prior_config: PriorConfig = PriorConfig(),
        noise_params: NoiseParameters = NoiseParameters(),
    ):
        if not HAS_DYNESTY:
            raise ImportError("dynesty required. Install with: pip install dynesty")

        self.forward_model = forward_model
        self.prior_config = prior_config
        self.noise_params = noise_params
        self.elements = forward_model.elements
        self.n_elements = len(self.elements)
        # T_eV + log_ne + (n_elements - 1) concentrations on the simplex.
        self.ndim = 2 + (self.n_elements - 1)

        logger.info(
            f"NestedSampler initialized: {self.n_elements} elements, "
            f"{self.ndim} dimensions, T range={prior_config.T_eV_range} eV"
        )

    def _prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Transform unit cube ``[0, 1]^n`` to physical parameter space."""
        x = np.zeros_like(u)

        T_min, T_max = self.prior_config.T_eV_range
        x[0] = T_min + u[0] * (T_max - T_min)

        log_ne_min, log_ne_max = self.prior_config.log_ne_range
        x[1] = log_ne_min + u[1] * (log_ne_max - log_ne_min)

        # Concentrations via stick-breaking on a Dirichlet(alpha, ...) prior.
        if self.n_elements > 1:
            alpha = self.prior_config.concentration_alpha
            remaining = 1.0
            for i in range(self.n_elements - 1):
                beta_sample = stats.beta.ppf(u[2 + i], alpha, alpha * (self.n_elements - 1 - i))
                x[2 + i] = remaining * beta_sample
                remaining -= x[2 + i]

        return x

    def _params_to_concentrations(self, params: np.ndarray) -> np.ndarray:
        """Convert parameter vector to full concentration array (sum-to-one)."""
        if self.n_elements == 1:
            return np.array([1.0])
        conc = np.zeros(self.n_elements)
        conc[:-1] = params[2:]
        conc[-1] = max(0.0, 1.0 - np.sum(conc[:-1]))
        return conc

    def _log_likelihood(self, params: np.ndarray, observed: np.ndarray) -> float:
        """Compute log-likelihood for nested sampling."""
        T_eV = params[0]
        log_ne = params[1]
        concentrations = self._params_to_concentrations(params)

        if T_eV <= 0 or np.any(concentrations < 0) or np.any(concentrations > 1):
            return -np.inf

        try:
            predicted = self.forward_model.forward_numpy(T_eV, log_ne, concentrations)

            sigma_read = self.noise_params.readout_noise
            dark = self.noise_params.dark_current
            gain = self.noise_params.gain
            # Match the NumPyro-side `bayesian_model` + module-level
            # `log_likelihood` exactly: Poisson shot-noise term is
            # `predicted / gain`, not raw `predicted`. Use `maximum(...,
            # 1e-10)` instead of `abs` so a negative predicted value
            # (possible after polynomial baseline) is flagged rather than
            # silently flipped positive. Bug surfaced 2026-05-19 by AI
            # physics review — was making MCMC posterior vs nested-
            # sampling-evidence Bayes factors incoherent.
            pred_safe = np.maximum(predicted, 1e-10)
            variance = pred_safe / gain + sigma_read**2 + dark

            residuals = observed - predicted
            log_lik = -0.5 * np.sum(residuals**2 / variance + np.log(2 * np.pi * variance))

            if not np.isfinite(log_lik):
                return -np.inf
            return float(log_lik)
        except Exception:  # pragma: no cover
            return -np.inf

    def run(
        self,
        observed: np.ndarray,
        nlive: int = 100,
        dlogz: float = 0.1,
        sample: str = "auto",
        bound: str = "multi",
        seed: int = 42,
        maxiter: Optional[int] = None,
        maxcall: Optional[int] = None,
        verbose: bool = True,
    ) -> NestedSamplingResult:
        """Run nested sampling."""
        observed_np = np.asarray(observed)

        def loglike(params):
            return self._log_likelihood(params, observed_np)

        rstate = np.random.default_rng(seed)

        logger.info(f"Starting nested sampling: nlive={nlive}, dlogz={dlogz}, ndim={self.ndim}")

        sampler = _DynestyNestedSampler(
            loglike,
            self._prior_transform,
            self.ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            rstate=rstate,
        )

        sampler.run_nested(
            dlogz=dlogz,
            maxiter=maxiter,
            maxcall=maxcall,
            print_progress=verbose,
        )

        results = sampler.results

        samples = results.samples  # (n_samples, ndim)
        weights = np.exp(results.logwt - results.logwt.max())
        weights /= weights.sum()

        log_evidence = float(results.logz[-1])
        log_evidence_err = float(results.logzerr[-1])
        information = float(results.information[-1]) if hasattr(results, "information") else 0.0

        T_samples = samples[:, 0]
        log_ne_samples = samples[:, 1]

        T_mean = float(np.average(T_samples, weights=weights))
        T_std = float(np.sqrt(np.average((T_samples - T_mean) ** 2, weights=weights)))
        log_ne_mean = float(np.average(log_ne_samples, weights=weights))
        log_ne_std = float(
            np.sqrt(np.average((log_ne_samples - log_ne_mean) ** 2, weights=weights))
        )

        conc_samples = np.array([self._params_to_concentrations(s) for s in samples])
        conc_mean: Dict[str, float] = {}
        conc_std: Dict[str, float] = {}
        for i, el in enumerate(self.elements):
            c_samples = conc_samples[:, i]
            c_mean = float(np.average(c_samples, weights=weights))
            c_std = float(np.sqrt(np.average((c_samples - c_mean) ** 2, weights=weights)))
            conc_mean[el] = c_mean
            conc_std[el] = c_std

        result = NestedSamplingResult(
            samples={
                "T_eV": T_samples,
                "log_ne": log_ne_samples,
                "concentrations": conc_samples,
            },
            weights=weights,
            log_evidence=log_evidence,
            log_evidence_err=log_evidence_err,
            information=information,
            T_eV_mean=T_mean,
            T_eV_std=T_std,
            log_ne_mean=log_ne_mean,
            log_ne_std=log_ne_std,
            concentrations_mean=conc_mean,
            concentrations_std=conc_std,
            n_live=nlive,
            n_iterations=int(results.niter),
            n_calls=int(np.sum(results.ncall)),
        )

        logger.info(
            f"Nested sampling complete: ln(Z) = {log_evidence:.2f} +/- {log_evidence_err:.2f}, "
            f"T = {T_mean:.3f} +/- {T_std:.3f} eV"
        )
        return result

    def fit(
        self,
        model: BayesianForwardModel,  # noqa: ARG002 - kept for Protocol parity
        data: Any,
        **kwargs: Any,
    ) -> SamplerResult:
        """:class:`Sampler` Protocol adapter wrapping :meth:`run`."""
        observed = getattr(data, "flux", data)
        result = self.run(observed, **kwargs)
        return SamplerResult(
            posterior_samples=result.samples,
            log_evidence=result.log_evidence,
            diagnostics={
                "log_evidence_err": result.log_evidence_err,
                "information": result.information,
                "n_live": result.n_live,
                "n_iterations": result.n_iterations,
                "n_calls": result.n_calls,
            },
            metadata={
                "sampler": "NestedSampler",
                "elements": list(self.elements),
            },
        )


DynestyNestedSampler = NestedSampler


# ---------------------------------------------------------------------------
# Convenience wrapper (legacy dict-returning API)
# ---------------------------------------------------------------------------


def run_mcmc(
    forward_model: BayesianForwardModel,
    observed: np.ndarray,
    prior_config: PriorConfig = PriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    """Convenience wrapper that runs MCMC and returns a legacy ``dict``.

    For full functionality (diagnostics, ArviZ integration), instantiate
    :class:`MCMCSampler` directly.
    """
    sampler = MCMCSampler(forward_model, prior_config, noise_params)
    result = sampler.run(
        observed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        progress_bar=False,
    )

    return {
        "samples": result.samples,
        "T_eV_mean": result.T_eV_mean,
        "T_eV_std": result.T_eV_std,
        "log_ne_mean": result.log_ne_mean,
        "log_ne_std": result.log_ne_std,
        "concentrations_mean": result.concentrations_mean,
        "concentrations_std": result.concentrations_std,
        "n_e_mean": result.n_e_mean,
        "T_K_mean": result.T_K_mean,
    }


__all__ = [
    "Sampler",
    "SamplerResult",
    "NumPyroNUTSSampler",
    "DynestyNestedSampler",
    "MCMCSampler",
    "NestedSampler",
    "run_mcmc",
    "HAS_ARVIZ",
    "HAS_DYNESTY",
]

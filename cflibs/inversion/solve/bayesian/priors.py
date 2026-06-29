"""Bayesian priors and parameter declarations (T1-6).

This module hosts:

* :class:`PriorConfig` / :class:`TwoZonePriorConfig` -- legacy configuration
  dataclasses consumed by :func:`bayesian_model` / :func:`two_zone_bayesian_model`.
* :class:`NoiseParameters` -- detector-noise hyperparameters.
* :class:`ConvergenceStatus` -- MCMC convergence label.
* :class:`Parameter` -- declarative parameter spec usable by both NumPyro
  (``Parameter.numpyro_sample``) and nested samplers (``Parameter.cube_transform``).
  See ADR-0001 / T1-6 spec section 3.
* ``create_temperature_prior`` / ``create_density_prior`` /
  ``create_concentration_prior`` -- convenience helpers that build NumPyro
  distributions for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

try:  # optional JAX
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover - JAX not installed
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

try:  # optional NumPyro
    import numpyro
    import numpyro.distributions as dist

    HAS_NUMPYRO = True
except ImportError:  # pragma: no cover - numpyro not installed
    HAS_NUMPYRO = False
    numpyro = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Detector noise
# ---------------------------------------------------------------------------


@dataclass
class NoiseParameters:
    """Noise model parameters for LIBS spectra.

    Combines Poisson shot noise (signal-dependent), Gaussian readout noise
    (signal-independent), and an additive dark-current offset.

    Attributes
    ----------
    readout_noise : float
        RMS readout noise in counts.
    dark_current : float
        Dark current per pixel in counts.
    gain : float
        Detector gain (counts / photon).
    """

    readout_noise: float = 10.0
    dark_current: float = 1.0
    gain: float = 1.0


# ---------------------------------------------------------------------------
# Legacy prior configurations
# ---------------------------------------------------------------------------


@dataclass
class PriorConfig:
    """Configuration for Bayesian priors on plasma parameters (single-zone).

    Attributes
    ----------
    T_eV_range : Tuple[float, float]
        Temperature range in eV (default 0.5-3.0).
    log_ne_range : Tuple[float, float]
        ``log_{10}(n_e)`` range (default 15-19; ``n_e`` in cm^-3).
    concentration_alpha : float
        Dirichlet prior concentration parameter (default 1.0 = uniform on simplex).
        With ``nominal_mole_fracs`` set, this is the total Dirichlet concentration
        ``sum(alpha)``; use ~50-80 for a WEAK feedstock prior (per-element sigma
        ~2-3 wt%, larger than expected DED drift, so the posterior stays
        data-dominated and the prior never pins the answer to nominal).
    baseline_degree : int
        Chebyshev polynomial baseline degree (0 = no baseline, max 5).
    baseline_scale : Optional[float]
        Prior scale for baseline coefficients. ``None`` -> auto
        ``0.1 * max(observed)`` at sampling time.
    nominal_mole_fracs : Optional[np.ndarray]
        Weakly-informative DED feedstock prior: number(mole) fractions aligned to
        the forward model's element order (sum to 1). When set, the concentration
        Dirichlet is centered on it (``alpha = concentration_alpha * x_nom``, mean
        exactly ``x_nom``) instead of symmetric. ``None`` (default) -> symmetric
        Dirichlet (no prior). Build it with :meth:`nominal_mole_fracs_from_wt`.
    """

    T_eV_range: Tuple[float, float] = (0.5, 3.0)
    log_ne_range: Tuple[float, float] = (15.0, 19.0)
    concentration_alpha: float = 1.0
    baseline_degree: int = 0
    baseline_scale: Optional[float] = None
    nominal_mole_fracs: Optional[Any] = None

    @staticmethod
    def nominal_mole_fracs_from_wt(
        nominal_wt: Dict[str, float], elements: Tuple[str, ...]
    ) -> "np.ndarray":
        """Build the nominal mole-fraction vector aligned to ``elements``.

        Converts a feedstock wt% (or mass-fraction) composition to number
        fractions ``x_i = (w_i/M_i) / sum_j(w_j/M_j)`` over the constrained set,
        in the EXACT order the forward model uses, so the Dirichlet components
        line up with the sampled concentrations.
        """
        from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES

        w = np.array([max(float(nominal_wt.get(e, 0.0)), 0.0) for e in elements])
        m = np.array([float(STANDARD_ATOMIC_MASSES.get(e, 50.0)) for e in elements])
        moles = np.where(m > 0, w / m, 0.0)
        total = moles.sum()
        if total <= 0:
            raise ValueError("nominal composition has no mass on the given elements")
        return moles / total

    @classmethod
    def geological(cls, **kwargs) -> "PriorConfig":
        """Geological sample prior: broad T/ne, sparse concentrations."""
        defaults = {"concentration_alpha": 0.5, "T_eV_range": (0.5, 2.0)}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def metallurgical(cls, **kwargs) -> "PriorConfig":
        """Metallurgical alloy prior: moderate T, peaked concentrations."""
        defaults = {"concentration_alpha": 2.0, "T_eV_range": (0.6, 1.5)}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def uninformative(cls, **kwargs) -> "PriorConfig":
        """Maximally uninformative prior: uniform on the simplex."""
        defaults = {"concentration_alpha": 1.0}
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class TwoZonePriorConfig:
    """Prior configuration for the two-zone plasma model.

    See ``bayesian/forward.py::TwoZoneBayesianForwardModel`` for the geometry.

    Parameters
    ----------
    T_core_eV_range : tuple of float
        Core temperature range in eV (default 0.8-3.0).
    T_shell_eV_range : tuple of float
        Shell temperature range in eV (default 0.3-2.0).
    log_ne_range : tuple of float
        ``log_{10}(n_e)`` range (default 15-19).
    concentration_alpha : float
        Dirichlet concentration parameter (default 1.0).
    shell_fraction_range : tuple of float
        Shell fraction of total plasma length (default 0.1-0.9).
    optical_depth_scale_range : tuple of float
        Optical depth multiplier range (default 0.01-10.0).
    enforce_T_ordering : bool
        If True, penalise ``T_core < T_shell`` (default True).
    baseline_degree : int
        Polynomial baseline degree (default 3).
    mcwhirter_penalty_scale : float
        McWhirter penalty strength (default 10.0; 0 disables).
    max_delta_E_eV : float
        Maximum energy gap for McWhirter criterion (default 3.0 eV).
    """

    T_core_eV_range: Tuple[float, float] = (0.8, 3.0)
    T_shell_eV_range: Tuple[float, float] = (0.3, 2.0)
    log_ne_range: Tuple[float, float] = (15.0, 19.0)
    concentration_alpha: float = 1.0
    shell_fraction_range: Tuple[float, float] = (0.1, 0.9)
    optical_depth_scale_range: Tuple[float, float] = (0.01, 10.0)
    enforce_T_ordering: bool = True
    baseline_degree: int = 3
    mcwhirter_penalty_scale: float = 10.0
    max_delta_E_eV: float = 3.0


# ---------------------------------------------------------------------------
# Convergence status
# ---------------------------------------------------------------------------


class ConvergenceStatus(Enum):
    """MCMC convergence status label."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    WARNING = "warning"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Declarative Parameter (T1-6 spec section 3)
# ---------------------------------------------------------------------------


PriorKind = Literal["uniform", "loguniform", "normal", "truncnormal", "dirichlet"]


@dataclass(frozen=True)
class Parameter:
    """Declarative parameter spec consumable by both NumPyro and nested samplers.

    Pattern source: petitRADTRANS ``RetrievalConfig.Parameter`` (D-P1) +
    LMFIT ``Parameter``. The pair ``numpyro_sample()`` / ``cube_transform()``
    lets one declaration drive both NumPyro NUTS and dynesty nested sampling
    without forking the prior plumbing.

    Notes
    -----
    The Dirichlet special case is intentionally NOT decomposed per-coordinate
    -- :class:`Parameter` declares scalar priors only. Composition simplices
    (``concentrations``) are handled by ``DirichletConcentrationPrior`` or by
    dynesty's stick-breaking transform inside :class:`NestedSampler`.

    Parameters
    ----------
    name : str
        Site name used by NumPyro / column name in dynesty output.
    prior_low : float
        Lower bound (uniform / loguniform) or low truncation (truncnormal).
    prior_high : float
        Upper bound (uniform / loguniform) or high truncation (truncnormal).
    prior_kind : PriorKind
        One of ``"uniform"``, ``"loguniform"``, ``"normal"``, ``"truncnormal"``,
        ``"dirichlet"``. (``"dirichlet"`` is reserved and not consumable by
        scalar adapters; see note above.)
    vary : bool
        If False, the parameter is held fixed at ``prior_mean`` (or
        ``prior_low`` if mean is unset).
    expr : Optional[str]
        Optional algebraic constraint string ("1 - C_Cr - C_Ni"); informational
        only -- not enforced by the adapters here.
    prior_mean : Optional[float]
        Mean for normal / truncnormal priors. Also used as the fixed value when
        ``vary=False``.
    prior_std : Optional[float]
        Standard deviation for normal / truncnormal priors.
    """

    name: str
    prior_low: float
    prior_high: float
    prior_kind: PriorKind = "uniform"
    vary: bool = True
    expr: Optional[str] = None
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None

    def numpyro_sample(self) -> Any:
        """Emit a NumPyro sample site and return the draw.

        Returns
        -------
        jax.numpy.ndarray
            Sampled value (or fixed scalar if ``vary=False``).

        Raises
        ------
        ImportError
            If NumPyro is not installed.
        ValueError
            On unsupported ``prior_kind``.
        """
        if not HAS_NUMPYRO:
            raise ImportError("NumPyro required for Parameter.numpyro_sample()")
        if not self.vary:
            value = self.prior_mean if self.prior_mean is not None else self.prior_low
            return jnp.asarray(value)
        if self.prior_kind == "uniform":
            return numpyro.sample(self.name, dist.Uniform(self.prior_low, self.prior_high))
        if self.prior_kind == "loguniform":
            return self._numpyro_sample_loguniform()
        if self.prior_kind == "normal":
            return self._numpyro_sample_normal()
        if self.prior_kind == "truncnormal":
            return self._numpyro_sample_truncnormal()
        if self.prior_kind == "dirichlet":
            raise ValueError(
                "Parameter('dirichlet') is reserved -- compose Dirichlet at the model layer "
                "or use dynesty stick-breaking in NestedSampler._prior_transform"
            )
        raise ValueError(f"Unsupported prior_kind: {self.prior_kind!r}")

    def _numpyro_sample_loguniform(self) -> Any:
        """NumPyro draw for the ``loguniform`` prior_kind."""
        log_low = jnp.log10(jnp.asarray(self.prior_low))
        log_high = jnp.log10(jnp.asarray(self.prior_high))
        log_x = numpyro.sample(self.name + "_log", dist.Uniform(log_low, log_high))
        return numpyro.deterministic(self.name, jnp.power(10.0, log_x))

    def _numpyro_sample_normal(self) -> Any:
        """NumPyro draw for the ``normal`` prior_kind."""
        if self.prior_mean is None or self.prior_std is None:
            raise ValueError(
                f"Parameter {self.name!r}: 'normal' prior requires prior_mean and prior_std"
            )
        return numpyro.sample(self.name, dist.Normal(self.prior_mean, self.prior_std))

    def _numpyro_sample_truncnormal(self) -> Any:
        """NumPyro draw for the ``truncnormal`` prior_kind."""
        if self.prior_mean is None or self.prior_std is None:
            raise ValueError(
                f"Parameter {self.name!r}: 'truncnormal' prior requires prior_mean and prior_std"
            )
        return numpyro.sample(
            self.name,
            dist.TruncatedNormal(
                self.prior_mean,
                self.prior_std,
                low=self.prior_low,
                high=self.prior_high,
            ),
        )

    def cube_transform(self, u: float) -> float:
        """Inverse-CDF transform from a unit-cube draw ``u`` in [0, 1].

        Adapter to dynesty-style nested-sampling prior transforms.

        Parameters
        ----------
        u : float
            Uniform draw in ``[0, 1]``.

        Returns
        -------
        float
            Physical-space value.
        """
        if not self.vary:
            return float(self.prior_mean if self.prior_mean is not None else self.prior_low)
        if self.prior_kind == "uniform":
            return float(self.prior_low + u * (self.prior_high - self.prior_low))
        if self.prior_kind == "loguniform":
            log_lo = np.log10(self.prior_low)
            log_hi = np.log10(self.prior_high)
            return float(10.0 ** (log_lo + u * (log_hi - log_lo)))
        if self.prior_kind == "normal":
            if self.prior_mean is None or self.prior_std is None:
                raise ValueError(
                    f"Parameter {self.name!r}: 'normal' prior requires prior_mean and prior_std"
                )
            from scipy.stats import norm  # lazy: scipy is heavy

            return float(norm.ppf(u, loc=self.prior_mean, scale=self.prior_std))
        if self.prior_kind == "truncnormal":
            if self.prior_mean is None or self.prior_std is None:
                raise ValueError(
                    f"Parameter {self.name!r}: 'truncnormal' prior requires prior_mean and prior_std"
                )
            from scipy.stats import truncnorm

            a = (self.prior_low - self.prior_mean) / self.prior_std
            b = (self.prior_high - self.prior_mean) / self.prior_std
            return float(truncnorm.ppf(u, a, b, loc=self.prior_mean, scale=self.prior_std))
        if self.prior_kind == "dirichlet":
            raise ValueError(
                "Parameter('dirichlet').cube_transform is not defined -- use stick-breaking"
            )
        raise ValueError(f"Unsupported prior_kind: {self.prior_kind!r}")


# ---------------------------------------------------------------------------
# NumPyro distribution factories (legacy convenience)
# ---------------------------------------------------------------------------


def create_temperature_prior(
    T_min_eV: float = 0.5,
    T_max_eV: float = 3.0,
    prior_type: str = "uniform",
) -> Any:
    """Create a temperature prior distribution.

    Parameters
    ----------
    T_min_eV, T_max_eV : float
        Temperature range in eV.
    prior_type : str
        ``"uniform"`` (default), ``"normal"`` (TruncatedNormal centred on the
        midpoint), or any unrecognised string (treated as uniform).

    Returns
    -------
    numpyro.distributions.Distribution
        Prior distribution.
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required")

    if prior_type == "uniform":
        return dist.Uniform(T_min_eV, T_max_eV)
    if prior_type == "normal":
        mean = (T_min_eV + T_max_eV) / 2
        std = (T_max_eV - T_min_eV) / 4
        return dist.TruncatedNormal(mean, std, low=T_min_eV, high=T_max_eV)
    return dist.Uniform(T_min_eV, T_max_eV)


def create_density_prior(
    log_ne_min: float = 15.0,
    log_ne_max: float = 19.0,
    prior_type: str = "uniform",
) -> Any:
    """Create an electron-density prior distribution.

    Returns a prior on ``log_{10}(n_e)``. ``"uniform"`` corresponds to the
    Jeffreys / log-uniform prior on ``n_e``.

    Parameters
    ----------
    log_ne_min, log_ne_max : float
        Bounds on ``log_{10}(n_e)``.
    prior_type : str
        ``"uniform"`` (default) or ``"normal"`` (TruncatedNormal).

    Returns
    -------
    numpyro.distributions.Distribution
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required")

    if prior_type == "uniform":
        return dist.Uniform(log_ne_min, log_ne_max)
    if prior_type == "normal":
        mean = (log_ne_min + log_ne_max) / 2
        std = (log_ne_max - log_ne_min) / 4
        return dist.TruncatedNormal(mean, std, low=log_ne_min, high=log_ne_max)
    return dist.Uniform(log_ne_min, log_ne_max)


def create_concentration_prior(
    n_elements: int,
    alpha: float = 1.0,
    known_concentrations: Optional[Dict[int, float]] = None,
) -> Any:
    """Create a concentration (Dirichlet) prior distribution.

    Parameters
    ----------
    n_elements : int
        Number of elements (length of the simplex).
    alpha : float
        Dirichlet concentration parameter (1.0 = uniform on simplex).
    known_concentrations : Optional[dict[int, float]]
        Optional ``{element_idx: value}`` map; for each known element the
        Dirichlet alpha is increased to peak near the supplied value.

    Returns
    -------
    numpyro.distributions.Dirichlet
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required")

    alphas = jnp.ones(n_elements) * alpha
    if known_concentrations:
        for idx, value in known_concentrations.items():
            alphas = alphas.at[idx].set(alpha * (1 + 10 * value))
    return dist.Dirichlet(alphas)


__all__ = [
    "NoiseParameters",
    "PriorConfig",
    "TwoZonePriorConfig",
    "ConvergenceStatus",
    "Parameter",
    "PriorKind",
    "create_temperature_prior",
    "create_density_prior",
    "create_concentration_prior",
]

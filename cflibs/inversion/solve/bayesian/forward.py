"""Bayesian forward models and NumPyro graph builders (T1-6).

This module hosts the JAX-compiled CF-LIBS forward models that map plasma
parameters ``(T, n_e, concentrations)`` to a synthetic spectrum, plus the
NumPyro graph builders that wire them up with priors:

* :class:`BayesianForwardModel` -- single-zone LTE forward model.
* :class:`TwoZoneBayesianForwardModel` -- core+shell self-reversed model.
* :func:`log_likelihood` -- module-level log-likelihood used by dynesty.
* :func:`bayesian_model` / :func:`two_zone_bayesian_model` -- NumPyro graph
  builders that combine the forward model with the priors from
  :mod:`cflibs.inversion.solve.bayesian.priors`.

Atomic-data carriers (:class:`AtomicDataArrays`, :func:`load_atomic_data`,
:func:`partition_function`, :func:`mcwhirter_log_penalty`, and a few JAX-real
helpers) live in the sibling :mod:`atomic` module.

Notes
-----
The unified forward kernel at :func:`cflibs.radiation.kernels.forward_model`
(T1-2) consumes :class:`cflibs.core.jax_runtime.AtomicSnapshot`, whose
partition-function representation differs from the legacy
:class:`AtomicDataArrays` direct-sum path. To preserve rtol=1e-5 parity with
pre-T1-6 ``BayesianForwardModel`` output under existing tests, we keep the
legacy partition path in :meth:`BayesianForwardModel._compute_spectrum` and
expose :meth:`AtomicDataArrays.from_snapshot` (in :mod:`atomic`) so callers
that already hold a snapshot can convert it into the legacy carrier. A
follow-up bead will migrate the sampler-facing path to the kernel once
parity testing across a wider grid is in place.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .atomic import (
    AtomicDataArrays,
    _as_jax_real,
    _compute_instrument_sigma,
    _JAX_C_LIGHT,
    _JAX_E_CHARGE,
    _JAX_EV_TO_J,
    _JAX_EV_TO_K,
    _JAX_H_PLANCK,
    _JAX_M_E,
    _JAX_M_PROTON,
    _JAX_REAL_DTYPE,
    _JAX_SAHA_CONST_CM3,
    _resolve_total_species_density_cm3,
    load_atomic_data,
    logger,
    mcwhirter_log_penalty,
    partition_function,
)
from .priors import (
    HAS_JAX,
    HAS_NUMPYRO,
    NoiseParameters,
    PriorConfig,
    TwoZonePriorConfig,
)

# ---------------------------------------------------------------------------
# Optional-deps gateway
# ---------------------------------------------------------------------------

if HAS_JAX:
    import jax.numpy as jnp

    from cflibs.radiation.profiles import _voigt_profile_kernel_jax
else:  # pragma: no cover
    jnp = None  # type: ignore[assignment]
    _voigt_profile_kernel_jax = None  # type: ignore[assignment]

if HAS_NUMPYRO:
    import numpyro
    import numpyro.distributions as dist
else:  # pragma: no cover
    numpyro = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Single-zone forward model
# ---------------------------------------------------------------------------


class BayesianForwardModel:
    """Bayesian forward model for CF-LIBS spectra (single-zone LTE).

    JAX-compatible mapping of plasma parameters ``(T, n_e, concentrations)``
    to a synthetic spectrum. Physics includes Saha-Boltzmann populations,
    Voigt profiles via the Weideman rational Faddeeva approximation, Stark
    broadening with temperature scaling, and Doppler broadening with the
    per-line mass dependence.

    Parameters
    ----------
    db_path : str
        Path to the atomic database.
    elements : list of str
        Elements to include.
    wavelength_range : tuple of float
        Wavelength range ``(wl_min, wl_max)`` in nm.
    wavelength_grid : np.ndarray, optional
        Custom wavelength grid; auto-generated if ``None``.
    pixels : int
        Number of pixels for the auto-generated grid (default 2048).
    instrument_fwhm_nm : float, optional
        Instrument FWHM in nm (default 0.05). Mutually exclusive with
        ``resolving_power``.
    resolving_power : float, optional
        Spectrometer resolving power ``R = lambda / Delta lambda``. When set,
        FWHM varies as ``lambda / R`` across the spectrum.
    """

    def __init__(
        self,
        db_path: str,
        elements: List[str],
        wavelength_range: Tuple[float, float],
        wavelength_grid: Optional[np.ndarray] = None,
        pixels: int = 2048,
        instrument_fwhm_nm: Optional[float] = None,
        resolving_power: Optional[float] = None,
    ):
        if not HAS_JAX:
            raise ImportError("JAX required. Install with: pip install jax jaxlib")

        has_fwhm = instrument_fwhm_nm is not None
        has_rp = resolving_power is not None
        if has_fwhm and has_rp:
            raise ValueError(
                "resolving_power and instrument_fwhm_nm are mutually exclusive. "
                "Set one or the other, not both."
            )
        if not has_fwhm and not has_rp:
            instrument_fwhm_nm = 0.05

        self.elements = elements
        self.wavelength_range = wavelength_range
        self.instrument_fwhm_nm = instrument_fwhm_nm if instrument_fwhm_nm is not None else 0.05
        self.resolving_power = resolving_power

        if wavelength_grid is not None:
            self.wavelength = _as_jax_real(wavelength_grid)
        else:
            wl_min = _as_jax_real(wavelength_range[0])
            wl_max = _as_jax_real(wavelength_range[1])
            self.wavelength = jnp.linspace(wl_min, wl_max, pixels, dtype=_JAX_REAL_DTYPE)

        self.atomic_data = load_atomic_data(db_path, elements, wavelength_range)

        # Pre-compute Chebyshev Vandermonde matrix for polynomial baseline.
        wl_np = np.asarray(self.wavelength)
        self._wl_norm = 2.0 * (wl_np - wl_np[0]) / max(wl_np[-1] - wl_np[0], 1e-6) - 1.0
        max_baseline_degree = 5
        self._max_baseline_degree = max_baseline_degree
        self._baseline_basis = np.polynomial.chebyshev.chebvander(
            self._wl_norm, max_baseline_degree
        )
        if HAS_JAX:
            self._baseline_basis_jax = jnp.array(self._baseline_basis, dtype=_JAX_REAL_DTYPE)

        logger.info(
            f"BayesianForwardModel: {len(elements)} elements, "
            f"{len(self.wavelength)} wavelengths, "
            f"{len(self.atomic_data.wavelength_nm)} lines"
        )

    def _load_atomic_data(self, db_path: str) -> AtomicDataArrays:
        """Load atomic data -- delegates to :func:`load_atomic_data`."""
        return load_atomic_data(db_path, self.elements, self.wavelength_range)

    def forward(
        self,
        T_eV: float,
        log_ne: float,
        concentrations: Any,
        total_species_density_cm3: Optional[float] = None,
    ):
        """Compute synthetic spectrum for given plasma parameters."""
        T_eV = _as_jax_real(T_eV)
        log_ne = _as_jax_real(log_ne)
        concentrations = _as_jax_real(concentrations)
        n_e = jnp.power(_as_jax_real(10.0), log_ne)
        return self._compute_spectrum(
            T_eV,
            n_e,
            concentrations,
            total_species_density_cm3=total_species_density_cm3,
        )

    def forward_numpy(
        self,
        T_eV: float,
        log_ne: float,
        concentrations: np.ndarray,
        total_species_density_cm3: Optional[float] = None,
    ) -> np.ndarray:
        """Compute synthetic spectrum using NumPy arrays (dynesty compatibility)."""
        conc_jax = _as_jax_real(concentrations)
        result = self.forward(
            T_eV,
            log_ne,
            conc_jax,
            total_species_density_cm3=total_species_density_cm3,
        )
        return np.array(result)

    @staticmethod
    def _partition_function(T_K: float, coeffs: Any) -> Any:
        """Evaluate polynomial partition function (delegates to module helper)."""
        return partition_function(T_K, coeffs)

    def _compute_spectrum(
        self,
        T_eV: float,
        n_e: float,
        concentrations: Any,
        total_species_density_cm3: Optional[float] = None,
    ):
        """Compute spectrum with full physics (legacy partition path)."""
        data = self.atomic_data
        T_eV = _as_jax_real(T_eV)
        n_e = _as_jax_real(n_e)
        concentrations = _as_jax_real(concentrations)
        T_K = T_eV * _JAX_EV_TO_K
        total_species_density = _resolve_total_species_density_cm3(n_e, total_species_density_cm3)

        U0 = self._partition_function(T_K, data.partition_coeffs[:, 0])
        U1 = self._partition_function(T_K, data.partition_coeffs[:, 1])
        IP_I = data.ionization_potentials[:, 0]

        saha_factor = (_JAX_SAHA_CONST_CM3 / n_e) * (T_eV**1.5)
        ratio_ion_neutral = saha_factor * (U1 / U0) * jnp.exp(-IP_I / T_eV)
        frac_neutral = 1.0 / (1.0 + ratio_ion_neutral)
        frac_ion = ratio_ion_neutral / (1.0 + ratio_ion_neutral)

        el_idx = data.element_idx
        ion_stage = data.ion_stage
        pop_fraction = jnp.where(ion_stage == 0, frac_neutral[el_idx], frac_ion[el_idx])
        U_val = jnp.where(ion_stage == 0, U0[el_idx], U1[el_idx])

        element_conc = concentrations[el_idx]
        N_species_total = element_conc * total_species_density
        N_species = N_species_total * pop_fraction

        n_upper = N_species * (data.gk / U_val) * jnp.exp(-data.ek_ev / T_eV)

        epsilon = (
            (_JAX_H_PLANCK * _JAX_C_LIGHT / (4 * jnp.pi * data.wavelength_nm * _as_jax_real(1e-9)))
            * data.aki
            * n_upper
        )

        mass_kg = data.mass_amu * _JAX_M_PROTON
        sigma_doppler = data.wavelength_nm * jnp.sqrt(
            _as_jax_real(2.0) * T_eV * _JAX_EV_TO_J / (mass_kg * _JAX_C_LIGHT**2)
        )

        sigma_inst = _compute_instrument_sigma(
            data.wavelength_nm, self.instrument_fwhm_nm, self.resolving_power
        )
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        REF_NE = 1.0e16
        REF_T_EV = 0.86173
        binding_energy = jnp.maximum(data.ip_ev - data.ek_ev, 0.1)
        n_eff = (ion_stage + 1) * jnp.sqrt(13.605 / binding_energy)
        w_est = 2.0e-5 * (data.wavelength_nm / 500.0) ** 2 * (n_eff**4)
        w_est = jnp.clip(w_est, 0.0001, 0.5)
        w_ref = jnp.where(jnp.isnan(data.stark_w), w_est, data.stark_w)
        factor_ne = n_e / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -data.stark_alpha)
        gamma_stark = w_ref * factor_ne * factor_T

        diff = self.wavelength[:, None] - data.wavelength_nm[None, :]
        profile = _voigt_profile_kernel_jax(diff, sigma_total[None, :], gamma_stark[None, :])

        intensity = jnp.sum(epsilon * profile, axis=1)
        intensity = jnp.clip(intensity, 0.0, 1e12)
        return intensity


# ---------------------------------------------------------------------------
# Two-zone (core+shell) forward model
# ---------------------------------------------------------------------------


class TwoZoneBayesianForwardModel:
    """Two-zone plasma forward model for self-reversed LIBS spectra.

    Models a hot core (temperature ``T_core``) surrounded by a cooler shell
    (temperature ``T_shell``).

    .. math::

        I_{\\mathrm{obs}} = I_{\\mathrm{core}} e^{-\\tau_{\\mathrm{shell}}}
        + I_{\\mathrm{shell}} \\frac{1 - e^{-\\tau_{\\mathrm{shell}}}}{\\tau_{\\mathrm{shell}}}

    where the optical depth profile is computed from the absorption coefficient
    ``kappa_0 = (pi e^2 / m_e c) f n_lower``.
    """

    def __init__(
        self,
        db_path: str,
        elements: List[str],
        wavelength_range: Tuple[float, float],
        wavelength_grid: Optional[np.ndarray] = None,
        pixels: int = 2048,
        instrument_fwhm_nm: Optional[float] = None,
        resolving_power: Optional[float] = None,
    ):
        if not HAS_JAX:
            raise ImportError("JAX required. Install with: pip install jax jaxlib")

        has_fwhm = instrument_fwhm_nm is not None
        has_rp = resolving_power is not None
        if has_fwhm and has_rp:
            raise ValueError("resolving_power and instrument_fwhm_nm are mutually exclusive.")
        if not has_fwhm and not has_rp:
            instrument_fwhm_nm = 0.05

        self.elements = elements
        self.wavelength_range = wavelength_range
        self.instrument_fwhm_nm = instrument_fwhm_nm if instrument_fwhm_nm is not None else 0.05
        self.resolving_power = resolving_power

        if wavelength_grid is not None:
            self.wavelength = _as_jax_real(wavelength_grid)
        else:
            wl_min = _as_jax_real(wavelength_range[0])
            wl_max = _as_jax_real(wavelength_range[1])
            self.wavelength = jnp.linspace(wl_min, wl_max, pixels, dtype=_JAX_REAL_DTYPE)

        self.atomic_data = load_atomic_data(db_path, elements, wavelength_range)

        logger.info(
            f"TwoZoneBayesianForwardModel: {len(elements)} elements, "
            f"{len(self.wavelength)} wavelengths, "
            f"{len(self.atomic_data.wavelength_nm)} lines"
        )

    def _compute_zone_spectrum(
        self,
        T_eV: float,
        n_e: float,
        concentrations,
        total_species_density_cm3: Optional[float] = None,
    ):
        """Compute emission spectrum and absorption profile for one zone."""
        data = self.atomic_data
        T_eV = _as_jax_real(T_eV)
        n_e = _as_jax_real(n_e)
        concentrations = _as_jax_real(concentrations)
        T_K = T_eV * _JAX_EV_TO_K
        total_species_density = _resolve_total_species_density_cm3(n_e, total_species_density_cm3)

        U0 = partition_function(T_K, data.partition_coeffs[:, 0])
        U1 = partition_function(T_K, data.partition_coeffs[:, 1])
        IP_I = data.ionization_potentials[:, 0]

        saha_factor = (_JAX_SAHA_CONST_CM3 / n_e) * (T_eV**1.5)
        ratio_ion_neutral = saha_factor * (U1 / U0) * jnp.exp(-IP_I / T_eV)
        frac_neutral = 1.0 / (1.0 + ratio_ion_neutral)
        frac_ion = ratio_ion_neutral / (1.0 + ratio_ion_neutral)

        el_idx = data.element_idx
        ion_stage = data.ion_stage
        pop_fraction = jnp.where(ion_stage == 0, frac_neutral[el_idx], frac_ion[el_idx])
        U_val = jnp.where(ion_stage == 0, U0[el_idx], U1[el_idx])

        element_conc = concentrations[el_idx]
        N_species = element_conc * total_species_density * pop_fraction

        n_upper = N_species * (data.gk / U_val) * jnp.exp(-data.ek_ev / T_eV)

        epsilon = (
            (_JAX_H_PLANCK * _JAX_C_LIGHT / (4 * jnp.pi * data.wavelength_nm * _as_jax_real(1e-9)))
            * data.aki
            * n_upper
        )

        mass_kg = data.mass_amu * _JAX_M_PROTON
        sigma_doppler = data.wavelength_nm * jnp.sqrt(
            _as_jax_real(2.0) * T_eV * _JAX_EV_TO_J / (mass_kg * _JAX_C_LIGHT**2)
        )
        sigma_inst = _compute_instrument_sigma(
            data.wavelength_nm, self.instrument_fwhm_nm, self.resolving_power
        )
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        REF_NE = 1.0e16
        REF_T_EV = 0.86173
        binding_energy = jnp.maximum(data.ip_ev - data.ek_ev, 0.1)
        n_eff = (ion_stage + 1) * jnp.sqrt(13.605 / binding_energy)
        w_est = 2.0e-5 * (data.wavelength_nm / 500.0) ** 2 * (n_eff**4)
        w_est = jnp.clip(w_est, 0.0001, 0.5)
        w_ref = jnp.where(jnp.isnan(data.stark_w), w_est, data.stark_w)
        factor_ne = n_e / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -data.stark_alpha)
        gamma_stark = w_ref * factor_ne * factor_T

        diff = self.wavelength[:, None] - data.wavelength_nm[None, :]
        profile = _voigt_profile_kernel_jax(diff, sigma_total[None, :], gamma_stark[None, :])

        intensity = jnp.sum(epsilon * profile, axis=1)
        intensity = jnp.clip(intensity, 0.0, 1e12)

        ei_ev = data.ei_ev if data.ei_ev is not None else jnp.zeros_like(data.ek_ev)
        n_lower = N_species * (1.0 / U_val) * jnp.exp(-ei_ev / T_eV)
        f_osc = data.f_osc if data.f_osc is not None else jnp.ones_like(data.aki) * 1e-2
        kappa_0 = (jnp.pi * _JAX_E_CHARGE**2 / (_JAX_M_E * _JAX_C_LIGHT)) * f_osc * n_lower
        absorption = jnp.sum(kappa_0 * profile, axis=1)
        absorption = jnp.clip(absorption, 0.0, 1e12)

        return intensity, absorption

    def forward(
        self,
        T_core_eV: float,
        T_shell_eV: float,
        log_ne: float,
        concentrations,
        shell_fraction: float,
        optical_depth_scale: float,
        total_species_density_cm3: Optional[float] = None,
    ):
        """Compute observed spectrum from the two-zone model."""
        T_core_eV = _as_jax_real(T_core_eV)
        T_shell_eV = _as_jax_real(T_shell_eV)
        log_ne = _as_jax_real(log_ne)
        concentrations = _as_jax_real(concentrations)
        shell_fraction = _as_jax_real(shell_fraction)
        optical_depth_scale = _as_jax_real(optical_depth_scale)
        n_e = jnp.power(_as_jax_real(10.0), log_ne)

        I_core, _ = self._compute_zone_spectrum(
            T_core_eV, n_e, concentrations, total_species_density_cm3=total_species_density_cm3
        )
        I_shell, kappa_shell = self._compute_zone_spectrum(
            T_shell_eV, n_e, concentrations, total_species_density_cm3=total_species_density_cm3
        )

        tau_shell = kappa_shell * optical_depth_scale * shell_fraction
        tau_safe = jnp.maximum(tau_shell, 1e-30)
        exp_neg_tau = jnp.exp(-tau_safe)
        source_term = (1.0 - exp_neg_tau) / tau_safe
        I_obs = I_core * exp_neg_tau + I_shell * source_term
        return jnp.clip(I_obs, 0.0, 1e12)

    def forward_numpy(
        self,
        T_core_eV: float,
        T_shell_eV: float,
        log_ne: float,
        concentrations: np.ndarray,
        shell_fraction: float,
        optical_depth_scale: float,
        total_species_density_cm3: Optional[float] = None,
    ) -> np.ndarray:
        """NumPy wrapper for forward model (dynesty compatibility)."""
        conc_jax = _as_jax_real(concentrations)
        result = self.forward(
            T_core_eV,
            T_shell_eV,
            log_ne,
            conc_jax,
            shell_fraction,
            optical_depth_scale,
            total_species_density_cm3=total_species_density_cm3,
        )
        return np.array(result)


# ---------------------------------------------------------------------------
# Log-likelihood and NumPyro graph builders
# ---------------------------------------------------------------------------


def log_likelihood(
    predicted: Any,
    observed: Any,
    noise_params: NoiseParameters = NoiseParameters(),
) -> float:
    """Compute log-likelihood for an observed spectrum given a predicted one.

    Noise model combines Poisson shot noise and Gaussian readout noise:

    .. code-block:: text

        variance = predicted / gain + readout_noise^2 + dark_current
    """
    pred_safe = jnp.maximum(predicted, 1e-10)
    variance = (
        pred_safe / noise_params.gain + noise_params.readout_noise**2 + noise_params.dark_current
    )
    residual = observed - pred_safe
    log_lik = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance) + residual**2 / variance)
    return log_lik


def bayesian_model(
    forward_model: "BayesianForwardModel",
    observed,
    prior_config: PriorConfig = PriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
):
    """NumPyro probabilistic model for single-zone CF-LIBS Bayesian inference."""
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required. Install with: pip install numpyro")

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
    "BayesianForwardModel",
    "TwoZoneBayesianForwardModel",
    "log_likelihood",
    "bayesian_model",
    "two_zone_bayesian_model",
]

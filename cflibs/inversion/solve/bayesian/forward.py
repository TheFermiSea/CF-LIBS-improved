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
the guarded :func:`partition_function` delegator, :func:`mcwhirter_log_penalty`,
and a few JAX-real helpers) live in the sibling :mod:`atomic` module.  The
two-zone forward model threads each species' ``[t_min, t_max]`` window + ``g0``
floor (carried on :class:`AtomicDataArrays`) into that delegator so every
Bayesian ``U(T)`` is clamped/floored by the ONE shared guarded evaluator.

Notes
-----
After ADR-0001 T1-6, :meth:`BayesianForwardModel._compute_spectrum` calls
the unified forward kernel at :func:`cflibs.radiation.kernels.forward_model`
directly. The kernel consumes an :class:`cflibs.core.jax_runtime.AtomicSnapshot`
built from :meth:`cflibs.atomic.AtomicDatabase.snapshot`, eliminating the
:func:`_atomic_data_arrays_from_snapshot` adapter from the sampler path.
The legacy :class:`AtomicDataArrays` carrier (and :func:`load_atomic_data`)
remains on the module for back-compat exports and for two-zone callers that
still consume it. Convention change: the snapshot path uses Irwin (base-10
log) polynomial partition coefficients matching the canonical database
schema, whereas the pre-migration ``_compute_spectrum`` interpreted the
same coefficients as natural-log Irwin -- absolute spectrum scale therefore
differs from pre-T1-6 output. Downstream MCMC / dynesty consumers normalise
by likelihood so the migration is transparent to them.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from cflibs.core.constants import EV_TO_K

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
    partition_function,
)
from .likelihood import log_likelihood
from .priors import HAS_JAX

# ---------------------------------------------------------------------------
# Optional-deps gateway
# ---------------------------------------------------------------------------

if HAS_JAX:
    import jax.numpy as jnp

    from cflibs.radiation.profiles import _voigt_profile_kernel_jax
else:  # pragma: no cover
    jnp = None  # type: ignore[assignment]
    _voigt_profile_kernel_jax = None  # type: ignore[assignment]


def _stage_bound(bound: Any, stage_idx: int) -> Any:
    """Slice a ``(n_elements, n_stages)`` partition-guard array at ``stage_idx``.

    Returns ``None`` when the bound array is absent (legacy
    :class:`AtomicDataArrays` built without guard arrays), in which case the
    guarded evaluator falls back to its unbounded form for that stage.  The
    slice is a static array operation, so it is jit / vmap safe.
    """
    if bound is None:
        return None
    return bound[:, stage_idx]


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
    apply_self_absorption : bool, default False
        When True, route the unified kernel through its optically-thick
        radiative-transfer branch ``B_lambda * (1 - exp(-kappa * L))`` instead of
        the optically-thin emission sum. Default keeps the historical
        optically-thin behaviour; set True (with ``path_length_m`` > 0) to make
        self-absorption reachable from the Bayesian forward model.
    path_length_m : float, default 0.0
        Plasma path length (m) consulted only when ``apply_self_absorption`` is
        True.
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
        apply_self_absorption: bool = False,
        path_length_m: float = 0.0,
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

        if apply_self_absorption and path_length_m <= 0.0:
            raise ValueError(
                "apply_self_absorption=True requires path_length_m > 0 " f"(got {path_length_m})"
            )

        self.elements = elements
        self.wavelength_range = wavelength_range
        self.instrument_fwhm_nm = instrument_fwhm_nm if instrument_fwhm_nm is not None else 0.05
        self.resolving_power = resolving_power
        self.apply_self_absorption = apply_self_absorption
        self.path_length_m = path_length_m

        if wavelength_grid is not None:
            self.wavelength = _as_jax_real(wavelength_grid)
        else:
            wl_min = _as_jax_real(wavelength_range[0])
            wl_max = _as_jax_real(wavelength_range[1])
            self.wavelength = jnp.linspace(wl_min, wl_max, pixels, dtype=_JAX_REAL_DTYPE)

        self.atomic_data = load_atomic_data(db_path, elements, wavelength_range)

        # Build snapshot + instrument once for the unified forward kernel
        # (ADR-0001 T1-6). The snapshot's jit-friendly arrays flow through
        # :func:`cflibs.radiation.kernels.forward_model` on every call to
        # :meth:`_compute_spectrum`.
        from cflibs.atomic import AtomicDatabase  # noqa: PLC0415
        from cflibs.instrument.model import InstrumentModel  # noqa: PLC0415

        atomic_db = AtomicDatabase(db_path)
        self.snapshot = atomic_db.snapshot(
            elements=list(elements),
            wavelength_range=wavelength_range,
        )
        if self.resolving_power is not None:
            self.instrument = InstrumentModel(
                resolution_fwhm_nm=0.0,
                resolving_power=float(self.resolving_power),
            )
        else:
            self.instrument = InstrumentModel(
                resolution_fwhm_nm=float(self.instrument_fwhm_nm),
            )

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

    def _compute_spectrum(
        self,
        T_eV: float,
        n_e: float,
        concentrations: Any,
        total_species_density_cm3: Optional[float] = None,
    ):
        """Compute spectrum via the unified forward kernel (ADR-0001 T1-6).

        Routes to :func:`cflibs.radiation.kernels.forward_model` with
        ``BroadeningMode.PHYSICAL_DOPPLER`` plus per-line Stark Voigt and
        instrument-sigma folding -- the same physics knobs the legacy
        direct-summation path used. The legacy
        :func:`_atomic_data_arrays_from_snapshot` adapter is no longer
        invoked from this code path.

        Batching contract
        -----------------
        The single supported batching mode is **JAX ``vmap``**, as used
        internally by NumPyro NUTS when ``MCMCSampler`` is configured
        with ``chain_method='vectorized'`` (the project default; see
        :func:`test_mcmc_sampler_default_chain_method_is_vectorized`).
        Under that path JAX adds a *leading* chain axis to every traced
        input, so ``concentrations`` typically arrives with shape
        ``(num_chains, n_elements)``. The species-density broadcast
        below tolerates that leading axis transparently via the
        ``[..., i]`` element selector, and the resulting
        ``plasma_state.species[el]`` leaves carry the chain axis
        through ``forward_model``.

        **Manual batching is NOT supported.** Calling
        ``_compute_spectrum`` directly with a Python-side stacked
        ``concentrations`` of shape ``(batch, n_elements)`` is not a
        public path: in that scenario ``total_species_density`` is a
        scalar and does *not* broadcast against the leading batch axis
        on its own (PR #186's species-dict refactor is benign for vmap
        but does not enable a free manual-batching mode -- the existing
        ``test_compute_spectrum_supports_vmap_chain_axis`` test passes
        with both the pre-PR and post-PR bodies because vmap re-traces
        the function under the leading axis either way). Any future
        caller that wants explicit (non-vmap) batching must:

        1. Reshape ``total_species_density`` to
           ``total_species_density[..., None]`` *before* the
           ``concentrations * total_species_density`` product, so the
           scalar density broadcasts cleanly against the leading batch
           axis.
        2. Also batch the scalar ``T_eV`` and ``n_e`` inputs --
           ``forward_model`` currently assumes scalar JAX tracers, not
           arrays, for those.

        In short: if you need to run many compositions concurrently,
        wrap this method in ``jax.vmap`` rather than stacking arrays
        in Python. The dict-comprehension over ``self.elements`` plus
        the ``[..., i]`` indexing has been deliberately chosen for
        vmap compatibility; do not "optimize" it back to per-index
        scalar lookups (``concentrations[i] * total_density``) without
        verifying both the vmap test
        (:func:`test_compute_spectrum_supports_vmap_chain_axis`) and
        a fresh manual-batch test still hold.
        """
        from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: PLC0415
        from cflibs.radiation.kernels import forward_model  # noqa: PLC0415
        from cflibs.radiation.profiles import BroadeningMode  # noqa: PLC0415

        T_eV = _as_jax_real(T_eV)
        n_e = _as_jax_real(n_e)
        concentrations = _as_jax_real(concentrations)
        total_species_density = _resolve_total_species_density_cm3(n_e, total_species_density_cm3)

        # Build a SingleZoneLTEPlasma whose pytree leaves carry the traced
        # MCMC inputs. Bypass ``__init__`` -- it logs an f-string formatted
        # against T_e, which fails on JAX tracers.
        plasma_state = object.__new__(SingleZoneLTEPlasma)
        plasma_state.T_e = T_eV * EV_TO_K
        plasma_state.n_e = n_e
        # Compute element densities by broadcast, then split along the element
        # axis. The ``[..., i]`` indexing pattern preserves any leading batch
        # axes added by ``vmap`` (e.g. NUTS ``chain_method='vectorized'``).
        density_per_element = concentrations * total_species_density
        plasma_state.species = {
            el: density_per_element[..., i] for i, el in enumerate(self.elements)
        }
        plasma_state.T_g = None
        plasma_state.pressure = None

        intensity = forward_model(
            plasma_state,
            self.snapshot,
            self.instrument,
            self.wavelength,
            broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
            path_length_m=self.path_length_m,
            apply_self_absorption=self.apply_self_absorption,
            fold_instrument_sigma=True,
            apply_stark=True,
            total_species_density_cm3=total_species_density,
        )
        # Preserve the legacy non-negativity / overflow guard so downstream
        # likelihoods see clipped intensities.
        return jnp.clip(intensity, 0.0, 1e12)


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

        # Guarded U(T): clamp T to the per-species [t_min, t_max] fit window
        # and floor at g0 (the SAME guard + coefficients the manifold snapshot
        # bakes; routed through the ONE shared evaluator).  The bounds arrive as
        # static (n_elements, n_stages) arrays on ``AtomicDataArrays`` so no
        # Python provider call enters the NUTS jit trace.
        U0 = partition_function(
            T_K,
            data.partition_coeffs[:, 0],
            t_min=_stage_bound(data.partition_t_min, 0),
            t_max=_stage_bound(data.partition_t_max, 0),
            g0=_stage_bound(data.partition_g0, 0),
        )
        U1 = partition_function(
            T_K,
            data.partition_coeffs[:, 1],
            t_min=_stage_bound(data.partition_t_min, 1),
            t_max=_stage_bound(data.partition_t_max, 1),
            g0=_stage_bound(data.partition_g0, 1),
        )
        U2 = partition_function(
            T_K,
            data.partition_coeffs[:, 2],
            t_min=_stage_bound(data.partition_t_min, 2),
            t_max=_stage_bound(data.partition_t_max, 2),
            g0=_stage_bound(data.partition_g0, 2),
        )
        IP_I = data.ionization_potentials[:, 0]
        IP_II = data.ionization_potentials[:, 1]

        # Saha balance over neutral / singly / doubly ionized (John et al. 2023).
        # Truncating at the singly-ionized stage under-counts the ladder for
        # low-second-IP elements (e.g. Ca II->III IP = 11.87 eV, Mg) whose
        # doubly-ionized fraction exceeds 1% at core T ~ 1.3 eV. Triply+ is
        # safely negligible (<1%) in the ps-LIBS regime, so we stop at III.
        # Step ratios (each relative to the next-lower stage) share the same
        # Saha prefactor (S / n_e) * T_eV^1.5; the n_e in the denominator is the
        # sampled electron density, consistent with the charge-balance closure
        # n_e = n_I*0 + n_II*1 + n_III*2 = n_II + 2 n_III enforced implicitly
        # through the Saha n_e dependence (see test_saha_charge_balance_holds).
        saha_factor = (_JAX_SAHA_CONST_CM3 / n_e) * (T_eV**1.5)
        r1 = saha_factor * (U1 / U0) * jnp.exp(-IP_I / T_eV)  # n_II / n_I
        r2 = saha_factor * (U2 / U1) * jnp.exp(-IP_II / T_eV)  # n_III / n_II
        denom = 1.0 + r1 + r1 * r2
        frac_neutral = 1.0 / denom
        frac_ion = r1 / denom
        frac_doubly = (r1 * r2) / denom

        el_idx = data.element_idx
        ion_stage = data.ion_stage
        pop_fraction = jnp.where(
            ion_stage == 0,
            frac_neutral[el_idx],
            jnp.where(ion_stage == 1, frac_ion[el_idx], frac_doubly[el_idx]),
        )
        U_val = jnp.where(
            ion_stage == 0,
            U0[el_idx],
            jnp.where(ion_stage == 1, U1[el_idx], U2[el_idx]),
        )

        element_conc = concentrations[el_idx]
        N_species = element_conc * total_species_density * pop_fraction

        # N_species is in cm⁻³ (project convention). The emissivity formula
        # below uses SI constants (h, c, λ in m) and therefore needs n_upper
        # in m⁻³. The single-zone kernels.forward_model path applies this
        # same `* 1.0e6` conversion at its emissivity site (kernels.py:506);
        # the TwoZone path was missing it, making every TwoZone emission
        # 10⁶× too small. Bug surfaced 2026-05-19 by AI physics review.
        n_upper = N_species * (data.gk / U_val) * jnp.exp(-data.ek_ev / T_eV)
        n_upper_m3 = n_upper * _as_jax_real(1.0e6)

        epsilon = (
            (_JAX_H_PLANCK * _JAX_C_LIGHT / (4 * jnp.pi * data.wavelength_nm * _as_jax_real(1e-9)))
            * data.aki
            * n_upper_m3
        )

        mass_kg = data.mass_amu * _JAX_M_PROTON
        # 1D Maxwell-Boltzmann standard deviation: sigma = (lambda/c) * sqrt(kT/m).
        # The earlier sqrt(2 kT/m) form is the most-probable-speed v_p, NOT the
        # std of the line-of-sight velocity distribution; it over-broadened the
        # Gaussian core by sqrt(2) ~ 1.414. Canonical source of truth:
        # cflibs/radiation/profiles.doppler_width / doppler_sigma_jax and the
        # single-zone kernel cflibs/radiation/kernels._per_line_doppler_sigma,
        # both of which use sqrt(kT/m) with no factor of 2.
        sigma_doppler = data.wavelength_nm * jnp.sqrt(
            T_eV * _JAX_EV_TO_J / (mass_kg * _JAX_C_LIGHT**2)
        )
        sigma_inst = _compute_instrument_sigma(
            data.wavelength_nm, self.instrument_fwhm_nm, self.resolving_power
        )
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        # ``data.stark_w`` (and the w_est fallback) is the stored electron-impact
        # FWHM at REF_NE = 1e17 cm^-3, T = 10000 K (single source of truth:
        # cflibs/radiation/stark.py). The Voigt kernel wants a Lorentzian HWHM,
        # so scale to live n_e and halve. The earlier (n_e/1e16) with no 0.5
        # over-broadened every Stark line by x20 at ps-LIBS densities (A4-CONV-2).
        REF_NE = 1.0e17
        REF_T_EV = 0.86173
        binding_energy = jnp.maximum(data.ip_ev - data.ek_ev, 0.1)
        n_eff = (ion_stage + 1) * jnp.sqrt(13.605 / binding_energy)
        w_est = 2.0e-5 * (data.wavelength_nm / 500.0) ** 2 * (n_eff**4)
        w_est = jnp.clip(w_est, 0.0001, 0.5)
        w_ref = jnp.where(jnp.isnan(data.stark_w), w_est, data.stark_w)
        factor_ne = n_e / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -data.stark_alpha)
        gamma_stark = 0.5 * w_ref * factor_ne * factor_T

        diff = self.wavelength[:, None] - data.wavelength_nm[None, :]
        profile = _voigt_profile_kernel_jax(diff, sigma_total[None, :], gamma_stark[None, :])

        intensity = jnp.sum(epsilon * profile, axis=1)
        intensity = jnp.clip(intensity, 0.0, 1e12)

        ei_ev = data.ei_ev if data.ei_ev is not None else jnp.zeros_like(data.ek_ev)
        # Same cm⁻³→m⁻³ conversion as n_upper above — kappa_0 uses SI
        # constants so n_lower must be in m⁻³.
        n_lower = N_species * (1.0 / U_val) * jnp.exp(-ei_ev / T_eV)
        n_lower_m3 = n_lower * _as_jax_real(1.0e6)
        f_osc = data.f_osc if data.f_osc is not None else jnp.ones_like(data.aki) * 1e-2
        kappa_0 = (jnp.pi * _JAX_E_CHARGE**2 / (_JAX_M_E * _JAX_C_LIGHT)) * f_osc * n_lower_m3
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
# Re-exports
#
# The module-level :func:`log_likelihood` (and the exact Poisson/Cash variant
# :func:`_poisson_cash_log_likelihood`) live in the sibling :mod:`likelihood`
# module; the NumPyro graph builders :func:`bayesian_model` /
# :func:`two_zone_bayesian_model` live in :mod:`models`. They are re-exported
# here for back-compat so the historical ``from ...bayesian.forward import X``
# import paths keep working. The split keeps every ``bayesian/*.py`` file under
# the 800-LOC limit required by ADR-0001 / T1-6 spec section 6.
# ---------------------------------------------------------------------------

from .models import bayesian_model, two_zone_bayesian_model  # noqa: E402

__all__ = [
    "BayesianForwardModel",
    "TwoZoneBayesianForwardModel",
    "log_likelihood",
    "bayesian_model",
    "two_zone_bayesian_model",
]

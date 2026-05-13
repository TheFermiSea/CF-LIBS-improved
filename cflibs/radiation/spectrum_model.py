"""
Forward spectrum model that ties together all components.
"""

import numpy as np
from typing import Optional, Tuple

from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver, SahaBoltzmannSolverJax
from cflibs.atomic.database import AtomicDatabase
from cflibs.instrument.model import InstrumentModel
from cflibs.radiation.profiles import (
    BroadeningMode,
    doppler_width,
)
from cflibs.instrument.convolution import apply_instrument_function
from cflibs.core.constants import H_PLANCK, C_LIGHT, KB, EV_TO_K
from cflibs.core.logging_config import get_logger

jit = jit_if_available  # local alias preserves existing @jit decorator sites

logger = get_logger("radiation.spectrum_model")


def _emissivity_line_table(transitions: list, populations: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Build (line_wavelengths_nm, line_emissivities) arrays for LDM dispatch.

    Mirrors the population-lookup + ``ε = (hc / 4πλ) · A_ki · n_k`` loop in
    :func:`cflibs.radiation.emissivity.calculate_spectrum_emissivity` but
    returns the per-line catalog instead of immediately broadening — the
    LDM kernel needs the raw line list. Transitions without an entry in
    ``populations`` are skipped (same convention as the legacy path).
    """
    wls: list[float] = []
    emis: list[float] = []
    for trans in transitions:
        key = (trans.element, trans.ionization_stage, round(trans.E_k_ev, 8))
        if key not in populations:
            continue
        n_k = populations[key]
        wl_m = trans.wavelength_nm * 1e-9
        n_k_m3 = n_k * 1e6
        epsilon = (H_PLANCK * C_LIGHT / (4 * np.pi * wl_m)) * trans.A_ki * n_k_m3
        wls.append(trans.wavelength_nm)
        emis.append(epsilon)
    return np.asarray(wls, dtype=np.float64), np.asarray(emis, dtype=np.float64)


def planck_radiance(wavelength_nm: np.ndarray, T_eV: float) -> np.ndarray:
    """
    Calculate spectral radiance of a blackbody in W m^-2 nm^-1 sr^-1.
    """
    wl_m = wavelength_nm * 1e-9
    T_K = T_eV * EV_TO_K

    # B_lambda = (2hc^2 / lambda^5) / (exp(hc / (lambda k T)) - 1)
    # Units: W m^-3 sr^-1. To get W m^-2 nm^-1 sr^-1, multiply by 1e-9.
    exponent = (H_PLANCK * C_LIGHT) / (wl_m * KB * T_K)
    exponent = np.clip(exponent, None, 700.0)  # avoid overflow

    B_m3 = (2.0 * H_PLANCK * C_LIGHT**2 / (wl_m**5)) / (np.exp(exponent) - 1.0)
    return B_m3 * 1e-9


class SpectrumModel:
    """
    Forward model for computing synthetic LIBS spectra.

    This class integrates:
    - Plasma state (temperature, density, composition)
    - Saha-Boltzmann solver (ionization and excitation balance)
    - Line emissivity calculations
    - Instrument response and convolution
    """

    def __init__(
        self,
        plasma: SingleZoneLTEPlasma,
        atomic_db: AtomicDatabase,
        instrument: InstrumentModel,
        lambda_min: float,
        lambda_max: float,
        delta_lambda: float,
        path_length_m: float = 0.01,  # 1 cm default
        use_jax: bool = False,
        broadening_mode: BroadeningMode = BroadeningMode.LEGACY,
    ):
        """
        Initialize spectrum model.

        Parameters
        ----------
        plasma : SingleZoneLTEPlasma
            Plasma state
        atomic_db : AtomicDatabase
            Atomic database
        instrument : InstrumentModel
            Instrument model
        lambda_min : float
            Minimum wavelength in nm
        lambda_max : float
            Maximum wavelength in nm
        delta_lambda : float
            Wavelength step in nm
        path_length_m : float
            Plasma path length in meters (for optically thin approximation)
        use_jax : bool
            Use JAX acceleration for broadening when available
        broadening_mode : BroadeningMode
            Broadening mode. LEGACY uses a single scalar sigma plus
            downstream instrument convolution. NIST_PARITY uses per-line
            sigma from resolving power (no downstream convolution).
            PHYSICAL_DOPPLER uses per-line physical Doppler width plus
            downstream instrument convolution.
        """
        if broadening_mode == BroadeningMode.NIST_PARITY and not instrument.is_resolving_power_mode:
            raise ValueError(
                "NIST_PARITY broadening mode requires InstrumentModel with resolving_power set. "
                "Use InstrumentModel.from_resolving_power(R) or set resolving_power field."
            )

        self.plasma = plasma
        self.atomic_db = atomic_db
        self.instrument = instrument
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.delta_lambda = delta_lambda
        self.path_length_m = path_length_m
        self.use_jax = use_jax
        self.broadening_mode = broadening_mode

        # Create wavelength grid
        self.wavelength = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)

        # Initialize solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        logger.info(
            f"Initialized SpectrumModel: λ=[{lambda_min:.1f}, {lambda_max:.1f}] nm, "
            f"Δλ={delta_lambda:.3f} nm, {len(self.wavelength)} points, "
            f"mode={broadening_mode.value}"
        )

    _FALLBACK_MASSES = {
        "H": 1.008,
        "He": 4.003,
        "Li": 6.941,
        "Be": 9.012,
        "B": 10.81,
        "C": 12.01,
        "N": 14.01,
        "O": 16.00,
        "Na": 22.99,
        "Mg": 24.31,
        "Al": 26.98,
        "Si": 28.09,
        "P": 30.97,
        "S": 32.07,
        "K": 39.10,
        "Ca": 40.08,
        "Ti": 47.87,
        "V": 50.94,
        "Cr": 52.00,
        "Mn": 54.94,
        "Fe": 55.85,
        "Co": 58.93,
        "Ni": 58.69,
        "Cu": 63.55,
        "Zn": 65.38,
        "Sr": 87.62,
        "Ba": 137.33,
        "W": 183.84,
    }

    def _get_element_mass(self, element: str) -> float:
        """Get atomic mass for an element, with fallback."""
        mass = self.atomic_db.get_atomic_mass(element)
        if mass is not None:
            return mass
        fallback = self._FALLBACK_MASSES.get(element)
        if fallback is not None:
            logger.warning("No DB mass for %s; using fallback %.2f amu", element, fallback)
            return fallback
        raise ValueError(
            f"No atomic mass found for element {element!r} in database or fallback table"
        )

    def _compute_sigma_per_line(self, transitions: list, populations: dict) -> Optional[np.ndarray]:
        """
        Compute per-line Gaussian sigma array based on broadening mode.

        Returns None for LEGACY mode (use scalar sigma instead).
        """
        if self.broadening_mode == BroadeningMode.LEGACY:
            return None

        # Collect sigmas only for transitions that have populations
        sigmas = []
        for trans in transitions:
            key = (trans.element, trans.ionization_stage, round(trans.E_k_ev, 8))
            if key not in populations:
                continue

            if self.broadening_mode == BroadeningMode.NIST_PARITY:
                sig = self.instrument.sigma_at_wavelength(trans.wavelength_nm)
            elif self.broadening_mode in (
                BroadeningMode.PHYSICAL_DOPPLER,
                BroadeningMode.LDM_GAUSSIAN,
            ):
                mass = self._get_element_mass(trans.element)
                fwhm = doppler_width(trans.wavelength_nm, self.plasma.T_e_eV, mass)
                sig = fwhm / 2.355
            else:
                raise ValueError(f"Unsupported broadening mode: {self.broadening_mode!r}")

            sigmas.append(sig)

        return np.array(sigmas) if sigmas else np.array([])

    def compute_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute synthetic spectrum.

        Returns
        -------
        wavelength : array
            Wavelength grid in nm
        intensity : array
            Spectral intensity in W m^-2 nm^-1 sr^-1

        Notes
        -----
        Thin wrapper over the unified :func:`cflibs.radiation.kernels.forward_model`
        kernel (ADR-0001 T1-2). Saha-Boltzmann populations are still computed
        by the detailed-levels :class:`SahaBoltzmannSolver` so that this code
        path remains numerically identical (rtol<1e-12, atol<1e-7) to its
        pre-T1-2 output. The legacy populations dict is converted to a
        per-line ``n_upper`` array and injected into the kernel via the
        ``_precomputed_n_upper_per_line`` parameter; the kernel then handles
        the per-line broadening + optional radiative-transfer step.

        Downstream scipy instrument convolution is retained for LEGACY /
        PHYSICAL_DOPPLER / LDM_GAUSSIAN modes (NIST_PARITY folds instrument
        broadening into the per-line sigma).
        """
        from cflibs.radiation.kernels import forward_model

        # Validate plasma
        self.plasma.validate()

        # 1. Solve Saha-Boltzmann for level populations (detailed-levels path).
        logger.debug("Solving Saha-Boltzmann equations...")
        populations = self.solver.solve_plasma(self.plasma)

        # 2. Build the AtomicSnapshot with the same min_relative_intensity
        #    filter the legacy path applied per element.
        min_ri = 0.01 if self.broadening_mode == BroadeningMode.NIST_PARITY else 10.0
        snapshot = self.atomic_db.snapshot(
            elements=list(self.plasma.species.keys()),
            wavelength_range=(self.lambda_min, self.lambda_max),
            min_relative_intensity=min_ri,
        )
        n_lines = int(np.asarray(snapshot.line_wavelengths_nm).shape[0])
        logger.debug(f"Snapshot has {n_lines} transitions")

        # 3. Convert dict-based populations -> per-line n_upper array so the
        #    kernel can consume them without re-running Saha-Boltzmann.
        n_upper_per_line = np.zeros(n_lines, dtype=np.float64)
        if n_lines:
            line_E_k = np.asarray(snapshot.line_E_k_ev)
            line_sp_idx = np.asarray(snapshot.line_species_index)
            for li in range(n_lines):
                el, stage = snapshot.species[int(line_sp_idx[li])]
                key = (el, stage, round(float(line_E_k[li]), 8))
                n_upper_per_line[li] = populations.get(key, 0.0)

        # 4. Optional LDM sigma_grid (only for LDM_GAUSSIAN dispatch).
        sigma_grid = None
        if self.broadening_mode == BroadeningMode.LDM_GAUSSIAN and n_lines:
            from cflibs.radiation.ldm import build_sigma_grid

            line_mass_amu = np.array(
                [self._get_element_mass(el) for el, _stage in snapshot.species],
                dtype=np.float64,
            )[np.asarray(snapshot.line_species_index)]
            T_eV = self.plasma.T_e_eV
            wl_nm = np.asarray(snapshot.line_wavelengths_nm)
            sigma_D = wl_nm * np.sqrt(
                (T_eV * 1.602176634e-19) / (line_mass_amu * 1.67262192369e-27 * (2.99792458e8) ** 2)
            )
            sigma_D = np.maximum(sigma_D, 1e-6)
            sigma_grid = build_sigma_grid(sigma_D)

        # 5. Run the unified kernel. The four broadening modes map to:
        #      LEGACY            -> scalar Gaussian sigma; downstream conv.
        #      NIST_PARITY       -> per-line instrument sigma; no downstream.
        #      PHYSICAL_DOPPLER  -> per-line Doppler sigma; no Stark; no fold;
        #                          downstream conv.
        #      LDM_GAUSSIAN      -> LDM/DIT Gaussian path; downstream conv.
        wl_jnp = jnp.asarray(self.wavelength, dtype=jnp.float64) if HAS_JAX else self.wavelength
        if n_lines:
            intensity = forward_model(
                self.plasma,
                snapshot,
                self.instrument,
                wl_jnp,
                sigma_grid=sigma_grid,
                broadening_mode=self.broadening_mode,
                path_length_m=self.path_length_m,
                apply_self_absorption=True,
                fold_instrument_sigma=(self.broadening_mode == BroadeningMode.NIST_PARITY),
                apply_stark=False,
                _precomputed_n_upper_per_line=n_upper_per_line,
            )
            intensity = np.asarray(intensity)
        else:
            # No transitions in band: emit zeros, then optionally Planck-RT-
            # squelch through the same RT step. Path matches legacy behaviour.
            intensity = np.zeros_like(self.wavelength)
            if self.path_length_m > 0:
                B_lambda = planck_radiance(self.wavelength, self.plasma.T_e_eV)
                intensity = B_lambda * 0.0

        # 6. Apply instrument response curve (host-side multiplication; not
        #    part of the kernel because it is data-driven).
        if self.instrument.response_curve is not None:
            logger.debug("Applying instrument response...")
            intensity = self.instrument.apply_response(self.wavelength, intensity)

        # 7. Apply downstream instrument convolution for the modes that need
        #    it (NIST_PARITY folds instrument broadening into per-line sigma,
        #    so it skips this step).
        if self.broadening_mode != BroadeningMode.NIST_PARITY:
            if self.instrument.is_resolving_power_mode:
                mid_wl = 0.5 * (self.lambda_min + self.lambda_max)
                sigma_conv = self.instrument.sigma_at_wavelength(mid_wl)
            else:
                sigma_conv = self.instrument.resolution_sigma_nm

            if sigma_conv > 0:
                logger.debug("Applying instrument function (sigma=%.4f nm)...", sigma_conv)
                if self.use_jax:
                    from cflibs.instrument.convolution import apply_instrument_function_jax

                    intensity = apply_instrument_function_jax(
                        self.wavelength, intensity, sigma_conv
                    )
                else:
                    intensity = apply_instrument_function(self.wavelength, intensity, sigma_conv)
            else:
                logger.debug("Skipping instrument convolution (sigma=0)")
        else:
            logger.debug("Skipping instrument convolution (NIST_PARITY mode)")

        logger.info("Spectrum computation complete")

        return self.wavelength, intensity


# ---------------------------------------------------------------------------
# JAX-accelerated forward spectrum model
# ---------------------------------------------------------------------------
#
# The JAX variant fuses the per-wavelength operations that follow the
# Saha-Boltzmann solve into a single jit'd kernel:
#
#   1. Per-line Gaussian broadening (vmap over lines, sum reduction)
#   2. Planck radiance evaluation
#   3. Uniform-slab radiative transfer: I = B * (1 - exp(-kappa * L))
#   4. Optional Gaussian instrument convolution
#
# This is the hot path that previously ran 100% on CPU regardless of
# ``JAX_PLATFORMS`` because every step lived in NumPy. The numerical
# behaviour is identical (within float precision) to ``SpectrumModel`` —
# the unit test ``tests/radiation/test_spectrum_model_jax.py`` enforces
# ``rtol=1e-5, atol=1e-7`` parity on a multi-element synthetic plasma.


if HAS_JAX:

    @jit
    def _planck_radiance_jax(wavelength_nm: jnp.ndarray, T_eV: jnp.ndarray) -> jnp.ndarray:
        """JAX Planck radiance in W m^-2 nm^-1 sr^-1 — identical to the NumPy form."""
        wl_m = wavelength_nm * 1e-9
        T_K = T_eV * EV_TO_K
        exponent = (H_PLANCK * C_LIGHT) / (wl_m * KB * T_K)
        exponent = jnp.clip(exponent, max=700.0)  # avoid overflow
        # Use the same (exp(x) - 1) form as the NumPy reference — keeps
        # bit-for-bit numerical parity at the rtol=1e-5 tolerance.
        B_m3 = (2.0 * H_PLANCK * C_LIGHT**2 / (wl_m**5)) / (jnp.exp(exponent) - 1.0)
        return B_m3 * 1e-9

    @jit
    def _broaden_per_line_jax(
        wavelength_grid: jnp.ndarray,
        line_wavelengths: jnp.ndarray,
        line_intensities: jnp.ndarray,
        sigmas: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sum of per-line Gaussians on the wavelength grid.

        Implementation uses a (N_wl, N_lines) outer-product broadcast so the
        whole emissivity calculation is one BLAS-friendly tensor op — the
        kind of compute pattern XLA fuses extremely well on GPU.
        """
        diff = wavelength_grid[:, None] - line_wavelengths[None, :]
        sig = jnp.maximum(sigmas, 1e-12)[None, :]
        x = diff / sig
        norm = sig * jnp.sqrt(2.0 * jnp.pi)
        profiles = jnp.exp(-0.5 * x * x) / norm
        weighted = line_intensities[None, :] * profiles
        return jnp.sum(weighted, axis=1)

    @jit
    def _radiative_transfer_jax(
        wavelength: jnp.ndarray,
        emissivity: jnp.ndarray,
        T_eV: jnp.ndarray,
        path_length_m: jnp.ndarray,
    ) -> jnp.ndarray:
        """Uniform-slab radiative transfer — matches the NumPy expression."""
        B = _planck_radiance_jax(wavelength, T_eV)
        kappa = emissivity / (B + 1e-100)
        return B * (-jnp.expm1(-kappa * path_length_m))

    @jit
    def _gaussian_kernel_jax(
        sigma_nm: jnp.ndarray, delta_wl: jnp.ndarray, kernel_size: int
    ) -> jnp.ndarray:
        """Gaussian convolution kernel — same shape and normalisation as the NumPy version."""
        n_sigma = 5.0
        kernel_wl = jnp.linspace(-n_sigma * sigma_nm, n_sigma * sigma_nm, kernel_size)
        kernel = jnp.exp(-0.5 * (kernel_wl / sigma_nm) ** 2)
        return kernel / jnp.sum(kernel)

else:  # pragma: no cover - JAX should be installed in this repo

    def _planck_radiance_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _broaden_per_line_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _radiative_transfer_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")


def planck_radiance_jax(wavelength_nm, T_eV) -> "jnp.ndarray":
    """Public JAX entry point for the Planck radiance.

    Mirrors :func:`planck_radiance` but operates on jnp arrays end-to-end
    so callers can compose it with other JAX kernels without paying the
    H2D copy cost on every call.
    """
    if not HAS_JAX:  # pragma: no cover
        raise ImportError("JAX is not installed; install jax + jaxlib")
    return _planck_radiance_jax(jnp.asarray(wavelength_nm), jnp.asarray(T_eV))


class SpectrumModelJax(SpectrumModel):
    """JAX-accelerated drop-in companion to :class:`SpectrumModel`.

    The ionization balance and atomic-data lookups still go through the
    Python ``SahaBoltzmannSolverJax`` (which produces JAX-evaluated values
    but materialises them as Python floats at the boundary). The
    wavelength-grid-sized arithmetic — emissivity broadening, Planck
    radiance, radiative transfer and instrument convolution — runs on
    ``jax.numpy`` arrays through ``_broaden_per_line_jax``,
    ``_radiative_transfer_jax`` and ``apply_instrument_function_jax``.

    Numerical equivalence with :class:`SpectrumModel` is asserted by
    ``tests/radiation/test_spectrum_model_jax.py`` within
    ``rtol=1e-5, atol=1e-7``.
    """

    def __init__(
        self,
        plasma: SingleZoneLTEPlasma,
        atomic_db: AtomicDatabase,
        instrument: InstrumentModel,
        lambda_min: float,
        lambda_max: float,
        delta_lambda: float,
        path_length_m: float = 0.01,
        broadening_mode: BroadeningMode = BroadeningMode.LEGACY,
    ):
        if not HAS_JAX:  # pragma: no cover - defensive
            raise ImportError("SpectrumModelJax requires JAX. Install with `pip install jax`.")
        # Initialise the parent so all attributes (wavelength grid,
        # validation, fallback masses, ...) are inherited verbatim.
        super().__init__(
            plasma=plasma,
            atomic_db=atomic_db,
            instrument=instrument,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            delta_lambda=delta_lambda,
            path_length_m=path_length_m,
            use_jax=True,  # informational; we own the JAX path
            broadening_mode=broadening_mode,
        )
        # Replace the NumPy solver with the JAX one — public surface is the
        # same so downstream code (acceptance tests, benchmark harness)
        # doesn't notice.
        self.solver = SahaBoltzmannSolverJax(atomic_db)
        # Pre-stage the wavelength grid as a jnp array so the convolution
        # path doesn't pay H2D cost on every call.
        self._wavelength_jax = jnp.asarray(self.wavelength, dtype=jnp.float64)

    def compute_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute synthetic spectrum on the configured wavelength grid.

        Returns NumPy arrays so existing callers (acceptance tests, the
        benchmark gate, the Bayesian inverter) don't have to special-case
        the device-array boundary.
        """
        self.plasma.validate()

        # 1. Saha-Boltzmann (uses JAX kernels internally).
        logger.debug("Solving Saha-Boltzmann (JAX)...")
        populations = self.solver.solve_plasma(self.plasma)

        # 2. Transitions — same filter logic as the NumPy variant.
        logger.debug("Loading transitions...")
        all_transitions = []
        for element in self.plasma.species.keys():
            min_ri = 0.01 if self.broadening_mode == BroadeningMode.NIST_PARITY else 10.0
            transitions = self.atomic_db.get_transitions(
                element,
                wavelength_min=self.lambda_min,
                wavelength_max=self.lambda_max,
                min_relative_intensity=min_ri,
            )
            all_transitions.extend(transitions)

        # 3. Per-line emissivity table (NumPy is fine here — it is a small
        #    array indexed by transition; the heavy work is the broadening).
        line_wavelengths_list = []
        line_emissivities_list = []
        line_sigmas_list = []
        for trans in all_transitions:
            key = (trans.element, trans.ionization_stage, round(trans.E_k_ev, 8))
            if key not in populations:
                continue
            n_k = populations[key]
            wl_m = trans.wavelength_nm * 1e-9
            n_k_m3 = n_k * 1e6
            epsilon = (H_PLANCK * C_LIGHT / (4 * np.pi * wl_m)) * trans.A_ki * n_k_m3
            line_wavelengths_list.append(trans.wavelength_nm)
            line_emissivities_list.append(epsilon)
            if self.broadening_mode == BroadeningMode.LEGACY:
                T_eV = self.plasma.T_e_eV
                line_sigmas_list.append(0.01 * np.sqrt(T_eV / 0.86))
            elif self.broadening_mode == BroadeningMode.NIST_PARITY:
                line_sigmas_list.append(self.instrument.sigma_at_wavelength(trans.wavelength_nm))
            elif self.broadening_mode == BroadeningMode.PHYSICAL_DOPPLER:
                mass = self._get_element_mass(trans.element)
                fwhm = doppler_width(trans.wavelength_nm, self.plasma.T_e_eV, mass)
                line_sigmas_list.append(fwhm / 2.355)
            else:  # pragma: no cover - already validated in __init__
                raise ValueError(f"Unsupported broadening mode: {self.broadening_mode!r}")

        if not line_wavelengths_list:
            zero = np.zeros_like(self.wavelength)
            return self.wavelength, zero

        line_wl_jnp = jnp.asarray(line_wavelengths_list, dtype=jnp.float64)
        line_em_jnp = jnp.asarray(line_emissivities_list, dtype=jnp.float64)
        line_sig_jnp = jnp.asarray(line_sigmas_list, dtype=jnp.float64)

        # 4. Per-line broadening — full GPU work.
        emissivity_jnp = _broaden_per_line_jax(
            self._wavelength_jax, line_wl_jnp, line_em_jnp, line_sig_jnp
        )

        # 5. Radiative transfer (Planck + slab self-absorption).
        intensity_jnp = _radiative_transfer_jax(
            self._wavelength_jax,
            emissivity_jnp,
            jnp.asarray(self.plasma.T_e_eV),
            jnp.asarray(self.path_length_m),
        )

        # 6. Instrument response curve.
        if self.instrument.response_curve is not None:
            logger.debug("Applying instrument response (JAX)...")
            # If the instrument is the JAX variant, this stays on-device.
            intensity_np = self.instrument.apply_response(
                self.wavelength, np.asarray(intensity_jnp)
            )
            intensity_jnp = jnp.asarray(intensity_np)

        # 7. Instrument function (Gaussian convolution).
        if self.broadening_mode != BroadeningMode.NIST_PARITY:
            if self.instrument.is_resolving_power_mode:
                mid_wl = 0.5 * (self.lambda_min + self.lambda_max)
                sigma_conv = self.instrument.sigma_at_wavelength(mid_wl)
            else:
                sigma_conv = self.instrument.resolution_sigma_nm
            if sigma_conv > 0:
                from cflibs.instrument.convolution import apply_instrument_function_jax

                intensity = apply_instrument_function_jax(
                    self.wavelength, np.asarray(intensity_jnp), sigma_conv
                )
            else:
                intensity = np.asarray(intensity_jnp)
        else:
            intensity = np.asarray(intensity_jnp)

        logger.info("Spectrum computation (JAX) complete")
        return self.wavelength, intensity

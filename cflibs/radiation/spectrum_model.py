"""
Forward spectrum model that ties together all components.
"""

import numpy as np
from typing import Dict, Tuple

from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.plasma.saha_boltzmann import (
    SahaBoltzmannSolver,
    SahaBoltzmannSolverJax,
    SpeciesStageState,
)
from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES
from cflibs.instrument.model import InstrumentModel
from cflibs.radiation.profiles import BroadeningMode
from cflibs.instrument.convolution import apply_instrument_function
from cflibs.core.constants import H_PLANCK, C_LIGHT, KB, EV_TO_K
from cflibs.core.logging_config import get_logger

jit = jit_if_available  # local alias preserves existing @jit decorator sites

logger = get_logger("radiation.spectrum_model")


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
        apply_stark: bool = True,
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
        apply_stark : bool, default True
            If True, include the per-line Lorentzian Stark width on the
            Voigt path (``PHYSICAL_DOPPLER`` only — ignored by LEGACY,
            NIST_PARITY, LDM_GAUSSIAN). Matches the default used by the
            manifold (``forward_from_snapshot``) and Bayesian
            (``BayesianForwardModel._compute_spectrum``) paths, which
            both default to True. Set False to reproduce the pre-Wave-1
            Doppler-only template behaviour (e.g. for parity tests).
            Physics rationale: at n_e ~ 10^17 cm^-3 the Lorentzian Stark
            width is comparable to or exceeds the Gaussian (Doppler +
            instrument) width — Aragón & Aguilera 2008,
            *Spectrochim. Acta B* 63, 893.
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
        self.apply_stark = apply_stark

        # Create wavelength grid
        self.wavelength = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)

        # Initialize solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        logger.info(
            f"Initialized SpectrumModel: λ=[{lambda_min:.1f}, {lambda_max:.1f}] nm, "
            f"Δλ={delta_lambda:.3f} nm, {len(self.wavelength)} points, "
            f"mode={broadening_mode.value}"
        )

    # Canonical standard atomic-mass table (cflibs.atomic.masses).
    _FALLBACK_MASSES = STANDARD_ATOMIC_MASSES

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

    def _build_n_upper_per_line(
        self,
        snapshot,
        species_states: Dict[Tuple[str, int], SpeciesStageState],
        n_lines: int,
    ) -> np.ndarray:
        """Per-line upper-level populations from per-species Saha states.

        Computes ``n_k = n_stage * (g_k / U) * exp(-E_k / kT)`` directly from
        each transition's own ``(g_k, E_k)`` carried by the lines table.
        The historical implementation joined against the ``energy_levels``
        table on a ``round(E_k, 8)`` float key; the two tables encode the
        same physical level with ~1e-7 eV differences, so ~98 % of lookups
        missed and the lines were silently dropped (audit 2026-06-09 F1,
        bead CF-LIBS-improved-z3cg).

        Lines whose upper level lies above the IPD-lowered ionization cutoff
        get zero population (the level has merged into the continuum) —
        preserving the cutoff semantics of
        :meth:`SahaBoltzmannSolver.solve_level_population`.
        """
        n_upper_per_line = np.zeros(n_lines, dtype=np.float64)
        if not n_lines:
            return n_upper_per_line

        T_e_eV = self.plasma.T_e_eV
        line_E_k = np.asarray(snapshot.line_E_k_ev, dtype=np.float64)
        line_g_k = np.asarray(snapshot.line_g_k, dtype=np.float64)
        line_sp_idx = np.asarray(snapshot.line_species_index)
        for sp_idx, (element, stage) in enumerate(snapshot.species):
            state = species_states.get((element, stage))
            if state is None:
                continue
            sel = np.flatnonzero(line_sp_idx == sp_idx)
            if sel.size == 0:
                continue
            E_k = line_E_k[sel]
            n_line = (
                state.number_density_cm3
                * (line_g_k[sel] / state.partition_function)
                * np.exp(-E_k / T_e_eV)
            )
            n_line[E_k > state.max_energy_ev] = 0.0
            n_upper_per_line[sel] = n_line
        return n_upper_per_line

    def _build_ldm_sigma_grid(self, snapshot, n_lines: int):
        """Build the optional LDM sigma_grid (only for ``LDM_GAUSSIAN`` dispatch).

        Returns ``None`` for any other broadening mode or when there are no
        lines, matching the legacy guard.
        """
        if not (self.broadening_mode == BroadeningMode.LDM_GAUSSIAN and n_lines):
            return None

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
        return build_sigma_grid(sigma_D)

    def _run_forward_kernel(
        self, snapshot, n_lines: int, sigma_grid, n_upper_per_line: np.ndarray
    ) -> np.ndarray:
        """Run the unified kernel (or emit the empty-band fallback).

        The four broadening modes map to:
          LEGACY            -> scalar Gaussian sigma; downstream conv.
          NIST_PARITY       -> per-line instrument sigma; no downstream.
          PHYSICAL_DOPPLER  -> per-line Doppler sigma; no Stark; no fold;
                              downstream conv.
          LDM_GAUSSIAN      -> LDM/DIT Gaussian path; downstream conv.
        """
        from cflibs.radiation.kernels import forward_model

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
                apply_stark=self.apply_stark,
                _precomputed_n_upper_per_line=n_upper_per_line,
            )
            return np.asarray(intensity)

        # No transitions in band: emit zeros, then optionally Planck-RT-
        # squelch through the same RT step. Path matches legacy behaviour.
        intensity = np.zeros_like(self.wavelength)
        if self.path_length_m > 0:
            B_lambda = planck_radiance(self.wavelength, self.plasma.T_e_eV)
            intensity = B_lambda * 0.0
        return intensity

    def _downstream_convolution_sigma(self) -> float:
        """Pick the scalar instrument sigma for the downstream convolution step."""
        if self.instrument.is_resolving_power_mode:
            mid_wl = 0.5 * (self.lambda_min + self.lambda_max)
            return self.instrument.sigma_at_wavelength(mid_wl)
        return self.instrument.resolution_sigma_nm

    def _apply_downstream_convolution(self, intensity: np.ndarray) -> np.ndarray:
        """Apply downstream instrument convolution for the modes that need it.

        NIST_PARITY folds instrument broadening into the per-line sigma, so it
        skips this step entirely. The selected backend (JAX vs scipy) matches
        ``self.use_jax``.
        """
        if self.broadening_mode == BroadeningMode.NIST_PARITY:
            logger.debug("Skipping instrument convolution (NIST_PARITY mode)")
            return intensity

        sigma_conv = self._downstream_convolution_sigma()
        if sigma_conv <= 0:
            logger.debug("Skipping instrument convolution (sigma=0)")
            return intensity

        logger.debug("Applying instrument function (sigma=%.4f nm)...", sigma_conv)
        if self.use_jax:
            from cflibs.instrument.convolution import apply_instrument_function_jax

            return apply_instrument_function_jax(self.wavelength, intensity, sigma_conv)
        return apply_instrument_function(self.wavelength, intensity, sigma_conv)

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
        kernel (ADR-0001 T1-2). The Saha ionization balance and partition
        functions are computed by the detailed :class:`SahaBoltzmannSolver`
        (direct-sum U with IPD truncation); per-line upper-level populations
        are then derived from each transition's own ``(g_k, E_k)`` via
        :meth:`_build_n_upper_per_line` and injected into the kernel via the
        ``_precomputed_n_upper_per_line`` parameter; the kernel then handles
        the per-line broadening + optional radiative-transfer step.

        Downstream scipy instrument convolution is retained for LEGACY /
        PHYSICAL_DOPPLER / LDM_GAUSSIAN modes (NIST_PARITY folds instrument
        broadening into the per-line sigma).
        """
        # Validate plasma
        self.plasma.validate()

        # 1. Solve the Saha ionization balance per (element, stage).
        logger.debug("Solving Saha-Boltzmann equations...")
        species_states = self.solver.solve_species_states(self.plasma)

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

        # 3. Per-line n_upper from the species states + each line's own
        #    (g_k, E_k) — no float-keyed energy_levels join (audit F1).
        n_upper_per_line = self._build_n_upper_per_line(snapshot, species_states, n_lines)

        # 4. Optional LDM sigma_grid (only for LDM_GAUSSIAN dispatch).
        sigma_grid = self._build_ldm_sigma_grid(snapshot, n_lines)

        # 5. Run the unified kernel (or empty-band fallback).
        intensity = self._run_forward_kernel(snapshot, n_lines, sigma_grid, n_upper_per_line)

        # 6. Apply instrument response curve (host-side multiplication; not
        #    part of the kernel because it is data-driven).
        if self.instrument.response_curve is not None:
            logger.debug("Applying instrument response...")
            intensity = self.instrument.apply_response(self.wavelength, intensity)

        # 7. Apply downstream instrument convolution for the modes that need it.
        intensity = self._apply_downstream_convolution(intensity)

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

    The ionization balance and atomic-data lookups go through the Python
    ``SahaBoltzmannSolverJax`` (which produces JAX-evaluated values but
    materialises them as Python floats at the boundary). The
    wavelength-grid-sized arithmetic — emissivity broadening (Gaussian or
    Voigt, honouring ``apply_stark``), Planck radiance, radiative transfer
    and instrument convolution — runs on ``jax.numpy`` arrays through the
    unified :func:`cflibs.radiation.kernels.forward_model` kernel used by
    the inherited :meth:`SpectrumModel.compute_spectrum` (this subclass no
    longer overrides ``compute_spectrum``; it only swaps in the JAX solver).

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
        apply_stark: bool = True,
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
            apply_stark=apply_stark,
        )
        # Replace the NumPy solver with the JAX one — public surface is the
        # same so downstream code (acceptance tests, benchmark harness)
        # doesn't notice.
        self.solver = SahaBoltzmannSolverJax(atomic_db)
        # Pre-stage the wavelength grid as a jnp array so the convolution
        # path doesn't pay H2D cost on every call.
        self._wavelength_jax = jnp.asarray(self.wavelength, dtype=jnp.float64)

    # NOTE (audit Family 1): ``compute_spectrum`` is intentionally NOT
    # overridden here. The base :meth:`SpectrumModel.compute_spectrum`
    # already runs the whole wavelength-grid-sized arithmetic on JAX:
    #
    #   * populations come from ``self.solver`` — which this subclass swaps
    #     to ``SahaBoltzmannSolverJax`` in ``__init__``;
    #   * the per-line emissivity, broadening and radiative-transfer math is
    #     delegated to :func:`cflibs.radiation.kernels.forward_model`, which
    #     is fully ``jax.numpy``-native (it receives a ``jnp`` wavelength
    #     grid via the ``HAS_JAX`` branch in the base method);
    #   * the downstream instrument convolution uses
    #     ``apply_instrument_function_jax`` because ``self.use_jax`` is True.
    #
    # The previous override re-implemented the kernel with a pure-Gaussian
    # sum (``_broaden_per_line_jax``) and silently dropped ``apply_stark``,
    # so ``SpectrumModelJax(apply_stark=True)`` produced Stark-free
    # (Gaussian-only) lines while the base class produced Voigt. Deleting
    # the override restores Voigt/Stark fidelity AND keeps the JAX dispatch,
    # because ``forward_model`` honours ``apply_stark`` on the
    # ``PHYSICAL_DOPPLER`` path via ``_voigt_sum_per_line``.

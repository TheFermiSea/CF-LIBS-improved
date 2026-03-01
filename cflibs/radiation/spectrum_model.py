"""
Forward spectrum model that ties together all components.
"""

import numpy as np
from typing import Optional, Tuple

from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.atomic.database import AtomicDatabase
from cflibs.instrument.model import InstrumentModel
from cflibs.radiation.emissivity import calculate_spectrum_emissivity
from cflibs.radiation.profiles import (
    BroadeningMode,
    doppler_width,
)
from cflibs.instrument.convolution import apply_instrument_function
from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.spectrum_model")


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
            key = (trans.element, trans.ionization_stage, trans.E_k_ev)
            if key not in populations:
                continue

            if self.broadening_mode == BroadeningMode.NIST_PARITY:
                sig = self.instrument.sigma_at_wavelength(trans.wavelength_nm)
            elif self.broadening_mode == BroadeningMode.PHYSICAL_DOPPLER:
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
        """
        # Validate plasma
        self.plasma.validate()

        # 1. Solve Saha-Boltzmann for level populations
        logger.debug("Solving Saha-Boltzmann equations...")
        populations = self.solver.solve_plasma(self.plasma)

        # 2. Get transitions for all species
        logger.debug("Loading transitions...")
        all_transitions = []
        for element in self.plasma.species.keys():
            transitions = self.atomic_db.get_transitions(
                element,
                wavelength_min=self.lambda_min,
                wavelength_max=self.lambda_max,
                min_relative_intensity=10.0,  # Filter weak lines
            )
            all_transitions.extend(transitions)

        logger.debug(f"Found {len(all_transitions)} transitions")

        # 3. Calculate line emissivity with mode-dependent broadening
        logger.debug(f"Calculating line emissivity (mode={self.broadening_mode.value})...")

        if self.broadening_mode == BroadeningMode.LEGACY:
            # Original behavior: single scalar sigma
            T_eV = self.plasma.T_e_eV
            sigma_nm = 0.01 * np.sqrt(T_eV / 0.86)
        else:
            # Per-line sigma array
            sigma_nm = self._compute_sigma_per_line(all_transitions, populations)

        emissivity = calculate_spectrum_emissivity(
            all_transitions, populations, self.wavelength, sigma_nm, use_jax=self.use_jax
        )

        # 4. Convert to intensity (optically thin: I = ε * L)
        intensity = emissivity * self.path_length_m

        # 5. Apply instrument response
        if self.instrument.response_curve is not None:
            logger.debug("Applying instrument response...")
            intensity = self.instrument.apply_response(self.wavelength, intensity)

        # 6. Apply instrument function (convolution)
        # NIST_PARITY: broadening is fully captured in per-line profiles, skip convolution
        # LEGACY and PHYSICAL_DOPPLER: apply downstream instrument convolution
        if self.broadening_mode != BroadeningMode.NIST_PARITY:
            # Determine convolution sigma: use resolving power at midpoint if available,
            # otherwise fall back to the fixed resolution_sigma_nm.
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

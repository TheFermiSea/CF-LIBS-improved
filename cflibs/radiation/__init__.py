"""
Radiation and spectral calculations.

This module provides:
- Line emissivity and opacity calculations
- Radiative transfer solvers (single zone, multi-zone)
- Continuum emission (Bremsstrahlung, recombination, etc.)
"""

from cflibs.radiation.emissivity import calculate_line_emissivity, calculate_spectrum_emissivity
from cflibs.radiation.profiles import (
    BroadeningMode,
    gaussian_profile,
    apply_gaussian_broadening,
    apply_gaussian_broadening_per_line,
    doppler_width,
    resolving_power_sigma,
)
from cflibs.radiation.ldm import (
    DEFAULT_N_SIGMA,
    broaden_lines_ldm,
    build_sigma_grid,
    ldm_broaden,
)
from cflibs.radiation.spectrum_model import (
    SpectrumModel,
    SpectrumModelJax,
    planck_radiance,
    planck_radiance_jax,
)

__all__ = [
    "BroadeningMode",
    "DEFAULT_N_SIGMA",
    "calculate_line_emissivity",
    "calculate_spectrum_emissivity",
    "gaussian_profile",
    "apply_gaussian_broadening",
    "apply_gaussian_broadening_per_line",
    "broaden_lines_ldm",
    "build_sigma_grid",
    "ldm_broaden",
    "doppler_width",
    "resolving_power_sigma",
    "SpectrumModel",
    "SpectrumModelJax",
    "planck_radiance",
    "planck_radiance_jax",
]

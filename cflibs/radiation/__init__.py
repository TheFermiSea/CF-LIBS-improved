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
from cflibs.radiation.spectrum_model import (
    SpectrumModel,
    SpectrumModelJax,
    planck_radiance,
    planck_radiance_jax,
)
from cflibs.radiation.batch import (
    compute_spectrum_batch,
    compute_spectrum_grid,
    compute_spectrum_ensemble,
)

__all__ = [
    "BroadeningMode",
    "calculate_line_emissivity",
    "calculate_spectrum_emissivity",
    "gaussian_profile",
    "apply_gaussian_broadening",
    "apply_gaussian_broadening_per_line",
    "doppler_width",
    "resolving_power_sigma",
    "SpectrumModel",
    "SpectrumModelJax",
    "planck_radiance",
    "planck_radiance_jax",
    "compute_spectrum_batch",
    "compute_spectrum_grid",
    "compute_spectrum_ensemble",
]

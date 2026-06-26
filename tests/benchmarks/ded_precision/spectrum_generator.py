"""Forward-model synthetic DED spectra at known composition (DED-PLAN step 4).

Wraps the production chunked forward model (``_ChunkedForward``) so we can
generate a clean spectrum at any constrained-set composition + plasma state,
and a noisy single shot (with plasma T/n_e jitter applied by re-forward-
modelling, per the noise model's source D).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .noise_model import DEDNoiseParams, apply_ded_noise

K_PER_EV = 11604.518  # Kelvin per eV


def _wt_to_number(comp_wt: Dict[str, float], elements: Sequence[str]):
    """Convert a wt% composition dict to the forward model's number fractions."""
    from cflibs.inversion.solve.full_spectrum import _mass_to_number_fractions

    mass = {e: float(comp_wt[e]) / 100.0 for e in elements}
    return _mass_to_number_fractions(mass, list(elements))


def make_forward(db_path: str, elements: Sequence[str], wl_grid, instrument_fwhm_nm: float = 0.1):
    """Build a reusable chunked forward model for a fixed element set + grid."""
    from cflibs.inversion.solve.full_spectrum import _ChunkedForward

    return _ChunkedForward(
        db_path,
        list(elements),
        np.asarray(wl_grid, dtype=float),
        instrument_fwhm_nm=instrument_fwhm_nm,
    )


def clean_spectrum(
    forward, comp_wt: Dict[str, float], elements: Sequence[str], T_K: float, ne_cm3: float
) -> np.ndarray:
    """Clean (noise-free) forward spectrum at the given composition + plasma state."""
    num = _wt_to_number(comp_wt, elements)
    spec = forward.spectrum_numpy(T_K / K_PER_EV, float(np.log10(ne_cm3)), num)
    return np.asarray(spec, dtype=float)


def noisy_shot(
    forward,
    comp_wt: Dict[str, float],
    elements: Sequence[str],
    T_K: float,
    ne_cm3: float,
    params: DEDNoiseParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """One noisy DED shot: jitter the plasma state (source D, re-forward-model),
    then apply shot/Poisson/readout/baseline noise (A,B,C,E)."""
    T_j, ne_j = params.sample_plasma_jitter(T_K, ne_cm3, rng)
    clean = clean_spectrum(forward, comp_wt, elements, T_j, ne_j)
    return apply_ded_noise(clean, params, rng)


def default_grid(window, step_nm: float = 0.02) -> np.ndarray:
    """Wavelength grid fine enough that a +/-1.5 sigma line window has many points."""
    lo, hi = window
    return np.arange(lo, hi + step_nm, step_nm)

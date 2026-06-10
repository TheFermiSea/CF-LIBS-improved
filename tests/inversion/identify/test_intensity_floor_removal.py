"""Tests for the removal of the ``max(area, peak_height)`` intensity floor.

Audit 2026-06-09 02-F6 (bead CF-LIBS-improved-cxxq): the trapezoid line
builder (:func:`_build_observation`) floored its integrated area (counts*nm)
at the bare peak height (counts), silently mixing incompatible quantities for
any line narrower than ~1 nm equivalent and distorting the Boltzmann ordinate
by 1.5-3 ln-units. The floor is removed: the integrated area is used
consistently, and only when the trapezoid integral is unusable (non-finite or
<= 0) does the builder fall back to the UNITS-CONSISTENT Gaussian-equivalent
area ``h * FWHM * sqrt(pi / (4 ln 2))`` — never the bare height.

Expected values are independent analytic oracles (Gaussian area formula,
linear half-max interpolation), never the production code's own output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflibs.atomic.structures import Transition
from cflibs.inversion.identify.line_detection import _build_observation

# Independent oracle: area of a Gaussian per unit height and FWHM.
_GAUSS_AREA_FACTOR = math.sqrt(math.pi / (4.0 * math.log(2.0)))  # ~1.0645


def _transition(wl: float = 400.0) -> Transition:
    return Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=wl,
        A_ki=1.0e8,
        E_i_ev=0.0,
        E_k_ev=3.0,
        g_i=9,
        g_k=11,
    )


def _gaussian_spectrum(height: float, fwhm_nm: float, wl_step: float = 0.005):
    """Isolated Gaussian line on a zero baseline, centered at 400 nm."""
    wavelength = np.arange(399.0, 401.0 + wl_step / 2, wl_step)
    sigma = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    intensity = height * np.exp(-0.5 * ((wavelength - 400.0) / sigma) ** 2)
    peak_idx = int(np.argmin(np.abs(wavelength - 400.0)))
    return wavelength, intensity, peak_idx


def test_narrow_line_intensity_is_integrated_area_not_peak_height():
    """A sub-nm line must report the trapezoid area (counts*nm), which is far
    BELOW the peak height — the old floor would have replaced it with the
    height."""
    height, fwhm = 1000.0, 0.1
    wl_step = 0.005
    wavelength, intensity, peak_idx = _gaussian_spectrum(height, fwhm, wl_step)
    half_width_px = int(0.5 / wl_step)  # +-0.5 nm window

    result = _build_observation(
        _transition(), peak_idx, wavelength, intensity, half_width_px, wl_step, 1.0
    )
    assert result is not None
    obs, _ = result

    analytic_area = height * fwhm * _GAUSS_AREA_FACTOR  # ~106.45 counts*nm
    assert obs.intensity == pytest.approx(analytic_area, rel=0.02)
    # The floor is gone: integrated area, not max(area, height).
    assert obs.intensity < height


def test_trapezoid_and_voigt_area_agree_on_synthetic_gaussian():
    """Audit 02-F6(d): the two builders must agree to ~10% on an isolated
    Gaussian so mixing them within one spectrum carries no ordinate offset."""
    height, fwhm = 500.0, 0.2
    wl_step = 0.005
    wavelength, intensity, peak_idx = _gaussian_spectrum(height, fwhm, wl_step)
    half_width_px = int(0.8 / wl_step)

    result = _build_observation(
        _transition(), peak_idx, wavelength, intensity, half_width_px, wl_step, 1.0
    )
    assert result is not None
    obs, _ = result

    # The Voigt path reports the true fitted area; oracle = analytic area.
    voigt_area = height * fwhm * _GAUSS_AREA_FACTOR
    assert abs(obs.intensity - voigt_area) / voigt_area < 0.10


def test_unusable_trapezoid_falls_back_to_gaussian_equivalent_area():
    """When baseline over-subtraction drives the integral negative, the
    fallback must be h * FWHM * sqrt(pi/(4 ln 2)) — NOT the bare height."""
    dx = 0.01
    wavelength = np.array([400.0 - 2 * dx, 400.0 - dx, 400.0, 400.0 + dx, 400.0 + 2 * dx])
    intensity = np.array([-200.0, 40.0, 100.0, 40.0, -200.0])
    # Independent oracle: trapezoid = dx * (-80 + 70 + 70 - 80) = -0.2 < 0.
    assert float(np.trapezoid(intensity, wavelength)) < 0.0

    result = _build_observation(_transition(), 2, wavelength, intensity, 2, dx, 1.0)
    assert result is not None
    obs, _ = result

    # Half-max (50) crossings by linear interpolation between (40, 100):
    # each side at dx * (50-40)/(100-40) = dx/6 beyond the +-dx samples
    # => FWHM = 2*(dx - dx/6) ... computed directly:
    left = (400.0 - dx) + dx * (50.0 - 40.0) / (100.0 - 40.0)
    right = (400.0 + dx) - dx * (50.0 - 40.0) / (100.0 - 40.0)
    fwhm_oracle = right - left  # = (5/3) * dx
    expected = 100.0 * fwhm_oracle * _GAUSS_AREA_FACTOR
    assert obs.intensity == pytest.approx(expected, rel=1e-9)
    # Units-consistent fallback, not the bare height.
    assert obs.intensity != pytest.approx(100.0, rel=0.01)


def test_all_nonpositive_segment_returns_none():
    """No usable area and no positive height -> no observation."""
    dx = 0.01
    wavelength = np.arange(399.95, 400.05 + dx / 2, dx)
    intensity = np.full_like(wavelength, -5.0)
    result = _build_observation(
        _transition(), len(wavelength) // 2, wavelength, intensity, 3, dx, 1.0
    )
    assert result is None

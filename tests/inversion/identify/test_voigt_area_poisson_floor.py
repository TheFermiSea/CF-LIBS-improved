"""Tests for the Poisson-equivalent floor on the Voigt-fit line builder.

The trapezoid line-building path (:func:`_build_observation`) floors its area
uncertainty at the shot-noise level ``sqrt(sum counts) * wl_step``. The
Voigt-fit path (:func:`_build_observation_from_fit`) previously floored only at
``1e-6``, letting an over-confident covariance-based fit report a sigma far
below the shot-noise limit and thus an over-weighted Boltzmann-plot point.

These tests pin the new behaviour: the Voigt path is floored at the same
shot-noise-equivalent level (``sqrt(area * wl_step)``), the floor can be
disabled, and it never *lowers* an already-honest (larger) fit uncertainty.
"""

from __future__ import annotations

import math

from cflibs.atomic.structures import Transition
from cflibs.inversion.deconvolution import VoigtFitResult
from cflibs.inversion.identify.line_detection import (
    _build_observation_from_fit,
    _poisson_area_floor,
)


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


def _fit(area: float, area_uncertainty: float) -> VoigtFitResult:
    return VoigtFitResult(
        center_nm=400.0,
        amplitude=area,
        sigma_gaussian=0.02,
        gamma_lorentzian=0.01,
        area=area,
        area_uncertainty=area_uncertainty,
        residual_rms=0.0,
    )


def test_poisson_area_floor_matches_shot_noise_form():
    area, wl_step = 1.0e8, 0.005
    expected = math.sqrt(area * wl_step)
    assert math.isclose(_poisson_area_floor(area, wl_step, scale=1.0), math.sqrt(area * wl_step))
    assert math.isclose(_poisson_area_floor(area, wl_step, scale=1.0), expected)
    # Disabled / degenerate inputs -> no floor.
    assert math.isclose(_poisson_area_floor(area, wl_step, scale=0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(_poisson_area_floor(area, 0.0, scale=1.0), 0.0, abs_tol=1e-12)
    assert math.isclose(_poisson_area_floor(-5.0, wl_step, scale=1.0), 0.0, abs_tol=1e-12)


def test_overconfident_voigt_sigma_is_raised_to_poisson_floor():
    # Confident fit: covariance sigma far below shot noise.
    area, wl_step = 1.0e8, 0.005
    fit = _fit(area=area, area_uncertainty=1.0)  # absurdly small for area=1e8
    obs, _ = _build_observation_from_fit(
        _transition(), fit, ground_state_threshold_ev=0.1, wl_step=wl_step
    )
    expected_floor = math.sqrt(area * wl_step)
    assert math.isclose(obs.intensity_uncertainty, expected_floor)
    assert obs.intensity_uncertainty > fit.area_uncertainty


def test_honest_large_voigt_sigma_is_not_lowered():
    # A fit whose own uncertainty already exceeds the floor must be preserved.
    area, wl_step = 1.0e8, 0.005
    big_sigma = 10.0 * math.sqrt(area * wl_step)
    fit = _fit(area=area, area_uncertainty=big_sigma)
    obs, _ = _build_observation_from_fit(
        _transition(), fit, ground_state_threshold_ev=0.1, wl_step=wl_step
    )
    assert math.isclose(obs.intensity_uncertainty, big_sigma)


def test_floor_disabled_reproduces_legacy_behaviour():
    # scale=0 (or wl_step=0) reproduces the old max(area_unc, 1e-6) floor.
    fit = _fit(area=1.0e8, area_uncertainty=1.0)
    obs, _ = _build_observation_from_fit(
        _transition(),
        fit,
        ground_state_threshold_ev=0.1,
        wl_step=0.005,
        poisson_floor_scale=0.0,
    )
    assert math.isclose(obs.intensity_uncertainty, 1.0)  # area_uncertainty, no Poisson floor

    obs0, _ = _build_observation_from_fit(
        _transition(), fit, ground_state_threshold_ev=0.1, wl_step=0.0
    )
    assert math.isclose(obs0.intensity_uncertainty, 1.0)


def test_inner_1e6_guard_still_applies():
    # Tiny area + tiny step: Poisson floor is below 1e-6, inner guard wins.
    fit = _fit(area=1.0e-9, area_uncertainty=1.0e-12)
    obs, _ = _build_observation_from_fit(
        _transition(), fit, ground_state_threshold_ev=0.1, wl_step=1.0e-9
    )
    assert math.isclose(obs.intensity_uncertainty, 1e-6)

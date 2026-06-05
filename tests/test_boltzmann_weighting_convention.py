"""Regression guard for the Boltzmann WLS weighting convention (audit Family I).

Bug: ``BoltzmannPlotFitter._compute_weights`` returns inverse-variance weights
``1/sigma^2`` (correct as the weight on the *squared* residual). Those were
passed straight into ``numpy.polyfit(..., w=W)``, but polyfit applies ``w`` to
the *unsquared* residual and minimises ``sum((w*r)^2)`` — so the effective
weighting became ``1/sigma^4``. That non-physical over-weighting collapses the
fit onto the few highest-SNR points and biases the slope (and hence T).

numpy's own docs are explicit: "use ``w[i] = 1/sigma(y[i])``" (not 1/sigma**2).

These tests build a heteroscedastic Boltzmann plot, compute the slope under both
the correct (1/sigma^2) and the buggy (1/sigma^4) conventions from first
principles (closed-form WLS normal equations), and assert the fitter matches the
correct one — and is distinguishable from the buggy one. They fail on the
pre-fix code and pass after it.
"""

import math

import numpy as np
import pytest

from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, FitMethod, LineObservation


def _obs(wavelength_nm: float, e_k_ev: float, y_value: float, rel_sigma: float) -> LineObservation:
    """Build a LineObservation whose Boltzmann-plot point is (e_k_ev, y_value).

    With ``intensity = exp(y) * g*A / lambda`` the plot ordinate
    ``ln(I*lambda/(g*A)) == y_value`` exactly, and (aki weighting off) the
    fitter's per-point sigma_y equals ``rel_sigma`` (the relative intensity
    uncertainty).
    """
    g_k = 2
    a_ki = 1e8
    intensity = math.exp(y_value) * g_k * a_ki / wavelength_nm
    return LineObservation(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=rel_sigma * intensity,
        E_k_ev=e_k_ev,
        g_k=g_k,
        A_ki=a_ki,
        aki_uncertainty=None,
    )


def _wls_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Closed-form WLS slope where ``w`` is the weight on the squared residual."""
    s_w = np.sum(w)
    s_wx = np.sum(w * x)
    s_wy = np.sum(w * y)
    s_wxx = np.sum(w * x * x)
    s_wxy = np.sum(w * x * y)
    det = s_w * s_wxx - s_wx * s_wx
    return (s_w * s_wxy - s_wx * s_wy) / det


# Heteroscedastic data: two high-SNR points define a slope ~ -1.0; two low-SNR
# points sit off that line and pull the correct (1/sigma^2) slope away from it,
# while the buggy (1/sigma^4) fit ignores them.
_POINTS = [
    # (wavelength_nm, E_k_ev, y_value, rel_sigma)
    (300.0, 1.0, 0.00, 0.02),
    (310.0, 2.0, -1.00, 0.02),
    (320.0, 3.0, -1.70, 0.20),
    (330.0, 4.0, -2.55, 0.20),
]


@pytest.mark.unit
@pytest.mark.physics
def test_boltzmann_fit_uses_inverse_variance_not_inverse_fourth():
    obs = [_obs(*p) for p in _POINTS]
    x = np.array([p[1] for p in _POINTS])
    y = np.array([p[2] for p in _POINTS])
    sigma = np.array([p[3] for p in _POINTS])

    slope_correct = _wls_slope(x, y, 1.0 / sigma**2)  # physical WLS
    slope_buggy = _wls_slope(x, y, 1.0 / sigma**4)  # pre-fix effective weighting

    # The dataset must actually distinguish the two conventions, else the test
    # proves nothing.
    assert abs(slope_correct - slope_buggy) > 1e-2

    fitter = BoltzmannPlotFitter(
        method=FitMethod.SIGMA_CLIP,
        outlier_sigma=1e9,  # disable rejection
        max_iterations=1,
    )
    result = fitter.fit(obs, aki_uncertainty_weighting=False)

    # Fitter must match the physical 1/sigma^2 WLS slope...
    assert result.slope == pytest.approx(slope_correct, rel=1e-6, abs=1e-9)
    # ...and must NOT match the old 1/sigma^4 behaviour.
    assert abs(result.slope - slope_buggy) > 1e-3


@pytest.mark.unit
@pytest.mark.physics
def test_weighted_fit_helper_matches_inverse_variance():
    """The covariance-returning _weighted_fit path uses the same convention."""
    x = np.array([p[1] for p in _POINTS])
    y = np.array([p[2] for p in _POINTS])
    sigma = np.array([p[3] for p in _POINTS])
    slope_correct = _wls_slope(x, y, 1.0 / sigma**2)

    fitter = BoltzmannPlotFitter()
    slope, intercept, slope_err, intercept_err, cov = fitter._weighted_fit(x, y, sigma)
    assert slope == pytest.approx(slope_correct, rel=1e-6, abs=1e-9)
    assert cov is not None and cov[0, 0] > 0

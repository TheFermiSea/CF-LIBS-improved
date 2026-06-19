"""Unit tests for the opt-in ODR (errors-in-variables) Boltzmann fit.

Candidate B1-odr-boltzmann. Verifies the cited *physical* properties of the
orthogonal-distance-regression slope estimator added to
:class:`cflibs.inversion.physics.boltzmann.BoltzmannPlotFitter`:

1. **Bias reduction under x-noise** — when the upper-level energies ``E_k``
   carry non-negligible error, weighted OLS attenuates the Boltzmann-plot
   slope toward zero (regression-dilution / errors-in-variables bias), which
   biases the recovered temperature ``T = -1/(k_B m)`` *high*. The ODR fit
   accounts for uncertainty on both axes and recovers ``T`` with less bias.
   Asserted as a Monte-Carlo ensemble bias comparison (Boggs & Rodgers 1990;
   Aragón & Aguilera 2008).
2. **OLS limit** — with zero x-noise ODR reduces to weighted OLS; the slopes
   are ``allclose``.
3. **Reporting** — T and a finite 2x2 slope/intercept covariance are returned.
4. **Default unchanged** — the default fitter does not touch the ODR path and
   keeps the ``"sigma_clip"`` method label.

All inputs are synthetic; no database is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.core.constants import KB_EV
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, LineObservation

pytestmark = pytest.mark.unit


def _observations_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma_y: float,
    sigma_x: float | None = None,
) -> list[LineObservation]:
    """Build LineObservations whose ``y_value``/``E_k_ev`` equal given x, y.

    ``y_value = ln(I * lambda / (g * A))``; with ``lambda = g = A = 1`` we get
    ``y_value = ln(I)``, so ``I = exp(y)``. ``y_uncertainty = sigma_I / I`` is
    set to the target relative ``sigma_y``. An optional per-line
    ``E_k_uncertainty`` attribute is attached so the ODR path can read it.
    """
    obs: list[LineObservation] = []
    for xi, yi in zip(x, y):
        intensity = float(np.exp(yi))
        o = LineObservation(
            wavelength_nm=1.0,
            intensity=intensity,
            intensity_uncertainty=intensity * sigma_y,  # rel. unc -> y_uncertainty == sigma_y
            element="Fe",
            ionization_stage=1,
            E_k_ev=float(xi),
            g_k=1,
            A_ki=1.0,
        )
        if sigma_x is not None:
            # Attribute the dataclass does not declare; the ODR path reads it
            # via getattr. Strictly opt-in metadata.
            o.E_k_uncertainty = float(sigma_x)  # type: ignore[attr-defined]
        obs.append(o)
    return obs


def _true_slope(temperature_K: float) -> float:
    return -1.0 / (KB_EV * temperature_K)


def test_zero_x_noise_odr_matches_ols() -> None:
    """With no x-noise, ODR reduces to weighted OLS (allclose)."""
    T_true = 9000.0
    slope = _true_slope(T_true)
    intercept = 5.0
    n = 25
    x = np.linspace(1.0, 6.0, n)
    rng = np.random.default_rng(123)
    sigma_y = 0.05
    y = slope * x + intercept + rng.normal(0.0, sigma_y, n)

    # No per-line x-error and odr_x_uncertainty=0 -> OLS limit.
    obs = _observations_from_xy(x, y, sigma_y=sigma_y, sigma_x=None)

    ols = BoltzmannPlotFitter().fit(obs, aki_uncertainty_weighting=False)
    odr = BoltzmannPlotFitter(use_odr=True, odr_x_uncertainty=0.0).fit(
        obs, aki_uncertainty_weighting=False
    )

    assert ols.fit_method == "sigma_clip"
    assert odr.fit_method == "sigma_clip_odr"
    # Slopes (and hence temperatures) agree to tight tolerance in the OLS limit.
    np.testing.assert_allclose(odr.slope, ols.slope, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(odr.temperature_K, ols.temperature_K, rtol=1e-6)


def test_odr_reports_temperature_and_covariance() -> None:
    """ODR returns a finite T and a finite, symmetric 2x2 covariance."""
    T_true = 8000.0
    slope = _true_slope(T_true)
    n = 20
    x = np.linspace(1.0, 5.0, n)
    rng = np.random.default_rng(7)
    sigma_y = 0.05
    sigma_x = 0.20
    y = slope * x + 4.0 + rng.normal(0.0, sigma_y, n)
    obs = _observations_from_xy(x, y, sigma_y=sigma_y, sigma_x=sigma_x)

    res = BoltzmannPlotFitter(use_odr=True).fit(obs, aki_uncertainty_weighting=False)

    assert np.isfinite(res.temperature_K)
    assert res.temperature_K > 0
    cov = res.covariance_matrix
    assert cov is not None
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, rtol=0, atol=1e-12)  # symmetric
    # Diagonal matches reported 1-sigma errors.
    np.testing.assert_allclose(np.sqrt(cov[0, 0]), res.slope_uncertainty, rtol=1e-9)
    assert np.isfinite(res.temperature_uncertainty_K)
    assert res.temperature_uncertainty_K > 0


def test_odr_reduces_temperature_bias_under_x_noise() -> None:
    """Monte-Carlo: ODR slope has less bias than OLS when E_k carries error.

    The OLS slope is attenuated toward zero by x-noise (regression dilution),
    biasing the recovered temperature high. ODR corrects this to first order,
    so its ensemble-mean recovered T is closer to the truth than OLS.
    """
    T_true = 9000.0
    slope_true = _true_slope(T_true)
    intercept = 5.0
    n = 30
    sigma_x = 0.30  # substantial E_k uncertainty (eV)
    sigma_y = 0.10
    x_true = np.linspace(1.0, 6.0, n)

    n_trials = 120
    T_ols = np.empty(n_trials)
    T_odr = np.empty(n_trials)

    ols_fitter = BoltzmannPlotFitter()
    odr_fitter = BoltzmannPlotFitter(use_odr=True, odr_x_uncertainty=sigma_x)

    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        x_obs = x_true + rng.normal(0.0, sigma_x, n)
        y_obs = slope_true * x_true + intercept + rng.normal(0.0, sigma_y, n)

        obs = _observations_from_xy(x_obs, y_obs, sigma_y=sigma_y, sigma_x=sigma_x)

        T_ols[trial] = ols_fitter.fit(obs, aki_uncertainty_weighting=False).temperature_K
        T_odr[trial] = odr_fitter.fit(obs, aki_uncertainty_weighting=False).temperature_K

    # Discard any non-finite (population-inversion) draws defensively.
    ols_ok = np.isfinite(T_ols)
    odr_ok = np.isfinite(T_odr)
    assert ols_ok.mean() > 0.95
    assert odr_ok.mean() > 0.95

    bias_ols = abs(np.mean(T_ols[ols_ok]) - T_true)
    bias_odr = abs(np.mean(T_odr[odr_ok]) - T_true)

    # Physical property under test: ODR is less biased than OLS.
    assert bias_odr < bias_ols, f"ODR bias {bias_odr:.1f} K not < OLS bias {bias_ols:.1f} K"
    # And meaningfully so: at least a 2x reduction in absolute temperature bias.
    assert bias_odr < 0.5 * bias_ols

    # Sanity: OLS over-estimates T (attenuated |slope| -> larger T).
    assert np.mean(T_ols[ols_ok]) > T_true


def test_default_fit_does_not_use_odr() -> None:
    """The default fitter ignores the ODR path; method label is unchanged."""
    T_true = 7000.0
    slope = _true_slope(T_true)
    n = 15
    x = np.linspace(1.0, 4.0, n)
    rng = np.random.default_rng(99)
    y = slope * x + 3.0 + rng.normal(0.0, 0.03, n)
    obs = _observations_from_xy(x, y, sigma_y=0.03, sigma_x=0.1)

    fitter = BoltzmannPlotFitter()
    assert fitter.use_odr is False
    res = fitter.fit(obs, aki_uncertainty_weighting=False)
    assert res.fit_method == "sigma_clip"

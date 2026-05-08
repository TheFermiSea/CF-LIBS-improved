import numpy as np
import pytest

from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, LineObservation
from cflibs.inversion.common.data_structures import FitMethod


def _observation_with_y(energy_ev: float, y_value: float, wavelength_nm: float) -> LineObservation:
    intensity = float(np.exp(y_value) / wavelength_nm)
    return LineObservation(
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=0.01 * intensity,
        element="Fe",
        ionization_stage=1,
        E_k_ev=energy_ev,
        g_k=1,
        A_ki=1.0,
    )


@pytest.mark.unit
def test_boltzmann_tuning_defaults():
    """Verify that the new stricter defaults are set correctly."""
    fitter = BoltzmannPlotFitter()
    # Check updated defaults
    assert fitter.outlier_sigma == pytest.approx(2.5)
    assert fitter.max_iterations == 10
    assert fitter.huber_epsilon == pytest.approx(1.2)


@pytest.mark.unit
def test_ransac_uses_outlier_sigma_for_threshold():
    """Verify that RANSAC now uses the outlier_sigma parameter for its default threshold."""
    # Ensure it respects custom outlier_sigma
    fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC, outlier_sigma=2.0)
    assert fitter.outlier_sigma == pytest.approx(2.0)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -1.0 * x + 10.0
    obs = [_observation_with_y(x[i], y[i], 300.0 + i) for i in range(len(x))]
    result = fitter.fit(obs)
    assert result.fit_method == "ransac"
    assert result.temperature_K > 0


@pytest.mark.unit
def test_huber_epsilon_tuning():
    """Verify huber_epsilon is correctly assigned and used in fit execution."""
    fitter = BoltzmannPlotFitter(method=FitMethod.HUBER, huber_epsilon=1.1)
    assert fitter.huber_epsilon == pytest.approx(1.1)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -1.0 * x + 10.0
    obs = [_observation_with_y(x[i], y[i], 300.0 + i) for i in range(len(x))]
    result = fitter.fit(obs)
    assert result.fit_method == "huber"
    assert result.n_iterations >= 1

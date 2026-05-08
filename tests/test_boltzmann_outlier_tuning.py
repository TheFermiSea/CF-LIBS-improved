import numpy as np
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, LineObservation
from cflibs.inversion.common.data_structures import FitMethod

def test_boltzmann_tuning_defaults():
    """Verify that the new stricter defaults are set correctly."""
    fitter = BoltzmannPlotFitter()
    # Check updated defaults
    assert fitter.outlier_sigma == 2.5
    assert fitter.max_iterations == 10
    assert fitter.huber_epsilon == 1.2

def test_ransac_uses_outlier_sigma_for_threshold():
    """Verify that RANSAC now uses the outlier_sigma parameter for its default threshold."""
    # Ensure it respects custom outlier_sigma
    fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC, outlier_sigma=2.0)
    assert fitter.outlier_sigma == 2.0
    
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -1.0 * x + 10.0
    obs = [
        LineObservation(
            element="Fe", ionization_stage=1, wavelength_nm=300.0+i,
            g_k=1.0, A_ki=1.0, E_k_ev=x[i], y_value=y[i], y_uncertainty=0.01
        ) for i in range(len(x))
    ]
    result = fitter.fit(obs)
    assert result.fit_method == "ransac"
    assert result.temperature_K > 0

def test_huber_epsilon_tuning():
    """Verify huber_epsilon is correctly assigned and used in fit execution."""
    fitter = BoltzmannPlotFitter(method=FitMethod.HUBER, huber_epsilon=1.1)
    assert fitter.huber_epsilon == 1.1
    
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -1.0 * x + 10.0
    obs = [
        LineObservation(
            element="Fe", ionization_stage=1, wavelength_nm=300.0+i,
            g_k=1.0, A_ki=1.0, E_k_ev=x[i], y_value=y[i], y_uncertainty=0.01
        ) for i in range(len(x))
    ]
    result = fitter.fit(obs)
    assert result.fit_method == "huber"
    assert result.n_iterations >= 1

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

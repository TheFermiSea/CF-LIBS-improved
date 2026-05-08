import numpy as np
import pytest
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, LineObservation, FitMethod

def test_aki_uncertainty_weighting_impact():
    """Verify that aki_uncertainty_weighting actually changes the fit result."""
    # Create synthetic observations where one line has a huge A_ki uncertainty
    # and is also an outlier in the Boltzmann plot.
    obs = [
        LineObservation(element="Fe", ionization_stage=1, wavelength_nm=300.0, 
                        E_k_ev=1.0, g_k=2, A_ki=1e8, y_value=10.0, y_uncertainty=0.1, aki_uncertainty=0.01),
        LineObservation(element="Fe", ionization_stage=1, wavelength_nm=310.0, 
                        E_k_ev=2.0, g_k=2, A_ki=1e8, y_value=9.0, y_uncertainty=0.1, aki_uncertainty=0.01),
        # This line is an outlier (y should be ~8.0) and has huge A_ki uncertainty (50%)
        LineObservation(element="Fe", ionization_stage=1, wavelength_nm=320.0, 
                        E_k_ev=3.0, g_k=2, A_ki=1e8, y_value=12.0, y_uncertainty=0.1, aki_uncertainty=0.5),
    ]
    
    fitter = BoltzmannPlotFitter(method=FitMethod.SIGMA_CLIP, outlier_sigma=10.0) # Disable outlier rejection for this test
    
    # Fit without A_ki weighting (uniform weighting for intensity errors)
    res_uniform = fitter.fit(obs, aki_uncertainty_weighting=False)
    
    # Fit with A_ki weighting (should down-weight the third line)
    res_weighted = fitter.fit(obs, aki_uncertainty_weighting=True)
    
    # The weighted fit should have a slope closer to -1.0 (the slope of the first two points)
    # whereas the uniform fit will be pulled up by the outlier at E_k=3.0.
    assert res_weighted.slope < res_uniform.slope
    assert abs(res_weighted.slope - (-1.0)) < abs(res_uniform.slope - (-1.0))

if __name__ == "__main__":
    pytest.main([__file__])

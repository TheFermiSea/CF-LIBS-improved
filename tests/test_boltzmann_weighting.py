import math

import pytest
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, LineObservation, FitMethod


def _observation_with_y(
    wavelength_nm: float,
    e_k_ev: float,
    y_value: float,
    y_uncertainty: float,
    aki_uncertainty: float,
) -> LineObservation:
    g_k = 2
    a_ki = 1e8
    intensity = math.exp(y_value) * g_k * a_ki / wavelength_nm
    return LineObservation(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=y_uncertainty * intensity,
        E_k_ev=e_k_ev,
        g_k=g_k,
        A_ki=a_ki,
        aki_uncertainty=aki_uncertainty,
    )


@pytest.mark.unit
def test_aki_uncertainty_weighting_impact():
    """Verify that aki_uncertainty_weighting actually changes the fit result."""
    # Create synthetic observations where one line has a huge A_ki uncertainty
    # and is also an outlier in the Boltzmann plot.
    obs = [
        _observation_with_y(300.0, 1.0, 10.0, 0.1, 0.01),
        _observation_with_y(310.0, 2.0, 9.0, 0.1, 0.01),
        # This line is an outlier (y should be ~8.0) and has huge A_ki uncertainty (50%)
        _observation_with_y(320.0, 3.0, 12.0, 0.1, 0.5),
    ]

    fitter = BoltzmannPlotFitter(
        method=FitMethod.SIGMA_CLIP,
        outlier_sigma=1e6,
        max_iterations=1,
    )

    # Fit without A_ki weighting (uniform weighting for intensity errors)
    res_uniform = fitter.fit(obs, aki_uncertainty_weighting=False)

    # Fit with A_ki weighting (should down-weight the third line)
    res_weighted = fitter.fit(obs, aki_uncertainty_weighting=True)

    # The weighted fit should have a slope closer to -1.0 (the slope of the first two points)
    # whereas the uniform fit will be pulled up by the outlier at E_k=3.0.
    assert res_weighted.slope < res_uniform.slope
    assert abs(res_weighted.slope - (-1.0)) < abs(res_uniform.slope - (-1.0))

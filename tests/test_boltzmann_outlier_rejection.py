import numpy as np
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter
from cflibs.inversion.common.data_structures import LineObservation, FitMethod

def test_iterative_outlier_rejection_robustness():
    np.random.seed(42)
    x_true = np.linspace(2, 6, 20)
    slope_true = -1.0
    intercept_true = 10.0
    y_true = slope_true * x_true + intercept_true

    y_obs = y_true + np.random.normal(0, 0.005, len(x_true))
    err = np.ones_like(x_true) * 0.005

    y_obs[15:18] += 2.0

    observations = []
    for x, y, e in zip(x_true, y_obs, err):
        lam = 500.0
        g = 1
        A = 1e8
        intensity = np.exp(y) * g * A / lam
        intensity_err = e * intensity
        obs = LineObservation(
            wavelength_nm=lam,
            intensity=intensity,
            intensity_uncertainty=intensity_err,
            element="Fe",
            ionization_stage=1,
            E_k_ev=x,
            g_k=g,
            A_ki=A,
            aki_uncertainty=None,
        )
        observations.append(obs)

    fitter = BoltzmannPlotFitter(method=FitMethod.SIGMA_CLIP, outlier_sigma=2.0, max_iterations=5)
    res_no_rej = fitter.fit(observations, reject_outliers=False)

    res_rej = fitter.fit(observations, reject_outliers=True)

    bias_no_rej = abs((res_no_rej.slope - slope_true)/slope_true) * 100
    bias_rej = abs((res_rej.slope - slope_true)/slope_true) * 100

    assert bias_no_rej > 5.0, f"Expected bias > 5%, got {bias_no_rej:.2f}%"
    assert bias_rej < 0.5, f"Expected bias < 0.5%, got {bias_rej:.2f}%"

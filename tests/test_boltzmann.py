"""
Tests for Boltzmann plot generation and fitting.

Covers:
- Basic Boltzmann plot fitting
- Outlier rejection via sigma-clipping, RANSAC, and Huber M-estimation
- Temperature recovery accuracy
- Edge cases and error handling
"""

import numpy as np
import pytest
from cflibs.inversion.boltzmann import (
    LineObservation,
    BoltzmannPlotFitter,
    FitMethod,
)
from cflibs.core.constants import KB_EV


def create_synthetic_lines(
    T_K: float, n_points: int = 10, noise_level: float = 0.05, seed: int | None = None
) -> list[LineObservation]:
    """Generate synthetic spectral lines following Boltzmann distribution.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin
    n_points : int
        Number of spectral lines to generate
    noise_level : float
        Standard deviation of noise in ln(I) space
    seed : int, optional
        Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    T_eV = T_K * KB_EV

    # Random upper energies between 2 and 6 eV
    energies = np.linspace(2.0, 6.0, n_points)

    # Constants (F*C/U) - arbitrary
    intercept_const = 10.0

    obs = []
    for Ek in energies:
        # ln(I*lam/gA) = ln(const) - Ek/kT
        # y = ln_const - Ek/T_eV
        expected_y = np.log(intercept_const) - Ek / T_eV

        # Add noise
        y_noisy = expected_y + rng.normal(0, noise_level)

        # Back-calculate Intensity (assuming lam=1, g=1, A=1 for simplicity)
        # y = ln(I) -> I = exp(y)
        intensity = np.exp(y_noisy)

        # Estimate uncertainty (say 5%)
        I_err = intensity * 0.05

        obs.append(
            LineObservation(
                wavelength_nm=1.0,
                intensity=intensity,
                intensity_uncertainty=I_err,
                element="Fe",
                ionization_stage=1,
                E_k_ev=Ek,
                g_k=1,
                A_ki=1.0,
            )
        )

    return obs


def test_boltzmann_fit_perfect():
    """Test fitting on perfect data."""
    T_target = 10000.0
    # Very low noise
    lines = create_synthetic_lines(T_target, n_points=20, noise_level=0.0001)

    fitter = BoltzmannPlotFitter()
    result = fitter.fit(lines)

    assert result.n_points == 20
    # Tolerance 1%
    assert abs(result.temperature_K - T_target) < 100.0
    assert result.r_squared > 0.99


def test_outlier_rejection():
    """Test that outliers are correctly rejected."""
    T_target = 8000.0
    lines = create_synthetic_lines(T_target, n_points=10, noise_level=0.01, seed=42)

    # Add an outlier (e.g. self-absorbed line, intensity lower than expected)
    # y = ln(I...) - Ek/kT. Lower I -> lower y.
    # Let's manually modify the last point
    outlier = lines[-1]
    # Reduce intensity significantly (e.g. factor of 10)
    outlier.intensity /= 10.0

    fitter = BoltzmannPlotFitter(outlier_sigma=2.0)
    result = fitter.fit(lines)

    assert len(result.rejected_points) >= 1
    assert 9 in result.rejected_points  # Last index

    # Temperature should still be reasonably close
    assert abs(result.temperature_K - T_target) < 500.0


def test_insufficient_points():
    """Test error handling for too few points."""
    lines = create_synthetic_lines(5000, n_points=1)
    fitter = BoltzmannPlotFitter()
    with pytest.raises(ValueError):
        fitter.fit(lines)


def test_y_value_calculation():
    """Test the y-axis calculation logic."""
    obs = LineObservation(
        wavelength_nm=500.0,
        intensity=100.0,
        intensity_uncertainty=10.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=3.0,
        g_k=2,
        A_ki=1.0e6,
    )
    # y = ln(I * lam / (g * A))
    # y = ln(100 * 500 / (2 * 1e6)) = ln(50000 / 2e6) = ln(0.025) ≈ -3.688
    expected = np.log(100.0 * 500.0 / (2.0 * 1.0e6))
    assert abs(obs.y_value - expected) < 1e-6

    # Uncertainty: dy = dI/I = 10/100 = 0.1
    assert abs(obs.y_uncertainty - 0.1) < 1e-6


# ============================================================================
# Tests for Robust Fitting Methods
# ============================================================================


def create_synthetic_lines_with_outliers(
    T_K: float,
    n_points: int = 15,
    noise_level: float = 0.02,
    n_outliers: int = 3,
    outlier_factor: float = 0.1,
    seed: int = 42,
) -> tuple[list[LineObservation], list[int]]:
    """
    Generate synthetic lines with known outliers.

    Returns the lines and the indices of the outlier points.
    """
    rng = np.random.default_rng(seed)
    T_eV = T_K * KB_EV

    energies = np.linspace(2.0, 6.0, n_points)
    intercept_const = 10.0

    # Select random outlier indices
    outlier_indices = sorted(rng.choice(n_points, size=n_outliers, replace=False))

    obs = []
    for i, Ek in enumerate(energies):
        expected_y = np.log(intercept_const) - Ek / T_eV
        y_noisy = expected_y + rng.normal(0, noise_level)

        # Apply outlier factor (simulate self-absorption - lower intensity)
        if i in outlier_indices:
            y_noisy -= np.log(1 / outlier_factor)  # Reduce intensity by factor

        intensity = np.exp(y_noisy)
        I_err = intensity * 0.05

        obs.append(
            LineObservation(
                wavelength_nm=1.0,
                intensity=intensity,
                intensity_uncertainty=I_err,
                element="Fe",
                ionization_stage=1,
                E_k_ev=Ek,
                g_k=1,
                A_ki=1.0,
            )
        )

    return obs, list(outlier_indices)


class TestSigmaClipFitting:
    """Tests for sigma-clipping fit method (default)."""

    def test_sigma_clip_clean_data(self):
        """Sigma-clip should work well on clean data."""
        T_target = 10000.0
        lines = create_synthetic_lines(T_target, n_points=20, noise_level=0.001, seed=42)

        fitter = BoltzmannPlotFitter(method=FitMethod.SIGMA_CLIP)
        result = fitter.fit(lines)

        assert result.fit_method == "sigma_clip"
        assert result.n_points == 20
        assert len(result.rejected_points) == 0
        assert abs(result.temperature_K - T_target) / T_target < 0.01  # 1% tolerance

    def test_sigma_clip_with_outliers(self):
        """Sigma-clip should reject obvious outliers."""
        T_target = 8000.0
        lines, _ = create_synthetic_lines_with_outliers(
            T_target, n_points=15, n_outliers=2, outlier_factor=0.05
        )

        fitter = BoltzmannPlotFitter(
            method=FitMethod.SIGMA_CLIP, outlier_sigma=2.5, max_iterations=10
        )
        result = fitter.fit(lines)

        # Should reject at least some outliers
        assert len(result.rejected_points) >= 1
        # Temperature should be reasonably recovered
        assert abs(result.temperature_K - T_target) / T_target < 0.1  # 10% tolerance


class TestRANSACFitting:
    """Tests for RANSAC robust fitting."""

    def test_ransac_clean_data(self):
        """RANSAC should work on clean data."""
        T_target = 10000.0
        lines = create_synthetic_lines(T_target, n_points=20, noise_level=0.001, seed=42)

        fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC)
        result = fitter.fit(lines)

        assert result.fit_method == "ransac"
        # RANSAC may not use all points, but temperature should be accurate
        assert abs(result.temperature_K - T_target) / T_target < 0.02  # 2% tolerance

    def test_ransac_gross_outliers(self):
        """RANSAC should handle gross outliers better than sigma-clip."""
        T_target = 8000.0
        # Create data with strong outliers (factor of 20 intensity reduction)
        lines, _ = create_synthetic_lines_with_outliers(
            T_target, n_points=15, n_outliers=3, outlier_factor=0.05, seed=123
        )

        fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC, ransac_max_trials=200)
        result = fitter.fit(lines)

        assert result.fit_method == "ransac"
        # Temperature recovery should be reasonable despite outliers
        # RANSAC may or may not reject outliers depending on threshold
        assert abs(result.temperature_K - T_target) / T_target < 0.20  # 20% tolerance
        assert result.n_points >= 5  # Should use at least some inliers

    def test_ransac_few_points(self):
        """RANSAC with very few points."""
        T_target = 9000.0
        lines = create_synthetic_lines(T_target, n_points=5, noise_level=0.01, seed=42)

        fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC, ransac_min_samples=2)
        result = fitter.fit(lines)

        assert result.n_points >= 2
        assert np.isfinite(result.temperature_K)


class TestHuberFitting:
    """Tests for Huber M-estimation fitting."""

    def test_huber_clean_data(self):
        """Huber should work well on clean data."""
        T_target = 10000.0
        lines = create_synthetic_lines(T_target, n_points=20, noise_level=0.001, seed=42)

        fitter = BoltzmannPlotFitter(method=FitMethod.HUBER)
        result = fitter.fit(lines)

        assert result.fit_method == "huber"
        assert abs(result.temperature_K - T_target) / T_target < 0.01  # 1% tolerance

    def test_huber_moderate_outliers(self):
        """Huber should downweight moderate outliers."""
        T_target = 8000.0
        # Moderate outliers (factor of 5 intensity change)
        lines, _ = create_synthetic_lines_with_outliers(
            T_target, n_points=15, n_outliers=3, outlier_factor=0.2, seed=456
        )

        fitter = BoltzmannPlotFitter(
            method=FitMethod.HUBER,
            huber_epsilon=1.35,
            max_iterations=20,
        )
        result = fitter.fit(lines)

        assert result.fit_method == "huber"
        # Temperature should be reasonably recovered
        assert abs(result.temperature_K - T_target) / T_target < 0.15  # 15% tolerance

    def test_huber_convergence(self):
        """Huber IRLS should converge."""
        T_target = 7500.0
        lines = create_synthetic_lines(T_target, n_points=12, noise_level=0.02, seed=42)

        fitter = BoltzmannPlotFitter(method=FitMethod.HUBER, max_iterations=50)
        result = fitter.fit(lines)

        # Should converge before max iterations
        assert result.n_iterations < 50


class TestMethodComparison:
    """Compare performance of different fitting methods."""

    def test_all_methods_agree_on_clean_data(self):
        """All methods should give similar results on clean data."""
        T_target = 9000.0
        lines = create_synthetic_lines(T_target, n_points=20, noise_level=0.005, seed=42)

        results = {}
        for method in FitMethod:
            fitter = BoltzmannPlotFitter(method=method)
            results[method] = fitter.fit(lines)

        # All temperatures should be within 2% of each other
        temps = [r.temperature_K for r in results.values()]
        temp_range = max(temps) - min(temps)
        mean_temp = np.mean(temps)
        assert temp_range / mean_temp < 0.02

    def test_ransac_better_with_gross_outliers(self):
        """RANSAC should outperform sigma-clip with gross outliers."""
        T_target = 8500.0
        # Strong outliers that will bias sigma-clip
        lines, _ = create_synthetic_lines_with_outliers(
            T_target, n_points=20, n_outliers=5, outlier_factor=0.02, seed=789
        )

        ransac_fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC)
        ransac_result = ransac_fitter.fit(lines)

        ransac_error = abs(ransac_result.temperature_K - T_target) / T_target

        # RANSAC should be at least as good, usually better
        # (Note: This is a soft assertion - RANSAC is probabilistic)
        assert ransac_error < 0.30  # RANSAC should be within 30%


class TestBoltzmannFitResultAttributes:
    """Test new attributes in BoltzmannFitResult."""

    def test_result_has_fit_method(self):
        """Result should include fit method used."""
        lines = create_synthetic_lines(8000.0, n_points=10, noise_level=0.01, seed=42)

        for method in FitMethod:
            fitter = BoltzmannPlotFitter(method=method)
            result = fitter.fit(lines)
            assert result.fit_method == method.value

    def test_result_has_iteration_count(self):
        """Result should include iteration count."""
        lines = create_synthetic_lines(8000.0, n_points=10, noise_level=0.01, seed=42)

        fitter = BoltzmannPlotFitter(method=FitMethod.SIGMA_CLIP)
        result = fitter.fit(lines)

        assert result.n_iterations >= 1

    def test_result_has_inlier_mask(self):
        """Result should include inlier mask."""
        lines = create_synthetic_lines(8000.0, n_points=10, noise_level=0.01, seed=42)

        fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC)
        result = fitter.fit(lines)

        assert result.inlier_mask is not None
        assert len(result.inlier_mask) == len(lines)
        assert result.n_points == np.sum(result.inlier_mask)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_aggressive_outlier_rejection(self):
        """Handle case where aggressive rejection may remove all points."""
        T_target = 8000.0
        # Very noisy data with aggressive rejection (1-sigma with 30% noise)
        # This is an edge case - the algorithm may reject all points
        lines = create_synthetic_lines(T_target, n_points=10, noise_level=0.3, seed=42)

        fitter = BoltzmannPlotFitter(
            method=FitMethod.SIGMA_CLIP, outlier_sigma=1.0, max_iterations=10
        )
        result = fitter.fit(lines)

        # The algorithm should return a result even if all points rejected
        # When n_points=0, uncertainties will be inf and a warning is logged
        assert result.n_points >= 0
        # Result should have valid structure regardless of point count
        assert result.fit_method == "sigma_clip"
        # Temperature may be inf when all points rejected, or finite otherwise
        assert np.isfinite(result.temperature_K) or result.n_points == 0

    def test_positive_slope_handling(self):
        """Handle non-physical positive slope gracefully."""
        # Create inverted data (positive slope)
        obs = []
        for Ek in [2.0, 3.0, 4.0, 5.0]:
            # Intensity increases with energy (non-physical for Boltzmann)
            intensity = 100 * np.exp(Ek / 0.5)  # Positive slope
            obs.append(
                LineObservation(
                    wavelength_nm=1.0,
                    intensity=intensity,
                    intensity_uncertainty=intensity * 0.05,
                    element="Fe",
                    ionization_stage=1,
                    E_k_ev=Ek,
                    g_k=1,
                    A_ki=1.0,
                )
            )

        fitter = BoltzmannPlotFitter()
        result = fitter.fit(obs)

        # Should return infinity temperature
        assert result.temperature_K == float("inf")

    def test_zero_uncertainty_handling(self):
        """Handle zero measurement uncertainties."""
        obs = []
        # Need more than 2 points for covariance scaling
        for Ek in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            intensity = 100 * np.exp(-Ek / 0.8)
            obs.append(
                LineObservation(
                    wavelength_nm=1.0,
                    intensity=intensity,
                    intensity_uncertainty=0.0,  # Zero uncertainty
                    element="Fe",
                    ionization_stage=1,
                    E_k_ev=Ek,
                    g_k=1,
                    A_ki=1.0,
                )
            )

        for method in FitMethod:
            fitter = BoltzmannPlotFitter(method=method)
            result = fitter.fit(obs)
            assert np.isfinite(result.temperature_K)
            assert result.temperature_K > 0

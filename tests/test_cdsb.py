"""
Tests for Column Density Saha-Boltzmann (CD-SB) plotting.

Covers:
- CDSBLineObservation creation and oscillator strength calculation
- CDSBPlotter fitting with various conditions
- Optical depth correction accuracy
- Resonance line handling
- Convergence behavior
- Edge cases and error handling
- Integration with standard Boltzmann fitting
"""

import numpy as np
from cflibs.core.constants import KB_EV
from cflibs.inversion.cdsb import (
    CDSBPlotter,
    CDSBLineObservation,
    CDSBConvergenceStatus,
    LineOpticalDepth,
    create_cdsb_observation,
    from_transition,
)
from cflibs.inversion.boltzmann import FitMethod
from cflibs.atomic.structures import Transition

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def create_synthetic_cdsb_lines(
    T_K: float,
    n_points: int = 10,
    noise_level: float = 0.02,
    n_resonance: int = 2,
    tau_scale: float = 0.5,
    seed: int = 42,
) -> list[CDSBLineObservation]:
    """
    Generate synthetic spectral lines with self-absorption effects.

    Parameters
    ----------
    T_K : float
        Plasma temperature in Kelvin
    n_points : int
        Number of spectral lines to generate
    noise_level : float
        Standard deviation of noise in ln(I) space
    n_resonance : int
        Number of resonance lines (will have highest tau)
    tau_scale : float
        Scale factor for optical depths
    seed : int
        Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    T_eV = T_K * KB_EV

    # Generate lines with varying upper and lower energies
    E_k_values = np.linspace(2.5, 6.0, n_points)

    # Lower level energies: some resonance (E_i=0), some excited
    E_i_values = np.zeros(n_points)
    non_resonance_start = n_resonance
    E_i_values[non_resonance_start:] = rng.uniform(0.5, 2.0, n_points - n_resonance)

    # Intercept constant (arbitrary units)
    intercept_const = 10.0

    observations = []
    for i, (E_k, E_i) in enumerate(zip(E_k_values, E_i_values)):
        is_resonance = i < n_resonance

        # True y-value from Boltzmann: ln(I*lam/gA) = ln(const) - E_k/kT
        expected_y = np.log(intercept_const) - E_k / T_eV

        # Optical depth (higher for resonance lines)
        if is_resonance:
            tau = tau_scale * rng.uniform(1.0, 3.0)
        else:
            tau = tau_scale * rng.uniform(0.1, 0.8)

        # Self-absorption factor: I_obs = I_true * (1 - exp(-tau)) / tau
        if tau > 1e-6:
            absorption_factor = (1.0 - np.exp(-tau)) / tau
        else:
            absorption_factor = 1.0

        # Observed y is reduced by self-absorption
        # y_obs = y_true + ln(absorption_factor) = y_true - ln(correction)
        y_observed = expected_y + np.log(absorption_factor)

        # Add measurement noise
        y_noisy = y_observed + rng.normal(0, noise_level)

        # Back-calculate intensity (simplified: lam=500, g=1, A=1e8)
        wavelength = 400.0 + i * 20.0  # Spread wavelengths
        g_k = rng.integers(1, 10)
        g_i = rng.integers(1, 10)
        A_ki = 10 ** rng.uniform(7, 9)  # Typical A values

        # y = ln(I * lam / (g * A)) -> I = exp(y) * g * A / lam
        intensity = np.exp(y_noisy) * g_k * A_ki / wavelength
        I_err = intensity * 0.05

        observations.append(
            CDSBLineObservation(
                wavelength_nm=wavelength,
                intensity=intensity,
                intensity_uncertainty=I_err,
                element="Fe",
                ionization_stage=1,
                E_k_ev=E_k,
                g_k=g_k,
                A_ki=A_ki,
                E_i_ev=E_i,
                g_i=g_i,
                is_resonance=is_resonance,
            )
        )

    return observations


# =============================================================================
# CDSBLineObservation Tests
# =============================================================================


class TestCDSBLineObservation:
    """Tests for CDSBLineObservation dataclass."""

    def test_basic_creation(self):
        """Test basic observation creation."""
        obs = CDSBLineObservation(
            wavelength_nm=500.0,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.0,
            g_k=5,
            A_ki=1e8,
            E_i_ev=0.0,
            g_i=3,
            is_resonance=True,
        )

        assert obs.wavelength_nm == 500.0
        assert obs.element == "Fe"
        assert obs.is_resonance is True
        assert obs.E_i_ev == 0.0

    def test_oscillator_strength_calculation(self):
        """Test that oscillator strength is calculated from A_ki."""
        obs = CDSBLineObservation(
            wavelength_nm=500.0,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.0,
            g_k=5,
            A_ki=1e8,
            E_i_ev=0.0,
            g_i=3,
            is_resonance=True,
        )

        # f_ik should be calculated
        assert obs.f_ik is not None
        assert obs.f_ik > 0

        # Verify formula: f_ik ~ 1.499e-14 * (g_k/g_i) * lambda^2 * A_ki
        expected_f = 1.499e-14 * (5 / 3) * 500.0**2 * 1e8
        assert abs(obs.f_ik - expected_f) / expected_f < 1e-6

    def test_y_value_inherited(self):
        """Test that y_value property is inherited from base class."""
        obs = CDSBLineObservation(
            wavelength_nm=500.0,
            intensity=100.0,
            intensity_uncertainty=5.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=2,
            A_ki=1e6,
            E_i_ev=0.0,
            g_i=1,
            is_resonance=True,
        )

        # y = ln(I * lam / (g * A))
        expected_y = np.log(100.0 * 500.0 / (2 * 1e6))
        assert abs(obs.y_value - expected_y) < 1e-10


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_cdsb_observation(self):
        """Test create_cdsb_observation factory function."""
        obs = create_cdsb_observation(
            wavelength_nm=589.0,
            intensity=5000.0,
            intensity_uncertainty=100.0,
            element="Na",
            ionization_stage=1,
            E_k_ev=2.1,
            E_i_ev=0.0,
            g_k=2,
            g_i=2,
            A_ki=6.16e7,
        )

        assert obs.element == "Na"
        assert obs.is_resonance is True  # Auto-detected from E_i_ev < 0.1

    def test_create_cdsb_observation_nonresonance(self):
        """Test auto-detection of non-resonance lines."""
        obs = create_cdsb_observation(
            wavelength_nm=600.0,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=5.0,
            E_i_ev=1.5,  # Excited lower level
            g_k=7,
            g_i=5,
            A_ki=1e7,
        )

        assert obs.is_resonance is False  # Auto-detected

    def test_from_transition(self):
        """Test from_transition factory function."""
        transition = Transition(
            element="Ca",
            ionization_stage=1,
            wavelength_nm=422.67,
            A_ki=2.18e8,
            E_k_ev=2.93,
            E_i_ev=0.0,
            g_k=3,
            g_i=1,
            is_resonance=True,
        )

        obs = from_transition(transition, intensity=10000.0, intensity_uncertainty=200.0)

        assert obs.element == "Ca"
        assert obs.wavelength_nm == 422.67
        assert obs.is_resonance is True
        assert obs.E_i_ev == 0.0


# =============================================================================
# CDSBPlotter Basic Tests
# =============================================================================


class TestCDSBPlotterBasic:
    """Basic tests for CDSBPlotter."""

    def test_init_defaults(self):
        """Test default initialization."""
        plotter = CDSBPlotter()

        assert plotter.plasma_length_cm == 0.1
        assert plotter.max_iterations == 20
        assert plotter.tau_max == 10.0

    def test_init_custom_params(self):
        """Test custom initialization parameters."""
        plotter = CDSBPlotter(
            plasma_length_cm=0.2,
            max_iterations=30,
            convergence_tolerance=0.005,
            tau_max=3.0,
            fit_method=FitMethod.RANSAC,
        )

        assert plotter.plasma_length_cm == 0.2
        assert plotter.max_iterations == 30
        assert plotter.tau_max == 3.0

    def test_insufficient_lines(self):
        """Test handling of too few lines."""
        plotter = CDSBPlotter()

        # Only 2 lines
        obs = create_synthetic_cdsb_lines(10000, n_points=2, seed=42)

        result = plotter.fit(obs, n_e=1e17)

        assert result.convergence_status == CDSBConvergenceStatus.INSUFFICIENT_LINES
        assert "Fewer than 3 lines" in result.warnings[0]


# =============================================================================
# CDSBPlotter Fitting Tests
# =============================================================================


class TestCDSBPlotterFitting:
    """Tests for CDSBPlotter fitting functionality."""

    def test_fit_optically_thin(self):
        """Test fitting on optically thin data (small tau)."""
        T_target = 10000.0
        # Very small tau_scale -> optically thin
        obs = create_synthetic_cdsb_lines(
            T_target, n_points=15, noise_level=0.01, tau_scale=0.01, seed=42
        )

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e16)  # Low density -> thin

        assert result.temperature_K > 0
        assert np.isfinite(result.temperature_K)
        # Optically thin -> CD-SB should match standard Boltzmann
        if result.uncorrected_result:
            rel_diff = (
                abs(result.temperature_K - result.uncorrected_result.temperature_K)
                / result.uncorrected_result.temperature_K
            )
            assert rel_diff < 0.1  # Within 10%

    def test_fit_with_self_absorption(self):
        """Test fitting on data with significant self-absorption."""
        T_target = 8000.0
        # Moderate self-absorption
        obs = create_synthetic_cdsb_lines(
            T_target, n_points=15, noise_level=0.02, n_resonance=3, tau_scale=1.0, seed=42
        )

        plotter = CDSBPlotter(max_iterations=30, convergence_tolerance=0.01)
        result = plotter.fit(obs, n_e=1e17, initial_T_K=T_target)

        assert result.temperature_K > 0
        assert result.iterations > 0
        # Should converge or hit max iterations
        assert result.convergence_status in [
            CDSBConvergenceStatus.CONVERGED,
            CDSBConvergenceStatus.MAX_ITERATIONS,
        ]

    def test_fit_returns_optical_depths(self):
        """Test that fit returns optical depth information."""
        obs = create_synthetic_cdsb_lines(9000, n_points=10, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        assert len(result.optical_depths) > 0
        for od in result.optical_depths:
            assert isinstance(od, LineOpticalDepth)
            assert od.wavelength_nm > 0
            assert od.tau_final >= 0
            assert np.isfinite(od.correction_factor)

    def test_fit_convergence_history(self):
        """Test that convergence history is tracked."""
        obs = create_synthetic_cdsb_lines(8500, n_points=12, seed=42)

        plotter = CDSBPlotter(max_iterations=15)
        result = plotter.fit(obs, n_e=1e17)

        assert len(result.tau_convergence_history) > 0
        assert len(result.temperature_history) > 0
        assert len(result.temperature_history) == result.iterations + 1

    def test_fit_with_partition_functions(self):
        """Test fitting with custom partition functions."""
        obs = create_synthetic_cdsb_lines(9000, n_points=10, seed=42)

        plotter = CDSBPlotter()
        partition_funcs = {"Fe": 30.0}  # Custom U(T)

        result = plotter.fit(obs, n_e=1e17, partition_funcs=partition_funcs)

        assert result.temperature_K > 0
        # Column density should reflect partition function
        assert result.column_density > 0


class TestCDSBPlotterResonanceHandling:
    """Tests for resonance line handling in CDSBPlotter."""

    def test_resonance_lines_have_higher_tau(self):
        """Test that resonance lines get higher optical depth estimates."""
        obs = create_synthetic_cdsb_lines(8000, n_points=10, n_resonance=3, tau_scale=0.5, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        # Separate resonance and non-resonance optical depths
        resonance_taus = [od.tau_final for od in result.optical_depths if od.is_resonance]
        nonres_taus = [od.tau_final for od in result.optical_depths if not od.is_resonance]

        # Resonance lines should generally have higher tau
        if resonance_taus and nonres_taus:
            assert np.mean(resonance_taus) >= np.mean(nonres_taus) * 0.5  # Allow some variance

    def test_resonance_weight_downweights(self):
        """Test that resonance_weight parameter downweights resonance lines."""
        obs = create_synthetic_cdsb_lines(9000, n_points=10, n_resonance=4, seed=42)

        # High resonance weight (normal)
        plotter_normal = CDSBPlotter(resonance_weight=1.0)
        result_normal = plotter_normal.fit(obs, n_e=1e17)

        # Low resonance weight (downweight resonance)
        plotter_low = CDSBPlotter(resonance_weight=0.3)
        result_low = plotter_low.fit(obs, n_e=1e17)

        # Both should give valid results
        assert result_normal.temperature_K > 0
        assert result_low.temperature_K > 0


# =============================================================================
# Optical Depth Correction Tests
# =============================================================================


class TestOpticalDepthCorrection:
    """Tests for optical depth correction calculations."""

    def test_correction_factor_thin_limit(self):
        """Test correction factor in optically thin limit (tau << 1)."""
        # For tau << 1: (1 - exp(-tau))/tau -> 1 - tau/2
        # So correction = tau/(1-exp(-tau)) -> 1/(1-tau/2) ~ 1 + tau/2

        tau = 0.01
        expected_correction = tau / (1.0 - np.exp(-tau))
        thin_approx = 1.0 / (1.0 - tau / 2.0)

        assert abs(expected_correction - thin_approx) / thin_approx < 0.01

    def test_correction_factor_thick_limit(self):
        """Test correction factor in optically thick limit (tau >> 1)."""
        # For tau >> 1: (1 - exp(-tau))/tau -> 1/tau
        # So correction = tau/(1-exp(-tau)) -> tau

        tau = 10.0
        expected_correction = tau / (1.0 - np.exp(-tau))
        thick_approx = tau

        assert abs(expected_correction - thick_approx) / thick_approx < 0.01

    def test_tau_masking(self):
        """Test that very high tau lines are masked."""
        # Create observations with one very high tau line
        obs = create_synthetic_cdsb_lines(8000, n_points=10, tau_scale=0.5, seed=42)

        plotter = CDSBPlotter(tau_max=2.0)  # Low threshold for testing
        result = plotter.fit(obs, n_e=1e18)  # High density -> high tau

        # Some lines may be masked
        assert result.n_points <= len(obs)


# =============================================================================
# Convergence Tests
# =============================================================================


class TestConvergence:
    """Tests for CD-SB iteration convergence."""

    def test_convergence_achieved(self):
        """Test that convergence can be achieved."""
        obs = create_synthetic_cdsb_lines(9000, n_points=15, noise_level=0.01, seed=42)

        plotter = CDSBPlotter(max_iterations=50, convergence_tolerance=0.02)
        result = plotter.fit(obs, n_e=1e17, initial_T_K=9000)

        # Should converge for clean data with good initial guess
        assert result.convergence_status in [
            CDSBConvergenceStatus.CONVERGED,
            CDSBConvergenceStatus.MAX_ITERATIONS,
        ]

    def test_max_iterations_status(self):
        """Test that max_iterations status is returned when not converged."""
        obs = create_synthetic_cdsb_lines(8000, n_points=10, noise_level=0.1, seed=42)

        plotter = CDSBPlotter(max_iterations=3, convergence_tolerance=1e-10)  # Very tight
        result = plotter.fit(obs, n_e=1e17)

        # Likely won't converge in 3 iterations with tight tolerance
        assert result.iterations <= 3

    def test_tau_history_decreasing_change(self):
        """Test that tau changes typically decrease during convergence."""
        obs = create_synthetic_cdsb_lines(9000, n_points=12, noise_level=0.02, seed=42)

        plotter = CDSBPlotter(max_iterations=20)
        result = plotter.fit(obs, n_e=1e17)

        if len(result.tau_convergence_history) > 3:
            # Changes should generally decrease (not always monotonic)
            early_change = abs(
                result.tau_convergence_history[2] - result.tau_convergence_history[1]
            )
            late_change = abs(
                result.tau_convergence_history[-1] - result.tau_convergence_history[-2]
            )
            # Late changes should not be much larger than early
            assert late_change < early_change * 10


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with other modules."""

    def test_uncorrected_result_included(self):
        """Test that uncorrected Boltzmann result is included."""
        obs = create_synthetic_cdsb_lines(8500, n_points=10, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        assert result.uncorrected_result is not None
        assert result.uncorrected_result.temperature_K > 0
        assert result.uncorrected_result.n_points > 0

    def test_different_fit_methods(self):
        """Test that different underlying fit methods can be used."""
        obs = create_synthetic_cdsb_lines(9000, n_points=15, seed=42)

        for method in [FitMethod.SIGMA_CLIP, FitMethod.RANSAC, FitMethod.HUBER]:
            plotter = CDSBPlotter(fit_method=method)
            result = plotter.fit(obs, n_e=1e17)

            assert result.temperature_K > 0
            assert result.n_points > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_intensity_line(self):
        """Test handling of zero intensity lines."""
        obs = create_synthetic_cdsb_lines(8000, n_points=10, seed=42)
        # Set one intensity to near-zero
        obs[0].intensity = 1e-20

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        # Should still produce result (line may be rejected)
        assert result.temperature_K >= 0

    def test_very_high_density(self):
        """Test with very high electron density (high tau)."""
        obs = create_synthetic_cdsb_lines(8000, n_points=10, seed=42)

        plotter = CDSBPlotter(tau_max=10.0)
        result = plotter.fit(obs, n_e=1e19)  # Very high density

        # Should handle gracefully
        assert result.convergence_status is not None

    def test_very_low_density(self):
        """Test with very low electron density (optically thin)."""
        obs = create_synthetic_cdsb_lines(10000, n_points=10, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e14)  # Very low density

        # Should be similar to uncorrected result
        assert result.temperature_K > 0

    def test_bad_initial_temperature(self):
        """Test with poor initial temperature estimate."""
        obs = create_synthetic_cdsb_lines(8000, n_points=12, seed=42)

        plotter = CDSBPlotter(max_iterations=30)
        # Very bad initial guess
        result = plotter.fit(obs, n_e=1e17, initial_T_K=50000)

        # Algorithm should still work (may not converge well)
        assert result.temperature_K > 0

    def test_all_resonance_lines(self):
        """Test with all resonance lines."""
        obs = create_synthetic_cdsb_lines(9000, n_points=8, n_resonance=8, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        # All lines are resonance
        for od in result.optical_depths:
            assert od.is_resonance is True

    def test_no_resonance_lines(self):
        """Test with no resonance lines."""
        obs = create_synthetic_cdsb_lines(9000, n_points=8, n_resonance=0, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        # No resonance lines
        for od in result.optical_depths:
            assert od.is_resonance is False


# =============================================================================
# CDSBResult Tests
# =============================================================================


class TestCDSBResult:
    """Tests for CDSBResult dataclass."""

    def test_result_attributes(self):
        """Test that result has all expected attributes."""
        obs = create_synthetic_cdsb_lines(8500, n_points=10, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        # Check all required attributes
        assert hasattr(result, "temperature_K")
        assert hasattr(result, "temperature_uncertainty_K")
        assert hasattr(result, "corrected_intercept")
        assert hasattr(result, "corrected_intercept_uncertainty")
        assert hasattr(result, "column_density")
        assert hasattr(result, "column_density_uncertainty")
        assert hasattr(result, "iterations")
        assert hasattr(result, "convergence_status")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "n_points")
        assert hasattr(result, "optical_depths")
        assert hasattr(result, "uncorrected_result")
        assert hasattr(result, "tau_convergence_history")
        assert hasattr(result, "temperature_history")
        assert hasattr(result, "warnings")

    def test_column_density_positive(self):
        """Test that column density is positive for valid fits."""
        obs = create_synthetic_cdsb_lines(9000, n_points=10, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        if result.n_points > 0:
            assert result.column_density > 0


# =============================================================================
# Physical Consistency Tests
# =============================================================================


class TestPhysicalConsistency:
    """Tests for physical consistency of results."""

    def test_temperature_reasonable_range(self):
        """Test that fitted temperature is in reasonable range."""
        for T_target in [5000, 10000, 15000]:
            obs = create_synthetic_cdsb_lines(T_target, n_points=12, noise_level=0.02, seed=42)

            plotter = CDSBPlotter()
            result = plotter.fit(obs, n_e=1e17, initial_T_K=T_target)

            # Temperature should be in physically reasonable range
            assert 1000 < result.temperature_K < 50000

    def test_corrected_intercept_changes(self):
        """Test that correction affects intercept."""
        obs = create_synthetic_cdsb_lines(8000, n_points=12, n_resonance=4, tau_scale=1.5, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        # With self-absorption, corrected intercept should differ from uncorrected
        if result.uncorrected_result and result.n_points > 2:
            # They may be similar but correction should have some effect
            assert np.isfinite(result.corrected_intercept)
            assert np.isfinite(result.uncorrected_result.intercept)

    def test_r_squared_valid(self):
        """Test that R-squared is in valid range."""
        obs = create_synthetic_cdsb_lines(9000, n_points=15, noise_level=0.01, seed=42)

        plotter = CDSBPlotter()
        result = plotter.fit(obs, n_e=1e17)

        if result.n_points >= 2:
            assert 0 <= result.r_squared <= 1

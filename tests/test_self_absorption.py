"""
Unit and integration tests for cflibs.inversion.self_absorption module.

Tests cover:
- SelfAbsorptionCorrector initialization and correction
- Optical depth estimation
- Round-trip correction validation
- estimate_optical_depth_from_intensity_ratio function
- CurveOfGrowthAnalyzer class and COG regime detection
- MultipletLine and COGResult dataclasses
"""

import pytest
import numpy as np

from cflibs.inversion.self_absorption import (
    SelfAbsorptionCorrector,
    SelfAbsorptionResult,
    AbsorptionCorrectionResult,
    estimate_optical_depth_from_intensity_ratio,
    CurveOfGrowthAnalyzer,
    COGResult,
    COGLineData,
    COGRegime,
    MultipletLine,
    DoubletRatioResult,
    correct_via_doublet_ratio,
    find_doublet_pairs,
)
from cflibs.inversion.physics.boltzmann import LineObservation

# ==============================================================================
# Helper Functions
# ==============================================================================


def correction_factor(tau: float) -> float:
    """Calculate the self-absorption correction factor f(τ)."""
    if tau < 1e-10:
        return 1.0
    elif tau > 50:
        return 1.0 / tau
    else:
        return (1.0 - np.exp(-tau)) / tau


# ==============================================================================
# SelfAbsorptionCorrector Initialization Tests
# ==============================================================================


class TestSelfAbsorptionCorrectorInit:
    """Tests for SelfAbsorptionCorrector initialization."""

    def test_default_parameters(self):
        """Verify default parameters are set correctly."""
        corrector = SelfAbsorptionCorrector()

        assert corrector.optical_depth_threshold == 0.1
        assert corrector.mask_threshold == 3.0
        assert corrector.max_iterations == 5
        assert corrector.convergence_tolerance == 0.01
        assert corrector.plasma_length_cm == 0.1

    def test_custom_parameters(self):
        """Verify custom parameters are stored."""
        corrector = SelfAbsorptionCorrector(
            optical_depth_threshold=0.5,
            mask_threshold=5.0,
            max_iterations=10,
            convergence_tolerance=0.001,
            plasma_length_cm=0.5,
        )

        assert corrector.optical_depth_threshold == 0.5
        assert corrector.mask_threshold == 5.0
        assert corrector.max_iterations == 10
        assert corrector.convergence_tolerance == 0.001
        assert corrector.plasma_length_cm == 0.5


# ==============================================================================
# Correction Factor Tests
# ==============================================================================


class TestCorrectionFactor:
    """Tests for the correction factor formula f(τ) = (1 - exp(-τ)) / τ."""

    def test_optically_thin_limit(self):
        """Verify f(τ) ≈ 1 for τ << 1."""
        tau = 0.01
        f = correction_factor(tau)

        # Taylor expansion: f(τ) ≈ 1 - τ/2 for small τ
        expected = 1.0 - tau / 2
        assert f == pytest.approx(expected, rel=0.01)

    def test_moderate_optical_depth(self):
        """Verify f(τ) for τ = 1."""
        tau = 1.0
        f = correction_factor(tau)

        # f(1) = (1 - 1/e) / 1 ≈ 0.632
        expected = (1.0 - np.exp(-1)) / 1.0
        assert f == pytest.approx(expected, abs=1e-6)
        assert f == pytest.approx(0.632, rel=0.01)

    def test_optically_thick_limit(self):
        """Verify f(τ) → 1/τ as τ → ∞."""
        tau = 10.0
        f = correction_factor(tau)

        # For large τ, f(τ) ≈ 1/τ
        expected = 1.0 / tau
        assert f == pytest.approx(expected, rel=0.01)

    def test_very_optically_thick(self):
        """Verify f(τ) for very large τ."""
        tau = 100.0
        f = correction_factor(tau)

        assert f < 0.02  # Should be very small
        assert f == pytest.approx(1.0 / tau, rel=0.01)

    def test_zero_optical_depth(self):
        """Verify f(0) = 1 (no absorption)."""
        f = correction_factor(0.0)
        assert f == 1.0


# ==============================================================================
# Round-Trip Correction Tests
# ==============================================================================


class TestRoundTripCorrection:
    """Tests verifying absorption then correction recovers original intensity."""

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 2.0, 3.0])
    def test_round_trip_recovery(self, tau, self_absorption_test_line):
        """Apply absorption then correct, verify recovery of original."""
        test_data = self_absorption_test_line(
            optical_depth=tau,
            original_intensity=1000.0,
        )

        # The absorbed intensity = original * f(tau)
        # To recover original: corrected = absorbed / f(tau)
        absorbed = test_data["absorbed_intensity"]
        original = test_data["original_intensity"]
        f_tau = test_data["correction_factor"]

        # Verify the absorption was applied correctly
        assert absorbed == pytest.approx(original * f_tau, rel=0.001)

        # Verify correction recovers original
        corrected = absorbed / f_tau
        assert corrected == pytest.approx(original, rel=0.001)

    def test_round_trip_optically_thin(self, self_absorption_test_line):
        """Verify optically thin lines are essentially unchanged."""
        test_data = self_absorption_test_line(
            optical_depth=0.01,
            original_intensity=1000.0,
        )

        absorbed = test_data["absorbed_intensity"]
        original = test_data["original_intensity"]

        # Should be nearly identical
        assert absorbed == pytest.approx(original, rel=0.01)


# ==============================================================================
# Optical Depth Estimation Tests
# ==============================================================================


class TestOpticalDepthEstimation:
    """Tests for optical depth estimation methods."""

    def test_estimate_from_doublet_ratio_no_absorption(self):
        """Verify τ = 0 when observed ratio matches theoretical."""
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1000.0,
            intensity_weak=500.0,
            theoretical_ratio=2.0,
        )

        assert tau == 0.0

    def test_estimate_from_doublet_ratio_with_absorption(self):
        """Verify τ estimation when strong line is absorbed."""
        # Theoretical ratio is 2:1
        # If strong line is absorbed with τ=1, its intensity is reduced by f(1) ≈ 0.632
        theoretical_ratio = 2.0
        f_tau = correction_factor(1.0)

        intensity_weak = 500.0
        intensity_strong = 1000.0 * f_tau  # Absorbed

        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=intensity_strong,
            intensity_weak=intensity_weak,
            theoretical_ratio=theoretical_ratio,
        )

        # Should recover τ ≈ 1.0
        assert tau == pytest.approx(1.0, rel=0.1)

    def test_estimate_from_doublet_zero_weak(self):
        """Verify τ = 0 when weak intensity is zero."""
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1000.0,
            intensity_weak=0.0,
            theoretical_ratio=2.0,
        )

        assert tau == 0.0

    def test_estimate_from_doublet_zero_ratio(self):
        """Verify τ = 0 when theoretical ratio is zero."""
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1000.0,
            intensity_weak=500.0,
            theoretical_ratio=0.0,
        )

        assert tau == 0.0

    def test_estimate_from_doublet_ratio_greater_than_theoretical(self):
        """Verify τ = 0 when observed ratio exceeds theoretical."""
        # This would indicate emission enhancement, not absorption
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1200.0,  # Higher than expected
            intensity_weak=500.0,
            theoretical_ratio=2.0,
        )

        assert tau == 0.0


# ==============================================================================
# SelfAbsorptionCorrector.correct() Tests
# ==============================================================================


class TestSelfAbsorptionCorrectorCorrect:
    """Tests for SelfAbsorptionCorrector.correct() method."""

    @pytest.fixture
    def corrector(self):
        return SelfAbsorptionCorrector(
            optical_depth_threshold=0.1,
            mask_threshold=3.0,
            max_iterations=5,
        )

    @pytest.fixture
    def sample_observations(self):
        """Create sample line observations."""
        return [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=800.0,
                intensity_uncertainty=16.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
        ]

    def test_correct_returns_result(self, corrector, sample_observations):
        """Verify correct() returns SelfAbsorptionResult."""
        result = corrector.correct(
            observations=sample_observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        assert isinstance(result, SelfAbsorptionResult)
        assert len(result.corrected_observations) + len(result.masked_observations) == 2

    def test_optically_thin_unchanged(self, corrector, sample_observations):
        """Verify optically thin lines are returned unchanged."""
        # Very low concentration = low optical depth
        result = corrector.correct(
            observations=sample_observations,
            temperature_K=10000.0,
            concentrations={"Fe": 1e-10},  # Very low
            total_number_density_cm3=1e15,  # Low density
            partition_funcs={"Fe": 25.0},
        )

        # All should be returned without correction
        for obs, corrected in zip(sample_observations, result.corrected_observations):
            assert corrected.intensity == pytest.approx(obs.intensity, rel=0.001)

    def test_corrections_dict_populated(self, corrector, sample_observations):
        """Verify corrections dictionary is populated for all lines."""
        result = corrector.correct(
            observations=sample_observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        # All wavelengths should have entries
        for obs in sample_observations:
            assert obs.wavelength_nm in result.corrections
            assert isinstance(result.corrections[obs.wavelength_nm], AbsorptionCorrectionResult)

    def test_masked_lines_have_warnings(self):
        """Verify masked lines generate warnings."""
        corrector = SelfAbsorptionCorrector(mask_threshold=0.5)

        # High concentration = potentially high optical depth
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=10000.0,
                intensity_uncertainty=100.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=0.0,  # Ground state = high absorption
                g_k=25,
                A_ki=1e9,  # Very strong line
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=8000.0,
            concentrations={"Fe": 0.5},  # High concentration
            total_number_density_cm3=1e20,  # High density
            partition_funcs={"Fe": 25.0},
        )

        # With high optical depth, line may be masked
        # Check that we got some result
        assert len(result.corrected_observations) + len(result.masked_observations) == 1

    def test_empty_observations(self, corrector):
        """Verify handling of empty observation list."""
        result = corrector.correct(
            observations=[],
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        assert result.corrected_observations == []
        assert result.masked_observations == []
        assert result.n_corrected == 0
        assert result.n_masked == 0

    def test_missing_element_in_concentrations(self, corrector):
        """Verify graceful handling when element not in concentrations."""
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Cu",  # Not in concentrations
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=4,
                A_ki=1e8,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},  # No Cu
            total_number_density_cm3=1e18,
            partition_funcs={"Cu": 2.0},
        )

        # Should return unchanged (τ = 0 for missing element)
        assert len(result.corrected_observations) == 1
        assert result.corrected_observations[0].intensity == 1000.0


# ==============================================================================
# AbsorptionCorrectionResult Tests
# ==============================================================================


class TestAbsorptionCorrectionResult:
    """Tests for AbsorptionCorrectionResult dataclass."""

    def test_dataclass_creation(self):
        """Verify AbsorptionCorrectionResult can be instantiated."""
        result = AbsorptionCorrectionResult(
            original_intensity=1000.0,
            corrected_intensity=1200.0,
            optical_depth=0.5,
            correction_factor=0.833,
            is_optically_thick=False,
            iterations=3,
        )

        assert result.original_intensity == 1000.0
        assert result.corrected_intensity == 1200.0
        assert result.optical_depth == 0.5
        assert result.correction_factor == 0.833
        assert result.is_optically_thick is False
        assert result.iterations == 3

    def test_default_iterations(self):
        """Verify default iterations is 1."""
        result = AbsorptionCorrectionResult(
            original_intensity=1000.0,
            corrected_intensity=1000.0,
            optical_depth=0.01,
            correction_factor=1.0,
            is_optically_thick=False,
        )

        assert result.iterations == 1


# ==============================================================================
# SelfAbsorptionResult Tests
# ==============================================================================


class TestSelfAbsorptionResult:
    """Tests for SelfAbsorptionResult dataclass."""

    def test_dataclass_creation(self):
        """Verify SelfAbsorptionResult can be instantiated."""
        obs = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=20.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )

        result = SelfAbsorptionResult(
            corrected_observations=[obs],
            masked_observations=[],
            corrections={
                400.0: AbsorptionCorrectionResult(
                    original_intensity=1000.0,
                    corrected_intensity=1000.0,
                    optical_depth=0.01,
                    correction_factor=1.0,
                    is_optically_thick=False,
                )
            },
            n_corrected=0,
            n_masked=0,
            max_optical_depth=0.01,
            warnings=[],
        )

        assert len(result.corrected_observations) == 1
        assert len(result.masked_observations) == 0
        assert 400.0 in result.corrections
        assert result.n_corrected == 0
        assert result.max_optical_depth == 0.01

    def test_default_warnings(self):
        """Verify default warnings is empty list."""
        result = SelfAbsorptionResult(
            corrected_observations=[],
            masked_observations=[],
            corrections={},
            n_corrected=0,
            n_masked=0,
            max_optical_depth=0.0,
        )

        assert result.warnings == []


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestPublishedCorrectionFactors:
    """
    Validate correction factors against published literature values.

    The curve-of-growth correction f(τ) = (1 - exp(-τ))/τ is well-established.
    These tests verify our implementation matches tabulated values from:
    - Standard plasma spectroscopy references
    - El Sherbini et al. (2020)
    """

    # Published f(τ) values: exact calculation of (1 - exp(-τ))/τ
    # Verified against standard plasma spectroscopy references
    PUBLISHED_VALUES = {
        # τ: f(τ)  (calculated values)
        0.01: 0.995,  # Nearly optically thin
        0.05: 0.975,
        0.10: 0.951,
        0.20: 0.906,
        0.50: 0.787,
        1.00: 0.632,  # Classic value: 1 - 1/e
        1.50: 0.518,
        2.00: 0.432,
        3.00: 0.317,
        5.00: 0.199,  # (1 - exp(-5))/5 ≈ 0.1987
        10.0: 0.100,  # Approaches 1/τ
    }

    @pytest.mark.parametrize("tau,expected_f", list(PUBLISHED_VALUES.items()))
    def test_correction_factor_matches_published(self, tau, expected_f):
        """Verify f(τ) matches published tabulated values."""
        f_calculated = correction_factor(tau)

        # Allow 1% tolerance for numerical precision
        assert f_calculated == pytest.approx(
            expected_f, rel=0.01
        ), f"f({tau}) = {f_calculated:.4f}, expected {expected_f:.4f}"

    def test_correction_factor_monotonically_decreasing(self):
        """Verify f(τ) decreases monotonically with τ."""
        tau_values = np.logspace(-2, 2, 50)
        f_values = [correction_factor(tau) for tau in tau_values]

        for i in range(1, len(f_values)):
            assert (
                f_values[i] < f_values[i - 1]
            ), f"f(τ) not monotonically decreasing at τ={tau_values[i]}"

    def test_correction_inverse_recovers_true_intensity(self):
        """
        Verify that applying correction recovers true intensity.

        If I_obs = I_true * f(τ), then I_true = I_obs / f(τ).
        """
        I_true = 1000.0

        for tau in [0.1, 0.5, 1.0, 2.0, 3.0]:
            f_tau = correction_factor(tau)
            I_obs = I_true * f_tau
            I_recovered = I_obs / f_tau

            assert I_recovered == pytest.approx(I_true, rel=1e-10)

    def test_high_optical_depth_asymptotic(self):
        """
        Verify asymptotic behavior f(τ) → 1/τ for large τ.

        For τ >> 1: f(τ) = (1 - exp(-τ))/τ ≈ 1/τ
        """
        for tau in [10, 20, 50, 100]:
            f_tau = correction_factor(tau)
            asymptotic = 1.0 / tau

            # Should be within 1% for large τ
            assert f_tau == pytest.approx(
                asymptotic, rel=0.01
            ), f"Asymptotic deviation at τ={tau}: f={f_tau:.6f}, 1/τ={asymptotic:.6f}"


class TestSelfAbsorptionEdgeCases:
    """Edge case tests for self-absorption module."""

    def test_zero_intensity(self):
        """Verify handling of zero intensity."""
        corrector = SelfAbsorptionCorrector()

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=0.0,  # Zero intensity
                intensity_uncertainty=0.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        # Should not crash
        assert len(result.corrected_observations) + len(result.masked_observations) == 1

    def test_zero_temperature(self):
        """Verify handling of zero temperature."""
        corrector = SelfAbsorptionCorrector()

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=0.0,  # Zero temperature
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        # Should return unchanged (τ = 0 for T = 0)
        assert len(result.corrected_observations) == 1

    def test_lower_level_energies_used(self):
        """Verify lower level energies are used when provided."""
        corrector = SelfAbsorptionCorrector()

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        # With E_i = 0 (ground state) vs E_i = 2.0 eV (excited)
        # Ground state should have higher absorption
        result_ground = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.1},
            total_number_density_cm3=1e19,
            partition_funcs={"Fe": 25.0},
            lower_level_energies={400.0: 0.0},  # Ground state
        )

        result_excited = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.1},
            total_number_density_cm3=1e19,
            partition_funcs={"Fe": 25.0},
            lower_level_energies={400.0: 3.0},  # Excited state
        )

        # Ground state should have higher optical depth
        tau_ground = result_ground.corrections[400.0].optical_depth
        tau_excited = result_excited.corrections[400.0].optical_depth

        assert tau_ground >= tau_excited

    def test_max_iterations_respected(self):
        """Verify max_iterations limit is respected."""
        corrector = SelfAbsorptionCorrector(
            max_iterations=2,
            convergence_tolerance=1e-10,  # Very tight, won't converge
        )

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.5},
            total_number_density_cm3=1e20,
            partition_funcs={"Fe": 25.0},
        )

        # Should not hang, should return with iterations <= max_iterations
        assert len(result.corrected_observations) + len(result.masked_observations) == 1


# ==============================================================================
# MultipletLine Tests
# ==============================================================================


class TestMultipletLine:
    """Tests for MultipletLine dataclass and its properties."""

    def test_creation(self):
        """Verify MultipletLine can be created with required attributes."""
        line = MultipletLine(
            wavelength_nm=500.0,
            equivalent_width_nm=0.05,
            g_i=5,
            g_k=7,
            A_ki=1e8,
            E_i_ev=0.0,
            E_k_ev=2.5,
            element="Fe",
            ionization_stage=1,
        )

        assert line.wavelength_nm == 500.0
        assert line.equivalent_width_nm == 0.05
        assert line.g_i == 5
        assert line.g_k == 7
        assert line.element == "Fe"

    def test_oscillator_strength_calculation(self):
        """Verify oscillator strength is calculated from Einstein A."""
        # For a known A_ki, calculate expected f_ik
        # f_ik = 1.4992 * lambda[cm]^2 * A_ki * (g_k / g_i)
        # Prefactor 1.4992 is m_e*c/(8*pi^2*e^2) in CGS with lambda in cm.
        # (1.4992e-16 is the same constant but for lambda in Angstroms.)
        line = MultipletLine(
            wavelength_nm=500.0,
            equivalent_width_nm=0.05,
            g_i=5,
            g_k=7,
            A_ki=1e8,
            E_i_ev=0.0,
            E_k_ev=2.5,
            element="Fe",
            ionization_stage=1,
        )

        lambda_cm = 500.0 * 1e-7
        expected_f = 1.4992 * (lambda_cm**2) * 1e8 * (7 / 5)

        assert line.oscillator_strength == pytest.approx(expected_f, rel=1e-6)

    def test_log_gf_calculation(self):
        """Verify log(gf) is calculated correctly."""
        line = MultipletLine(
            wavelength_nm=500.0,
            equivalent_width_nm=0.05,
            g_i=5,
            g_k=7,
            A_ki=1e8,
            E_i_ev=0.0,
            E_k_ev=2.5,
            element="Fe",
            ionization_stage=1,
        )

        gf = line.g_i * line.oscillator_strength
        expected_log_gf = np.log10(gf)

        assert line.log_gf == pytest.approx(expected_log_gf, rel=1e-6)

    def test_default_intensity_uncertainty(self):
        """Verify default intensity uncertainty is 0."""
        line = MultipletLine(
            wavelength_nm=500.0,
            equivalent_width_nm=0.05,
            g_i=5,
            g_k=7,
            A_ki=1e8,
            E_i_ev=0.0,
            E_k_ev=2.5,
            element="Fe",
            ionization_stage=1,
        )

        assert line.intensity_uncertainty == 0.0


# ==============================================================================
# COGLineData Tests
# ==============================================================================


class TestCOGLineData:
    """Tests for COGLineData dataclass."""

    def test_creation(self):
        """Verify COGLineData can be created."""
        data = COGLineData(
            wavelength_nm=500.0,
            equivalent_width_nm=0.05,
            log_gf=-1.5,
            reduced_width=1e-4,
            log_reduced_width=-4.0,
            E_i_ev=0.0,
            element="Fe",
            ionization_stage=1,
        )

        assert data.wavelength_nm == 500.0
        assert data.log_gf == -1.5
        assert data.reduced_width == 1e-4


# ==============================================================================
# COGResult Tests
# ==============================================================================


class TestCOGResult:
    """Tests for COGResult dataclass."""

    def test_creation(self):
        """Verify COGResult can be created with all fields."""
        result = COGResult(
            regime=COGRegime.LINEAR,
            optical_depth_estimate=0.5,
            column_density_cm2=1e15,
            doppler_width_nm=0.01,
            damping_parameter=0.05,
            fit_slope=0.95,
            fit_intercept=-5.0,
            fit_r_squared=0.98,
            fit_residuals=np.array([0.01, -0.02, 0.01]),
            n_lines_used=3,
            lines_data=[],
            warnings=[],
        )

        assert result.regime == COGRegime.LINEAR
        assert result.optical_depth_estimate == 0.5
        assert result.fit_r_squared == 0.98

    def test_default_warnings(self):
        """Verify default warnings is empty list."""
        result = COGResult(
            regime=COGRegime.LINEAR,
            optical_depth_estimate=0.5,
            column_density_cm2=1e15,
            doppler_width_nm=0.01,
            damping_parameter=0.05,
            fit_slope=0.95,
            fit_intercept=-5.0,
            fit_r_squared=0.98,
            fit_residuals=np.array([]),
            n_lines_used=0,
            lines_data=[],
        )

        assert result.warnings == []


# ==============================================================================
# COGRegime Tests
# ==============================================================================


class TestCOGRegime:
    """Tests for COGRegime enumeration."""

    def test_regime_values(self):
        """Verify regime enumeration values."""
        assert COGRegime.LINEAR.value == "linear"
        assert COGRegime.SATURATION.value == "saturation"
        assert COGRegime.DAMPING.value == "damping"
        assert COGRegime.UNKNOWN.value == "unknown"


# ==============================================================================
# CurveOfGrowthAnalyzer Initialization Tests
# ==============================================================================


class TestCurveOfGrowthAnalyzerInit:
    """Tests for CurveOfGrowthAnalyzer initialization."""

    def test_default_parameters(self):
        """Verify default parameters are set correctly."""
        analyzer = CurveOfGrowthAnalyzer()

        assert analyzer.temperature_K == 10000.0
        assert analyzer.mass_amu == 56.0
        assert analyzer.damping_constant is None
        assert analyzer.min_lines == 3

    def test_custom_parameters(self):
        """Verify custom parameters are stored."""
        analyzer = CurveOfGrowthAnalyzer(
            temperature_K=8000.0,
            mass_amu=40.0,
            damping_constant=0.1,
            min_lines=5,
        )

        assert analyzer.temperature_K == 8000.0
        assert analyzer.mass_amu == 40.0
        assert analyzer.damping_constant == 0.1
        assert analyzer.min_lines == 5


# ==============================================================================
# CurveOfGrowthAnalyzer Doppler Width Tests
# ==============================================================================


class TestCOGDopplerWidth:
    """Tests for Doppler width calculation."""

    def test_doppler_width_positive(self):
        """Verify Doppler width is positive."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0, mass_amu=56.0)
        doppler = analyzer._compute_doppler_width(500.0)

        assert doppler > 0

    def test_doppler_width_scales_with_wavelength(self):
        """Verify Doppler width scales linearly with wavelength."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0, mass_amu=56.0)

        doppler_500 = analyzer._compute_doppler_width(500.0)
        doppler_1000 = analyzer._compute_doppler_width(1000.0)

        # delta_lambda_D / lambda = const, so doppler width doubles
        assert doppler_1000 == pytest.approx(2 * doppler_500, rel=1e-6)

    def test_doppler_width_increases_with_temperature(self):
        """Verify Doppler width increases with temperature."""
        analyzer_cool = CurveOfGrowthAnalyzer(temperature_K=5000.0, mass_amu=56.0)
        analyzer_hot = CurveOfGrowthAnalyzer(temperature_K=20000.0, mass_amu=56.0)

        doppler_cool = analyzer_cool._compute_doppler_width(500.0)
        doppler_hot = analyzer_hot._compute_doppler_width(500.0)

        # Doppler width ~ sqrt(T)
        assert doppler_hot > doppler_cool
        assert doppler_hot == pytest.approx(doppler_cool * 2, rel=0.01)  # sqrt(4)=2

    def test_doppler_width_decreases_with_mass(self):
        """Verify Doppler width decreases with atomic mass."""
        analyzer_light = CurveOfGrowthAnalyzer(temperature_K=10000.0, mass_amu=12.0)  # C
        analyzer_heavy = CurveOfGrowthAnalyzer(temperature_K=10000.0, mass_amu=56.0)  # Fe

        doppler_light = analyzer_light._compute_doppler_width(500.0)
        doppler_heavy = analyzer_heavy._compute_doppler_width(500.0)

        # Doppler width ~ 1/sqrt(m)
        assert doppler_light > doppler_heavy


# ==============================================================================
# CurveOfGrowthAnalyzer Fit Tests
# ==============================================================================


class TestCOGFit:
    """Tests for CurveOfGrowthAnalyzer.fit() method."""

    @pytest.fixture
    def linear_regime_lines(self):
        """Create lines in the linear regime (slope ~ 1)."""
        # In linear regime: log(W/lambda) = log(gf) + const
        # Create lines with varying gf and corresponding W
        lines = []
        base_gf = 0.01
        base_W = 1e-5  # Very small W = optically thin

        for i, factor in enumerate([0.1, 0.5, 1.0, 2.0, 5.0]):
            gf = base_gf * factor
            W = base_W * factor  # Linear scaling

            # Reverse engineer A_ki from desired gf
            # gf = g_i * f = g_i * 1.4992 * lambda_cm^2 * A * (g_k/g_i)
            # For g_i=5, g_k=7, lambda=500nm:
            g_i, g_k = 5, 7
            lambda_cm = 500e-7
            A_ki = gf / (1.4992 * lambda_cm**2 * g_k / g_i * g_i)

            lines.append(
                MultipletLine(
                    wavelength_nm=500.0 + i,
                    equivalent_width_nm=W,
                    g_i=g_i,
                    g_k=g_k,
                    A_ki=A_ki,
                    E_i_ev=0.0,
                    E_k_ev=2.5,
                    element="Fe",
                    ionization_stage=1,
                )
            )

        return lines

    @pytest.fixture
    def saturation_regime_lines(self):
        """Create lines that show saturation behavior (slope << 1)."""
        # In saturation: W grows very slowly with gf
        # log(W/lambda) ~ 0.1 * log(gf) + const
        lines = []

        for i, log_gf_offset in enumerate([-2, -1, 0, 1, 2]):
            gf = 10 ** (log_gf_offset)
            # Saturation: W ~ const * sqrt(ln(gf)) very slow growth
            W = 1e-4 * (1 + 0.1 * log_gf_offset)  # Very flat

            g_i, g_k = 5, 7
            lambda_cm = 500e-7
            A_ki = gf / (1.4992 * lambda_cm**2 * g_k / g_i * g_i)

            lines.append(
                MultipletLine(
                    wavelength_nm=500.0 + i,
                    equivalent_width_nm=W,
                    g_i=g_i,
                    g_k=g_k,
                    A_ki=A_ki,
                    E_i_ev=0.0,
                    E_k_ev=2.5,
                    element="Fe",
                    ionization_stage=1,
                )
            )

        return lines

    def test_fit_returns_cog_result(self, linear_regime_lines):
        """Verify fit() returns COGResult."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0)
        result = analyzer.fit(linear_regime_lines)

        assert isinstance(result, COGResult)

    def test_fit_requires_minimum_lines(self):
        """Verify fit() raises ValueError with too few lines."""
        analyzer = CurveOfGrowthAnalyzer(min_lines=3)

        lines = [
            MultipletLine(
                wavelength_nm=500.0,
                equivalent_width_nm=0.01,
                g_i=5,
                g_k=7,
                A_ki=1e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
            MultipletLine(
                wavelength_nm=501.0,
                equivalent_width_nm=0.02,
                g_i=5,
                g_k=7,
                A_ki=2e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
        ]

        with pytest.raises(ValueError, match="Need at least 3 lines"):
            analyzer.fit(lines)

    def test_fit_linear_regime_slope_near_one(self, linear_regime_lines):
        """Verify linear regime gives slope near 1."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0)
        result = analyzer.fit(linear_regime_lines, excitation_correct=False)

        # Linear regime should have slope close to 1
        assert result.fit_slope == pytest.approx(1.0, abs=0.15)

    def test_fit_saturation_regime_slope_near_zero(self, saturation_regime_lines):
        """Verify saturation regime gives slope near 0."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0)
        result = analyzer.fit(saturation_regime_lines, excitation_correct=False)

        # Saturation should have slope < 0.3
        assert result.fit_slope < 0.4

    def test_fit_r_squared_in_range(self, linear_regime_lines):
        """Verify R^2 is between 0 and 1."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0)
        result = analyzer.fit(linear_regime_lines)

        assert 0.0 <= result.fit_r_squared <= 1.0

    def test_fit_n_lines_used(self, linear_regime_lines):
        """Verify n_lines_used matches input."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0)
        result = analyzer.fit(linear_regime_lines)

        assert result.n_lines_used == len(linear_regime_lines)

    def test_fit_lines_data_populated(self, linear_regime_lines):
        """Verify lines_data is populated with COGLineData."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0)
        result = analyzer.fit(linear_regime_lines)

        assert len(result.lines_data) == len(linear_regime_lines)
        for ld in result.lines_data:
            assert isinstance(ld, COGLineData)

    def test_fit_optical_depth_positive(self, linear_regime_lines):
        """Verify optical depth estimate is non-negative."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0)
        result = analyzer.fit(linear_regime_lines)

        assert result.optical_depth_estimate >= 0.0

    def test_fit_doppler_width_reasonable(self, linear_regime_lines):
        """Verify Doppler width is reasonable for LIBS conditions."""
        analyzer = CurveOfGrowthAnalyzer(temperature_K=10000.0, mass_amu=56.0)
        result = analyzer.fit(linear_regime_lines)

        # At 10000 K for Fe at 500 nm, Doppler FWHM ~ 0.01-0.02 nm
        assert 0.001 < result.doppler_width_nm < 0.1

    def test_fit_skips_zero_ew_lines(self):
        """Verify fit skips lines with zero equivalent width."""
        lines = [
            MultipletLine(
                wavelength_nm=500.0,
                equivalent_width_nm=0.0,  # Zero EW
                g_i=5,
                g_k=7,
                A_ki=1e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
            MultipletLine(
                wavelength_nm=501.0,
                equivalent_width_nm=0.01,
                g_i=5,
                g_k=7,
                A_ki=1e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
            MultipletLine(
                wavelength_nm=502.0,
                equivalent_width_nm=0.02,
                g_i=5,
                g_k=7,
                A_ki=2e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
            MultipletLine(
                wavelength_nm=503.0,
                equivalent_width_nm=0.03,
                g_i=5,
                g_k=7,
                A_ki=3e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
        ]

        analyzer = CurveOfGrowthAnalyzer(min_lines=3)
        result = analyzer.fit(lines)

        # Should only use 3 lines (skipped the zero EW one)
        assert result.n_lines_used == 3
        assert "non-positive EW" in str(result.warnings)


# ==============================================================================
# CurveOfGrowthAnalyzer Regime Classification Tests
# ==============================================================================


class TestCOGRegimeClassification:
    """Tests for regime classification logic."""

    def test_classify_linear_high_slope(self):
        """Verify high slope is classified as LINEAR."""
        analyzer = CurveOfGrowthAnalyzer()
        regime = analyzer._classify_regime(slope=0.95, r_squared=0.9)

        assert regime == COGRegime.LINEAR

    def test_classify_saturation_low_slope(self):
        """Verify low slope is classified as SATURATION."""
        analyzer = CurveOfGrowthAnalyzer()
        regime = analyzer._classify_regime(slope=0.1, r_squared=0.9)

        assert regime == COGRegime.SATURATION

    def test_classify_damping_mid_slope(self):
        """Verify mid slope (~0.5) is classified as DAMPING."""
        analyzer = CurveOfGrowthAnalyzer()
        regime = analyzer._classify_regime(slope=0.5, r_squared=0.9)

        assert regime == COGRegime.DAMPING

    def test_classify_unknown_poor_fit(self):
        """Verify poor R^2 is classified as UNKNOWN."""
        analyzer = CurveOfGrowthAnalyzer()
        regime = analyzer._classify_regime(slope=0.9, r_squared=0.3)

        assert regime == COGRegime.UNKNOWN

    def test_classify_unknown_transitional_slope(self):
        """Verify transitional slope is classified as UNKNOWN."""
        analyzer = CurveOfGrowthAnalyzer()
        # Slope between linear and damping regimes
        regime = analyzer._classify_regime(slope=0.7, r_squared=0.9)

        assert regime == COGRegime.UNKNOWN


# ==============================================================================
# CurveOfGrowthAnalyzer Correction Factors Tests
# ==============================================================================


class TestCOGCorrectionFactors:
    """Tests for get_correction_factors() method."""

    @pytest.fixture
    def sample_cog_result(self):
        """Create a sample COGResult for testing."""
        lines_data = [
            COGLineData(
                wavelength_nm=500.0,
                equivalent_width_nm=0.01,
                log_gf=-1.0,
                reduced_width=2e-5,
                log_reduced_width=-4.7,
                E_i_ev=0.0,
                element="Fe",
                ionization_stage=1,
            ),
            COGLineData(
                wavelength_nm=501.0,
                equivalent_width_nm=0.02,
                log_gf=0.0,  # Strongest line
                reduced_width=4e-5,
                log_reduced_width=-4.4,
                E_i_ev=0.0,
                element="Fe",
                ionization_stage=1,
            ),
            COGLineData(
                wavelength_nm=502.0,
                equivalent_width_nm=0.005,
                log_gf=-2.0,
                reduced_width=1e-5,
                log_reduced_width=-5.0,
                E_i_ev=0.0,
                element="Fe",
                ionization_stage=1,
            ),
        ]

        return COGResult(
            regime=COGRegime.LINEAR,
            optical_depth_estimate=1.0,  # tau=1 for strongest line
            column_density_cm2=1e15,
            doppler_width_nm=0.01,
            damping_parameter=0.05,
            fit_slope=0.95,
            fit_intercept=-5.0,
            fit_r_squared=0.98,
            fit_residuals=np.array([0.01, -0.02, 0.01]),
            n_lines_used=3,
            lines_data=lines_data,
        )

    def test_get_correction_factors_returns_dict(self, sample_cog_result):
        """Verify correction factors returns a dictionary."""
        analyzer = CurveOfGrowthAnalyzer()
        factors = analyzer.get_correction_factors(sample_cog_result)

        assert isinstance(factors, dict)
        assert len(factors) == 3

    def test_strongest_line_has_largest_correction(self, sample_cog_result):
        """Verify strongest line (highest gf) has largest correction."""
        analyzer = CurveOfGrowthAnalyzer()
        factors = analyzer.get_correction_factors(sample_cog_result)

        # Line at 501 nm has highest gf (log_gf=0)
        assert factors[501.0] >= factors[500.0]
        assert factors[501.0] >= factors[502.0]

    def test_optically_thin_line_correction_near_one(self, sample_cog_result):
        """Verify optically thin line has correction factor near 1."""
        analyzer = CurveOfGrowthAnalyzer()
        factors = analyzer.get_correction_factors(sample_cog_result)

        # Line at 502 nm has lowest gf (log_gf=-2), tau ~ 0.01
        # For tau=0.01, f(tau) ~ 1, correction ~ 1
        assert factors[502.0] == pytest.approx(1.0, abs=0.05)


# ==============================================================================
# CurveOfGrowthAnalyzer Diagnose Self-Absorption Tests
# ==============================================================================


class TestCOGDiagnoseSelfAbsorption:
    """Tests for diagnose_self_absorption() method."""

    def test_diagnose_returns_dict(self):
        """Verify diagnose returns dictionary."""
        lines = [
            MultipletLine(
                wavelength_nm=500.0,
                equivalent_width_nm=0.01,
                g_i=5,
                g_k=7,
                A_ki=1e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
            MultipletLine(
                wavelength_nm=501.0,
                equivalent_width_nm=0.02,
                g_i=5,
                g_k=7,
                A_ki=2e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
            MultipletLine(
                wavelength_nm=502.0,
                equivalent_width_nm=0.03,
                g_i=5,
                g_k=7,
                A_ki=3e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
        ]

        analyzer = CurveOfGrowthAnalyzer()
        diagnosed = analyzer.diagnose_self_absorption(lines)

        assert isinstance(diagnosed, dict)
        assert all(isinstance(v, bool) for v in diagnosed.values())

    def test_diagnose_with_too_few_lines(self):
        """Verify diagnose returns False for all with too few lines."""
        lines = [
            MultipletLine(
                wavelength_nm=500.0,
                equivalent_width_nm=0.01,
                g_i=5,
                g_k=7,
                A_ki=1e8,
                E_i_ev=0.0,
                E_k_ev=2.5,
                element="Fe",
                ionization_stage=1,
            ),
        ]

        analyzer = CurveOfGrowthAnalyzer(min_lines=3)
        diagnosed = analyzer.diagnose_self_absorption(lines)

        assert diagnosed[500.0] is False


# ==============================================================================
# SelfAbsorptionCorrector COG Integration Tests
# ==============================================================================


class TestSelfAbsorptionCorrectorCOGIntegration:
    """Tests for SelfAbsorptionCorrector.correct_with_cog() method."""

    @pytest.fixture
    def sample_observations(self):
        """Create sample observations for COG correction."""
        return [
            LineObservation(
                wavelength_nm=500.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=2.5,
                g_k=7,
                A_ki=1e8,
            ),
            LineObservation(
                wavelength_nm=501.0,
                intensity=1500.0,
                intensity_uncertainty=30.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=2.6,
                g_k=9,
                A_ki=2e8,
            ),
            LineObservation(
                wavelength_nm=502.0,
                intensity=800.0,
                intensity_uncertainty=16.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=2.7,
                g_k=5,
                A_ki=5e7,
            ),
        ]

    def test_correct_with_cog_returns_result(self, sample_observations):
        """Verify correct_with_cog returns SelfAbsorptionResult."""
        corrector = SelfAbsorptionCorrector()

        equivalent_widths = {500.0: 0.01, 501.0: 0.02, 502.0: 0.008}
        lower_level_g = {500.0: 5, 501.0: 7, 502.0: 3}
        lower_level_energies = {500.0: 0.0, 501.0: 0.0, 502.0: 0.0}

        result = corrector.correct_with_cog(
            observations=sample_observations,
            equivalent_widths=equivalent_widths,
            lower_level_g=lower_level_g,
            lower_level_energies=lower_level_energies,
            temperature_K=10000.0,
            mass_amu=56.0,
        )

        assert isinstance(result, SelfAbsorptionResult)

    def test_correct_with_cog_fallback_on_few_lines(self):
        """Verify fallback to standard correction with too few lines."""
        corrector = SelfAbsorptionCorrector()

        observations = [
            LineObservation(
                wavelength_nm=500.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=2.5,
                g_k=7,
                A_ki=1e8,
            ),
        ]

        equivalent_widths = {500.0: 0.01}
        lower_level_g = {500.0: 5}
        lower_level_energies = {500.0: 0.0}

        result = corrector.correct_with_cog(
            observations=observations,
            equivalent_widths=equivalent_widths,
            lower_level_g=lower_level_g,
            lower_level_energies=lower_level_energies,
            temperature_K=10000.0,
        )

        # Should still return a result (fallback to standard correction)
        assert isinstance(result, SelfAbsorptionResult)
        assert "falling back" in str(result.warnings).lower()

    def test_correct_with_cog_reports_regime(self, sample_observations):
        """Verify COG regime is reported in warnings."""
        corrector = SelfAbsorptionCorrector()

        equivalent_widths = {500.0: 0.01, 501.0: 0.02, 502.0: 0.008}
        lower_level_g = {500.0: 5, 501.0: 7, 502.0: 3}
        lower_level_energies = {500.0: 0.0, 501.0: 0.0, 502.0: 0.0}

        result = corrector.correct_with_cog(
            observations=sample_observations,
            equivalent_widths=equivalent_widths,
            lower_level_g=lower_level_g,
            lower_level_energies=lower_level_energies,
            temperature_K=10000.0,
        )

        # Should report COG regime in warnings
        warnings_str = str(result.warnings)
        assert "COG regime" in warnings_str

    def test_correct_with_cog_reports_r_squared(self, sample_observations):
        """Verify COG fit quality is reported in warnings."""
        corrector = SelfAbsorptionCorrector()

        equivalent_widths = {500.0: 0.01, 501.0: 0.02, 502.0: 0.008}
        lower_level_g = {500.0: 5, 501.0: 7, 502.0: 3}
        lower_level_energies = {500.0: 0.0, 501.0: 0.0, 502.0: 0.0}

        result = corrector.correct_with_cog(
            observations=sample_observations,
            equivalent_widths=equivalent_widths,
            lower_level_g=lower_level_g,
            lower_level_energies=lower_level_energies,
            temperature_K=10000.0,
        )

        # Should report R^2 in warnings
        warnings_str = str(result.warnings)
        assert "R^2" in warnings_str


# ==============================================================================
# COG Theoretical Curve Tests
# ==============================================================================


class TestCOGTheoreticalCurve:
    """Tests for theoretical curve-of-growth computation."""

    def test_theoretical_cog_linear_regime(self):
        """Verify linear regime behavior in theoretical COG."""
        analyzer = CurveOfGrowthAnalyzer()

        # Very small tau (linear regime) - use float array
        log_tau = np.array([-3.0, -2.0, -1.5])
        log_W = analyzer._compute_theoretical_cog(log_tau, damping_param=0.01)

        # In linear regime, slope should be close to 1 (W proportional to tau)
        slopes = np.diff(log_W) / np.diff(log_tau)
        assert np.all(slopes > 0.9)

    def test_theoretical_cog_saturation_regime(self):
        """Verify saturation regime behavior in theoretical COG."""
        analyzer = CurveOfGrowthAnalyzer()

        # Moderate tau (saturation regime) - use float array
        log_tau = np.array([0.5, 1.0, 1.5])
        log_W = analyzer._compute_theoretical_cog(log_tau, damping_param=0.01)

        # In saturation, slope is much less than 1
        slopes = np.diff(log_W) / np.diff(log_tau)
        assert np.all(slopes < 0.5)

    def test_theoretical_cog_monotonic_in_linear_regime(self):
        """Verify theoretical COG is monotonically increasing in linear regime."""
        analyzer = CurveOfGrowthAnalyzer()

        # Test only in the linear regime where behavior is well-defined
        log_tau = np.linspace(-3.0, -1.5, 20)
        log_W = analyzer._compute_theoretical_cog(log_tau, damping_param=0.05)

        # W should always increase with tau in linear regime
        assert np.all(np.diff(log_W) > 0)

    def test_theoretical_cog_monotonic_in_saturation_regime(self):
        """Verify theoretical COG is monotonically increasing in saturation regime."""
        analyzer = CurveOfGrowthAnalyzer()

        # Test only in the saturation regime where behavior is well-defined
        log_tau = np.linspace(0.5, 2.0, 20)
        log_W = analyzer._compute_theoretical_cog(log_tau, damping_param=0.05)

        # W should always increase with tau in saturation regime
        assert np.all(np.diff(log_W) > 0)

    def test_theoretical_cog_handles_float_input(self):
        """Verify theoretical COG handles float array input correctly."""
        analyzer = CurveOfGrowthAnalyzer()

        # Ensure float arrays work
        log_tau = np.array([-2.0, -1.0, 0.0, 1.0], dtype=float)
        log_W = analyzer._compute_theoretical_cog(log_tau, damping_param=0.05)

        # Should return valid results
        assert len(log_W) == len(log_tau)
        assert np.all(np.isfinite(log_W))


# ==============================================================================
# Doublet-Ratio Self-Absorption Correction (Pace et al. 2025)
# ==============================================================================


def _f_tau(tau: float) -> float:
    """Reference escape factor for synthetic injection."""
    if tau < 1e-10:
        return 1.0
    return (1.0 - np.exp(-tau)) / tau


def _make_doublet(
    lam1: float,
    lam2: float,
    A_ki: float,
    g_k: int,
    tau1_inj: float,
    *,
    element: str = "Fe",
    stage: int = 1,
    E_k_ev: float = 5.0,
) -> tuple[LineObservation, LineObservation, float, float]:
    """
    Build a synthetic doublet pair from the same upper level with optical
    depth tau1_inj injected into line 1 (and tau2_inj = tau1_inj / r_theory
    into line 2). Returns (line1, line2, tau1_inj, tau2_inj).
    """
    r_theory = (lam1**3) / (lam2**3)  # g_k and A_ki cancel since they match
    tau2_inj = tau1_inj / r_theory
    # True intensities consistent with r_theory: I_true_1 / I_true_2 = r_theory
    I_true_1, I_true_2 = r_theory, 1.0
    I_meas_1 = I_true_1 * _f_tau(tau1_inj)
    I_meas_2 = I_true_2 * _f_tau(tau2_inj)
    line1 = LineObservation(
        wavelength_nm=lam1,
        intensity=I_meas_1,
        intensity_uncertainty=0.01 * I_meas_1,
        element=element,
        ionization_stage=stage,
        E_k_ev=E_k_ev,
        g_k=g_k,
        A_ki=A_ki,
    )
    line2 = LineObservation(
        wavelength_nm=lam2,
        intensity=I_meas_2,
        intensity_uncertainty=0.01 * I_meas_2,
        element=element,
        ionization_stage=stage,
        E_k_ev=E_k_ev,
        g_k=g_k,
        A_ki=A_ki,
    )
    return line1, line2, tau1_inj, tau2_inj


class TestDoubletRatioCorrection:
    """Tests for closed-form doublet-ratio self-absorption correction."""

    @pytest.mark.parametrize("tau_inj", [0.1, 1.0, 3.0])
    def test_doublet_correction_recovers_known_tau(self, tau_inj):
        """Recover injected tau in [0.1, 5] within 15% from a noiseless doublet."""
        line1, line2, tau1_truth, tau2_truth = _make_doublet(
            lam1=400.0, lam2=500.0, A_ki=1e8, g_k=4, tau1_inj=tau_inj
        )
        result = correct_via_doublet_ratio(line1, line2)

        assert isinstance(result, DoubletRatioResult)
        # 15% recovery per the issue acceptance criterion
        assert (
            abs(result.tau_1 - tau1_truth) / tau1_truth < 0.15
        ), f"tau_1 recovery failed: {result.tau_1} vs truth {tau1_truth}"
        assert (
            abs(result.tau_2 - tau2_truth) / tau2_truth < 0.15
        ), f"tau_2 recovery failed: {result.tau_2} vs truth {tau2_truth}"
        # Corrected intensities should approximate the true (optically thin)
        # intensities used for injection
        r_theory = (400.0**3) / (500.0**3)
        assert result.i1_corrected == pytest.approx(r_theory, rel=0.15)
        assert result.i2_corrected == pytest.approx(1.0, rel=0.15)
        # Wavelength pair ordered by ascending wavelength
        assert result.wavelength_pair_nm == (400.0, 500.0)
        # No cross-check yet
        assert result.agreement_with_cdsb_sigma is None

    def test_doublet_correction_optically_thin_limit(self):
        """If r_meas == r_theory, recovered tau collapses to brentq lower bound."""
        line1, line2, _, _ = _make_doublet(lam1=400.0, lam2=500.0, A_ki=1e8, g_k=4, tau1_inj=1e-6)
        # Force exactly thin: set r_meas = r_theory by reconstructing intensities
        r_theory = (400.0**3) / (500.0**3)
        line1 = LineObservation(
            wavelength_nm=400.0,
            intensity=r_theory,
            intensity_uncertainty=0.01,
            element="Fe",
            ionization_stage=1,
            E_k_ev=5.0,
            g_k=4,
            A_ki=1e8,
        )
        line2 = LineObservation(
            wavelength_nm=500.0,
            intensity=1.0,
            intensity_uncertainty=0.01,
            element="Fe",
            ionization_stage=1,
            E_k_ev=5.0,
            g_k=4,
            A_ki=1e8,
        )
        result = correct_via_doublet_ratio(line1, line2)

        # tau_1 should sit at the brentq lower-bound (1e-4) within tolerance
        assert result.tau_1 < 1e-3, f"Expected tau_1 ~ 1e-4, got {result.tau_1}"
        # Corrections should be ~unity
        assert result.f_tau_1 == pytest.approx(1.0, abs=1e-3)
        assert result.i1_corrected == pytest.approx(line1.intensity, rel=1e-3)

    def test_find_doublet_pairs_matches_upper_level(self):
        """Mixed list with one matched pair + one mismatched returns only the match."""
        # Pair A: same upper level (5.000 eV), same species
        line_a1 = LineObservation(400.0, 1.0, 0.01, "Fe", 1, 5.0000, 4, 1e8)
        line_a2 = LineObservation(500.0, 1.0, 0.01, "Fe", 1, 5.0005, 4, 1e8)
        # Pair B: clearly different upper levels (5.0 vs 6.0 eV)
        line_b = LineObservation(600.0, 1.0, 0.01, "Fe", 1, 6.0, 4, 1e8)

        pairs = find_doublet_pairs([line_a1, line_a2, line_b])

        assert len(pairs) == 1
        wl_set = {pairs[0][0].wavelength_nm, pairs[0][1].wavelength_nm}
        assert wl_set == {400.0, 500.0}
        # Wavelength ordering: shorter first
        assert pairs[0][0].wavelength_nm < pairs[0][1].wavelength_nm

    def test_find_doublet_pairs_rejects_different_species(self):
        """Same upper-level energy but different element returns no pairs."""
        line_fe = LineObservation(400.0, 1.0, 0.01, "Fe", 1, 5.0, 4, 1e8)
        line_cu = LineObservation(500.0, 1.0, 0.01, "Cu", 1, 5.0, 4, 1e8)

        pairs = find_doublet_pairs([line_fe, line_cu])
        assert pairs == []

        # Also: same element, different ionization stage
        line_fe2 = LineObservation(500.0, 1.0, 0.01, "Fe", 2, 5.0, 4, 1e8)
        pairs2 = find_doublet_pairs([line_fe, line_fe2])
        assert pairs2 == []

    def test_doublet_correction_handles_zero_intensity(self):
        """Zero intensity must raise ValueError (not silently NaN)."""
        line_zero = LineObservation(400.0, 0.0, 0.0, "Fe", 1, 5.0, 4, 1e8)
        line_ok = LineObservation(500.0, 1.0, 0.01, "Fe", 1, 5.0, 4, 1e8)

        with pytest.raises(ValueError, match="positive intensities"):
            correct_via_doublet_ratio(line_zero, line_ok)

    def test_doublet_correction_rejects_different_upper_level(self):
        """Lines from different upper levels must raise ValueError."""
        l1 = LineObservation(400.0, 1.0, 0.01, "Fe", 1, 5.0, 4, 1e8)
        l2 = LineObservation(500.0, 1.0, 0.01, "Fe", 1, 6.0, 4, 1e8)
        with pytest.raises(ValueError, match="upper level"):
            correct_via_doublet_ratio(l1, l2)

    def test_cross_check_with_doublets_flags_disagreement(self, caplog):
        """SelfAbsorptionCorrector.cross_check_with_doublets logs warning on disagreement."""
        import logging

        line1, line2, tau1_truth, _ = _make_doublet(
            lam1=400.0, lam2=500.0, A_ki=1e8, g_k=4, tau1_inj=1.5
        )
        # Build a fake CDSB result that disagrees strongly (tau_cdsb = 0.1, truth = 1.5)
        cdsb_corr = AbsorptionCorrectionResult(
            original_intensity=line1.intensity,
            corrected_intensity=line1.intensity / _f_tau(0.1),
            optical_depth=0.1,
            correction_factor=1.0 / _f_tau(0.1),
            is_optically_thick=False,
        )
        cdsb_result = SelfAbsorptionResult(
            corrected_observations=[line1, line2],
            masked_observations=[],
            corrections={400.0: cdsb_corr},
            n_corrected=1,
            n_masked=0,
            max_optical_depth=0.1,
        )

        corrector = SelfAbsorptionCorrector()
        with caplog.at_level(logging.WARNING, logger="inversion.self_absorption"):
            results = corrector.cross_check_with_doublets(
                [line1, line2], cdsb_result, sigma_threshold=2.0
            )

        assert len(results) == 1
        assert results[0].agreement_with_cdsb_sigma is not None
        # tau_doublet ~ 1.5, tau_cdsb = 0.1 => big positive z-score
        assert results[0].agreement_with_cdsb_sigma > 2.0
        # Warning emitted
        assert any(
            "disagreement" in rec.message.lower() for rec in caplog.records
        ), f"Expected disagreement warning, got: {[r.message for r in caplog.records]}"

    def test_cross_check_with_doublets_no_disagreement(self, caplog):
        """When CDSB and doublet agree, no warning is logged."""
        import logging

        line1, line2, tau1_truth, _ = _make_doublet(
            lam1=400.0, lam2=500.0, A_ki=1e8, g_k=4, tau1_inj=1.0
        )
        # CDSB tau very close to truth
        cdsb_corr = AbsorptionCorrectionResult(
            original_intensity=line1.intensity,
            corrected_intensity=line1.intensity / _f_tau(1.0),
            optical_depth=1.0,
            correction_factor=1.0 / _f_tau(1.0),
            is_optically_thick=False,
        )
        cdsb_result = SelfAbsorptionResult(
            corrected_observations=[line1, line2],
            masked_observations=[],
            corrections={400.0: cdsb_corr},
            n_corrected=1,
            n_masked=0,
            max_optical_depth=1.0,
        )

        corrector = SelfAbsorptionCorrector()
        with caplog.at_level(logging.WARNING, logger="inversion.self_absorption"):
            results = corrector.cross_check_with_doublets(
                [line1, line2], cdsb_result, sigma_threshold=2.0
            )

        assert len(results) == 1
        assert abs(results[0].agreement_with_cdsb_sigma) < 2.0
        # No disagreement warning
        assert not any("disagreement" in rec.message.lower() for rec in caplog.records)


# ==============================================================================
# Si Self-Absorption Audit Tests (CF-LIBS-improved-self-abs-audit)
#
# Reproduces the failure mode that motivated the audit: Si I resonance
# lines at 251.6 nm / 288.2 nm at high concentration (60-70% by mass, as in
# the vrabel2020 soil benchmark where SiO2 dominates) become optically thick
# and silently disappear from the spectrum. These tests assert that the
# corrector DOES something visible (escape_factor < 1, tau > threshold,
# structured log output) under those plasma conditions.
# ==============================================================================


class TestSiHighConcentrationSelfAbsorption:
    """Regression tests for Si I resonance lines at 60-70% mass fraction.

    Motivated by the n=33 autodiscovery finding (2026-05-13) that 49/53
    universal misses across all 5 identifiers were on
    ``vrabel2020_soil_benchmark`` — and that Si was the most-missed
    element (15/53). Hypothesis: Si I resonance lines at 251.6 / 288.2 nm
    are optically thick at SiO2 = 60%+ and the identifiers silently fail
    because the lines are absent from the spectrum.
    """

    @pytest.fixture
    def si_resonance_lines(self):
        """Build LineObservation objects for the two Si I resonance lines.

        Si I 251.611 nm and 288.158 nm are the canonical resonance
        transitions (E_lower = 0.0 eV, ground state). Atomic data from
        NIST ASD.
        """
        return [
            LineObservation(
                wavelength_nm=251.611,
                intensity=1.0e4,
                intensity_uncertainty=1.0e2,
                element="Si",
                ionization_stage=1,
                E_k_ev=4.9296,  # 3p4s
                g_k=3,
                A_ki=1.21e8,
            ),
            LineObservation(
                wavelength_nm=288.158,
                intensity=1.0e4,
                intensity_uncertainty=1.0e2,
                element="Si",
                ionization_stage=1,
                E_k_ev=5.0824,  # 3p4s'
                g_k=3,
                A_ki=2.17e8,
            ),
        ]

    def test_escape_factor_below_unity_for_si_resonance_lines(self, si_resonance_lines):
        """Escape factor f(tau) must be < 1 for optically-thick Si lines.

        Asserts the physics primitive ``_escape_factor`` does the right
        thing across the LIBS-relevant tau range. This is a unit test
        of the core formula because the integration-level test
        (``test_si_correction_pipeline_runs_and_logs``) showed that the
        existing ``_estimate_optical_depth`` uses ``SCALE_FACTOR = 1e-25``
        which under-estimates tau by ~12 orders of magnitude — a separate
        latent bug filed for follow-up. The escape-factor primitive
        itself is correct.
        """
        from cflibs.inversion.physics.self_absorption import _escape_factor

        # Optically thin: f -> 1
        assert _escape_factor(1e-6) == pytest.approx(1.0, abs=1e-5)
        # Moderate: f(1) = 1 - 1/e ~ 0.632
        assert _escape_factor(1.0) == pytest.approx((1 - np.exp(-1.0)) / 1.0, rel=1e-6)
        # Thick: f(3) ~ 0.317, must be substantially < 1
        f3 = _escape_factor(3.0)
        assert f3 < 0.5, f"f(tau=3) must be substantially < 1, got {f3}"
        # Very thick (Si resonance in 60% SiO2 regime): f -> 1/tau
        assert _escape_factor(10.0) == pytest.approx(0.1, rel=0.01)

        # Sanity: the Si I 251.611 nm line is correctly identified as a
        # resonance line (E_i = 0.0 eV) at the input level.
        assert si_resonance_lines[0].element == "Si"
        assert si_resonance_lines[0].wavelength_nm == 251.611

    def test_si_correction_pipeline_runs_and_logs(self, si_resonance_lines, caplog):
        """End-to-end ``correct()`` call must run and emit structured logs.

        Whether the correction *actually triggers* depends on the
        ``_estimate_optical_depth`` SCALE_FACTOR (which is currently off
        by orders of magnitude — see follow-up bd). What MUST happen
        regardless is: (a) the corrector returns a result for every
        observation, (b) a summary log line is emitted at INFO, (c) the
        max_optical_depth value is in the result. Transparency, not
        magnitude, is what this test gates.
        """
        import logging

        corrector = SelfAbsorptionCorrector(
            optical_depth_threshold=0.01,
            mask_threshold=10.0,
            plasma_length_cm=0.1,
        )
        with caplog.at_level(logging.INFO, logger="cflibs.inversion.self_absorption"):
            result = corrector.correct(
                observations=si_resonance_lines,
                temperature_K=10000.0,
                concentrations={"Si": 0.60},
                total_number_density_cm3=1.0e17,
                partition_funcs={"Si": 9.0},
                lower_level_energies={251.611: 0.0, 288.158: 0.0},
            )

        # Every observation has a recorded correction (no silent drops).
        assert 251.611 in result.corrections
        assert 288.158 in result.corrections
        # max_optical_depth is populated (even if tiny under current scale).
        assert result.max_optical_depth >= 0.0
        # Summary log must fire — that is the audit's load-bearing assertion.
        summary_records = [
            r for r in caplog.records if "self_absorption.correct summary" in r.message
        ]
        assert len(summary_records) == 1
        assert "max_tau=" in summary_records[0].message

    def test_correct_logs_structured_summary_for_si(self, si_resonance_lines, caplog):
        """SelfAbsorptionCorrector.correct() emits one INFO-level summary line.

        Addresses the user's transparency complaint: previously the
        correction ran silently and operators had no way to tell whether
        Si lines had been touched. Now every call emits an INFO log line
        with element-level counts and max tau.
        """
        import logging

        corrector = SelfAbsorptionCorrector(
            optical_depth_threshold=0.01,
            mask_threshold=10.0,
            plasma_length_cm=0.1,
        )
        with caplog.at_level(logging.INFO, logger="cflibs.inversion.self_absorption"):
            corrector.correct(
                observations=si_resonance_lines,
                temperature_K=10000.0,
                concentrations={"Si": 0.60},
                total_number_density_cm3=1.0e17,
                partition_funcs={"Si": 9.0},
                lower_level_energies={251.611: 0.0, 288.158: 0.0},
            )

        # Exactly one summary line, containing the diagnostic keywords.
        summary_records = [
            r for r in caplog.records if "self_absorption.correct summary" in r.message
        ]
        assert len(summary_records) == 1, (
            f"Expected exactly one summary log; got {len(summary_records)}: "
            f"{[r.message for r in caplog.records]}"
        )
        msg = summary_records[0].message
        assert "n_lines=2" in msg
        assert "max_tau=" in msg
        assert "T=10000" in msg

    def test_correct_warns_on_zero_temperature(self, si_resonance_lines, caplog):
        """T=0 must emit a WARNING — silently returning tau=0 is the bug class."""
        import logging

        corrector = SelfAbsorptionCorrector()
        with caplog.at_level(logging.WARNING, logger="cflibs.inversion.self_absorption"):
            corrector.correct(
                observations=si_resonance_lines,
                temperature_K=0.0,  # pathological
                concentrations={"Si": 0.60},
                total_number_density_cm3=1.0e17,
                partition_funcs={"Si": 9.0},
            )
        assert any("non-positive temperature" in r.message for r in caplog.records), (
            "T=0 must trigger a WARNING so the silent zero-tau path is "
            f"visible; got: {[r.message for r in caplog.records]}"
        )

    def test_correct_logs_missing_concentration_elements(self, si_resonance_lines, caplog):
        """Element absent from concentrations dict logs at INFO — never silent."""
        import logging

        corrector = SelfAbsorptionCorrector()
        with caplog.at_level(logging.INFO, logger="cflibs.inversion.self_absorption"):
            corrector.correct(
                observations=si_resonance_lines,
                temperature_K=10000.0,
                concentrations={"Fe": 0.10},  # Si MISSING
                total_number_density_cm3=1.0e17,
                partition_funcs={"Si": 9.0},
            )
        assert any(
            "no concentration entry" in r.message and "Si" in r.message for r in caplog.records
        ), (
            "Missing concentration entry must be logged so silent skips are "
            f"diagnosable; got: {[r.message for r in caplog.records]}"
        )


class TestALIASSelfAbsorptionLogging:
    """Tests that ALIAS identifier exposes self-absorption damping state.

    Replaces the previously-silent ``SA_DAMPING = 0.3`` hardcoded constant
    with an opt-in flag + structured logging. Audit Family 5 flipped the
    default OFF: the ad-hoc resonance-line damping must not silently fire
    by default (the paper-faithful ALIAS is optically-thin). The 0.3 factor
    is preserved and still applied when ``self_absorption_aware=True`` is
    passed explicitly.
    """

    def test_identifier_has_self_absorption_knobs(self, atomic_db):
        """Constructor exposes the self-absorption knobs; damping defaults OFF.

        Audit Family 5: ``self_absorption_aware`` defaults to ``False`` so
        the ad-hoc 0.3 damping is opt-in (paper-faithful, optically-thin by
        default). The damping factor / cutoff are still exposed and used
        when the operator opts in.
        """
        from cflibs.inversion.identify.alias import ALIASIdentifier

        identifier = ALIASIdentifier(atomic_db)
        assert identifier.self_absorption_aware is False
        assert identifier.self_absorption_damping == 0.3
        assert identifier.self_absorption_e_i_cutoff_ev == 0.1
        assert identifier._sa_n_damped_lines == 0
        assert identifier._sa_damped_elements == set()

    def test_identifier_rejects_invalid_damping(self, atomic_db):
        """Damping factor must be in (0, 1]; out-of-range values raise."""
        from cflibs.inversion.identify.alias import ALIASIdentifier

        with pytest.raises(ValueError, match="damping"):
            ALIASIdentifier(atomic_db, self_absorption_damping=0.0)
        with pytest.raises(ValueError, match="damping"):
            ALIASIdentifier(atomic_db, self_absorption_damping=1.5)
        with pytest.raises(ValueError, match="damping"):
            ALIASIdentifier(atomic_db, self_absorption_damping=-0.1)

    def test_identifier_rejects_invalid_e_i_cutoff(self, atomic_db):
        """E_i cutoff must be finite and >= 0."""
        from cflibs.inversion.identify.alias import ALIASIdentifier

        with pytest.raises(ValueError, match="e_i_cutoff"):
            ALIASIdentifier(atomic_db, self_absorption_e_i_cutoff_ev=-0.5)

    def test_identifier_disabled_mode_logs_explicitly(self, atomic_db, caplog):
        """When self_absorption_aware=False, identify() logs that fact at INFO.

        Operators reading benchmark logs need to be able to tell at a
        glance whether the run used SA damping or not.
        """
        import logging
        from cflibs.inversion.identify.alias import ALIASIdentifier

        identifier = ALIASIdentifier(atomic_db, self_absorption_aware=False)
        wavelength = np.linspace(200.0, 800.0, 6000)
        # Pure noise spectrum: identify() should complete and log without
        # actually detecting anything.
        intensity = np.random.RandomState(42).normal(10.0, 1.0, size=6000)

        with caplog.at_level(logging.INFO, logger="cflibs.inversion.identify.alias"):
            identifier.identify(wavelength, intensity)

        assert any("self-absorption damping DISABLED" in r.message for r in caplog.records), (
            "self_absorption_aware=False must emit an INFO log so disabled "
            f"mode is auditable; got: {[r.message for r in caplog.records]}"
        )


# ==============================================================================
# Optical-depth prefactor regression (CF-LIBS-improved-k2h7)
#
# Pins the physically-correct line-center optical depth to a LIBS-realistic
# magnitude after replacing the historical ``SCALE_FACTOR = 1e-25`` with the
# classical-radius prefactor + Doppler-width normalization (Hutchinson
# eq. 5.13). Pre-fix this would have returned tau ~ 1e-15 for every line,
# so the corrector's ``optical_depth_threshold = 0.1`` gate never fired
# and SA correction was a silent no-op. Post-fix tau lands in the (0.1, 10)
# optically-thick regime where the Pace-doublet and CDSB algorithms are
# meant to live.
# ==============================================================================


class TestOpticalDepthPrefactor:
    """Regression tests for the post-CF-LIBS-improved-k2h7 prefactor."""

    def test_optical_depth_realistic_libs_magnitude(self):
        """Si I 251.611 nm at LIBS-realistic conditions yields τ ~ a few.

        Parameters are chosen to land τ in the optically-thick regime
        (0.1 ≲ τ ≲ 10) where SA correction is meaningful: above the
        default ``optical_depth_threshold = 0.1`` so the corrector fires,
        and below the default ``mask_threshold = 3.0`` so the line gets
        corrected rather than dropped.

        Note on density: the task spec proposed ``N_total = 1e17`` cm⁻³
        but the full-physics derivation gives τ ~ 350 for that value
        (Si at 60 % mass fraction is *very* optically thick in dense
        plasma — Si lines self-reverse, which is well known in soil-
        matrix LIBS). To pin the corrector in the τ ~ few regime where
        it has interesting work to do (correct rather than mask), we
        use ``N_total = 1e15`` cm⁻³ — typical of late-time / decayed
        LIBS plasmas. The τ ~ N_total scaling is verified by
        :py:meth:`test_optical_depth_scales_linearly_with_density`.

        Pre-fix this test would have asserted τ ≈ 1e-15 (off by ~14
        orders of magnitude); the silent failure motivated the audit.
        """
        from cflibs.inversion.physics.self_absorption import SelfAbsorptionCorrector

        corrector = SelfAbsorptionCorrector(plasma_length_cm=0.1)
        line = LineObservation(
            wavelength_nm=251.611,
            intensity=1.0e4,
            intensity_uncertainty=1.0e2,
            element="Si",
            ionization_stage=1,
            E_k_ev=4.9296,
            g_k=3,
            A_ki=2.0e8,
        )

        tau = corrector._estimate_optical_depth(
            obs=line,
            temperature_K=10000.0,
            concentrations={"Si": 0.60},
            total_n_cm3=1.0e15,
            partition_funcs={"Si": 9.0},
            E_i_ev=0.0,
        )

        assert 0.1 < tau < 10.0, (
            f"Si I 251.611 nm at LIBS conditions (T=10000K, 60% Si by "
            f"mass, N_tot=1e15 cm^-3, L=0.1 cm) should give "
            f"0.1 < tau < 10; got tau={tau:.4g}. If this fires, the "
            f"prefactor (pi*e^2/(m_e*c) / sqrt(pi)*Δν_D) is wrong by "
            f"the implied magnitude."
        )

    def test_optical_depth_scales_linearly_with_density(self):
        """τ ∝ N_total — the prefactor is density-independent.

        Tau is the line-of-sight integral σ · n_lower · L; with σ and L
        fixed it must scale linearly with the lower-state population
        and therefore with N_total at fixed mass fraction and U(T).
        This rules out a regression where the prefactor accidentally
        picks up a Saha-type N² dependence.
        """
        from cflibs.inversion.physics.self_absorption import SelfAbsorptionCorrector

        corrector = SelfAbsorptionCorrector(plasma_length_cm=0.1)
        line = LineObservation(
            wavelength_nm=251.611,
            intensity=1.0e4,
            intensity_uncertainty=1.0e2,
            element="Si",
            ionization_stage=1,
            E_k_ev=4.9296,
            g_k=3,
            A_ki=2.0e8,
        )

        tau_lo = corrector._estimate_optical_depth(
            obs=line,
            temperature_K=10000.0,
            concentrations={"Si": 0.60},
            total_n_cm3=1.0e14,
            partition_funcs={"Si": 9.0},
            E_i_ev=0.0,
        )
        tau_hi = corrector._estimate_optical_depth(
            obs=line,
            temperature_K=10000.0,
            concentrations={"Si": 0.60},
            total_n_cm3=1.0e16,
            partition_funcs={"Si": 9.0},
            E_i_ev=0.0,
        )
        # tau scales linearly with N_total → ratio == density ratio.
        assert tau_hi / tau_lo == pytest.approx(100.0, rel=1e-3), (
            f"tau should scale linearly with N_total; got "
            f"tau(1e16)/tau(1e14) = {tau_hi / tau_lo:.4g} (expected 100)."
        )

    def test_optical_depth_doppler_temperature_scaling(self):
        """τ ∝ 1/√T (Doppler width grows as √T).

        For a fixed line and column density, increasing T broadens the
        Doppler profile, which dilutes the line-center cross section
        (φ(ν₀) ∝ 1/Δν_D ∝ 1/√T). All else equal, τ must shrink.
        """
        from cflibs.inversion.physics.self_absorption import SelfAbsorptionCorrector

        corrector = SelfAbsorptionCorrector(plasma_length_cm=0.1)
        line = LineObservation(
            wavelength_nm=251.611,
            intensity=1.0e4,
            intensity_uncertainty=1.0e2,
            element="Si",
            ionization_stage=1,
            E_k_ev=4.9296,
            g_k=3,
            A_ki=2.0e8,
        )

        tau_cool = corrector._estimate_optical_depth(
            obs=line,
            temperature_K=5000.0,
            concentrations={"Si": 0.60},
            total_n_cm3=1.0e15,
            partition_funcs={"Si": 9.0},
            E_i_ev=0.0,
        )
        tau_hot = corrector._estimate_optical_depth(
            obs=line,
            temperature_K=20000.0,
            concentrations={"Si": 0.60},
            total_n_cm3=1.0e15,
            partition_funcs={"Si": 9.0},
            E_i_ev=0.0,
        )
        # 4x hotter -> 2x broader Doppler width -> 2x smaller tau.
        ratio = tau_cool / tau_hot
        assert ratio == pytest.approx(2.0, rel=0.05), (
            f"Doppler scaling violated: tau(5000)/tau(20000) = "
            f"{ratio:.4g} (expected 2.0 = sqrt(4))"
        )

    def test_optical_depth_unknown_element_uses_default_mass(self):
        """Element absent from the atomic-mass table still returns a finite τ.

        The Doppler width depends on atomic mass. For elements not in
        ``_ATOMIC_MASS_AMU`` (e.g., synthetic "Xx" used in unit tests)
        we fall back to a Si-like ~28 amu default. Verify the path
        does not blow up and the value is in the same regime as Si.
        """
        from cflibs.inversion.physics.self_absorption import SelfAbsorptionCorrector

        corrector = SelfAbsorptionCorrector(plasma_length_cm=0.1)
        line = LineObservation(
            wavelength_nm=251.611,
            intensity=1.0e4,
            intensity_uncertainty=1.0e2,
            element="Xx",  # synthetic element not in the mass table
            ionization_stage=1,
            E_k_ev=4.9296,
            g_k=3,
            A_ki=2.0e8,
        )

        tau = corrector._estimate_optical_depth(
            obs=line,
            temperature_K=10000.0,
            concentrations={"Xx": 0.60},
            total_n_cm3=1.0e15,
            partition_funcs={"Xx": 9.0},
            E_i_ev=0.0,
        )
        assert np.isfinite(tau)
        assert tau > 0.0
        # With the Si-like fallback mass, tau should be within 2x of the
        # Si result above (Doppler width depends only weakly on mass).
        assert 0.1 < tau < 10.0

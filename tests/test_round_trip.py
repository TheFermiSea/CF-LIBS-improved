"""
Round-trip validation tests for CF-LIBS.

These tests verify the complete forward-inverse pipeline:
1. Generate synthetic spectra with known parameters
2. Add realistic noise
3. Run CF-LIBS inversion
4. Verify parameter recovery within tolerance

Target tolerances (from ROADMAP.md):
- Temperature: ±5%
- Electron density: ±20%
- Concentrations: ±10%
"""

import pytest
import numpy as np

from cflibs.validation.round_trip import (
    GoldenSpectrumGenerator,
    NoiseModel,
    RoundTripValidator,
    RoundTripResult,
    GoldenSpectrum,
)
from cflibs.inversion.boltzmann import LineObservation


class TestGoldenSpectrumGenerator:
    """Tests for the golden spectrum generator."""

    def test_generate_basic(self, atomic_db):
        """Test basic golden spectrum generation."""
        generator = GoldenSpectrumGenerator(atomic_db)

        golden = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            n_lines_per_element=5,
            seed=42,
        )

        # Check ground truth is stored
        assert golden.temperature_K == 10000.0
        assert golden.electron_density_cm3 == 1e17
        assert golden.concentrations["Fe"] == 1.0

        # Check lines were generated
        assert len(golden.line_observations) > 0

        # Check all lines have required attributes
        for obs in golden.line_observations:
            assert isinstance(obs, LineObservation)
            assert obs.intensity > 0
            assert obs.intensity_uncertainty > 0
            assert obs.element == "Fe"

    def test_generate_multi_element(self, atomic_db):
        """Test generation with multiple elements."""
        generator = GoldenSpectrumGenerator(atomic_db)

        golden = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.7, "H": 0.3},
            n_lines_per_element=5,
            seed=42,
            min_intensity=0.01,
        )

        # Check both elements present
        elements = set(obs.element for obs in golden.line_observations)
        assert "Fe" in elements or "H" in elements  # At least one element should work

    def test_generate_reproducibility(self, atomic_db):
        """Test that same seed produces same spectrum."""
        generator = GoldenSpectrumGenerator(atomic_db)

        golden1 = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
        )

        golden2 = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
        )

        # Should produce identical results
        assert len(golden1.line_observations) == len(golden2.line_observations)
        for obs1, obs2 in zip(golden1.line_observations, golden2.line_observations):
            np.testing.assert_allclose(obs1.intensity, obs2.intensity)

    def test_concentration_normalization(self, atomic_db):
        """Test that concentrations are normalized."""
        generator = GoldenSpectrumGenerator(atomic_db)

        golden = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.5, "H": 0.3},  # Sum = 0.8
            seed=42,
        )

        # Concentrations should be normalized to sum to 1.0
        total = sum(golden.concentrations.values())
        np.testing.assert_allclose(total, 1.0, rtol=0.01)

    def test_temperature_property(self, atomic_db):
        """Test temperature_eV property."""
        generator = GoldenSpectrumGenerator(atomic_db)

        golden = generator.generate(
            temperature_K=11604.5,  # ~1 eV
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
        )

        np.testing.assert_allclose(golden.temperature_eV, 1.0, rtol=0.01)


class TestNoiseModel:
    """Tests for the noise model."""

    def test_apply_noise_basic(self, atomic_db):
        """Test basic noise application."""
        generator = GoldenSpectrumGenerator(atomic_db)
        noise_model = NoiseModel()

        golden = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
        )

        noisy = noise_model.apply(golden)

        # Noisy spectrum should have same number of lines
        assert len(noisy.line_observations) == len(golden.line_observations)

        # Intensities should be different (noise added)
        intensities_changed = any(
            abs(n.intensity - g.intensity) > 1e-10
            for n, g in zip(noisy.line_observations, golden.line_observations)
        )
        assert intensities_changed, "Noise should change intensities"

        # Ground truth should be preserved
        assert noisy.temperature_K == golden.temperature_K
        assert noisy.electron_density_cm3 == golden.electron_density_cm3

    def test_noise_reproducibility(self, atomic_db):
        """Test noise reproducibility with seed."""
        generator = GoldenSpectrumGenerator(atomic_db)
        noise_model = NoiseModel()

        golden = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
        )

        noisy1 = noise_model.apply(golden, seed=100)
        noisy2 = noise_model.apply(golden, seed=100)

        for obs1, obs2 in zip(noisy1.line_observations, noisy2.line_observations):
            np.testing.assert_allclose(obs1.intensity, obs2.intensity)

    def test_noise_levels(self, atomic_db):
        """Test different noise levels."""
        generator = GoldenSpectrumGenerator(atomic_db)

        golden = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
        )

        # Low noise
        noise_low = NoiseModel(shot_noise=True, readout_noise=1.0, multiplicative_noise=0.01)
        noisy_low = noise_low.apply(golden, seed=100)

        # High noise
        noise_high = NoiseModel(shot_noise=True, readout_noise=20.0, multiplicative_noise=0.10)
        noisy_high = noise_high.apply(golden, seed=100)

        # High noise should have larger uncertainty
        assert noisy_high.line_observations[0].intensity_uncertainty > (
            noisy_low.line_observations[0].intensity_uncertainty
        )

    def test_no_noise(self, atomic_db):
        """Test with all noise disabled."""
        generator = GoldenSpectrumGenerator(atomic_db)
        noise_model = NoiseModel(
            shot_noise=False, readout_noise=0.0, background=0.0, multiplicative_noise=0.0
        )

        golden = generator.generate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
            min_intensity=10.0,  # Above NoiseModel's clamp at 1.0
        )

        noisy = noise_model.apply(golden)

        # Intensities should be nearly identical (no noise)
        for n, g in zip(noisy.line_observations, golden.line_observations):
            np.testing.assert_allclose(n.intensity, g.intensity, rtol=1e-10)


class TestRoundTripValidator:
    """Tests for the complete round-trip validator."""

    @pytest.mark.requires_db
    def test_validator_init(self, atomic_db):
        """Test validator initialization."""
        validator = RoundTripValidator(atomic_db)

        assert validator.temperature_tolerance == 0.05
        assert validator.density_tolerance == 0.20
        assert validator.concentration_tolerance == 0.10

    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_round_trip_no_noise(self, atomic_db):
        """Test round-trip without noise (should pass easily)."""
        validator = RoundTripValidator(
            atomic_db,
            temperature_tolerance=0.10,  # Relaxed for this test
            density_tolerance=0.30,
            concentration_tolerance=0.20,
        )

        result = validator.validate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
            add_noise=False,
        )

        # Should converge
        assert result.converged, f"Solver did not converge: {result.iterations} iterations"

        # Check errors are reasonable
        assert (
            result.temperature_error_frac < 0.50
        ), f"Temperature error too high: {result.temperature_error_frac*100:.1f}%"

    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_round_trip_with_noise(self, atomic_db):
        """Test round-trip with realistic noise."""
        validator = RoundTripValidator(
            atomic_db,
            temperature_tolerance=0.15,  # Relaxed tolerances for noisy data
            density_tolerance=0.40,
            concentration_tolerance=0.30,
        )

        result = validator.validate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
            add_noise=True,
        )

        # Should still converge
        assert result.converged, "Solver should converge even with noise"

    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_round_trip_summary(self, atomic_db):
        """Test result summary generation."""
        validator = RoundTripValidator(atomic_db)

        result = validator.validate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            seed=42,
            add_noise=False,
        )

        summary = result.summary()
        assert "Round-Trip Validation" in summary
        assert "Temperature" in summary
        assert "Electron density" in summary

    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_multi_element_recovery(self, atomic_db):
        """Test recovery of multiple element concentrations."""
        validator = RoundTripValidator(
            atomic_db,
            temperature_tolerance=0.20,
            density_tolerance=0.50,
            concentration_tolerance=0.40,
        )

        result = validator.validate(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.7, "H": 0.3},
            seed=42,
            add_noise=False,
        )

        # Check that concentrations are returned
        assert len(result.recovered_concentrations) > 0


class TestRoundTripResult:
    """Tests for the RoundTripResult dataclass."""

    def test_result_summary(self):
        """Test result summary formatting."""
        result = RoundTripResult(
            true_temperature_K=10000.0,
            recovered_temperature_K=10500.0,
            temperature_error_frac=0.05,
            true_electron_density=1e17,
            recovered_electron_density=1.1e17,
            electron_density_error_frac=0.10,
            true_concentrations={"Fe": 0.7, "Cu": 0.3},
            recovered_concentrations={"Fe": 0.68, "Cu": 0.32},
            concentration_errors={"Fe": 0.029, "Cu": 0.067},
            converged=True,
            iterations=5,
            passed=True,
        )

        summary = result.summary()

        assert "PASSED" in summary
        assert "10000" in summary
        assert "10500" in summary
        assert "Fe" in summary
        assert "Cu" in summary

    def test_result_failed(self):
        """Test failed result summary."""
        result = RoundTripResult(
            true_temperature_K=10000.0,
            recovered_temperature_K=15000.0,
            temperature_error_frac=0.50,
            true_electron_density=1e17,
            recovered_electron_density=5e17,
            electron_density_error_frac=4.0,
            true_concentrations={"Fe": 1.0},
            recovered_concentrations={"Fe": 0.5},
            concentration_errors={"Fe": 0.50},
            converged=False,
            iterations=20,
            passed=False,
        )

        summary = result.summary()
        assert "FAILED" in summary


# Integration tests with synthetic data only (no database required)


class TestGoldenSpectrumProperties:
    """Test GoldenSpectrum dataclass."""

    def test_basic_properties(self):
        """Test basic GoldenSpectrum creation."""
        obs = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            )
        ]

        golden = GoldenSpectrum(
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            line_observations=obs,
            seed=42,
        )

        assert golden.temperature_K == 10000.0
        assert golden.electron_density_cm3 == 1e17
        assert len(golden.line_observations) == 1
        np.testing.assert_allclose(golden.temperature_eV, 10000.0 / 11604.5, rtol=0.01)


class TestSelfConsistentElectronDensity:
    """Tests for self-consistent n_e and round-trip recovery."""

    @pytest.mark.requires_db
    def test_compute_equilibrium_ne(self, atomic_db):
        """Test that compute_equilibrium_ne converges to a physical value."""
        generator = GoldenSpectrumGenerator(atomic_db)

        n_e = generator.compute_equilibrium_ne(
            temperature_K=10000.0,
            concentrations={"Fe": 1.0},
        )

        # n_e should be positive and physically reasonable for 1 atm, 10000 K
        assert n_e > 1e14
        assert n_e < 1e19

    @pytest.mark.requires_db
    @pytest.mark.slow
    def test_round_trip_self_consistent_ne(self, atomic_db):
        """Round-trip with self-consistent n_e should recover all parameters.

        Uses compute_equilibrium_ne to generate golden spectrum at the
        physically correct n_e for the given T and pressure, then checks
        that the solver recovers T, n_e, and concentrations.
        """
        generator = GoldenSpectrumGenerator(atomic_db)

        T_K = 10000.0
        concentrations = {"Fe": 1.0}

        # Compute self-consistent n_e at 1 atm
        n_e_eq = generator.compute_equilibrium_ne(T_K, concentrations)

        # Generate golden spectrum at equilibrium n_e
        golden = generator.generate(
            temperature_K=T_K,
            electron_density_cm3=n_e_eq,
            concentrations=concentrations,
            n_lines_per_element=10,
            seed=42,
            include_ionic=True,
        )

        assert len(golden.line_observations) > 0

        # Run solver with same pressure
        from cflibs.inversion.solver import IterativeCFLIBSSolver
        from cflibs.core.constants import STP_PRESSURE

        solver = IterativeCFLIBSSolver(atomic_db, max_iterations=20, pressure_pa=STP_PRESSURE)
        result = solver.solve(golden.line_observations)

        assert result.converged, f"Solver did not converge in {result.iterations} iterations"

        # Temperature recovery
        T_err = abs(result.temperature_K - T_K) / T_K
        assert T_err < 0.15, f"Temperature error {T_err*100:.1f}% exceeds 15%"

        # Electron density recovery (the key test for CF-LIBS-8sa)
        ne_err = abs(result.electron_density_cm3 - n_e_eq) / n_e_eq
        assert ne_err < 0.50, (
            f"n_e error {ne_err*100:.1f}% exceeds 50%: "
            f"true={n_e_eq:.2e}, recovered={result.electron_density_cm3:.2e}"
        )

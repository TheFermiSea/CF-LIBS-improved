"""
Unit and integration tests for cflibs.inversion.quality module.

Tests cover:
- QualityMetrics dataclass
- QualityAssessor threshold classifications
- compute_reconstruction_chi_squared function
"""

import pytest
import numpy as np

from cflibs.inversion.quality import (
    QualityMetrics,
    QualityAssessor,
    compute_reconstruction_chi_squared,
)
from cflibs.inversion.boltzmann import LineObservation

# ==============================================================================
# QualityMetrics Dataclass Tests
# ==============================================================================


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_dataclass_creation(self):
        """Verify QualityMetrics can be instantiated with all fields."""
        metrics = QualityMetrics(
            r_squared_boltzmann=0.95,
            r_squared_by_element={"Fe": 0.96, "Cu": 0.94},
            temperature_by_element={"Fe": 10000.0, "Cu": 10200.0},
            inter_element_t_std_K=100.0,
            inter_element_t_std_frac=0.01,
            saha_boltzmann_consistency=0.05,
            t_boltzmann_K=10000.0,
            t_saha_K=10100.0,
            closure_residual=0.02,
            chi_squared=15.0,
            reduced_chi_squared=1.5,
            n_degrees_freedom=10,
            quality_flag="good",
            warnings=["test warning"],
        )

        assert metrics.r_squared_boltzmann == 0.95
        assert metrics.r_squared_by_element["Fe"] == 0.96
        assert metrics.temperature_by_element["Cu"] == 10200.0
        assert metrics.inter_element_t_std_K == 100.0
        assert metrics.quality_flag == "good"
        assert len(metrics.warnings) == 1

    def test_dataclass_defaults(self):
        """Verify default values are set correctly."""
        metrics = QualityMetrics(r_squared_boltzmann=0.90)

        assert metrics.r_squared_by_element == {}
        assert metrics.temperature_by_element == {}
        assert metrics.inter_element_t_std_K == 0.0
        assert metrics.closure_residual == 0.0
        assert metrics.quality_flag == "unknown"
        assert metrics.warnings == []

    def test_to_dict(self):
        """Verify serialization to dictionary."""
        metrics = QualityMetrics(
            r_squared_boltzmann=0.95,
            inter_element_t_std_frac=0.02,
            saha_boltzmann_consistency=0.05,
            closure_residual=0.01,
            reduced_chi_squared=1.2,
            quality_flag="excellent",
        )

        d = metrics.to_dict()

        assert d["r_squared_boltzmann"] == 0.95
        assert d["inter_element_t_std_frac"] == 0.02
        assert d["saha_boltzmann_consistency"] == 0.05
        assert d["closure_residual"] == 0.01
        assert d["reduced_chi_squared"] == 1.2
        assert d["quality_flag"] == "excellent"


# ==============================================================================
# QualityAssessor Tests
# ==============================================================================


class TestQualityAssessorInit:
    """Tests for QualityAssessor initialization."""

    def test_default_thresholds(self):
        """Verify default thresholds are set correctly."""
        assessor = QualityAssessor()

        assert assessor.THRESHOLDS["r_squared"]["excellent"] == 0.95
        assert assessor.THRESHOLDS["r_squared"]["good"] == 0.90
        assert assessor.THRESHOLDS["r_squared"]["acceptable"] == 0.80
        assert assessor.THRESHOLDS["closure"]["excellent"] == 0.01
        assert assessor.THRESHOLDS["closure"]["good"] == 0.05
        assert assessor.THRESHOLDS["closure"]["acceptable"] == 0.10

    def test_custom_weights(self):
        """Verify custom weights are stored."""
        assessor = QualityAssessor(
            r_squared_weight=2.0,
            consistency_weight=0.5,
            closure_weight=1.5,
        )

        assert assessor.r_squared_weight == 2.0
        assert assessor.consistency_weight == 0.5
        assert assessor.closure_weight == 1.5


class TestQualityAssessorLevels:
    """Tests for QualityAssessor._get_level method."""

    @pytest.fixture
    def assessor(self):
        return QualityAssessor()

    def test_r_squared_excellent(self, assessor):
        """Verify R² > 0.95 classified as excellent."""
        level = assessor._get_level("r_squared", 0.98, higher_is_better=True)
        assert level == "excellent"

    def test_r_squared_good(self, assessor):
        """Verify 0.90 < R² < 0.95 classified as good."""
        level = assessor._get_level("r_squared", 0.92, higher_is_better=True)
        assert level == "good"

    def test_r_squared_acceptable(self, assessor):
        """Verify 0.80 < R² < 0.90 classified as acceptable."""
        level = assessor._get_level("r_squared", 0.85, higher_is_better=True)
        assert level == "acceptable"

    def test_r_squared_poor(self, assessor):
        """Verify R² < 0.80 classified as poor."""
        level = assessor._get_level("r_squared", 0.65, higher_is_better=True)
        assert level == "poor"

    def test_closure_excellent(self, assessor):
        """Verify closure < 0.01 classified as excellent."""
        level = assessor._get_level("closure", 0.005, higher_is_better=False)
        assert level == "excellent"

    def test_closure_good(self, assessor):
        """Verify 0.01 < closure < 0.05 classified as good."""
        level = assessor._get_level("closure", 0.03, higher_is_better=False)
        assert level == "good"

    def test_closure_acceptable(self, assessor):
        """Verify 0.05 < closure < 0.10 classified as acceptable."""
        level = assessor._get_level("closure", 0.07, higher_is_better=False)
        assert level == "acceptable"

    def test_closure_poor(self, assessor):
        """Verify closure > 0.10 classified as poor."""
        level = assessor._get_level("closure", 0.15, higher_is_better=False)
        assert level == "poor"

    def test_boundary_r_squared_excellent(self, assessor):
        """Test boundary at exactly 0.95 (should be excellent)."""
        level = assessor._get_level("r_squared", 0.95, higher_is_better=True)
        assert level == "excellent"

    def test_boundary_closure_excellent(self, assessor):
        """Test boundary at exactly 0.01 (should be excellent)."""
        level = assessor._get_level("closure", 0.01, higher_is_better=False)
        assert level == "excellent"


class TestQualityAssessorDetermineFlag:
    """Tests for QualityAssessor._determine_quality_flag method."""

    @pytest.fixture
    def assessor(self):
        return QualityAssessor()

    def test_all_excellent(self, assessor):
        """All metrics excellent should give excellent overall."""
        flag = assessor._determine_quality_flag(
            r_squared=0.98,
            saha_consistency=0.05,
            t_std_frac=0.02,
            closure_residual=0.005,
        )
        assert flag == "excellent"

    def test_worst_metric_wins(self, assessor):
        """Overall flag should match worst individual metric."""
        # One poor metric among otherwise excellent
        flag = assessor._determine_quality_flag(
            r_squared=0.98,
            saha_consistency=0.05,
            t_std_frac=0.02,
            closure_residual=0.15,  # Poor
        )
        assert flag == "poor"

    def test_mixed_good_acceptable(self, assessor):
        """Mix of good and acceptable should give acceptable."""
        flag = assessor._determine_quality_flag(
            r_squared=0.92,  # Good
            saha_consistency=0.15,  # Good
            t_std_frac=0.12,  # Acceptable
            closure_residual=0.03,  # Good
        )
        assert flag == "acceptable"


class TestQualityAssessorAssess:
    """Integration tests for QualityAssessor.assess method."""

    @pytest.fixture
    def assessor(self):
        return QualityAssessor()

    def test_assess_returns_metrics(self, assessor, synthetic_line_observations):
        """Verify assess returns a QualityMetrics object."""
        observations = synthetic_line_observations(n_lines=10, element="Fe")

        metrics = assessor.assess(
            observations=observations,
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            ionization_potentials={"Fe": 7.87},
            partition_funcs_I={"Fe": 25.0},
            partition_funcs_II={"Fe": 15.0},
        )

        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.r_squared_boltzmann <= 1.0
        assert metrics.closure_residual >= 0.0

    def test_assess_closure_warning(self, assessor, synthetic_line_observations):
        """Verify warning is added when closure residual > 0.05."""
        observations = synthetic_line_observations(n_lines=10, element="Fe")

        metrics = assessor.assess(
            observations=observations,
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.5},  # Sum < 1.0
            ionization_potentials={"Fe": 7.87},
            partition_funcs_I={"Fe": 25.0},
            partition_funcs_II={"Fe": 15.0},
        )

        # Closure residual should be 0.5
        assert metrics.closure_residual == pytest.approx(0.5, abs=0.01)
        assert any("Closure residual" in w for w in metrics.warnings)

    def test_assess_few_observations(self, assessor):
        """Verify graceful handling with < 3 observations."""
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

        metrics = assessor.assess(
            observations=observations,
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            ionization_potentials={"Fe": 7.87},
            partition_funcs_I={"Fe": 25.0},
            partition_funcs_II={"Fe": 15.0},
        )

        # R² should be 0.0 for fewer than 3 points
        assert metrics.r_squared_boltzmann == 0.0


# ==============================================================================
# compute_reconstruction_chi_squared Tests
# ==============================================================================


class TestComputeChiSquared:
    """Tests for compute_reconstruction_chi_squared function."""

    def test_perfect_match(self):
        """Verify chi² = 0 for identical spectra."""
        measured = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        modeled = measured.copy()

        chi2, reduced_chi2, n_dof = compute_reconstruction_chi_squared(measured, modeled)

        assert chi2 == pytest.approx(0.0, abs=1e-10)

    def test_with_known_residuals(self):
        """Verify chi² scales correctly with residual magnitude."""
        measured = np.array([100.0, 100.0, 100.0, 100.0])
        modeled = np.array([110.0, 110.0, 110.0, 110.0])  # 10 off
        uncertainties = np.array([10.0, 10.0, 10.0, 10.0])

        chi2, reduced_chi2, n_dof = compute_reconstruction_chi_squared(
            measured, modeled, uncertainties
        )

        # Each residual contributes (10/10)^2 = 1
        # chi2 = 4 * 1 = 4
        assert chi2 == pytest.approx(4.0, abs=0.01)

    def test_poisson_uncertainties(self):
        """Verify Poisson uncertainties are used when none provided."""
        measured = np.array([100.0, 400.0, 900.0])
        modeled = measured + np.array([10.0, 20.0, 30.0])

        chi2, reduced_chi2, n_dof = compute_reconstruction_chi_squared(measured, modeled)

        # Uncertainties should be sqrt(measured)
        # Residuals / uncertainties = [10/10, 20/20, 30/30] = [1, 1, 1]
        assert chi2 == pytest.approx(3.0, abs=0.01)

    def test_length_mismatch_raises(self):
        """Verify ValueError when spectrum lengths don't match."""
        measured = np.array([100.0, 200.0, 300.0])
        modeled = np.array([100.0, 200.0])

        with pytest.raises(ValueError, match="lengths must match"):
            compute_reconstruction_chi_squared(measured, modeled)

    def test_zero_uncertainty_skipped(self):
        """Verify points with zero uncertainty are skipped."""
        measured = np.array([100.0, 200.0, 300.0])
        modeled = np.array([110.0, 200.0, 300.0])
        uncertainties = np.array([10.0, 0.0, 10.0])  # Middle point has zero

        chi2, reduced_chi2, n_dof = compute_reconstruction_chi_squared(
            measured, modeled, uncertainties
        )

        # Only first and last points contribute
        # First: (10/10)^2 = 1, Last: (0/10)^2 = 0
        assert chi2 == pytest.approx(1.0, abs=0.01)

    def test_degrees_of_freedom(self):
        """Verify degrees of freedom calculation (n_valid - 3)."""
        measured = np.ones(100)
        modeled = np.ones(100)
        uncertainties = np.ones(100)

        chi2, reduced_chi2, n_dof = compute_reconstruction_chi_squared(
            measured, modeled, uncertainties
        )

        # n_dof = 100 - 3 = 97
        assert n_dof == 97

    def test_reduced_chi_squared(self):
        """Verify reduced chi² = chi² / n_dof."""
        measured = np.ones(13)
        modeled = np.ones(13) + 0.5
        uncertainties = np.ones(13) * 0.5

        chi2, reduced_chi2, n_dof = compute_reconstruction_chi_squared(
            measured, modeled, uncertainties
        )

        # Each point contributes (0.5/0.5)^2 = 1
        # chi2 = 13, n_dof = 13 - 3 = 10
        # reduced_chi2 = 13 / 10 = 1.3
        assert chi2 == pytest.approx(13.0, abs=0.01)
        assert n_dof == 10
        assert reduced_chi2 == pytest.approx(1.3, abs=0.01)


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


class TestQualityEdgeCases:
    """Edge case tests for quality module."""

    def test_empty_observations(self):
        """Verify handling of empty observation list."""
        assessor = QualityAssessor()

        metrics = assessor.assess(
            observations=[],
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            ionization_potentials={"Fe": 7.87},
            partition_funcs_I={"Fe": 25.0},
            partition_funcs_II={"Fe": 15.0},
        )

        assert metrics.r_squared_boltzmann == 0.0
        assert metrics.r_squared_by_element == {}

    def test_single_element_no_t_std(self, synthetic_line_observations):
        """Verify inter-element T std is 0 with single element."""
        assessor = QualityAssessor()
        observations = synthetic_line_observations(n_lines=10, element="Fe")

        metrics = assessor.assess(
            observations=observations,
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            ionization_potentials={"Fe": 7.87},
            partition_funcs_I={"Fe": 25.0},
            partition_funcs_II={"Fe": 15.0},
        )

        assert metrics.inter_element_t_std_frac == 0.0

    def test_negative_concentration_handled(self, synthetic_line_observations):
        """Verify negative concentrations don't cause crash."""
        assessor = QualityAssessor()
        observations = synthetic_line_observations(n_lines=10, element="Fe")

        # This shouldn't happen in practice, but should be handled gracefully
        metrics = assessor.assess(
            observations=observations,
            temperature_K=10000.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": -0.5},  # Negative
            ionization_potentials={"Fe": 7.87},
            partition_funcs_I={"Fe": 25.0},
            partition_funcs_II={"Fe": 15.0},
        )

        # Closure residual should be |(-0.5) - 1.0| = 1.5
        assert metrics.closure_residual == pytest.approx(1.5, abs=0.01)
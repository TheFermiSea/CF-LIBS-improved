"""
Tests for uncertainty propagation utilities.

Tests the `cflibs.inversion.physics.uncertainty` module which provides automatic
correlation-aware uncertainty propagation using the `uncertainties` package.

Requirements: uncertainties>=3.2.0
"""

import pytest
import numpy as np

# Mark entire module as requiring uncertainties package
pytestmark = pytest.mark.requires_uncertainty

# Skip all tests if uncertainties not available
pytest.importorskip("uncertainties")


class TestCreateBoltzmannUncertainties:
    """Tests for create_boltzmann_uncertainties function."""

    def test_with_covariance_matrix(self):
        """Test creating correlated uncertainties from covariance matrix."""
        from cflibs.inversion.physics.uncertainty import create_boltzmann_uncertainties

        slope = -1.0
        intercept = 5.0
        # Covariance matrix with correlation
        cov = np.array([[0.01, 0.005], [0.005, 0.04]])

        slope_u, intercept_u = create_boltzmann_uncertainties(slope, intercept, cov)

        assert slope_u.nominal_value == pytest.approx(slope)
        assert intercept_u.nominal_value == pytest.approx(intercept)
        assert slope_u.std_dev == pytest.approx(np.sqrt(0.01))
        assert intercept_u.std_dev == pytest.approx(np.sqrt(0.04))

    def test_without_covariance_matrix(self):
        """Test fallback to independent uncertainties."""
        from cflibs.inversion.physics.uncertainty import create_boltzmann_uncertainties

        slope = -1.0
        intercept = 5.0
        slope_err = 0.1
        intercept_err = 0.2

        slope_u, intercept_u = create_boltzmann_uncertainties(
            slope, intercept, None, slope_err, intercept_err
        )

        assert slope_u.nominal_value == pytest.approx(slope)
        assert intercept_u.nominal_value == pytest.approx(intercept)
        assert slope_u.std_dev == pytest.approx(slope_err)
        assert intercept_u.std_dev == pytest.approx(intercept_err)

    def test_correlation_preserved(self):
        """Test that correlations are preserved for correlated operations."""
        from cflibs.inversion.physics.uncertainty import create_boltzmann_uncertainties

        slope = -1.0
        intercept = 5.0
        # Perfectly correlated
        cov = np.array([[0.01, 0.01], [0.01, 0.01]])

        slope_u, intercept_u = create_boltzmann_uncertainties(slope, intercept, cov)

        # For perfectly correlated variables, slope - slope = 0 exactly
        diff = slope_u - slope_u
        assert diff.nominal_value == pytest.approx(0.0)
        assert diff.std_dev == pytest.approx(0.0)


class TestTemperatureFromSlope:
    """Tests for temperature_from_slope function."""

    def test_basic_conversion(self):
        """Test temperature calculation from slope."""
        from cflibs.inversion.physics.uncertainty import temperature_from_slope
        from uncertainties import ufloat
        from cflibs.core.constants import KB_EV

        # slope = -1 / (k_B[eV/K] * T[K])
        # For T = 10000 K, slope ≈ -1.16 eV^-1
        T_expected_K = 10000.0
        slope = -1.0 / (KB_EV * T_expected_K)

        slope_u = ufloat(slope, 0.01)
        T_K_u = temperature_from_slope(slope_u)

        assert T_K_u.nominal_value == pytest.approx(T_expected_K, rel=0.01)
        # Uncertainty should be non-zero
        assert T_K_u.std_dev > 0


class TestClosurePropagation:
    """Tests for closure equation uncertainty propagation."""

    def test_standard_closure_sum_to_one(self):
        """Test that concentrations sum to 1 regardless of uncertainties."""
        from cflibs.inversion.physics.uncertainty import propagate_through_closure_standard
        from uncertainties import ufloat

        intercepts_u = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.3),
            "Al": ufloat(7.0, 0.4),
        }
        partition_funcs = {"Fe": 25.0, "Si": 15.0, "Al": 10.0}

        concentrations_u = propagate_through_closure_standard(intercepts_u, partition_funcs)

        # Sum of concentrations should be exactly 1
        total = sum(c.nominal_value for c in concentrations_u.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_standard_closure_propagates_uncertainty(self):
        """Test that uncertainties are propagated through closure."""
        from cflibs.inversion.physics.uncertainty import propagate_through_closure_standard
        from uncertainties import ufloat

        intercepts_u = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.3),
        }
        partition_funcs = {"Fe": 25.0, "Si": 15.0}

        concentrations_u = propagate_through_closure_standard(intercepts_u, partition_funcs)

        # All uncertainties should be positive (propagated)
        for conc_u in concentrations_u.values():
            assert conc_u.std_dev > 0

    def test_matrix_closure_fixed_element(self):
        """Test matrix mode closure with fixed element concentration."""
        from cflibs.inversion.physics.uncertainty import propagate_through_closure_matrix
        from uncertainties import ufloat

        intercepts_u = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.3),
        }
        partition_funcs = {"Fe": 25.0, "Si": 15.0}

        concentrations_u = propagate_through_closure_matrix(
            intercepts_u, partition_funcs, matrix_element="Fe", matrix_fraction=0.9
        )

        # Matrix element should have concentration = matrix_fraction
        assert concentrations_u["Fe"].nominal_value == pytest.approx(0.9)

    def test_oxide_closure_normalizes_oxide_weighted_total(self):
        """Test oxide mode normalizes using the oxide-weighted closure sum."""
        from cflibs.inversion.physics.uncertainty import propagate_through_closure_oxide
        from uncertainties import ufloat

        intercepts_u = {
            "Fe": ufloat(np.log(2.0), 0.1),
            "Si": ufloat(0.0, 0.1),
        }
        partition_funcs = {"Fe": 1.0, "Si": 1.0}
        oxide_stoichiometry = {"Fe": 2.0, "Si": 1.0}

        concentrations_u = propagate_through_closure_oxide(
            intercepts_u,
            partition_funcs,
            oxide_stoichiometry,
        )

        oxide_total = sum(
            concentrations_u[element].nominal_value * oxide_stoichiometry.get(element, 1.0)
            for element in concentrations_u
        )
        assert oxide_total == pytest.approx(1.0, abs=1e-10)

    def test_standard_closure_rejects_invalid_abundance_multiplier(self):
        """Test invalid abundance multipliers fail fast."""
        from cflibs.inversion.physics.uncertainty import propagate_through_closure_standard
        from uncertainties import ufloat

        intercepts_u = {"Fe": ufloat(10.0, 0.5)}
        partition_funcs = {"Fe": 25.0}

        with pytest.raises(ValueError, match="abundance_multipliers\\['Fe'\\]"):
            propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers={"Fe": float("nan")},
            )

    def test_correlation_in_closure(self):
        """Test that correlations affect closure uncertainty properly.

        When intercepts are correlated (from same Boltzmann fit), the
        relative uncertainties in concentrations should be smaller than
        with independent errors.
        """
        from cflibs.inversion.physics.uncertainty import propagate_through_closure_standard
        from uncertainties import ufloat, correlated_values

        # Case 1: Independent uncertainties
        intercepts_indep = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.5),
        }
        partition_funcs = {"Fe": 25.0, "Si": 25.0}

        conc_indep = propagate_through_closure_standard(intercepts_indep, partition_funcs)
        fe_uncert_indep = conc_indep["Fe"].std_dev

        # Case 2: Perfectly correlated (both from same measurement)
        # When q_Fe and q_Si move together, ratio stays more constant
        cov = np.array([[0.25, 0.25], [0.25, 0.25]])  # Perfect correlation
        q_fe, q_si = correlated_values([10.0, 8.0], cov)
        intercepts_corr = {"Fe": q_fe, "Si": q_si}

        conc_corr = propagate_through_closure_standard(intercepts_corr, partition_funcs)
        fe_uncert_corr = conc_corr["Fe"].std_dev

        # Correlated case should have smaller relative uncertainty
        # because common-mode errors cancel in the ratio
        assert fe_uncert_corr < fe_uncert_indep


class TestExtractValuesAndUncertainties:
    """Tests for extract_values_and_uncertainties function."""

    def test_extraction(self):
        """Test extracting nominal values and uncertainties from dict."""
        from cflibs.inversion.physics.uncertainty import extract_values_and_uncertainties
        from uncertainties import ufloat

        data = {
            "Fe": ufloat(0.6, 0.05),
            "Si": ufloat(0.3, 0.03),
            "Al": ufloat(0.1, 0.02),
        }

        nominal, uncert = extract_values_and_uncertainties(data)

        assert nominal["Fe"] == pytest.approx(0.6)
        assert nominal["Si"] == pytest.approx(0.3)
        assert nominal["Al"] == pytest.approx(0.1)
        assert uncert["Fe"] == pytest.approx(0.05)
        assert uncert["Si"] == pytest.approx(0.03)
        assert uncert["Al"] == pytest.approx(0.02)


class TestBoltzmannFitResultCovariance:
    """Tests for covariance_matrix attribute in BoltzmannFitResult."""

    def test_sigma_clip_produces_covariance(self):
        """Test that sigma_clip fitting produces covariance matrix."""
        from cflibs.inversion.physics.boltzmann import (
            BoltzmannPlotFitter,
            LineObservation,
            FitMethod,
        )

        # Create synthetic data for Fe I with known slope
        observations = [
            LineObservation(
                wavelength_nm=300.0 + i * 10,
                intensity=1000.0 * np.exp(-E / 0.8),  # T ~ 9300 K
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=E,
                g_k=5,
                A_ki=1e8,
            )
            for i, E in enumerate([2.0, 3.0, 4.0, 5.0, 6.0])  # Need >2 points
        ]

        fitter = BoltzmannPlotFitter(method=FitMethod.SIGMA_CLIP)
        result = fitter.fit(observations)

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (2, 2)
        # Diagonal elements should be positive
        assert result.covariance_matrix[0, 0] > 0
        assert result.covariance_matrix[1, 1] > 0

    def test_ransac_produces_covariance(self):
        """Test that RANSAC fitting produces covariance matrix."""
        from cflibs.inversion.physics.boltzmann import (
            BoltzmannPlotFitter,
            LineObservation,
            FitMethod,
        )

        observations = [
            LineObservation(
                wavelength_nm=300.0 + i * 10,
                intensity=1000.0 * np.exp(-E / 0.8),
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=E,
                g_k=5,
                A_ki=1e8,
            )
            for i, E in enumerate([2.0, 3.0, 4.0, 5.0, 6.0])
        ]

        fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC)
        result = fitter.fit(observations)

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (2, 2)

    def test_huber_produces_covariance(self):
        """Test that Huber fitting produces covariance matrix."""
        from cflibs.inversion.physics.boltzmann import (
            BoltzmannPlotFitter,
            LineObservation,
            FitMethod,
        )

        observations = [
            LineObservation(
                wavelength_nm=300.0 + i * 10,
                intensity=1000.0 * np.exp(-E / 0.8),
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=E,
                g_k=5,
                A_ki=1e8,
            )
            for i, E in enumerate([2.0, 3.0, 4.0, 5.0, 6.0])
        ]

        fitter = BoltzmannPlotFitter(method=FitMethod.HUBER)
        result = fitter.fit(observations)

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (2, 2)


class TestCFLIBSResultFields:
    """Tests for new CFLIBSResult fields."""

    def test_result_has_new_fields(self):
        """Test that CFLIBSResult has the new uncertainty fields."""
        from cflibs.inversion.solve.iterative import CFLIBSResult

        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.9, "Si": 0.1},
            concentration_uncertainties={"Fe": 0.05, "Si": 0.02},
            iterations=5,
            converged=True,
            quality_metrics={"r_squared_last": 0.98},
            electron_density_uncertainty_cm3=1e15,
            boltzmann_covariance=np.array([[0.01, 0.005], [0.005, 0.04]]),
        )

        assert result.electron_density_uncertainty_cm3 == 1e15
        assert result.boltzmann_covariance is not None
        assert result.boltzmann_covariance.shape == (2, 2)

    def test_result_default_values(self):
        """Test that new fields have sensible defaults."""
        from cflibs.inversion.solve.iterative import CFLIBSResult

        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=0.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            concentration_uncertainties={},
            iterations=1,
            converged=True,
        )

        assert result.electron_density_uncertainty_cm3 == 0.0
        assert result.boltzmann_covariance is None


# =============================================================================
# Monte Carlo UQ Tests
# =============================================================================


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_result_creation(self):
        """Test creating a MonteCarloResult with valid data."""
        from cflibs.inversion.physics.uncertainty import MonteCarloResult, PerturbationType

        result = MonteCarloResult(
            n_samples=100,
            n_successful=95,
            T_samples=np.array([10000.0] * 95),
            ne_samples=np.array([1e17] * 95),
            concentration_samples={"Fe": np.array([0.9] * 95)},
            T_mean=10000.0,
            T_std=500.0,
            T_ci_68=(9500.0, 10500.0),
            T_ci_95=(9000.0, 11000.0),
            ne_mean=1e17,
            ne_std=1e16,
            ne_ci_68=(9e16, 1.1e17),
            ne_ci_95=(8e16, 1.2e17),
            concentrations_mean={"Fe": 0.9},
            concentrations_std={"Fe": 0.05},
            concentrations_ci_68={"Fe": (0.85, 0.95)},
            concentrations_ci_95={"Fe": (0.80, 1.0)},
            perturbation_type=PerturbationType.SPECTRAL_NOISE,
            seed=42,
        )

        assert result.n_samples == 100
        assert result.n_successful == 95
        assert result.success_rate == pytest.approx(0.95)
        assert result.T_relative_uncertainty == pytest.approx(0.05)

    def test_summary_table(self):
        """Test that summary_table returns a formatted string."""
        from cflibs.inversion.physics.uncertainty import MonteCarloResult, PerturbationType

        result = MonteCarloResult(
            n_samples=100,
            n_successful=95,
            T_samples=np.array([10000.0] * 95),
            ne_samples=np.array([1e17] * 95),
            concentration_samples={"Fe": np.array([0.9] * 95), "Si": np.array([0.1] * 95)},
            T_mean=10000.0,
            T_std=500.0,
            T_ci_68=(9500.0, 10500.0),
            T_ci_95=(9000.0, 11000.0),
            ne_mean=1e17,
            ne_std=1e16,
            ne_ci_68=(9e16, 1.1e17),
            ne_ci_95=(8e16, 1.2e17),
            concentrations_mean={"Fe": 0.9, "Si": 0.1},
            concentrations_std={"Fe": 0.05, "Si": 0.02},
            concentrations_ci_68={"Fe": (0.85, 0.95), "Si": (0.08, 0.12)},
            concentrations_ci_95={"Fe": (0.80, 1.0), "Si": (0.06, 0.14)},
            perturbation_type=PerturbationType.SPECTRAL_NOISE,
            seed=42,
        )

        table = result.summary_table()
        assert "Monte Carlo" in table
        assert "10000" in table  # Temperature
        assert "Fe" in table
        assert "Si" in table
        assert "95.0%" in table  # Success rate

    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        from cflibs.inversion.physics.uncertainty import MonteCarloResult, PerturbationType

        # Create correlated data
        np.random.seed(42)
        T_samples = np.random.normal(10000, 500, 100)
        # n_e positively correlated with T
        ne_samples = T_samples * 1e13 + np.random.normal(0, 1e15, 100)
        fe_samples = np.random.uniform(0.85, 0.95, 100)

        result = MonteCarloResult(
            n_samples=100,
            n_successful=100,
            T_samples=T_samples,
            ne_samples=ne_samples,
            concentration_samples={"Fe": fe_samples},
            T_mean=float(np.mean(T_samples)),
            T_std=float(np.std(T_samples)),
            T_ci_68=(9500.0, 10500.0),
            T_ci_95=(9000.0, 11000.0),
            ne_mean=float(np.mean(ne_samples)),
            ne_std=float(np.std(ne_samples)),
            ne_ci_68=(9e16, 1.1e17),
            ne_ci_95=(8e16, 1.2e17),
            concentrations_mean={"Fe": float(np.mean(fe_samples))},
            concentrations_std={"Fe": float(np.std(fe_samples))},
            concentrations_ci_68={"Fe": (0.85, 0.95)},
            concentrations_ci_95={"Fe": (0.80, 1.0)},
            perturbation_type=PerturbationType.SPECTRAL_NOISE,
            seed=42,
        )

        corr_data = result.correlation_matrix()

        assert corr_data["matrix"] is not None
        assert corr_data["matrix"].shape == (3, 3)  # T, n_e, C_Fe
        assert "T" in corr_data["labels"]
        assert "n_e" in corr_data["labels"]
        # T and n_e should be correlated
        assert corr_data["T_ne_corr"] > 0.5


class TestAtomicDataUncertainty:
    """Tests for AtomicDataUncertainty dataclass."""

    def test_default_uncertainty(self):
        """Test default atomic data uncertainty."""
        from cflibs.inversion.physics.uncertainty import AtomicDataUncertainty

        adu = AtomicDataUncertainty()
        assert adu.default_A_uncertainty == 0.10  # 10%

        # Should return default for any wavelength
        assert adu.get_uncertainty(500.0) == 0.10
        assert adu.get_uncertainty(300.0) == 0.10

    def test_per_line_uncertainties(self):
        """Test per-line atomic data uncertainties."""
        from cflibs.inversion.physics.uncertainty import AtomicDataUncertainty

        adu = AtomicDataUncertainty(
            default_A_uncertainty=0.10,
            per_line_uncertainties={
                500.0: 0.03,  # High quality line
                600.0: 0.25,  # Low quality line
            },
        )

        assert adu.get_uncertainty(500.0) == 0.03
        assert adu.get_uncertainty(600.0) == 0.25
        assert adu.get_uncertainty(700.0) == 0.10  # Default for unknown


class TestPerturbationType:
    """Tests for PerturbationType enum."""

    def test_perturbation_types(self):
        """Test that all perturbation types are defined."""
        from cflibs.inversion.physics.uncertainty import PerturbationType

        assert PerturbationType.SPECTRAL_NOISE.value == "spectral_noise"
        assert PerturbationType.ATOMIC_DATA.value == "atomic_data"
        assert PerturbationType.COMBINED.value == "combined"


class TestMonteCarloUQUnit:
    """Unit tests for MonteCarloUQ class (no database required)."""

    def test_initialization(self):
        """Test MonteCarloUQ initialization."""
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ
        from unittest.mock import MagicMock

        mock_solver = MagicMock()
        mc = MonteCarloUQ(mock_solver, n_samples=100, seed=42)

        assert mc.n_samples == 100
        assert mc.seed == 42
        assert mc.solver is mock_solver

    def test_perturbation_spectral_noise(self):
        """Test that spectral noise perturbation modifies intensities."""
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ, PerturbationType
        from cflibs.inversion.physics.boltzmann import LineObservation
        from unittest.mock import MagicMock

        mock_solver = MagicMock()
        mc = MonteCarloUQ(mock_solver, n_samples=10, seed=42)

        observations = [
            LineObservation(
                wavelength_nm=500.0,
                intensity=1000.0,
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=5,
                A_ki=1e8,
            )
        ]

        rng = np.random.default_rng(42)
        perturbed = mc._perturb_observations(
            observations,
            rng,
            noise_fraction=0.05,
            atomic_uncertainty=None,
            perturbation_type=PerturbationType.SPECTRAL_NOISE,
        )

        # Intensity should be perturbed
        assert perturbed[0].intensity != observations[0].intensity
        # A_ki should be unchanged
        assert perturbed[0].A_ki == observations[0].A_ki

    def test_perturbation_atomic_data(self):
        """Test that atomic data perturbation modifies A_ki values."""
        from cflibs.inversion.physics.uncertainty import (
            MonteCarloUQ,
            PerturbationType,
            AtomicDataUncertainty,
        )
        from cflibs.inversion.physics.boltzmann import LineObservation
        from unittest.mock import MagicMock

        mock_solver = MagicMock()
        mc = MonteCarloUQ(mock_solver, n_samples=10, seed=42)

        observations = [
            LineObservation(
                wavelength_nm=500.0,
                intensity=1000.0,
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=5,
                A_ki=1e8,
            )
        ]

        rng = np.random.default_rng(42)
        adu = AtomicDataUncertainty(default_A_uncertainty=0.10)

        perturbed = mc._perturb_observations(
            observations,
            rng,
            noise_fraction=None,
            atomic_uncertainty=adu,
            perturbation_type=PerturbationType.ATOMIC_DATA,
        )

        # A_ki should be perturbed
        assert perturbed[0].A_ki != observations[0].A_ki
        # Intensity should be unchanged (no spectral noise)
        assert perturbed[0].intensity == observations[0].intensity

    def test_perturbation_combined(self):
        """Test that combined perturbation modifies both intensity and A_ki."""
        from cflibs.inversion.physics.uncertainty import (
            MonteCarloUQ,
            PerturbationType,
            AtomicDataUncertainty,
        )
        from cflibs.inversion.physics.boltzmann import LineObservation
        from unittest.mock import MagicMock

        mock_solver = MagicMock()
        mc = MonteCarloUQ(mock_solver, n_samples=10, seed=42)

        observations = [
            LineObservation(
                wavelength_nm=500.0,
                intensity=1000.0,
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=5,
                A_ki=1e8,
            )
        ]

        rng = np.random.default_rng(42)
        adu = AtomicDataUncertainty(default_A_uncertainty=0.10)

        perturbed = mc._perturb_observations(
            observations,
            rng,
            noise_fraction=0.05,
            atomic_uncertainty=adu,
            perturbation_type=PerturbationType.COMBINED,
        )

        # Both should be perturbed
        assert perturbed[0].intensity != observations[0].intensity
        assert perturbed[0].A_ki != observations[0].A_ki

    def test_process_results_statistics(self):
        """Test that process_results computes correct statistics."""
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ, PerturbationType
        from cflibs.inversion.solve.iterative import CFLIBSResult
        from unittest.mock import MagicMock

        mock_solver = MagicMock()
        mc = MonteCarloUQ(mock_solver, n_samples=100, seed=42)

        # Create mock results with known distribution
        np.random.seed(42)
        T_values = np.random.normal(10000, 500, 100)
        ne_values = np.random.normal(1e17, 1e16, 100)
        fe_values = np.random.uniform(0.85, 0.95, 100)

        mock_results = []
        for i in range(100):
            result = CFLIBSResult(
                temperature_K=T_values[i],
                temperature_uncertainty_K=0.0,
                electron_density_cm3=ne_values[i],
                concentrations={"Fe": fe_values[i]},
                concentration_uncertainties={},
                iterations=5,
                converged=True,
            )
            mock_results.append(result)

        mc_result = mc._process_results(mock_results, PerturbationType.SPECTRAL_NOISE)

        # Check that statistics are computed correctly
        assert mc_result.n_successful == 100
        assert mc_result.T_mean == pytest.approx(np.mean(T_values), rel=0.01)
        assert mc_result.T_std == pytest.approx(np.std(T_values, ddof=1), rel=0.01)
        assert mc_result.ne_mean == pytest.approx(np.mean(ne_values), rel=0.01)

    def test_process_results_handles_failures(self):
        """Test that process_results handles failed runs correctly."""
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ, PerturbationType
        from cflibs.inversion.solve.iterative import CFLIBSResult
        from unittest.mock import MagicMock

        mock_solver = MagicMock()
        mc = MonteCarloUQ(mock_solver, n_samples=10, seed=42)

        # Mix of successful and failed results
        mock_results = [
            CFLIBSResult(
                temperature_K=10000.0,
                temperature_uncertainty_K=0.0,
                electron_density_cm3=1e17,
                concentrations={"Fe": 0.9},
                concentration_uncertainties={},
                iterations=5,
                converged=True,
            ),
            None,  # Failed
            CFLIBSResult(
                temperature_K=10500.0,
                temperature_uncertainty_K=0.0,
                electron_density_cm3=1.1e17,
                concentrations={"Fe": 0.85},
                concentration_uncertainties={},
                iterations=5,
                converged=True,
            ),
            None,  # Failed
            CFLIBSResult(
                temperature_K=9500.0,
                temperature_uncertainty_K=0.0,
                electron_density_cm3=9e16,
                concentrations={"Fe": 0.95},
                concentration_uncertainties={},
                iterations=5,
                converged=True,
            ),
        ]

        mc_result = mc._process_results(mock_results, PerturbationType.SPECTRAL_NOISE)

        assert mc_result.n_successful == 3
        assert len(mc_result.failed_indices) == 2
        assert 1 in mc_result.failed_indices
        assert 3 in mc_result.failed_indices


class TestRunMonteCarloUQFunction:
    """Tests for run_monte_carlo_uq convenience function."""

    def test_convenience_function_exists(self):
        """Test that run_monte_carlo_uq function is importable."""
        from cflibs.inversion.physics.uncertainty import run_monte_carlo_uq

        assert callable(run_monte_carlo_uq)


class TestMonteCarloResultCompareWithBayesian:
    """Tests for MonteCarloResult.compare_with_bayesian method."""

    def test_agreement_when_results_match(self):
        """Test that comparison shows agreement for matching results."""
        from cflibs.inversion.physics.uncertainty import MonteCarloResult, PerturbationType
        from cflibs.core.constants import EV_TO_K

        # Create MC result
        mc_result = MonteCarloResult(
            n_samples=100,
            n_successful=100,
            T_samples=np.array([10000.0] * 100),
            ne_samples=np.array([1e17] * 100),
            concentration_samples={"Fe": np.array([0.9] * 100)},
            T_mean=10000.0,
            T_std=500.0,
            T_ci_68=(9500.0, 10500.0),
            T_ci_95=(9000.0, 11000.0),
            ne_mean=1e17,
            ne_std=1e16,
            ne_ci_68=(9e16, 1.1e17),
            ne_ci_95=(8e16, 1.2e17),
            concentrations_mean={"Fe": 0.9},
            concentrations_std={"Fe": 0.05},
            concentrations_ci_68={"Fe": (0.85, 0.95)},
            concentrations_ci_95={"Fe": (0.80, 1.0)},
            perturbation_type=PerturbationType.SPECTRAL_NOISE,
            seed=42,
        )

        # Create mock Bayesian result with matching values
        class MockBayesianResult:
            T_eV_mean = 10000.0 / EV_TO_K  # Same T in eV
            T_eV_std = 500.0 / EV_TO_K
            n_e_mean = 1e17
            log_ne_mean = 17.0
            log_ne_std = 0.05

        comparison = mc_result.compare_with_bayesian(MockBayesianResult())

        assert comparison["T_agreement"] is True
        assert comparison["ne_agreement"] is True
        assert comparison["ci_overlap_T"] is True
        assert comparison["ci_overlap_ne"] is True
        assert "AGREE" in comparison["summary"]

    def test_disagreement_when_results_differ(self):
        """Test that comparison shows disagreement for different results."""
        from cflibs.inversion.physics.uncertainty import MonteCarloResult, PerturbationType
        from cflibs.core.constants import EV_TO_K

        # Create MC result with T=10000 K
        mc_result = MonteCarloResult(
            n_samples=100,
            n_successful=100,
            T_samples=np.array([10000.0] * 100),
            ne_samples=np.array([1e17] * 100),
            concentration_samples={"Fe": np.array([0.9] * 100)},
            T_mean=10000.0,
            T_std=200.0,  # Tight uncertainty
            T_ci_68=(9800.0, 10200.0),
            T_ci_95=(9600.0, 10400.0),  # Tight CI
            ne_mean=1e17,
            ne_std=1e15,
            ne_ci_68=(9.9e16, 1.01e17),
            ne_ci_95=(9.8e16, 1.02e17),
            concentrations_mean={"Fe": 0.9},
            concentrations_std={"Fe": 0.05},
            concentrations_ci_68={"Fe": (0.85, 0.95)},
            concentrations_ci_95={"Fe": (0.80, 1.0)},
            perturbation_type=PerturbationType.SPECTRAL_NOISE,
            seed=42,
        )

        # Create mock Bayesian result with very different T
        class MockBayesianResult:
            T_eV_mean = 15000.0 / EV_TO_K  # Very different T
            T_eV_std = 200.0 / EV_TO_K
            n_e_mean = 1e18  # Very different n_e
            log_ne_mean = 18.0
            log_ne_std = 0.05

        comparison = mc_result.compare_with_bayesian(MockBayesianResult(), tolerance=0.1)

        assert comparison["T_agreement"] is False
        assert comparison["T_difference"] > 0.1
        assert "DISAGREE" in comparison["summary"]


@pytest.mark.requires_uncertainty
class TestSolveWithUncertaintyConsistency:
    """Regression tests: solve() and solve_with_uncertainty() must agree nominally."""

    @pytest.fixture
    def mock_db(self):
        from unittest.mock import MagicMock
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.atomic.structures import PartitionFunction

        db = MagicMock(spec=AtomicDatabase)
        db.get_ionization_potential.return_value = 7.0
        coeffs_I = [3.2188, 0, 0, 0, 0]  # log(25) constant partition function
        db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
            element=el,
            ionization_stage=sp,
            coefficients=coeffs_I,
            t_min=1000,
            t_max=20000,
            source="test",
        )
        return db

    def test_nominal_concentrations_agree_neutral_only(self, mock_db):
        """solve() and solve_with_uncertainty() concentrations agree for neutral-only obs."""
        from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, LineObservation

        T_eV = 1.0
        solver = IterativeCFLIBSSolver(mock_db, max_iterations=15)

        obs = []
        for el, intercept in [("A", 10.0), ("B", 10.0)]:
            for E in [1.0, 2.0, 3.0, 4.0]:
                y = intercept - E / T_eV
                obs.append(
                    LineObservation(
                        500.0, np.exp(y) / 500.0, 0.01 * np.exp(y) / 500.0, el, 1, E, 1, 1.0
                    )
                )

        result_det = solver.solve(obs)
        result_uq = solver.solve_with_uncertainty(obs)

        for el in result_det.concentrations:
            assert result_det.concentrations[el] == pytest.approx(
                result_uq.concentrations[el], abs=0.02
            ), f"Concentration for {el} disagrees between solve() and solve_with_uncertainty()"

    def test_nominal_concentrations_agree_mixed_stage(self, mock_db):
        """solve() and solve_with_uncertainty() agree for mixed neutral+ionic observations."""
        from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, LineObservation
        from cflibs.core.constants import SAHA_CONST_CM3

        T_eV = 1.0
        n_e_init = 1.0e17
        ip = 7.0
        wavelength_nm = 500.0
        saha_offset = np.log((SAHA_CONST_CM3 / n_e_init) * (T_eV**1.5))
        common_intercept = 8.0

        solver = IterativeCFLIBSSolver(mock_db, max_iterations=15)
        obs = []

        # Neutral lines for element A
        for E in [1.0, 2.0, 3.0]:
            y = common_intercept - E / T_eV
            intensity = np.exp(y) / wavelength_nm
            obs.append(
                LineObservation(
                    wavelength_nm, intensity, max(intensity * 0.02, 1e-10), "A", 1, E, 1, 1.0
                )
            )

        # Ionic lines for element A
        for E in [4.0, 5.0, 6.0]:
            y = common_intercept + saha_offset - (ip + E) / T_eV
            intensity = np.exp(y) / wavelength_nm
            obs.append(
                LineObservation(
                    wavelength_nm, intensity, max(intensity * 0.02, 1e-10), "A", 2, E, 1, 1.0
                )
            )

        result_det = solver.solve(obs)
        result_uq = solver.solve_with_uncertainty(obs)

        # Nominal concentrations must agree within 2%
        assert result_det.concentrations["A"] == pytest.approx(
            result_uq.concentrations["A"], abs=0.02
        )

    def test_temperature_uncertainty_is_positive(self, mock_db):
        """solve_with_uncertainty() should propagate a non-zero temperature uncertainty."""
        from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, LineObservation

        T_eV = 1.0
        solver = IterativeCFLIBSSolver(mock_db, max_iterations=15)

        obs = []
        for el in ["A", "B"]:
            for E in [1.0, 2.0, 3.0, 4.0]:
                y = 10.0 - E / T_eV
                intensity = np.exp(y) / 500.0
                obs.append(
                    LineObservation(
                        500.0, intensity, max(intensity * 0.02, 1e-10), el, 1, E, 1, 1.0
                    )
                )

        result = solver.solve_with_uncertainty(obs)
        assert result.temperature_uncertainty_K > 0.0

    def test_mixed_stage_multi_element_uncertainty_is_bounded(self, mock_db):
        """solve_with_uncertainty() should keep mixed-stage T uncertainty bounded."""
        from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, LineObservation
        from cflibs.core.constants import SAHA_CONST_CM3

        rng = np.random.default_rng(123)
        T_eV = 1.0
        n_e_init = 1.0e17
        wavelength_nm = 500.0
        saha_offset = np.log((SAHA_CONST_CM3 / n_e_init) * (T_eV**1.5))
        solver = IterativeCFLIBSSolver(mock_db, max_iterations=15)

        observations = []
        for element, intercept in [("A", 8.0), ("B", 7.5)]:
            for E_k in [1.0, 2.0, 3.0, 4.0]:
                intensity = np.exp(intercept - E_k / T_eV) / wavelength_nm
                sigma = max(intensity * 0.02, 1e-12)
                noisy_intensity = max(intensity + rng.normal(0.0, sigma), 1e-12)
                observations.append(
                    LineObservation(
                        wavelength_nm,
                        noisy_intensity,
                        sigma,
                        element,
                        1,
                        E_k,
                        1,
                        1.0,
                    )
                )

            for E_k in [4.5, 5.5, 6.5, 7.5]:
                intensity = np.exp(intercept + saha_offset - (7.0 + E_k) / T_eV) / wavelength_nm
                sigma = max(intensity * 0.02, 1e-12)
                noisy_intensity = max(intensity + rng.normal(0.0, sigma), 1e-12)
                observations.append(
                    LineObservation(
                        wavelength_nm,
                        noisy_intensity,
                        sigma,
                        element,
                        2,
                        E_k,
                        1,
                        1.0,
                    )
                )

        result_det = solver.solve(observations)
        result_uq = solver.solve_with_uncertainty(observations)

        for element in result_det.concentrations:
            assert result_det.concentrations[element] == pytest.approx(
                result_uq.concentrations[element], abs=0.02
            )
            assert result_uq.concentration_uncertainties[element] > 0.0

        assert result_uq.boltzmann_covariance is not None
        assert result_uq.boltzmann_covariance.shape == (2, 2)
        assert result_uq.quality_metrics["boltzmann_covariance_element"] == "A"
        assert result_uq.temperature_uncertainty_K > 0.0


# ============================================================================
# Audit Family 7: completing the uncertainty variance budget
# ============================================================================


class TestSahaFactorNeUFloat:
    """Bug (a): saha_factor_with_uncertainty must propagate an n_e UFloat."""

    def test_exact_ne_gives_no_extra_variance(self):
        """A plain float n_e is treated as exact (S has no n_e contribution)."""
        from cflibs.inversion.physics.uncertainty import saha_factor_with_uncertainty
        from uncertainties import ufloat

        T_eV_u = ufloat(1.0, 0.05)
        S_float = saha_factor_with_uncertainty(T_eV_u, 1e17, 7.9, 25.0, 15.0, 6.04e21)

        # std comes only from T; recompute with T exact to isolate.
        T_exact = ufloat(1.0, 0.0)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            S_t_exact = saha_factor_with_uncertainty(T_exact, 1e17, 7.9, 25.0, 15.0, 6.04e21)
        assert S_t_exact.std_dev == pytest.approx(0.0, abs=1e-30)
        assert S_float.std_dev > 0.0  # only T contributes

    def test_ne_ufloat_adds_relative_variance(self):
        """S scales as 1/n_e, so a 10% n_e UFloat -> ~10% relative term in S.

        Independent oracle: for S = k / n_e (T exact), sigma_S / S = sigma_ne / n_e.
        """
        from cflibs.inversion.physics.uncertainty import saha_factor_with_uncertainty
        from uncertainties import ufloat
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            T_exact = ufloat(1.0, 0.0)
            ne_u = ufloat(1e17, 0.10 * 1e17)
            S = saha_factor_with_uncertainty(T_exact, ne_u, 7.9, 25.0, 15.0, 6.04e21)

        assert S.std_dev / S.nominal_value == pytest.approx(0.10, rel=1e-6)


class TestClosurePropagatesNeUncertainty:
    """Bug (a) gate: propagate_through_closure with an n_e-uncertain multiplier
    must yield a nonzero sigma_C contribution."""

    def test_uncertain_abundance_multiplier_gives_nonzero_sigma_c(self):
        from cflibs.inversion.physics.uncertainty import (
            propagate_through_closure_standard,
            saha_factor_with_uncertainty,
        )
        from uncertainties import ufloat
        import warnings as _w

        # Exact intercepts so the ONLY uncertainty source is the n_e UFloat
        # threaded through the Saha (1 + n_II/n_I) multipliers.
        intercepts_u = {"Fe": ufloat(10.0, 0.0), "Si": ufloat(8.0, 0.0)}
        partition_funcs = {"Fe": 25.0, "Si": 15.0}

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            T_u = ufloat(1.0, 0.0)
            ne_u = ufloat(1e17, 0.10 * 1e17)  # 10%-uncertain n_e
            mults = {
                "Fe": 1.0 + saha_factor_with_uncertainty(T_u, ne_u, 7.9, 25.0, 15.0, 6.04e21),
                "Si": 1.0 + saha_factor_with_uncertainty(T_u, ne_u, 8.15, 15.0, 10.0, 6.04e21),
            }

        conc_u = propagate_through_closure_standard(
            intercepts_u, partition_funcs, abundance_multipliers=mults
        )

        # n_e variance must reach sigma_C (non-zero contribution)
        for el, cu in conc_u.items():
            assert cu.std_dev > 0.0, f"sigma_C[{el}] should be non-zero from n_e UFloat"

        # And exact multipliers must give zero sigma_C (control)
        conc0 = propagate_through_closure_standard(
            intercepts_u, partition_funcs, abundance_multipliers={"Fe": 2.0, "Si": 2.0}
        )
        for cu in conc0.values():
            assert cu.std_dev == pytest.approx(0.0, abs=1e-12)


@pytest.mark.requires_uncertainty
class TestSolveWithUncertaintyNeBudget:
    """Bug (a) wiring: solve_with_uncertainty(n_e_relative_uncertainty=...) must
    enlarge sigma_C without moving the point estimates (concentrations/T/n_e)."""

    @pytest.fixture
    def mock_db(self):
        from unittest.mock import MagicMock
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.atomic.structures import PartitionFunction

        db = MagicMock(spec=AtomicDatabase)
        # Element-dependent IPs so the two elements have DIFFERENT Saha
        # ionization balance. A common-mode (perfectly correlated) n_e error
        # cancels in closure when every element shares the SAME multiplier;
        # distinct IPs break that degeneracy so the n_e variance reaches sigma_C.
        ip_map = {"A": 6.0, "B": 9.0}
        db.get_ionization_potential.side_effect = lambda el, sp: ip_map.get(el, 7.0)
        coeffs_I = [3.2188, 0, 0, 0, 0]
        db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
            element=el,
            ionization_stage=sp,
            coefficients=coeffs_I,
            t_min=1000,
            t_max=20000,
            source="test",
        )
        return db

    def _make_mixed_stage_obs(self):
        from cflibs.inversion.solve.iterative import LineObservation
        from cflibs.core.constants import SAHA_CONST_CM3

        T_eV = 1.0
        n_e_init = 1.0e17
        wavelength_nm = 500.0
        saha_offset = np.log((SAHA_CONST_CM3 / n_e_init) * (T_eV**1.5))
        ip_map = {"A": 6.0, "B": 9.0}  # match the fixture's per-element IPs
        obs = []
        for element, intercept in [("A", 8.0), ("B", 7.5)]:
            ip = ip_map[element]
            for E_k in [1.0, 2.0, 3.0]:
                intensity = np.exp(intercept - E_k / T_eV) / wavelength_nm
                obs.append(
                    LineObservation(
                        wavelength_nm,
                        intensity,
                        max(intensity * 0.02, 1e-12),
                        element,
                        1,
                        E_k,
                        1,
                        1.0,
                    )
                )
            for E_k in [4.5, 5.5, 6.5]:
                intensity = np.exp(intercept + saha_offset - (ip + E_k) / T_eV) / wavelength_nm
                obs.append(
                    LineObservation(
                        wavelength_nm,
                        intensity,
                        max(intensity * 0.02, 1e-12),
                        element,
                        2,
                        E_k,
                        1,
                        1.0,
                    )
                )
        return obs

    def test_ne_uncertainty_enlarges_sigma_c_without_moving_point_estimates(self, mock_db):
        from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

        solver = IterativeCFLIBSSolver(mock_db, max_iterations=15)
        obs = self._make_mixed_stage_obs()

        res_base = solver.solve_with_uncertainty(obs, n_e_relative_uncertainty=0.0)
        res_ne = solver.solve_with_uncertainty(obs, n_e_relative_uncertainty=0.10)

        # Point estimates (concentrations, T, n_e) must be identical: this is an
        # uncertainty-only change.
        assert res_base.temperature_K == pytest.approx(res_ne.temperature_K, rel=1e-12)
        assert res_base.electron_density_cm3 == pytest.approx(
            res_ne.electron_density_cm3, rel=1e-12
        )
        for el in res_base.concentrations:
            assert res_base.concentrations[el] == pytest.approx(res_ne.concentrations[el], rel=1e-9)

        # sigma_C must be strictly larger with the extra n_e variance for at
        # least one element (the mixed-stage elements pick up the Saha term).
        enlarged = any(
            res_ne.concentration_uncertainties[el]
            > res_base.concentration_uncertainties[el] + 1e-12
            for el in res_base.concentrations
        )
        assert enlarged, "n_e uncertainty did not enlarge any sigma_C"


class TestRunMonteCarloUQDefaultsToAtomicData:
    """Bug (b): the convenience run_monte_carlo_uq must include atomic-data
    (A_ki) perturbation by default."""

    def test_default_perturbation_is_combined(self):
        """Default convenience call forwards COMBINED (spectral + atomic data)."""
        from cflibs.inversion.physics.uncertainty import (
            MonteCarloUQ,
            PerturbationType,
            run_monte_carlo_uq,
        )
        from cflibs.inversion.physics.boltzmann import LineObservation
        from unittest.mock import MagicMock, patch

        observations = [LineObservation(500.0, 1000.0, 50.0, "Fe", 1, 3.0, 5, 1e8)]
        captured = {}

        def spy_run(self, obs, **kwargs):
            captured["perturbation_type"] = kwargs.get("perturbation_type")
            return None  # short-circuit the (mocked) solver pipeline

        with patch.object(MonteCarloUQ, "run", spy_run):
            run_monte_carlo_uq(MagicMock(), observations, n_samples=2)

        assert captured["perturbation_type"] == PerturbationType.COMBINED

    def test_atomic_data_actually_perturbs_aki(self):
        """COMBINED perturbation modifies A_ki (atomic-data variance is included)."""
        from cflibs.inversion.physics.uncertainty import (
            MonteCarloUQ,
            PerturbationType,
            AtomicDataUncertainty,
        )
        from cflibs.inversion.physics.boltzmann import LineObservation
        from unittest.mock import MagicMock

        mc = MonteCarloUQ(MagicMock(), n_samples=1, seed=1)
        obs = [LineObservation(500.0, 1000.0, 50.0, "Fe", 1, 3.0, 5, 1e8)]
        rng = np.random.default_rng(1)
        perturbed = mc._perturb_observations(
            obs,
            rng,
            0.05,
            AtomicDataUncertainty(default_A_uncertainty=0.10),
            PerturbationType.COMBINED,
        )
        assert perturbed[0].A_ki != obs[0].A_ki


class TestLowCountNoiseIsUnbiased:
    """Bug (c): low-count (ps-LIBS 10-50 count) MC intensity perturbation must
    not be biased upward. The old additive-Gaussian +1.0 floor truncated only
    downward excursions, biasing the mean up."""

    def _perturb_many(self, true_intensity, n, noise_model, noise_fraction):
        from cflibs.inversion.physics.uncertainty import (
            MonteCarloUQ,
            PerturbationType,
        )
        from cflibs.inversion.physics.boltzmann import LineObservation
        from unittest.mock import MagicMock

        mc = MonteCarloUQ(MagicMock(), n_samples=n, seed=7)
        rng = np.random.default_rng(7)
        obs = [LineObservation(500.0, true_intensity, true_intensity, "Fe", 1, 3.0, 5, 1e8)]
        sampled = []
        for _ in range(n):
            p = mc._perturb_observations(
                obs, rng, noise_fraction, None, PerturbationType.SPECTRAL_NOISE, noise_model
            )
            sampled.append(p[0].intensity)
        return np.array(sampled)

    def test_poisson_mean_matches_true_low_count(self):
        """Poisson model: E[Poisson(I)] = I exactly (independent oracle)."""
        from cflibs.inversion.physics.uncertainty import NoiseModel

        true = 20.0  # low-count ps-LIBS line
        sampled = self._perturb_many(true, 40000, NoiseModel.POISSON, noise_fraction=None)
        # Within MC error (~ sqrt(true/n)) the mean equals the true rate, no
        # upward bias.
        assert sampled.mean() == pytest.approx(true, abs=0.15)

    def test_gaussian_floor_no_longer_biases_to_one(self):
        """Gaussian model: floor moved from +1.0 to 0.0 (no upward +1 floor bias).

        Independent oracle: re-sample the SAME perturbed intensities and apply
        the OLD vs NEW floor. The old +1.0 floor must give a strictly larger
        (more upward-biased) mean than the new 0.0 floor for a low-count line.
        """
        from cflibs.inversion.physics.uncertainty import NoiseModel

        true = 2.0  # low-count line where the floor bites
        # Reproduce the additive-Gaussian draws the perturber would use, then
        # compare floors directly (oracle independent of the production clip).
        rng = np.random.default_rng(99)
        sigma = 1.0 * true  # noise_fraction=1.0 -> sigma = true
        raw = true + rng.normal(0.0, sigma, size=200000)
        mean_old_floor = np.maximum(raw, 1.0).mean()  # buggy +1.0 floor
        mean_new_floor = np.maximum(raw, 0.0).mean()  # fixed 0.0 floor

        assert mean_old_floor > mean_new_floor, "new floor must reduce upward bias"
        assert mean_old_floor - true > mean_new_floor - true

        # And the production Gaussian path must never clip up to the old 1.0
        # floor: with floor=0.0 some draws land in [0, 1).
        gauss = self._perturb_many(true, 50000, NoiseModel.GAUSSIAN, noise_fraction=1.0)
        assert np.any(gauss < 1.0), "fixed floor should allow intensities below 1.0"

"""
Tests for uncertainty propagation utilities.

Tests the `cflibs.inversion.uncertainty` module which provides automatic
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
        from cflibs.inversion.uncertainty import create_boltzmann_uncertainties

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
        from cflibs.inversion.uncertainty import create_boltzmann_uncertainties

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
        from cflibs.inversion.uncertainty import create_boltzmann_uncertainties

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
        from cflibs.inversion.uncertainty import temperature_from_slope
        from uncertainties import ufloat
        from cflibs.core.constants import KB_EV, EV_TO_K

        # slope = -1/(kB * T_eV)
        # For T = 10000 K = 0.862 eV, slope = -1.16
        T_expected_K = 10000.0
        T_expected_eV = T_expected_K / EV_TO_K
        slope = -1.0 / (KB_EV * T_expected_eV)

        slope_u = ufloat(slope, 0.01)
        T_K_u = temperature_from_slope(slope_u)

        assert T_K_u.nominal_value == pytest.approx(T_expected_K, rel=0.01)
        # Uncertainty should be non-zero
        assert T_K_u.std_dev > 0


class TestClosurePropagation:
    """Tests for closure equation uncertainty propagation."""

    def test_standard_closure_sum_to_one(self):
        """Test that concentrations sum to 1 regardless of uncertainties."""
        from cflibs.inversion.uncertainty import propagate_through_closure_standard
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
        from cflibs.inversion.uncertainty import propagate_through_closure_standard
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
        from cflibs.inversion.uncertainty import propagate_through_closure_matrix
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
        from cflibs.inversion.uncertainty import propagate_through_closure_oxide
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
        from cflibs.inversion.uncertainty import propagate_through_closure_standard
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
        from cflibs.inversion.uncertainty import propagate_through_closure_standard
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
        from cflibs.inversion.uncertainty import extract_values_and_uncertainties
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
        from cflibs.inversion.boltzmann import (
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
        from cflibs.inversion.boltzmann import (
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
        from cflibs.inversion.boltzmann import (
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
        from cflibs.inversion.solver import CFLIBSResult

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
        from cflibs.inversion.solver import CFLIBSResult

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
        from cflibs.inversion.uncertainty import MonteCarloResult, PerturbationType

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
        from cflibs.inversion.uncertainty import MonteCarloResult, PerturbationType

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
        from cflibs.inversion.uncertainty import MonteCarloResult, PerturbationType

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
        from cflibs.inversion.uncertainty import AtomicDataUncertainty

        adu = AtomicDataUncertainty()
        assert adu.default_A_uncertainty == 0.10  # 10%

        # Should return default for any wavelength
        assert adu.get_uncertainty(500.0) == 0.10
        assert adu.get_uncertainty(300.0) == 0.10

    def test_per_line_uncertainties(self):
        """Test per-line atomic data uncertainties."""
        from cflibs.inversion.uncertainty import AtomicDataUncertainty

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
        from cflibs.inversion.uncertainty import PerturbationType

        assert PerturbationType.SPECTRAL_NOISE.value == "spectral_noise"
        assert PerturbationType.ATOMIC_DATA.value == "atomic_data"
        assert PerturbationType.COMBINED.value == "combined"


class TestMonteCarloUQUnit:
    """Unit tests for MonteCarloUQ class (no database required)."""

    def test_initialization(self):
        """Test MonteCarloUQ initialization."""
        from cflibs.inversion.uncertainty import MonteCarloUQ
        from unittest.mock import MagicMock

        mock_solver = MagicMock()
        mc = MonteCarloUQ(mock_solver, n_samples=100, seed=42)

        assert mc.n_samples == 100
        assert mc.seed == 42
        assert mc.solver is mock_solver

    def test_perturbation_spectral_noise(self):
        """Test that spectral noise perturbation modifies intensities."""
        from cflibs.inversion.uncertainty import MonteCarloUQ, PerturbationType
        from cflibs.inversion.boltzmann import LineObservation
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
        from cflibs.inversion.uncertainty import (
            MonteCarloUQ,
            PerturbationType,
            AtomicDataUncertainty,
        )
        from cflibs.inversion.boltzmann import LineObservation
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
        from cflibs.inversion.uncertainty import (
            MonteCarloUQ,
            PerturbationType,
            AtomicDataUncertainty,
        )
        from cflibs.inversion.boltzmann import LineObservation
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
        from cflibs.inversion.uncertainty import MonteCarloUQ, PerturbationType
        from cflibs.inversion.solver import CFLIBSResult
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
        from cflibs.inversion.uncertainty import MonteCarloUQ, PerturbationType
        from cflibs.inversion.solver import CFLIBSResult
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
        from cflibs.inversion.uncertainty import run_monte_carlo_uq

        assert callable(run_monte_carlo_uq)


class TestMonteCarloResultCompareWithBayesian:
    """Tests for MonteCarloResult.compare_with_bayesian method."""

    def test_agreement_when_results_match(self):
        """Test that comparison shows agreement for matching results."""
        from cflibs.inversion.uncertainty import MonteCarloResult, PerturbationType
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
        from cflibs.inversion.uncertainty import MonteCarloResult, PerturbationType
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
        from cflibs.inversion.solver import IterativeCFLIBSSolver, LineObservation

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
        from cflibs.inversion.solver import IterativeCFLIBSSolver, LineObservation
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
        from cflibs.inversion.solver import IterativeCFLIBSSolver, LineObservation

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

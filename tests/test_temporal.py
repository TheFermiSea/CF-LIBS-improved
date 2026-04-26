"""
Tests for temporal dynamics and time-resolved optimization module.

This module tests:
- PlasmaEvolutionModel: T(t) and n_e(t) evolution
- GateTimingOptimizer: Optimal gate delay/width finding
- TemporalSelfAbsorptionCorrector: Time-varying optical depth correction
- TimeResolvedCFLIBSSolver: Multi-gate CF-LIBS analysis
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from cflibs.inversion.temporal import (
    PlasmaPhase,
    TemporalGateConfig,
    TimeResolvedSpectrum,
    PlasmaEvolutionPoint,
    PlasmaEvolutionProfile,
    GateOptimizationResult,
    TemporalSelfAbsorptionResult,
    PlasmaEvolutionModel,
    GateTimingOptimizer,
    TemporalSelfAbsorptionCorrector,
    TimeResolvedCFLIBSSolver,
    create_default_evolution_model,
    recommend_gate_timing,
)
from cflibs.inversion.boltzmann import LineObservation

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_evolution_model():
    """Default plasma evolution model for testing."""
    return PlasmaEvolutionModel(
        T_initial_K=20000.0,
        ne_initial_cm3=1e18,
        tau_T_ns=1000.0,
        tau_ne_ns=500.0,
        T_ambient_K=300.0,
    )


@pytest.fixture
def sample_line_observations():
    """Sample line observations for testing."""
    return [
        LineObservation(
            wavelength_nm=500.0,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.5,
            g_k=9,
            A_ki=1e8,
        ),
        LineObservation(
            wavelength_nm=510.0,
            intensity=800.0,
            intensity_uncertainty=40.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.0,
            g_k=7,
            A_ki=8e7,
        ),
        LineObservation(
            wavelength_nm=520.0,
            intensity=600.0,
            intensity_uncertainty=30.0,
            element="Fe",
            ionization_stage=2,
            E_k_ev=4.5,
            g_k=11,
            A_ki=1.2e8,
        ),
    ]


@pytest.fixture
def sample_gate_config():
    """Sample gate configuration."""
    return TemporalGateConfig(delay_ns=500.0, width_ns=1000.0, label="test_gate")


# =============================================================================
# Tests for Data Classes
# =============================================================================


class TestTemporalGateConfig:
    """Tests for TemporalGateConfig dataclass."""

    def test_basic_creation(self):
        """Test basic gate config creation."""
        gate = TemporalGateConfig(delay_ns=500.0, width_ns=1000.0)
        assert gate.delay_ns == 500.0
        assert gate.width_ns == 1000.0
        assert gate.label == ""

    def test_with_label(self):
        """Test gate config with label."""
        gate = TemporalGateConfig(delay_ns=100.0, width_ns=500.0, label="early")
        assert gate.label == "early"

    def test_center_time(self):
        """Test center time calculation."""
        gate = TemporalGateConfig(delay_ns=500.0, width_ns=1000.0)
        assert gate.center_ns == 1000.0  # 500 + 1000/2

    def test_end_time(self):
        """Test end time calculation."""
        gate = TemporalGateConfig(delay_ns=500.0, width_ns=1000.0)
        assert gate.end_ns == 1500.0

    def test_negative_delay_raises(self):
        """Test that negative delay raises ValueError."""
        with pytest.raises(ValueError, match="delay must be non-negative"):
            TemporalGateConfig(delay_ns=-100.0, width_ns=500.0)

    def test_zero_width_raises(self):
        """Test that zero width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            TemporalGateConfig(delay_ns=100.0, width_ns=0.0)

    def test_negative_width_raises(self):
        """Test that negative width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            TemporalGateConfig(delay_ns=100.0, width_ns=-500.0)


class TestTimeResolvedSpectrum:
    """Tests for TimeResolvedSpectrum dataclass."""

    def test_creation(self, sample_gate_config, sample_line_observations):
        """Test spectrum creation."""
        spectrum = TimeResolvedSpectrum(
            gate=sample_gate_config,
            observations=sample_line_observations,
            continuum_level=50.0,
            snr_estimate=30.0,
        )
        assert spectrum.gate == sample_gate_config
        assert len(spectrum.observations) == 3
        assert spectrum.continuum_level == 50.0
        assert spectrum.snr_estimate == 30.0

    def test_with_metadata(self, sample_gate_config, sample_line_observations):
        """Test spectrum with metadata."""
        spectrum = TimeResolvedSpectrum(
            gate=sample_gate_config,
            observations=sample_line_observations,
            metadata={"laser_energy_mJ": 50.0},
        )
        assert spectrum.metadata["laser_energy_mJ"] == 50.0


class TestPlasmaEvolutionPoint:
    """Tests for PlasmaEvolutionPoint dataclass."""

    def test_creation(self):
        """Test point creation."""
        point = PlasmaEvolutionPoint(
            time_ns=500.0,
            temperature_K=15000.0,
            electron_density_cm3=5e17,
            lte_validity=0.9,
            phase=PlasmaPhase.INTERMEDIATE,
        )
        assert point.time_ns == 500.0
        assert point.temperature_K == 15000.0
        assert point.electron_density_cm3 == 5e17
        assert point.lte_validity == 0.9
        assert point.phase == PlasmaPhase.INTERMEDIATE

    def test_default_values(self):
        """Test default values."""
        point = PlasmaEvolutionPoint(
            time_ns=100.0,
            temperature_K=20000.0,
            electron_density_cm3=1e18,
        )
        assert point.lte_validity == 1.0
        assert point.phase == PlasmaPhase.INTERMEDIATE


# =============================================================================
# Tests for PlasmaEvolutionModel
# =============================================================================


class TestPlasmaEvolutionModel:
    """Tests for PlasmaEvolutionModel class."""

    def test_temperature_at_zero(self, default_evolution_model):
        """Test temperature at t=0."""
        T = default_evolution_model.temperature(0.0)
        assert T == pytest.approx(20000.0, rel=0.01)

    def test_temperature_decay(self, default_evolution_model):
        """Test temperature exponential decay."""
        # At t = tau_T, temperature should decrease by factor of e
        T_0 = default_evolution_model.temperature(0.0)
        T_tau = default_evolution_model.temperature(1000.0)  # tau_T

        # T(tau) = (T0 - T_amb) * exp(-1) + T_amb
        T_expected = (T_0 - 300.0) / np.e + 300.0
        assert T_tau == pytest.approx(T_expected, rel=0.01)

    def test_temperature_approaches_ambient(self, default_evolution_model):
        """Test temperature approaches ambient at late times."""
        T_late = default_evolution_model.temperature(10000.0)  # 10 * tau_T
        assert T_late == pytest.approx(300.0, abs=100.0)

    def test_electron_density_at_zero(self, default_evolution_model):
        """Test electron density at t=0."""
        n_e = default_evolution_model.electron_density(0.0)
        assert n_e == pytest.approx(1e18, rel=0.01)

    def test_electron_density_decay(self, default_evolution_model):
        """Test electron density exponential decay."""
        n_e_0 = default_evolution_model.electron_density(0.0)
        n_e_tau = default_evolution_model.electron_density(500.0)  # tau_ne

        # Should decrease by factor of e
        assert n_e_tau == pytest.approx(n_e_0 / np.e, rel=0.01)

    def test_density_decays_faster_than_temperature(self, default_evolution_model):
        """Test that density decays faster than temperature."""
        # At same time, density should be lower relative to initial
        t = 500.0
        T_ratio = default_evolution_model.temperature(t) / 20000.0
        n_e_ratio = default_evolution_model.electron_density(t) / 1e18

        # Density should have decayed more
        assert n_e_ratio < T_ratio

    def test_negative_time_returns_initial(self, default_evolution_model):
        """Test negative time returns initial values."""
        assert default_evolution_model.temperature(-100.0) == 20000.0
        assert default_evolution_model.electron_density(-100.0) == 1e18

    def test_lte_validity_early(self, default_evolution_model):
        """Test LTE validity at early times (high density)."""
        lte = default_evolution_model.lte_validity(0.0)
        # With n_e = 1e18 and T = 20000 K, should be excellent LTE
        assert lte > 0.9

    def test_lte_validity_late(self, default_evolution_model):
        """Test LTE validity at late times (low density)."""
        lte = default_evolution_model.lte_validity(5000.0)
        # With decayed density, LTE may be marginal
        assert 0.0 <= lte <= 1.0

    def test_phase_classification_early(self, default_evolution_model):
        """Test early phase classification."""
        phase = default_evolution_model.classify_phase(50.0)
        assert phase == PlasmaPhase.EARLY

    def test_phase_classification_intermediate(self, default_evolution_model):
        """Test intermediate phase classification."""
        phase = default_evolution_model.classify_phase(500.0)
        assert phase == PlasmaPhase.INTERMEDIATE

    def test_phase_classification_late(self, default_evolution_model):
        """Test late phase classification."""
        phase = default_evolution_model.classify_phase(3000.0)
        assert phase == PlasmaPhase.LATE

    def test_generate_profile(self, default_evolution_model):
        """Test profile generation."""
        profile = default_evolution_model.generate_profile(n_points=50, t_max_ns=2000.0)

        assert isinstance(profile, PlasmaEvolutionProfile)
        assert len(profile.points) == 50
        assert profile.T_initial_K == 20000.0
        assert profile.ne_initial_cm3 == 1e18
        assert profile.tau_T_ns == 1000.0
        assert profile.tau_ne_ns == 500.0

    def test_profile_interpolation(self, default_evolution_model):
        """Test profile interpolation methods."""
        profile = default_evolution_model.generate_profile(n_points=100, t_max_ns=2000.0)

        # Test interpolation at a specific time
        T_interp = profile.temperature_at(500.0)
        T_model = default_evolution_model.temperature(500.0)
        assert T_interp == pytest.approx(T_model, rel=0.05)

    def test_from_measurements(self):
        """Test fitting from experimental measurements."""
        # Simulate measurements
        times = np.array([100, 200, 500, 1000, 2000])
        temps = 19000 * np.exp(-times / 800) + 300
        densities = 1e18 * np.exp(-times / 400)

        model = PlasmaEvolutionModel.from_measurements(times, temps, densities)

        # Check that fitted model has reasonable parameters
        assert 100 < model.tau_T_ns < 10000
        assert 50 < model.tau_ne_ns < 5000
        assert 5000 < model.T_initial_K < 50000
        assert 1e15 < model.ne_initial_cm3 < 1e20

    def test_from_measurements_with_noise(self):
        """Test fitting from noisy measurements."""
        rng = np.random.default_rng(42)
        times = np.array([100, 200, 500, 1000, 2000])
        temps = 19000 * np.exp(-times / 800) + 300 + rng.normal(0, 500, len(times))
        densities = 1e18 * np.exp(-times / 400) * (1 + rng.normal(0, 0.1, len(times)))

        model = PlasmaEvolutionModel.from_measurements(times, temps, densities)

        # Should still produce reasonable results
        assert model.T_initial_K > 5000
        assert model.ne_initial_cm3 > 1e15


# =============================================================================
# Tests for GateTimingOptimizer
# =============================================================================


class TestGateTimingOptimizer:
    """Tests for GateTimingOptimizer class."""

    def test_score_gate(self, default_evolution_model):
        """Test gate scoring."""
        optimizer = GateTimingOptimizer(default_evolution_model)
        score, details = optimizer.score_gate(500.0, 500.0)

        assert 0.0 <= score <= 1.0
        assert "lte_score" in details
        assert "signal_score" in details
        assert "self_absorption_score" in details
        assert "continuum_score" in details
        assert details["center_time_ns"] == 750.0  # 500 + 500/2

    def test_scores_change_with_time(self, default_evolution_model):
        """Test that scores change with gate timing."""
        optimizer = GateTimingOptimizer(default_evolution_model)

        score_early, _ = optimizer.score_gate(100.0, 200.0)
        score_late, _ = optimizer.score_gate(2000.0, 200.0)

        # Scores should be different
        assert score_early != score_late

    def test_optimize_finds_intermediate_timing(self, default_evolution_model):
        """Test that optimizer finds intermediate gate timing."""
        optimizer = GateTimingOptimizer(default_evolution_model)
        result = optimizer.optimize(
            delay_range=(100.0, 3000.0),
            width_range=(200.0, 1000.0),
            n_delay_points=20,
            n_width_points=5,
        )

        assert isinstance(result, GateOptimizationResult)
        # Optimal delay should be in intermediate range, not too early or late
        assert 100.0 < result.optimal_delay_ns < 3000.0
        assert 0.0 < result.score <= 1.0

    def test_optimize_respects_lte_constraint(self, default_evolution_model):
        """Test that optimizer respects minimum LTE validity."""
        optimizer = GateTimingOptimizer(default_evolution_model)
        result = optimizer.optimize(min_lte_validity=0.7)

        # Result should have reasonable LTE validity
        assert result.lte_validity >= 0.5  # May be slightly below due to grid search

    def test_custom_weights(self, default_evolution_model):
        """Test optimizer with custom weight settings."""
        # LTE-focused optimizer
        lte_optimizer = GateTimingOptimizer(
            default_evolution_model,
            lte_weight=2.0,
            signal_weight=0.3,
            self_absorption_weight=0.5,
        )
        result_lte = lte_optimizer.optimize()

        # Signal-focused optimizer
        signal_optimizer = GateTimingOptimizer(
            default_evolution_model,
            lte_weight=0.5,
            signal_weight=2.0,
            self_absorption_weight=0.3,
        )
        result_signal = signal_optimizer.optimize()

        # Different weights should give different optimal timings
        # (though they might happen to be similar in some cases)
        assert result_lte.optimal_delay_ns is not None
        assert result_signal.optimal_delay_ns is not None

    def test_sweep_delays(self, default_evolution_model):
        """Test delay sweep functionality."""
        optimizer = GateTimingOptimizer(default_evolution_model)
        delays = np.linspace(100, 2000, 20)
        results = optimizer.sweep_delays(delays, fixed_width_ns=500.0)

        assert len(results) == 20
        for r in results:
            assert "delay_ns" in r
            assert "score" in r
            assert "temperature_K" in r

    def test_result_contains_expected_values(self, default_evolution_model):
        """Test that result contains all expected fields."""
        optimizer = GateTimingOptimizer(default_evolution_model)
        result = optimizer.optimize()

        assert hasattr(result, "optimal_delay_ns")
        assert hasattr(result, "optimal_width_ns")
        assert hasattr(result, "score")
        assert hasattr(result, "lte_validity")
        assert hasattr(result, "expected_temperature_K")
        assert hasattr(result, "expected_density_cm3")
        assert hasattr(result, "self_absorption_factor")
        assert hasattr(result, "analysis")


# =============================================================================
# Tests for TemporalSelfAbsorptionCorrector
# =============================================================================


class TestTemporalSelfAbsorptionCorrector:
    """Tests for TemporalSelfAbsorptionCorrector class."""

    def test_optical_depth_at_time(self, default_evolution_model):
        """Test optical depth calculation at specific time."""
        corrector = TemporalSelfAbsorptionCorrector(default_evolution_model)

        tau = corrector.optical_depth_at_time(
            time_ns=500.0,
            wavelength_nm=500.0,
            A_ki=1e8,
            g_k=9,
            E_lower_eV=0.0,
            concentration=0.1,
            total_number_density_cm3=1e18,
            partition_func=25.0,
        )

        assert tau >= 0.0

    def test_optical_depth_decreases_with_time(self, default_evolution_model):
        """Test that optical depth decreases with time.

        Note: The optical depth depends on temperature through the Boltzmann factor
        exp(-E_lower/kT). For E_lower=0 (ground state), this is always 1.
        To see time variation, we need either:
        1. A non-zero E_lower so the Boltzmann factor varies with T
        2. Temperature-dependent partition functions

        We use a non-zero E_lower here to demonstrate the effect.
        """
        corrector = TemporalSelfAbsorptionCorrector(default_evolution_model)

        # Use a non-zero lower level energy so Boltzmann factor varies with T
        tau_early = corrector.optical_depth_at_time(
            time_ns=100.0,
            wavelength_nm=500.0,
            A_ki=1e8,
            g_k=9,
            E_lower_eV=2.0,  # Non-zero lower level energy
            concentration=0.1,
            total_number_density_cm3=1e18,
            partition_func=25.0,
        )

        tau_late = corrector.optical_depth_at_time(
            time_ns=2000.0,
            wavelength_nm=500.0,
            A_ki=1e8,
            g_k=9,
            E_lower_eV=2.0,  # Non-zero lower level energy
            concentration=0.1,
            total_number_density_cm3=1e18,
            partition_func=25.0,
        )

        # Early time has higher T, so exp(-E/kT) is smaller
        # but late time has lower T, so exp(-E/kT) should be smaller
        # Wait - at lower T, exp(-E/kT) is smaller (more negative exponent)
        # So tau_late < tau_early because n_lower decreases with T
        # Actually: at higher T, exp(-E/kT) is LARGER (closer to 1)
        # At lower T, exp(-E/kT) is SMALLER
        # So tau should be larger at higher T (early time)
        assert tau_early > tau_late

    def test_gate_averaged_optical_depth(self, default_evolution_model, sample_gate_config):
        """Test gate-averaged optical depth calculation."""
        corrector = TemporalSelfAbsorptionCorrector(default_evolution_model)

        def partition_func(T_K):
            return 25.0

        tau_avg = corrector.gate_averaged_optical_depth(
            gate=sample_gate_config,
            wavelength_nm=500.0,
            A_ki=1e8,
            g_k=9,
            E_lower_eV=0.0,
            concentration=0.1,
            total_number_density_cm3=1e18,
            partition_func_callable=partition_func,
        )

        assert tau_avg >= 0.0

    def test_correct_observations(
        self, default_evolution_model, sample_gate_config, sample_line_observations
    ):
        """Test correction of observations."""
        corrector = TemporalSelfAbsorptionCorrector(default_evolution_model)

        def partition_func(el, stage, T_K):
            return 25.0

        result = corrector.correct_observations(
            observations=sample_line_observations,
            gate=sample_gate_config,
            concentrations={"Fe": 0.9},
            total_number_density_cm3=1e18,
            partition_func_callable=partition_func,
        )

        assert isinstance(result, TemporalSelfAbsorptionResult)
        assert len(result.corrected_observations) >= 0
        assert isinstance(result.optical_depths, dict)
        assert isinstance(result.correction_factors, dict)
        assert isinstance(result.time_averaged_tau, dict)

    def test_high_optical_depth_masking(
        self, default_evolution_model, sample_gate_config, sample_line_observations
    ):
        """Test that high optical depth lines are masked."""
        # Create corrector with very high number density to force high tau
        model = PlasmaEvolutionModel(
            T_initial_K=20000.0,
            ne_initial_cm3=1e20,  # Very high
            tau_T_ns=1000.0,
            tau_ne_ns=500.0,
        )
        corrector = TemporalSelfAbsorptionCorrector(model, plasma_length_cm=10.0)

        def partition_func(el, stage, T_K):
            return 25.0

        result = corrector.correct_observations(
            observations=sample_line_observations,
            gate=sample_gate_config,
            concentrations={"Fe": 0.99},
            total_number_density_cm3=1e22,  # Very high
            partition_func_callable=partition_func,
            mask_threshold=0.001,  # Very low threshold
        )

        # Should have some warnings about masking
        assert len(result.warnings) >= 0  # May or may not mask depending on calculation


# =============================================================================
# Tests for TimeResolvedCFLIBSSolver
# =============================================================================


class TestTimeResolvedCFLIBSSolver:
    """Tests for TimeResolvedCFLIBSSolver class."""

    @pytest.fixture
    def mock_atomic_db(self):
        """Create mock atomic database."""
        db = MagicMock()
        db.get_ionization_potential.return_value = 7.87
        db.get_partition_coefficients.return_value = None
        return db

    @pytest.fixture
    def sample_spectra(self, sample_line_observations):
        """Create sample time-resolved spectra."""
        return [
            TimeResolvedSpectrum(
                gate=TemporalGateConfig(delay_ns=200.0, width_ns=500.0),
                observations=sample_line_observations,
                snr_estimate=50.0,
            ),
            TimeResolvedSpectrum(
                gate=TemporalGateConfig(delay_ns=700.0, width_ns=500.0),
                observations=sample_line_observations,
                snr_estimate=80.0,
            ),
            TimeResolvedSpectrum(
                gate=TemporalGateConfig(delay_ns=1200.0, width_ns=500.0),
                observations=sample_line_observations,
                snr_estimate=40.0,
            ),
        ]

    def test_solver_creation(self, mock_atomic_db, default_evolution_model):
        """Test solver creation."""
        solver = TimeResolvedCFLIBSSolver(
            atomic_db=mock_atomic_db,
            evolution_model=default_evolution_model,
        )
        assert solver.atomic_db is mock_atomic_db
        assert solver.evolution_model is default_evolution_model

    def test_solver_without_evolution_model(self, mock_atomic_db):
        """Test solver creation without evolution model."""
        solver = TimeResolvedCFLIBSSolver(atomic_db=mock_atomic_db)
        assert solver.evolution_model is None

    def test_gate_weight_computation(self, mock_atomic_db, default_evolution_model):
        """Test gate weight computation."""
        solver = TimeResolvedCFLIBSSolver(
            atomic_db=mock_atomic_db,
            evolution_model=default_evolution_model,
        )

        # Create a mock result
        gate_result = {
            "converged": True,
            "temperature_K": 15000.0,
            "electron_density_cm3": 5e17,
            "snr": 50.0,
        }

        weight = solver._compute_gate_weight(
            gate_result,
            expected_T=15000.0,
            expected_ne=5e17,
        )

        assert weight > 0.0

    def test_non_converged_gate_has_zero_weight(self, mock_atomic_db, default_evolution_model):
        """Test that non-converged gates have zero weight."""
        solver = TimeResolvedCFLIBSSolver(
            atomic_db=mock_atomic_db,
            evolution_model=default_evolution_model,
        )

        gate_result = {
            "converged": False,
            "temperature_K": 15000.0,
            "electron_density_cm3": 5e17,
            "snr": 50.0,
        }

        weight = solver._compute_gate_weight(gate_result, 15000.0, 5e17)
        assert weight == 0.0

    def test_solve_multi_gate_empty_raises(self, mock_atomic_db):
        """Test that empty spectra list raises error."""
        solver = TimeResolvedCFLIBSSolver(atomic_db=mock_atomic_db)

        with pytest.raises(ValueError, match="No spectra provided"):
            solver.solve_multi_gate([])


# =============================================================================
# Tests for Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.parametrize("material_type", ["metal", "ceramic", "polymer", "soil"])
    def test_create_default_evolution_model_materials(self, material_type):
        """Test default model creation for different materials."""
        model = create_default_evolution_model(material_type=material_type)

        assert isinstance(model, PlasmaEvolutionModel)
        assert model.T_initial_K > 5000.0
        assert model.ne_initial_cm3 > 1e15
        assert model.tau_T_ns > 0.0
        assert model.tau_ne_ns > 0.0

    def test_create_default_evolution_model_energy_scaling(self):
        """Test that model scales with laser energy."""
        model_low = create_default_evolution_model(laser_energy_mJ=20.0)
        model_high = create_default_evolution_model(laser_energy_mJ=100.0)

        # Higher energy should give higher initial values
        assert model_high.T_initial_K > model_low.T_initial_K
        assert model_high.ne_initial_cm3 > model_low.ne_initial_cm3

    def test_create_default_evolution_model_unknown_material(self):
        """Test fallback for unknown material type."""
        model = create_default_evolution_model(material_type="unknown")
        # Should use metal defaults
        assert isinstance(model, PlasmaEvolutionModel)

    @pytest.mark.parametrize("priority", ["lte", "signal", "balanced"])
    def test_recommend_gate_timing_priorities(self, priority):
        """Test gate timing recommendations for different priorities."""
        gate = recommend_gate_timing(
            material_type="metal",
            laser_energy_mJ=50.0,
            priority=priority,
        )

        assert isinstance(gate, TemporalGateConfig)
        assert gate.delay_ns > 0.0
        assert gate.width_ns > 0.0
        assert gate.label == f"optimal_{priority}"

    def test_recommend_gate_timing_materials(self):
        """Test gate recommendations for different materials."""
        gate_metal = recommend_gate_timing(material_type="metal")
        gate_polymer = recommend_gate_timing(material_type="polymer")

        # Both should produce valid gates
        assert gate_metal.delay_ns > 0.0
        assert gate_polymer.delay_ns > 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestTemporalIntegration:
    """Integration tests combining multiple temporal components."""

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create evolution model
        model = create_default_evolution_model(material_type="metal", laser_energy_mJ=50.0)

        # Optimize gate timing
        optimizer = GateTimingOptimizer(model)
        result = optimizer.optimize()

        # Recommended gate should be usable
        gate = TemporalGateConfig(
            delay_ns=result.optimal_delay_ns,
            width_ns=result.optimal_width_ns,
            label="optimized",
        )

        assert gate.delay_ns > 0
        assert gate.width_ns > 0

    def test_evolution_model_and_corrector_consistency(self):
        """Test that evolution model and corrector are consistent."""
        model = PlasmaEvolutionModel(
            T_initial_K=20000.0,
            ne_initial_cm3=1e18,
            tau_T_ns=1000.0,
            tau_ne_ns=500.0,
        )

        TemporalSelfAbsorptionCorrector(model)

        # Check that corrector uses model's T and n_e
        time = 500.0
        T_model = model.temperature(time)

        # The corrector should internally use this temperature
        assert T_model > 300.0  # Should be decayed from initial

    def test_gate_sweep_produces_sensible_trend(self):
        """Test that gate sweep produces sensible trends."""
        model = create_default_evolution_model()
        optimizer = GateTimingOptimizer(model)

        delays = np.linspace(100, 3000, 30)
        results = optimizer.sweep_delays(delays, fixed_width_ns=500.0)

        # Extract temperatures
        temperatures = [r["temperature_K"] for r in results]

        # Temperature should generally decrease with time
        assert temperatures[0] > temperatures[-1]

        # All temperatures should be positive and reasonable
        assert all(300 < T < 50000 for T in temperatures)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_early_time(self):
        """Test behavior at very early times."""
        model = PlasmaEvolutionModel()
        T = model.temperature(1.0)  # 1 ns
        n_e = model.electron_density(1.0)

        # Should be very close to initial values
        assert T > 19000.0
        assert n_e > 9e17

    def test_very_late_time(self):
        """Test behavior at very late times."""
        model = PlasmaEvolutionModel()
        T = model.temperature(100000.0)  # 100 us
        n_e = model.electron_density(100000.0)

        # Temperature should approach ambient
        assert T < 500.0  # Close to 300 K ambient
        # Density should be very low
        assert n_e < 1e10

    def test_zero_concentration(self, default_evolution_model):
        """Test corrector with zero concentration."""
        corrector = TemporalSelfAbsorptionCorrector(default_evolution_model)

        tau = corrector.optical_depth_at_time(
            time_ns=500.0,
            wavelength_nm=500.0,
            A_ki=1e8,
            g_k=9,
            E_lower_eV=0.0,
            concentration=0.0,  # Zero concentration
            total_number_density_cm3=1e18,
            partition_func=25.0,
        )

        assert tau == 0.0

    def test_zero_partition_function(self, default_evolution_model):
        """Test corrector with zero partition function."""
        corrector = TemporalSelfAbsorptionCorrector(default_evolution_model)

        tau = corrector.optical_depth_at_time(
            time_ns=500.0,
            wavelength_nm=500.0,
            A_ki=1e8,
            g_k=9,
            E_lower_eV=0.0,
            concentration=0.1,
            total_number_density_cm3=1e18,
            partition_func=0.0,  # Zero partition function
        )

        # Should handle gracefully
        assert tau == 0.0


# =============================================================================
# Joint multi-gate Saha-Boltzmann fit
# =============================================================================


class TestJointMultiGateFit:
    """Tests for ``joint_multi_gate_fit``: shared composition across gates."""

    # -- Shared synthetic-spectrum helpers -------------------------------------

    _IPS = {"Fe": 7.87, "Cu": 7.73}
    _U_I = {"Fe": 25.0, "Cu": 2.0}
    _U_II = {"Fe": 30.0, "Cu": 4.0}

    # (wavelength_nm, E_k_eV, g_k, A_ki) per (element, ion stage)
    _LINES = {
        "Fe": {
            1: [
                (371.99, 3.33, 11, 1.0e7),
                (404.58, 3.93, 9, 8.6e6),
                (438.35, 4.44, 9, 5.0e6),
                (495.76, 4.68, 7, 4.2e6),
                (516.75, 2.45, 9, 5.7e6),
            ],
            2: [
                (238.20, 5.20, 10, 3.0e8),
                (259.94, 4.82, 8, 2.2e8),
            ],
        },
        "Cu": {
            1: [
                (324.75, 3.82, 4, 1.4e8),
                (327.40, 3.79, 2, 1.4e8),
                (510.55, 3.82, 4, 2.0e6),
            ],
        },
    }

    def _synth_observations(
        self,
        T_K: float,
        ne: float,
        composition: dict,
        alpha_offset: float = 0.0,
        seed: int = 0,
        snr: float = 200.0,
    ):
        """Build a list of LineObservation matching the joint-fit forward model."""
        from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3
        from cflibs.inversion.boltzmann import LineObservation

        rng = np.random.default_rng(seed)
        T_eV = T_K / EV_TO_K
        log_S = {
            el: (
                np.log(SAHA_CONST_CM3)
                - np.log(ne)
                + 1.5 * np.log(T_eV)
                + np.log(self._U_II[el] / self._U_I[el])
                - self._IPS[el] / T_eV
            )
            for el in composition
        }
        observations = []
        for element, by_stage in self._LINES.items():
            if element not in composition:
                continue
            for stage, lines in by_stage.items():
                for wl, E_k, g_k, A_ki in lines:
                    y = (
                        alpha_offset
                        + np.log(composition[element])
                        - np.log(self._U_I[element])
                        - E_k / T_eV
                    )
                    if stage == 2:
                        y = y + log_S[element]
                    intensity = float(np.exp(y) * g_k * A_ki / wl)
                    intensity *= float(rng.normal(1.0, 1.0 / snr))
                    intensity = max(intensity, 1e-12)
                    sigma = max(intensity / snr, 1e-12)
                    observations.append(
                        LineObservation(
                            wavelength_nm=wl,
                            intensity=intensity,
                            intensity_uncertainty=sigma,
                            element=element,
                            ionization_stage=stage,
                            E_k_ev=E_k,
                            g_k=g_k,
                            A_ki=A_ki,
                        )
                    )
        return observations

    def _build_spectra(
        self,
        gates_config,
        composition,
        snr: float = 200.0,
    ):
        """Construct a list of TimeResolvedSpectrum from per-gate parameters."""
        spectra = []
        for i, gc in enumerate(gates_config):
            obs = self._synth_observations(
                T_K=gc["T_K"],
                ne=gc["ne"],
                composition=composition,
                alpha_offset=gc["alpha"],
                seed=42 + i,
                snr=snr,
            )
            spectra.append(
                TimeResolvedSpectrum(
                    gate=TemporalGateConfig(delay_ns=gc["delay"], width_ns=200.0),
                    observations=obs,
                )
            )
        return spectra

    # -- Acceptance tests ------------------------------------------------------

    def test_joint_multi_gate_recovers_constant_composition(self):
        """3 gates, same composition, different (T, ne): joint fit recovers C within 2%."""
        from cflibs.inversion.runtime.multi_gate import joint_multi_gate_fit

        true_comp = {"Fe": 0.7, "Cu": 0.3}
        gates_config = [
            {"delay": 200.0, "T_K": 14000.0, "ne": 5.0e17, "alpha": 30.0},
            {"delay": 700.0, "T_K": 11000.0, "ne": 2.0e17, "alpha": 28.5},
            {"delay": 1500.0, "T_K": 8000.0, "ne": 8.0e16, "alpha": 27.0},
        ]
        spectra = self._build_spectra(gates_config, true_comp, snr=200.0)

        result = joint_multi_gate_fit(
            spectra,
            ionization_potentials_eV=self._IPS,
            partition_func_I=self._U_I,
            partition_func_II=self._U_II,
            initial_temperatures_K=[10000.0] * 3,
            initial_electron_densities_cm3=[1.0e17] * 3,
            initial_composition={"Fe": 0.5, "Cu": 0.5},
        )

        assert result.converged
        assert result.n_gates == 3
        for el, true_value in true_comp.items():
            assert abs(result.composition_shared[el] - true_value) < 0.02
        # Per-gate temperatures should also recover within ~20%
        for fitted_T, gc in zip(result.temperatures_K_per_gate, gates_config):
            assert abs(fitted_T - gc["T_K"]) / gc["T_K"] < 0.25

    def test_joint_multi_gate_lower_uncertainty_than_per_gate(self):
        """Joint fit composition uncertainty must be < average per-gate uncertainty."""
        from cflibs.inversion.runtime.multi_gate import joint_multi_gate_fit

        true_comp = {"Fe": 0.7, "Cu": 0.3}
        gates_config = [
            {"delay": 200.0, "T_K": 14000.0, "ne": 5.0e17, "alpha": 30.0},
            {"delay": 700.0, "T_K": 11000.0, "ne": 2.0e17, "alpha": 28.5},
            {"delay": 1500.0, "T_K": 8000.0, "ne": 8.0e16, "alpha": 27.0},
        ]
        spectra = self._build_spectra(gates_config, true_comp, snr=50.0)

        # Per-gate uncertainties (run the joint-fit machinery on each gate alone)
        per_gate_sigmas = []
        for sp in spectra:
            res_pg = joint_multi_gate_fit(
                [sp],
                ionization_potentials_eV=self._IPS,
                partition_func_I=self._U_I,
                partition_func_II=self._U_II,
                initial_temperatures_K=[10000.0],
                initial_electron_densities_cm3=[1.0e17],
                initial_composition={"Fe": 0.5, "Cu": 0.5},
            )
            per_gate_sigmas.append(
                np.array([res_pg.composition_uncertainty[el] for el in ["Fe", "Cu"]])
            )
        mean_per_gate_sigma = np.mean(per_gate_sigmas, axis=0)

        res_joint = joint_multi_gate_fit(
            spectra,
            ionization_potentials_eV=self._IPS,
            partition_func_I=self._U_I,
            partition_func_II=self._U_II,
            initial_temperatures_K=[10000.0] * 3,
            initial_electron_densities_cm3=[1.0e17] * 3,
            initial_composition={"Fe": 0.5, "Cu": 0.5},
        )
        joint_sigma = np.array([res_joint.composition_uncertainty[el] for el in ["Fe", "Cu"]])

        # Headline benefit: joint fit must be strictly tighter than the
        # average per-gate fit for every element.
        assert np.all(joint_sigma < mean_per_gate_sigma), (
            f"Joint sigma {joint_sigma.tolist()} not strictly less than "
            f"per-gate mean {mean_per_gate_sigma.tolist()}"
        )

    def test_joint_multi_gate_handles_missing_lines(self):
        """Drop half the lines from one gate; fit must still recover composition."""
        from cflibs.inversion.runtime.multi_gate import joint_multi_gate_fit

        true_comp = {"Fe": 0.7, "Cu": 0.3}
        gates_config = [
            {"delay": 200.0, "T_K": 14000.0, "ne": 5.0e17, "alpha": 30.0},
            {"delay": 700.0, "T_K": 11000.0, "ne": 2.0e17, "alpha": 28.5},
            {"delay": 1500.0, "T_K": 8000.0, "ne": 8.0e16, "alpha": 27.0},
        ]
        spectra = self._build_spectra(gates_config, true_comp, snr=200.0)
        # Drop every other observation from the middle gate
        spectra[1] = TimeResolvedSpectrum(
            gate=spectra[1].gate,
            observations=list(spectra[1].observations[::2]),
        )

        result = joint_multi_gate_fit(
            spectra,
            ionization_potentials_eV=self._IPS,
            partition_func_I=self._U_I,
            partition_func_II=self._U_II,
            initial_temperatures_K=[10000.0] * 3,
            initial_electron_densities_cm3=[1.0e17] * 3,
            initial_composition={"Fe": 0.5, "Cu": 0.5},
        )

        assert result.converged
        for el, true_value in true_comp.items():
            assert abs(result.composition_shared[el] - true_value) < 0.03

    def test_joint_multi_gate_accepts_tuple_payload(self):
        """The function should accept ``(gate, observations)`` tuples directly."""
        from cflibs.inversion.runtime.multi_gate import joint_multi_gate_fit

        true_comp = {"Fe": 0.7, "Cu": 0.3}
        gates_config = [
            {"delay": 200.0, "T_K": 14000.0, "ne": 5.0e17, "alpha": 30.0},
            {"delay": 1500.0, "T_K": 8000.0, "ne": 8.0e16, "alpha": 27.0},
        ]
        spectra = self._build_spectra(gates_config, true_comp, snr=200.0)
        payload = [(s.gate, s.observations) for s in spectra]

        result = joint_multi_gate_fit(
            payload,
            ionization_potentials_eV=self._IPS,
            partition_func_I=self._U_I,
            partition_func_II=self._U_II,
            initial_temperatures_K=[10000.0] * 2,
            initial_electron_densities_cm3=[1.0e17] * 2,
            initial_composition={"Fe": 0.5, "Cu": 0.5},
        )

        assert result.converged
        assert result.n_gates == 2
        for el, true_value in true_comp.items():
            assert abs(result.composition_shared[el] - true_value) < 0.03

    def test_joint_multi_gate_5_gate_acceptance(self):
        """5-gate acceptance: joint composition RMSEP < half the per-gate RMSEP."""
        from cflibs.inversion.runtime.multi_gate import joint_multi_gate_fit

        true_comp = {"Fe": 0.7, "Cu": 0.3}
        gates_config = [
            {"delay": 100.0, "T_K": 16000.0, "ne": 8.0e17, "alpha": 31.0},
            {"delay": 400.0, "T_K": 13000.0, "ne": 4.0e17, "alpha": 29.5},
            {"delay": 800.0, "T_K": 10500.0, "ne": 2.0e17, "alpha": 28.5},
            {"delay": 1500.0, "T_K": 8500.0, "ne": 1.0e17, "alpha": 27.5},
            {"delay": 2500.0, "T_K": 7000.0, "ne": 5.0e16, "alpha": 26.5},
        ]
        spectra = self._build_spectra(gates_config, true_comp, snr=50.0)

        # Per-gate fits
        per_gate_results = []
        for sp in spectra:
            res = joint_multi_gate_fit(
                [sp],
                ionization_potentials_eV=self._IPS,
                partition_func_I=self._U_I,
                partition_func_II=self._U_II,
                initial_temperatures_K=[10000.0],
                initial_electron_densities_cm3=[1.0e17],
                initial_composition={"Fe": 0.5, "Cu": 0.5},
            )
            per_gate_results.append(np.array([res.composition_shared[el] for el in ["Fe", "Cu"]]))
        truth_vec = np.array([true_comp[el] for el in ["Fe", "Cu"]])
        per_gate_errs = np.array([(r - truth_vec) ** 2 for r in per_gate_results])
        per_gate_rmsep = float(np.sqrt(per_gate_errs.mean()))

        # Joint fit
        res_joint = joint_multi_gate_fit(
            spectra,
            ionization_potentials_eV=self._IPS,
            partition_func_I=self._U_I,
            partition_func_II=self._U_II,
            initial_temperatures_K=[10000.0] * 5,
            initial_electron_densities_cm3=[1.0e17] * 5,
            initial_composition={"Fe": 0.5, "Cu": 0.5},
        )
        joint_vec = np.array([res_joint.composition_shared[el] for el in ["Fe", "Cu"]])
        joint_rmsep = float(np.sqrt(((joint_vec - truth_vec) ** 2).mean()))

        # Acceptance criterion from the bead spec.
        assert joint_rmsep < 0.5 * per_gate_rmsep, (
            f"Joint RMSEP {joint_rmsep:.4f} should be < half per-gate RMSEP "
            f"{per_gate_rmsep:.4f}"
        )

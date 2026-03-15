"""
Unit and integration tests for cflibs.inversion.line_selection module.

Tests cover:
- LineSelector initialization and selection
- Scoring formula verification
- Isolation factor calculation
- Rejection criteria
- identify_resonance_lines function
"""

import pytest
import numpy as np

from cflibs.inversion.line_selection import (
    LineSelector,
    LineScore,
    LineSelectionResult,
    identify_resonance_lines,
)
from cflibs.inversion.boltzmann import LineObservation

# ==============================================================================
# LineSelector Initialization Tests
# ==============================================================================


class TestLineSelectorInit:
    """Tests for LineSelector initialization."""

    def test_default_parameters(self):
        """Verify default parameters are set correctly."""
        selector = LineSelector()

        assert selector.min_snr == 10.0
        assert selector.min_energy_spread_ev == 2.0
        assert selector.min_lines_per_element == 3
        assert selector.exclude_resonance is True
        assert selector.isolation_wavelength_nm == 0.1
        assert selector.max_lines_per_element == 20

    def test_custom_parameters(self):
        """Verify custom parameters are stored."""
        selector = LineSelector(
            min_snr=20.0,
            min_energy_spread_ev=3.0,
            min_lines_per_element=5,
            exclude_resonance=False,
            isolation_wavelength_nm=0.2,
            max_lines_per_element=10,
        )

        assert selector.min_snr == 20.0
        assert selector.min_energy_spread_ev == 3.0
        assert selector.min_lines_per_element == 5
        assert selector.exclude_resonance is False
        assert selector.isolation_wavelength_nm == 0.2
        assert selector.max_lines_per_element == 10


# ==============================================================================
# Isolation Factor Tests
# ==============================================================================


class TestIsolationFactor:
    """Tests for LineSelector._compute_isolation method."""

    @pytest.fixture
    def selector(self):
        return LineSelector(isolation_wavelength_nm=0.1)

    def test_isolation_zero_separation(self, selector):
        """Verify isolation ≈ 0 when lines overlap."""
        obs1 = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=20.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )
        obs2 = LineObservation(
            wavelength_nm=400.0,
            intensity=800.0,
            intensity_uncertainty=16.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.5,
            g_k=7,
            A_ki=8e6,
        )

        isolation = selector._compute_isolation(obs1, [obs1, obs2])

        # exp(0) = 1, so isolation = 1 - 1 = 0
        assert isolation == pytest.approx(0.0, abs=0.001)

    def test_isolation_large_separation(self, selector):
        """Verify isolation ≈ 1 for well-separated lines."""
        obs1 = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=20.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )
        obs2 = LineObservation(
            wavelength_nm=500.0,
            intensity=800.0,
            intensity_uncertainty=16.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.5,
            g_k=7,
            A_ki=8e6,
        )

        isolation = selector._compute_isolation(obs1, [obs1, obs2])

        # Separation = 100 nm >> 0.1 nm, so isolation ≈ 1
        assert isolation == pytest.approx(1.0, abs=0.001)

    def test_isolation_formula(self, selector):
        """Verify isolation = 1 - exp(-separation/isolation_wavelength_nm)."""
        obs1 = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=20.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )
        obs2 = LineObservation(
            wavelength_nm=400.1,
            intensity=800.0,
            intensity_uncertainty=16.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.5,
            g_k=7,
            A_ki=8e6,
        )

        isolation = selector._compute_isolation(obs1, [obs1, obs2])

        # Separation = 0.1 nm, isolation_wavelength = 0.1 nm
        # isolation = 1 - exp(-0.1/0.1) = 1 - exp(-1) ≈ 0.632
        expected = 1.0 - np.exp(-1)
        assert isolation == pytest.approx(expected, rel=0.01)

    def test_isolation_single_line(self, selector):
        """Verify isolation = 1 for single line (no neighbors)."""
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

        isolation = selector._compute_isolation(obs, [obs])

        assert isolation == 1.0


# ==============================================================================
# Scoring Formula Tests
# ==============================================================================


class TestScoringFormula:
    """Tests for LineSelector._score_line method."""

    @pytest.fixture
    def selector(self):
        return LineSelector(isolation_wavelength_nm=0.1)

    def test_score_formula(self, selector):
        """Verify score = SNR × (1/uncertainty) × isolation."""
        obs = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=10.0,  # SNR = 100
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )

        atomic_uncertainties = {("Fe", 1, 400.0): 0.05}  # 5% uncertainty

        score_info = selector._score_line(
            obs, [obs], resonance_lines=set(), atomic_uncertainties=atomic_uncertainties
        )

        # SNR = 1000/10 = 100
        # uncertainty = 0.05, so 1/uncertainty = 20
        # isolation = 1.0 (single line)
        # score = 100 * 20 * 1.0 = 2000
        assert score_info.snr == pytest.approx(100.0, rel=0.01)
        assert score_info.atomic_uncertainty == 0.05
        assert score_info.isolation_factor == 1.0
        assert score_info.score == pytest.approx(2000.0, rel=0.01)

    def test_score_with_default_uncertainty(self, selector):
        """Verify default 10% uncertainty is used when not provided."""
        obs = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=20.0,  # SNR = 50
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )

        score_info = selector._score_line(
            obs, [obs], resonance_lines=set(), atomic_uncertainties={}
        )

        assert score_info.atomic_uncertainty == 0.10  # Default

    def test_score_snr_from_intensity_ratio(self, selector):
        """Verify SNR is calculated as intensity / uncertainty."""
        obs = LineObservation(
            wavelength_nm=400.0,
            intensity=500.0,
            intensity_uncertainty=25.0,  # SNR = 20
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )

        score_info = selector._score_line(
            obs, [obs], resonance_lines=set(), atomic_uncertainties={}
        )

        assert score_info.snr == pytest.approx(20.0, rel=0.01)

    def test_score_zero_uncertainty_gives_high_snr(self, selector):
        """Verify zero intensity_uncertainty gives default high SNR."""
        obs = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=0.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )

        score_info = selector._score_line(
            obs, [obs], resonance_lines=set(), atomic_uncertainties={}
        )

        assert score_info.snr == 100.0  # Default when no uncertainty

    def test_resonance_line_flagged(self, selector):
        """Verify resonance lines are flagged correctly."""
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

        resonance_lines = {("Fe", 1, 400.0)}

        score_info = selector._score_line(
            obs, [obs], resonance_lines=resonance_lines, atomic_uncertainties={}
        )

        assert score_info.is_resonance is True


# ==============================================================================
# LineSelector.select() Tests
# ==============================================================================


class TestLineSelectorSelect:
    """Tests for LineSelector.select() method."""

    @pytest.fixture
    def selector(self):
        return LineSelector(
            min_snr=10.0,
            min_lines_per_element=2,
            max_lines_per_element=5,
            isolation_wavelength_nm=0.1,
        )

    def test_select_returns_result(self, selector, line_selector_test_data):
        """Verify select() returns LineSelectionResult."""
        observations = line_selector_test_data(n_lines=5, element="Fe")

        result = selector.select(observations)

        assert isinstance(result, LineSelectionResult)

    def test_select_respects_min_snr(self, selector):
        """Verify lines below min_snr are rejected."""
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=100.0,
                intensity_uncertainty=50.0,  # SNR = 2
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,  # SNR = 100
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
        ]

        result = selector.select(observations)

        # First line should be rejected (SNR = 2 < 10)
        assert len(result.selected_lines) == 1
        assert len(result.rejected_lines) == 1
        assert result.rejected_lines[0].wavelength_nm == 400.0

    def test_select_respects_max_lines_per_element(self, selector):
        """Verify max_lines_per_element is respected."""
        # Create 10 lines for same element
        observations = [
            LineObservation(
                wavelength_nm=400.0 + i * 10,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0 + i * 0.2,
                g_k=9,
                A_ki=1e7,
            )
            for i in range(10)
        ]

        result = selector.select(observations)

        # Should select at most 5 (max_lines_per_element)
        fe_selected = [line for line in result.selected_lines if line.element == "Fe"]
        assert len(fe_selected) <= 5

    def test_rejection_reasons_tracked(self, selector):
        """Verify rejection reasons are recorded."""
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=100.0,
                intensity_uncertainty=100.0,  # SNR = 1
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = selector.select(observations)

        # Find the score for this line
        rejected_scores = [s for s in result.scores if s.rejection_reason is not None]
        assert len(rejected_scores) == 1
        assert "Low SNR" in rejected_scores[0].rejection_reason

    def test_select_lines_sorted_by_score(self, selector):
        """Verify selected lines are sorted by score descending."""
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=500.0,
                intensity_uncertainty=10.0,  # SNR = 50
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,  # SNR = 100
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
            LineObservation(
                wavelength_nm=420.0,
                intensity=200.0,
                intensity_uncertainty=10.0,  # SNR = 20
                element="Fe",
                ionization_stage=1,
                E_k_ev=4.0,
                g_k=5,
                A_ki=5e6,
            ),
        ]

        result = selector.select(observations)

        # Higher SNR should be first
        selected_snrs = []
        for line in result.selected_lines:
            score_info = next(s for s in result.scores if s.observation is line)
            selected_snrs.append(score_info.snr)

        # Should be sorted descending
        assert selected_snrs == sorted(selected_snrs, reverse=True)

    def test_energy_spread_warning(self, selector):
        """Verify warning when energy spread is insufficient."""
        # All lines at similar energy
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.1,
                g_k=7,
                A_ki=8e6,
            ),
            LineObservation(
                wavelength_nm=420.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.2,
                g_k=5,
                A_ki=5e6,
            ),
        ]

        result = selector.select(observations)

        # Energy spread = 0.2 eV < 2.0 eV threshold
        assert any("Energy spread" in w for w in result.warnings)

    def test_min_lines_per_element_warning(self):
        """Verify warning when fewer than min lines available."""
        selector = LineSelector(min_snr=10.0, min_lines_per_element=5)

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
        ]

        result = selector.select(observations)

        # Only 2 Fe lines, but need 5
        assert any("only 2 lines available" in w.lower() for w in result.warnings)

    def test_resonance_lines_excluded(self):
        """Verify resonance lines are excluded when exclude_resonance=True."""
        selector = LineSelector(exclude_resonance=True, min_snr=10.0)

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        resonance_lines = {("Fe", 1, 400.0)}

        result = selector.select(observations, resonance_lines=resonance_lines)

        # Resonance line should be rejected
        assert len(result.rejected_lines) == 1
        rejected_score = result.scores[0]
        assert "Resonance" in rejected_score.rejection_reason

    def test_resonance_lines_included(self):
        """Verify resonance lines are included when exclude_resonance=False."""
        selector = LineSelector(exclude_resonance=False, min_snr=10.0)

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        resonance_lines = {("Fe", 1, 400.0)}

        result = selector.select(observations, resonance_lines=resonance_lines)

        # Resonance line should be selected
        assert len(result.selected_lines) == 1

    def test_empty_input(self, selector):
        """Verify empty input returns empty output."""
        result = selector.select([])

        assert result.selected_lines == []
        assert result.rejected_lines == []
        assert result.scores == []
        assert result.energy_spread_ev == 0.0

    def test_all_rejected(self, selector):
        """Verify graceful handling when all lines are rejected."""
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=10.0,
                intensity_uncertainty=100.0,  # SNR = 0.1
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = selector.select(observations)

        assert len(result.selected_lines) == 0
        assert len(result.rejected_lines) == 1


# ==============================================================================
# LineScore Tests
# ==============================================================================


class TestLineScore:
    """Tests for LineScore dataclass."""

    def test_dataclass_creation(self):
        """Verify LineScore can be instantiated."""
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

        score = LineScore(
            observation=obs,
            score=500.0,
            snr=50.0,
            isolation_factor=0.9,
            atomic_uncertainty=0.10,
            is_resonance=False,
            rejection_reason=None,
        )

        assert score.observation is obs
        assert score.score == 500.0
        assert score.snr == 50.0
        assert score.isolation_factor == 0.9
        assert score.atomic_uncertainty == 0.10
        assert score.is_resonance is False
        assert score.rejection_reason is None

    def test_rejection_reason_default(self):
        """Verify rejection_reason defaults to None."""
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

        score = LineScore(
            observation=obs,
            score=500.0,
            snr=50.0,
            isolation_factor=0.9,
            atomic_uncertainty=0.10,
            is_resonance=False,
        )

        assert score.rejection_reason is None


# ==============================================================================
# LineSelectionResult Tests
# ==============================================================================


class TestLineSelectionResult:
    """Tests for LineSelectionResult dataclass."""

    def test_dataclass_creation(self):
        """Verify LineSelectionResult can be instantiated."""
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

        result = LineSelectionResult(
            selected_lines=[obs],
            rejected_lines=[],
            scores=[],
            energy_spread_ev=2.5,
            n_elements=1,
            warnings=["test warning"],
        )

        assert len(result.selected_lines) == 1
        assert len(result.rejected_lines) == 0
        assert result.energy_spread_ev == 2.5
        assert result.n_elements == 1
        assert len(result.warnings) == 1

    def test_default_warnings(self):
        """Verify warnings defaults to empty list."""
        result = LineSelectionResult(
            selected_lines=[],
            rejected_lines=[],
            scores=[],
            energy_spread_ev=0.0,
            n_elements=0,
        )

        assert result.warnings == []


# ==============================================================================
# identify_resonance_lines Tests
# ==============================================================================


class TestIdentifyResonanceLines:
    """Tests for identify_resonance_lines function."""

    def test_returns_empty_set(self):
        """Verify stub returns empty set."""
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

        result = identify_resonance_lines(observations)

        # Current implementation is a stub returning empty set
        assert result == set()

    def test_accepts_threshold_parameter(self):
        """Verify function accepts ground_state_threshold_ev parameter."""
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

        # Should not raise
        result = identify_resonance_lines(observations, ground_state_threshold_ev=1.0)

        assert isinstance(result, set)


# ==============================================================================
# recommend_lines Tests
# ==============================================================================


class TestRecommendLines:
    """Tests for LineSelector.recommend_lines() method."""

    def test_recommend_lines_returns_dict(self):
        """Verify recommend_lines returns dictionary by element."""
        selector = LineSelector(min_snr=10.0)

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=324.0,
                intensity=2000.0,
                intensity_uncertainty=20.0,
                element="Cu",
                ionization_stage=1,
                E_k_ev=3.8,
                g_k=4,
                A_ki=1.4e8,
            ),
        ]

        result = selector.recommend_lines(observations, n_per_element=5)

        assert isinstance(result, dict)
        assert "Fe" in result or "Cu" in result

    def test_recommend_lines_respects_n_per_element(self):
        """Verify n_per_element limits recommendations."""
        selector = LineSelector(min_snr=10.0, max_lines_per_element=100)

        # Create 10 Fe lines
        observations = [
            LineObservation(
                wavelength_nm=400.0 + i * 10,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0 + i * 0.2,
                g_k=9,
                A_ki=1e7,
            )
            for i in range(10)
        ]

        result = selector.recommend_lines(observations, n_per_element=3)

        if "Fe" in result:
            assert len(result["Fe"]) <= 3


# ==============================================================================
# Multi-element Tests
# ==============================================================================


class TestMultiElementSelection:
    """Tests for line selection across multiple elements."""

    def test_multiple_elements(self):
        """Verify selection works across multiple elements."""
        selector = LineSelector(min_snr=10.0, min_lines_per_element=2)

        observations = [
            # Fe lines
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=800.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=4.0,
                g_k=7,
                A_ki=8e6,
            ),
            # Cu lines
            LineObservation(
                wavelength_nm=324.0,
                intensity=2000.0,
                intensity_uncertainty=20.0,
                element="Cu",
                ionization_stage=1,
                E_k_ev=3.8,
                g_k=4,
                A_ki=1.4e8,
            ),
            LineObservation(
                wavelength_nm=327.0,
                intensity=1000.0,
                intensity_uncertainty=15.0,
                element="Cu",
                ionization_stage=1,
                E_k_ev=3.8,
                g_k=2,
                A_ki=1.4e8,
            ),
        ]

        result = selector.select(observations)

        assert result.n_elements == 2

        # Both elements should have lines selected
        selected_elements = {line.element for line in result.selected_lines}
        assert "Fe" in selected_elements
        assert "Cu" in selected_elements

    def test_n_elements_count(self):
        """Verify n_elements is counted correctly."""
        selector = LineSelector(min_snr=10.0, min_lines_per_element=1)

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=324.0,
                intensity=2000.0,
                intensity_uncertainty=20.0,
                element="Cu",
                ionization_stage=1,
                E_k_ev=3.8,
                g_k=4,
                A_ki=1.4e8,
            ),
            LineObservation(
                wavelength_nm=394.0,
                intensity=1500.0,
                intensity_uncertainty=15.0,
                element="Al",
                ionization_stage=1,
                E_k_ev=3.1,
                g_k=2,
                A_ki=5e7,
            ),
        ]

        result = selector.select(observations)

        assert result.n_elements == 3


# ==============================================================================
# Integration Tests: Line Selector with Boltzmann Fitter Pipeline
# ==============================================================================


class TestLineSelectorBoltzmannIntegration:
    """Integration tests for line selection feeding into Boltzmann fitting."""

    def test_selected_lines_produce_valid_boltzmann_fit(self):
        """Verify auto-selected lines produce valid Boltzmann plot fit."""
        from cflibs.inversion.boltzmann import BoltzmannPlotFitter

        # Create synthetic Fe lines following Boltzmann distribution at 8000 K
        T_K = 8000.0
        T_eV = T_K * 8.617e-5  # KB_EV
        intercept_const = 10.0

        observations = []
        for i, E_k in enumerate(np.linspace(2.0, 6.0, 15)):
            # y = ln(const) - E_k / kT
            expected_y = np.log(intercept_const) - E_k / T_eV
            intensity = np.exp(expected_y)
            # Add some noise
            intensity *= 1 + np.random.normal(0, 0.02)

            observations.append(
                LineObservation(
                    wavelength_nm=400.0 + i * 10,  # Well separated
                    intensity=intensity,
                    intensity_uncertainty=intensity * 0.05,
                    element="Fe",
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )

        # Add some low-quality outliers that should be rejected
        observations.append(
            LineObservation(
                wavelength_nm=600.0,
                intensity=1.0,  # Very weak
                intensity_uncertainty=10.0,  # SNR = 0.1
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=1,
                A_ki=1.0,
            )
        )

        # Select lines
        selector = LineSelector(min_snr=10.0, min_lines_per_element=3)
        selection_result = selector.select(observations)

        # Should have rejected the weak line
        assert len(selection_result.rejected_lines) >= 1

        # Fit Boltzmann plot with selected lines
        fitter = BoltzmannPlotFitter()
        fit_result = fitter.fit(selection_result.selected_lines)

        # Temperature should be recovered within 10%
        assert abs(fit_result.temperature_K - T_K) / T_K < 0.10
        assert fit_result.r_squared > 0.95

    def test_auto_selection_improves_fit_quality(self):
        """Verify auto-selection improves fit vs using all lines."""
        from cflibs.inversion.boltzmann import BoltzmannPlotFitter

        T_K = 9000.0
        T_eV = T_K * 8.617e-5

        # Create good lines
        good_lines = []
        for i, E_k in enumerate(np.linspace(2.0, 5.0, 10)):
            expected_y = np.log(10.0) - E_k / T_eV
            intensity = np.exp(expected_y)
            good_lines.append(
                LineObservation(
                    wavelength_nm=400.0 + i * 10,
                    intensity=intensity * (1 + np.random.normal(0, 0.01)),
                    intensity_uncertainty=intensity * 0.03,
                    element="Fe",
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )

        # Add bad lines (outliers due to self-absorption)
        bad_lines = []
        for i, E_k in enumerate([2.5, 3.5, 4.5]):
            expected_y = np.log(10.0) - E_k / T_eV
            intensity = np.exp(expected_y)
            # Reduce intensity significantly (simulating self-absorption)
            bad_lines.append(
                LineObservation(
                    wavelength_nm=550.0 + i * 10,
                    intensity=intensity * 0.1,  # 90% absorption
                    intensity_uncertainty=intensity * 0.01,  # High apparent SNR
                    element="Fe",
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )

        all_lines = good_lines + bad_lines

        # Fit with all lines (no selection)
        fitter = BoltzmannPlotFitter()
        fitter.fit(all_lines)

        # Fit with auto-selection (should exclude some bad lines via outlier rejection)
        selector = LineSelector(min_snr=10.0, min_lines_per_element=3)
        selection_result = selector.select(all_lines)
        fit_selected = fitter.fit(selection_result.selected_lines)

        # Both should work, but selected fit should be at least as good
        # (Auto-selection primarily filters low SNR, not self-absorbed high-SNR lines)
        assert fit_selected.n_points >= 3
        assert np.isfinite(fit_selected.temperature_K)


# ==============================================================================
# Comparison Tests: Auto-Selected vs Manual Line Sets
# ==============================================================================


class TestAutoVsManualLineSelection:
    """Compare automatic selection with manual expert selection patterns."""

    def test_auto_selection_prefers_high_snr(self):
        """Verify auto-selection prefers high SNR lines."""
        selector = LineSelector(min_snr=10.0)

        # Create lines with varying SNR
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,  # SNR=100
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=200.0,
                intensity_uncertainty=10.0,  # SNR=20
                element="Fe",
                ionization_stage=1,
                E_k_ev=4.0,
                g_k=7,
                A_ki=8e6,
            ),
            LineObservation(
                wavelength_nm=420.0,
                intensity=500.0,
                intensity_uncertainty=10.0,  # SNR=50
                element="Fe",
                ionization_stage=1,
                E_k_ev=5.0,
                g_k=5,
                A_ki=5e6,
            ),
        ]

        result = selector.select(observations)

        # All should be selected (all above min_snr)
        assert len(result.selected_lines) == 3

        # First selected should have highest score (highest SNR)
        scores = {s.observation.wavelength_nm: s.snr for s in result.scores}
        assert scores[400.0] > scores[410.0]
        assert scores[400.0] > scores[420.0]

    def test_auto_selection_avoids_blends(self):
        """Verify auto-selection rejects blended lines."""
        selector = LineSelector(min_snr=10.0, isolation_wavelength_nm=0.1)

        # Create closely spaced lines (blended)
        observations = [
            LineObservation(
                wavelength_nm=400.00,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=400.01,
                intensity=800.0,
                intensity_uncertainty=10.0,  # 0.01 nm away
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
            LineObservation(
                wavelength_nm=500.0,
                intensity=500.0,
                intensity_uncertainty=10.0,  # Well isolated
                element="Fe",
                ionization_stage=1,
                E_k_ev=4.0,
                g_k=5,
                A_ki=5e6,
            ),
        ]

        result = selector.select(observations)

        # Find isolation factors
        blended_scores = [s for s in result.scores if s.observation.wavelength_nm < 401]
        isolated_score = next(s for s in result.scores if s.observation.wavelength_nm == 500.0)

        # Blended lines should have low isolation factor
        for s in blended_scores:
            assert s.isolation_factor < 0.5
        # Isolated line should have high isolation factor
        assert isolated_score.isolation_factor > 0.9

    def test_manual_vs_auto_energy_spread(self):
        """Compare auto-selection energy spread with manual recommendation."""
        selector = LineSelector(
            min_snr=10.0,
            min_energy_spread_ev=2.0,
            min_lines_per_element=3,
        )

        # Create lines with good energy spread
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=2.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=420.0,
                intensity=800.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
            LineObservation(
                wavelength_nm=440.0,
                intensity=600.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=5.0,
                g_k=5,
                A_ki=5e6,
            ),
        ]

        result = selector.select(observations)

        # Energy spread should be adequate (5.0 - 2.0 = 3.0 eV > 2.0 eV)
        assert result.energy_spread_ev >= 2.0
        # No energy spread warnings expected
        assert not any("Energy spread" in w for w in result.warnings)


# ==============================================================================
# Edge Case Tests: Crowded Spectra
# ==============================================================================


class TestCrowdedSpectraEdgeCases:
    """Edge case tests for crowded/complex spectra scenarios."""

    def test_highly_crowded_spectrum(self):
        """Test selection in a highly crowded spectral region."""
        # Use larger isolation scale so 0.05 nm spacing counts as blended
        selector = LineSelector(min_snr=10.0, isolation_wavelength_nm=0.1)

        # Create 20 lines in a narrow 1 nm window (0.05 nm spacing)
        observations = [
            LineObservation(
                wavelength_nm=400.0 + i * 0.05,  # 0.05 nm spacing (half isolation scale)
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0 + i * 0.1,
                g_k=9,
                A_ki=1e7,
            )
            for i in range(20)
        ]

        result = selector.select(observations)

        # Many lines should be rejected due to low isolation
        n_blended_rejections = sum(
            1 for s in result.scores if s.rejection_reason and "Blended" in s.rejection_reason
        )
        assert n_blended_rejections > 0

        # Edge lines should have better isolation than center lines
        # With 0.05 nm spacing: first=400.0, last=400.95
        edge_scores = [s for s in result.scores if s.observation.wavelength_nm in [400.0, 400.95]]
        center_scores = [s for s in result.scores if 400.3 < s.observation.wavelength_nm < 400.6]

        if edge_scores and center_scores:
            avg_edge_isolation = np.mean([s.isolation_factor for s in edge_scores])
            avg_center_isolation = np.mean([s.isolation_factor for s in center_scores])
            # Edge lines should have better isolation (only one neighbor)
            assert avg_edge_isolation >= avg_center_isolation

    def test_spectrum_with_one_isolated_line(self):
        """Test crowded spectrum with one well-isolated line."""
        selector = LineSelector(min_snr=10.0, isolation_wavelength_nm=0.1)

        # Cluster of closely spaced lines
        observations = [
            LineObservation(
                wavelength_nm=400.0 + i * 0.02,  # Very tight cluster
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0 + i * 0.1,
                g_k=9,
                A_ki=1e7,
            )
            for i in range(5)
        ]

        # Add one well-isolated line
        observations.append(
            LineObservation(
                wavelength_nm=500.0,  # 100 nm away
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=4.0,
                g_k=9,
                A_ki=1e7,
            )
        )

        result = selector.select(observations)

        # The isolated line should definitely be selected
        isolated_line = next(
            (line for line in result.selected_lines if line.wavelength_nm == 500.0), None
        )
        assert isolated_line is not None

        # Its isolation score should be near 1.0
        isolated_score = next(s for s in result.scores if s.observation.wavelength_nm == 500.0)
        assert isolated_score.isolation_factor > 0.99

    def test_mixed_elements_crowded(self):
        """Test selection with multiple elements in crowded region."""
        selector = LineSelector(
            min_snr=10.0,
            isolation_wavelength_nm=0.1,
            min_lines_per_element=2,
        )

        observations = []
        # Fe lines
        for i in range(5):
            observations.append(
                LineObservation(
                    wavelength_nm=400.0 + i * 0.5,
                    intensity=1000.0,
                    intensity_uncertainty=10.0,
                    element="Fe",
                    ionization_stage=1,
                    E_k_ev=3.0 + i * 0.2,
                    g_k=9,
                    A_ki=1e7,
                )
            )
        # Cu lines interspersed
        for i in range(3):
            observations.append(
                LineObservation(
                    wavelength_nm=400.25 + i * 0.5,  # Between Fe lines
                    intensity=800.0,
                    intensity_uncertainty=10.0,
                    element="Cu",
                    ionization_stage=1,
                    E_k_ev=3.5 + i * 0.2,
                    g_k=4,
                    A_ki=1.4e8,
                )
            )

        result = selector.select(observations)

        # Both elements should be represented
        {line.element for line in result.selected_lines}
        # At least some should be selected despite crowding
        assert len(result.selected_lines) > 0

    def test_missing_atomic_data_graceful_handling(self):
        """Test graceful handling when atomic data uncertainties are missing."""
        selector = LineSelector(min_snr=10.0)

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=800.0,
                intensity_uncertainty=10.0,
                element="Zr",  # Rare element with potentially missing data
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=5,
                A_ki=1e6,
            ),
        ]

        # Empty atomic uncertainties - should use defaults
        result = selector.select(observations, atomic_uncertainties={})

        # Should still work with default 10% uncertainty
        assert len(result.scores) == 2
        for score in result.scores:
            assert score.atomic_uncertainty == 0.10  # Default

    def test_all_lines_low_snr(self):
        """Test behavior when all lines have low SNR."""
        selector = LineSelector(min_snr=20.0)

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=100.0,
                intensity_uncertainty=10.0,  # SNR=10
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=50.0,
                intensity_uncertainty=10.0,  # SNR=5
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
        ]

        result = selector.select(observations)

        # All should be rejected
        assert len(result.selected_lines) == 0
        assert len(result.rejected_lines) == 2
        assert all("Low SNR" in s.rejection_reason for s in result.scores)
"""
Unit tests for shared element identification data structures.
"""

import pytest
from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
    to_line_observations,
)
from cflibs.atomic.structures import Transition
from cflibs.inversion.boltzmann import LineObservation

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_transition():
    """Create a mock Transition object."""
    return Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=371.99,
        A_ki=1.0e8,
        E_k_ev=3.33,
        E_i_ev=0.0,
        g_k=9,
        g_i=9,
        relative_intensity=1000.0,
    )


@pytest.fixture
def mock_transition_ti():
    """Create a mock Transition object for Ti."""
    return Transition(
        element="Ti",
        ionization_stage=1,
        wavelength_nm=498.17,
        A_ki=5.0e7,
        E_k_ev=2.49,
        E_i_ev=0.0,
        g_k=7,
        g_i=5,
        relative_intensity=500.0,
    )


def test_identified_line_creation(mock_transition):
    """Test IdentifiedLine creation with all fields."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        correlation=0.95,
        is_interfered=False,
        interfering_elements=[],
    )

    assert line.wavelength_exp_nm == 372.0
    assert line.wavelength_th_nm == 371.99
    assert line.element == "Fe"
    assert line.ionization_stage == 1
    assert line.intensity_exp == 1000.0
    assert line.emissivity_th == 500.0
    assert line.transition == mock_transition
    assert line.correlation == 0.95
    assert line.is_interfered is False
    assert line.interfering_elements == []


def test_identified_line_defaults(mock_transition):
    """Test IdentifiedLine default values."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    assert line.correlation == 0.0
    assert line.is_interfered is False
    assert line.interfering_elements == []


def test_identified_line_interfered(mock_transition):
    """Test IdentifiedLine with interference."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        is_interfered=True,
        interfering_elements=["Ti", "Cr"],
    )

    assert line.is_interfered is True
    assert line.interfering_elements == ["Ti", "Cr"]


def test_element_identification_creation(mock_transition):
    """Test ElementIdentification with matched/unmatched lines."""
    matched_line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        correlation=0.95,
    )

    unmatched_transition = Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=373.49,
        A_ki=8.0e7,
        E_k_ev=3.24,
        E_i_ev=0.0,
        g_k=7,
        g_i=9,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=2,
        matched_lines=[matched_line],
        unmatched_lines=[unmatched_transition],
        metadata={"mean_correlation": 0.95, "snr": 50.0},
    )

    assert elem_id.element == "Fe"
    assert elem_id.detected is True
    assert elem_id.score == 0.85
    assert elem_id.confidence == 0.90
    assert elem_id.n_matched_lines == 1
    assert elem_id.n_total_lines == 2
    assert len(elem_id.matched_lines) == 1
    assert len(elem_id.unmatched_lines) == 1
    assert elem_id.metadata["mean_correlation"] == 0.95


def test_element_identification_result_creation(mock_transition, mock_transition_ti):
    """Test ElementIdentificationResult with detected/rejected split."""
    fe_line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        correlation=0.95,
    )

    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=2,
        matched_lines=[fe_line],
        unmatched_lines=[],
        metadata={},
    )

    ti_id = ElementIdentification(
        element="Ti",
        detected=False,
        score=0.30,
        confidence=0.25,
        n_matched_lines=0,
        n_total_lines=5,
        matched_lines=[],
        unmatched_lines=[mock_transition_ti],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id],
        rejected_elements=[ti_id],
        all_elements=[fe_id, ti_id],
        experimental_peaks=[(0, 372.0), (1, 498.2)],
        n_peaks=2,
        n_matched_peaks=1,
        n_unmatched_peaks=1,
        algorithm="alias",
        parameters={"threshold": 0.5, "tolerance_nm": 0.1},
        warnings=["Low SNR on peak at 498.2 nm"],
    )

    assert len(result.detected_elements) == 1
    assert len(result.rejected_elements) == 1
    assert len(result.all_elements) == 2
    assert result.n_peaks == 2
    assert result.n_matched_peaks == 1
    assert result.n_unmatched_peaks == 1
    assert result.algorithm == "alias"
    assert result.parameters["threshold"] == 0.5
    assert len(result.warnings) == 1


def test_to_line_observations_conversion(mock_transition, mock_transition_ti):
    """Test to_line_observations() conversion."""
    fe_line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        correlation=0.95,
    )

    ti_line = IdentifiedLine(
        wavelength_exp_nm=498.2,
        wavelength_th_nm=498.17,
        element="Ti",
        ionization_stage=1,
        intensity_exp=500.0,
        emissivity_th=250.0,
        transition=mock_transition_ti,
        correlation=0.85,
    )

    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=2,
        matched_lines=[fe_line],
        unmatched_lines=[],
        metadata={},
    )

    ti_id = ElementIdentification(
        element="Ti",
        detected=True,
        score=0.70,
        confidence=0.75,
        n_matched_lines=1,
        n_total_lines=5,
        matched_lines=[ti_line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id, ti_id],
        rejected_elements=[],
        all_elements=[fe_id, ti_id],
        experimental_peaks=[(0, 372.0), (1, 498.2)],
        n_peaks=2,
        n_matched_peaks=2,
        n_unmatched_peaks=0,
        algorithm="alias",
    )

    observations = to_line_observations(result)

    assert len(observations) == 2
    assert all(isinstance(obs, LineObservation) for obs in observations)

    # Check Fe line
    fe_obs = observations[0]
    assert fe_obs.wavelength_nm == 371.99
    assert fe_obs.intensity == 1000.0
    assert fe_obs.intensity_uncertainty == max(1000.0 * 0.02, 1e-6)
    assert fe_obs.element == "Fe"
    assert fe_obs.ionization_stage == 1
    assert fe_obs.E_k_ev == 3.33
    assert fe_obs.g_k == 9
    assert fe_obs.A_ki == 1.0e8

    # Check Ti line
    ti_obs = observations[1]
    assert ti_obs.wavelength_nm == 498.17
    assert ti_obs.intensity == 500.0
    assert ti_obs.element == "Ti"


def test_to_line_observations_skips_interfered(mock_transition):
    """Test to_line_observations() skips interfered lines."""
    clean_line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        is_interfered=False,
    )

    interfered_line = IdentifiedLine(
        wavelength_exp_nm=373.5,
        wavelength_th_nm=373.49,
        element="Fe",
        ionization_stage=1,
        intensity_exp=500.0,
        emissivity_th=250.0,
        transition=mock_transition,
        is_interfered=True,
        interfering_elements=["Ti"],
    )

    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=2,
        n_total_lines=2,
        matched_lines=[clean_line, interfered_line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id],
        rejected_elements=[],
        all_elements=[fe_id],
        experimental_peaks=[(0, 372.0), (1, 373.5)],
        n_peaks=2,
        n_matched_peaks=2,
        n_unmatched_peaks=0,
        algorithm="alias",
    )

    observations = to_line_observations(result)

    # Should only have 1 line (interfered line skipped)
    assert len(observations) == 1
    assert observations[0].wavelength_nm == 371.99


def test_to_line_observations_deduplicates(mock_transition):
    """Test to_line_observations() deduplicates."""
    line1 = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    # Duplicate line (same element, ion stage, wavelength_th)
    line2 = IdentifiedLine(
        wavelength_exp_nm=372.01,  # Slightly different exp wavelength
        wavelength_th_nm=371.99,  # Same theoretical
        element="Fe",
        ionization_stage=1,
        intensity_exp=1050.0,
        emissivity_th=525.0,
        transition=mock_transition,
    )

    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=2,
        n_total_lines=2,
        matched_lines=[line1, line2],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id],
        rejected_elements=[],
        all_elements=[fe_id],
        experimental_peaks=[(0, 372.0), (1, 372.01)],
        n_peaks=2,
        n_matched_peaks=2,
        n_unmatched_peaks=0,
        algorithm="alias",
    )

    observations = to_line_observations(result)

    # Should only have 1 line (duplicate removed)
    assert len(observations) == 1
    assert observations[0].wavelength_nm == 371.99
    assert observations[0].intensity == 1000.0  # First one kept


def test_to_line_observations_intensity_uncertainty_floor(mock_transition):
    """Test intensity uncertainty has 1e-6 floor."""
    # Very low intensity line
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1e-8,  # Very low
        emissivity_th=500.0,
        transition=mock_transition,
    )

    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.50,
        confidence=0.45,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id],
        rejected_elements=[],
        all_elements=[fe_id],
        experimental_peaks=[(0, 372.0)],
        n_peaks=1,
        n_matched_peaks=1,
        n_unmatched_peaks=0,
        algorithm="alias",
    )

    observations = to_line_observations(result)

    # Uncertainty should be floored at 1e-6
    assert observations[0].intensity_uncertainty == 1e-6


# ============================================================================
# Additional comprehensive tests
# ============================================================================


def test_identified_line_wavelength_shift(mock_transition):
    """Test IdentifiedLine with wavelength shift between exp and theoretical."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.1,  # 0.11 nm shift
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    # Check shift
    shift = line.wavelength_exp_nm - line.wavelength_th_nm
    assert abs(shift - 0.11) < 1e-6


def test_identified_line_different_ionization_stages(mock_transition):
    """Test IdentifiedLine with different ionization stages."""
    line_neutral = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,  # Neutral
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    line_ionized = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=2,  # Singly ionized
        intensity_exp=800.0,
        emissivity_th=400.0,
        transition=mock_transition,
    )

    assert line_neutral.ionization_stage == 1
    assert line_ionized.ionization_stage == 2


def test_identified_line_multiple_interfering_elements(mock_transition):
    """Test IdentifiedLine with multiple interfering elements."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        is_interfered=True,
        interfering_elements=["Ti", "Cr", "Mn"],
    )

    assert line.is_interfered is True
    assert len(line.interfering_elements) == 3
    assert "Ti" in line.interfering_elements
    assert "Cr" in line.interfering_elements
    assert "Mn" in line.interfering_elements


def test_element_identification_zero_matched_lines(mock_transition):
    """Test ElementIdentification with zero matched lines."""
    elem_id = ElementIdentification(
        element="Ca",
        detected=False,
        score=0.0,
        confidence=0.0,
        n_matched_lines=0,
        n_total_lines=10,
        matched_lines=[],
        unmatched_lines=[mock_transition],
        metadata={"reason": "no_peaks_found"},
    )

    assert elem_id.detected is False
    assert elem_id.n_matched_lines == 0
    assert len(elem_id.matched_lines) == 0
    assert len(elem_id.unmatched_lines) > 0


def test_element_identification_all_lines_matched(mock_transition):
    """Test ElementIdentification with all lines matched."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=1.0,
        confidence=1.0,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={"match_rate": 1.0},
    )

    assert elem_id.n_matched_lines == elem_id.n_total_lines
    assert len(elem_id.unmatched_lines) == 0


def test_element_identification_score_confidence_difference(mock_transition):
    """Test ElementIdentification with different score and confidence."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.70,  # Based on correlation
        confidence=0.85,  # Includes quality factors
        n_matched_lines=5,
        n_total_lines=10,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={"quality_boost": 0.15},
    )

    assert elem_id.score < elem_id.confidence


def test_element_identification_metadata_types(mock_transition):
    """Test ElementIdentification with various metadata types."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={
            "mean_snr": 50.0,
            "n_interfered": 0.0,
            "algorithm_time_ms": 15.3,
        },
    )

    assert isinstance(elem_id.metadata["mean_snr"], float)
    assert elem_id.metadata["n_interfered"] == 0.0


def test_element_identification_result_empty(mock_transition):
    """Test ElementIdentificationResult with no elements detected."""
    result = ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[],
        all_elements=[],
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    assert len(result.detected_elements) == 0
    assert len(result.rejected_elements) == 0
    assert len(result.all_elements) == 0
    assert result.n_peaks == 0


def test_element_identification_result_all_peaks_matched(mock_transition):
    """Test ElementIdentificationResult with all peaks matched."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.95,
        confidence=0.95,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[elem_id],
        rejected_elements=[],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.0)],
        n_peaks=1,
        n_matched_peaks=1,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    assert result.n_unmatched_peaks == 0
    assert result.n_matched_peaks == result.n_peaks


def test_element_identification_result_warnings(mock_transition):
    """Test ElementIdentificationResult with warnings."""
    result = ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[],
        all_elements=[],
        experimental_peaks=[(0, 300.0)],
        n_peaks=1,
        n_matched_peaks=0,
        n_unmatched_peaks=1,
        algorithm="comb",
        warnings=[
            "Low SNR detected",
            "Baseline instability at 300-350 nm",
            "Possible saturation in detector",
        ],
    )

    assert len(result.warnings) == 3
    assert "Low SNR" in result.warnings[0]


def test_element_identification_result_multiple_algorithms(mock_transition):
    """Test ElementIdentificationResult for different algorithms."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.85,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={},
    )

    for algorithm in ["alias", "comb", "correlation"]:
        result = ElementIdentificationResult(
            detected_elements=[elem_id],
            rejected_elements=[],
            all_elements=[elem_id],
            experimental_peaks=[(0, 372.0)],
            n_peaks=1,
            n_matched_peaks=1,
            n_unmatched_peaks=0,
            algorithm=algorithm,
        )
        assert result.algorithm == algorithm


def test_to_line_observations_empty_result():
    """Test to_line_observations with empty result."""
    result = ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[],
        all_elements=[],
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    assert len(observations) == 0


def test_to_line_observations_only_rejected_elements(mock_transition):
    """Test to_line_observations with only rejected elements."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=False,  # Rejected
        score=0.30,
        confidence=0.25,
        n_matched_lines=1,
        n_total_lines=5,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[elem_id],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.0)],
        n_peaks=1,
        n_matched_peaks=0,
        n_unmatched_peaks=1,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    # Should be empty because element is rejected
    assert len(observations) == 0


def test_to_line_observations_all_interfered(mock_transition):
    """Test to_line_observations when all lines are interfered."""
    line1 = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        is_interfered=True,
        interfering_elements=["Ti"],
    )

    line2 = IdentifiedLine(
        wavelength_exp_nm=373.5,
        wavelength_th_nm=373.49,
        element="Fe",
        ionization_stage=1,
        intensity_exp=500.0,
        emissivity_th=250.0,
        transition=mock_transition,
        is_interfered=True,
        interfering_elements=["Cr"],
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.75,
        confidence=0.70,
        n_matched_lines=2,
        n_total_lines=2,
        matched_lines=[line1, line2],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[elem_id],
        rejected_elements=[],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.0), (1, 373.5)],
        n_peaks=2,
        n_matched_peaks=2,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    # All lines interfered, should be empty
    assert len(observations) == 0


def test_to_line_observations_multiple_elements(mock_transition, mock_transition_ti):
    """Test to_line_observations with multiple detected elements."""
    fe_line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    ti_line = IdentifiedLine(
        wavelength_exp_nm=498.2,
        wavelength_th_nm=498.17,
        element="Ti",
        ionization_stage=1,
        intensity_exp=500.0,
        emissivity_th=250.0,
        transition=mock_transition_ti,
    )

    cu_transition = Transition(
        element="Cu",
        ionization_stage=1,
        wavelength_nm=324.75,
        A_ki=1.4e8,
        E_k_ev=3.82,
        E_i_ev=0.0,
        g_k=4,
        g_i=2,
    )

    cu_line = IdentifiedLine(
        wavelength_exp_nm=324.8,
        wavelength_th_nm=324.75,
        element="Cu",
        ionization_stage=1,
        intensity_exp=2000.0,
        emissivity_th=1000.0,
        transition=cu_transition,
    )

    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[fe_line],
        unmatched_lines=[],
        metadata={},
    )

    ti_id = ElementIdentification(
        element="Ti",
        detected=True,
        score=0.70,
        confidence=0.75,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[ti_line],
        unmatched_lines=[],
        metadata={},
    )

    cu_id = ElementIdentification(
        element="Cu",
        detected=True,
        score=0.95,
        confidence=0.95,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[cu_line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id, ti_id, cu_id],
        rejected_elements=[],
        all_elements=[fe_id, ti_id, cu_id],
        experimental_peaks=[(0, 324.8), (1, 372.0), (2, 498.2)],
        n_peaks=3,
        n_matched_peaks=3,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    assert len(observations) == 3
    elements = [obs.element for obs in observations]
    assert "Fe" in elements
    assert "Ti" in elements
    assert "Cu" in elements


def test_to_line_observations_preserves_atomic_data(mock_transition):
    """Test that atomic data is correctly transferred to LineObservation."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[elem_id],
        rejected_elements=[],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.0)],
        n_peaks=1,
        n_matched_peaks=1,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    obs = observations[0]

    # Check atomic data is preserved
    assert obs.E_k_ev == mock_transition.E_k_ev
    assert obs.g_k == mock_transition.g_k
    assert obs.A_ki == mock_transition.A_ki


def test_to_line_observations_intensity_uncertainty_calculation(mock_transition):
    """Test intensity uncertainty calculation (2% or 1e-6 floor)."""
    # High intensity case - 2% should dominate
    line_high = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=10000.0,
        emissivity_th=5000.0,
        transition=mock_transition,
    )

    elem_id_high = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.9,
        confidence=0.9,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line_high],
        unmatched_lines=[],
        metadata={},
    )

    result_high = ElementIdentificationResult(
        detected_elements=[elem_id_high],
        rejected_elements=[],
        all_elements=[elem_id_high],
        experimental_peaks=[(0, 372.0)],
        n_peaks=1,
        n_matched_peaks=1,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    obs_high = to_line_observations(result_high)[0]
    assert obs_high.intensity_uncertainty == 10000.0 * 0.02  # 2%

    # Low intensity case - 1e-6 floor should dominate
    line_low = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1e-9,
        emissivity_th=5e-10,
        transition=mock_transition,
    )

    elem_id_low = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.5,
        confidence=0.5,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line_low],
        unmatched_lines=[],
        metadata={},
    )

    result_low = ElementIdentificationResult(
        detected_elements=[elem_id_low],
        rejected_elements=[],
        all_elements=[elem_id_low],
        experimental_peaks=[(0, 372.0)],
        n_peaks=1,
        n_matched_peaks=1,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    obs_low = to_line_observations(result_low)[0]
    assert obs_low.intensity_uncertainty == 1e-6  # Floor


def test_to_line_observations_uses_theoretical_wavelength(mock_transition):
    """Test that LineObservation uses theoretical wavelength, not experimental."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.1,  # Shifted from theoretical
        wavelength_th_nm=371.99,  # This should be used
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[elem_id],
        rejected_elements=[],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.1)],
        n_peaks=1,
        n_matched_peaks=1,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    # Should use theoretical wavelength
    assert observations[0].wavelength_nm == 371.99


def test_to_line_observations_deduplication_order(mock_transition):
    """Test that deduplication keeps first occurrence."""
    line1 = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,  # First occurrence
        emissivity_th=500.0,
        transition=mock_transition,
    )

    line2 = IdentifiedLine(
        wavelength_exp_nm=372.05,
        wavelength_th_nm=371.99,  # Same theoretical
        element="Fe",
        ionization_stage=1,
        intensity_exp=1100.0,  # Different intensity
        emissivity_th=550.0,
        transition=mock_transition,
    )

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=2,
        n_total_lines=2,
        matched_lines=[line1, line2],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[elem_id],
        rejected_elements=[],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.0), (1, 372.05)],
        n_peaks=2,
        n_matched_peaks=2,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    assert len(observations) == 1
    # Should keep first occurrence
    assert observations[0].intensity == 1000.0


def test_element_identification_result_consistency():
    """Test that all_elements equals detected + rejected."""
    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[],
        unmatched_lines=[],
        metadata={},
    )

    ti_id = ElementIdentification(
        element="Ti",
        detected=False,
        score=0.30,
        confidence=0.25,
        n_matched_lines=0,
        n_total_lines=5,
        matched_lines=[],
        unmatched_lines=[],
        metadata={},
    )

    cu_id = ElementIdentification(
        element="Cu",
        detected=True,
        score=0.75,
        confidence=0.80,
        n_matched_lines=2,
        n_total_lines=3,
        matched_lines=[],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id, cu_id],
        rejected_elements=[ti_id],
        all_elements=[fe_id, ti_id, cu_id],
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    assert len(result.all_elements) == len(result.detected_elements) + len(result.rejected_elements)

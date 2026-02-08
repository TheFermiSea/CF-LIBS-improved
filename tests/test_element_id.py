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

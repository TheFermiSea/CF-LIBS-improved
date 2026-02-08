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


def test_identified_line_with_multiple_interferers(mock_transition):
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

    assert len(line.interfering_elements) == 3
    assert "Ti" in line.interfering_elements
    assert "Cr" in line.interfering_elements
    assert "Mn" in line.interfering_elements


def test_identified_line_high_correlation(mock_transition):
    """Test IdentifiedLine with very high correlation."""
    line = IdentifiedLine(
        wavelength_exp_nm=371.99,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
        correlation=0.999,
    )

    assert line.correlation > 0.99
    assert line.wavelength_exp_nm == line.wavelength_th_nm  # Perfect match


def test_identified_line_low_correlation(mock_transition):
    """Test IdentifiedLine with very low correlation."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.5,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=100.0,
        emissivity_th=500.0,
        transition=mock_transition,
        correlation=0.1,
    )

    assert line.correlation < 0.2
    assert abs(line.wavelength_exp_nm - line.wavelength_th_nm) > 0.5  # Poor match


def test_element_identification_zero_matched_lines(mock_transition):
    """Test ElementIdentification with no matched lines."""
    unmatched = Transition(
        element="Ca",
        ionization_stage=1,
        wavelength_nm=422.67,
        A_ki=2.0e8,
        E_k_ev=2.93,
        E_i_ev=0.0,
        g_k=3,
        g_i=1,
    )

    elem_id = ElementIdentification(
        element="Ca",
        detected=False,
        score=0.0,
        confidence=0.0,
        n_matched_lines=0,
        n_total_lines=10,
        matched_lines=[],
        unmatched_lines=[unmatched],
        metadata={"reason": "no_peaks_found"},
    )

    assert elem_id.n_matched_lines == 0
    assert elem_id.detected is False
    assert elem_id.score == 0.0
    assert len(elem_id.matched_lines) == 0


def test_element_identification_all_lines_matched(mock_transition):
    """Test ElementIdentification with all lines matched."""
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

    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.95,
        confidence=0.98,
        n_matched_lines=5,
        n_total_lines=5,
        matched_lines=[matched_line] * 5,  # Simplified for test
        unmatched_lines=[],
        metadata={"match_rate": 1.0},
    )

    assert elem_id.n_matched_lines == elem_id.n_total_lines
    assert len(elem_id.unmatched_lines) == 0


def test_element_identification_metadata_preservation(mock_transition):
    """Test that metadata is preserved in ElementIdentification."""
    metadata = {
        "fingerprint": 0.87,
        "n_active_teeth": 8,
        "n_total_teeth": 12,
        "mean_snr": 45.5,
        "custom_metric": 123.456,
    }

    elem_id = ElementIdentification(
        element="Cu",
        detected=True,
        score=0.87,
        confidence=0.90,
        n_matched_lines=8,
        n_total_lines=12,
        matched_lines=[],
        unmatched_lines=[],
        metadata=metadata,
    )

    assert elem_id.metadata["fingerprint"] == 0.87
    assert elem_id.metadata["n_active_teeth"] == 8
    assert elem_id.metadata["custom_metric"] == 123.456


def test_element_identification_result_empty():
    """Test ElementIdentificationResult with no elements."""
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
    assert len(result.all_elements) == 0
    assert result.n_peaks == 0


def test_element_identification_result_all_matched():
    """Test ElementIdentificationResult where all peaks are matched."""
    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.90,
        confidence=0.92,
        n_matched_lines=3,
        n_total_lines=5,
        matched_lines=[],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[elem_id],
        rejected_elements=[],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.0), (1, 373.5), (2, 375.0)],
        n_peaks=3,
        n_matched_peaks=3,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    assert result.n_matched_peaks == result.n_peaks
    assert result.n_unmatched_peaks == 0


def test_element_identification_result_all_unmatched():
    """Test ElementIdentificationResult where no peaks are matched."""
    result = ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[],
        all_elements=[],
        experimental_peaks=[(0, 400.0), (1, 500.0)],
        n_peaks=2,
        n_matched_peaks=0,
        n_unmatched_peaks=2,
        algorithm="alias",
    )

    assert result.n_matched_peaks == 0
    assert result.n_unmatched_peaks == result.n_peaks


def test_element_identification_result_warnings():
    """Test ElementIdentificationResult with warnings."""
    result = ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[],
        all_elements=[],
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm="correlation",
        warnings=[
            "Low SNR detected",
            "Possible saturation at 500nm",
            "Baseline drift detected",
        ],
    )

    assert len(result.warnings) == 3
    assert "Low SNR detected" in result.warnings


def test_element_identification_result_multiple_algorithms():
    """Test that algorithm field correctly identifies different algorithms."""
    for alg in ["alias", "comb", "correlation"]:
        result = ElementIdentificationResult(
            detected_elements=[],
            rejected_elements=[],
            all_elements=[],
            experimental_peaks=[],
            n_peaks=0,
            n_matched_peaks=0,
            n_unmatched_peaks=0,
            algorithm=alg,
        )
        assert result.algorithm == alg


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
    assert isinstance(observations, list)


def test_to_line_observations_rejected_elements_ignored(mock_transition):
    """Test that rejected elements are not converted."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    rejected_id = ElementIdentification(
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
        rejected_elements=[rejected_id],
        all_elements=[rejected_id],
        experimental_peaks=[(0, 372.0)],
        n_peaks=1,
        n_matched_peaks=0,
        n_unmatched_peaks=1,
        algorithm="comb",
    )

    observations = to_line_observations(result)
    # Should be empty because element was rejected
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

    cu_id = ElementIdentification(
        element="Cu",
        detected=True,
        score=0.95,
        confidence=0.98,
        n_matched_lines=1,
        n_total_lines=3,
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
    elements_found = set(obs.element for obs in observations)
    assert elements_found == {"Fe", "Ti", "Cu"}


def test_to_line_observations_different_ionization_stages(mock_transition):
    """Test to_line_observations with multiple ionization stages."""
    fe_i_line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=1000.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    fe_ii_transition = Transition(
        element="Fe",
        ionization_stage=2,
        wavelength_nm=259.94,
        A_ki=2.7e8,
        E_k_ev=4.77,
        E_i_ev=0.0,
        g_k=10,
        g_i=10,
    )

    fe_ii_line = IdentifiedLine(
        wavelength_exp_nm=260.0,
        wavelength_th_nm=259.94,
        element="Fe",
        ionization_stage=2,
        intensity_exp=800.0,
        emissivity_th=400.0,
        transition=fe_ii_transition,
    )

    fe_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.85,
        confidence=0.90,
        n_matched_lines=2,
        n_total_lines=10,
        matched_lines=[fe_i_line, fe_ii_line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[fe_id],
        rejected_elements=[],
        all_elements=[fe_id],
        experimental_peaks=[(0, 260.0), (1, 372.0)],
        n_peaks=2,
        n_matched_peaks=2,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)

    assert len(observations) == 2
    ion_stages = [obs.ionization_stage for obs in observations]
    assert 1 in ion_stages
    assert 2 in ion_stages


def test_to_line_observations_atomic_data_preserved(mock_transition):
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
        n_total_lines=2,
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
    # Check all atomic data preserved
    assert obs.E_k_ev == mock_transition.E_k_ev
    assert obs.g_k == mock_transition.g_k
    assert obs.A_ki == mock_transition.A_ki
    assert obs.element == mock_transition.element
    assert obs.ionization_stage == mock_transition.ionization_stage


def test_to_line_observations_intensity_uncertainty_calculation(mock_transition):
    """Test intensity uncertainty is 2% of intensity."""
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
        n_total_lines=2,
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

    expected_uncertainty = max(1000.0 * 0.02, 1e-6)
    assert observations[0].intensity_uncertainty == expected_uncertainty


def test_to_line_observations_uses_theoretical_wavelength(mock_transition):
    """Test that LineObservation uses theoretical wavelength, not experimental."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.05,  # Experimental (shifted)
        wavelength_th_nm=371.99,  # Theoretical (database)
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
        n_total_lines=2,
        matched_lines=[line],
        unmatched_lines=[],
        metadata={},
    )

    result = ElementIdentificationResult(
        detected_elements=[elem_id],
        rejected_elements=[],
        all_elements=[elem_id],
        experimental_peaks=[(0, 372.05)],
        n_peaks=1,
        n_matched_peaks=1,
        n_unmatched_peaks=0,
        algorithm="comb",
    )

    observations = to_line_observations(result)

    # Should use theoretical wavelength from transition
    assert observations[0].wavelength_nm == 371.99
    assert observations[0].intensity == 1000.0  # But experimental intensity


def test_element_identification_score_vs_confidence():
    """Test distinction between score and confidence."""
    elem_id = ElementIdentification(
        element="Fe",
        detected=True,
        score=0.75,  # Raw score
        confidence=0.85,  # Confidence (higher due to quality factors)
        n_matched_lines=10,
        n_total_lines=15,
        matched_lines=[],
        unmatched_lines=[],
        metadata={"quality_boost": 0.10},
    )

    # Confidence can be higher than score
    assert elem_id.confidence > elem_id.score
    assert elem_id.score == 0.75
    assert elem_id.confidence == 0.85


def test_identified_line_zero_intensity(mock_transition):
    """Test IdentifiedLine with zero intensity (edge case)."""
    line = IdentifiedLine(
        wavelength_exp_nm=372.0,
        wavelength_th_nm=371.99,
        element="Fe",
        ionization_stage=1,
        intensity_exp=0.0,
        emissivity_th=500.0,
        transition=mock_transition,
    )

    assert line.intensity_exp == 0.0


def test_element_identification_result_peak_consistency():
    """Test that peak counts are internally consistent."""
    result = ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=[],
        all_elements=[],
        experimental_peaks=[(0, 400.0), (1, 500.0), (2, 600.0)],
        n_peaks=3,
        n_matched_peaks=1,
        n_unmatched_peaks=2,
        algorithm="comb",
    )

    # n_peaks should equal matched + unmatched
    assert result.n_peaks == result.n_matched_peaks + result.n_unmatched_peaks
    # Number of peak tuples should match n_peaks
    assert len(result.experimental_peaks) == result.n_peaks
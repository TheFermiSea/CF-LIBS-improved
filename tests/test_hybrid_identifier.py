import numpy as np
import pytest
from unittest.mock import MagicMock
from cflibs.inversion.identify.hybrid import HybridIdentifier
from cflibs.inversion.element_id import ElementIdentification, ElementIdentificationResult

@pytest.fixture
def mock_deps(mocker):
    # Mock the identifiers that are imported locally in identify()
    mock_nnls = mocker.patch("cflibs.inversion.spectral_nnls_identifier.SpectralNNLSIdentifier")
    mock_alias = mocker.patch("cflibs.inversion.alias_identifier.ALIASIdentifier")
    mock_comb = mocker.patch("cflibs.inversion.comb_identifier.CombIdentifier")
    mock_corr = mocker.patch("cflibs.inversion.correlation_identifier.CorrelationIdentifier")
    return mock_nnls, mock_alias, mock_comb, mock_corr

def create_mock_result(elements_detected):
    all_elements = []
    for el in ["Fe", "Mg", "Ca", "Na"]:
        detected = el in elements_detected
        all_elements.append(ElementIdentification(
            element=el,
            detected=detected,
            score=1.0 if detected else 0.0,
            confidence=1.0 if detected else 0.0,
            n_matched_lines=1 if detected else 0,
            n_total_lines=1,
            matched_lines=[],
            unmatched_lines=[],
            metadata={}
        ))
    return ElementIdentificationResult(
        detected_elements=[e for e in all_elements if e.detected],
        rejected_elements=[e for e in all_elements if not e.detected],
        all_elements=all_elements,
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm="test",
        parameters={}
    )

def test_hybrid_identifier_quorum_2_of_3(mock_deps):
    mock_nnls, mock_alias, mock_comb, mock_corr = mock_deps
    
    # Fe: 3 votes (detected)
    # Mg: 2 votes (detected)
    # Ca: 1 vote (rejected)
    # Na: 0 votes (rejected)
    
    mock_nnls.return_value.identify.return_value = create_mock_result(["Fe", "Mg", "Ca", "Na"])
    mock_alias.return_value.identify.return_value = create_mock_result(["Fe", "Mg", "Ca"])
    mock_comb.return_value.identify.return_value = create_mock_result(["Fe", "Mg"])
    mock_corr.return_value.identify.return_value = create_mock_result(["Fe"])
    
    hi = HybridIdentifier(
        atomic_db=MagicMock(),
        basis_library=MagicMock(),
        elements=["Fe", "Mg", "Ca", "Na"],
        quorum_threshold=2,
        mode="quorum"
    )
    
    res = hi.identify(np.array([1.0]), np.array([1.0]))
    detected = {e.element for e in res.detected_elements}
    
    assert "Fe" in detected
    assert "Mg" in detected
    assert "Ca" not in detected
    assert "Na" not in detected

def test_hybrid_identifier_union_1_of_3(mock_deps):
    mock_nnls, mock_alias, mock_comb, mock_corr = mock_deps
    
    mock_nnls.return_value.identify.return_value = create_mock_result(["Fe", "Mg", "Ca", "Na"])
    mock_alias.return_value.identify.return_value = create_mock_result(["Fe"])
    mock_comb.return_value.identify.return_value = create_mock_result(["Mg"])
    mock_corr.return_value.identify.return_value = create_mock_result(["Ca"])
    
    hi = HybridIdentifier(
        atomic_db=MagicMock(),
        basis_library=MagicMock(),
        elements=["Fe", "Mg", "Ca", "Na"],
        mode="union"
    )
    
    res = hi.identify(np.array([1.0]), np.array([1.0]))
    detected = {e.element for e in res.detected_elements}
    
    assert "Fe" in detected
    assert "Mg" in detected
    assert "Ca" in detected
    assert "Na" not in detected

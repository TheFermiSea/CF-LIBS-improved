import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from cflibs.inversion.identify.hybrid import HybridIdentifier
from cflibs.inversion.common.element_id import (
    TRACE_ELEMENTS, 
    get_default_algorithm, 
    ElementIdentification, 
    ElementIdentificationResult
)

@patch("cflibs.inversion.alias_identifier.ALIASIdentifier")
@patch("cflibs.inversion.comb_identifier.CombIdentifier")
@patch("cflibs.inversion.correlation_identifier.CorrelationIdentifier")
def test_hybrid_identifier_quorum(mock_corr_cls, mock_comb_cls, mock_alias_cls):
    mock_db = MagicMock()
    elements = ["Fe", "V"]
    
    # Setup mocks
    def make_res(detected_elements):
        all_elements = [
            ElementIdentification(el, el in detected_elements, 0.8, 0.8, 0, 0, [], [])
            for el in elements
        ]
        return ElementIdentificationResult(
            detected_elements=[e for e in all_elements if e.detected],
            rejected_elements=[e for e in all_elements if not e.detected],
            all_elements=all_elements,
            experimental_peaks=[],
            n_peaks=0,
            n_matched_peaks=0,
            n_unmatched_peaks=0,
            algorithm="test"
        )

    mock_alias_cls.return_value.identify.return_value = make_res(["Fe", "V"])
    mock_comb_cls.return_value.identify.return_value = make_res(["Fe"])
    mock_corr_cls.return_value.identify.return_value = make_res(["Fe", "V"])

    # Test quorum=2
    hybrid = HybridIdentifier(mock_db, elements, quorum=2)
    res = hybrid.identify(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    
    detected = {e.element for e in res.detected_elements}
    assert "Fe" in detected # 3 votes
    assert "V" in detected  # 2 votes (alias, correlation)
    
    # Test quorum=3
    hybrid.quorum = 3
    res = hybrid.identify(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    detected = {e.element for e in res.detected_elements}
    assert "Fe" in detected
    assert "V" not in detected # only 2 votes

def test_trace_elements_defaults():
    assert "V" in TRACE_ELEMENTS
    assert get_default_algorithm("V") == "hybrid"
    assert get_default_algorithm("Fe") == "alias"

def test_hybrid_identifier_union_mode():
    with patch("cflibs.inversion.alias_identifier.ALIASIdentifier") as mock_alias_cls, \
         patch("cflibs.inversion.comb_identifier.CombIdentifier") as mock_comb_cls, \
         patch("cflibs.inversion.correlation_identifier.CorrelationIdentifier") as mock_corr_cls:
        
        mock_db = MagicMock()
        elements = ["Au"]
        
        def make_res(detected_elements):
            all_elements = [
                ElementIdentification(el, el in detected_elements, 0.8, 0.8, 0, 0, [], [])
                for el in elements
            ]
            return ElementIdentificationResult(
                detected_elements=[e for e in all_elements if e.detected],
                rejected_elements=[e for e in all_elements if not e.detected],
                all_elements=all_elements,
                experimental_peaks=[],
                n_peaks=0,
                n_matched_peaks=0,
                n_unmatched_peaks=0,
                algorithm="test"
            )

        mock_alias_cls.return_value.identify.return_value = make_res([])
        mock_comb_cls.return_value.identify.return_value = make_res(["Au"])
        mock_corr_cls.return_value.identify.return_value = make_res([])

        # Test quorum=1 (union)
        hybrid = HybridIdentifier(mock_db, elements, quorum=1)
        res = hybrid.identify(np.array([1.0]), np.array([10.0]))
        assert "Au" in {e.element for e in res.detected_elements}

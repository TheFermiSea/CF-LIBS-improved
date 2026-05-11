import pytest
import numpy as np
from unittest.mock import MagicMock
from cflibs.inversion.identify.hybrid import HybridIdentifier
from cflibs.inversion.common.element_id import get_element_tier, ElementTier, get_default_identifier_config

def test_element_tiers():
    assert get_element_tier("Fe") == ElementTier.MAJOR
    assert get_element_tier("Si") == ElementTier.MAJOR
    assert get_element_tier("Li") == ElementTier.MINOR
    assert get_element_tier("Au") == ElementTier.TRACE
    assert get_element_tier("U") == ElementTier.TRACE

def test_default_config():
    trace_cfg = get_default_identifier_config("Au")
    assert trace_cfg["algorithm"] == "hybrid_quorum"
    assert trace_cfg["min_votes"] == 2
    
    major_cfg = get_default_identifier_config("Fe")
    assert major_cfg["algorithm"] == "alias"

@pytest.fixture
def mock_deps():
    db = MagicMock()
    lib = MagicMock()
    return db, lib

def test_hybrid_quorum_logic(mock_deps):
    db, lib = mock_deps
    elements = ["Fe", "Au"]
    
    # Setup mock results
    def make_mock_result(detected_elements):
        res = MagicMock()
        res.detected_elements = [MagicMock(element=e) for e in detected_elements]
        res.all_elements = [MagicMock(element=e, score=0.8) for e in elements]
        res.experimental_peaks = []
        res.n_peaks = 0
        res.n_matched_peaks = 0
        res.n_unmatched_peaks = 0
        return res

    # Fe: 3 votes, Au: 1 vote
    alias_res = make_mock_result(["Fe", "Au"])
    comb_res = make_mock_result(["Fe"])
    corr_res = make_mock_result(["Fe"])
    
    # Patch the imports inside identify
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("cflibs.inversion.alias_identifier.ALIASIdentifier", lambda **kwargs: MagicMock(identify=lambda w, i: alias_res))
        mp.setattr("cflibs.inversion.comb_identifier.CombIdentifier", lambda **kwargs: MagicMock(identify=lambda w, i: comb_res))
        mp.setattr("cflibs.inversion.correlation_identifier.CorrelationIdentifier", lambda **kwargs: MagicMock(identify=lambda w, i: corr_res))
        mp.setattr("cflibs.inversion.spectral_nnls_identifier.SpectralNNLSIdentifier", lambda **kwargs: MagicMock(identify=lambda w, i: make_mock_result(elements)))

        identifier = HybridIdentifier(db, lib, elements, min_votes=2, nnls_screening=False)
        wavelength = np.linspace(200, 800, 1000)
        intensity = np.random.rand(1000)
        
        result = identifier.identify(wavelength, intensity)
        
        detected = {e.element for e in result.detected_elements}
        assert "Fe" in detected
        assert "Au" not in detected # Only 1 vote (alias)
        
        # Check metadata
        au_id = next(e for e in result.all_elements if e.element == "Au")
        assert au_id.metadata["votes"] == 1
        assert au_id.metadata["alias_detected"] is True
        assert au_id.metadata["comb_detected"] is False
        assert au_id.metadata["correlation_detected"] is False

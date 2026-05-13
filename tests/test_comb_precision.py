import numpy as np
import pytest
from unittest.mock import MagicMock
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.inversion.element_id import is_element_detected

def test_tier2_detection_rules():
    """Verify that Mn/Na/K have stricter detection thresholds."""
    # Standard element (e.g., Fe)
    assert is_element_detected("Fe", score=0.13, n_active=2, min_score=0.12, min_active=2) is True
    
    # Mn with same score (should fail because Tier-2 min_score is 0.15)
    assert is_element_detected("Mn", score=0.13, n_active=2, min_score=0.12, min_active=2) is False
    
    # Mn with high score but only 1 line (should fail because Tier-2 min_active is 2)
    assert is_element_detected("Mn", score=0.20, n_active=1, min_score=0.12, min_active=1) is False
    
    # Mn with high score and 2 lines (should pass)
    assert is_element_detected("Mn", score=0.20, n_active=2, min_score=0.12, min_active=1) is True

def test_comb_identifier_default_parameters():
    """Ensure CombIdentifier defaults are tightened as requested."""
    db = MagicMock()
    ci = CombIdentifier(atomic_db=db)
    assert ci.min_correlation == 0.12
    assert ci.relative_threshold_scale == 2.0

@pytest.mark.parametrize("element", ["Mn", "Na", "K"])
def test_trace_element_fp_suppression(element):
    """Test that trace elements are suppressed when SNR is low or lines are few."""
    # Case: Score is above global minimum but below Tier-2 minimum
    assert is_element_detected(element, score=0.13, n_active=2, min_score=0.12, min_active=2) is False
    # Case: Score is high but only 1 line
    assert is_element_detected(element, score=0.25, n_active=1, min_score=0.12, min_active=1) is False

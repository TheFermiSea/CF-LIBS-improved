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
    """Ensure CombIdentifier defaults are set for improved recall.
    
    The defaults were lowered from the tightened PR #166 values to address
    the FN-bound issue (CF-LIBS-improved-5thd). The comb algorithm was
    producing F1=0.0303 due to excessive False Negatives (low recall).
    
    New defaults:
    - min_correlation: 0.05 (was 0.12) - lowers fingerprint detection threshold
    - tooth_activation_threshold: 0.3 (was 0.5) - lowers per-tooth activation threshold
    - max_lines_per_element: 50 (unchanged)
    """
    db = MagicMock()
    ci = CombIdentifier(atomic_db=db)
    # New relaxed defaults for improved recall
    assert ci.min_correlation == 0.05
    assert ci.tooth_activation_threshold == 0.3
    assert ci.max_lines_per_element == 50
    assert ci.relative_threshold_scale == 2.0

@pytest.mark.parametrize("element", ["Mn", "Na", "K"])
def test_trace_element_fp_suppression(element):
    """Test that trace elements are suppressed when SNR is low or lines are few.
    
    Tier-2 elements (Mn/Na/K) still have stricter per-tooth activation thresholds
    via the tier2_tooth_activation_threshold knob (default 0.7), which is applied
    via max(global, tier2) to ensure FP suppression without affecting non-Tier-2 elements.
    """
    # Case: Score is above global minimum but below Tier-2 minimum
    assert is_element_detected(element, score=0.13, n_active=2, min_score=0.12, min_active=2) is False
    # Case: Score is high but only 1 line
    assert is_element_detected(element, score=0.25, n_active=1, min_score=0.12, min_active=1) is False

import numpy as np
import math
from unittest.mock import MagicMock
from cflibs.inversion.identify.alias import ALIASIdentifier

def test_alias_default_thresholds_lowered():
    """Verify that the default thresholds have been lowered as requested."""
    db = MagicMock()
    identifier = ALIASIdentifier(db)
    # detection_threshold: 0.02 -> 0.01 (50% reduction)
    assert identifier.detection_threshold == 0.01
    # intensity_threshold_factor: 3.0 -> 2.0 (33% reduction)
    assert identifier.intensity_threshold_factor == 2.0

def test_alias_detection_logic_with_lowered_threshold():
    """Verify that the lower threshold allows detection of lower-confidence candidates."""
    db = MagicMock()
    identifier = ALIASIdentifier(db) # uses default 0.01
    
    # Case: CL = 0.015. 
    # With old threshold (0.02), this would be rejected.
    # With new threshold (0.01), this should be accepted (assuming N_expected >= 10).
    
    CL = 0.015
    N_expected = 10 # No adaptive multiplier (sqrt(10/10) = 1)
    
    adaptive_dt = identifier.detection_threshold
    if N_expected > 0 and N_expected < 10:
        adaptive_dt *= min(3.0, math.sqrt(10.0 / N_expected))
    
    detected = CL >= adaptive_dt
    assert detected is True, f"CL {CL} should be detected with threshold {identifier.detection_threshold}"

    # Verify it would have failed with old threshold
    old_threshold = 0.02
    assert CL < old_threshold

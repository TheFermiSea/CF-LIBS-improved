import numpy as np
import pytest
from unittest.mock import MagicMock
from cflibs.inversion.identify.alias import ALIASIdentifier
from cflibs.atomic.database import AtomicDatabase

def test_alias_default_thresholds():
    """
    Verify that the ALIASIdentifier default thresholds have been lowered
    as per the requirements for aa1100_substrate candidate surfacing.
    """
    db = MagicMock(spec=AtomicDatabase)
    identifier = ALIASIdentifier(db)
    
    # Verify lowered defaults
    assert identifier.intensity_threshold_factor == 2.0
    assert identifier.detection_threshold == 0.01
    assert identifier.boltzmann_r2_min == 0.6

def test_alias_threshold_override():
    """
    Verify that thresholds can still be overridden by the user.
    """
    db = MagicMock(spec=AtomicDatabase)
    identifier = ALIASIdentifier(
        db, 
        intensity_threshold_factor=5.0, 
        detection_threshold=0.1, 
        boltzmann_r2_min=0.95
    )
    
    assert identifier.intensity_threshold_factor == 5.0
    assert identifier.detection_threshold == 0.1
    assert identifier.boltzmann_r2_min == 0.95

def test_alias_initialization_validation():
    """
    Verify that invalid threshold values still raise errors.
    """
    db = MagicMock(spec=AtomicDatabase)
    with pytest.raises(ValueError, match="boltzmann_r2_min must be finite and in"):
        ALIASIdentifier(db, boltzmann_r2_min=1.5)

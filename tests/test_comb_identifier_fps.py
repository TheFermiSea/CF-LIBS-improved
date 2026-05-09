import numpy as np
import pytest
from unittest.mock import MagicMock
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition

@pytest.fixture
def mock_db():
    db = MagicMock(spec=AtomicDatabase)
    db.get_available_elements.return_value = ["Fe", "Mn", "Na", "K"]
    
    # Mock transitions for Mn (Tier-2)
    mn_transitions = [
        Transition(wavelength_nm=403.076, element="Mn", ionization_stage=1, A_ki=1e7, g_k=6, E_k_ev=3.07, E_i_ev=0.0),
        Transition(wavelength_nm=403.307, element="Mn", ionization_stage=1, A_ki=1e7, g_k=6, E_k_ev=3.07, E_i_ev=0.0),
        Transition(wavelength_nm=403.449, element="Mn", ionization_stage=1, A_ki=1e7, g_k=6, E_k_ev=3.07, E_i_ev=0.0),
        Transition(wavelength_nm=404.0, element="Mn", ionization_stage=1, A_ki=1e7, g_k=6, E_k_ev=3.07, E_i_ev=0.0),
    ]
    
    # Mock transitions for Fe (Major)
    fe_transitions = [
        Transition(wavelength_nm=372.0, element="Fe", ionization_stage=1, A_ki=1e8, g_k=1, E_k_ev=3.3, E_i_ev=0.0),
        Transition(wavelength_nm=373.0, element="Fe", ionization_stage=1, A_ki=1e8, g_k=1, E_k_ev=3.3, E_i_ev=0.0),
    ]
    
    def get_transitions(element, **kwargs):
        if element == "Mn": return mn_transitions
        if element == "Fe": return fe_transitions
        return []
        
    db.get_transitions.side_effect = get_transitions
    return db

def test_mn_fp_rejection(mock_db):
    """Test that Mn is rejected when only 2 weak lines match (N<3 requirement)."""
    identifier = CombIdentifier(mock_db, min_active_teeth=2)
    
    # Create a spectrum with no Mn lines (only noise and Fe lines)
    wavelength = np.linspace(350, 450, 10000)
    intensity = np.random.normal(0.01, 0.001, len(wavelength))
    
    # Add 2 weak 'fake' Mn peaks to trigger 2-line detection (which should now be rejected)
    # 403.076 and 403.307
    idx1 = np.argmin(np.abs(wavelength - 403.076))
    idx2 = np.argmin(np.abs(wavelength - 403.307))
    intensity[idx1-2:idx1+3] += 0.5
    intensity[idx2-2:idx2+3] += 0.5
    
    result = identifier.identify(wavelength, intensity)
    
    # Mn should be rejected because it needs 3 lines
    mn_id = next((e for e in result.all_elements if e.element == "Mn"), None)
    assert mn_id is not None
    assert mn_id.detected is False
    assert mn_id.n_matched_lines == 2

def test_mn_detection_with_three_lines(mock_db):
    """Test that Mn is detected when 3 lines match."""
    identifier = CombIdentifier(mock_db, min_active_teeth=2)
    
    wavelength = np.linspace(350, 450, 10000)
    intensity = np.random.normal(0.01, 0.001, len(wavelength))
    
    # Add 3 strong Mn peaks
    for wl in [403.076, 403.307, 403.449]:
        idx = np.argmin(np.abs(wavelength - wl))
        intensity[idx-2:idx+3] += 1.0
        
    result = identifier.identify(wavelength, intensity)
    
    mn_id = next((e for e in result.all_elements if e.element == "Mn"), None)
    assert mn_id is not None
    assert mn_id.detected is True
    assert mn_id.n_matched_lines >= 3

def test_default_thresholds():
    """Verify that default thresholds are tightened."""
    identifier = CombIdentifier(MagicMock())
    assert identifier.min_correlation == 0.10
    assert identifier.relative_threshold_scale == 1.5

import numpy as np
import pytest
from unittest.mock import MagicMock
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.atomic.structures import Transition

def test_mn_fp_rejection():
    """
    Verify that Mn (Tier-2 element) is rejected when only one line is present,
    even if min_active_teeth is set to 1.
    """
    # Mock atomic database with one Mn line and two Fe lines
    mock_db = MagicMock()
    
    mn_trans = Transition(
        element="Mn", ionization_stage=1, wavelength_nm=403.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0
    )
    fe_trans1 = Transition(
        element="Fe", ionization_stage=1, wavelength_nm=404.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0
    )
    fe_trans2 = Transition(
        element="Fe", ionization_stage=1, wavelength_nm=405.0,
        A_ki=1e8, g_k=6, g_i=4, E_k_ev=3.0, E_i_ev=0.0
    )
    
    def get_transitions(element, **kwargs):
        if element == "Mn":
            return [mn_trans]
        if element == "Fe":
            return [fe_trans1, fe_trans2]
        return []
        
    mock_db.get_transitions.side_effect = get_transitions
    mock_db.get_available_elements.return_value = ["Mn", "Fe"]
    
    # Create spectrum with a peak at Mn 403.0 and Fe 404.0, 405.0
    wavelength = np.linspace(400, 410, 1000)
    intensity = np.zeros_like(wavelength)
    
    def add_peak(wl, amp):
        idx = np.argmin(np.abs(wavelength - wl))
        # Use a wider window for the peak to ensure it's detected by the comb
        mask = np.abs(wavelength - wl) < 0.1
        intensity[mask] += amp * np.exp(-0.5 * ((wavelength[mask] - wl)/0.02)**2)

    add_peak(403.0, 100.0) # Mn line
    add_peak(404.0, 100.0) # Fe line 1
    add_peak(405.0, 100.0) # Fe line 2
    
    # Add some noise
    np.random.seed(42)
    intensity += np.random.normal(0, 1, len(intensity))
    
    # identifier with min_active_teeth=1 to test the override for Mn
    identifier = CombIdentifier(mock_db, min_active_teeth=1)
    result = identifier.identify(wavelength, intensity)
    
    detected_elements = [e.element for e in result.detected_elements]
    
    # Fe should be detected (2 lines)
    assert "Fe" in detected_elements
    
    # Mn should NOT be detected (only 1 line, and it's a Tier-2 element)
    assert "Mn" not in detected_elements

def test_na_k_fp_rejection():
    """Verify Na and K also require 2 lines."""
    mock_db = MagicMock()
    
    na_trans = Transition(
        element="Na", ionization_stage=1, wavelength_nm=589.0,
        A_ki=1e8, g_k=6, g_i=2, E_k_ev=3.0, E_i_ev=0.0
    )
    k_trans = Transition(
        element="K", ionization_stage=1, wavelength_nm=766.0,
        A_ki=1e8, g_k=6, g_i=2, E_k_ev=3.0, E_i_ev=0.0
    )
    
    def get_transitions(element, **kwargs):
        if element == "Na": return [na_trans]
        if element == "K": return [k_trans]
        return []
        
    mock_db.get_transitions.side_effect = get_transitions
    mock_db.get_available_elements.return_value = ["Na", "K"]
    
    wavelength = np.linspace(500, 800, 3000)
    intensity = np.zeros_like(wavelength)
    
    def add_peak(wl, amp):
        mask = np.abs(wavelength - wl) < 0.1
        intensity[mask] += amp * np.exp(-0.5 * ((wavelength[mask] - wl)/0.02)**2)

    add_peak(589.0, 100.0)
    add_peak(766.0, 100.0)
    
    identifier = CombIdentifier(mock_db, min_active_teeth=1)
    result = identifier.identify(wavelength, intensity)
    
    detected_elements = [e.element for e in result.detected_elements]
    assert "Na" not in detected_elements
    assert "K" not in detected_elements

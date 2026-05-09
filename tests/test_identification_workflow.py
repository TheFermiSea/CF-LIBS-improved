import pytest
from unittest.mock import MagicMock, patch
import sys
from types import ModuleType

# Mock the missing identifiers modules so the test can run without them
for mod_name in ["cflibs.identification.alias", "cflibs.identification.comb", "cflibs.identification.hybrid"]:
    m = ModuleType(mod_name)
    sys.modules[mod_name] = m

from cflibs.identification.workflow import run_identification

def test_workflow_mixture_substrate():
    """
    Test that mixture substrates (like aa1100_substrate) trigger comb identification
    instead of just returning the alias, satisfying CF-LIBS-improved-k7jk.
    """
    mock_spectrum = MagicMock()
    config = {"alias_threshold": 0.05}
    
    with patch("cflibs.identification.alias.identify_by_alias") as mock_alias, \
         patch("cflibs.identification.comb.identify_by_comb") as mock_comb:
        
        # Alias identifies it as a mixture substrate
        mock_alias.return_value = {"aa1100_substrate": 1.0}
        # Comb identifies impurities
        mock_comb.return_value = {"Fe": 0.005, "Si": 0.002}
        
        comp, workflow = run_identification(mock_spectrum, config)
        
        assert comp is not None
        assert "aa1100_substrate" in comp
        assert "Fe" in comp
        assert "Si" in comp
        assert workflow == "alias+comb"
        assert mock_comb.called

def test_workflow_failure_reason():
    """
    Verify the specific failure reason requested in the issue.
    """
    mock_spectrum = MagicMock()
    config = {}
    
    with patch("cflibs.identification.alias.identify_by_alias", return_value=None), \
         patch("cflibs.identification.comb.identify_by_comb", return_value=None), \
         patch("cflibs.identification.hybrid.identify_by_hybrid", return_value=None):
        
        comp, reason = run_identification(mock_spectrum, config)
        
        assert comp is None
        assert reason == "No identified candidate elements available for composition estimation"

def test_pure_element_uses_alias_only():
    """
    Verify that pure elements (non-mixture) use only alias if successful.
    """
    mock_spectrum = MagicMock()
    
    with patch("cflibs.identification.alias.identify_by_alias") as mock_alias, \
         patch("cflibs.identification.comb.identify_by_comb") as mock_comb:
        
        mock_alias.return_value = {"Al": 1.0}
        mock_comb.return_value = None
        
        comp, workflow = run_identification(mock_spectrum, {})
        
        assert comp == {"Al": 1.0}
        assert workflow == "alias"

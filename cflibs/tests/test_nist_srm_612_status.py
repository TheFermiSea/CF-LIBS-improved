import pytest
from cflibs.datasets.registry import DATASET_REGISTRY, EXCLUDED_DATASETS

def test_nist_srm_612_registration():
    """Verify nist_srm_612 is registered but correctly marked as pending spectra."""
    assert "nist_srm_612" in DATASET_REGISTRY
    entry = DATASET_REGISTRY["nist_srm_612"]
    
    assert entry["loader"] is None
    assert entry["status"] == "pending_spectra"
    assert "acquisition_plan" in entry
    
    actions_str = str(entry["acquisition_plan"]["actions"])
    assert "IRAP" in actions_str
    assert "U Malaga" in actions_str
    assert "ORDaR" in actions_str

def test_nist_srm_612_exclusion():
    """Ensure nist_srm_612 is explicitly excluded from accuracy gates for now."""
    assert "nist_srm_612" in EXCLUDED_DATASETS

def test_nist_srm_612_no_synthesis_only_directive():
    """Enforce the project directive against synthesis-only data for this ID."""
    entry = DATASET_REGISTRY["nist_srm_612"]
    directive = entry.get("acquisition_plan", {}).get("directive", "")
    assert "synthesis cannot be the sole data source" in directive.lower()
    assert "real measured spectra are required" in directive.lower()

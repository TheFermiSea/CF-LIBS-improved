import math
import pytest
from dataclasses import dataclass
from cflibs.inversion.common.element_id import get_wavelength_tolerance

@dataclass
class MockTransition:
    stark_width_nm: float = None

def test_wavelength_tolerance_stark_available():
    # R=5000, wl=500nm -> fwhm_inst = 0.1nm
    # stark = 0.05nm
    # expected = sqrt(0.1^2 + 0.05^2) = sqrt(0.01 + 0.0025) = sqrt(0.0125) approx 0.1118
    transition = MockTransition(stark_width_nm=0.05)
    tol = get_wavelength_tolerance(wavelength_nm=500.0, transition=transition, resolving_power=5000.0)
    expected = math.sqrt(0.1**2 + 0.05**2)
    assert math.isclose(tol, expected)

def test_wavelength_tolerance_stark_missing_fallback():
    # stark is None -> fallback to 0.05
    transition = MockTransition(stark_width_nm=None)
    tol = get_wavelength_tolerance(wavelength_nm=500.0, transition=transition, resolving_power=5000.0)
    assert tol == 0.05

def test_wavelength_tolerance_stark_zero_fallback():
    # stark is 0.0 -> fallback to 0.05
    transition = MockTransition(stark_width_nm=0.0)
    tol = get_wavelength_tolerance(wavelength_nm=500.0, transition=transition, resolving_power=5000.0)
    assert tol == 0.05

def test_wavelength_tolerance_no_transition_fallback():
    # transition is None -> fallback to 0.05
    tol = get_wavelength_tolerance(wavelength_nm=500.0, transition=None, resolving_power=5000.0)
    assert tol == 0.05

def test_wavelength_tolerance_custom_fallback():
    # Verify custom fallback is honored
    tol = get_wavelength_tolerance(wavelength_nm=500.0, transition=None, resolving_power=5000.0, fallback=0.02)
    assert tol == 0.02

import math
from dataclasses import dataclass
from typing import Optional

from cflibs.inversion.common.element_id import get_wavelength_tolerance


@dataclass
class MockTransition:
    """Mock matching the real ``Transition`` Stark fields (electron-impact FWHM
    at REF_NE=1e17, T=10000 K — the single-source-of-truth convention)."""

    stark_w: Optional[float] = None
    stark_alpha: Optional[float] = None


def test_wavelength_tolerance_stark_available():
    # R=5000, wl=500nm -> fwhm_inst = 0.1nm.
    # stark_w = 0.05 nm = the stored electron-impact FWHM at REF_NE=1e17,
    # T=10000K. The helper defaults to those exact Konjević reference
    # conditions, so omega_stark (FWHM) == 0.05 nm exactly (A4-CONV-2 fix; the
    # pre-fix runtime produced the x20-inflated 1.0 nm here).
    # Expected tol = sqrt(0.1^2 + 0.05^2).
    transition = MockTransition(stark_w=0.05)
    tol = get_wavelength_tolerance(
        wavelength_nm=500.0, transition=transition, resolving_power=5000.0
    )
    expected = math.sqrt(0.1**2 + 0.05**2)
    assert math.isclose(tol, expected, rel_tol=1e-6)


def test_wavelength_tolerance_stark_missing_fallback():
    # stark_w is None -> fallback to 0.05
    transition = MockTransition(stark_w=None)
    tol = get_wavelength_tolerance(
        wavelength_nm=500.0, transition=transition, resolving_power=5000.0
    )
    assert tol == 0.05


def test_wavelength_tolerance_stark_zero_fallback():
    # stark_w is 0.0 -> fallback to 0.05
    transition = MockTransition(stark_w=0.0)
    tol = get_wavelength_tolerance(
        wavelength_nm=500.0, transition=transition, resolving_power=5000.0
    )
    assert tol == 0.05


def test_wavelength_tolerance_no_transition_fallback():
    # transition is None -> fallback to 0.05
    tol = get_wavelength_tolerance(wavelength_nm=500.0, transition=None, resolving_power=5000.0)
    assert tol == 0.05


def test_wavelength_tolerance_custom_fallback():
    # Verify custom fallback is honored
    tol = get_wavelength_tolerance(
        wavelength_nm=500.0, transition=None, resolving_power=5000.0, fallback=0.02
    )
    assert tol == 0.02

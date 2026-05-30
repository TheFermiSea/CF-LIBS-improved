import math
from dataclasses import dataclass
from typing import Optional

from cflibs.inversion.common.element_id import get_wavelength_tolerance


@dataclass
class MockTransition:
    """Mock matching the real ``Transition`` Stark fields (HWHM at REF_NE=1e16, T=10000 K)."""

    stark_w: Optional[float] = None
    stark_alpha: Optional[float] = None


def test_wavelength_tolerance_stark_available():
    # R=5000, wl=500nm -> fwhm_inst = 0.1nm.
    # stark_w = 0.05 nm HWHM at REF_NE=1e16, T=10000K. The helper scales to the
    # Konjević reference (n_e=1e17, T=10000K) by default — so HWHM scales by 10x
    # to 0.5 nm and FWHM is 1.0 nm. Expected tol = sqrt(0.1^2 + 1.0^2).
    transition = MockTransition(stark_w=0.05)
    tol = get_wavelength_tolerance(
        wavelength_nm=500.0, transition=transition, resolving_power=5000.0
    )
    expected = math.sqrt(0.1**2 + 1.0**2)
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

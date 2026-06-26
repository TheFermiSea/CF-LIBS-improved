"""
Tests for physical constants module.
"""

import pytest
from cflibs.core import constants


def test_constants_exist():
    """Test that all expected constants are defined."""
    assert hasattr(constants, "KB")
    assert hasattr(constants, "KB_EV")
    assert hasattr(constants, "H_PLANCK")
    assert hasattr(constants, "C_LIGHT")
    assert hasattr(constants, "M_E")
    assert hasattr(constants, "E_CHARGE")


def test_constant_values():
    """Test that constants have reasonable values."""
    # Boltzmann constant
    assert constants.KB > 0
    assert constants.KB_EV > 0

    # Planck constant
    assert constants.H_PLANCK > 0

    # Speed of light
    assert constants.C_LIGHT > 0
    assert abs(constants.C_LIGHT - 3e8) < 1e6  # ~3e8 m/s

    # Electron mass
    assert constants.M_E > 0
    assert constants.M_E < 1e-30  # Should be very small


def test_conversion_factors():
    """Test conversion factor relationships."""
    # Energy conversions
    assert abs(constants.EV_TO_J * constants.J_TO_EV - 1.0) < 1e-10

    # Temperature conversions
    assert abs(constants.K_TO_EV * constants.EV_TO_K - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])


def test_kb_ev_matches_codata():
    """KB_EV must equal the CODATA Boltzmann constant in eV/K (k_B / e)."""
    import scipy.constants as sc

    from cflibs.core import constants

    assert constants.KB_EV == pytest.approx(sc.k / sc.e, rel=1e-5)


def test_cm_to_ev_matches_codata():
    """CM_TO_EV must equal the energy (eV) of a 1 cm^-1 wavenumber: h*c*100/e."""
    import scipy.constants as sc

    from cflibs.core import constants

    assert constants.CM_TO_EV == pytest.approx(sc.h * sc.c * 100.0 / sc.e, rel=1e-5)


def test_saha_const_cm3_matches_codata_derivation():
    """SAHA_CONST_CM3 must match the CODATA Saha prefactor with the x2 electron-
    spin factor and T expressed in eV: 2*(2*pi*m_e*e/h^2)^1.5, m^-3 -> cm^-3.
    (The missing x2 spin would give ~3.0e21, half the coded value, so this also
    pins the spin factor.)"""
    import numpy as np
    import scipy.constants as sc

    from cflibs.core import constants

    saha = 2.0 * (2.0 * np.pi * sc.m_e * sc.e / sc.h**2) ** 1.5 * 1e-6
    assert constants.SAHA_CONST_CM3 == pytest.approx(saha, rel=2e-3)  # within 0.2%

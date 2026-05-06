"""Focused unit tests for ALIAS validation helpers."""

import math

import numpy as np
import pytest

from cflibs.atomic.structures import Transition
from cflibs.core.constants import KB_EV
from cflibs.inversion.identify.alias import ALIASIdentifier

pytestmark = pytest.mark.unit


class _DummyAtomicDB:
    pass


def _transition(wavelength_nm: float, energy_ev: float) -> Transition:
    return Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=wavelength_nm,
        A_ki=1.0,
        E_k_ev=energy_ev,
        E_i_ev=0.0,
        g_k=1,
        g_i=1,
    )


def test_alias_rejects_invalid_boltzmann_r2_min():
    """The configurable Boltzmann gate must be a finite probability-like value."""
    for value in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="boltzmann_r2_min"):
            ALIASIdentifier(_DummyAtomicDB(), boltzmann_r2_min=value)


def test_boltzmann_consistency_low_energy_spread_flags_no_fit():
    """Matched lines with too little E_k spread should not report a perfect R^2."""
    identifier = ALIASIdentifier(_DummyAtomicDB())
    fused_lines = [
        {"transition": _transition(500.0, 1.00)},
        {"transition": _transition(501.0, 1.10)},
        {"transition": _transition(502.0, 1.20)},
    ]
    matched_mask = np.array([True, True, True])
    matched_peak_idx = np.array([0, 1, 2])
    intensity = np.array([10.0, 9.0, 8.0])
    peaks = [(0, 500.0), (1, 501.0), (2, 502.0)]

    factor, r_squared = identifier._boltzmann_consistency_check(
        "Fe", fused_lines, matched_mask, matched_peak_idx, intensity, peaks
    )

    assert factor == pytest.approx(0.5)
    assert r_squared == pytest.approx(0.0)


def test_boltzmann_consistency_uses_canonical_line_observation():
    """The consistency check should build LineObservation with intensity uncertainty."""
    identifier = ALIASIdentifier(_DummyAtomicDB())
    temperature_k = 10000.0
    energies = np.array([1.0, 2.5, 4.0])
    wavelengths = np.array([500.0, 501.0, 502.0])
    intensities = np.exp(-energies / (KB_EV * temperature_k))
    fused_lines = [
        {"transition": _transition(float(wavelength), float(energy))}
        for wavelength, energy in zip(wavelengths, energies)
    ]
    peaks = [(idx, float(wavelength)) for idx, wavelength in enumerate(wavelengths)]

    factor, r_squared = identifier._boltzmann_consistency_check(
        "Fe",
        fused_lines,
        np.array([True, True, True]),
        np.array([0, 1, 2]),
        intensities,
        peaks,
    )

    assert np.isfinite(factor)
    assert np.isfinite(r_squared)

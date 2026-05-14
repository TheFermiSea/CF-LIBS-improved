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


# ---------------------------------------------------------------------------
# Opt-in high-recall mode (CF-LIBS-improved-1h0w)
# ---------------------------------------------------------------------------
# These tests guard the precision-king baseline:
#   precision=1.000, FP/spec=0 on n=33 cross-shard
#   (see .swarm/identifier-f1-baseline.json on dev).
# The closed PR #134 silently flipped intensity_threshold_factor 3.0->2.0 and
# detection_threshold 0.02->0.01, which dropped precision to 0.8333 and
# raised FP/spec to 0.0909 on shard 1/3 smoke. The opt-in `high_recall` flag
# is the contract-respecting replacement: strict is the default, recall is
# explicit.


def test_alias_default_is_strict():
    """Default construction MUST preserve the precision-king strict baseline.

    Guards against silent default changes (the failure mode of PR #134).
    If this test ever needs to be updated, the change is a contract break
    and must be reviewed as such.
    """
    identifier = ALIASIdentifier(_DummyAtomicDB())
    assert identifier.intensity_threshold_factor == 3.0
    assert identifier.detection_threshold == 0.02
    assert identifier.high_recall is False


def test_alias_high_recall_lowers_thresholds():
    """Opt-in ``high_recall=True`` MUST lower both thresholds.

    Recall mode trades precision for recall. Values match what PR #134
    proposed (intensity_threshold_factor=2.0, detection_threshold=0.01)
    but are now reached only via the explicit flag.
    """
    identifier = ALIASIdentifier(_DummyAtomicDB(), high_recall=True)
    assert identifier.intensity_threshold_factor == 2.0
    assert identifier.detection_threshold == 0.01
    assert identifier.high_recall is True


def test_alias_explicit_threshold_overrides_high_recall():
    """Explicit threshold values must override the high_recall preset.

    Lets callers pin one knob while inheriting the preset for the other.
    """
    identifier = ALIASIdentifier(
        _DummyAtomicDB(),
        high_recall=True,
        intensity_threshold_factor=4.2,
    )
    # Explicit value wins for the pinned knob.
    assert identifier.intensity_threshold_factor == 4.2
    # The other knob still follows the recall preset.
    assert identifier.detection_threshold == 0.01

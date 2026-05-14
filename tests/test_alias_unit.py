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


# ---------------------------------------------------------------------------
# Opt-in adaptive Boltzmann R^2 gate (CF-LIBS-improved-ftp1)
# ---------------------------------------------------------------------------
# These tests exercise the three-mode gate via the ``_r2_gate_rejects``
# helper that ``identify`` calls on every candidate with N_matched >= 3.
# The helper isolates the gate decision from the rest of the pipeline so
# the byte-identical "fixed" default and the cold-T relaxation are
# independently verifiable without building a full synthetic spectrum.
# The diagnosis behind ``adaptive_t`` is PR #172 (Vrabel universal-miss
# root cause): cold plasma (T < ~5500 K) yields short effective E_k
# spans on real-but-real-element line sets, which depresses R^2 below
# the static 0.85 floor even though the identification is physically
# correct. The "fixed" default preserves the precision=1.000 / FP=0
# baseline on n=33 cross-shard.


def test_r2_gate_fixed_mode_byte_identical():
    """Default ``r2_gate_mode="fixed"`` MUST be byte-identical to the
    historical static-0.85 gate.

    Guards the precision-king baseline (precision=1.000, FP/spec=0 on
    n=33 cross-shard): default construction and explicit ``"fixed"``
    must produce the same rejection decision on the same inputs.
    """
    default_id = ALIASIdentifier(_DummyAtomicDB())
    explicit_id = ALIASIdentifier(_DummyAtomicDB(), r2_gate_mode="fixed")

    # Mode attribute exposed for downstream inspection.
    assert default_id.r2_gate_mode == "fixed"
    assert explicit_id.r2_gate_mode == "fixed"

    # Below the 0.85 floor: both reject.
    for boltz_r2 in (0.0, 0.3, 0.5, 0.84):
        assert default_id._r2_gate_rejects(boltz_r2, N_matched=5) is True
        assert explicit_id._r2_gate_rejects(boltz_r2, N_matched=5) is True

    # At/above the floor: both accept.
    for boltz_r2 in (0.85, 0.9, 1.0):
        assert default_id._r2_gate_rejects(boltz_r2, N_matched=5) is False
        assert explicit_id._r2_gate_rejects(boltz_r2, N_matched=5) is False

    # N_matched < 3 short-circuits in both modes (the upstream
    # "min 3 matches" gate handles that case; this gate is a no-op).
    assert default_id._r2_gate_rejects(0.0, N_matched=2) is False
    assert explicit_id._r2_gate_rejects(0.0, N_matched=2) is False


def test_r2_gate_adaptive_t_admits_cold_plasma_with_moderate_r2():
    """Vrabel-style cold plasma: T < threshold AND r2 in [0.3, 0.85).

    Under ``r2_gate_mode="adaptive_t"`` the cold-T relaxation activates
    and a moderate R^2 (0.4) clears the gate. Under the default
    ``"fixed"`` mode the same R^2 is rejected. The element is
    constructed with N_matched=5 and ``boltz_r2 = 0.4`` per the plan.
    """
    fixed_id = ALIASIdentifier(_DummyAtomicDB())
    adaptive_id = ALIASIdentifier(
        _DummyAtomicDB(),
        r2_gate_mode="adaptive_t",
        r2_gate_t_quality_threshold=5500.0,
    )

    # Vrabel-style cold plasma: monkey-patch the runtime temperature
    # estimate the way identify() would have set it via
    # _estimate_plasma_temperature.
    fixed_id._estimated_T = 4000.0
    adaptive_id._estimated_T = 4000.0

    # Fixed mode: r2=0.4 < boltzmann_r2_min=0.85 → reject.
    assert fixed_id._r2_gate_rejects(0.4, N_matched=5) is True
    # Adaptive_t mode: cold plasma + r2 above the 0.3 cold floor → admit.
    assert adaptive_id._r2_gate_rejects(0.4, N_matched=5) is False


def test_r2_gate_adaptive_t_rejects_warm_plasma_with_low_r2():
    """BHVO-2-style warm plasma: adaptive_t MUST NOT loosen the gate.

    For T >= threshold the cold-plasma relaxation does not fire and
    both modes apply the strict 0.85 floor. Guards against the gate
    silently degrading warm-plasma precision.
    """
    fixed_id = ALIASIdentifier(_DummyAtomicDB())
    adaptive_id = ALIASIdentifier(
        _DummyAtomicDB(),
        r2_gate_mode="adaptive_t",
        r2_gate_t_quality_threshold=5500.0,
    )

    # Warm plasma — above the cold-T threshold.
    fixed_id._estimated_T = 7000.0
    adaptive_id._estimated_T = 7000.0

    # Both modes reject r2=0.4 < 0.85.
    assert fixed_id._r2_gate_rejects(0.4, N_matched=5) is True
    assert adaptive_id._r2_gate_rejects(0.4, N_matched=5) is True

    # Edge case: exactly at the threshold — cold-T branch does not fire
    # (the predicate is strict ``T < threshold``), so the strict gate
    # still applies and r2=0.4 is rejected.
    adaptive_id._estimated_T = 5500.0
    assert adaptive_id._r2_gate_rejects(0.4, N_matched=5) is True


def test_r2_gate_disabled_admits_everything():
    """``r2_gate_mode="disabled"`` MUST bypass the gate entirely.

    Control-cell measurement for the sweep — disabling the gate should
    let any boltz_r2 through, including pathological zeros, regardless
    of plasma temperature.
    """
    identifier = ALIASIdentifier(_DummyAtomicDB(), r2_gate_mode="disabled")

    # No matter the temperature, the gate never rejects.
    for est_T in (4000.0, 7000.0, 12000.0):
        identifier._estimated_T = est_T
        for boltz_r2 in (0.0, 0.4, 0.5, 0.84, 0.85, 1.0):
            assert identifier._r2_gate_rejects(boltz_r2, N_matched=5) is False

    # And with _estimated_T unset (None) the disabled gate still passes.
    identifier._estimated_T = None
    assert identifier._r2_gate_rejects(0.0, N_matched=5) is False


def test_r2_gate_validation():
    """Constructor MUST validate ``r2_gate_mode`` and ``r2_gate_t_quality_threshold``.

    Invalid mode strings and non-positive thresholds raise ValueError
    before any state is committed to the instance.
    """
    # Invalid mode name.
    with pytest.raises(ValueError, match="r2_gate_mode"):
        ALIASIdentifier(_DummyAtomicDB(), r2_gate_mode="bogus")

    # Non-positive threshold rejects.
    with pytest.raises(ValueError, match="r2_gate_t_quality_threshold"):
        ALIASIdentifier(_DummyAtomicDB(), r2_gate_t_quality_threshold=-1.0)
    with pytest.raises(ValueError, match="r2_gate_t_quality_threshold"):
        ALIASIdentifier(_DummyAtomicDB(), r2_gate_t_quality_threshold=0.0)

    # Non-finite threshold rejects.
    with pytest.raises(ValueError, match="r2_gate_t_quality_threshold"):
        ALIASIdentifier(_DummyAtomicDB(), r2_gate_t_quality_threshold=math.inf)
    with pytest.raises(ValueError, match="r2_gate_t_quality_threshold"):
        ALIASIdentifier(_DummyAtomicDB(), r2_gate_t_quality_threshold=math.nan)

    # All three valid modes construct without error.
    for mode in ("fixed", "adaptive_t", "disabled"):
        ALIASIdentifier(_DummyAtomicDB(), r2_gate_mode=mode)

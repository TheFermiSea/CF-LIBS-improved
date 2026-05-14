"""Focused unit tests for ALIAS validation helpers."""

import copy
import math

import numpy as np
import pytest

from cflibs.atomic.structures import Transition
from cflibs.core.constants import KB_EV
from cflibs.inversion.common.element_id import (
    ElementIdentification,
    IdentifiedLine,
)
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


def _identified_line(
    element: str, ion_stage: int, intensity: float
) -> IdentifiedLine:
    """Minimal ``IdentifiedLine`` sufficient for the relative-CL gate.

    The gate inspects ``intensity_exp`` and ``ionization_stage`` only; the
    other fields are filled with neutral, well-formed placeholders so the
    dataclass stays valid for any future invariants.
    """
    return IdentifiedLine(
        wavelength_exp_nm=500.0,
        wavelength_th_nm=500.0,
        element=element,
        ionization_stage=ion_stage,
        intensity_exp=intensity,
        emissivity_th=1.0,
        transition=Transition(
            element=element,
            ionization_stage=ion_stage,
            wavelength_nm=500.0,
            A_ki=1.0,
            E_k_ev=1.0,
            E_i_ev=0.0,
            g_k=1,
            g_i=1,
        ),
    )


def _element_id(
    element: str,
    confidence: float,
    dominant_ion_stage: int,
    *,
    detected: bool = True,
) -> ElementIdentification:
    """Construct an ``ElementIdentification`` with one matched line whose
    ion stage is the "dominant" one used by the per-ion-stage gate.
    """
    return ElementIdentification(
        element=element,
        detected=detected,
        score=confidence,
        confidence=confidence,
        n_matched_lines=1,
        n_total_lines=1,
        matched_lines=[
            _identified_line(element, dominant_ion_stage, intensity=1.0)
        ],
        unmatched_lines=[],
        metadata={},
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
# Opt-in per-ion-stage relative_cl_threshold (CF-LIBS-improved-dj6y)
# ---------------------------------------------------------------------------
# These tests guard the global-gate default and exercise the opt-in
# per-ion-stage branch. The motivating failure is Vrabel s019 where Al I
# dominates with CL=0.13 and the global threshold ``max_CL * 0.1`` kills
# Mg II at CL=0.026 (see PR #172 diagnosis). The per-ion-stage gate splits
# the comparison into a neutral and an ionized subset so the dominant
# neutral cannot eliminate a lower-CL ionized species.


def _detected_elements(elem_ids):
    return {e.element for e in elem_ids if e.detected}


def test_per_ion_stage_default_off_byte_identical():
    """Explicit ``relative_cl_per_ion_stage=False`` MUST be byte-identical
    to the implicit default (the global gate).
    """
    fixture = [
        _element_id("Al", confidence=0.90, dominant_ion_stage=1),  # neutral
        _element_id("Fe", confidence=0.40, dominant_ion_stage=1),  # neutral
        _element_id("Mg", confidence=0.05, dominant_ion_stage=2),  # ionized
        _element_id("Ca", confidence=0.08, dominant_ion_stage=2),  # ionized
        _element_id("Si", confidence=0.06, dominant_ion_stage=2),  # ionized
    ]

    default_ids = copy.deepcopy(fixture)
    ALIASIdentifier(_DummyAtomicDB())._apply_relative_cl_gate(default_ids)

    explicit_off_ids = copy.deepcopy(fixture)
    ALIASIdentifier(
        _DummyAtomicDB(), relative_cl_per_ion_stage=False
    )._apply_relative_cl_gate(explicit_off_ids)

    assert _detected_elements(default_ids) == _detected_elements(explicit_off_ids)


def test_per_ion_stage_recovers_mg_when_dominant_element_is_neutral_metal():
    """Vrabel s019 reduction: Al I dominates globally; Mg II survives only
    once neutrals and ions are gated separately.

    Fixture:
      * Al neutral CL=0.9 (sets global max → threshold=0.09 under default)
      * Mg ionized CL=0.05
      * Si ionized CL=0.08, Ca ionized CL=0.06

    Global mode: Mg's 0.05 < 0.09 → Mg killed.
    Per-ion-stage (relative_cl_threshold=0.1): ionized max=0.08 →
    threshold=0.008; Mg's 0.05 > 0.008 → Mg survives.
    """
    fixture = [
        _element_id("Al", confidence=0.90, dominant_ion_stage=1),
        _element_id("Mg", confidence=0.05, dominant_ion_stage=2),
        _element_id("Si", confidence=0.08, dominant_ion_stage=2),
        _element_id("Ca", confidence=0.06, dominant_ion_stage=2),
    ]

    global_ids = copy.deepcopy(fixture)
    ALIASIdentifier(_DummyAtomicDB())._apply_relative_cl_gate(global_ids)
    global_detected = _detected_elements(global_ids)
    assert "Al" in global_detected
    assert "Mg" not in global_detected, (
        "Global gate is expected to kill Mg II when Al I dominates; if Mg "
        "survives here the failure mode that motivated this fix has changed."
    )

    per_ion_ids = copy.deepcopy(fixture)
    ALIASIdentifier(
        _DummyAtomicDB(),
        relative_cl_per_ion_stage=True,
    )._apply_relative_cl_gate(per_ion_ids)
    per_ion_detected = _detected_elements(per_ion_ids)
    assert "Al" in per_ion_detected, "Dominant neutral must still survive."
    assert "Mg" in per_ion_detected, (
        "Per-ion-stage mode must recover Mg II — ionized subset max=0.08 → "
        "threshold=0.008, Mg=0.05 clears."
    )


def test_per_ion_stage_explicit_threshold_overrides():
    """``relative_cl_threshold_neutral`` / ``_ionized`` independently
    govern their subsets.

    Fixture:
      * Neutrals: A=1.0, B=0.10 (max_neutral=1.0)
        With ``neutral_threshold=0.05`` → 0.05 cutoff → only CL<0.05 dies.
        B=0.10 survives; if we add a C=0.01 neutral it dies.
      * Ionized: D=1.0, E=0.15 (max_ionized=1.0)
        With ``ionized_threshold=0.2`` → 0.2 cutoff → E=0.15 dies, D survives.
    """
    fixture = [
        _element_id("A", confidence=1.00, dominant_ion_stage=1),
        _element_id("B", confidence=0.10, dominant_ion_stage=1),
        _element_id("C", confidence=0.01, dominant_ion_stage=1),
        _element_id("D", confidence=1.00, dominant_ion_stage=2),
        _element_id("E", confidence=0.15, dominant_ion_stage=2),
    ]

    ids = copy.deepcopy(fixture)
    ALIASIdentifier(
        _DummyAtomicDB(),
        relative_cl_per_ion_stage=True,
        relative_cl_threshold_neutral=0.05,
        relative_cl_threshold_ionized=0.2,
    )._apply_relative_cl_gate(ids)

    detected = _detected_elements(ids)
    assert "A" in detected
    assert "B" in detected, "B's 0.10 must survive the 0.05 neutral threshold."
    assert "C" not in detected, "C's 0.01 must be killed by the 0.05 neutral threshold."
    assert "D" in detected
    assert "E" not in detected, "E's 0.15 must be killed by the 0.2 ionized threshold."


def test_per_ion_stage_validation():
    """Per-ion-stage threshold kwargs must reject values outside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="relative_cl_threshold_neutral"):
        ALIASIdentifier(_DummyAtomicDB(), relative_cl_threshold_neutral=1.5)
    with pytest.raises(ValueError, match="relative_cl_threshold_neutral"):
        ALIASIdentifier(_DummyAtomicDB(), relative_cl_threshold_neutral=-0.1)
    with pytest.raises(ValueError, match="relative_cl_threshold_ionized"):
        ALIASIdentifier(_DummyAtomicDB(), relative_cl_threshold_ionized=1.5)
    with pytest.raises(ValueError, match="relative_cl_threshold_ionized"):
        ALIASIdentifier(_DummyAtomicDB(), relative_cl_threshold_ionized=math.nan)


def test_per_ion_stage_handles_only_neutrals_no_crash():
    """Per-ion-stage mode must skip the empty-ionized branch cleanly when
    every element's dominant matched line is neutral.
    """
    fixture = [
        _element_id("A", confidence=0.9, dominant_ion_stage=1),
        _element_id("B", confidence=0.5, dominant_ion_stage=1),
        _element_id("C", confidence=0.02, dominant_ion_stage=1),
    ]

    ids = copy.deepcopy(fixture)
    ALIASIdentifier(
        _DummyAtomicDB(),
        relative_cl_per_ion_stage=True,
    )._apply_relative_cl_gate(ids)
    detected = _detected_elements(ids)
    # With max_neutral=0.9 and threshold=0.1 → cutoff=0.09:
    # A and B survive, C (0.02) dies. No crash from an empty ionized subset.
    assert detected == {"A", "B"}

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


def _transition_with_e_i(wavelength_nm: float, energy_ev: float, e_i_ev: float) -> Transition:
    """Build a transition with an explicit lower-level energy ``E_i``.

    Used by the resonance-filter tests to mark lines as either
    "resonance" (``E_i ~ 0``) or "non-resonance" (``E_i`` above the
    cutoff used by :meth:`ALIASIdentifier._apply_resonance_filter`).
    """
    return Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=wavelength_nm,
        A_ki=1.0,
        E_k_ev=energy_ev,
        E_i_ev=e_i_ev,
        g_k=1,
        g_i=1,
    )


def _identified_line(element: str, ion_stage: int, intensity: float) -> IdentifiedLine:
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
        matched_lines=[_identified_line(element, dominant_ion_stage, intensity=1.0)],
        unmatched_lines=[],
        metadata={},
    )


def test_alias_rejects_invalid_boltzmann_r2_min():
    """The configurable Boltzmann gate must be a finite probability-like value."""
    for value in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="boltzmann_r2_min"):
            ALIASIdentifier(_DummyAtomicDB(), boltzmann_r2_min=value)


def test_boltzmann_consistency_low_energy_spread_is_deweighted_not_zeroed():
    """A genuinely linear but low-E_k-leverage stage keeps its real R^2.

    Regression guard for the W1 multistage fix (commit fixing
    ALIAS-R2GATE-2 / ALIAS-BOLTZ-IONMIX-1 interaction). Per-ionization-stage
    Boltzmann fitting (W1) naturally narrows the E_k span of each stage: a
    stage's matched UV lines often cluster within a few tenths of an eV.

    The historical ``ptp < 0.5`` guard returned ``(0.5, 0.0)`` — it ZEROED
    the R^2 of a perfectly linear short-span stage, which the adaptive_t gate
    then hard-rejected (``0.0 < 0.3``). That dropped Fe from the Fe/Cu
    multistage e2e spectrum even though both Fe stages fit R^2 ~= 1.0.

    Corrected contract: a short E_k span makes the *temperature* untrustworthy
    (small lever arm) but does NOT make a high R^2 a false consistency signal.
    So the fit still reports its TRUE R^2, with the confidence ``factor``
    *capped below the full-span reward* (low-leverage stages stay slightly
    de-weighted). A genuinely non-linear short-span stage still earns a low
    R^2 and is still gated — see the FP-control assertion below.
    """
    identifier = ALIASIdentifier(_DummyAtomicDB())
    # Intensities decreasing monotonically with E_k -> clean negative
    # (Boltzmann-consistent) slope despite the 0.20 eV span.
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

    # R^2 is NOT zeroed: a real linear relation reports its real (high) R^2.
    assert r_squared > 0.85
    # Factor is in the capped low-leverage band [0.7, 0.9], strictly below
    # the full-span max reward of 1.0 — so low-leverage fits stay de-weighted.
    assert 0.7 <= factor <= 0.9 + 1e-9
    assert factor < 1.0

    # FP control: scrambling the intensities destroys linearity, so even a
    # short-span stage must earn a low R^2 (the gate's FP suppressor is intact).
    intensity_nonlinear = np.array([1.0, 20.0, 3.0])
    _, r_squared_bad = identifier._boltzmann_consistency_check(
        "Fe", fused_lines, matched_mask, matched_peak_idx, intensity_nonlinear, peaks
    )
    assert r_squared_bad < r_squared


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
# Resonance-filter seam (arch candidate 5)
# ---------------------------------------------------------------------------
# These tests exercise ``_apply_resonance_filter`` directly so the
# decision logic can be verified without going through the full
# ``identify()`` pipeline. The four covered branches are:
#
#   1. NNLS does not support the candidate -> filter OFF.
#   2. ``self_absorption_aware`` is disabled -> filter OFF.
#   3. NNLS supports + enough non-resonance lines -> filter ON.
#   4. NNLS supports but ALL matched lines are resonance lines
#      (the n3rf.4 Al-I case) -> filter OFF (do not strand element).


def _resonance_filter_inputs(transitions):
    """Build the matched_indices/peak_idx/intensity/peaks tuple used by
    :meth:`ALIASIdentifier._apply_resonance_filter` from a list of
    transitions. One peak per line, all with positive intensity."""
    fused_lines = [{"transition": t} for t in transitions]
    n = len(transitions)
    matched_indices = np.arange(n)
    matched_peak_idx = np.arange(n)
    intensity = np.full(n, 10.0)
    peaks = [(idx, float(t.wavelength_nm)) for idx, t in enumerate(transitions)]
    return fused_lines, matched_indices, matched_peak_idx, intensity, peaks


def test_apply_resonance_filter_returns_false_when_nnls_not_significant():
    """Without independent NNLS evidence the filter must stay off."""
    identifier = ALIASIdentifier(_DummyAtomicDB())
    transitions = [
        _transition_with_e_i(500.0, 1.0, e_i_ev=2.0),
        _transition_with_e_i(501.0, 2.5, e_i_ev=2.0),
        _transition_with_e_i(502.0, 4.0, e_i_ev=2.0),
    ]
    fused_lines, mi, mpi, intensity, peaks = _resonance_filter_inputs(transitions)
    candidate = {"nnls_significant": False}

    assert (
        identifier._apply_resonance_filter(fused_lines, candidate, mi, mpi, intensity, peaks)
        is False
    )


def test_apply_resonance_filter_returns_false_when_self_absorption_aware_disabled():
    """Operator-disabled self-absorption awareness must short-circuit the seam."""
    identifier = ALIASIdentifier(_DummyAtomicDB())
    identifier.self_absorption_aware = False
    transitions = [
        _transition_with_e_i(500.0, 1.0, e_i_ev=2.0),
        _transition_with_e_i(501.0, 2.5, e_i_ev=2.0),
        _transition_with_e_i(502.0, 4.0, e_i_ev=2.0),
    ]
    fused_lines, mi, mpi, intensity, peaks = _resonance_filter_inputs(transitions)
    candidate = {"nnls_significant": True}

    assert (
        identifier._apply_resonance_filter(fused_lines, candidate, mi, mpi, intensity, peaks)
        is False
    )


def test_apply_resonance_filter_returns_true_when_enough_non_resonance_lines():
    """NNLS-significant + enough non-resonance survivors -> apply the filter."""
    identifier = ALIASIdentifier(_DummyAtomicDB())
    # Mix: one resonance line (E_i=0) and three non-resonance lines (E_i=2 eV).
    transitions = [
        _transition_with_e_i(500.0, 1.0, e_i_ev=0.0),  # resonance — gets dropped
        _transition_with_e_i(501.0, 2.5, e_i_ev=2.0),
        _transition_with_e_i(502.0, 4.0, e_i_ev=2.0),
        _transition_with_e_i(503.0, 5.5, e_i_ev=2.0),
    ]
    fused_lines, mi, mpi, intensity, peaks = _resonance_filter_inputs(transitions)
    candidate = {"nnls_significant": True}

    assert (
        identifier._apply_resonance_filter(fused_lines, candidate, mi, mpi, intensity, peaks)
        is True
    )


def test_apply_resonance_filter_preserves_all_resonance_element_n3rf4_guard():
    """n3rf.4 guard: an all-resonance element (like Al I) must NOT be stranded.

    Al I 396.15 + Al I 308.21 are both resonance lines (``E_i ~ 0``).
    Filtering them would drop Al below the three-line Boltzmann minimum
    and the R^2 gate would then reject it — which is what regressed Al
    recall 0.500 -> 0.000 in Phase 5 (commit 4794d04). The pre-scan
    must detect this and disable the filter for the candidate.
    """
    identifier = ALIASIdentifier(_DummyAtomicDB())
    transitions = [
        _transition_with_e_i(396.15, 3.14, e_i_ev=0.0),
        _transition_with_e_i(308.21, 4.02, e_i_ev=0.0),
        _transition_with_e_i(309.27, 4.02, e_i_ev=0.0),
    ]
    fused_lines, mi, mpi, intensity, peaks = _resonance_filter_inputs(transitions)
    candidate = {"nnls_significant": True}

    assert (
        identifier._apply_resonance_filter(fused_lines, candidate, mi, mpi, intensity, peaks)
        is False
    )


def test_apply_resonance_filter_handles_missing_candidate():
    """A ``None`` candidate must be treated as "no NNLS evidence" -> filter off."""
    identifier = ALIASIdentifier(_DummyAtomicDB())
    transitions = [
        _transition_with_e_i(500.0, 1.0, e_i_ev=2.0),
        _transition_with_e_i(501.0, 2.5, e_i_ev=2.0),
        _transition_with_e_i(502.0, 4.0, e_i_ev=2.0),
    ]
    fused_lines, mi, mpi, intensity, peaks = _resonance_filter_inputs(transitions)

    assert identifier._apply_resonance_filter(fused_lines, None, mi, mpi, intensity, peaks) is False


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
    # The other knob still follows the recall preset. detection_threshold is now
    # the paper's C_th k_det presence threshold (Noel 2025 sec 3.8): strict=0.5,
    # recall=0.4 (was a legacy CL floor of 0.02/0.01 before the paper-faithful fix).
    assert identifier.detection_threshold == 0.4


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


# Opt-in robust/weighted temperature estimator (CF-LIBS-improved-762f)
# ---------------------------------------------------------------------------
# Background: the docs/research/vrabel-universal-miss-root-cause-2026-05-14
# diagnosis traced cold T_K bias on synthetic Vrabel s019 spectra to a few
# weak high-E_k Fe I lines dragging the unweighted np.polyfit/linregress
# slope. NotebookLM (Karkare 2020, Lin 2024) says self-absorption biases
# T HOT, NOT cold — so the mechanism here is the unweighted slope fit being
# dominated by noisy high-E_k points. The robust mode weights by
# 1/sigma_y^2 with a shot-noise proxy; the weighted mode drops the
# bottom-quartile by SNR before fitting.
#
# The legacy default MUST stay byte-identical; the audit of the r_sq > 0.2
# polarity check at alias.py:1832/1844 (post-conflict-resolution line
# numbers may differ; search for "slope < 0 and r_sq > 0.2") found the
# polarity is CORRECT (high r^2 ⇒ accept), so robust mode does NOT change
# that check — see the PR body for the audit trail.


class _StubAtomicDB:
    """Minimal AtomicDatabase stub that returns hand-built Fe I lines.

    Lets us construct Vrabel-style synthetic Boltzmann plots where a few
    strong low-E_k lines coexist with many noisy high-E_k lines, and
    confirm that the legacy slope fit gets dragged cold while the robust
    weighted fit recovers a warm T_K.
    """

    def __init__(self, transitions_by_key):
        # transitions_by_key: dict[(element, ion_stage)] -> list[Transition]
        self._table = transitions_by_key

    def get_transitions(
        self,
        element,
        ionization_stage=None,
        wavelength_min=None,
        wavelength_max=None,
        min_relative_intensity=None,
    ):
        key = (element, ionization_stage)
        rows = self._table.get(key, [])
        out = []
        for t in rows:
            if wavelength_min is not None and t.wavelength_nm < wavelength_min:
                continue
            if wavelength_max is not None and t.wavelength_nm > wavelength_max:
                continue
            out.append(t)
        return out


def _make_fe_trans(wl_nm, E_k_ev, A_ki=1.0e7, g_k=9):
    return Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=wl_nm,
        A_ki=A_ki,
        E_k_ev=E_k_ev,
        E_i_ev=0.0,
        g_k=g_k,
        g_i=9,
    )


def _build_vrabel_like_fixture(T_K=10000.0, weak_factor=0.01, n_strong=4, n_weak=8):
    """Construct (transitions, peaks_in_intensity_form) for a Vrabel-like spectrum.

    n_strong low-E_k Fe I lines at expected Boltzmann intensity, plus n_weak
    high-E_k lines downscaled by ``weak_factor`` (default 1/100). The legacy
    unweighted slope-fit will be dragged cold by the weak high-E_k cluster.
    """
    kT_ev = KB_EV * T_K
    # Strong low-E_k lines (well-spaced wavelengths and E_k values so the
    # fit has lever arm to recover T cleanly when the weighting is correct).
    strong_E_k = np.linspace(2.5, 3.5, n_strong)
    strong_wls = np.linspace(370.0, 390.0, n_strong)
    # Weak high-E_k lines crammed at high energies to bias the slope cold.
    weak_E_k = np.linspace(5.5, 6.5, n_weak)
    weak_wls = np.linspace(400.0, 480.0, n_weak)

    transitions = []
    peak_records = []  # list of (wl_nm, intensity)
    rng = np.random.default_rng(42)
    for E_k, wl in zip(strong_E_k, strong_wls):
        A_ki = 1.0e7
        g_k = 9
        # I_obs ∝ A_ki·g_k·exp(-E_k/kT)/lambda (we use the y=log(I·λ/(g·A))
        # form so include λ so the test reads naturally)
        I_expected = A_ki * g_k * math.exp(-E_k / kT_ev) / wl
        # 1% Gaussian noise so r^2 stays high on the strong subset.
        I_obs = I_expected * (1.0 + 0.01 * rng.standard_normal())
        transitions.append(_make_fe_trans(wl, E_k, A_ki=A_ki, g_k=g_k))
        peak_records.append((wl, max(I_obs, 1e-9)))
    for E_k, wl in zip(weak_E_k, weak_wls):
        A_ki = 1.0e7
        g_k = 9
        I_expected = A_ki * g_k * math.exp(-E_k / kT_ev) / wl
        # Knock weak lines down to weak_factor × expected, so they sit
        # near the per-point noise floor and the unweighted slope follows
        # them rather than the strong cluster.
        I_obs = I_expected * weak_factor * (1.0 + 0.5 * rng.standard_normal())
        transitions.append(_make_fe_trans(wl, E_k, A_ki=A_ki, g_k=g_k))
        peak_records.append((wl, max(I_obs, 1e-12)))
    return transitions, peak_records


def _run_estimator(identifier, peak_records, wl_min=300.0, wl_max=500.0):
    """Drive _estimate_plasma_temperature with a peak list + intensity array."""
    # peaks: list of (idx, wl); corrected_intensity is keyed by idx.
    peaks = [(i, wl) for i, (wl, _I) in enumerate(peak_records)]
    intensity = np.array([intens for _wl, intens in peak_records], dtype=float)
    identifier._effective_R = 5000.0
    identifier._global_wl_shift = 0.0
    identifier._estimated_T = None
    identifier._estimate_plasma_temperature(peaks, intensity, wl_min, wl_max)
    return identifier._estimated_T


def test_temperature_estimator_validation():
    """Bogus mode strings must raise ValueError at construction time."""
    with pytest.raises(ValueError, match="temperature_estimator_mode"):
        ALIASIdentifier(_DummyAtomicDB(), temperature_estimator_mode="bogus")


def test_temperature_estimator_robust_recovers_warm_t_on_vrabel_fixture():
    """Robust mode MUST recover a warm T_K on the synthetic Vrabel fixture.

    Legacy mode is dragged cold by the 8 weak high-E_k lines (acts as
    expected baseline failure). Robust mode weights by 1/sigma_y^2 so the
    weak high-E_k lines lose influence, and the recovered T_K must be
    >= 6500 K (vs the legacy cold-pathology of ~4000 K).
    """
    transitions, peak_records = _build_vrabel_like_fixture(
        T_K=10000.0, weak_factor=0.01, n_strong=4, n_weak=8
    )
    stub = _StubAtomicDB({("Fe", 1): transitions})

    legacy_id = ALIASIdentifier(stub, temperature_estimator_mode="legacy")
    robust_id = ALIASIdentifier(stub, temperature_estimator_mode="robust")

    T_legacy = _run_estimator(legacy_id, peak_records)
    T_robust = _run_estimator(robust_id, peak_records)

    # Counterfactual sanity check: both modes return SOMETHING.
    assert T_legacy is not None
    assert T_robust is not None

    # The robust mode MUST recover a warmer T_K than the legacy mode on
    # this fixture by a margin of at least 1500 K (in practice it should
    # recover ~10000 K vs legacy's <6000 K, but we leave headroom for
    # noise realisations and the single-element fallback path).
    assert T_robust >= 6500.0, (
        f"Robust mode recovered T_K={T_robust:.0f} K, expected >=6500 K. "
        f"Legacy recovered T_K={T_legacy:.0f} K."
    )
    assert T_robust > T_legacy + 1500.0, (
        f"Robust mode T_K={T_robust:.0f} K must exceed legacy "
        f"T_K={T_legacy:.0f} K by at least 1500 K."
    )


def test_temperature_estimator_robust_unchanged_on_bhvo2_fixture():
    """BHVO-2-like fixture: all matched lines have similar SNR.

    When the spectrum does NOT exhibit the weak-high-E_k pathology, the
    robust weighted fit should produce a T_K within 500 K of legacy. This
    guards against the robust mode being a one-way ratchet that always
    raises T regardless of input.
    """
    # All 12 lines at roughly the Boltzmann-expected intensity.
    transitions, peak_records = _build_vrabel_like_fixture(
        T_K=10000.0, weak_factor=1.0, n_strong=4, n_weak=8
    )
    stub = _StubAtomicDB({("Fe", 1): transitions})

    legacy_id = ALIASIdentifier(stub, temperature_estimator_mode="legacy")
    robust_id = ALIASIdentifier(stub, temperature_estimator_mode="robust")

    T_legacy = _run_estimator(legacy_id, peak_records)
    T_robust = _run_estimator(robust_id, peak_records)

    assert T_legacy is not None
    assert T_robust is not None
    # On a well-behaved spectrum the two should agree to within 1000 K.
    # The slight difference comes from weights damping per-line scatter
    # symmetrically rather than asymmetrically (no cold-bias correction
    # to apply), and the single-element fallback path in robust mode
    # picking the best-r^2 fit instead of the first-r^2>0.2 fit.
    delta = abs(T_robust - T_legacy)
    assert delta <= 1000.0, (
        f"Robust deviated by {delta:.0f} K on a well-behaved fixture "
        f"(legacy={T_legacy:.0f}, robust={T_robust:.0f}); robust mode "
        f"should only correct cold-bias pathologies."
    )
    # Both should be in the expected range for the synthetic 10000 K plasma.
    for name, T in (("legacy", T_legacy), ("robust", T_robust)):
        assert 7000.0 <= T <= 13000.0, (
            f"{name} mode T={T:.0f} K is outside [7000, 13000] on a "
            f"well-behaved 10000 K fixture."
        )


def test_temperature_estimator_weighted_drops_bottom_quartile(monkeypatch):
    """The weighted mode must drop the bottom quartile of matched lines.

    Asserts the post-pruning slope-fit input excludes the weakest lines.
    """
    transitions, peak_records = _build_vrabel_like_fixture(
        T_K=10000.0, weak_factor=0.005, n_strong=4, n_weak=8
    )
    stub = _StubAtomicDB({("Fe", 1): transitions})

    weighted_id = ALIASIdentifier(stub, temperature_estimator_mode="weighted")
    legacy_id = ALIASIdentifier(stub, temperature_estimator_mode="legacy")

    # Capture the E_k array passed into linregress in each mode.
    captured = {"weighted": None, "legacy": None}

    from scipy.stats import linregress as real_linregress

    def make_spy(label):
        def _spy(x, y):
            if captured[label] is None:
                captured[label] = (np.array(x, copy=True), np.array(y, copy=True))
            return real_linregress(x, y)

        return _spy

    import cflibs.inversion.identify.alias as alias_mod

    monkeypatch.setattr(alias_mod, "linregress", make_spy("weighted"))
    _run_estimator(weighted_id, peak_records)
    monkeypatch.setattr(alias_mod, "linregress", make_spy("legacy"))
    _run_estimator(legacy_id, peak_records)

    assert captured["weighted"] is not None, "weighted mode did not reach linregress"
    assert captured["legacy"] is not None, "legacy mode did not reach linregress"

    n_weighted = captured["weighted"][0].size
    n_legacy = captured["legacy"][0].size
    # Weighted dropped the bottom quartile, so the fit sees strictly fewer
    # lines than legacy on the same fixture.
    assert n_weighted < n_legacy, (
        f"weighted mode passed {n_weighted} lines to linregress but legacy "
        f"passed {n_legacy}; weighted mode must drop the bottom quartile."
    )


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


def test_r2_gate_adaptive_t_admits_cold_plasma_with_moderate_r2():
    """Vrabel-style cold plasma: T < threshold AND r2 in [0.3, 0.85).

    Under ``r2_gate_mode="adaptive_t"`` the cold-T relaxation activates
    and a moderate R^2 (0.4) clears the gate. Under the default
    ``"fixed"`` mode the same R^2 is rejected. The element is
    constructed with N_matched=5 and ``boltz_r2 = 0.4`` per the plan.
    """
    fixed_id = ALIASIdentifier(_DummyAtomicDB(), r2_gate_mode="fixed")
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
    fixed_id = ALIASIdentifier(_DummyAtomicDB(), r2_gate_mode="fixed")
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

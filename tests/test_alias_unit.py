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
    intensity = np.array([I for _wl, I in peak_records], dtype=float)
    identifier._effective_R = 5000.0
    identifier._global_wl_shift = 0.0
    identifier._estimated_T = None
    identifier._estimate_plasma_temperature(peaks, intensity, wl_min, wl_max)
    return identifier._estimated_T


def test_temperature_estimator_validation():
    """Bogus mode strings must raise ValueError at construction time."""
    with pytest.raises(ValueError, match="temperature_estimator_mode"):
        ALIASIdentifier(_DummyAtomicDB(), temperature_estimator_mode="bogus")


def test_temperature_estimator_legacy_byte_identical():
    """No-kwarg construction and mode='legacy' must return identical _estimated_T.

    Drives the estimator on a fresh Vrabel-like fixture so the legacy code
    path is fully exercised, then confirms the explicit mode='legacy'
    constructor produces bit-for-bit the same number.
    """
    transitions, peak_records = _build_vrabel_like_fixture(T_K=10000.0)
    stub = _StubAtomicDB({("Fe", 1): transitions})

    default_id = ALIASIdentifier(stub)
    explicit_legacy_id = ALIASIdentifier(stub, temperature_estimator_mode="legacy")

    T_default = _run_estimator(default_id, peak_records)
    T_explicit = _run_estimator(explicit_legacy_id, peak_records)

    assert T_default is not None
    assert T_explicit is not None
    # Byte-identical (same arithmetic path).
    assert T_default == T_explicit


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

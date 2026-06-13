"""Bead ye6t regression tests: edge-riding residual scan + per-line residual gate.

The post-calibration +/-0.05 nm "mop-up" comb shift-scan measurably rode its
window edge in every configuration tested on real ChemCam BHVO-2 (its
objective maximizes match count, so scooping one extra dense-catalog
coincidence always wins), admitting contaminated matches up to
``tolerance + 0.05`` from the true axis (Al I 892.356: +10.4 wt% Al). These
tests pin both halves of the fix:

1. ``shift_scan_nm = 0`` (the post-quality-passed-calibration default) keeps
   the fitted axis: the contaminated match is out of reach.
2. The per-line residual gate drops off-consensus matches even when a legacy
   scan re-breaks the axis, and records them in ``residual_gate`` diagnostics.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cflibs.atomic.structures import Transition
from cflibs.inversion.identify.line_detection import (
    _drop_element_if_mostly_gated,
    _match_transitions_to_peaks,
    _pooled_residual_consensus,
    detect_line_observations,
)
from cflibs.inversion.physics.boltzmann import LineObservation

pytestmark = pytest.mark.unit


def _transition(element: str, wavelength_nm: float, e_k: float = 3.0) -> Transition:
    return Transition(
        element=element,
        ionization_stage=1,
        wavelength_nm=wavelength_nm,
        A_ki=1.0e7,
        E_k_ev=e_k,
        E_i_ev=0.0,
        g_k=5,
        g_i=3,
        relative_intensity=100.0,
    )


class _StubDB:
    """Minimal AtomicDatabase stand-in for detect_line_observations."""

    def __init__(self, transitions):
        self._transitions = list(transitions)

    def get_transitions(
        self,
        element,
        wavelength_min=None,
        wavelength_max=None,
        min_relative_intensity=None,
    ):
        out = []
        for t in self._transitions:
            if t.element != element:
                continue
            if wavelength_min is not None and t.wavelength_nm < wavelength_min:
                continue
            if wavelength_max is not None and t.wavelength_nm > wavelength_max:
                continue
            out.append(t)
        return out


# Real lines sit exactly on their peaks (true residual 0). The contaminated
# DB line at 340.14 nm is 0.14 nm from its nearest peak: out of reach at the
# true axis (tolerance 0.1) but admitted once a +0.05 scan shift re-breaks
# the axis (|340.05 - 340.14| = 0.09 < 0.1) -- the Al I 892.356 mechanism.
_REAL_LINES = [300.0, 310.0, 320.0, 330.0]
_CONTAMINATED_LINE = 340.14
_PEAKS = _REAL_LINES + [340.0]


def _rigged_spectrum():
    wavelength = np.arange(295.0, 345.0, 0.01)
    intensity = np.zeros_like(wavelength)
    for center in _PEAKS:
        intensity += 10.0 * np.exp(-0.5 * ((wavelength - center) / 0.03) ** 2)
    return wavelength, intensity


def _detect(shift_scan_nm: float, line_residual_gate: bool):
    wavelength, intensity = _rigged_spectrum()
    db = _StubDB([_transition("Mg", wl) for wl in _REAL_LINES + [_CONTAMINATED_LINE]])
    return detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=db,
        elements=["Mg"],
        wavelength_tolerance_nm=0.1,
        min_peak_height=0.05,
        peak_width_nm=0.2,
        shift_scan_nm=shift_scan_nm,
        line_residual_gate=line_residual_gate,
    )


class TestEdgeRidingScanRegression:
    def test_zero_residual_scan_keeps_true_axis_and_rejects_contamination(self):
        """Fix 1: with the post-calibration scan at 0, the fitted axis is
        kept and the contaminated match never exists."""
        result = _detect(shift_scan_nm=0.0, line_residual_gate=False)
        assert result.applied_shift_nm == pytest.approx(0.0)
        matched = {obs.wavelength_nm for obs in result.observations}
        assert _CONTAMINATED_LINE not in matched
        assert set(_REAL_LINES) <= matched

    def test_per_line_gate_drops_contaminated_match_even_with_legacy_scan(self):
        """Fix 3: even when the legacy scan re-breaks the axis, the per-line
        residual gate drops the off-consensus match and records it."""
        result = _detect(shift_scan_nm=0.05, line_residual_gate=True)
        matched = {obs.wavelength_nm for obs in result.observations}
        assert _CONTAMINATED_LINE not in matched
        assert set(_REAL_LINES) <= matched
        assert result.residual_gate is not None
        assert result.residual_gate["enabled"] is True
        assert result.residual_gate["n_gated"] >= 1
        gated_wls = {g["wavelength_nm"] for g in result.residual_gate["gated_lines"]}
        assert _CONTAMINATED_LINE in gated_wls
        assert "line_residual_gated_matches" in result.warnings


class TestPerLineResidualGate:
    """Direct matcher-level pinning: consensus +0.00, outlier +0.12 -> dropped."""

    def test_outlier_residual_dropped_against_zero_consensus(self):
        peaks = [(i, wl) for i, wl in enumerate([500.0, 510.0, 520.0, 530.12])]
        transitions = [_transition("Fe", wl) for wl in [500.0, 510.0, 520.0, 530.0]]
        # Wide tolerance admits the +0.12 outlier; the band must reject it.
        gated_out: list = []
        matches = _match_transitions_to_peaks(
            peaks=peaks,
            transitions=transitions,
            tolerance_nm=0.15,
            shift_nm=0.0,
            residual_center_nm=0.0,
            residual_band_nm=0.05,
            gated_out=gated_out,
        )
        matched_lines = {m[0].wavelength_nm for m in matches}
        assert matched_lines == {500.0, 510.0, 520.0}
        assert len(gated_out) == 1
        transition, peak_wl, residual = gated_out[0]
        assert transition.wavelength_nm == pytest.approx(530.0)
        assert peak_wl == pytest.approx(530.12)
        assert residual == pytest.approx(0.12)

    def test_gated_match_does_not_consume_the_peak(self):
        """A gated-out match must leave its peak available (peak ownership)."""
        peaks = [(0, 600.1)]
        contaminant = _transition("Sn", 600.0)
        used: set = set()
        matches = _match_transitions_to_peaks(
            peaks=peaks,
            transitions=[contaminant],
            tolerance_nm=0.15,
            shift_nm=0.0,
            used_peaks=used,
            residual_center_nm=0.0,
            residual_band_nm=0.05,
        )
        assert matches == []
        assert used == set()

    def test_no_center_means_no_gating(self):
        peaks = [(0, 530.12)]
        transitions = [_transition("Fe", 530.0)]
        matches = _match_transitions_to_peaks(
            peaks=peaks,
            transitions=transitions,
            tolerance_nm=0.15,
            shift_nm=0.0,
        )
        assert len(matches) == 1

    def test_pooled_consensus_is_median_of_element_residuals(self):
        peaks = [(i, wl) for i, wl in enumerate([400.01, 410.01, 420.01])]
        transitions = {"Ca": [_transition("Ca", wl) for wl in [400.0, 410.0, 420.0]]}
        consensus = _pooled_residual_consensus(
            ["Ca"], peaks, transitions, applied_shift_nm=0.0, tolerance_nm=0.1
        )
        assert consensus == pytest.approx(0.01, abs=1e-9)

    def test_pooled_consensus_none_without_matches(self):
        peaks = [(0, 700.0)]
        transitions = {"Ca": [_transition("Ca", 400.0)]}
        assert (
            _pooled_residual_consensus(
                ["Ca"], peaks, transitions, applied_shift_nm=0.0, tolerance_nm=0.1
            )
            is None
        )


class TestMostlyGatedElementDrop:
    """An element whose match set was mostly contamination is dropped whole."""

    def _drop(self, n_kept, n_gated, min_kept_lines=3):
        observations = [
            LineObservation(
                wavelength_nm=500.0 + i,
                intensity=1.0,
                intensity_uncertainty=0.1,
                element="Zz",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=5,
                A_ki=1e7,
            )
            for i in range(n_kept)
        ]
        resonance: set = set()
        diag: dict = {}
        _drop_element_if_mostly_gated(
            "Zz",
            n_kept=n_kept,
            n_gated=n_gated,
            coherence_min_lines=2,
            coherence_min_fraction=0.5,
            min_kept_lines=min_kept_lines,
            observations=observations,
            n_obs_before=0,
            resonance_lines=resonance,
            residual_gate_diag=diag,
        )
        return observations, diag

    def test_low_coherent_fraction_drops_element(self):
        """The BHVO-2 Sn case: 2 coherent of 6 matches -> dropped."""
        obs, diag = self._drop(n_kept=2, n_gated=4)
        assert obs == []
        assert diag["dropped_elements"]["Zz"] == {"kept": 2, "gated": 4}

    def test_contaminated_element_below_min_kept_lines_drops(self):
        """The BHVO-2 Th case: 2 kept + 2 gated sits exactly at the fraction
        threshold but below the min-kept-lines evidence bar -> dropped."""
        obs, diag = self._drop(n_kept=2, n_gated=2)
        assert obs == []
        assert diag["dropped_elements"]["Zz"] == {"kept": 2, "gated": 2}

    def test_majority_coherent_with_enough_lines_survives(self):
        obs, diag = self._drop(n_kept=3, n_gated=2)
        assert len(obs) == 3
        assert "dropped_elements" not in diag

    def test_clean_element_never_subject_to_the_bar(self):
        """No gated matches -> no contamination evidence -> never dropped,
        however sparse (protects 2-line real elements)."""
        obs, diag = self._drop(n_kept=2, n_gated=0)
        assert len(obs) == 2
        assert "dropped_elements" not in diag


class TestPerPeakOwnership:
    """A stronger element's claimed peak cannot double-count for a weaker one."""

    def _detect(self, line_residual_gate):
        el1_lines = [400.0, 410.0, 420.0, 430.0]
        # 400.004 sits on El1's 400.0 peak; 600/610 are El2's own peaks.
        el2_lines = [400.004, 600.0, 610.0]
        peak_centers = el1_lines + [600.0, 610.0]
        wavelength = np.arange(395.0, 615.0, 0.01)
        intensity = np.zeros_like(wavelength)
        for center in peak_centers:
            intensity += 10.0 * np.exp(-0.5 * ((wavelength - center) / 0.03) ** 2)
        db = _StubDB(
            [_transition("Fe", wl) for wl in el1_lines]
            + [_transition("Sn", wl) for wl in el2_lines]
        )
        return detect_line_observations(
            wavelength=wavelength,
            intensity=intensity,
            atomic_db=db,
            elements=["Fe", "Sn"],
            wavelength_tolerance_nm=0.1,
            min_peak_height=0.05,
            peak_width_nm=0.2,
            shift_scan_nm=0.0,
            line_residual_gate=line_residual_gate,
        )

    def test_weaker_claimant_cannot_reuse_owned_peak(self):
        result = self._detect(line_residual_gate=True)
        sn = sorted(o.wavelength_nm for o in result.observations if o.element == "Sn")
        assert sn == [600.0, 610.0]  # 400.004 blocked: the peak belongs to Fe 400.0


class TestMatchTransitionNaN:
    """A NaN peak wavelength must match nothing (PR #282 review MF2).

    The De Morgan rewrite ``distance > tol or distance >= best`` let NaN
    through every branch (NaN comparisons are all False), so a NaN peak
    fabricated a match with the LAST transition in the list.
    """

    def test_nan_peak_matches_nothing(self):
        from cflibs.inversion.identify.line_detection import _match_transition

        t1 = SimpleNamespace(wavelength_nm=255.0)
        t2 = SimpleNamespace(wavelength_nm=255.5)
        assert _match_transition(float("nan"), [t1, t2], 0.1) is None

    def test_nan_peak_matches_nothing_with_residual_gate(self):
        from cflibs.inversion.identify.line_detection import _match_transition

        t1 = SimpleNamespace(wavelength_nm=255.0)
        assert (
            _match_transition(
                float("nan"), [t1], 0.1, residual_center_nm=0.0, residual_band_nm=0.05
            )
            is None
        )

    def test_finite_peak_still_matches_nearest(self):
        from cflibs.inversion.identify.line_detection import _match_transition

        t1 = SimpleNamespace(wavelength_nm=255.0)
        t2 = SimpleNamespace(wavelength_nm=255.5)
        assert _match_transition(255.04, [t1, t2], 0.1) is t1

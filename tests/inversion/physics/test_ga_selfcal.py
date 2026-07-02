"""Unit tests for in-plasma relative-g·A self-calibration (audit Issue 1a).

The physics under test is the exact, model-free same-upper-level identity::

    I_i / I_j = (A_i lambda_j) / (A_j lambda_i)

so in the Boltzmann ordinate ``y = ln(I lambda / (g A))`` every line sharing an
upper level lands on the same point. A deviation is a direct measurement of the
line's RELATIVE A_ki error. These tests build synthetic optically-thin line
groups (no DB needed), corrupt A_ki, and verify recovery, shrinkage, and the
exclusion of self-absorption-suspect lines.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import pytest

from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.physics.ga_selfcal import (
    RelativeGACalibration,
    apply_relative_ga_correction,
    find_shared_upper_groups,
    measure_relative_ga_residuals,
    self_calibrate_relative_ga,
)

pytestmark = pytest.mark.unit


def _obs(wl, intensity, *, element="Fe", stage=1, E_k=5.0, g_k=6, A_ki=1e8, unc_frac=0.001):
    return LineObservation(
        wavelength_nm=wl,
        intensity=intensity,
        intensity_uncertainty=unc_frac * intensity,
        element=element,
        ionization_stage=stage,
        E_k_ev=E_k,
        g_k=g_k,
        A_ki=A_ki,
    )


def _shared_group(a_true, wavelengths, *, base=1.0, E_k=5.0, g_k=6, unc_frac=0.001, **kw):
    """Build an optically-thin same-upper-level group.

    Intensity of an optically-thin line is I ∝ A_ki n_k / lambda; with a shared
    upper level (shared n_k, g_k) that is ``base * A_true / lambda``. The DB A_ki
    fed to the inversion may differ from the true one (that is the error we
    measure), but the *intensities* always reflect the TRUE A_ki.
    """
    return [
        _obs(
            wl,
            base * at / wl,
            E_k=E_k,
            g_k=g_k,
            A_ki=at,  # placeholder; caller overrides DB A_ki as needed
            unc_frac=unc_frac,
            **kw,
        )
        for at, wl in zip(a_true, wavelengths)
    ]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def test_grouping_by_fingerprint_shared_upper_level():
    # Two lines share upper level (same E_k, g_k); a third has a different E_k.
    obs = [
        _obs(400.0, 1.0, E_k=5.0, g_k=6),
        _obs(500.0, 1.0, E_k=5.0, g_k=6),
        _obs(450.0, 1.0, E_k=4.0, g_k=6),
    ]
    groups = find_shared_upper_groups(obs)
    assert len(groups) == 1
    g = groups[0]
    assert sorted(g.indices) == [0, 1]
    assert g.resolved_by_id is False
    assert g.element == "Fe" and g.ionization_stage == 1


def test_grouping_separates_elements_and_stages():
    obs = [
        _obs(400.0, 1.0, element="Fe", stage=1, E_k=5.0, g_k=6),
        _obs(500.0, 1.0, element="Fe", stage=1, E_k=5.0, g_k=6),
        _obs(410.0, 1.0, element="Fe", stage=2, E_k=5.0, g_k=6),
        _obs(510.0, 1.0, element="Fe", stage=2, E_k=5.0, g_k=6),
        _obs(420.0, 1.0, element="Ti", stage=1, E_k=5.0, g_k=6),
    ]
    groups = find_shared_upper_groups(obs)
    # Fe I pair and Fe II pair; the lone Ti line is not a group.
    assert len(groups) == 2
    assert {(g.element, g.ionization_stage) for g in groups} == {("Fe", 1), ("Fe", 2)}


def test_grouping_drops_singletons_and_bad_intensity():
    obs = [
        _obs(400.0, 1.0, E_k=5.0, g_k=6),  # singleton E_k=5
        _obs(410.0, -1.0, E_k=6.0, g_k=6),  # bad intensity
        _obs(420.0, 1.0, E_k=6.0, g_k=6),  # would-be partner but only one valid
    ]
    groups = find_shared_upper_groups(obs)
    assert groups == []


# ---------------------------------------------------------------------------
# Exact ratio identity
# ---------------------------------------------------------------------------
def test_same_upper_level_ratio_identity_holds():
    a_true = [1e8, 3e8]
    wl = [400.0, 500.0]
    obs = _shared_group(a_true, wl)
    i0, i1 = obs[0].intensity, obs[1].intensity
    predicted = (a_true[0] * wl[1]) / (a_true[1] * wl[0])
    assert math.isclose(i0 / i1, predicted, rel_tol=1e-12)


def test_clean_group_residuals_are_zero():
    a_true = [1e8, 2e8, 5e7]
    wl = [400.0, 450.0, 500.0]
    obs = _shared_group(a_true, wl)  # DB A_ki == true A_ki
    groups = find_shared_upper_groups(obs)
    res = measure_relative_ga_residuals(obs, groups)
    assert len(res) == 3
    assert max(abs(r.residual) for r in res) < 1e-9


# ---------------------------------------------------------------------------
# Recovery of injected relative-A_ki error
# ---------------------------------------------------------------------------
def test_recovers_injected_relative_error():
    # Group of 4 sharing an upper level. Intensities reflect the TRUE A_ki.
    a_true = np.array([1e8, 1e8, 1e8, 1e8])
    wl = [400.0, 430.0, 470.0, 500.0]
    obs = _shared_group(a_true, wl, unc_frac=1e-4)
    # Corrupt the DB A_ki of lines 1 and 2 (leave 0 and 3 as anchors).
    corrupt = {1: 1.5, 2: 0.6}
    for i, f in corrupt.items():
        obs[i] = LineObservation(
            wavelength_nm=obs[i].wavelength_nm,
            intensity=obs[i].intensity,
            intensity_uncertainty=obs[i].intensity_uncertainty,
            element=obs[i].element,
            ionization_stage=obs[i].ionization_stage,
            E_k_ev=obs[i].E_k_ev,
            g_k=obs[i].g_k,
            A_ki=a_true[i] * f,  # DB (corrupted) A_ki
            aki_uncertainty=0.5,  # fabricated grade-D uncertainty
        )
    groups = find_shared_upper_groups(obs)
    res = measure_relative_ga_residuals(obs, groups)
    by_idx = {r.index: r for r in res}
    # Residual of a corrupted line ~ -ln(factor) (up to the small group mean).
    assert by_idx[1].residual < 0  # factor 1.5 -> A too big -> y too low
    assert by_idx[2].residual > 0  # factor 0.6 -> A too small -> y too high

    corrected, calib = apply_relative_ga_correction(obs, res, shrinkage="empirical_bayes")
    # Relative calibration: corrected A_ki RATIOS must match TRUE ratios.
    a_corr = np.array([c.A_ki for c in corrected])
    a_corr_ratio = a_corr / a_corr[0]
    a_true_ratio = a_true / a_true[0]
    np.testing.assert_allclose(a_corr_ratio, a_true_ratio, rtol=2e-3)
    # The correction is relative (the group's absolute scale is unmeasurable),
    # so the RATIO of recovered corrections tracks the ratio of injected
    # factors: c_1/c_2 == (1/1.5) / (1/0.6) == 0.6/1.5 == 0.4.
    assert math.isclose(
        calib.corrections[1] / calib.corrections[2], 0.6 / 1.5, rel_tol=5e-3
    )
    # Corrected lines get the measured residual as their new A_ki uncertainty.
    assert calib.correction_sigma[1] == corrected[1].aki_uncertainty
    assert corrected[1].aki_uncertainty < 0.5  # replaced the fabricated 0.5


def test_residual_tracks_injected_factor_correlation():
    # A range of injected factors on a large group; residual ≈ -ln(factor).
    n = 8
    a_true = np.full(n, 1e8)
    wl = list(np.linspace(400.0, 520.0, n))
    obs = _shared_group(a_true, wl, unc_frac=1e-5)
    rng = np.random.default_rng(0)
    factors = np.exp(rng.uniform(-0.6, 0.6, size=n))  # ~0.55x .. 1.8x
    for i in range(n):
        obs[i] = LineObservation(
            wavelength_nm=obs[i].wavelength_nm,
            intensity=obs[i].intensity,
            intensity_uncertainty=obs[i].intensity_uncertainty,
            element=obs[i].element,
            ionization_stage=obs[i].ionization_stage,
            E_k_ev=obs[i].E_k_ev,
            g_k=obs[i].g_k,
            A_ki=a_true[i] * factors[i],
        )
    groups = find_shared_upper_groups(obs)
    res = measure_relative_ga_residuals(obs, groups)
    by_idx = {r.index: r.residual for r in res}
    r_vec = np.array([by_idx[i] for i in range(n)])
    target = -np.log(factors)
    # Both are mean-removed within the group, so correlate the centered vectors.
    corr = np.corrcoef(r_vec - r_vec.mean(), target - target.mean())[0, 1]
    assert corr > 0.999


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------
def test_shrinkage_suppresses_noise_only_residuals():
    # Small residual buried in large per-line uncertainty -> shrink to no-op.
    a_true = np.array([1e8, 1e8, 1e8])
    wl = [400.0, 450.0, 500.0]
    obs = _shared_group(a_true, wl, unc_frac=0.5)  # very noisy
    # Corrupt by a tiny factor well within the noise.
    obs[1] = LineObservation(
        wavelength_nm=obs[1].wavelength_nm,
        intensity=obs[1].intensity,
        intensity_uncertainty=obs[1].intensity_uncertainty,
        element=obs[1].element,
        ionization_stage=obs[1].ionization_stage,
        E_k_ev=obs[1].E_k_ev,
        g_k=obs[1].g_k,
        A_ki=a_true[1] * 1.02,
    )
    groups = find_shared_upper_groups(obs)
    res = measure_relative_ga_residuals(obs, groups)
    corrected, calib = apply_relative_ga_correction(obs, res, shrinkage="empirical_bayes")
    assert calib.tau2 == 0.0
    assert calib.n_lines_corrected == 0
    # A_ki unchanged.
    for c, o in zip(corrected, obs):
        assert c.A_ki == o.A_ki


def test_fixed_shrinkage_and_disable():
    a_true = np.array([1e8, 1e8])
    wl = [400.0, 500.0]
    obs = _shared_group(a_true, wl, unc_frac=1e-5)
    obs[1] = LineObservation(
        wavelength_nm=obs[1].wavelength_nm,
        intensity=obs[1].intensity,
        intensity_uncertainty=obs[1].intensity_uncertainty,
        element=obs[1].element,
        ionization_stage=obs[1].ionization_stage,
        E_k_ev=obs[1].E_k_ev,
        g_k=obs[1].g_k,
        A_ki=a_true[1] * 2.0,
    )
    groups = find_shared_upper_groups(obs)
    res = measure_relative_ga_residuals(obs, groups)
    # shrinkage=None disables entirely.
    _, calib_off = apply_relative_ga_correction(obs, res, shrinkage=None)
    assert calib_off.n_lines_corrected == 0
    # Fixed 50% shrinkage halves the log-correction.
    _, calib_half = apply_relative_ga_correction(obs, res, shrinkage=0.5)
    _, calib_full = apply_relative_ga_correction(obs, res, shrinkage=1.0)
    log_half = math.log(calib_half.corrections[1])
    log_full = math.log(calib_full.corrections[1])
    assert math.isclose(log_half, 0.5 * log_full, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Self-absorption exclusion
# ---------------------------------------------------------------------------
def test_excluded_line_not_corrected_and_not_anchor():
    a_true = np.array([1e8, 1e8, 1e8])
    wl = [400.0, 450.0, 500.0]
    obs = _shared_group(a_true, wl, unc_frac=1e-4)
    # Line 0 is optically thick: its measured intensity is suppressed, so its
    # A_ki looks too large (negative residual). Simulate by halving intensity.
    obs[0] = LineObservation(
        wavelength_nm=obs[0].wavelength_nm,
        intensity=obs[0].intensity * 0.5,
        intensity_uncertainty=obs[0].intensity_uncertainty * 0.5,
        element=obs[0].element,
        ionization_stage=obs[0].ionization_stage,
        E_k_ev=obs[0].E_k_ev,
        g_k=obs[0].g_k,
        A_ki=a_true[0],
    )
    groups = find_shared_upper_groups(obs)
    exclude = [True, False, False]
    res = measure_relative_ga_residuals(obs, groups, exclude_mask=exclude)
    corrected, calib = apply_relative_ga_correction(obs, res)
    # The thick line is not corrected (would have been chased as an A error).
    assert 0 not in calib.corrections
    assert corrected[0].A_ki == obs[0].A_ki
    # The two thin anchors agree, so they are not spuriously corrected either.
    assert calib.n_lines_corrected == 0


# ---------------------------------------------------------------------------
# No-op / degenerate
# ---------------------------------------------------------------------------
def test_no_groups_is_identity():
    obs = [_obs(400.0, 1.0, E_k=5.0), _obs(500.0, 1.0, E_k=4.0)]
    corrected, calib = self_calibrate_relative_ga(obs)
    assert corrected == list(obs)
    assert isinstance(calib, RelativeGACalibration)
    assert calib.n_lines_corrected == 0


# ---------------------------------------------------------------------------
# atomic_db upp_level_id path (lightweight fake)
# ---------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, table):
        self._table = table

    def execute(self, query, params):
        element, sp_num = params
        return _FakeCursor(
            [
                (wl, ek, uid)
                for (el, sn, wl, ek, uid) in self._table
                if el == element and sn == sp_num
            ]
        )


class _FakeDB:
    """Minimal atomic_db exposing _get_connection() with a lines table."""

    def __init__(self, table):
        self._table = table

    @contextmanager
    def _get_connection(self):
        yield _FakeConn(self._table)


def test_grouping_uses_upp_level_id_when_available():
    # Two lines with DIFFERENT E_k but the DB says they share upp_level_id.
    # (Contrived: exercises the id path taking precedence over fingerprint.)
    obs = [
        _obs(400.0, 1.0, E_k=5.0, g_k=6),
        _obs(500.0, 1.0, E_k=5.0001, g_k=8),  # different g_k -> fingerprint would split
    ]
    table = [
        ("Fe", 1, 400.0, 5.0, "026001.000100"),
        ("Fe", 1, 500.0, 5.0001, "026001.000100"),
    ]
    db = _FakeDB(table)
    groups = find_shared_upper_groups(obs, db, ek_tol_ev=1e-3, wl_tol_nm=0.05)
    assert len(groups) == 1
    assert groups[0].resolved_by_id is True
    assert sorted(groups[0].indices) == [0, 1]

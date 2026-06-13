"""J3 parity — jittable line matching + gates vs the frozen reference oracle.

ADR-0004 §4 row 4 / D5 (the tightest contract in the program: exact discrete
logic + identical arithmetic). Each test feeds **identical padded inputs** to
the *real* reference helper (imported from
:mod:`cflibs.inversion.identify.line_detection`, never reimplemented) and to the
``cflibs.jitpipe.identify`` jit kernel, then asserts the §4 tolerance:

* greedy match peak-index sets identical;
* comb P/R/F1 rtol 1e-12; passes flags exact;
* ``applied_shift_nm`` exactly equal; accepted/kept element sets identical;
* residual consensus rtol 1e-12; veto keep-set identical;
* observation line-key sets identical; n_gated / dropped maps equal;
* intensities + sigma rtol 1e-10 (trapezoid path).

Plus the four mandatory hazard micro-fixtures (§5.3), vmap smoke (B=16), grad
smoke (finite), no-SQLite-in-kernel guard, and padding invariance (rerun at the
next pad size => bit-identical on the valid region).

Run (never the full suite — 600s watchdog)::

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/jitpipe/test_parity_j3.py -q --timeout=300
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.atomic.structures import Transition  # noqa: E402
from cflibs.inversion.identify import line_detection as ref  # noqa: E402
from cflibs.jitpipe import identify as J  # noqa: E402

# conftest forces CPU + x64; assert it so a misconfigured runner fails loudly.
assert jax.config.jax_enable_x64, "J3 parity requires fp64 (conftest forces it)"


# ---------------------------------------------------------------------------
# Host-side fixture builders — turn reference (peaks, transitions) into the
# padded FrontEndSnapshot the kernel consumes, with NO loss of information.
# ---------------------------------------------------------------------------


def _mk_transition(element, stage, wl, *, a_ki=1.0, e_k=1.0, e_i=0.0, g_k=2, is_res=None):
    return Transition(
        element=element,
        ionization_stage=stage,
        wavelength_nm=float(wl),
        A_ki=float(a_ki),
        E_k_ev=float(e_k),
        E_i_ev=float(e_i),
        g_k=int(g_k),
        g_i=1,
        is_resonance=is_res,
    )


def build_snapshot(peaks, transitions_by_element, *, e_max, k_comb, p_max):
    """Build a :class:`FrontEndSnapshot` from reference peaks + comb transitions.

    ``peaks`` is the reference ``List[(idx, wl)]``; ``transitions_by_element`` is
    the comb-ranked ``Dict[str, List[Transition]]`` (already in the catalog
    order the reference iterates). Element-slot order = dict insertion order =
    the order the reference's element loop walks. Padded to ``(e_max, k_comb)``
    / ``(p_max,)``.
    """
    elements = list(transitions_by_element.keys())
    n_e = len(elements)
    assert n_e <= e_max and len(peaks) <= p_max

    peak_wl = np.zeros(p_max)
    peak_idx = np.full(p_max, -1, dtype=np.int64)
    peak_mask = np.zeros(p_max, dtype=bool)
    for j, (idx, wl) in enumerate(peaks):
        peak_wl[j] = wl
        peak_idx[j] = idx
        peak_mask[j] = True

    comb_wl = np.zeros((e_max, k_comb))
    comb_ek = np.zeros((e_max, k_comb))
    comb_gk = np.ones((e_max, k_comb))
    comb_aki = np.ones((e_max, k_comb))
    comb_ei = np.zeros((e_max, k_comb))
    comb_stage = np.zeros((e_max, k_comb), dtype=np.int64)
    comb_res = np.full((e_max, k_comb), -1, dtype=np.int64)
    comb_mask = np.zeros((e_max, k_comb), dtype=bool)
    element_mask = np.zeros(e_max, dtype=bool)
    full_n = np.zeros(e_max, dtype=np.int64)

    for e, el in enumerate(elements):
        trans = transitions_by_element[el]
        element_mask[e] = True
        full_n[e] = len(trans)
        for k, t in enumerate(trans[:k_comb]):
            comb_wl[e, k] = t.wavelength_nm
            comb_ek[e, k] = t.E_k_ev
            comb_gk[e, k] = t.g_k
            comb_aki[e, k] = t.A_ki
            comb_ei[e, k] = t.E_i_ev
            comb_stage[e, k] = t.ionization_stage
            comb_res[e, k] = -1 if t.is_resonance is None else int(bool(t.is_resonance))
            comb_mask[e, k] = True

    return (
        J.FrontEndSnapshot(
            peak_wavelength_nm=jnp.asarray(peak_wl),
            peak_index=jnp.asarray(peak_idx),
            peak_mask=jnp.asarray(peak_mask),
            comb_wavelength_nm=jnp.asarray(comb_wl),
            comb_E_k_ev=jnp.asarray(comb_ek),
            comb_g_k=jnp.asarray(comb_gk),
            comb_A_ki=jnp.asarray(comb_aki),
            comb_E_i_ev=jnp.asarray(comb_ei),
            comb_stage=jnp.asarray(comb_stage),
            comb_is_resonance=jnp.asarray(comb_res),
            comb_mask=jnp.asarray(comb_mask),
            element_mask=jnp.asarray(element_mask),
            full_n_lines=jnp.asarray(full_n),
        ),
        elements,
    )


# ---------------------------------------------------------------------------
# Fixtures — micro hazard cases + a richer multi-element/multi-shift case.
# ---------------------------------------------------------------------------


def _two_elements_one_peak():
    """Hazard: two elements competing for one peak (greedy order-dependence)."""
    peaks = [(10, 400.00), (20, 500.00), (30, 600.00)]
    # Element A: strong (higher f1 later) lines near 400/500; B: near 400 only.
    a = [_mk_transition("A", 1, 400.02), _mk_transition("A", 1, 500.02)]
    b = [_mk_transition("B", 1, 400.04)]
    return peaks, {"A": a, "B": b}


def _np_isclose_tie():
    """Hazard: two shifts whose total F1 are within np.isclose of each other."""
    # Symmetric layout: shifts +/-d give identical match counts/F1; the smaller
    # |shift| (here 0) must win the tie. Two lines straddling a peak.
    peaks = [(10, 400.0), (20, 410.0)]
    a = [_mk_transition("A", 1, 400.0), _mk_transition("A", 1, 410.0)]
    return peaks, {"A": a}


def _gated_peak_rescued():
    """Hazard: a gated line that does NOT consume a peak (gated-without-consuming).

    Element A's 3 coherent lines set the consensus at 0 on peaks 0/1/2. Element
    B's single line is within tolerance of peak 3 but +0.07 nm off-consensus,
    with no coherent alternative -> it must be *gated* (flag set) and must NOT
    consume peak 3 (verified n_gated == 1 in :func:`test_gated_hazard_fires`).
    """
    peaks = [(0, 400.00), (1, 401.00), (2, 402.00), (3, 410.00)]
    a = [
        _mk_transition("A", 1, 400.00),
        _mk_transition("A", 1, 401.00),
        _mk_transition("A", 1, 402.00),
    ]
    b = [_mk_transition("B", 1, 409.93)]  # peak3 - line = +0.07, off-consensus
    return peaks, {"A": a, "B": b}


def _mostly_gated_drop():
    """Hazard: element retroactively dropped (kept < min, peaks stay claimed).

    Strong element M's 6 coherent lines set the consensus at 0 (its peaks
    dominate the pooled median). Element X has 2 coherent lines (peaks 0/1) and
    3 off-consensus lines (+0.06 nm) with no coherent alternative -> 2 kept,
    3 gated -> kept < 3 -> X is dropped entirely; its 2 claimed peaks stay
    claimed (never rolled back). Mirrors the BHVO-2 Sn confounder.
    """
    peaks = [(i, 500.0 + i) for i in range(6)] + [(10 + i, 400.0 + i) for i in range(5)]
    m = [_mk_transition("M", 1, 500.0 + i) for i in range(6)]
    x = [
        _mk_transition("X", 1, 400.0),
        _mk_transition("X", 1, 401.0),
        _mk_transition("X", 1, 402.06),
        _mk_transition("X", 1, 403.06),
        _mk_transition("X", 1, 404.06),
    ]
    return peaks, {"M": m, "X": x}


def _multi_element_multi_shift():
    """Richer case: 3 elements, several peaks, a real applied shift."""
    rng = np.random.default_rng(7)
    peaks = [(i, 350.0 + i * 1.3 + 0.07) for i in range(12)]  # global +0.07 nm
    fe = [_mk_transition("Fe", 1, 350.0 + i * 1.3, e_k=0.3 * i, a_ki=1e7) for i in range(6)]
    ti = [_mk_transition("Ti", 2, 351.3 + i * 1.3, e_k=0.2 * i, a_ki=5e6) for i in range(4)]
    # Sn confounder: a few lines scattered, mostly off the global shift.
    sn = [_mk_transition("Sn", 1, 352.0 + rng.uniform(0, 8.0)) for _ in range(5)]
    return peaks, {"Fe": fe, "Ti": ti, "Sn": sn}


# ---------------------------------------------------------------------------
# Reference-call helpers — invoke the REAL reference functions on the same data.
# ---------------------------------------------------------------------------

TOL = 0.1


def ref_score_grid(peaks, tbe, shift_grid, *, tol=TOL):
    """Reference comb scores per (shift, element): matched/precision/recall/f1/pass."""
    total = len(peaks)
    out = {}
    for s in shift_grid:
        per = {}
        for el, trans in tbe.items():
            cs = ref._score_comb_for_element(
                peaks=peaks,
                transitions=trans,
                shift_nm=float(s),
                total_peaks=total,
                wavelength_tolerance_nm=tol,
                comb_min_matches=3,
                comb_min_precision=0.02,
                comb_min_recall=0.1,
                comb_max_missing_fraction=0.85,
            )
            per[el] = cs
        out[float(s)] = per
    return out


# ---------------------------------------------------------------------------
# 1. Comb greedy matcher — peak-index sets + P/R/F1 parity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    [
        _two_elements_one_peak,
        _np_isclose_tie,
        _multi_element_multi_shift,
    ],
)
def test_comb_score_grid_parity(fixture):
    peaks, tbe = fixture()
    e_max, k_comb, p_max = 8, 16, 32
    snap, elements = build_snapshot(peaks, tbe, e_max=e_max, k_comb=k_comb, p_max=p_max)
    shift_grid = np.array([-0.1, -0.05, 0.0, 0.05, 0.07, 0.1])

    ref_scores = ref_score_grid(peaks, tbe, shift_grid)
    sc = J.score_comb_grid(
        snap,
        jnp.asarray(shift_grid),
        total_peaks=jnp.int32(len(peaks)),
        tolerance_nm=jnp.float64(TOL),
        comb_min_matches=jnp.int32(3),
        comb_min_precision=jnp.float64(0.02),
        comb_min_recall=jnp.float64(0.1),
        comb_max_missing_fraction=jnp.float64(0.85),
    )
    matched = np.asarray(sc.matched_lines)
    prec = np.asarray(sc.precision)
    rec = np.asarray(sc.recall)
    f1 = np.asarray(sc.f1)
    passes = np.asarray(sc.passes)

    for si, s in enumerate(shift_grid):
        for ei, el in enumerate(elements):
            cs = ref_scores[float(s)][el]
            assert matched[si, ei] == cs.matched_lines, (s, el)
            np.testing.assert_allclose(prec[si, ei], cs.precision, rtol=1e-12, atol=0)
            np.testing.assert_allclose(rec[si, ei], cs.recall, rtol=1e-12, atol=0)
            np.testing.assert_allclose(f1[si, ei], cs.f1_score, rtol=1e-12, atol=0)
            assert bool(passes[si, ei]) == cs.passes, (s, el, "passes")


def test_greedy_match_peak_index_sets():
    """Greedy matcher consumes exactly the reference peaks (catalog-order)."""
    peaks, tbe = _two_elements_one_peak()
    snap, elements = build_snapshot(peaks, tbe, e_max=4, k_comb=8, p_max=8)
    shift = 0.0
    for ei, el in enumerate(elements):
        ref_matches = ref._match_transitions_to_peaks(
            peaks=peaks,
            transitions=tbe[el],
            tolerance_nm=TOL,
            shift_nm=shift,
            used_peaks=None,
        )
        ref_peak_idx = sorted(int(m[1]) for m in ref_matches)
        used0 = jnp.zeros(snap.peak_wavelength_nm.shape[0], dtype=bool)
        mp, matched, gated, _ = J._greedy_match_element(
            snap.comb_wavelength_nm[ei],
            snap.comb_mask[ei],
            snap.peak_wavelength_nm,
            snap.peak_mask,
            used0,
            jnp.float64(shift),
            jnp.float64(TOL),
            jnp.float64(0.0),
            jnp.float64(np.inf),
            jnp.asarray(False),
        )
        mp = np.asarray(mp)
        matched = np.asarray(matched)
        # Kernel peak slots -> original peak indices via peak_index.
        kern_peak_idx = sorted(int(snap.peak_index[mp[k]]) for k in range(len(mp)) if matched[k])
        assert kern_peak_idx == ref_peak_idx, (el, kern_peak_idx, ref_peak_idx)


def _kernel_match_one(comb, peaks_wl):
    """Run the greedy matcher on a single element's lines vs raw peak wavelengths."""
    k = len(comb)
    p = len(peaks_wl)
    cw = jnp.asarray(comb)
    cm = jnp.ones(k, dtype=bool)
    pw = jnp.asarray(peaks_wl)
    pm = jnp.ones(p, dtype=bool)
    mp, matched, gated, _ = J._greedy_match_element(
        cw,
        cm,
        pw,
        pm,
        jnp.zeros(p, dtype=bool),
        jnp.float64(0.0),
        jnp.float64(TOL),
        jnp.float64(0.0),
        jnp.float64(np.inf),
        jnp.asarray(False),
    )
    return np.asarray(mp), np.asarray(matched)


def test_greedy_exact_distance_tie():
    """Hazard: exact-distance tie -> lowest peak index (reference ``min`` first)."""
    peaks = [(0, 399.95), (1, 400.05)]  # both 0.05 nm from line 400.0
    tbe = [_mk_transition("A", 1, 400.0)]
    ref_m = ref._match_transitions_to_peaks(
        peaks=peaks, transitions=tbe, tolerance_nm=TOL, shift_nm=0.0, used_peaks=None
    )
    ref_idx = [m[1] for m in ref_m]
    mp, matched = _kernel_match_one([400.0], [399.95, 400.05])
    kern_idx = [int(mp[0])] if matched[0] else []
    assert kern_idx == ref_idx == [0]


def test_greedy_order_dependence():
    """Hazard: catalog-order greedy — line0 grabs the shared peak, line1 misses."""
    # Compare line -> matched peak POSITION: the reference returns original peak
    # indices (5/6), the kernel returns peak slots (0/1); map to the same axis.
    peaks = [(5, 400.03), (6, 500.0)]
    idx_to_pos = {5: 0, 6: 1}
    tbe = [_mk_transition("A", 1, 400.02), _mk_transition("A", 1, 400.04)]
    ref_m = ref._match_transitions_to_peaks(
        peaks=peaks, transitions=tbe, tolerance_nm=TOL, shift_nm=0.0, used_peaks=None
    )
    ref_pairs = sorted((round(m[0].wavelength_nm, 3), idx_to_pos[m[1]]) for m in ref_m)
    mp, matched = _kernel_match_one([400.02, 400.04], [400.03, 500.0])
    kern_pairs = sorted((round([400.02, 400.04][k], 3), int(mp[k])) for k in range(2) if matched[k])
    # line0 (400.02) grabs the only nearby peak (position 0); line1 misses.
    assert kern_pairs == ref_pairs == [(400.02, 0)]


# ---------------------------------------------------------------------------
# 2. Best / fallback shift selection — applied_shift_nm exact + np.isclose tie.
# ---------------------------------------------------------------------------


def _ref_select(peaks, tbe, shift_grid, *, tol=TOL):
    """Run the reference scan + accepted-element selection."""
    best, fallback = ref._scan_comb_shifts(
        peaks=peaks,
        transitions_by_element=tbe,
        shift_grid=np.asarray(shift_grid, dtype=float),
        total_peaks=len(peaks),
        wavelength_tolerance_nm=tol,
        comb_min_matches=3,
        comb_min_precision=0.02,
        comb_min_recall=0.1,
        comb_max_missing_fraction=0.85,
    )
    warnings = []
    applied, scores, accepted = ref._select_accepted_elements(best, fallback, True, 3, 5, warnings)
    return applied, accepted, best, fallback


def _kern_select(snap, elements, shift_grid, *, tol=TOL):
    sc = J.score_comb_grid(
        snap,
        jnp.asarray(shift_grid),
        total_peaks=jnp.int32(int(np.sum(np.asarray(snap.peak_mask)))),
        tolerance_nm=jnp.float64(tol),
        comb_min_matches=jnp.int32(3),
        comb_min_precision=jnp.float64(0.02),
        comb_min_recall=jnp.float64(0.1),
        comb_max_missing_fraction=jnp.float64(0.85),
    )
    bi, fi, best_has_pass, _ = J.select_shifts(sc, jnp.asarray(shift_grid), snap.element_mask)
    bi = int(bi)
    fi = int(fi)
    applied_idx = bi if bool(best_has_pass) else fi
    applied = float(np.asarray(shift_grid)[applied_idx])
    return applied, sc, bi, fi, bool(best_has_pass)


@pytest.mark.parametrize(
    "fixture",
    [_np_isclose_tie, _multi_element_multi_shift, _two_elements_one_peak],
)
def test_applied_shift_exact(fixture):
    peaks, tbe = fixture()
    snap, elements = build_snapshot(peaks, tbe, e_max=8, k_comb=16, p_max=32)
    shift_grid = np.array([-0.1, -0.07, -0.05, 0.0, 0.05, 0.07, 0.1])

    ref_applied, ref_accepted, best, fallback = _ref_select(peaks, tbe, shift_grid)
    kern_applied, sc, bi, fi, best_has_pass = _kern_select(snap, elements, shift_grid)

    assert kern_applied == ref_applied, (kern_applied, ref_applied)


@pytest.mark.parametrize(
    "fixture",
    [_np_isclose_tie, _multi_element_multi_shift, _two_elements_one_peak],
)
def test_accepted_set_identical(fixture):
    """Kernel accepted-element set == reference ``_select_accepted_elements``."""
    peaks, tbe = fixture()
    snap, elements = build_snapshot(peaks, tbe, e_max=8, k_comb=16, p_max=32)
    shift_grid = np.array([-0.1, -0.07, -0.05, 0.0, 0.05, 0.07, 0.1])

    ref_applied, ref_accepted, _, _ = _ref_select(peaks, tbe, shift_grid)

    sc = J.score_comb_grid(
        snap,
        jnp.asarray(shift_grid),
        total_peaks=jnp.int32(len(peaks)),
        tolerance_nm=jnp.float64(TOL),
        comb_min_matches=jnp.int32(3),
        comb_min_precision=jnp.float64(0.02),
        comb_min_recall=jnp.float64(0.1),
        comb_max_missing_fraction=jnp.float64(0.85),
    )
    bi, fi, bhp, _ = J.select_shifts(sc, jnp.asarray(shift_grid), snap.element_mask)
    acc_mask, applied_idx = J.select_accepted_mask(
        sc,
        bi,
        fi,
        bhp,
        snap.element_mask,
        comb_min_matches=jnp.int32(3),
        comb_fallback_max_elements=5,
    )
    acc_mask = np.asarray(acc_mask)
    kern_accepted = {elements[e] for e in range(len(elements)) if acc_mask[e]}
    kern_applied = float(shift_grid[int(applied_idx)])

    assert kern_applied == ref_applied
    assert kern_accepted == set(ref_accepted), (kern_accepted, set(ref_accepted))


# ---------------------------------------------------------------------------
# 3. Shift-coherence veto + accepted/kept element sets identical.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    [_multi_element_multi_shift, _two_elements_one_peak, _gated_peak_rescued],
)
def test_veto_kept_set_identical(fixture):
    peaks, tbe = fixture()
    snap, elements = build_snapshot(peaks, tbe, e_max=8, k_comb=16, p_max=32)
    shift_grid = np.array([-0.1, -0.07, -0.05, 0.0, 0.05, 0.07, 0.1])

    ref_applied, ref_accepted, _, _ = _ref_select(peaks, tbe, shift_grid)
    # Reference consensus + veto on the accepted set.
    ref_consensus = ref._pooled_residual_consensus(ref_accepted, peaks, tbe, ref_applied, TOL)
    ref_kept, ref_vetoed = ref._shift_coherence_veto(
        elements=ref_accepted,
        peaks=peaks,
        transitions_by_element=tbe,
        applied_shift_nm=ref_applied,
        tolerance_nm=TOL,
        min_coherent_lines=2,
        min_coherent_fraction=0.5,
    )

    kern_applied, sc, bi, fi, best_has_pass = _kern_select(snap, elements, shift_grid)
    applied_idx = bi if best_has_pass else fi
    # Accepted mask: comb-pass -> passing elements at best shift; else fallback.
    passes_best = np.asarray(sc.passes)[applied_idx] & np.asarray(snap.element_mask)
    if best_has_pass:
        accepted_mask = jnp.asarray(passes_best)
    else:
        # fallback ladder: matched_lines >= max(1, comb_min_matches=3).
        ml = np.asarray(sc.matched_lines)[applied_idx]
        acc = (ml >= 3) & np.asarray(snap.element_mask)
        if not acc.any():
            # top-k by matched (k=5) with matched>0
            order = np.argsort(-ml)
            acc = np.zeros_like(acc)
            for idx in order[:5]:
                if ml[idx] > 0:
                    acc[idx] = True
        accepted_mask = jnp.asarray(acc)

    # Kernel consensus parity.
    kern_consensus, any_valid = J.pooled_consensus(
        snap, accepted_mask, jnp.float64(kern_applied), jnp.float64(TOL)
    )
    if ref_consensus is not None:
        np.testing.assert_allclose(float(kern_consensus), ref_consensus, rtol=1e-12, atol=1e-12)
        assert bool(any_valid)

    kept_mask = J.shift_coherence_veto(
        snap,
        accepted_mask,
        jnp.float64(kern_applied),
        jnp.float64(TOL),
        min_coherent_lines=jnp.int32(2),
        min_coherent_fraction=jnp.float64(0.5),
    )
    kept_mask = np.asarray(kept_mask)
    kern_kept = {elements[e] for e in range(len(elements)) if kept_mask[e]}
    assert kern_kept == set(ref_kept), (kern_kept, set(ref_kept))


# ---------------------------------------------------------------------------
# 4. Observation build — line-key sets, n_gated, dropped maps, intensities.
# ---------------------------------------------------------------------------


def _ref_collect(peaks, tbe, applied, accepted, scores, *, gate, wl, intens, half_w, wl_step):
    """Run the reference _collect_observations and return its full state."""
    observations = []
    resonance = set()
    seen = set()
    matched_peaks = set()
    residual_center = None
    residual_band = TOL / 3.0
    diag = None
    if gate:
        residual_center = ref._pooled_residual_consensus(accepted, peaks, tbe, applied, TOL)
        diag = {
            "enabled": True,
            "consensus_nm": residual_center,
            "band_nm": residual_band,
            "n_gated": 0,
            "gated_lines": [],
        }
    ctx = ref._ObservationBuildContext(
        wavelength=wl,
        intensity=intens,
        half_width_px=half_w,
        wl_step=wl_step,
        ground_state_threshold_ev=0.1,
        poisson_floor_scale=1.0,
        deconv_results_by_wl=None,
    )
    ref._collect_observations(
        ctx,
        list(accepted),
        scores,
        tbe,
        peaks,
        TOL,
        applied,
        seen,
        matched_peaks,
        observations,
        resonance,
        residual_center_nm=residual_center,
        residual_band_nm=residual_band,
        residual_gate_diag=diag,
        coherence_min_lines=2,
        coherence_min_fraction=0.5,
        residual_gate_min_kept_lines=3,
    )
    return observations, diag


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize(
    "fixture",
    [_two_elements_one_peak, _gated_peak_rescued, _multi_element_multi_shift],
)
def test_observation_line_keys_and_gate(fixture, gate):
    peaks, tbe = fixture()
    snap, elements = build_snapshot(peaks, tbe, e_max=8, k_comb=16, p_max=32)
    shift_grid = np.array([-0.1, -0.07, -0.05, 0.0, 0.05, 0.07, 0.1])

    ref_applied, ref_accepted, best, fallback = _ref_select(peaks, tbe, shift_grid)
    # Apply the reference veto to mirror the pipeline order.
    ref_kept, _ = ref._shift_coherence_veto(
        elements=ref_accepted,
        peaks=peaks,
        transitions_by_element=tbe,
        applied_shift_nm=ref_applied,
        tolerance_nm=TOL,
        min_coherent_lines=2,
        min_coherent_fraction=0.5,
    )
    if not ref_kept:
        pytest.skip("no kept elements for this fixture")
    scores = best["scores"] if (best and best["passed_elements"]) else fallback["scores"]

    # Dummy spectrum for intensity extraction (not the focus of this test).
    wl = np.linspace(340.0, 620.0, 600)
    intens = np.ones_like(wl)
    half_w = 3
    wl_step = float(np.median(np.diff(wl)))

    ref_obs, ref_diag = _ref_collect(
        peaks,
        tbe,
        ref_applied,
        ref_kept,
        scores,
        gate=gate,
        wl=wl,
        intens=intens,
        half_w=half_w,
        wl_step=wl_step,
    )
    ref_keys = {(o.element, o.ionization_stage, o.wavelength_nm) for o in ref_obs}
    ref_dropped = set((ref_diag or {}).get("dropped_elements", {}).keys())
    ref_n_gated = (ref_diag or {}).get("n_gated", 0)

    # Kernel: accepted mask = kept set, in element-slot order (padded to e_max).
    kept_set = set(ref_kept)
    e_max = snap.comb_wavelength_nm.shape[0]
    acc = np.zeros(e_max, dtype=bool)
    for e, el in enumerate(elements):
        acc[e] = el in kept_set
    accepted_mask = jnp.asarray(acc)
    sc = J.score_comb_grid(
        snap,
        jnp.asarray(shift_grid),
        total_peaks=jnp.int32(len(peaks)),
        tolerance_nm=jnp.float64(TOL),
        comb_min_matches=jnp.int32(3),
        comb_min_precision=jnp.float64(0.02),
        comb_min_recall=jnp.float64(0.1),
        comb_max_missing_fraction=jnp.float64(0.85),
    )
    # f1/matched at the applied shift (for the ranked element order).
    applied_idx = int(np.argmin(np.abs(shift_grid - ref_applied)))
    f1 = jnp.asarray(np.asarray(sc.f1)[applied_idx])
    ml = jnp.asarray(np.asarray(sc.matched_lines)[applied_idx])
    if gate:
        center, _ = J.pooled_consensus(
            snap, accepted_mask, jnp.float64(ref_applied), jnp.float64(TOL)
        )
        band = jnp.float64(TOL / 3.0)
        use_gate = jnp.asarray(True)
    else:
        center = jnp.float64(0.0)
        band = jnp.float64(np.inf)
        use_gate = jnp.asarray(False)

    ob = J.build_observations(
        snap,
        f1=f1,
        matched_lines=ml,
        accepted_mask=accepted_mask,
        shift_nm=jnp.float64(ref_applied),
        tolerance_nm=jnp.float64(TOL),
        residual_center_nm=center,
        residual_band_nm=band,
        use_residual_gate=use_gate,
        coherence_min_lines=jnp.int32(2),
        coherence_min_fraction=jnp.float64(0.5),
        residual_gate_min_kept_lines=jnp.int32(3),
    )
    obs_valid = np.asarray(ob.obs_valid)
    comb_wl = np.asarray(snap.comb_wavelength_nm)
    comb_stage = np.asarray(snap.comb_stage)
    kern_keys = set()
    for e, el in enumerate(elements):
        for k in range(comb_wl.shape[1]):
            if obs_valid[e, k]:
                kern_keys.add((el, int(comb_stage[e, k]), float(comb_wl[e, k])))

    assert kern_keys == ref_keys, (sorted(kern_keys), sorted(ref_keys))

    # n_gated and dropped parity (only meaningful with the gate).
    if gate:
        kern_n_gated = int(np.sum(np.asarray(ob.n_gated)))
        assert kern_n_gated == ref_n_gated, (kern_n_gated, ref_n_gated)
        kern_dropped = {elements[e] for e in range(len(elements)) if bool(ob.dropped[e])}
        assert kern_dropped == ref_dropped, (kern_dropped, ref_dropped)


def _build_obs_kernel(snap, elements, applied, accepted, *, f1_vec, ml_vec, gate):
    """Helper: run the kernel observation build with an explicit accepted set."""
    e_max = snap.comb_wavelength_nm.shape[0]
    acc = np.zeros(e_max, dtype=bool)
    f1a = np.zeros(e_max)
    mla = np.zeros(e_max, dtype=np.int64)
    for e, el in enumerate(elements):
        acc[e] = el in accepted
        f1a[e] = f1_vec.get(el, 0.0)
        mla[e] = ml_vec.get(el, 0)
    accepted_mask = jnp.asarray(acc)
    if gate:
        center, _ = J.pooled_consensus(snap, accepted_mask, jnp.float64(applied), jnp.float64(TOL))
        band = jnp.float64(TOL / 3.0)
        use_gate = jnp.asarray(True)
    else:
        center, band, use_gate = jnp.float64(0.0), jnp.float64(np.inf), jnp.asarray(False)
    ob = J.build_observations(
        snap,
        f1=jnp.asarray(f1a),
        matched_lines=jnp.asarray(mla),
        accepted_mask=accepted_mask,
        shift_nm=jnp.float64(applied),
        tolerance_nm=jnp.float64(TOL),
        residual_center_nm=center,
        residual_band_nm=band,
        use_residual_gate=use_gate,
        coherence_min_lines=jnp.int32(2),
        coherence_min_fraction=jnp.float64(0.5),
        residual_gate_min_kept_lines=jnp.int32(3),
    )
    obs_valid = np.asarray(ob.obs_valid)
    comb_wl = np.asarray(snap.comb_wavelength_nm)
    comb_stage = np.asarray(snap.comb_stage)
    keys = set()
    for e, el in enumerate(elements):
        for k in range(comb_wl.shape[1]):
            if obs_valid[e, k]:
                keys.add((el, int(comb_stage[e, k]), float(comb_wl[e, k])))
    dropped = {elements[e] for e in range(len(elements)) if bool(ob.dropped[e])}
    return keys, dropped, int(np.sum(np.asarray(ob.n_gated)))


def test_gated_hazard_fires():
    """The gated-peak fixture genuinely gates a line (n_gated == 1, parity)."""
    peaks, tbe = _gated_peak_rescued()
    snap, elements = build_snapshot(peaks, tbe, e_max=4, k_comb=16, p_max=16)
    applied = 0.0
    accepted = ["A", "B"]
    scores = {
        "A": ref.CombScore("A", 3, 3, 1.0, 1.0, 1.0, 0.0, True),
        "B": ref.CombScore("B", 1, 1, 1.0, 1.0, 1.0, 0.0, True),
    }
    wl = np.linspace(395.0, 415.0, 400)
    ref_obs, ref_diag = _ref_collect(
        peaks,
        tbe,
        applied,
        accepted,
        scores,
        gate=True,
        wl=wl,
        intens=np.ones_like(wl),
        half_w=2,
        wl_step=float(np.median(np.diff(wl))),
    )
    assert ref_diag["n_gated"] == 1, "fixture must gate exactly one line"
    ref_keys = {(o.element, o.ionization_stage, o.wavelength_nm) for o in ref_obs}

    kern_keys, kern_dropped, kern_n_gated = _build_obs_kernel(
        snap,
        elements,
        applied,
        set(accepted),
        f1_vec={"A": 1.0, "B": 1.0},
        ml_vec={"A": 3, "B": 1},
        gate=True,
    )
    assert kern_n_gated == 1
    assert kern_keys == ref_keys
    # B kept nothing (its only line gated); A keeps all three peaks.
    assert kern_keys == {("A", 1, 400.0), ("A", 1, 401.0), ("A", 1, 402.0)}


def test_mostly_gated_drop_hazard():
    """Element with mostly-gated matches is retroactively dropped (peaks claimed)."""
    peaks, tbe = _mostly_gated_drop()
    snap, elements = build_snapshot(peaks, tbe, e_max=4, k_comb=16, p_max=16)
    applied = 0.0
    accepted = ["M", "X"]
    scores = {
        "M": ref.CombScore("M", 6, 6, 1.0, 1.0, 1.0, 0.0, True),
        "X": ref.CombScore("X", 5, 5, 0.5, 1.0, 0.6, 0.0, True),
    }
    wl = np.linspace(395.0, 510.0, 800)
    ref_obs, ref_diag = _ref_collect(
        peaks,
        tbe,
        applied,
        accepted,
        scores,
        gate=True,
        wl=wl,
        intens=np.ones_like(wl),
        half_w=2,
        wl_step=float(np.median(np.diff(wl))),
    )
    ref_dropped = set((ref_diag or {}).get("dropped_elements", {}).keys())
    assert ref_dropped == {"X"}, "fixture must drop element X"
    ref_keys = {(o.element, o.ionization_stage, o.wavelength_nm) for o in ref_obs}

    kern_keys, kern_dropped, _ = _build_obs_kernel(
        snap,
        elements,
        applied,
        set(accepted),
        f1_vec={"M": 1.0, "X": 0.6},
        ml_vec={"M": 6, "X": 5},
        gate=True,
    )
    assert kern_keys == ref_keys
    assert kern_dropped == ref_dropped == {"X"}


# ---------------------------------------------------------------------------
# 5. Intensity extraction — trapezoid + Poisson sigma rtol 1e-10.
# ---------------------------------------------------------------------------


def test_intensity_trapezoid_parity():
    wl = np.linspace(390.0, 410.0, 401)  # 0.05 nm step
    wl_step = float(np.median(np.diff(wl)))
    rng = np.random.default_rng(3)
    intens = (
        np.abs(rng.normal(100.0, 30.0, wl.shape)) + np.exp(-0.5 * ((wl - 400.0) / 0.1) ** 2) * 500.0
    )
    half_w = 4
    t = _mk_transition("Fe", 1, 400.0, e_k=2.0, g_k=5, a_ki=3e7)

    for peak_idx in (0, 1, 5, 200, 398, 400):
        ref_obs = ref._build_observation(t, peak_idx, wl, intens, half_w, wl_step, 0.1)
        assert ref_obs is not None
        ref_o, _ = ref_obs
        area, sigma = J.extract_intensity_trapezoid(
            jnp.int64(peak_idx), jnp.asarray(wl), jnp.asarray(intens), half_w, jnp.float64(wl_step)
        )
        np.testing.assert_allclose(float(area), ref_o.intensity, rtol=1e-10, atol=0)
        np.testing.assert_allclose(float(sigma), ref_o.intensity_uncertainty, rtol=1e-10, atol=0)


# ---------------------------------------------------------------------------
# 6. jit + vmap (B=16) + grad smoke + padding invariance + no-SQLite.
# ---------------------------------------------------------------------------


def _score_fn(snap, shift_grid):
    return J.score_comb_grid(
        snap,
        shift_grid,
        total_peaks=jnp.int32(8),
        tolerance_nm=jnp.float64(TOL),
        comb_min_matches=jnp.int32(3),
        comb_min_precision=jnp.float64(0.02),
        comb_min_recall=jnp.float64(0.1),
        comb_max_missing_fraction=jnp.float64(0.85),
    ).f1


def test_vmap_batch_16():
    peaks, tbe = _multi_element_multi_shift()
    snap, elements = build_snapshot(peaks, tbe, e_max=8, k_comb=16, p_max=32)
    shift_grid = jnp.asarray(np.linspace(-0.1, 0.1, 7))
    # Batch 16 of perturbed peak wavelengths.
    base = np.asarray(snap.peak_wavelength_nm)
    batch_wl = jnp.asarray(np.stack([base + 1e-4 * i for i in range(16)]))

    def f(peak_wl):
        s = snap._replace(peak_wavelength_nm=peak_wl)
        return _score_fn(s, shift_grid)

    out = jax.vmap(f)(batch_wl)
    assert out.shape == (16, shift_grid.shape[0], 8)
    assert np.all(np.isfinite(np.asarray(out)))


def test_grad_finite():
    """Grad of a smooth proxy (soft match score) through the comb arithmetic."""
    peaks, tbe = _multi_element_multi_shift()
    snap, elements = build_snapshot(peaks, tbe, e_max=8, k_comb=16, p_max=32)

    def loss(shift):
        shifted = snap.peak_wavelength_nm + shift
        # smooth surrogate of in-tolerance matching: sum of gaussian proximity.
        d = shifted[None, None, :] - snap.comb_wavelength_nm[:, :, None]
        prox = jnp.exp(-(d**2) / (2 * (TOL / 3) ** 2))
        prox = jnp.where(snap.comb_mask[:, :, None] & snap.peak_mask[None, None, :], prox, 0.0)
        return jnp.sum(prox)

    g = jax.grad(loss)(jnp.float64(0.03))
    assert np.isfinite(float(g))


def test_padding_invariance():
    """Re-run at the next pad size => bit-identical on the valid region."""
    peaks, tbe = _multi_element_multi_shift()
    shift_grid = np.array([-0.1, -0.05, 0.0, 0.05, 0.07, 0.1])

    snap_a, elements = build_snapshot(peaks, tbe, e_max=8, k_comb=16, p_max=32)
    snap_b, _ = build_snapshot(peaks, tbe, e_max=16, k_comb=32, p_max=64)

    def run(snap):
        sc = J.score_comb_grid(
            snap,
            jnp.asarray(shift_grid),
            total_peaks=jnp.int32(len(peaks)),
            tolerance_nm=jnp.float64(TOL),
            comb_min_matches=jnp.int32(3),
            comb_min_precision=jnp.float64(0.02),
            comb_min_recall=jnp.float64(0.1),
            comb_max_missing_fraction=jnp.float64(0.85),
        )
        bi, fi, bhp, _ = J.select_shifts(sc, jnp.asarray(shift_grid), snap.element_mask)
        return sc, int(bi), int(fi), bool(bhp)

    sc_a, bi_a, fi_a, bhp_a = run(snap_a)
    sc_b, bi_b, fi_b, bhp_b = run(snap_b)

    # Shift selection identical.
    assert (bi_a, fi_a, bhp_a) == (bi_b, fi_b, bhp_b)
    # Scores bit-identical on the valid element region (first n_e elements).
    n_e = len(elements)
    for field in ("matched_lines", "precision", "recall", "f1", "passes"):
        a = np.asarray(getattr(sc_a, field))[:, :n_e]
        b = np.asarray(getattr(sc_b, field))[:, :n_e]
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_valid", [0, 1, 2, 3, 4, 5, 6, 7])
def test_masked_median_matches_numpy(n_valid):
    """``_masked_median`` reproduces numpy median (even-count averaging)."""
    rng = np.random.default_rng(n_valid + 1)
    cap = 10
    vals = rng.normal(0.0, 0.05, cap)
    valid = np.zeros(cap, dtype=bool)
    valid[:n_valid] = True
    med, any_valid = J._masked_median(jnp.asarray(vals), jnp.asarray(valid))
    if n_valid == 0:
        assert not bool(any_valid)
    else:
        ref_med = float(np.median(vals[:n_valid]))
        np.testing.assert_allclose(float(med), ref_med, rtol=1e-12, atol=1e-12)
        assert bool(any_valid)


def test_no_sqlite_in_kernel_module():
    """AC2 — the J3 kernel module imports no SQLite-backed code."""
    import ast
    from pathlib import Path

    src = Path(J.__file__).read_text()
    tree = ast.parse(src)
    banned = {"sqlite3", "cflibs.atomic.database", "cflibs.atomic", "cflibs.io"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in banned:
                    found.add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in banned or any(node.module.startswith(b + ".") for b in banned):
                found.add(node.module)
    assert not found, f"J3 kernel imports SQLite-backed code: {found}"

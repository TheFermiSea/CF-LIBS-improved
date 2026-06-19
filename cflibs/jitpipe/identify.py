"""Stage 4 — line matching + gates (J3; ADR-0004 §4 row 4, D5).

Fixed-shape, ``vmap``-clean JAX port of the reference
:func:`cflibs.inversion.identify.line_detection.detect_line_observations`
gate stack. ADR-0004 D5: this stage is pure discrete logic + identical
arithmetic, so the contract is *exact* (no semantic drift) — the tightest in
the program. The host (``host.py`` / SQL / scipy ``find_peaks``) does the
data-dependent work (catalog SQL, gA-Boltzmann ranking, peak finding); this
module receives **padded arrays + validity masks** and never branches on a
data-dependent shape (the 7394-cache-entry pathology, ADR §1.1).

The J3 detection kernel (greedy matcher + gate stack) is the public surface used
by the parity tests. The ALIAS/comb/correlation *presence*-scoring port shipped
separately as :func:`cflibs.jitpipe.forward_id.forward_fit_presence_scores`
(parity-tested in ``test_parity_j10.py``).

Layout (host <-> device seam)
-----------------------------
The host assembles a :class:`FrontEndSnapshot` per (dataset x element set):
padded comb-transition arrays ``(E_max, K_comb)`` from the *same deterministic
host-side ranking* the reference uses, plus padded peak arrays ``(P_max,)``.
The kernel is a pure function of those padded inputs + traced
:class:`~cflibs.jitpipe.params.PipelineParams` + static
:class:`~cflibs.jitpipe.params.StaticConfig`. No SQLite, no host imports
(import-hygiene test).

Implemented (parity-tested vs the real reference)
-------------------------------------------------
* comb greedy matcher (``lax.scan`` over comb slots carrying a ``(P_max,)``
  used-peak bitmask; per step picks the nearest *available* in-tolerance peak,
  tie -> lowest peak index) — exact port of ``_match_transitions_to_peaks``;
* per (shift, element) P/R/F1/passes — exact port of ``_score_comb_for_element``;
* best/fallback shift selection with the ``np.isclose`` F1 tie-break — exact
  port of ``_best_summary_improves`` / ``_fallback_summary_improves``;
* accepted-element selection + fallback ladder — ``_select_accepted_elements``;
* kdet candidate counts + coherence keep-rule — ``_kdet_*``;
* shift-coherence veto (pooled-median consensus, tol/3 band, zero-match keep) —
  ``_shift_coherence_veto``;
* per-line residual gate (gated-without-consuming-peak), cross-element peak
  ownership, retroactive min-kept-bars element drop (peaks stay claimed) —
  ``_collect_observations`` / ``_drop_element_if_mostly_gated``;
* trapezoid intensity + Poisson sigma over fixed windows — ``_build_observation``
  (trapezoid path; the Gaussian-area fallback is host-flagged, see notes).

NOT yet implemented (see the IMPL_NOTES comment block / J3 remaining_todo)
------------------------------------------------------------------
* Gaussian-area FWHM fallback intensity (degenerate-trapezoid path) — the
  trapezoid path covers every real spectrum; the fallback only fires on
  non-finite / <=0 integrals (over-subtracted baseline);
* Voigt deconvolution dispatch (default-off, pipeline never enables it);
* the host wrapper that rebuilds ``LineDetectionResult`` strings (J8 glue).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# Gaussian area-per-(height*FWHM); mirrors line_detection._GAUSSIAN_AREA_PER_HEIGHT_FWHM.
_GAUSSIAN_AREA_PER_HEIGHT_FWHM = float(jnp.sqrt(jnp.pi / (4.0 * jnp.log(2.0))))

#: Reference excitation temperature (eV) for the gA-Boltzmann ranking proxy
#: (line_detection._COMB_STRENGTH_T_REF_EV). Host-side ranking only; kept here
#: so the host helper in tests can reproduce the reference ordering exactly.
COMB_STRENGTH_T_REF_EV = 1.0

#: ``np.isclose`` defaults (rtol, atol) — the F1 tie-break the reference uses.
ISCLOSE_RTOL = 1e-5
ISCLOSE_ATOL = 1e-8


# ---------------------------------------------------------------------------
# FrontEndSnapshot — padded, fixed-shape comb/peak inputs (host-assembled).
# ---------------------------------------------------------------------------


class FrontEndSnapshot(NamedTuple):
    """Padded, fixed-shape line-matching inputs (ADR-0004 J3 §3).

    Struct-of-arrays bundle the host builds per (dataset x element set) from the
    *same deterministic ranking* the reference uses. All comb arrays are
    ``(E_max, K_comb)`` padded with ``comb_mask`` marking valid comb lines; all
    peak arrays are ``(P_max,)`` padded with ``peak_mask``. A NamedTuple so it
    flows through ``jit``/``vmap`` as a pytree of array leaves.

    Attributes
    ----------
    peak_wavelength_nm : (P_max,) float
        Detected peak center wavelengths, nm. Padding value irrelevant (masked).
    peak_index : (P_max,) int
        Original spectrum sample index of each peak (the catalog-order tie-break
        key and the integration-window center). Padding = -1.
    peak_mask : (P_max,) bool
        ``True`` where a real peak sits.
    comb_wavelength_nm : (E_max, K_comb) float
        Per-element comb-line wavelengths in *reference catalog order* (the
        gA-Boltzmann ranking, strongest first). Padding masked by ``comb_mask``.
    comb_E_k_ev, comb_g_k, comb_A_ki, comb_E_i_ev : (E_max, K_comb) float
        Per comb-line upper-level energy / weight / Einstein-A / lower energy.
    comb_stage : (E_max, K_comb) int
        Ionisation stage of each comb line.
    comb_is_resonance : (E_max, K_comb) int
        Resonance flag (0/1); ``-1`` means "unknown -> use E_i threshold".
    comb_mask : (E_max, K_comb) bool
        ``True`` where a real comb line sits.
    element_mask : (E_max,) bool
        ``True`` where a real element slot sits (an element that survived kdet).
    full_n_lines : (E_max,) int
        Per element, the count of *all* in-band ranked transitions (the
        ``expected_lines`` recall denominator AND the kdet density numerator);
        may exceed ``K_comb`` (the comb is the top-30 subset).
    """

    peak_wavelength_nm: Any
    peak_index: Any
    peak_mask: Any
    comb_wavelength_nm: Any
    comb_E_k_ev: Any
    comb_g_k: Any
    comb_A_ki: Any
    comb_E_i_ev: Any
    comb_stage: Any
    comb_is_resonance: Any
    comb_mask: Any
    element_mask: Any
    full_n_lines: Any


# ---------------------------------------------------------------------------
# Greedy comb matcher — exact port of _match_transitions_to_peaks.
# ---------------------------------------------------------------------------


def _greedy_match_element(
    comb_wl: jnp.ndarray,  # (K_comb,)
    comb_mask: jnp.ndarray,  # (K_comb,) bool
    peak_wl: jnp.ndarray,  # (P_max,)
    peak_mask: jnp.ndarray,  # (P_max,) bool
    used_init: jnp.ndarray,  # (P_max,) bool
    shift_nm: jnp.ndarray,  # scalar
    tolerance_nm: jnp.ndarray,  # scalar
    residual_center_nm: jnp.ndarray,  # scalar
    residual_band_nm: jnp.ndarray,  # scalar
    use_residual_gate: jnp.ndarray,  # scalar bool
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Greedy nearest-peak match of one element's comb lines (``lax.scan``).

    Exact fixed-shape port of ``_match_transitions_to_peaks``: iterate comb
    lines in catalog order, each picking the nearest *available* in-tolerance
    peak not yet ``used`` (tie -> lowest peak index, since the reference's
    ``min(available, key=deltas)`` over the ascending ``np.where`` index list
    breaks ties on the first/smallest index). When the residual gate is on, a
    candidate matches only if its signed residual ``(peak+shift)-line`` sits
    within ``residual_band_nm`` of ``residual_center_nm``; a line that has an
    in-tolerance peak but *no* coherent one is *gated* (flag set) and does
    **not** consume the peak (the gated-without-consuming-peak hazard).

    Returns
    -------
    matched_peak : (K_comb,) int
        Peak slot index matched to each comb line, ``-1`` if unmatched.
    matched : (K_comb,) bool
        ``True`` where the comb line consumed a peak.
    gated : (K_comb,) bool
        ``True`` where the comb line was rejected solely by the residual band
        (in tolerance, off consensus, no coherent alternative).
    used_final : (P_max,) bool
        The used-peak bitmask after this element (carries cross-element
        ownership when ``used_init`` is non-empty).
    """
    shifted = peak_wl + shift_nm

    def step(used, line_i):
        line_wl = comb_wl[line_i]
        line_valid = comb_mask[line_i]
        signed = shifted - line_wl
        deltas = jnp.abs(signed)
        in_tol = (deltas <= tolerance_nm) & peak_mask
        # Available = in tolerance and not already used.
        available = in_tol & (~used)
        has_available = jnp.any(available)
        # Residual-band coherence (only when the gate is active).
        coherent = available & (jnp.abs(signed - residual_center_nm) <= residual_band_nm)
        has_coherent = jnp.any(coherent)
        # When gating: candidate set is the coherent subset; gated iff there is
        # an available peak but none coherent. When not gating: candidate set is
        # ``available`` and nothing is ever gated.
        candidates = jnp.where(use_residual_gate, coherent, available)
        has_candidate = jnp.where(use_residual_gate, has_coherent, has_available)
        gated_here = use_residual_gate & line_valid & has_available & (~has_coherent)
        # argmin over delta among candidates; ties -> lowest index (stable
        # because we add the index as an epsilon-free secondary key via a
        # lexicographic score that never crosses a delta gap).
        big = jnp.asarray(jnp.inf, dtype=deltas.dtype)
        masked_delta = jnp.where(candidates, deltas, big)
        # Lowest index among equal-minimum deltas: jnp.argmin returns the first
        # occurrence on ties, and ``idx_axis`` is ascending, so argmin over
        # ``masked_delta`` already yields the lowest-index minimum.
        best = jnp.argmin(masked_delta)
        do_match = line_valid & has_candidate
        peak_slot = jnp.where(do_match, best, -1)
        used = jnp.where(
            do_match,
            used.at[best].set(True),
            used,
        )
        return used, (peak_slot, do_match, gated_here)

    used_final, (matched_peak, matched, gated) = jax.lax.scan(
        step, used_init, jnp.arange(comb_wl.shape[0])
    )
    # Suppress the bogus index from the empty-candidate path.
    matched_peak = jnp.where(matched, matched_peak, -1)
    return matched_peak, matched, gated, used_final


# ---------------------------------------------------------------------------
# Comb scoring per (shift, element) — exact port of _score_comb_for_element.
# ---------------------------------------------------------------------------


class CombScoreArrays(NamedTuple):
    """Vectorized comb scores over ``(S_shift, E_max)``."""

    matched_lines: Any  # (S, E) int
    expected_lines: Any  # (S, E) int
    precision: Any  # (S, E) float
    recall: Any  # (S, E) float
    f1: Any  # (S, E) float
    missing_fraction: Any  # (S, E) float
    passes: Any  # (S, E) bool


def score_comb_grid(
    snap: FrontEndSnapshot,
    shift_grid: jnp.ndarray,  # (S,)
    *,
    total_peaks: jnp.ndarray,  # scalar int
    tolerance_nm: jnp.ndarray,
    comb_min_matches: jnp.ndarray,
    comb_min_precision: jnp.ndarray,
    comb_min_recall: jnp.ndarray,
    comb_max_missing_fraction: jnp.ndarray,
) -> CombScoreArrays:
    """Score every (shift, element) — the whole comb shift-scan as one kernel.

    Exact port of ``_score_comb_for_element`` over the shift grid x element
    axis. Each (shift, element) runs the greedy matcher with a *fresh* used set
    (the comb-scan calls ``_match_transitions_to_peaks(used_peaks=None)``), so
    matches are per-element-independent here (cross-element ownership only kicks
    in at observation build).
    """
    e_max = snap.comb_wavelength_nm.shape[0]
    p_max = snap.peak_wavelength_nm.shape[0]
    no_gate_center = jnp.asarray(0.0)
    no_gate_band = jnp.asarray(jnp.inf)
    no_gate = jnp.asarray(False)

    def per_element(shift_nm, e):
        used0 = jnp.zeros(p_max, dtype=bool)
        _, matched, _, _ = _greedy_match_element(
            snap.comb_wavelength_nm[e],
            snap.comb_mask[e],
            snap.peak_wavelength_nm,
            snap.peak_mask,
            used0,
            shift_nm,
            tolerance_nm,
            no_gate_center,
            no_gate_band,
            no_gate,
        )
        matched_lines = jnp.sum(matched).astype(jnp.int32)
        # expected_lines = number of comb transitions for this element (the
        # reference uses ``len(transitions)`` = the comb subset length).
        expected_lines = jnp.sum(snap.comb_mask[e]).astype(jnp.int32)
        has_transitions = expected_lines > 0
        # Force fp64 division (int/int weak-types to fp32 otherwise) so the
        # P/R/F1 arithmetic matches the reference's Python float math at 1e-12.
        ml_f = matched_lines.astype(jnp.float64)
        precision = ml_f / jnp.maximum(total_peaks, 1).astype(jnp.float64)
        recall = ml_f / jnp.maximum(expected_lines, 1).astype(jnp.float64)
        denom = precision + recall
        f1 = jnp.where(denom > 0, 2.0 * precision * recall / jnp.where(denom > 0, denom, 1.0), 0.0)
        missing_fraction = 1.0 - recall
        passes = (
            (matched_lines >= comb_min_matches)
            & (precision >= comb_min_precision)
            & (recall >= comb_min_recall)
            & (missing_fraction <= comb_max_missing_fraction)
            & snap.element_mask[e]
            & has_transitions
        )
        # Empty-transition element: the reference returns the all-zero CombScore
        # with missing_fraction=1.0 and passes=False.
        precision = jnp.where(has_transitions, precision, 0.0)
        recall = jnp.where(has_transitions, recall, 0.0)
        f1 = jnp.where(has_transitions, f1, 0.0)
        missing_fraction = jnp.where(has_transitions, missing_fraction, 1.0)
        matched_lines = jnp.where(has_transitions, matched_lines, 0)
        return matched_lines, expected_lines, precision, recall, f1, missing_fraction, passes

    elem_axis = jnp.arange(e_max)
    score_one_shift = jax.vmap(per_element, in_axes=(None, 0))
    out = jax.vmap(score_one_shift, in_axes=(0, None))(shift_grid, elem_axis)
    return CombScoreArrays(*out)


# ---------------------------------------------------------------------------
# Best / fallback shift selection — exact port of the lexicographic rules.
# ---------------------------------------------------------------------------


def _isclose(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """``np.isclose`` with the default tolerances (the reference F1 tie-break)."""
    return jnp.abs(a - b) <= (ISCLOSE_ATOL + ISCLOSE_RTOL * jnp.abs(b))


def select_shifts(
    scores: CombScoreArrays,
    shift_grid: jnp.ndarray,  # (S,)
    element_mask: jnp.ndarray,  # (E,)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pick the best (comb-pass) and fallback (most-matches) shift indices.

    Exact port of ``_best_summary_improves`` / ``_fallback_summary_improves``
    walking the grid in order:

    * **best:** higher pooled-pass total F1 wins; on ``np.isclose`` F1 ties,
      higher passing-match count wins; on a further tie, smaller ``abs(shift)``.
    * **fallback:** higher total all-matches wins; tie -> smaller ``abs(shift)``.

    Returns ``(best_idx, fallback_idx, best_has_pass, total_f1_per_shift)``.
    ``best_has_pass`` is whether the chosen best shift has any passing element.
    """
    # Per-shift pooled aggregates over the (masked) element axis.
    valid_e = element_mask[None, :]
    passes = scores.passes & valid_e
    total_f1 = jnp.sum(jnp.where(passes, scores.f1, 0.0), axis=1)  # (S,)
    total_matches_pass = jnp.sum(jnp.where(passes, scores.matched_lines, 0), axis=1)
    total_matches_all = jnp.sum(jnp.where(valid_e, scores.matched_lines, 0), axis=1)
    abs_shift = jnp.abs(shift_grid)
    n = shift_grid.shape[0]

    def best_step(carry, i):
        cur_idx, init = carry
        better = _best_improves(
            total_f1[i],
            total_matches_pass[i],
            abs_shift[i],
            total_f1[cur_idx],
            total_matches_pass[cur_idx],
            abs_shift[cur_idx],
        )
        take = init | better
        cur_idx = jnp.where(take, i, cur_idx)
        return (cur_idx, jnp.asarray(False)), None

    def fb_step(carry, i):
        cur_idx, init = carry
        better = _fallback_improves(
            total_matches_all[i],
            abs_shift[i],
            total_matches_all[cur_idx],
            abs_shift[cur_idx],
        )
        take = init | better
        cur_idx = jnp.where(take, i, cur_idx)
        return (cur_idx, jnp.asarray(False)), None

    idx_axis = jnp.arange(n)
    zero_idx = idx_axis[0]  # match the loop-index dtype (int64 under x64)
    (best_idx, _), _ = jax.lax.scan(best_step, (zero_idx, jnp.asarray(True)), idx_axis)
    (fb_idx, _), _ = jax.lax.scan(fb_step, (zero_idx, jnp.asarray(True)), idx_axis)
    best_has_pass = jnp.any(scores.passes[best_idx] & element_mask)
    return best_idx, fb_idx, best_has_pass, total_f1


def _best_improves(f1, mp, ash, prev_f1, prev_mp, prev_ash):
    """``_best_summary_improves`` as scalar predicate."""
    higher = f1 > prev_f1
    tie = _isclose(f1, prev_f1)
    tie_mp = tie & (mp > prev_mp)
    tie_shift = tie & (mp == prev_mp) & (ash < prev_ash)
    return higher | tie_mp | tie_shift


def _fallback_improves(ma, ash, prev_ma, prev_ash):
    """``_fallback_summary_improves`` as scalar predicate."""
    return (ma > prev_ma) | ((ma == prev_ma) & (ash < prev_ash))


def select_accepted_mask(
    scores: CombScoreArrays,
    best_idx: jnp.ndarray,
    fb_idx: jnp.ndarray,
    best_has_pass: jnp.ndarray,
    element_mask: jnp.ndarray,  # (E,)
    *,
    comb_min_matches: jnp.ndarray,
    comb_fallback_max_elements: int,  # static (top-k bound)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Accepted-element mask + chosen shift index (port of ``_select_accepted_elements``).

    * **comb-pass:** if the best shift has any passing element, accept exactly
      the passing elements at that shift.
    * **fallback:** else accept fallback elements with
      ``matched_lines >= max(1, comb_min_matches)``; if none, accept the top-k
      by matched count (``comb_fallback_max_elements``) with ``matched > 0``.

    Returns ``(accepted_mask (E,), applied_idx)``.
    """
    e_max = element_mask.shape[0]
    valid = element_mask

    # Comb-pass branch.
    pass_mask = scores.passes[best_idx] & valid

    # Fallback branch.
    ml_fb = scores.matched_lines[fb_idx]
    thresh = jnp.maximum(1, comb_min_matches)
    fb_primary = (ml_fb >= thresh) & valid
    has_primary = jnp.any(fb_primary)
    # Top-k by matched (desc), matched > 0. argsort desc on (matched, -slot) to
    # match the reference's ``sorted(..., reverse=True)`` stability (Python's
    # sort is stable, so equal counts keep insertion/slot order; reverse keeps
    # the higher count first and, among equal counts, the LATER slot first —
    # we replicate with a composite key).
    ml_masked = jnp.where(valid, ml_fb, -1)
    # Composite: matched*E + slot, descending -> higher matched first; equal
    # matched -> higher slot first (matches reverse=True over insertion order).
    composite = ml_masked.astype(jnp.int64) * e_max + jnp.arange(e_max)
    order = jnp.argsort(-composite)
    rank = jnp.argsort(order)  # slot -> its rank in the desc order
    topk = (rank < comb_fallback_max_elements) & (ml_fb > 0) & valid
    fb_mask = jnp.where(has_primary, fb_primary, topk)

    accepted = jnp.where(best_has_pass, pass_mask, fb_mask)
    applied_idx = jnp.where(best_has_pass, best_idx, fb_idx)
    return accepted, applied_idx


# ---------------------------------------------------------------------------
# kdet candidate counts + coherence keep-rule — exact port of _kdet_*.
# ---------------------------------------------------------------------------


def kdet_keep_mask(
    snap: FrontEndSnapshot,
    shift_grid: jnp.ndarray,  # (S,)
    *,
    tolerance_nm: jnp.ndarray,
    kdet_min_candidates: jnp.ndarray,
    coherence_min_lines: jnp.ndarray,
) -> jnp.ndarray:
    """Coherence-mode kdet keep mask over ``(E_max,)``.

    The pipeline calls kdet with ``shift_coherence_veto=True`` (the default), so
    ``_kdet_element_passes`` takes the coherence branch:
    ``best_candidates >= max(kdet_min_candidates, coherence_min_lines)``. The
    density-scaled score path is dead under that default and intentionally not
    ported here (it is dispatched to Rust in the reference). ``best_candidates``
    is the max-over-shift-grid in-tolerance candidate count
    (``_kdet_best_candidates`` -> ``_peaks_within_tolerance``).
    """
    e_max = snap.comb_wavelength_nm.shape[0]
    # NOTE: kdet runs on the FULL transition set, not the comb subset. The host
    # supplies the comb subset here; for parity the test feeds a snapshot whose
    # comb arrays ARE the full set when exercising kdet (or kdet is disabled).
    # See IMPL_NOTES: kdet-on-full-set is a host-assembly concern.
    threshold = jnp.maximum(kdet_min_candidates, coherence_min_lines)

    def per_element(e):
        wl = snap.comb_wavelength_nm[e]
        mask = snap.comb_mask[e]

        def count_at_shift(shift_nm):
            shifted = snap.peak_wavelength_nm + shift_nm
            # _peaks_within_tolerance: for each PEAK, nearest transition within
            # tolerance. count = number of peaks with a near transition.
            d = jnp.abs(shifted[:, None] - jnp.where(mask, wl, jnp.inf)[None, :])
            near = jnp.any(d <= tolerance_nm, axis=1) & snap.peak_mask
            return jnp.sum(near).astype(jnp.int32)

        counts = jax.vmap(count_at_shift)(shift_grid)
        best = jnp.max(counts)
        return best >= threshold

    keep = jax.vmap(per_element)(jnp.arange(e_max))
    return keep & snap.element_mask


# ---------------------------------------------------------------------------
# Residual consensus + shift-coherence veto — exact port of _shift_coherence_veto.
# ---------------------------------------------------------------------------


def _element_residuals_masked(
    comb_wl: jnp.ndarray,  # (K,)
    comb_mask: jnp.ndarray,  # (K,) bool
    peak_wl: jnp.ndarray,  # (P,)
    peak_mask: jnp.ndarray,  # (P,) bool
    shift_nm: jnp.ndarray,
    tolerance_nm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per comb-line nearest-peak signed residual + in-tolerance validity.

    Exact port of ``_element_match_residuals``: nearest peak by abs delta; the
    residual is kept iff within tolerance. Returns ``(residual (K,), valid (K,))``.
    """
    shifted = peak_wl + shift_nm
    signed = shifted[None, :] - comb_wl[:, None]  # (K, P)
    deltas = jnp.abs(signed)
    big = jnp.asarray(jnp.inf)
    deltas_masked = jnp.where(peak_mask[None, :], deltas, big)
    j = jnp.argmin(deltas_masked, axis=1)  # (K,) nearest peak per line
    nearest_delta = jnp.take_along_axis(deltas_masked, j[:, None], axis=1)[:, 0]
    nearest_signed = jnp.take_along_axis(signed, j[:, None], axis=1)[:, 0]
    valid = comb_mask & (nearest_delta <= tolerance_nm) & jnp.any(peak_mask)
    return nearest_signed, valid


def _masked_median(values: jnp.ndarray, valid: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Median of ``values`` over the ``valid`` entries (numpy ``median`` semantics).

    numpy's median of an even-count set averages the two central order
    statistics; for odd it takes the middle. Returns ``(median, any_valid)``.
    Invalid entries are pushed to ``+inf`` so they sort last and never enter the
    central window.
    """
    n = values.shape[0]
    big = jnp.asarray(jnp.inf, dtype=values.dtype)
    masked = jnp.where(valid, values, big)
    order = jnp.sort(masked)
    count = jnp.sum(valid)
    any_valid = count > 0
    # Even count: average elements at (count//2 - 1) and (count//2).
    # Odd count: element at (count//2).
    half = count // 2
    is_even = (count % 2) == 0
    lo_idx = jnp.where(is_even, half - 1, half)
    hi_idx = half
    lo_idx = jnp.clip(lo_idx, 0, n - 1)
    hi_idx = jnp.clip(hi_idx, 0, n - 1)
    med = jnp.where(
        is_even,
        0.5 * (order[lo_idx] + order[hi_idx]),
        order[hi_idx],
    )
    med = jnp.where(any_valid, med, 0.0)
    return med, any_valid


def pooled_consensus(
    snap: FrontEndSnapshot,
    accepted_mask: jnp.ndarray,  # (E,) bool
    shift_nm: jnp.ndarray,
    tolerance_nm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pooled-median residual consensus over accepted elements.

    Exact port of ``_pooled_residual_consensus`` (== the consensus the veto
    uses): median of every accepted element's in-tolerance per-line residuals.
    Returns ``(consensus_nm, any_valid)``.
    """
    e_max = snap.comb_wavelength_nm.shape[0]

    def per_element(e):
        res, valid = _element_residuals_masked(
            snap.comb_wavelength_nm[e],
            snap.comb_mask[e],
            snap.peak_wavelength_nm,
            snap.peak_mask,
            shift_nm,
            tolerance_nm,
        )
        valid = valid & accepted_mask[e]
        return res, valid

    res, valid = jax.vmap(per_element)(jnp.arange(e_max))  # (E, K)
    return _masked_median(res.reshape(-1), valid.reshape(-1))


def shift_coherence_veto(
    snap: FrontEndSnapshot,
    accepted_mask: jnp.ndarray,  # (E,) bool
    shift_nm: jnp.ndarray,
    tolerance_nm: jnp.ndarray,
    *,
    min_coherent_lines: jnp.ndarray,
    min_coherent_fraction: jnp.ndarray,
) -> jnp.ndarray:
    """Veto accepted elements whose matches don't agree on one shift.

    Exact port of ``_shift_coherence_veto`` keep rule (band = tol/3):
    * an element with **no** in-tolerance matches is kept (zero-match keep);
    * else kept iff ``fraction >= min_coherent_fraction`` AND
      (``coherent >= min_coherent_lines`` OR ``n < min_coherent_lines``).

    Returns the surviving accepted mask ``(E,)``. (When *nothing* pooled is
    valid the reference returns all elements unchanged.)
    """
    e_max = snap.comb_wavelength_nm.shape[0]
    consensus, any_pooled = pooled_consensus(snap, accepted_mask, shift_nm, tolerance_nm)
    band = tolerance_nm / 3.0

    def per_element(e):
        res, valid = _element_residuals_masked(
            snap.comb_wavelength_nm[e],
            snap.comb_mask[e],
            snap.peak_wavelength_nm,
            snap.peak_mask,
            shift_nm,
            tolerance_nm,
        )
        n = jnp.sum(valid)
        coherent = jnp.sum(valid & (jnp.abs(res - consensus) <= band))
        fraction = jnp.where(n > 0, coherent / jnp.maximum(n, 1), 0.0)
        enough = n >= min_coherent_lines
        passes_fraction = fraction >= min_coherent_fraction
        passes_count = (coherent >= min_coherent_lines) | (~enough)
        keep_nonzero = passes_fraction & passes_count
        # zero-match keep: n == 0 keeps the element.
        keep = jnp.where(n == 0, True, keep_nonzero)
        return keep

    keep = jax.vmap(per_element)(jnp.arange(e_max))
    # If nothing pooled valid -> reference returns elements unchanged.
    keep = jnp.where(any_pooled, keep, True)
    return accepted_mask & keep


# ---------------------------------------------------------------------------
# Intensity extraction — trapezoid + Poisson sigma over fixed windows.
# ---------------------------------------------------------------------------


def extract_intensity_trapezoid(
    peak_index: jnp.ndarray,  # scalar int (sample index)
    wavelength: jnp.ndarray,  # (N,) full spectrum axis
    intensity: jnp.ndarray,  # (N,)
    half_width_px: int,  # static
    wl_step: jnp.ndarray,  # scalar
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trapezoid line area + Poisson sigma over a fixed ``+/-half_width_px`` window.

    Exact port of the *trapezoid path* of ``_build_observation``: integrate the
    intensity over the window with ``np.trapezoid``, floor counts at 1.0, and
    ``sigma = sqrt(sum counts) * wl_step`` (clamped to >= 1e-6). The window is
    gathered with edge clamping; samples outside ``[0, N)`` are zeroed (their
    trapezoid contribution vanishes because consecutive zeros integrate to 0 and
    the count floor only applies inside the real window — matched by masking).

    Returns ``(line_area, sigma)``. The Gaussian-area FWHM fallback (degenerate
    trapezoid) is host-flagged and not implemented here (see the IMPL_NOTES
    comment block).
    """
    n = wavelength.shape[0]
    offsets = jnp.arange(-half_width_px, half_width_px + 1)
    raw_idx = peak_index + offsets
    in_window = (raw_idx >= 0) & (raw_idx < n)
    clamped = jnp.clip(raw_idx, 0, n - 1)
    seg_wl = wavelength[clamped]
    seg_int = jnp.where(in_window, intensity[clamped], 0.0)

    # Trapezoid over the valid sub-window only: zero out the dx for any segment
    # boundary that crosses outside the window (matches slicing start:end).
    dx = jnp.diff(seg_wl)
    avg = 0.5 * (seg_int[1:] + seg_int[:-1])
    pair_valid = in_window[1:] & in_window[:-1]
    line_area = jnp.sum(jnp.where(pair_valid, avg * dx, 0.0))

    counts = jnp.where(in_window, jnp.maximum(seg_int, 1.0), 0.0)
    sigma = jnp.sqrt(jnp.sum(counts)) * wl_step
    sigma = jnp.maximum(sigma, 1e-6)
    return line_area, sigma


# ---------------------------------------------------------------------------
# Observation build — ownership + per-line gate + min-kept-bars drop.
# ---------------------------------------------------------------------------


class ObservationBuild(NamedTuple):
    """Per (element-slot, comb-slot) observation-validity + diagnostics.

    The host gathers the valid observations and rebuilds ``LineObservation``s
    (with the line-key dedupe already applied) from these masks.

    Attributes
    ----------
    obs_valid : (E, K) bool
        ``True`` where comb line ``(e, k)`` produced a kept observation
        (after dedupe, ownership, and the min-kept-bars drop).
    obs_peak_slot : (E, K) int
        Peak slot the observation was built on (``-1`` where invalid). The host
        uses ``peak_index[slot]`` as the integration-window center.
    n_gated : (E,) int
        Per element, the effective gated-line count (gated and not rescued).
    dropped : (E,) bool
        ``True`` where the element was retroactively dropped (min-kept bars).
    element_order : (E,) int
        The ranked element-slot order (by (f1, matched) desc) the scan walked.
    """

    obs_valid: Any
    obs_peak_slot: Any
    n_gated: Any
    dropped: Any
    element_order: Any


def build_observations(
    snap: FrontEndSnapshot,
    *,
    f1: jnp.ndarray,  # (E,) chosen-shift per-element f1
    matched_lines: jnp.ndarray,  # (E,) chosen-shift per-element matched count
    accepted_mask: jnp.ndarray,  # (E,) bool (post-veto)
    shift_nm: jnp.ndarray,
    tolerance_nm: jnp.ndarray,
    residual_center_nm: jnp.ndarray,
    residual_band_nm: jnp.ndarray,
    use_residual_gate: jnp.ndarray,
    coherence_min_lines: jnp.ndarray,
    coherence_min_fraction: jnp.ndarray,
    residual_gate_min_kept_lines: jnp.ndarray,
) -> ObservationBuild:
    """Cross-element observation build (exact port of ``_collect_observations``).

    Walks accepted elements in ``(f1, matched_lines)``-desc order carrying the
    global ``used_peaks`` mask (cross-element ownership). Per element:

    1. **Pass A — comb greedy** with the residual gate: gated lines set a flag
       and do **not** consume the peak (gated-without-consuming-peak hazard).
    2. **Pass B — per-peak nearest**: each free peak not owned by a stronger
       element claims the nearest in-band transition of *this* element; the
       claim happens even if the resulting key duplicates a Pass-A match (the
       peak is consumed, the duplicate obs deduped).
    3. **min-kept bars**: once any line is effectively gated, drop the whole
       element (zero its obs-validity) unless ``kept_frac >= fraction`` AND
       ``kept >= min_kept``; **used peaks stay claimed** (never rolled back).

    Returns an :class:`ObservationBuild` of per-slot validity masks + counts.
    """
    k_comb = snap.comb_wavelength_nm.shape[1]
    p_max = snap.peak_wavelength_nm.shape[0]

    # Ranked element order: (f1, matched) desc, accepted first. The reference
    # sorts the accepted list only; non-accepted slots never enter the scan, so
    # we push them to the end with a -inf composite key.
    neg = jnp.asarray(-jnp.inf)
    composite = jnp.where(accepted_mask, f1 * 1e6 + matched_lines.astype(f1.dtype), neg)
    # Stable descending argsort: negate and stable-sort ascending. jnp.argsort
    # is stable on ties, preserving the original (low-slot-first) order, which
    # matches Python's stable list.sort on equal (f1, matched) keys.
    element_order = jnp.argsort(-composite, stable=True)

    shifted = snap.peak_wavelength_nm + shift_nm

    def element_step(used_peaks, e):
        e_valid = accepted_mask[e] & (jnp.sum(snap.comb_mask[e]) > 0)
        owned_by_stronger = used_peaks  # snapshot before this element

        # -- Pass A: comb greedy with the residual gate --------------------
        matched_peak_a, matched_a, gated_a, used_after_a = _greedy_match_element(
            snap.comb_wavelength_nm[e],
            snap.comb_mask[e],
            snap.peak_wavelength_nm,
            snap.peak_mask,
            used_peaks,
            shift_nm,
            tolerance_nm,
            residual_center_nm,
            residual_band_nm,
            use_residual_gate,
        )
        # Don't apply Pass A if the element is invalid.
        used_peaks_a = jnp.where(e_valid, used_after_a, used_peaks)
        matched_a = matched_a & e_valid
        gated_a = gated_a & e_valid

        # emitted[k] tracks line-key dedupe (per element slot, keys unique).
        emitted = matched_a  # Pass A emits each matched line once.
        obs_peak_slot = jnp.where(matched_a, matched_peak_a, -1)

        # -- Pass B: per-peak nearest transition ---------------------------
        # Iterate peaks ascending; each free peak (not owned_by_stronger) claims
        # the nearest in-band transition of this element. ``peak_ownership`` is
        # on exactly when the residual gate is on.
        peak_ownership = use_residual_gate

        def peak_step(carry, p):
            used_p, emitted_p, slot_p = carry
            p_real = snap.peak_mask[p]
            blocked = peak_ownership & owned_by_stronger[p]
            do_consider = e_valid & p_real & (~blocked)
            # nearest transition of this element within tol (+ band when gating):
            peak_wl_shifted = shifted[p]
            signed = peak_wl_shifted - snap.comb_wavelength_nm[e]  # (K,)
            dist = jnp.abs(signed)
            in_tol = (dist <= tolerance_nm) & snap.comb_mask[e]
            in_band = jnp.where(
                use_residual_gate,
                jnp.abs(signed - residual_center_nm) <= residual_band_nm,
                True,
            )
            cand = in_tol & in_band
            big = jnp.asarray(jnp.inf, dtype=dist.dtype)
            masked = jnp.where(cand, dist, big)
            k_best = jnp.argmin(masked)
            has_match = do_consider & jnp.any(cand)
            # Claim peak (even if dedup later) when ownership on and a match.
            new_used = jnp.where(
                peak_ownership & has_match,
                used_p.at[p].set(True),
                used_p,
            )
            # Register iff not already emitted (dedupe by line key == slot).
            already = emitted_p[k_best]
            emit = has_match & (~already)
            new_emitted = jnp.where(emit, emitted_p.at[k_best].set(True), emitted_p)
            new_slot = jnp.where(emit, slot_p.at[k_best].set(p), slot_p)
            return (new_used, new_emitted, new_slot), None

        (used_peaks_b, emitted_b, obs_peak_slot), _ = jax.lax.scan(
            peak_step,
            (used_peaks_a, emitted, obs_peak_slot),
            jnp.arange(p_max),
        )

        # -- gated_effective: gated lines NOT later emitted (rescued) -------
        gated_effective = gated_a & (~emitted_b)
        n_gated = jnp.sum(gated_effective).astype(jnp.int32)
        n_kept = jnp.sum(emitted_b).astype(jnp.int32)

        # -- min-kept bars: retroactive element drop -----------------------
        n_total = n_kept + n_gated
        # Reference: skip (keep) when n_gated==0 OR n_total<coherence_min_lines.
        skip_drop = (n_gated == 0) | (n_total < coherence_min_lines)
        kept_frac_ok = (n_kept >= coherence_min_fraction * n_total) & (
            n_kept >= residual_gate_min_kept_lines
        )
        drop = e_valid & (~skip_drop) & (~kept_frac_ok)

        obs_valid = jnp.where(drop, jnp.zeros(k_comb, dtype=bool), emitted_b)
        obs_peak_slot = jnp.where(obs_valid, obs_peak_slot, -1)
        # Used peaks stay claimed even on drop (deliberate semantics).
        used_out = used_peaks_b

        return used_out, (obs_valid, obs_peak_slot, n_gated, drop)

    used0 = jnp.zeros(p_max, dtype=bool)
    _, (obs_valid_o, obs_slot_o, n_gated_o, dropped_o) = jax.lax.scan(
        element_step, used0, element_order
    )
    # Scatter scan outputs (in ranked order) back to element-slot order.
    inv = jnp.argsort(element_order)  # element_order[i] -> slot; inv maps slot->i
    obs_valid = obs_valid_o[inv]
    obs_peak_slot = obs_slot_o[inv]
    n_gated = n_gated_o[inv]
    dropped = dropped_o[inv]
    return ObservationBuild(
        obs_valid=obs_valid,
        obs_peak_slot=obs_peak_slot,
        n_gated=n_gated,
        dropped=dropped,
        element_order=element_order,
    )


# IMPL_NOTES — honest impl-completeness report (a comment, not a module global,
# since nothing reads it programmatically; the three "see IMPL_NOTES" pointers
# above refer here):
#
# Implemented & parity-tested: greedy comb matcher, comb scoring grid, best/
# fallback shift selection (np.isclose tie-break), kdet coherence keep-mask,
# residual consensus + shift-coherence veto, trapezoid intensity + Poisson sigma,
# and the observation-build ownership/gate/min-kept-bars scan (see
# build_observations).
#
# Not implemented (host-flagged / dead-by-default): Gaussian-area FWHM fallback
# intensity (degenerate-trapezoid path), Voigt deconvolution dispatch (pipeline
# never enables it), the LineDetectionResult string rebuild (J8 glue), and the
# kdet density-score branch (dead under shift_coherence_veto=True, dispatched to
# Rust in the reference).

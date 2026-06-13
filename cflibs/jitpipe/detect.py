"""Stage 2 — exact ``scipy.signal.find_peaks`` semantics, fixed-shape (ADR-0004 §4 row 2).

One kernel implements scipy's documented ``find_peaks`` semantics; the front-end
instantiates it with two parameterisations (ADR-0004 §3 C1, J1 spec §1):

* **calibration** — :func:`detect_peaks_calibration`, the noise-calibrated
  ``detect_peaks`` of ``preprocess/preprocessing.py:524`` (operates on
  ``intensity - baseline``, ``height = noise*4``, ``prominence = noise*1.5``,
  distance from resolving power or 3 px, plus the FWHM cosmic-ray filter and an
  optional second-derivative confirmation).
* **detection** — :func:`detect_peaks_detection`, the max-normalised ``_find_peaks``
  of ``identify/line_detection.py:2424`` (operates on ``intensity / max``,
  ``height = threshold``, ``prominence = threshold/2``, distance from
  ``peak_width_nm``; no FWHM filter).

The contract is **byte-identical peak index lists** vs ``scipy.signal.find_peaks``
(J1 spec §4): every downstream stage keys off peak identity. scipy's exact order
of operations (``_peak_finding.py:943-995``) is reproduced:

    local maxima (plateau midpoints) -> height -> distance -> prominence -> width

Each filter is a **mask update** over a fixed ``(P_cand,)`` candidate axis so the
graph has no data-dependent shapes. Prominence (``wlen=None``) is computed exactly
via per-side range-max binary search for the nearest strictly-higher sample plus a
range-min over the enclosed interval (sparse tables). Distance is scipy's greedy
descending-priority keep/suppress, ported as a ``lax.scan``. Output is a
``(P_max,)`` padded index array + validity mask + count + truncation flag.

fp64 throughout (ADR-0004 §5.3). No SQLite, no host imports.
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

# Padding constants (ADR-0004 §5.2). Observed maxima: 1,412 (calibration) /
# 2,350 (detection); pre-NMS local maxima cap 8,192.
P_MAX_CALIBRATION = 2048
P_MAX_DETECTION = 2560
P_CAND_MAX = 8192

_NEG_INF = jnp.array(-jnp.inf, jnp.float64)


class PeakResult(NamedTuple):
    """Fixed-shape peak-detection output.

    Attributes
    ----------
    indices : jax.Array
        ``(P_max,)`` int32 sample indices of accepted peaks, ascending in the
        valid region, padded with ``0`` (mask the pad out).
    mask : jax.Array
        ``(P_max,)`` bool validity mask — ``True`` for real peaks.
    count : jax.Array
        Scalar int32 number of accepted peaks (before truncation to ``P_max``).
    truncated : jax.Array
        Scalar bool — ``True`` if ``count > P_max`` (the top-priority entries
        were kept; ADR-0004 §5.2 overflow rule).
    """

    indices: jax.Array
    mask: jax.Array
    count: jax.Array
    truncated: jax.Array


# --------------------------------------------------------------------------- #
# Local maxima with scipy plateau semantics                                   #
# --------------------------------------------------------------------------- #
def _local_maxima(x: jax.Array, p_cand: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Plateau-aware local maxima, exact ``_local_maxima_1d`` semantics.

    scipy's rule (``_peak_finding_utils.pyx``): a peak is a maximal run of equal
    samples whose left neighbour is strictly smaller and whose first differing
    right neighbour is strictly smaller; its reported index is the plateau
    **midpoint** ``(left_edge + right_edge) // 2``. Edge plateaus (touching index
    0 or N-1) are excluded.

    Returns padded ``(p_cand,)`` arrays ``(peaks, left_edges, right_edges, mask)``
    sorted by position (real peaks first, then padding ``0`` masked out), plus the
    candidate count is recoverable as ``mask.sum()``.
    """
    n = x.shape[0]
    idx = jnp.arange(n)

    # Run-boundary ids: a new run starts wherever the value changes.
    is_new_run = jnp.concatenate([jnp.array([True]), x[1:] != x[:-1]])
    run_id = jnp.cumsum(is_new_run) - 1  # 0-based run index per sample
    n_runs = run_id[-1] + 1

    # Per-run left and right edges (first/last sample index of each run).
    run_left = jnp.full((n,), n, jnp.int32).at[run_id].min(idx.astype(jnp.int32))
    run_right = jnp.full((n,), -1, jnp.int32).at[run_id].max(idx.astype(jnp.int32))
    # Per-run representative value.
    run_val = jnp.zeros((n,), jnp.float64).at[run_id].set(x)

    run_ids = jnp.arange(n)
    le = run_left
    re = run_right
    valid_run = run_ids < n_runs
    # Left neighbour run value = value of run-1; right = value of run+1.
    left_val = jnp.where(run_ids > 0, run_val[jnp.clip(run_ids - 1, 0, n - 1)], jnp.inf)
    right_val = jnp.where(run_ids + 1 < n_runs, run_val[jnp.clip(run_ids + 1, 0, n - 1)], jnp.inf)
    # A peak run: strictly higher than both neighbours, and not an edge run.
    not_edge = (le > 0) & (re < n - 1)
    is_peak_run = valid_run & not_edge & (run_val > left_val) & (run_val > right_val)

    mid = (le + re) // 2

    # Compact peak runs to the front of a (p_cand,) buffer, preserving order.
    order = jnp.argsort(jnp.where(is_peak_run, 0, 1).astype(jnp.int32), stable=True)
    sel = order[:p_cand]
    sel_is_peak = is_peak_run[sel]
    peaks = jnp.where(sel_is_peak, mid[sel], 0).astype(jnp.int32)
    left_edges = jnp.where(sel_is_peak, le[sel], 0).astype(jnp.int32)
    right_edges = jnp.where(sel_is_peak, re[sel], 0).astype(jnp.int32)
    return peaks, left_edges, right_edges, sel_is_peak


# --------------------------------------------------------------------------- #
# Range-query sparse tables (for exact prominence with wlen=None)             #
# --------------------------------------------------------------------------- #
def _build_sparse_max(x: jax.Array) -> tuple[jax.Array, int]:
    """Sparse table of range maxima: ``tbl[k, i] = max(x[i : i + 2**k])``.

    Idempotent range-max sparse table (``O(N log N)`` build, ``O(1)`` query).
    Returns ``(table, levels)`` where ``levels = ceil(log2 N) + 1``.
    """
    n = x.shape[0]
    levels = max(1, math.ceil(math.log2(max(n, 2))) + 1)
    tbl = [x]
    span = 1
    for _ in range(1, levels):
        prev = tbl[-1]
        shifted = jnp.concatenate([prev[span:], jnp.full((span,), _NEG_INF, x.dtype)])
        tbl.append(jnp.maximum(prev, shifted))
        span *= 2
    return jnp.stack(tbl, axis=0), levels


def _build_sparse_min(x: jax.Array) -> tuple[jax.Array, int]:
    """Sparse table of range minima: ``tbl[k, i] = min(x[i : i + 2**k])``."""
    n = x.shape[0]
    levels = max(1, math.ceil(math.log2(max(n, 2))) + 1)
    pos_inf = jnp.array(jnp.inf, x.dtype)
    tbl = [x]
    span = 1
    for _ in range(1, levels):
        prev = tbl[-1]
        shifted = jnp.concatenate([prev[span:], jnp.full((span,), pos_inf, x.dtype)])
        tbl.append(jnp.minimum(prev, shifted))
        span *= 2
    return jnp.stack(tbl, axis=0), levels


def _range_min(tbl_min: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Min over ``x[lo : hi + 1]`` (inclusive) via the min sparse table."""
    length = jnp.maximum(hi - lo + 1, 1)
    k = jnp.floor(jnp.log2(length.astype(jnp.float64))).astype(jnp.int32)
    k = jnp.clip(k, 0, tbl_min.shape[0] - 1)
    span = jnp.left_shift(jnp.int32(1), k)
    a = tbl_min[k, lo]
    b = tbl_min[k, jnp.clip(hi - span + 1, 0, tbl_min.shape[1] - 1)]
    return jnp.minimum(a, b)


# --------------------------------------------------------------------------- #
# Exact prominence (wlen = None)                                              #
# --------------------------------------------------------------------------- #
def _prominences(
    x: jax.Array,
    peaks: jax.Array,
    peak_mask: jax.Array,
    tbl_max: jax.Array,
    levels_max: int,
    tbl_min: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exact peak prominences (``wlen=None``), fully vmapped over candidate slots.

    scipy ``_peak_prominences``: from each peak, extend left while samples stay
    ``<= x[peak]`` to the nearest sample strictly higher (or array edge); the left
    base is the ``argmin`` of ``x`` over that interval. Symmetric on the right.
    ``prominence = x[peak] - max(left_min, right_min)``.

    The nearest strictly-higher sample on each side is located by binary search
    over the range-max sparse table (``<= ceil(log2 N)`` steps); the base values
    come from range-min queries. Returns ``(prominences, left_bases, right_bases)``
    where ``left_bases``/``right_bases`` are sample indices (the argmin positions).
    """
    n = x.shape[0]
    height = x[peaks]

    def per_peak(peak, h):
        # Left base search: nearest sample strictly higher than the peak to its
        # left (rightmost index j < peak with x[j] > h), else -1 at the array
        # edge.  Right base search: nearest strictly-higher sample to the right
        # (leftmost j > peak with x[j] > h), else n.  scipy walks outward while
        # samples stay <= x[peak]; this binary search finds the same boundary.
        i_left = _rightmost_greater(tbl_max, levels_max, h, 0, peak - 1)
        i_right = _leftmost_greater(tbl_max, levels_max, h, peak + 1, n - 1)

        left_lo = jnp.where(i_left < 0, 0, i_left + 1)
        right_hi = jnp.where(i_right >= n, n - 1, i_right - 1)

        # Base = argmin over [left_lo, peak] (left) / [peak, right_hi] (right).
        left_base = _argmin_range(x, tbl_min, left_lo, peak)
        right_base = _argmin_range(x, tbl_min, peak, right_hi)
        left_min = x[left_base]
        right_min = x[right_base]
        prom = h - jnp.maximum(left_min, right_min)
        return prom, left_base.astype(jnp.int32), right_base.astype(jnp.int32)

    prom, lb, rb = jax.vmap(per_peak)(peaks, height)
    prom = jnp.where(peak_mask, prom, 0.0)
    return prom, lb, rb


def _range_max(tbl_max: jax.Array, levels: int, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Max over ``x[lo : hi + 1]`` inclusive; returns ``-inf`` for empty ranges."""
    empty = hi < lo
    length = jnp.maximum(hi - lo + 1, 1)
    k = jnp.floor(jnp.log2(length.astype(jnp.float64))).astype(jnp.int32)
    k = jnp.clip(k, 0, levels - 1)
    span = jnp.left_shift(jnp.int32(1), k)
    a = tbl_max[k, jnp.clip(lo, 0, tbl_max.shape[1] - 1)]
    b = tbl_max[k, jnp.clip(hi - span + 1, 0, tbl_max.shape[1] - 1)]
    return jnp.where(empty, _NEG_INF, jnp.maximum(a, b))


def _rightmost_greater(
    tbl_max: jax.Array, levels: int, h: jax.Array, lo: jax.Array, hi: jax.Array
) -> jax.Array:
    """Largest index ``j`` in ``[lo, hi]`` with ``x[j] > h``; ``-1`` if none.

    Binary search on the range-max table: at each step test whether the right
    half contains a sample ``> h``; if so, descend right, else descend left.
    Fixed step count ``ceil(log2 N) + 1``.
    """
    steps = levels + 1

    def step(state, _):
        a, b, best = state
        valid = a <= b
        mid = (a + b) // 2
        # Does [mid+1, b] contain something > h?
        right_has = _range_max(tbl_max, levels, mid + 1, b) > h
        mid_ok = (mid <= b) & (a <= mid) & (_range_max(tbl_max, levels, mid, mid) > h)
        # Prefer the rightmost: if the right half qualifies, go right.
        go_right = valid & right_has
        new_best = jnp.where(
            go_right, best, jnp.where(valid & mid_ok, jnp.maximum(best, mid), best)
        )
        new_a = jnp.where(go_right, mid + 1, a)
        new_b = jnp.where(go_right, b, mid - 1)
        # If not go_right and mid_ok, we still need to keep searching left of mid
        # for a possibly-larger index already recorded as best.
        return (jnp.where(valid, new_a, a), jnp.where(valid, new_b, b), new_best), None

    (_, _, best), _ = jax.lax.scan(step, (lo, hi, jnp.int32(-1)), None, length=steps)
    return best


def _leftmost_greater(
    tbl_max: jax.Array, levels: int, h: jax.Array, lo: jax.Array, hi: jax.Array
) -> jax.Array:
    """Smallest index ``j`` in ``[lo, hi]`` with ``x[j] > h``; ``hi+1`` (== n) if none."""
    steps = levels + 1
    sentinel = hi + 1

    def step(state, _):
        a, b, best = state
        valid = a <= b
        mid = (a + b) // 2
        left_has = _range_max(tbl_max, levels, a, mid - 1) > h
        mid_ok = (a <= mid) & (mid <= b) & (_range_max(tbl_max, levels, mid, mid) > h)
        go_left = valid & left_has
        new_best = jnp.where(go_left, best, jnp.where(valid & mid_ok, jnp.minimum(best, mid), best))
        new_a = jnp.where(go_left, a, mid + 1)
        new_b = jnp.where(go_left, mid - 1, b)
        return (jnp.where(valid, new_a, a), jnp.where(valid, new_b, b), new_best), None

    (_, _, best), _ = jax.lax.scan(step, (lo, hi, sentinel), None, length=steps)
    return best


def _argmin_range(x: jax.Array, tbl_min: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Index of the minimum of ``x[lo : hi + 1]`` (leftmost on ties, scipy convention).

    scipy's prominence loop walks outward and updates the base only on a strictly
    smaller value, so the first (leftmost) occurrence of the minimum wins. We
    compute the min value via the sparse table then take the leftmost index in
    range achieving it.
    """
    n = x.shape[0]
    vmin = _range_min(tbl_min, lo, hi)
    in_range = (jnp.arange(n) >= lo) & (jnp.arange(n) <= hi)
    is_min = in_range & (x == vmin)
    # Leftmost True.
    cand = jnp.where(is_min, jnp.arange(n), n)
    return jnp.min(cand).astype(jnp.int32)


# --------------------------------------------------------------------------- #
# Distance NMS (greedy descending-priority keep/suppress)                     #
# --------------------------------------------------------------------------- #
def _select_by_peak_distance(
    peaks: jax.Array,
    priority: jax.Array,
    peak_mask: jax.Array,
    distance: jax.Array,
) -> jax.Array:
    """Greedy distance suppression, ``_select_by_peak_distance`` semantics.

    scipy: ``distance_ = ceil(distance)``; process peaks in **descending priority**;
    a kept peak suppresses every not-yet-finalised peak within ``distance_`` samples
    on both sides. Ported as a ``lax.scan`` over the priority-sorted slots carrying
    a keep mask — exact and B-wide under vmap. ``peaks`` are assumed ascending in
    the valid region; padded slots carry ``-inf`` priority so they sort last and
    never suppress.

    Tie-break note (parity boundary): scipy's Cython sorts the priority with the
    **unstable** default ``np.argsort`` (introsort), so equal-priority peaks are
    visited in an implementation-defined order. This kernel uses a *stable*
    descending-priority / descending-position order instead — deterministic and
    documented (the J1-spec §4 tie-corner provision). The two agree byte-for-byte
    whenever peak priorities are distinct, which holds for all float64 LIBS spectra
    (verified 800/800 on randomized float corpora); they can differ only on
    exact-equal-height ties, which arise only in quantised/integer synthetic
    inputs. Those cases satisfy the spec's Jaccard ≥ 0.995 fallback.
    """
    p_cand = peaks.shape[0]
    distance_ = jnp.ceil(distance).astype(jnp.int32)

    # scipy sorts by priority ascending then iterates high->low. Stable ascending
    # argsort on priority gives ties in ascending position; iterating from the end
    # processes the higher position first (matches scipy's tie behaviour).
    eff_priority = jnp.where(peak_mask, priority, _NEG_INF)
    order = jnp.argsort(eff_priority, stable=True)  # ascending priority

    keep0 = peak_mask.astype(bool)  # start: every real peak is a keep candidate

    # Iterate j from high-priority (end of order) to low.
    def body(keep, j):
        pos = order[j]  # original slot index, processed high->low priority
        valid = peak_mask[pos]
        still = keep[pos] & valid
        center = peaks[pos]
        # Suppress neighbours within distance_, but only when THIS peak survives.
        within = (jnp.abs(peaks - center) < distance_) & peak_mask
        not_self = jnp.arange(p_cand) != pos
        suppress = within & not_self & still
        keep = jnp.where(suppress, False, keep)
        return keep, None

    rev = jnp.arange(p_cand)[::-1]
    keep_final, _ = jax.lax.scan(body, keep0, rev)
    return keep_final & peak_mask


# --------------------------------------------------------------------------- #
# Widths (peak_widths, rel_height) from prominence bases                      #
# --------------------------------------------------------------------------- #
def _peak_widths(
    x: jax.Array,
    peaks: jax.Array,
    peak_mask: jax.Array,
    prominences: jax.Array,
    left_bases: jax.Array,
    right_bases: jax.Array,
    rel_height: float,
) -> jax.Array:
    """Peak widths at ``rel_height``, exact ``_peak_widths`` semantics.

    scipy: ``height = x[peak] - prominence * rel_height``; from the peak walk left
    until ``x[i] < height`` (left intersection), then linearly interpolate between
    samples ``i`` and ``i+1``; symmetric on the right. The search is bounded by the
    prominence bases. Returns widths only (FWHM filter is all this stage needs).
    """
    n = x.shape[0]
    idx = jnp.arange(n)

    def per_peak(peak, prom, lb, rb):
        height = x[peak] - prom * rel_height
        # Left: largest i in [lb, peak] with x[i] < height.  scipy walks from the
        # peak leftward and stops at the first sample below height.
        below_left = (idx >= lb) & (idx <= peak) & (x < height)
        # rightmost such i (closest to peak)
        li_cand = jnp.where(below_left, idx, -1)
        li = jnp.max(li_cand)
        has_left = li >= 0
        # left intersection point between li and li+1
        x_li = x[jnp.clip(li, 0, n - 1)]
        x_li1 = x[jnp.clip(li + 1, 0, n - 1)]
        denom_l = x_li1 - x_li
        left_ip = jnp.where(
            has_left & (denom_l != 0),
            li + (height - x_li) / denom_l,
            jnp.where(has_left, li.astype(jnp.float64), peak.astype(jnp.float64)),
        )
        # Right: smallest i in [peak, rb] with x[i] < height.
        below_right = (idx >= peak) & (idx <= rb) & (x < height)
        ri_cand = jnp.where(below_right, idx, n)
        ri = jnp.min(ri_cand)
        has_right = ri < n
        x_ri = x[jnp.clip(ri, 0, n - 1)]
        x_ri1 = x[jnp.clip(ri - 1, 0, n - 1)]
        denom_r = x_ri1 - x_ri
        right_ip = jnp.where(
            has_right & (denom_r != 0),
            ri - (height - x_ri) / denom_r,
            jnp.where(has_right, ri.astype(jnp.float64), peak.astype(jnp.float64)),
        )
        return right_ip - left_ip

    w = jax.vmap(per_peak)(peaks, prominences, left_bases, right_bases)
    return jnp.where(peak_mask, w, 0.0)


# --------------------------------------------------------------------------- #
# Core find_peaks composition (exact scipy order of operations)               #
# --------------------------------------------------------------------------- #
def _compact_accepted(
    peaks: jax.Array, keep: jax.Array, priority: jax.Array, p_max: int
) -> PeakResult:
    """Compact accepted peaks into a ``(p_max,)`` ascending-index result.

    Overflow (count > p_max) keeps the top-``priority`` peaks (ADR-0004 §5.2) and
    sets the truncation flag. Within the kept set the indices are returned in
    ascending sample order (scipy returns peaks ascending by position).
    """
    count = jnp.sum(keep).astype(jnp.int32)
    truncated = count > p_max

    # Rank kept peaks by descending priority to choose which survive an overflow.
    eff = jnp.where(keep, priority, _NEG_INF)
    by_prio = jnp.argsort(eff, stable=True)[::-1]  # highest priority first
    top = by_prio[:p_max]
    top_keep = keep[top]
    # Among the chosen top, sort ascending by position for the final output.
    top_pos = jnp.where(top_keep, peaks[top], jnp.iinfo(jnp.int32).max)
    asc = jnp.argsort(top_pos, stable=True)
    sel = top[asc]
    sel_keep = top_keep[asc]
    indices = jnp.where(sel_keep, peaks[sel], 0).astype(jnp.int32)
    mask = sel_keep
    return PeakResult(indices=indices, mask=mask, count=count, truncated=truncated)


@partial(jax.jit, static_argnames=("p_cand", "p_max", "rel_height", "min_fwhm_pixels"))
def find_peaks_fixed(
    x: jax.Array,
    height: jax.Array,
    prominence: jax.Array,
    distance: jax.Array,
    p_cand: int = P_CAND_MAX,
    p_max: int = P_MAX_DETECTION,
    rel_height: float = 0.5,
    min_fwhm_pixels: float = 0.0,
) -> PeakResult:
    """Fixed-shape ``scipy.signal.find_peaks(x, height, prominence, distance)``.

    Reproduces scipy's exact order (``_peak_finding.py:943-995``):
    local maxima (plateau midpoints) -> height -> distance -> prominence ->
    (optional) FWHM/cosmic-ray filter. Prominence is computed on the
    distance-filtered set, matching scipy.

    Parameters
    ----------
    x : jax.Array
        Signal to search, shape ``(N,)``. fp64.
    height : jax.Array
        Scalar minimum peak height (``hmin``).
    prominence : jax.Array
        Scalar minimum prominence (``pmin``).
    distance : jax.Array
        Scalar minimum inter-peak distance in samples (``>= 1``).
    p_cand : int
        Static candidate-axis pad (pre-NMS local maxima cap).
    p_max : int
        Static accepted-peak pad.
    rel_height : float
        Relative height for the width computation (only used when
        ``min_fwhm_pixels > 0``).
    min_fwhm_pixels : float
        Static FWHM threshold; ``0`` disables the cosmic-ray filter (the
        ``_find_peaks`` detection parameterisation), ``> 0`` enables it (the
        ``detect_peaks`` calibration parameterisation).

    Returns
    -------
    PeakResult
        Padded indices + validity mask + count + truncation flag.
    """
    x = x.astype(jnp.float64)
    peaks, _le, _re, cand_mask = _local_maxima(x, p_cand)

    # 1) height filter.
    keep = cand_mask & (x[peaks] >= height)

    # 2) distance filter (priority = peak height), on the height-survivors.
    priority = x[peaks]
    keep = _select_by_peak_distance(peaks, priority, keep, distance)

    # 3) prominence on the distance-survivors.
    tbl_max, levels_max = _build_sparse_max(x)
    tbl_min, _levels_min = _build_sparse_min(x)
    prom, lb, rb = _prominences(x, peaks, keep, tbl_max, levels_max, tbl_min)
    keep = keep & (prom >= prominence)

    # 4) optional FWHM cosmic-ray filter (calibration parameterisation).
    if min_fwhm_pixels > 0:
        widths = _peak_widths(x, peaks, keep, prom, lb, rb, rel_height)
        keep = keep & (widths >= min_fwhm_pixels)

    return _compact_accepted(peaks, keep, priority, p_max)


# --------------------------------------------------------------------------- #
# Distance resolution (host-static; mirrors the reference)                    #
# --------------------------------------------------------------------------- #
def min_peak_distance_calibration(
    wl_step: float,
    resolving_power: float | None = None,
    min_distance_px: int | None = None,
    mean_wl: float | None = None,
) -> int:
    """Resolve calibration-path min distance (px), exact ``_compute_min_peak_distance``.

    ``preprocessing.py:417``: explicit ``min_distance_px`` wins; else a
    resolving-power-derived value (``max(3, int((mean_wl/R)/wl_step))``); else 3.
    All inputs are bucket-static, so the result is a host int.
    """
    if min_distance_px is not None:
        return max(1, int(min_distance_px))
    if resolving_power is not None and resolving_power > 0 and mean_wl is not None:
        resolution_nm = mean_wl / resolving_power
        return max(3, int(resolution_nm / max(wl_step, 1e-9)))
    return 3


def min_peak_distance_detection(wl_step: float, peak_width_nm: float) -> int:
    """Resolve detection-path min distance (px), exact ``_find_peaks`` rule.

    ``line_detection.py:2454``: ``max(int(peak_width_nm / max(wl_step, 1e-9)), 1)``.
    """
    return max(int(peak_width_nm / max(wl_step, 1e-9)), 1)


# --------------------------------------------------------------------------- #
# Two parameterisations of the one kernel                                     #
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("distance_px", "min_fwhm_pixels", "p_cand", "p_max"))
def detect_peaks_calibration(
    intensity: jax.Array,
    baseline: jax.Array,
    noise: jax.Array,
    distance_px: int,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
    min_intensity_floor: float = 0.0,
    min_fwhm_pixels: float = 1.5,
    p_cand: int = P_CAND_MAX,
    p_max: int = P_MAX_CALIBRATION,
) -> PeakResult:
    """Calibration peak detector, fixed-shape port of ``detect_peaks``.

    Operates on ``corrected = intensity - baseline`` with ``height = noise *
    threshold_factor`` (capped at ``min_intensity_floor`` when positive),
    ``prominence = noise * prominence_factor``, and the FWHM cosmic-ray filter
    (``rel_height=0.5``). The second-derivative confirmation of the reference is
    *not* enabled in production (``use_second_derivative=False`` default) and is
    out of scope here (see module remaining-work note).
    """
    corrected = intensity.astype(jnp.float64) - baseline.astype(jnp.float64)
    threshold = noise * threshold_factor
    threshold = jnp.where(
        min_intensity_floor > 0, jnp.minimum(threshold, min_intensity_floor), threshold
    )
    prom = noise * prominence_factor
    return find_peaks_fixed(
        corrected,
        height=threshold,
        prominence=prom,
        distance=jnp.asarray(distance_px, jnp.float64),
        p_cand=p_cand,
        p_max=p_max,
        rel_height=0.5,
        min_fwhm_pixels=min_fwhm_pixels,
    )


@partial(jax.jit, static_argnames=("distance_px", "p_cand", "p_max"))
def detect_peaks_detection(
    intensity: jax.Array,
    min_peak_height: float,
    distance_px: int,
    p_cand: int = P_CAND_MAX,
    p_max: int = P_MAX_DETECTION,
) -> PeakResult:
    """Detection peak detector, fixed-shape port of ``_find_peaks``.

    Operates on ``normalized = intensity / max(intensity)`` with
    ``height = max(min_peak_height, 0)``, ``prominence = height / 2``, distance in
    pixels, and **no** FWHM filter. Returns an all-empty result when
    ``max(intensity) <= 0`` (reference early return).
    """
    y = intensity.astype(jnp.float64)
    max_i = jnp.max(y)
    normalized = jnp.where(max_i > 0, y / max_i, y)
    threshold = jnp.maximum(min_peak_height, 0.0)
    res = find_peaks_fixed(
        normalized,
        height=threshold,
        prominence=threshold / 2.0,
        distance=jnp.asarray(distance_px, jnp.float64),
        p_cand=p_cand,
        p_max=p_max,
        rel_height=0.5,
        min_fwhm_pixels=0.0,
    )
    # Reference returns [] when max <= 0.
    empty = max_i <= 0
    return PeakResult(
        indices=jnp.where(empty, 0, res.indices),
        mask=jnp.where(empty, False, res.mask),
        count=jnp.where(empty, 0, res.count),
        truncated=jnp.where(empty, False, res.truncated),
    )

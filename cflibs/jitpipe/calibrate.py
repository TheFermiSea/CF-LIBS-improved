"""Stage 3 — fixed-shape, vmap-clean wavelength calibration kernel (ADR-0004 J2).

Jittable port of the **global single-axis calibrator**
:func:`cflibs.inversion.preprocess.wavelength_calibration.calibrate_wavelength_axis`
(the inner core re-used by the segmented driver). This module holds the
device-side kernel only: padded arrays + validity masks, no data-dependent
shapes, no SQLite, no host imports (import-hygiene test). The host wrapper
(:mod:`cflibs.jitpipe.host`) builds the reference line pool from the snapshot
and reconstitutes ``WavelengthCalibrationResult``; this kernel never touches
the DB.

Algorithm (ADR-0004 §4 row 3, J2 spec §3)
-----------------------------------------
1. **Banded pairs.** Per peak, ``searchsorted`` a ``±max_pair_window_nm``
   window into the sorted line pool, take up to ``K_pair`` nearest slots ->
   ``(P_max, K_pair)`` candidate (x=peak_wl, y=line_wl, weight, line_id) +
   validity mask. Flattened to ``C = P_max * K_pair`` candidate slots.
2. **Hypotheses (replace sequential RANSAC).**
   * *shift*: every live candidate slot IS a 1-point hypothesis
     (``shift = y - x``); exhaustive over all ``C`` slots.
   * *affine / quadratic*: ``H`` deterministic stratified samples drawn from
     distinct peak strata (counter-based ``jax.random``; distinct-x by
     construction), closed-form 2/3-point weighted fit.
3. **Per-hypothesis scoring.** Residuals vs all ``C`` slots (H-chunked via
   ``lax.map``); inlier mask at ``inlier_tolerance_nm``; score =
   ``(upper_bound_unique_inliers, -masked_median_residual)`` where the upper
   bound is ``min(#unique inlier peaks, #unique inlier lines)`` — NOT a
   per-hypothesis greedy dedupe.
4. **Winner + exact dedupe.** The argmax-score hypothesis goes through the
   exact greedy one-to-one dedupe (``_dedupe_one_to_one`` parity) as a
   ``lax.scan`` carrying ``(P_max,)``/``(L_max,)`` used-boolean masks.
5. **Refine.** 2 rounds of masked weighted normal equations (centered-and-
   scaled x; deliberate deviation vs the reference SVD lstsq, J2 spec §3).
6. **BIC + gates.** All 3 models computed unconditionally; validity mask =
   (enough points) & (monotonic on grid) & finite; best = masked argmin BIC;
   the 5 quality thresholds become a branchless gate vector; on failure the
   kernel returns the identity correction + an int reason code.

The returned :class:`CalibrationKernelResult` is a pytree (vmap/grad clean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

# --------------------------------------------------------------------------
# Static shape constants (ADR-0004 §4: P_max=2048, K_pair=48, H=4096).
# Kept module-level so the host wrapper / tests can pad to the same shapes.
# --------------------------------------------------------------------------

#: Default banded fan-out per peak (observed ≤32 at the ±2.0 nm window).
K_PAIR_DEFAULT = 48
#: Default stratified-hypothesis count for affine/quadratic.
H_AFFINE_DEFAULT = 4096
#: Default H-chunk block size (≤100 MB live residual matrix, J2 §5).
H_BLOCK_DEFAULT = 256
#: Number of refine rounds (J2 spec §3 "fixed 2 rounds").
N_REFINE = 2

#: Model ids — index into the per-model arrays (shift / affine / quadratic).
MODEL_SHIFT = 0
MODEL_AFFINE = 1
MODEL_QUADRATIC = 2
_MODEL_PARAM_COUNT = (1, 2, 3)
_MODEL_MIN_POINTS = (1, 2, 3)

#: Quality-gate reason codes (host maps to today's ``quality_reason`` strings).
REASON_PASSED = 0
REASON_INSUFFICIENT_INLIERS = 1
REASON_LOW_PEAK_MATCH_FRACTION = 2
REASON_RMSE_TOO_HIGH = 3
REASON_INSUFFICIENT_SPAN = 4
REASON_CORRECTION_TOO_LARGE = 5
REASON_NO_VALID_MODEL_FIT = 6

#: Map for the host wrapper (mirrors reference ``quality_reason`` strings).
REASON_STRINGS: dict[int, str] = {
    REASON_PASSED: "passed",
    REASON_INSUFFICIENT_INLIERS: "insufficient_inliers",
    REASON_LOW_PEAK_MATCH_FRACTION: "low_peak_match_fraction",
    REASON_RMSE_TOO_HIGH: "rmse_too_high",
    REASON_INSUFFICIENT_SPAN: "insufficient_span",
    REASON_CORRECTION_TOO_LARGE: "correction_too_large",
    REASON_NO_VALID_MODEL_FIT: "no_valid_model_fit",
}

#: Model-class labels for the host wrapper / parity contract (c).
MODEL_STRINGS: dict[int, str] = {
    MODEL_SHIFT: "shift",
    MODEL_AFFINE: "affine",
    MODEL_QUADRATIC: "quadratic",
}


@dataclass
class CalibrationKernelResult:
    """Device-side result of the global wavelength calibrator (pytree).

    All array fields are JAX arrays so the struct is a valid ``vmap``/``grad``
    leaf bundle. The host wrapper reconstitutes a
    :class:`~cflibs.inversion.preprocess.wavelength_calibration.WavelengthCalibrationResult`.

    Attributes
    ----------
    corrected_wavelength : array, shape (W_max,)
        Corrected axis when ``quality_passed`` else the identity axis.
    coefficients : array, shape (3,)
        Best-model coefficients in the reference ``_eval_model`` convention
        (shift: ``(b,0,0)``; affine: ``(a,b,0)``; quadratic: ``(c2,c1,c0)``),
        padded to length 3.
    model_id : int scalar
        Selected model class (``MODEL_SHIFT`` / ``MODEL_AFFINE`` /
        ``MODEL_QUADRATIC``).
    bic, rmse_nm : float scalar
        Best-model BIC / inlier RMSE.
    n_inliers : int scalar
        Robust (deduped) inlier count of the best model.
    matched_peak_fraction : float scalar
        Unique inlier peaks / total live peaks.
    quality_passed : bool scalar
        Whether the branchless gate vector passed.
    reason_code : int scalar
        Quality-gate reason code (see ``REASON_*``).
    success : bool scalar
        Whether any monotonic model fit was found.
    robust_mask : array, shape (C,)
        Boolean robust (deduped) inlier mask of the best model over the
        ``C = P_max * K_pair`` candidate slots. Exposed so the segmented driver
        can recover the inlier-anchor wavelengths (``x[robust_mask]``) for the
        ye6t coverage gate without a second pass.
    """

    corrected_wavelength: Any
    coefficients: Any
    model_id: Any
    bic: Any
    rmse_nm: Any
    n_inliers: Any
    matched_peak_fraction: Any
    quality_passed: Any
    reason_code: Any
    success: Any
    robust_mask: Any = None


def _ckr_flatten(r: CalibrationKernelResult):
    children = (
        r.corrected_wavelength,
        r.coefficients,
        r.model_id,
        r.bic,
        r.rmse_nm,
        r.n_inliers,
        r.matched_peak_fraction,
        r.quality_passed,
        r.reason_code,
        r.success,
        r.robust_mask,
    )
    return children, None


def _ckr_unflatten(_aux: Any, children: tuple) -> CalibrationKernelResult:
    return CalibrationKernelResult(*children)


jax.tree_util.register_pytree_node(CalibrationKernelResult, _ckr_flatten, _ckr_unflatten)


# --------------------------------------------------------------------------
# Banded candidate pairs (replaces the nested Python loop, ref :441-476).
# --------------------------------------------------------------------------


def build_banded_pairs(
    peak_wl: jnp.ndarray,
    peak_amp: jnp.ndarray,
    peak_mask: jnp.ndarray,
    line_wl: jnp.ndarray,
    line_strength: jnp.ndarray,
    line_mask: jnp.ndarray,
    max_pair_window_nm: float,
    k_pair: int,
) -> tuple[jnp.ndarray, ...]:
    """Banded (P_max, K_pair) candidate pairs via ``searchsorted``.

    Mirrors the reference :func:`_build_candidate_pairs` (window
    ``|line - peak| <= max_pair_window_nm``, weight
    ``sqrt(peak_amp_norm * line_strength_norm)``) but with a fixed per-peak
    fan-out ``K_pair`` (J2 spec §3). For each live peak the ``K_pair`` lines
    nearest in wavelength are kept; invalid / out-of-window / padding slots are
    masked out.

    ``line_wl`` MUST be sorted ascending (host responsibility, matching the
    snapshot line pool). Returns ``(x, y, peak_id, line_id, weight, mask)``
    each shape ``(P_max * K_pair,)``.
    """
    p_max = peak_wl.shape[0]
    # Normalisation by the masked max (ref divides by np.max over the live set).
    amp_max = jnp.maximum(jnp.max(jnp.where(peak_mask, peak_amp, -jnp.inf)), 1e-12)
    str_max = jnp.maximum(jnp.max(jnp.where(line_mask, line_strength, -jnp.inf)), 1e-12)
    peak_amp_n = peak_amp / amp_max
    line_strength_n = line_strength / str_max

    # For each peak, find the insertion point and gather a symmetric K_pair
    # window of nearest line indices (by |line_wl - peak|). We over-gather a
    # 2*K window around the insertion point then take the K nearest.
    half = k_pair  # candidate radius before selecting K nearest
    ins = jnp.searchsorted(line_wl, peak_wl)  # (P_max,)
    # Offsets [-half, ..., half-1] -> 2*half candidate columns.
    offs = jnp.arange(-half, half)  # (2K,)
    cand_idx = ins[:, None] + offs[None, :]  # (P_max, 2K)
    cand_idx = jnp.clip(cand_idx, 0, line_wl.shape[0] - 1)
    cand_in_range = (ins[:, None] + offs[None, :] >= 0) & (
        ins[:, None] + offs[None, :] < line_wl.shape[0]
    )
    cand_line_wl = line_wl[cand_idx]  # (P_max, 2K)
    cand_line_mask = line_mask[cand_idx] & cand_in_range
    delta = jnp.abs(cand_line_wl - peak_wl[:, None])
    within = delta <= max_pair_window_nm
    valid = within & cand_line_mask & peak_mask[:, None]
    # Select the K nearest valid candidates per peak: sort by (invalid_first,
    # delta) and take the first K columns.
    sort_key = jnp.where(valid, delta, jnp.inf)
    order = jnp.argsort(sort_key, axis=1)  # (P_max, 2K)
    take = order[:, :k_pair]  # (P_max, K)
    sel_line_local = jnp.take_along_axis(cand_idx, take, axis=1)  # (P_max, K)
    sel_valid = jnp.take_along_axis(valid, take, axis=1)  # (P_max, K)

    y = line_wl[sel_line_local]  # (P_max, K)
    x = jnp.broadcast_to(peak_wl[:, None], (p_max, k_pair))
    line_id = sel_line_local
    peak_id = jnp.broadcast_to(jnp.arange(p_max)[:, None], (p_max, k_pair))
    pa = jnp.broadcast_to(peak_amp_n[:, None], (p_max, k_pair))
    ls = line_strength_n[sel_line_local]
    weight = jnp.sqrt(jnp.maximum(pa, 1e-6) * jnp.maximum(ls, 1e-6))

    flat = (-1,)
    return (
        x.reshape(flat),
        y.reshape(flat),
        peak_id.reshape(flat),
        line_id.reshape(flat),
        jnp.where(sel_valid, weight, 0.0).reshape(flat),
        sel_valid.reshape(flat),
    )


# --------------------------------------------------------------------------
# Closed-form weighted fits (centered-and-scaled x; deliberate deviation).
# --------------------------------------------------------------------------


def _center_scale(x: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(x0, half_span)`` for centering x to the masked midpoint/half-span.

    J2 spec §3 required numerical deviation: raw-nm normal equations with
    ``x^2 ~ 2.5e5`` lose ~10 digits, so we form the Gram matrices on the
    centered/scaled coordinate ``u = (x - x0) / half_span`` and map the
    coefficients back to the reference (raw-nm) convention host/eval-side.
    """
    xm = jnp.where(mask, x, jnp.nan)
    lo = jnp.nanmin(xm)
    hi = jnp.nanmax(xm)
    lo = jnp.where(jnp.isfinite(lo), lo, 0.0)
    hi = jnp.where(jnp.isfinite(hi), hi, 0.0)
    x0 = 0.5 * (lo + hi)
    half = jnp.maximum(0.5 * (hi - lo), 1e-9)
    return x0, half


def _fit_weighted(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    mask: jnp.ndarray,
    model_id: int,
) -> jnp.ndarray:
    """Closed-form masked weighted fit. Returns coeffs in raw-nm ``_eval_model`` order.

    * shift -> ``(b, 0, 0)`` with ``b = sum(w*(y-x)) / sum(w)``
    * affine -> ``(a, b, 0)`` via centered normal equations, mapped to raw nm
    * quadratic -> ``(c2, c1, c0)`` via centered normal equations, mapped to raw nm
    """
    wm = jnp.where(mask, jnp.clip(w, 1e-8, None), 0.0)
    sw = jnp.maximum(jnp.sum(wm), 1e-30)

    if model_id == MODEL_SHIFT:
        b = jnp.sum(wm * (y - x)) / sw
        return jnp.array([b, 0.0, 0.0], dtype=x.dtype)

    x0, half = _center_scale(x, mask)
    u = (x - x0) / half
    deg = 1 if model_id == MODEL_AFFINE else 2
    # Vandermonde in u, columns [1, u, u^2][: deg+1].
    cols = [jnp.ones_like(u), u, u * u][: deg + 1]
    vand = jnp.stack(cols, axis=1)  # (C, deg+1)
    wv = wm[:, None] * vand
    gram = vand.T @ wv  # (deg+1, deg+1)
    rhs = vand.T @ (wm * y)  # (deg+1,)
    # Ridge floor for conditioning when the fit is rank-deficient/empty.
    gram = gram + jnp.eye(deg + 1, dtype=x.dtype) * 1e-12
    coef_u = jnp.linalg.solve(gram, rhs)  # ascending powers of u

    # Map y = sum_j coef_u[j] * ((x - x0)/half)^j back to powers of x.
    if model_id == MODEL_AFFINE:
        c0u, c1u = coef_u[0], coef_u[1]
        a = c1u / half
        b = c0u - c1u * x0 / half
        return jnp.array([a, b, 0.0], dtype=x.dtype)
    # quadratic: c0u + c1u*u + c2u*u^2, u=(x-x0)/h
    c0u, c1u, c2u = coef_u[0], coef_u[1], coef_u[2]
    h = half
    c2 = c2u / (h * h)
    c1 = c1u / h - 2.0 * c2u * x0 / (h * h)
    c0 = c0u - c1u * x0 / h + c2u * x0 * x0 / (h * h)
    return jnp.array([c2, c1, c0], dtype=x.dtype)


def _eval_model(x: jnp.ndarray, coef: jnp.ndarray, model_id: int) -> jnp.ndarray:
    """Evaluate the calibration model (reference ``_eval_model`` convention)."""
    if model_id == MODEL_SHIFT:
        return x + coef[0]
    if model_id == MODEL_AFFINE:
        return coef[0] * x + coef[1]
    return coef[0] * x * x + coef[1] * x + coef[2]


# --------------------------------------------------------------------------
# Exact greedy one-to-one dedupe (parity port of ref :209-230 as lax.scan).
# --------------------------------------------------------------------------


def _dedupe_mask(
    residual: jnp.ndarray,
    peak_id: jnp.ndarray,
    line_id: jnp.ndarray,
    inlier_mask: jnp.ndarray,
    p_max: int,
    l_max: int,
) -> jnp.ndarray:
    """Greedy one-to-one dedupe -> boolean selected mask over candidate slots.

    Exact port of :func:`_dedupe_one_to_one`: sort live inliers by residual
    ascending, greedily accept a slot iff its peak and line are both still
    unused. Implemented as a ``lax.scan`` carrying ``(P_max,)``/``(L_max,)``
    used-boolean masks (J2 spec §3). Ties in residual are broken by original
    slot order (``argsort`` is stable on the ``(residual, index)`` key, matching
    NumPy's stable ``np.argsort``).
    """
    c = residual.shape[0]
    # Stable sort by residual; masked-out slots pushed to the end with +inf.
    key = jnp.where(inlier_mask, residual, jnp.inf)
    order = jnp.argsort(key, stable=True)  # ascending; ties keep slot order

    def step(carry, slot):
        used_p, used_l, sel = carry
        is_in = inlier_mask[slot]
        p = peak_id[slot]
        ln = line_id[slot]
        free = is_in & (~used_p[p]) & (~used_l[ln])
        used_p = used_p.at[p].set(used_p[p] | free)
        used_l = used_l.at[ln].set(used_l[ln] | free)
        sel = sel.at[slot].set(free)
        return (used_p, used_l, sel), None

    init = (
        jnp.zeros(p_max, dtype=bool),
        jnp.zeros(l_max, dtype=bool),
        jnp.zeros(c, dtype=bool),
    )
    (_, _, sel), _ = lax.scan(step, init, order)
    return sel


# --------------------------------------------------------------------------
# Per-model fit + BIC (refine rounds; parity of ref _ransac_fit tail).
# --------------------------------------------------------------------------


def _compute_bic(rss: jnp.ndarray, n: jnp.ndarray, k: int) -> jnp.ndarray:
    """BIC parity (ref :233): ``inf`` if ``n<=k or n<=0`` else n*log(rss/n)+k*log(n)."""
    rss = jnp.maximum(rss, 1e-12)
    nf = n.astype(jnp.float64)
    bic = nf * jnp.log(rss / nf) + k * jnp.log(nf)
    return jnp.where((n <= k) | (n <= 0), jnp.inf, bic)


def _refine_and_bic(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    peak_id: jnp.ndarray,
    line_id: jnp.ndarray,
    cand_mask: jnp.ndarray,
    seed_mask: jnp.ndarray,
    model_id: int,
    inlier_tol: float,
    p_max: int,
    l_max: int,
    n_peaks_total: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Refit on ``seed_mask``, re-dedupe, then final fit + RMSE + BIC + frac.

    Mirrors :func:`_refine_robust_inliers` + :func:`_build_model_fit`: refit on
    the seed inlier set, recompute residuals/dedupe over ALL slots to get the
    robust inlier set, then final refit on that set with RMSE/BIC.
    """
    # Round 1: refit on the seed (winner-deduped) set.
    coef = _fit_weighted(x, y, w, seed_mask, model_id)
    robust = seed_mask
    for _ in range(N_REFINE):
        resid = jnp.abs(_eval_model(x, coef, model_id) - y)
        inl = (resid <= inlier_tol) & cand_mask
        robust = _dedupe_mask(resid, peak_id, line_id, inl, p_max, l_max)
        coef = _fit_weighted(x, y, w, robust, model_id)

    # Final stats on the robust set.
    pred = _eval_model(x, coef, model_id)
    res = jnp.where(robust, pred - y, 0.0)
    n_in = jnp.sum(robust)
    rss = jnp.sum(res * res)
    rmse = jnp.sqrt(rss / jnp.maximum(n_in, 1))
    k = _MODEL_PARAM_COUNT[model_id]
    bic = _compute_bic(rss, n_in, k)
    # Unique inlier peaks / total live peaks.
    uniq_peaks = jnp.zeros(p_max, dtype=bool).at[peak_id].max(robust)
    matched_frac = jnp.sum(uniq_peaks) / jnp.maximum(n_peaks_total, 1)
    return coef, bic, rmse, n_in, matched_frac, robust


# --------------------------------------------------------------------------
# Upper-bound hypothesis scoring (parity-anchored parallel RANSAC, J2 §3).
# --------------------------------------------------------------------------


def _unique_count_upper_bound(
    inlier_mask: jnp.ndarray,
    peak_id: jnp.ndarray,
    line_id: jnp.ndarray,
    p_max: int,
    l_max: int,
) -> jnp.ndarray:
    """``min(#unique inlier peaks, #unique inlier lines)`` — the dedupe upper bound."""
    up = jnp.zeros(p_max, dtype=bool).at[peak_id].max(inlier_mask)
    ul = jnp.zeros(l_max, dtype=bool).at[line_id].max(inlier_mask)
    return jnp.minimum(jnp.sum(up), jnp.sum(ul))


def _exact_dedupe_count(
    residual: jnp.ndarray,
    peak_id: jnp.ndarray,
    line_id: jnp.ndarray,
    inlier_mask: jnp.ndarray,
    p_max: int,
    l_max: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact greedy one-to-one dedupe ``(n_inliers, masked_median_residual)``.

    The reference ranks each RANSAC hypothesis by its **greedy-deduped** inlier
    count (``_dedupe_one_to_one`` -> ``(n_in, median)``, ref :285-290), NOT the
    parallel upper bound. The upper bound ``min(#unique peaks, #unique lines)``
    can overcount a hypothesis whose peak↔line assignment is many-to-one, which
    lets a marginal affine win the per-model search and then edge out shift on
    the post-refine BIC — the J2 §7 R8 model-class flip. Scoring with the exact
    dedupe restores the reference ranking; it is one ``lax.scan`` per hypothesis
    (same cost as the winner dedupe), used only for the slope models on the
    segmented path (the shift model is 1-point exhaustive and never flips).
    """
    sel = _dedupe_mask(residual, peak_id, line_id, inlier_mask, p_max, l_max)
    n_in = jnp.sum(sel)
    med = _masked_median(residual, sel)
    return n_in, med


def _score_hypothesis(
    coef: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    peak_id: jnp.ndarray,
    line_id: jnp.ndarray,
    cand_mask: jnp.ndarray,
    model_id: int,
    inlier_tol: float,
    p_max: int,
    l_max: int,
    exact_dedupe: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Score one hypothesis: ``(count, -masked_median_residual)``.

    ``count`` is the parallel upper bound ``min(#unique peaks, #unique lines)``
    by default (J2 §3 fast path), or the exact greedy-dedupe inlier count when
    ``exact_dedupe`` is set (J2 §7 R8 model-flip fix — matches the reference
    RANSAC score ``(n_in, median)`` exactly).
    """
    resid = jnp.abs(_eval_model(x, coef, model_id) - y)
    inl = (resid <= inlier_tol) & cand_mask
    if exact_dedupe:
        return _exact_dedupe_count(resid, peak_id, line_id, inl, p_max, l_max)
    ub = _unique_count_upper_bound(inl, peak_id, line_id, p_max, l_max)
    # Masked median of inlier residuals (tie-break term in the score).
    med = _masked_median(resid, inl)
    return ub, med


def _masked_median(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Median of ``values`` over ``True`` entries of ``mask`` (fixed-shape)."""
    n = jnp.sum(mask)
    big = jnp.where(mask, values, jnp.inf)
    s = jnp.sort(big)  # inliers first (ascending), padding +inf at the end
    # median index handling for even/odd counts.
    half = n // 2
    lo = s[jnp.clip(half - 1, 0, s.shape[0] - 1)]
    hi = s[jnp.clip(half, 0, s.shape[0] - 1)]
    even = (n % 2) == 0
    med = jnp.where(even, 0.5 * (lo + hi), hi)
    return jnp.where(n > 0, med, jnp.inf)


# --------------------------------------------------------------------------
# Per-model winner search (shift exhaustive; affine/quad stratified).
# --------------------------------------------------------------------------


def _winner_seed_mask(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    peak_id: jnp.ndarray,
    line_id: jnp.ndarray,
    cand_mask: jnp.ndarray,
    model_id: int,
    inlier_tol: float,
    p_max: int,
    l_max: int,
    h_affine: int,
    h_block: int,
    key: jnp.ndarray,
    exact_dedupe: bool = False,
) -> jnp.ndarray:
    """Find the best hypothesis and return its (pre-refine) inlier seed mask.

    * shift: every live candidate slot IS a 1-point hypothesis
      ``coef=(y-x,0,0)``, scored exhaustively (J2 spec §3).
    * affine/quadratic: ``h_affine`` stratified counter-PRNG hypotheses; each
      draws ``min_pts`` distinct peak strata and closed-form fits them.
    Scoring is the upper-bound ``(unique-inlier-count, -masked-median)`` by
    default, or the exact greedy-dedupe ``(n_in, -median)`` when ``exact_dedupe``
    is set (J2 §7 R8: matches the reference RANSAC ranking on the slope models so
    the per-model winner — and therefore the model class — never flips);
    H-chunked via ``lax.map`` over blocks of ``h_block``.
    """
    c = x.shape[0]

    if model_id == MODEL_SHIFT:
        # Exhaustive: hypothesis i has shift b_i = y_i - x_i. The shift score is
        # always the cheap upper bound: a 1-point shift hypothesis has one (x,y)
        # so the dedupe is a no-op and the upper bound equals the exact count.
        shifts = y - x  # (C,)

        def score_one(b):
            coef = jnp.array([b, 0.0, 0.0], dtype=x.dtype)
            ub, med = _score_hypothesis(
                coef, x, y, peak_id, line_id, cand_mask, model_id, inlier_tol, p_max, l_max
            )
            # Invalidate dead slots so their (b undefined) never win.
            return ub, med

        ubs, meds = _chunked_score(score_one, shifts, h_block)
        # Dead candidate slots cannot be a valid 1-pt hypothesis.
        ubs = jnp.where(cand_mask, ubs, -1)
        best = _argmax_score(ubs, meds)
        best_b = shifts[best]
        best_coef = jnp.array([best_b, 0.0, 0.0], dtype=x.dtype)
    else:
        min_pts = _MODEL_MIN_POINTS[model_id]
        # Stratified sampling: sample min_pts distinct *peaks*, then one live
        # candidate slot per sampled peak. Distinct-x by construction.
        samples = _stratified_samples(
            x, peak_id, cand_mask, min_pts, h_affine, p_max, key
        )  # (h_affine, min_pts) slot indices, or -1 when a draw is degenerate.

        def fit_score(sample):
            valid = jnp.all(sample >= 0)
            s = jnp.clip(sample, 0, c - 1)
            sub_mask = jnp.zeros(c, dtype=bool).at[s].set(True) & valid
            coef = _fit_weighted(x, y, w, sub_mask, model_id)
            ub, med = _score_hypothesis(
                coef,
                x,
                y,
                peak_id,
                line_id,
                cand_mask,
                model_id,
                inlier_tol,
                p_max,
                l_max,
                exact_dedupe=exact_dedupe,
            )
            return jnp.where(valid, ub, -1), med, coef

        ubs, meds, coefs = _chunked_fit_score(fit_score, samples, h_block)
        best = _argmax_score(ubs, meds)
        best_coef = coefs[best]

    # Seed inlier mask of the winning hypothesis (input to refine).
    resid = jnp.abs(_eval_model(x, best_coef, model_id) - y)
    inl = (resid <= inlier_tol) & cand_mask
    return _dedupe_mask(resid, peak_id, line_id, inl, p_max, l_max)


def _argmax_score(ubs: jnp.ndarray, meds: jnp.ndarray) -> jnp.ndarray:
    """Argmax over ``(ub desc, med asc)`` — lexicographic like the reference score."""
    # Encode as a single key: prefer larger ub, then smaller med.
    # med is non-negative (residual); use a large multiplier so ub dominates.
    key = ubs.astype(jnp.float64) - meds / (1.0 + jnp.max(jnp.where(jnp.isfinite(meds), meds, 0.0)))
    return jnp.argmax(key)


def _chunked_score(fn, params: jnp.ndarray, h_block: int):
    """``lax.map`` ``fn`` over H in blocks (memory bound). Returns (ubs, meds).

    ``batch_size=h_block`` bounds the live (H×C) residual matrix (J2 §5: the
    H-block keeps live residuals ≤100 MB rather than materialising all H at
    once). ``batch_size`` must divide the leading axis, so it is clipped to it.
    """
    bs = max(1, min(int(h_block), int(params.shape[0])))
    ubs, meds = jax.lax.map(fn, params, batch_size=bs)
    return ubs, meds


def _chunked_fit_score(fn, samples: jnp.ndarray, h_block: int):
    bs = max(1, min(int(h_block), int(samples.shape[0])))
    ubs, meds, coefs = jax.lax.map(fn, samples, batch_size=bs)
    return ubs, meds, coefs


def _stratified_samples(
    x: jnp.ndarray,
    peak_id: jnp.ndarray,
    cand_mask: jnp.ndarray,
    min_pts: int,
    h_affine: int,
    p_max: int,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Draw ``h_affine`` samples of ``min_pts`` distinct-peak candidate slots.

    For each hypothesis: pick ``min_pts`` distinct peaks uniformly from the live
    peaks, then one live candidate slot per picked peak. Returns slot indices
    ``(h_affine, min_pts)`` (or ``-1`` markers when the live set is too small).
    Counter-based ``jax.random`` -> deterministic per ``(key, shapes)``.
    """
    c = x.shape[0]
    # Live peaks: a peak is live iff it has >=1 live candidate slot.
    peak_has = jnp.zeros(p_max, dtype=bool).at[peak_id].max(cand_mask)
    n_live_peaks = jnp.sum(peak_has)
    # Per-peak: a representative live slot (lowest index) for that peak.
    slot_idx = jnp.arange(c)
    rep_slot = jnp.full(p_max, c, dtype=jnp.int32)
    rep_slot = rep_slot.at[peak_id].min(jnp.where(cand_mask, slot_idx, c).astype(jnp.int32))
    # Compact list of live peak ids (sorted; padding = p_max).
    live_peaks = jnp.where(peak_has, jnp.arange(p_max), p_max)
    live_peaks = jnp.sort(live_peaks)  # live ids first, padding (p_max) last

    keys = jax.random.split(key, h_affine)

    def draw(k):
        # Pick min_pts distinct positions in [0, n_live_peaks).
        perm = jax.random.permutation(k, p_max)  # positions into live_peaks
        # Take the first min_pts positions that index live peaks.
        chosen_pos = perm[:min_pts]
        chosen_peak = live_peaks[chosen_pos]
        valid = chosen_pos < n_live_peaks
        chosen_peak = jnp.where(valid, chosen_peak, p_max)
        slots = jnp.where((chosen_peak < p_max), rep_slot[jnp.clip(chosen_peak, 0, p_max - 1)], c)
        # Mark degenerate (dead) slots as -1.
        ok = (slots < c) & (n_live_peaks >= min_pts)
        return jnp.where(ok, slots, -1)

    return jax.lax.map(draw, keys)


# --------------------------------------------------------------------------
# Quality gate (branchless vector; parity of ref :96-133).
# --------------------------------------------------------------------------


def _quality_gate(
    corrected: jnp.ndarray,
    robust_mask: jnp.ndarray,
    n_inliers: jnp.ndarray,
    matched_frac: jnp.ndarray,
    rmse: jnp.ndarray,
    peak_wl: jnp.ndarray,
    peak_id: jnp.ndarray,
    wavelength: jnp.ndarray,
    wl_mask: jnp.ndarray,
    min_inliers: float,
    min_peak_match_fraction: float,
    max_rmse_nm: float,
    min_inlier_span_fraction: float,
    max_abs_correction_nm: float,
    p_max: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Branchless 5-gate vector -> ``(passed, reason_code)`` (ref order preserved)."""
    # Inlier peak span: ptp over unique inlier peak wavelengths.
    uniq_peaks = jnp.zeros(p_max, dtype=bool).at[peak_id].max(robust_mask)
    n_uniq = jnp.sum(uniq_peaks)
    pk = jnp.where(uniq_peaks, peak_wl, jnp.nan)
    pmin = jnp.nanmin(pk)
    pmax_ = jnp.nanmax(pk)
    peak_span = jnp.where(n_uniq >= 2, pmax_ - pmin, 0.0)
    full_span = jnp.maximum(_masked_ptp(wavelength, wl_mask), 1e-9)
    span_frac = peak_span / full_span
    # Max |correction| over the live axis.
    corr = jnp.where(wl_mask, corrected - wavelength, 0.0)
    max_abs_corr = jnp.max(jnp.abs(corr))

    g1 = n_inliers >= min_inliers
    g2 = matched_frac >= min_peak_match_fraction
    g3 = rmse <= max_rmse_nm
    g4 = span_frac >= min_inlier_span_fraction
    g5 = max_abs_corr <= max_abs_correction_nm
    passed = g1 & g2 & g3 & g4 & g5
    # First failing gate sets the reason (ref returns on first failure).
    reason = jnp.where(
        ~g1,
        REASON_INSUFFICIENT_INLIERS,
        jnp.where(
            ~g2,
            REASON_LOW_PEAK_MATCH_FRACTION,
            jnp.where(
                ~g3,
                REASON_RMSE_TOO_HIGH,
                jnp.where(~g4, REASON_INSUFFICIENT_SPAN, REASON_CORRECTION_TOO_LARGE),
            ),
        ),
    )
    reason = jnp.where(passed, REASON_PASSED, reason)
    return passed, reason


def _masked_ptp(v: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    vm = jnp.where(mask, v, jnp.nan)
    return jnp.nanmax(vm) - jnp.nanmin(vm)


def _is_monotonic(coef: jnp.ndarray, model_id: int, wavelength: jnp.ndarray, wl_mask: jnp.ndarray):
    """Ref ``_is_monotonic_on_grid``: all diffs of corrected live axis > 0."""
    corrected = _eval_model(wavelength, coef, model_id)
    # Only consider adjacent live samples; padding diffs are ignored.
    d = jnp.diff(corrected)
    valid_pair = wl_mask[1:] & wl_mask[:-1]
    return jnp.all(jnp.where(valid_pair, d > 0, True))


# --------------------------------------------------------------------------
# Public kernel.
# --------------------------------------------------------------------------


def calibrate_axis_kernel(
    peak_wl: jnp.ndarray,
    peak_amp: jnp.ndarray,
    peak_mask: jnp.ndarray,
    line_wl: jnp.ndarray,
    line_strength: jnp.ndarray,
    line_mask: jnp.ndarray,
    wavelength: jnp.ndarray,
    wl_mask: jnp.ndarray,
    *,
    inlier_tolerance_nm: float = 0.08,
    max_pair_window_nm: float = 2.0,
    quality_min_inliers: float = 12.0,
    quality_min_peak_match_fraction: float = 0.0,
    quality_max_rmse_nm: float = 0.10,
    quality_min_inlier_span_fraction: float = 0.25,
    quality_max_abs_correction_nm: float = 2.5,
    apply_quality_gate: bool = True,
    candidate_models: tuple[int, ...] = (MODEL_SHIFT, MODEL_AFFINE, MODEL_QUADRATIC),
    k_pair: int = K_PAIR_DEFAULT,
    h_affine: int = H_AFFINE_DEFAULT,
    h_block: int = H_BLOCK_DEFAULT,
    exact_dedupe_score: bool = True,
    seed: int = 42,
) -> CalibrationKernelResult:
    """Fixed-shape global wavelength calibrator (parity of ``calibrate_wavelength_axis``).

    All array inputs are padded; ``*_mask`` arrays mark valid entries.
    ``line_wl`` MUST be sorted ascending. ``candidate_models`` / ``k_pair`` /
    ``h_affine`` / ``h_block`` / ``exact_dedupe_score`` are Python-static (they
    shape the graph).

    ``exact_dedupe_score`` (default True) ranks slope-model hypotheses by the
    exact greedy one-to-one dedupe count — the reference RANSAC score — so the
    per-model winner (and hence the BIC-selected model class) matches the
    reference on the ye6t Al-doublet confounder (J2 §7 R8). Set False for the
    cheaper parallel upper-bound score where the model-flip hazard is absent.
    """
    p_max = peak_wl.shape[0]
    l_max = line_wl.shape[0]
    n_peaks_total = jnp.sum(peak_mask)

    x, y, peak_id, line_id, weight, cand_mask = build_banded_pairs(
        peak_wl,
        peak_amp,
        peak_mask,
        line_wl,
        line_strength,
        line_mask,
        max_pair_window_nm,
        k_pair,
    )
    peak_id = peak_id.astype(jnp.int32)
    line_id = line_id.astype(jnp.int32)

    base_key = jax.random.PRNGKey(seed)

    # Fit every candidate model unconditionally (model lattice, J2 spec §3).
    coefs, bics, rmses, n_ins, fracs, robusts, monos = [], [], [], [], [], [], []
    for m in candidate_models:
        # Per-model seed offset mirrors the reference rng seeding by k.
        mk = jax.random.fold_in(base_key, _MODEL_PARAM_COUNT[m])
        seed_mask = _winner_seed_mask(
            x,
            y,
            weight,
            peak_id,
            line_id,
            cand_mask,
            m,
            inlier_tolerance_nm,
            p_max,
            l_max,
            h_affine,
            h_block,
            mk,
            exact_dedupe=exact_dedupe_score,
        )
        coef, bic, rmse, n_in, frac, robust = _refine_and_bic(
            x,
            y,
            weight,
            peak_id,
            line_id,
            cand_mask,
            seed_mask,
            m,
            inlier_tolerance_nm,
            p_max,
            l_max,
            n_peaks_total,
        )
        mono = _is_monotonic(coef, m, wavelength, wl_mask)
        # A model fit is valid iff it has >= min_pts robust inliers and is
        # monotonic and finite BIC (parity: non-monotonic/None fits dropped).
        valid = (n_in >= _MODEL_MIN_POINTS[m]) & mono & jnp.isfinite(bic)
        coefs.append(coef)
        bics.append(jnp.where(valid, bic, jnp.inf))
        rmses.append(rmse)
        n_ins.append(n_in)
        fracs.append(frac)
        robusts.append(robust)
        monos.append(valid)

    coefs = jnp.stack(coefs)  # (M, 3)
    bics = jnp.stack(bics)  # (M,)
    rmses = jnp.stack(rmses)
    n_ins = jnp.stack(n_ins)
    fracs = jnp.stack(fracs)
    robusts = jnp.stack(robusts)  # (M, C)
    monos = jnp.stack(monos)

    any_valid = jnp.any(monos)
    # Best = masked argmin BIC; ties broken by model order (stable: argmin
    # returns the first min, and candidate_models is ordered shift<affine<quad).
    best = jnp.argmin(bics)
    best_model_id = jnp.array(candidate_models, dtype=jnp.int32)[best]
    best_coef = coefs[best]
    best_bic = bics[best]
    best_rmse = rmses[best]
    best_n_in = n_ins[best]
    best_frac = fracs[best]
    best_robust = robusts[best]

    # Corrected axis for the selected model (lax.switch over the static model
    # list so the right `_eval_model` branch runs for the traced `best`).
    corrected_branches = [
        (lambda c=coefs[i], mm=candidate_models[i]: _eval_model(wavelength, c, mm))
        for i in range(len(candidate_models))
    ]
    corrected_full = lax.switch(best, corrected_branches)

    # Quality gate (branchless, full 5-vector with the model-correct correction).
    gate_passed, reason = _quality_gate(
        corrected_full,
        best_robust,
        best_n_in,
        best_frac,
        best_rmse,
        peak_wl,
        peak_id,
        wavelength,
        wl_mask,
        quality_min_inliers,
        quality_min_peak_match_fraction,
        quality_max_rmse_nm,
        quality_min_inlier_span_fraction,
        quality_max_abs_correction_nm,
        p_max,
    )

    quality_passed = jnp.where(jnp.array(apply_quality_gate), gate_passed & any_valid, any_valid)
    reason = jnp.where(
        any_valid,
        jnp.where(jnp.array(apply_quality_gate), reason, REASON_PASSED),
        REASON_NO_VALID_MODEL_FIT,
    )
    corrected = jnp.where(quality_passed, corrected_full, wavelength)

    return CalibrationKernelResult(
        corrected_wavelength=corrected,
        coefficients=best_coef,
        model_id=jnp.where(any_valid, best_model_id, -1),
        bic=best_bic,
        rmse_nm=best_rmse,
        n_inliers=best_n_in,
        matched_peak_fraction=best_frac,
        quality_passed=quality_passed,
        reason_code=reason,
        success=any_valid,
        robust_mask=best_robust,
    )


# ==========================================================================
# Segmented driver (parity of ``calibrate_wavelength_axis_segmented``, J2 §3).
# ==========================================================================
#
# Fixed-shape port of the reference segmented orchestration. The reference flow
# (``:1397``) is:
#
#   detect_ccd_seams (:829)  -> always-computed global fit (:1518)
#     -> global ye6t coverage gate (:1537, re-entrant shift refit :1554)
#     -> per-segment Python loop (:1212/:1080):
#          sparse-segment model restriction (:1121)
#          per-segment fit + trust gate (:976) + coverage gate (:1015) +
#          global-disagreement gate (:1163)
#          fallback 1 = global offset (:1192)
#     -> neighbour fallback (:1287)  [only when global fit unavailable]
#     -> seam-monotonicity restore cascade (:1302)
#     -> revert-to-global gates (cumulative shift > 0.5 :1632; residual
#        non-monotonicity :1641).
#
# This kernel replaces every host construct with fixed-shape JAX:
#   * seams: rolling median of diff(wl) via sliding-window ``jnp.sort``
#     (window 2*W+1), ``segment_id = cumsum(seam_mask)`` clipped to SEG_max;
#   * per-segment calibration: ``lax.map`` of the global kernel over SEG_max
#     padded segment slots (one peak/wl mask per slot);
#   * sparse-segment restriction + coverage degrade-to-shift = a **model
#     lattice** (the segment is calibrated twice — full model set and shift
#     only — and the gate *selects* the precomputed result, no recursion);
#   * trust / coverage / disagreement gates -> branchless masks;
#   * neighbour fallback -> masked argmin |i-j| + segment-masked median offset;
#   * seam-monotonicity restore -> ``lax.scan`` over SEG_max carrying a
#     cumulative shift (exact port of :1302);
#   * revert gates -> branchless select of segmented vs global axis.

#: Max segments in the jit graph (ADR-0004 §4; observed <=11 on csa_planetary).
SEG_MAX_DEFAULT = 16
#: Seam-detector rolling-median half-window (reference default ``window=51``).
SEAM_WINDOW_DEFAULT = 51
#: Dense-anchor-hull fraction (reference ``_COVERAGE_DENSE_HULL_ALPHA``).
COVERAGE_DENSE_HULL_ALPHA = 0.8

#: Per-segment status codes (host maps to the reference status strings).
SEG_STATUS_GLOBAL = 0
SEG_STATUS_FIT = 1
SEG_STATUS_NEIGHBOR = 2
SEG_STATUS_STRINGS: dict[int, str] = {
    SEG_STATUS_GLOBAL: "global",
    SEG_STATUS_FIT: "fit",
    SEG_STATUS_NEIGHBOR: "neighbor",
}

#: Segmented revert reason codes (host maps to reference ``segmented_reverted``).
SEG_REVERT_NONE = 0
SEG_REVERT_LARGE_SEAM_SHIFT = 1
SEG_REVERT_RESIDUAL_NON_MONOTONIC = 2


@dataclass
class SegmentedKernelResult:
    """Device-side result of the segmented wavelength calibrator (pytree).

    Mirrors the reference :func:`calibrate_wavelength_axis_segmented` outputs the
    host wrapper needs to reconstitute a ``WavelengthCalibrationResult`` with
    ``model="segmented"`` (or the inherited global model on the seam-free /
    reverted paths).

    Attributes
    ----------
    corrected_wavelength : array, shape (W_max,)
        Piecewise-corrected axis (or the global axis on the seam-free / reverted
        paths), identity-padded beyond ``wl_mask``.
    n_segments : int scalar
        Number of live segments (``cumsum(seam_mask)`` max + 1, clipped to
        SEG_max).
    seam_count : int scalar
        Number of detected seams.
    segment_status : array, shape (SEG_max,)
        Per-segment status code (``SEG_STATUS_*``); padding slots = -1.
    segment_model_id : array, shape (SEG_max,)
        Per-segment selected model class (after the coverage gate); -1 padding.
    total_inliers : int scalar
        Sum of accepted-fit segment inlier counts.
    quality_passed : bool scalar
        Aggregate gate verdict (any segment fit -> True; else inherits global).
    reverted_code : int scalar
        Revert reason (``SEG_REVERT_*``); 0 when the segmented axis was kept.
    global_result : CalibrationKernelResult
        The (coverage-gated) global single-axis fit — the seam-free answer and
        the per-segment fallback source.
    """

    corrected_wavelength: Any
    n_segments: Any
    seam_count: Any
    segment_status: Any
    segment_model_id: Any
    total_inliers: Any
    quality_passed: Any
    reverted_code: Any
    global_result: Any


def _skr_flatten(r: SegmentedKernelResult):
    children = (
        r.corrected_wavelength,
        r.n_segments,
        r.seam_count,
        r.segment_status,
        r.segment_model_id,
        r.total_inliers,
        r.quality_passed,
        r.reverted_code,
        r.global_result,
    )
    return children, None


def _skr_unflatten(_aux: Any, children: tuple) -> SegmentedKernelResult:
    return SegmentedKernelResult(*children)


jax.tree_util.register_pytree_node(SegmentedKernelResult, _skr_flatten, _skr_unflatten)


# --------------------------------------------------------------------------
# CCD seam detection (parity of detect_ccd_seams :829, rolling-median-of-diffs).
# --------------------------------------------------------------------------


def detect_ccd_seams_kernel(
    wavelength: jnp.ndarray,
    wl_mask: jnp.ndarray,
    *,
    ratio_threshold: float = 3.0,
    window: int = SEAM_WINDOW_DEFAULT,
    min_local_window: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fixed-shape seam mask + segment ids (parity of :func:`detect_ccd_seams`).

    Reference (``:874-888``): ``dl = diff(wl)``; per gap ``i`` a *local* rolling
    median over ``dl[i-w : i+w+1]`` with ``w = max(window, min_local_window)``;
    a gap is a seam when ``dl[i] / max(local_med, 1e-12) > ratio_threshold``.

    The Python ``np.median`` rolling loop (``:881-884``) becomes a single
    sliding-window ``jnp.sort`` over a ``(2w+1)``-wide gathered window per gap
    (J2 spec §3). Padding gaps (either endpoint masked) are excluded from both
    the median windows and the seam test, exactly as the reference operates on
    the live axis only.

    Returns ``(seam_mask, segment_id)`` over the gap axis / sample axis:

    * ``seam_mask`` shape ``(W_max - 1,)`` — True at gap ``i`` between live
      samples ``i`` and ``i+1``;
    * ``segment_id`` shape ``(W_max,)`` — ``cumsum(seam_mask)`` over live
      samples, so sample 0 is segment 0 and each seam opens the next segment;
      padding samples carry the last live segment id.
    """
    dl = jnp.diff(wavelength)  # (W_max-1,)
    gap_valid = wl_mask[1:] & wl_mask[:-1]  # both endpoints live
    n_gaps = dl.shape[0]
    w = int(max(int(window), int(min_local_window)))

    # Sliding window of [-w, ..., +w] gathered gaps, masked to live gaps only.
    offs = jnp.arange(-w, w + 1)  # (2w+1,)
    gap_idx = jnp.arange(n_gaps)
    win_idx = gap_idx[:, None] + offs[None, :]  # (n_gaps, 2w+1)
    in_range = (win_idx >= 0) & (win_idx < n_gaps)
    win_idx_c = jnp.clip(win_idx, 0, n_gaps - 1)
    win_vals = dl[win_idx_c]
    win_live = in_range & gap_valid[win_idx_c]
    # Masked median over each window: live values first (ascending), padding
    # pushed to +inf, then pick the median index of the live count.
    big = jnp.where(win_live, win_vals, jnp.inf)
    srt = jnp.sort(big, axis=1)  # (n_gaps, 2w+1)
    cnt = jnp.sum(win_live, axis=1)  # (n_gaps,)
    half = cnt // 2
    last = srt.shape[1] - 1
    lo = jnp.take_along_axis(srt, jnp.clip(half - 1, 0, last)[:, None], axis=1)[:, 0]
    hi = jnp.take_along_axis(srt, jnp.clip(half, 0, last)[:, None], axis=1)[:, 0]
    even = (cnt % 2) == 0
    local_med = jnp.where(even, 0.5 * (lo + hi), hi)
    local_med = jnp.where(cnt > 0, local_med, jnp.inf)

    local_med = jnp.maximum(local_med, 1e-12)
    ratio = dl / local_med
    seam_mask = (ratio > float(ratio_threshold)) & gap_valid

    # The reference returns no seams on too-short axes (:871). Guard it here so
    # short live axes degrade to a single segment.
    n_live = jnp.sum(wl_mask)
    too_short = n_live < max(3, 2 * int(min_local_window))
    seam_mask = jnp.where(too_short, False, seam_mask)

    # segment_id over samples: sample 0 -> 0; a seam at gap i opens segment for
    # sample i+1 onward. cumsum of seam_mask aligned to the *right* sample.
    seg_step = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), seam_mask.astype(jnp.int32)])
    segment_id = jnp.cumsum(seg_step)
    # Padding samples inherit the last live id (they never enter any segment).
    return seam_mask, segment_id.astype(jnp.int32)


# --------------------------------------------------------------------------
# Dense-anchor-hull coverage gate (parity of :916-1012, JAX/fixed-shape).
# --------------------------------------------------------------------------


def _dense_anchor_hull(anchor_wl: jnp.ndarray, anchor_mask: jnp.ndarray, alpha: float):
    """Shortest interval holding ``ceil(alpha*n)`` sorted live anchors.

    Parity of :func:`_dense_anchor_hull` (``:916``): the highest-density anchor
    interval, robust against stray circularly-matched anchors that stretch a
    plain min/max hull (the ye6t hazard). Fixed-shape: anchors are padded; live
    anchors are sorted to the front (padding -> +inf), and we scan every
    possible window of width ``k`` over the C slots. ``k`` is a traced int used
    only in gather indices (never a shape), so this is jittable as-is (J2 §3).
    """
    c = anchor_wl.shape[0]
    n = jnp.sum(anchor_mask)
    srt = jnp.sort(jnp.where(anchor_mask, anchor_wl, jnp.inf))  # live first, asc
    k = jnp.maximum(jnp.ceil(alpha * n.astype(jnp.float64)).astype(jnp.int32), 1)
    k = jnp.minimum(k, jnp.maximum(n, 1))

    # widths[i] = srt[i + k - 1] - srt[i] for i in [0, n - k]; +inf elsewhere.
    idx = jnp.arange(c)
    right = jnp.clip(idx + k - 1, 0, c - 1)
    width = srt[right] - srt
    valid_start = (idx + k - 1) < n  # window fully inside the live anchors
    width = jnp.where(valid_start, width, jnp.inf)
    i_star = jnp.argmin(width)
    hull_lo = srt[i_star]
    hull_hi = srt[jnp.clip(i_star + k - 1, 0, c - 1)]
    has = n > 0
    hull_lo = jnp.where(has, hull_lo, jnp.nan)
    hull_hi = jnp.where(has, hull_hi, jnp.nan)
    return hull_lo, hull_hi


def _eval_model_traced(xv: jnp.ndarray, coef: jnp.ndarray, model_id: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the calibration model with a *traced* model id (``lax.switch``).

    The coefficient convention is model-specific (affine ``(a, b, 0)`` is NOT
    the quadratic ``(c2, c1, c0)`` order), so the coverage gate — which sees a
    traced ``model_id`` — must dispatch the right :func:`_eval_model` branch.
    """
    return lax.switch(
        jnp.clip(model_id, 0, 2),
        [
            lambda: _eval_model(xv, coef, MODEL_SHIFT),
            lambda: _eval_model(xv, coef, MODEL_AFFINE),
            lambda: _eval_model(xv, coef, MODEL_QUADRATIC),
        ],
    )


def _slope_correction(coef: jnp.ndarray, xv: jnp.ndarray, model_id: jnp.ndarray) -> jnp.ndarray:
    """Correction ``corr(x) = model(x) - x`` for a slope model (affine/quad)."""
    return _eval_model_traced(xv, coef, model_id) - xv


def _coverage_extrapolation_nm(
    seg_lo: jnp.ndarray,
    seg_hi: jnp.ndarray,
    coef: jnp.ndarray,
    model_id: jnp.ndarray,
    is_slope: jnp.ndarray,
    hull_lo: jnp.ndarray,
    hull_hi: jnp.ndarray,
) -> jnp.ndarray:
    """Worst-case correction drift from the anchor hull to a segment edge.

    Parity of :func:`_coverage_extrapolation_nm` (``:938``): a slope model's
    correction keeps changing past its anchors, so we return
    ``max(|corr(edge) - corr(nearest clamped hull edge)|)`` over both ends.
    A ``shift`` model (constant correction) returns 0; a non-finite hull
    (no anchors) returns +inf.
    """
    finite = jnp.isfinite(hull_lo) & jnp.isfinite(hull_hi)

    drift = jnp.array(0.0, dtype=seg_lo.dtype)
    for edge, hull in (
        (seg_lo, jnp.minimum(jnp.maximum(hull_lo, seg_lo), seg_hi)),
        (seg_hi, jnp.minimum(jnp.maximum(hull_hi, seg_lo), seg_hi)),
    ):
        drift = jnp.maximum(
            drift,
            jnp.abs(
                _slope_correction(coef, edge, model_id) - _slope_correction(coef, hull, model_id)
            ),
        )
    drift = jnp.where(finite, drift, jnp.inf)
    # A pure shift model has constant correction => 0 drift (reference :955).
    return jnp.where(is_slope, drift, 0.0)


def _segment_anchor_coverage(
    anchor_wl: jnp.ndarray,
    robust_mask: jnp.ndarray,
    coef: jnp.ndarray,
    model_id: jnp.ndarray,
    is_slope: jnp.ndarray,
    seg_lo: jnp.ndarray,
    seg_hi: jnp.ndarray,
    local_px: jnp.ndarray,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(span_fraction, extrapolation_nm, extrapolation_px)`` (parity :990)."""
    hull_lo, hull_hi = _dense_anchor_hull(anchor_wl, robust_mask, alpha)
    seg_span = jnp.maximum(seg_hi - seg_lo, 1e-9)
    finite = jnp.isfinite(hull_lo) & jnp.isfinite(hull_hi)
    span_fraction = jnp.where(finite, (hull_hi - hull_lo) / seg_span, 0.0)
    extrap_nm = _coverage_extrapolation_nm(
        seg_lo, seg_hi, coef, model_id, is_slope, hull_lo, hull_hi
    )
    extrap_px = extrap_nm / jnp.maximum(local_px, 1e-9)
    return span_fraction, extrap_nm, extrap_px


# --------------------------------------------------------------------------
# Per-segment calibration (model lattice: full model set + shift-only).
# --------------------------------------------------------------------------


def _segment_masks(
    segment_id: jnp.ndarray,
    wl_mask: jnp.ndarray,
    peak_wl: jnp.ndarray,
    peak_mask: jnp.ndarray,
    wavelength: jnp.ndarray,
    s: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-segment wl-mask, peak-mask, edges, n_pts, local pixel for slot ``s``.

    A wl sample belongs to segment ``s`` iff it is live AND ``segment_id == s``.
    A peak belongs to segment ``s`` iff it is live and its wavelength falls in
    the segment's ``[lo, hi]`` span (matching the reference ``wavelength[a:b]``
    slice that carries the peaks detected inside that channel).
    """
    seg_wl_mask = wl_mask & (segment_id == s)
    n_pts = jnp.sum(seg_wl_mask)
    seg_lo = jnp.min(jnp.where(seg_wl_mask, wavelength, jnp.inf))
    seg_hi = jnp.max(jnp.where(seg_wl_mask, wavelength, -jnp.inf))
    seg_lo = jnp.where(n_pts > 0, seg_lo, 0.0)
    seg_hi = jnp.where(n_pts > 0, seg_hi, 0.0)
    # Local pixel = median live gap of this segment.
    gaps = jnp.diff(wavelength)
    gap_in = seg_wl_mask[1:] & seg_wl_mask[:-1]
    local_px = _masked_median(jnp.abs(gaps), gap_in)
    local_px = jnp.where(jnp.isfinite(local_px), local_px, 0.0)
    # A peak belongs to the segment iff live and inside [lo, hi].
    seg_peak_mask = peak_mask & (peak_wl >= seg_lo) & (peak_wl <= seg_hi) & (n_pts > 0)
    return seg_wl_mask, seg_peak_mask, seg_lo, seg_hi, local_px, n_pts


def calibrate_segmented_kernel(
    peak_wl: jnp.ndarray,
    peak_amp: jnp.ndarray,
    peak_mask: jnp.ndarray,
    line_wl: jnp.ndarray,
    line_strength: jnp.ndarray,
    line_mask: jnp.ndarray,
    wavelength: jnp.ndarray,
    wl_mask: jnp.ndarray,
    *,
    inlier_tolerance_nm: float = 0.08,
    max_pair_window_nm: float = 2.0,
    seam_ratio_threshold: float = 3.0,
    seam_window: int = SEAM_WINDOW_DEFAULT,
    min_segment_points: int = 16,
    segment_min_inliers: float = 10.0,
    segment_max_rmse_nm: float = 0.06,
    segment_max_global_disagreement_nm: float = 0.5,
    sparse_segment_points: int = 400,
    affine_coverage_gate: bool = True,
    coverage_min_anchor_span_fraction: float = 0.6,
    coverage_max_extrapolation_px: float = 1.0,
    candidate_models: tuple[int, ...] = (MODEL_SHIFT, MODEL_AFFINE),
    sparse_segment_models: tuple[int, ...] = (MODEL_SHIFT,),
    k_pair: int = K_PAIR_DEFAULT,
    h_affine: int = H_AFFINE_DEFAULT,
    h_block: int = H_BLOCK_DEFAULT,
    seg_max: int = SEG_MAX_DEFAULT,
    seed: int = 42,
) -> SegmentedKernelResult:
    """Fixed-shape segmented calibrator (parity of ``calibrate_wavelength_axis_segmented``).

    All array inputs are padded; ``*_mask`` arrays mark valid entries.
    ``line_wl`` MUST be sorted ascending. Static (graph-shaping) arguments:
    ``candidate_models`` / ``sparse_segment_models`` / ``k_pair`` / ``h_affine``
    / ``h_block`` / ``seg_max`` / ``seam_window``.

    The flow mirrors the reference §3 driver: seams -> always-computed global fit
    (with the global ye6t coverage gate as a model-lattice select) -> per-segment
    vmap of the global kernel (sparse restriction + coverage gate + trust gate +
    global-disagreement gate) -> global-offset fallback -> seam-monotonicity
    restore -> revert-to-global gates.
    """
    alpha = COVERAGE_DENSE_HULL_ALPHA

    # ------------------------------------------------------------------ seams
    seam_mask, segment_id = detect_ccd_seams_kernel(
        wavelength,
        wl_mask,
        ratio_threshold=seam_ratio_threshold,
        window=seam_window,
    )
    seam_count = jnp.sum(seam_mask)
    # Clip segment ids to seg_max-1 so an over-segmented axis never indexes out
    # of the padded segment axis (reference clips SEG_max=16; observed <=11).
    segment_id = jnp.minimum(segment_id, seg_max - 1)
    n_segments = jnp.minimum(jnp.max(jnp.where(wl_mask, segment_id, 0)) + 1, seg_max)

    # ----------------------------------------------------------- global fit
    # Always-computed global single-axis fit. Apply the global ye6t coverage
    # gate as a model lattice: compute the full-model fit AND a shift-only fit,
    # then SELECT shift when the slope model's anchors do not cover the axis —
    # same answer as the reference re-entrant shift refit (:1554), no recursion.
    global_full = calibrate_axis_kernel(
        peak_wl,
        peak_amp,
        peak_mask,
        line_wl,
        line_strength,
        line_mask,
        wavelength,
        wl_mask,
        inlier_tolerance_nm=inlier_tolerance_nm,
        max_pair_window_nm=max_pair_window_nm,
        apply_quality_gate=True,
        candidate_models=candidate_models,
        k_pair=k_pair,
        h_affine=h_affine,
        h_block=h_block,
        seed=seed,
    )
    global_shift = calibrate_axis_kernel(
        peak_wl,
        peak_amp,
        peak_mask,
        line_wl,
        line_strength,
        line_mask,
        wavelength,
        wl_mask,
        inlier_tolerance_nm=inlier_tolerance_nm,
        max_pair_window_nm=max_pair_window_nm,
        apply_quality_gate=True,
        candidate_models=(MODEL_SHIFT,),
        k_pair=k_pair,
        h_affine=h_affine,
        h_block=h_block,
        seed=seed,
    )

    g_x, _gy, g_pid, _gl, _gw, g_cmask = build_banded_pairs(
        peak_wl,
        peak_amp,
        peak_mask,
        line_wl,
        line_strength,
        line_mask,
        max_pair_window_nm,
        k_pair,
    )
    g_pid = g_pid.astype(jnp.int32)
    g_anchor_wl = g_x  # x = peak_wl[peak_id] broadcast
    full_lo = jnp.min(jnp.where(wl_mask, wavelength, jnp.inf))
    full_hi = jnp.max(jnp.where(wl_mask, wavelength, -jnp.inf))
    g_gaps = jnp.diff(wavelength)
    g_gap_in = wl_mask[1:] & wl_mask[:-1]
    g_local_px = _masked_median(jnp.abs(g_gaps), g_gap_in)
    g_local_px = jnp.where(jnp.isfinite(g_local_px), g_local_px, 0.0)

    # Coverage fails iff the global fit is a slope model AND under-covers.
    g_is_slope = global_full.model_id != MODEL_SHIFT
    g_span_frac, _g_extrap_nm, g_extrap_px = _segment_anchor_coverage(
        g_anchor_wl,
        global_full.robust_mask & g_cmask,
        global_full.coefficients,
        global_full.model_id,
        g_is_slope,
        full_lo,
        full_hi,
        g_local_px,
        alpha,
    )
    g_cov_fail = (
        affine_coverage_gate
        & global_full.success
        & g_is_slope
        & (
            (g_span_frac < coverage_min_anchor_span_fraction)
            | (g_extrap_px > coverage_max_extrapolation_px)
        )
    )
    global_result = _select_kernel_result(g_cov_fail, global_shift, global_full)

    # global offset over the axis (per-sample correction the fallback uses).
    global_corrected = jnp.where(
        global_result.success, global_result.corrected_wavelength, wavelength
    )
    global_offset = global_corrected - wavelength

    # --------------------------------------------------- per-segment lattice
    seg_ids = jnp.arange(seg_max)

    def run_one_segment(s):
        seg_wl_mask, seg_peak_mask, seg_lo, seg_hi, local_px, n_pts = _segment_masks(
            segment_id, wl_mask, peak_wl, peak_mask, wavelength, s
        )
        # Per-segment seed mirrors the reference ``random_seed + index`` (:1132).
        seg_seed = seed + s
        # Full-model fit (well-populated segment).
        full = calibrate_axis_kernel(
            peak_wl,
            peak_amp,
            seg_peak_mask,
            line_wl,
            line_strength,
            line_mask,
            wavelength,
            seg_wl_mask,
            inlier_tolerance_nm=inlier_tolerance_nm,
            max_pair_window_nm=max_pair_window_nm,
            apply_quality_gate=False,
            candidate_models=candidate_models,
            k_pair=k_pair,
            h_affine=h_affine,
            h_block=h_block,
            seed=seg_seed,
        )
        # Shift-only fit (sparse segment AND coverage degrade target).
        shift = calibrate_axis_kernel(
            peak_wl,
            peak_amp,
            seg_peak_mask,
            line_wl,
            line_strength,
            line_mask,
            wavelength,
            seg_wl_mask,
            inlier_tolerance_nm=inlier_tolerance_nm,
            max_pair_window_nm=max_pair_window_nm,
            apply_quality_gate=False,
            candidate_models=sparse_segment_models,
            k_pair=k_pair,
            h_affine=h_affine,
            h_block=h_block,
            seed=seg_seed,
        )
        return _resolve_one_segment(
            s,
            full,
            shift,
            seg_wl_mask,
            seg_peak_mask,
            seg_lo,
            seg_hi,
            local_px,
            n_pts,
            peak_wl,
            peak_amp,
            line_wl,
            line_strength,
            line_mask,
            wavelength,
            global_offset,
            max_pair_window_nm,
            k_pair,
            min_segment_points,
            sparse_segment_points,
            segment_min_inliers,
            segment_max_rmse_nm,
            segment_max_global_disagreement_nm,
            affine_coverage_gate,
            coverage_min_anchor_span_fraction,
            coverage_max_extrapolation_px,
            alpha,
        )

    # ``lax.map`` keeps only one segment's residual matrix live at a time.
    seg_corr, seg_status, seg_model, seg_inliers = jax.lax.map(run_one_segment, seg_ids)

    # ------------------------------------------------- stitch corrected axis
    # Each sample takes ITS segment's corrected value: gather row ``segment_id``
    # column ``i`` from ``seg_corr`` (shape (seg_max, W_max)). Where a segment
    # fell back to global, the per-segment correction already IS
    # ``wavelength + global_offset`` (see _resolve_one_segment), so this is exact.
    samp_idx = jnp.arange(seg_corr.shape[1])
    samp_corr = seg_corr[segment_id, samp_idx]  # (W_max,)
    corrected = jnp.where(wl_mask, samp_corr, wavelength)

    # --------------------------------------- seam-monotonicity restore (:1302)
    corrected, cumulative_shift, _n_clamped = _restore_seam_monotonicity(
        corrected, segment_id, wl_mask, seg_max
    )

    # ------------------------------------------------------- revert-to-global
    # Residual non-monotonicity on the live axis (should be impossible, :1641).
    d = jnp.diff(corrected)
    pair_live = wl_mask[1:] & wl_mask[:-1]
    residual_non_mono = jnp.any(jnp.where(pair_live, d <= 0, False))
    revert_large = (seam_count > 0) & (cumulative_shift > 0.5)
    revert_resid = (seam_count > 0) & residual_non_mono & (~revert_large)
    reverted_code = jnp.where(
        revert_large,
        SEG_REVERT_LARGE_SEAM_SHIFT,
        jnp.where(revert_resid, SEG_REVERT_RESIDUAL_NON_MONOTONIC, SEG_REVERT_NONE),
    )
    revert = revert_large | revert_resid

    # Seam-free axis: degrade to the global fit (:1572). Reverted axis: global.
    use_global_axis = (seam_count == 0) | revert
    final_corrected = jnp.where(use_global_axis, global_corrected, corrected)

    # ------------------------------------------------------------- aggregate
    seg_live = seg_ids < n_segments
    n_fit = jnp.sum((seg_status == SEG_STATUS_FIT) & seg_live)
    total_inliers = jnp.sum(jnp.where(seg_live, seg_inliers, 0))
    # Quality: any segment fit -> passed; else inherit the global verdict (:1360).
    seg_quality = jnp.where(n_fit > 0, True, global_result.quality_passed)
    quality_passed = jnp.where(use_global_axis, global_result.quality_passed, seg_quality)

    seg_status_out = jnp.where(seg_live, seg_status, -1)
    seg_model_out = jnp.where(seg_live & (seg_status == SEG_STATUS_FIT), seg_model, -1)

    return SegmentedKernelResult(
        corrected_wavelength=final_corrected,
        n_segments=n_segments,
        seam_count=seam_count,
        segment_status=seg_status_out,
        segment_model_id=seg_model_out,
        total_inliers=total_inliers,
        quality_passed=quality_passed,
        reverted_code=jnp.where(use_global_axis & (seam_count > 0), reverted_code, SEG_REVERT_NONE),
        global_result=global_result,
    )


def _select_kernel_result(
    pick_b: jnp.ndarray, a: CalibrationKernelResult, b: CalibrationKernelResult
) -> CalibrationKernelResult:
    """Branchless select between two :class:`CalibrationKernelResult` pytrees.

    ``pick_b`` True -> take ``a`` (the degrade target); False -> take ``b``.
    Implemented as a leaf-wise ``jnp.where`` so it stays jit/vmap clean (a model
    lattice select, not a Python branch).
    """

    def sel(la, lb):
        return jnp.where(pick_b, la, lb)

    return jax.tree_util.tree_map(sel, a, b)


def _resolve_one_segment(
    s,
    full: CalibrationKernelResult,
    shift: CalibrationKernelResult,
    seg_wl_mask,
    seg_peak_mask,
    seg_lo,
    seg_hi,
    local_px,
    n_pts,
    peak_wl,
    peak_amp,
    line_wl,
    line_strength,
    line_mask,
    wavelength,
    global_offset,
    max_pair_window_nm,
    k_pair,
    min_segment_points,
    sparse_segment_points,
    segment_min_inliers,
    segment_max_rmse_nm,
    segment_max_global_disagreement_nm,
    affine_coverage_gate,
    coverage_min_anchor_span_fraction,
    coverage_max_extrapolation_px,
    alpha,
):
    """Apply the per-segment gate lattice (parity of :func:`_fit_one_segment`).

    Returns ``(seg_corrected_axis, status_code, model_id, accepted_inliers)``.
    The flow: sparse restriction -> trust gate -> coverage gate (model-lattice
    select to shift) -> re-check trust -> global-disagreement gate -> accept or
    fall back to the global offset over this segment.
    """
    # Sparse-segment restriction (:1121): few points -> shift-only model set.
    is_sparse = n_pts < sparse_segment_points
    base = _select_kernel_result(is_sparse, shift, full)

    # Trust gate (:976): success & model != none & inliers & rmse.
    def trusted_of(res):
        return (
            res.success
            & (res.n_inliers >= segment_min_inliers)
            & (res.rmse_nm <= segment_max_rmse_nm)
        )

    trusted0 = trusted_of(base)

    # Coverage gate (:1015): only for trusted slope models. Degrade to shift.
    # Re-derive the candidate geometry for the segment so the robust mask aligns
    # with the candidate slots (x = peak_wl[peak_id] broadcast = the anchors).
    x_s, _y_s, _pid_s, _lid_s, _w_s, cmask_s = build_banded_pairs(
        peak_wl,
        peak_amp,
        seg_peak_mask,
        line_wl,
        line_strength,
        line_mask,
        max_pair_window_nm,
        k_pair,
    )
    base_is_slope = base.model_id != MODEL_SHIFT
    span_frac, _extrap_nm, extrap_px = _segment_anchor_coverage(
        x_s,
        base.robust_mask & cmask_s,
        base.coefficients,
        base.model_id,
        base_is_slope,
        seg_lo,
        seg_hi,
        local_px,
        alpha,
    )
    cov_fail = (
        affine_coverage_gate
        & trusted0
        & base_is_slope
        & (
            (span_frac < coverage_min_anchor_span_fraction)
            | (extrap_px > coverage_max_extrapolation_px)
        )
    )
    seg_res = _select_kernel_result(cov_fail, shift, base)
    trusted = trusted_of(seg_res)

    # Global-disagreement gate (:1163): a trusted segment fit whose median
    # correction departs from the global correction by more than the bound is a
    # catalog alias -> demote to the global fallback.
    seg_offset_med = _masked_median(seg_res.corrected_wavelength - wavelength, seg_wl_mask)
    global_offset_med = _masked_median(global_offset, seg_wl_mask)
    disagreement = jnp.abs(seg_offset_med - global_offset_med)
    disagree_fail = trusted & (disagreement > segment_max_global_disagreement_nm)
    trusted = trusted & (~disagree_fail)

    # Reference only fits at all when n_pts >= min_segment_points (:1120).
    eligible = n_pts >= min_segment_points
    accept_fit = trusted & eligible

    # Accepted -> the segment's own corrected axis (over its samples); else the
    # global single-axis correction over this segment (fallback 1, :1192).
    fit_offset = seg_res.corrected_wavelength - wavelength
    seg_offset = jnp.where(accept_fit, fit_offset, global_offset)
    seg_corrected = wavelength + seg_offset

    status = jnp.where(accept_fit, SEG_STATUS_FIT, SEG_STATUS_GLOBAL)
    model_id = jnp.where(accept_fit, seg_res.model_id, -1)
    accepted_inliers = jnp.where(accept_fit, seg_res.n_inliers, 0)
    return seg_corrected, status, model_id.astype(jnp.int32), accepted_inliers.astype(jnp.int32)


def _restore_seam_monotonicity(
    corrected: jnp.ndarray,
    segment_id: jnp.ndarray,
    wl_mask: jnp.ndarray,
    seg_max: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shift downstream segments up to remove seam overlaps (parity :1302).

    The reference walks seam boundaries ``k=1..n_seg-1``; at each, if the last
    corrected sample of segment ``k-1`` is >= the first of segment ``k`` it adds
    ``deficit + 1e-6`` to all of segment ``k`` onward, cascading the shift. We
    port it as a ``lax.scan`` over the ``seg_max`` segments carrying the running
    cumulative shift; per-segment additive shifts are then scattered to samples.

    Returns ``(corrected, cumulative_shift, n_clamped)``.
    """
    # Per-segment first/last live corrected sample BEFORE any restore shift.
    seg_ids = jnp.arange(seg_max)

    def seg_edges(s):
        in_s = wl_mask & (segment_id == s)
        first = jnp.min(jnp.where(in_s, corrected, jnp.inf))
        last = jnp.max(jnp.where(in_s, corrected, -jnp.inf))
        live = jnp.any(in_s)
        return first, last, live

    firsts, lasts, lives = jax.vmap(seg_edges)(seg_ids)

    def step(carry, s):
        cum_shift, prev_last, max_clamp, n_clamped = carry
        live = lives[s]
        first = firsts[s] + cum_shift  # this segment after the running shift
        # Deficit vs the previous live segment's (already-shifted) last sample.
        deficit = prev_last - first
        need = live & (s > 0) & (deficit >= 0)
        add = jnp.where(need, deficit + 1e-6, 0.0)
        cum_shift = cum_shift + add
        seg_shift = jnp.where(live, cum_shift, 0.0)
        new_last = jnp.where(live, lasts[s] + cum_shift, prev_last)
        max_clamp = jnp.maximum(max_clamp, add)
        n_clamped = n_clamped + need.astype(jnp.int32)
        return (cum_shift, new_last, max_clamp, n_clamped), seg_shift

    init = (
        jnp.array(0.0, dtype=corrected.dtype),
        -jnp.inf,
        jnp.array(0.0, corrected.dtype),
        jnp.array(0, jnp.int32),
    )
    (cum_total, _pl, _mc, n_clamped), seg_shifts = lax.scan(step, init, seg_ids)
    # Scatter the per-segment additive shift back to its samples.
    samp_shift = jnp.where(wl_mask, seg_shifts[segment_id], 0.0)
    corrected = corrected + samp_shift
    return corrected, cum_total, n_clamped

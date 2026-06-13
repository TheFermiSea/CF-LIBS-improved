"""Stage J4 — fixed-shape fit kernels: line selection, Boltzmann/SB-graph, closure.

Implements ADR-0004 §4 rows 5/6/9 as batched, masked, ``jit``/``vmap``-clean JAX
kernels over ``(B, E, N)`` padded arrays. The whole point (ADR §1.1, the
7394-cache-entry pathology) is that NO data-dependent shape ever appears inside a
trace: every stage is a pure mask transform on fixed-shape arrays, with failures
surfaced as *quality flags*, never exceptions.

Three back-end stages live here:

* **Line selection** (§2) — pure mask transform: score = SNR × (1/sigma_atomic) ×
  isolation, sequential boolean gates, per-element ``lax.top_k`` (static K=20).
  Contract: exact selected-set equality for tie-free inputs; deterministic
  tiebreak (lower original index wins, matching the Python stable sort).
* **SB-graph / common-slope** (§3) — the Saha-Boltzmann graph with unit weights
  is *algebraically identical* to the centered common-slope kernel (the arrow
  matrix's Schur complement). We therefore reuse the existing bit-exact JAX twins
  :func:`_saha_correct_kernel` / :func:`_common_slope_kernel` instead of any
  on-device ``lstsq``. Contract: slope/intercepts rtol 1e-10 vs
  ``np.linalg.lstsq`` on identical rows.
* **Closure** (§4) — standard / matrix / oxide masked linear algebra lifted one
  batch axis from the existing lax closures, plus ILR-as-standard (the documented
  identity round-trip) and the keystone-degeneracy gate. No ``pure_callback``.

A shared :func:`masked_median` (sort-with-+inf-fill, average the two central
order statistics) is exact ``np.median`` for both parities and is reused by the
weight cap here and (per the spec) by J5/J6.

Physics-only: this module imports only ``jax`` / ``jax.numpy`` and the existing
physics JAX twins. No SQLite, no host imports, no ``sklearn``/``torch``/
``jax.nn``/``jaxopt`` (ruff TID251).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from cflibs.inversion.physics.boltzmann_jax import batched_boltzmann_fit
from cflibs.inversion.solve.iterative import _common_slope_kernel, _saha_correct_kernel

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot

# Static top-K per element (physics/line_selection.py:145, max_lines_per_element).
TOP_K_PER_ELEMENT = 20

# Default atomic-data relative uncertainty when none supplied
# (physics/line_selection.py:338, ``atomic_uncertainties.get(key, 0.10)``).
DEFAULT_ATOMIC_UNCERTAINTY = 0.10

# Default SNR when no intensity uncertainty is provided
# (physics/line_selection.py:334, ``snr = 100.0``).
DEFAULT_SNR = 100.0

# Isolation-gate threshold (physics/line_selection.py:258, ``isolation < 0.5``).
ISOLATION_GATE = 0.5

# Number of fixed sigma-clip iterations (boltzmann.py:345, ``range(max_iterations)``
# with the default ``max_iterations=10``). CPU early termination is an idempotent
# fixed point, so the fixed-K scan reaches the identical result.
SIGMA_CLIP_ITERS = 10

# Outlier rejection threshold in residual-sigma units (boltzmann.py:107, default
# ``outlier_sigma=2.5``).
OUTLIER_SIGMA = 2.5

# Log-ratio clip floor for ILR parity (physics/closure.py:34).
LOGRATIO_CLIP_FLOOR = 1e-10


# ---------------------------------------------------------------------------
# Shared masked-statistics helpers (reused by J5/J6 per spec §3, §8)
# ---------------------------------------------------------------------------


def masked_median(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Exact ``np.median`` of the masked entries along the last axis.

    Padded / invalid entries (``mask == False``) are pushed to ``+inf`` by the
    sort and excluded from the order-statistic gather, so the result is the
    median of the *valid* entries only. Matches ``numpy.median`` for both even
    and odd valid counts (averages the two central order statistics).

    Parameters
    ----------
    values : array, shape ``(..., N)``
        Values to take the median over (last axis).
    mask : array of bool, shape ``(..., N)``
        Valid-entry mask; ``False`` entries are ignored.

    Returns
    -------
    array, shape ``(...)``
        Per-row median of the valid entries; ``0.0`` where no entry is valid.
    """
    values = jnp.asarray(values, dtype=jnp.float64)
    mask_b = jnp.asarray(mask, dtype=bool)
    n_valid = jnp.sum(mask_b.astype(jnp.float64), axis=-1)  # (...,)

    # Push invalid entries to +inf so they sort to the high end and never land
    # in a central order-statistic slot for any valid count.
    filled = jnp.where(mask_b, values, jnp.inf)
    srt = jnp.sort(filled, axis=-1)  # (..., N), valid entries first, ascending

    n = jnp.maximum(n_valid, 1.0)
    lo = jnp.floor((n - 1.0) / 2.0).astype(jnp.int32)  # (...,)
    hi = jnp.floor(n / 2.0).astype(jnp.int32)  # (...,)

    lo_val = jnp.take_along_axis(srt, lo[..., None], axis=-1)[..., 0]
    hi_val = jnp.take_along_axis(srt, hi[..., None], axis=-1)[..., 0]
    med = 0.5 * (lo_val + hi_val)
    return jnp.where(n_valid > 0.0, med, 0.0)


def cap_weights(weights: jnp.ndarray, mask: jnp.ndarray, cap: jnp.ndarray) -> jnp.ndarray:
    """Masked port of ``_cap_boltzmann_weights`` (iterative.py:2911-2944).

    Clips each row's weights to ``cap × median(valid finite-positive weights)``.
    Entries at or below the cap are untouched; a non-positive ``cap`` disables
    the clip. The median is taken over the finite, strictly-positive *valid*
    entries only (matching the reference, which masks before calling).

    Parameters
    ----------
    weights : array, shape ``(..., N)``
        Per-line weights (inverse variance), already zeroed/ignored where masked.
    mask : array of bool, shape ``(..., N)``
        Valid-line mask.
    cap : scalar array
        Multiplier ``K``. ``cap <= 0`` returns ``weights`` unchanged.

    Returns
    -------
    array
        Clipped weights (same shape as ``weights``).
    """
    weights = jnp.asarray(weights, dtype=jnp.float64)
    mask_b = jnp.asarray(mask, dtype=bool)
    cap = jnp.asarray(cap, dtype=jnp.float64)

    finite_pos = mask_b & jnp.isfinite(weights) & (weights > 0.0)
    med = masked_median(weights, finite_pos)  # (...,)
    has_pos = jnp.any(finite_pos, axis=-1)  # (...,)
    med_ok = has_pos & jnp.isfinite(med) & (med > 0.0)

    cap_value = (cap * med)[..., None]
    clipped = jnp.minimum(weights, cap_value)
    # Apply only where the cap is enabled AND a usable median exists.
    apply = (cap > 0.0) & med_ok
    return jnp.where(apply[..., None], clipped, weights)


# ---------------------------------------------------------------------------
# §2 Line selection — pure mask transform
# ---------------------------------------------------------------------------


def line_scores(
    wavelength_nm: jnp.ndarray,
    intensity: jnp.ndarray,
    intensity_unc: jnp.ndarray,
    atomic_unc: jnp.ndarray,
    mask: jnp.ndarray,
    isolation_scale_nm: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Score lines exactly as ``LineSelector._score_line`` (line_selection.py:322-389).

    SNR = I/sigma_I (or :data:`DEFAULT_SNR` when sigma_I <= 0); isolation =
    ``1 - exp(-Δλ_min / scale)`` over all *valid* lines (masked pairwise min,
    line_selection.py:362-389); score = SNR × (1/atomic_unc) × isolation, with
    atomic_unc defaulting to :data:`DEFAULT_ATOMIC_UNCERTAINTY` upstream.

    All arrays are shape ``(B, L)``. Returns a dict of per-line ``(B, L)``
    arrays: ``score``, ``snr``, ``isolation``.
    """
    wavelength_nm = jnp.asarray(wavelength_nm, dtype=jnp.float64)
    intensity = jnp.asarray(intensity, dtype=jnp.float64)
    intensity_unc = jnp.asarray(intensity_unc, dtype=jnp.float64)
    atomic_unc = jnp.asarray(atomic_unc, dtype=jnp.float64)
    mask_b = jnp.asarray(mask, dtype=bool)
    scale = jnp.asarray(isolation_scale_nm, dtype=jnp.float64)

    # SNR: I/sigma_I, else DEFAULT_SNR when no usable uncertainty.
    has_unc = intensity_unc > 0.0
    snr = jnp.where(has_unc, intensity / jnp.where(has_unc, intensity_unc, 1.0), DEFAULT_SNR)

    # Isolation: min |Δλ| to any *other valid* line (line_selection.py:362-389).
    # The reference loops over all observations and skips ``other is obs``; only
    # valid (present) lines exist there, so we mask invalid neighbours to +inf.
    dwl = jnp.abs(wavelength_nm[:, :, None] - wavelength_nm[:, None, :])  # (B, L, L)
    big = jnp.inf
    # Exclude self (diagonal) and invalid neighbours.
    eye = jnp.eye(wavelength_nm.shape[-1], dtype=bool)[None, :, :]
    neighbour_ok = mask_b[:, None, :] & (~eye)
    dwl = jnp.where(neighbour_ok, dwl, big)
    min_sep = jnp.min(dwl, axis=-1)  # (B, L)
    # No neighbours -> +inf -> isolation 1.0 (line_selection.py:382-383).
    has_neighbour = jnp.isfinite(min_sep)
    scale_safe = jnp.where(scale > 0.0, scale, 1.0)
    isolation = jnp.where(has_neighbour, 1.0 - jnp.exp(-min_sep / scale_safe), 1.0)

    # Score = SNR × (1/atomic_unc) × isolation (line_selection.py:346-351).
    atomic_ok = atomic_unc > 0.0
    inv_atomic = jnp.where(atomic_ok, 1.0 / jnp.where(atomic_ok, atomic_unc, 1.0), 1.0)
    score = jnp.where(atomic_ok, snr * inv_atomic * isolation, snr * isolation)

    # Zero out scores for padded lines so they never win a top-K slot.
    score = jnp.where(mask_b, score, -jnp.inf)
    return {"score": score, "snr": snr, "isolation": isolation}


def select_lines(
    wavelength_nm: jnp.ndarray,
    intensity: jnp.ndarray,
    intensity_unc: jnp.ndarray,
    atomic_unc: jnp.ndarray,
    element_idx: jnp.ndarray,
    is_resonance: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    n_elements: int,
    min_snr: jnp.ndarray,
    isolation_scale_nm: jnp.ndarray,
    energy_ev: jnp.ndarray | None = None,
    exclude_resonance: bool = False,
    top_k: int = TOP_K_PER_ELEMENT,
) -> dict[str, jnp.ndarray]:
    """Fixed-shape line selection (``LineSelector.select``, line_selection.py:176-290).

    Pure mask transform on ``(B, L)`` inputs. Sequential gates (SNR, optional
    resonance, isolation) are a boolean AND; per-element top-K is a static
    ``lax.top_k`` over a lexicographically packed (score, -index) key so ties
    break to the lower original index, matching the Python stable sort.

    Parameters
    ----------
    wavelength_nm, intensity, intensity_unc, atomic_unc : array ``(B, L)``
        Per-line atomic/measured quantities.
    element_idx : int array ``(B, L)``
        Element index ``0..n_elements-1`` per line (``-1``/invalid for padding).
    is_resonance : bool array ``(B, L)``
        Resonance-line flag (gate only applies when ``exclude_resonance``).
    mask : bool array ``(B, L)``
        Valid-line (present) mask.
    n_elements : int
        Static element-axis length ``E``.
    min_snr, isolation_scale_nm : scalar arrays
        Traced selection knobs.
    energy_ev : array ``(B, L)``, optional
        Per-line upper-level energy ``E_k`` [eV], used only for the per-element
        ``spread_ev`` diagnostic (host-side warning, not a gate). When ``None``
        the spread is reported as ``0.0`` (the diagnostic is simply unavailable).
    exclude_resonance : bool
        Static gate toggle (default off, matching the validated CLI default).
    top_k : int
        Static per-element cap (default :data:`TOP_K_PER_ELEMENT` = 20).

    Returns
    -------
    dict
        ``selected_mask`` ``(B, L)`` bool: lines kept by the gates+top-K.
        ``score``/``snr``/``isolation`` ``(B, L)``: per-line diagnostics.
        ``topk_index`` ``(B, E, top_k)`` int: per-element selected line indices
        (``-1`` for empty slots) — the fixed ``(B, E, N)`` selection used by the
        fit, so the variable-length list form is never materialised.
        ``topk_valid`` ``(B, E, top_k)`` bool: which top-K slots hold a real line.
        ``spread_ev`` / ``n_valid`` ``(B, E)``: per-element diagnostics
        (energy-spread / min-lines are host-side warnings, not gates).
    """
    mask_b = jnp.asarray(mask, dtype=bool)
    element_idx = jnp.asarray(element_idx, dtype=jnp.int32)
    is_resonance = jnp.asarray(is_resonance, dtype=bool)
    B, L = mask_b.shape

    scored = line_scores(
        wavelength_nm, intensity, intensity_unc, atomic_unc, mask_b, isolation_scale_nm
    )
    snr = scored["snr"]
    isolation = scored["isolation"]
    score = scored["score"]

    # Sequential gates (line_selection.py:241-264) collapsed to a boolean AND.
    min_snr = jnp.asarray(min_snr, dtype=jnp.float64)
    gate_snr = snr >= min_snr
    gate_iso = isolation >= ISOLATION_GATE
    gate_res = jnp.ones_like(mask_b) if not exclude_resonance else (~is_resonance)
    gate = mask_b & gate_snr & gate_iso & gate_res

    # Per-element top-K with an EXACT lexicographic (score desc, index asc) order
    # so equal scores keep input order — bit-identical to the Python stable
    # descending sort (line_selection.py:276). We sort within each element by a
    # composite key built from two stable sorts (radix-style): first by index
    # ascending (the secondary key), then by score descending (the primary key).
    # ``jnp.argsort`` is stable, so the second sort preserves the index order on
    # genuine score ties — no float perturbation, no ordering flips.
    gated_score = jnp.where(gate, score, -jnp.inf)  # (B, L)

    def _per_element(e: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        in_el = element_idx == e  # (B, L)
        elig = in_el & gate  # (B, L)
        el_score = jnp.where(elig, gated_score, -jnp.inf)  # (B, L)
        # Stable argsort by -score keeps the lower original index on genuine
        # score ties (jnp.argsort is stable), bit-identical to the Python stable
        # descending sort — no float perturbation, no ordering flips.
        order = jnp.argsort(-el_score, axis=1, stable=True)  # (B, L)
        sel = order[:, :top_k].astype(jnp.int32)  # (B, top_k) line indices
        sel_score = jnp.take_along_axis(el_score, sel, axis=1)  # (B, top_k)
        valid = jnp.isfinite(sel_score)  # finite => genuine pick
        sel_idx = jnp.where(valid, sel, -1)
        return sel_idx, valid

    sel_list = [_per_element(e) for e in range(n_elements)]
    topk_index = jnp.stack([s[0] for s in sel_list], axis=1)  # (B, E, top_k)
    topk_valid = jnp.stack([s[1] for s in sel_list], axis=1)  # (B, E, top_k)

    # selected_mask: a line is selected iff it appears in any element's top-K.
    # Scatter (avoids jax.nn one-hot, which is banned by TID251).
    selected_mask = jnp.zeros((B, L), dtype=bool)
    flat_idx = topk_index.reshape(B, -1)  # (B, E*top_k)
    flat_valid = topk_valid.reshape(B, -1)
    safe_idx = jnp.where(flat_valid, flat_idx, 0)
    batch_ar = jnp.arange(B)[:, None]
    selected_mask = selected_mask.at[batch_ar, safe_idx].max(flat_valid)

    # Per-element diagnostics: energy spread + valid-line count over GATED lines
    # (line_selection.py:292-320 — warning strings, here numeric).
    if energy_ev is None:
        energy_ev = jnp.zeros_like(score)
    spread_ev, n_valid = _per_element_diagnostics(
        element_idx, gate, jnp.asarray(energy_ev, dtype=jnp.float64), n_elements
    )

    return {
        "selected_mask": selected_mask,
        "score": score,
        "snr": snr,
        "isolation": isolation,
        "topk_index": topk_index,
        "topk_valid": topk_valid,
        "spread_ev": spread_ev,
        "n_valid": n_valid,
    }


def _per_element_diagnostics(
    element_idx: jnp.ndarray,
    gate: jnp.ndarray,
    energy_ev: jnp.ndarray,
    n_elements: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-element energy spread + gated-line count (numeric diagnostics)."""
    energy_ev = jnp.asarray(energy_ev, dtype=jnp.float64)
    spreads = []
    counts = []
    for e in range(n_elements):
        in_el = (element_idx == e) & gate  # (B, L)
        cnt = jnp.sum(in_el.astype(jnp.float64), axis=1)  # (B,)
        e_masked_hi = jnp.where(in_el, energy_ev, -jnp.inf)
        e_masked_lo = jnp.where(in_el, energy_ev, jnp.inf)
        emax = jnp.max(e_masked_hi, axis=1)
        emin = jnp.min(e_masked_lo, axis=1)
        spread = jnp.where(cnt > 0.0, emax - emin, 0.0)
        spreads.append(spread)
        counts.append(cnt)
    return jnp.stack(spreads, axis=1), jnp.stack(counts, axis=1)


# ---------------------------------------------------------------------------
# §3 SB-graph / common-slope (Schur-equivalence: NO on-device lstsq)
# ---------------------------------------------------------------------------


def sb_graph_fit(
    x: jnp.ndarray,
    y: jnp.ndarray,
    stage: jnp.ndarray,
    mask: jnp.ndarray,
    ip: jnp.ndarray,
    ln_S: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Pooled Saha-Boltzmann graph fit with unit weights — WITHOUT ``lstsq``.

    The reference (``_fit_saha_boltzmann_graph`` / ``_solve_sb_graph_lstsq``,
    iterative.py:1454-1539, 2818-2886) solves a global ``(N_rows, 1+E)`` design
    ``A = [x | element-dummies]`` by ``np.linalg.lstsq`` with **unit weights**.
    With unit weights the normal matrix ``AᵀA`` is *arrow-shaped* (one dense
    slope column/row + a diagonal intercept block). Its Schur complement reduces
    to the centered common-slope estimator::

        m   = Σ_s Σ_l (x − x̄_s)(y − ȳ_s) / Σ_s Σ_l (x − x̄_s)²
        q_s = ȳ_s − m·x̄_s

    i.e. the SB-graph with unit weights *is* :func:`_common_slope_kernel` with
    ``w = mask`` on the ion-shifted coordinates. We therefore reuse the existing
    bit-exact JAX twins :func:`_saha_correct_kernel` (the ion shift) and
    :func:`_common_slope_kernel` — no general solve runs on device. (Confirmed by
    audit Q2, ``02-inversion-solver.md:378``.)

    Parameters
    ----------
    x : array ``(E, N)``
        Per-element upper-level energies ``E_k`` [eV].
    y : array ``(E, N)``
        Per-element Boltzmann-plot values ``ln(I λ / g A)``.
    stage : int array ``(E, N)``
        Ionization stage (1 neutral, 2 singly ionized).
    mask : bool array ``(E, N)``
        Validity mask. The reference screens ``A_ki<=0 | g_k<=0 | I<=0`` and
        non-finite ``x``/``y`` (iterative.py:2803-2811); callers fold those into
        this mask.
    ip : scalar array
        Ionization potential [eV] (ion-line x shift; reference uses ``ip*(z-1)``).
    ln_S : scalar array
        Saha log-offset (ion-line y shift; reference uses ``ln_S*(z-1)``).

    Returns
    -------
    dict
        ``slope`` (scalar), ``intercepts`` ``(E,)``, ``slope_variance`` (scalar),
        ``r_squared`` (scalar), ``n_rows`` (scalar), ``valid`` (scalar bool:
        global system over-determined, ``n_rows >= 1+E_active`` and ``>= 3``,
        mirroring iterative.py:2844).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    stage = jnp.asarray(stage, dtype=jnp.int32)
    mask_b = jnp.asarray(mask, dtype=bool)
    ip = jnp.asarray(ip, dtype=jnp.float64)
    ln_S = jnp.asarray(ln_S, dtype=jnp.float64)

    # Ion shift (reference: x += ip for z>1, y -= ln_S for z>1). The twin uses a
    # ``log_correction`` named arg = ln_S; both apply only to stage==2.
    x_corr, y_corr = _saha_correct_kernel(x, y, stage, ip, jnp.float64(0.0), ln_S)

    # Unit weights = mask (the SB-graph design point: per-line weighting recreates
    # Fe over-attribution, iterative.py:2783-2795).
    w = mask_b.astype(jnp.float64)
    res = _common_slope_kernel(x_corr, y_corr, w, mask_b)

    n_valid_per_el = res["n_valid_per_el"]  # (E,)
    n_rows = jnp.sum(n_valid_per_el)
    # Active elements = those contributing at least one row. The reference dummy
    # column exists per element present in ``elements``; under-determination is
    # ``n_rows < 1 + E`` (iterative.py:2844). We use active-element count so a
    # padded (all-masked) element slot does not spuriously inflate ``E``.
    e_active = jnp.sum(n_valid_per_el > 0.0)
    valid = (n_rows >= 3.0) & (n_rows >= (1.0 + e_active))

    return {
        "slope": res["slope"],
        "intercepts": res["intercepts"],
        "slope_variance": res["slope_variance"],
        "r_squared": res["r_squared"],
        "n_rows": n_rows,
        "valid": valid,
    }


def common_slope_fit(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    mask: jnp.ndarray,
    weight_cap: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Pooled common-slope Boltzmann plane (``_fit_common_boltzmann_plane``).

    Reuses :func:`_common_slope_kernel` verbatim after applying the per-element
    masked weight cap (``_cap_boltzmann_weights``, iterative.py:1398, 2911-2944)
    via :func:`cap_weights`. All arrays ``(E, N)``; add a batch axis with
    :func:`jax.vmap` to get ``(B, E, N)``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    mask_b = jnp.asarray(mask, dtype=bool)
    w_capped = cap_weights(w, mask_b, weight_cap)
    res = _common_slope_kernel(x, y, w_capped, mask_b)
    return res


# ---------------------------------------------------------------------------
# §3 robust single-element fitters (fixed-K reweighting scans)
# ---------------------------------------------------------------------------


def sigma_clip_fit(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    n_iters: int = SIGMA_CLIP_ITERS,
    outlier_sigma: float = OUTLIER_SIGMA,
) -> dict[str, jnp.ndarray]:
    """Fixed-K sigma-clip WLS via ``lax.scan`` of masked reweighting (§3).

    Mirrors ``BoltzmannFitter._fit_sigma_clip`` / ``_fit_sigma_clip_jax``
    (boltzmann.py:326-535): iterate weighted-LS, reject points whose residual
    exceeds ``outlier_sigma · std(residuals)`` (unweighted ``np.std`` over the
    current inliers), and refit. CPU early termination (``not np.any(bad)`` or
    ``std_res == 0``) is an idempotent fixed point — once no point is rejected,
    further iterations leave the inlier set unchanged — so the fixed-K scan
    reaches the identical result. The inner WLS is the bit-exact
    :func:`batched_boltzmann_fit`.

    Inputs are ``(B, N)`` batches (one Boltzmann plot per row). Returns a dict
    with ``slope``/``intercept``/``T_K``/``R_squared``/``inlier_mask`` ``(B,...)``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    mask0 = jnp.asarray(mask, dtype=bool)

    def _body(inlier, _):
        fit = batched_boltzmann_fit(x, y, w, inlier)
        m = fit.slope[:, None]
        c = fit.intercept[:, None]
        y_pred = c + m * x
        resid = y - y_pred
        # Unweighted std over current inliers (boltzmann.py:387 ``np.std``).
        inlier_f = inlier.astype(jnp.float64)
        n = jnp.sum(inlier_f, axis=1, keepdims=True)
        n_safe = jnp.maximum(n, 1.0)
        mean_r = jnp.sum(resid * inlier_f, axis=1, keepdims=True) / n_safe
        var_r = jnp.sum(((resid - mean_r) ** 2) * inlier_f, axis=1, keepdims=True) / n_safe
        std_r = jnp.sqrt(var_r)
        # std_res == 0 -> break (no rejection): keep the inlier set unchanged.
        bad = (jnp.abs(resid) > outlier_sigma * std_r) & inlier & (std_r > 0.0)
        # Never drop below 2 inliers (boltzmann.py:351 ``len(x) < 2`` break).
        new_inlier = inlier & (~bad)
        enough = jnp.sum(new_inlier.astype(jnp.float64), axis=1, keepdims=True) >= 2.0
        new_inlier = jnp.where(enough, new_inlier, inlier)
        return new_inlier, None

    inlier, _ = jax.lax.scan(_body, mask0, None, length=n_iters)
    fit = batched_boltzmann_fit(x, y, w, inlier)
    return {
        "slope": fit.slope,
        "intercept": fit.intercept,
        "T_K": fit.T_K,
        "R_squared": fit.R_squared,
        "sigma_slope": fit.sigma_slope,
        "inlier_mask": inlier,
        "n_valid": fit.n_valid,
    }


# ---------------------------------------------------------------------------
# §4 Closure (standard / matrix / oxide / ILR) + keystone gate
# ---------------------------------------------------------------------------


def closure_standard(
    intercepts: jnp.ndarray,
    partition: jnp.ndarray,
    mult: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Standard closure ``Σ C_s = 1`` (closure.py:537-593; lax twin :567-572).

    Missing-U elements are masked out (≡ the ``continue`` at closure.py:570-571).
    All arrays ``(E,)``; vmap for a batch axis. Returns concentrations ``(E,)``.
    """
    intercepts = jnp.asarray(intercepts, dtype=jnp.float64)
    partition = jnp.asarray(partition, dtype=jnp.float64)
    mult = jnp.asarray(mult, dtype=jnp.float64)
    mask_f = jnp.asarray(mask, dtype=jnp.float64)
    rel = mult * partition * jnp.exp(intercepts) * mask_f
    total = jnp.sum(rel)
    return jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)


def closure_matrix(
    intercepts: jnp.ndarray,
    partition: jnp.ndarray,
    mult: jnp.ndarray,
    mask: jnp.ndarray,
    matrix_index: int,
    matrix_fraction: jnp.ndarray,
) -> jnp.ndarray:
    """Matrix closure: one element pinned to ``matrix_fraction`` (closure.py:595-660).

    The reference falls back to standard when the matrix element is missing; that
    resolves statically at array build (the caller passes ``matrix_index < 0`` to
    request the fallback, mirroring iterative.py:579-587).
    """
    intercepts = jnp.asarray(intercepts, dtype=jnp.float64)
    partition = jnp.asarray(partition, dtype=jnp.float64)
    mult = jnp.asarray(mult, dtype=jnp.float64)
    mask_f = jnp.asarray(mask, dtype=jnp.float64)
    matrix_fraction = jnp.asarray(matrix_fraction, dtype=jnp.float64)
    if matrix_index < 0:
        return closure_standard(intercepts, partition, mult, mask)
    rel = mult * partition * jnp.exp(intercepts) * mask_f
    rel_m = rel[matrix_index]
    F = rel_m / matrix_fraction
    return jnp.where(F > 0.0, rel / jnp.where(F > 0.0, F, 1.0), 0.0)


def closure_oxide(
    intercepts: jnp.ndarray,
    partition: jnp.ndarray,
    mult: jnp.ndarray,
    mask: jnp.ndarray,
    oxide_factors: jnp.ndarray,
) -> jnp.ndarray:
    """Oxide closure: ``Σ (C_s · factor_s) = 1`` (closure.py:662-727; twin :603-610).

    ``oxide_factors`` is the ``(E,)`` per-element stoichiometry vector (default
    1.0 = metal, closure.py:712). Returns ELEMENTAL fractions normalized so the
    oxides sum to 1.
    """
    intercepts = jnp.asarray(intercepts, dtype=jnp.float64)
    partition = jnp.asarray(partition, dtype=jnp.float64)
    mult = jnp.asarray(mult, dtype=jnp.float64)
    mask_f = jnp.asarray(mask, dtype=jnp.float64)
    oxide_factors = jnp.asarray(oxide_factors, dtype=jnp.float64)
    rel = mult * partition * jnp.exp(intercepts) * mask_f
    total_oxide = jnp.sum(rel * oxide_factors)
    return jnp.where(total_oxide > 0.0, rel / jnp.where(total_oxide > 0.0, total_oxide, 1.0), 0.0)


def closure_ilr(
    intercepts: jnp.ndarray,
    partition: jnp.ndarray,
    mult: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """ILR closure as the standard closure with the 1e-10 clip (closure.py:729-801).

    The reference ILR is an *identity round-trip*: it normalizes
    ``U_s exp(q_s)`` to the simplex, maps to ILR coords via the Helmert basis,
    and inverts — recovering the same simplex up to the ``LOGRATIO_CLIP_FLOOR``
    clip and the ``sorted(intercepts)`` element ordering (which does not change
    per-element values). On strictly-positive compositions this is *exactly*
    standard closure (audit 02-inversion-solver.md:364-365). We therefore
    compute the standard normalization and apply the same clip floor, giving
    bit-identical results without a ``pure_callback``.
    """
    intercepts = jnp.asarray(intercepts, dtype=jnp.float64)
    partition = jnp.asarray(partition, dtype=jnp.float64)
    mult = jnp.asarray(mult, dtype=jnp.float64)
    mask_f = jnp.asarray(mask, dtype=jnp.float64)
    rel = mult * partition * jnp.exp(intercepts) * mask_f
    total = jnp.sum(rel)
    simplex = jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)
    # ILR round-trip = exp(clr) / sum, clr = log(clip(s)) - mean(log(clip(s))).
    # On the simplex (sum=1) this reduces to clip-then-renormalize.
    clipped = jnp.where(mask_f > 0.0, jnp.clip(simplex, LOGRATIO_CLIP_FLOOR, None), 0.0)
    denom = jnp.sum(clipped)
    return jnp.where(denom > 0.0, clipped / jnp.where(denom > 0.0, denom, 1.0), 0.0)


def keystone_degenerate(
    concentrations: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    threshold: float = 0.8,
    min_elements: int = 4,
) -> jnp.ndarray:
    """Keystone-collapse gate (``ClosureEquation.validate_degeneracy``, closure.py:495-535).

    Returns ``True`` when at least ``min_elements`` valid elements are present
    and any single concentration exceeds ``threshold``. Closes lax seam (iv) of
    ADR-0004 §4 row 10. The iterative solver passes ``min_elements=4``.
    """
    concentrations = jnp.asarray(concentrations, dtype=jnp.float64)
    mask_b = jnp.asarray(mask, dtype=bool)
    n_present = jnp.sum(mask_b.astype(jnp.float64), axis=-1)
    eff_min = float(max(int(min_elements), 2))
    max_c = jnp.max(jnp.where(mask_b, concentrations, -jnp.inf), axis=-1)
    return (n_present >= eff_min) & (max_c > threshold)


# ---------------------------------------------------------------------------
# Stage entry point (J0 skeleton signature, now implemented)
# ---------------------------------------------------------------------------


def boltzmann_fit(
    line_intensities: Any,
    line_index: Any,
    line_mask: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Per-species Boltzmann/SB-graph fit for identified lines (J4 entry point).

    Thin orchestrator over the kernels in this module. The padded ``(E, N)``
    arrays are assembled by the host adapter (``jitpipe.parity.fit_parity``) from
    the snapshot; this function runs the pooled SB-graph fit (the production
    geological estimator) and returns ``(slope, intercepts, r_squared)`` plus the
    validity flag. The per-stage host wiring lands at J7 integration; the kernels
    above are the parity surface.
    """
    raise NotImplementedError(
        "boltzmann_fit host wiring lands at J7 integration; the J4 parity surface "
        "is the kernels (select_lines, sb_graph_fit, common_slope_fit, "
        "sigma_clip_fit, closure_*, keystone_degenerate, masked_median)."
    )

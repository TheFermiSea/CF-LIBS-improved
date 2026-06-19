"""Stage 1 — baseline + noise, fixed-shape fp64 JAX kernels (ADR-0004 §4 row 1).

Port of ``cflibs/inversion/preprocess/preprocessing.py`` (the frozen oracle)
into jit/vmap-clean kernels with **no data-dependent shapes**: window sizes and
iteration counts are host-computed static ints, failures surface as quality
flags rather than exceptions, and every reduction over a candidate set runs at a
padded fixed length with a validity mask.

Baselines
---------
* :func:`snip_baseline` — direct LLS-transform SNIP, fixed iteration count.
  The canonical jit baseline (pure vector ops). rtol 1e-12 vs the reference.
* :func:`median_baseline` / :func:`percentile_baseline` — sliding-window gather
  ``(N, W)`` + ``jnp.sort`` + gather at the median/percentile rank, row-chunked
  via :func:`jax.lax.map` to cap the silva2022 worst-case live memory. Exact
  (sort-based), rtol 1e-12.
* :func:`als_baseline` — fixed-``max_iters`` IRLS with an early-converged freeze
  (``where`` on a converged flag) and a banded LDLᵀ solve of the pentadiagonal
  SPD system ``(W + λDᵀD) z = W y`` as a ``lax.scan`` recurrence. ALS tolerance
  ``max|Δz| ≤ 1e-6·scale(y)``.

Noise
-----
* :func:`estimate_noise` — masked 3-iteration sigma-clipped MAD with the
  reference's two early-break conditions frozen via ``where``. Exact (rtol 1e-12).

Conventions
-----------
``W`` (median/percentile window), ``num_iterations`` (SNIP), ``max_iters`` (ALS)
and ``pad_width`` (SNIP) are **static** Python ints because the wavelength axis
is constant per compile bucket (ADR-0004 §5.2). They are never traced. fp64 is
mandatory (ADR-0004 §5.3): the LLS exp/log round-trip and the λ=1e6 ALS Gram are
the precision-sensitive spots.

No SQLite, no host imports (import-hygiene test); arrays in, arrays out.

Consumption status (2026-06)
----------------------------
These J1 kernels are parity-tested (``tests/jitpipe/test_parity_j1.py``) but are
NOT yet composed into the live pipeline: ``run_one`` /
``host.run_front_end_ondevice`` apply the host-side response multiply on-device
and do not currently run a jitpipe baseline/noise stage. Per ADR-0004's staged
plan a parity-tested-but-unwired kernel is an intentional intermediate state, not
dead code — wiring this stage into the front-end is the residual integration gap.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# The reference applies these LLS/MAD constants; mirror them exactly so the
# arithmetic is identical (rtol 1e-12 contract).
_MAD_TO_SIGMA = 1.4826  # MAD -> Gaussian sigma scale (preprocessing.py:343)
_SNIP_DEFAULT_ITERS = 40  # preprocessing.py:98
_SIGMA_CLIP_ITERS = 3  # preprocessing.py:340
_SIGMA_CLIP_K = 3.0  # preprocessing.py:346
_SIGMA_CLIP_MIN_KEEP = 10  # preprocessing.py:347
_NOISE_FLOOR = 1e-10  # preprocessing.py:344,357


# --------------------------------------------------------------------------- #
# SNIP baseline                                                               #
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("num_iterations", "order"))
def snip_baseline(
    intensity: jax.Array,
    num_iterations: int = _SNIP_DEFAULT_ITERS,
    order: int = 2,
) -> jax.Array:
    """SNIP baseline (LLS transform + fixed clipping), exact reference port.

    Mirrors ``estimate_baseline_snip`` (``preprocessing.py:95``): LLS forward
    transform, ``pad_width = min(num_iterations, N-1)`` reflect pad, ``p`` from
    ``pad_width`` down to 1 clipping ``v = min(v, (v[i-p]+v[i+p])/2)``, then the
    LLS inverse. ``pad_width`` is host-static (N constant per bucket), so the
    clip loop is a static Python ``for`` — no traced shapes.

    Parameters
    ----------
    intensity : jax.Array
        Raw intensities, shape ``(N,)``. fp64.
    num_iterations : int
        Static clip iteration count (default 40).
    order : int
        Static LLS order (``2`` = log-log-sqrt, the reference default).

    Returns
    -------
    jax.Array
        Baseline, shape ``(N,)``.
    """
    n = intensity.shape[0]
    y = jnp.maximum(intensity.astype(jnp.float64), 0.0)

    if order >= 1:
        y = jnp.log(jnp.sqrt(y + 1.0) + 1.0)
    if order >= 2:
        y = jnp.log(y + 1.0)

    if n <= 1:
        # Reference returns intensity.copy() for N < 2; an LLS round-trip on a
        # length-1 array is degenerate, so short-circuit (parity edge).
        return intensity.astype(jnp.float64)

    pad_width = min(num_iterations, n - 1)
    v = jnp.pad(y, pad_width, mode="reflect")

    # p from pad_width down to 1; pad_width is a static int (bucket-fixed N).
    for p in range(pad_width, 0, -1):
        avg = (v[: v.shape[0] - 2 * p] + v[2 * p :]) / 2.0
        mid = v[p : v.shape[0] - p]
        v = v.at[p : v.shape[0] - p].set(jnp.minimum(mid, avg))

    v = v[pad_width : pad_width + n]

    if order >= 2:
        v = jnp.exp(v) - 1.0
    if order >= 1:
        v = (jnp.exp(v) - 1.0) ** 2 - 1.0
        v = jnp.maximum(v, 0.0)
    return v


# --------------------------------------------------------------------------- #
# Sliding-window order-statistic baselines (median / percentile)             #
# --------------------------------------------------------------------------- #
def _window_pts(n: int, spacing: float, window_nm: float) -> int:
    """Resolve the odd, clamped sliding-window width in pixels (host-static).

    Exact port of the window resolution shared by ``estimate_baseline`` and
    ``estimate_baseline_percentile`` (``preprocessing.py:48-56,308-316``). Pure
    Python — N and ``spacing`` are bucket-constant, so the result is a static
    int that keys the kernel shape (never traced).
    """
    if not (spacing > 0) or spacing != spacing or spacing in (float("inf"),):
        spacing = 1.0
    window_pts = max(3, int(window_nm / spacing))
    max_window = n if n % 2 == 1 else n - 1
    window_pts = min(window_pts, max(3, max_window))
    if window_pts % 2 == 0:
        window_pts += 1
    return window_pts


def _sliding_windows(y: jax.Array, window: int) -> jax.Array:
    """Build the ``(N, window)`` reflect-padded sliding-window matrix.

    Reproduces ``scipy.ndimage``'s default ``mode='reflect'`` boundary for an
    odd, origin-centred window: pad ``half = window // 2`` on each side with
    ``mode='reflect'`` (scipy "reflect" == NumPy "symmetric": edge sample is
    duplicated), then gather contiguous ``window``-length slices.
    """
    half = window // 2
    yp = jnp.pad(y, half, mode="symmetric")
    idx = jnp.arange(y.shape[0])[:, None] + jnp.arange(window)[None, :]
    return yp[idx]


@partial(jax.jit, static_argnames=("window", "row_chunk"))
def median_baseline(intensity: jax.Array, window: int, row_chunk: int = 4096) -> jax.Array:
    """Median-filter baseline via sliding-window sort, exact reference port.

    Equivalent to ``scipy.ndimage.median_filter(intensity, size=window)`` for an
    odd window (``estimate_baseline``, ``preprocessing.py:57``): gather the
    ``(N, window)`` window matrix, ``jnp.sort`` each row, take the middle rank.
    Row-chunked via :func:`jax.lax.map` so the window matrix never fully
    materialises (caps the silva2022 53,717×1,001 worst case).

    Parameters
    ----------
    intensity : jax.Array
        Raw intensities, shape ``(N,)``. fp64.
    window : int
        Static odd window width in pixels (use :func:`_window_pts`).
    row_chunk : int
        Static row-chunk size for ``lax.map`` (memory knob, not a parity knob).
    """
    y = intensity.astype(jnp.float64)
    mid = window // 2
    win = _sliding_windows(y, window)
    return jax.lax.map(lambda row: jnp.sort(row)[mid], win, batch_size=row_chunk)


@partial(jax.jit, static_argnames=("window", "percentile", "row_chunk"))
def percentile_baseline(
    intensity: jax.Array, window: int, percentile: float = 10.0, row_chunk: int = 4096
) -> jax.Array:
    """Percentile-filter baseline via sliding-window sort, exact reference port.

    Equivalent to ``scipy.ndimage.percentile_filter(intensity, percentile, size)``
    (``estimate_baseline_percentile``, ``preprocessing.py:317``). scipy's
    ``_rank_filter`` (``scipy/ndimage/_filters.py:1960-1969``) selects the
    ``rank``-th order statistic where ``rank = window - 1`` when
    ``percentile == 100`` and ``rank = int(window * percentile / 100.0)``
    otherwise (floor, no interpolation).

    Parameters
    ----------
    intensity : jax.Array
        Raw intensities, shape ``(N,)``. fp64.
    window : int
        Static odd window width in pixels.
    percentile : float
        Percentile in ``[0, 100]`` (default 10, the reference default).
    row_chunk : int
        Static ``lax.map`` chunk size.
    """
    y = intensity.astype(jnp.float64)
    pct = percentile + 100.0 if percentile < 0.0 else percentile
    rank = window - 1 if pct == 100.0 else int(window * pct / 100.0)
    rank = min(max(rank, 0), window - 1)
    win = _sliding_windows(y, window)
    return jax.lax.map(lambda row: jnp.sort(row)[rank], win, batch_size=row_chunk)


# --------------------------------------------------------------------------- #
# ALS baseline (fixed IRLS + banded LDLT pentadiagonal solve)                 #
# --------------------------------------------------------------------------- #
def _solve_pentadiagonal_ldlt(
    d0: jax.Array, d1: jax.Array, d2: jax.Array, rhs: jax.Array
) -> jax.Array:
    r"""Solve a symmetric pentadiagonal SPD system ``A z = rhs`` via banded LDLᵀ.

    ``A`` is symmetric with diagonals ``d0`` (main, len N), ``d1`` (first
    super/sub, len N-1) and ``d2`` (second super/sub, len N-2). The factorisation
    ``A = L D Lᵀ`` with unit-lower-banded ``L`` (bandwidth 2) is computed by a
    forward ``lax.scan`` recurrence; forward and back substitution are two more
    scans. Exact and deterministic — the ADR-mandated replacement for
    ``scipy.sparse.linalg.spsolve`` (matrix-free CG rejected: λ=1e6 is too
    ill-conditioned for a fixed small iteration count).
    """
    n = d0.shape[0]

    # Factorisation: carry the previous two D values and the two L sub-band
    # entries needed to form each new row. l1[i] = L[i, i-1], l2[i] = L[i, i-2].
    def fac_step(carry, i):
        dprev1, dprev2, l1_prev, l2_prev = carry
        # L[i, i-2] = d2[i-2] / D[i-2]
        l2_i = jnp.where(i >= 2, d2[jnp.clip(i - 2, 0, n - 1)] / dprev2, 0.0)
        # L[i, i-1] = (d1[i-1] - L[i,i-2]*L[i-1,i-2]*D[i-2]) / D[i-1]
        l1_i = jnp.where(
            i >= 1,
            (d1[jnp.clip(i - 1, 0, n - 1)] - l2_i * l1_prev * dprev2) / dprev1,
            0.0,
        )
        # D[i] = d0[i] - L[i,i-1]^2 D[i-1] - L[i,i-2]^2 D[i-2]
        d_i = (
            d0[i]
            - jnp.where(i >= 1, l1_i * l1_i * dprev1, 0.0)
            - jnp.where(i >= 2, l2_i * l2_i * dprev2, 0.0)
        )
        new_carry = (d_i, dprev1, l1_i, l2_i)
        return new_carry, (d_i, l1_i, l2_i)

    init = (jnp.array(1.0, jnp.float64),) * 4
    _, (d_diag, l1, l2) = jax.lax.scan(fac_step, init, jnp.arange(n))

    # Forward solve  L y = rhs  (unit lower, bandwidth 2).
    def fwd_step(carry, i):
        yprev1, yprev2 = carry
        yi = (
            rhs[i] - jnp.where(i >= 1, l1[i] * yprev1, 0.0) - jnp.where(i >= 2, l2[i] * yprev2, 0.0)
        )
        return (yi, yprev1), yi

    _, yvec = jax.lax.scan(fwd_step, (jnp.array(0.0, jnp.float64),) * 2, jnp.arange(n))

    # Diagonal solve  D w = y.
    wvec = yvec / d_diag

    # Back solve  Lᵀ z = w  (unit upper, bandwidth 2), reverse order.
    def bwd_step(carry, i):
        znext1, znext2 = carry
        # z[i] = w[i] - L[i+1,i] z[i+1] - L[i+2,i] z[i+2]
        zi = (
            wvec[i]
            - jnp.where(i + 1 <= n - 1, l1[jnp.clip(i + 1, 0, n - 1)] * znext1, 0.0)
            - jnp.where(i + 2 <= n - 1, l2[jnp.clip(i + 2, 0, n - 1)] * znext2, 0.0)
        )
        return (zi, znext1), zi

    _, zrev = jax.lax.scan(
        bwd_step, (jnp.array(0.0, jnp.float64),) * 2, jnp.arange(n), reverse=True
    )
    return zrev


def _als_penalty_bands(n: int, lam: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pentadiagonal bands of ``λ DᵀD`` for the second-difference operator ``D``.

    ``D`` is the ``(n-2, n)`` second-difference matrix (``estimate_baseline_als``,
    ``preprocessing.py:260-261``). ``DᵀD`` is symmetric pentadiagonal with the
    well-known stencil ``[1, -4, 6, -4, 1]`` interior, tapering at the four
    boundary rows. Returns ``(diag0, diag1, diag2)``.
    """
    d0 = jnp.full((n,), 6.0, jnp.float64)
    d0 = d0.at[0].set(1.0).at[1].set(5.0).at[n - 2].set(5.0).at[n - 1].set(1.0)
    d1 = jnp.full((n - 1,), -4.0, jnp.float64)
    d1 = d1.at[0].set(-2.0).at[n - 2].set(-2.0)
    d2 = jnp.ones((n - 2,), jnp.float64)
    return lam * d0, lam * d1, lam * d2


@partial(jax.jit, static_argnames=("max_iters",))
def als_baseline(
    intensity: jax.Array,
    lam: float = 1e6,
    p: float = 0.01,
    max_iters: int = 20,
    tol: float = 1e-4,
) -> jax.Array:
    """ALS baseline (fixed-iteration IRLS, banded LDLᵀ solve).

    Port of ``estimate_baseline_als`` (``preprocessing.py:190``). Runs exactly
    ``max_iters`` IRLS steps; the reference's early ``break`` on weight
    convergence is reproduced by a converged-flag freeze (``where``) so the
    returned baseline matches the early-converged reference result. Each inner
    solve of ``(W + λDᵀD) z = W y`` uses :func:`_solve_pentadiagonal_ldlt`.

    Parameters
    ----------
    intensity : jax.Array
        Raw intensities, shape ``(N,)``. fp64.
    lam : float
        Smoothness penalty (default 1e6, the reference default).
    p : float
        Asymmetry parameter (default 0.01).
    max_iters : int
        Static IRLS iteration cap (default 20).
    tol : float
        Convergence tolerance on the weight vector (default 1e-4).

    Returns
    -------
    jax.Array
        Baseline, shape ``(N,)``.
    """
    y = intensity.astype(jnp.float64)
    n = y.shape[0]
    pen0, pen1, pen2 = _als_penalty_bands(n, lam)

    def body(carry, _):
        w, z, done = carry
        # Inner solve with current weights (W + λDᵀD) z_new = w y.
        d0 = w + pen0
        z_new = _solve_pentadiagonal_ldlt(d0, pen1, pen2, w * y)
        w_new = jnp.where(y > z_new, p, 1.0 - p)
        # Reference convergence: ||w_new - w|| / max(||w||, 1e-10) < tol.
        rel = jnp.linalg.norm(w_new - w) / jnp.maximum(jnp.linalg.norm(w), 1e-10)
        converged = rel < tol
        # On the converged step the reference sets z=z_new then breaks, so it
        # keeps z_new but does NOT update w. Freeze everything once done.
        new_done = done | converged
        z_out = jnp.where(done, z, z_new)
        w_out = jnp.where(done | converged, w, w_new)
        return (w_out, z_out, new_done), None

    w0 = jnp.ones((n,), jnp.float64)
    (_, z_final, _), _ = jax.lax.scan(body, (w0, y, jnp.array(False)), None, length=max_iters)
    return z_final


# --------------------------------------------------------------------------- #
# Noise                                                                       #
# --------------------------------------------------------------------------- #
def _masked_median(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Median of the masked-in entries of ``values`` (fixed-shape, exact).

    Masked-out entries are pushed to ``+inf`` so a full sort leaves the valid
    samples in their original order at the front; the median is then the average
    of the two central valid ranks (NumPy ``np.median`` convention: lower-mid
    and upper-mid averaged for even counts). Exact vs ``np.median`` on the kept
    subset.
    """
    n = values.shape[0]
    count = jnp.sum(mask)
    keyed = jnp.where(mask, values, jnp.inf)
    srt = jnp.sort(keyed)
    # NumPy median over `count` valid elements: indices (count-1)//2 and count//2.
    lo = jnp.clip((count - 1) // 2, 0, n - 1)
    hi = jnp.clip(count // 2, 0, n - 1)
    med = 0.5 * (srt[lo] + srt[hi])
    return jnp.where(count > 0, med, 0.0)


@jax.jit
def estimate_noise(intensity: jax.Array, baseline: jax.Array) -> jax.Array:
    """Sigma-clipped MAD noise, masked fixed-shape port of ``estimate_noise``.

    Port of ``preprocessing.py:320``: 3 iterations of 3-σ MAD clipping with the
    two reference early-breaks (``sigma < 1e-10`` and ``kept < 10``) frozen via a
    ``done`` mask so the iteration that would have ``break``-ed in the reference
    leaves the keep-mask unchanged. The shrinking-residual-array trick is
    replaced by an ``(N,)`` keep-mask. Returns the noise floor exactly as the
    reference (``max(1e-10, percentile(|resid|, 95) * 1e-6)``) when the final
    MAD underflows.

    Parameters
    ----------
    intensity : jax.Array
        Raw intensities, shape ``(N,)``. fp64.
    baseline : jax.Array
        Baseline estimate, shape ``(N,)``. fp64.

    Returns
    -------
    jax.Array
        Scalar noise sigma. fp64.
    """
    residuals = intensity.astype(jnp.float64) - baseline.astype(jnp.float64)
    n = residuals.shape[0]
    keep0 = jnp.ones((n,), bool)

    def body(carry, _):
        keep, done = carry
        med = _masked_median(residuals, keep)
        absdev = jnp.abs(residuals - med)
        mad = _masked_median(absdev, keep)
        sigma = mad * _MAD_TO_SIGMA
        # Reference: `if sigma < 1e-10: break` (no further clipping).
        break_sigma = sigma < _NOISE_FLOOR
        new_keep = keep & (absdev < _SIGMA_CLIP_K * sigma)
        # Reference: `if np.sum(mask) < 10: break` (keep PREVIOUS residuals).
        break_count = jnp.sum(new_keep) < _SIGMA_CLIP_MIN_KEEP
        # Once done, freeze. The sigma break stops before applying the mask;
        # the count break also stops before adopting new_keep.
        will_break = break_sigma | break_count
        keep_out = jnp.where(done | will_break, keep, new_keep)
        done_out = done | will_break
        return (keep_out, done_out), None

    (keep_final, _), _ = jax.lax.scan(
        body, (keep0, jnp.array(False)), None, length=_SIGMA_CLIP_ITERS
    )

    med = _masked_median(residuals, keep_final)
    mad = _masked_median(jnp.abs(residuals - med), keep_final)
    noise = mad * _MAD_TO_SIGMA

    # Reference floor on underflow (preprocessing.py:357-358).
    floor = jnp.maximum(_NOISE_FLOOR, jnp.nanpercentile(jnp.abs(residuals), 95.0) * 1e-6)
    return jnp.where(noise < _NOISE_FLOOR, floor, noise)


@jax.jit
def robust_normalize(intensity: jax.Array, percentile: float = 95.0) -> jax.Array:
    """Percentile-scale normalisation, exact port of ``robust_normalize``.

    ``preprocessing.py:727``: divide by ``percentile(intensity, q)`` when that
    scale exceeds ``1e-10``, else return a copy.
    """
    y = intensity.astype(jnp.float64)
    scale = jnp.percentile(y, percentile)
    return jnp.where(scale > _NOISE_FLOOR, y / scale, y)

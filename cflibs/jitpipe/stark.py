"""Stage 6b — Stark n_e diagnostic as a fixed-shape, vmap-clean JAX kernel (J6).

ADR-0004 §4 row 8, §5.1.1, risk R9. This module ports the Stark-width
electron-density measurement (the frozen oracle
``cflibs/inversion/physics/stark_ne.py::measure_stark_ne``) and its per-iteration
solver coupling (``cflibs/inversion/solve/iterative.py::_estimate_ne_from_stark_multi``)
into a single jittable, vmappable kernel.

Why a redesign was needed
-------------------------
The reference does, *inside a data-dependent Python loop over candidates*:

* a **SQLite query** for same-species multiplet neighbours
  (``_has_strong_multiplet_neighbour``);
* a **host** ``scipy.optimize.least_squares`` trust-region fit with an
  adaptive, data-dependent ``nfev`` (``_fit_lorentz_fwhm``);
* a Python ``list.sort`` + ``break``-at-``max_lines``; half-max crossing loops;
  irregular per-candidate window sizes.

None of these survive ``jit``/``vmap`` — they are exactly the
data-dependent-shape pathology ADR-0004 §1.1 is built to eliminate. This module
replaces every breaker with a fixed-shape equivalent:

* **Fixed windows** ``(C, W)``: raw samples gathered around the recentred peak
  (recentre = masked argmax — jittable); irregular physical widths become
  per-candidate *validity masks* over the fixed ``W`` samples. No resampling
  (window-extraction parity is risk R9 — see ``extract_windows``).
* **Vmapped fixed-iteration Levenberg–Marquardt** (``K=20``) on the same
  pinned-Gaussian Voigt 5-parameter least-squares basin as the reference, with
  ``5x5`` damped normal equations solved by :func:`jax.numpy.linalg.solve`.
  Feasibility (``area > 0``, ``gamma > 0``) is by exp-transform rather than box
  bounds. Per-candidate convergence is a *mask*, not a ``break``. The contract
  is on the **solution**, not the optimiser path (ADR §4 / R9.c): both fitters
  minimise the identical smooth basin, so they agree to rtol 1e-3 when both
  converge, and disagreement on near-degenerate profiles is caught by the same
  rel-RMSE / resolvability gates rather than by chasing trf-vs-LM parity.
* The Voigt model uses the project's jittable Faddeeva kernel
  (``cflibs.radiation.profiles._voigt_profile_kernel_jax``, Weideman 1994), which
  on the fp64 CPU backend matches scipy's ``wofz`` to ~15 digits — so the two
  least-squares basins are identical up to round-off.
* **Gates -> masks**; ``max_lines`` ranking -> ``top_k``; median / MAD /
  cohort-trim -> masked median (two fixed passes); rejection reasons -> uint8
  quality codes.

The multiplet-blend gate is *atomic-data-only* apart from a weak Boltzmann
factor, so it is precomputed against the full catalogue at snapshot-build time
or evaluated on-device against the snapshot line table — never a DB query in the
loop (see :func:`multiplet_blend_mask`).

fp64 is mandatory throughout (ADR-0004 §5.3; ``radiation/profiles.py`` is the
authority that the Voigt fit must never run fp32). No SQLite, no host imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp

from cflibs.radiation.profiles import _voigt_profile_kernel_jax

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot

# ---------------------------------------------------------------------------
# Physical / model constants (mirror the frozen reference exactly).
# ---------------------------------------------------------------------------

#: Reference density the stored ``stark_w`` is tabulated at (cm^-3).
#: ``cflibs/radiation/stark.py::REF_NE``.
REF_NE: float = 1.0e17
#: Reference temperature the stored ``stark_w`` is tabulated at (K).
#: ``cflibs/radiation/stark.py::REF_T_K``.
REF_T_K: float = 10000.0

#: Olivero–Longbothum (1977) Voigt-FWHM coefficients
#: (``cflibs/radiation/stark.py::deconvolve_stark_fwhm`` ``a``/``b``).
_OL_A: float = 0.5346
_OL_B: float = 0.2166

#: FWHM / sigma for a Gaussian (``stark_ne.py::_FWHM_PER_SIGMA`` = 2.3548...).
_FWHM_PER_SIGMA: float = 2.0 * 1.1774100225154747  # 2*sqrt(2 ln 2)

#: T floor used by the reference width-law inversion (``estimate_ne_from_stark``
#: clamps ``T_eff = max(T_K, 1000.0)``).
_T_FLOOR_K: float = 1000.0

#: Loose physical plausibility band for a per-line n_e (cm^-3).
#: ``stark_ne.py::NE_SANITY_MIN_CM3`` / ``NE_SANITY_MAX_CM3``.
NE_SANITY_MIN_CM3: float = 1.0e14
NE_SANITY_MAX_CM3: float = 1.0e20

#: Resolvability floor: a Lorentzian narrower than this fraction of the pinned
#: Gaussian FWHM is unresolved (``stark_ne.py:564``).
_RESOLVABILITY_FRAC: float = 0.05

#: Default Stark temperature exponent when the catalogue value is missing
#: (``stark_ne.py:525`` -> ``alpha if alpha is not None else 0.5``;
#: ``estimate_ne_from_stark`` uses the same 0.5 default).
DEFAULT_STARK_ALPHA: float = 0.5

#: uint8 quality / rejection codes (the device-side replacement for the
#: reference's ``rejected`` reason-string dict). 0 == accepted.
QC_OK: int = 0
QC_PAD: int = 1  # padding slot, never a real candidate
QC_FIT_NONFINITE: int = 2  # LM produced a non-finite area/gamma
QC_POOR_FIT: int = 3  # rel-RMSE above the gate
QC_UNRESOLVED: int = 4  # lorentz_fwhm below the resolvability floor
QC_IMPLAUSIBLE_NE: int = 5  # n_e outside the sanity band
QC_COHORT_OUTLIER: int = 6  # > 1 decade from the cohort log10 median

#: Default fixed iteration count for the LM solve (ADR §4: "K~20").
LM_ITERS: int = 20
#: LM damping schedule constants (Marquardt up/down factors + bounds).
_LM_LAMBDA0: float = 1.0e-3
_LM_LAMBDA_UP: float = 3.0
_LM_LAMBDA_DOWN: float = 0.3
_LM_LAMBDA_MIN: float = 1.0e-12
_LM_LAMBDA_MAX: float = 1.0e9


# ---------------------------------------------------------------------------
# Result container (a NamedTuple is a pytree, so it flows through jit/vmap).
# ---------------------------------------------------------------------------


class StarkFitResult(NamedTuple):
    """Per-candidate LM fit output (all leaves are device arrays, shape ``(C,)``).

    Attributes
    ----------
    area : array
        Fitted Voigt area (always positive — exp-transformed).
    center : array
        Fitted line centre, nm.
    gamma : array
        Fitted Lorentzian HWHM, nm (always positive — exp-transformed).
    c0, c1 : array
        Fitted linear-baseline intercept / slope.
    lorentz_fwhm : array
        ``2 * gamma`` — the Lorentzian (Stark) FWHM, nm.
    rel_rmse : array
        Residual RMS / (max-min of the window samples) — the reference's
        ``rel_rmse`` (``stark_ne.py:311-312``).
    converged : array of bool
        Whether the LM step norm fell below tolerance within ``LM_ITERS``.
    """

    area: Any
    center: Any
    gamma: Any
    c0: Any
    c1: Any
    lorentz_fwhm: Any
    rel_rmse: Any
    converged: Any


class StarkNeResult(NamedTuple):
    """Full-stage output (the device-side analogue of ``StarkNeDiagnostics``).

    Attributes
    ----------
    fit : StarkFitResult
        Per-candidate LM fit, shape ``(C,)`` leaves.
    ne_per_line : array, shape (C,)
        Per-candidate electron density, cm^-3 (NaN where the candidate is
        rejected or padded).
    quality : array of uint8, shape (C,)
        Per-candidate quality code (see ``QC_*``). 0 == accepted.
    valid : array of bool, shape (C,)
        ``quality == QC_OK`` — the accepted, cohort-trimmed diagnostic set.
    ne_median : array, scalar
        Robust combined n_e over the valid set (NaN when none valid).
    ne_scatter : array, scalar
        1.4826 * MAD of the valid n_e (0 for <2 lines).
    n_lines : array, scalar int
        Number of valid lines that fed the median.
    """

    fit: StarkFitResult
    ne_per_line: Any
    quality: Any
    valid: Any
    ne_median: Any
    ne_scatter: Any
    n_lines: Any


# ---------------------------------------------------------------------------
# Piece 1 — pure-algebra width-law inversion + robust combine (AC5).
# Parity vs iterative.py:1250-1280 at rtol 1e-12.
# ---------------------------------------------------------------------------


def deconvolve_stark_fwhm(
    measured_fwhm_nm: Any,
    instrument_fwhm_nm: Any,
    doppler_fwhm_nm: Any,
) -> Any:
    """Jittable Olivero–Longbothum Lorentzian-FWHM recovery.

    Bit-faithful port of ``cflibs.radiation.stark.deconvolve_stark_fwhm``:
    recover the Lorentzian (Stark) FWHM ``f_L`` from the measured Voigt FWHM
    ``f_V`` and the combined Gaussian FWHM ``f_G = hypot(f_inst, f_dopp)`` by
    inverting ``f_V = a f_L + sqrt(b f_L^2 + f_G^2)`` (``a=0.5346, b=0.2166``).

    Branches in the reference become ``jnp.where`` so the result is identical on
    every code path:

    * ``f_V <= 0`` or ``f_V <= f_G`` -> 0 (Gaussian already exceeds the width);
    * ``f_G <= 0`` -> ``f_V`` (no Gaussian to remove);
    * negative discriminant -> plain quadrature subtraction
      ``sqrt(max(f_V^2 - f_G^2, 0))``;
    * otherwise the physical (smaller, positive) quadratic root, falling back to
      the larger root when the smaller is non-positive.
    """
    f_v = jnp.asarray(measured_fwhm_nm)
    f_g = jnp.hypot(jnp.asarray(instrument_fwhm_nm), jnp.asarray(doppler_fwhm_nm))

    qa = _OL_A * _OL_A - _OL_B  # 0.0692 > 0
    qb = -2.0 * _OL_A * f_v
    qc = f_v * f_v - f_g * f_g
    disc = qb * qb - 4.0 * qa * qc
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))

    root_small = (-qb - sqrt_disc) / (2.0 * qa)
    root_large = (-qb + sqrt_disc) / (2.0 * qa)
    quad_root = jnp.where(root_small <= 0.0, root_large, root_small)
    quad_root = jnp.maximum(quad_root, 0.0)

    quadrature = jnp.sqrt(jnp.maximum(f_v * f_v - f_g * f_g, 0.0))
    f_l = jnp.where(disc < 0.0, quadrature, quad_root)

    # f_G == 0 -> return f_V verbatim (reference short-circuit).
    f_l = jnp.where(f_g <= 0.0, f_v, f_l)
    # Gaussian already accounts for the full width, or non-positive measurement.
    f_l = jnp.where(f_v <= f_g, 0.0, f_l)
    f_l = jnp.where(f_v <= 0.0, 0.0, f_l)
    return f_l


def estimate_ne_from_stark(
    measured_fwhm_nm: Any,
    T_K: Any,
    stark_w_ref: Any,
    stark_alpha: Any,
    instrument_fwhm_nm: Any = 0.0,
    doppler_fwhm_nm: Any = 0.0,
) -> Any:
    """Jittable port of ``cflibs.radiation.stark.estimate_ne_from_stark``.

    ``n_e = REF_NE * (w_stark / w_ref) * (max(T, 1000) / REF_T_K)^alpha`` where
    ``w_stark`` is the deconvolved Lorentzian FWHM. Returns NaN (the device
    sentinel for the reference's ``None``) when ``w_ref <= 0``, ``w_stark <= 0``,
    or the result is non-finite/non-positive.
    """
    w_ref = jnp.asarray(stark_w_ref)
    alpha = jnp.asarray(stark_alpha)
    w_stark = deconvolve_stark_fwhm(measured_fwhm_nm, instrument_fwhm_nm, doppler_fwhm_nm)
    t_eff = jnp.maximum(jnp.asarray(T_K), _T_FLOOR_K)
    n_e = REF_NE * (w_stark / w_ref) * (t_eff / REF_T_K) ** alpha
    bad = (w_ref <= 0.0) | (w_stark <= 0.0) | ~jnp.isfinite(n_e) | (n_e <= 0.0)
    return jnp.where(bad, jnp.nan, n_e)


def _masked_median(values: Any, mask: Any) -> Any:
    """Median of ``values`` over the ``True`` entries of ``mask`` (fixed shape).

    Uses the numpy ``'linear'`` (== ``np.median``) interpolation convention:
    odd count -> middle element; even count -> mean of the two central elements.
    Implemented with a fixed-shape sort that pushes masked entries to ``+inf``
    so they fall past the valid prefix; the valid count selects the central
    rank(s) with linear interpolation.
    """
    values = jnp.asarray(values, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    n = jnp.sum(mask).astype(jnp.float64)
    big = jnp.asarray(jnp.inf, dtype=jnp.float64)
    keyed = jnp.where(mask, values, big)
    s = jnp.sort(keyed)  # valid values first (ascending), padding (+inf) last

    # Central ranks for an n-element ascending list, numpy 'linear' convention.
    lo = jnp.floor((n - 1.0) / 2.0)
    hi = jnp.ceil((n - 1.0) / 2.0)
    lo_i = jnp.clip(lo, 0.0, jnp.asarray(values.shape[0] - 1, dtype=jnp.float64)).astype(jnp.int32)
    hi_i = jnp.clip(hi, 0.0, jnp.asarray(values.shape[0] - 1, dtype=jnp.float64)).astype(jnp.int32)
    med = 0.5 * (s[lo_i] + s[hi_i])
    return jnp.where(n > 0, med, jnp.nan)


def combine_ne(ne_per_line: Any, valid: Any) -> tuple[Any, Any, Any]:
    """Robust median + 1.4826*MAD combine over the valid per-line densities.

    Bit-faithful port of ``_estimate_ne_from_stark_multi`` (iterative.py
    1273-1280): ``ne_median = median(valid)``;
    ``scatter = 1.4826*MAD`` for ``n >= 2`` (falling back to ``std`` when the MAD
    is exactly 0), ``0`` for a single line, NaN/0 when none valid.

    Returns ``(ne_median, scatter, n_lines)``.
    """
    ne = jnp.asarray(ne_per_line, dtype=jnp.float64)
    valid = jnp.asarray(valid, dtype=bool) & jnp.isfinite(ne)
    n = jnp.sum(valid)
    ne_median = _masked_median(ne, valid)

    abs_dev = jnp.abs(ne - ne_median)
    mad = _masked_median(abs_dev, valid)
    # Population std over the valid set (numpy default ddof=0).
    mean = jnp.where(n > 0, jnp.sum(jnp.where(valid, ne, 0.0)) / jnp.maximum(n, 1), jnp.nan)
    var = jnp.where(
        n > 0,
        jnp.sum(jnp.where(valid, (ne - mean) ** 2, 0.0)) / jnp.maximum(n, 1),
        jnp.nan,
    )
    std = jnp.sqrt(var)
    scatter_ge2 = jnp.where(mad > 0, 1.4826 * mad, std)
    scatter = jnp.where(n >= 2, scatter_ge2, 0.0)
    scatter = jnp.where(n > 0, scatter, 0.0)
    return ne_median, scatter, n


def stark_ne_from_widths(
    lorentz_fwhm_nm: Any,
    stark_w_ref_nm: Any,
    stark_alpha: Any,
    valid: Any,
    T_K: Any,
    *,
    instrument_fwhm_nm: Any = 0.0,
    doppler_fwhm_nm: Any = 0.0,
) -> tuple[Any, Any, Any, Any]:
    """Solver-coupling kernel: per-line re-inversion + robust combine (AC5).

    Drops directly into J7's solve body for per-iteration n_e re-inversion at
    the current temperature. The diagnostics carry already-deconvolved
    Lorentzian widths (instrument/Doppler are 0), so this is pure algebra over a
    fixed ``(D,)`` array — matching ``_estimate_ne_from_stark_multi`` at rtol
    1e-12.

    Returns ``(ne_per_line, ne_median, scatter, n_lines)``.
    """
    ne_per_line = estimate_ne_from_stark(
        lorentz_fwhm_nm,
        T_K,
        stark_w_ref_nm,
        stark_alpha,
        instrument_fwhm_nm=instrument_fwhm_nm,
        doppler_fwhm_nm=doppler_fwhm_nm,
    )
    keep = jnp.asarray(valid, dtype=bool) & jnp.isfinite(ne_per_line)
    ne_median, scatter, n = combine_ne(ne_per_line, keep)
    return ne_per_line, ne_median, scatter, n


# ---------------------------------------------------------------------------
# Piece 2 — vmapped fixed-iteration Levenberg–Marquardt Voigt fit.
# Parity vs the scipy trf fit (_fit_lorentz_fwhm) at rtol 1e-3 on the solution.
# ---------------------------------------------------------------------------


def _voigt_model(params: Any, wl: Any, center0: Any, sigma_g: Any) -> Any:
    """Pinned-Gaussian Voigt + linear baseline, parameterised as the reference.

    ``params = (log_area, x0, log_gamma, c0, c1)`` (exp-transform on area/gamma
    for feasibility). The model mirrors ``stark_ne.py::_fit_lorentz_fwhm._model``:
    ``area * voigt(wl; x0, sigma_g, gamma) + c0 + c1 * (wl - center0)``, where
    ``center0`` is the recentred window centre (the reference's ``center_nm``).
    """
    log_area, x0, log_gamma, c0, c1 = params
    area = jnp.exp(log_area)
    gamma = jnp.exp(log_gamma)
    prof = area * _voigt_profile_kernel_jax(wl - x0, sigma_g, gamma)
    return prof + c0 + c1 * (wl - center0)


def _fit_one_lm(
    wl: Any,
    inten: Any,
    mask: Any,
    center0: Any,
    sigma_g: Any,
    window_half_nm: Any,
    init: Any,
    n_iters: int,
) -> tuple[Any, Any]:
    """One fixed-iteration Levenberg–Marquardt solve over a single window.

    Minimises ``sum_i mask_i * (model_i - inten_i)^2`` over the 5 transformed
    parameters via damped Gauss–Newton (``(J^T J + lambda diag(J^T J)) dp =
    J^T r``). Masked (padding) samples contribute zero residual and zero
    Jacobian, so the fixed ``W`` length never changes the fit. The damping
    follows a Marquardt up/down schedule keyed on whether the trial step reduced
    the masked SSE.

    Returns ``(params, converged)``.
    """
    w = jnp.asarray(mask, dtype=jnp.float64)

    # Box constraints mirroring the reference scipy bounds (stark_ne.py:299-300):
    # x0 in [center0 - half/2, center0 + half/2]; gamma in (0, 10*half];
    # area > 0. The reparameterization is exp-transform on area/gamma, so the
    # upper gamma bound becomes a clamp on log_gamma; this keeps LM inside the
    # same physical basin and prevents blow-up on near-degenerate profiles.
    half = jnp.maximum(window_half_nm, 1e-6)
    x0_lo = center0 - 0.5 * half
    x0_hi = center0 + 0.5 * half
    log_gamma_hi = jnp.log(10.0 * half)
    log_gamma_lo = jnp.asarray(jnp.log(1e-12))
    log_area_hi = jnp.asarray(jnp.log(1e30))
    log_area_lo = jnp.asarray(jnp.log(1e-30))

    def clamp(p: Any) -> Any:
        log_area, x0, log_gamma, c0, c1 = p
        log_area = jnp.clip(log_area, log_area_lo, log_area_hi)
        x0 = jnp.clip(x0, x0_lo, x0_hi)
        log_gamma = jnp.clip(log_gamma, log_gamma_lo, log_gamma_hi)
        return jnp.stack([log_area, x0, log_gamma, c0, c1])

    def residual(p: Any) -> Any:
        return (_voigt_model(p, wl, center0, sigma_g) - inten) * w

    def sse(p: Any) -> Any:
        r = residual(p)
        return jnp.sum(r * r)

    jac_fn = jax.jacobian(residual)

    def body(carry, _):
        p, lam, prev_sse, conv = carry
        r = residual(p)
        jmat = jac_fn(p)  # (W, 5)
        jtj = jmat.T @ jmat  # (5, 5)
        jtr = jmat.T @ r  # (5,)
        diag = jnp.diag(jnp.diag(jtj))
        a = jtj + lam * diag + 1e-12 * jnp.eye(5)
        step = jnp.linalg.solve(a, -jtr)
        p_new = clamp(p + step)
        step = p_new - p  # effective (post-clamp) step for the convergence test

        new_sse = sse(p_new)
        improved = new_sse < prev_sse
        p_out = jnp.where(improved, p_new, p)
        lam_out = jnp.where(
            improved,
            jnp.maximum(lam * _LM_LAMBDA_DOWN, _LM_LAMBDA_MIN),
            jnp.minimum(lam * _LM_LAMBDA_UP, _LM_LAMBDA_MAX),
        )
        sse_out = jnp.where(improved, new_sse, prev_sse)
        step_norm = jnp.sqrt(jnp.sum(step * step))
        # Converged once an accepted step is negligibly small.
        conv_out = conv | (improved & (step_norm < 1e-10))
        return (p_out, lam_out, sse_out, conv_out), None

    init = jnp.asarray(init, dtype=jnp.float64)
    carry0 = (init, jnp.asarray(_LM_LAMBDA0), sse(init), jnp.asarray(False))
    (p_final, _, _, conv), _ = jax.lax.scan(body, carry0, xs=None, length=n_iters)
    return p_final, conv


def _init_params(wl: Any, inten: Any, mask: Any, center0: Any, gaussian_fwhm_nm: Any) -> Any:
    """Initial transformed parameter vector mirroring the reference x0.

    Reference (``stark_ne.py:280-298``): ``baseline0 = min(median(left edge),
    median(right edge))``; ``peak0 = max(inten) - baseline0``;
    ``area0 = peak0 * gaussian_fwhm``; ``gamma0 = 0.1 * gaussian_fwhm``;
    ``x0_center = center_nm``. We mask padding before the reductions and clamp
    ``area0``/``peak0`` positive so the log-transform is finite even for a
    degenerate window (the QC mask rejects those downstream).
    """
    w = jnp.asarray(mask, dtype=bool)
    big = jnp.asarray(jnp.inf, dtype=jnp.float64)
    small = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    inten_valid_max = jnp.max(jnp.where(w, inten, small))

    # Edge medians over the valid prefix: the reference takes the first/last
    # ``edge = max(3, len/8)`` *contiguous* samples of the masked window. With a
    # left-packed window (extract_windows guarantees this) the valid samples are
    # the leading ``n`` entries, so the same contiguous-edge slices apply.
    n = jnp.sum(w.astype(jnp.int32))
    idx = jnp.arange(wl.shape[0])
    edge = jnp.maximum(3, (n // 8))
    left_edge = w & (idx < edge)
    right_edge = w & (idx >= (n - edge)) & (idx < n)
    base_left = _masked_median(inten, left_edge)
    base_right = _masked_median(inten, right_edge)
    baseline0 = jnp.minimum(base_left, base_right)

    peak0 = jnp.maximum(inten_valid_max - baseline0, 1e-30)
    area0 = peak0 * gaussian_fwhm_nm
    gamma0 = 0.1 * gaussian_fwhm_nm
    log_area0 = jnp.log(jnp.maximum(area0, 1e-30))
    log_gamma0 = jnp.log(jnp.maximum(gamma0, 1e-30))
    del big
    return jnp.stack([log_area0, center0, log_gamma0, baseline0, jnp.asarray(0.0)])


def fit_lorentz_fwhm_lm(
    wl: Any,
    inten: Any,
    mask: Any,
    center0: Any,
    gaussian_fwhm_nm: Any,
    *,
    n_iters: int = LM_ITERS,
) -> StarkFitResult:
    """Vmappable pinned-Gaussian Voigt LM fit over fixed ``(C, W)`` windows.

    Parameters
    ----------
    wl, inten : array, shape (C, W)
        Per-candidate raw window samples (wavelength / intensity). Padding
        samples are flagged by ``mask`` and contribute nothing.
    mask : array of bool, shape (C, W)
        Validity mask — ``True`` for real samples, left-packed.
    center0 : array, shape (C,)
        Recentred window centre per candidate, nm (the reference ``center_nm``).
    gaussian_fwhm_nm : array, shape (C,)
        Pinned Gaussian FWHM ``hypot(instrument, doppler)`` per candidate.
    n_iters : int
        Fixed LM iteration count (static; default :data:`LM_ITERS`).

    Returns
    -------
    StarkFitResult
        Per-candidate fit with ``(C,)`` leaves. ``lorentz_fwhm = 2 * gamma`` and
        ``rel_rmse = sqrt(mean(masked residual^2)) / (max-min of valid samples)``
        exactly as ``stark_ne.py:311-313``.
    """
    sigma_g = jnp.maximum(jnp.asarray(gaussian_fwhm_nm) / _FWHM_PER_SIGMA, 1e-6)

    def one(wl_i, inten_i, mask_i, center_i, sg_i, fwhm_i):
        init = _init_params(wl_i, inten_i, mask_i, center_i, fwhm_i)
        # Physical half-window = max |wl - center| over the valid samples; this
        # is the reference ``window_nm`` the scipy box bounds key on.
        window_half = jnp.max(jnp.where(mask_i, jnp.abs(wl_i - center_i), 0.0))
        params, conv = _fit_one_lm(
            wl_i, inten_i, mask_i, center_i, sg_i, window_half, init, n_iters
        )
        log_area, x0, log_gamma, c0, c1 = params
        area = jnp.exp(log_area)
        gamma = jnp.exp(log_gamma)

        w = jnp.asarray(mask_i, dtype=jnp.float64)
        n = jnp.maximum(jnp.sum(w), 1.0)
        model = _voigt_model(params, wl_i, center_i, sg_i)
        resid = (model - inten_i) * w
        rms = jnp.sqrt(jnp.sum(resid * resid) / n)
        big = jnp.asarray(jnp.inf, dtype=jnp.float64)
        small = jnp.asarray(-jnp.inf, dtype=jnp.float64)
        vmax = jnp.max(jnp.where(mask_i, inten_i, small))
        vmin = jnp.min(jnp.where(mask_i, inten_i, big))
        scale = vmax - vmin
        rel_rmse = jnp.where(scale > 0, rms / scale, 1.0)
        finite = jnp.isfinite(area) & jnp.isfinite(gamma) & (area > 0)
        return StarkFitResult(
            area=area,
            center=x0,
            gamma=gamma,
            c0=c0,
            c1=c1,
            lorentz_fwhm=2.0 * gamma,
            rel_rmse=rel_rmse,
            converged=conv & finite,
        )

    return jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0))(
        wl, inten, mask, center0, sigma_g, jnp.asarray(gaussian_fwhm_nm)
    )


# ---------------------------------------------------------------------------
# Piece 3 — window extraction (raw-sample gather; risk R9).
# ---------------------------------------------------------------------------


def extract_windows(
    wavelength: Any,
    intensity: Any,
    center_idx: Any,
    half_widths: Any,
    W: int,
) -> tuple[Any, Any, Any]:
    """Gather fixed ``(C, W)`` raw-sample windows around each candidate.

    *Never resamples* — windows are byte-identical slices of the source
    spectrum (risk R9). For candidate ``c`` the window is the ``W`` contiguous
    samples ``[center_idx_c - W//2 : center_idx_c - W//2 + W]``; out-of-bounds
    indices are clamped and flagged invalid in the returned mask. The
    ``half_widths`` (nm) define the physical fit window: samples beyond
    ``|wl - wl[center_idx]| > half_width`` are masked out, so the irregular
    per-candidate physical width becomes a mask over the fixed ``W`` length.

    The validity mask is **left-packed** is NOT guaranteed by this raw gather
    (the physical-width mask can punch holes); :func:`recenter_and_pack` is the
    helper that produces a left-packed window when the init/edge-median logic
    needs one. For LM the holes are harmless (masked residuals are zero).

    Parameters
    ----------
    wavelength, intensity : array, shape (N,)
        Source spectrum axes.
    center_idx : array of int, shape (C,)
        Index of the recentred peak per candidate.
    half_widths : array, shape (C,)
        Physical half-window per candidate, nm.
    W : int
        Fixed window length (static).

    Returns
    -------
    wl_win, inten_win : array, shape (C, W)
        Gathered raw samples.
    mask : array of bool, shape (C, W)
        ``True`` where the sample is in-bounds AND within the physical window.
    """
    wavelength = jnp.asarray(wavelength)
    intensity = jnp.asarray(intensity)
    n = wavelength.shape[0]
    center_idx = jnp.asarray(center_idx)
    half = W // 2
    offsets = jnp.arange(W) - half  # (W,)
    raw_idx = center_idx[:, None] + offsets[None, :]  # (C, W)
    in_bounds = (raw_idx >= 0) & (raw_idx < n)
    gather_idx = jnp.clip(raw_idx, 0, n - 1)
    wl_win = wavelength[gather_idx]
    inten_win = intensity[gather_idx]
    center_wl = wavelength[jnp.clip(center_idx, 0, n - 1)][:, None]
    within = jnp.abs(wl_win - center_wl) <= half_widths[:, None]
    mask = in_bounds & within
    return wl_win, inten_win, mask


def recenter_idx(
    wavelength: Any,
    intensity: Any,
    center_nm: Any,
    search_nm: Any,
) -> Any:
    """Index of the local intensity maximum near ``center_nm`` (masked argmax).

    Jittable port of ``stark_ne.py::_recenter_on_local_peak``: restrict to
    ``|wl - center_nm| <= search_nm`` and take the argmax there. When fewer than
    3 samples fall in the window the reference returns ``center_nm`` unchanged;
    here we return the index of the spectrum sample nearest ``center_nm`` so the
    downstream window is still anchored sensibly (the candidate is gate-rejected
    via its mask if the window is too sparse).
    """
    wavelength = jnp.asarray(wavelength)
    intensity = jnp.asarray(intensity)
    center_nm = jnp.asarray(center_nm)
    search_nm = jnp.asarray(search_nm)
    in_win = jnp.abs(wavelength - center_nm) <= search_nm
    enough = jnp.sum(in_win) >= 3
    small = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    keyed = jnp.where(in_win, jnp.asarray(intensity, dtype=jnp.float64), small)
    peak_idx = jnp.argmax(keyed)
    nearest_idx = jnp.argmin(jnp.abs(wavelength - center_nm))
    return jnp.where(enough, peak_idx, nearest_idx)


# ---------------------------------------------------------------------------
# Piece 3b — gate masks (QC codes) + cohort trim + full-stage driver.
# ---------------------------------------------------------------------------


def apply_quality_gates(
    fit: StarkFitResult,
    ne_per_line: Any,
    gaussian_fwhm_nm: Any,
    candidate_mask: Any,
    *,
    max_fit_rel_rmse: float,
) -> Any:
    """Reduce the reference's gate ladder to a per-candidate uint8 QC code.

    Mirrors, in order, ``stark_ne.py``:

    * ``fit is None`` / non-finite area-gamma -> :data:`QC_FIT_NONFINITE`
      (the reference's ``fit_failed``);
    * ``rel_rmse > max_fit_rel_rmse`` -> :data:`QC_POOR_FIT` (``poor_fit``);
    * ``lorentz_fwhm < 0.05 * gauss`` -> :data:`QC_UNRESOLVED` (``unresolved``);
    * n_e outside ``[1e14, 1e20]`` or NaN -> :data:`QC_IMPLAUSIBLE_NE`.

    Padding / non-candidate slots get :data:`QC_PAD`. The first failing gate in
    that order wins (matching the reference's ``continue`` chain). Cohort-trim
    (:data:`QC_COHORT_OUTLIER`) is applied separately, after the median.
    """
    is_cand = jnp.asarray(candidate_mask, dtype=bool)
    gauss = jnp.asarray(gaussian_fwhm_nm)

    finite_ok = fit.converged & jnp.isfinite(fit.area) & jnp.isfinite(fit.gamma) & (fit.area > 0)
    rmse_ok = fit.rel_rmse <= max_fit_rel_rmse
    resolved = fit.lorentz_fwhm >= (_RESOLVABILITY_FRAC * gauss)
    ne_ok = jnp.isfinite(ne_per_line) & (ne_per_line >= NE_SANITY_MIN_CM3)
    ne_ok = ne_ok & (ne_per_line <= NE_SANITY_MAX_CM3)

    code = jnp.full(is_cand.shape, QC_OK, dtype=jnp.uint8)
    code = jnp.where(~finite_ok, jnp.uint8(QC_FIT_NONFINITE), code)
    code = jnp.where(finite_ok & ~rmse_ok, jnp.uint8(QC_POOR_FIT), code)
    code = jnp.where(finite_ok & rmse_ok & ~resolved, jnp.uint8(QC_UNRESOLVED), code)
    code = jnp.where(finite_ok & rmse_ok & resolved & ~ne_ok, jnp.uint8(QC_IMPLAUSIBLE_NE), code)
    code = jnp.where(is_cand, code, jnp.uint8(QC_PAD))
    return code


def cohort_trim_mask(ne_per_line: Any, accepted: Any, max_log10_dev: float = 1.0) -> Any:
    """Log-space cohort outlier trim (``stark_ne.py::_trim_cohort_outliers``).

    For ``n >= 3`` accepted lines, reject any whose ``log10(n_e)`` is more than
    ``max_log10_dev`` decades from the accepted-cohort log10 median. For ``n <
    3`` nothing is trimmed (reference short-circuit). Returns the boolean
    *survivor* mask (``True`` == kept).
    """
    accepted = jnp.asarray(accepted, dtype=bool) & jnp.isfinite(ne_per_line)
    n = jnp.sum(accepted)
    log_ne = jnp.log10(jnp.where(accepted, ne_per_line, 1.0))
    med = _masked_median(log_ne, accepted)
    within = jnp.abs(log_ne - med) <= max_log10_dev
    survivor = jnp.where(n >= 3, accepted & within, accepted)
    return survivor


def measure_stark_ne_jit(
    wl_win: Any,
    inten_win: Any,
    mask: Any,
    center0: Any,
    gaussian_fwhm_nm: Any,
    stark_w_ref_nm: Any,
    stark_alpha: Any,
    candidate_mask: Any,
    T_K: Any,
    *,
    max_fit_rel_rmse: float = 0.25,
    n_iters: int = LM_ITERS,
) -> StarkNeResult:
    """Full fixed-shape Stark-n_e stage over a padded candidate set ``(C, W)``.

    This is the device-side analogue of the reference ``measure_stark_ne`` from
    the per-candidate Voigt fit onward (candidate *selection* — SNR / isolation
    / source-class / multiplet-blend ranking — is the host/snapshot side; see
    module docstring and :func:`multiplet_blend_mask`). Given the gathered
    windows it:

    1. LM-fits each window (:func:`fit_lorentz_fwhm_lm`);
    2. inverts the width law per line (:func:`estimate_ne_from_stark`);
    3. applies the gate ladder -> QC codes (:func:`apply_quality_gates`);
    4. cohort-trims (:func:`cohort_trim_mask`) and combines
       (median + 1.4826*MAD, :func:`combine_ne`).

    Parameters
    ----------
    wl_win, inten_win, mask : array, shape (C, W)
        Gathered windows + validity mask (from :func:`extract_windows`).
    center0 : array, shape (C,)
        Recentred window centre, nm.
    gaussian_fwhm_nm : array, shape (C,)
        Pinned Gaussian FWHM ``hypot(instrument, doppler)``, nm.
    stark_w_ref_nm, stark_alpha : array, shape (C,)
        Per-candidate reference Stark width / temperature exponent (snapshot).
    candidate_mask : array of bool, shape (C,)
        ``True`` for real (selected) candidates, ``False`` for padding.
    T_K : scalar
        Current plasma temperature, K.
    max_fit_rel_rmse : float
        rel-RMSE acceptance gate (reference default 0.25).
    n_iters : int
        Fixed LM iteration count.

    Returns
    -------
    StarkNeResult
    """
    fit = fit_lorentz_fwhm_lm(wl_win, inten_win, mask, center0, gaussian_fwhm_nm, n_iters=n_iters)
    # The pinned-Gaussian fit already removed instrument+Doppler, so the
    # width-law inversion takes instrument/Doppler = 0 (matches the reference's
    # StarkDiagnosticLine construction at stark_ne.py:601-604).
    ne_per_line = estimate_ne_from_stark(
        fit.lorentz_fwhm,
        T_K,
        stark_w_ref_nm,
        stark_alpha,
        instrument_fwhm_nm=0.0,
        doppler_fwhm_nm=0.0,
    )
    quality = apply_quality_gates(
        fit, ne_per_line, gaussian_fwhm_nm, candidate_mask, max_fit_rel_rmse=max_fit_rel_rmse
    )
    accepted = quality == QC_OK
    survivor = cohort_trim_mask(ne_per_line, accepted)
    # Tag trimmed-but-accepted lines with the cohort-outlier QC code.
    quality = jnp.where(accepted & ~survivor, jnp.uint8(QC_COHORT_OUTLIER), quality)
    valid = quality == QC_OK
    ne_median, scatter, n = combine_ne(ne_per_line, valid)
    return StarkNeResult(
        fit=fit,
        ne_per_line=ne_per_line,
        quality=quality,
        valid=valid,
        ne_median=ne_median,
        ne_scatter=scatter,
        n_lines=n,
    )


# ---------------------------------------------------------------------------
# Multiplet-blend gate — atomic-data-only, evaluated on-device against the
# snapshot line table (NO DB query in the loop). ADR §3 bullet 1.
# ---------------------------------------------------------------------------


def multiplet_blend_mask(
    cand_line_index: Any,
    cand_window_nm: Any,
    snapshot: "PipelineSnapshot",
    T_eV: Any,
    *,
    strength_fraction: float = 0.25,
) -> Any:
    """Same-species multiplet-blend rejection as a masked reduction (no SQLite).

    Device-side replacement for ``stark_ne.py::_has_strong_multiplet_neighbour``
    (which queries SQLite *inside the candidate loop*). For each candidate line
    (indexed into the snapshot's per-line tables) we evaluate every snapshot line
    of the *same species* within the candidate's fit window and reject when any
    companion's thermal ``g_k * A_ki * exp(-E_k/T_eV)`` strength exceeds
    ``strength_fraction`` of the candidate's own — exactly the reference
    predicate, but as a ``(C, N_lines)`` masked reduction at the current T.

    The candidate itself (``|Δλ| < 0.05 nm``) is excluded, mirroring
    ``stark_ne.py:363``.

    Parameters
    ----------
    cand_line_index : array of int, shape (C,)
        Index of each candidate into the snapshot per-line tables.
    cand_window_nm : array, shape (C,)
        Fit-window half-width per candidate, nm.
    snapshot : PipelineSnapshot
        Atomic snapshot (per-line wavelength / species / g_k / A_ki / E_k).
    T_eV : scalar
        Current temperature, eV.
    strength_fraction : float
        Reference threshold (0.25).

    Returns
    -------
    array of bool, shape (C,)
        ``True`` where a strong same-species companion blends the candidate
        (i.e. the candidate should be *rejected*).
    """
    wl_all = jnp.asarray(snapshot.line_wavelength_nm)
    sp_all = jnp.asarray(snapshot.line_species_index)
    gk_all = jnp.asarray(snapshot.line_g_k)
    aki_all = jnp.asarray(snapshot.line_A_ki)
    ek_all = jnp.asarray(snapshot.line_E_k_ev)

    cand_idx = jnp.asarray(cand_line_index)
    cand_wl = wl_all[cand_idx]
    cand_sp = sp_all[cand_idx]
    cand_gk = gk_all[cand_idx]
    cand_aki = aki_all[cand_idx]
    cand_ek = ek_all[cand_idx]

    t_eff = jnp.maximum(jnp.asarray(T_eV), 0.1)

    def strength(g, a, e):
        return g * a * jnp.exp(-e / t_eff)

    own = strength(cand_gk, cand_aki, cand_ek)  # (C,)
    comp = strength(gk_all, aki_all, ek_all)  # (N,)

    same_sp = sp_all[None, :] == cand_sp[:, None]  # (C, N)
    in_win = jnp.abs(wl_all[None, :] - cand_wl[:, None]) <= cand_window_nm[:, None]
    not_self = jnp.abs(wl_all[None, :] - cand_wl[:, None]) >= 0.05
    strong = comp[None, :] > (strength_fraction * own[:, None])
    hit = same_sp & in_win & not_self & strong
    has_blend = jnp.any(hit, axis=1)
    # Reference returns False when own strength is non-positive.
    return jnp.where(own > 0, has_blend, jnp.asarray(False))


# ---------------------------------------------------------------------------
# Stage entry point (J0 stub signature) — thin convenience wrapper.
# ---------------------------------------------------------------------------


def stark_electron_density(
    line_widths_nm: Any,
    line_index: Any,
    temperature_eV: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Solver-coupling entry point: n_e from already-measured Stark widths (J6).

    Convenience wrapper around :func:`stark_ne_from_widths` that pulls the
    per-line reference Stark width / exponent from the ``snapshot`` line table
    by ``line_index`` and combines into a single robust n_e at the supplied
    temperature. This is the J7-facing per-iteration coupling (closes lax seam
    (iii); see spec §1). The window-extraction + LM measurement path is
    :func:`measure_stark_ne_jit`.

    Parameters
    ----------
    line_widths_nm : array, shape (D,)
        Measured Lorentzian (Stark) FWHM per diagnostic line, nm. Instrument /
        Doppler are assumed already removed (pinned-Gaussian fit convention).
    line_index : array of int, shape (D,)
        Index of each diagnostic into the snapshot per-line tables.
    temperature_eV : scalar
        Plasma temperature, eV.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (``line_stark_w`` / ``line_stark_alpha``).
    params : PipelineParams
        Traced knobs (unused here; kept for stage-signature uniformity).
    static : StaticConfig
        Static config (unused here; kept for stage-signature uniformity).

    Returns
    -------
    array, scalar
        Robust electron-density estimate, cm^-3 (NaN when no line is usable).
    """
    del params, static
    from cflibs.core.constants import EV_TO_K

    idx = jnp.asarray(line_index)
    w_ref = jnp.asarray(snapshot.line_stark_w)[idx]
    alpha = jnp.asarray(snapshot.line_stark_alpha)[idx]
    alpha = jnp.where(alpha == 0.0, DEFAULT_STARK_ALPHA, alpha)
    valid = w_ref > 0.0
    t_k = jnp.asarray(temperature_eV) * EV_TO_K
    _, ne_median, _, _ = stark_ne_from_widths(jnp.asarray(line_widths_nm), w_ref, alpha, valid, t_k)
    return ne_median

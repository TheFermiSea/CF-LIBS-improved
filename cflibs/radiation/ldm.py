"""
Line Distribution Method (LDM/DIT) broadening kernel for CF-LIBS.

This module implements the Radis-style line-distribution-method specialized
to the Gaussian-dominant CF-LIBS regime, following van den Bekerom & Pannier
(2021), JQSRT 261 107476.

The algorithm projects N lines onto a small log-σ grid (~16-32 layers) using
bilinear weights, then performs one FFT convolution per σ layer. This
collapses the dominant ``O(N_lines · N_λ)`` per-line broadcasting cost (the
``(N_λ, N_lines)`` profile matrix produced by ``voigt_spectrum_jax`` or the
inline Humlicek W4 block in ``manifold.generator``) to ``O(N_σ · N_λ · log
N_λ)``. The manifold sweep over ``(T, n_e, composition)`` reuses the same
σ-grid layout across every grid point so the dominant scatter+convolution
work is the inner-loop variable.

API surface
-----------

* :func:`ldm_broaden` — pure JAX kernel, vmap-friendly. The wavelength grid
  must be uniformly spaced (CF-LIBS convention).
* :func:`build_sigma_grid` — host-side helper that constructs the log-σ
  layer grid from observed Gaussian widths.

Notes
-----

The 2-D Voigt extension (additional ``γ_L`` axis) is deferred to a follow-up
bead — the kernel signature accepts ``gamma_grid`` / ``line_gammas`` keyword
arguments and raises ``NotImplementedError`` when they are provided to keep
the API stable.

References
----------

van den Bekerom, D., & Pannier, E. (2021). A discrete integral transform for
rapid spectral synthesis. *JQSRT*, 261, 107476.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cflibs.core.jax_runtime import (
    HAS_JAX,
    JaxMemoryPolicy,
    jax,
    jax_default_real_dtype,
    jax_policy,
    jit_if_available,
    jnp,
)
from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.ldm")


__all__ = [
    "DEFAULT_N_SIGMA",
    "build_sigma_grid",
    "ldm_broaden",
]


# Default number of σ-grid layers. Spec §3 allows ``N_σ ∈ [16, 32]`` with 24
# as the recommended default (Radis ships dx_σ ≈ 0.2; we use ~0.15).
DEFAULT_N_SIGMA: int = 24
_MIN_N_SIGMA: int = 4


def build_sigma_grid(
    line_sigmas: np.ndarray | jnp.ndarray,
    *,
    n_sigma: int = DEFAULT_N_SIGMA,
    sigma_min_factor: float = 0.5,
    sigma_max_factor: float = 2.0,
    floor_nm: float = 1e-6,
) -> np.ndarray:
    """Construct a log-spaced σ layer grid from observed per-line widths.

    Spec §3: ``sigma_min = sigma_min_factor · min_i σ_i``,
    ``sigma_max = sigma_max_factor · max_i σ_i``. The output is always a
    numpy array (host-side; static under jit since manifold bounds are
    config-fixed). Callers that need a ``jnp`` array should ``jnp.asarray``
    the result.

    Parameters
    ----------
    line_sigmas : array-like, shape (N_lines,)
        Per-line Gaussian σ values in nm. Must be strictly positive.
    n_sigma : int, optional
        Number of σ layers. Defaults to :data:`DEFAULT_N_SIGMA` (24).
        Must satisfy ``n_sigma >= 4``.
    sigma_min_factor, sigma_max_factor : float, optional
        Bracket factors applied to ``min`` / ``max`` of ``line_sigmas``.
    floor_nm : float, optional
        Lower clamp applied to ``sigma_min`` to keep ``log(σ)`` finite when
        the catalog contains zero-width lines.

    Returns
    -------
    sigma_grid : ndarray, shape (n_sigma,)
        Log-spaced σ grid in nm.

    Raises
    ------
    ValueError
        If ``n_sigma`` is below 4 or ``line_sigmas`` is empty.
    """
    if n_sigma < _MIN_N_SIGMA:
        raise ValueError(f"n_sigma must be >= {_MIN_N_SIGMA}; got {n_sigma}")

    arr = np.asarray(line_sigmas, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("line_sigmas must contain at least one entry")
    if not np.isfinite(arr).all():
        raise ValueError("line_sigmas contains non-finite values")

    sigma_min = float(np.maximum(sigma_min_factor * arr.min(), floor_nm))
    sigma_max = float(np.maximum(sigma_max_factor * arr.max(), sigma_min * 1.5))

    # log-spaced grid: σ_k = σ_min * (σ_max/σ_min)^(k/(N-1))
    grid = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), n_sigma))
    return grid


def _gaussian_fft_kernels(
    sigma_grid: jnp.ndarray,
    n_fft: int,
    dlam: jnp.ndarray,
    dtype: Any,
) -> jnp.ndarray:
    """Precompute the FFT-domain Gaussian kernel for each σ layer.

    Returns the (N_σ, n_fft//2 + 1) complex rfft of a circularly-shifted
    Gaussian normalised to unit area. We construct the kernel in the
    frequency domain analytically: the rfft of a continuous Gaussian
    ``g(x) = (1/(σ√(2π))) exp(-x²/(2σ²))`` evaluated on a uniform grid is
    well-approximated by ``exp(-2π² σ² f²)`` for the positive frequency
    half. This sidesteps the FFT cost on the kernel itself and keeps the
    expression differentiable.
    """
    # FFT frequency axis: f_k = k / (n_fft · dlam), for k = 0..n_fft//2
    n_half = n_fft // 2 + 1
    k_idx = jnp.arange(n_half, dtype=dtype)
    freq = k_idx / (jnp.asarray(n_fft, dtype=dtype) * dlam)  # cycles per nm
    # G_hat(f) = exp(-2 π² σ² f²) for unit-area Gaussian
    two_pi_sq = jnp.asarray(2.0 * np.pi**2, dtype=dtype)
    sigma_col = sigma_grid[:, None].astype(dtype)
    g_hat = jnp.exp(-two_pi_sq * (sigma_col * freq[None, :]) ** 2)
    return g_hat


@jit_if_available
def _ldm_broaden_impl(
    line_wavelengths: jnp.ndarray,
    line_intensities: jnp.ndarray,
    line_sigmas: jnp.ndarray,
    wavelength_grid: jnp.ndarray,
    sigma_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Pure-JAX LDM kernel — see :func:`ldm_broaden` for the public docstring."""
    # All compute happens in the dtype of the wavelength grid so callers can
    # pick fp32/fp64 via JaxMemoryPolicy upstream.
    dtype = wavelength_grid.dtype
    line_wavelengths = line_wavelengths.astype(dtype)
    line_intensities = line_intensities.astype(dtype)
    line_sigmas = line_sigmas.astype(dtype)
    sigma_grid = sigma_grid.astype(dtype)

    n_lambda = wavelength_grid.shape[0]
    n_sigma = sigma_grid.shape[0]

    dlam = wavelength_grid[1] - wavelength_grid[0]
    wl0 = wavelength_grid[0]

    # --- Step 1: locate σ_i in the log-σ grid ---
    log_sigma_grid = jnp.log(sigma_grid)
    log_sigmas = jnp.log(jnp.maximum(line_sigmas, jnp.asarray(1e-30, dtype=dtype)))

    # searchsorted returns the insertion point; subtract 1 and clamp to
    # [0, N_σ-2] to get the left-edge layer index per spec §4.
    k = jnp.clip(
        jnp.searchsorted(sigma_grid, line_sigmas) - 1,
        0,
        n_sigma - 2,
    ).astype(jnp.int32)

    log_sg_k = log_sigma_grid[k]
    log_sg_kp1 = log_sigma_grid[k + 1]
    alpha = jnp.clip(
        (log_sigmas - log_sg_k)
        / jnp.maximum(log_sg_kp1 - log_sg_k, jnp.asarray(1e-30, dtype=dtype)),
        0.0,
        1.0,
    )

    # --- Step 2: locate λ_i on the uniform wavelength grid ---
    # 3-point parabolic (Lagrange) scatter in λ: a delta source at
    # fractional offset β ∈ [-0.5, 0.5] from the nearest bin spreads to
    # three bins (j-1, j, j+1) with weights chosen to match its 0th, 1st
    # and 2nd moments. This eliminates the sub-pixel shift error that
    # plain bilinear scatter exhibits when σ ~ Δλ — the dominant LDM
    # accuracy cap on the parity test (spec §8 AC#1). Weights derived
    # from moment matching:
    #
    #     w_{-1} = β(β-1)/2
    #     w_0   = 1 - β²
    #     w_{+1} = β(β+1)/2
    #
    # (sum=1, ∑i·w_i=β, ∑i²·w_i=β² for β∈[0,1] interpreted relative to
    # j-1; equivalent symmetric form when β∈[-0.5,0.5] relative to j.)
    j_raw = (line_wavelengths - wl0) / dlam
    j_center = jnp.clip(jnp.round(j_raw).astype(jnp.int32), 1, n_lambda - 2)
    beta = jnp.clip(j_raw - j_center.astype(dtype), -0.5, 0.5)
    j_lo = j_center - 1
    j_hi = j_center + 1

    half = jnp.asarray(0.5, dtype=dtype)
    w_lo = half * beta * (beta - 1.0)
    w_ctr = 1.0 - beta * beta
    w_hi = half * beta * (beta + 1.0)

    # --- Step 3: scatter into the (N_σ, N_λ) ledger via segment_sum ---
    # Each line is a unit-area delta in continuous wavelength space; on a
    # discrete grid with spacing dλ, its density representation is
    # ``intensity / dλ`` (so that the discrete sum approximates the
    # continuous integral). The FFT-domain Gaussian kernel below has unit
    # *continuous* area, which the IFFT samples as ``dλ · g_density``; the
    # ``1/dλ`` here cancels that and yields the same per-line Gaussian
    # density as ``apply_gaussian_broadening_per_line``.
    inv_dlam = jnp.asarray(1.0, dtype=dtype) / dlam
    weighted_intensity = line_intensities * inv_dlam

    # 6-way bilinear-in-σ × Lagrange-3-in-λ scatter:
    # (σ_lo, σ_hi) × (j-1, j, j+1).
    one_minus_alpha = 1.0 - alpha
    w_klo_jlo = one_minus_alpha * w_lo * weighted_intensity
    w_klo_jctr = one_minus_alpha * w_ctr * weighted_intensity
    w_klo_jhi = one_minus_alpha * w_hi * weighted_intensity
    w_khi_jlo = alpha * w_lo * weighted_intensity
    w_khi_jctr = alpha * w_ctr * weighted_intensity
    w_khi_jhi = alpha * w_hi * weighted_intensity

    n_lambda_i32 = jnp.asarray(n_lambda, dtype=jnp.int32)
    seg_klo_jlo = k * n_lambda_i32 + j_lo
    seg_klo_jctr = k * n_lambda_i32 + j_center
    seg_klo_jhi = k * n_lambda_i32 + j_hi
    seg_khi_jlo = (k + 1) * n_lambda_i32 + j_lo
    seg_khi_jctr = (k + 1) * n_lambda_i32 + j_center
    seg_khi_jhi = (k + 1) * n_lambda_i32 + j_hi
    num_segments = int(n_sigma * n_lambda)

    flat_klo_jlo = jax.ops.segment_sum(
        w_klo_jlo, seg_klo_jlo, num_segments=num_segments, indices_are_sorted=False
    )
    flat_klo_jctr = jax.ops.segment_sum(
        w_klo_jctr, seg_klo_jctr, num_segments=num_segments, indices_are_sorted=False
    )
    flat_klo_jhi = jax.ops.segment_sum(
        w_klo_jhi, seg_klo_jhi, num_segments=num_segments, indices_are_sorted=False
    )
    flat_khi_jlo = jax.ops.segment_sum(
        w_khi_jlo, seg_khi_jlo, num_segments=num_segments, indices_are_sorted=False
    )
    flat_khi_jctr = jax.ops.segment_sum(
        w_khi_jctr, seg_khi_jctr, num_segments=num_segments, indices_are_sorted=False
    )
    flat_khi_jhi = jax.ops.segment_sum(
        w_khi_jhi, seg_khi_jhi, num_segments=num_segments, indices_are_sorted=False
    )
    flat_acc = (
        flat_klo_jlo + flat_klo_jctr + flat_klo_jhi + flat_khi_jlo + flat_khi_jctr + flat_khi_jhi
    )
    ledger = flat_acc.reshape((n_sigma, n_lambda))

    # --- Step 4: FFT-convolve each σ-layer with its Gaussian kernel ---
    # FFT length: spec §4 calls for ``next_pow2(N_λ + 6·max(σ_grid)/Δλ)``.
    # ``n_lambda`` is concrete under jit (shape-static), so we resolve the
    # FFT length to a Python int at trace time using a safe upper bound:
    # ``2 * next_pow2(N_λ)``. For all CF-LIBS manifold grids with σ < ~1 nm
    # and Δλ ~ 0.01 nm, this comfortably covers the 6σ-tail clearance.
    n_fft = int(_next_pow2(2 * n_lambda))

    ledger_padded = jnp.zeros((n_sigma, n_fft), dtype=dtype)
    ledger_padded = ledger_padded.at[:, :n_lambda].set(ledger)

    g_hat = _gaussian_fft_kernels(sigma_grid, n_fft, dlam, dtype)

    fft_ledger = jnp.fft.rfft(ledger_padded, n=n_fft, axis=1)
    fft_conv = fft_ledger * g_hat
    conv = jnp.fft.irfft(fft_conv, n=n_fft, axis=1)

    spectrum = jnp.sum(conv[:, :n_lambda], axis=0)
    return spectrum


def _next_pow2(n: int) -> int:
    """Return the smallest power of two >= n (n >= 1)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def ldm_broaden(
    line_wavelengths: jnp.ndarray,
    line_intensities: jnp.ndarray,
    line_sigmas: jnp.ndarray,
    wavelength_grid: jnp.ndarray,
    sigma_grid: jnp.ndarray,
    *,
    gamma_grid: jnp.ndarray | None = None,
    line_gammas: jnp.ndarray | None = None,
    memory_policy: JaxMemoryPolicy | None = None,
) -> jnp.ndarray:
    """Broaden a line catalog onto a uniform wavelength grid via LDM.

    The kernel projects each line onto a 1-D log-σ grid using bilinear
    weights, then convolves each σ-layer with a Gaussian of the
    corresponding width using a real FFT. The output is the sum across all
    σ-layers — identical (to ``O(dx_σ²)``) to the per-line Gaussian
    broadening returned by
    :func:`cflibs.radiation.profiles.apply_gaussian_broadening_per_line`.

    Parameters
    ----------
    line_wavelengths : array, shape (N_lines,)
        Line center wavelengths in nm.
    line_intensities : array, shape (N_lines,)
        Integrated line intensities (area under each Gaussian).
    line_sigmas : array, shape (N_lines,)
        Per-line Gaussian σ in nm. Must be strictly positive; values outside
        the σ-grid range are clipped to the boundary layers.
    wavelength_grid : array, shape (N_λ,)
        Uniform wavelength grid in nm.
    sigma_grid : array, shape (N_σ,)
        Log-spaced σ grid in nm. Construct once per manifold (or per call
        for one-off use) via :func:`build_sigma_grid`. Treated as jit-static
        layout — the caller is responsible for ensuring the grid bounds
        bracket the per-line σ range with reasonable margin.
    gamma_grid, line_gammas : array, optional
        Reserved for the 2-D Voigt extension. Currently unused; passing
        either argument raises :class:`NotImplementedError`.
    memory_policy : JaxMemoryPolicy, optional
        Currently unused on this code path (precision is governed by the
        input dtype). Accepted for forward compatibility with T1-2 dispatch.

    Returns
    -------
    spectrum : array, shape (N_λ,)
        Sum of per-line Gaussians on the wavelength grid.

    Raises
    ------
    ImportError
        If JAX is unavailable.
    NotImplementedError
        If ``gamma_grid`` or ``line_gammas`` is provided (2-D Voigt
        extension is scope for a follow-up bead).
    ValueError
        If the wavelength grid has fewer than two samples (cannot infer Δλ).
    """
    if not HAS_JAX:  # pragma: no cover - JAX is part of cflibs's local extras
        raise ImportError(
            "JAX is required for cflibs.radiation.ldm.ldm_broaden. "
            "Install with `pip install jax jaxlib`."
        )
    if gamma_grid is not None or line_gammas is not None:
        raise NotImplementedError(
            "2-D Voigt LDM (gamma_grid / line_gammas) is reserved for a "
            "follow-up bead; only the 1-D Gaussian path is implemented."
        )

    _ = memory_policy  # forward-compat: T1-2 will route policy through here.

    line_wavelengths = jnp.asarray(line_wavelengths)
    line_intensities = jnp.asarray(line_intensities)
    line_sigmas = jnp.asarray(line_sigmas)
    wavelength_grid = jnp.asarray(wavelength_grid)
    sigma_grid = jnp.asarray(sigma_grid)

    if wavelength_grid.ndim != 1 or wavelength_grid.shape[0] < 2:
        raise ValueError(
            "wavelength_grid must be 1-D with at least two samples; "
            f"got shape {tuple(wavelength_grid.shape)}"
        )

    # Promote to the wavelength grid's dtype so downstream segment_sum and
    # rfft run at a single precision tier.
    target_dtype = wavelength_grid.dtype
    if line_wavelengths.dtype != target_dtype:
        line_wavelengths = line_wavelengths.astype(target_dtype)
    if line_intensities.dtype != target_dtype:
        line_intensities = line_intensities.astype(target_dtype)
    if line_sigmas.dtype != target_dtype:
        line_sigmas = line_sigmas.astype(target_dtype)
    if sigma_grid.dtype != target_dtype:
        sigma_grid = sigma_grid.astype(target_dtype)

    if line_wavelengths.shape[0] == 0:
        return jnp.zeros_like(wavelength_grid)

    return _ldm_broaden_impl(
        line_wavelengths,
        line_intensities,
        line_sigmas,
        wavelength_grid,
        sigma_grid,
    )


def broaden_lines_ldm(
    line_wavelengths: np.ndarray | jnp.ndarray,
    line_intensities: np.ndarray | jnp.ndarray,
    line_sigmas: np.ndarray | jnp.ndarray,
    wavelength_grid: np.ndarray | jnp.ndarray,
    *,
    sigma_grid: np.ndarray | jnp.ndarray | None = None,
    n_sigma: int = DEFAULT_N_SIGMA,
    memory_policy: JaxMemoryPolicy | None = None,
) -> np.ndarray:
    """Host-side LDM driver — auto-builds ``sigma_grid`` when not supplied.

    This is the entry point used by :class:`cflibs.radiation.SpectrumModel`
    when ``BroadeningMode.LDM_GAUSSIAN`` is selected. It builds the σ-grid
    from the observed line widths (numpy-side), uploads to the device, and
    returns a numpy array so the caller does not have to special-case the
    device-array boundary.

    Parameters
    ----------
    line_wavelengths, line_intensities, line_sigmas : array-like
        Per-line catalogs. See :func:`ldm_broaden`.
    wavelength_grid : array-like, shape (N_λ,)
        Uniform wavelength grid in nm.
    sigma_grid : array-like, optional
        Pre-built σ-grid (numpy or jax). When ``None``, constructed via
        :func:`build_sigma_grid`.
    n_sigma : int, optional
        Number of σ layers when auto-building the grid.
    memory_policy : JaxMemoryPolicy, optional
        Reserved for forward compatibility with T1-2 dispatch.

    Returns
    -------
    spectrum : ndarray, shape (N_λ,)
        Broadened spectrum on the wavelength grid.
    """
    if not HAS_JAX:  # pragma: no cover - JAX is part of cflibs's local extras
        raise ImportError(
            "JAX is required for cflibs.radiation.ldm.broaden_lines_ldm. "
            "Install with `pip install jax jaxlib`."
        )

    policy = memory_policy if memory_policy is not None else jax_policy()

    # Match the caller's dtype convention via JaxMemoryPolicy. We use the
    # process-default real dtype unless the caller's wavelength grid is
    # already a jnp array (in which case respect its dtype).
    if hasattr(wavelength_grid, "dtype") and isinstance(wavelength_grid.dtype, jnp.dtype):
        dtype = wavelength_grid.dtype
    else:
        dtype = policy.real_dtype if HAS_JAX else jax_default_real_dtype()

    line_wl = np.asarray(line_wavelengths, dtype=np.float64).reshape(-1)
    line_int = np.asarray(line_intensities, dtype=np.float64).reshape(-1)
    line_sig = np.asarray(line_sigmas, dtype=np.float64).reshape(-1)
    wl_grid = np.asarray(wavelength_grid, dtype=np.float64).reshape(-1)

    if line_wl.size == 0:
        return np.zeros_like(wl_grid)

    if sigma_grid is None:
        sigma_grid_arr = build_sigma_grid(line_sig, n_sigma=n_sigma)
    else:
        sigma_grid_arr = np.asarray(sigma_grid, dtype=np.float64).reshape(-1)

    spectrum = ldm_broaden(
        jnp.asarray(line_wl, dtype=dtype),
        jnp.asarray(line_int, dtype=dtype),
        jnp.asarray(line_sig, dtype=dtype),
        jnp.asarray(wl_grid, dtype=dtype),
        jnp.asarray(sigma_grid_arr, dtype=dtype),
        memory_policy=policy,
    )
    return np.asarray(spectrum)

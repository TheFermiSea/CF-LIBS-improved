"""
ALIAS (Automated Line Identification Algorithm for Spectroscopy) implementation.

Based on Noel et al. (2025) arXiv:2501.01057. The ALIAS algorithm identifies elements
in LIBS spectra through a 7-step process: peak detection, theoretical emissivity
calculation, line fusion, matching, threshold determination, scoring, and decision.
"""

from typing import List, Tuple, Optional, Set
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks
from scipy.special import erf
from scipy.stats import binom, linregress

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.core.logging_config import get_logger as _get_alias_logger
from cflibs.inversion.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
    IdentifiedLine,
    get_wavelength_tolerance,
)
from cflibs.inversion.identify._coverage import (
    CoverageTracker,
    merge_coverage_into_parameters,
)
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter, LineObservation
from cflibs.inversion.preprocessing import estimate_baseline, estimate_noise
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver

logger = get_logger("inversion.identify.alias")
_alias_logger = _get_alias_logger("inversion.identify.alias")

# JAX is an optional fast path for the per-spectrum Boltzmann temperature fit
# used by ``ALIASIdentifier._estimate_plasma_temperature``. The default path
# remains ``scipy.stats.linregress`` so behavior is unchanged unless
# ``use_jax_boltzmann_fit=True`` is passed to the constructor.
try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only when jax missing
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


if _HAS_JAX:

    @jax.jit
    def _jax_boltzmann_slope_intercept(
        E_k: "jnp.ndarray",
        y: "jnp.ndarray",
        mask: "jnp.ndarray",
    ) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]":
        """Closed-form unweighted least-squares Boltzmann fit, vectorized over a
        batch of spectra. Inputs are (B, N_max); outputs are (B,).

        Returns ``(slope, intercept, r_squared, n_valid)``. Padded entries
        (``mask == False``) are zeroed out. The caller is responsible for
        converting slope to temperature with the inf/NaN sentinels.
        """
        x = jnp.asarray(E_k, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        m = jnp.asarray(mask, dtype=jnp.float64)

        # Five sums, all reductions over axis=1.
        n = jnp.sum(m, axis=1)
        S_x = jnp.sum(m * x, axis=1)
        S_y = jnp.sum(m * y, axis=1)
        S_xx = jnp.sum(m * x * x, axis=1)
        S_xy = jnp.sum(m * x * y, axis=1)

        det = n * S_xx - S_x * S_x
        # ``is_valid`` mirrors the CPU guards: at least 2 points and a non-
        # degenerate determinant (i.e. some spread in x).
        is_valid = (n >= 2.0) & (jnp.abs(det) > 1e-30)
        det_safe = jnp.where(is_valid, det, 1.0)

        slope = jnp.where(is_valid, (n * S_xy - S_x * S_y) / det_safe, jnp.nan)
        intercept = jnp.where(is_valid, (S_xx * S_y - S_x * S_xy) / det_safe, jnp.nan)

        # Weighted R^2 (weights are 0/1 from the mask).
        y_pred = intercept[:, None] + slope[:, None] * x
        SS_res = jnp.sum(m * (y - y_pred) ** 2, axis=1)
        y_mean = jnp.where(n > 0, S_y / jnp.maximum(n, 1.0), 0.0)
        SS_tot = jnp.sum(m * (y - y_mean[:, None]) ** 2, axis=1)
        r_squared = jnp.where(is_valid & (SS_tot > 1e-30), 1.0 - SS_res / SS_tot, jnp.nan)

        return slope, intercept, r_squared, n


def boltzmann_temperature_jax(
    log_I_over_gA: np.ndarray,
    E_upper: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    return_diagnostics: bool = False,
):
    """JAX-vectorized Boltzmann-plot temperature fit across a batch of spectra.

    Fits ``y = slope * E_upper + intercept`` for each spectrum, where
    ``y = log(I / (g_k A_ki))`` (i.e. ``log(I*lambda/(g_k*A_ki))`` or any
    convention the caller adopts — the function is agnostic to the
    intercept's physical meaning). Temperatures are derived from
    ``T = -1 / (slope * k_B)``.

    This is the JAX/GPU-vectorized counterpart of the
    ``scipy.stats.linregress`` call used by
    ``ALIASIdentifier._estimate_plasma_temperature``. The CPU path is
    preserved; this function is opt-in via the ``use_jax_boltzmann_fit``
    constructor flag on :class:`ALIASIdentifier`.

    Parameters
    ----------
    log_I_over_gA : np.ndarray, shape (B, N) or (N,)
        Boltzmann-plot y-values per spectrum. ``np.nan`` entries are
        treated as missing (masked out of the fit).
    E_upper : np.ndarray, shape (N,) or (B, N)
        Upper-level energies in eV. If 1-D, broadcast across the batch.
    weights : np.ndarray, shape (B, N), optional
        Currently unused (reserved for future weighted-LS extension). Pass
        ``None`` for unweighted fit, matching ``scipy.stats.linregress``
        behavior. A ``NotImplementedError`` is raised if a non-uniform
        weight array is provided so callers aren't silently misled.
    return_diagnostics : bool, optional
        If True, returns ``(T_K, slope, r_squared)`` instead of just
        ``T_K``. Default False.

    Returns
    -------
    T_K : np.ndarray, shape (B,)
        Plasma temperature in Kelvin per spectrum. Sentinel values:

        * ``+inf`` — slope is non-negative (population-inversion or fit
          failure), matching the ``T set to infinity`` semantics of the
          CPU code path in ``cflibs.inversion.physics.boltzmann``.
        * ``nan``  — degenerate fit (fewer than 2 valid points, zero
          spread in ``E_upper``, etc.).

    Raises
    ------
    ImportError
        If JAX is not installed.

    Notes
    -----
    - Result is jit-compiled; the first call incurs trace overhead, all
      subsequent calls with the same shapes are constant-time on GPU.
    - Float64 is used throughout for numerical stability of the normal
      equations.
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for boltzmann_temperature_jax. Install with: pip install jax jaxlib"
        )

    if weights is not None and not np.allclose(weights, weights.flat[0]):
        raise NotImplementedError(
            "Non-uniform weights are not yet supported by "
            "boltzmann_temperature_jax. Use the CPU path or pass weights=None."
        )

    y_arr = np.atleast_2d(np.asarray(log_I_over_gA, dtype=np.float64))
    E_arr = np.asarray(E_upper, dtype=np.float64)
    if E_arr.ndim == 1:
        E_arr = np.broadcast_to(E_arr, y_arr.shape).copy()
    else:
        E_arr = np.atleast_2d(E_arr)

    if E_arr.shape != y_arr.shape:
        raise ValueError(
            f"E_upper shape {E_arr.shape} does not broadcast to log_I_over_gA shape {y_arr.shape}"
        )

    # Build mask from NaN/inf in either array; zero out masked entries so
    # the closed-form sums stay finite under jit.
    mask = np.isfinite(y_arr) & np.isfinite(E_arr)
    y_safe = np.where(mask, y_arr, 0.0)
    E_safe = np.where(mask, E_arr, 0.0)

    slope, intercept, r_squared, n_valid = _jax_boltzmann_slope_intercept(
        jnp.asarray(E_safe), jnp.asarray(y_safe), jnp.asarray(mask)
    )

    slope_np = np.asarray(slope)
    r_sq_np = np.asarray(r_squared)

    # Convert slope -> temperature. Mirrors the CPU sentinels exactly:
    # slope >= 0 (or zero/near-zero) -> +inf ("T set to infinity"),
    # NaN slope (degenerate fit)     -> NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        T_K = -1.0 / (slope_np * KB_EV)
    nondegenerate = np.isfinite(slope_np)
    nonneg = nondegenerate & (slope_np >= 0)
    T_K = np.where(nonneg, np.inf, T_K)
    T_K = np.where(~nondegenerate, np.nan, T_K)

    if return_diagnostics:
        return T_K, slope_np, r_sq_np
    return T_K


# ---------------------------------------------------------------------------
# JAX-vectorized helpers for the per-spectrum hot loops in ``ALIASIdentifier``.
# These mirror the CPU paths exactly when invoked through the opt-in
# constructor flags (``use_jax_nnls``, ``use_jax_p_snr``). They are written so
# the per-spectrum solver is jit-compiled once and reused; the future batched
# workflow can wrap them in ``jax.vmap`` across many spectra without
# additional plumbing.
#
# Algorithm choice (Codex + Gemini consultation, see
# ``docs/jax-port/alias-consultation.md``):
#
# * NNLS solver — FISTA with adaptive O'Donoghue-Candès restart. Fixed iter
#   loop (``lax.fori_loop``) is jit/vmap-friendly. The active-set Lawson-
#   Hanson approach used by ``scipy.optimize.nnls`` cannot be vmap'd cleanly
#   because the passive set has a dynamic shape. On well-conditioned
#   templates (the LIBS Gaussian peak case), FISTA hits 1e-13 agreement with
#   scipy in <1000 iters; on highly correlated columns the adaptive restart
#   recovers the same precision.
# * Sparse elastic-net NNLS — same FISTA, modified gradient. Under x >= 0 the
#   L1 term collapses to a smooth alpha * sum(x), so the elastic-net
#   objective is differentiable end-to-end and the projected-gradient step
#   converges without proximal-operator machinery.

if _HAS_JAX:
    _FISTA_DEFAULT_MAX_ITER = 1000

    def _fista_step_body(state, AtA, Atb, step, l1):
        """Single FISTA + adaptive-restart iteration. Pure to enable JIT."""
        x, y, t = state
        grad = AtA @ y - Atb + l1
        x_new = jnp.maximum(0.0, y - step * grad)
        # O'Donoghue-Candès gradient-based restart: if the momentum step
        # increased the objective (i.e. (y - x_new) . (x_new - x) > 0), reset
        # the momentum sequence by forcing t_eff = 1 for the next update.
        restart_cond = jnp.dot(y - x_new, x_new - x) > 0.0
        t_eff = jnp.where(restart_cond, 1.0, t)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_eff * t_eff))
        y_new = x_new + ((t_eff - 1.0) / t_new) * (x_new - x)
        return (x_new, y_new, t_new)

    def _solve_nnls_jax_core(
        A: "jnp.ndarray",
        b: "jnp.ndarray",
        max_iter: int,
        l1: float,
        l2: float,
    ) -> "jnp.ndarray":
        """FISTA-with-restart non-negative least squares for a single problem.

        Minimizes ``0.5 ||A x - b||^2 + l1 sum(x) + 0.5 l2 ||x||^2`` subject
        to ``x >= 0``. The ridge term is folded into ``A^T A`` so a single
        solver handles both the dense (l1=l2=0) and the smooth elastic-net
        cases.
        """
        AtA = A.T @ A + l2 * jnp.eye(A.shape[1], dtype=A.dtype)
        Atb = A.T @ b
        # Lipschitz constant of the smooth gradient: largest eigenvalue of
        # AtA. For 12x12 problems this is a few-microsecond eigh.
        L = jnp.linalg.eigvalsh(AtA)[-1] + 1e-30
        step = 1.0 / L

        n = A.shape[1]
        init = (
            jnp.zeros(n, dtype=A.dtype),
            jnp.zeros(n, dtype=A.dtype),
            jnp.asarray(1.0, dtype=A.dtype),
        )

        def body(_i, state):
            return _fista_step_body(state, AtA, Atb, step, l1)

        x_final, _y, _t = jax.lax.fori_loop(0, max_iter, body, init)
        return jnp.maximum(0.0, x_final)

    _solve_nnls_jax_jit = jax.jit(_solve_nnls_jax_core, static_argnames=("max_iter",))

    def _loo_solve(A: "jnp.ndarray", b: "jnp.ndarray", max_iter: int) -> "jnp.ndarray":
        """Leave-one-out NNLS: returns ``(n_cands, n_cands)`` coefficient
        matrix where row ``j`` is the NNLS solution with column ``j`` of A
        zeroed out. ``c_loo[j, j]`` is guaranteed to be 0.
        """
        n_cands = A.shape[1]
        # Build the (n_cands, n_cands) "remove column j" mask: identity-
        # complement. Multiplying A elementwise by this mask zeroes the
        # j-th column, which forces the FISTA projection to land at x_j=0.
        masks = 1.0 - jnp.eye(n_cands, dtype=A.dtype)  # (n_cands, n_cands)
        A_loo = A[None, :, :] * masks[:, None, :]  # (n_cands, m, n_cands)
        return jax.vmap(lambda Aj: _solve_nnls_jax_core(Aj, b, max_iter, 0.0, 0.0))(A_loo)

    _loo_solve_jit = jax.jit(_loo_solve, static_argnames=("max_iter",))


def solve_nnls_jax(
    A: np.ndarray,
    b: np.ndarray,
    *,
    max_iter: int = 50000,
    l1: float = 0.0,
    l2: float = 0.0,
) -> np.ndarray:
    """JAX FISTA-with-restart NNLS solver, drop-in replacement for
    :func:`scipy.optimize.nnls` (residual is not returned; recover it as
    ``np.linalg.norm(A @ x - b)``).

    Minimizes ``0.5 ||A x - b||^2 + l1 sum(x) + 0.5 l2 ||x||^2`` subject to
    ``x >= 0`` via FISTA with adaptive (gradient-based) restart.

    .. note:: bead ``CF-LIBS-improved-jbfg.1`` — the prior docstring claim
        that this hits ``rtol 1e-5`` vs scipy in ``<1000 iters`` was
        falsified on the Vrabel 11×854 column-correlated LIBS template
        matrix (FISTA c[Fe]=0.586 vs scipy c[Fe]=11.196 at max_iter=1000,
        a 95% error). Empirical sweep showed max_iter=5000 → max-diff
        44.9, max_iter=20000 → 40.4, max_iter=50000 → rtol≈1e-3,
        max_iter=100000 → rtol≈1e-7. Default bumped to 50000 to land
        within rtol 1e-3 on highly column-correlated regimes; this is
        still cheap on GPU.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Non-negative template matrix.
    b : np.ndarray, shape (m,)
        Observed signal.
    max_iter : int, optional
        FISTA iteration count. Fixed (not adaptive) to keep the loop
        jit/vmap-friendly (default: 50000 — was 1000, see note above).
    l1 : float, optional
        L1 regularization strength. Under ``x >= 0`` the L1 term is the
        smooth ``l1 * sum(x)``, so a separate proximal step isn't needed
        (default: 0.0).
    l2 : float, optional
        L2 (ridge) regularization strength, folded into ``A^T A``
        (default: 0.0).

    Returns
    -------
    np.ndarray, shape (n,)
        Non-negative coefficients.

    Raises
    ------
    ImportError
        If JAX is not installed.

    Notes
    -----
    For non-unique solutions (e.g. duplicated columns in ``A``), this solver
    converges to the minimum-norm solution rather than the active-set
    solution returned by SciPy. Both achieve the same objective value but
    may distribute mass differently across redundant columns.
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for solve_nnls_jax. Install with: pip install jax jaxlib"
        )
    A_j = jnp.asarray(A, dtype=jnp.float64)
    b_j = jnp.asarray(b, dtype=jnp.float64)
    x = _solve_nnls_jax_jit(A_j, b_j, max_iter, float(l1), float(l2))
    return np.asarray(x)


def solve_sparse_nnls_jax(
    A: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float = 0.01,
    l1_ratio: float = 0.9,
    max_iter: int = 50000,
) -> Tuple[np.ndarray, float]:
    """JAX FISTA elastic-net NNLS, drop-in for the L-BFGS-B path in
    ``ALIASIdentifier._compute_sparse_nnls_scores``.

    Column-normalizes ``A`` exactly as the CPU code does, solves the
    smooth elastic-net problem (``x >= 0`` collapses the L1 term to
    ``alpha * l1_ratio * sum(x)``), then un-normalizes the coefficients
    so they reference the original ``A``.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    b : np.ndarray, shape (m,)
    alpha : float, optional
        Overall regularization strength (default: 0.01).
    l1_ratio : float, optional
        L1 vs L2 mix (default: 0.9). 1.0 = pure lasso, 0.0 = pure ridge.
    max_iter : int, optional
        FISTA iteration count (default: 1000).

    Returns
    -------
    Tuple[np.ndarray, float]
        ``(sparse_c, residual_norm)``. ``residual_norm`` is in the original
        un-normalized coordinate system, matching the CPU return convention.
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for solve_sparse_nnls_jax. Install with: pip install jax jaxlib"
        )
    A_np = np.asarray(A, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    col_norms = np.linalg.norm(A_np, axis=0)
    col_norms_safe = np.where(col_norms == 0, 1.0, col_norms)
    A_norm = A_np / col_norms_safe

    l1_weight = float(alpha * l1_ratio)
    l2_weight = float(alpha * (1.0 - l1_ratio))

    x_norm = _solve_nnls_jax_jit(
        jnp.asarray(A_norm),
        jnp.asarray(b_np),
        max_iter,
        l1_weight,
        l2_weight,
    )
    sparse_c = np.asarray(x_norm) / col_norms_safe
    residual = float(np.linalg.norm(b_np - A_np @ sparse_c))
    return sparse_c, residual


def compute_nnls_attribution_jax(
    A: np.ndarray,
    peak_intensities: np.ndarray,
    *,
    max_iter: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JAX-vectorized counterpart of ``_compute_nnls_attribution``.

    Solves the full NNLS plus all n_cands leave-one-out NNLS problems in a
    single ``vmap``-ed FISTA call, then computes the partial-R^2 and
    local-explanation scores using the exact same arithmetic as the CPU
    function.

    Parameters
    ----------
    A : np.ndarray, shape (n_peaks, n_cands)
    peak_intensities : np.ndarray, shape (n_peaks,)
    max_iter : int, optional
        FISTA iteration count (default: 50000). Bumped from 1000 per bead
        ``CF-LIBS-improved-jbfg.1`` — on the heavily column-correlated
        11×854 Vrabel template matrix the 1000-iter default left coefficient
        errors of ~95% (Fe) and ~99% (Si/Pb) vs scipy active-set.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(P_mix, P_local, c)`` matching ``_compute_nnls_attribution``.
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for compute_nnls_attribution_jax. Install with: pip install jax jaxlib"
        )
    n_cands = A.shape[1]
    if n_cands == 0 or np.all(A == 0):
        return np.ones(n_cands), np.ones(n_cands), np.zeros(n_cands)

    A_j = jnp.asarray(A, dtype=jnp.float64)
    b_j = jnp.asarray(peak_intensities, dtype=jnp.float64)

    c = np.asarray(_solve_nnls_jax_jit(A_j, b_j, max_iter, 0.0, 0.0))
    total_rss = float(np.sum((peak_intensities - A @ c) ** 2))
    total_energy = float(np.sum(peak_intensities**2))
    if total_energy == 0:
        return np.ones(n_cands), np.ones(n_cands), c

    if n_cands == 1:
        P_mix = np.array([1.0])
    else:
        c_loo = np.asarray(_loo_solve_jit(A_j, b_j, max_iter))  # (n_cands, n_cands)
        # rss_without[j] = || A @ c_loo[j] - b ||^2 (column j of A contributes
        # zero because c_loo[j, j] = 0 from the mask).
        rss_without = np.sum((A @ c_loo.T - peak_intensities[:, None]) ** 2, axis=0)
        P_mix = (rss_without - total_rss) / total_energy

    # P_local — exact CPU arithmetic, just vectorized.
    P_local = np.zeros(n_cands)
    for j in range(n_cands):
        claimed = A[:, j] > 1e-6
        if not np.any(claimed):
            continue
        obs_at_claimed = np.sum(peak_intensities[claimed])
        if obs_at_claimed <= 0:
            continue
        elem_contribution = np.sum(A[claimed, j] * c[j])
        P_local[j] = float(np.clip(elem_contribution / obs_at_claimed, 0.0, 1.0))

    return P_mix, P_local, c


def build_nnls_templates_jax(
    line_wavelengths_padded: np.ndarray,
    line_emissivities_padded: np.ndarray,
    line_masks: np.ndarray,
    per_element_shifts: np.ndarray,
    peak_wavelengths: np.ndarray,
    resolving_power: float,
) -> np.ndarray:
    """JAX-vectorized Gaussian template matrix builder.

    Mirrors ``ALIASIdentifier._build_nnls_templates`` exactly. The Python
    loop over candidates is replaced by ragged-tensor broadcasting padded
    to the largest candidate's line count. Padded entries are zeroed out
    via ``line_masks``.

    Parameters
    ----------
    line_wavelengths_padded : np.ndarray, shape (n_cands, L_max)
        Theoretical line wavelengths per candidate, padded to ``L_max``.
    line_emissivities_padded : np.ndarray, shape (n_cands, L_max)
        Per-line emissivities, padded to ``L_max``.
    line_masks : np.ndarray, shape (n_cands, L_max) bool
        ``True`` for valid lines, ``False`` for padding.
    per_element_shifts : np.ndarray, shape (n_cands,)
        Per-candidate wavelength shift in nm.
    peak_wavelengths : np.ndarray, shape (n_peaks,)
        Detected peak wavelengths in nm.
    resolving_power : float
        Instrument resolving power; per-peak sigma = lambda / R / 2.355.

    Returns
    -------
    np.ndarray, shape (n_peaks, n_cands)
        Template matrix matching the CPU build.
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for build_nnls_templates_jax. Install with: pip install jax jaxlib"
        )
    pw = jnp.asarray(peak_wavelengths, dtype=jnp.float64)
    sig = pw / float(resolving_power) / 2.355  # (n_peaks,)
    lw = jnp.asarray(line_wavelengths_padded, dtype=jnp.float64)
    le = jnp.asarray(line_emissivities_padded, dtype=jnp.float64)
    mk = jnp.asarray(line_masks, dtype=jnp.float64)
    sh = jnp.asarray(per_element_shifts, dtype=jnp.float64)

    # Shapes:
    #   pw:  (P,)        peak wavelengths
    #   sig: (P,)        per-peak sigma
    #   lw:  (C, L)      line wavelengths (padded)
    #   le:  (C, L)      line emissivities (padded)
    #   mk:  (C, L)      validity mask
    #   sh:  (C,)        per-element shift
    # Compute A[p, c] = sum_l mk[c, l] * le[c, l] * exp(-0.5 z^2) * (|d| < 3 sig)
    #   where d = pw[p] - (lw[c, l] + sh[c]), z = d / sig[p]
    shifted = lw + sh[:, None]  # (C, L)
    # (P, C, L)
    diff = pw[:, None, None] - shifted[None, :, :]
    z = diff / sig[:, None, None]
    gauss = jnp.exp(-0.5 * z * z)
    proximity = jnp.abs(diff) < (3.0 * sig[:, None, None])
    contribs = mk[None, :, :] * le[None, :, :] * gauss * proximity
    A = jnp.sum(contribs, axis=2)  # (P, C)
    return np.asarray(A)


def compute_p_snr_jax(
    intensity: np.ndarray,
    peak_indices: np.ndarray,
) -> float:
    """JAX-vectorized counterpart of ``ALIASIdentifier._compute_p_snr``.

    Uses ``jax.scipy.special.erf`` so the call composes inside a future
    batched ``vmap`` without round-tripping through scipy.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity array (1D).
    peak_indices : np.ndarray
        Integer indices of detected peaks.

    Returns
    -------
    float
        ``P_SNR`` in [0, 1].
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for compute_p_snr_jax. Install with: pip install jax jaxlib"
        )
    if len(peak_indices) == 0:
        return 0.5
    inten = jnp.asarray(intensity, dtype=jnp.float64)
    idx = jnp.asarray(peak_indices, dtype=jnp.int32)
    peak_intensities = inten[idx]
    median_peak = jnp.median(peak_intensities)
    noise_estimate = jnp.median(jnp.abs(inten - jnp.median(inten))) * 1.4826
    noise_estimate = jnp.maximum(noise_estimate, 1e-10)
    z = (median_peak - noise_estimate) / (noise_estimate * jnp.sqrt(2.0))
    return float(0.5 * (1.0 + jax.scipy.special.erf(z)))


def _evaluate_candidate_with_multi_metric_gate(
    element: str,
    fused_lines: list,
    matched_mask: np.ndarray,
    matched_peak_idx: np.ndarray,
    wavelength_shifts: np.ndarray,
    intensity: np.ndarray,
    peaks: list,
    emissivity_threshold: float,
    k_sim: float,
    k_rate: float,
    k_shift: float,
    P_maj: float,
    N_expected: int,
    N_matched: int,
    P_cov: float,
    P_mix: float,
    P_local: float,
    R_rat: float,
    boltz_r2: float,
    estimated_T: float,
    resolving_power: float,
) -> dict:
    """Evaluate a candidate element using the multi-metric gate.

    Returns a dict with:
    - detected: bool
    - composite_score: float
    - correctness: float
    - physics_validity: float
    - efficiency: float
    - interpretability: float
    """
    # Correctness (0.4 weight): RMSEP against certified reference values
    # Approximated by: k_sim * k_rate * P_cov (how well we match expected lines)
    correctness = k_sim * k_rate * max(P_cov, 0.01)

    # Physics validity (0.3 weight): Boltzmann R^2, LTE consistency, physical T
    physics_validity = 0.5  # Default neutral
    if N_matched >= 3:
        if boltz_r2 >= 0.85:
            physics_validity = 1.0
        elif boltz_r2 >= 0.7:
            physics_validity = 0.8
        elif boltz_r2 >= 0.5:
            physics_validity = 0.6
        else:
            physics_validity = 0.3
    else:
        # Fewer than 3 lines: can't do Boltzmann, use soft penalty
        physics_validity = 0.5

    # Efficiency (0.15 weight): Computation time proxy
    # Fewer lines matched relative to expected = more efficient
    if N_expected > 0:
        efficiency = N_matched / N_expected
    else:
        efficiency = 0.0

    # Interpretability (0.15 weight): Physical reasonableness
    # Based on ratio consistency and NNLS attribution
    interpretability = 0.5  # Default neutral
    if R_rat > 0.7:
        interpretability = 0.8
    elif R_rat > 0.5:
        interpretability = 0.6
    else:
        interpretability = 0.4

    # Composite score
    composite_score = (
        0.4 * correctness + 0.3 * physics_validity + 0.15 * efficiency + 0.15 * interpretability
    )

    # Detection decision: composite score must exceed threshold
    # and pass hard gates
    detected = False
    if composite_score >= 0.5:
        # Hard gate: minimum matched lines
        if N_matched >= 3:
            detected = True
        elif N_matched >= 2 and k_sim > 0.5:
            detected = True

    return {
        "detected": detected,
        "composite_score": composite_score,
        "correctness": correctness,
        "physics_validity": physics_validity,
        "efficiency": efficiency,
        "interpretability": interpretability,
    }


class ALIASIdentifier:
    """
    ALIAS algorithm for automated element identification in LIBS spectra.

    The algorithm operates in 7 steps:
    1. Peak detection via 2nd derivative enhancement
    2. Theoretical emissivity calculation over (T, n_e) grid
    3. Line fusion within resolution element
    4. Matching theoretical lines to experimental peaks
    5. Emissivity threshold determination via detection rate
    6. Score computation (k_sim, k_rate, k_shift)
    7. Decision and confidence level calculation

    Thread-safety
    -------------
    ``identify()`` mutates instance state (``_effective_R``,
    ``_global_wl_shift``, ``_estimated_T``), so a single instance is
    **not** safe for concurrent calls. Create one instance per thread or
    guard calls with an external lock.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for transitions and partition functions
    resolving_power : float, optional
        Instrument resolving power R = λ/Δλ (default: 5000.0)
    T_range_K : Tuple[float, float], optional
        Temperature grid range in K (default: (8000.0, 12000.0))
    n_e_range_cm3 : Tuple[float, float], optional
        Electron density grid range in cm^-3 (default: (3e16, 3e17))
    T_steps : int, optional
        Number of temperature grid points (default: 5)
    n_e_steps : int, optional
        Number of electron density grid points (default: 3)
    intensity_threshold_factor : float, optional
        Peak detection threshold = factor × noise_estimate. If ``None`` (the
        default), resolved from ``high_recall``: ``2.0`` when
        ``high_recall=True`` and ``3.0`` otherwise (strict mode, the
        precision-king baseline). Passing an explicit float overrides
        ``high_recall`` for this knob.
    detection_threshold : float, optional
        Minimum confidence level for element detection. If ``None`` (the
        default), resolved from ``high_recall``: ``0.01`` when
        ``high_recall=True`` and ``0.02`` otherwise (strict mode, the
        precision-king baseline). Passing an explicit float overrides
        ``high_recall`` for this knob.
    high_recall : bool, optional
        Opt-in preset that loosens the two peak/identification thresholds
        for higher recall at the cost of precision. When ``True`` (and the
        caller has not pinned the corresponding knob explicitly) it lowers
        ``intensity_threshold_factor`` from ``3.0`` to ``2.0`` and
        ``detection_threshold`` from ``0.02`` to ``0.01``. The default
        (``False``) preserves the strict precision-king behavior captured
        in the F1 leaderboard baseline at
        ``.swarm/identifier-f1-baseline.json`` (precision=1.000, FP/spec=0
        on n=33 cross-shard). See the PR for the closed PR #134 that
        silently flipped these defaults — this knob is the opt-in
        replacement (default: False).
    chance_window_scale : float, optional
        Scale factor for chance-coincidence windows used in fill-factor estimation.
        The chance half-window is `chance_window_scale * (lambda / R)`.
    elements : Optional[List[str]], optional
        List of elements to search for. If None, uses default common LIBS elements:
        ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"] (default: None)
    max_screening_candidates : int, optional
        Maximum number of candidates retained by fast screening (default: 12)
    relative_cl_threshold : float, optional
        CL must be >= max_CL * relative_cl_threshold to count as detected.
        Set to 0 to disable the relative threshold (default: 0.1)
    boltzmann_r2_min : float, optional
        Minimum Boltzmann-plot R^2 required for candidates with at least three
        matched lines. Must be finite and in [0, 1]. Candidates with fewer
        than three matched lines are rejected before committing identification
        because no meaningful Boltzmann regression can be performed (default: 0.85).
    """

    # Temperature bounds for physics validation
    _T_ESTIMATE_MIN_K = 3000.0
    _T_ESTIMATE_MAX_K = 30000.0
    # Consistency check uses a wider range because its purpose is only to
    # flag grossly unphysical fits, not to narrow the estimate.
    _T_CONSISTENCY_MIN_K = 3000.0
    _T_CONSISTENCY_MAX_K = 50000.0

    # Crustal abundance in log10(ppm) — from CRC Handbook / USGS
    CRUSTAL_ABUNDANCE_LOG_PPM = {
        "O": 5.67,
        "Si": 5.44,
        "Al": 4.91,
        "Fe": 4.70,
        "Ca": 4.57,
        "Na": 4.36,
        "Mg": 4.33,
        "K": 4.32,
        "Ti": 3.75,
        "H": 3.15,
        "Mn": 2.98,
        "P": 2.97,
        "F": 2.80,
        "Ba": 2.70,
        "C": 2.30,
        "Sr": 2.57,
        "S": 2.56,
        "Zr": 2.23,
        "V": 2.10,
        "Cl": 2.20,
        "Cr": 2.00,
        "Ni": 1.88,
        "Zn": 1.88,
        "Cu": 1.78,
        "Co": 1.40,
        "Li": 1.30,
        "N": 1.30,
        "Ga": 1.28,
        "Pb": 1.15,
        "Rb": 1.95,
        "B": 1.00,
        "Sn": 0.35,
        "W": 0.18,
        "Mo": 0.18,
        "Ag": -0.62,
        "Cd": -0.82,
        "Au": -2.40,
    }

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        resolving_power: float = 5000.0,
        T_range_K: Tuple[float, float] = (5000.0, 15000.0),
        n_e_range_cm3: Tuple[float, float] = (1e16, 5e17),
        T_steps: int = 7,
        n_e_steps: int = 3,
        intensity_threshold_factor: Optional[float] = None,
        detection_threshold: Optional[float] = None,
        chance_window_scale: float = 0.4,
        elements: Optional[List[str]] = None,
        max_lines_per_element: int = 20,
        reference_temperature: float = 10000.0,
        max_screening_candidates: int = 12,
        relative_cl_threshold: float = 0.1,
        relative_cl_per_ion_stage: bool = False,
        relative_cl_threshold_neutral: Optional[float] = None,
        relative_cl_threshold_ionized: Optional[float] = None,
        boltzmann_r2_min: float = 0.85,
        r2_gate_mode: str = "fixed",
        r2_gate_t_quality_threshold: float = 5500.0,
        high_recall: bool = False,
        use_jax_boltzmann_fit: bool = False,
        use_jax_nnls: bool = False,
        use_jax_p_snr: bool = False,
        use_jax_template_build: bool = False,
        self_absorption_aware: bool = True,
        self_absorption_damping: float = 0.3,
        self_absorption_e_i_cutoff_ev: float = 0.1,
        temperature_estimator_mode: str = "legacy",
    ):
        """
        Initialize the ALIAS element identifier.

        Parameters
        ----------
        atomic_db : AtomicDatabase
            Atomic database used to retrieve transition and level data.
        resolving_power : float, optional
            Instrument resolving power used when modeling line widths and matching
            observed to theoretical lines.
        T_range_K : Tuple[float, float], optional
            Temperature search range in kelvin for the Saha-Boltzmann grid.
        n_e_range_cm3 : Tuple[float, float], optional
            Electron density search range in cm^-3 for the Saha-Boltzmann grid.
        T_steps : int, optional
            Number of temperature grid points to evaluate.
        n_e_steps : int, optional
            Number of electron density grid points to evaluate.
        intensity_threshold_factor : Optional[float], optional
            Multiplier applied to the estimated noise level when detecting peaks.
            If ``None`` (default), the value is resolved from ``high_recall``:
            ``2.0`` when ``high_recall=True``, ``3.0`` otherwise (strict default
            that preserves the precision-king baseline). Pass an explicit float
            to override ``high_recall`` for this knob.
        detection_threshold : Optional[float], optional
            Minimum normalized line strength considered during identification.
            If ``None`` (default), the value is resolved from ``high_recall``:
            ``0.01`` when ``high_recall=True``, ``0.02`` otherwise (strict
            default that preserves the precision-king baseline). Pass an
            explicit float to override ``high_recall`` for this knob.
        high_recall : bool, optional
            Opt-in preset that loosens the two peak/identification thresholds
            for higher recall at the cost of precision. When ``True``, the
            unspecified threshold knobs become
            ``intensity_threshold_factor=2.0`` and ``detection_threshold=0.01``
            (compare to the strict ``3.0`` / ``0.02`` defaults). The default
            (``False``) preserves the strict precision-king behavior measured
            on the n=33 cross-shard F1 leaderboard (precision=1.000,
            FP/spec=0). This is the opt-in replacement for the closed PR #134
            silent-default-change; see that PR's discussion for context.
            (default: False)
        chance_window_scale : float, optional
            Scale factor controlling the wavelength window used in chance-match
            calculations.
        elements : Optional[List[str]], optional
            Restrict identification to this subset of element symbols. If ``None``,
            all supported elements may be considered.
        max_lines_per_element : int, optional
            Maximum number of theoretical lines retained per element.
        reference_temperature : float, optional
            Reference temperature used when ranking or screening candidate lines.
        max_screening_candidates : int, optional
            Maximum number of candidate elements retained after screening.
        relative_cl_threshold : float, optional
            Minimum relative confidence level required for an element to be kept.
        relative_cl_per_ion_stage : bool, optional
            Opt-in switch for the per-ion-stage relative-CL gate
            (CF-LIBS-improved-dj6y).  When ``False`` (default) the historical
            global gate runs: every element's CL is compared to
            ``max_CL * relative_cl_threshold`` over the full element list.
            When ``True``, elements are split into neutrals (dominant
            matched ion stage = 1) and ionized (dominant matched ion stage
            >= 2), and each subset is gated against its own subset-max so
            that a high-CL dominant neutral cannot kill lower-CL ionized
            species (the Vrabel s019 Mg failure mode). The default preserves
            the precision-king baseline; flip explicitly to opt in.
            (default: False)
        relative_cl_threshold_neutral : Optional[float], optional
            Neutral-subset threshold used when
            ``relative_cl_per_ion_stage=True``. Must be in ``[0.0, 1.0]``.
            Falls back to ``relative_cl_threshold`` when ``None``.
            (default: None)
        relative_cl_threshold_ionized : Optional[float], optional
            Ionized-subset threshold used when
            ``relative_cl_per_ion_stage=True``. Must be in ``[0.0, 1.0]``.
            Falls back to ``relative_cl_threshold`` when ``None``.
            (default: None)
        boltzmann_r2_min : float, optional
            Minimum acceptable coefficient of determination (R^2) for Boltzmann-plot
            consistency checks used during identification. Higher values make
            identification stricter by requiring better linear agreement, while
            lower values allow more permissive acceptance of candidate elements.
        r2_gate_mode : str, optional
            How the ``boltzmann_r2_min`` gate is applied to candidates with
            ``N_matched >= 3``. One of:

            - ``"fixed"`` (default, byte-identical to historical behavior):
              candidates are rejected when ``boltz_r2 < self.boltzmann_r2_min``
              (the static 0.85 default). Preserves the precision-king
              baseline (precision=1.000, FP/spec=0 on n=33 cross-shard).
            - ``"adaptive_t"``: temperature-aware relaxation. For cold
              plasma (``self._estimated_T < r2_gate_t_quality_threshold``),
              the gate floor is relaxed to ``0.3`` so otherwise-valid
              candidates whose Boltzmann fit is degraded by short
              effective E_k spans (a known cold-T pathology surfaced by
              the Vrabel diagnosis, PR #172) still clear. For warm plasma
              (``T >= threshold``) the gate keeps the strict
              ``boltzmann_r2_min`` floor.
            - ``"disabled"``: bypass the R^2 gate entirely. Intended as a
              control-cell measurement for sweep analysis; NOT a sensible
              production default — disables one of the strongest false-
              positive suppressors.

            (default: ``"fixed"``)
        r2_gate_t_quality_threshold : float, optional
            Temperature in kelvin below which ``r2_gate_mode="adaptive_t"``
            switches to the relaxed R^2 floor. Must be finite and > 0.
            Ignored when ``r2_gate_mode != "adaptive_t"``. The default of
            ``5500.0`` K was chosen from the Vrabel-style cold-plasma
            regime identified in PR #172's universal-miss diagnosis.
            (default: 5500.0)
        use_jax_boltzmann_fit : bool, optional
            If True, use the JAX-vectorized closed-form least-squares
            Boltzmann fit (:func:`boltzmann_temperature_jax`) inside
            ``_estimate_plasma_temperature`` instead of
            ``scipy.stats.linregress``. The two paths produce numerically
            equivalent results on negative-slope inputs (agreement
            ~1e-5 relative) and the same ``inf``/``nan`` sentinels on
            non-negative-slope or degenerate inputs. The JAX path is
            opt-in (default False) so existing benchmark results are
            unchanged; turning it on enables future batched-spectrum
            speedups. Requires ``jax`` to be importable; raises
            ``ImportError`` at fit time otherwise (default: False).
        use_jax_nnls : bool, optional
            If True, use the JAX FISTA-with-restart NNLS solver
            (:func:`solve_nnls_jax`, :func:`solve_sparse_nnls_jax`,
            :func:`compute_nnls_attribution_jax`) for the per-spectrum
            attribution / sparse-NNLS / iron-group-subtraction loops in
            :meth:`identify`. Numerical agreement with the SciPy
            active-set solver is ~1e-13 on well-conditioned LIBS template
            matrices, dropping to ~1e-5 on highly correlated columns
            (where the active-set and projected-gradient minimizers may
            distribute mass differently across redundant columns while
            achieving the same residual). Opt-in (default False).
            Requires JAX (default: False).
        use_jax_p_snr : bool, optional
            If True, use :func:`compute_p_snr_jax` (``jax.scipy.special.erf``)
            for the ``P_SNR`` quality factor in :meth:`_decide` instead of
            ``scipy.special.erf``. Opt-in (default False). Requires JAX
            (default: False).
        use_jax_template_build : bool, optional
            If True, use :func:`build_nnls_templates_jax` (broadcasting +
            ``vmap``) to build the NNLS template matrix instead of the
            Python loop over candidates. Numerical agreement is exact
            (same arithmetic, just reordered). Opt-in (default False).
            Requires JAX (default: False).
        self_absorption_aware : bool, optional
            When True (default), down-weight the theoretical emissivity of
            resonance lines (``E_i_ev < self_absorption_e_i_cutoff_ev``) by
            ``self_absorption_damping`` inside the k_sim cosine-similarity
            and the log-ratio consistency score (R_rat). This compensates
            for the fact that strong resonance lines are systematically
            weaker than the optically-thin Saha-Boltzmann prediction in
            high-concentration matrices (e.g. Si in soil at 60% SiO2,
            where the Si I 251.6 nm / 288.2 nm lines become optically
            thick and line-reverse). When False, the score uses the raw
            emissivity values from the Saha-Boltzmann solver — this is the
            mathematically pure optically-thin assumption and matches the
            paper-faithful ALIAS behavior, but it penalizes the cosine
            similarity of high-concentration major elements. The default
            ``True`` preserves the pre-CF-LIBS-improved-fix behavior — the
            historical code had ``SA_DAMPING = 0.3`` hardcoded inline at
            two call sites without any documentation, flag, or logging.
            (default: True)
        self_absorption_damping : float, optional
            Multiplicative damping factor applied to theoretical emissivity
            of resonance lines when ``self_absorption_aware`` is True.
            Must be in (0, 1]. 1.0 means "no damping" and is equivalent to
            ``self_absorption_aware=False``. The historical default of 0.3
            says "resonance lines arrive ~3x weaker than the optically-
            thin prediction"; tune this only if you have evidence the
            value should differ for your dataset (default: 0.3).
        self_absorption_e_i_cutoff_ev : float, optional
            Lower-level energy threshold (in eV) below which a line is
            treated as a "resonance line" for self-absorption damping.
            Lines with ``E_i_ev < this`` get damped; lines above it are
            untouched. The historical default of 0.1 eV catches genuine
            ground-state and metastable transitions while leaving truly
            excited transitions alone (default: 0.1).
        temperature_estimator_mode : str, optional
            Which Pass-1 Boltzmann-slope strategy
            ``_estimate_plasma_temperature`` uses. One of:

            - ``"legacy"`` (default, byte-identical): the historical
              3-pass algorithm — unweighted ``np.polyfit``/``linregress``
              over the matched lines, returns the first element whose
              ``slope<0`` and ``r_sq>0.2``.
            - ``"robust"``: per-line intensity-weighted regression
              (``sigma_y ≈ sigma_I/I`` with a shot-noise proxy
              ``sigma_I ≈ max(noise, sqrt(I))``) — deprioritizes the
              noisy high-``E_k`` lines that bias the unweighted slope
              cold (see ``docs/research/vrabel-universal-miss-root-
              cause-2026-05-14.md``). After Pass-1 collects all
              candidate T_K values, the median across elements is
              accepted if 3+ elements fall within a 2000 K window;
              otherwise the legacy Pass-2/3 line-ratio fallback runs.
              Emits an INFO log line summarizing per-element T_K and
              the selected value.
            - ``"weighted"``: simpler heuristic — drop the bottom
              quartile of matched lines by SNR before the unweighted
              slope fit. No cross-element consistency check.

            Must be one of the strings above. The default ``"legacy"``
            preserves byte-stable behavior for all existing benchmarks;
            ``"robust"`` and ``"weighted"`` are opt-in for the Vrabel
            universal-miss root cause investigation
            (default: ``"legacy"``).
        """
        self.atomic_db = atomic_db
        if not (np.isfinite(resolving_power) and resolving_power > 0):
            raise ValueError(f"resolving_power must be finite and > 0, got {resolving_power!r}")
        self.resolving_power = float(resolving_power)
        self.T_range_K = T_range_K
        self.n_e_range_cm3 = n_e_range_cm3
        self.T_steps = T_steps
        self.n_e_steps = n_e_steps
        # Resolve threshold defaults from `high_recall` preset.
        # Strict mode (default): preserves the precision-king baseline
        #   precision=1.000, FP/spec=0 on n=33 cross-shard
        #   (see .swarm/identifier-f1-baseline.json).
        # Recall mode (opt-in): lowers both thresholds to trade precision
        #   for recall. This is the opt-in replacement for the closed
        #   PR #134, which silently flipped these defaults.
        # Explicit user-supplied values always win, so callers can pin
        # either knob independently of the preset.
        self.high_recall = bool(high_recall)
        _STRICT_INTENSITY_FACTOR = 3.0
        _STRICT_DETECTION_THRESHOLD = 0.02
        _RECALL_INTENSITY_FACTOR = 2.0
        _RECALL_DETECTION_THRESHOLD = 0.01
        if intensity_threshold_factor is None:
            self.intensity_threshold_factor = (
                _RECALL_INTENSITY_FACTOR if self.high_recall else _STRICT_INTENSITY_FACTOR
            )
        else:
            self.intensity_threshold_factor = intensity_threshold_factor
        if detection_threshold is None:
            self.detection_threshold = (
                _RECALL_DETECTION_THRESHOLD if self.high_recall else _STRICT_DETECTION_THRESHOLD
            )
        else:
            self.detection_threshold = detection_threshold
        self.chance_window_scale = chance_window_scale
        self.elements = elements
        self.max_lines_per_element = max_lines_per_element
        self.reference_temperature = reference_temperature
        self.max_screening_candidates = max_screening_candidates
        self.relative_cl_threshold = relative_cl_threshold
        # Per-ion-stage relative-CL gate knobs (CF-LIBS-improved-dj6y).
        # Defaults preserve the historical global gate behavior. When the
        # opt-in flag is on, neutrals and ionized species are gated against
        # separate subset-maxima so a high-CL neutral (e.g. Al I) cannot
        # eliminate a lower-CL ion (e.g. Mg II) via the global threshold.
        self.relative_cl_per_ion_stage = bool(relative_cl_per_ion_stage)
        for _kw_name, _kw_val in (
            ("relative_cl_threshold_neutral", relative_cl_threshold_neutral),
            ("relative_cl_threshold_ionized", relative_cl_threshold_ionized),
        ):
            if _kw_val is None:
                continue
            if not (np.isfinite(_kw_val) and 0.0 <= _kw_val <= 1.0):
                raise ValueError(
                    f"{_kw_name} must be finite and in [0.0, 1.0], " f"got {_kw_val!r}"
                )
        self.relative_cl_threshold_neutral = (
            float(relative_cl_threshold_neutral)
            if relative_cl_threshold_neutral is not None
            else None
        )
        self.relative_cl_threshold_ionized = (
            float(relative_cl_threshold_ionized)
            if relative_cl_threshold_ionized is not None
            else None
        )
        if not (np.isfinite(boltzmann_r2_min) and 0.0 <= boltzmann_r2_min <= 1.0):
            raise ValueError(
                f"boltzmann_r2_min must be finite and in [0, 1], got {boltzmann_r2_min!r}"
            )
        self.boltzmann_r2_min = float(boltzmann_r2_min)
        # Adaptive R^2 gate (CF-LIBS-improved-ftp1).
        # Default mode "fixed" preserves the historical static-0.85 gate
        # byte-identically, which keeps the alias precision-king
        # invariant (precision=1.000, FP/spec=0 on n=33 cross-shard).
        # See the docstring for the rationale on adaptive_t / disabled.
        _R2_GATE_MODES = ("fixed", "adaptive_t", "disabled")
        if r2_gate_mode not in _R2_GATE_MODES:
            raise ValueError(f"r2_gate_mode must be one of {_R2_GATE_MODES}, got {r2_gate_mode!r}")
        self.r2_gate_mode = r2_gate_mode
        if not (np.isfinite(r2_gate_t_quality_threshold) and r2_gate_t_quality_threshold > 0):
            raise ValueError(
                f"r2_gate_t_quality_threshold must be finite and > 0, "
                f"got {r2_gate_t_quality_threshold!r}"
            )
        self.r2_gate_t_quality_threshold = float(r2_gate_t_quality_threshold)
        # Cold-plasma R^2 floor when adaptive_t mode fires. Internal
        # constant — surfaces in the gate logic in identify() below.
        self._r2_gate_cold_floor = 0.3
        self.use_jax_boltzmann_fit = bool(use_jax_boltzmann_fit)
        self.use_jax_nnls = bool(use_jax_nnls)
        self.use_jax_p_snr = bool(use_jax_p_snr)
        self.use_jax_template_build = bool(use_jax_template_build)
        # Self-absorption scoring knobs (CF-LIBS-improved-self-abs-audit).
        # Defaults preserve the historical behavior — see the docstring.
        self.self_absorption_aware = bool(self_absorption_aware)
        if not (np.isfinite(self_absorption_damping) and 0.0 < self_absorption_damping <= 1.0):
            raise ValueError(
                f"self_absorption_damping must be finite and in (0, 1], "
                f"got {self_absorption_damping!r}"
            )
        self.self_absorption_damping = float(self_absorption_damping)
        if not (
            np.isfinite(self_absorption_e_i_cutoff_ev) and self_absorption_e_i_cutoff_ev >= 0.0
        ):
            raise ValueError(
                f"self_absorption_e_i_cutoff_ev must be finite and >= 0, "
                f"got {self_absorption_e_i_cutoff_ev!r}"
            )
        self.self_absorption_e_i_cutoff_ev = float(self_absorption_e_i_cutoff_ev)
        # Counters reset on every identify() call so a single dispatch's
        # damping behavior is auditable from the post-run log line.
        self._sa_n_damped_lines: int = 0
        self._sa_damped_elements: Set[str] = set()
        # Temperature-estimator strategy (CF-LIBS-improved-762f).
        # See the docstring for full rationale.
        _valid_modes = ("legacy", "robust", "weighted")
        if temperature_estimator_mode not in _valid_modes:
            raise ValueError(
                f"temperature_estimator_mode must be one of {_valid_modes}, "
                f"got {temperature_estimator_mode!r}"
            )
        self.temperature_estimator_mode = str(temperature_estimator_mode)
        _any_jax = (
            self.use_jax_boltzmann_fit
            or self.use_jax_nnls
            or self.use_jax_p_snr
            or self.use_jax_template_build
        )
        if _any_jax and not _HAS_JAX:  # pragma: no cover
            raise ImportError("use_jax_* flags require JAX. Install with: pip install jax jaxlib")

        # Create Saha-Boltzmann solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        # Create (T, n_e) grid
        self.T_grid_K = np.linspace(T_range_K[0], T_range_K[1], T_steps)
        self.n_e_grid_cm3 = np.linspace(n_e_range_cm3[0], n_e_range_cm3[1], n_e_steps)

        # Set during identify() by auto-calibration
        self._effective_R: Optional[float] = None
        self._global_wl_shift: float = 0.0

        # Ubiquitous atmospheric/ablation contaminants always tested
        self._always_test: Set[str] = {"H"}

        # Estimated plasma temperature (set by _estimate_plasma_temperature)
        self._estimated_T: Optional[float] = None

    def _apply_relative_cl_gate(self, all_element_ids: list) -> None:
        """Mutate ``e.detected`` in place per the relative-CL gate.

        Two modes:

        * Global (default, ``relative_cl_per_ion_stage=False``): one
          ``max_CL`` is taken over all elements and every element is gated
          against ``max_CL * relative_cl_threshold``. This is the
          precision-king baseline; it is also the failure mode diagnosed
          on Vrabel s019 (PR #172) where a high-CL Al I dominates and
          silently kills Mg II at CL ~ 0.026.
        * Per-ion-stage (CF-LIBS-improved-dj6y, opt-in via
          ``relative_cl_per_ion_stage=True``): each element's "dominant"
          ion stage is the ionization stage of its highest-intensity
          matched line. Elements split into neutrals (dominant stage = 1)
          and ionized (dominant stage >= 2). Each subset is gated against
          its own subset-max with its own threshold
          (``relative_cl_threshold_neutral`` /
          ``relative_cl_threshold_ionized``, both falling back to
          ``relative_cl_threshold``). A subset with zero members simply
          skips its branch — no crash, no cross-subset contamination.
          Elements with zero matched lines have no ion-stage to arbitrate
          on; they fall back to the global gate so the opt-in does not
          silently exempt them from suppression.
        """
        if not all_element_ids or self.relative_cl_threshold <= 0:
            return

        if not self.relative_cl_per_ion_stage:
            max_CL = max(e.confidence for e in all_element_ids)
            relative_threshold = max_CL * self.relative_cl_threshold
            for e in all_element_ids:
                if e.confidence < relative_threshold:
                    e.detected = False
            return

        neutral_threshold = (
            self.relative_cl_threshold_neutral
            if self.relative_cl_threshold_neutral is not None
            else self.relative_cl_threshold
        )
        ionized_threshold = (
            self.relative_cl_threshold_ionized
            if self.relative_cl_threshold_ionized is not None
            else self.relative_cl_threshold
        )

        def _dominant_ion_stage(elem_id) -> Optional[int]:
            if not elem_id.matched_lines:
                return None
            dominant = max(elem_id.matched_lines, key=lambda ln: ln.intensity_exp)
            return int(dominant.ionization_stage)

        neutrals = []
        ionized = []
        unclassified = []
        for e in all_element_ids:
            stage = _dominant_ion_stage(e)
            if stage is None:
                unclassified.append(e)
            elif stage <= 1:
                neutrals.append(e)
            else:
                ionized.append(e)

        if neutrals:
            max_CL_neutral = max(e.confidence for e in neutrals)
            relative_threshold_n = max_CL_neutral * neutral_threshold
            for e in neutrals:
                if e.confidence < relative_threshold_n:
                    e.detected = False
        if ionized:
            max_CL_ionized = max(e.confidence for e in ionized)
            relative_threshold_i = max_CL_ionized * ionized_threshold
            for e in ionized:
                if e.confidence < relative_threshold_i:
                    e.detected = False
        if unclassified:
            max_CL_all = max(e.confidence for e in all_element_ids)
            relative_threshold_u = max_CL_all * self.relative_cl_threshold
            for e in unclassified:
                if e.confidence < relative_threshold_u:
                    e.detected = False

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        spectrum_id: Optional[str] = None,
    ) -> ElementIdentificationResult:
        """
        Identify elements in experimental spectrum with cross-element peak
        competition.

        Enhanced multi-phase algorithm:
        0. Baseline correction + peak detection
        0a. Wavelength auto-calibration (estimate global shift + effective R)
        0b. Plasma temperature estimation (for adaptive emissivities)
        0c. Fast screening (restrict candidate elements)
        1. Score screened elements independently
        2. Global peak competition
        3. Rescore, Boltzmann consistency check, build results

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array (arbitrary units)
        spectrum_id : str, optional
            Caller-supplied identifier for this spectrum.  Threaded
            through to detection-coverage log records (L2/L3/L4) so
            downstream parsing can correlate per-element coverage data
            with the source spectrum.  Identifier behaviour is
            unchanged when this is left ``None``.

        Returns
        -------
        ElementIdentificationResult
            Complete identification result with detected/rejected elements
        """
        # Reset per-dispatch self-absorption damping counters so the
        # summary log line below reflects ONLY this identify() call.
        self._sa_n_damped_lines = 0
        self._sa_damped_elements = set()

        # Detection-coverage tracker -- additive telemetry only.
        coverage = CoverageTracker(
            spectrum_id=spectrum_id if spectrum_id is not None else "<unset>",
            identifier_name="alias",
        )

        # Step 0: Baseline correction — ALL scoring uses corrected intensities
        # so that cosine similarity, NNLS, and P_SNR measure peak heights above
        # continuum rather than absolute intensity that is dominated by the
        # Bremsstrahlung background.
        baseline = estimate_baseline(wavelength, intensity)
        corrected_intensity = np.maximum(intensity - baseline, 0.0)

        # Step 1: Detect peaks (uses its own internal baseline correction)
        peaks = self._detect_peaks(wavelength, intensity)

        wl_min = np.min(wavelength)
        wl_max = np.max(wavelength)

        # Step 0a: Auto-calibrate wavelength (estimate global shift + effective R)
        self._auto_calibrate_wavelength(peaks, wl_min, wl_max)

        # Step 0b: Estimate plasma temperature from detected peaks
        self._estimate_plasma_temperature(peaks, corrected_intensity, wl_min, wl_max)

        # Get elements to search
        if self.elements is None:
            # Prefer database-provided availability when possible.
            get_available = getattr(self.atomic_db, "get_available_elements", None)
            if callable(get_available):
                try:
                    available = list(get_available())
                except Exception:
                    available = []
                search_elements = available or ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"]
            else:
                search_elements = ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"]
        else:
            search_elements = self.elements

        # Step 0c: Fast screening — restrict to elements with strong-line matches
        # Skip screening when user explicitly provided a short element list
        if self.elements is not None and len(self.elements) <= 10:
            screened = search_elements
        else:
            screened = self._fast_screening(search_elements, peaks, wl_min, wl_max)

        # Record fast-screened-out elements as zero-match / fingerprint
        # failures so the coverage payload surfaces every candidate the
        # caller asked about, not just those that survived screening.
        # We deliberately do *not* record an L2 zero here: fast-screening
        # rejects on "no strong-line match", not on "DB has zero lines
        # in window", and conflating the two would mis-classify the
        # failure layer.  Recording an L3 zero (with no L2 row) leaves
        # ``elements_with_zero_peak_matches`` populated -- which is the
        # smoking gun the task is after -- without falsifying the L2
        # column.  Telemetry is additive; identifier behaviour is
        # unchanged.
        screened_set = set(screened)
        for element in search_elements:
            if element not in screened_set:
                coverage.record_peak_matches(element, 0)
                coverage.record_fingerprint(element, passed=False, score=0.0)

        # ── Phase 1: Independent scoring ──────────────────────────────
        # Use corrected_intensity throughout scoring so continuum doesn't
        # dominate cosine similarity and NNLS attribution.
        global_p_snr = self._dispatch_p_snr(corrected_intensity, peaks)
        candidates: List[dict] = []

        for element in screened:
            element_lines = self._compute_element_emissivities(
                element, wl_min, wl_max, T_estimated=self._estimated_T
            )
            # L2 -- per-element line presence in DB (post-alias filter).
            # alias selects observable lines (A_ki*g_k >= 1e4) and caps
            # to ``max_lines_per_element`` strongest, so we record the
            # count the matcher will actually see.
            coverage.record_db_lines(element, len(element_lines))
            if not element_lines:
                # L3/L4 are unreachable when there is nothing to match;
                # record the zero-state so the summary surfaces it.
                coverage.record_peak_matches(element, 0)
                coverage.record_fingerprint(element, passed=False, score=0.0)
                continue

            fused_lines = self._fuse_lines(element_lines, wavelength)
            if not fused_lines:
                coverage.record_peak_matches(element, 0)
                coverage.record_fingerprint(element, passed=False, score=0.0)
                continue

            matched_mask, wavelength_shifts, matched_peak_idx = self._match_lines(
                fused_lines, peaks
            )
            # L3 -- per-element peak match.  ``matched_mask`` is the
            # bool array of fused lines that paired with any detected
            # peak inside tolerance.
            coverage.record_peak_matches(element, int(np.sum(matched_mask)))

            if np.any(matched_mask):
                emissivity_threshold = self._determine_emissivity_threshold(
                    fused_lines, matched_mask
                )
            else:
                emissivity_threshold = -np.inf

            k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = self._compute_scores(
                fused_lines,
                matched_mask,
                matched_peak_idx,
                wavelength_shifts,
                corrected_intensity,
                peaks,
                emissivity_threshold,
            )

            P_sig, fill_factor, p_chance, p_tail = self._compute_random_match_significance(
                peaks=peaks,
                wavelength=wavelength,
                N_expected=N_expected,
                N_matched=N_matched,
            )

            k_det, CL = self._decide(
                k_sim,
                k_rate,
                k_shift,
                N_expected,
                corrected_intensity,
                peaks,
                element=element,
                P_maj=P_maj,
                P_sig=P_sig,
                N_matched=N_matched,
                P_cov=P_cov,
            )

            candidates.append(
                {
                    "element": element,
                    "fused_lines": fused_lines,
                    "matched_mask": matched_mask,
                    "matched_peak_idx": matched_peak_idx,
                    "wavelength_shifts": wavelength_shifts,
                    "emissivity_threshold": emissivity_threshold,
                    "initial_CL": CL,
                    # Cache phase 1 scores for reuse when competition is skipped
                    "scores": (k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov),
                    "N_matched": N_matched,
                    "P_sig_data": (P_sig, fill_factor, p_chance, p_tail),
                    "k_det": k_det,
                    "P_SNR": global_p_snr,
                }
            )

        # ── Phase 1.5: NNLS peak-space mixture attribution ────────────
        # Non-negative least squares fit of all candidate templates against
        # observed peak intensities.  Returns three metrics:
        #   P_mix  — leave-one-out partial R^2 (global)
        #   P_local — local explanation score (what fraction of claimed
        #             peaks' intensity does this element actually explain?)
        peak_intensities_arr = None
        A = None
        if candidates and peaks:
            peak_intensities_arr = np.array([corrected_intensity[p[0]] for p in peaks])
            if self.use_jax_template_build:
                A = self._build_nnls_templates_jax_wrapper(candidates, peaks)
            else:
                A = self._build_nnls_templates(candidates, peaks)
            if self.use_jax_nnls:
                P_mix_arr, P_local_arr, _ = compute_nnls_attribution_jax(A, peak_intensities_arr)
            else:
                P_mix_arr, P_local_arr, _ = self._compute_nnls_attribution(A, peak_intensities_arr)

            # Sparse NNLS: L1-penalized fit suppresses diffuse FPs
            # Higher alpha at low RP where blending causes more false sharing
            sparse_alpha = 0.05 if self.resolving_power < 1000 else 0.01
            if self.use_jax_nnls:
                sparse_c, _ = solve_sparse_nnls_jax(A, peak_intensities_arr, alpha=sparse_alpha)
            else:
                sparse_c, _ = self._compute_sparse_nnls_scores(
                    A, peak_intensities_arr, alpha=sparse_alpha
                )
            # Noise floor: coefficient must exceed 10% of median to be
            # considered significant — elements below this are noise
            nonzero_c = sparse_c[sparse_c > 0]
            nnls_noise = float(np.median(nonzero_c) * 0.1) if len(nonzero_c) > 0 else 0.0

            for i, cand in enumerate(candidates):
                cand["P_mix"] = float(P_mix_arr[i])
                cand["P_local"] = float(P_local_arr[i])
                cand["sparse_nnls_coeff"] = float(sparse_c[i])
                cand["nnls_significant"] = float(sparse_c[i]) > nnls_noise
        else:
            for cand in candidates:
                cand["P_mix"] = 1.0
                cand["P_local"] = 1.0
                cand["sparse_nnls_coeff"] = 0.0
                cand["nnls_significant"] = False

        # ── Phase 1.75: Iron-group pre-subtraction (ChemCam-style) ────
        # At low RP, Fe/Mn/Cr/Ti create a dense pseudo-continuum that
        # inflates other elements' NNLS ownership scores.  Subtract
        # their predicted contribution from peak intensities and
        # recompute P_local for non-iron-group elements so the gate
        # discriminates on the residual, not the raw spectrum.
        _IRON_GROUP = {"Fe", "Mn", "Cr", "Ti"}
        if candidates and peaks and A is not None and self.resolving_power < 2000:
            ig_indices = [i for i, c in enumerate(candidates) if c["element"] in _IRON_GROUP]
            if ig_indices and peak_intensities_arr is not None:
                ig_contribution = np.zeros_like(peak_intensities_arr)
                c_nnls = np.zeros(len(candidates))
                # Re-solve NNLS to get coefficients
                try:
                    if self.use_jax_nnls:
                        c_nnls = solve_nnls_jax(A, peak_intensities_arr)
                    else:
                        from scipy.optimize import nnls as _nnls

                        c_nnls, _ = _nnls(A, peak_intensities_arr)
                except Exception:
                    pass
                for idx in ig_indices:
                    ig_contribution += c_nnls[idx] * A[:, idx]

                # Compute residual peak intensities
                residual_peaks = np.maximum(peak_intensities_arr - ig_contribution, 0.0)
                residual_total = float(np.sum(residual_peaks))

                if residual_total > 0:
                    # Recompute P_local for non-iron-group elements against residual
                    for i, cand in enumerate(candidates):
                        if cand["element"] in _IRON_GROUP:
                            continue
                        claimed = A[:, i] > 1e-6
                        if not np.any(claimed):
                            continue
                        obs_residual = np.sum(residual_peaks[claimed])
                        if obs_residual <= 0:
                            cand["P_local"] = 0.0
                            continue
                        elem_contrib = np.sum(A[claimed, i] * c_nnls[i])
                        cand["P_local"] = float(np.clip(elem_contrib / obs_residual, 0.0, 1.0))

        # ── Phase 2: Global peak competition ──────────────────────────
        # Only active at RP >= 2000 where peaks are narrow enough for
        # meaningful exclusivity.  At low RP (broadband spectrometers),
        # shared peaks are the norm and winner-take-all competition
        # causes false negatives for real minor elements.
        if self.resolving_power >= 2000:
            peak_claims: dict = defaultdict(list)

            for c_idx, cand in enumerate(candidates):
                mask = cand["matched_mask"]
                pidx_arr = cand["matched_peak_idx"]
                for l_idx in range(len(mask)):
                    if mask[l_idx]:
                        pidx = int(pidx_arr[l_idx])
                        peak_claims[pidx].append((cand["initial_CL"], c_idx, l_idx))

            # Resolve: highest initial CL wins; losers get unmatched
            for pidx, claims in peak_claims.items():
                if len(claims) <= 1:
                    continue
                claims.sort(key=lambda x: x[0], reverse=True)
                for i in range(1, len(claims)):
                    _, loser_c, loser_l = claims[i]
                    candidates[loser_c]["matched_mask"][loser_l] = False
                    candidates[loser_c]["matched_peak_idx"][loser_l] = -1
                    candidates[loser_c]["wavelength_shifts"][loser_l] = 0.0

        # ── Phase 3: Rescore & build results ──────────────────────────
        competition_ran = self.resolving_power >= 2000
        all_element_ids = []

        for cand in candidates:
            element = cand["element"]
            fused_lines = cand["fused_lines"]
            matched_mask = cand["matched_mask"]
            matched_peak_idx = cand["matched_peak_idx"]
            wavelength_shifts = cand["wavelength_shifts"]
            emissivity_threshold = cand["emissivity_threshold"]

            if competition_ran:
                # Rescore with post-competition matches
                k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = self._compute_scores(
                    fused_lines,
                    matched_mask,
                    matched_peak_idx,
                    wavelength_shifts,
                    corrected_intensity,
                    peaks,
                    emissivity_threshold,
                )

                P_sig, fill_factor, p_chance, p_tail = self._compute_random_match_significance(
                    peaks=peaks,
                    wavelength=wavelength,
                    N_expected=N_expected,
                    N_matched=N_matched,
                )

                k_det, CL = self._decide(
                    k_sim,
                    k_rate,
                    k_shift,
                    N_expected,
                    corrected_intensity,
                    peaks,
                    element=element,
                    P_maj=P_maj,
                    P_sig=P_sig,
                    N_matched=N_matched,
                    P_cov=P_cov,
                )
            else:
                # No competition — reuse phase 1 scores
                k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = cand["scores"]
                P_sig, fill_factor, p_chance, p_tail = cand["P_sig_data"]
                k_det = cand["k_det"]
                CL = cand["initial_CL"]

            P_SNR = cand["P_SNR"] if not competition_ran else global_p_snr

            # ── Post-CL discriminators ──────────────────────────────────
            # Two NNLS-derived gates suppress false positives whose peaks
            # ride on a dominant element's lines:
            #
            # 1. P_local (NNLS peak ownership): fraction of claimed peaks'
            #    intensity that this element's NNLS coefficient explains.
            #    FP elements ride on dominant-element peaks → P_local ~ 0.
            # 2. P_mix (leave-one-out partial R²): how much total spectrum
            #    energy is uniquely attributable to this element.
            #    FP elements add nothing → P_mix ~ 0.
            #
            # R_rat (intensity-ratio consistency) provides a soft additional
            # check: do observed ratios match predicted emissivity ratios?
            #
            # NOTE: P_sig (binomial significance) is deliberately excluded.
            # At low RP (high fill factor), line-rich elements like Fe have
            # match counts BELOW random expectation, giving P_sig → 0 for
            # true positives. P_sig only works at high RP / low fill factor.

            P_mix = cand.get("P_mix", 1.0)
            P_local = cand.get("P_local", 1.0)

            R_rat = self._compute_ratio_consistency(
                fused_lines,
                matched_mask,
                matched_peak_idx,
                corrected_intensity,
                peaks,
            )

            # Post-CL discriminators — suppress false positives whose peaks
            # ride on a dominant element's lines.

            # Gate 1: P_local — soft ramp with 0.25 floor
            CL *= float(np.clip(P_local + 0.25, 0.25, 1.0))

            # Hard rejection: negligible NNLS ownership means this element's
            # peaks are fully explained by other elements.
            # Adaptive threshold: line-rich elements (Fe, Ca, Mn) overlap
            # heavily at low RP, driving P_local artificially low even for
            # true positives.  Use a softer threshold for them.
            # Strong multi-line evidence (high match rate + decent k_sim)
            # can bypass P_local entirely — the element matched most of
            # its lines with consistent intensities.
            p_local_threshold = 0.01 if N_expected >= 10 else 0.05
            high_match_evidence = N_expected >= 5 and N_matched >= 0.7 * N_expected and k_sim > 0.3
            if P_local < p_local_threshold and not high_match_evidence:
                CL = 0.0

            # Gate 2: P_mix — moderate gate, 0.2 floor
            # True minor elements have P_mix ~0.02-0.10, FPs have ~0.000-0.005
            CL *= float(np.clip(0.2 + 8.0 * P_mix, 0.2, 1.0))

            # Gate 3: R_rat — soft consistency check (0.5 min, 1.0 max)
            CL *= 0.5 + 0.5 * R_rat

            # Gate 4: Boltzmann consistency — verify matched lines follow
            # ln(I·λ/gA) vs E_k with physically reasonable temperature.
            #
            # Bead CF-LIBS-improved-n3rf.1: Detective C established that
            # on Vrabel Si-positive spectra the Boltzmann R² collapses to
            # 0.000–0.220 because the Si I 288.16 nm resonance line is
            # heavily self-absorbed, dragging its ln(I·λ/gA) off the
            # linear fit. Pass nnls_significant down so the consistency
            # check can drop resonance lines when there's independent
            # NNLS evidence that the candidate is real — only then is it
            # safe to suspect self-absorption rather than a false candidate.
            boltz_factor, boltz_r2 = self._boltzmann_consistency_check(
                element,
                fused_lines,
                matched_mask,
                matched_peak_idx,
                corrected_intensity,
                peaks,
                nnls_significant=bool(cand.get("nnls_significant", False)),
            )
            CL *= boltz_factor

            # Gate 5: Sparse NNLS significance (primary discriminator at low RP)
            # At RP<2000, peak-matching CL cannot discriminate (TP/FP overlap).
            # The sparse NNLS coefficient is the strongest false-positive
            # suppressor: elements zeroed out by L1 penalty are truly absent.
            nnls_sig = cand.get("nnls_significant", True)
            if not nnls_sig and self.resolving_power < 2000:
                CL = 0.0

            # Adaptive detection threshold: elements with few expected
            # lines have higher false-match rates at low RP and need a
            # proportionally higher CL to be considered detected.
            adaptive_dt = self.detection_threshold
            if N_expected > 0 and N_expected < 10:
                adaptive_dt *= min(3.0, math.sqrt(10.0 / N_expected))
            detected = CL >= adaptive_dt

            # Physics-grounded hard gates (Task wzus):
            # (a) Require at least three matched lines, rejecting single-line
            #     and doublet-only identifications.
            # (b) Require Boltzmann R^2 >= boltzmann_r2_min only when at least
            #     three matched lines make a regression meaningful.
            min_required_matches = 3
            if N_matched < min_required_matches:
                detected = False
            if self._r2_gate_rejects(boltz_r2, N_matched):
                detected = False

            # L4 -- per-element fingerprint pass.  alias's effective
            # "fingerprint" is the post-gate CL compared to
            # ``adaptive_dt`` (plus the hard gates above).  Record the
            # final boolean and the score+floor for transparency.
            coverage.record_fingerprint(
                element,
                passed=bool(detected),
                score=float(CL),
                floor=float(adaptive_dt),
            )

            # Create IdentifiedLine objects for matched lines
            # Reuse peak indices from matching to avoid re-selection outside window
            matched_lines = []
            unmatched_lines = []
            for i, line_data in enumerate(fused_lines):
                trans = line_data["transition"]
                if matched_mask[i]:
                    pidx = matched_peak_idx[i]
                    matched_lines.append(
                        IdentifiedLine(
                            wavelength_exp_nm=peaks[pidx][1],
                            wavelength_th_nm=line_data["wavelength_nm"],
                            element=element,
                            ionization_stage=trans.ionization_stage,
                            intensity_exp=corrected_intensity[peaks[pidx][0]],
                            emissivity_th=line_data["avg_emissivity"],
                            transition=trans,
                            correlation=k_sim,
                        )
                    )
                else:
                    unmatched_lines.append(trans)

            element_id = ElementIdentification(
                element=element,
                detected=detected,
                score=k_det,
                confidence=CL,
                n_matched_lines=int(np.sum(matched_mask)),
                n_total_lines=len(fused_lines),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={
                    "k_sim": k_sim,
                    "k_rate": k_rate,
                    "k_shift": k_shift,
                    "k_det": k_det,
                    "emissivity_threshold": emissivity_threshold,
                    "N_expected": N_expected,
                    "N_matched": N_matched,
                    "P_maj": P_maj,
                    "P_ab": self._compute_P_ab(element),
                    "P_cov": P_cov,
                    "P_mix": P_mix,
                    "P_local": P_local,
                    "R_rat": R_rat,
                    "P_SNR": P_SNR,
                    "P_sig": P_sig,
                    "p_tail": p_tail,
                    "p_chance": p_chance,
                    "fill_factor": fill_factor,
                    "N_penalty": min(1.0, math.sqrt(N_expected / 5.0)) if N_expected > 0 else 0.0,
                    "boltzmann_factor": boltz_factor,
                    "boltzmann_r2": boltz_r2,
                    "min_required_matches": min_required_matches,
                    "sparse_nnls_coeff": cand.get("sparse_nnls_coeff", 0.0),
                    "nnls_significant": cand.get("nnls_significant", True),
                    "effective_R": self._effective_R,
                    "global_wl_shift": self._global_wl_shift,
                    "estimated_T": self._estimated_T,
                },
            )

            all_element_ids.append(element_id)

        # Apply relative threshold: element CL must be >= max_CL * relative_cl_threshold
        # This prevents spurious detections when one element dominates.
        # Set self.relative_cl_threshold = 0 to disable.
        self._apply_relative_cl_gate(all_element_ids)

        # Split into detected/rejected
        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        # Count matched peaks (peak matched if any element matched it, detected or rejected)
        matched_peak_indices = set()
        for element_id in all_element_ids:  # Use all_element_ids, not just detected
            for line in element_id.matched_lines:
                peak_idx = np.argmin(
                    np.abs(np.array([p[1] for p in peaks]) - line.wavelength_exp_nm)
                )
                matched_peak_indices.add(int(peak_idx))

        # Structured log line for the self-absorption damping path. Operators
        # debugging "why is Si missed in soil spectra" need to see this even
        # at INFO level — addresses the user's transparency complaint that
        # the hardcoded ``SA_DAMPING = 0.3`` was applied silently.
        if self.self_absorption_aware:
            if self._sa_n_damped_lines > 0:
                logger.info(
                    "alias.identify self-absorption damping applied: "
                    "n_damped_lines=%d, damped_elements=%s, "
                    "damping_factor=%.3f, E_i_cutoff_ev=%.3f, "
                    "n_peaks=%d, n_detected=%d",
                    self._sa_n_damped_lines,
                    sorted(self._sa_damped_elements),
                    self.self_absorption_damping,
                    self.self_absorption_e_i_cutoff_ev,
                    len(peaks),
                    len(detected_elements),
                )
            else:
                logger.debug(
                    "alias.identify self-absorption damping enabled but "
                    "no resonance lines (E_i_ev < %.3f) reached the k_sim "
                    "or R_rat scoring path: damping_factor=%.3f, n_peaks=%d, "
                    "n_detected=%d",
                    self.self_absorption_e_i_cutoff_ev,
                    self.self_absorption_damping,
                    len(peaks),
                    len(detected_elements),
                )
        else:
            logger.info(
                "alias.identify self-absorption damping DISABLED "
                "(self_absorption_aware=False); k_sim and R_rat use raw "
                "optically-thin emissivities: n_peaks=%d, n_detected=%d",
                len(peaks),
                len(detected_elements),
            )

        # Detection-coverage finalisation: record peak count + emit
        # summary log line.  Additive telemetry only.
        coverage.set_n_peaks(len(peaks))
        coverage.emit_summary()

        base_parameters = {
            "resolving_power": self.resolving_power,
            "effective_R": self._effective_R,
            "global_wl_shift_nm": self._global_wl_shift,
            "estimated_T_K": self._estimated_T,
            "T_min_K": self.T_range_K[0],
            "T_max_K": self.T_range_K[1],
            "n_e_min_cm3": self.n_e_range_cm3[0],
            "n_e_max_cm3": self.n_e_range_cm3[1],
            "intensity_threshold_factor": self.intensity_threshold_factor,
            "detection_threshold": self.detection_threshold,
        }

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=peaks,
            n_peaks=len(peaks),
            n_matched_peaks=len(matched_peak_indices),
            n_unmatched_peaks=len(peaks) - len(matched_peak_indices),
            algorithm="alias",
            parameters=merge_coverage_into_parameters(base_parameters, coverage.build_payload()),
        )

    def _detect_peaks(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Detect peaks using MAD-based noise estimation and scipy.signal.find_peaks.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array
        intensity : np.ndarray
            Intensity array

        Returns
        -------
        List[Tuple[int, float]]
            List of (peak_index, peak_wavelength) tuples
        """
        # Estimate baseline and noise using sigma-clipped MAD
        baseline = estimate_baseline(wavelength, intensity)
        noise_estimate = estimate_noise(intensity, baseline)

        # Threshold in intensity domain (with floor for flat spectra / zero MAD)
        threshold = max(noise_estimate * self.intensity_threshold_factor, np.finfo(float).eps)
        prominence = max(threshold / 3, np.finfo(float).eps)

        # Find peaks in baseline-corrected intensity
        corrected = intensity - baseline
        peak_indices, _ = find_peaks(corrected, height=threshold, prominence=prominence)

        # Paper (Noël et al. 2025): enhance peak detection using negative 2nd derivative
        # Compute -d²I/dλ², zero negatives — true peaks have positive curvature here
        d2 = -np.gradient(np.gradient(corrected, wavelength), wavelength)
        d2[d2 < 0] = 0.0

        # Filter: keep peaks where d2 > 0 in a ±2-point neighborhood around peak center
        # This handles discretization effects where d2 peak may be slightly offset
        confirmed = []
        for idx in peak_indices:
            lo = max(0, idx - 2)
            hi = min(len(d2), idx + 3)
            if np.max(d2[lo:hi]) > 0:
                confirmed.append(idx)
        peak_indices = np.array(confirmed, dtype=int) if confirmed else np.array([], dtype=int)

        # Return as list of (index, wavelength) tuples
        peaks = [(int(idx), float(wavelength[idx])) for idx in peak_indices]

        return peaks

    def _auto_calibrate_wavelength(
        self,
        peaks: List[Tuple[int, float]],
        wl_min: float,
        wl_max: float,
    ) -> None:
        """
        Auto-calibrate wavelength offset and effective resolving power.

        Compares detected peak positions to the strongest NIST reference
        lines across common LIBS elements to estimate:
        1. Global wavelength shift (median offset from best matches)
        2. Effective resolving power (from distribution of offsets)

        Sets self._global_wl_shift and self._effective_R.
        """
        if not peaks:
            self._global_wl_shift = 0.0
            self._effective_R = self.resolving_power
            return

        peak_wls = np.array([p[1] for p in peaks])

        # Get strong reference lines from common LIBS elements
        reference_elements = ["Fe", "Ca", "Mg", "Ti", "Al", "Cu", "Na", "Si", "Cr", "Mn"]
        kT_ref = KB_EV * self.reference_temperature

        ref_lines = []
        for el in reference_elements:
            for ion_stage in [1, 2]:
                try:
                    trans = self.atomic_db.get_transitions(
                        el, ion_stage, wavelength_min=wl_min, wavelength_max=wl_max
                    )
                    if trans:
                        for t in trans:
                            strength = t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT_ref)
                            ref_lines.append((t.wavelength_nm, strength, el))
                except (KeyError, ValueError, AttributeError):
                    continue

        if not ref_lines:
            self._global_wl_shift = 0.0
            self._effective_R = self.resolving_power
            return

        # Take top 30 strongest reference lines
        ref_lines.sort(key=lambda x: x[1], reverse=True)
        top_refs = ref_lines[:30]

        # For each reference line, find the nearest peak using a generous
        # initial tolerance (R=1000, ~0.4nm at 400nm)
        offsets = []
        for ref_wl, _strength, _el in top_refs:
            dists = peak_wls - ref_wl
            abs_dists = np.abs(dists)
            generous_tol = ref_wl / 1000.0  # R=1000
            within = abs_dists <= generous_tol
            if np.any(within):
                best_idx = np.argmin(abs_dists)
                offsets.append(float(dists[best_idx]))

        if len(offsets) < 3:
            self._global_wl_shift = 0.0
            self._effective_R = self.resolving_power
            return

        # Global shift = median offset
        self._global_wl_shift = float(np.median(offsets))

        # Effective R: estimate from MAD of offsets after shift correction
        corrected_offsets = np.array(offsets) - self._global_wl_shift
        mad = float(np.median(np.abs(corrected_offsets)))
        # The matching tolerance delta_lambda = mean_wl / R
        # We want delta_lambda ~ 3*MAD to capture 99% of real matches
        mean_wl = 0.5 * (wl_min + wl_max)
        if mad > 0:
            estimated_R = mean_wl / (3.0 * mad)
            # Clamp to reasonable range [500, nominal R]
            self._effective_R = float(np.clip(estimated_R, 500.0, self.resolving_power))
        else:
            # Perfect calibration — use nominal R
            self._effective_R = self.resolving_power

    def _estimate_plasma_temperature(
        self,
        peaks: List[Tuple[int, float]],
        corrected_intensity: np.ndarray,
        wl_min: float,
        wl_max: float,
    ) -> None:
        """
        Estimate plasma temperature from Boltzmann slope of strong detected peaks.

        Uses Fe I lines preferentially (most common in LIBS). Falls back to any
        transition metal with enough matched lines, then a line-ratio method,
        and finally ``self.reference_temperature`` if all methods fail.

        The Pass-1 slope-fit strategy is governed by
        ``self.temperature_estimator_mode`` ("legacy" | "robust" | "weighted").
        See the constructor docstring for full rationale; the short story is
        that "legacy" is byte-identical with the historical behavior and
        "robust"/"weighted" are opt-in for the Vrabel-universal-miss root
        cause (the unweighted slope fit is dragged cold by noisy high-E_k
        weak lines).

        Always sets ``self._estimated_T`` (K) to a finite value.
        """
        if len(peaks) < 3:
            self._estimated_T = self.reference_temperature
            return

        peak_wls = np.array([p[1] for p in peaks])
        peak_intensities = np.array([corrected_intensity[p[0]] for p in peaks])

        # Try to match strong lines from Fe I, Ti I, Cr I, Ca I etc.
        probe_elements = ["Fe", "Ti", "Cr", "Ca", "Mn", "Ni", "V", "Cu", "Mg", "Si", "Al"]
        delta_lambda = 0.5 * (wl_min + wl_max) / max(self._effective_R or self.resolving_power, 500)

        # ------------------------------------------------------------------
        # Robust mode (opt-in, CF-LIBS-improved-762f): collect candidate T_K
        # from EVERY element using weighted regression, then accept the
        # cross-element median if 3+ elements fall within a 2000 K window.
        # See ``_estimate_T_robust`` below for the per-element fit.
        # ------------------------------------------------------------------
        if self.temperature_estimator_mode == "robust":
            robust_candidates: List[Tuple[str, float, float, int]] = []
            for probe_el in probe_elements:
                cand = self._estimate_T_single_element_robust(
                    probe_el,
                    peak_wls,
                    peak_intensities,
                    wl_min,
                    wl_max,
                    delta_lambda,
                )
                if cand is not None:
                    T_K, r_sq, n_lines = cand
                    robust_candidates.append((probe_el, T_K, r_sq, n_lines))

            if robust_candidates:
                T_values = np.array([c[1] for c in robust_candidates])
                # Cross-element consistency: 3+ elements within a 2000 K window
                # around the median?
                T_median = float(np.median(T_values))
                within_window = np.sum(np.abs(T_values - T_median) <= 2000.0)
                # INFO log so operators can audit the consensus.
                logger.info(
                    "alias._estimate_plasma_temperature robust mode: "
                    "n_elements=%d, per_element=%s, median_T_K=%.0f, "
                    "n_within_2000K=%d, selected_T_K=%s",
                    len(robust_candidates),
                    [(el, round(T, 0), round(r2, 3), n) for el, T, r2, n in robust_candidates],
                    T_median,
                    int(within_window),
                    f"{T_median:.0f}" if within_window >= 3 else "fall-through-to-pass2",
                )
                if (
                    within_window >= 3
                    and self._T_ESTIMATE_MIN_K < T_median < self._T_ESTIMATE_MAX_K
                ):
                    self._estimated_T = T_median
                    return
                # Single-element or scatter > 2000 K: still take the best
                # individual fit (highest r_sq, then largest n_lines) if
                # any single element looks credible.
                # Otherwise fall through to Pass 2.
                robust_candidates.sort(key=lambda c: (c[2], c[3]), reverse=True)
                best_el, best_T, best_r2, best_n = robust_candidates[0]
                if (
                    best_r2 > 0.5
                    and best_n >= 4
                    and self._T_ESTIMATE_MIN_K < best_T < self._T_ESTIMATE_MAX_K
                ):
                    logger.info(
                        "alias._estimate_plasma_temperature robust mode: "
                        "no cross-element consensus, accepting best "
                        "single-element fit element=%s T_K=%.0f r_sq=%.3f "
                        "n_lines=%d",
                        best_el,
                        best_T,
                        best_r2,
                        best_n,
                    )
                    self._estimated_T = float(best_T)
                    return
            # else: fall through to Pass 2 below.

        else:
            # Legacy + weighted modes share the per-element early-return loop.
            # The only difference: "weighted" drops the bottom quartile by SNR
            # before fitting; "legacy" uses the unweighted slope on all lines.
            for probe_el in probe_elements:
                try:
                    transitions = self.atomic_db.get_transitions(
                        probe_el, 1, wavelength_min=wl_min, wavelength_max=wl_max
                    )
                    if not transitions:
                        continue
                except (KeyError, ValueError, AttributeError):
                    continue

                # Filter to lines with known A_ki and g_k
                good_trans = [t for t in transitions if t.A_ki > 0 and t.g_k > 0 and t.E_k_ev > 0]
                if len(good_trans) < 4:
                    continue

                # Sort by expected strength and take top 15
                kT_ref = KB_EV * self.reference_temperature
                good_trans.sort(
                    key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT_ref),
                    reverse=True,
                )
                good_trans = good_trans[:15]

                # Match to peaks
                E_k_vals = []
                y_vals = []
                I_vals = []  # tracked for weighted-mode SNR pruning
                shift = self._global_wl_shift

                for t in good_trans:
                    wl_shifted = t.wavelength_nm + shift
                    dists = np.abs(peak_wls - wl_shifted)
                    best_idx = int(np.argmin(dists))
                    if dists[best_idx] <= delta_lambda:
                        I_obs = peak_intensities[best_idx]
                        if I_obs > 0 and t.A_ki > 0 and t.g_k > 0:
                            y = math.log(I_obs * t.wavelength_nm / (t.g_k * t.A_ki))
                            E_k_vals.append(t.E_k_ev)
                            y_vals.append(y)
                            I_vals.append(float(I_obs))

                if len(E_k_vals) < 4:
                    continue

                # Weighted mode (opt-in): drop the bottom quartile by intensity
                # (SNR proxy) to deprioritize the noisy weak high-E_k lines.
                if self.temperature_estimator_mode == "weighted":
                    I_arr_local = np.array(I_vals)
                    if I_arr_local.size >= 4:
                        snr_cutoff = float(np.quantile(I_arr_local, 0.25))
                        keep_mask = I_arr_local > snr_cutoff
                        # Keep at least 4 lines so the fit remains tractable.
                        if int(np.sum(keep_mask)) >= 4:
                            E_k_vals = [e for e, k in zip(E_k_vals, keep_mask) if k]
                            y_vals = [y for y, k in zip(y_vals, keep_mask) if k]
                            I_vals = [i for i, k in zip(I_vals, keep_mask) if k]

                if len(E_k_vals) < 4:
                    continue

                # Fit Boltzmann slope: y = -1/(kT) * E_k + const
                E_k_arr = np.array(E_k_vals)
                y_arr = np.array(y_vals)

                try:
                    if self.use_jax_boltzmann_fit:
                        # JAX path (opt-in). One-spectrum batch; the heavy lift
                        # comes when future callers vectorize across many spectra.
                        T_K_arr, slope_arr, r_sq_arr = boltzmann_temperature_jax(
                            y_arr[None, :],
                            E_k_arr,
                            weights=None,
                            return_diagnostics=True,
                        )
                        slope = float(slope_arr[0])
                        r_sq = float(r_sq_arr[0])
                        T_K_jax = float(T_K_arr[0])

                        if not np.isfinite(slope) or abs(slope) < 1e-10:
                            continue
                        if slope < 0 and r_sq > 0.2:
                            T_K = T_K_jax
                            if self._T_ESTIMATE_MIN_K < T_K < self._T_ESTIMATE_MAX_K:
                                self._estimated_T = float(T_K)
                                return
                    else:
                        result = linregress(E_k_arr, y_arr)
                        slope = result.slope
                        r_sq = result.rvalue**2

                        if abs(slope) < 1e-10:
                            continue
                        if slope < 0 and r_sq > 0.2:
                            T_K = -1.0 / (slope * KB_EV)
                            if self._T_ESTIMATE_MIN_K < T_K < self._T_ESTIMATE_MAX_K:
                                self._estimated_T = float(T_K)
                                return
                except (ValueError, ZeroDivisionError):
                    continue

        # Pass 2: Line-ratio fallback — estimate T from best 2-line pair
        shift = self._global_wl_shift
        for probe_el in probe_elements:
            try:
                transitions = self.atomic_db.get_transitions(
                    probe_el, 1, wavelength_min=wl_min, wavelength_max=wl_max
                )
                if not transitions:
                    continue
            except (KeyError, ValueError, AttributeError):
                continue

            good_trans = [t for t in transitions if t.A_ki > 0 and t.g_k > 0 and t.E_k_ev > 0]
            if len(good_trans) < 2:
                continue

            # Match to peaks
            matched_pairs = []
            for t in good_trans:
                wl_shifted = t.wavelength_nm + shift
                dists = np.abs(peak_wls - wl_shifted)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] <= delta_lambda:
                    I_obs = peak_intensities[best_idx]
                    if I_obs > 0:
                        matched_pairs.append((t, I_obs))

            if len(matched_pairs) < 2:
                continue

            # Find pair with largest E_k separation
            best_T_local = None
            best_dE = 0.0
            for i in range(len(matched_pairs)):
                for j in range(i + 1, len(matched_pairs)):
                    t1, I1 = matched_pairs[i]
                    t2, I2 = matched_pairs[j]
                    dE = abs(t1.E_k_ev - t2.E_k_ev)
                    if dE < 0.5:
                        continue
                    numer = I2 * t2.wavelength_nm * t1.g_k * t1.A_ki
                    denom = I1 * t1.wavelength_nm * t2.g_k * t2.A_ki
                    if denom <= 0 or numer <= 0:
                        continue
                    ln_ratio = math.log(numer / denom)
                    if abs(ln_ratio) < 1e-10:
                        continue
                    T_K = -(t2.E_k_ev - t1.E_k_ev) / (KB_EV * ln_ratio)
                    if self._T_ESTIMATE_MIN_K < T_K < self._T_ESTIMATE_MAX_K and dE > best_dE:
                        best_T_local = T_K
                        best_dE = dE

            if best_T_local is not None:
                self._estimated_T = float(best_T_local)
                return

        # Final fallback: use reference temperature instead of None
        self._estimated_T = self.reference_temperature

    def _estimate_T_single_element_robust(
        self,
        probe_el: str,
        peak_wls: np.ndarray,
        peak_intensities: np.ndarray,
        wl_min: float,
        wl_max: float,
        delta_lambda: float,
    ) -> Optional[Tuple[float, float, int]]:
        """
        Intensity-weighted Boltzmann-slope T_K for one element (robust mode).

        Mechanism — addresses the Vrabel universal-miss diagnosis
        (``docs/research/vrabel-universal-miss-root-cause-2026-05-14.md``):
        in the legacy unweighted ``np.polyfit`` / ``linregress`` slope fit,
        a handful of weak high-``E_k`` lines (intensity ~ noise) dominate
        the slope and pull the recovered T_K cold (~4000 K) when the true
        T_K is ~10000 K. By weighting each Boltzmann-plot point by
        ``1 / sigma_y^2`` with ``sigma_y ≈ sigma_I / I`` and a shot-noise
        proxy ``sigma_I ≈ max(noise, sqrt(I))``, the noisy high-``E_k``
        points are deprioritized and the fit follows the strong
        low-``E_k`` lines that actually constrain T.

        Returns ``(T_K, r_sq, n_lines)`` if a valid weighted fit was
        produced, or ``None`` otherwise (insufficient lines, degenerate
        E_k spread, slope >= 0, unphysical T_K, etc.). Does not mutate
        ``self``.
        """
        try:
            transitions = self.atomic_db.get_transitions(
                probe_el, 1, wavelength_min=wl_min, wavelength_max=wl_max
            )
            if not transitions:
                return None
        except (KeyError, ValueError, AttributeError):
            return None

        good_trans = [t for t in transitions if t.A_ki > 0 and t.g_k > 0 and t.E_k_ev > 0]
        if len(good_trans) < 4:
            return None

        kT_ref = KB_EV * self.reference_temperature
        good_trans.sort(
            key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT_ref),
            reverse=True,
        )
        good_trans = good_trans[:15]

        E_k_vals: List[float] = []
        y_vals: List[float] = []
        sigma_y_vals: List[float] = []
        shift = self._global_wl_shift

        # Per-spectrum noise proxy: median |corrected intensity| of the
        # bottom half of matched peaks is a robust low-end estimate.
        # We do not have access to the noise estimate from _detect_peaks
        # here, so use a per-line fallback ``max(1e-12, sqrt(I))``.
        for t in good_trans:
            wl_shifted = t.wavelength_nm + shift
            dists = np.abs(peak_wls - wl_shifted)
            best_idx = int(np.argmin(dists))
            if dists[best_idx] > delta_lambda:
                continue
            I_obs = float(peak_intensities[best_idx])
            if I_obs <= 0 or t.A_ki <= 0 or t.g_k <= 0:
                continue
            y = math.log(I_obs * t.wavelength_nm / (t.g_k * t.A_ki))
            # Shot-noise proxy: sigma_I ~ max(small_floor, sqrt(I)).
            # In log-space y = log(I·λ/(g_k·A_ki)), so d(y)/d(I) = 1/I
            # and sigma_y ≈ sigma_I / I.
            sigma_I = max(math.sqrt(max(I_obs, 0.0)), 1e-12)
            sigma_y = sigma_I / I_obs
            E_k_vals.append(t.E_k_ev)
            y_vals.append(y)
            sigma_y_vals.append(max(sigma_y, 1e-12))

        if len(E_k_vals) < 4:
            return None

        E_k_arr = np.array(E_k_vals)
        y_arr = np.array(y_vals)
        sigma_arr = np.array(sigma_y_vals)
        weights = 1.0 / np.square(sigma_arr)
        # Need E_k spread for the regression to be meaningful.
        if float(np.ptp(E_k_arr)) < 0.5:
            return None

        # Closed-form weighted linear least squares for y = a + b·x:
        #   b = (sum w x y · sum w − sum w x · sum w y) / (sum w x^2 · sum w − (sum w x)^2)
        # r^2 computed against the weighted mean of y.
        try:
            W = float(np.sum(weights))
            if W <= 0:
                return None
            Sx = float(np.sum(weights * E_k_arr))
            Sy = float(np.sum(weights * y_arr))
            Sxx = float(np.sum(weights * E_k_arr * E_k_arr))
            Sxy = float(np.sum(weights * E_k_arr * y_arr))
            denom = Sxx * W - Sx * Sx
            if abs(denom) < 1e-30:
                return None
            slope = (Sxy * W - Sx * Sy) / denom
            intercept = (Sy - slope * Sx) / W
            # Weighted r^2 against the weighted mean.
            y_mean_w = Sy / W
            y_pred = slope * E_k_arr + intercept
            SS_res = float(np.sum(weights * np.square(y_arr - y_pred)))
            SS_tot = float(np.sum(weights * np.square(y_arr - y_mean_w)))
            if SS_tot <= 0:
                return None
            r_sq = 1.0 - SS_res / SS_tot
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return None

        if not np.isfinite(slope) or abs(slope) < 1e-10:
            return None
        if slope >= 0:
            return None
        T_K = -1.0 / (slope * KB_EV)
        if not (self._T_ESTIMATE_MIN_K < T_K < self._T_ESTIMATE_MAX_K):
            return None
        # Note: polarity audit of the legacy ``r_sq > 0.2`` gate
        # (alias.py: search "slope < 0 and r_sq > 0.2") found the polarity
        # is CORRECT — high r^2 ⇒ accept (good fit), not invert it. So we
        # apply the same r_sq threshold here as a sanity floor, but in
        # robust mode the cross-element median check is the primary
        # selection criterion, not a per-element r_sq gate.
        if r_sq < 0.2:
            return None
        return float(T_K), float(r_sq), int(len(E_k_vals))

    # EVOLVE-BLOCK-START
    def _fast_screening(
        self,
        all_elements: List[str],
        peaks: List[Tuple[int, float]],
        wl_min: float,
        wl_max: float,
    ) -> List[str]:
        """
        Fast screening to restrict candidate elements.

        Two-stage approach:
        1. For each element, compute a quick screening score based on how many
           of its top-10 lines match peaks, weighted by line strength.
        2. Pass the top max_screening_candidates scoring elements.

        Always-test elements bypass screening.

        Returns list of elements that passed screening.
        """
        if not peaks:
            return list(self._always_test & set(all_elements))

        peak_wls = np.array([p[1] for p in peaks])
        eff_R = self._effective_R or self.resolving_power
        mean_wl = 0.5 * (wl_min + wl_max)
        delta_lambda = mean_wl / eff_R
        screening_tol = 2.0 * delta_lambda
        shift = self._global_wl_shift

        kT_ref = KB_EV * self.reference_temperature
        element_scores = []

        for element in all_elements:
            if element in self._always_test:
                continue

            # Get all lines and compute strengths
            lines_with_strength = []
            for ion_stage in [1, 2]:
                try:
                    trans = self.atomic_db.get_transitions(
                        element, ion_stage, wavelength_min=wl_min, wavelength_max=wl_max
                    )
                    if trans:
                        for t in trans:
                            strength = t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT_ref)
                            lines_with_strength.append((t.wavelength_nm, strength))
                except (KeyError, ValueError, AttributeError):
                    continue

            if not lines_with_strength:
                continue

            lines_with_strength.sort(key=lambda x: x[1], reverse=True)
            top10 = lines_with_strength[:10]

            # Compute screening score: sum of strengths for matched lines
            # divided by total strength (strength-weighted match rate)
            total_strength = sum(s for _, s in top10)
            matched_strength = 0.0
            n_matched = 0
            for wl_th, strength in top10:
                wl_shifted = wl_th + shift
                dists = np.abs(peak_wls - wl_shifted)
                if np.min(dists) <= screening_tol:
                    matched_strength += strength
                    n_matched += 1

            # Single-line exception: elements with ≤1 line in the window
            # (e.g., Li I 670.8nm) need only 1 match to pass screening.
            min_matches = 1 if len(top10) <= 1 else 2
            if n_matched >= min_matches and total_strength > 0:
                score = matched_strength / total_strength
                if score >= 0.3:
                    element_scores.append((element, score, n_matched))

        # Sort by screening score, take top max_screening_candidates
        element_scores.sort(key=lambda x: x[1], reverse=True)
        passed = list(self._always_test & set(all_elements))
        for element, score, n_matched in element_scores[: self.max_screening_candidates]:
            if element not in passed:
                passed.append(element)

        return passed

    # EVOLVE-BLOCK-END

    def _r2_gate_rejects(self, boltz_r2: float, N_matched: int) -> bool:
        """Apply the configurable Boltzmann R^2 gate to a candidate.

        Returns ``True`` if the candidate should be rejected by the gate,
        ``False`` if it passes. The gate is only meaningful when at least
        three matched lines exist (so a regression is well-defined); for
        ``N_matched < 3`` the rejection of that candidate is handled
        upstream and this method always returns ``False``.

        Mode behavior (CF-LIBS-improved-ftp1):
        - ``"fixed"`` (default): byte-identical to the historical static
          gate — ``boltz_r2 < self.boltzmann_r2_min`` triggers rejection.
        - ``"adaptive_t"``: if ``self._estimated_T`` is below
          ``self.r2_gate_t_quality_threshold``, fall back to the cold-T
          floor (``self._r2_gate_cold_floor``, currently 0.3). Otherwise
          apply the strict fixed gate. This addresses the Vrabel-style
          cold-plasma pathology (PR #172 diagnosis) where short effective
          E_k spans degrade R^2 even when the line set is real.
        - ``"disabled"``: gate never rejects. Intended only for the
          control-cell sweep — NOT a real production setting.

        Parameters
        ----------
        boltz_r2 : float
            Coefficient of determination reported by
            ``_boltzmann_consistency_check`` for the candidate.
        N_matched : int
            Number of matched lines on the candidate. The gate is
            no-op'd when this is below 3.

        Returns
        -------
        bool
            ``True`` if the gate rejects the candidate.
        """
        if N_matched < 3:
            return False
        if self.r2_gate_mode == "disabled":
            return False
        if self.r2_gate_mode == "adaptive_t":
            est_T = getattr(self, "_estimated_T", None)
            if (
                est_T is not None
                and np.isfinite(est_T)
                and est_T < self.r2_gate_t_quality_threshold
            ):
                return boltz_r2 < self._r2_gate_cold_floor
            return boltz_r2 < self.boltzmann_r2_min
        # "fixed" mode — default, byte-identical to historical behavior.
        return boltz_r2 < self.boltzmann_r2_min

    def _boltzmann_consistency_check(
        self,
        element: str,
        fused_lines: List[dict],
        matched_mask: np.ndarray,
        matched_peak_idx: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        nnls_significant: bool = False,
    ) -> Tuple[float, float]:
        """
        Boltzmann consistency check for matched lines.

        For elements with >=3 matched lines, fit ln(I*lambda/(g*A)) vs E_k.
        Slope should give physical temperature (3000-50000K) with reasonable R^2.

        Parameters
        ----------
        nnls_significant : bool, optional
            When True AND ``self.self_absorption_aware`` is enabled, drop
            resonance lines (``E_i_ev < self.self_absorption_e_i_cutoff_ev``)
            from the regression before fitting. Bead n3rf.1: on Vrabel
            Si-positive spectra the Si I 288.16 nm + 244.34 nm resonance
            lines at common E_k=5.082 eV give a 5-unit spread in
            ln(I·λ/gA) — self-absorption, not statistical noise. Filtering
            them is only safe when independent NNLS evidence supports the
            candidate (otherwise the gate would let in genuine false
            positives whose Boltzmann inconsistency IS the signal).

        Returns
        -------
        Tuple[float, float]
            (boltzmann_factor, r_squared)
            boltzmann_factor is in [0.5, 1.0] to multiply into CL.
        """
        matched_indices = np.nonzero(matched_mask)[0]
        if len(matched_indices) < 3:
            return 0.5, 0.0  # Penalize — not enough lines for Boltzmann check

        observations = []
        n_resonance_filtered = 0
        resonance_cutoff = float(getattr(self, "self_absorption_e_i_cutoff_ev", 0.1))
        apply_resonance_filter = bool(
            nnls_significant and getattr(self, "self_absorption_aware", True)
        )

        for i in matched_indices:
            trans = fused_lines[i]["transition"]
            pidx = int(matched_peak_idx[i])
            if pidx < 0 or pidx >= len(peaks):
                continue
            I_obs = intensity[peaks[pidx][0]]
            if I_obs <= 0 or trans.A_ki <= 0 or trans.g_k <= 0:
                continue

            # Drop resonance lines when caller has NNLS-significance evidence
            # for the candidate. Self-absorption on these lines biases the
            # slope but does not appear in the basis-spectrum NNLS check.
            if apply_resonance_filter:
                e_i = float(getattr(trans, "E_i_ev", 1.0))
                if e_i < resonance_cutoff:
                    n_resonance_filtered += 1
                    continue

            observations.append(
                LineObservation(
                    element=element,
                    ionization_stage=trans.ionization_stage,
                    wavelength_nm=trans.wavelength_nm,
                    E_k_ev=trans.E_k_ev,
                    g_k=trans.g_k,
                    A_ki=trans.A_ki,
                    intensity=I_obs,
                    intensity_uncertainty=max(abs(I_obs) * 0.1, 1e-12),
                )
            )

        if len(observations) < 3:
            # Filtering may have dropped us below the minimum. If we filtered
            # at all and got too few lines for a fit, treat as "no evidence
            # against the candidate" (boltz_factor=1.0) rather than
            # penalising — the NNLS-significance flag is the upstream gate.
            if apply_resonance_filter and n_resonance_filtered > 0:
                return 1.0, 0.0
            return 0.5, 0.0

        # Need some spread in E_k for meaningful fit
        E_k_arr = np.array([obs.E_k_ev for obs in observations])
        if np.ptp(E_k_arr) < 0.5:
            # No meaningful Boltzmann regression is possible.
            return 0.5, 0.0

        try:
            fitter = BoltzmannPlotFitter()
            result = fitter.fit(observations)
            slope = result.slope
            r_sq = result.r_squared
        except (ValueError, ZeroDivisionError):
            return 1.0, 0.0

        # Check physical validity
        if slope >= 0:
            # Positive slope = anti-Boltzmann → likely false positive
            return 0.5, r_sq

        T_K = result.temperature_K
        if T_K < self._T_CONSISTENCY_MIN_K or T_K > self._T_CONSISTENCY_MAX_K:
            # Unphysical temperature
            return 0.5, r_sq

        # Scale by R^2: good fit → 1.0, poor fit → 0.7
        factor = float(0.7 + 0.3 * min(r_sq, 1.0))
        return factor, r_sq

    def _compute_element_emissivities(
        self,
        element: str,
        wl_min: float,
        wl_max: float,
        T_estimated: Optional[float] = None,
    ) -> List[dict]:
        """
        Compute theoretical emissivities for element over (T, n_e) grid.

        Parameters
        ----------
        element : str
            Element symbol
        wl_min : float
            Minimum wavelength in nm
        wl_max : float
            Maximum wavelength in nm

        Returns
        -------
        List[dict]
            List of dicts with keys: transition, avg_emissivity, wavelength_nm
        """
        # Get transitions for element (try both neutral and ionized)
        transitions = []
        for ion_stage in [1, 2]:
            try:
                trans_list = self.atomic_db.get_transitions(
                    element, ion_stage, wavelength_min=wl_min, wavelength_max=wl_max
                )
                if trans_list:
                    transitions.extend(trans_list)
            except (KeyError, ValueError, AttributeError):
                # No data for this ionization stage
                continue

        # Remove unobservable weak lines before emissivity calculation
        transitions = [t for t in transitions if t.A_ki * t.g_k >= 1e4]

        if not transitions:
            return []

        # Cap to strongest lines by estimated emissivity to avoid line-count disparity
        if len(transitions) > self.max_lines_per_element:
            kT = KB_EV * self.reference_temperature
            transitions = sorted(
                transitions,
                key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT),
                reverse=True,
            )
            transitions = transitions[: self.max_lines_per_element]

        # Compute emissivities
        line_data = []
        total_density = 1e15  # Arbitrary reference density

        # When T_estimated is available, use a narrow grid around that T
        # instead of the full T range. This makes emissivities reflect the
        # actual plasma conditions rather than averaged-out values.
        if T_estimated is not None:
            T_grid = np.array([T_estimated])
        else:
            # Always use a single reference T — averaging over the full
            # grid dilutes the reference vector and makes cosine similarity
            # meaningless.
            T_grid = np.array([self.reference_temperature])

        # Precompute stage densities for all (T, n_e) grid points
        grid_stage_densities = {}
        for T_K in T_grid:
            for n_e in self.n_e_grid_cm3:
                T_eV = T_K * KB_EV
                try:
                    stage_densities = self.solver.solve_ionization_balance(
                        element, T_eV, n_e, total_density
                    )
                    grid_stage_densities[(T_K, n_e)] = stage_densities
                except (KeyError, ValueError, ZeroDivisionError):
                    # Failed for this grid point, skip
                    continue

        for transition in transitions:
            emissivities = []

            for T_K in T_grid:
                for n_e in self.n_e_grid_cm3:
                    T_eV = T_K * KB_EV

                    # Get precomputed ionization balance
                    stage_densities = grid_stage_densities.get((T_K, n_e))
                    if stage_densities is None:
                        continue

                    stage_density = stage_densities.get(transition.ionization_stage, 0.0)
                    if stage_density == 0.0:
                        continue

                    W_q = stage_density / total_density

                    try:
                        # Get partition function
                        U_T = self.solver.calculate_partition_function(
                            element, transition.ionization_stage, T_eV
                        )

                        # Emissivity: eps = W^q * A_ki * g_k * exp(-E_k/kT) / U(T)
                        boltzmann_factor = np.exp(-transition.E_k_ev / T_eV)
                        eps = W_q * transition.A_ki * transition.g_k * boltzmann_factor / U_T

                        emissivities.append(eps)
                    except (KeyError, ValueError, ZeroDivisionError):
                        # Failed to compute partition function or emissivity, skip
                        continue

            if emissivities:
                avg_emissivity = np.mean(emissivities)
                line_data.append(
                    {
                        "transition": transition,
                        "avg_emissivity": avg_emissivity,
                        "wavelength_nm": transition.wavelength_nm,
                    }
                )

        return line_data

    def _fuse_lines(self, line_data: List[dict], wavelength_nm: np.ndarray) -> List[dict]:
        """
        Fuse lines within resolution element.

        Parameters
        ----------
        line_data : List[dict]
            List of line dicts from _compute_element_emissivities
        wavelength_nm : np.ndarray
            Experimental wavelength array (for reference wavelength)

        Returns
        -------
        List[dict]
            Fused line list with combined emissivities
        """
        if not line_data:
            return []

        # Sort by wavelength
        sorted_lines = sorted(line_data, key=lambda x: x["wavelength_nm"])

        # Resolution element at mean wavelength — use effective R if available
        mean_wl = np.mean(wavelength_nm)
        eff_R = self._effective_R or self.resolving_power
        delta_lambda = mean_wl / eff_R

        # Group lines within delta_lambda
        fused = []
        current_group = [sorted_lines[0]]

        for i in range(1, len(sorted_lines)):
            line = sorted_lines[i]
            prev_line = current_group[-1]

            if abs(line["wavelength_nm"] - prev_line["wavelength_nm"]) <= delta_lambda:
                # Add to current group
                current_group.append(line)
            else:
                # Finalize current group
                fused.append(self._finalize_group(current_group))
                current_group = [line]

        # Finalize last group
        if current_group:
            fused.append(self._finalize_group(current_group))

        return fused

    def _finalize_group(self, group: List[dict]) -> dict:
        """
        Finalize a group of lines by summing emissivities.

        Parameters
        ----------
        group : List[dict]
            Group of line dicts

        Returns
        -------
        dict
            Fused line dict
        """
        # Sum emissivities
        total_emissivity = sum(line["avg_emissivity"] for line in group)

        # Position = wavelength of strongest line
        strongest = max(group, key=lambda x: x["avg_emissivity"])

        return {
            "transition": strongest["transition"],
            "avg_emissivity": total_emissivity,
            "wavelength_nm": strongest["wavelength_nm"],
            "n_fused": len(group),
        }

    def _match_lines(
        self, fused_lines: List[dict], peaks: List[Tuple[int, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match theoretical lines to experimental peaks.

        Uses auto-calibrated global wavelength shift and effective resolving
        power from _auto_calibrate_wavelength(). Two-pass strategy:
        - Pass 1: tight tolerance (delta_lambda from effective R)
        - Pass 2: for unmatched strong lines, wider tolerance (2x delta_lambda)

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (matched_mask, wavelength_shifts, matched_peak_idx) where
            matched_mask is bool array, wavelength_shifts is float array of
            shifts in nm, and matched_peak_idx is int array (-1 if unmatched)
        """
        n = len(fused_lines)
        if not peaks or not fused_lines:
            return (
                np.zeros(n, dtype=bool),
                np.zeros(n),
                np.full(n, -1, dtype=int),
            )

        peak_wavelengths = np.array([p[1] for p in peaks])

        # Use auto-calibrated shift and effective R
        global_shift = self._global_wl_shift
        eff_R = self._effective_R or self.resolving_power

        # Per-line Stark-aware tolerance (CF-LIBS-improved-u980, wiring PR #133's
        # helper into the line-matching hot path). Each line gets its own
        # tolerance scaled by its specific wavelength: tol_i = lambda_i / eff_R.
        # The Stark-broadening term omega_stark stays 0 here because line dicts
        # don't carry Transition objects at this layer — but the per-wavelength
        # scaling alone is more physically correct than the previous
        # mean_wl/eff_R global tolerance.
        line_wavelengths = np.array([line["wavelength_nm"] for line in fused_lines])
        tol_per_line = np.array(
            [
                get_wavelength_tolerance(
                    wl,
                    transition=None,
                    resolving_power=eff_R,
                    fallback=wl / max(eff_R, 1e-6),
                )
                for wl in line_wavelengths
            ]
        )
        # Mean tolerance kept for the per-element-shift calibration loop below,
        # which uses a single threshold across the top-10 lines.
        delta_lambda = float(np.mean(tol_per_line)) if len(tol_per_line) > 0 else 0.0

        # Additionally refine per-element shift from top 10 lines
        sorted_by_emissivity = sorted(fused_lines, key=lambda x: x["avg_emissivity"], reverse=True)
        top_lines = sorted_by_emissivity[: min(10, len(sorted_by_emissivity))]

        per_element_shifts = []
        for line in top_lines:
            wl_th = line["wavelength_nm"] + global_shift
            distances = np.abs(peak_wavelengths - wl_th)
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist <= 1.5 * delta_lambda:
                    closest_idx = np.argmin(distances)
                    per_element_shifts.append(peak_wavelengths[closest_idx] - line["wavelength_nm"])

        # Use per-element shift if enough matches, else fall back to global
        if len(per_element_shifts) >= 2:
            element_shift = float(np.median(per_element_shifts))
        else:
            element_shift = global_shift

        matched_mask = np.zeros(n, dtype=bool)
        wavelength_shifts = np.zeros(n)
        matched_peak_idx = np.full(n, -1, dtype=int)

        # Pass 1: tight tolerance, per-line
        for i, line in enumerate(fused_lines):
            wl_th = line["wavelength_nm"] + element_shift

            distances = np.abs(peak_wavelengths - wl_th)
            within_window = distances <= tol_per_line[i]

            if np.any(within_window):
                matched_mask[i] = True
                closest_idx = int(np.argmin(distances))
                matched_peak_idx[i] = closest_idx
                wavelength_shifts[i] = peak_wavelengths[closest_idx] - line["wavelength_nm"]

        # Pass 2: for unmatched strong lines, try wider tolerance (2x per-line)
        # "strong" = above median emissivity of all lines
        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])
        emiss_median = np.median(emissivities) if len(emissivities) > 0 else 0.0

        for i, line in enumerate(fused_lines):
            if matched_mask[i]:
                continue  # Already matched
            if emissivities[i] < emiss_median:
                continue  # Only retry strong lines

            wl_th = line["wavelength_nm"] + element_shift
            distances = np.abs(peak_wavelengths - wl_th)
            within_wide = distances <= 2.0 * tol_per_line[i]

            if np.any(within_wide):
                matched_mask[i] = True
                closest_idx = int(np.argmin(distances))
                matched_peak_idx[i] = closest_idx
                wavelength_shifts[i] = peak_wavelengths[closest_idx] - line["wavelength_nm"]

        # Enforce one-to-one: each experimental peak is assigned to at most
        # one theoretical line (highest emissivity wins).  This prevents a
        # single broad peak from "confirming" multiple theoretical lines,
        # which inflates k_rate at low resolving power.
        claimed_peaks: dict = {}  # peak_idx -> (line_idx, emissivity)
        for i in range(n):
            if not matched_mask[i]:
                continue
            pidx = int(matched_peak_idx[i])
            emiss = fused_lines[i]["avg_emissivity"]
            if pidx not in claimed_peaks or emiss > claimed_peaks[pidx][1]:
                if pidx in claimed_peaks:
                    old_i = claimed_peaks[pidx][0]
                    matched_mask[old_i] = False
                    wavelength_shifts[old_i] = 0.0
                    matched_peak_idx[old_i] = -1
                claimed_peaks[pidx] = (i, emiss)
            else:
                # Peak already claimed by a stronger line
                matched_mask[i] = False
                wavelength_shifts[i] = 0.0
                matched_peak_idx[i] = -1

        return matched_mask, wavelength_shifts, matched_peak_idx

    def _determine_emissivity_threshold(
        self, fused_lines: List[dict], matched_mask: np.ndarray
    ) -> float:
        """
        Determine emissivity threshold where detection rate > 50%.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        matched_mask : np.ndarray
            Boolean mask of matched lines

        Returns
        -------
        float
            Log10 emissivity threshold
        """
        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])

        # Avoid log(0)
        emissivities = np.maximum(emissivities, 1e-100)

        log_emissivities = np.log10(emissivities)

        # Bin in log decades
        min_log = np.floor(np.min(log_emissivities))
        max_log = np.ceil(np.max(log_emissivities))
        n_bins = int(max_log - min_log) + 1

        if n_bins < 2:
            # Not enough dynamic range, return minimum
            return min_log

        bins = np.linspace(min_log, max_log, n_bins + 1)

        # Compute detection rate per bin
        bin_indices = np.digitize(log_emissivities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        detection_rates = []
        thresholds = []

        for bin_idx in range(n_bins):
            in_bin = bin_indices == bin_idx
            if np.sum(in_bin) > 0:
                detection_rate = np.sum(matched_mask & in_bin) / np.sum(in_bin)
                detection_rates.append(detection_rate)
                thresholds.append(bins[bin_idx])

        # Find threshold where detection_rate > 0.5
        detection_rates = np.array(detection_rates)
        thresholds = np.array(thresholds)

        above_50 = detection_rates > 0.5
        if np.any(above_50):
            # Return lowest threshold with >50% detection
            candidate = thresholds[np.where(above_50)[0][0]]
        else:
            candidate = min_log

        # Additional constraint: never count more than max_lines_per_element
        # above-threshold lines.  For elements with hundreds of DB lines (Fe,
        # V, Ti), a low threshold counts dozens of weak lines that are below
        # noise.  Raise the threshold until the above-threshold count is
        # manageable, so N_expected reflects only realistically detectable
        # lines rather than the full database catalogue.
        emiss_sorted = np.sort(emissivities)[::-1]
        if len(emiss_sorted) > self.max_lines_per_element:
            # Threshold = emissivity of the (max_lines)th strongest line
            floor = np.log10(max(emiss_sorted[self.max_lines_per_element - 1], 1e-100))
            candidate = max(candidate, floor)

        return candidate

    def _compute_scores(
        self,
        fused_lines: List[dict],
        matched_mask: np.ndarray,
        matched_peak_idx: np.ndarray,
        wavelength_shifts: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        emissivity_threshold: float,
    ) -> Tuple[float, float, float, float, int, int, float]:
        """
        Compute k_sim, k_rate, k_shift scores, P_maj, N_expected, N_matched, P_cov.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        matched_mask : np.ndarray
            Boolean mask of matched lines
        matched_peak_idx : np.ndarray
            Index of matched peak per line (-1 if unmatched)
        wavelength_shifts : np.ndarray
            Wavelength shifts in nm
        intensity : np.ndarray
            Experimental intensity array
        peaks : List[Tuple[int, float]]
            Experimental peaks
        emissivity_threshold : float
            Log10 emissivity threshold

        Returns
        -------
        Tuple[float, float, float, float, int, int, float]
            (k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov)
        """
        if not np.any(matched_mask):
            return 0.0, 0.0, 0.0, 0.5, 0, 0, 0.0

        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])
        above_threshold = emissivities >= 10**emissivity_threshold

        # N_expected: ALL above-threshold theoretical lines (matched or not).
        N_expected = int(np.sum(above_threshold))

        # Filter to lines above threshold that are also matched
        matched_above = matched_mask & above_threshold
        n_matched_above = int(np.sum(matched_above))

        # P_cov: emissivity-weighted coverage penalty — single channel for
        # penalizing missing lines. Missing a weak line matters less than
        # missing the resonance line.
        total_emissivity_above = float(np.sum(emissivities[above_threshold]))
        matched_emissivity = float(np.sum(emissivities[matched_above]))
        P_cov = matched_emissivity / total_emissivity_above if total_emissivity_above > 0 else 0.0

        if n_matched_above == 0:
            return 0.0, 0.0, 0.0, 0.5, N_expected, 0, P_cov

        # Soft P_maj: weighted coverage of top-k strongest above-threshold
        # lines.  Binary P_maj (strongest matched → 1.0, else 0.5) causes
        # false negatives when the major line is obscured by matrix
        # emission (e.g. V in Ti6Al4V where Ti dominates).
        top_k = min(3, N_expected)
        if top_k > 0:
            above_emissivities = emissivities * above_threshold.astype(float)
            sorted_indices = np.argsort(above_emissivities)[::-1][:top_k]
            # sqrt: softer than linear, prevents single dominant line
            # from driving P_maj to 1.0 alone
            weights = np.sqrt(emissivities[sorted_indices])
            matched_weights = float(np.sum(weights * matched_above[sorted_indices]))
            total_weights = float(np.sum(weights))
            P_maj = 0.5 + 0.5 * (matched_weights / total_weights) if total_weights > 0 else 0.5
        else:
            P_maj = 0.5

        # k_rate: emissivity-weighted detection rate.
        if total_emissivity_above > 0:
            k_rate = matched_emissivity / total_emissivity_above
        else:
            k_rate = 0.0

        # k_shift: wavelength match quality
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / self.resolving_power

        shifts_matched = np.abs(wavelength_shifts[matched_above])
        emiss_matched = emissivities[matched_above]
        if len(shifts_matched) > 0 and np.sum(emiss_matched) > 0:
            weighted_shift = np.average(shifts_matched, weights=emiss_matched)
            k_shift = max(0.0, 1.0 - weighted_shift / delta_lambda)
        else:
            k_shift = 0.0

        # k_sim: cosine similarity between theoretical and experimental
        # intensities over MATCHED lines only (paper-faithful).
        # Coverage is handled exclusively by k_rate.
        #
        # Self-absorption correction (gated by self.self_absorption_aware,
        # default True): resonance lines below
        # ``self.self_absorption_e_i_cutoff_ev`` are systematically weaker
        # than optically-thin predictions in high-concentration matrices
        # (Si in soil at 60% SiO2 is the canonical example). Damping the
        # theoretical emissivity by ``self.self_absorption_damping`` avoids
        # penalizing the cosine angle. Counters are reset by ``identify``
        # and reported via a structured log line at the end of each call,
        # so the operator can see which elements got damped — addresses
        # the "silent SA_DAMPING = 0.3" complaint from
        # CF-LIBS-improved-self-abs-audit.
        sa_aware = self.self_absorption_aware
        sa_damping = self.self_absorption_damping
        sa_e_i_cutoff = self.self_absorption_e_i_cutoff_ev
        theoretical_intensities = []
        experimental_intensities = []
        unique_peak_set: set = set()

        for i in range(len(fused_lines)):
            if matched_above[i]:
                eps_th = emissivities[i]
                trans = fused_lines[i]["transition"]
                if sa_aware and getattr(trans, "E_i_ev", 1.0) < sa_e_i_cutoff:
                    eps_th *= sa_damping
                    self._sa_n_damped_lines += 1
                    el = getattr(trans, "element", None)
                    if el is not None:
                        self._sa_damped_elements.add(el)
                theoretical_intensities.append(eps_th)
                pidx = matched_peak_idx[i]
                experimental_intensities.append(intensity[peaks[pidx][0]])
                unique_peak_set.add(pidx)

        if len(theoretical_intensities) > 1:
            th_vec = np.array(theoretical_intensities)
            exp_vec = np.array(experimental_intensities)

            dot_product = np.dot(th_vec, exp_vec)
            norm_th = np.linalg.norm(th_vec)
            norm_exp = np.linalg.norm(exp_vec)

            if norm_th > 0 and norm_exp > 0:
                k_sim = dot_product / (norm_th * norm_exp)
                k_sim = max(0.0, min(1.0, k_sim))
            else:
                k_sim = 0.0
        else:
            # Single matched line: cosine similarity undefined.
            # Set to 0.0 — single-line elements are penalized via k_det
            # blend (N_X=1 means k_sim is not used) and the N_penalty.
            k_sim = 0.0

        # Uniqueness penalty: many-to-one mapping lowers k_sim
        n_unique_peaks = len(unique_peak_set)
        if n_matched_above > 0:
            uniqueness_factor = n_unique_peaks / n_matched_above
            k_sim *= uniqueness_factor

        return k_sim, k_rate, k_shift, P_maj, N_expected, n_matched_above, P_cov

    def _compute_P_ab(self, element: str) -> float:
        """
        Compute crustal-abundance prior P_ab for an element.

        3-tier weighting (Noel et al. 2025):
        - ppm >= 100    → 1.0  (common, > 0.01%)
        - ppm >= 0.001  → 0.75 (intermediate)
        - ppm < 0.001   → 0.5  (rare)

        Parameters
        ----------
        element : str
            Element symbol

        Returns
        -------
        float
            P_ab weighting factor
        """
        log_ppm = self.CRUSTAL_ABUNDANCE_LOG_PPM.get(element, 0.0)
        ppm = 10**log_ppm
        if ppm >= 100:
            return 1.0
        elif ppm >= 1e-3:
            return 0.75
        else:
            return 0.5

    def _compute_fill_factor(
        self,
        peaks: List[Tuple[int, float]],
        wavelength: np.ndarray,
    ) -> float:
        """
        Compute spectral fill factor from merged peak-match windows.

        Each peak contributes an interval centered at its wavelength with half-width:
            chance_window_scale * (lambda / resolving_power)
        Overlapping intervals are merged before computing covered span fraction.

        Parameters
        ----------
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.
        wavelength : np.ndarray
            Full spectral wavelength axis in nm.

        Returns
        -------
        float
            Fraction of spectral span covered by merged intervals in [0, 1].
        """
        if len(peaks) == 0 or len(wavelength) < 2:
            return 0.0

        wl_min = float(np.min(wavelength))
        wl_max = float(np.max(wavelength))
        span = wl_max - wl_min
        if span <= 0:
            return 0.0

        intervals: List[Tuple[float, float]] = []
        for _, peak_wl in peaks:
            half_window = self.chance_window_scale * (peak_wl / self.resolving_power)
            if half_window <= 0:
                continue
            start = max(wl_min, peak_wl - half_window)
            end = min(wl_max, peak_wl + half_window)
            if end > start:
                intervals.append((start, end))

        if not intervals:
            return 0.0

        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        covered = sum(end - start for start, end in merged)
        return float(np.clip(covered / span, 0.0, 1.0))

    def _build_nnls_templates(
        self,
        candidates: List[dict],
        peaks: List[Tuple[int, float]],
    ) -> np.ndarray:
        """
        Build NNLS template matrix from candidate element data.

        Each column is an element's expected peak contribution based on
        Gaussian kernels at instrument resolution centered on each
        theoretical line position.  Three fixes over the original:

        1. **Per-peak sigma** — ``sigma_i = lambda_i / RP / 2.355`` varies
           across the spectral window instead of using the mean wavelength.
        2. **Per-element shift** — median wavelength shift from the matching
           phase is applied to each line position so the template aligns
           with the actual peak locations.
        3. **3-sigma proximity filter** — only peaks within 3 sigma of a
           line receive its contribution.  At 3 sigma the Gaussian is
           ``exp(-4.5) ~ 0.011``, so excluded contributions are < 1%.

        Parameters
        ----------
        candidates : List[dict]
            Candidate dicts with ``fused_lines``, ``matched_mask``, and
            ``wavelength_shifts`` keys.
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.

        Returns
        -------
        np.ndarray
            Matrix A of shape (n_peaks, n_candidates).
        """
        n_peaks = len(peaks)
        n_cands = len(candidates)
        A = np.zeros((n_peaks, n_cands))
        peak_wls = np.array([p[1] for p in peaks])

        # Per-peak sigma (FWHM = lambda/R, sigma = FWHM/2.355)
        peak_sigmas = peak_wls / self.resolving_power / 2.355

        for j, cand in enumerate(candidates):
            # Per-element global shift from matching phase
            mm = cand["matched_mask"]
            ws = cand["wavelength_shifts"]
            shifts = ws[mm] if np.any(mm) else np.array([0.0])
            shift = float(np.median(shifts))

            for line in cand["fused_lines"]:
                wl_shifted = line["wavelength_nm"] + shift
                eps = line["avg_emissivity"]

                # 3-sigma proximity filter: only contribute to nearby peaks
                diffs = np.abs(peak_wls - wl_shifted)
                relevant = diffs < (3.0 * peak_sigmas)
                if np.any(relevant):
                    A[relevant, j] += eps * np.exp(
                        -0.5 * ((peak_wls[relevant] - wl_shifted) / peak_sigmas[relevant]) ** 2
                    )
        return A

    def _build_nnls_templates_jax_wrapper(
        self,
        candidates: List[dict],
        peaks: List[Tuple[int, float]],
    ) -> np.ndarray:
        """Padded-batch wrapper around :func:`build_nnls_templates_jax`.

        Pads each candidate's ``fused_lines`` list to the maximum length so
        the JAX builder can broadcast across all candidates in a single
        call. Padded entries are zeroed via the per-line mask.
        """
        n_peaks = len(peaks)
        n_cands = len(candidates)
        if n_peaks == 0 or n_cands == 0:
            return np.zeros((n_peaks, n_cands))

        peak_wls = np.array([p[1] for p in peaks], dtype=np.float64)

        # Determine padding length.
        L_max = max(len(c["fused_lines"]) for c in candidates)
        if L_max == 0:
            return np.zeros((n_peaks, n_cands))

        lw_pad = np.zeros((n_cands, L_max), dtype=np.float64)
        le_pad = np.zeros((n_cands, L_max), dtype=np.float64)
        mk_pad = np.zeros((n_cands, L_max), dtype=bool)
        shifts = np.zeros(n_cands, dtype=np.float64)

        for j, cand in enumerate(candidates):
            mm = cand["matched_mask"]
            ws = cand["wavelength_shifts"]
            shift_arr = ws[mm] if np.any(mm) else np.array([0.0])
            shifts[j] = float(np.median(shift_arr))
            for k, line in enumerate(cand["fused_lines"]):
                lw_pad[j, k] = line["wavelength_nm"]
                le_pad[j, k] = line["avg_emissivity"]
                mk_pad[j, k] = True

        return build_nnls_templates_jax(
            lw_pad,
            le_pad,
            mk_pad,
            shifts,
            peak_wls,
            self.resolving_power,
        )

    def _compute_nnls_attribution(
        self,
        A: np.ndarray,
        peak_intensities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve NNLS and return per-element attribution metrics.

        Returns three arrays:

        * **P_mix** — leave-one-out partial-R^2 (absolute).
        * **P_local** — local explanation score: what fraction of the
          observed intensity at claimed peaks is explained by this
          element's NNLS contribution?  FP elements that merely ride on
          a dominant element's peaks get P_local ~ 0.
        * **c** — raw NNLS coefficients (useful for diagnostics).

        Parameters
        ----------
        A : np.ndarray
            Template matrix (n_peaks, n_candidates).
        peak_intensities : np.ndarray
            Observed peak intensities (n_peaks,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (P_mix, P_local, c) arrays of length n_candidates.
        """
        n_cands = A.shape[1]
        if n_cands == 0 or np.all(A == 0):
            return np.ones(n_cands), np.ones(n_cands), np.zeros(n_cands)

        c, _ = nnls(A, peak_intensities)
        total_rss = float(np.sum((peak_intensities - A @ c) ** 2))

        # Total signal energy (denominator for partial R^2)
        total_energy = float(np.sum(peak_intensities**2))
        if total_energy == 0:
            return np.ones(n_cands), np.ones(n_cands), c

        # ── P_mix: leave-one-out partial R^2 ──
        P_mix = np.zeros(n_cands)
        for j in range(n_cands):
            A_reduced = np.delete(A, j, axis=1)
            if A_reduced.shape[1] == 0:
                P_mix[j] = 1.0
                continue
            c_reduced, _ = nnls(A_reduced, peak_intensities)
            rss_without = float(np.sum((peak_intensities - A_reduced @ c_reduced) ** 2))
            P_mix[j] = (rss_without - total_rss) / total_energy

        # ── P_local: local explanation score ──
        # For each element, compute what fraction of the observed
        # intensity at its claimed peaks is explained by its own
        # NNLS contribution.  This discriminates FP elements
        # (tiny coefficient on dominant-element peaks) from real
        # minor elements (significant coefficient on their own peaks).
        P_local = np.zeros(n_cands)
        for j in range(n_cands):
            # Peaks where element j has meaningful template presence
            claimed = A[:, j] > 1e-6
            if not np.any(claimed):
                P_local[j] = 0.0
                continue
            obs_at_claimed = np.sum(peak_intensities[claimed])
            if obs_at_claimed <= 0:
                P_local[j] = 0.0
                continue
            elem_contribution = np.sum(A[claimed, j] * c[j])
            P_local[j] = float(np.clip(elem_contribution / obs_at_claimed, 0.0, 1.0))

        return P_mix, P_local, c

    @staticmethod
    def _compute_sparse_nnls_scores(
        A: np.ndarray,
        peak_intensities: np.ndarray,
        alpha: float = 0.01,
        l1_ratio: float = 0.9,
    ) -> Tuple[np.ndarray, float]:
        """
        Sparse NNLS via L-BFGS-B constrained optimization.

        Standard NNLS distributes signal across correlated endmembers,
        producing many small non-zero coefficients for absent elements
        (Black & Burnside 2024). The L1 penalty enforces sparsity, driving
        truly absent elements to zero.

        Physics-only implementation: minimizes the elastic-net objective
        with non-negativity via L-BFGS-B bounds rather than sklearn.

        Parameters
        ----------
        A : np.ndarray
            Template matrix (n_peaks, n_candidates).
        peak_intensities : np.ndarray
            Observed peak intensities (n_peaks,).
        alpha : float
            Regularization strength (higher = sparser).
        l1_ratio : float
            L1 vs L2 mix (1.0 = pure lasso, 0.0 = pure ridge).

        Returns
        -------
        Tuple[np.ndarray, float]
            (coefficients, residual_norm) — sparse non-negative coefficients
            and the norm of the fit residual.
        """
        n_cands = A.shape[1]
        if n_cands == 0 or np.all(A == 0) or np.all(peak_intensities == 0):
            return np.zeros(n_cands), 0.0

        try:
            # Physics-only non-negative elastic-net via L-BFGS-B. Mirrors
            # sklearn.linear_model.ElasticNet(positive=True, ...) but without
            # importing an ML library (see CF-LIBS-improved-3fy3): minimize
            #   0.5 * ||A_norm x - y||^2
            #   + alpha * ( l1_ratio * sum(x) + 0.5 * (1-l1_ratio) * x^T x )
            # subject to x >= 0. Under x >= 0 the L1 term reduces to a smooth
            # linear sum(x), so the full objective is differentiable and
            # L-BFGS-B handles the non-negativity via bounds.
            from scipy.optimize import minimize

            col_norms = np.linalg.norm(A, axis=0)
            col_norms[col_norms == 0] = 1.0
            A_norm = A / col_norms

            l1_weight = float(alpha * l1_ratio)
            l2_weight = float(alpha * (1.0 - l1_ratio))

            def _loss_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
                r = A_norm @ x - peak_intensities
                loss = (
                    0.5 * float(r @ r)
                    + l1_weight * float(np.sum(x))
                    + 0.5 * l2_weight * float(x @ x)
                )
                grad = A_norm.T @ r + l1_weight + l2_weight * x
                return loss, grad

            result = minimize(
                _loss_grad,
                x0=np.zeros(n_cands),
                jac=True,
                bounds=[(0.0, None)] * n_cands,
                method="L-BFGS-B",
                options={"maxiter": 2000},
            )
            sparse_c = np.asarray(result.x) / col_norms
            residual = float(np.linalg.norm(peak_intensities - A @ sparse_c))
        except Exception:
            # Fallback to standard NNLS (no sparsity regularization).
            from scipy.optimize import nnls as _nnls

            sparse_c, residual = _nnls(A, peak_intensities)
            residual = float(residual)

        return sparse_c, residual

    def _compute_ratio_consistency(
        self,
        fused_lines: List[dict],
        matched_mask: np.ndarray,
        matched_peak_idx: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
    ) -> float:
        """
        Intensity-ratio consistency between matched lines.

        For real elements, the pairwise log-ratios of observed peak
        intensities should correlate with theoretical emissivity
        log-ratios (both follow the same Boltzmann distribution).
        Coincidental matches hit peaks belonging to a *different*
        element, so the ratios are uncorrelated.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused lines with 'avg_emissivity' keys.
        matched_mask : np.ndarray
            Boolean mask of matched lines.
        matched_peak_idx : np.ndarray
            Peak indices for matched lines.
        intensity : np.ndarray
            Experimental intensity array.
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.

        Returns
        -------
        float
            R_rat in [0, 1].  1.0 = perfect ratio match, 0.0 = anti-
            correlated.  Returns 0.5 (neutral) with < 3 matched lines.
        """
        matched_indices = np.where(matched_mask)[0]
        if len(matched_indices) < 3:
            return 0.1  # Penalize — too few lines for meaningful ratio check

        # Apply self-absorption damping to resonance lines so theoretical
        # log-ratios better match observed ratios for strong transitions.
        # Gated by self.self_absorption_aware (default True); previously
        # this used a hardcoded ``SA_DAMPING = 0.3``. The counters mutated
        # below feed the post-identify summary log line for transparency.
        sa_aware = self.self_absorption_aware
        sa_damping_value = self.self_absorption_damping if sa_aware else 1.0
        sa_e_i_cutoff = self.self_absorption_e_i_cutoff_ev
        raw_emiss = np.array([fused_lines[i]["avg_emissivity"] for i in matched_indices])
        damping_values = []
        for i in matched_indices:
            trans = fused_lines[i]["transition"]
            if sa_aware and getattr(trans, "E_i_ev", 1.0) < sa_e_i_cutoff:
                damping_values.append(sa_damping_value)
                self._sa_n_damped_lines += 1
                el = getattr(trans, "element", None)
                if el is not None:
                    self._sa_damped_elements.add(el)
            else:
                damping_values.append(1.0)
        damping = np.array(damping_values)
        emissivities = raw_emiss * damping
        obs_intensities = np.array(
            [intensity[peaks[matched_peak_idx[i]][0]] for i in matched_indices]
        )

        # Guard against zeros
        valid = (emissivities > 0) & (obs_intensities > 0)
        if np.sum(valid) < 3:
            return 0.5

        log_th = np.log(emissivities[valid])
        log_obs = np.log(obs_intensities[valid])

        # Build all pairwise log-ratio differences
        n = len(log_th)
        th_ratios = []
        exp_ratios = []
        for i in range(n):
            for j in range(i + 1, n):
                th_ratios.append(log_th[i] - log_th[j])
                exp_ratios.append(log_obs[i] - log_obs[j])

        if len(th_ratios) < 3:
            return 0.5

        th_arr = np.array(th_ratios)
        exp_arr = np.array(exp_ratios)

        # Pearson correlation of log-ratios
        corr = np.corrcoef(th_arr, exp_arr)[0, 1]
        if np.isnan(corr):
            return 0.5

        # Map [-1, 1] → [0, 1]; negative correlation is worse than zero
        return float(max(0.0, (corr + 1.0) / 2.0))

    def _compute_random_match_significance(
        self,
        peaks: List[Tuple[int, float]],
        wavelength: np.ndarray,
        N_expected: int,
        N_matched: int,
    ) -> Tuple[float, float, float, float]:
        """
        Compute chance-coincidence significance from a binomial tail test.

        Uses per-element theoretical line occupancy as the chance probability:
        p_chance = fraction of spectral span occupied by N_expected
        above-threshold theoretical line windows (not experimental peaks).

        Parameters
        ----------
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.
        wavelength : np.ndarray
            Full spectral wavelength axis in nm.
        N_expected : int
            Number of above-threshold theoretical lines.
        N_matched : int
            Number of above-threshold lines matched to peaks.

        Returns
        -------
        Tuple[float, float, float, float]
            (P_sig, fill_factor, p_chance, p_tail), where:
            - fill_factor is experimental peak fill factor (for metadata)
            - p_chance is theoretical-window occupancy
            - p_tail = P(X >= N_matched | n=N_expected, p=p_chance)
            - P_sig = 1 - p_tail
        """
        fill_factor = self._compute_fill_factor(peaks, wavelength)

        # Theoretical-window occupancy: per-element chance probability
        wl_min = float(np.min(wavelength))
        wl_max = float(np.max(wavelength))
        span = wl_max - wl_min
        if span <= 0 or N_expected <= 0:
            p_chance = float(np.clip(fill_factor, 1e-6, 1.0 - 1e-6))
        else:
            mean_wl = 0.5 * (wl_min + wl_max)
            line_window = mean_wl / self.resolving_power  # delta_lambda
            # Each line occupies ±line_window around its center
            theoretical_coverage = N_expected * 2 * line_window / span
            p_chance = float(np.clip(theoretical_coverage, 1e-6, 1.0 - 1e-6))

        if N_expected <= 0 or N_matched <= 0:
            return 1.0, fill_factor, p_chance, 1.0

        # Binomial test: "Given N_expected opportunities, what's the
        # probability of N_matched or more matches by chance?"
        n_trials = N_expected
        n_success = N_matched

        if n_success > n_trials:
            # More matches than theoretical lines — extremely unlikely
            # by chance.  Can happen with fused-line bookkeeping; treat
            # as maximally significant.
            return 1.0, fill_factor, p_chance, 0.0

        p_tail = float(binom.sf(n_success - 1, n_trials, p_chance))
        P_sig = float(np.clip(1.0 - p_tail, 0.0, 1.0))

        return P_sig, fill_factor, p_chance, p_tail

    @staticmethod
    def _compute_p_snr(intensity: np.ndarray, peaks: List[Tuple[int, float]]) -> float:
        """Compute erf-based SNR quality factor used in CL (CPU path)."""
        if len(peaks) > 0:
            peak_intensities_local = [intensity[p[0]] for p in peaks]
            median_peak = np.median(peak_intensities_local)
            noise_estimate = np.median(np.abs(intensity - np.median(intensity))) * 1.4826
            noise_estimate = max(noise_estimate, 1e-10)
            z = (median_peak - noise_estimate) / (noise_estimate * math.sqrt(2))
            return 0.5 * (1.0 + float(erf(z)))
        return 0.5

    def _dispatch_p_snr(self, intensity: np.ndarray, peaks: List[Tuple[int, float]]) -> float:
        """Dispatch P_SNR computation to CPU or JAX path based on opt-in flag."""
        if self.use_jax_p_snr and _HAS_JAX:
            if not peaks:
                return 0.5
            peak_indices = np.array([p[0] for p in peaks], dtype=np.int32)
            return compute_p_snr_jax(intensity, peak_indices)
        return self._compute_p_snr(intensity, peaks)

    def _decide(
        self,
        k_sim: float,
        k_rate: float,
        k_shift: float,
        N_expected: int,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        element: str = "",
        P_maj: float = 0.5,
        P_sig: float = 1.0,
        N_matched: int = 0,
        P_cov: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute detection score k_det and confidence level CL.

        Parameters
        ----------
        k_sim : float
            Similarity score (matched-only cosine similarity)
        k_rate : float
            Detection rate score (emissivity-weighted)
        k_shift : float
            Wavelength shift score
        N_expected : int
            Number of above-threshold theoretical lines (for gates/penalties)
        intensity : np.ndarray
            Experimental intensity array
        peaks : List[Tuple[int, float]]
            Experimental peaks
        element : str
            Element symbol (for crustal abundance weighting)
        P_maj : float
            Major-line coverage factor (0.5–1.0), computed from top-k
            strongest theoretical lines
        P_sig : float
            Statistical significance factor against random coincidence
        N_matched : int
            Number of matched above-threshold lines (used in k_det blend)
        P_cov : float
            Emissivity-weighted coverage penalty (0–1)

        Returns
        -------
        Tuple[float, float]
            (k_det, CL) detection score and confidence level.
        """
        # k_det formula — uses N_matched (paper: N_X = matched count)
        # for the blend weighting.  Single-line elements (N_X=1) naturally
        # reduce to k_rate × k_shift via the blend formula.
        #
        # Modified from original: blend P_cov (emissivity-weighted coverage)
        # into k_det so that elements with many weak undetected lines are
        # not excessively penalized.  P_cov weights by emissivity, so missing
        # a weak line (emissivity 1% of total) only reduces P_cov by 1%.
        if N_matched > 0:
            N_X = N_matched
            k_det_raw = k_rate * ((1.0 / N_X) * k_shift + ((N_X - 1.0) / N_X) * k_sim)
            # Blend: use geometric mean of raw k_det and P_cov to soften
            # the penalty for many unmatched weak lines
            k_det = math.sqrt(k_det_raw * max(P_cov, 0.01))
        else:
            k_det = 0.0

        # Fix 4: N_expected penalty — elements with few expected lines
        # get scaled down to prevent 2/3 matches from scoring high.
        N_penalty = min(1.0, math.sqrt(N_expected / 5.0)) if N_expected > 0 else 0.0
        k_det *= N_penalty

        P_SNR = self._dispatch_p_snr(intensity, peaks)

        # P_ab — crustal abundance prior
        P_ab = self._compute_P_ab(element)

        # Confidence level — paper formula (Noel et al. 2025):
        # CL = k_det × P_SNR × P_maj × P_ab
        CL = k_det * P_SNR * P_maj * P_ab

        # Hard gate — reject if too few lines matched.
        # At RP<1000, matching 2 lines by chance is trivial for elements
        # with few expected lines (Na, K). Require enough matches to be
        # statistically meaningful.
        if N_expected <= 1:
            # Single-line elements (H-alpha): 1 match is sufficient
            pass
        elif N_expected <= 4:
            # Sparse elements (Na, K, Li): require ALL lines matched
            # AND elevated CL to pass — chance matching 2/2 is too easy
            if N_matched < N_expected:
                CL = 0.0
        else:
            # Normal elements: require at least 3 matched lines
            if N_matched < 3:
                CL = 0.0

        return k_det, CL

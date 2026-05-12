"""
Full-spectrum NNLS element identifier for CF-LIBS.

Replaces peak-matching (ALIAS) with full-spectrum forward-model decomposition:
at estimated (T, ne), the observed spectrum is decomposed as a non-negative
linear combination of single-element basis spectra via NNLS.

Architecture: Preprocess → Estimate (T, ne) → Decompose (NNLS) → Validate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from cflibs.core.logging_config import get_logger
from cflibs.inversion.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.preprocessing import estimate_baseline

if TYPE_CHECKING:
    from cflibs.manifold.basis_index import BasisIndex
    from cflibs.manifold.basis_library import BasisLibrary

logger = get_logger("inversion.spectral_nnls")


# ---------------------------------------------------------------------------
# JAX NNLS kernel (opt-in fast path for GPU batching)
#
# Default solver is ``scipy.optimize.nnls`` (Lawson--Hanson active set);
# when ``use_jax_nnls=True`` is passed to :class:`SpectralNNLSIdentifier`
# we replace the per-identify() solve with a hand-rolled FISTA solver on
# the Gram form. FISTA is convex, jit-friendly, and vmap-able across
# row-masked subsets of the basis matrix --- which is exactly what
# :func:`cflibs.inversion.identify.model_selection.bic_prune_elements`
# needs. The two paths agree on the residual norm to ~1e-5 rtol; the
# coefficient vector itself may differ slightly on rank-deficient
# problems (NNLS has multiple minimizers in that case, and Lawson--Hanson
# vs FISTA select different vertices of the feasible polytope). See
# ``docs/jax-port/nnls-consultation.md`` for the Codex+Opus synthesis
# this design follows.
try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only when jax missing
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


if _HAS_JAX:
    from functools import partial

    @partial(jax.jit, static_argnames=("max_iter",))
    def _jax_nnls_gram_fista(
        G: "jnp.ndarray",
        c: "jnp.ndarray",
        max_iter: int = 300,
        ridge: float = 1e-12,
    ) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]":
        """FISTA-accelerated projected gradient NNLS in Gram form.

        Solves ``min_x>=0  0.5 * x^T G x - c^T x`` which is the Gram form
        of ``min_x>=0  0.5 * ||A x - b||^2`` with ``G = A^T A`` and
        ``c = A^T b``. Returns ``(x, kkt_viol, primal_obj)``.

        Notes
        -----
        - ``ridge`` adds ``ridge * I`` to ``G`` for stability on
          rank-deficient bases. Default 1e-12 is small enough that it
          does not perturb well-conditioned solves.
        - Iteration count is fixed (``lax.scan``) to keep the function
          jit-able with static shapes. 300 iterations matches the
          convergence target from the consultation
          (``||x_fista - x_exact|| / ||x_exact|| ~ 1e-4 .. 1e-5``).
        """
        n = G.shape[-1]
        G_reg = G + ridge * jnp.eye(n, dtype=G.dtype)

        # Step size 1/L with L = spectral norm of G. ``jnp.linalg.norm``
        # with default ord=None on a 2-D matrix returns the Frobenius
        # norm, so explicitly request the spectral (operator) norm.
        L = jnp.linalg.norm(G_reg, ord=2) + 1e-30
        alpha = 1.0 / L

        x0 = jnp.zeros(n, dtype=G.dtype)

        def body(carry, _):
            x_prev, y_prev, t_prev = carry
            grad = G_reg @ y_prev - c
            x_new = jnp.maximum(y_prev - alpha * grad, 0.0)
            t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_prev * t_prev))
            y_new = x_new + ((t_prev - 1.0) / t_new) * (x_new - x_prev)
            return (x_new, y_new, t_new), None

        init = (x0, x0, jnp.asarray(1.0, dtype=G.dtype))
        (x_final, _, _), _ = jax.lax.scan(body, init, xs=None, length=max_iter)

        # KKT residual: ``min(x, grad)`` should be ~0 elementwise at
        # optimum. For inactive constraints (x > 0) require |grad| small;
        # for active ones (x == 0) require grad >= 0.
        grad_final = G_reg @ x_final - c
        kkt = jnp.where(x_final > 0, jnp.abs(grad_final), jnp.maximum(-grad_final, 0.0))
        kkt_viol = jnp.max(kkt)

        primal_obj = 0.5 * jnp.dot(x_final, G_reg @ x_final) - jnp.dot(c, x_final)
        return x_final, kkt_viol, primal_obj


def nnls_jax(
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int = 300,
    ridge: float = 1e-12,
    return_diagnostics: bool = False,
):
    """JAX FISTA NNLS solver. Drop-in for ``scipy.optimize.nnls``.

    Solves ``min_x>=0 ||A x - b||_2`` via FISTA on the Gram form
    ``G = A^T A``, ``c = A^T b``. Designed to match
    :func:`scipy.optimize.nnls` on **residual norm** to rtol ~1e-5;
    coefficient agreement is rtol ~1e-4 in the well-conditioned case
    and may be looser on rank-deficient problems (multiple NNLS
    minimizers exist there).

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Design matrix.
    b : np.ndarray, shape (m,)
        Right-hand side.
    max_iter : int, optional
        Number of FISTA iterations (fixed, no early exit). Default 300
        per the consultation. Raise to 500-1000 if KKT violation is too
        high on your problem.
    ridge : float, optional
        Tiny Tikhonov regularization on the Gram matrix for stability.
        Default 1e-12.
    return_diagnostics : bool, optional
        If True, returns ``(x, residual_norm, kkt_viol)`` instead of
        just ``(x, residual_norm)``. Default False (matches scipy.nnls
        return signature).

    Returns
    -------
    x : np.ndarray, shape (n,)
        Non-negative least-squares solution.
    residual_norm : float
        ``||A x - b||_2`` (matches scipy.nnls).
    kkt_viol : float, optional
        Returned only if ``return_diagnostics=True``.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for nnls_jax. Install with: pip install jax jaxlib"
        )

    A_arr = np.asarray(A, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A_arr.shape}")
    if b_arr.ndim != 1:
        raise ValueError(f"b must be 1-D, got shape {b_arr.shape}")
    if A_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(
            f"A.shape[0]={A_arr.shape[0]} does not match b.shape[0]={b_arr.shape[0]}"
        )

    # Form Gram system in float64.
    G_np = A_arr.T @ A_arr
    c_np = A_arr.T @ b_arr

    x_jax, kkt_viol, _obj = _jax_nnls_gram_fista(
        jnp.asarray(G_np), jnp.asarray(c_np), int(max_iter), float(ridge)
    )
    x = np.asarray(x_jax, dtype=np.float64)
    residual_norm = float(np.linalg.norm(A_arr @ x - b_arr))

    if return_diagnostics:
        return x, residual_norm, float(np.asarray(kkt_viol))
    return x, residual_norm


def nnls_jax_batch(
    A: np.ndarray,
    b: np.ndarray,
    row_masks: np.ndarray,
    max_iter: int = 300,
    ridge: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched NNLS via FISTA, vmapped over per-batch row masks.

    Solves ``min_x>=0 || (A * mask_b)^T x - b ||`` for each batch
    element ``b``, where ``mask_b`` zeros out specific rows of the basis
    matrix. This is the pattern needed by BIC backward elimination,
    where each iteration masks out a different candidate element.

    Parameters
    ----------
    A : np.ndarray, shape (n_components, n_pixels)
        Full basis matrix. (Note: same convention as
        ``BasisLibrary.get_basis_matrix_interp`` --- rows are
        components, columns are pixels.)
    b : np.ndarray, shape (n_pixels,)
        Observed spectrum.
    row_masks : np.ndarray, shape (B, n_components)
        Boolean (or 0/1 float) masks. Each row selects which components
        are active for that batch element. Masked-out components are
        forced to coefficient 0.
    max_iter : int, optional
        FISTA iterations. Default 300.
    ridge : float, optional
        Tikhonov regularization. Default 1e-12.

    Returns
    -------
    X : np.ndarray, shape (B, n_components)
        Solutions. Masked-out components are guaranteed exactly 0.
    residual_norms : np.ndarray, shape (B,)
        ``||A_b^T x_b - b||`` for each batch element.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "JAX is required for nnls_jax_batch. Install with: pip install jax jaxlib"
        )

    A_arr = np.asarray(A, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    masks_arr = np.asarray(row_masks, dtype=np.float64)
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A_arr.shape}")
    if b_arr.ndim != 1 or b_arr.shape[0] != A_arr.shape[1]:
        raise ValueError(
            f"b shape {b_arr.shape} incompatible with A shape {A_arr.shape}"
        )
    if masks_arr.ndim != 2 or masks_arr.shape[1] != A_arr.shape[0]:
        raise ValueError(
            f"row_masks shape {masks_arr.shape} incompatible with "
            f"A.shape[0]={A_arr.shape[0]}"
        )

    # Build per-batch Gram systems via einsum. A is (n_comp, n_pix);
    # masked design matrix M_b is (n_pix, n_comp) with M_b[:, j] =
    # mask_b[j] * A[j, :].T.  G_b = (A * mask[:, None]) @ (A * mask[:, None]).T
    # i.e. G_b[i, j] = mask_b[i] * mask_b[j] * sum_k A[i, k] * A[j, k].
    # In einsum: G_b[i, j] = mask[b, i] * mask[b, j] * (A @ A.T)[i, j].
    AAt = A_arr @ A_arr.T  # (n_comp, n_comp)
    G = masks_arr[:, :, None] * masks_arr[:, None, :] * AAt[None, :, :]
    c = masks_arr * (A_arr @ b_arr)[None, :]  # (B, n_comp)

    # vmap can't carry a static_argnames over the leading batch axis, so
    # close over max_iter and ridge before vmapping.
    _iter = int(max_iter)
    _ridge = float(ridge)

    def _solve_one(G_b, c_b):
        return _jax_nnls_gram_fista(G_b, c_b, _iter, _ridge)

    solver = jax.vmap(_solve_one, in_axes=(0, 0))
    X_jax, _kkt, _obj = solver(jnp.asarray(G), jnp.asarray(c))
    X = np.asarray(X_jax, dtype=np.float64)
    # Force masked-out components to exactly 0 (ridge + FISTA can leave
    # tiny ~1e-15 residuals on inactive entries because mask multiplies
    # both G rows/cols but not the initial x = 0 update step).
    X = X * (masks_arr > 0)

    # Per-batch residual norms.
    # predicted_b = sum_j X[b, j] * A[j, :]  -> (B, n_pix)
    predicted = X @ A_arr  # (B, n_pix)
    residual_norms = np.linalg.norm(predicted - b_arr[None, :], axis=1)

    return X, residual_norms


class SpectralNNLSIdentifier:
    """
    Element identification via NNLS decomposition of the full observed
    spectrum into single-element basis spectra.

    At a given (T, ne), the observed spectrum is modeled as:

        observed(λ) ≈ Σᵢ cᵢ · basis_i(λ; T, ne) + continuum(λ)

    where basis_i is the pre-computed synthetic spectrum of element i.
    NNLS enforces cᵢ ≥ 0.  Elements with cᵢ above a significance
    threshold (SNR > detection_snr) are reported as detected.

    Parameters
    ----------
    basis_library : BasisLibrary
        Pre-computed single-element basis library.
    basis_index : BasisIndex or None
        FAISS index for fast (T, ne) estimation.  If None, uses
        ``fallback_T_K`` and ``fallback_ne_cm3`` directly.
    detection_snr : float
        Minimum coefficient SNR for element detection (default: 3.0).
    continuum_degree : int
        Degree of polynomial continuum added to basis matrix (default: 3).
        Set to -1 to disable continuum fitting.
    fallback_T_K : float
        Temperature to use if basis_index is None (default: 8000.0).
    fallback_ne_cm3 : float
        Electron density to use if basis_index is None (default: 1e17).

    Notes
    -----
    Not thread-safe: identify() caches estimated (T, ne) on the instance.
    """

    def __init__(
        self,
        basis_library: BasisLibrary,
        basis_index: Optional[BasisIndex] = None,
        detection_snr: float = 3.0,
        continuum_degree: int = 3,
        fallback_T_K: float = 8000.0,
        fallback_ne_cm3: float = 1e17,
        use_jax_nnls: bool = False,
        jax_nnls_max_iter: int = 300,
    ):
        self.basis_library = basis_library
        self.basis_index = basis_index
        self.detection_snr = detection_snr
        self.continuum_degree = continuum_degree
        self.fallback_T_K = fallback_T_K
        self.fallback_ne_cm3 = fallback_ne_cm3
        self.use_jax_nnls = bool(use_jax_nnls)
        self.jax_nnls_max_iter = int(jax_nnls_max_iter)
        if self.use_jax_nnls and not _HAS_JAX:  # pragma: no cover
            raise ImportError(
                "use_jax_nnls=True requires JAX. "
                "Install with: pip install jax jaxlib"
            )

        # Warn if basis library includes ionization stages > II
        if hasattr(basis_library, "config") and hasattr(basis_library.config, "ionization_stages"):
            for stage in basis_library.config.ionization_stages:
                if stage > 2:
                    logger.warning(
                        "Basis library includes ionization stage %d; "
                        "stages > II are uncommon in LTE CF-LIBS and may "
                        "produce unreliable results.",
                        stage,
                    )

        # Cached per identify() call
        self._estimated_T: Optional[float] = None
        self._estimated_ne: Optional[float] = None

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """
        Identify elements in an observed LIBS spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm.
        intensity : np.ndarray
            Intensity array (arbitrary units).

        Returns
        -------
        ElementIdentificationResult
            Detected and rejected elements with metadata.
        """
        # Input validation
        wavelength = np.asarray(wavelength, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)
        if wavelength.ndim != 1 or intensity.ndim != 1:
            raise ValueError("wavelength and intensity must be 1-D arrays")
        if len(wavelength) == 0 or len(intensity) == 0:
            raise ValueError("wavelength and intensity must be non-empty")
        if len(wavelength) != len(intensity):
            raise ValueError(
                f"wavelength ({len(wavelength)}) and intensity ({len(intensity)}) "
                "must have the same length"
            )

        # Ensure monotonically increasing wavelength (sort if needed)
        sort_idx = np.argsort(wavelength)
        if not np.array_equal(sort_idx, np.arange(len(wavelength))):
            logger.debug("Sorting wavelength array to ensure monotonic order")
            wavelength = wavelength[sort_idx]
            intensity = intensity[sort_idx]

        # Step 1: Preprocess — baseline subtraction
        baseline = estimate_baseline(wavelength, intensity)
        corrected = np.maximum(intensity - baseline, 0.0)

        # Step 2: Resample to basis library wavelength grid FIRST,
        # then area-normalize on that grid (grid-independent normalization)
        lib_wl = self.basis_library.wavelength
        observed_resampled = np.interp(lib_wl, wavelength, corrected, left=0.0, right=0.0)

        # Area-normalize on the library grid to match basis library convention
        area = np.trapezoid(observed_resampled, lib_wl)
        if area > 1e-20:
            observed_resampled = observed_resampled / area

        # Step 3: Estimate (T, ne) via FAISS or fallback
        if self.basis_index is not None and self.basis_index.is_built:
            T_est, ne_est, _details = self.basis_index.estimate_plasma_params(
                observed_resampled, k=50
            )
        else:
            T_est = self.fallback_T_K
            ne_est = self.fallback_ne_cm3

        self._estimated_T = T_est
        self._estimated_ne = ne_est

        # Step 4: Retrieve basis matrix at estimated (T, ne)
        basis_matrix = self.basis_library.get_basis_matrix_interp(T_est, ne_est)
        # basis_matrix shape: (n_elements, n_pixels)
        elements = self.basis_library.elements
        n_elements = len(elements)

        # Step 5: Build augmented matrix with continuum polynomials
        A = self._build_augmented_matrix(basis_matrix, lib_wl)

        # Step 6: Solve NNLS (SciPy Lawson-Hanson by default; opt-in JAX
        # FISTA path when use_jax_nnls=True --- residual norms agree to
        # ~1e-5 rtol, coefficient agreement is rtol ~1e-4 in the
        # well-conditioned case)
        if self.use_jax_nnls:
            coefficients, residual_norm = nnls_jax(
                A.T, observed_resampled, max_iter=self.jax_nnls_max_iter
            )
        else:
            coefficients, residual_norm = nnls(A.T, observed_resampled)

        # Extract element coefficients (first n_elements) and continuum
        element_coeffs = coefficients[:n_elements]

        # Step 7: Compute significance (SNR) for each element
        residual = observed_resampled - A.T @ coefficients
        n_params = A.shape[0]
        dof = max(len(residual) - n_params, 1)
        residual_var = float(np.sum(residual**2) / dof) if len(residual) > 0 else 1e-20
        # Use a realistic noise floor: 1% of the signal variance
        signal_var = float(np.var(observed_resampled)) if len(observed_resampled) > 0 else 1e-20
        noise_floor = max(signal_var * 1e-4, 1e-20)
        residual_var = max(residual_var, noise_floor)

        # Coefficient uncertainties from (A^T A)^-1 diagonal
        AtA = A @ A.T
        try:
            AtA_inv_diag = np.diag(np.linalg.inv(AtA + 1e-12 * np.eye(len(AtA))))
            sigma_coeffs = np.sqrt(np.maximum(residual_var * AtA_inv_diag[:n_elements], 0.0))
        except np.linalg.LinAlgError:
            logger.debug("AtA inversion failed; using fallback uncertainty estimate (0.1)")
            sigma_coeffs = np.ones(n_elements) * 0.1

        snr = element_coeffs / np.maximum(sigma_coeffs, 1e-10)

        # Step 8: Build results
        all_element_ids: List[ElementIdentification] = []

        for i, element in enumerate(elements):
            coeff = float(element_coeffs[i])
            element_snr = float(snr[i])
            detected = element_snr >= self.detection_snr and coeff > 1e-10

            # Compute concentration estimate (fraction of total element signal)
            total_element_signal = float(np.sum(element_coeffs))
            concentration = coeff / total_element_signal if total_element_signal > 0 else 0.0

            element_id = ElementIdentification(
                element=element,
                detected=detected,
                score=min(element_snr / 10.0, 1.0),  # Normalize to 0-1
                confidence=min(element_snr / 10.0, 1.0),
                n_matched_lines=0,  # Full-spectrum: no individual line matching
                n_total_lines=0,
                matched_lines=[],
                unmatched_lines=[],
                metadata={
                    "nnls_coefficient": coeff,
                    "nnls_snr": element_snr,
                    "sigma_coeff": float(sigma_coeffs[i]),
                    "concentration_estimate": concentration,
                    "estimated_T_K": T_est,
                    "estimated_ne_cm3": ne_est,
                },
            )
            all_element_ids.append(element_id)

        # Split by detection
        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=[],  # Full-spectrum: no peak list
            n_peaks=0,
            n_matched_peaks=0,
            n_unmatched_peaks=0,
            algorithm="spectral_nnls",
            parameters={
                "detection_snr": self.detection_snr,
                "continuum_degree": self.continuum_degree,
                "estimated_T_K": T_est,
                "estimated_ne_cm3": ne_est,
                "residual_norm": float(residual_norm),
                "n_elements_tested": n_elements,
                "n_detected": len(detected_elements),
            },
        )

    def _build_augmented_matrix(
        self,
        basis_matrix: np.ndarray,
        wavelength: np.ndarray,
    ) -> np.ndarray:
        """
        Build augmented matrix with element basis + polynomial continuum.

        Returns shape (n_elements + n_poly, n_pixels).
        """
        components = [basis_matrix]

        if self.continuum_degree >= 0:
            # Normalized wavelength for numerical stability
            wl_min, wl_max = wavelength[0], wavelength[-1]
            wl_norm = (wavelength - wl_min) / max(wl_max - wl_min, 1e-10)

            poly_cols = []
            for deg in range(self.continuum_degree + 1):
                col = wl_norm**deg
                # Normalize polynomial columns to similar scale as basis
                col_norm = np.sum(np.abs(col))
                if col_norm > 1e-20:
                    col /= col_norm
                poly_cols.append(col.reshape(1, -1))

            components.append(np.vstack(poly_cols))

        return np.vstack(components)

"""
Full-spectrum NNLS element identifier for CF-LIBS.

Replaces peak-matching (ALIAS) with full-spectrum forward-model decomposition:
at estimated (T, ne), the observed spectrum is decomposed as a non-negative
linear combination of single-element basis spectra via NNLS.

Architecture: Preprocess → Estimate (T, ne) → Decompose (NNLS) → Validate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

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
    ):
        self.basis_library = basis_library
        self.basis_index = basis_index
        self.detection_snr = detection_snr
        self.continuum_degree = continuum_degree
        self.fallback_T_K = fallback_T_K
        self.fallback_ne_cm3 = fallback_ne_cm3

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

        # Step 6: Solve NNLS
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

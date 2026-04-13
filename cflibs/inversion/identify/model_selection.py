"""
BIC-based model selection for spectral NNLS element identification.

Implements backward elimination of elements using the Bayesian Information
Criterion (BIC) and Boltzmann consistency filtering.  Starting from all
NNLS-detected elements, iteratively removes spurious elements whose removal
improves (decreases) BIC, then validates survivors via Boltzmann linearity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.stats import linregress

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.model_selection")


@dataclass
class ModelSelectionResult:
    """Result of BIC-based element selection."""

    selected_elements: List[str]
    removed_elements: List[str]
    concentrations: Dict[str, float]
    bic_final: float
    bic_initial: float
    boltzmann_results: Dict[str, dict] = field(default_factory=dict)


def _compute_bic(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Compute the Bayesian Information Criterion.

    BIC = n * ln(RSS / n) + k * ln(n)

    Parameters
    ----------
    observed : np.ndarray
        Observed spectrum (n_pixels,).
    predicted : np.ndarray
        Model-predicted spectrum (n_pixels,).
    k : int
        Number of free parameters.

    Returns
    -------
    float
        BIC value (lower is better).
    """
    n = len(observed)
    residuals = observed - predicted
    rss = float(np.sum(residuals**2))
    if rss <= 0.0:
        rss = np.finfo(np.float64).tiny
    return n * np.log(rss / n) + k * np.log(n)


def _solve_nnls_subset(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    element_mask: np.ndarray,
    n_elements: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve NNLS for a subset of elements (plus all continuum columns).

    Parameters
    ----------
    observed : np.ndarray
        Observed spectrum (n_pixels,).
    basis_matrix : np.ndarray
        Full basis matrix (n_components, n_pixels).  First *n_elements* rows
        are element basis spectra; remaining rows are continuum columns.
    element_mask : np.ndarray
        Boolean mask of length *n_elements* selecting active elements.
    n_elements : int
        Number of element rows at the start of *basis_matrix*.

    Returns
    -------
    coeffs : np.ndarray
        NNLS coefficients for the active rows (n_active,).
    predicted : np.ndarray
        Reconstructed spectrum (n_pixels,).
    """
    n_continuum = basis_matrix.shape[0] - n_elements
    continuum_mask = np.ones(n_continuum, dtype=bool)
    full_mask = np.concatenate([element_mask, continuum_mask])
    active_basis = basis_matrix[full_mask, :]

    if active_basis.shape[0] == 0:
        return np.array([]), np.zeros_like(observed)

    coeffs, _ = nnls(active_basis.T, observed)
    predicted = active_basis.T @ coeffs
    return coeffs, predicted


def bic_prune_elements(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    element_list: List[str],
    element_coefficients: np.ndarray,
    noise_variance: float,
) -> ModelSelectionResult:
    """Backward elimination of elements using BIC.

    BIC = n * ln(RSS / n) + k * ln(n)

    where n = number of spectral pixels, k = number of free parameters
    (number of elements + continuum terms), RSS = residual sum of squares.

    Starting from all detected elements, iteratively removes the element
    with the smallest coefficient.  If BIC decreases (model improves with
    fewer parameters), the element was spurious -- keep it removed.
    If BIC increases, restore it and stop.

    Parameters
    ----------
    observed : np.ndarray
        Observed spectrum (n_pixels,).
    basis_matrix : np.ndarray
        Full basis matrix including continuum columns (n_components, n_pixels).
        First ``len(element_list)`` rows are element basis spectra.
    element_list : List[str]
        Element symbols corresponding to first rows of *basis_matrix*.
    element_coefficients : np.ndarray
        NNLS coefficients for all components (n_components,).
    noise_variance : float
        Estimated noise variance of the observed spectrum.

    Returns
    -------
    ModelSelectionResult
    """
    n_elements = len(element_list)
    n_continuum = basis_matrix.shape[0] - n_elements

    # Step 1: identify elements with nonzero coefficients
    el_coeffs = element_coefficients[:n_elements]
    active_mask = el_coeffs > 0.0

    # If no elements are active, return empty result
    if not np.any(active_mask):
        return ModelSelectionResult(
            selected_elements=[],
            removed_elements=list(element_list),
            concentrations={},
            bic_final=_compute_bic(observed, np.zeros_like(observed), n_continuum),
            bic_initial=_compute_bic(observed, np.zeros_like(observed), n_continuum),
        )

    # Step 2: compute initial BIC with all active elements
    k_initial = int(np.sum(active_mask)) + n_continuum
    _, predicted_initial = _solve_nnls_subset(observed, basis_matrix, active_mask, n_elements)
    bic_current = _compute_bic(observed, predicted_initial, k_initial)
    bic_initial = bic_current

    logger.debug(
        "BIC pruning: starting with %d elements, BIC=%.2f",
        int(np.sum(active_mask)),
        bic_current,
    )

    # Step 3: sort active elements by coefficient (ascending -- smallest first)
    active_indices = np.where(active_mask)[0]
    sorted_indices = active_indices[np.argsort(el_coeffs[active_indices])]

    removed = []

    # Step 4: backward elimination
    for idx in sorted_indices:
        # Skip if this is the last remaining element
        if int(np.sum(active_mask)) <= 1:
            break

        # Tentatively remove element
        trial_mask = active_mask.copy()
        trial_mask[idx] = False

        k_trial = int(np.sum(trial_mask)) + n_continuum
        _, predicted_trial = _solve_nnls_subset(observed, basis_matrix, trial_mask, n_elements)
        bic_trial = _compute_bic(observed, predicted_trial, k_trial)

        if bic_trial < bic_current:
            # Removal improved the model -- element was spurious
            active_mask[idx] = False
            bic_current = bic_trial
            removed.append(element_list[idx])
            logger.debug(
                "Removed %s (coeff=%.4e), BIC %.2f -> %.2f",
                element_list[idx],
                el_coeffs[idx],
                bic_current,
                bic_trial,
            )
        else:
            # Removal worsened the model -- stop elimination
            logger.debug(
                "Keeping %s (coeff=%.4e), BIC would increase %.2f -> %.2f; stopping",
                element_list[idx],
                el_coeffs[idx],
                bic_current,
                bic_trial,
            )
            break

    # Step 5: final NNLS solve with surviving elements
    final_coeffs, _ = _solve_nnls_subset(observed, basis_matrix, active_mask, n_elements)

    # Build concentrations dict (element coefficients only, normalized)
    n_active_elements = int(np.sum(active_mask))
    el_final_coeffs = final_coeffs[:n_active_elements]
    total = float(np.sum(el_final_coeffs))
    active_elements = [element_list[i] for i in range(n_elements) if active_mask[i]]

    if total > 0.0:
        concentrations = {el: float(c / total) for el, c in zip(active_elements, el_final_coeffs)}
    else:
        concentrations = {el: 0.0 for el in active_elements}

    # Elements that were inactive from the start (zero coefficient)
    initially_inactive = [element_list[i] for i in range(n_elements) if el_coeffs[i] <= 0.0]

    return ModelSelectionResult(
        selected_elements=active_elements,
        removed_elements=removed + initially_inactive,
        concentrations=concentrations,
        bic_final=bic_current,
        bic_initial=bic_initial,
    )


def boltzmann_consistency_filter(
    element: str,
    wavelength: np.ndarray,
    observed: np.ndarray,
    basis_spectrum: np.ndarray,
    transitions: list,
    T_estimated_K: float,
) -> dict:
    """Check if an element's matched lines follow Boltzmann statistics.

    For each transition with significant contribution in the basis spectrum,
    compute ln(I * lambda / (g * A)) vs E_k.  Fit a line -- slope should give
    a physical temperature (3000--50000 K) with R^2 > 0.5.

    Parameters
    ----------
    element : str
        Element symbol.
    wavelength : np.ndarray
        Wavelength grid (n_pixels,).
    observed : np.ndarray
        Observed spectrum (n_pixels,).
    basis_spectrum : np.ndarray
        Element's basis spectrum (n_pixels,).
    transitions : list
        List of transition objects with attributes: ``wavelength_nm``,
        ``A_ki``, ``g_k``, ``E_k_ev``.
    T_estimated_K : float
        Estimated plasma temperature in K.

    Returns
    -------
    dict
        Keys: ``T_K`` (fitted temperature), ``R_squared``, ``n_lines``,
        ``passes`` (bool).
    """
    if not transitions:
        return {"T_K": 0.0, "R_squared": 0.0, "n_lines": 0, "passes": False}

    # Find peaks in the basis spectrum above a threshold
    basis_max = np.max(basis_spectrum)
    if basis_max <= 0.0:
        return {"T_K": 0.0, "R_squared": 0.0, "n_lines": 0, "passes": False}

    threshold = 0.01 * basis_max

    E_k_values = []
    y_values = []

    for trans in transitions:
        wl_nm = trans.wavelength_nm
        A_ki = trans.A_ki
        g_k = trans.g_k
        E_k = trans.E_k_ev

        if A_ki <= 0.0 or g_k <= 0:
            continue

        # Find the nearest pixel to this transition's wavelength
        idx = int(np.argmin(np.abs(wavelength - wl_nm)))

        # Check if the basis spectrum has a significant contribution here
        if basis_spectrum[idx] < threshold:
            continue

        # Get observed intensity at this wavelength
        intensity = observed[idx]
        if intensity <= 0.0:
            continue

        # Boltzmann plot ordinate: ln(I * lambda / (g_k * A_ki))
        y = np.log(intensity * wl_nm / (g_k * A_ki))
        E_k_values.append(E_k)
        y_values.append(y)

    n_lines = len(E_k_values)
    if n_lines < 3:
        return {"T_K": 0.0, "R_squared": 0.0, "n_lines": n_lines, "passes": False}

    E_k_arr = np.array(E_k_values)
    y_arr = np.array(y_values)

    result = linregress(E_k_arr, y_arr)
    slope = result.slope
    r_squared = result.rvalue**2

    # Convert slope to temperature: slope = -1 / (kB * T)
    # T = -1 / (kB * slope)
    if slope >= 0.0:
        # Non-physical (positive slope means inverted population)
        return {
            "T_K": 0.0,
            "R_squared": r_squared,
            "n_lines": n_lines,
            "passes": False,
        }

    T_K = -1.0 / (KB_EV * slope)

    # Validation criteria
    passes = bool(r_squared > 0.5 and 3000.0 <= T_K <= 50000.0)

    return {
        "T_K": float(T_K),
        "R_squared": float(r_squared),
        "n_lines": n_lines,
        "passes": passes,
    }

"""
Information-criterion model selection for spectral NNLS element identification.

Implements backward elimination of elements using a selectable information
criterion and Boltzmann consistency filtering.  Starting from all
NNLS-detected elements, iteratively removes spurious elements whose removal
improves (decreases) the criterion, then validates survivors via Boltzmann
linearity.

Four criteria share a single backward-elimination entry point
(:func:`prune_elements`), each differing only in the parameter-count penalty
added to the Gaussian log-likelihood term ``n * ln(RSS / n)``:

* ``BIC``  -- Bayesian Information Criterion, penalty ``k * ln(n)``
  (Schwarz 1978; Bozdogan 1987).
* ``AIC``  -- Akaike Information Criterion, penalty ``2 * k``
  (Akaike 1974).
* ``AICC`` -- AIC with the small-sample correction
  ``2 * k + 2k(k + 1) / (n - k - 1)`` (Hurvich & Tsai, Biometrika 1989).
* ``SPIC`` -- Spectral Information Criterion, a line-strength-weighted hybrid
  of the AICc and BIC penalties (Webb, Lee, Carswell & Milakovic, MNRAS 2021,
  "Getting the model right: an information criterion for spectroscopy",
  arXiv:2009.08336).

All four preserve the BIC entry point's interface: :func:`bic_prune_elements`
remains a thin ``criterion=Criterion.BIC`` wrapper so existing callers are
unaffected (additive, no point-estimate change to the default path).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.stats import linregress

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.inversion.identify.spectral_nnls import (
    _HAS_JAX,
    nnls_jax,
    nnls_jax_batch,
)

logger = get_logger("inversion.model_selection")


class Criterion(Enum):
    """Selectable information criterion for backward elimination.

    Members
    -------
    BIC
        Bayesian Information Criterion (default; penalty ``k * ln(n)``).
    AIC
        Akaike Information Criterion (penalty ``2 * k``).
    AICC
        AIC with the Hurvich & Tsai (1989) small-sample correction
        (penalty ``2 * k + 2k(k + 1) / (n - k - 1)``).
    SPIC
        Spectral Information Criterion (Webb et al. 2021); a line-strength-
        weighted hybrid of the AICc and BIC penalties.
    """

    BIC = "bic"
    AIC = "aic"
    AICC = "aicc"
    SPIC = "spic"


@dataclass
class ModelSelectionResult:
    """Result of information-criterion element selection.

    Notes
    -----
    The ``bic_final`` / ``bic_initial`` fields are retained for backward
    compatibility and always hold the value of whichever criterion drove the
    selection (the default criterion is :attr:`Criterion.BIC`).  The
    ``criterion`` field records which criterion was used so callers selecting
    AIC / AICc / SpIC can interpret the score correctly.
    """

    selected_elements: List[str]
    removed_elements: List[str]
    concentrations: Dict[str, float]
    bic_final: float
    bic_initial: float
    boltzmann_results: Dict[str, dict] = field(default_factory=dict)
    criterion: Criterion = Criterion.BIC


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
    return _gaussian_log_likelihood_term(observed, predicted) + k * np.log(n)


def _gaussian_log_likelihood_term(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the residual-sum-of-squares log-likelihood term ``n * ln(RSS / n)``.

    This is the deviance-style data-fit term shared by every information
    criterion in this module (the ``chi^2``-analogue under the assumption of
    homoscedastic Gaussian noise).  Returning it from one helper guarantees
    that the BIC, AIC, AICc, and SpIC penalties are added to an identical fit
    term, so criterion *differences* reflect only the penalty.

    Parameters
    ----------
    observed : np.ndarray
        Observed spectrum (n_pixels,).
    predicted : np.ndarray
        Model-predicted spectrum (n_pixels,).

    Returns
    -------
    float
        ``n * ln(RSS / n)`` with ``RSS`` floored at the smallest positive
        float to avoid ``log(0)``.
    """
    n = len(observed)
    residuals = observed - predicted
    rss = float(np.sum(residuals**2))
    if rss <= 0.0:
        rss = np.finfo(np.float64).tiny
    return n * np.log(rss / n)


def _compute_aic(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Compute the Akaike Information Criterion.

    AIC = n * ln(RSS / n) + 2 * k

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
        AIC value (lower is better).

    References
    ----------
    Akaike, H. (1974). "A new look at the statistical model identification."
    IEEE Transactions on Automatic Control, 19(6), 716-723.
    """
    return _gaussian_log_likelihood_term(observed, predicted) + 2.0 * k


def _compute_aicc(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Compute the corrected Akaike Information Criterion (small-sample AICc).

    AICc = AIC + 2 * k * (k + 1) / (n - k - 1)
         = n * ln(RSS / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

    The correction term is strictly positive whenever ``n > k + 1`` and grows
    as ``n`` shrinks toward ``k + 1``, so AICc penalizes more heavily than AIC
    at small sample sizes and converges to AIC as ``n -> infinity``.  When
    ``n <= k + 1`` the correction is undefined; ``+inf`` is returned so the
    model is rejected (it has at least as many parameters as data points).

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
        AICc value (lower is better), or ``+inf`` when ``n <= k + 1``.

    References
    ----------
    Hurvich, C. M., & Tsai, C.-L. (1989). "Regression and time series model
    selection in small samples." Biometrika, 76(2), 297-307.
    """
    n = len(observed)
    aic = _compute_aic(observed, predicted, k)
    denom = n - k - 1
    if denom <= 0:
        return float("inf")
    return aic + 2.0 * k * (k + 1) / denom


def _compute_spic(
    observed: np.ndarray,
    predicted: np.ndarray,
    line_strengths: np.ndarray,
    k_per_component: int = 1,
    f: float = 0.5,
) -> float:
    """Compute the Spectral Information Criterion (SpIC).

    Webb et al. (2021), Eq. (3):

    .. math::

        \\mathrm{SpIC} = \\chi^2 + \\sum_{a=1}^{Q}
        \\left[ \\frac{2 f k_a R_a}{R_a - k_a - 1}
                + (1 - f) k_a \\ln(R_a) \\right]

    The :math:`\\chi^2` data-fit term is replaced here by the shared Gaussian
    log-likelihood term ``n * ln(RSS / n)`` for parity with the other criteria
    in this module.  Crucially, SpIC weights each component's penalty by a
    *line-strength* measure :math:`R_a` (Webb et al. Eq. 5) rather than the
    global pixel count :math:`N`.  This is the defining advantage over AICc and
    BIC, which "treat all model parameters as being of equal importance" --
    inappropriate when weak, near-threshold components sit alongside strong
    ones (Webb et al. 2021).

    The weighting parameter :math:`f \\in [0, 1]` interpolates between the AICc
    analogue (:math:`f = 1`, first bracket term only) and the BIC analogue
    (:math:`f = 0`, second bracket term only); :math:`f = 1/2` is the
    recommended hybrid.  At equal fit the SpIC penalty per component uses the
    component's own :math:`R_a` in place of :math:`N`; because a detected
    spectral component's effective :math:`R_a` is far smaller than the full
    pixel count, SpIC's penalty -- and hence its effective parameter cost --
    is smaller than AICc's at equal fit, matching the cited result that SpIC
    "requires fewer absorption components to achieve a similar goodness of
    fit."

    A lower bound :math:`R_a \\ge k_a + 2` is imposed (Webb et al. Section 2.4)
    so the first bracket term cannot diverge at :math:`R_a = k_a + 1`.

    Parameters
    ----------
    observed : np.ndarray
        Observed spectrum (n_pixels,).
    predicted : np.ndarray
        Model-predicted spectrum (n_pixels,).
    line_strengths : np.ndarray
        Per-component line-strength measures :math:`R_a` (Webb et al. Eq. 5),
        one per active component, length ``Q``.
    k_per_component : int, optional
        Number of free parameters per component, :math:`k_a`.  For
        element-presence NNLS each element contributes a single nonnegative
        amplitude, so the default is 1.
    f : float, optional
        Hybrid weighting parameter :math:`f \\in [0, 1]`.  ``f = 1`` -> AICc
        analogue, ``f = 0`` -> BIC analogue, ``f = 0.5`` (default) -> hybrid.

    Returns
    -------
    float
        SpIC value (lower is better).

    References
    ----------
    Webb, J. K., Lee, C.-C., Carswell, R. F., & Milakovic, D. (2021).
    "Getting the model right: an information criterion for spectroscopy."
    MNRAS, 501(2), 2268-2278. arXiv:2009.08336.
    """
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"SpIC weighting f must lie in [0, 1], got {f}")
    chi2_term = _gaussian_log_likelihood_term(observed, predicted)
    k_a = float(k_per_component)
    # Lower bound R_a >= k_a + 2 (Webb et al. Section 2.4) to keep the first
    # bracket term finite at R_a = k_a + 1.
    r_a = np.maximum(np.asarray(line_strengths, dtype=np.float64), k_a + 2.0)
    aicc_like = 2.0 * f * k_a * r_a / (r_a - k_a - 1.0)
    bic_like = (1.0 - f) * k_a * np.log(r_a)
    penalty = float(np.sum(aicc_like + bic_like))
    return chi2_term + penalty


def _component_line_strengths(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    coeffs: np.ndarray,
    active_indices: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    """Compute per-component line-strength measures :math:`R_a` (Webb Eq. 5).

    Webb et al. (2021) define :math:`R_a = \\sum_j r_{a,j}` where each
    :math:`r_{a,j}` parameterises "both the effective number of pixels and
    line strength information" for one component, and in the idealised unit-
    error case reduces to the effective number of pixels a line influences.
    The natural NNLS-element analogue is the noise-normalised summed model
    contribution of that element's basis spectrum:

    .. math::

        R_a = \\sum_i \\frac{c_a \\, B_{a,i}}{\\sigma_i}

    where :math:`c_a` is the element's NNLS amplitude, :math:`B_{a,i}` its
    basis spectrum, and :math:`\\sigma_i = \\sqrt{\\mathrm{noise\\_variance}}`
    the (homoscedastic) per-pixel error.  This satisfies Webb's three
    requirements: it increases with line strength, scales with the error
    array, and is computed per component so multiple species are handled
    simultaneously.

    Parameters
    ----------
    observed : np.ndarray
        Observed spectrum (n_pixels,) -- accepted for interface symmetry; the
        strength measure is taken from the model contribution, not the data.
    basis_matrix : np.ndarray
        Full basis matrix (n_components, n_pixels).
    coeffs : np.ndarray
        NNLS coefficients for all components (n_components,).
    active_indices : np.ndarray
        Indices (into the element rows) of the active components, length ``Q``.
    noise_variance : float
        Estimated per-pixel noise variance.

    Returns
    -------
    np.ndarray
        Per-component strength measures :math:`R_a`, length ``Q``.

    References
    ----------
    Webb, J. K., et al. (2021), MNRAS 501, 2268, Eqs. (5)-(6).
    """
    sigma = float(np.sqrt(noise_variance)) if noise_variance > 0.0 else 1.0
    strengths = np.empty(len(active_indices), dtype=np.float64)
    for a, idx in enumerate(active_indices):
        contribution = coeffs[idx] * basis_matrix[idx, :]
        strengths[a] = float(np.sum(np.abs(contribution)) / sigma)
    return strengths


def _full_coeffs_from_active(
    active_coeffs: np.ndarray,
    active_mask: np.ndarray,
    n_elements: int,
    n_continuum: int,
) -> np.ndarray:
    """Scatter active-row NNLS coefficients back into a full-component vector.

    :func:`_solve_nnls_subset` returns coefficients only for the rows it
    actually solved (active elements followed by all continuum columns).  SpIC
    needs each *element* coefficient indexed by its original basis row, so this
    rebuilds the dense ``(n_elements + n_continuum,)`` coefficient vector with
    zeros in the removed-element slots.

    Parameters
    ----------
    active_coeffs : np.ndarray
        Coefficients for active rows, ordered [active elements, all continuum].
    active_mask : np.ndarray
        Boolean mask of length *n_elements* selecting active elements.
    n_elements : int
        Number of element rows.
    n_continuum : int
        Number of continuum rows.

    Returns
    -------
    np.ndarray
        Dense coefficient vector of length ``n_elements + n_continuum``.
    """
    full = np.zeros(n_elements + n_continuum, dtype=np.float64)
    active_element_rows = np.where(active_mask)[0]
    n_active = len(active_element_rows)
    if n_active > 0:
        full[active_element_rows] = active_coeffs[:n_active]
    if n_continuum > 0:
        full[n_elements:] = active_coeffs[n_active:]
    return full


def _compute_criterion(
    observed: np.ndarray,
    predicted: np.ndarray,
    k: int,
    criterion: Criterion,
    *,
    basis_matrix: Optional[np.ndarray] = None,
    full_coeffs: Optional[np.ndarray] = None,
    active_element_indices: Optional[np.ndarray] = None,
    noise_variance: float = 1.0,
    spic_f: float = 0.5,
) -> float:
    """Dispatch to the requested information criterion (lower is better).

    Every criterion shares the Gaussian log-likelihood fit term
    ``n * ln(RSS / n)`` and differs only in the parameter-count penalty, so
    ``_compute_criterion(..., Criterion.BIC)`` is numerically identical to
    :func:`_compute_bic`.  This single dispatcher is what gives AIC, AICc, and
    SpIC the *same* backward-elimination interface as BIC.

    Parameters
    ----------
    observed, predicted : np.ndarray
        Observed and model-predicted spectra (n_pixels,).
    k : int
        Total number of free parameters (active elements + continuum terms).
        Used by BIC / AIC / AICc.
    criterion : Criterion
        Which criterion to evaluate.
    basis_matrix, full_coeffs, active_element_indices : optional
        Required only for :attr:`Criterion.SPIC`; used to compute the per-
        component line-strength measures :math:`R_a` (Webb et al. Eq. 5).
    noise_variance : float, optional
        Per-pixel noise variance (SpIC strength normalisation).
    spic_f : float, optional
        SpIC hybrid weighting :math:`f \\in [0, 1]` (default 0.5, the
        recommended AICc/BIC hybrid).

    Returns
    -------
    float
        Criterion value.
    """
    if criterion is Criterion.BIC:
        return _compute_bic(observed, predicted, k)
    if criterion is Criterion.AIC:
        return _compute_aic(observed, predicted, k)
    if criterion is Criterion.AICC:
        return _compute_aicc(observed, predicted, k)
    # SpIC
    if basis_matrix is None or full_coeffs is None or active_element_indices is None:
        raise ValueError("Criterion.SPIC requires basis_matrix, full_coeffs, and indices")
    if len(active_element_indices) == 0:
        # No active components: SpIC penalty sum is empty, reduce to fit term.
        return _gaussian_log_likelihood_term(observed, predicted)
    strengths = _component_line_strengths(
        observed,
        basis_matrix,
        full_coeffs,
        active_element_indices,
        noise_variance,
    )
    return _compute_spic(observed, predicted, strengths, k_per_component=1, f=spic_f)


def _solve_nnls_subset(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    element_mask: np.ndarray,
    n_elements: int,
    use_jax_nnls: bool = False,
    jax_nnls_max_iter: int = 300,
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
    use_jax_nnls : bool, optional
        If True, solve via :func:`nnls_jax` (FISTA on the Gram form)
        instead of ``scipy.optimize.nnls`` (Lawson--Hanson). The two
        agree on residual norm to rtol ~1e-5 but can pick different
        minimizers on rank-deficient problems. Default False.
    jax_nnls_max_iter : int, optional
        FISTA iteration count when ``use_jax_nnls=True``. Default 300.

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

    if use_jax_nnls:
        coeffs, _ = nnls_jax(active_basis.T, observed, max_iter=jax_nnls_max_iter)
    else:
        coeffs, _ = nnls(active_basis.T, observed)
    predicted = active_basis.T @ coeffs
    return coeffs, predicted


def _empty_selection_result(
    observed: np.ndarray,
    element_list: List[str],
    n_continuum: int,
) -> ModelSelectionResult:
    """Build the result returned when no elements have nonzero coefficients."""
    return ModelSelectionResult(
        selected_elements=[],
        removed_elements=list(element_list),
        concentrations={},
        bic_final=_compute_bic(observed, np.zeros_like(observed), n_continuum),
        bic_initial=_compute_bic(observed, np.zeros_like(observed), n_continuum),
    )


def _compute_prebatch_predictions(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    active_mask: np.ndarray,
    sorted_indices: np.ndarray,
    n_continuum: int,
    jax_nnls_max_iter: int,
) -> Dict[int, np.ndarray]:
    """Pre-compute every cumulative leave-one-out trial in one vmapped call.

    Builds the sequence of full-component masks corresponding to
    "cumulatively remove sorted_indices[:k]" for k=1..len. Pads row-mask
    space to the full n_components (= n_elements + n_continuum) so we can
    call :func:`nnls_jax_batch` on *basis_matrix* directly.
    """
    cum_masks = np.tile(active_mask, (len(sorted_indices), 1)).astype(np.float64)
    for k, idx in enumerate(sorted_indices):
        cum_masks[k:, idx] = 0.0  # remove this element from all subsequent trials
    full_masks = np.concatenate(
        [cum_masks, np.ones((len(sorted_indices), n_continuum), dtype=np.float64)],
        axis=1,
    )
    _coeffs_batch, _ = nnls_jax_batch(
        basis_matrix,
        observed,
        full_masks,
        max_iter=jax_nnls_max_iter,
    )
    # predicted_b = sum_j coeffs[b, j] * basis_matrix[j, :]
    return {
        int(sorted_indices[k]): _coeffs_batch[k] @ basis_matrix for k in range(len(sorted_indices))
    }


def _trial_prediction(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    trial_mask: np.ndarray,
    idx: int,
    n_elements: int,
    prebatch_predictions: Optional[Dict[int, np.ndarray]],
    prebatch_valid: bool,
    use_jax_nnls: bool,
    jax_nnls_max_iter: int,
) -> np.ndarray:
    """Reconstruct the predicted spectrum for a single leave-one-out trial."""
    if prebatch_predictions is not None and prebatch_valid:
        return prebatch_predictions[int(idx)]
    _, predicted_trial = _solve_nnls_subset(
        observed,
        basis_matrix,
        trial_mask,
        n_elements,
        use_jax_nnls=use_jax_nnls,
        jax_nnls_max_iter=jax_nnls_max_iter,
    )
    return predicted_trial


def _build_concentrations(
    final_coeffs: np.ndarray,
    active_elements: List[str],
    n_active_elements: int,
) -> Dict[str, float]:
    """Normalize element coefficients into a concentration dict."""
    el_final_coeffs = final_coeffs[:n_active_elements]
    total = float(np.sum(el_final_coeffs))
    if total > 0.0:
        return {el: float(c / total) for el, c in zip(active_elements, el_final_coeffs)}
    return {el: 0.0 for el in active_elements}


def _criterion_for_mask(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    active_mask: np.ndarray,
    n_elements: int,
    n_continuum: int,
    criterion: Criterion,
    noise_variance: float,
    spic_f: float,
    *,
    predicted: Optional[np.ndarray] = None,
    use_jax_nnls: bool = False,
    jax_nnls_max_iter: int = 300,
) -> float:
    """Score one active-element configuration under *criterion*.

    Solves the subset NNLS once (unless a cached *predicted* is supplied for
    the BIC/AIC/AICc fast paths), then dispatches via :func:`_compute_criterion`.
    SpIC always requires the coefficients, so any cached prediction is ignored
    for that criterion and a fresh solve is performed.
    """
    k = int(np.sum(active_mask)) + n_continuum
    need_coeffs = criterion is Criterion.SPIC
    if predicted is None or need_coeffs:
        active_coeffs, predicted = _solve_nnls_subset(
            observed,
            basis_matrix,
            active_mask,
            n_elements,
            use_jax_nnls=use_jax_nnls,
            jax_nnls_max_iter=jax_nnls_max_iter,
        )
        full_coeffs = _full_coeffs_from_active(active_coeffs, active_mask, n_elements, n_continuum)
    else:
        full_coeffs = None
    return _compute_criterion(
        observed,
        predicted,
        k,
        criterion,
        basis_matrix=basis_matrix,
        full_coeffs=full_coeffs,
        active_element_indices=np.where(active_mask)[0],
        noise_variance=noise_variance,
        spic_f=spic_f,
    )


def prune_elements(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    element_list: List[str],
    element_coefficients: np.ndarray,
    noise_variance: float,
    criterion: Criterion = Criterion.BIC,
    spic_f: float = 0.5,
    use_jax_nnls: bool = False,
    jax_nnls_max_iter: int = 300,
    jax_batch_trials: bool = False,
) -> ModelSelectionResult:
    """Backward elimination of elements using a selectable information criterion.

    All four criteria add a parameter-count penalty to the shared Gaussian
    log-likelihood term ``n * ln(RSS / n)`` (``n`` = number of spectral pixels,
    RSS = residual sum of squares):

    * ``BIC``  : ``+ k * ln(n)``                       (Schwarz 1978).
    * ``AIC``  : ``+ 2 * k``                           (Akaike 1974).
    * ``AICC`` : ``+ 2 * k + 2k(k + 1) / (n - k - 1)`` (Hurvich & Tsai 1989).
    * ``SPIC`` : line-strength-weighted AICc/BIC hybrid (Webb et al. 2021).

    where ``k`` is the number of free parameters (active elements + continuum
    terms).  Starting from all detected elements, the routine iteratively
    removes the element with the smallest coefficient; if the criterion
    decreases (model improves with fewer parameters) the element was spurious
    and stays removed, otherwise it is restored and elimination stops.

    Selecting a different criterion changes *only* the diagnostic / pruning
    behaviour -- it is an additive model-selection option, not a change to the
    underlying NNLS point estimate.

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
        Estimated noise variance of the observed spectrum (used by the SpIC
        line-strength normalisation; ignored by BIC / AIC / AICc).
    criterion : Criterion, optional
        Information criterion driving the pruning.  Default
        :attr:`Criterion.BIC` (identical to the legacy behaviour).
    spic_f : float, optional
        SpIC hybrid weighting :math:`f \\in [0, 1]` (``1`` = AICc analogue,
        ``0`` = BIC analogue, ``0.5`` = recommended hybrid).  Ignored unless
        ``criterion is Criterion.SPIC``.  Default 0.5.
    use_jax_nnls : bool, optional
        Route inner NNLS solves through :func:`nnls_jax` (FISTA, GPU-
        batchable) instead of ``scipy.optimize.nnls`` (Lawson--Hanson,
        CPU). Residual norms agree to ~1e-5 rtol; coefficient agreement
        is rtol ~1e-4 in the well-conditioned case. Default False.
    jax_nnls_max_iter : int, optional
        FISTA iteration count. Default 300.
    jax_batch_trials : bool, optional
        Only meaningful when ``use_jax_nnls=True`` and
        ``criterion is Criterion.BIC``. If True, compute every leave-one-out
        *trial* in a single :func:`nnls_jax_batch` call (vmapped), then walk
        through them sequentially according to the acceptance/rejection logic.
        This is exact when each accepted removal matches the upfront
        sequential plan; otherwise it falls back to per-trial solves. The
        other criteria need the per-trial coefficients (SpIC) or are cheap
        enough that batching is bypassed. Default False.

    Returns
    -------
    ModelSelectionResult
        ``bic_final`` / ``bic_initial`` carry the selected criterion's score;
        ``criterion`` records which criterion was used.

    References
    ----------
    Akaike, H. (1974). IEEE Trans. Automat. Contr. 19, 716.
    Hurvich, C. M., & Tsai, C.-L. (1989). Biometrika 76, 297.
    Webb, J. K., et al. (2021). MNRAS 501, 2268. arXiv:2009.08336.
    """
    if use_jax_nnls and not _HAS_JAX:  # pragma: no cover
        raise ImportError("use_jax_nnls=True requires JAX. " "Install with: pip install jax jaxlib")
    n_elements = len(element_list)
    n_continuum = basis_matrix.shape[0] - n_elements

    # Step 1: identify elements with nonzero coefficients
    el_coeffs = element_coefficients[:n_elements]
    active_mask = el_coeffs > 0.0

    # If no elements are active, return empty result
    if not np.any(active_mask):
        result = _empty_selection_result(observed, element_list, n_continuum)
        result.criterion = criterion
        return result

    # Step 2: compute initial criterion with all active elements
    _, predicted_initial = _solve_nnls_subset(
        observed,
        basis_matrix,
        active_mask,
        n_elements,
        use_jax_nnls=use_jax_nnls,
        jax_nnls_max_iter=jax_nnls_max_iter,
    )
    score_current = _criterion_for_mask(
        observed,
        basis_matrix,
        active_mask,
        n_elements,
        n_continuum,
        criterion,
        noise_variance,
        spic_f,
        predicted=predicted_initial,
        use_jax_nnls=use_jax_nnls,
        jax_nnls_max_iter=jax_nnls_max_iter,
    )
    score_initial = score_current

    logger.debug(
        "%s pruning: starting with %d elements, score=%.2f",
        criterion.name,
        int(np.sum(active_mask)),
        score_current,
    )

    # Step 3: sort active elements by coefficient (ascending -- smallest first)
    active_indices = np.where(active_mask)[0]
    sorted_indices = active_indices[np.argsort(el_coeffs[active_indices])]

    # Step 3b (optional): pre-compute every leave-one-out trial in one
    # vmapped call. Only the BIC fast path uses cached *predictions* (SpIC
    # needs per-trial coefficients, so it always re-solves). We assume the
    # "all removals succeed" plan and validate sequentially below.
    prebatch_predictions: Optional[Dict[int, np.ndarray]] = None
    if use_jax_nnls and jax_batch_trials and criterion is Criterion.BIC and len(sorted_indices) > 0:
        prebatch_predictions = _compute_prebatch_predictions(
            observed,
            basis_matrix,
            active_mask,
            sorted_indices,
            n_continuum,
            jax_nnls_max_iter,
        )

    removed = []
    prebatch_valid = True  # becomes False after first rejected removal

    # Step 4: backward elimination
    for idx in sorted_indices:
        # Skip if this is the last remaining element
        if int(np.sum(active_mask)) <= 1:
            break

        # Tentatively remove element
        trial_mask = active_mask.copy()
        trial_mask[idx] = False

        if prebatch_predictions is not None and prebatch_valid:
            predicted_trial = _trial_prediction(
                observed,
                basis_matrix,
                trial_mask,
                idx,
                n_elements,
                prebatch_predictions,
                prebatch_valid,
                use_jax_nnls,
                jax_nnls_max_iter,
            )
            score_trial = _criterion_for_mask(
                observed,
                basis_matrix,
                trial_mask,
                n_elements,
                n_continuum,
                criterion,
                noise_variance,
                spic_f,
                predicted=predicted_trial,
                use_jax_nnls=use_jax_nnls,
                jax_nnls_max_iter=jax_nnls_max_iter,
            )
        else:
            score_trial = _criterion_for_mask(
                observed,
                basis_matrix,
                trial_mask,
                n_elements,
                n_continuum,
                criterion,
                noise_variance,
                spic_f,
                use_jax_nnls=use_jax_nnls,
                jax_nnls_max_iter=jax_nnls_max_iter,
            )

        if score_trial < score_current:
            # Removal improved the model -- element was spurious
            active_mask[idx] = False
            removed.append(element_list[idx])
            logger.debug(
                "Removed %s (coeff=%.4e), %s %.2f -> %.2f",
                element_list[idx],
                el_coeffs[idx],
                criterion.name,
                score_current,
                score_trial,
            )
            score_current = score_trial
        else:
            # Removal worsened the model -- stop elimination
            logger.debug(
                "Keeping %s (coeff=%.4e), %s would increase %.2f -> %.2f; stopping",
                element_list[idx],
                el_coeffs[idx],
                criterion.name,
                score_current,
                score_trial,
            )
            # Pre-batched trials past this point are stale (they assumed
            # this removal succeeded). Invalidate so the *next* loop
            # iteration --- if reached via a future revision --- falls
            # back to per-trial solves. We also break, so this is mostly
            # a safety net.
            prebatch_valid = False
            break

    # Step 5: final NNLS solve with surviving elements
    final_coeffs, _ = _solve_nnls_subset(
        observed,
        basis_matrix,
        active_mask,
        n_elements,
        use_jax_nnls=use_jax_nnls,
        jax_nnls_max_iter=jax_nnls_max_iter,
    )

    # Build concentrations dict (element coefficients only, normalized)
    n_active_elements = int(np.sum(active_mask))
    active_elements = [element_list[i] for i in range(n_elements) if active_mask[i]]
    concentrations = _build_concentrations(final_coeffs, active_elements, n_active_elements)

    # Elements that were inactive from the start (zero coefficient)
    initially_inactive = [element_list[i] for i in range(n_elements) if el_coeffs[i] <= 0.0]

    return ModelSelectionResult(
        selected_elements=active_elements,
        removed_elements=removed + initially_inactive,
        concentrations=concentrations,
        bic_final=score_current,
        bic_initial=score_initial,
        criterion=criterion,
    )


def bic_prune_elements(
    observed: np.ndarray,
    basis_matrix: np.ndarray,
    element_list: List[str],
    element_coefficients: np.ndarray,
    noise_variance: float,
    use_jax_nnls: bool = False,
    jax_nnls_max_iter: int = 300,
    jax_batch_trials: bool = False,
) -> ModelSelectionResult:
    """Backward elimination of elements using BIC (thin :func:`prune_elements` wrapper).

    Preserved as the canonical BIC entry point for backward compatibility.
    Equivalent to ``prune_elements(..., criterion=Criterion.BIC)``.

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
    use_jax_nnls : bool, optional
        Route inner NNLS solves through :func:`nnls_jax` (FISTA, GPU-
        batchable) instead of ``scipy.optimize.nnls`` (Lawson--Hanson,
        CPU). Residual norms agree to ~1e-5 rtol; coefficient agreement
        is rtol ~1e-4 in the well-conditioned case. Default False.
    jax_nnls_max_iter : int, optional
        FISTA iteration count. Default 300.
    jax_batch_trials : bool, optional
        Only meaningful when ``use_jax_nnls=True``. If True, compute
        every leave-one-out *trial* in a single :func:`nnls_jax_batch`
        call (vmapped), then walk through them sequentially according to
        the BIC acceptance/rejection logic. This is exact when each
        accepted removal happens to match the upfront sequential plan;
        otherwise it falls back to per-trial solves whenever the
        active mask diverges from the pre-batched assumption. Default
        False (per-trial path). Useful for small batch GPU runs where
        the launch overhead per individual NNLS solve dominates the
        FISTA cost.

    Returns
    -------
    ModelSelectionResult
    """
    return prune_elements(
        observed,
        basis_matrix,
        element_list,
        element_coefficients,
        noise_variance,
        criterion=Criterion.BIC,
        use_jax_nnls=use_jax_nnls,
        jax_nnls_max_iter=jax_nnls_max_iter,
        jax_batch_trials=jax_batch_trials,
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

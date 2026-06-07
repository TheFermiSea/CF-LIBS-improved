"""
Top-K candidate prefilter for Bayesian MCMC element inference.

Reduces the candidate element set from 50+ (periodic table) to K_max (default 15)
using SpectralNNLS decomposition before passing to the Bayesian forward model.
This is mandatory — full-element MCMC is intractable.

The prefilter uses NNLS coefficients (physical spectral contribution) as the
ranking signal, with SNR as a noise gate. To guard against temperature-sensitive
trace elements being missed at a single (T, ne) estimate, the prefilter
optionally evaluates at multiple temperatures and takes the union.

Design reviewed by Gemini 3.1 Pro (2026-04-13):
- SNR gate + coefficient ranking (not SNR-only ranking)
- Multi-T evaluation union for robustness
- Aggregate across ionization stages (MAX coefficient)
- K_min floor to prevent degenerate fits
- force_include for known matrix elements
"""

from __future__ import annotations

import copy as _copy_module
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.candidate_prefilter")

# Regex to extract base element symbol from species like "Fe I", "Fe II", etc.
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def _extract_element_symbol(species: str) -> str:
    """Extract base element symbol from a species string (e.g., 'Fe II' → 'Fe')."""
    m = _ELEMENT_RE.match(species.strip())
    return m.group(1) if m else species.strip()


def _collect_nnls_results(
    identifier,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    multi_t_offsets: List[float],
) -> list:
    """Run NNLS at the base (T, ne) plus optional T-offset evaluations.

    Returns the pooled list of identification results: the base result followed
    by one result per successful temperature offset. Offsets that raise are
    logged at debug level and skipped (best-effort multi-T robustness).

    Parameters
    ----------
    identifier : SpectralNNLSIdentifier
        Configured NNLS identifier with basis library loaded.
    wavelength : np.ndarray
        Observed spectrum wavelengths in nm.
    intensity : np.ndarray
        Observed spectrum intensities.
    multi_t_offsets : list of float
        Temperature offsets in Kelvin for multi-T evaluation. Empty disables it.

    Returns
    -------
    list
        Identification results (base first, then per-offset results).
    """
    # Step 1: Run NNLS at base (T, ne)
    base_result = identifier.identify(wavelength, intensity)
    all_results = [base_result]

    # Step 2: Multi-T robustness — run at T ± offsets
    if multi_t_offsets:
        # Use explicit is-not-None checks: a valid numeric estimate of exactly
        # 0.0 is falsy, so `or fallback` would silently discard it. Preserve
        # None -> fallback behavior; only stop treating 0.0 as missing.
        _est_T = getattr(identifier, "_estimated_T", None)
        base_T = _est_T if _est_T is not None else identifier.fallback_T_K
        _est_ne = getattr(identifier, "_estimated_ne", None)
        base_ne = _est_ne if _est_ne is not None else identifier.fallback_ne_cm3

        for offset in multi_t_offsets:
            T_offset = max(base_T + offset, 3000.0)  # Floor at 3000 K (physical minimum)
            # Deep copy to avoid aliasing mutable state (_estimated_T, _estimated_ne)
            id_copy = _copy_module.deepcopy(identifier)
            id_copy.basis_index = None  # Force fallback path
            id_copy.fallback_T_K = T_offset
            id_copy.fallback_ne_cm3 = base_ne
            try:
                offset_result = id_copy.identify(wavelength, intensity)
                all_results.append(offset_result)
            except Exception:
                logger.debug("Multi-T offset %.0f K failed; skipping", offset)

    return all_results


def _aggregate_element_coefficients(
    all_results: list,
    min_snr: float,
) -> Dict[str, float]:
    """Pool identifications, gate by SNR, aggregate MAX coefficient per element.

    Lines failing the SNR gate or with a non-positive coefficient are dropped.
    Ionization stages of the same element are collapsed to the maximum NNLS
    coefficient observed for that element.

    Parameters
    ----------
    all_results : list
        Identification results to pool (as produced by ``_collect_nnls_results``).
    min_snr : float
        Minimum NNLS coefficient SNR for inclusion.

    Returns
    -------
    dict of str to float
        Element symbol → aggregated (MAX) NNLS coefficient.
    """
    element_coefficients: Dict[str, float] = defaultdict(float)

    for result in all_results:
        for eid in result.all_elements:
            snr = eid.metadata.get("nnls_snr", 0.0)
            coeff = eid.metadata.get("nnls_coefficient", 0.0)

            if snr < min_snr or coeff <= 0:
                continue

            # Aggregate across ionization stages: MAX coefficient per element
            element = _extract_element_symbol(eid.element)
            element_coefficients[element] = max(element_coefficients[element], coeff)

    return element_coefficients


def _pad_to_k_min(
    selected: List[str],
    element_coefficients: Dict[str, float],
    k_min: int,
) -> List[str]:
    """Pad ``selected`` from the ranked rejected pool until it reaches ``k_min``.

    Appends not-yet-selected elements in descending-coefficient order until the
    selection has at least ``k_min`` entries (or the pool is exhausted). Mutates
    and returns ``selected``.

    Parameters
    ----------
    selected : list of str
        Currently selected element symbols (modified in place).
    element_coefficients : dict of str to float
        Element symbol → aggregated NNLS coefficient (the full candidate pool).
    k_min : int
        Minimum number of candidates to return.

    Returns
    -------
    list of str
        The (in-place modified) ``selected`` list.
    """
    if len(selected) < k_min:
        all_ranked = [
            el for el, _ in sorted(element_coefficients.items(), key=lambda x: x[1], reverse=True)
        ]
        for el in all_ranked:
            if el not in selected:
                selected.append(el)
            if len(selected) >= k_min:
                break

    return selected


def select_candidate_elements(
    identifier,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    force_include: Optional[List[str]] = None,
    k_max: int = 15,
    k_min: int = 3,
    min_snr: float = 3.0,
    coeff_ratio: float = 1e-4,
    multi_t_offsets: Optional[List[float]] = None,
) -> List[str]:
    """Select top-K candidate elements for Bayesian MCMC inference.

    Uses SpectralNNLSIdentifier as a fast prefilter to reduce the element
    candidate set from 50+ to K_max before passing to BayesianForwardModel.

    The algorithm:
    1. Run NNLS identification at the estimated (T, ne)
    2. Optionally run at T ± offsets and pool results (union)
    3. Filter by SNR > min_snr (noise gate)
    4. Aggregate across ionization stages (MAX coefficient per element)
    5. Keep elements where coefficient > coeff_ratio * max_coefficient
    6. Sort by coefficient descending, take top K_max
    7. If fewer than K_min, pad from rejected list
    8. Union with force_include

    Parameters
    ----------
    identifier : SpectralNNLSIdentifier
        Configured NNLS identifier with basis library loaded.
    wavelength : np.ndarray
        Observed spectrum wavelengths in nm.
    intensity : np.ndarray
        Observed spectrum intensities.
    force_include : list of str, optional
        Elements to always include (e.g., ["Fe"] for steel samples).
        These do not count against k_max unless len(force_include) > k_max.
    k_max : int
        Maximum number of candidate elements (default: 15).
    k_min : int
        Minimum number of candidates to return (default: 3).
    min_snr : float
        Minimum NNLS coefficient SNR for inclusion (default: 3.0).
    coeff_ratio : float
        Adaptive threshold: keep elements with coefficient > ratio * max (default: 1e-4).
    multi_t_offsets : list of float, optional
        Temperature offsets in Kelvin for multi-T evaluation (default: [-1500, +1500]).
        Set to [] or None to disable.

    Returns
    -------
    list of str
        Selected element symbols, sorted by NNLS significance (descending).
    """
    if force_include is None:
        force_include = []
    if multi_t_offsets is None:
        multi_t_offsets = [-1500.0, 1500.0]
    if k_min > k_max:
        raise ValueError(f"k_min ({k_min}) must not exceed k_max ({k_max})")
    if k_max < 1:
        raise ValueError(f"k_max must be at least 1, got {k_max}")

    # Steps 1-2: Run NNLS at base (T, ne) plus optional T-offset evaluations
    all_results = _collect_nnls_results(identifier, wavelength, intensity, multi_t_offsets)

    # Step 3: Pool all identifications, filter by SNR, aggregate by element
    element_coefficients = _aggregate_element_coefficients(all_results, min_snr)

    # Step 4: Adaptive threshold
    if not element_coefficients:
        logger.warning("No elements passed SNR threshold; returning force_include only")
        return list(dict.fromkeys(force_include))[:k_max] or []

    max_coeff = max(element_coefficients.values())
    threshold = coeff_ratio * max_coeff

    # Step 5: Filter and rank
    ranked = sorted(
        [(el, coeff) for el, coeff in element_coefficients.items() if coeff >= threshold],
        key=lambda x: x[1],
        reverse=True,
    )

    # Step 6: Apply K_max cap (excluding force_include)
    forced_set = set(f.strip() for f in force_include)
    dynamic_candidates = [el for el, _ in ranked if el not in forced_set]
    dynamic_slots = max(k_max - len(forced_set), 0)
    selected = list(forced_set) + dynamic_candidates[:dynamic_slots]

    # Step 7: Apply K_min floor — pad from rejected if needed
    selected = _pad_to_k_min(selected, element_coefficients, k_min)

    logger.info(
        "Prefilter: %d/%d elements selected (threshold=%.2e, max_coeff=%.2e)",
        len(selected),
        len(element_coefficients),
        threshold,
        max_coeff,
    )

    return selected

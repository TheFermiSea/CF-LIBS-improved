"""Backward-compatible shim — use cflibs.inversion.common.element_id."""

from cflibs.inversion.common.element_id import *  # noqa: F401,F403


def is_element_detected(
    element: str,
    score: float,
    n_active: int,
    min_score: float,
    min_active: int,
) -> bool:
    """
    Determine if an element is detected based on fingerprint score and line count.

    Applies stricter thresholds for Tier-2 elements (Mn, Na, K) to reduce false
    positives in crowded or low-SNR spectra.
    """
    # Tier-2 elements (Mn, Na, K) are prone to false positives due to crowded
    # resonance lines and low SNR. We enforce a higher score floor and
    # at least 2 active teeth for these elements.
    effective_min_score = min_score
    effective_min_active = min_active

    if element in {"Mn", "Na", "K"}:
        # Penalize low-confidence Mn/Na/K detections
        effective_min_score = max(min_score, 0.15)
        effective_min_active = max(min_active, 2)

    return score >= effective_min_score and n_active >= effective_min_active

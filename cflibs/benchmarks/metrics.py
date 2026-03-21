"""
Compositional accuracy metrics for CF-LIBS benchmark evaluation.

All functions operate on composition vectors represented as dictionaries
mapping element symbols to number fractions (or mass fractions).  The
functions handle edge cases such as zero or negative values by clamping
to a small epsilon before taking logarithms.

References
----------
- Aitchison, J. (1986) "The Statistical Analysis of Compositional Data"
- Egozcue et al. (2003) "Isometric Logratio Transformations for
  Compositional Data Analysis"
"""

from typing import Dict, Tuple

import numpy as np

# Minimum value used to replace zeros/negatives before log transforms.
_EPSILON = 1e-12


def _to_positive_array(
    composition: Dict[str, float],
    elements: list[str],
) -> np.ndarray:
    """Extract a positive numpy array from a composition dict.

    Parameters
    ----------
    composition : Dict[str, float]
        Element symbol -> fraction mapping.
    elements : list[str]
        Ordered element keys to extract.

    Returns
    -------
    np.ndarray
        1-D array of positive values (zeros/negatives clipped to epsilon).
    """
    arr = np.array([composition.get(el, 0.0) for el in elements], dtype=np.float64)
    return np.clip(arr, _EPSILON, None)


# ---------------------------------------------------------------------------
# Aitchison distance
# ---------------------------------------------------------------------------


def aitchison_distance(
    c_true: Dict[str, float],
    c_pred: Dict[str, float],
) -> float:
    """Compute the Aitchison distance between two compositions.

    The Aitchison distance is the Euclidean norm of the difference of
    centered log-ratio (CLR) vectors.  It is scale-invariant: multiplying
    all components by a constant does not change the distance.

    .. math::

        d_A(x, y) = \\sqrt{\\sum_i \\left(
            \\ln \\frac{x_i}{g(x)} - \\ln \\frac{y_i}{g(y)}
        \\right)^2}

    where :math:`g(\\cdot)` is the geometric mean.

    Parameters
    ----------
    c_true : Dict[str, float]
        Ground truth composition.
    c_pred : Dict[str, float]
        Predicted composition.

    Returns
    -------
    float
        Aitchison distance (non-negative).
    """
    elements = sorted(set(c_true.keys()) | set(c_pred.keys()))
    if not elements:
        return 0.0

    x = _to_positive_array(c_true, elements)
    y = _to_positive_array(c_pred, elements)

    clr_x = np.log(x) - np.mean(np.log(x))
    clr_y = np.log(y) - np.mean(np.log(y))

    return float(np.sqrt(np.sum((clr_x - clr_y) ** 2)))


# ---------------------------------------------------------------------------
# CLR transform
# ---------------------------------------------------------------------------


def clr_transform(c: Dict[str, float]) -> Dict[str, float]:
    """Centered log-ratio (CLR) transform.

    .. math::

        \\text{clr}(x)_i = \\ln \\frac{x_i}{g(x)}

    Parameters
    ----------
    c : Dict[str, float]
        Composition (element -> fraction).

    Returns
    -------
    Dict[str, float]
        CLR-transformed values keyed by element symbol.
    """
    elements = sorted(c.keys())
    if not elements:
        return {}

    arr = _to_positive_array(c, elements)
    log_arr = np.log(arr)
    clr_vals = log_arr - np.mean(log_arr)

    return {el: float(v) for el, v in zip(elements, clr_vals)}


# ---------------------------------------------------------------------------
# ILR transform
# ---------------------------------------------------------------------------


def _helmert_basis(d: int) -> np.ndarray:
    """Construct a (d-1) x d Helmert sub-composition basis.

    Each row i (0-indexed) contrasts the first i+1 components against
    the (i+2)-th component, with appropriate normalisation so the rows
    are orthonormal in the Aitchison inner product.

    Parameters
    ----------
    d : int
        Number of compositional parts.

    Returns
    -------
    np.ndarray
        Shape ``(d-1, d)`` orthonormal contrast matrix.
    """
    basis = np.zeros((d - 1, d))
    for i in range(d - 1):
        r = i + 1
        basis[i, :r] = 1.0 / r
        basis[i, r] = -1.0
        basis[i] *= np.sqrt(r / (r + 1.0))
    return basis


def ilr_transform(c: Dict[str, float]) -> np.ndarray:
    """Isometric log-ratio (ILR) transform using the Helmert sub-composition.

    Maps a D-part composition to a (D-1)-dimensional real vector that
    lives in an unconstrained Euclidean space.  Euclidean distance in
    ILR coordinates equals the Aitchison distance.

    Parameters
    ----------
    c : Dict[str, float]
        Composition (element -> fraction).

    Returns
    -------
    np.ndarray
        ILR coordinates, shape ``(D-1,)``.
    """
    elements = sorted(c.keys())
    d = len(elements)
    if d <= 1:
        return np.array([], dtype=np.float64)

    arr = _to_positive_array(c, elements)
    log_arr = np.log(arr)

    basis = _helmert_basis(d)
    return basis @ log_arr


def ilr_inverse(coords: np.ndarray, elements: list[str]) -> Dict[str, float]:
    """Inverse ILR transform: map ILR coordinates back to a composition.

    Parameters
    ----------
    coords : np.ndarray
        ILR coordinates, shape ``(D-1,)``.
    elements : list[str]
        Ordered element symbols (length D).

    Returns
    -------
    Dict[str, float]
        Composition that sums to 1.
    """
    d = len(elements)
    if d <= 1:
        return {elements[0]: 1.0} if elements else {}

    basis = _helmert_basis(d)
    log_arr = basis.T @ coords
    arr = np.exp(log_arr)
    arr /= arr.sum()
    return {el: float(v) for el, v in zip(elements, arr)}


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------


def rmse_composition(
    c_true: Dict[str, float],
    c_pred: Dict[str, float],
) -> float:
    """Root mean squared error between two compositions.

    Parameters
    ----------
    c_true : Dict[str, float]
        Ground truth composition.
    c_pred : Dict[str, float]
        Predicted composition.

    Returns
    -------
    float
        RMSE (non-negative).
    """
    elements = sorted(set(c_true.keys()) | set(c_pred.keys()))
    if not elements:
        return 0.0

    true_arr = np.array([c_true.get(el, 0.0) for el in elements], dtype=np.float64)
    pred_arr = np.array([c_pred.get(el, 0.0) for el in elements], dtype=np.float64)

    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))


def per_element_error(
    c_true: Dict[str, float],
    c_pred: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """Per-element absolute and relative error.

    Parameters
    ----------
    c_true : Dict[str, float]
        Ground truth composition.
    c_pred : Dict[str, float]
        Predicted composition.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Mapping of element -> (absolute_error, relative_error).
        Relative error is ``inf`` when the true value is zero.
    """
    elements = sorted(set(c_true.keys()) | set(c_pred.keys()))
    result: Dict[str, Tuple[float, float]] = {}

    for el in elements:
        t = c_true.get(el, 0.0)
        p = c_pred.get(el, 0.0)
        abs_err = abs(p - t)
        rel_err = abs_err / t if t > 0 else float("inf") if abs_err > 0 else 0.0
        result[el] = (abs_err, rel_err)

    return result

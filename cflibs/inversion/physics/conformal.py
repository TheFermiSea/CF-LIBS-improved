"""
Distribution-free, finite-sample prediction intervals for CF-LIBS outputs.

This module wraps a CF-LIBS composition (or any scalar) point estimate with
*conformal* prediction intervals. Conformal prediction is an **additive
diagnostic**: it never changes the inversion point estimate. It consumes a held
out calibration set (predictions paired with known ground truth) and returns an
interval around the point estimate that carries a finite-sample, distribution
free marginal-coverage guarantee under exchangeability.

Two estimators are provided:

1. **Split conformal prediction** (Lei et al., JASA 2018). Given calibration
   nonconformity scores ``s_i`` (e.g. absolute residuals ``|y_i - point_i|``),
   the level-:math:`\\alpha` threshold is the
   :math:`\\lceil (1-\\alpha)(n+1) \\rceil / n` empirical quantile of the scores,
   equivalently the :math:`(1-\\alpha)(1 + 1/n)` quantile. The interval is then
   ``point +/- q_hat`` — a *constant width* band, since the calibration
   threshold does not depend on the test point.

2. **Conformalized quantile regression (CQR)** (Romano, Patterson & Candes,
   NeurIPS 2019). Given calibration *lower* and *upper* conditional-quantile
   predictions and the calibration truths, the conformity score is
   :math:`E_i = \\max\\{\\hat{q}_{lo}(x_i) - y_i,\\ y_i - \\hat{q}_{hi}(x_i)\\}`
   (negative inside the interval, positive outside). Its
   :math:`\\lceil (1-\\alpha)(n+1) \\rceil / n` empirical quantile :math:`E` is
   the calibration correction; test intervals become
   :math:`[\\hat{q}_{lo}(x) - E,\\ \\hat{q}_{hi}(x) + E]`. Because the base
   quantiles already track the conditional spread, CQR adapts to
   heteroscedastic noise and is typically *no wider* than the constant-width
   split-CP band while retaining the same marginal-coverage guarantee — the
   central result of Romano et al. (2019).

Both estimators give, for an exchangeable calibration/test draw of size
:math:`n` and a fresh test point,

.. math::

    \\Pr\\{Y_{n+1} \\in \\hat{C}(X_{n+1})\\} \\ge 1 - \\alpha ,

with the upper bound :math:`1 - \\alpha + 1/(n+1)` when scores are almost surely
distinct.

This module is **physics-only**: it depends on numpy alone (no learned models,
no sklearn/torch/etc.). The quantile predictions consumed by CQR are supplied
by the caller — this module performs only the distribution-free calibration.

References
----------
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
  "Distribution-Free Predictive Inference for Regression."
  *Journal of the American Statistical Association*, 113(523), 1094-1111.
- Romano, Y., Patterson, E., & Candes, E. (2019). "Conformalized Quantile
  Regression." *Advances in Neural Information Processing Systems 32* (NeurIPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.conformal")

__all__ = [
    "ConformalInterval",
    "conformal_rank",
    "conformal_quantile_level",
    "split_conformal",
    "conformal_interval",
    "cqr_conformity_scores",
    "cqr_calibrate",
    "cqr_interval",
    "conformalize_split",
    "conformalize_cqr",
]


@dataclass(frozen=True)
class ConformalInterval:
    """A conformal prediction interval with its calibration metadata.

    Attributes
    ----------
    lo, hi : np.ndarray
        Lower and upper interval endpoints (broadcast to the test shape).
    q_hat : float
        Calibration threshold (split-CP radius, or CQR correction ``E``).
    alpha : float
        Requested miscoverage level; nominal coverage is ``1 - alpha``.
    n_cal : int
        Number of finite calibration scores used.
    method : str
        ``"split"`` or ``"cqr"``.
    """

    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    q_hat: float
    alpha: float
    n_cal: int
    method: str

    @property
    def width(self) -> NDArray[np.float64]:
        """Pointwise interval width ``hi - lo``."""
        return self.hi - self.lo


def _validate_alpha(alpha: float) -> float:
    """Coerce and range-check the miscoverage level ``alpha`` in (0, 1)."""
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1); got {alpha!r}.")
    return alpha


def _finite_scores(cal_scores: ArrayLike, *, name: str = "cal_scores") -> NDArray[np.float64]:
    """Flatten ``cal_scores`` to finite float64 values, validating non-emptiness."""
    scores = np.asarray(cal_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values.")
    if finite.size != scores.size:
        logger.warning(
            "Dropping %d non-finite value(s) from %s (%d -> %d).",
            scores.size - finite.size,
            name,
            scores.size,
            finite.size,
        )
    return finite


def conformal_rank(n: int, alpha: float) -> int:
    """Finite-sample conformal *order-statistic rank* :math:`k` (1-indexed).

    Returns :math:`k = \\lceil (1-\\alpha)(n+1) \\rceil`, the rank of the
    calibration score used as the split-conformal threshold (Lei et al., 2018):
    the threshold is the :math:`k`-th *smallest* of the ``n`` calibration scores.
    When :math:`k > n` (i.e. ``(1-alpha)(n+1) > n``), ``n`` is too small to
    certify the requested coverage and the conformal interval is unbounded; this
    is signalled by returning ``k = n + 1`` (a rank that no score attains).

    Parameters
    ----------
    n : int
        Number of (finite) calibration scores; must be >= 1.
    alpha : float
        Miscoverage level in (0, 1).

    Returns
    -------
    int
        1-indexed order-statistic rank :math:`k` in ``[1, n + 1]``.
    """
    alpha = _validate_alpha(alpha)
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}.")
    return int(np.ceil((1.0 - alpha) * (n + 1)))


def conformal_quantile_level(n: int, alpha: float) -> float:
    """Finite-sample conformal quantile *level* for ``n`` calibration scores.

    Returns the rank level :math:`\\lceil (1-\\alpha)(n+1) \\rceil / n` used by
    split conformal prediction (Lei et al., 2018), equivalently the
    :math:`(1-\\alpha)(1 + 1/n)` level. This is the inflated quantile that yields
    finite-sample marginal coverage; it is clipped to ``1.0`` when
    ``(1-alpha)(n+1) > n`` (i.e. when ``n`` is too small to certify the level,
    in which case the conservative interval is unbounded).

    Note
    ----
    This is the *level* (in (0, 1]); the calibration threshold is selected by
    order-statistic *rank* (:func:`conformal_rank`), not by passing this level to
    :func:`numpy.quantile`, whose interpolation indexing (position ``q*(n-1)``)
    does not coincide with the conformal rank ``ceil((1-alpha)(n+1))``.

    Parameters
    ----------
    n : int
        Number of (finite) calibration scores; must be >= 1.
    alpha : float
        Miscoverage level in (0, 1).

    Returns
    -------
    float
        Quantile level in (0, 1].
    """
    alpha = _validate_alpha(alpha)
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}.")
    level = np.ceil((1.0 - alpha) * (n + 1)) / n
    return float(min(level, 1.0))


def _threshold_from_rank(
    scores: NDArray[np.float64], alpha: float, *, label: str, kind: str
) -> float:
    """Select the conformal order statistic; return +inf if rank exceeds ``n``."""
    n = scores.size
    k = conformal_rank(n, alpha)
    if k > n:
        logger.warning(
            "n=%d too small to certify %.0f%% %s coverage; returning %s=+inf.",
            n,
            100.0 * (1.0 - alpha),
            label,
            kind,
        )
        return float("inf")
    # k is 1-indexed; the k-th smallest is index k-1 of the sorted scores.
    return float(np.partition(scores, k - 1)[k - 1])


def split_conformal(cal_scores: ArrayLike, alpha: float) -> float:
    """Split-conformal threshold ``q_hat`` from calibration nonconformity scores.

    Computes the :math:`\\lceil (1-\\alpha)(n+1) \\rceil / n`-empirical quantile of
    the calibration nonconformity scores (Lei et al., JASA 2018). For the
    symmetric residual score :math:`s_i = |y_i - \\hat{y}_i|`, the resulting
    ``q_hat`` is the half-width of a constant-width prediction band that attains
    marginal coverage :math:`\\ge 1 - \\alpha` under exchangeability.

    Parameters
    ----------
    cal_scores : array_like
        Nonconformity scores on the calibration set (e.g. absolute residuals).
        Non-finite entries are dropped with a warning.
    alpha : float
        Miscoverage level in (0, 1).

    Returns
    -------
    float
        The calibration threshold ``q_hat`` (``+inf`` if ``n`` is too small to
        certify the requested level).

    Notes
    -----
    The threshold is the :math:`k`-th *smallest* calibration score with
    :math:`k = \\lceil (1-\\alpha)(n+1) \\rceil` (:func:`conformal_rank`) — an
    *attained* calibration score, which gives the :math:`\\le 1 - \\alpha +
    1/(n+1)` upper coverage bound for distinct scores. This rank selection is
    used in preference to :func:`numpy.quantile`, whose interpolation indexing
    (position ``q*(n-1)``) does not match the conformal rank.
    """
    scores = _finite_scores(cal_scores)
    return _threshold_from_rank(scores, alpha, label="", kind="q_hat")


def conformal_interval(
    point: ArrayLike, q_hat: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric split-conformal interval ``(point - q_hat, point + q_hat)``.

    Parameters
    ----------
    point : array_like
        Point estimate(s) to wrap (e.g. CF-LIBS concentrations). Shape preserved.
    q_hat : float
        Calibration threshold from :func:`split_conformal` (must be >= 0).

    Returns
    -------
    lo, hi : np.ndarray
        Lower and upper interval endpoints, same shape as ``point``.
    """
    q = float(q_hat)
    if q < 0.0:
        raise ValueError(f"q_hat must be non-negative; got {q!r}.")
    pt = np.asarray(point, dtype=np.float64)
    return pt - q, pt + q


def cqr_conformity_scores(
    cal_lower: ArrayLike, cal_upper: ArrayLike, cal_truth: ArrayLike
) -> NDArray[np.float64]:
    """CQR conformity scores ``E_i = max(lo_i - y_i, y_i - hi_i)``.

    Implements the conformity score of Romano, Patterson & Candes (NeurIPS
    2019): negative when the truth falls inside the predicted quantile interval,
    positive (the signed distance to the nearer endpoint) when it falls outside.

    Parameters
    ----------
    cal_lower, cal_upper : array_like
        Calibration lower and upper conditional-quantile predictions
        (typically the ``alpha/2`` and ``1 - alpha/2`` quantiles).
    cal_truth : array_like
        Calibration ground-truth responses.

    Returns
    -------
    np.ndarray
        One conformity score per calibration point (1-D, float64).
    """
    lo = np.asarray(cal_lower, dtype=np.float64).ravel()
    hi = np.asarray(cal_upper, dtype=np.float64).ravel()
    y = np.asarray(cal_truth, dtype=np.float64).ravel()
    if not (lo.shape == hi.shape == y.shape):
        raise ValueError(
            "cal_lower, cal_upper, cal_truth must share a shape; "
            f"got {lo.shape}, {hi.shape}, {y.shape}."
        )
    return np.maximum(lo - y, y - hi)


def cqr_calibrate(
    cal_lower: ArrayLike, cal_upper: ArrayLike, cal_truth: ArrayLike, alpha: float
) -> float:
    """CQR calibration correction ``E`` (conformalized-quantile threshold).

    Computes the conformity scores via :func:`cqr_conformity_scores` then their
    :math:`\\lceil (1-\\alpha)(n+1) \\rceil / n`-empirical quantile (Romano et al.,
    2019). The result ``E`` is added/subtracted to the base quantile endpoints
    by :func:`cqr_interval`; ``E`` may be negative, which *shrinks* an
    over-conservative base interval while preserving the coverage guarantee.

    Parameters
    ----------
    cal_lower, cal_upper, cal_truth : array_like
        Calibration lower/upper quantile predictions and truths.
    alpha : float
        Miscoverage level in (0, 1).

    Returns
    -------
    float
        Calibration correction ``E`` (``+inf`` if ``n`` is too small to certify).
    """
    scores = cqr_conformity_scores(cal_lower, cal_upper, cal_truth)
    finite = _finite_scores(scores, name="cqr conformity scores")
    return _threshold_from_rank(finite, alpha, label="CQR", kind="E")


def cqr_interval(
    test_lower: ArrayLike, test_upper: ArrayLike, correction: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Conformalized quantile interval ``(lo - E, hi + E)``.

    Parameters
    ----------
    test_lower, test_upper : array_like
        Base lower/upper quantile predictions at the test point(s).
    correction : float
        Calibration correction ``E`` from :func:`cqr_calibrate`.

    Returns
    -------
    lo, hi : np.ndarray
        Conformalized interval endpoints, broadcast to the inputs' shape.
    """
    e = float(correction)
    lo = np.asarray(test_lower, dtype=np.float64)
    hi = np.asarray(test_upper, dtype=np.float64)
    return lo - e, hi + e


def conformalize_split(
    point: ArrayLike,
    cal_scores: ArrayLike,
    alpha: float,
    *,
    test_point: Union[ArrayLike, None] = None,
) -> ConformalInterval:
    """End-to-end split-CP wrapper returning a :class:`ConformalInterval`.

    Convenience entry point: calibrate ``q_hat`` from ``cal_scores`` then build a
    symmetric band around ``test_point`` (default: ``point``).

    Parameters
    ----------
    point : array_like
        Point estimate used as the default band center.
    cal_scores : array_like
        Calibration nonconformity scores.
    alpha : float
        Miscoverage level in (0, 1).
    test_point : array_like, optional
        Point(s) to wrap; defaults to ``point``.

    Returns
    -------
    ConformalInterval
    """
    scores = _finite_scores(cal_scores)
    q_hat = split_conformal(scores, alpha)
    target = point if test_point is None else test_point
    lo, hi = conformal_interval(target, q_hat) if np.isfinite(q_hat) else _unbounded(target)
    return ConformalInterval(
        lo=lo, hi=hi, q_hat=q_hat, alpha=_validate_alpha(alpha), n_cal=scores.size, method="split"
    )


def conformalize_cqr(
    test_lower: ArrayLike,
    test_upper: ArrayLike,
    cal_lower: ArrayLike,
    cal_upper: ArrayLike,
    cal_truth: ArrayLike,
    alpha: float,
) -> ConformalInterval:
    """End-to-end CQR wrapper returning a :class:`ConformalInterval`.

    Parameters
    ----------
    test_lower, test_upper : array_like
        Base quantile predictions at the test point(s).
    cal_lower, cal_upper, cal_truth : array_like
        Calibration quantile predictions and truths.
    alpha : float
        Miscoverage level in (0, 1).

    Returns
    -------
    ConformalInterval
    """
    e = cqr_calibrate(cal_lower, cal_upper, cal_truth, alpha)
    n_cal = np.asarray(cal_truth, dtype=np.float64).size
    if np.isfinite(e):
        lo, hi = cqr_interval(test_lower, test_upper, e)
    else:
        lo, hi = _unbounded(test_lower)
    return ConformalInterval(
        lo=lo, hi=hi, q_hat=e, alpha=_validate_alpha(alpha), n_cal=n_cal, method="cqr"
    )


def _unbounded(
    ref: ArrayLike,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a (-inf, +inf) interval broadcast to ``ref``'s shape."""
    shape = np.asarray(ref, dtype=np.float64).shape
    return np.full(shape, -np.inf), np.full(shape, np.inf)

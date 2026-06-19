"""TARP posterior-coverage diagnostic for CF-LIBS Bayesian inversion.

This module implements the **Tests of Accuracy with Random Points (TARP)**
multivariate coverage test of Lemos, Coogan, Hezaveh & Perreault-Levasseur
(ICML 2023). TARP is an *evaluation-free* diagnostic: it operates on posterior
**samples only** and never needs the posterior density. This makes it a natural
fit for CF-LIBS, where the Bayesian inversion (`cflibs.inversion.solve.bayesian`)
returns NUTS/dynesty samples of plasma parameters (T, n_e, log-ratio composition)
rather than a tractable density.

This is a **purely additive diagnostic**: it does not touch the inversion point
estimate. It answers the question "are my posterior credible intervals trustworthy
(neither over- nor under-confident)?" given a validation set of (posterior samples,
ground-truth parameter) pairs -- typically obtained by inverting a corpus of
synthetic spectra with known composition.

Algorithm (Lemos et al. 2023, Algorithm 2)
------------------------------------------
For a validation set of ``n_obj`` test objects, each with ``n_samples`` posterior
draws in a ``dim``-dimensional parameter space and a known true parameter:

1. (Optionally) standardize every parameter axis using the pooled mean/std of the
   samples so the L2 metric is not dominated by one physical scale (e.g. n_e in
   cm^-3 vs. mass fractions in [0, 1]).
2. For each object draw a random reference point ``r``. Following the paper's
   reference implementation the default draws ``r`` uniformly in the bounding
   hypercube of the (standardized) samples.
3. Form the TARP region as the L2 ball centred on ``r`` with radius
   ``||theta_true - r||``, and compute the *posterior mass* the samples assign to
   it: ``f = mean( ||theta_sample - r|| < ||theta_true - r|| )``.
4. The set ``{f_i}`` over objects is distributed as ``Uniform(0, 1)`` **iff** the
   posteriors are calibrated (Theorem in Lemos et al. 2023). The Expected Coverage
   Probability at nominal credibility level ``c`` is the empirical CDF
   ``ECP(c) = mean( f_i <= c )``; a perfectly calibrated estimator yields
   ``ECP(c) = c`` (the diagonal).

Interpretation
--------------
- ECP curve **on the diagonal**: calibrated.
- ECP **below the diagonal** (curve sags right/down): the credible regions are too
  small -> **over-confident / under-coverage** (the most dangerous failure for
  reporting CF-LIBS composition error bars).
- ECP **above the diagonal**: credible regions too large -> under-confident /
  conservative.

References
----------
- Lemos, P., Coogan, A., Hezaveh, Y. & Perreault-Levasseur, L. "Sampling-Based
  Accuracy Testing of Posterior Estimators for General Inference." Proceedings of
  the 40th International Conference on Machine Learning (ICML), 2023.
  arXiv:2302.03026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.coverage")

__all__ = ["TARPCoverageResult", "tarp_coverage"]


@dataclass
class TARPCoverageResult:
    """Result of a TARP posterior-coverage test.

    Attributes
    ----------
    credibility : np.ndarray
        Nominal credibility levels in ``(0, 1)`` at which the Expected Coverage
        Probability is evaluated. Shape ``(n_alpha,)``.
    ecp : np.ndarray
        Empirical Expected Coverage Probability ``ECP(c) = mean(f_i <= c)`` at each
        ``credibility`` level. Shape ``(n_alpha,)``. A perfectly calibrated
        estimator gives ``ecp == credibility`` (the diagonal).
    coverage_fractions : np.ndarray
        Per-object TARP statistic ``f_i`` (the posterior mass inside the TARP region
        for object ``i``). Shape ``(n_obj,)``. Under calibration these are
        ``Uniform(0, 1)``.
    ks_statistic : float
        Kolmogorov-Smirnov distance between the empirical distribution of
        ``coverage_fractions`` and ``Uniform(0, 1)``. Small (~0) means calibrated.
    ks_pvalue : float
        Two-sided KS test p-value against ``Uniform(0, 1)``. A small p-value
        (e.g. < 0.05) is evidence of mis-calibration.
    ecp_bias : float
        Signed mean deviation of the ECP curve from the diagonal,
        ``mean(ecp - credibility)``. Negative -> under-coverage (over-confident);
        positive -> over-coverage (under-confident).
    max_abs_deviation : float
        Maximum absolute deviation ``max|ecp - credibility|`` of the ECP curve from
        the diagonal -- a worst-case calibration error analogous to the KS statistic
        but measured on the ECP curve itself.
    n_obj : int
        Number of test objects.
    n_samples : int
        Number of posterior samples per object.
    dim : int
        Parameter-space dimensionality.
    warnings : list
        Non-fatal issues encountered (e.g. zero-variance parameter axes).
    """

    credibility: np.ndarray
    ecp: np.ndarray
    coverage_fractions: np.ndarray
    ks_statistic: float
    ks_pvalue: float
    ecp_bias: float
    max_abs_deviation: float
    n_obj: int
    n_samples: int
    dim: int
    warnings: list = field(default_factory=list)

    @property
    def is_calibrated(self) -> bool:
        """Heuristic calibration verdict.

        ``True`` when the ECP curve stays within 0.1 of the diagonal everywhere and
        the KS test does not reject uniformity at the 5% level. This is a convenience
        flag for triage, not a substitute for inspecting the ECP curve.
        """
        return self.max_abs_deviation < 0.1 and self.ks_pvalue > 0.05

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary (curve arrays as lists)."""
        return {
            "credibility": self.credibility.tolist(),
            "ecp": self.ecp.tolist(),
            "ks_statistic": float(self.ks_statistic),
            "ks_pvalue": float(self.ks_pvalue),
            "ecp_bias": float(self.ecp_bias),
            "max_abs_deviation": float(self.max_abs_deviation),
            "is_calibrated": self.is_calibrated,
            "n_obj": self.n_obj,
            "n_samples": self.n_samples,
            "dim": self.dim,
            "warnings": list(self.warnings),
        }


def _draw_reference_points(
    samples: np.ndarray,
    references: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one reference point per object.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape ``(n_obj, n_samples, dim)`` (already standardized
        if standardization is requested by the caller).
    references : {"uniform", "normal"}
        ``"uniform"`` (default, per Lemos et al. 2023) draws each reference point
        uniformly in the global bounding hypercube of all samples. ``"normal"`` draws
        from a global Gaussian matched to the pooled sample mean/std.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Reference points, shape ``(n_obj, dim)``.
    """
    n_obj, _, dim = samples.shape
    pooled = samples.reshape(-1, dim)
    if references == "uniform":
        lo = pooled.min(axis=0)
        hi = pooled.max(axis=0)
        # Guard against a degenerate (zero-width) axis.
        span = hi - lo
        span = np.where(span > 0.0, span, 1.0)
        return lo + rng.random((n_obj, dim)) * span
    if references == "normal":
        mu = pooled.mean(axis=0)
        sigma = pooled.std(axis=0)
        sigma = np.where(sigma > 0.0, sigma, 1.0)
        return mu + rng.standard_normal((n_obj, dim)) * sigma
    raise ValueError(f"references must be 'uniform' or 'normal', got {references!r}")


def tarp_coverage(
    samples: np.ndarray,
    theta_true: np.ndarray,
    n_alpha: int = 100,
    references: str = "uniform",
    standardize: bool = True,
    seed: Optional[int] = None,
) -> TARPCoverageResult:
    r"""Run the TARP posterior-coverage test (Lemos et al. 2023, arXiv:2302.03026).

    TARP is an evaluation-free multivariate coverage diagnostic that uses only
    posterior **samples** (no density). For each test object it draws a random
    reference point ``r`` and measures the fraction of posterior samples that lie
    closer to ``r`` than the true parameter does. Over a validation set these
    fractions are ``Uniform(0, 1)`` if and only if the posteriors are calibrated; the
    Expected Coverage Probability curve (the empirical CDF of those fractions) then
    follows the diagonal.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape ``(n_obj, n_samples, dim)``. ``n_obj`` is the number
        of validation objects, ``n_samples`` the posterior draws per object, ``dim``
        the parameter-space dimensionality. A 2-D array ``(n_samples, dim)`` for a
        single object is accepted and promoted to ``(1, n_samples, dim)``.
    theta_true : np.ndarray
        Ground-truth parameters, shape ``(n_obj, dim)`` (or ``(dim,)`` for a single
        object). Must use the same parameter ordering as ``samples``.
    n_alpha : int, optional
        Number of nominal credibility levels at which to evaluate the ECP curve.
        Levels are placed on a uniform grid in ``[0, 1]``. Default 100.
    references : {"uniform", "normal"}, optional
        Reference-point distribution. ``"uniform"`` (default, as in the paper) draws
        in the bounding hypercube of the samples; ``"normal"`` draws from a Gaussian
        matched to the pooled samples.
    standardize : bool, optional
        If ``True`` (default), z-score every parameter axis using the pooled
        sample mean/std before computing L2 distances, so no single physical scale
        (e.g. ``n_e`` vs. mass fractions) dominates the metric. The true parameters
        are standardized with the *same* transform.
    seed : int, optional
        Seed for reproducible reference-point sampling.

    Returns
    -------
    TARPCoverageResult
        Dataclass holding the ECP curve, per-object coverage fractions, and scalar
        calibration statistics (KS distance/p-value to uniform, ECP bias, max
        deviation from the diagonal).

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``dim`` mismatches between ``samples`` and
        ``theta_true``, ``n_alpha < 2``, there are fewer than 2 objects (a single
        object cannot estimate a coverage *distribution*), or inputs are non-finite.

    Notes
    -----
    Computational cost is ``O(n_obj * n_samples * dim)`` and fully vectorized in
    NumPy. The diagnostic is additive and read-only with respect to the inversion: it
    consumes already-computed posteriors and never alters a point estimate.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n_obj, n_samples, dim = 2000, 800, 2
    >>> # Calibrated toy model (Lemos et al. 2023): truth and samples both drawn from
    >>> # N(theta_bar, 1); the truth is a genuine posterior draw, so f_i ~ U(0, 1).
    >>> theta_bar = rng.uniform(-5, 5, size=(n_obj, dim))
    >>> theta_true = theta_bar + rng.normal(size=(n_obj, dim))
    >>> samples = theta_bar[:, None, :] + rng.normal(size=(n_obj, n_samples, dim))
    >>> res = tarp_coverage(samples, theta_true, seed=1)
    >>> bool(res.ks_statistic < 0.1)
    True

    References
    ----------
    Lemos, P., Coogan, A., Hezaveh, Y. & Perreault-Levasseur, L. "Sampling-Based
    Accuracy Testing of Posterior Estimators for General Inference." ICML 2023,
    arXiv:2302.03026.
    """
    samples = np.asarray(samples, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)

    # Promote single-object inputs to the canonical 3-D / 2-D shapes.
    if samples.ndim == 2:
        samples = samples[None, :, :]
    if theta_true.ndim == 1:
        theta_true = theta_true[None, :]

    if samples.ndim != 3:
        raise ValueError(
            f"samples must have shape (n_obj, n_samples, dim); got ndim={samples.ndim}"
        )
    if theta_true.ndim != 2:
        raise ValueError(f"theta_true must have shape (n_obj, dim); got ndim={theta_true.ndim}")

    n_obj, n_samples, dim = samples.shape
    if theta_true.shape[0] != n_obj:
        raise ValueError(
            f"n_obj mismatch: samples has {n_obj} objects, theta_true has " f"{theta_true.shape[0]}"
        )
    if theta_true.shape[1] != dim:
        raise ValueError(f"dim mismatch: samples dim={dim}, theta_true dim={theta_true.shape[1]}")
    if n_alpha < 2:
        raise ValueError(f"n_alpha must be >= 2, got {n_alpha}")
    if n_obj < 2:
        raise ValueError(
            "TARP estimates a coverage *distribution* over objects; need >= 2 objects, "
            f"got {n_obj}"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples contains non-finite values (nan/inf)")
    if not np.all(np.isfinite(theta_true)):
        raise ValueError("theta_true contains non-finite values (nan/inf)")

    warnings_list: list = []

    # Standardize each axis using pooled sample statistics so L2 is scale-free.
    if standardize:
        pooled = samples.reshape(-1, dim)
        mu = pooled.mean(axis=0)
        sigma = pooled.std(axis=0)
        zero_var = sigma <= 0.0
        if np.any(zero_var):
            warnings_list.append(
                f"{int(zero_var.sum())} parameter axis/axes had zero sample variance; "
                "left un-scaled."
            )
        sigma = np.where(zero_var, 1.0, sigma)
        samples = (samples - mu) / sigma
        theta_true = (theta_true - mu) / sigma

    rng = np.random.default_rng(seed)
    refs = _draw_reference_points(samples, references, rng)  # (n_obj, dim)

    # Distance from the reference to the truth, per object: (n_obj,)
    d_true = np.linalg.norm(theta_true - refs, axis=1)
    # Distance from the reference to each posterior sample: (n_obj, n_samples)
    d_samples = np.linalg.norm(samples - refs[:, None, :], axis=2)

    # TARP statistic f_i: posterior mass inside the ball of radius d_true.
    # Strict "<" matches the paper's region definition; ties (a sample exactly on the
    # truth's shell) are vanishingly rare for continuous posteriors.
    coverage_fractions = np.mean(d_samples < d_true[:, None], axis=1)  # (n_obj,)

    # ECP(c) = empirical CDF of {f_i} = fraction of objects with f_i <= c.
    credibility = np.linspace(0.0, 1.0, n_alpha)
    # Broadcast compare: (n_alpha, n_obj) -> mean over objects.
    ecp = np.mean(coverage_fractions[None, :] <= credibility[:, None], axis=1)  # (n_alpha,)

    # Calibration statistic: KS distance of {f_i} to Uniform(0, 1).
    ks_statistic, ks_pvalue = _ks_uniform(coverage_fractions)

    deviation = ecp - credibility
    ecp_bias = float(np.mean(deviation))
    max_abs_deviation = float(np.max(np.abs(deviation)))

    logger.debug(
        "TARP coverage: n_obj=%d n_samples=%d dim=%d KS=%.4f (p=%.3f) bias=%.4f " "max_dev=%.4f",
        n_obj,
        n_samples,
        dim,
        ks_statistic,
        ks_pvalue,
        ecp_bias,
        max_abs_deviation,
    )

    return TARPCoverageResult(
        credibility=credibility,
        ecp=ecp,
        coverage_fractions=coverage_fractions,
        ks_statistic=float(ks_statistic),
        ks_pvalue=float(ks_pvalue),
        ecp_bias=ecp_bias,
        max_abs_deviation=max_abs_deviation,
        n_obj=n_obj,
        n_samples=n_samples,
        dim=dim,
        warnings=warnings_list,
    )


def _ks_uniform(values: np.ndarray) -> Tuple[float, float]:
    """KS distance and two-sided p-value of ``values`` against Uniform(0, 1).

    Uses :func:`scipy.stats.kstest` when SciPy is available; otherwise falls back to
    the one-sample KS statistic (supremum of ``|F_n(x) - x|``) and the asymptotic
    Kolmogorov p-value, so the diagnostic still degrades gracefully without SciPy.

    Parameters
    ----------
    values : np.ndarray
        Sample of coverage fractions, expected to lie in ``[0, 1]``.

    Returns
    -------
    tuple of float
        ``(ks_statistic, p_value)``.
    """
    try:
        from scipy import stats  # local import: keep SciPy optional at module load

        result = stats.kstest(values, "uniform")
        return float(result.statistic), float(result.pvalue)
    except Exception:  # pragma: no cover - exercised only without SciPy
        x = np.sort(np.asarray(values, dtype=float))
        n = x.size
        cdf = np.arange(1, n + 1) / n
        d_plus = np.max(cdf - x)
        d_minus = np.max(x - (np.arange(0, n) / n))
        ks = float(max(d_plus, d_minus))
        # Asymptotic Kolmogorov survival function (Q(t) two-sided tail).
        t = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks
        j = np.arange(1, 101)
        q = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * (j**2) * t**2))
        p = float(min(max(q, 0.0), 1.0))
        return ks, p

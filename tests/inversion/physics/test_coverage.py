"""Tests for the TARP posterior-coverage diagnostic (Lemos et al. 2023).

These tests use only synthetic Gaussian "posteriors" with known ground truth -- no
atomic DB, no JAX, no real inversion -- so they run in well under a second.

The synthetic validation set follows the exact toy model used in the TARP paper
(Lemos et al. 2023, Sec. 4 / Fig. 2) and reproduced by Detecting-model-misspecification
(arXiv) for their calibration sanity checks:

    posterior centre   theta_bar_i ~ U(-5, 5)          (the estimator's mean)
    true value         theta*_i    ~ N(theta_bar_i, sigma^2)   (correct DGP)
    posterior samples  theta_ij    ~ N(theta_bar_i, (spread*sigma)^2)

So ``spread == 1`` is **calibrated** (samples drawn from the same law that produced
the truth), ``spread < 1`` is **over-confident** (too narrow), and ``spread > 1`` is
**under-confident** (too wide). Note the truth is centred on ``theta_bar_i``, *not*
on itself -- centring the cloud on the truth would make every f_i pile up at 0.5 and
is *not* a calibrated posterior.

Acceptance criterion (from the candidate specification), two arms:

1. **Calibrated** samples: the ECP curve tracks the diagonal and the KS distance of
   the per-object coverage fractions to Uniform(0, 1) is small.
2. **Over-confident** samples (shrunk spread): the ECP curve lies *below* the
   diagonal -- TARP detects the under-coverage (negative ECP bias, large KS, and
   ECP(c) < c throughout the upper credibility band).

We also pin the exact mathematics with independent oracles (a delta posterior and a
deterministic 1-D reference recomputation) so the test really checks the algorithm
rather than tautologically re-deriving the implementation.

Reference
---------
Lemos, P., Coogan, A., Hezaveh, Y. & Perreault-Levasseur, L. "Sampling-Based
Accuracy Testing of Posterior Estimators for General Inference." ICML 2023,
arXiv:2302.03026.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.physics.coverage import TARPCoverageResult, tarp_coverage


def _toy_posteriors(
    n_obj: int,
    n_samples: int,
    dim: int,
    spread: float,
    seed: int,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the TARP-paper Gaussian toy validation set.

    Parameters
    ----------
    spread : float
        Posterior-width multiplier. ``1.0`` -> calibrated, ``< 1`` -> over-confident,
        ``> 1`` -> under-confident.
    """
    rng = np.random.default_rng(seed)
    theta_bar = rng.uniform(-5.0, 5.0, size=(n_obj, dim))  # posterior centre
    theta_true = theta_bar + sigma * rng.normal(size=(n_obj, dim))  # correct DGP
    samples = theta_bar[:, None, :] + spread * sigma * rng.normal(size=(n_obj, n_samples, dim))
    return samples, theta_true


def test_calibrated_gaussian_is_near_diagonal() -> None:
    """Calibrated samples: ECP ~ diagonal, small KS, uniformity not rejected."""
    samples, theta_true = _toy_posteriors(n_obj=2000, n_samples=1000, dim=3, spread=1.0, seed=0)
    res = tarp_coverage(samples, theta_true, n_alpha=100, seed=11)

    assert isinstance(res, TARPCoverageResult)
    # Per-object coverage fractions ~ Uniform(0, 1): mean ~ 0.5, small KS.
    assert res.coverage_fractions.mean() == pytest.approx(0.5, abs=0.03)
    assert res.ks_statistic < 0.05, f"KS too large for calibrated posterior: {res.ks_statistic}"
    # ECP curve hugs the diagonal everywhere.
    assert res.max_abs_deviation < 0.05, f"ECP off diagonal: {res.max_abs_deviation}"
    assert abs(res.ecp_bias) < 0.02
    # Uniformity should not be rejected at the 5% level.
    assert res.ks_pvalue > 0.05
    assert res.is_calibrated


def test_overconfident_lies_below_diagonal() -> None:
    """Over-confident (shrunk) samples: ECP below the diagonal => under-coverage.

    This is the primary acceptance test of the candidate spec. The signature, exactly
    as described in Lemos et al. 2023 for over-confident posteriors:

    - the f_i distribution becomes bimodal (mass near 0 and near 1),
    - the overall ECP curve sits *below* the diagonal (negative bias),
    - across the *upper* credibility band the credible regions badly under-cover:
      ECP(c) < c, e.g. a "90% credible region" contains the truth far less than 90%
      of the time,
    - the KS test decisively rejects uniformity of the coverage fractions.
    """
    samples, theta_true = _toy_posteriors(n_obj=2000, n_samples=1000, dim=3, spread=0.3, seed=0)
    res = tarp_coverage(samples, theta_true, n_alpha=100, seed=11)

    # Overall the ECP curve lies below the diagonal => under-coverage detected.
    assert res.ecp_bias < -0.02, f"expected under-coverage bias, got {res.ecp_bias}"

    # The hallmark under-coverage band: high nominal credibility, low actual coverage.
    upper = (res.credibility >= 0.5) & (res.credibility <= 0.95)
    assert np.all(res.ecp[upper] < res.credibility[upper]), "upper band must under-cover"
    # A nominal ~90% credible region should contain the truth well under 90% of cases.
    ecp_at_90 = res.ecp[np.searchsorted(res.credibility, 0.9)]
    assert ecp_at_90 < 0.8, f"90% region over-covers ({ecp_at_90:.3f}); not over-confident"

    # Bimodal coverage fractions (paper's diagnostic histogram signature).
    f = res.coverage_fractions
    assert np.mean(f < 0.1) > 0.15
    assert np.mean(f > 0.9) > 0.15

    # Strong calibration failure.
    assert res.ks_statistic > 0.1
    assert res.ks_pvalue < 0.01
    assert not res.is_calibrated


def test_underconfident_lies_above_diagonal() -> None:
    """Under-confident (inflated) samples: ECP above the diagonal => over-coverage.

    For too-wide posteriors the coverage fractions concentrate near 0.5 (the truth is
    typically interior to the inflated cloud), so the ECP curve rises *above* the
    diagonal in the upper credibility band: a nominal "70% region" already contains
    the truth far more than 70% of the time.
    """
    samples, theta_true = _toy_posteriors(n_obj=2000, n_samples=1000, dim=3, spread=2.0, seed=0)
    res = tarp_coverage(samples, theta_true, n_alpha=100, seed=11)

    assert res.ecp_bias > 0.02, f"expected over-coverage bias, got {res.ecp_bias}"
    upper = (res.credibility >= 0.5) & (res.credibility <= 0.9)
    assert np.all(res.ecp[upper] > res.credibility[upper]), "upper band must over-cover"
    assert res.ks_statistic > 0.1
    assert not res.is_calibrated


def test_ecp_curve_monotone_and_bounded() -> None:
    """ECP is a (non-strict) monotone CDF bounded in [0, 1] with a sane endpoint."""
    samples, theta_true = _toy_posteriors(n_obj=300, n_samples=200, dim=2, spread=1.0, seed=3)
    res = tarp_coverage(samples, theta_true, n_alpha=50, seed=5)

    assert np.all(np.diff(res.ecp) >= -1e-12), "ECP must be non-decreasing"
    assert np.all(res.ecp >= 0.0) and np.all(res.ecp <= 1.0)
    # At credibility 1.0, every coverage fraction satisfies f_i <= 1 -> ECP == 1.
    assert res.ecp[-1] == pytest.approx(1.0)
    assert np.all(res.coverage_fractions >= 0.0)
    assert np.all(res.coverage_fractions <= 1.0)
    assert res.coverage_fractions.shape == (300,)


def test_delta_posterior_oracle() -> None:
    """Independent oracle: a posterior collapsed exactly onto the truth.

    Every sample equals the truth, so for any reference point the truth's distance to
    the reference equals each sample's distance. With a strict ``<`` region NO sample
    is strictly closer, hence ``f_i == 0`` for all objects and ECP(c) == 1 for every
    c >= 0 (because 0 <= c).
    """
    rng = np.random.default_rng(0)
    theta_true = rng.normal(size=(50, 2))
    samples = np.repeat(theta_true[:, None, :], 100, axis=1)  # delta posterior

    res = tarp_coverage(samples, theta_true, n_alpha=20, seed=1, standardize=False)

    assert np.allclose(res.coverage_fractions, 0.0)
    assert np.allclose(res.ecp, 1.0)


def test_one_dimensional_reference_oracle() -> None:
    """Deterministic oracle: recompute one object's f_i by hand using the same RNG.

    We reproduce the internal reference-point draw recipe exactly (uniform in the
    sample bounding box, ``numpy.random.default_rng(seed)``) and verify the reported
    coverage fraction matches a hand recomputation of
    ``mean(|sample - r| < |truth - r|)``.
    """
    # Two objects so the >=2 guard passes; we check object 0 against the oracle.
    samples = np.stack([np.arange(11, dtype=float), np.arange(11, dtype=float) + 100.0]).reshape(
        2, 11, 1
    )
    theta_true = np.array([[5.0], [105.0]])

    res = tarp_coverage(
        samples,
        theta_true,
        n_alpha=10,
        references="uniform",
        standardize=False,
        seed=42,
    )

    # Reproduce the internal reference draw: one uniform draw per object (object 0 first).
    rng = np.random.default_rng(42)
    pooled = samples.reshape(-1, 1)
    lo, hi = pooled.min(axis=0), pooled.max(axis=0)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    refs = lo + rng.random((2, 1)) * span
    r0 = refs[0, 0]
    d_true = abs(5.0 - r0)
    d_samples = np.abs(np.arange(11, dtype=float) - r0)
    expected_f0 = float(np.mean(d_samples < d_true))

    assert res.coverage_fractions[0] == pytest.approx(expected_f0)


def test_single_object_rejected() -> None:
    """A single object cannot estimate a coverage distribution -> ValueError."""
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(100, 3))  # (n_samples, dim) single object
    theta_true = rng.normal(size=(3,))  # (dim,)
    with pytest.raises(ValueError, match="2 objects"):
        tarp_coverage(samples, theta_true)


def test_input_validation() -> None:
    """Shape, finiteness and parameter validation raise informative ValueErrors."""
    rng = np.random.default_rng(0)
    good = rng.normal(size=(10, 50, 2))
    truth = rng.normal(size=(10, 2))

    with pytest.raises(ValueError, match="dim mismatch"):
        tarp_coverage(good, rng.normal(size=(10, 3)))
    with pytest.raises(ValueError, match="n_obj mismatch"):
        tarp_coverage(good, rng.normal(size=(9, 2)))
    with pytest.raises(ValueError, match="n_alpha"):
        tarp_coverage(good, truth, n_alpha=1)
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tarp_coverage(bad, truth)
    with pytest.raises(ValueError, match="references"):
        tarp_coverage(good, truth, references="cauchy")


def test_reproducible_with_seed() -> None:
    """Same seed => identical coverage fractions and ECP."""
    samples, theta_true = _toy_posteriors(n_obj=100, n_samples=80, dim=2, spread=1.0, seed=1)
    a = tarp_coverage(samples, theta_true, seed=123)
    b = tarp_coverage(samples, theta_true, seed=123)
    assert np.array_equal(a.coverage_fractions, b.coverage_fractions)
    assert np.array_equal(a.ecp, b.ecp)


def test_to_dict_is_serializable() -> None:
    """to_dict returns JSON-friendly types only."""
    import json

    samples, theta_true = _toy_posteriors(n_obj=50, n_samples=40, dim=2, spread=1.0, seed=2)
    res = tarp_coverage(samples, theta_true, n_alpha=10, seed=2)
    payload = res.to_dict()
    json.loads(json.dumps(payload))  # round-trips without error
    assert payload["n_obj"] == 50
    assert len(payload["ecp"]) == 10

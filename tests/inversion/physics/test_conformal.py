"""Tests for split conformal prediction + CQR (cflibs.inversion.physics.conformal).

These tests use INDEPENDENT oracles (hand-computed order statistics, an explicit
empirical-coverage Monte Carlo over fresh synthetic draws) — never the
production formula re-expressed.

Acceptance tests (from the candidate spec):

1. Finite-sample marginal coverage: on a synthetic calibration+test split from a
   known noise model, empirical test coverage is >= 1 - alpha for
   alpha in {0.1, 0.2} (averaged over many independent splits, the conformal
   guarantee 1 - alpha <= coverage <= 1 - alpha + 1/(n+1)).
2. CQR central result: under heteroscedastic noise, conformalized quantile
   intervals are no wider (on average) than the constant-width split-CP band at
   matched coverage.

References
----------
- Lei et al., JASA 2018 (split conformal).
- Romano, Patterson & Candes, NeurIPS 2019 (CQR).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cflibs.inversion.physics.conformal import (
    ConformalInterval,
    conformal_interval,
    conformal_quantile_level,
    conformal_rank,
    conformalize_cqr,
    conformalize_split,
    cqr_calibrate,
    cqr_conformity_scores,
    cqr_interval,
    split_conformal,
)

# --------------------------------------------------------------------------- #
# Unit-level: quantile level and threshold against hand-computed oracles
# --------------------------------------------------------------------------- #


def test_quantile_level_matches_ceil_formula():
    # ceil((1-alpha)(n+1))/n, hand-computed.
    # n=9, alpha=0.1: ceil(0.9*10)/9 = 9/9 = 1.0
    assert conformal_quantile_level(9, 0.1) == pytest.approx(1.0)
    # n=19, alpha=0.1: ceil(0.9*20)/19 = 18/19
    assert conformal_quantile_level(19, 0.1) == pytest.approx(18.0 / 19.0)
    # n=100, alpha=0.2: ceil(0.8*101)/100 = ceil(80.8)/100 = 81/100
    assert conformal_quantile_level(100, 0.2) == pytest.approx(0.81)


def test_quantile_level_clipped_to_one_when_n_too_small():
    # n=4, alpha=0.1: ceil(0.9*5)/4 = ceil(4.5)/4 = 5/4 -> clipped to 1.0
    assert conformal_quantile_level(4, 0.1) == pytest.approx(1.0)


def test_conformal_rank_matches_ceil_formula():
    # k = ceil((1-alpha)(n+1)), hand-computed.
    assert conformal_rank(10, 0.2) == 9  # ceil(0.8*11)=ceil(8.8)=9
    assert conformal_rank(10, 0.1) == 10  # ceil(0.9*11)=ceil(9.9)=10
    assert conformal_rank(19, 0.1) == 18  # ceil(0.9*20)=18
    # n too small: rank exceeds n -> signalled by k = n+1 region.
    assert conformal_rank(4, 0.1) == 5  # ceil(0.9*5)=ceil(4.5)=5 > 4


def test_split_conformal_picks_correct_order_statistic():
    # Scores 1..10. The conformal threshold is the k-th SMALLEST score,
    # k = ceil((1-alpha)(n+1)).
    scores = np.arange(1.0, 11.0)
    # alpha=0.2 -> k=ceil(0.8*11)=9 -> 9th smallest = 9.
    assert split_conformal(scores, 0.2) == pytest.approx(9.0)
    # alpha=0.1 -> k=ceil(0.9*11)=10 -> 10th smallest = 10.
    assert split_conformal(scores, 0.1) == pytest.approx(10.0)
    # Shuffled input must give the same order statistic (rank, not position).
    shuffled = np.random.default_rng(0).permutation(scores)
    assert split_conformal(shuffled, 0.2) == pytest.approx(9.0)


def test_split_conformal_unbounded_when_n_too_small():
    # n=4, alpha=0.1 requires the (5/4)-quantile -> not certifiable -> +inf.
    assert split_conformal([0.1, 0.2, 0.3, 0.4], 0.1) == np.inf


def test_conformal_interval_symmetric_and_shape_preserving():
    point = np.array([0.2, 0.5, 0.8])
    lo, hi = conformal_interval(point, 0.1)
    np.testing.assert_allclose(lo, point - 0.1)
    np.testing.assert_allclose(hi, point + 0.1)
    assert lo.shape == point.shape


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        split_conformal([1.0, 2.0, 3.0], 0.0)
    with pytest.raises(ValueError):
        split_conformal([1.0, 2.0, 3.0], 1.0)


def test_negative_q_hat_rejected():
    with pytest.raises(ValueError):
        conformal_interval(0.5, -0.1)


# --------------------------------------------------------------------------- #
# CQR conformity score: independent oracle
# --------------------------------------------------------------------------- #


def test_cqr_conformity_score_sign_and_value():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    y = np.array([0.5, 1.3, -0.2])  # inside, above by 0.3, below by 0.2
    e = cqr_conformity_scores(lo, hi, y)
    # inside: max(0-0.5, 0.5-1) = max(-0.5,-0.5) = -0.5  (negative)
    # above:  max(0-1.3, 1.3-1) = max(-1.3, 0.3) = 0.3   (positive)
    # below:  max(0-(-0.2), -0.2-1) = max(0.2,-1.2) = 0.2 (positive)
    np.testing.assert_allclose(e, [-0.5, 0.3, 0.2])


def test_cqr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cqr_conformity_scores([0.0, 0.0], [1.0], [0.5, 0.5])


def test_cqr_correction_can_shrink_overconservative_interval():
    # Base interval is far too wide ([-10, 10]); truths tightly around 0.
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=200)
    lo = np.full_like(y, -10.0)
    hi = np.full_like(y, 10.0)
    e = cqr_calibrate(lo, hi, y, 0.1)
    # Conformity scores are all strongly negative -> E < 0 -> interval shrinks.
    assert e < 0.0
    t_lo, t_hi = cqr_interval(-10.0, 10.0, e)
    assert (t_hi - t_lo) < 20.0


# --------------------------------------------------------------------------- #
# ACCEPTANCE TEST 1: finite-sample marginal coverage >= 1 - alpha
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_split_cp_finite_sample_coverage(alpha):
    """Split-CP empirical coverage >= 1 - alpha (averaged over fresh splits).

    Known noise model: a constant CF-LIBS point estimate of a scalar
    concentration with additive Gaussian measurement noise. Nonconformity score
    is the absolute residual. We Monte-Carlo over many independent
    calibration/test draws; the conformal guarantee is on the *expected*
    coverage, 1 - alpha <= E[cov] <= 1 - alpha + 1/(n+1).
    """
    rng = np.random.default_rng(1234)
    point = 0.42  # fixed point estimate (additive: never altered)
    sigma = 0.05
    n_cal = 200
    n_test = 200
    n_trials = 400

    coverages = np.empty(n_trials)
    for t in range(n_trials):
        cal_residuals = rng.normal(0.0, sigma, size=n_cal)
        cal_scores = np.abs(cal_residuals)  # |y - point| with truth = point + noise
        q_hat = split_conformal(cal_scores, alpha)
        lo, hi = conformal_interval(point, q_hat)
        test_truth = point + rng.normal(0.0, sigma, size=n_test)
        covered = np.mean((test_truth >= lo) & (test_truth <= hi))
        coverages[t] = covered

    mean_cov = float(np.mean(coverages))
    # Finite-sample guarantee: mean coverage at or above nominal (small MC slack).
    assert (
        mean_cov >= (1.0 - alpha) - 0.01
    ), f"split-CP mean coverage {mean_cov:.4f} < nominal {1 - alpha:.2f}"
    # Upper bound 1 - alpha + 1/(n+1): the band is not absurdly over-conservative.
    assert mean_cov <= (1.0 - alpha) + 1.0 / (n_cal + 1) + 0.02


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_cqr_finite_sample_coverage_heteroscedastic(alpha):
    """CQR empirical coverage >= 1 - alpha under heteroscedastic noise.

    Noise model: y = f(x) + sigma(x) * eps, sigma(x) increasing in x. Base
    conditional quantiles are the *Gaussian-oracle* quantiles f(x) +/- z * s(x)
    where s(x) is a deliberately MIS-scaled estimate of sigma(x) (so the base
    quantiles are imperfect and CQR must correct them).
    """
    rng = np.random.default_rng(99)
    n_cal = 250
    n_test = 250
    n_trials = 300
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    def sample(n):
        x = rng.uniform(0.0, 1.0, size=n)
        sigma = 0.02 + 0.20 * x  # heteroscedastic
        y = 0.5 + 0.0 * x + sigma * rng.standard_normal(n)
        return x, y, sigma

    def base_quantiles(x):
        # Deliberately mis-scaled spread estimate (0.7x true), so base intervals
        # under-cover and CQR must inflate them to restore the guarantee.
        s_hat = 0.7 * (0.02 + 0.20 * x)
        lo = 0.5 - z * s_hat
        hi = 0.5 + z * s_hat
        return lo, hi

    coverages = np.empty(n_trials)
    for t in range(n_trials):
        xc, yc, _ = sample(n_cal)
        clo, chi = base_quantiles(xc)
        e = cqr_calibrate(clo, chi, yc, alpha)
        xt, yt, _ = sample(n_test)
        tlo, thi = base_quantiles(xt)
        ilo, ihi = cqr_interval(tlo, thi, e)
        coverages[t] = np.mean((yt >= ilo) & (yt <= ihi))

    mean_cov = float(np.mean(coverages))
    assert (
        mean_cov >= (1.0 - alpha) - 0.01
    ), f"CQR mean coverage {mean_cov:.4f} < nominal {1 - alpha:.2f}"


# --------------------------------------------------------------------------- #
# ACCEPTANCE TEST 2: CQR no wider than constant-width split-CP under
# heteroscedastic noise (central CQR result, Romano et al. 2019)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_cqr_no_wider_than_split_cp_heteroscedastic(alpha):
    """Under heteroscedastic noise, mean CQR width <= mean split-CP width.

    Both methods are calibrated to the SAME alpha on the SAME calibration data,
    and both attain >= 1 - alpha coverage. The constant-width split-CP band must
    be wide enough for the noisiest region, so it over-covers the quiet region
    and is wider on average than the locally-adaptive CQR band — the central
    result of Romano et al. (2019).
    """
    rng = np.random.default_rng(2024)
    n_cal = 600
    n_test = 4000
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    def sample(n):
        x = rng.uniform(0.0, 1.0, size=n)
        sigma = 0.01 + 0.30 * x  # strongly heteroscedastic
        y = 0.5 + sigma * rng.standard_normal(n)
        return x, y

    def base_quantiles(x):
        s_hat = 0.01 + 0.30 * x  # well-specified conditional spread
        return 0.5 - z * s_hat, 0.5 + z * s_hat

    xc, yc = sample(n_cal)

    # --- Split-CP: absolute residual to the constant mean prediction 0.5 ---
    split_q = split_conformal(np.abs(yc - 0.5), alpha)
    split_width = 2.0 * split_q  # constant-width band

    # --- CQR on the same calibration data ---
    clo, chi = base_quantiles(xc)
    e = cqr_calibrate(clo, chi, yc, alpha)

    # Evaluate widths and coverage on a fresh test set.
    xt, yt = sample(n_test)
    tlo, thi = base_quantiles(xt)
    cqr_lo, cqr_hi = cqr_interval(tlo, thi, e)
    cqr_widths = cqr_hi - cqr_lo

    split_lo = 0.5 - split_q
    split_hi = 0.5 + split_q
    split_cov = np.mean((yt >= split_lo) & (yt <= split_hi))
    cqr_cov = np.mean((yt >= cqr_lo) & (yt <= cqr_hi))

    # Both achieve (approximately) nominal coverage.
    assert split_cov >= (1.0 - alpha) - 0.02
    assert cqr_cov >= (1.0 - alpha) - 0.02

    mean_cqr_width = float(np.mean(cqr_widths))
    # CQR is no wider than the constant-width split-CP band (strictly narrower
    # here, with a small tolerance to avoid flakiness on the boundary).
    assert (
        mean_cqr_width <= split_width + 1e-9
    ), f"CQR mean width {mean_cqr_width:.4f} > split-CP width {split_width:.4f}"


# --------------------------------------------------------------------------- #
# End-to-end wrappers
# --------------------------------------------------------------------------- #


def test_conformalize_split_wrapper():
    rng = np.random.default_rng(7)
    scores = np.abs(rng.normal(0.0, 0.1, size=100))
    ci = conformalize_split(0.3, scores, 0.1)
    assert isinstance(ci, ConformalInterval)
    assert ci.method == "split"
    assert ci.n_cal == 100
    assert ci.q_hat > 0.0
    np.testing.assert_allclose(ci.lo, 0.3 - ci.q_hat)
    np.testing.assert_allclose(ci.hi, 0.3 + ci.q_hat)
    np.testing.assert_allclose(ci.width, 2.0 * ci.q_hat)


def test_conformalize_cqr_wrapper():
    rng = np.random.default_rng(8)
    y = rng.normal(0.0, 1.0, size=150)
    clo = np.full_like(y, -1.0)
    chi = np.full_like(y, 1.0)
    ci = conformalize_cqr(-1.0, 1.0, clo, chi, y, 0.2)
    assert ci.method == "cqr"
    assert ci.n_cal == 150
    assert np.isfinite(ci.q_hat)
    assert ci.hi >= ci.lo

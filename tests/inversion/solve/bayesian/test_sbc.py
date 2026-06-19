"""Simulation-Based Calibration (SBC) rank-uniformity harness tests.

These tests pin the additive SBC diagnostic added to
:mod:`cflibs.inversion.solve.bayesian.diagnostics`, which implements
Algorithm 1 of Talts, Betancourt, Simpson, Vehtari & Gelman, "Validating
Bayesian Inference Algorithms with Simulation-Based Calibration",
arXiv:1804.06788 (2018).

The self-consistency theorem of that paper states that, for an *exact*
posterior sampler, the rank of the prior draw among the ``L`` posterior draws
is uniformly distributed on ``{0, ..., L}``. The acceptance test exercises a
conjugate Gaussian model where the posterior is known in closed form:

* a Gaussian prior ``mu ~ N(mu_0, tau^2)`` with a Gaussian likelihood of known
  variance ``sigma^2`` over ``n_obs`` observations has the conjugate Gaussian
  posterior

      Var_post = 1 / (1/tau^2 + n_obs/sigma^2)
      mean_post = Var_post * (mu_0/tau^2 + sum(y)/sigma^2)

* Using the EXACT posterior gives uniform ranks (chi2 p > 0.05).
* A deliberately mis-scaled (too narrow) posterior produces a U-shaped rank
  histogram detected as non-uniform (p < 0.01).

The whole suite is synthetic, DB-free, and runs in well under a minute
(watchdog-safe).
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.solve.bayesian.diagnostics import (
    SBCResult,
    run_sbc,
    sbc_rank,
    sbc_ranks,
    sbc_uniformity_test,
)

# ---------------------------------------------------------------------------
# Conjugate Gaussian model (closed-form posterior) used as ground truth.
# ---------------------------------------------------------------------------

PRIOR_MEAN = 0.0
PRIOR_STD = 2.0
LIKELIHOOD_STD = 1.0
N_OBS = 5


def _prior_sample(rng: np.random.Generator) -> float:
    """Draw mu ~ N(PRIOR_MEAN, PRIOR_STD**2)."""
    return float(rng.normal(PRIOR_MEAN, PRIOR_STD))


def _simulate(theta: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate N_OBS observations y_i ~ N(mu, LIKELIHOOD_STD**2)."""
    return rng.normal(float(theta), LIKELIHOOD_STD, size=N_OBS)


def _posterior_params(data: np.ndarray, scale: float = 1.0) -> tuple[float, float]:
    """Closed-form conjugate-Gaussian posterior (mean, std).

    ``scale`` multiplies the posterior std: ``scale=1`` is exact,
    ``scale<1`` deliberately over-concentrates (too-narrow / overconfident).
    """
    prior_prec = 1.0 / PRIOR_STD**2
    like_prec = N_OBS / LIKELIHOOD_STD**2
    post_var = 1.0 / (prior_prec + like_prec)
    post_mean = post_var * (PRIOR_MEAN * prior_prec + np.sum(data) / LIKELIHOOD_STD**2)
    return float(post_mean), float(np.sqrt(post_var) * scale)


def _exact_posterior_sample(data: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_draws from the EXACT conjugate Gaussian posterior."""
    mean, std = _posterior_params(data, scale=1.0)
    return rng.normal(mean, std, size=n_draws)


def _narrow_posterior_sample(
    data: np.ndarray, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n_draws from a deliberately TOO-NARROW (overconfident) posterior."""
    mean, std = _posterior_params(data, scale=0.5)
    return rng.normal(mean, std, size=n_draws)


# ---------------------------------------------------------------------------
# Unit tests for the low-level rank primitives.
# ---------------------------------------------------------------------------


def test_sbc_rank_counts_strictly_less():
    """sbc_rank returns #{posterior draws < prior draw} (Talts et al. Eq. 2)."""
    posterior = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    # Three draws (-1, 0, 1) are strictly less than 1.5.
    assert sbc_rank(1.5, posterior) == 3
    # All draws less -> rank == L.
    assert sbc_rank(100.0, posterior) == len(posterior)
    # No draws less -> rank == 0.
    assert sbc_rank(-100.0, posterior) == 0


def test_sbc_rank_range_is_zero_to_L():
    """Ranks are bounded in {0, ..., L} for arbitrary inputs."""
    rng = np.random.default_rng(0)
    posterior = rng.normal(size=20)
    for prior in rng.normal(size=50):
        r = sbc_rank(float(prior), posterior, rng=rng)
        assert 0 <= r <= len(posterior)


def test_sbc_rank_tie_break_stays_in_range():
    """Exact ties are randomised but the rank stays in {0, ..., L}."""
    posterior = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    rng = np.random.default_rng(1)
    ranks = [sbc_rank(1.0, posterior, rng=rng) for _ in range(200)]
    assert all(0 <= r <= len(posterior) for r in ranks)
    # With three tied draws, the tie-break must produce more than one value.
    assert len(set(ranks)) > 1


def test_sbc_ranks_vectorised_matches_scalar():
    """Vectorised sbc_ranks agrees with the scalar sbc_rank (no ties)."""
    rng = np.random.default_rng(7)
    n_sims, n_draws = 30, 25
    prior = rng.normal(size=n_sims)
    post = rng.normal(size=(n_sims, n_draws))
    ranks = sbc_ranks(prior, post, param_names=["mu"])["mu"]
    expected = np.array([sbc_rank(prior[i], post[i]) for i in range(n_sims)])
    np.testing.assert_array_equal(ranks, expected)


def test_sbc_ranks_multiparam_shapes():
    """Multi-parameter ranks return one array per parameter, all (n_sims,)."""
    rng = np.random.default_rng(3)
    n_sims, n_draws, n_params = 40, 15, 3
    prior = rng.normal(size=(n_sims, n_params))
    post = rng.normal(size=(n_sims, n_draws, n_params))
    ranks = sbc_ranks(prior, post, param_names=["a", "b", "c"])
    assert set(ranks) == {"a", "b", "c"}
    for arr in ranks.values():
        assert arr.shape == (n_sims,)
        assert np.all((arr >= 0) & (arr <= n_draws))


def test_sbc_ranks_shape_validation():
    """Mismatched simulation / parameter counts raise ValueError."""
    rng = np.random.default_rng(0)
    prior = rng.normal(size=10)
    post = rng.normal(size=(9, 20))  # wrong n_sims
    with pytest.raises(ValueError, match="simulation-count mismatch"):
        sbc_ranks(prior, post)


# ---------------------------------------------------------------------------
# Uniformity test on synthetic rank distributions.
# ---------------------------------------------------------------------------


def test_uniformity_test_passes_for_uniform_ranks():
    """Genuinely uniform ranks give a large chi-square p-value."""
    rng = np.random.default_rng(11)
    n_sims, L = 4000, 99
    ranks = {"mu": rng.integers(0, L + 1, size=n_sims)}
    p_values, chi2, n_bins = sbc_uniformity_test(ranks, n_posterior_draws=L)
    assert n_bins >= 2
    assert p_values["mu"] > 0.05


def test_uniformity_test_flags_ushaped_ranks():
    """A U-shaped (mass at 0 and L) rank histogram is flagged non-uniform."""
    rng = np.random.default_rng(13)
    n_sims, L = 4000, 99
    # Concentrate mass at the extremes -> overconfident posterior signature.
    extremes = rng.choice([0, L], size=n_sims, p=[0.5, 0.5])
    p_values, _chi2, _bins = sbc_uniformity_test({"mu": extremes}, n_posterior_draws=L)
    assert p_values["mu"] < 0.01


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST: full SBC harness on the conjugate Gaussian model.
# ---------------------------------------------------------------------------


def test_acceptance_exact_posterior_is_uniform():
    """ACCEPTANCE: the EXACT conjugate posterior yields uniform ranks (p > 0.05).

    This is the positive half of the SBC self-consistency check from Talts
    et al. (arXiv:1804.06788): a correctly-calibrated sampler produces a flat
    rank histogram, so the chi-square uniformity test must NOT reject.
    """
    result = run_sbc(
        prior_sample_fn=_prior_sample,
        simulate_fn=_simulate,
        posterior_sample_fn=_exact_posterior_sample,
        n_sims=2000,
        n_posterior_draws=99,
        param_names=["mu"],
        seed=2024,
    )
    assert isinstance(result, SBCResult)
    assert result.n_sims == 2000
    assert result.n_posterior_draws == 99
    assert result.p_values["mu"] > 0.05, (
        "Exact posterior must yield uniform SBC ranks; "
        f"got p={result.p_values['mu']:.4f}, chi2={result.chi2['mu']:.2f}"
    )
    assert result.is_calibrated(alpha=0.05)["mu"] is True


def test_acceptance_narrow_posterior_is_ushaped_and_rejected():
    """ACCEPTANCE: a too-narrow posterior gives a U-shaped, non-uniform histogram.

    This is the negative half of the SBC check: an overconfident (too narrow)
    posterior pushes the prior draw into the extreme ranks 0 and L, producing
    the characteristic bathtub histogram, which the chi-square test must reject
    (p < 0.01).
    """
    result = run_sbc(
        prior_sample_fn=_prior_sample,
        simulate_fn=_simulate,
        posterior_sample_fn=_narrow_posterior_sample,
        n_sims=2000,
        n_posterior_draws=99,
        param_names=["mu"],
        seed=2024,
    )
    assert result.p_values["mu"] < 0.01, (
        "Too-narrow posterior must be detected as non-uniform; "
        f"got p={result.p_values['mu']:.4g}, chi2={result.chi2['mu']:.2f}"
    )
    assert result.is_calibrated(alpha=0.05)["mu"] is False

    # Confirm the rejection is specifically the U-shape (extreme-rank excess),
    # not an arbitrary deviation: the outer bins must hold more than uniform.
    L = result.n_posterior_draws
    ranks = result.ranks["mu"]
    n_bins = result.n_bins
    edges = np.linspace(0, L + 1, n_bins + 1)
    counts, _ = np.histogram(ranks, bins=edges)
    uniform_expected = result.n_sims / n_bins
    outer_mass = counts[0] + counts[-1]
    assert outer_mass > 2.0 * uniform_expected, (
        "Overconfident posterior should pile rank mass at the extremes; "
        f"outer bins held {outer_mass} vs uniform {2 * uniform_expected:.0f}"
    )


def test_run_sbc_validates_arguments():
    """run_sbc rejects non-positive simulation / draw counts."""
    with pytest.raises(ValueError, match="n_sims must be positive"):
        run_sbc(
            _prior_sample,
            _simulate,
            _exact_posterior_sample,
            n_sims=0,
            n_posterior_draws=10,
        )
    with pytest.raises(ValueError, match="n_posterior_draws must be positive"):
        run_sbc(
            _prior_sample,
            _simulate,
            _exact_posterior_sample,
            n_sims=10,
            n_posterior_draws=0,
        )


def test_run_sbc_is_reproducible():
    """A fixed seed produces identical ranks across runs."""
    kwargs = dict(
        prior_sample_fn=_prior_sample,
        simulate_fn=_simulate,
        posterior_sample_fn=_exact_posterior_sample,
        n_sims=200,
        n_posterior_draws=49,
        param_names=["mu"],
        seed=99,
    )
    r1 = run_sbc(**kwargs)
    r2 = run_sbc(**kwargs)
    np.testing.assert_array_equal(r1.ranks["mu"], r2.ranks["mu"])
    assert r1.p_values["mu"] == r2.p_values["mu"]

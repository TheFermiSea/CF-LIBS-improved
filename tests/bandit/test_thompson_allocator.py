"""Tests for the Thompson-sampling allocator (T2.3).

Covers:
- Synthetic 6-arm bandit regret simulation (acceptance #1: ≥70%
  allocation to the two good arms).
- Determinism at fixed seed.
- Bootstrap-CI tightening vs. equal allocation
  (acceptance #3: ≥30% tighter).
- Edge cases: single-arm degenerate, all-arms-tied,
  single-observation arm, NaN/Inf rejection, out-of-range arm indices.
- ``round_robin_warmup_schedule`` invariants.
- ``posterior_summary`` shape and per-arm bookkeeping.

Run with::

    pytest tests/bandit/test_thompson_allocator.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.bandit import ThompsonAllocator
from cflibs.bandit.thompson_allocator import (
    bootstrap_ci_width,
    round_robin_warmup_schedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_arm_pull(
    arm_idx: int,
    rng: np.random.Generator,
    true_means: np.ndarray,
    true_sigma: float = 0.02,
) -> float:
    """Sample one observation of arm ``arm_idx``.

    Models d_A as ``Normal(true_means[arm], true_sigma)``.  ``d_A`` is
    bounded below by 0; we clip negative draws to a tiny epsilon so
    the simulated regime stays in the realistic range.
    """
    draw = rng.normal(loc=true_means[arm_idx], scale=true_sigma)
    return float(max(draw, 1e-6))


def _run_bandit_sweep(
    *,
    n_arms: int,
    n_iters: int,
    warmup_n: int,
    true_means: np.ndarray,
    true_sigma: float,
    allocator_seed: int,
    env_seed: int,
) -> tuple[ThompsonAllocator, list[int], dict[int, list[float]]]:
    """Replay the parameter_sweep.py warmup-then-bandit policy.

    Returns
    -------
    allocator : ThompsonAllocator
        The trained allocator (with full per-arm history).
    arm_history : list[int]
        The arm index pulled at each iteration (length == ``n_iters``).
    rewards_per_arm : dict[int, list[float]]
        Per-arm raw observations in pull-order.
    """
    allocator = ThompsonAllocator(
        n_arms=n_arms,
        lower_is_better=True,
        random_state=allocator_seed,
    )
    env_rng = np.random.default_rng(env_seed)

    warmup = round_robin_warmup_schedule(n_arms, warmup_n)
    arm_history: list[int] = []
    rewards_per_arm: dict[int, list[float]] = {i: [] for i in range(n_arms)}

    for i in range(n_iters):
        if i < len(warmup):
            arm = warmup[i]
        else:
            arm = allocator.select_arm()
        value = _simulate_arm_pull(arm, env_rng, true_means, true_sigma)
        allocator.update(arm, value)
        arm_history.append(arm)
        rewards_per_arm[arm].append(value)

    return allocator, arm_history, rewards_per_arm


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_synthetic_six_arm_70pct_to_top_two() -> None:
    """Acceptance #1: ≥70% of post-warmup iters to the two good arms.

    Setup: 6 arms, 2 have d_A ≈ 0.05 (good), 4 have d_A ≈ 0.20 (bad),
    warmup=2 obs/arm, total budget=48 iters.  Average across seeds to
    confirm the property holds in expectation rather than for one
    lucky env-seed.
    """
    n_arms = 6
    n_iters = 48
    warmup_n = 2
    true_means = np.array([0.05, 0.05, 0.20, 0.20, 0.20, 0.20])
    good_arms = {0, 1}

    good_fractions = []
    for env_seed in range(20):
        _, arm_history, _ = _run_bandit_sweep(
            n_arms=n_arms,
            n_iters=n_iters,
            warmup_n=warmup_n,
            true_means=true_means,
            true_sigma=0.02,
            allocator_seed=1234 + env_seed,
            env_seed=env_seed,
        )
        post_warmup = arm_history[n_arms * warmup_n :]
        assert len(post_warmup) == n_iters - n_arms * warmup_n == 36
        good_pulls = sum(1 for a in post_warmup if a in good_arms)
        good_fractions.append(good_pulls / len(post_warmup))

    mean_good_frac = float(np.mean(good_fractions))
    # Acceptance: ≥ 0.70.  Single-seed runs can dip below 0.70 by
    # chance, but the mean over 20 seeds should comfortably exceed
    # the threshold.
    assert mean_good_frac >= 0.70, (
        f"Bandit allocated only {mean_good_frac:.1%} of post-warmup pulls "
        f"to the top-2 arms; want ≥ 70%. Per-seed fractions: {good_fractions}"
    )


@pytest.mark.unit
def test_bootstrap_ci_tighter_than_equal_allocation() -> None:
    """Acceptance #3: best-arm CI ≥30% tighter than equal allocation.

    Equal allocation: 8 iters per arm = 8 observations on the best arm.
    Bandit allocation should yield significantly more observations on
    the best arm, so the percentile-bootstrap CI on its mean shrinks.
    Average over multiple env seeds to smooth the comparison.
    """
    n_arms = 6
    n_iters = 48
    warmup_n = 2
    true_means = np.array([0.05, 0.05, 0.20, 0.20, 0.20, 0.20])

    bandit_widths: list[float] = []
    equal_widths: list[float] = []
    n_seeds = 20
    for env_seed in range(n_seeds):
        # --- Bandit run ---
        _, _, b_rewards = _run_bandit_sweep(
            n_arms=n_arms,
            n_iters=n_iters,
            warmup_n=warmup_n,
            true_means=true_means,
            true_sigma=0.02,
            allocator_seed=4242 + env_seed,
            env_seed=env_seed,
        )
        # Pick the arm with the most pulls as the bandit's "best."
        best_arm_bandit = max(range(n_arms), key=lambda a: len(b_rewards[a]))
        w_bandit = bootstrap_ci_width(
            b_rewards[best_arm_bandit], n_boot=2000, random_state=env_seed
        )

        # --- Equal-allocation run ---
        eq_rng = np.random.default_rng(env_seed + 9999)
        eq_rewards: list[list[float]] = [[] for _ in range(n_arms)]
        per_arm = n_iters // n_arms
        for arm in range(n_arms):
            for _ in range(per_arm):
                eq_rewards[arm].append(
                    _simulate_arm_pull(arm, eq_rng, true_means, true_sigma=0.02)
                )
        # Best arm under equal allocation: lowest sample mean.
        best_arm_eq = min(range(n_arms), key=lambda a: float(np.mean(eq_rewards[a])))
        w_equal = bootstrap_ci_width(
            eq_rewards[best_arm_eq], n_boot=2000, random_state=env_seed
        )

        bandit_widths.append(w_bandit)
        equal_widths.append(w_equal)

    mean_bandit_w = float(np.mean(bandit_widths))
    mean_equal_w = float(np.mean(equal_widths))
    # Tighter CI ⇒ smaller width.  Require ≥ 30% reduction.
    rel_reduction = (mean_equal_w - mean_bandit_w) / mean_equal_w
    assert rel_reduction >= 0.30, (
        f"Bandit CI width {mean_bandit_w:.4f} vs equal {mean_equal_w:.4f}: "
        f"reduction = {rel_reduction:.1%}, want ≥ 30%."
    )


@pytest.mark.unit
def test_regret_converges_to_optimal() -> None:
    """Cumulative regret grows sub-linearly.

    Cumulative regret over the post-warmup phase should be much smaller
    than the worst-case (all bad-arm pulls) reference.  This is a
    weaker check than the 70% allocation test but it directly probes
    the "converges to optimal" property called out in the task spec.
    """
    n_arms = 6
    n_iters = 60
    warmup_n = 2
    true_means = np.array([0.05, 0.05, 0.20, 0.20, 0.20, 0.20])
    optimal_mean = float(true_means.min())
    worst_extra = float(true_means.max() - optimal_mean)

    regrets = []
    for env_seed in range(10):
        _, arm_history, _ = _run_bandit_sweep(
            n_arms=n_arms,
            n_iters=n_iters,
            warmup_n=warmup_n,
            true_means=true_means,
            true_sigma=0.02,
            allocator_seed=7000 + env_seed,
            env_seed=env_seed,
        )
        post_warmup = arm_history[n_arms * warmup_n :]
        cum_regret = sum(true_means[a] - optimal_mean for a in post_warmup)
        # Worst-case regret over the same horizon (all-bad-arm policy).
        worst_case = worst_extra * len(post_warmup)
        regrets.append(cum_regret / worst_case)

    mean_normalized_regret = float(np.mean(regrets))
    # We expect the bandit to be much better than the worst policy.
    assert mean_normalized_regret <= 0.30, (
        f"Mean normalized regret {mean_normalized_regret:.3f} > 0.30; "
        "bandit failed to concentrate on optimal arms."
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_determinism_at_fixed_seed() -> None:
    """Two allocators with the same seed produce the same arm sequence."""
    rewards = [0.05, 0.05, 0.20, 0.20, 0.20, 0.20, 0.05, 0.20, 0.05, 0.20]

    def _seq(seed: int) -> list[int]:
        alloc = ThompsonAllocator(n_arms=6, random_state=seed)
        warmup = round_robin_warmup_schedule(6, warmup_n=2)
        out: list[int] = []
        env_rng = np.random.default_rng(99)
        for i in range(20):
            arm = warmup[i] if i < len(warmup) else alloc.select_arm()
            value = max(
                env_rng.normal(rewards[arm % len(rewards)], 0.02), 1e-6
            )
            alloc.update(arm, value)
            out.append(arm)
        return out

    a = _seq(seed=12345)
    b = _seq(seed=12345)
    assert a == b, f"Allocator non-deterministic at fixed seed: {a} vs {b}"


@pytest.mark.unit
def test_different_seeds_produce_different_sequences() -> None:
    """Sanity: distinct seeds give distinct arm sequences post-warmup."""
    n_arms = 6
    warmup_n = 2

    def _post_warmup_seq(seed: int) -> list[int]:
        alloc = ThompsonAllocator(n_arms=n_arms, random_state=seed)
        warmup = round_robin_warmup_schedule(n_arms, warmup_n)
        env_rng = np.random.default_rng(0)
        seq: list[int] = []
        for i in range(30):
            arm = warmup[i] if i < len(warmup) else alloc.select_arm()
            value = max(env_rng.normal(0.1, 0.05), 1e-6)
            alloc.update(arm, value)
            if i >= len(warmup):
                seq.append(arm)
        return seq

    assert _post_warmup_seq(1) != _post_warmup_seq(2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_single_arm_degenerate() -> None:
    """A 1-arm allocator always returns arm 0."""
    alloc = ThompsonAllocator(n_arms=1, random_state=0)
    for _ in range(50):
        assert alloc.select_arm() == 0
    alloc.update(0, 0.1)
    assert alloc.select_arm() == 0


@pytest.mark.unit
def test_all_arms_tied_starts_uniform_then_separates() -> None:
    """With identical observations on every arm the posterior stays close.

    We don't require *exact* uniformity (Thompson sampling is
    randomized), but over many selections every arm should be picked
    a meaningful number of times.
    """
    alloc = ThompsonAllocator(n_arms=4, random_state=42)
    for arm in range(4):
        alloc.update(arm, 0.10)
        alloc.update(arm, 0.10)

    counts = np.zeros(4, dtype=int)
    for _ in range(2000):
        counts[alloc.select_arm()] += 1
    # Each arm should get at least 5% of pulls.
    assert all(c >= 100 for c in counts), (
        f"Tied arms got highly non-uniform pulls: {counts.tolist()}"
    )


@pytest.mark.unit
def test_single_observation_arm_does_not_crash() -> None:
    """An arm with n=1 has S=0 and must not produce NaN samples."""
    alloc = ThompsonAllocator(n_arms=3, random_state=0)
    alloc.update(0, 0.05)
    alloc.update(1, 0.20)
    # arm 2 has zero observations
    for _ in range(50):
        choice = alloc.select_arm()
        assert choice in (0, 1, 2)


@pytest.mark.unit
def test_update_rejects_nan_and_inf() -> None:
    alloc = ThompsonAllocator(n_arms=2, random_state=0)
    with pytest.raises(ValueError):
        alloc.update(0, float("nan"))
    with pytest.raises(ValueError):
        alloc.update(0, float("inf"))


@pytest.mark.unit
def test_update_rejects_out_of_range_arm() -> None:
    alloc = ThompsonAllocator(n_arms=3, random_state=0)
    with pytest.raises(IndexError):
        alloc.update(-1, 0.1)
    with pytest.raises(IndexError):
        alloc.update(3, 0.1)


@pytest.mark.unit
def test_constructor_validates_args() -> None:
    with pytest.raises(ValueError):
        ThompsonAllocator(n_arms=0)
    with pytest.raises(ValueError):
        ThompsonAllocator(n_arms=1, prior_alpha=0.0)
    with pytest.raises(ValueError):
        ThompsonAllocator(n_arms=1, prior_beta=-1.0)
    with pytest.raises(ValueError):
        ThompsonAllocator(n_arms=1, prior_kappa=0.0)


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_posterior_summary_shape_and_fields() -> None:
    alloc = ThompsonAllocator(n_arms=3, random_state=0)
    alloc.update(0, 0.05)
    alloc.update(0, 0.06)
    alloc.update(1, 0.20)
    # Skip prob_best for speed in the structural test.
    summary = alloc.posterior_summary(prob_best_samples=0)
    assert len(summary) == 3
    for i, s in enumerate(summary):
        assert s["arm"] == i
        assert "n_pulls" in s
        assert "posterior_mean" in s
        assert "posterior_var" in s
        assert "observed_mean" in s
        assert "observed_var" in s
        assert "prob_best" in s
    assert summary[0]["n_pulls"] == 2
    assert summary[1]["n_pulls"] == 1
    assert summary[2]["n_pulls"] == 0
    # The arm with the better mean should pull its posterior mean lower
    # than the prior μ₀ = 0.1; arm 1 (single obs at 0.20) is pulled up.
    assert summary[0]["posterior_mean"] < 0.10
    assert summary[1]["posterior_mean"] > 0.10


@pytest.mark.unit
def test_posterior_summary_prob_best_sums_to_one() -> None:
    alloc = ThompsonAllocator(n_arms=4, random_state=7)
    alloc.update(0, 0.04)
    alloc.update(0, 0.05)
    alloc.update(1, 0.20)
    alloc.update(2, 0.22)
    alloc.update(3, 0.19)
    summary = alloc.posterior_summary(prob_best_samples=512)
    total = sum(s["prob_best"] for s in summary)
    assert abs(total - 1.0) < 1e-9, f"prob_best sums to {total}, expected 1.0"
    # The clearly-best arm should dominate.
    best = max(summary, key=lambda s: s["prob_best"])
    assert best["arm"] == 0


# ---------------------------------------------------------------------------
# Warmup schedule
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_round_robin_warmup_schedule() -> None:
    assert round_robin_warmup_schedule(3, 2) == [0, 1, 2, 0, 1, 2]
    assert round_robin_warmup_schedule(4, 0) == []
    assert round_robin_warmup_schedule(2, 3) == [0, 1, 0, 1, 0, 1]


@pytest.mark.unit
def test_round_robin_warmup_schedule_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        round_robin_warmup_schedule(0, 1)
    with pytest.raises(ValueError):
        round_robin_warmup_schedule(2, -1)


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_ci_width_shrinks_with_n() -> None:
    rng = np.random.default_rng(0)
    small = rng.normal(0.1, 0.02, size=8)
    large = rng.normal(0.1, 0.02, size=40)
    w_small = bootstrap_ci_width(small, n_boot=1000, random_state=0)
    w_large = bootstrap_ci_width(large, n_boot=1000, random_state=0)
    assert w_large < w_small, f"CI width did not shrink: {w_large} vs {w_small}"


@pytest.mark.unit
def test_bootstrap_ci_width_handles_tiny_samples() -> None:
    assert np.isnan(bootstrap_ci_width([], random_state=0))
    assert np.isnan(bootstrap_ci_width([0.1], random_state=0))


# ---------------------------------------------------------------------------
# n_pulls / history accessors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_n_pulls_and_history_accessors() -> None:
    alloc = ThompsonAllocator(n_arms=3, random_state=0)
    alloc.update(0, 0.05)
    alloc.update(2, 0.20)
    alloc.update(0, 0.06)
    assert alloc.n_pulls() == [2, 0, 1]
    assert alloc.n_pulls(0) == 2
    assert alloc.history(0) == [0.05, 0.06]
    assert alloc.history(1) == []
    with pytest.raises(IndexError):
        alloc.history(5)
    with pytest.raises(IndexError):
        alloc.n_pulls(7)


@pytest.mark.unit
def test_lower_is_better_flag_inverts_selection() -> None:
    """With lower_is_better=False, the high-reward arm should win."""
    alloc = ThompsonAllocator(n_arms=3, lower_is_better=False, random_state=0)
    # arm 2 is consistently *higher*
    for _ in range(8):
        alloc.update(0, 0.05)
        alloc.update(1, 0.10)
        alloc.update(2, 0.50)
    counts = np.zeros(3, dtype=int)
    for _ in range(1000):
        counts[alloc.select_arm()] += 1
    assert int(np.argmax(counts)) == 2

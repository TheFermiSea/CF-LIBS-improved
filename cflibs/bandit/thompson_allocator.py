"""Thompson-sampling allocator with a Gaussian–Normal-Inverse-Gamma model.

This module implements a small, dependency-light Thompson-sampling
allocator used by :mod:`scripts.parameter_sweep` to redistribute its
iteration budget across cells (arms) that produce continuous d_A
(Aitchison composition distance) observations.

Design summary
--------------

Each arm has a Normal–Inverse-Gamma (NIG) prior on its mean and
variance::

    σ²  ~ InverseGamma(α₀, β₀)
    μ   ~ Normal(μ₀, σ²/κ₀)

After ``n`` observations of an arm with running mean ``x̄`` and
sum-of-squared-deviations ``S = Σᵢ(xᵢ − x̄)²``, the posterior is::

    κₙ = κ₀ + n
    μₙ = (κ₀·μ₀ + n·x̄) / κₙ
    αₙ = α₀ + n/2
    βₙ = β₀ + S/2 + κ₀·n·(x̄ − μ₀)² / (2·κₙ)

A Thompson sample for the arm's mean is drawn by sampling
``σ² ~ InverseGamma(αₙ, βₙ)`` and then ``μ ~ Normal(μₙ, sqrt(σ²/κₙ))``.

``select_arm()`` draws one such sample per arm and returns the
``argmin`` (when ``lower_is_better=True``) or ``argmax`` otherwise.
Ties are broken by arm index.

References
----------
- Gelman et al., *Bayesian Data Analysis* 3rd ed., §3.3 (NIG conjugacy).
- Russo et al., *A Tutorial on Thompson Sampling*, 2018.
- Chapelle & Li, *An Empirical Evaluation of Thompson Sampling*, NIPS 2011.

See also
--------
``docs/bandit-allocator-consultation.md`` for the design rationale and
the cross-model (Codex + Gemini) consultation that informed the
prior-choice and tie-breaking decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats as _scipy_stats

# Floor on β to avoid invgamma.rvs collapsing to 0 when an arm has had
# only identical observations.  1e-12 keeps the posterior numerically
# well-conditioned without affecting any reasonable d_A scale.
_BETA_FLOOR = 1e-12

# Default prior — chosen so the prior median of σ² is ≈ (0.1)², which
# matches the d_A scatter seen on Vrabel-class smoke runs.  α₀ = 1
# makes the inverse-gamma proper but very broad; κ₀ = 0.01 keeps the
# prior on μ weakly informative so warmup data dominates by n ≈ 4.
_DEFAULT_MU0 = 0.1
_DEFAULT_KAPPA0 = 0.01
_DEFAULT_ALPHA0 = 1.0
_DEFAULT_BETA0 = 0.01  # ⇒ E[σ²] = β₀/(α₀−1); proper only for α₀>1, so we use median


@dataclass
class _ArmState:
    """Per-arm sufficient statistics for the Gaussian–NIG model.

    ``mean`` and ``m2`` are maintained with Welford's online algorithm
    so that ``S = m2`` is the running sum-of-squared-deviations.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Σ(xᵢ − x̄)² accumulated incrementally
    last_value: Optional[float] = None
    history: List[float] = field(default_factory=list)


class ThompsonAllocator:
    """Thompson-sampling allocator over ``n_arms`` continuous-reward arms.

    Parameters
    ----------
    n_arms : int
        Number of arms (cells) to allocate over. Must be >= 1.
    prior_alpha : float, default 1.0
        Shape parameter ``α₀`` of the inverse-gamma prior on variance.
    prior_beta : float, default 0.01
        Scale parameter ``β₀`` of the inverse-gamma prior on variance.
    prior_mu : float, default 0.1
        Prior mean ``μ₀`` for each arm.
    prior_kappa : float, default 0.01
        Prior pseudo-count ``κ₀`` for the mean.
    lower_is_better : bool, default True
        If True, ``select_arm()`` returns the arm with the *smallest*
        Thompson-sampled mean (appropriate for d_A and other "loss"
        metrics).  If False, returns the largest.
    random_state : int | np.random.Generator | None
        Seed or generator for reproducibility.  ``None`` uses fresh
        OS entropy; an integer seeds a fresh ``np.random.default_rng``;
        a Generator is used as-is.

    Notes
    -----
    The allocator does NOT enforce a warmup schedule itself — callers
    (e.g. ``parameter_sweep.py``) are responsible for round-robin
    warmup pulls before delegating selection to ``select_arm()``.  This
    keeps the allocator pure and easy to test in isolation, and makes
    the warmup schedule explicit in the calling code.
    """

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = _DEFAULT_ALPHA0,
        prior_beta: float = _DEFAULT_BETA0,
        prior_mu: float = _DEFAULT_MU0,
        prior_kappa: float = _DEFAULT_KAPPA0,
        lower_is_better: bool = True,
        random_state: Optional[int | np.random.Generator] = None,
    ) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if prior_alpha <= 0:
            raise ValueError(f"prior_alpha must be > 0, got {prior_alpha}")
        if prior_beta <= 0:
            raise ValueError(f"prior_beta must be > 0, got {prior_beta}")
        if prior_kappa <= 0:
            raise ValueError(f"prior_kappa must be > 0, got {prior_kappa}")

        self.n_arms = int(n_arms)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.prior_mu = float(prior_mu)
        self.prior_kappa = float(prior_kappa)
        self.lower_is_better = bool(lower_is_better)
        self._arms: List[_ArmState] = [_ArmState() for _ in range(self.n_arms)]

        if random_state is None:
            self._rng = np.random.default_rng()
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(int(random_state))

    # ------------------------------------------------------------------
    # Posterior parameters
    # ------------------------------------------------------------------

    def _posterior_params(self, arm: _ArmState) -> tuple[float, float, float, float]:
        """Return ``(μₙ, κₙ, αₙ, βₙ)`` for the given arm."""
        n = arm.n
        if n == 0:
            return (
                self.prior_mu,
                self.prior_kappa,
                self.prior_alpha,
                self.prior_beta,
            )
        kappa_n = self.prior_kappa + n
        mu_n = (self.prior_kappa * self.prior_mu + n * arm.mean) / kappa_n
        alpha_n = self.prior_alpha + 0.5 * n
        delta = arm.mean - self.prior_mu
        beta_n = (
            self.prior_beta + 0.5 * arm.m2 + 0.5 * self.prior_kappa * n * delta * delta / kappa_n
        )
        beta_n = max(beta_n, _BETA_FLOOR)
        return mu_n, kappa_n, alpha_n, beta_n

    def _sample_mean(self, arm: _ArmState) -> float:
        """Draw one Thompson sample of the arm's mean from the posterior."""
        mu_n, kappa_n, alpha_n, beta_n = self._posterior_params(arm)
        # σ² ~ InverseGamma(αₙ, βₙ); scipy uses `scale=β`.
        # Generate via gamma reciprocal to avoid scipy.stats's global
        # random_state plumbing (we want exact control via self._rng).
        gamma_draw = self._rng.gamma(shape=alpha_n, scale=1.0 / beta_n)
        # Floor σ² so very-tiny gamma draws don't underflow Normal scale.
        sigma2 = 1.0 / max(gamma_draw, 1e-300)
        scale = np.sqrt(sigma2 / kappa_n)
        return float(self._rng.normal(loc=mu_n, scale=scale))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """Return the index of the arm with the best Thompson sample.

        For ``lower_is_better=True`` this is the ``argmin`` of the
        per-arm samples; otherwise it is the ``argmax``.  Ties are
        broken by arm index (numpy's default argmin/argmax behaviour).
        """
        samples = np.array([self._sample_mean(arm) for arm in self._arms])
        if self.lower_is_better:
            return int(np.argmin(samples))
        return int(np.argmax(samples))

    def update(self, arm_idx: int, value: float) -> None:
        """Update arm ``arm_idx`` with a new observation ``value``.

        Uses Welford's online algorithm so that ``arm.m2`` is the
        running sum-of-squared-deviations (= ``S`` in the NIG update).
        """
        if not 0 <= arm_idx < self.n_arms:
            raise IndexError(f"arm_idx={arm_idx} out of range for n_arms={self.n_arms}")
        value_f = float(value)
        if not np.isfinite(value_f):
            raise ValueError(f"update value must be finite, got {value_f} for arm {arm_idx}")
        arm = self._arms[arm_idx]
        arm.n += 1
        delta = value_f - arm.mean
        arm.mean += delta / arm.n
        delta2 = value_f - arm.mean
        arm.m2 += delta * delta2
        arm.last_value = value_f
        arm.history.append(value_f)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def posterior_summary(self, prob_best_samples: int = 256) -> List[dict]:
        """Return per-arm posterior summary statistics.

        Parameters
        ----------
        prob_best_samples : int, default 256
            Number of Monte-Carlo draws used to estimate ``prob_best``,
            i.e. the posterior probability that the arm is optimal.
            ``prob_best_samples=0`` skips the (relatively expensive)
            ``prob_best`` estimate and returns it as ``None``.

        Returns
        -------
        list of dict
            One entry per arm with keys ``arm``, ``n_pulls``,
            ``posterior_mean``, ``posterior_var``, ``observed_mean``,
            ``observed_var``, ``prob_best``.
        """
        summaries: List[dict] = []
        for idx, arm in enumerate(self._arms):
            mu_n, kappa_n, alpha_n, beta_n = self._posterior_params(arm)
            # Mode of inverse-gamma is β / (α + 1); fall back to mean
            # β / (α − 1) when α > 1 since that is more familiar.
            if alpha_n > 1.0:
                post_var_mean = beta_n / (alpha_n - 1.0)
            else:
                post_var_mean = float("nan")
            # Unbiased sample variance for the observed data.
            if arm.n >= 2:
                obs_var = arm.m2 / (arm.n - 1)
            else:
                obs_var = float("nan")
            summaries.append(
                {
                    "arm": idx,
                    "n_pulls": arm.n,
                    "posterior_mean": float(mu_n),
                    "posterior_var": float(post_var_mean),
                    "observed_mean": float(arm.mean) if arm.n > 0 else float("nan"),
                    "observed_var": float(obs_var),
                    "prob_best": None,
                }
            )

        if prob_best_samples > 0 and self.n_arms > 0:
            counts = np.zeros(self.n_arms, dtype=np.int64)
            # Use a *separate* short-lived generator so prob_best
            # estimation does not consume self._rng entropy (and so
            # the allocator's selection sequence stays deterministic
            # whether or not summaries are inspected).
            rng = np.random.default_rng(self._rng.integers(0, 2**31 - 1))
            for _ in range(int(prob_best_samples)):
                samples = np.empty(self.n_arms)
                for i, arm in enumerate(self._arms):
                    mu_n, kappa_n, alpha_n, beta_n = self._posterior_params(arm)
                    sigma2 = 1.0 / max(rng.gamma(alpha_n, 1.0 / beta_n), 1e-300)
                    samples[i] = rng.normal(mu_n, np.sqrt(sigma2 / kappa_n))
                if self.lower_is_better:
                    counts[int(np.argmin(samples))] += 1
                else:
                    counts[int(np.argmax(samples))] += 1
            probs = counts / float(prob_best_samples)
            for i, summary in enumerate(summaries):
                summary["prob_best"] = float(probs[i])

        return summaries

    def n_pulls(self, arm_idx: Optional[int] = None) -> int | List[int]:
        """Pull counts: total per-arm list, or for a single arm if specified."""
        if arm_idx is None:
            return [arm.n for arm in self._arms]
        if not 0 <= arm_idx < self.n_arms:
            raise IndexError(f"arm_idx={arm_idx} out of range for n_arms={self.n_arms}")
        return self._arms[arm_idx].n

    def history(self, arm_idx: int) -> List[float]:
        """Return the full observation history for an arm."""
        if not 0 <= arm_idx < self.n_arms:
            raise IndexError(f"arm_idx={arm_idx} out of range for n_arms={self.n_arms}")
        return list(self._arms[arm_idx].history)


def round_robin_warmup_schedule(n_arms: int, warmup_n: int) -> List[int]:
    """Return the warmup arm-pull order: ``[0,1,...,n-1, 0,1,...,n-1, ...]``.

    Deterministic and independent of ``random_state`` so that a sweep
    with ``--bandit warmup_n`` always starts with the same first
    ``n_arms × warmup_n`` arm assignments.

    Parameters
    ----------
    n_arms : int
        Number of arms.
    warmup_n : int
        Warmup pulls per arm. ``warmup_n=0`` returns an empty list
        (i.e. no warmup; every pull is bandit-driven from the start —
        in practice you almost always want at least ``warmup_n=1``).
    """
    if n_arms < 1:
        raise ValueError(f"n_arms must be >= 1, got {n_arms}")
    if warmup_n < 0:
        raise ValueError(f"warmup_n must be >= 0, got {warmup_n}")
    schedule: List[int] = []
    for _ in range(warmup_n):
        schedule.extend(range(n_arms))
    return schedule


def bootstrap_ci_width(
    values: Sequence[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> float:
    """Width of the percentile-bootstrap CI on the mean of ``values``.

    Convenience helper for the acceptance criterion that the
    bandit-best-arm CI is ≥ 30% tighter than equal-allocation would
    have produced.  Returns ``nan`` for ``len(values) < 2``.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    rng = np.random.default_rng(random_state)
    boot_means = np.empty(int(n_boot))
    n = arr.size
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(arr[idx].mean())
    alpha = 1.0 - float(confidence)
    lo = np.quantile(boot_means, alpha / 2.0)
    hi = np.quantile(boot_means, 1.0 - alpha / 2.0)
    return float(hi - lo)


__all__ = [
    "ThompsonAllocator",
    "bootstrap_ci_width",
    "round_robin_warmup_schedule",
]


# Silence the unused scipy import when stats is only needed as a
# numerical reference; importing scipy.stats at module load gives
# users a clear ImportError if the env is missing scipy.
_ = _scipy_stats

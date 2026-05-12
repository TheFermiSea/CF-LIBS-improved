# Thompson-Sampling Bandit Allocator — Design Consultation (T2.3)

**Date:** 2026-05-12
**Issue:** `CF-LIBS-improved-7yfp`
**Parent epic:** `CF-LIBS-improved-7lht`
**Branch:** `feat/bandit-allocator`

## Motivation

`scripts/parameter_sweep.py` (T1.1, PR #129) runs a single config N times.
The next natural step is a *multi-cell* sweep: M candidate configs × K
iterations each, total budget B = M·K. At equal allocation, B/M of the
budget goes to every cell — including obviously-bad cells whose d_A
(Aitchison composition distance) is already clearly worse than the
front-runners after the first couple of pulls. Treating each cell as
the arm of a multi-armed bandit lets the allocator concentrate budget
on the most promising cells, tightening the bootstrap CI on the best
cell without paying any additional compute.

Acceptance: a 6-arm × 48-iter sweep where 2 arms have d_A ≈ 0.05 and
4 have d_A ≈ 0.20 must allocate ≥ 70% of post-warmup iters to the two
good arms, and the bootstrap CI on the best arm must be ≥ 30% tighter
than equal allocation.

## Cross-model consultation

Both models were queried via the CLIAPIProxy (localhost:8317) on
2026-05-12.

- **gpt-5.4-mini** (Codex-class, full review)
- **gemini-3.1-flash-lite-preview** (Gemini Flash)

Both converged on the same recommendation:

> **Use Gaussian likelihood with a conjugate Normal–Inverse-Gamma
> (NIG) prior, sample with Thompson sampling, round-robin warmup,
> watch tiny-sample variance underestimation.**

### Consensus design

#### 1. Likelihood / prior choice — NIG over Beta or Student-t

| Option | Verdict |
|---|---|
| **Beta-Bernoulli** | Rejected: d_A is continuous on [0, ~0.5], not binary. Thresholding wastes information and degrades ranking. |
| **Gaussian + Normal–Inverse-Gamma** | **Chosen.** Closed-form conjugate update for both mean and variance. Posterior predictive is Student-t with `2α` degrees of freedom, which is naturally heavy-tailed at small `n_pulls` — exactly what we want for cold-start. |
| **Explicit Student-t likelihood** | Rejected for first cut: requires MCMC or variational inference. The Gaussian-NIG posterior predictive *is already* Student-t, which we get for free. |

The NIG prior parameters:

- `μ₀` (prior mean of arm mean): centered at a neutral value (0.1, midway
  between "good" and "bad" d_A scales seen on Vrabel).
- `κ₀` (prior pseudo-count for mean): **0.01** — weakly informative.
- `α₀` (shape of inverse-gamma on variance): **1.0** — gives a proper
  but very broad prior on variance.
- `β₀` (scale of inverse-gamma): chosen so prior median variance
  ≈ `(0.1)²`, giving the model a baseline expectation of typical d_A
  scatter without overcommitting.

Closed-form posterior update for an arm with `n` observations of mean
`x̄` and sum-of-squared-deviations `S = Σ(xᵢ − x̄)²`:

```
κₙ = κ₀ + n
μₙ = (κ₀·μ₀ + n·x̄) / κₙ
αₙ = α₀ + n/2
βₙ = β₀ + 0.5·S + (κ₀·n·(x̄ − μ₀)²) / (2·κₙ)
```

A Thompson sample is drawn by:
1. `σ² ~ InverseGamma(αₙ, βₙ)` (scipy: `invgamma.rvs(αₙ, scale=βₙ)`),
2. `μ ~ Normal(μₙ, sqrt(σ²/κₙ))`,
3. Return `μ` as the arm's sampled mean.

For *lower-is-better*, the allocator picks `argmin` of the samples.

#### 2. Thompson sampling over UCB / Bayes-UCB

Both consultants agreed:

- **TS** integrates uncertainty natively, is simple to implement, and
  is empirically the most sample-efficient option for small budgets in
  benchmark studies of bandit algorithms (Chapelle & Li 2011).
- **UCB** is deterministic and can over-explore noisy arms early.
- **Bayes-UCB** is principled but sensitive to quantile choice (0.95
  vs 0.99) — adds a tuning knob without consistent gain in this
  regime.

#### 3. Cold-start strategy

- **Warmup**: round-robin `warmup_n` pulls per arm before any bandit
  decisions. Default `warmup_n=2` gives 12 deterministic pulls in a
  6-arm sweep. This guarantees every arm has at least 2 observations
  before posterior comparison.
- **Round-robin order** rather than random order makes the warmup
  deterministic at fixed seed (important for the byte-identical
  `--bandit 0` regression case in this PR's acceptance criteria).
- After warmup, all decisions go through Thompson sampling — no
  forced exploration epsilon. The Gaussian-NIG posterior naturally
  re-explores when an arm's posterior is wider than another's,
  which addresses Gemini's variance-underestimation concern.

#### 4. Pitfall to watch

**Variance underestimation at tiny n.** With `n=2` and a flat
likelihood prior, sample variance can be much smaller than the true
arm variance, leading the bandit to lock onto a lucky arm before its
real performance reveals itself. We mitigate via:

- **κ₀ = 0.01 + n** (not κ₀ = 0.001): keeps the posterior pulled
  toward the broad prior for several observations after warmup.
- **β₀ chosen so prior median variance ≈ (0.1)²**: gives the model a
  baseline scatter expectation that prevents an arm with two
  near-identical observations from claiming σ² ≈ 0.
- **Numerical floor on β** in the sampler: clamp `βₙ` to `≥ 1e-9` so
  `invgamma.rvs` never returns 0 for a deterministic arm.

## Numerical-stability notes

- `S = Σ(xᵢ − x̄)²` is updated *online* via Welford's algorithm to
  avoid catastrophic cancellation when the arm has many similar
  observations.
- A single observation does not contribute to `S` (it would be 0
  anyway), so the variance estimate from the prior dominates until
  n ≥ 2.

## Tie-breaking

When two arms produce identical Thompson samples (rare, only at
n=0/1 with seed = 0), we break ties by arm index. This makes the
algorithm deterministic at fixed seed, which the acceptance test
relies on.

## Out of scope for T2.3

- Adaptive bandit "phases" (early-explore, late-exploit) — Thompson
  sampling handles this implicitly via shrinking posteriors.
- Non-stationary rewards / change-point detection — d_A is
  stationary within a single pipeline config; if you change
  `--vrabel-max-shots` mid-sweep the bandit assumption is broken,
  but so is the rest of the sweep.
- Best-arm identification with PAC guarantees — out of scope; the
  acceptance criteria (70% allocation, 30% CI tightening) are
  empirical, not theoretical.
- True multi-cell sweep CLI (`--cells` flag combining configs) — for
  T2.3 we expose the allocator and wire it in to the existing
  seed-iteration loop; multi-cell config grids land in a follow-up.

## Implementation

The allocator lives in `cflibs/bandit/thompson_allocator.py`. It has
no JAX, scipy, or pandas dependencies beyond what `parameter_sweep`
already imports (numpy + scipy.stats are already there). The
`parameter_sweep.py` integration is purely opt-in via `--bandit N`;
`--bandit 0` (default) preserves byte-identical behavior with the
T1.1 baseline.

## References

- Russo et al., "A Tutorial on Thompson Sampling," 2018.
- Chapelle & Li, "An Empirical Evaluation of Thompson Sampling," NIPS 2011.
- Gelman et al., *Bayesian Data Analysis* 3rd ed., Ch. 3 (NIG conjugacy).

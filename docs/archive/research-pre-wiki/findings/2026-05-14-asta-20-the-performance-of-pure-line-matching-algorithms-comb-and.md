---
id: 2026-05-14-asta-20
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_20_0
status: refuted
surprise: -0.4464
prior: 0.7083
posterior: 0.4219
is_surprising: false
tags: [comb, correlation, spectral_nnls, resolving-power]
relates_to_prs: []
---
# Asta Experiment #20: Refuted

## Hypothesis

> The performance of pure line-matching algorithms (`comb` and `correlation`) is heavily dependent on the spectrometer's resolving power, showing a strong positive correlation with `rp_estimate`, whereas the inversion-based `spectral_nnls` is statistically robust to variations in resolving power.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.4219
- **Surprise:** -0.4464
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 139.5s

## Asta analysis

The code successfully executed and generated the required Spearman rank correlation coefficients and p-values to evaluate the hypothesis. The results reveal that for the 'comb' algorithm, there is a statistically significant correlation (p = 0.0003), but it is strongly negative (Corr = -0.5863), contradicting the expectation of a positive correlation. For the 'correlation' algorithm, the correlation is positive (Corr = 0.3322) but falls just short of statistical significance (p = 0.0589). Finally, as hypothesized, the 'spectral_nnls' algorithm showed no significant correlation with resolving power (Corr = -0.0071, p = 0.9686), confirming its robustness to variations in resolving power. Overall, the hypothesis is partially rejected: while `spectral_nnls` is indeed robust to resolving power, the line-matching algorithms do not exhibit the hypothesized strong positive correlation (with `comb` showing a significant negative correlation instead).

## What this means for us

Weak refute: comb correlation with rp_estimate is significant but NEGATIVE (Spearman ρ=-0.59, p=0.0003) — higher rp *hurts* comb, likely because finer rp produces more candidate peaks that overload the matching algorithm. Correlation is weakly positive (ρ=0.33, p=0.06). Spectral_nnls is rp-invariant as predicted. The comb result is genuinely surprising and worth a focused follow-up.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_20_0
```

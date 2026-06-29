---
id: 2026-05-14-asta-10
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_10_0
status: refuted
surprise: -0.5925
prior: 0.7083
posterior: 0.3281
is_surprising: true
tags: [hybrid_union, literature-ceiling, recall-vs-precision]
relates_to_prs: []
---
# Asta Experiment #10: Refuted

## Hypothesis

> The current macro_f1 ceiling of ~0.69 for `hybrid_union` (falling short of the ~0.85 literature ceiling) is fundamentally driven by a recall bottleneck rather than poor precision, meaning the algorithm systematically misses true elements more frequently than it predicts false ones.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.3281
- **Surprise:** -0.5925
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 133.4s

## Asta analysis

The code successfully executed and generated the requested metrics and statistical tests for the 'hybrid_union' configuration. The mean precision across the 33 spectra is 0.7501, and the mean recall is 0.7009. Although the mean recall is numerically lower than the precision, the statistical tests (testing if precision is significantly greater than recall) returned p-values of 0.0622 (Wilcoxon signed-rank test) and 0.1923 (Paired t-test). Since both p-values are greater than the standard alpha level of 0.05, the difference is not statistically significant. We fail to reject the null hypothesis, suggesting that the current F1 gap for 'hybrid_union' is not fundamentally driven by a significant recall bottleneck relative to precision.

## What this means for us

REFUTED: precision (0.75) ≈ recall (0.70), no statistically significant gap (paired t p=0.19). The 0.69→0.85 macro-F1 gap is NOT a pure recall problem — both metrics need to improve in lockstep. This kills the simple 'just push recall' strategy. The hard recall ceiling finding (#28) below is consistent with this: even union of all 5 identifiers caps at 0.67 recall.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_10_0
```

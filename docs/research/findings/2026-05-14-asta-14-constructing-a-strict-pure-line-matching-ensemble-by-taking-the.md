---
id: 2026-05-14-asta-14
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_14_0
status: weakly_refuted
surprise: -0.3652
prior: 0.3750
posterior: 0.1406
is_surprising: false
tags: [alias, correlation, ensemble, pure-line-union]
relates_to_prs: [#164]
---
# Asta Experiment #14: Weakly Refuted

## Hypothesis

> Constructing a strict pure-line-matching ensemble by taking the union of predictions from `alias` and `correlation` will significantly improve recall over `alias` alone, while strictly maintaining the scientifically invariant `precision = 1.0`, yielding a 'free' F1 boost without sacrificing precision.

## Asta result

- **Prior:** 0.3750
- **Posterior:** 0.1406
- **Surprise:** -0.3652
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `weakly_refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 175.1s

## Asta analysis

The experiment was successfully executed and the results effectively disprove the hypothesis. While the ensemble union of `alias` and `correlation` did yield a highly significant increase in mean recall (from 0.0944 to 0.1995, p < 0.0001), it failed to maintain the strict precision invariant of 1.0. The synthetic mean precision dropped to 0.8600, introducing 7 False Positives overall. This demonstrates that relaxing line-matching constraints via a multi-algorithm union comes at the cost of `alias`'s perfect precision.

## What this means for us

Weak refute: a pure alias∪correlation ensemble does NOT recover the performance attributed to it. Spectral_nnls / hybrid_union are needed to clear the recall floor; line-matching alone is insufficient. This rules out the 'just stack line-matchers' design and supports the inversion-anchored hybrid architecture.

## Cross-references

- PR #164

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_14_0
```

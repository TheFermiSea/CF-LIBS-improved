---
id: 2026-05-14-asta-01
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_1_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [spectral_nnls, cross-dataset, matrix-effects, vrabel-soil]
relates_to_prs: []
---
# Asta Experiment #1: Refuted

## Hypothesis

> The spectral_nnls identifier exhibits significantly lower F1 scores and higher performance variance on the Vrabel soil dataset compared to the Aalto minerals dataset, due to complex multi-element soil matrices disrupting composition inversion.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 123.3s

## Asta analysis

The code successfully executed and evaluated the cross-dataset stability of the `spectral_nnls` identifier. The summary statistics reveal an unexpected result: the Aalto Minerals dataset (n=3) achieved a mean F1 score of 0.1538 and a variance of 0.0000, while the Vrabel Soil Benchmark dataset (n=21) achieved a significantly higher mean F1 score of 0.4455 and a variance of 0.0280. The Mann-Whitney U test yielded a p-value of 0.0144, indicating a statistically significant difference in median performance between the two datasets, but in the opposite direction of the hypothesis (Vrabel performed better). Furthermore, Levene's test returned a p-value of 0.1095, suggesting no statistically significant difference in variance between the two groups. This contradicts the hypothesis that complex soil matrices lead to lower F1 scores and higher variance compared to minerals for this algorithm.

## What this means for us

Spectral_nnls actually performs *better* on Vrabel soils (F1=0.45) than Aalto minerals (F1=0.15). The Aalto-minerals n=3 sample is too small to draw conclusions, but this refutes the assumption that more complex matrices uniformly hurt composition inversion. The next investigation should isolate *which* Aalto minerals fail — there may be a small-n outlier effect rather than a true matrix-complexity signal.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_1_0
```

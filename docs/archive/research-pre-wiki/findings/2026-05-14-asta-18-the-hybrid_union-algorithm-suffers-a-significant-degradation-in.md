---
id: 2026-05-14-asta-18
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_18_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [hybrid_union, vrabel-soil, aalto-minerals, matrix-effects]
relates_to_prs: []
---
# Asta Experiment #18: Refuted

## Hypothesis

> The `hybrid_union` algorithm suffers a significant degradation in F1 score on the `vrabel2020_soil_benchmark` dataset compared to the `aalto_libs` dataset because the complex soil matrices increase the false negative rate (lowering recall), while its precision remains stable across both datasets.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 173.0s

## Asta analysis

The experiment was successfully implemented. The programmer correctly extracted the F1, precision, and recall metrics for the `hybrid_union` algorithm and performed the required statistical comparisons between the `aalto_libs` and `vrabel2020_soil_benchmark` datasets. The results show that mean recall indeed dropped significantly on the Vrabel dataset (from 1.0000 down to 0.6157, p < 0.05). However, contrary to the hypothesis, precision did not remain stable; it increased significantly on the Vrabel dataset (from 0.2500 to 0.7144, p < 0.05). Furthermore, the F1 score actually improved significantly on the Vrabel dataset (from 0.4000 to 0.6441, p < 0.05) rather than degrading. Therefore, the hypothesis is refuted: while the complex soil matrices do lower the recall, the overall F1 score does not suffer a degradation, and precision is certainly not stable across the two datasets.

## What this means for us

REFUTED on direction: hybrid_union performs BETTER on Vrabel soil (F1 significantly improved over Aalto). Both precision AND recall are higher on Vrabel — the soil corpus is apparently a *better* regime for the algorithm than minerals, likely due to (a) richer label cardinality and (b) higher rp. The Aalto n=3 sample is the outlier.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_18_0
```

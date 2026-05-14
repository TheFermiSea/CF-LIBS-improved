---
id: 2026-05-14-asta-21
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_21_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [hybrid_union, spectral_nnls, matrix-complexity, vrabel-soil]
relates_to_prs: []
---
# Asta Experiment #21: Refuted

## Hypothesis

> The `hybrid_union` algorithm is robust to matrix complexity, maintaining a stable F1 score across different geological contexts, whereas `spectral_nnls` degrades significantly on the spectrally complex Vrabel soils compared to Aalto minerals. This demonstrates that composition inversion algorithms are highly sensitive to matrix effects, while consensus voting provides cross-dataset stability.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 151.7s

## Asta analysis

The code successfully executed and generated the required deliverables. The mean F1 scores reveal that both algorithms actually perform better on the Vrabel dataset (hybrid_union: 0.644, spectral_nnls: 0.445) compared to the Aalto dataset (hybrid_union: 0.400, spectral_nnls: 0.154). The two-way ANOVA results show significant main effects for both the identifier (p = 0.0004) and the dataset (p = 0.0019). However, the interaction term between identifier and dataset is not statistically significant (p = 0.7705). This contradicts the hypothesis: `spectral_nnls` does not degrade significantly more on Vrabel soils compared to Aalto minerals than `hybrid_union` does. Instead, both algorithms experience a similar performance boost on the Vrabel dataset, indicating that the relative performance gap between the two algorithms remains consistent across these two distinct matrix complexities.

## What this means for us

REFUTED: no significant identifier×dataset interaction (ANOVA p=0.77). Both algorithms perform better on Vrabel than on Aalto. The 'composition inversion is matrix-sensitive' claim is not supported on this corpus. Whatever's special about Aalto's performance is more likely an n=3 + corpus-construction issue.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_21_0
```

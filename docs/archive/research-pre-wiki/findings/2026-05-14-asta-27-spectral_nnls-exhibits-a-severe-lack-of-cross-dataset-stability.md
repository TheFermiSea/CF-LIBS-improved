---
id: 2026-05-14-asta-27
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_27_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [spectral_nnls, alias, cross-dataset-stability]
relates_to_prs: []
---
# Asta Experiment #27: Refuted

## Hypothesis

> `spectral_nnls` exhibits a severe lack of cross-dataset stability, showing a statistically significant degradation in F1 score and a surge in False Positives on the complex `vrabel2020_soil_benchmark` compared to the simpler `aalto_libs` minerals, whereas `alias` maintains perfect precision (1.0) and stable metric distributions across all dataset matrix complexities.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 112.2s

## Asta analysis

The experiment successfully executed the statistical tests to evaluate identifier performance stability across varying dataset complexities. For `spectral_nnls`, the Kruskal-Wallis tests revealed a highly significant variance across datasets for both F1 score (p=0.0044) and False Positives per spectrum (p=0.0005). Interestingly, the data contradicts the hypothesized direction of degradation: `spectral_nnls` performed worst on the 'simple' `aalto_libs` minerals (mean F1: 0.15, mean FP: 11.0) and better on the 'complex' `bhvo2_usgs` basalt (mean F1: 0.53, mean FP: 3.67) and `vrabel2020_soil_benchmark` (mean F1: 0.44, mean FP: 5.76). Secondly, the hypothesis that `alias` maintains perfect precision across all matrices was proven decisively false. While `alias` achieved a perfect mean precision of 1.0 on `aalto_libs`, its precision dropped to 0.667 on `bhvo2_usgs` and degraded further to 0.381 on `vrabel2020_soil_benchmark`, demonstrating considerable variance. In conclusion, while `spectral_nnls` does indeed lack cross-dataset stability, the specific matrix complexity assumption was backwards, and `alias` is highly vulnerable to false positives when transitioning from simple minerals to complex soils.

## What this means for us

REFUTED on direction: spectral_nnls performs WORST on the simplest dataset (Aalto F1=0.15) and BEST on BHVO-2 (F1=0.53). The 'composition inversion is fragile to matrix complexity' narrative is dead. The n=3 Aalto sample appears to be a low-cardinality outlier, not a matrix-complexity outlier.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_27_0
```

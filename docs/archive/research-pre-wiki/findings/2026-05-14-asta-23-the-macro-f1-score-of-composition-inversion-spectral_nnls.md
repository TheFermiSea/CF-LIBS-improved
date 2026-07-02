---
id: 2026-05-14-asta-23
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_23_0
status: refuted
surprise: -0.4951
prior: 0.7083
posterior: 0.3906
is_surprising: true
tags: [spectral_nnls, alias, label-cardinality, f1]
relates_to_prs: []
---
# Asta Experiment #23: Refuted

## Hypothesis

> The macro F1 score of composition inversion (`spectral_nnls`) decays significantly as the label cardinality (number of true elements in the sample) increases due to signal blending, whereas the perfect precision of `alias` (precision=1.0) is entirely invariant to label cardinality.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.3906
- **Surprise:** -0.4951
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 186.2s

## Asta analysis

The code successfully loaded the dataset and performed the requested OLS regression analyses to evaluate the hypothesis. For `spectral_nnls` (predicting F1 from label cardinality), the model revealed a statistically significant relationship (R-squared = 0.652, p < 0.001). However, contrary to the hypothesis that F1 score would decay, the slope coefficient was positive (0.0432), indicating that F1 actually increased with higher label cardinality in this dataset snapshot. For `alias` (predicting precision from label cardinality), the model showed no statistically significant relationship (R-squared = 0.008, p = 0.628) with a slope coefficient of 0.0143. This confirms the second part of the hypothesis: the precision of the `alias` line-matching algorithm remains statistically invariant to the spectral complexity (label cardinality) of the sample.

## What this means for us

REFUTED on direction: spectral_nnls F1 *increases* with label cardinality (slope +0.043, p<0.001) — same finding as #5. The 'signal blending hurts the inversion' theory is wrong on this corpus. Possible explanation: higher-cardinality spectra force a more constrained inversion solution and stabilize NNLS.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_23_0
```

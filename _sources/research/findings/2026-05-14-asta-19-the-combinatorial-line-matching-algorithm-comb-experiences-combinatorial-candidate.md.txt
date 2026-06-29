---
id: 2026-05-14-asta-19
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_19_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [comb, bhvo2-basalt, failure-mode]
relates_to_prs: [#166]
---
# Asta Experiment #19: Refuted

## Hypothesis

> The combinatorial line-matching algorithm (`comb`) experiences combinatorial candidate explosion and fails significantly more often on the BHVO-2 basalt corpus compared to minerals or soils, owing to the highly complex and spectrally crowded geological matrix of basalt.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 165.0s

## Asta analysis

The experiment was successfully implemented. The code properly isolated the 'comb' algorithm results, categorized failures based on the F1 score and failure reason, and performed the necessary statistical tests. 

### Summary of Results and Findings

**Experiment Results:**
- **Aalto Minerals (n=3):** 0 failures, 100% success rate (0.0 failure rate)
- **BHVO-2 Basalt (n=9):** 9 failures, 100% failure rate
- **Vrabel Soil Benchmark (n=21):** 21 failures, 100% failure rate
- **Chi-Square Test (BHVO-2 vs Others):** p-value = 0.6653
- **Chi-Square Test (All Datasets):** p-value = 0.0000

**Findings:**
The hypothesis is rejected. While the 'comb' algorithm indeed failed completely (100% failure rate) on the complex geological matrix of the BHVO-2 dataset, it also completely failed on the Vrabel soil dataset (21 out of 21 trials). Because of this shared complete failure, the Chi-Square test comparing BHVO-2 specifically against the remaining datasets yielded an insignificant p-value of 0.6653. 

Interestingly, the overall Chi-Square test across all three datasets was highly significant (p = 0.0000). This indicates a strong dataset-dependent performance disparity, driven entirely by the algorithm succeeding 100% of the time on the simpler Aalto minerals corpus while failing completely on both soil and basalt matrices.

## What this means for us

REFUTED on dataset selectivity: comb fails on BHVO-2 (9/9) AND on Vrabel (21/21) — 100% failure on both. The hypothesis that BHVO-2 is uniquely hard for comb is wrong; comb's threshold is just globally too strict (cf. exp #24 below).

## Cross-references

- PR #166

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_19_0
```

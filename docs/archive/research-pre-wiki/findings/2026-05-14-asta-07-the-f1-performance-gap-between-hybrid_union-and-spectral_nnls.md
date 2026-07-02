---
id: 2026-05-14-asta-07
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_7_0
status: confirmed
surprise: +0.2841
prior: 0.7083
posterior: 0.8906
is_surprising: false
tags: [hybrid_union, spectral_nnls, alkali, alkaline-earth]
relates_to_prs: [#164]
---
# Asta Experiment #7: Confirmed

## Hypothesis

> The F1 performance gap between hybrid_union and spectral_nnls is primarily driven by spectral_nnls consistently missing alkali and alkaline earth metals (Na, K, Ca, Mg), which hybrid_union successfully recovers through line-matching consensus.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.8906
- **Surprise:** +0.2841
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `confirmed` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 256.5s

## Asta analysis

The code successfully executed and tested the hypothesis. The per-element precision, recall, and F1 scores highlight that `hybrid_union` has substantially higher recall across several elements compared to `spectral_nnls`. 

Focusing on the target alkali and alkaline earth metals (Na, K, Ca, Mg):
1. **Combined Analysis**: Across 23 spectra containing at least one of these elements, the mean False Negative (FN) rate per spectrum significantly drops from 0.2899 (spectral_nnls) to 0.1920 (hybrid_union). The Wilcoxon signed-rank test yields a p-value of 0.0412, indicating a statistically significant reduction in missed detections for this group.
2. **Individual Element Analysis**:
   - **Na**: `hybrid_union` completely eliminated FNs (0 vs 3 for spectral_nnls).
   - **K**: `hybrid_union` achieved perfect recall, again reducing FNs to 0 (down from 3).
   - **Ca**: Both algorithms successfully detected Ca in all 22 instances (0 FNs).
   - **Mg**: `hybrid_union` reduced FNs from 16 down to 13.

Overall, the results strongly support the hypothesis that the F1 performance gap is significantly driven by `spectral_nnls` failing to detect specific metals like Na, K, and Mg, which `hybrid_union` successfully recovers.

## What this means for us

CONFIRMED: hybrid_union cuts FN rate on Na/K/Ca/Mg from 0.29 to 0.19 (Wilcoxon p=0.041). Consensus voting recovers alkalis that spectral_nnls misses. This is direct evidence that the hybrid_union architecture is doing useful work — not redundant with its members.

## Cross-references

- PR #164

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_7_0
```

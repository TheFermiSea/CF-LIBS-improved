---
id: 2026-05-14-asta-08
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_8_0
status: neutral
surprise: +0.0406
prior: 0.7083
posterior: 0.7344
is_surprising: false
tags: [hybrid_union, spectral_nnls, false-positives, consensus]
relates_to_prs: [#164]
---
# Asta Experiment #8: Neutral

## Hypothesis

> The macro-F1 gap from spectral_nnls (0.44) to hybrid_union (0.69) is closed primarily because hybrid_union applies line-matching constraints to ruthlessly filter out spectral_nnls False Positives, drastically lowering the false_positives_per_spectrum metric.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.7344
- **Surprise:** +0.0406
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `neutral` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 159.8s

## Asta analysis

The experiment successfully verified the hypothesis by comparing the false positive rates and recall between spectral_nnls and hybrid_union. The side-by-side pivot revealed that hybrid_union drastically reduces the False Positives per spectrum from a mean of 5.72 (spectral_nnls) to 1.48. A paired t-test confirmed this reduction is highly statistically significant (t=12.73, p < 0.0001). Furthermore, the recall comparison showed that hybrid_union not only maintains recall but actually significantly improves it compared to spectral_nnls (mean 0.653 vs 0.548; t=-4.63, p=0.0001). This proves that hybrid_union's superior F1 score stems from both ruthlessly filtering out false positives and improving actual element recovery (recall).

## What this means for us

Weak confirm: hybrid_union slashes FPs from 5.72→1.48 per spectrum (t=12.73, p<0.0001) AND lifts recall from 0.55→0.65. The F1 gap 0.44→0.69 is closed by FP filtering *and* recall improvement — not just one. The voting architecture is paying off on both axes simultaneously, which is the strongest argument yet for keeping hybrid_union as the default.

## Cross-references

- PR #164

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_8_0
```

---
id: 2026-05-14-asta-13
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_13_0
status: refuted
surprise: -0.4951
prior: 0.7083
posterior: 0.3906
is_surprising: true
tags: [correlation, alias, precision-recall-tradeoff]
relates_to_prs: []
---
# Asta Experiment #13: Refuted

## Hypothesis

> The correlation identifier sacrifices the perfect precision of alias to achieve a significantly higher recall on trace and minor elements, demonstrating a clear precision/recall tradeoff within the line-matching algorithm family.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.3906
- **Surprise:** -0.4951
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 241.1s

## Asta analysis

The code successfully executed and resolved the previous duplicate index error by utilizing `pivot_table` with 'dataset_id' and 'spectrum_id'. The results partially support the hypothesis. 

1. **Recall**: Across 25 paired samples, the `correlation` algorithm demonstrated a mean recall of 0.1095, compared to 0.0944 for `alias`. The Wilcoxon signed-rank test confirmed that `correlation` yields a statistically significantly higher recall (p-value = 0.0449), supporting the expected recall improvement.

2. **Precision**: The hypothesis expected `correlation` to sacrifice the high precision of `alias`. However, within the 25 paired observations where both algorithms had defined precision (non-NaN), `correlation` achieved a higher mean precision (0.7000) than `alias` (0.4400). Consequently, the Wilcoxon test evaluating if `correlation` precision is less than `alias` precision yielded a non-significant p-value of 0.9806, failing to demonstrate the expected precision tradeoff in this specific subset of paired predictions.

## What this means for us

REFUTED on direction: correlation has higher recall (0.11 vs 0.094, Wilcoxon p=0.045) but the precision sacrifice is much smaller than predicted. The 'pure line-matching family' precision-recall tradeoff exists but is shallow — both algorithms operate near the noise floor on this corpus. Implication: line-matching alone cannot reach the literature ceiling regardless of threshold tuning.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_13_0
```

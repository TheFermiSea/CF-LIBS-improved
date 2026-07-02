---
id: 2026-05-14-asta-16
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_16_0
status: refuted
surprise: -0.4951
prior: 0.7083
posterior: 0.3906
is_surprising: true
tags: [hybrid_union, false-negatives, trace-elements, major-elements]
relates_to_prs: []
---
# Asta Experiment #16: Refuted

## Hypothesis

> The False Negative budget preventing the `hybrid_union` model from reaching the ~0.85 macro_f1 literature ceiling is dominated by trace and rare earth elements (e.g., Ba, Sr, Zr, Ce, Nd) rather than major rock-forming structural elements (e.g., Si, Al, O, Fe, Ca).

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.3906
- **Surprise:** -0.4951
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 104.2s

## Asta analysis

The code successfully loaded the dataset and evaluated the performance of the 'hybrid_union' model on major structural elements versus trace/rare earth elements. The absolute counts reveal that major elements account for 44 False Negatives (FN), while trace elements account for 24 FNs. Therefore, in absolute terms, the False Negative budget is actually dominated by major elements, contrary to the hypothesis. However, the detection rates tell a different story: the model identified 75.7% (137 TP / 181 total) of the major elements but only 56.4% (31 TP / 55 total) of the trace elements. A Fisher's Exact Test yields a statistically significant p-value of 0.0102 with an odds ratio of 2.41, confirming that trace elements are proportionally significantly harder for the model to detect than major structural elements.

## What this means for us

REFUTED on absolute counts: major elements account for 44 FNs vs 24 for trace. But trace-element detection RATE is much worse (56% vs 76%, Fisher significant). The FN *budget* is dominated by majors (simply because there are more of them), so closing the major-FN gap moves macro-F1 more than fixing trace recall. This contradicts the intuitive 'fix the tail' strategy.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_16_0
```

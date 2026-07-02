---
id: 2026-05-14-asta-24
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_24_0
status: confirmed
surprise: +0.3246
prior: 0.6667
posterior: 0.8750
is_surprising: false
tags: [comb, false-negatives, threshold-floor, recall]
relates_to_prs: [#166]
---
# Asta Experiment #24: Confirmed

## Hypothesis

> The extremely low F1 score of the comb algorithm (F1=0.0303) is driven primarily by excessive False Negatives (low recall) rather than False Positives, meaning its internal matching threshold is structurally too high across all datasets, acting as a draconian threshold floor rather than failing due to matrix physics.

## Asta result

- **Prior:** 0.6667
- **Posterior:** 0.8750
- **Surprise:** +0.3246
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `confirmed` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 134.8s

## Asta analysis

The code successfully executed and evaluated the 'comb' algorithm's performance metrics and configuration parameters. The analysis confirms the hypothesis: the 'comb' algorithm suffers from a massive imbalance, with a mean of 7.06 False Negatives (FN) compared to just 0.36 False Positives (FP) per spectrum. The paired Wilcoxon signed-rank test strongly supports this with a highly significant p-value of 1.45e-06. Furthermore, the extraction of the annotation thresholds reveals a highly rigid configuration across the runs (e.g., threshold_percentile strictly set at 85, min_active_teeth at 2, and min_correlation averaging ~0.098), which acts as an excessively strict threshold floor and drives the high False Negative rate.

## What this means for us

CONFIRMED (strongest signal in the run, surprise +0.32): comb's F1=0.03 is FN-dominated. Mean FN=7.06 vs mean FP=0.36 per spectrum (Wilcoxon p=1.45e-06). The threshold (threshold_percentile=85, min_active_teeth=2, min_correlation=0.7) is globally too rigid. This is the LEAST controversial fix on the board: loosen comb's thresholds, ride out the FP increase, and rely on hybrid_union consensus to filter. PR #166 took the OPPOSITE direction (tighter FP rejection for tier-2 Mn/Na/K) — worth investigating whether that was the right call for *this* algorithm specifically.

## Cross-references

- PR #166

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_24_0
```

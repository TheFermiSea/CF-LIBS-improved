---
id: 2026-05-14-asta-12
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_12_0
status: weakly_confirmed
surprise: +0.1380
prior: 0.7083
posterior: 0.7969
is_surprising: false
tags: [spectral_nnls, correlation, hybrid_union, trace-elements, kalium-sodium]
relates_to_prs: [#164, #166]
---
# Asta Experiment #12: Weakly Confirmed

## Hypothesis

> Certain trace and alkali elements (e.g., Na, K, Mn) are systematically missed (False Negatives) by `spectral_nnls` but over-detected (False Positives) by `correlation`, which causes `hybrid_union` to struggle with trace elements despite good performance on major elements.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.7969
- **Surprise:** +0.1380
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `weakly_confirmed` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 173.0s

## Asta analysis

The experiment ran successfully and generated the requested error rates. The results identify the top False Negative drivers for `spectral_nnls` as Al (74% FN rate), Mg (76%), Si (64%), P (100%), and Na (50%). Under `hybrid_union`, Na's False Negatives are completely resolved (0 FNs), and rates for Al, Mg, and Si are somewhat reduced, though P remains entirely undetected. Conversely, `correlation` generated False Positives mainly for Co, Zr, K, and Na. Crucially, the data reveals that `hybrid_union` experiences a massive spike in False Positives for K (8 FPs) and Na (15 FPs) compared to the base algorithms, indicating that while the hybrid approach recovers missing true elements (like Na), it introduces substantial over-prediction for these same elements.

## What this means for us

Weak confirm but with a critical twist: hybrid_union's K and Na FPs EXPLODE (8 K-FPs, 15 Na-FPs) relative to spectral_nnls baseline. Correlation is the FP-injector. The union rule is overcorrecting for FNs by adding noise — argues for the **2-of-3 majority** mode (#164) over straight union for these elements specifically.

## Cross-references

- PR #164
- PR #166

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_12_0
```

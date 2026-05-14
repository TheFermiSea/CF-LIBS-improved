---
id: 2026-05-14-asta-03
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_3_0
status: neutral
surprise: +0.0000
prior: 0.7083
posterior: 0.7083
is_surprising: false
tags: [correlation, comb, fwhm, no-signal]
relates_to_prs: [#153, #154]
---
# Asta Experiment #3: Neutral

## Hypothesis

> The correlation identifier's superior F1 score over the comb identifier is explained by its robustness to basis FWHM mismatches, whereas the comb algorithm's rigid peak-matching thresholds cause catastrophic recall degradation when FWHM mismatch increases.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.7083
- **Surprise:** +0.0000
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `neutral` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 138.8s

## Asta analysis

The code executed successfully, but it revealed a flaw in the hypothesis: the 'basis_fwhm_mismatch_nm' parameter does not exist in the 'annotations' JSON blob for either the 'comb' or 'correlation' identifiers. As a result, 0 rows contained a non-null FWHM mismatch value, and the correlation/regression analysis could not be performed. The script provided a helpful list of available annotation keys for both algorithms (e.g., 'wavelength_tolerance_nm' for correlation, 'min_active_teeth' and 'min_correlation' for comb). The hypothesis will need to be revised to test parameters that are actually utilized by these algorithms.

## What this means for us

Hypothesis untestable: `basis_fwhm_mismatch_nm` is not in the annotations blob. Asta inadvertently catalogued available annotation keys (`wavelength_tolerance_nm` for correlation; `min_active_teeth`, `min_correlation` for comb). Action: any future ablation testing FWHM sensitivity needs to log basis FWHM into annotations first.

## Cross-references

- PR #153
- PR #154

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_3_0
```

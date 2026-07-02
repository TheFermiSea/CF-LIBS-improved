---
id: 2026-05-14-asta-22
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_22_0
status: weakly_refuted
surprise: -0.3652
prior: 0.6250
posterior: 0.3906
is_surprising: false
tags: [spectral_nnls, alias, resolving-power, no-variance]
relates_to_prs: []
---
# Asta Experiment #22: Weakly Refuted

## Hypothesis

> Spectra with lower estimated resolving power (rp_estimate) cause a significant deterioration in precision for spectral_nnls due to line blending confusing the inversion algorithm, whereas the alias algorithm maintains its scientific invariant of 1.0 precision regardless of resolving power.

## Asta result

- **Prior:** 0.6250
- **Posterior:** 0.3906
- **Surprise:** -0.3652
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `weakly_refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 189.7s

## Asta analysis

The code executed successfully, but it revealed significant issues with the experiment's core assumptions. First, the dataset could not be split into 'Low RP' and 'High RP' categories because all 66 filtered records (33 for `alias` and 33 for `spectral_nnls`) fell into the 'Low RP' category (<= 30000.50). This indicates that the `rp_estimate` variable lacks variance across the dataset, with all spectra likely having an identical resolving power estimate of 30000.50. Consequently, the Mann-Whitney U test comparing precisions based on resolving power could not be performed.

Second, the validation step directly contradicted the hypothesis regarding the `alias` algorithm. While the hypothesis assumed `alias` maintains a strict scientific invariant of 1.0 precision, the per-spectrum mean precision for `alias` was computed as 0.5152 (with a minimum of 0.0000 and a maximum of 1.0000). This discrepancy likely occurs because instances where `alias` detects zero elements (True Positives = 0, False Positives = 0) result in a mathematically undefined precision that is represented as 0.0 at the row level, dragging down the mean.

Ultimately, the hypothesis could not be validated because the dataset lacks resolving power variability, and the premise regarding strict per-row precision for `alias` was flawed.

## What this means for us

Untestable: all 66 records fall in one rp bucket (<=30000.5). The n=33 corpus essentially has only two rp values (9433 BHVO-2 and 30000 Vrabel) — there is no continuous rp axis to regress on. Future rp-sensitivity experiments need a corpus with intermediate rp values, or this stays untestable.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_22_0
```

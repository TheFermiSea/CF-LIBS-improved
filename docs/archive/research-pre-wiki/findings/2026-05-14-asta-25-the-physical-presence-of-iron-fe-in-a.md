---
id: 2026-05-14-asta-25
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_25_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [spectral_nnls, iron, false-positives, matrix-effects]
relates_to_prs: []
---
# Asta Experiment #25: Refuted

## Hypothesis

> The physical presence of Iron (Fe) in a sample, which has a highly complex and dense emission spectrum, acts as a systematic confounder for `spectral_nnls`, causing a statistically significant increase in `false_positives_per_spectrum` compared to spectra lacking Iron.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 133.9s

## Asta analysis

The code executed successfully and performed the requested statistical analysis. The results directly contradict the hypothesis. Instead of increasing false positives, the presence of Iron (Fe) is associated with a statistically significant *decrease* in false positives for the `spectral_nnls` algorithm. Spectra with Iron present averaged 4.96 false positives per spectrum, while those without Iron averaged 9.60. The Welch's t-test yielded a t-statistic of -4.4710 with a p-value of 0.00656, indicating this difference is statistically significant. Therefore, the dense emission spectrum of Iron does not act as a systematic confounder inflating false positives in this dataset.

## What this means for us

REFUTED on direction: spectra WITH Fe have FEWER FPs (4.96 vs 9.60, Welch p=0.007). Iron's dense spectrum is apparently a *helpful* constraint for the inversion (it's a strong line in NIST that anchors fitting), not a confounder. Counterintuitive but actionable: do NOT remove Fe-rich basis lines as a 'noise' reduction strategy.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_25_0
```

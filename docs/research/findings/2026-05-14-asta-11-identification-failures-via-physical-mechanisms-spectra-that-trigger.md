---
id: 2026-05-14-asta-11
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_11_0
status: neutral
surprise: +0.0000
prior: 0.7083
posterior: 0.7083
is_surprising: false
tags: [alias, comb, correlation, resolving-power, no-signal]
relates_to_prs: []
---
# Asta Experiment #11: Neutral

## Hypothesis

> Identification Failures via Physical Mechanisms: Spectra that trigger identification failures (non-null `failure_reason`) across line-matching algorithms (`alias`, `correlation`, `comb`) have significantly lower spectrometer resolving power (`rp_estimate`) compared to successful spectra, implying that unresolved peak overlaps (Stark broadening) drive complete algorithmic failure.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.7083
- **Surprise:** +0.0000
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `neutral` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 163.2s

## Asta analysis

The experiment code executed successfully. The dataset was correctly loaded, filtered for the 'alias', 'comb', and 'correlation' algorithms, and the records were partitioned into successful and failed identifications based on the `failure_reason` column. However, the output reveals that there are 0 failed records with a valid (non-null) `rp_estimate` (while there are 99 successful records with a mean resolving power of ~21,713). Because the failed group had 0 records, the statistical tests (Welch's T-Test and Mann-Whitney U Test) could not be performed. This finding suggests either that there were no identification failures for these specific algorithms within the provided benchmark corpora, or that the `rp_estimate` metric is systematically missing (NaN) for spectra that completely fail the identification process.

## What this means for us

Zero failure-class records (failure_reason null for all 99 line-matching runs), so rp_estimate vs failure cannot be tested. The finding is that under current thresholds, none of {alias, comb, correlation} *bail out* on these 33 spectra — they all return *some* prediction (often empty or wrong). The failure-mode investigation would need to be re-cast as 'recall==0' rather than 'failure_reason raised' (cf. exp 26 which does that).

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_11_0
```

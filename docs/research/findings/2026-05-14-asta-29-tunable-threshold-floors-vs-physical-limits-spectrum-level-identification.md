---
id: 2026-05-14-asta-29
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_29_0
status: neutral
surprise: +0.0000
prior: 0.5417
posterior: 0.5417
is_surprising: false
tags: [alias, detection-threshold, failure-correlation, no-failures]
relates_to_prs: []
---
# Asta Experiment #29: Neutral

## Hypothesis

> Tunable Threshold Floors vs Physical Limits: Spectrum-level identification failures in `alias` are significantly correlated with more aggressive user-defined `detection_threshold` parameter values inside the JSON annotations, proving the 'recall ceiling' is artificially bound by configurable thresholds rather than missing self-absorption physics.

## Asta result

- **Prior:** 0.5417
- **Posterior:** 0.5417
- **Surprise:** +0.0000
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `neutral` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 154.6s

## Asta analysis

The experiment executed successfully. The programmer correctly implemented the fallback logic for loading the dataset, parsed the 'annotations' JSON to extract the 'detection_threshold', and segregated the records into 'failed' and 'succeeded' groups based on the presence of a 'failure_reason'. However, the results indicate that all 33 'alias' records in this dataset were classified as 'succeeded' (meaning 'failure_reason' was empty or null for all of them), resulting in 0 records in the 'failed' group. Consequently, the Mann-Whitney U test could not be performed. Furthermore, the summary statistics reveal that the 'detection_threshold' was identical across all 33 records (Median = 0.0200, IQR = 0.0000). Because there is no variance in the threshold parameter within this dataset, it is not possible to correlate threshold aggression with identification failures here.

## What this means for us

Untestable: zero failure-class alias records. Same issue as #4 / #11 — the 'failure' label is too narrow on this corpus.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_29_0
```

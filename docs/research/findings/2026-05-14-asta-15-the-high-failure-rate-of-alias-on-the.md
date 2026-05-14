---
id: 2026-05-14-asta-15
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_15_0
status: neutral
surprise: +0.0000
prior: 0.7500
posterior: 0.7500
is_surprising: false
tags: [aa1100, alias, self-absorption, missing-data]
relates_to_prs: []
---
# Asta Experiment #15: Neutral

## Hypothesis

> The high failure rate of `alias` on the `aa1100_substrate` (aluminum alloy) is caused by severe self-absorption of major Aluminum lines leading to complete spectrum rejection (0 predicted elements), whereas `spectral_nnls`, which utilizes full-spectrum physical forward modeling, successfully recovers Aluminum on the exact same failed spectra.

## Asta result

- **Prior:** 0.7500
- **Posterior:** 0.7500
- **Surprise:** +0.0000
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `neutral` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 110.7s

## Asta analysis

The code executed successfully and correctly implemented the search strategy to locate 'aa1100_substrate' records. However, it found zero matching records across all columns in the dataset. This outcome makes sense given the dataset metadata, which explicitly states that the 33 spectra are drawn only from three corpora: Aalto 74 minerals, BHVO-2 USGS Hawaiian basalt, and Vrabel 2020 soil benchmark. The mention of '13/24 aa1100_substrate records' in the dataset metadata appears to be contextual background information about the algorithm's performance on a broader or different dataset rather than the exact contents of 'all_records_fixed.csv'. Because the required samples are absent, the hypothesis regarding 'alias' failure on aluminum alloy cannot be tested on this specific dataset. The hypothesis must be adapted to evaluate matrix element self-absorption and recovery (e.g., Al or Fe) using the available basalts or soils.

## What this means for us

Untestable: aa1100_substrate not in the n=33 corpus. The previously-reported 13/24 aa1100 failure stat comes from a different (larger) dataset snapshot. Recommend either (a) add aa1100 spectra to the benchmark, or (b) explicitly scope the self-absorption investigation to a different corpus.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_15_0
```

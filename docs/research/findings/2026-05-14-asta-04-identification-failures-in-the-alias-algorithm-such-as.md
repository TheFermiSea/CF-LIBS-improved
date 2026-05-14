---
id: 2026-05-14-asta-04
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_4_0
status: neutral
surprise: +0.0000
prior: 0.2083
posterior: 0.2083
is_surprising: false
tags: [alias, detection-threshold, no-failures]
relates_to_prs: [#159]
---
# Asta Experiment #4: Neutral

## Hypothesis

> Identification failures in the alias algorithm (such as 'No identified candidate elements') are exclusively triggered when the 'detection_threshold' parameter in the annotations functions as an excessively strict tunable threshold floor, rather than being caused by physical self-absorption phenomena.

## Asta result

- **Prior:** 0.2083
- **Posterior:** 0.2083
- **Surprise:** +0.0000
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `neutral` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 115.8s

## Asta analysis

The code successfully executed and loaded the dataset to evaluate the hypothesis. However, the results indicate that there are zero 'Failed' records for the 'alias' algorithm within this specific dataset (which consists of Aalto minerals, BHVO-2 basalt, and Vrabel soil). All 33 'alias' runs were classified as successful and utilized an identical 'detection_threshold' of 0.02. Because there was no variance in the threshold and no failure class present in the data, the Welch's t-test and logistic regression could not be performed. The hypothesis cannot be evaluated with this specific dataset snapshot, likely because the known failing spectra (e.g., 'aa1100_substrate') are not included in the current corpora.

## What this means for us

Zero `alias` failures in the n=33 corpus, so the threshold-vs-physics question is unanswerable here. The lack of failures itself is a finding: alias post-PR-#159 (high_recall opt-in) didn't have its high-recall path enabled in this experiment, but the strict baseline still completed all 33 spectra. The 'failure' mode previously seen on aa1100 substrate is not reproduced on the current corpus.

## Cross-references

- PR #159

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_4_0
```

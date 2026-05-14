---
id: 2026-05-14-asta-17
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_17_0
status: neutral
surprise: +0.0000
prior: 0.7083
posterior: 0.7083
is_surprising: false
tags: [aa1100, alias, self-absorption, missing-data]
relates_to_prs: []
---
# Asta Experiment #17: Neutral

## Hypothesis

> The strict precision of `alias` leads to `aa1100_substrate` failures due to a missing self-absorption correction mechanism for the matrix element, making Aluminum (Al) the most frequent False Negative in these samples.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.7083
- **Surprise:** +0.0000
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `neutral` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 349.8s

## Asta analysis

The experiment was successfully implemented. The script loaded the dataset, correctly verified the absence of the 'aa1100' records, and printed the explicitly requested message stating that the `aa1100_substrate` data is missing and the hypothesis cannot be tested. It then halted execution as instructed. The reported error is simply a `SystemExit` exception caused by calling `sys.exit(0)` within an IPython/Jupyter environment, which does not invalidate the findings. The experiment correctly confirms that the data required to test the self-absorption hypothesis for the AA1100 aluminum alloy is absent from the current dataset.

## What this means for us

Dup of #15: aa1100 not in corpus. Asta exited cleanly. No action.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_17_0
```

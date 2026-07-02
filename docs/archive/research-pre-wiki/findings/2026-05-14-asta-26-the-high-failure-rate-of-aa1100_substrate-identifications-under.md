---
id: 2026-05-14-asta-26
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_26_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [alias, aa1100, resolving-power, self-absorption]
relates_to_prs: []
---
# Asta Experiment #26: Refuted

## Hypothesis

> The high failure rate of `aa1100_substrate` identifications under the `alias` algorithm is driven by a missing physical correction for line broadening, evidenced by a significantly lower resolving power (`rp_estimate`) in failed records compared to successful ones.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 281.7s

## Asta analysis

The code executed successfully after adapting the experiment methodology to define "failure" as a recall of 0 for all `alias` algorithm records. The statistical analysis yields interesting results that directly contradict the original hypothesis. The hypothesis suggested that identification failures were linked to a significantly lower resolving power (`rp_estimate`). However, the results show that failed records (recall == 0) actually possess a significantly *higher* mean resolving power (26,144.14) compared to successful records (17,542.65). Both the Welch's t-test (p = 0.0266) and the Mann-Whitney U test (p = 0.0323) confirm that this difference is statistically significant (p < 0.05). Rather than low resolution causing failures, it appears that excessively high resolving power (narrow spectral lines) might cause the `alias` algorithm to be overly strict in its line-matching, leading to missed elements (recall of 0).

## What this means for us

REFUTED on direction: failed alias records (recall=0) have HIGHER rp (26144 vs 17543, p=0.027). Higher resolving power → more peak candidates → more chances for alias to fail to converge on any match. This is the same pattern as comb (#20). Pure line-matching algorithms are non-monotonic in rp; this is a real physics phenomenon, not a bug.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_26_0
```

---
id: 2026-05-14-asta-28
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_28_0
status: confirmed
surprise: +0.2841
prior: 0.7083
posterior: 0.8906
is_surprising: false
tags: [super-union, literature-ceiling, recall-ceiling]
relates_to_prs: [#168]
---
# Asta Experiment #28: Confirmed

## Hypothesis

> The 0.16 macro-F1 gap to the literature ceiling (~0.85) is fundamentally constrained by a hard recall ceiling; even a 'super-union' of predictions across all 5 algorithms cannot achieve a mean recall of 0.85, indicating that the F1 gap is due to missing physical modeling rather than suboptimal algorithm combining.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.8906
- **Surprise:** +0.2841
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `confirmed` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 132.9s

## Asta analysis

The experiment was successfully executed and effectively tests the hypothesis. The programmer correctly grouped the data by `spectrum_id`, created a 'super-union' of predictions from all five algorithms (`alias`, `comb`, `correlation`, `spectral_nnls`, and `hybrid_union`), and calculated the synthetic recall for each spectrum. The resulting mean super-union recall across the 25 valid spectra evaluated was 0.6669. A one-sample t-test comparing this distribution to the literature ceiling of 0.85 yielded a t-statistic of -3.9619 and a highly significant one-sided p-value of 0.00029. This confirms that the maximum achievable recall with the current suite of algorithms falls significantly short of the 0.85 target. These results strongly support the hypothesis that the performance gap is constrained by a hard recall ceiling, indicating that further algorithm ensemble approaches are insufficient and that introducing missing physical modeling (e.g., LDM/Voigt broadening or Bayesian composition priors) is fundamentally required to close the gap.

## What this means for us

CONFIRMED: the synthetic union of ALL 5 identifiers achieves mean recall 0.667 — far below the 0.85 literature ceiling (one-sided t p=0.00029). **There is no algorithmic combination of the current 5 identifiers that can reach the literature ceiling on this corpus.** Closing the gap requires either (a) new physical modeling (self-absorption correction, Stark broadening, partition-function fixes — see PR #168), or (b) substantially different basis line selection.

## Cross-references

- PR #168

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_28_0
```

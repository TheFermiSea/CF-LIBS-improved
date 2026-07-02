---
id: 2026-05-14-asta-06
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_6_0
status: confirmed
surprise: +0.2841
prior: 0.7083
posterior: 0.8906
is_surprising: false
tags: [alias, recall, trace-metals, matrix-vs-trace]
relates_to_prs: [#159, #166]
---
# Asta Experiment #6: Confirmed

## Hypothesis

> The alias algorithm's strict precision constraint (precision=1.0) results in a disproportionately high False Negative rate for transition metals and rare earth elements (e.g., Mn, Fe, Ti, REEs) compared to major matrix elements (e.g., Si, Al, O).

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.8906
- **Surprise:** +0.2841
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `confirmed` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 165.5s

## Asta analysis

The experiment was successful. The code correctly loaded the dataset, filtered for the 'alias' identifier, parsed the true and predicted elements, and categorized them into 'Matrix' and 'Trace/Transition' groups. The recall for Matrix elements was calculated at 10.49% (17 TP, 145 FN), whereas the recall for Trace/Transition elements was significantly lower at 1.35% (1 TP, 73 FN). The chi-square test of independence resulted in a statistic of 4.7986 and a p-value of 0.0285. This statistically significant p-value (p < 0.05) supports the hypothesis that the strict precision constraints of the 'alias' identifier systematically bias against the detection of trace and transition metals compared to major matrix elements.

## What this means for us

CONFIRMED: alias recall is 10.49% on matrix elements vs 1.35% on trace/transition (chi^2 p=0.029). This is consistent with the literature ceiling: alias's strict precision constraint hurts trace elements disproportionately. PR #159's high_recall opt-in is the right direction; the question is whether tier-2 FP rejection (#166) is enough to keep precision high while loosening the threshold.

## Cross-references

- PR #159
- PR #166

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_6_0
```

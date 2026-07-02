---
id: 2026-05-14-asta-30
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_30_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [hybrid_union, alkali, alkaline-earth, false-negatives]
relates_to_prs: []
---
# Asta Experiment #30: Refuted

## Hypothesis

> Alkali and Alkaline-earth metals (e.g., Na, K, Li, Mg, Ca) account for a disproportionate share of the False Negatives in `hybrid_union`, suggesting that uncorrected physical mechanisms like self-absorption (line reversal at high concentrations) cause these elements to be systematically missed.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 96.7s

## Asta analysis

The code executed successfully and completed all the required steps. It correctly loaded the dataset, filtered for the 'hybrid_union' algorithm, categorized elements into 'Alkali/Alkaline-earth' and 'Other', and calculated the true positives (TP), false negatives (FN), and recall for each category. 

The results show that Alkali and Alkaline-earth metals achieved a recall of 82.47% (80 TP, 17 FN), whereas 'Other' elements had a much lower recall of 63.31% (88 TP, 51 FN). The Chi-square test yielded a statistic of 9.3174 and a highly significant p-value of 0.00227. 

While the statistical test confirms a significant difference in detection rates between the two groups, the direction of this effect directly contradicts the hypothesis. Alkali and Alkaline-earth elements do not suffer a systemic recall deficit; rather, they are detected significantly *better* than the rest of the elements by the hybrid_union algorithm. Therefore, the hypothesis that uncorrected self-absorption causes these elements to be systematically missed is rejected.

## What this means for us

REFUTED on direction: hybrid_union recall on Na/K/Li/Mg/Ca is 82.47% — HIGHER than on 'Other' elements (63.31%). Alkalis are not the FN drivers in hybrid_union; the consensus rule is recovering them well. The remaining FN budget is on majors (Si, Al — see empirical analysis below) and trace metals.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_30_0
```

---
id: 2026-05-14-asta-09
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_9_0
status: confirmed
surprise: +0.2841
prior: 0.7083
posterior: 0.8906
is_surprising: false
tags: [alias, alkali, alkaline-earth, recall-ceiling]
relates_to_prs: [#159]
---
# Asta Experiment #9: Confirmed

## Hypothesis

> The low recall ceiling of the precision-dominant `alias` identifier is systematically driven by False Negatives in highly volatile alkali and alkaline earth metals (Na, K, Ca, Mg), which suffer from self-absorption and bypass strict threshold invariants.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.8906
- **Surprise:** +0.2841
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `confirmed` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 151.1s

## Asta analysis

The code executed successfully and correctly evaluated the hypothesis. The results show that the False Negative (FN) rate for alkali and alkaline earth metals (Na, K, Ca, Mg) is 97.94% (95 FNs, 2 TPs), whereas the FN rate for all other elements is 88.49% (123 FNs, 16 TPs). The Chi-Square test yielded a statistically significant p-value of 0.0146 (Chi2 = 5.9609), indicating that alkali and alkaline earth metals are disproportionately missed by the 'alias' identifier compared to other elements. This strongly supports the hypothesis that the algorithm's recall ceiling is significantly impacted by failures to identify these highly volatile elements.

## What this means for us

CONFIRMED: alias FN rate is 97.94% on Na/K/Ca/Mg vs 88.49% on others (chi^2 p=0.015). The strict precision design hurts alkalis specifically — they're missed almost always. Self-absorption is the leading physical explanation, but the algorithm-level fix is loosening the threshold for this specific element subset (PR #159 is the generic version).

## Cross-references

- PR #159

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_9_0
```

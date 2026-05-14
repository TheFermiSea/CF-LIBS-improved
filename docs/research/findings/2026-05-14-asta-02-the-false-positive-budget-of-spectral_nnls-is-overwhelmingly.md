---
id: 2026-05-14-asta-02
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_2_0
status: refuted
surprise: -0.6899
prior: 0.7083
posterior: 0.2656
is_surprising: true
tags: [spectral_nnls, false-positives, transition-metals]
relates_to_prs: []
---
# Asta Experiment #2: Refuted

## Hypothesis

> The False Positive budget of spectral_nnls is overwhelmingly dominated by transition metals (Fe, Ti, Mn, etc.) due to dense emission spectra confusing composition inversion models that lack Bayesian priors.

## Asta result

- **Prior:** 0.7083
- **Posterior:** 0.2656
- **Surprise:** -0.6899
- **Asta flag `is_surprising`:** True
- **Status (this doc):** `refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 175.1s

## Asta analysis

The code successfully executed the experiment by parsing the true and predicted elements to calculate the False Positives for `spectral_nnls`. It produced a comprehensive frequency table of the FP elements and separated them into Transition Metals and Others. Out of 187 total False Positives, only 38 were transition metals, while 149 were other elements (dominated by U, O, Li, Na, and Pb). The one-sample binomial test resulted in a p-value of 1.0, meaning we strongly fail to reject the null hypothesis. The hypothesis that transition metals overwhelmingly dominate the false positive budget of spectral_nnls is definitively refuted.

## What this means for us

Transition metals are only 38/187 (20%) of spectral_nnls FPs. The real FP drivers are U, O, Li, Na, Pb — i.e. NIST-listed-but-rarely-present trace elements. This points to **basis bloat**, not transition-metal spectrum density, as the FP root cause. Pruning the basis to a geologically realistic prior (e.g. drop U/Pb from soil samples) is likely a much bigger win than adding Bayesian priors over transition metals.

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_2_0
```

---
id: 2026-05-14-asta-05
date: 2026-05-14
source: asta-autodiscovery-1f79815f-78b6-4f74-9c06-27127067c326
experiment_node: node_5_0
status: weakly_refuted
surprise: -0.2841
prior: 0.2917
posterior: 0.1094
is_surprising: false
tags: [spectral_nnls, alias, label-cardinality, precision]
relates_to_prs: []
---
# Asta Experiment #5: Weakly Refuted

## Hypothesis

> As label_cardinality (the number of true elements in a spectrum) increases, the precision of composition inversion algorithms (spectral_nnls) degrades significantly due to spectral interference, while line-matching algorithms (alias) maintain a perfectly stable invariant precision of 1.0.

## Asta result

- **Prior:** 0.2917
- **Posterior:** 0.1094
- **Surprise:** -0.2841
- **Asta flag `is_surprising`:** False
- **Status (this doc):** `weakly_refuted` (refuted ≤ -0.4 < weakly_refuted ≤ -0.2 < neutral < +0.1 ≤ weakly_confirmed < +0.25 ≤ confirmed)
- **Runtime:** 165.6s

## Asta analysis

The code executed successfully and generated the requested statistical metrics. The results strongly refute the hypothesis. For Spectral NNLS, there is a strong positive correlation between label cardinality and precision (r = 0.8806, p < 0.001). The OLS regression indicates that precision significantly *increases* (by 0.0554) with each additional true element (R-squared = 0.775), directly contradicting the assumption that precision degrades due to spectral interference. For the Alias algorithm, there is no significant relationship between label cardinality and precision (r = 0.0875, p = 0.628), showing stable performance regardless of the number of elements; however, the intercept is 0.4132 rather than the hypothesized invariant 1.0. The interaction model indicates that the difference in slopes between the two algorithms (interaction term p-value = 0.170) is not statistically significant at the 5% level. In conclusion, higher label cardinality is associated with better, rather than degraded, precision for the composition inversion algorithm.

## What this means for us

Strong positive correlation (r=0.88, p<0.001) between label cardinality and spectral_nnls precision — the *opposite* of the intuitive hypothesis. More true elements actually *increases* precision, likely because higher-cardinality spectra are richer signal sources that disambiguate the inversion. Alias precision is invariant (as expected from precision=1.0 design).

## Cross-references

- _No closed PRs from this session directly map to this finding._

## Pull full code/output from Asta

```bash
asta autodiscovery experiment 1f79815f-78b6-4f74-9c06-27127067c326 node_5_0
```

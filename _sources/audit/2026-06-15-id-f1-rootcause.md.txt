# Why identification F1 is stuck below 0.5 — root-cause investigation

> **Partially superseded (2026-06-16):** this investigation concluded the low F1 was "not the
> identification algorithms" and "not a stale corpus." The next-day ALIAS-faithful audit
> (`docs/audit/2026-06-16-alias-faithful-fix.md`) reversed **both** of those: the identifiers
> *were* deviating from their source papers (Comb median gate, ALIAS CL-decision/k_sim deflators)
> **and** the corpus *was* violating the papers' assumptions (the 224–265 nm Fe-forest window /
> injected shift / out-of-range T,RP). The "structural — information-starved window" finding (TL;DR
> item 1) was correct and is the through-line; read this doc with the 2026-06-16 conclusion as the
> current state.

**Date:** 2026-06-15 · **Trigger:** "F1<0.5 is still terrible, something must be wrong."
**Method:** 12-agent adversarial workflow (`libs-id-rootcause`, run `wf_bca77300-cb0`): 6 parallel
finders → ranking synthesis → 6 adversarial verifiers that ran live counterfactuals. Plus an
independent line-aliasing DB query (this author). All read-only, CPU-only.

## TL;DR

The low F1 is **real**, but it is **not** in the identification algorithms (Comb / ALIAS /
Correlation / forward_fit), **not** a stale corpus, **not** a metric bug, **not** trace-label
inflation, and **not** the atomic DB. Three genuine problems, in leverage order:

1. **[STRUCTURAL — highest leverage] The benchmark window is information-starved.** The corpus
   spectra cover only **224–265 nm**, a deep-UV iron-group line forest. At this window an **ideal
   detector caps at F1 ≈ 0.31** (verified counterfactual) because absent elements are spectrally
   *aliased onto Fe*. No algorithm can beat this ceiling. **Fix = regenerate the ID corpus with a
   diagnostic window (add ~285–450 nm) and/or balance the panel.** Expected ceiling 0.31 → 0.65–0.80.
2. **[CODE BUG — medium leverage] The shared noise/baseline estimator over-estimates noise ~1200–1600×**
   on these line-dense spectra, collapsing peak detection from ~100–700 real peaks to ~4–12. This
   crushes *recall*. But it is identifier-specific (Comb's core matching is provably invariant to it)
   and the naive fix backfires — needs careful re-tuning. **Code-only**, +0.05–0.15 for detection-gated
   identifiers; does **not** break the window ceiling alone.
3. **[MEASUREMENT HYGIENE — no F1 gain, but we were blind] `--max-spectra 40` silently tests one recipe.**
   Sorted-then-truncated selection makes the first 40 (and 120) spectra **100% `binary_Fe_Ni`**
   `{Fe:0.7, Ni:0.3}`. And **7 of 11 candidate elements never appear in any recipe** (Co is a
   1014-line FP magnet). Stratify/shuffle and trim/balance the panel.

**Implication for recent work:** the entire diag-weights / IEF / BIC tuning effort (J10/o3be) was
polishing identifiers against a benchmark with a **~0.31 absolute ceiling**. The diag-weights
"+0.045 vs Comb" is a *real relative* gain on this window, but no identifier change can lift the
absolute number much past ~0.5 here. Stop tuning identifiers for absolute F1 on `ak3.1.3`; fix the
corpus window + preprocessing first.

## What was REFUTED (do not spend effort here)

| Hypothesis | Verdict | Decisive evidence |
|---|---|---|
| **zfy2: corpus built with old/buggy forward model** | **REFUTED** | Full deterministic seed-42 replay with the *current* `BayesianForwardModel` + current DB reproduces every recipe: Pearson r = 0.997–1.000, recovered T_eV matches stored to 5 dp. Corpus `w2_fixedforward_v1`, created 2026-06-10 (= forward-model overhaul day). **Do not rebuild for staleness.** |
| **Trace/ppm labels inflate FN** | **REFUTED** | min `true_composition` value = **0.02**; truth set identical at thresholds 1e-4 … 0.02. `presence_threshold` is irrelevant. |
| **Eval-harness counting/matching bug** | **REFUTED** | `confusion_counts` recomputed byte-for-byte from true/pred lists on a live run (`all_ok=True`). Symbol-exact matching, no ion-state drift, correct micro-pooling. |
| **Atomic DB integrity (λ / gA / U(T))** | **REFUTED** | All 11 candidates have in-band lines with aki/gk/ek populated; 14 strong lines match NIST AIR to <0.5 pm; partition functions sane/monotone. |
| **Wavelength-axis shift** | **REFUTED** | Detected peaks align to current-DB line positions to median 0.006 nm (95% within 0.05 nm). |

## The structural ceiling — independent confirmation

Direct DB query: fraction of each absent element's strongest 40 lines (by aki·gk) within 0.05 nm
of *any* Fe line, per window:

| Window | Diagnostic elements (<70% Fe-aliased) | Typical aliasing |
|---|---|---|
| **224–265 nm (corpus)** | **0 of 10** | 97–100% |
| 285–450 nm | 2 of 10 (Ni 65%, Cu 67%) | 65–85% |
| 200–900 nm | 0–3 (tolerance saturates at high Fe density) | mixed |

In 224–265 nm there are **2045 Fe lines / ~40 nm ≈ one Fe line every 0.02 nm**, vs a **0.35 nm
resolution element at RP≈700** → ~17 Fe lines per resolved bin. There is **no Fe-free spectral
region**; any absent element's strong line is co-located with an Fe line, so no line-coincidence
identifier can separate present from absent. This is the precise mechanism behind the
"all four algorithms fail identically" signature.

**Ideal-detector counterfactual (verifier, ran live):** feed perfect peaks + present-if-≥2-of-top-40-DB-lines-coincide
→ recall 0.96–1.00 but **precision 0.18, F1 = 0.31–0.32** over the first 40 spectra. A perfect
detector does not cross 0.5 in this window.

## The preprocessing bug (real, but not the master lever)

`cflibs/inversion/preprocess/preprocessing.py` `estimate_baseline` (10 nm median) + `estimate_noise`:
on `pure_Fe_0000` (Imax 2.42e10) the 10 nm median baseline tracks the **line-forest envelope**
(~2.2e9, 12% of max) instead of the true zero floor (the raw spectrum has 13.8–15.7% exactly-zero
pixels). `estimate_noise` then returns **2.13e9 vs true detector noise 1.79e6 → ~1190×**. Detection
threshold (noise×3–4) lands at 26–43% of max, surviving only ~4–12 of ~100–700 real peaks.

- **It starves recall** (true Fe/Ni missed 50–78% of the time).
- **But it is NOT the shared cause:** Comb's core matching uses `_estimate_baseline_threshold`
  (percentile of positive residuals), *not* `estimate_noise`; monkeypatching `estimate_noise` left
  Comb's decisions **identical**. It mainly hurts ALIAS/Correlation.
- **The naive fix backfires:** raw-q20 MAD floor / cap@2% made ALIAS *worse* (F1 0.19→0.07–0.08);
  only a *higher* cap helped (cap@10% → 0.348). Needs SNIP/ALS baseline + a tuned noise floor.

## Recommended fix plan (priority order)

1. **Regenerate the ID benchmark corpus** with a diagnostic spectral window (extend into ~285–450 nm
   where Ni I 341/352, Cr 425, Mn 403, Al 396, Mg 285 have Fe-separable lines) and/or **balance the
   candidate panel** to elements actually exercised by the recipes (add Co/Al/Cu/Mg/Si/Ti/V recipes,
   or trim the panel). Highest leverage: ceiling 0.31 → 0.65–0.80. (Serves epic `a1xz` goal-first re-baseline.)
2. **Fix the preprocessing estimator**: replace the 10 nm median baseline with SNIP/ALS; estimate
   detector noise from the raw low-intensity quantile (not the median-corrected signal); re-tune the
   ALIAS noise cap. Code-only, +0.05–0.15 recall-side.
3. **Measurement hygiene** (bead `jtov` scoreboard): seeded shuffle before truncation OR run all 288;
   report **stratified by recipe** (pure_Fe / pure_Ni / binary_Fe_Ni / steel_like) and by cardinality;
   report **per-element FP** so the Co confounder is visible; companion metric on the restricted panel.

## Caveats

- The 0.05 nm aliasing tolerance is finer than the 0.35 nm resolution element, so at high Fe line
  density the "% aliased" metric saturates — but this *reinforces* the conclusion (no resolvable
  Fe-free bins). The load-bearing number is the verifier's empirical ideal-detector F1 ≈ 0.31.
- Gains in the fix plan are estimates; the corpus-regen ceiling (0.65–0.80) assumes the diagnostic
  window is actually added and should itself be re-measured.

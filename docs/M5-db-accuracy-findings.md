# M5 Atomic-Database Accuracy Findings (confound-controlled)

**Question:** does the new VALD-complete database improve CF-LIBS composition accuracy vs NIST?
**Harness:** `run_scoreboard(pipeline_impl="reference")`, paired per spectrum, **same-element-set
conditioned** RMSE (scored over the intersection of elements both DBs called present, so the
ID-density element-flip cannot distort the comparison). max_spectra=4, datasets bhvo2_chemcam +
supercam_labcal.

## Results (median conditioned RMSE, wt%)

| Variant | bhvo2_chemcam | supercam_labcal | vs NIST |
|---|---|---|---|
| **NIST** (28k graded) | **2.49** | **2.12** | baseline |
| VALD-complete (1.09M) | 5.17 | 4.00 | **~2× worse** (elem sets matched 4/4) |
| VALD grade-B-only (118k) | 20.1 | 17.8 | **~8× worse** (only 15/200 species have B lines) |

## Interpretation (honest)

1. **Raw VALD-complete REGRESSES ~2×** — and this is *real*, not the earlier ID-flip artifact
   (scored element sets were identical 4/4, so conditioned == raw). Mechanism: VALD's line
   *selection* draws ~20 lines/element from 1.09M lines that are **75% D-grade (Kurucz-theoretical)
   + 14% U**, polluting the Boltzmann/Saha quantitation. NIST's small *curated, graded* set fits cleaner.
2. **VALD grade-B-only is far worse** — VALD's experimental-grade (B) lines exist for only **15
   species** (118k lines, 11%). Filtering to them decimates coverage. "Use only the good lines"
   is not viable standalone.
3. **This confirms the standing hypothesis + reconciles everything**: completeness alone HURTS;
   the lever is **grade-aware selection**, not raw substitution. Consistent with the earlier Kurucz
   regression, the literature ("default NIST A/B; use VALD/Kurucz only to widen coverage, downweighted"),
   and the repo's own measurement — which found Kurucz BEATS NIST *only* because it used a curated
   24-strongest-line bundle + structured-GN, NOT the raw dump.

## The key meta-finding

**Database value is COUPLED to the selection regime.** In the production *reference pipeline with
default (intensity-only) selection*, NIST's curated graded set wins. A bigger DB only helps if the
**line selector is grade-aware** (prefer high-grade lines; fall back to fill coverage) or the
analytical set is curated — exactly the regime where the repo measured completeness winning.

## Viable paths NOT yet tested (the real M5 levers)

- **R4 — NIST-A/B ∪ VALD-backfill** (literature-recommended hybrid): NIST graded lines authoritative;
  add VALD lines ONLY where NIST lacks (element,ion,~λ) coverage, flagged/downweighted. Keeps NIST
  accuracy + adds completeness for gaps. Needs a merged DB (complete_atomic_db.py merge+dedup).
- **Grade-aware LineSelector** (Lever 1B, a *pipeline* change not a DB filter): make the per-element
  line selector prefer accuracy_grade A/B over D/U, falling back only to fill the min-lines floor.
  Tested with VALD-complete — this is the change that could let completeness win without pollution.
- **Curated-bundle regime**: run VALD through `build_atomic_bundle.py` (24 strongest/element) +
  structured-GN, apples-to-apples with the repo's winning measurement, to see if VALD-curated beats
  NIST-curated.

## Bottom line

The new databases do **not** improve accuracy by naive substitution into the current pipeline (raw
regresses; grade-B too sparse). The improvement, if it exists, requires **grade-aware selection** —
either the R4 hybrid DB or a grade-aware LineSelector — which are the concrete next experiments.

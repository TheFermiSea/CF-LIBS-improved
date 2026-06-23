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

## Grade-aware LineSelector (Lever 1B) — implemented, MEASURED no-op for VALD

Implemented `grade_aware_selection` (gated, default-off): feed grade-derived `aki_uncertainty`
into the selector so per-element top-N prefers A/B over D/U (U→worst, not the optimistic 0.10).
Found the selector was **grade-blind** (pipeline never passed `atomic_uncertainties`); the flag
activates the existing scoring path.

**Result: byte-identical RMSE on VALD-complete (ON == OFF).** Root cause: VALD's LIBS-major species
are **almost entirely D-grade (Kurucz-theoretical) with no A/B alternatives** — Si/Fe 1/Ca/Ti/Mn/Na/K
are all D/U. **You cannot prefer accurate lines that do not exist.** So VALD's ~2× regression is a
pure **gf-VALUE-accuracy** issue (Kurucz-theoretical D-grade < NIST graded for the same lines),
*not* a selection problem — unfixable by any selection change *within VALD*.

## DEFINITIVE conclusion

The new databases do **NOT** improve accuracy by substitution: VALD-complete is ~75% Kurucz-theoretical
D-grade, so its gf values are less accurate than NIST's graded set for the LIBS analytical lines, and
it regresses ~2×. Grade-aware selection can't recover it (no high-grade alternatives in VALD).

**The only path that can improve accuracy is R4 — NIST-A/B authoritative + VALD backfill ONLY where
NIST lacks coverage.** Keep NIST's accurate gf where it has lines; use VALD purely to *widen* coverage.
And the grade-aware selector built here is **exactly the mechanism R4 needs**: in a NIST∪VALD merged DB,
NIST lines are A/B and VALD backfill is D, so `grade_aware_selection=True` would prefer the NIST-accurate
lines. This is also the literature-recommended approach and the basis of the repo's own Kurucz-beats-NIST
measurement (curated graded set, not raw dump).

**Next experiment: build R4 (merged NIST-A/B ∪ VALD-backfill) and benchmark with grade-aware ON.**

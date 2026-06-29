This is a synthesis/writing task. I have all the data I need in the provided JSON (the protocol design matrix and the core NIST-vs-VALD run results). Let me write the markdown verdict and campaign plan.

# M5 Atomic-Database Accuracy Verdict & Campaign Plan

## 1. Initial Verdict — does VALD-complete improve composition accuracy vs NIST?

**No — and more importantly, the only run executed so far does NOT cleanly answer the question.** VALD-complete measured *worse* than NIST, but it was run in the **confounded full-pipeline regime** — the exact regime the prior failed Kurucz benchmark warned against. This is a "swap the whole DB into the dense pipeline" result, not a "data-quality, selection-held-constant" result. Treat it as a **red flag, not a verdict.**

### What was measured (confound-controlled on db-path only, but selection NOT equalized)

Identical `run_scoreboard(..., pipeline_impl="reference", preset=geological default)`, varying **only** the db path. NIST = `ASD_da/libs_production.db`; VALD = `output/vald_complete.db` (1.09M lines). `max_spectra=4`.

| Dataset | NIST RMSE med (wt%) | VALD RMSE med (wt%) | Delta (VALD−NIST) | Direction |
|---|---|---|---|---|
| `bhvo2_chemcam` (holdout) | 2.486 | 5.174 | **+2.689** | WORSE |
| `supercam_labcal` | 2.122 | 4.001 | **+1.878** | WORSE |

- `chemcam_calib` was **unrunnable** — its PDS CSV (`data/chemcam_calib/msl_ccam_libs_calib.csv`) is absent from both the worktree and main checkout (0 spectra). So this is **4 runs (2 datasets × 2 DBs), not 6.** All four: `n_ok=4, n_failed=0`.

### Confound checks (the honest part)

- **Scored-element SET matched** for both DBs on both datasets → the RMSE gap is *not* a dropped-element scoring denominator artifact. Good.
- **BUT the called-present set FLIPPED** on **3/4 bhvo2** and **2/4 supercam** spectra: VALD dropped Al/V and added Mn/Ti. This is the **ID-density perturbation signature** — the dense 1.09M-line list went through full `run_pipeline` line-identification with **no curated N-strongest-per-element cap and no equalized selection.** The selection regime changed between DBs, so the RMSE gap **cannot be attributed to gf/line-value quality** — line *selection* moved too.
- **PF/IP source differs** (VALD ships Barklem PF+IP; NIST uses its own PF source). Uncontrolled axis — the original 3rd confound.
- Dominant regressors are exactly the majors whose selection was perturbed: **supercam Al |err| +15.87 wt%**, **bhvo2 Fe +5.73 / Mg +4.55 wt%.** Bit-identical elements (K/Na/P on bhvo2; ~16 traces on supercam) confirm VALD only moved the majors it re-identified — an **ID-density effect, not a uniform data-quality shift.**

### Comparison to the documented Kurucz deltas

The repo's **curated-24-strongest-lines + structured-GN** measurement (selection held constant) found **Kurucz BEATS NIST**: bhvo2 +0.0042, chemcam_calib +0.0060, supercam_labcal +6.42 wt% (all *better*). This VALD full-pipeline run shows the **opposite sign** (bhvo2 −2.689, labcal −1.878, both *worse*).

**These are not contradictory — they are different regimes.** The Kurucz win came from curated, selection-equalized bundles; this VALD run is the uncurated dense-flood regime. The sign flip is precisely the signature the lesson predicted: *if VALD shows the "2× worse" artifact, a confound leaked back in.* It did. **The fair data-quality measurement (equalized top-24 selection) has not yet been run.** No adoption claim — for or against — is warranted from this run alone.

---

## 2. The Full Permutation Matrix — runs to execute next

The fix is to **vary ONLY the gf/line-value source** while freezing pipeline, solver, line-selection, and PF source. The harness already supports this: `build_pipeline_config` exposes the exact selection knobs (`max_lines_per_element`, `min_lines_per_element`, `min_relative_intensity`) that caused the original flood, and `run_scoreboard(..., config_overrides=...)` runs the identical production pipeline for any DB.

### FIXED config block — byte-identical for EVERY run (this is the contract)

```python
config_overrides = {
    "max_lines_per_element": 24,    # THE density equalizer: top-24-strongest per (element,ion) by gA-Boltzmann
    "min_lines_per_element": 3,     # identical recall floor — no DB drops an element on count
    "min_relative_intensity": 0.0,  # floor disabled; the top-24 cap does the equalizing
}
# preset=None (geological default), pipeline_impl="reference", PF=Barklem for BOTH,
# candidate set = scoreboard default, seed=20260610
```

Mechanism: matched **strongest-N-per-(element,ion)** with N=24 (not exact-wavelength intersection — NIST air vs VALD grids differ by ±pm, so intersection would silently shrink the set and re-introduce a coverage confound). The 24-cap forces a 1.09M-line DB to present exactly the same number of analytical slots/element as the 28k DB.

### Ranked campaign matrix

| # | DB variant | Datasets | max_spectra | Build? | Purpose |
|---|---|---|---|---|---|
| **R1** | NIST baseline | all 6 | SuperCam=40, rest=None | none | reference board; every delta measured against this |
| **R2** | VALD-complete | all 6 | same | none | **headline verdict** (selection now equalized — the fair re-run of §1) |
| **R3** | VALD grade-B-only (`WHERE accuracy_grade='B'`, 118k experimental-grade) | all 6 | same | **new DB** | **quality vs quantity** — does VALD's effect come from graded lines or D/U bulk? |
| **R4** | NIST-A/B ∪ VALD-backfill | all 6 | same | **merged DB** | production-recommended hybrid: curated strong-line gf + completeness backfill |
| **R5** | VALD-complete, N∈{12,48} | chemcam_calib + supercam_labcal | same | none | **density-robustness** — if VALD only wins at large N, that's density not quality |
| **R6** | NIST ∪ VALD union (NIST wins on collision) | all 6 | same | **merged DB** | coverage upper bound / dedup control (should ≈ R4; if not, dedup is the confound) |
| **R7** | Kurucz-full | all 6 | same | regenerate | **OPTIONAL** — confirm documented Kurucz deltas reproduce under THIS harness (sanity) |

### Datasets, sampling, metric

**Datasets (composition-truth-bearing only):**

| Dataset | Tier | include_holdout | Note |
|---|---|---|---|
| `chemcam_calib` | optimization | — | expect +0.0060 dir. — **NOTE: PDS CSV must be restored first (was absent in §1)** |
| `supercam_labcal` | optimization | — | largest documented win (+6.42 wt%) |
| `csa_planetary` | optimization | — | the documented Kurucz **REGRESSION** case — must be checked, not cherry-picked away |
| `silva2022` | optimization | — | independent calibration-standard matrix |
| `bhvo2_chemcam` | **holdout** | True | adoption-gate headline (+0.0042) |
| `supercam_scct` | **holdout** | True | Mars adoption gate (+4.86 wt%) |

Excluded: `nist_steel`/`aalto` (presence-only, no composition RMSE), `gibbons2024` (vault).

**Sampling:** `max_spectra=40, seed=20260610` for the two 547-spectrum SuperCam sets (cost); `max_spectra=None` for the small calib/bhvo2 sets. Fixed seed → **paired** per-spectrum comparison (same 40 spectra every variant).

**Metric & "is it real":**
- Primary: per-dataset `composition.rmse_wt_median` delta (VALD−NIST), O excluded. Win = **negative** delta.
- **Same-element-set conditioning:** restrict RMSE to elements called present by **BOTH** DBs (removes the ID-drop confound that contaminated §1 — exactly the Al/V flip that broke the last run).
- **Significance:** paired sign / Wilcoxon on per-spectrum `rmse_wt` (stored in `spectra[]`). A verdict needs **consistent direction across ≥2 datasets AND a paired test clearing per-spectrum noise** — not a single-dataset median flip.

### Run command form (config block byte-identical across all runs)

```bash
env PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu \
  /home/brian/code/CF-LIBS-improved/.venv/bin/python -c "
from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard, write_artifacts
b = run_scoreboard(AtomicDatabase('<DB>'), datasets=[...], include_holdout=True,
                   max_spectra=40, seed=20260610, pipeline_impl='reference',
                   config_overrides={'max_lines_per_element':24,'min_lines_per_element':3,'min_relative_intensity':0.0})
write_artifacts(b, 'output/m5_perm/<variant>')"
```

---

## 3. What to Build

Three DBs need constructing; the rest load as-is. Every merged/filtered DB **copies `partition_functions` + `species_physics` from VALD's Barklem set verbatim** so PF stays constant — variant differs only in `lines` rows. Each build gets a `provenance`/`source` audit and a **`source`-string equality assertion** before any run (the original 3rd confound was a PF source-string mismatch).

| Variant | Build recipe |
|---|---|
| **R3 — VALD grade-B-only** | One `CREATE TABLE lines AS SELECT ... FROM vald_complete WHERE accuracy_grade='B'` (~118k experimental-grade lines); copy `partition_functions` + `species_physics` verbatim. |
| **R4 — NIST-A/B ∪ VALD-backfill** | Use `complete_atomic_db.py` (already merges sources + dedups by element/ion/wavelength). NIST grade A/B/B+ lines authoritative; add VALD lines only for (element,ion,~λ) NIST lacks, flagged `provenance='vald_backfill'`. |
| **R6 — NIST ∪ VALD union** | `complete_atomic_db.py` union of both line tables; NIST wins on (element,ion,λ) collision. Control for R4 — should ≈ R4. |
| **R7 — Kurucz-full** (optional) | Regenerate `output/kurucz_complete.db` via `ingest_kurucz_atomic.py` + `complete_atomic_db.py`. |

**Prerequisites that must be fixed before R1/R2 are credible:**
1. **Restore `data/chemcam_calib/msl_ccam_libs_calib.csv`** (absent in both checkouts — it silently zeroed a documented signal dataset in §1). Without it the matrix is missing the +0.0060 reference point.
2. Symlink/stage `data/supercam_calib` and `supercam_scct`/`csa_planetary`/`silva2022` data into the worktree as needed (only labcal + bhvo2 were locally present in §1).

**Confounds to keep watching during builds:** (a) detected-peak-to-line **match tolerance must be R-derived (DB-independent)**, not absolute-λ, or the NIST-air/VALD grid offset tilts ID; (b) merged DBs (R4–R6) add **dedup logic as a new variable** — audit the provenance column, confirm union≈backfill or dedup is itself the confound; (c) confirm both DBs use the **same n_e route** via the recorded `ne_source` field (Stark-width coverage differs; hold the n_e path host-delegated).

---

## 4. Recommendation

**Do NOT adopt VALD-complete on the strength of the §1 run — and do not reject it either.** The §1 result is in the confounded dense-flood regime; its "VALD is 2× worse" finding is *expected behavior of an uncontrolled selection swap*, not evidence about gf data quality. Adopting or rejecting now would repeat the last benchmark's mistake (over-claiming from a confounded measurement).

**The real lever is almost certainly grade-aware selection, not raw completeness.** Two independent signals point this way:
1. The documented **curated-24-strongest + structured-GN** Kurucz win (selection held constant) went the *opposite* direction from the dense-flood VALD run — i.e., the *selection regime*, not the *line source*, dominated the outcome in both prior measurements.
2. §1's regression was driven entirely by **majors whose identification flipped** (Al, Fe, Mg) while ~16 traces stayed bit-identical — a selection/ID-density effect, not a uniform data-quality shift.

**Concrete next step (in order):**
1. **Restore `chemcam_calib` data + stage the other 4 datasets** into the worktree (unblocks the full 6-dataset board).
2. **Run R1 (NIST) and R2 (VALD-complete) with the equalized top-24 config** — this is the fair re-run of §1 and the headline verdict. If VALD now lands in the documented Kurucz direction (small win), the §1 worse-result was confound, confirmed.
3. **Run R3 (VALD grade-B-only)** — the decisive quality-vs-quantity test. **If grade-B beats VALD-complete, grade-aware selection is the real lever** and the production recommendation becomes R4 (NIST-A/B ∪ VALD-backfill), not raw completeness.
4. **Gate adoption** on: negative RMSE delta, **consistent across ≥2 datasets**, surviving **same-element-set conditioning** and a **paired Wilcoxon** — including the `csa_planetary` regression case (do not cherry-pick it away). Until then, **NIST baseline stays the shipped DB.**
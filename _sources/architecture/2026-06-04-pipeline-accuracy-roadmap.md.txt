# CF-LIBS Pipeline Accuracy Roadmap — BHVO-2 Basalt North-Star

**Date:** 2026-06-04
**Branch / HEAD:** `7aae680` (`fix(identify): default invert min_relative_intensity floor to 100`)
**North-star sample:** USGS BHVO-2 basalt, ChemCam spectra `data/bhvo2_usgs/chemcam_bhvo2_loc{1,2,3}_spectrum.csv`
**Certified truth:** `cflibs/benchmark/reference_compositions.py:56` `BHVO2_BASALT_USGS`, renormalized to the 10 cations (O omitted), sum = 1.
**Goal:** measured composition error on a certified geostandard at or below the literature "good CF-LIBS" band (≤10 % average relative error on majors, Safi 2019).

---

## 1. North-Star: BHVO-2 Composition Error, Before and After This Pass

The single number that defines success is the per-element composition error on real BHVO-2 ChemCam spectra, run through the production pipeline (`cflibs analyze` / `cflibs invert` → `detect_line_observations` → `LineSelector().select` → `IterativeCFLIBSSolver.solve`). Per-element values are wt % over the 10-cation basis renormalized to 100, mean over loc1/loc2/loc3.

### CRITICAL CAVEAT — two CLI entry points diverged this pass

The fixes landed **only in `invert_cmd`** (`cflibs/cli/main.py:161`), **not in `analyze_cmd`** (`cflibs/cli/main.py:326`), which is the path that produced the baseline. Re-running the exact baseline measurement at HEAD is **byte-identical** to the baseline — RMSE 33.69, MAE 18.99, Na 97.90 %. `analyze_cmd` still calls `detect_line_observations(...)` with the `min_relative_intensity` default (`main.py:347`, no floor) and a bare `LineSelector()` (`main.py:357`), so none of the IDENT-RYDBERG / SA fixes touch it. The improvement is real but is gated behind the `invert` entry point.

| Element | Certified (10-cation wt%) | BEFORE (analyze_cmd) | AFTER (invert, SA off) | AFTER (invert, SA on) | Verdict |
|---------|--------------------------:|---------------------:|-----------------------:|----------------------:|---------|
| Si | 41.99 | **1.11** (drop) | 21.45 | 26.79 | recovered, still low |
| Ti |  2.95 | **0.00** (drop) | 10.89 | 19.48 | recovered but over-predicted |
| Al | 12.85 | **0.00** (drop) | **0.00** (drop) | **0.00** (drop) | STILL DROPPED |
| Fe | 15.50 | **0.66** | 21.61 | 26.40 | recovered, now slightly over |
| Mn |  0.23 |  0.28 | **37.65** | 22.16 | REGRESSED (minor blows up) |
| Mg |  7.85 | **0.06** | **0.00** | **0.00** | STILL DROPPED |
| Ca | 14.67 | **0.00** (drop) | 8.39 | 5.16 | recovered, still low |
| Na |  2.97 | **97.90** (blowup) | **0.00** | **0.00** | blowup KILLED (now zero) |
| K  |  0.78 | **0.00** (drop) | **0.00** (drop) | **0.00** (drop) | STILL DROPPED |
| P  |  0.21 | **0.00** (drop) | **0.00** (drop) | **0.00** (drop) | STILL DROPPED |

**Aggregate north-star (RMSE / MAE, apples-to-apples across both reports):**

| Metric | BEFORE (analyze_cmd, baseline) | AFTER (invert, SA off) | AFTER (invert, SA on) |
|--------|-------------------------------:|-----------------------:|----------------------:|
| RMSE (wt%) | **33.69** | 14.83 | **11.97** |
| MAE (wt%)  | **18.99** | 10.29 | **9.87** |
| T (kK)     | 14.2–14.4 | 10.9 | 10.3 |
| n_e (cm⁻³) | 2.5e17 | 3.0e17 | 2.8e17 |

> **Aitchison distance is NOT apples-to-apples** between the two reports (baseline harness reported 15.5; re-measure harness computes 43.28 on the byte-identical baseline result — different ε/basis instantiation). It is dominated by the 3–4 hard-zero elements and barely moves; disregard it as a cross-report metric. RMSE/MAE and the per-element table reproduce exactly.

**Bottom line:** On the *intended* path (`invert`), RMSE dropped from **33.69 → 11.97 wt%** (~2.8×). The degenerate Na-dominated non-answer is gone — the result is now a noisy-but-balanced estimate, the qualitative shape the project wanted. On the *baseline* path (`analyze`), the number is **unchanged**. The single highest-leverage remaining action is structural: make the two CLI entry points agree.

---

## 2. What Was Fixed This Pass, and the Accuracy Delta

| Fix | File:line | What it does | Delta on north-star |
|-----|-----------|--------------|---------------------|
| **IDENT-RYDBERG / DET-2** — Na Rydberg pruning | `cflibs/cli/main.py:231` (invert path) | Default `min_relative_intensity=100.0` (was `None`) prunes the spurious high-lying Na I 413–421 nm Rydberg lines (A_ki ~1e5, unobservable at 1.2 eV) that drove Na to 98 %. | **Na 97.90 → 0.00**. The dominant baseline error (absErr 94.9) is eliminated. Biggest single win. |
| **B1** — wire SelfAbsorptionCorrector into solver | `cflibs/inversion/solve/iterative.py:790-797`, `:1006` `_apply_self_absorption_correction` | Outer Bulajic-2002 recursion: recompute τ from current plasma state each iteration, divide observed intensities by the curve-of-growth escape factor before Boltzmann/closure fit. Was 3000 LOC of dead code (zero callers). | RMSE 14.83 → **11.97** on real BHVO-2 (tamps Mn 37.65 → 22.16, nudges Si/Fe up). |
| **B1 safety** — opt-in + caps | `iterative.py:718` (`apply_self_absorption=False` default), `:721` `tau_cap=10.0`, `:769` high `mask_threshold` | Made correction **default-off** after it over-corrected optically-thin synthetic spectra; bounded I/f(τ) boost to ~10× literature regime; strong major lines corrected not dropped. | Prevents thin-data false positives; net-positive on thick BHVO-2. |
| **B3** — CDSB τ uses species density | `cflibs/inversion/physics/cdsb.py:267,288,333` (`number_densities` arg) | Initial τ now scales with absorbing-species number density (n_species·L) not n_e (was under-estimating τ ~40–70× for matrix majors). | Enables B1 to converge to "thick" not "everything thin". |
| **B2** — retain resonance lines when SA on | `cflibs/cli/main.py:264` (`exclude_resonance = not apply_self_absorption`), `iterative.py` | When SA enabled, retain low-E_i resonance lines (Ca II H/K, Na D, Mg I 285, Al I 396) and **correct** them rather than drop them (Aragón & Aguilera 2008). | Lets optically-thick majors reach the Boltzmann plot. |
| **aki_uncertainty weighting** (landed PR #215) | `cflibs/inversion/physics/boltzmann.py:119` (`aki_uncertainty_weighting=True` default) | Inverse-variance Boltzmann weighting by NIST A_ki grade. Already on by default. | Improves T accuracy / uncertainty calibration on D/E-grade alkalis. |

**Net:** Two of the three baseline failure modes are addressed — the Na blowup (killed) and Ca/Si/Fe/Ti drop (partially recovered). The **element-drop of Al/K/P/Mg** and a **new Mn/Ti over-attribution** remain. The B1 physics fix is now wired and demonstrably net-positive on real thick spectra, reversing its synthetic over-correction.

---

## 3. Ranked Remaining Pipeline Issues (Blocking the Goal)

Ranked by **expected composition-accuracy lift ÷ effort**. Each is a candidate bead. Stage tags: DET = detection, ID = identification, CF = CF-LIBS physics, COMP = composition/closure, PROD = productionization.

### Rank 1 — Unify `analyze_cmd` with `invert_cmd` (PROD / DET / ID)
- **file:line:** `cflibs/cli/main.py:347` (bare `detect_line_observations`, no `min_relative_intensity`), `:357` (bare `LineSelector()`). Contrast `invert_cmd` `:231`, `:265-272`.
- **kind:** productionization / entry-point divergence.
- **lift:** RMSE **33.69 → 11.97** on the default path — the entire delta of this pass is currently stranded behind `invert`. Single largest available lift.
- **effort:** trivial (port the IDENT-RYDBERG floor + tuned `LineSelector(min_snr, exclude_resonance, …)` + `apply_self_absorption` plumbing from `invert_cmd` to `analyze_cmd`).
- **literature:** N/A (internal consistency); validated by the byte-identical-baseline measurement above.

### Rank 2 — Recover dropped majors Al / K / P (DET / ID)
- **file:line:** `cflibs/inversion/identify/line_detection.py:453-456` comb gates (`comb_min_matches=3`, `comb_min_precision=0.02`, `comb_max_missing_fraction=0.85`); element prefilter `_kdet_filter_elements` at `line_detection.py:~1290-1325` (`kdet_min_score`/`kdet_min_candidates`, warning `"kdet_filtered_elements"`).
- **kind:** over-strict detection/identification gating.
- **lift:** **HIGH.** Al (12.85), K (0.78), P (0.21), Mg (7.85) are still hard-zero. Al + Mg alone are ~21 wt% of unrecoverable error; the RMSE floor cannot fall below ~10 wt% while they are dropped. Relax `comb_min_precision 0.02→0.01`, `comb_max_missing_fraction 0.85→0.95`, and `comb_min_matches 3→2` (DET-1, IDENT-COMB-MATCHED-THRESHOLD-4 — recovers K I 766.5/769.9 and Na D doublets).
- **effort:** small (threshold changes + kdet score relaxation).
- **literature:** Safi 2019 (<10 % only if majors are present); De Giacomo 2007 (<5 % meteorite majors). Recovering majors is the precondition for any literature-band result.

### Rank 3 — Fix Mn / Ti over-attribution (ID / CF)
- **file:line:** `cflibs/inversion/identify/line_detection.py` comb scoring (no resonance-preference / E_k physical reject); `cflibs/inversion/physics/closure.py` (closure normalizes any degenerate population).
- **kind:** misassignment — Mn (cert 0.23) predicted 22–38 %; Ti (cert 2.95) over-predicted 11–19 %. The Na degeneracy was replaced by a milder Mn/Ti one, not cured.
- **lift:** MEDIUM-HIGH. Mn alone is +22–37 wt% absolute error post-fix. Add an upper-level energy reject (E_k > kT + margin) mirroring the Na Rydberg fix, and a closure sanity guard that flags single-element-dominated populations (no guard caught Na=98 % or Mn=38 %).
- **effort:** small–medium.
- **literature:** De Giacomo 2007 (exclude ground-state-terminating + slow-emission lines, via Poggialini 2023); same physical filter that fixed Na generalizes to Mn/Ti.

### Rank 4 — Si locked to wrong ionization stage (ID / CF)
- **file:line:** `cflibs/inversion/identify/` ALIAS / comb stage handling (IDENT-ALIAS-SI-IONIZATION-STAGE-8, DET-8); ion-stage pooling in `line_detection.py` comb scoring.
- **kind:** wrong ionization stage — Si identified only as Si II (6 weak high-lying lines at 244/250/272 nm), Si I 288.2 nm (SNR ~51, the *correct* stage) missed. Corrupts the Saha ion→neutral mapping for the largest cation (Si, cert 41.99).
- **lift:** MEDIUM-HIGH (Si is 42 % of the basis; it currently tops out at 27 % even after fixes). Don't let high-lying weak lines win a stage when a strong resonance/low-E line of the other stage is present.
- **effort:** small.
- **literature:** physically impossible to be Si-II-dominant at 1.2 eV; Aguilera & Aragón 2004 Saha-Boltzmann self-consistency requires correct stage assignment.

### Rank 5 — Enforce charge balance (wire Anderson solver) (CF)
- **file:line:** `cflibs/plasma/anderson_solver.py` exists but is **not** wired into `cflibs/plasma/saha_boltzmann.py` (saha solve consumes n_e as a free input, does not close n_e = Σ ion charges).
- **kind:** missing physics constraint — n_e is a free parameter, not closed by charge neutrality.
- **lift:** MEDIUM. Removes a degree of freedom and stabilizes the Saha ion→neutral correction that B1 self-absorption now depends on. Prerequisite for a *trustworthy* Saha mapping.
- **effort:** medium.
- **literature:** Tognoni 2010 Spectrochim. Acta B 65, 1 §2.2/3.4 (CF-LIBS requires simultaneous Saha + charge neutrality); Ciucci 1999 Appl. Spectrosc. 53, 960.

### Rank 6 — Stark-width convention 20× mismatch on JAX/forward paths (CF / COMP)
- **file:line:** `cflibs/radiation/profiles.py` (FWHM/HWHM and 1e16/1e17 reference handling); DB stores FWHM@1e17, runtime consumers treat as HWHM@1e16 (A4-CONV-2).
- **kind:** unit-convention bug — every Stark width inflated ~20× on manifold/Bayesian/forward paths (e.g. Al I 396.15: 4.5 pm stored → 90 pm runtime). Default CPU solver partially immune (alias passes `transition=None`, `omega_stark=0`).
- **lift:** ~0 on the default solver; real composition bias on JAX/Bayesian/forward composition paths and any n_e-from-Stark diagnostic. Prerequisite for n_e-from-Stark.
- **effort:** small–medium.
- **literature:** El Sherbini 2005 Spectrochim. Acta B 60, 1573 (Δλ₀ = w_s·n_e); IRSAC RSC Adv. 2026 D6RA01889K (Ca II 393 Stark n_e).

### Rank 7 — Fix `polyfit` quartic-weighting bug (CF)
- **file:line:** `cflibs/inversion/physics/boltzmann.py:~890-895` (`np.polyfit(w=1/σ²)` squares weights → effective 1/σ⁴; JAX path made bug-compatible).
- **kind:** numerical correctness — weights passed to `polyfit` must be 1/σ, not 1/σ².
- **lift:** LOW-MEDIUM. Mainly improves T accuracy + uncertainty calibration on D/E-grade alkalis (Na/K), feeding the Saha correction. Cheap, removes a real correctness bug.
- **effort:** trivial.
- **literature:** Tognoni 2010 Eq. 5 (WLS form); Kramida 2024 NIST EPJ D (uncertainty protocol).

### Rank 8 — Wire ALIAS per-ionization-stage Boltzmann R² (ID)
- **file:line:** `cflibs/inversion/identify/` ALIAS Boltzmann fit (pools all ion stages into one regression → pooled R² collapses ~0.012 while per-stage is 0.997/0.999); fixed `R² ≥ 0.85` hard gate then rejects valid ps-LIBS fits (ALIAS-BOLTZ-IONMIX-1 / ALIAS-R2GATE-2, 2026-06-03 diagnosis).
- **kind:** wrong-denominator gate — fixed R² threshold with no literature basis collapses valid multi-stage fits.
- **lift:** MEDIUM (recall-bound). +0.05–0.12 alias macro-F1; gates whether Fe-group elements survive identification at all.
- **effort:** small–medium.
- **literature:** literature uses physical line-selection + WLS, not a blanket R² reject (Poggialini 2023 JAAS D3JA00130J).

### Rank 9 — Closure sanity guard / degeneracy detector (COMP)
- **file:line:** `cflibs/inversion/physics/closure.py`; `IterativeCFLIBSSolver.solve` quality metrics (`quality_metrics['boltzmann_r_squared']` and `n_elements_fit` return `None`).
- **kind:** missing guard / uninformative metrics — solver reports `converged=True`, `lte_mcwhirter_satisfied=True` on a garbage single-element-dominated answer (Q5-Metrics).
- **lift:** LOW direct composition lift, HIGH diagnostic value — would have caught Na=98 % and Mn=38 % automatically and is the acceptance-gate infrastructure for everything else.
- **effort:** small.
- **literature:** von Toussaint 2018 arXiv:1805.08301 (validation must be non-tautological).

### Rank 10 — Wavelength calibration + baseline propagation into matching (DET)
- **file:line:** `cflibs/inversion/preprocess/` (wavelength calibration not run before detection — DET-4); baseline subtraction in peak detection not propagated to line-matching (DET-3).
- **kind:** preprocessing gaps — uncorrected spectrometer drift propagates into all matches; baseline-embedded noise peaks matched as transitions.
- **lift:** LOW-MEDIUM (matrix-dependent; reduces spurious matches feeding Mn/Ti over-attribution).
- **effort:** small.
- **literature:** standard LIBS preprocessing (Poggialini 2023 review).

### Rank 11 — Hermann two-region (core + corona) SolverStrategy (CF)
- **file:line:** Bayesian `TwoZoneBayesianForwardModel` at `cflibs/inversion/solve/bayesian.py:2408` exists; deterministic Hermann two-region `SolverStrategy` **not** implemented (CF-LIBS-improved-mgp5). Production solver is single-zone.
- **kind:** missing physics model — single-zone ignores the cool absorbing border that self-absorbs major resonance lines.
- **lift:** ~2× oxide improvement reported, but **heavily overlaps B1 self-absorption** (both model the cool border). Defer: single-zone + SA captures most of the gradient-border error first.
- **effort:** LARGE (forward-model rewrite + nonlinear fit).
- **literature:** Hermann 2010/2014 Spectrochim. Acta B 100, 189 (~2× oxide); Anderson 2021 Spectrochim. Acta B 188, 106347 (SuperCam basalt).

### Rank 12 — Continuum emission (bremsstrahlung + free-bound) (CF)
- **file:line:** forward model `cflibs/radiation/` and inversion loop have no continuum term (DET-5 / D2-CONTINUUM-EMISSION).
- **kind:** missing physics — violates energy-balance closure in high-density regions; baseline noise masquerades as line intensity.
- **lift:** MEDIUM but matrix/density-dependent; mostly a forward-model fidelity / round-trip concern.
- **effort:** LARGE.
- **literature:** standard plasma-emission modeling; Poggialini 2023.

### Rank 13 — Partition-function regen for Fe/Ti/Cr (CF)
- **file:line:** `cflibs/plasma/partition_functions.py` polynomial U(T) is 1.7–2.6× too low at LIBS T for Fe/Ti/Cr (defect C), skewing inter-element ratios −31 %/+60 % on JAX/Bayesian/forward paths.
- **kind:** atomic-data error — affects JAX paths only (default CPU solver is direct-sum-immune).
- **lift:** ~0 F1 on default path; composition-correctness on JAX paths only.
- **effort:** medium (ingest Barklem & Collet 2016).
- **literature:** Barklem & Collet 2016 A&A 588, A96.

### Rank 14 — `cflibs analyze --format json` crash (PROD)
- **file:line:** `cflibs/cli/main.py:412` (`json.dumps(output, indent=2)` with no numpy-bool coercion; a numpy-bool in `quality_metrics` is not JSON-serializable → "Object of type bool is not JSON serializable").
- **kind:** productionization bug — table/csv paths work, json crashes.
- **lift:** ~0 on accuracy, but blocks machine-readable output / regression harnessing.
- **effort:** trivial (custom `default=` encoder or coerce numpy scalars).
- **literature:** N/A.

---

## 4. Literature Target and Our Gap

**What good CF-LIBS achieves on basalts / geostandards (by tier):**

| Tier | Result | Source |
|------|--------|--------|
| Best-case majors | trueness "better than 5 wt%" on majors | De Giacomo 2007 (meteorites) |
| Best-case majors | "better than 1 wt%" on a major | Cavalcanti OPC 2013 |
| Realistic target | average relative error **< 10 %** on fused glass | Safi 2019 (extended C-σ) |
| Synthetic floor | ~1 % rel. (Si worst at 3.74 %) — authors warn real is worse | Demidov 2022 PMC9573556 |
| Uncorrected SA | 30–250 % when SA neglected | Aragón (Al-into-vacuum) |
| Self-absorption prize | Mg-Ca 50:50: **27 % → 2 %** absolute when SA corrected | John & Anoop 2023 RSC Adv. 13, 29613 |
| Production reality | ChemCam/SuperCam use **calibration models** (69→408 standards), not pure CF-LIBS | Clegg 2017 SAB 129, 64; Anderson 2021 SAB 188, 106347 |

**North-star target:** ≤ **10 %** average relative error on majors (Safi 2019); aspirational ≤ 5 % (De Giacomo 2007). Adopt the **John & Anoop Mg-Ca 50:50 → 2 %** synthetic regression and a **BHVO-2 ≤ 10 % avg relative** real-sample regression as acceptance gates.

**Our gap:**

- BEFORE (analyze_cmd baseline): **RMSE 33.69 wt%** — far outside any literature-acceptable band; a degenerate Na=98 % non-answer (uncorrected-SA regime, 30–250 %, Aragón).
- AFTER (invert, SA on): **RMSE 11.97 wt%** — now within striking distance of the band but not there; ~1.2× over the 10 % target *and only on majors that survived detection*. Al/K/P/Mg are still hard-zero, so the true gap is larger than RMSE alone shows.
- The realistic literature floor on actual geostandards (well-tuned pure CF-LIBS) is **10–20 % relative on majors**, with **Si the perennial worst performer** (self-absorption + A_ki grade) — exactly our pattern (Si tops out at 27 % vs cert 42 %). Closing the last gap to ≤10 % requires Ranks 1–4 (recover dropped majors + fix Si stage + Mn/Ti over-attribution), then Ranks 5–7 (Saha/charge-balance correctness) to push surviving majors toward the 5–10 % floor.

---

## 5. What Was Checked and Found OK

- **Plasma parameters are sane.** T ~10.3–14.4 kK (0.9–1.24 eV), n_e ~2.5–3.0e17 cm⁻³ — both physically plausible for ps-LIBS. Solver converges in 3–6 iterations; `lte_mcwhirter_satisfied=True`. **The Saha/Boltzmann math is not the problem** — the failure is upstream in detection/identification (and partially in misassignment), not in the core inversion solver.
- **The B1 self-absorption wiring is correct and net-positive on real thick data.** SA-on improves RMSE 14.83 → 11.97 on BHVO-2 (opposite of its over-correction on thin synthetic data, consistent with the authors' thin-data caveat). Bulajic-2002 recursion recomputes τ from plasma state each iteration (`iterative.py:1006`).
- **B3 / B2 fixes verified present.** CDSB `number_densities` arg wired (`cdsb.py:267,333`); resonance retention coupled to `apply_self_absorption` (`main.py:264`).
- **aki_uncertainty inverse-variance Boltzmann weighting is on by default** (`boltzmann.py:119`), already addressing part of literature item "Boltzmann weighting".
- **Pipeline runs end-to-end with no crash** on all 3 spectra via the table/direct path. Only `--format json` crashes (Rank 14), which does not block obtaining the composition.
- **Self-absorption is NOT the binding constraint at the baseline.** The pipeline never gets far enough for SA to matter when 5 of 10 elements are dropped at detection — confirming the audit ordering: detection/ID first (Ranks 1–4), physics second.
- **The Na blowup root cause is confirmed and fixed.** Spurious high-lying Na I 413–421 nm Rydberg lines (A_ki ~1e5) were claiming bright peaks; the `min_relative_intensity=100` floor prunes them exactly as designed (Na 97.90 → 0.00).

---

### Acceptance gates to adopt
1. **BHVO-2 real-sample regression:** ≤ 10 % avg relative error on majors (Safi 2019) on `data/bhvo2_usgs/chemcam_bhvo2_loc{1,2,3}` via the unified CLI path.
2. **John & Anoop Mg-Ca 50:50 synthetic regression:** corrected composition within 2 % absolute (validates B1 SA without thin-data over-correction).
3. **No-degeneracy guard:** no single element > 80 % unless certified (would have caught Na=98 %, Mn=38 %).

### Citation index
Bulajic 2002 SAB 57, 339 (DOI 10.1016/S0584-8547(01)00398-6) · El Sherbini 2005 SAB 60, 1573 · Pace 2025 SAB (doublet-ratio) · John & Anoop 2023 RSC Adv. 13, 29613 · Poggialini/Palleschi 2023 JAAS D3JA00130J · IRSAC RSC Adv. 2026 D6RA01889K · Tognoni 2010 SAB 65, 1 · Aguilera & Aragón 2004 SAB 59, 1861 · Ciucci 1999 Appl. Spectrosc. 53, 960 · Hermann 2010/2014 SAB 100, 189 · De Giacomo 2007 · Cavalcanti OPC 2013 · Safi 2019 · Demidov 2022 PMC9573556 · Clegg 2017 SAB 129, 64 · Anderson 2021 SAB 188, 106347 · Kramida 2024 NIST EPJ D · Wiese & Fuhr 2009 JPCRD 38, 565 · Barklem & Collet 2016 A&A 588, A96 · von Toussaint 2018 arXiv:1805.08301.

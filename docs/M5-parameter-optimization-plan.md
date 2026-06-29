I'll synthesize the three designs into a unified plan. The designs are self-contained and consistent, so I'll produce the markdown directly.

# Unified Plan: From Magic Numbers to a Principled, Adaptive, Optimizable CF-LIBS Pipeline (M5)

The three designs converge on one architecture, not three competing ones. The first-principles derivation tells us *which* knobs are physically determined; the adaptive-parameterization design tells us *which* should track a runtime feature; the joint-optimizer design tells us *how* to safely tune what's genuinely left over — and the framework for all of it (`scripts/campaign1/` Optuna + scoreboard + tier splits) already exists. The unifying move is a **three-tier routing of every parameter**, then a small offline optimizer over only the residual free coefficients.

---

## 1. Three-Tier Routing of Every Parameter

Each parameter is classified as exactly one of: **DERIVE** (fixed by a physics formula), **ADAPT** (`param = f(runtime feature)`, a bounded interpretable rule), **JOINTLY OPTIMIZE** (genuinely free, tuned offline), or **DROP** (study-confirmed inert — pin at default, remove from the search space).

| Parameter | Default | Routing | Formula / Feature / Rule | Range or coeff | Source of truth |
|---|---|---|---|---|---|
| **min_lines_per_element** | 3 | **DERIVE** → acceptance criterion | Replace integer floor with master formula: accept element iff achieved `σ_T/T = (T·k_B)/(SNR·s_E·√(N−1)) ≤ target`. Hard statistical floor `N≥3` (slope+intercept = 2 DoF + 1 residual). | target σ_T/T ≈ 5%; floor N≥3, cap N≤5 (cliff past 5) | master formula; `boltzmann.py:1291` |
| **min_energy_spread_ev** | 2.0 | **DROP** (subsumed by DERIVE) | Falls out of the same σ_T/T criterion (`s_E ≥ (T·k_B)/((σ_T/T)·SNR·√(N−1))`). Study-confirmed inert; do not tune separately. | pinned / removed | master formula |
| **min_snr** | 10 | **DROP** (subsumed by DERIVE) | Same criterion; enters linearly so N dominates. Study-confirmed inert on this corpus. Keep only as a fixed detection-FPR guard (`SNR ≳ Φ⁻¹(1−α/N_pix)`). | pinned ~10 | master formula |
| **wavelength_tolerance_nm** | 0.1 | **ADAPT** | `clip(k_tol·λ_mid/R, 2·Δλ, tol_max)`; better form `sqrt((λ/R)² + σ_cal²)` using `cal.rmse_nm`. Recovers the confirmed supercam 0.1→0.2 retune "for free" as 1 FWHM. | k_tol≈1.0, tol_max≈0.3 | `line_detection.py:594` `_resolve_adaptive_tolerances` |
| **peak_width_nm** | 0.2 | **ADAPT** | Same FWHM rule: `clip(k_w·λ_mid/R, 2·Δλ, ...)`; integration half-width ~2–3× FWHM. Already R-aware. | shared k_w≈2 | `line_detection.py:613` |
| **isolation_wavelength_nm** | 0.1 | **ADAPT** | `k_iso·FWHM = k_iso·λ_mid/R`. Makes the blend kernel `1−exp(−Δλ/λ_iso)` measure separation in FWHM units. | k_iso≈1.0 | `line_selection.py:261/504` |
| **min_peak_height** | 0.01 | **ADAPT** | `max(h_floor, k_h·σ_noise/I_max)` using the per-spectrum MAD σ from `estimate_noise`. Floored at 0.01 → cannot regress the clean-spectrum optimum; rises on noisy field spectra. | k_h≈3–5, h_floor=0.01 | `preprocess/preprocessing.py:321` |
| **residual_shift_scan_nm** | 0.0 | **ADAPT** (drift-gated) | `cal.rmse_nm > τ_drift ? k_s·cal.rmse_nm : 0`. Directly resolves the documented supercam(+)/bhvo2(−) shift-scan confound. | τ_drift, k_s | `pipeline.py:620` |
| **max_lines_per_element** | 20 | **JOINTLY OPTIMIZE** | No physics floor — robustness/compute cap; saturated plateau. Implicitly `min(K, n_avail)`. | 12–40 | knob_space |
| **min_relative_intensity** | None | **JOINTLY OPTIMIZE** (likely keep None) | Catalog-strength floor, not a spectrum statistic. Any nonzero value is a one-sided cliff. | None / explore | knob_space |
| **shift_coherence_veto / line_residual_gate / stark_ne / exclude_resonance** (gates) | bool | **JOINTLY OPTIMIZE** (booleans, default-locked) | On/off policy switches; all-default is at/near Pareto-optimal. Their *internal* cutoffs are the real DERIVE-able knobs (not yet config-exposed — future work). | keep default | study §5 |
| **affine_coverage_gate / grade_aware_selection / peak_width_mode** | bool/enum | **DROP** | Bit-identically inert on this corpus. Pin at default; remove from search space. | pinned | study (5 inert knobs) |
| **wavelength_calibration** (gate) | True | **N/A — always-on input-quality lever** | Not a tunable number. | — | — |

**Net effect:** ~5 magic numbers (`min_lines`, `min_energy_spread`, `min_snr`, plus the 3 inert gates) collapse into **one physics target** (σ_T/T) and a set of **DROP** pins; 4 detection thresholds become **ADAPT** rules with ~6 shared coefficients; only **~3–5 genuinely free knobs** remain for the offline optimizer.

---

## 2. Optimizer + Tier-Based Protocol

**Optimizer — Optuna TPE (multivariate, `group=True`) as phase A; CMA-ES as phase B.**

- TPE natively handles the mixed continuous + int + categorical + **conditional** residual space, is sample-efficient for multi-minute evals, and is **verified import-clean of sklearn** (lives only in `scripts/`, never under `cflibs/`, so TID251 is untouched). The M5 landscape is *saturated plateaus with one-sided cliffs* — TPE tolerates this far better than CMA-ES, which wastes covariance adaptation on a flat plateau.
- **CMA-ES (numpy `cma` or Optuna `CmaEsSampler`)** is phase B only, restricted to the continuous coefficient subspace (`k_tol`, `k_iso`, `k_h`, `k_s`, `τ_drift`, `tol_max`) after TPE has frozen the discrete choices. This is the natural fit once everything is reduced to bounded continuous coefficients.
- **Rejected:** scikit-optimize (pulls sklearn — banned); Nevergrad (viable but redundant with the wired Optuna); `cflibs/evolution/` (mutates *code source*, not a typed numeric vector — wrong tool for threshold tuning).

**Objective — single composite scalar with a cross-dataset variance penalty (not full Pareto).** Reuse `objective.compute_fitness`, re-weighted for M5:

```
score_d  = 0.2·F1_d + 0.8·(1 − min(RMSE_d,10)/10)     # RMSE-first; F1 as death-penalty guard
fitness  = Σ w_d·score_d / Σ w_d                        # weighted mean across datasets
           − 0.5·Var_d(score_d)                         # anti-tug-of-war (supercam vs bhvo2)
           − 0.2·max(0, t_med/5 − 1)                    # runtime budget
```

- Flip `F1_WEIGHT/RMSE_WEIGHT` from 0.6/0.4 → **0.2/0.8** (module constants in `objective.py`; recorded in the frozen manifest).
- **Keep the `−0.5·Var_d` penalty** — it is the principled answer to the supercam/bhvo2 tug-of-war (a candidate that wins on one by wrecking the other is punished automatically, e.g. it kills `residual_shift_scan=0.1`).
- **Single scalar, not NSGA-II Pareto** — overkill for 2 datasets; the mean−λ·variance already encodes "improve the worst dataset," and the adoption tiebreak uses **best worst-dataset score** (the Pareto-min you actually want). Reserve Pareto only if a third pulling dataset appears.
- Use **median** RMSE per dataset (robust to one blown spectrum).

**Tier protocol (optimization / holdout / vault = train / val / test), structurally enforced:**

1. **Tune on the optimization tier only.** `objective.evaluate_overrides(section="optimization")` raises `HoldoutViolation` if any holdout/vault id leaks in. Tune on element-wt optimization datasets (`chemcam_calib`/`silva2022` train splits + `synthetic_fixedforward`).
2. **Validate on holdout — quota 1/week** (`enforce_holdout_quota`, human `--force` to exceed). This is the anti-overfit kill-switch: you cannot iterate against holdout, so it stays a true validation set. bhvo2/supercam-class spectra are the **holdout adoption gate**, *not* a tuning target.
3. **Final number on vault (gibbons2024) — one human run, end of program.** The honest generalization estimate.
4. **Report the train–test RMSE gap** (`optimization_RMSE − holdout_RMSE`) in `holdout_verdict.md`; a large positive gap is the overfit alarm.

**How many free params are safely fittable.** With ~4 spectra/dataset (~8–12 effective), the defensible budget is **≤ 3–5 free continuous params**. The routing above achieves exactly this: DROP removes 5 inert knobs; DERIVE collapses 3 into one target; ADAPT moves 4 thresholds into ~6 coefficients that are **fit per-instrument** (frozen, not per-spectrum DoF). The residual **JOINTLY OPTIMIZE** set is ~3–5 knobs — small enough that 150–300 TPE trials genuinely saturate it, and 2D interaction sweeps (`min_lines × max_lines`, `tolerance × residual_shift`) fall out for free.

**Missing piece to build first:** `objective.paired_bootstrap_delta_f1` is F1-only. Add `paired_bootstrap_delta_rmse` (~30 lines, mirrors the F1 version, 2000 resamples, adopt only if the 95% CI excludes 0). Set the adoption ΔRMSE threshold above the per-spectrum bootstrap noise floor (mirror `MIN_DELTA_F1=0.02`). **No sub-0.1-RMSE-unit win is trustworthy.**

---

## 3. Physics-Only Compliance + Interpretability

Both the optimizer and the shipped adaptive rules stay strictly inside the hard constraint:

- **Offline tuning never touches `cflibs/`.** Optuna/CMA-ES live in `scripts/campaign1/`; they pass numbers through the top-precedence `config_overrides` tier of `build_pipeline_config`. The shipped pipeline imports no optimizer. (`cflibs/evolution/` remains the *only* ML-permitted zone and is reserved for code-structure search, a separate layer with the same scoreboard backend and the same AST blocklist guarantee — not used for this numeric HPO.)
- **The adaptive rules are interpretable scaling laws, not learned models.** Every ADAPT rule has the shape `param = clip(k·feature, lo, hi)` — closed-form numpy, the same idiom as the existing `_resolve_adaptive_tolerances`. No `sklearn`/`torch`/`jax.nn`. The coefficients `k` are dimensionless physics multiples (FWHM-multiples, sigma-multiples); `lo`/`hi` are safety rails (Nyquist floor, legacy-constant fallback).
- **The winning vector ships as plain preset numbers** in `cflibs/inversion/pipeline.py` with a provenance comment (study dir, trial #, frozen-manifest sha256) — no learned weights, no optuna import in the shipped path.
- **The `None` sentinel is the wiring mechanism** (`pipeline.py:203`): `None` → adaptive rule fires; a float → user pin overrides (preserves reproducibility and the existing `_OVERRIDABLE_FIELDS` precedence).
- **DERIVE keeps the same property:** replacing `min_lines` with a σ_T/T acceptance test is *more* physics-interpretable than an integer magic number — it ties directly to the exact-propagation `σ_T = T²·k_B·σ_m` already in `boltzmann.py`.

This satisfies "shipped inference + adaptive rules stay physics-interpretable (rules, not learned black-box)" by construction at every layer.

---

## 4. Concrete First Prototype (smallest end-to-end step)

Goal: prove the loop and recover the confirmed manual retune **without** touching `cflibs/` physics yet.

```
# 0. One-time setup (scripts/ only):
#    - Reduce knob_space to the M5 sub-set; pin the 5 inert knobs at default (DROP).
#    - Flip F1_WEIGHT/RMSE_WEIGHT 0.6/0.4 -> 0.2/0.8 in objective.py.
#    - Add paired_bootstrap_delta_rmse + RMSE columns/gap line in holdout_verdict.md.

# 1. Init study; seed baseline + the confirmed retune (min_lines=2, wavelength_tolerance=0.2)
#    + looser-gates hypothesis as enqueue_trial seeds (NOT discovered by tuning on bhvo2).
JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/campaign1/driver.py init \
    --study-dir output/campaign1/m5-rmse --db ASD_da/libs_production.db \
    --datasets chemcam_calib,silva2022,synthetic_fixedforward \
    --target-trials 300 --fitness-version 2 --enqueue-from m5_seeds.json

# 2. Jointly optimize the ~5 load-bearing knobs (phase A TPE; CMA-ES phase B on the
#    continuous coefficient subspace) on the OPTIMIZATION tier only.
sbatch --array=0-15%16 --cpus-per-task=8 scripts/campaign1/slurm/worker_array.sbatch

# 3. Validate top-K + baseline on holdout (bhvo2/supercam-class), quota 1/week:
#    adoption gate G1-G4 + NEW RMSE bootstrap CI; pick best worst-dataset score;
#    report optimization_RMSE - holdout_RMSE gap.
sbatch scripts/campaign1/slurm/holdout_eval.sbatch  # TOP_K=5

# 4. Vault (gibbons2024): one human run at end-of-program.
# 5. Adopt: ship the winning override dict as plain preset numbers in pipeline.py.
```

**First ADAPT rule to land (phase 2):** make the existing R-derived FWHM tolerance the *default* (not opt-in) for `wavelength_tolerance_nm`/`peak_width_nm`/`isolation_wavelength_nm`. This **provably recovers the confirmed supercam 0.1→0.2 nm (−12% RMSE) retune** as "tolerance = 1 instrumental FWHM," generalizing to instruments the OAT sweep never saw — and it adds **zero** new optimizer DoF on the static path (coefficients `k≈1`, floored at the legacy constant).

**Expected gain vs the confirmed manual retune:**
- The static joint optimization should **match or modestly beat** the manual supercam −12% RMSE by finding the `min_lines × tolerance` interaction the OAT sweep couldn't see — but treat any gain beyond the manual retune that's < the ΔRMSE bootstrap floor as **noise, not signal**.
- The larger, defensible gain is from **ADAPT**: the R-derived tolerance recovers the −12% *per-instrument automatically*, and the drift-gated shift-scan resolves the supercam(+15%)/bhvo2(−48%) confound into a single rule — landing *each* dataset near its own optimum without a per-dataset compromise. That is the real win the static retune cannot capture.

---

## 5. Honest Caveats

1. **Small-N overfit is the dominant risk.** ~4 spectra/dataset cannot support 12–46 free params; the −0.056 bhvo2 "gain" in the study is within plausible small-N noise. **Mitigations:** the routing's ≤3–5-free-param budget, per-instrument (not per-spectrum) coefficients, the bootstrap-CI adoption gate, the one-shot vault, and a **zero-regression rule** (reject any candidate/coefficient set that regresses *any* optimization dataset vs the legacy constants). Nested CV is *not* worth it at n=4/dataset (inner folds n=2–3) — revisit only on a larger corpus.

2. **Tier misclassification trap.** The M5 sensitivity datasets bhvo2_chemcam/supercam are **holdout-only** in `splits.py`. Tuning RMSE directly on them is tuning against the validation set (forbidden). Tune on optimization-tier element-wt datasets; enter the confirmed `(min_lines=2, tol=0.2)` retune as a **seed trial validated through the gate**, never as a tuning target.

3. **Dataset tradeoff is real; a single global static config is a compromise.** The variance penalty handles the tug-of-war, but the remaining gains genuinely live in **per-calibration-quality adaptivity** — which is why the ADAPT tier (drift-gated shift-scan keyed on `cal.rmse_nm`, FWHM-keyed tolerance/isolation) matters more than squeezing the static vector. **Verify τ_drift cleanly separates bhvo2 (low rmse) from supercam (high rmse)** with a 2-point check before trusting it; if `cal.rmse_nm` doesn't separate them, fall back to scan=0.0 globally (the study's safe default).

4. **DERIVE rests on assumptions that can break.** The master formula assumes single-temperature LTE and `σ_y≈1/SNR` (Poisson-dominated). Self-absorption, non-LTE, or correlated calibration errors break the σ_y model, so a σ_T/T gate could spuriously reject good elements or admit biased ones. **Validate the target-σ_T gate against the bhvo2/supercam corpus before replacing the integer floor.** Note `s_E` must use the actual std-dev/Var(E_k) from the slope variance, not the current max−min range check (`line_selection.py:344`) — or apply a √12 uniform-fill correction.

5. **When adaptivity helps vs. adds noise.**
   - **Helps:** when the optimum *provably tracks a measurable condition* — R/FWHM (tolerance, isolation, width), per-spectrum noise σ (peak height), calibration residual (shift-scan). Floored at the legacy constant, the adaptive path *cannot* regress the validated clean-spectrum optimum and degrades to byte-identical legacy behavior when the feature is unavailable (R=None, calibration failed).
   - **Adds noise:** per-*spectrum* coefficients can inject between-spectrum variance into a composition map. Keep **coefficients per-instrument/per-campaign, only the feature per-spectrum**, and clip every rule. **Do NOT make `min_lines_per_element` adaptive** — it is the #1 fragility (RMSE up to 12.7; tightening past 5 catastrophic), and a per-element adaptive floor that raises it on line-rich elements would re-trigger Boltzmann-fit starvation. Keep it a constant floor (2–3); the max-cap self-adapts via `min(K, n_avail)`.

6. **The scheme reduces, but does not eliminate, tuning.** σ_T/T target, k_h, and the R/σ_cal blend constants are themselves new free parameters — ~5 magic numbers become ~2–3 physically-interpretable ones plus ~6 bounded coefficients. Every one must be **benchmark-gated via `run_scoreboard(config_overrides=...)`** on the optimization tier with holdout/vault validation before landing, and confirmed on a larger corpus before becoming a shipped default (the study's standing small-N caveat).
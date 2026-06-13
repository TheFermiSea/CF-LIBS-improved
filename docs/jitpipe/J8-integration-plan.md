# J8 Integration Plan — End-to-End Single-Spectrum jit Pipeline (Milestone M1)

**Status:** pre-wiring audit + composition plan. **Bead:** J8. **ADR:** [ADR-0004](../adr/ADR-0004-jittable-inversion-pipeline.md) §5.1/§5.2/§5.4; **Spec:** [`specs/J8-end-to-end-m1.md`](../adr/specs/J8-end-to-end-m1.md).

This document is the concrete plan to compose the nine independently-built stage
kernels (J1–J7, J10, J11) into one jitted single-spectrum graph (`run_one`) plus
the host glue that reconstitutes the reference result dataclasses. It is produced
from a read-only audit of the integrated `integ-jfan` worktree against the
frozen reference (`cflibs/inversion/pipeline.py:785` `run_pipeline`).

---

## 0. Headline finding — the orchestrators are still J0 stubs

The nine stages were each built **inside their own module as low-level kernels
with explicit padded-array signatures**, parity-tested directly. But every
**high-level orchestrator entry point that carries the `(snapshot, params,
static)` signature is still a J0 skeleton that `raise NotImplementedError`:**

| Module | Stub entry (raises) | Real kernel surface (implemented + parity-tested) |
|---|---|---|
| `pipeline.py` | `run_one`, `run_batch` | — (this is the J8 deliverable) |
| `forward.py` | `forward_spectrum` | (bridges to `radiation.kernels.forward_model` via `snapshot.to_atomic_snapshot`) |
| `calibrate.py` | `calibrate_wavelengths` | `calibrate_axis_kernel` |
| `identify.py` | `identify_lines` (this stub is for J10 presence-scoring) | `score_comb_grid` → `select_shifts` → `select_accepted_mask` → `kdet_keep_mask` → `shift_coherence_veto` → `build_observations` → `extract_intensity_trapezoid` |
| `fit.py` | `boltzmann_fit` | `select_lines`, `sb_graph_fit`, `common_slope_fit`, `sigma_clip_fit`, `closure_*`, `keystone_degenerate`, `masked_median` |
| `selfabs.py` | `correct_self_absorption` (5-arg adapter) | `correct_self_absorption_arrays` |
| `solve.py` | `iterative_solve` (explicitly says "host wrapper lands at J8") | `LaxKernelInputs.from_snapshot`, `scan_solve`, `joint_wls_solve` |
| `stark.py` | — (`stark_electron_density` IS implemented) | `measure_stark_ne_jit`, `stark_ne_from_widths`, `extract_windows`, `multiplet_blend_mask`, `stark_electron_density` |
| `preprocess.py` | — (all kernels implemented) | `snip_baseline`/`als_baseline`/`median_baseline`/`percentile_baseline`, `estimate_noise`, `robust_normalize` |

**Implication:** J8 is not merely "call the seven orchestrators in order." J8 must
(a) write the orchestration body for each stub by calling the real kernels, OR
(b) skip the per-stage orchestrators and call the kernels directly from
`pipeline.run_one`. The cleaner path is (b): `run_one` calls the kernels directly
(the kernels are the parity surface; the stub orchestrators were placeholders).
The stubbed `(snapshot, params, static)` orchestrators can either be deleted or
filled in as thin wrappers — but `run_one` should not depend on them.

This is the single largest piece of work and the reason none of the stages today
read a `PipelineParams` object (see §3).

---

## 1. Stage-by-stage data-flow graph (single-spectrum `run_one`)

Mirrors the reference stage order `run_pipeline`: response-multiply → calibrate →
detect → identify → (self-absorption) → stark → fit → solve. Shapes use the
shapes.md constants (`P_max`, `L_max`, `E_max`, `K_comb`, `W`, `C`, `N_wl`).
**Bold arrows** are the device data flow; *italic* notes are host glue (§2).

### Stage chain

```
                 raw (intensities (N_wl,), wavelengths_nm (N_wl,))
                          │  HOST: response-curve multiplier array (per-channel), §2.A
                          ▼
[J1 preprocess]   intensity_corr = intensity * response          (N_wl,) f64
   snip/als/median/percentile_baseline(intensity) -> baseline    (N_wl,) f64
   estimate_noise(intensity, baseline)            -> noise        scalar f64
   (robust_normalize for the detection-path height)
                          │
        ┌─────────────────┴───────────────────────────────┐
        ▼ calibration path                                 ▼ detection path
[J2 detect, calibration param.]                  [J2 detect, detection param.]
  detect_peaks_calibration(intensity,            detect_peaks_detection(intensity,
     baseline, noise, distance_px=...)              min_peak_height, distance_px=...)
   -> PeakResult{indices (P_max,) i32,             -> PeakResult (P_max,) (P_MAX_DETECTION)
       mask (P_max,) bool, count, truncated}
        │  HOST: peak_wl = wavelengths[indices]              │ HOST: same gather
        ▼                                                    │
[J3 calibrate]                                               │
  calibrate_axis_kernel(peak_wl, peak_amp, peak_mask,        │
     line_wl, line_strength, line_mask, wavelength, wl_mask) │
   -> CalibrationKernelResult{corrected_wavelength (W_max,), │
       coefficients (3,), model_id, quality_passed, ...}     │
        │  corrected axis feeds the detection-path λ axis ───┘
        │  HOST: build FrontEndSnapshot (§2.B) — the big one
        ▼
[J4 identify]  snap: FrontEndSnapshot{peak_* (P_max,), comb_* (E_max,K_comb), element_mask (E_max,)}
  score_comb_grid(snap, shift_grid (S,), tolerance_nm, comb_min_*) -> CombScoreArrays{f1,precision,
       recall,matched_lines,passes : (S,E_max)}
  select_shifts(scores, shift_grid, element_mask)  -> (best_idx, fb_idx, best_has_pass, total_f1)
  select_accepted_mask(scores, best_idx, fb_idx, ...) -> (accepted_mask (E_max,), applied_idx)
  kdet_keep_mask(snap, ...)                        -> keep (E_max,) bool
  shift_coherence_veto(snap, ...)                  -> veto-adjusted accepted (E_max,)
  build_observations(snap, f1, matched_lines, accepted_mask, shift_nm, tolerance_nm, ...)
       -> ObservationBuild{obs_valid (E,K) bool, obs_peak_slot (E,K) i32, n_gated (E,), dropped (E,)}
  extract_intensity_trapezoid(...)                 -> per-obs integrated intensity + Poisson sigma
        │  HOST: gather valid obs -> padded per-line arrays for the line axis
        │        (the (E, N_max) Boltzmann block + line_index into snapshot)
        ▼
        │  *** FAILURE GATE *** if Σ obs_valid == 0 -> failed mask (§4, AC4)
        ▼
[J5 self-absorption]  (only when static.apply_self_absorption)
  correct_self_absorption_arrays(intensity (L,), intensity_unc (L,), aki_unc (L,),
       line_index (L,), line_valid (L,), line_element (L,), pair_idx (P,2), pair_valid (P,), snapshot)
   -> SelfAbsorptionResult{intensity (L,), intensity_unc (L,), tau, method, suspect, n_corrected,...}
        │  corrected intensity + inflated sigma replace the obs intensities
        ▼
[J6 stark n_e]
  HOST: extract_windows + multiplet_blend_mask + candidate selection (SNR/isolation/source-class)
  measure_stark_ne_jit(wl_win (C,W), inten_win (C,W), mask (C,W), center0 (C,), gaussian_fwhm_nm (C,),
       stark_w_ref_nm (C,), stark_alpha (C,), candidate_mask (C,), T_K)
   -> StarkNeResult{ne_median scalar, ne_scatter, valid (C,), n_lines}
        │  ne_median pins the solve (audit F2); NaN -> solver fallback path
        ▼
[J7a fit (Boltzmann/SB-graph)]
  HOST: build Boltzmann ordinates y = ln(I λ / (g A)), x = E_k, per-element (E, N_max) block
        from corrected obs intensities + snapshot atomic data (this is glue, not a kernel)
  select_lines(wavelength,intensity,intensity_unc,atomic_unc,element_idx,is_resonance,mask,
       n_elements=E, min_snr, isolation_scale_nm, energy_ev, top_k)
   -> {selected_mask (B,L), topk_index (B,E,top_k), topk_valid, spread_ev, n_valid}
  (sb_graph_fit / common_slope_fit / sigma_clip_fit are the fit primitives; in the
   production geological path the SB-graph runs INSIDE scan_solve — see §5 note)
        ▼
[J7b solve]
  LaxKernelInputs.from_snapshot(lax_snapshot, x (E,N_max), y, w, stage, mask)   <- snapshot.to_lax_snapshot(elements)
  scan_solve(inp, init_T_K=1e4, init_ne_cm3=ne_median or 1e17, max_iters=static.max_iters,
       closure_mode=static.closure_mode, oxide_factors=..., t_tol_k, ne_tol_frac, pressure_pa, min_r2)
   -> LoopState{T_K, n_e_cm3, concentrations (E,), r_squared, converged, i, ...}
  (optional) joint_wls_solve(inp, n_e_cm3=ne_median, closure_mode, ...) -> JointWLSResult (production estimator)
        │  HOST: reconstitute CFLIBSResult (§2.D)
        ▼
                  CFLIBSResult{temperature_K, electron_density_cm3, concentrations: dict, ...}
```

### Shape-handoff table (device boundaries that must agree)

| Producer → Consumer | Producer output | Consumer expects | Status |
|---|---|---|---|
| preprocess → detect | `baseline (N_wl,) f64`, `noise scalar f64` | `detect_peaks_calibration(intensity, baseline, noise, ...)` | OK |
| detect → calibrate | `PeakResult.indices (P_max,) i32` + `mask` | calibrate needs `peak_wl (P_max,) f64` + `peak_amp` + `peak_mask` | **GAP (host gather), §2.B** |
| calibrate → detect/identify | `corrected_wavelength (W_max,) f64` | identify needs `peak_wavelength_nm (P_max,)` (= corrected_wl[indices]) | **GAP (host gather)** |
| identify → fit | `ObservationBuild.obs_valid (E,K)`, `obs_peak_slot (E,K)` + per-obs intensity | fit needs `(E, N_max)` Boltzmann block + `line_index` | **GAP (host gather/reshape), §2.C** |
| fit → solve | `select_lines.topk_index (B,E,top_k)` | `LaxKernelInputs.x/y/w/stage/mask (E, N_max)` | **GAP — different layout; host gather, §5** |
| stark → solve | `StarkNeResult.ne_median scalar` | `scan_solve(init_ne_cm3=...)` / `joint_wls_solve(n_e_cm3=...)` | OK (scalar pin) |
| selfabs → fit | `SelfAbsorptionResult.intensity (L,)`, `intensity_unc (L,)` | corrected intensities feed the Boltzmann `y` | OK (in-place replace) |
| solve → host | `LoopState` / `JointWLSResult` (E-indexed) | `CFLIBSResult.concentrations: Dict[str,float]` | **GAP (host unpack), §2.D** |

None of these gaps are *blocking mismatches* in the sense of incompatible
shapes that can't be reconciled — they are all **host-glue gathers that do not
exist yet** (the FrontEndSnapshot assembly, the line-block assembly, the result
unpack). See §6 for the one genuine blocker.

---

## 2. Host glue still required (the J8 host surface)

The kernels are device-pure; J8 must add the impure host code in `host.py`
(and complete `pipeline.py` / `parity.py`). None of this exists today.

### 2.A Response-curve multiplier
`run_pipeline:816-820` divides intensity by `SpectralResponseCorrection.apply`.
The per-channel multiplier array is the only thing that crosses to device
(ADR §5.1.3). Host computes it; `run_one` multiplies inside the trace.

### 2.B `FrontEndSnapshot` host assembly — **the largest J8 glue item**
`identify.py:18` and `:50` flag this explicitly: *"The host assembles a
`FrontEndSnapshot` per (dataset × element set)"* and *"the host wrapper that
rebuilds `LineDetectionResult` strings (J8 glue)."* There is **no builder in
`host.py`** today. It must:
1. Run scipy/host catalog SQL for the candidate elements (or gather from the
   `PipelineSnapshot` superset by element mask — ADR §5.1.2 prefers the latter).
2. Reproduce the reference gA-Boltzmann comb ranking (`COMB_STRENGTH_T_REF_EV`,
   strongest-first) to pad `comb_*` to `(E_max, K_comb)`.
3. Gather peak arrays `peak_wavelength_nm/peak_index/peak_mask (P_max,)` from the
   detect `PeakResult.indices` against the *calibrated* wavelength axis.
4. Build `shift_grid (S,)` (`_build_shift_grid`), `element_mask`, `full_n_lines`.

### 2.C Observation → line-block gather
After `build_observations`, the host gathers valid `(e,k)` obs into the padded
`(E, N_max)` line axis the fit/solve kernels consume: `line_index` into the
snapshot, integrated intensity, intensity_unc, `is_resonance`, `element_idx`,
`stage`, `E_k`. Mirrors the reference `_build_padded_arrays_from_obs`
(`iterative.py:2986-3020`) + `snapshot.reorder`.

### 2.D Stark window extraction + candidate selection (host portion)
`stark.py` docstring: *candidate selection (SNR/isolation/source-class/blend
ranking) is the host/snapshot side.* Host runs `extract_windows` +
`multiplet_blend_mask` against the corrected axis and snapshot, gathers
`(C, W)` windows, then calls the device `measure_stark_ne_jit`.

### 2.E `LaxKernelInputs` per-bucket assembly + `to_lax_snapshot`
`snapshot.to_lax_snapshot(elements)` already builds the lax `_AtomicSnapshot`
(implemented, J0). `LaxKernelInputs.from_snapshot` (implemented) consumes it plus
the `(E,N_max)` obs block. Host calls these — only the `(E,N_max)` block (§2.C)
is missing.

### 2.F Result reconstitution → reference dataclasses
J8 spec §2 names three: `WavelengthCalibrationResult` (from
`CalibrationKernelResult`), `LineDetectionResult` (from `ObservationBuild` +
intensities; the warning/diagnostic strings), `CFLIBSResult`
(`iterative.py:81` — `temperature_K`, `electron_density_cm3`,
`concentrations: Dict[str,float]`, `iterations`, `converged`, `quality_metrics`)
from `LoopState`/`JointWLSResult` (E-indexed arrays → element-keyed dicts via
`element_symbols`). `solve.py:695` says this CFLIBSResult unpack *"lands with J8."*

### 2.G Scoreboard `--pipeline {reference,jit}` dispatch
`scoreboard._score_spectrum:211` calls `run_pipeline(...)` unconditionally. J8
threads a `pipeline_kind` through `_score_spectrum`/`run_scoreboard` and, for
`jit`, calls a `run_one`→`(CFLIBSResult, diagnostics)` host wrapper that emits the
identical record schema incl. `stage_timings_s` and `"pipeline":"jit"`. Scoring
logic (`:194` candidate policy, presence rule, confusion, failure policy) is
untouched.

---

## 3. Consistency / convention-drift findings

### 3.1 PipelineParams (audit point 1) — **no stage reads it; the wrapper must map every field by hand**
`rg "params\.<field>"` across all stage modules returns **zero** real reads.
Every implemented kernel takes its knobs as **individual scalar-array keyword
args**, not a `PipelineParams` object. The only `(snapshot, params, static)`
signatures are the NotImplementedError stubs plus `stark_electron_density`
(`del params, static`) and `forward_fit_identify` (reads only
`static.broadening_mode`). Consequences:

- **`PipelineParams` is wired into nothing yet.** J8 owns the entire
  `PipelineParams` → kernel-kwarg fan-out. This is by design (traced leaves flow
  as values) but means the field-name mapping is unwritten and must be exact.
- **Name drift the wrapper must bridge** (PipelineParams field → kernel kwarg):
  - `t_tolerance_k` → `scan_solve(t_tol_k=...)`
  - `ne_tolerance_frac` → `scan_solve(ne_tol_frac=...)`
  - `min_boltzmann_r2` → `scan_solve(min_r2=...)`
  - `isolation_wavelength_nm` → `select_lines(isolation_scale_nm=...)`
  - `top_k_per_element` → `select_lines(top_k=...)` (rounded to int; but `top_k`
    is also `StaticConfig`-shaped — see 3.4)
  - `wavelength_tolerance_nm` → `identify.*(tolerance_nm=...)`
  - `residual_shift_scan_nm` / `global_shift_scan_nm` → identify shift-grid build
  - `pressure_pa`, `min_snr`, `min_peak_height`, `peak_width_nm` → pass straight.
- **Gate knobs absent from PipelineParams.** `comb_min_matches (=3)`,
  `comb_min_precision (=0.02)`, `comb_min_recall (=0.1)`,
  `comb_max_missing_fraction (=0.85)`, `comb_fallback_max_elements (=5)`,
  and calibrate's `inlier_tolerance_nm (=0.08)`, `quality_min_inliers (=12)`,
  `quality_max_rmse_nm (=0.10)` etc. are **not** `PipelineParams` fields. They are
  reference `detect_line_observations` / `calibrate_wavelength_axis` defaults
  (`line_detection.py:1113-1118`). Decision needed: either (a) add them to
  `PipelineParams` (keeps them traced/tunable — preferred for J11 knob-tuning), or
  (b) the host sources them from `AnalysisPipelineConfig` and passes them as
  closed-over constants. M1 parity only requires the reference defaults; (b) is the
  minimal path, (a) is the ADR-§6.3-aligned path.

### 3.2 PipelineSnapshot accessors (audit point 2) — **consistent**
All consumers read snapshot fields by the canonical names in `snapshot.py`
(`line_wavelength_nm`, `line_A_ki`, `line_stark_w`, `line_stark_alpha`,
`species_physics[:,0]=ip`, `oxide_stoichiometry`, `doublet_pairs/rho/r_thin`).
`stark_electron_density` reads `snapshot.line_stark_w[idx]` /
`line_stark_alpha[idx]`; `to_lax_snapshot` reads `partition_coeffs_stored`
(NaN-sentinel, lax convention) while `to_atomic_snapshot` reads `partition_coeffs`
(canonical re-fit) — this dual-poly split is **correct and documented**
(contracts.md eager-fallback section). No drift found.

### 3.3 Padding constants (audit point 4) — **consistent, but two distinct `P_max`**
`detect.py` defines `P_MAX_CALIBRATION` (=2048) and `P_MAX_DETECTION` (=2560),
matching shapes.md (calibration 2048 / detection 2560) and `P_CAND_MAX` (=8192).
identify's `FrontEndSnapshot` uses `(E_max, K_comb)` / `(P_max,)`. solve uses
`(E, N_max)`. No hardcoded divergent constant was found; the **two P_max values
are intentional** (calibration vs detection paths) — J8 must route the right one
to each path (calibration peaks → `P_MAX_CALIBRATION`; detection peaks →
`P_MAX_DETECTION`). The `bucket_for_n_lines` ladder in `host.py:152` matches
shapes.md (64…4096 powers of two).

### 3.4 `StaticConfig` vs `PipelineParams` overlap (audit point 1/4)
`top_k_per_element` lives in **both** `PipelineParams` (float, traced) and
implicitly as `select_lines(top_k=...)` which must be a **static int** (it sizes
the `(B,E,top_k)` array and keys the cache). Same tension for `max_iterations`
(traced budget) vs `StaticConfig.max_iters` (static trip count) — params.py
documents this split. J8 must take the **static** value from `StaticConfig` for
array shapes and may pass the **traced** value only where it is a pure threshold.
`StaticConfig` carries `closure_mode`, `broadening_mode`, `apply_self_absorption`,
`max_iters`, `bucket_id`, `n_species`, `level_pad`, `batch_size` — all consumed
correctly (`scan_solve(closure_mode=static.closure_mode)`,
`forward_fit_identify(static.broadening_mode)`).

### 3.5 fp64/fp32 policy (audit point 5) — **consistent (all fp64)**
No stage forces fp32 device math. The two `float32` text hits are a comment
(`stark.py:51`) and a `jnp.result_type(..., jnp.float32)` tiny-floor guard
(`forward_id.py:296`, harmless). `LaxKernelInputs.from_snapshot` casts everything
to `jnp.float64`; `find_peaks_fixed` casts `x` to f64; closure/SB-graph all f64.
Matches ADR §5.3 / contracts.md precision policy. No drift.

### 3.6 Result-type conventions — **consistent NamedTuple/dataclass pattern**
`PeakResult`, `CombScoreArrays`, `ObservationBuild`, `SelfAbsorptionResult`,
`StarkFitResult`/`StarkNeResult`, `LaxKernelInputs`, `JointWLSResult` are
NamedTuples (pytree leaves); `CalibrationKernelResult` is a dataclass with an
explicit `register_pytree_node`. All are valid vmap/grad bundles. No drift; J8
just needs adapters from these to the reference `WavelengthCalibrationResult` /
`LineDetectionResult` / `CFLIBSResult`.

---

## 4. Failure-policy parity (AC4)

Reference raises `ValueError` at zero observations (`pipeline.py:872-873`);
scoreboard scores all-FN (`:159-166`). The jit graph must **not raise** — at
`Σ ObservationBuild.obs_valid == 0` set a scalar `failed` mask, propagate
NaN-free pass-through state through fit/solve, and have the `run_one` host
wrapper interpret `failed` → emit the same all-FN record (NaN-free
concentrations, `converged=False`). This `failed` flag is a new output of the
identify→fit boundary that J8 must thread; no kernel raises.

---

## 5. The one structural decision: SB-graph ownership (fit vs solve)

`fit.sb_graph_fit` and `solve.scan_solve` **both** compute the SB-graph: `fit`
exposes it as a standalone `(E,N)` primitive, while `scan_solve._solve_iteration`
runs the SB-graph **inside** the iteration via the same `_common_slope_kernel`
twin (solve.py:437-442 region). They are math-identical (both reduce to
`_common_slope_kernel` on ion-shifted coords). For the **production geological
path** (`saha_boltzmann_graph=True`, `closure_mode="oxide"` — the only path that
runs in production, ADR §1.1), the solve owns the SB-graph and re-derives T/n_e
each iteration. **J8 decision:** for M1 parity, `run_one` should build the
`(E,N_max)` Boltzmann block once (host glue §2.C), feed `LaxKernelInputs`, and let
`scan_solve` own the fit — using `fit.select_lines` only for the selection gate
(SNR/isolation/top-K), **not** calling `fit.sb_graph_fit` separately. This avoids
a double fit and matches the reference solve loop. `joint_wls_solve` is the J8.5+
production estimator (ADR §6.1); for M1 the `scan_solve` initializer alone meets
the Tier-B contract.

---

## 6. Blocking mismatch

**One genuine blocker, and it is a "not-built-yet", not a shape conflict:**

> **B1 — the host glue layer (`host.py` candidate-set assembly, `FrontEndSnapshot`
> builder, `(E,N_max)` Boltzmann-block gather, Stark window extraction, result
> reconstitution) does not exist.** Every stage kernel was built and parity-tested
> in isolation against *hand-padded* inputs (via the test adapters), so the
> seam between stages — which is pure host gather/scatter — was never written.
> `pipeline.run_one`, `forward.forward_spectrum`, `calibrate.calibrate_wavelengths`,
> `identify.identify_lines`, `fit.boltzmann_fit`, `selfabs.correct_self_absorption`,
> `solve.iterative_solve` all still `raise NotImplementedError`. **This is the J8
> deliverable**, not a regression — but it means M1 cannot compose until §2.A–§2.F
> are written. No two kernels have *incompatible* shapes once the documented host
> gathers (`_build_padded_arrays_from_obs` / `reorder` pattern, ADR §5.1.2) run.

No fp64/fp32, padding-constant, or snapshot-accessor mismatch was found that would
break composition. The PipelineParams name-drift (§3.1) is a **wiring** issue (the
wrapper must map names), not a runtime shape break, but it WILL silently
mis-parameterize the graph if J8 maps `t_tolerance_k`→`t_tol_k` etc. incorrectly —
treat the §3.1 mapping table as a checklist.

---

## 7. Ordered task list to reach M1

1. **`host.py`: candidate-set assembly + element-mask gather** over the
   `PipelineSnapshot` superset → per-bucket padded line arrays (the
   `_build_padded_arrays_from_obs`/`reorder` pattern). Foundation for §2.B/§2.C/§2.E.
2. **`host.py`: `FrontEndSnapshot` builder** (§2.B) — gA-Boltzmann comb ranking,
   peak gather against the calibrated axis, shift grid. Parity-test against the
   reference `detect_line_observations` ranking on one fixture per dataset.
3. **`host.py`: Stark window extraction host wrapper** (§2.D) — `extract_windows` +
   `multiplet_blend_mask` + candidate selection → `(C,W)` windows.
4. **Decide PipelineParams gate-knob policy** (§3.1 (a) vs (b)). Minimal-for-M1:
   (b), source comb/calibrate gate defaults from `AnalysisPipelineConfig`. File a
   follow-up bead for (a) (J11 tunability).
5. **Write `pipeline.run_one`** as the composition (§1), calling the **real
   kernels directly** (not the stub orchestrators), with the §3.1 param map. Thread
   the `failed` mask (§4). Single-spectrum only (batch is J9).
6. **`host.py`: result reconstitution** (§2.F) → `WavelengthCalibrationResult`,
   `LineDetectionResult`, `CFLIBSResult` + the `failed`→all-FN path.
7. **Fill (or delete) the stub orchestrators** (`forward_spectrum`,
   `calibrate_wavelengths`, `identify_lines`(keep for J10), `boltzmann_fit`,
   `correct_self_absorption`, `iterative_solve`) as thin wrappers that delegate to
   the kernels — so the public `__init__` API resolves and the stub
   `NotImplementedError`s are gone.
8. **`parity.py`: complete the per-stage adapters** (`preprocess_parity`,
   `detect_parity`, `identify_parity`, `fit_parity`, `solve_parity`) so AC5
   stage-bisection works on the composed graph.
9. **Scoreboard `--pipeline` dispatch** (§2.G) — flag through
   `_score_spectrum`/`run_scoreboard`; `jit` path emits identical record schema.
10. **Tests** (`tests/jitpipe/test_parity_pipeline.py`,
    `test_scoreboard_dispatch.py`): end-to-end vs `run_pipeline` on the two M1
    datasets (bhvo2_chemcam, chemcam_calib), failure-policy fixture, graph-level
    padding invariance, compile-once-per-bucket (AC6). **Per CLAUDE.md: run only
    narrow subsets inside any sub-agent; background the full suite from the
    parent.** Commit after each step (watchdog discipline).
11. **`docs/jitpipe/divergence-ledger.md`** — create and seed with any M1
    adjudications (J8 spec §2).

**Sequencing note:** steps 1–3 (host gather) unblock everything; step 5 is the
integration spine; steps 9–11 are the M1 gate. `forward_id.py` (J10) and
`autodiff.py` (J11) are **off the M1 critical path** — they are the recall-endgame
and differentiability tracks, not part of the `run_one` parity composition.

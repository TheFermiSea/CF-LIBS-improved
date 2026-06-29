# 04 — Pipeline Defaults Audit: Validated Improvements Not Wired Into Default Paths

**Date:** 2026-06-09 · **Scope:** end-to-end wiring of `cflibs analyze` / `invert` / `bayesian` / `batch`, config surface, trust reporting, scripts-vs-CLI drift, reproducibility, dead entry points.
**Method:** read-only code trace + two live measurements on real ChemCam BHVO-2 (`data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv`), run today on `main` (6dd4045).

## 0. Headline measurement (reproduced today)

Same spectrum, same code, same element list — only opt-in flags differ:

| Configuration | RMSE (wt%) | Fe (cert 8.61) | Converged reported | User warning shown |
|---|---|---|---|---|
| **Default** (`analyze` with no flags): standard closure, no SB-graph | **10.288** | **39.04** (4.5× over) | `True` | none |
| **Best validated** (`--saha-boltzmann-graph --closure-mode oxide`) | **4.029** | 4.33 | `True` | none |

Artifacts: `output/bhvo2_measure/audit_default_baseline.json`, `output/bhvo2_measure/audit_sbgraph_oxide.json`.

The 2.55× accuracy improvement is fully implemented, tested, merged, and **off by default**. Both runs also silently used the 1-atm pressure-balance n_e fallback (logged as "physically non-standard for LIBS", never surfaced to the user) and both dropped Na/K/P entirely while reporting `converged=True`.

---

## 1. Default vs best-validated, stage by stage

| Stage | Default choice | Best-validated choice | Where the knob lives | CLI? | YAML? | Documented? |
|---|---|---|---|---|---|---|
| Wavelength calibration | **ON** (segmented robust cal + 0.05 nm residual shift-scan) — good | same | `_detect_and_select_lines(wavelength_calibration=True)` `cflibs/cli/main.py:180` | not exposed (always on) | no | no |
| Detection floor | `min_relative_intensity=None` + `top_k_per_element=60` + shift-coherence veto — good (the Na=98% fix) | same | `cflibs/cli/main.py:167-168`; library default in `identify/line_detection.py:883-884` is `None/None` (≠ CLI) | `--min-relative-intensity` | `analysis.min_relative_intensity` | partially (help text only) |
| Resonance lines | kept (`exclude_resonance=None→False`) — good | same (SA corrects them when SA on) | `cflibs/cli/main.py:327-328`; **`LineSelector` class default is still `True`** (`physics/line_selection.py:141`) | no flag | `analysis.exclude_resonance` | example YAML says `true` (wrong) |
| Identifier | direct transition matching (`detect_line_observations`); ALIAS/comb/NNLS/hybrid **never reachable from CLI** | hybrid/ALIAS for unknown samples | `cflibs/inversion/identify/` | **no** | no | no |
| Intercept extraction | per-element common-slope plane | **pooled Saha-Boltzmann graph** (`saha_boltzmann_graph=True`) — BHVO-2 Fe 39%→4% | `solve/iterative.py:874` (default `False`) | `analyze --saha-boltzmann-graph` only; `invert` config-key only; **batch/bayesian: no** | `analysis.saha_boltzmann_graph` (invert only) | **not in User_Guide/Quick_Start/API_Reference** |
| Closure | `standard` | **`oxide`** for geological samples (auto stoichiometry) | `solve/iterative.py:1593`; CLI default `cflibs/cli/main.py:1090-1092` | `analyze --closure-mode`; `invert` config-key only; **batch: no** | `analysis.closure_mode` | docs list only standard/matrix/oxide (CLI has 6 modes) |
| Boltzmann weight cap | **ON by default** (`boltzmann_weight_cap=5.0`) — good, landed #223 | same | `solve/iterative.py:873` | not exposed | **not wired** in `invert_cmd` (cli/main.py:523-535 omits it) | no |
| Self-absorption | OFF (`apply_self_absorption=False`) | ON for known optically-thick samples; correctly opt-in until thin/thick gating exists | `solve/iterative.py:867` | `analyze --apply-self-absorption`; `invert` config-key; batch: no | `analysis.apply_self_absorption` | no |
| n_e diagnostic | **1-atm pressure balance** (docstring: "physically invalid for LIBS") | Stark-width diagnostic (`stark_diagnostic=` param of `solve()`) | `solve/iterative.py:1594` | **not wired into any CLI path** | no | no |
| Boltzmann R² gate | ON (`min_boltzmann_r2=0.3`) — good | same | `solve/iterative.py:872` | not exposed | not wired | no |
| Bayesian prefilter | **not used** by `bayesian_cmd` | `select_candidate_elements` ("mandatory" per CLAUDE.md; used by `benchmark/bayesian_sparse_id.py:205-217`) | `inversion/candidate_prefilter.py` | **no** | no | CLAUDE.md only |
| Bayesian instrument model | fixed-FWHM default; no resolving-power, no prior config | resolving-power mode for the ps instrument | `solve/bayesian/forward.py:144` | **no `--resolving-power` on `bayesian`** | no | no |

### Per-command summary

- **`analyze`** — the only command exposing the best path, but all three accuracy flags default off. No config-file support (`--config` absent), so per-instrument settings can't be persisted.
- **`invert`** — gets the same good detection helper; SB-graph/closure/SA reachable **only** via YAML `analysis.*` keys; no CLI flags. Asymmetric with `analyze` (flags vs YAML, never both).
- **`batch`** (`cflibs/cli/main.py:748-831`) — **bypasses `_detect_and_select_lines` entirely**. It calls raw `detect_line_observations` (no top-K bound, no wavelength calibration, no shift-coherence defaults from the CLI helper) + bare `LineSelector()` (**`exclude_resonance=True`**, dropping Al/Na/Mg/Ca/K majors) + bare `IterativeCFLIBSSolver()` (standard closure, no SB-graph). **The batch path still contains the exact pre-fix wiring whose drift caused the Na=98% blowup that `_detect_and_select_lines`'s docstring warns about.**
- **`bayesian`** — no candidate prefilter (intractability guard), no resolving-power, no baseline/prior knobs, no quality gating of the trace beyond ArviZ summary.

---

## 2. Config-surface inventory

### 2.1 YAML schema (`cflibs/core/config.py`)
`validate_plasma_config` / `validate_instrument_config` only. **There is no schema or validation for the `analysis:` section** — typo'd keys (`saha_boltzman_graph`) are silently ignored and you get the default. `analysis.*` keys actually read by `invert_cmd` (cli/main.py:446-538): `elements, wavelength_tolerance_nm, min_peak_height, peak_width_nm, min_relative_intensity, resolving_power, apply_self_absorption, exclude_resonance, min_snr, min_energy_spread_ev, min_lines_per_element, isolation_wavelength_nm, max_lines_per_element, max_iterations, t_tolerance_k, ne_tolerance_frac, pressure_pa|pressure, self_absorption_column_density_cm3, self_absorption_plasma_length_cm, saha_boltzmann_graph, closure_mode, closure_kwargs, matrix_element, oxide_elements`.

**Not configurable anywhere (constructor-only):** `boltzmann_weight_cap`, `min_boltzmann_r2`, `apply_ipd`, `aki_uncertainty_weighting`, `two_region`, `self_absorption_tau_cap`, `self_absorption_mask_threshold`, `use_jax_boltzmann`, `use_lax_while_loop`, `stark_diagnostic`.

### 2.2 Conflicting defaults (CLI vs YAML examples vs class defaults)

| Knob | CLI helper default | Library/class default | `examples/inversion_config_example.yaml` | `docs/User_Guide.md` |
|---|---|---|---|---|
| `exclude_resonance` | `False` (cli/main.py:327-328) | **`True`** (`LineSelector`, physics/line_selection.py:141) | **`true` (line 9) — re-enables the major-deleting legacy gate** | `true` (line ~358) |
| `min_relative_intensity` | `None` | `None` | **`100.0` (line 20) — re-enables the floor that deletes Mg/K/Al** | `null` (inconsistent with example file) |
| `top_k_per_element` | `60` (cli/main.py:168) | **`None`** (identify/line_detection.py:884) — `batch` gets unbounded catalog | absent | absent |
| `closure_mode` | `standard` | `standard` | `standard` | docs list 3 modes; CLI offers 6 (`ilr`,`pwlr`,`dirichlet_residual` undocumented) |
| `resolving_power` | `None` | `None` | `5000.0` | absent |

A user who copies `examples/inversion_config_example.yaml` — the only shipped inversion config — gets the **legacy catastrophic detection settings** back.

### 2.3 Environment variables still gating physics/numerics

| Var | Site | Status |
|---|---|---|
| `CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION` | `solve/iterative.py:41` (seeds `use_jax_boltzmann` ctor flag, #275/866642a) and **`runtime/streaming.py:108` (still env-only — missed by the c5 lift)** | partially lifted |
| `CFLIBS_USE_LAX_WHILE_LOOP` | `solve/iterative.py:56` (seeds `use_lax_while_loop` ctor flag) | lifted |
| `CFLIBS_DISABLE_STARK_T_FACTOR` | `radiation/kernels.py:444` — **physics ablation toggle, env-only**, changes Stark widths | should be ctor/config flag |
| `CFLIBS_USE_JAX_IDENTIFIER` | `benchmark/unified.py:51,4054`; mutated by `parameter_sweep_server/server.py:292-302` | benchmark-only, env-only |
| `CFLIBS_BENCH_CHECKPOINT_PATH/_EVERY`, `CFLIBS_OUTPUT_FORMAT` | `benchmark/composition_eval.py:367-369`, `unified.py:4251` | benchmark plumbing |
| `SWEEP_*` (7 vars) | `parameter_sweep_server/server.py` | server plumbing |

### 2.4 Hardcoded constants that should be config
`_solve_python` initial state `T_K=10000.0`, `n_e=1e17` (solve/iterative.py:2078-2080); `pressure_pa=101325` (STP — wrong physics for LIBS, see §3); `self_absorption_column_density_cm3=1e16`, `plasma_length_cm=0.1`, `tau_cap=10`, `mask_threshold=1e6`; degeneracy threshold `0.8` (`physics/closure.py:498`); detection comb/kdet gates (`identify/line_detection.py:888-906`); residual shift scan `0.05 nm` / legacy `0.5 nm` (cli/main.py:181,270). For the user's instrument (T 0.5–1.3 eV, n_e 1e16–1e18) the T/ne seeds are fine; the 1-atm pressure default is the actionable one.

---

## 3. Trust / quality reporting gaps

The solver now has real gates (post-#220/#223): `boltzmann_degenerate` (slope≥0 or R²<0.3 holds T and forces `converged=False`), `closure_degenerate` (any element >0.8 → `converged=False`), `ne_from_stark` provenance, all in `quality_metrics` (solve/iterative.py:2041-2062, 1988-1999, 2157-2161). **Almost none of it reaches the user:**

1. **`invert` prints nothing about quality** — `_output_invert_result` (cli/main.py:418-424) prints T, n_e, concentrations. Not even `converged`. A degenerate solve is indistinguishable from a clean one on stdout.
2. **`analyze` table mode** (cli/main.py:673-687) prints `Converged` and the McWhirter warning only. It never prints R², `boltzmann_degenerate`, `closure_degenerate`, or `ne_from_stark`; there is no "this result is unreliable" banner. JSON mode dumps the raw `quality_metrics` dict with no interpretation.
3. **`converged=True` coexists with garbage**: today's default run reported `converged=True` while Fe was 4.5× over-attributed and Na/K/P were dropped to exactly 0. Nothing flags wholesale element loss — `measure_bhvo2_presence.py` reports "Dropped majors"; the CLI does not.
4. **n_e fallback warning is log-only**: "physically non-standard for LIBS… coarse last-resort estimate" goes to the logger (invisible at default INFO formatting in a pipe); `ne_from_stark=0.0` lands in quality_metrics but no CLI path renders it.
5. **lax path diverges**: `_solve_lax` quality_metrics = `{r_squared_last} + LTE` only (solve/iterative.py:2298) — **no `closure_degenerate`/`boltzmann_degenerate`/`ne_from_stark` keys**, and its convergence check (line 818) gates only on Boltzmann degeneracy, not closure degeneracy → opt-in JAX path can report `converged=True` on a keystone-collapsed composition.
6. **`batch`** records `converged` per row but no quality metrics; failed files are logged and silently skipped from the aggregate.
7. Exporters (`io/exporters.py:372,398,779,802`) do serialize `converged`+`quality_metrics`, so file outputs are better than stdout — but `if data["quality_metrics"]` silently omits the section when the dict is empty (lax/MC paths), which is the residual form of the "quality_metrics None" bug.

---

## 4. Scripts vs library drift

| Script | What it wires | vs CLI |
|---|---|---|
| `scripts/measure_bhvo2_presence.py` | **imports the private CLI helper** `from cflibs.cli.main import _detect_and_select_lines` + `IterativeCFLIBSSolver(saha_boltzmann_graph=…)`, oxide stoichiometry — i.e., the production path with the good flags | Aligned by construction (good), but depends on a private function; also reports RMSE/dropped-majors/FP diagnostics the CLI never shows |
| `scripts/validate_real_data.py` | full wavelength-calibration knob surface (`--wavelength-calibration-mode`, 6 quality-gate flags, lines 1234-1247), **auto resolving-power estimation** (`estimate_resolving_power`, line 503), 3 identifiers (ALIAS/Comb/Correlation) | CLI hardcodes segmented calibration with fixed gates, has **no resolving-power auto-detect** (user must pass `--resolving-power`), and exposes **no identifier at all** |
| `cflibs/benchmark/bayesian_sparse_id.py` | `select_candidate_elements` NNLS prefilter before MCMC (lines 205-217) | `bayesian_cmd` runs full-element MCMC — the documented-intractable configuration |
| `scripts/probe_saha_boltzmann_graph.py`, `scripts/diagnose_closure.py` | validated probes for the SB-graph and closure intercepts | diagnostics only, fine as scripts |

Net: the benchmark scripts both (a) exercise better configurations than the defaults and (b) carry capabilities (auto-R, calibration gates, dropped-major reporting, RMSE scoring) that belong in the library/CLI.

---

## 5. Reproducibility of the 4.03 wt% result

**Reproduced today** (RMSE 4.029 wt%):

```bash
python scripts/measure_bhvo2_presence.py --closure-mode oxide --saha-boltzmann-graph --label sbgraph_oxide
```

CLI-equivalent (same production path):

```bash
cflibs analyze data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv \
  --elements Si,Ti,Al,Fe,Mn,Mg,Ca,Na,K,P \
  --db-path ASD_da/libs_production.db \
  --closure-mode oxide --saha-boltzmann-graph
```

What's missing for a *user* to find this: **neither command appears anywhere in the documentation.** `docs/User_Guide.md` and `docs/Quick_Start_For_Scientists.md` never mention `--saha-boltzmann-graph` or `analyze --closure-mode`; `User_Guide`/`API_Reference` list only 3 of 6 closure modes; the only worked `analyze` examples are bare-default Fe-only runs. The numbers (10.33→5.60→4.29→4.03) live only in commit messages (#223/#224) and bead `CF-LIBS-improved-btcl`.

---

## 6. Dead/legacy entry points (zero external users → delete)

| Path | Why dead |
|---|---|
| `manifold-generator.py` | legacy, not MPI-aware (CLAUDE.md warns against it); superseded by `cflibs generate-manifold` |
| `cf-libs-analyzer.py` | standalone legacy analyzer with its own physics constants (`KB = 8.617e-5`) and own airPLS; fully superseded by `cflibs/inversion/` |
| `line-identifier.py` | legacy NNLS identifier; superseded by `cflibs/inversion/identify/spectral_nnls.py` |
| `saha-eggert.py`, `fix_boltzmann.py` | one-off legacy scripts |
| `datagen.py` | superseded by `datagen_v2.py` |
| `run_submit.py` | literally `print("Pretending to submit")` |
| 18 flat shims in `cflibs/inversion/` root (`line_detection.py`, `line_selection.py`, `quality.py`, `self_absorption.py`, `softmax_closure.py`, `boltzmann_jax.py`, `pca.py`, `hybrid.py`, `deconvolution.py`, `outliers.py`, `wavelength_calibration.py`, `matrix_effects.py`, `model_selection.py`, `result_base.py`, `spectral_refiner.py`, `streaming.py`, `temporal.py`, `daq_interface.py`) | each is a 3-line `import *` shim; internal importers measured at 1–6 each (e.g. `validate_real_data.py:43` uses the `wavelength_calibration` shim) — repoint and delete |
| CLAUDE.md claim "shims exist at all old flat paths (`from cflibs.inversion.solver import X` still works)" | **stale/false** — `cflibs.inversion.solver`, `.bayesian`, `.uncertainty`, `.closure`, `.boltzmann` all raise `ModuleNotFoundError` (verified). Fix the doc when deleting the rest |

---

## 7. Timeline of validated improvements (all confirmed opt-in at landing)

| Date | Commit/PR | Improvement | Default? |
|---|---|---|---|
| 2026-03-15 | d281b13 / #59 | ILR + Dirichlet-residual closures | opt-in |
| 2026-06-04 | 1efdc45 | robust wavelength calibration in CLI detection | **default-on** ✓ |
| 2026-06-04 | 1eec1c0 | detection cascade: gA-Boltzmann comb strength, per-element top-K, shift-coherence veto | **default-on in `analyze`/`invert`** ✓ (not `batch`) |
| 2026-06-05 | dfdb8a1 / #223 | Boltzmann weight cap (RMSE 14.2→10.3; oxide 5.6) | **default-on** (`cap=5.0`) ✓ |
| 2026-06-05 | 40a4c5b / #224 | pooled SB-graph intercepts (oxide+SB-graph 4.29; CSA 12.26) | **opt-in** ✗ |
| 2026-06-07 | 1d19cfb / #233 | Σ C=1 closure enforcement, true Hron PLR (point estimates changed → today's 4.03) | mixed |
| 2026-06-09 | 866642a / #275 | env vars → `use_jax_boltzmann`/`use_lax_while_loop` ctor flags | default seeds from env ✓ |

---

## 8. Change list: make the best path the default

Ordered; each line is one concrete edit. Items 1–4 close the headline gap; 5–10 close the trust gap; 11+ are hygiene.

1. **`cflibs/cli/main.py:1090-1092`** — `analyze --closure-mode` default `"standard"` → `"oxide"` *for geological workflows*; better: add `--preset {geological,metallurgical,default}` that maps to (closure, SB-graph, SA) bundles, with `geological = oxide + SB-graph`. Minimum viable: flip nothing else but document the preset.
2. **`cflibs/cli/main.py:1100-1109` + `589`** — `--saha-boltzmann-graph` → default **True** with a `--no-saha-boltzmann-graph` escape hatch (`argparse.BooleanOptionalAction`). It won on both ChemCam (10.33→9.06 standalone, 5.60→4.29 stacked) and CSA; #224 found no regression case.
3. **`cflibs/cli/main.py:534, 537`** — `invert` mirrors the same defaults (`analysis_cfg.get("saha_boltzmann_graph", True)`, `closure_mode` per preset) so YAML and flag paths agree.
4. **`cflibs/cli/main.py:771-795`** — rewrite `batch_cmd` to call `_detect_and_select_lines` and accept `--closure-mode/--saha-boltzmann-graph/--apply-self-absorption`, deleting the bare `LineSelector()`/`detect_line_observations` wiring (the surviving Na-blowup path).
5. **`cflibs/cli/main.py:418-424`** — `_output_invert_result`: print `converged`, R², and a `RESULT UNRELIABLE` banner when `boltzmann_degenerate`/`closure_degenerate`/`not converged`.
6. **`cflibs/cli/main.py:673-687`** — `_output_analyze_result`: same banner; render `ne_from_stark=0` as "n_e from 1-atm fallback (coarse)"; print dropped requested elements (`requested − concentrations>ε`, the script's "Dropped majors" check).
7. **`cflibs/inversion/solve/iterative.py:2298` & `818`** — lax path: add `closure_degenerate` gate to converged + emit the same degenerate/provenance keys as `_build_python_quality_metrics` (cross-path parity).
8. **`cflibs/cli/main.py:718-723`** — `bayesian_cmd`: insert `select_candidate_elements` prefilter (as in `benchmark/bayesian_sparse_id.py:205-217`) + add `--resolving-power`.
9. **`examples/inversion_config_example.yaml:9,20`** — `exclude_resonance: true` → `false`(or delete the key); `min_relative_intensity: 100.0` → `null`. Add `saha_boltzmann_graph: true`, `closure_mode: oxide` (commented with "geological").
10. **`docs/User_Guide.md` (§Inversion, ~line 340-380) + `Quick_Start_For_Scientists.md`** — document the reproducible BHVO-2 command from §5 and all 6 closure modes; fix `exclude_resonance` doc default.
11. **`cflibs/inversion/physics/line_selection.py:141`** — `LineSelector(exclude_resonance=True)` → `False` so the class default matches the validated CLI default (kills the batch/library trap class-wide).
12. **`cflibs/inversion/identify/line_detection.py:884`** — `top_k_per_element=None` → `60` (match CLI helper).
13. **`cflibs/cli/main.py:523-535`** — wire `boltzmann_weight_cap`, `min_boltzmann_r2` from `analysis_cfg` (currently constructor-only).
14. **`cflibs/inversion/runtime/streaming.py:108`** — finish the c5 env-var lift (constructor flag like iterative.py).
15. **`cflibs/radiation/kernels.py:444`** — `CFLIBS_DISABLE_STARK_T_FACTOR` env toggle → explicit kwarg/config.
16. **`cflibs/core/config.py`** — add `validate_analysis_config` rejecting unknown `analysis.*` keys (typos currently silently revert to defaults).
17. Add `analyze --config` so per-instrument settings (resolving power for the 1040 nm ps system, pressure, preset) persist in YAML instead of flags.
18. Delete legacy entry points per §6; update CLAUDE.md's stale flat-shim claim.
19. (Physics, larger) wire a Stark `stark_diagnostic` auto-pick (Hα / Si II / Fe I from `lines.stark_w`) into the CLI so the default n_e is not the documented-invalid 1-atm balance.

---

## Appendix: evidence pointers

- Default-vs-best measurements: `output/bhvo2_measure/audit_default_baseline.json`, `audit_sbgraph_oxide.json` (this audit).
- SB-graph rationale + measured matrix: commit 40a4c5b body (`standard 10.33 / oxide 5.60 / SB-graph 9.06 / SB-graph+oxide 4.29` ChemCam; CSA column).
- Weight-cap rationale: `solve/iterative.py:931-959` comment block (Fe I 382.0 nm 133× weight pathology).
- Root-cause of why defaults stay bad: bead `CF-LIBS-improved-btcl` (per-element Boltzmann R²≈0 on real data; common-slope fit masks it at pooled R²=0.92).
- Na=98% historical blowup + drift mechanism: `_detect_and_select_lines` docstring, `cflibs/cli/main.py:183-204`.

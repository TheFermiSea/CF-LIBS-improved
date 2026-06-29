# Repo Cleanliness & Engineering Health Audit

**Date:** 2026-06-09 · **Scope:** /home/brian/code/CF-LIBS-improved (main @ 6dd4045)
**Context:** single maintainer, zero external users — aggressive deletion is acceptable.
**Method:** read-only; full pytest suite NOT run (per watchdog policy); per-test targeted runs only.

---

## 1. Headline numbers

| Metric | Value |
|---|---|
| Python LOC in `cflibs/` | ~90,100 |
| Largest package | `cflibs/inversion/` (41,585) then `cflibs/benchmark/` (17,266) |
| Files > 800 lines | **33** |
| Tests collected | 3,004 |
| `ruff check cflibs/` | **0 violations** |
| `complexipy cflibs/ --max-complexity-allowed 14` | **0 functions over** (ngjd campaign held) |
| `vulture cflibs/ --min-confidence 90` | 26 hits (mostly `__exit__` false positives) |
| Local branches | **115** (~100 are squash-merged PR heads) |
| Remote branches | 55 |
| Open PRs | **0** (#220/#221/#222 CLOSED, #236 MERGED — nothing pending) |
| Worktrees | 10 (4 stale locked agent WTs, 5 wt-pf, 1 /tmp residual) |

Top 15 largest files (lines):

```
4926 cflibs/inversion/identify/alias.py
4391 cflibs/benchmark/unified.py
3175 cflibs/inversion/solve/iterative.py
2437 cflibs/inversion/physics/self_absorption.py
2119 cflibs/inversion/identify/line_detection.py
1903 cflibs/inversion/runtime/temporal.py
1482 cflibs/visualization/widgets.py
1482 cflibs/plasma/partition.py
1373 cflibs/inversion/identify/comb.py
1368 cflibs/manifold/generator.py
1359 cflibs/inversion/runtime/streaming.py
1338 cflibs/inversion/preprocess/wavelength_calibration.py
1300 cflibs/benchmark/dataset.py
1259 cflibs/inversion/physics/uncertainty.py
1227 cflibs/io/exporters.py
```

`alias.py`, `unified.py`, and `iterative.py` are the three sprawl hot-spots; everything else
is tolerable. Lint/complexity baseline is excellent — sprawl, not mess, is the problem.

---

## 2. FIX list — the 8 pre-existing full-suite failures (bead CF-LIBS-improved-td8w)

| # | Test | Verdict | Disposition |
|---|---|---|---|
| 1 | `tests/cli/test_analyze_invert_no_drift.py:97` `test_analyze_and_invert_select_identical_lines` | **stale-test** — pins the pre-PR#223 contract (`min_relative_intensity=100.0`); PR #223 retired the absolute floor (default now `None`, gA-Boltzmann top-K + shift-coherence veto replace it) and updated only the 2 unit tests, not the 2 `requires_db` integration tests | A complete rewrite already exists **uncommitted** in `/home/brian/code/wt-pf/g2-cli-drift` — commit it |
| 2 | `tests/cli/test_analyze_invert_no_drift.py:147` `test_default_floor_kills_na_rydberg_blowup` | **stale-test** — same root cause: asserts the deleted floor is the Na-Rydberg pruning mechanism | Same g2 worktree diff covers it |
| 3 | `tests/parameter_sweep_server/test_e2e.py:381` `test_sweep_server_amortization` | **external-infra / flaky perf gate** — asserts warm-server is <1/4 wall time of 8 cold subprocesses; timing-environment dependent | Mark `slow`+`benchmark` (exclude from default run) or relax ratio; not a code bug |
| 4 | `tests/scripts/test_bench_pause.py` `test_run_skips_when_flag_present` | **external-infra** — un-stubbed run path resolves real `sbatch` and **submitted live SLURM jobs 3083–3085** during reproduction | Hermetic fix (sbatch/ssh/rsync stubs + `BENCH_SKIP_SYNC` + xfail) drafted **uncommitted** in `/home/brian/code/wt-pf/g4-benchpause` — commit it |
| 5 | `tests/test_bayesian.py:~360` `TestBayesianForwardModel::test_forward_allows_decoupled_total_species_density` | **REAL BUG** (verified by running today): spectra for `total_species_density_cm3=2.5e16` vs `2.5e17` are **bit-identical**. The param is plumbed through `forward.py:329→357` into `radiation.kernels.forward_model(... total_species_density_cm3=...)` but is silently ignored downstream — a physics parameter no-op introduced by the kernel migration | Fix in `cflibs/radiation/kernels.py` (wire the override into species densities) or delete the parameter and the contract |
| 6 | `tests/test_bayesian.py` `TestMCMCSampling::test_mcmc_result_correlation_matrix` | **stale/underpowered test + minor real gap** (verified): with 10 warmup / 30 samples the T_eV chain has zero variance → correlation matrix row is NaN (`diag = [nan,1,1,1]`). Underpowered NUTS budget; `correlation_matrix()` also lacks a zero-variance guard | Raise warmup/samples or fix seed; add NaN guard in `correlation_matrix()`. Related: bead 359q (bayesian pytest isolation) |
| 7 | `tests/test_bayesian.py:~910` `TestNestedSampling::test_nested_sampling_result_model_comparison` | **stale-test** (verified): asserts literal `"Δln(Z)"` in the comparison report; report was reworded ("Bayesian Model Comparison … Interpretation: Very strong evidence for Model A") — logic is correct | One-line assert update |
| 8 | `tests/test_comb.py` `test_default_min_correlation_lowered` | **ALREADY FIXED** (verified passing): renamed to `test_default_min_correlation_remains_benchmark_gated` (asserts 0.12) in commit 6dd4045 / PR #278 | Close out in bead td8w |

Net: 4 stale tests (2 with ready uncommitted fixes), 2 external-infra, **1 real physics bug
(#5)**, 1 already fixed. None of the 8 indicates pipeline-accuracy regression.

---

## 3. DELETE list

### 3.1 Branches (~100 of 115 local; 55 remote need pruning)

- **DELETE — squash-merged PR heads, content fully on dev** (verified by sampling
  `gh pr list --head`): all 81 `cx/ngjd-*`, `cx/wave1-*`…`cx/wave4-*`; all 5 `arch/c*`;
  `ngjd/tier-a`, `ngjd/tier-b`; `chore/wave2-audit-doc`; `docs/wave2-audit-grounding`;
  `fix/identifier-count-invariance` (#217), `fix/nnls-relative-coeff-recall` (#216),
  all `fix/wave2-*` / `fix/wave3-*` (merged #224–#269 series).
  One command after verification: `git branch -D` the lot, `git push origin --delete` /
  `git fetch --prune` for remote.
- **DELETE — empty fix attempts**: `fix/pre-g1-bayesian` … `fix/pre-g5-sweep` have **zero
  commits** over dev. BUT first salvage uncommitted diffs (see §3.2).
- **REVIEW THEN DELETE — unmerged content** (do not bulk-delete):
  - `fix/physics-audit-wave-1`: PR #212 merged, but tip commit **33e24b0 "Refactor inversion
    import shims" is dated TODAY (2026-06-09 09:12) and unmerged** — it retargets canonical
    imports across benchmark/cli/identify and looks like the in-progress bead **38xr** work.
    **SALVAGE this commit** (cherry-pick onto a fresh branch) before deleting.
  - `feat/fe-ti-intercept-residual` (13 commits, no PR, last 2026-06-04, worktree
    `/tmp/cflibs-residual`): real fix commits (double-claimed flux guard, Poisson floor on
    Voigt area uncertainty, R²-gated T update, Boltzmann weight cap). Decide
    merge-or-abandon explicitly; several may already be superseded by #223/#236.
  - `fix/self-absorption-wiring-b1` (4 commits, no PR): its final commit re-imposes
    `min_relative_intensity=100` — **contradicts the later PR #223 direction**; likely
    superseded → delete after a 5-minute diff check.
- **Worktrees**: prune 4 stale locked `.claude/worktrees/agent-*` (based at 7043445,
  2026-05-08 — a month old; unlock+remove unless an agent is live); remove
  `/home/brian/code/wt-pf/g1..g5` after salvaging g2/g4 diffs; remove `/tmp/cflibs-residual`
  after deciding on its branch. Each wt-pf worktree also has a stray untracked
  `ASD_da/ASD_da` artifact.

### 3.2 Salvage-before-delete (uncommitted work sitting in worktrees)

1. `/home/brian/code/wt-pf/g2-cli-drift` — full rewrite of
   `tests/cli/test_analyze_invert_no_drift.py` to the post-#223 contract (fixes failures #1–2).
2. `/home/brian/code/wt-pf/g4-benchpause` — hermetic-env fix for `test_bench_pause.py`
   (fixes failure #4 and prevents accidental live SLURM submissions).
3. Commit `33e24b0` on `fix/physics-audit-wave-1` — the shim-kill (bead 38xr) work.

### 3.3 Root-level tracked clutter

| File | Why delete |
|---|---|
| `fix_boltzmann.py` | One-off March script that repaired a file-corruption artifact ("[...417 lines truncated") — job done, pure noise |
| `run_submit.py` | Entire content: `print("Pretending to submit")` |
| `line-identifier.py`, `cf-libs-analyzer.py` | Pre-package standalone NNLS identifier prototypes; functionality lives in `cflibs/inversion/identify/` |
| `saha-eggert.py` | Pre-package prototype with its own duplicate constants table |
| `manifold-generator.py` | Legacy; CLAUDE.md already warns "do not launch" — delete and drop the warning |
| `.adr-0001-baseline-cov.txt` (608 KB!), `.adr-0001-baseline.txt`, `.adr-0001-baseline-bench.json` | Tracked one-shot baseline artifacts; belong in `docs/adr/` or gone |
| `.swarm-progress.txt` | March swarm scratch file |
| `datagen.py` | Legacy Colab notebook (ruff-excluded); move to `docs/archive/legacy/` per its own README, or delete — `datagen_v2.py` is canonical |
| `tests/validation/validation/` | Accidentally committed nested duplicate of the validation tests + result JSONs; it's the thing `pyproject.toml` ruff `extend-exclude` works around — delete dir AND the exclude |

Untracked-but-large (no git action, disk hygiene only): `output/` 1.4 GB, `data/` 11 GB,
`spectral_manifold.h5`. `.gitignore` is adequate (minor: duplicate `target/` and
`!ASD_da/libs_production.db` entries). No `.beadhub` remnants on disk or in the index.

### 3.4 Scripts (91 entries in `scripts/`)

- **KEEP — production workflow (~30)**: `build_synthetic_id_corpus.py`,
  `benchmark_synthetic_identifiers.py`, `audit_synthetic_physics.py`,
  `validate_nist_parity.py`, `run_nist_validation.py`, `validate_real_data*.py`,
  `calibrate_alias.py`, `generate_model_library.py`, `parameter_sweep.py`,
  `run_unified_benchmark.py`, `run_aalto_benchmark.py`, `run_vrabel2020_*`,
  `aggregate_shards.py`, `generate_bayesian_test_corpus.py`, `fetch/generate_nist_reference_spectra.py`,
  `benchmarks/`, `hpc/`, `bench-pause.sh`/`bench-resume.sh`, `start_sweep_server.sh`,
  `orphan-claim-watchdog.{py,service,timer}`, telemetry trio, `element_confusion.py`,
  `failure_attribution.py`, `generate_benchmark_figures.py`, `generate_real_data_report.py`.
- **MOVE → `scripts/db_migrations/` (one-time DB builders, already applied; ~17)**:
  `add_missing_*` (4), `add_mo_ii.py`, `add_pr_i_and_expand_nd_i.py`,
  `cleanup_autoionizing_levels.py`, `expand_partition_functions.py`, `expand_stub_elements.py`,
  `fill_stark_gaps.py`, `ingest_stark_b.py`, `migrate_add_broadening_columns.py`,
  `populate_*` (4), `regenerate_partition_functions.py`, `fit_partition_coefficients.py`.
  (Or delete outright — the built DB is committed at `ASD_da/libs_production.db`.)
- **DELETE — one-off probes/experiments (~25)**: `debug_recall.py`, entire `scripts/diag/`
  (`gate_diag2.py`, `gate_fix_test.py`, `kdet_why.py`…), `probe_*.py` (4),
  `keystone_*.py` (2), `measure_bhvo2_presence.py`, `sweep_alias_fixes.py`,
  `sweep_boltzmann_r2_min.py`, `analyze_calibration_stress.py`, `analyze_threshold_pareto.py`,
  `plot_alias_diagnostics.py`, `compare_stark_vjbh.py`, `submit_post_alias_fix_benchmark.sh`,
  `submit_stark_vjbh_benchmark.sh`, `exp001_dashboard.py`, `exp001_status.sh`,
  `exp003_launch.sh`, `exp_chain.py`, `loop_24h_driver.sh`, `loop_iteration.sh`,
  `run_cell.sh`, `_report_data.py`, `report_aki_uncertainty_medians.py`,
  `diagnose_closure.py`, and keep only one of `verify_partition_functions.py` /
  `validate_partition_functions.py`.

---

## 4. CONSOLIDATE list

1. **Inversion flat shims (bead 38xr, in progress).** 18 three-line shims remain at
   `cflibs/inversion/{boltzmann_jax,daq_interface,deconvolution,hybrid,line_detection,
   line_selection,matrix_effects,model_selection,outliers,pca,quality,result_base,
   self_absorption,softmax_closure,spectral_refiner,streaming,temporal,
   wavelength_calibration}.py`. Internal `cflibs/` imports still routed through shims:
   **13 sites** — `cli/main.py` (×4: lines 253-254, 756-757), `benchmark/unified.py:1986`,
   `benchmark/synthetic_eval.py:43`, `manifold/vector_index.py:13`,
   `manifold/basis_index.py:387`, `inversion/runtime/streaming.py:79`,
   `inversion/identify/line_detection.py:15,588`, `inversion/solve/joint_optimizer.py:37`
   (+1 docstring). Tests/scripts: ~37 import sites. With zero users: retarget the 13+37
   sites, then **delete all 18 shims**. Note `cflibs.inversion.solver` is *already gone*
   (PR #210) and nothing broke — proof the rest can go. Commit 33e24b0 (§3.2) appears to
   already do much of this.
2. **`cflibs/benchmarks/` (plural) deprecation package** — 45 lines re-exporting
   `cflibs.benchmark`; 1 test imports it (incl. private `_KB_EV`/`_REFERENCE_LINES`).
   Retarget and delete the package.
3. **Atomic-mass tables** — canonical `cflibs/atomic/masses.py` (+`resolve_element_mass`
   from arch/c3) exists, but four in-package copies remain:
   `radiation/kernels.py:91 _FALLBACK_MASSES`, `radiation/spectrum_model.py:159
   _FALLBACK_MASSES` (copy of the copy), `inversion/physics/self_absorption.py:157
   _ATOMIC_MASS_AMU`, `benchmark/synthetic.py:42 STANDARD_MASSES`; plus 3 script-local
   tables (`populate_vdw_and_self_broadening.py:186`, `populate_species_physics.py:215`,
   `run_accuracy_ablation.py:83`). Point all at `cflibs.atomic.masses`.
4. **Constants** — `cflibs/benchmark/corpus.py:131 _KB_EV = 8.617333262e-5` duplicates
   `cflibs/core/constants.py:16 KB_EV`. Use `core.constants`.
5. **`alias.py` (4,926) / `unified.py` (4,391) / `iterative.py` (3,175)** — candidates for
   sub-module splits in a future wave; not urgent given complexity gates pass.
6. **Vulture true positives** (small): `identify/alias.py:661 estimated_T`,
   `identify/model_selection.py:211,407`, `physics/matrix_effects.py:960
   propagate_uncertainty` (accepted-but-unused arg), `solve/bayesian/samplers.py:407`,
   `manifold/batch_forward.py:59`.

---

## 5. Docs & CLAUDE.md accuracy

- **Stale (reference deleted `cflibs.inversion.solver` or pre-split layout; last touched
  2026-04-26/29):** `docs/API_Reference.md`,
  `docs/CF-LIBS_Codebase_Technical_Documentation.md`, `docs/user/Peak_Identification_Guide.md`,
  `docs/user/Quick_Start_Real_Data.md`, `docs/physics/Equations.md` (import examples).
  Update import paths or delete the two big reference docs (Sphinx + code are the truth).
- **Archive candidates:** 8+ one-off `*-consultation.md` files at `docs/` root
  (bandit-allocator, basis-libraries, dataset-sharding, jax-cache-and-pause,
  parameter-sweep ×2, parquet-results), `CODEEVOLVE_WAVE2_PLAN.md`, `exp001-autonomous-plan.md`
  → `docs/archive/`. The `docs/architecture/` audit series is current — keep.
- **CLAUDE.md errors found:** (1) "`from cflibs.inversion.solver import X` still works" —
  **false**, raises `ModuleNotFoundError` (verified); (2) "Two approaches in
  `cflibs/inversion/uncertainty.py`" and "`cflibs/inversion/bayesian.py`" — both paths no
  longer exist (now `inversion/physics/uncertainty.py`, `inversion/solve/bayesian/`);
  (3) Setup says `uv venv --python 3.12` but the working venv is Python 3.11.2.

---

## 6. Dependency hygiene

- Extras layout (`jax-cpu/metal/cuda`, `ci`, `local`, `cluster`) is sane and intentional.
- **Real findings (deptry 0.23 + inspection):** `IPython` is imported by
  `cflibs/visualization/widgets.py:41` but not declared in the `widgets` extra — add it;
  `duckdb` is in `parquet`/`ci` extras but unused inside `cflibs/` (used by tests/analysis —
  documented rationale exists in pyproject; keep). The other ~580 deptry flags are
  dev-tool/extra noise.
- `mypy python_version = "3.10"` vs `requires-python = ">=3.11"` — inconsistent; bump.
- `uv.lock` is gitignored (untracked) while `Cargo.lock` is tracked — intentional, fine.

---

## 7. Looks dead but ISN'T — keep

- `cflibs/bandit/`, `cflibs/observability/`, `cflibs/parameter_sweep_server/` — small but
  wired into the sweep/benchmark workflow (tests exist).
- `cflibs/evolution/` — physics-only blocklist enforcement; referenced by CLAUDE.md/CI.
- The 5–6 **identifier implementations** (alias/comb/correlation/spectral-NNLS/hybrid/
  hybrid-consensus) are deliberate competing algorithms benchmark-gated against each other —
  not copy-paste duplication.
- `scripts/orphan-claim-watchdog.*` systemd units, `bench-pause/resume.sh` — live ops tooling.
- `ASD_da/libs_production.db` (6 MB tracked) — deliberate (gitignore carve-out).
- `datagen_v2.py` at root — canonical DB generator, referenced by CLAUDE.md; root location
  is fine (or move to `scripts/`, low value).
- The 4 `.claude/worktrees/agent-*` are *locked*; confirm no live agent before pruning.

---

## 8. Priority order

1. **Salvage** g2/g4 worktree diffs + commit 33e24b0 (fixes 3 of 8 failures + closes 38xr).
2. **Fix the real bug** — `total_species_density_cm3` no-op in `radiation/kernels.py` (#5).
3. **Branch massacre** — delete ~100 squash-merged locals, prune remotes, remove worktrees.
4. **Root clutter purge** (§3.3) + `tests/validation/validation/` + ruff-exclude removal.
5. **Shim kill** (38xr) + `cflibs/benchmarks/` + mass-table consolidation.
6. **Scripts triage** (§3.4) and docs archive/update + CLAUDE.md corrections (§5).

# ML-Driven Algorithm-Search Program for CF-LIBS — Design

**Date:** 2026-06-10
**Status:** Design (pre-implementation)
**Scope:** The optimization *process* may use any ML/LLM tooling. The *shipped* algorithm stays
physics-only (ruff TID251 + `cflibs/evolution/evaluator.py` AST blocklist; bead CF-LIBS-improved-3fy3).
**Fitness function:** the goal-metric scoreboard (`cflibs/benchmark/scoreboard.py` +
`scoreboard_registry.py`, branch `overhaul/w4-scoreboard`; baseline committed at
`docs/benchmarks/SCOREBOARD-2026-06-10-baseline.md`).

**Where we are (measured, 2026-06-10 baseline, geological preset):**

| Dataset | P | R | F1 | RMSE wt% (med) | s/spectrum (med) |
|---|---|---|---|---|---|
| bhvo2_chemcam (4 real ChemCam) | 1.000 | 0.575 | 0.730 | 2.52 | 2.7 |
| aalto (74 real) | 0.937 | 0.471 | 0.627 | — (presence-only) | 0.8 |
| synthetic_fixedforward (74/288 sampled) | 0.583 | 0.148 | 0.236 | 43.9 (basis-mismatch by design) | 0.5 |

The weak link is **recall** (0.47–0.58 real, 0.15 synthetic) at near-perfect precision, plus
hard failures ("No usable spectral lines detected": 11/74 aalto, 46/74 synthetic). Identification
is where the search program should spend its budget. Precision (zero FP on BHVO-2) is the
asset every adoption gate must protect.

**Prior art in this repo (do not rediscover):** CodeEvolve Wave 1
(`docs/CODEEVOLVE_WAVE2_PLAN.md`) ran GPU Sep-CMA-ES over a 101-parameter ensemble combiner on
*cached* algorithm outputs for 74 Aalto spectra: F_beta(0.5) 0.654 → 0.914 at 35,300 evals/s.
Its known limitation — frozen caches, no end-to-end pipeline in the loop — is exactly what the
scoreboard now fixes. Also institutional memory: the "paper-faithful ALIAS" change (PR #229)
*regressed* F1 by 0.041 — every identifier-scoring change must be benchmark-gated.

---

## 1. Resource Survey

All resources were fetched and read 2026-06-10 (WebFetch summaries; MDPI paper identified via
web search after a 403).

| # | Resource | What it is | Maintained / installable | Verdict | Use here | Integration cost |
|---|---|---|---|---|---|---|
| 1 | [arXiv 2602.12259v2](https://arxiv.org/html/2602.12259v2) — "Think like a Scientist: Physics-guided LLM Agent for Equation Discovery" (KeplerAgent; Yang, Venkatachalam, Kianezhad, Vadgama, Yu) | LLM agent that first discovers structure (symmetries, functional patterns) from data, then *configures* PySR/PySINDy with those constraints; iterates via workspace + experience log. 42.3% symbolic accuracy on LSR-Transform vs 37.8% PySR alone. | Paper; no code release mentioned in the HTML. | **Imitate** | Campaign 2 pattern: a Claude session inspects the scoreboard feature tables, proposes physics-motivated features/operator sets/constraints, *then* launches PySR — instead of throwing raw features at PySR. | Low (it's a workflow pattern, not a dependency). |
| 2 | [merlresearch/llmphy](https://github.com/merlresearch/llmphy) | GPT-4 + MuJoCo loop estimating physical parameters of tray-stack scenes; reconstruction error fed back to the LLM. AISTATS 2026 artifact. | 2 commits, v1.0.0 (2026-04), AGPL-3.0, Python 3.7 + MuJoCo 2.1. | **Skip** | Domain mismatch (rigid-body scene reconstruction); its one transferable idea — LLM proposes parameters, simulator scores, error feeds back — is already the core of Campaigns 1/3. AGPL is also incompatible with adopting code into MIT-adjacent tooling. | — |
| 3 | [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | 144 agent skills (Agent Skills standard; works in Claude Code/Cursor/Codex). Bio/chem/clinical heavy; 78+ database lookups. MIT, 27.8k stars, v2.47.0 (2026-06). | Yes, very active (`npx skills add`). | **Skip** | Nothing LIBS/plasma/spectroscopy-specific; no algorithm-search machinery. Zero role in the fitness loop. (Optionally installable for interactive research sessions, but that is outside this program.) | — |
| 4 | [InternScience/Awesome-Scientific-Skills](https://github.com/InternScience/Awesome-Scientific-Skills) | Curated *link list* of scientific agent skills ("Phase 1: the reading list before the library"), 423 stars. | List only; no installable code. | **Skip** | Duplicates #3's discovery function; nothing executable. | — |
| 5 | [InternScience/MLEvolve](https://github.com/InternScience/MLEvolve) | Autonomous ML-engineering system: Monte Carlo Graph Search + multi-agent codegen (single-pass / stepwise / SEARCH-REPLACE patching) + experience memory (BM25 + FAISS over past plans/code/metrics) + stagnation detection + cross-branch fusion. #1 on MLE-bench (65.3% medal rate). 323 stars, 24 commits (2026-02). | Runnable but built around mle-bench Kaggle tasks; **no license specified** — code cannot be legally adopted. | **Imitate** | Campaign 3 driver patterns: experience-driven memory of (diff, per-dataset deltas, verdict) retrieved into prompts; adaptive codegen mode selection; stagnation detection triggering structural mutations (our `structural_mutation_cadence` already anticipates this). | Low-medium (re-implement 3 mechanisms; our `cflibs/evolution/` scaffolding + beads already hold half the state). |
| 6 | [mims-harvard/ToolUniverse](https://github.com/mims-harvard/ToolUniverse) | 1000+ biomedical tools/models/APIs behind one agent protocol (drug discovery, oncology, pharmacovigilance). Apache-2.0, active (v1.2.6, 2026-06). | Yes (`uv pip install tooluniverse`). | **Skip** | Entirely biomedical; our agents need exactly two "tools" (run scoreboard, query atomic DB) and both are internal CLIs. | — |
| 7 | [MITIBM-FastCoder/LessonL](https://github.com/MITIBM-FastCoder/LessonL) | NeurIPS 2025: multi-LLM code optimization where agents extract **lessons** from each attempt (solicitation → banking → selection) and share them; beats larger single models on ParEval/PolyBench. MIT, 4 commits. | Paper artifact (C++/CUDA benchmarks baked in); reproducible, not a maintained framework. | **Imitate** | Campaign 3 prompt protocol: after each candidate eval, solicit a 2–3 sentence lesson tied to the per-dataset deltas; bank in run dir (and `bd remember`); select top-k relevant lessons into the next batch's prompt. The mechanism is ~200 lines; adopting their C++ harness buys nothing. | Low. |
| 8 | [DEAP/deap](https://github.com/DEAP/deap) | Classic evolutionary-computation framework: GA/GP, CMA-ES, NSGA-II/III, SPEA2, MO-CMA-ES. LGPL-3.0, 6.4k stars; mature/slow-moving but stable; `pip install deap`. | Yes. | **Adopt (secondary)** | Campaign 1 phase B (CMA-ES restricted to the continuous knobs) and the multi-objective alternative (NSGA-II over F1/RMSE/runtime). Never ships — optimization layer only. | Low (pip; ask-tell loop wraps our SLURM evaluator). |
| 9 | [MilesCranmer/PySR](https://github.com/MilesCranmer/PySR) | Symbolic regression on a Julia backend (SymbolicRegression.jl): custom operators, custom losses, complexity constraints, accuracy-vs-complexity Pareto front, exports to SymPy/NumPy/JAX. Apache-2.0, v1.5.10 (2026-03), very active. | Yes (`pip install pysr`; Julia auto-installed on first import). | **Adopt** | The engine of Campaign 2: discovered expressions export to SymPy/NumPy and are transcribed into physics-legal shipped code with provenance. Custom loss lets us fit presence-decision scores with class-weighted margin loss. | Medium (Julia on vasp nodes; first-run precompilation — do it once per node image; runs only in the optimization layer). |
| 10 | [adaptive-intelligent-robotics/QDax](https://github.com/adaptive-intelligent-robotics/QDax) | Quality-Diversity in JAX: MAP-Elites, CVT-MAP-Elites, CMA-ME/CMA-MEGA, MOME, NSGA2 baselines; GPU-vectorized evaluation. MIT, v0.5.0 (2025-05), Python ≥3.11. | Yes (`pip install QDax`). | **Adopt for cached-output mode; imitate for end-to-end** | Behavior-descriptor search over (recall, precision): in the Wave-1-style *cached-outputs* regime the evaluation is JAX-vectorizable and QDax's GPU MAP-Elites/CMA-ME is the right tool on the V100S nodes. For *end-to-end* scoreboard evaluation the bottleneck is the CPU pipeline (seconds/spectrum, not jittable), so QDax's GPU speedup is void — there a 2-D archive grid (~60 lines, imitated) over SLURM-evaluated candidates suffices. | Medium (GPU mode needs the cache generator refreshed à la `generate_evolve_cache_v2.py`); trivial for the imitated archive. |
| 11 | [MDPI engproc 59(1):238](https://www.mdpi.com/2673-4591/59/1/238) — Tomar, Bansal & Singh 2023, "Metaheuristic Algorithms for Optimization: A Brief Review" | Short conference review of GA/PSO/ACO-class metaheuristics (page 403'd; identified via search). | Review only; no code. | **Skip** | Adds nothing beyond DEAP/Optuna documentation; no LIBS content. | — |

**Not on the list but load-bearing:** **Optuna** (TPE + `CmaEsSampler` + storage/dashboards,
actively maintained, never ships) is the recommended Campaign 1 phase-A driver because the knob
space is mixed continuous/integer/categorical/conditional, which TPE handles natively and
CMA-ES does not.

---

## 2. Fitness Protocol

### 2.1 Dataset split (optimization vs holdout vs vault)

Datasets = `w4-scoreboard` core registry + `w4-datasets` extended adapters
(`cflibs/benchmark/adapters_extended.py`: CSA planetary ~99 spectra element-wt, ChemCam
calibration ~250 element-wt, EMSLIBS-2019 100 presence-only, Silva-2022 soils 102 presence-only,
Gibbons-2024 ~175 N-doped MGS-1). Splits are **by sample/target identity, never by spectrum**
(multiple spectra of one target are correlated; splitting by spectrum leaks truth).

| Tier | Datasets | Role |
|---|---|---|
| **Optimization set** (the fitness function; queried freely) | `aalto` (74), `synthetic_fixedforward` (seeded 74/288 sample per the baseline), `chemcam_calib`-train (~60% of targets), `csa_planetary`-train (~60% of targets), `silva2022`-train (~60% of samples) | Computes fitness every evaluation. |
| **Holdout** (adoption gate only; each query logged) | `bhvo2_chemcam` (all 4 — the headline number stays out of the loop), `emslibs2019`, the held-out 40% target splits of `chemcam_calib` / `csa_planetary` / `silva2022` | Run only when a candidate passes the optimization-set gate. Budget: ≤1 holdout run per campaign phase per week, count tracked in the run ledger. |
| **Vault** (end-of-program only) | `gibbons2024` (never yet measured by any tuning loop) | Final report figure; one run, ever, per campaign. |

Split manifests (lists of `spectrum_id` + file sha256) are generated once, committed under
`docs/benchmarks/manifests/`, and hashed into every run record. `synthetic_fixedforward` is
tracked for **relative** movement only (it shares physics with the inversion — its absolute
numbers are never adoption evidence; see baseline doc).

### 2.2 Composite fitness (single-objective form)

Per dataset *d* in the optimization set, from `_aggregate_dataset` output:
micro-F1 `F1_d`, median composition RMSE `R_d` (element-wt datasets only), median wall
`t_d` (s/spectrum), and FP count `FP_d`.

```
score_d  = 0.6 * F1_d  +  0.4 * (1 - min(R_d, R_cap)/R_cap)        R_cap = 10 wt%
           (presence-only datasets: score_d = F1_d)

Fitness  = Σ_d w_d * score_d / Σ_d w_d
           - 0.5 * Var_d(score_d)                                   # cross-matrix overfitting penalty
           - 0.2 * max(0, t_med/t_budget - 1)                       # runtime: free under budget,
                                                                    # linear penalty beyond t_budget = 5 s/spectrum
DEATH PENALTIES (fitness = -inf):
  - blocklist violation (Campaign 2/3 candidates)                   # existing evaluator behaviour
  - any real-dataset FP_d > FP_d(baseline) + 1                      # precision is the asset
  - any dataset with n_failed > 1.25 * baseline failures            # no trading failures for F1
```

Weights `w_d`: real element-wt datasets 1.0, real presence-only 0.7, synthetic 0.5 — extends the
existing `EvolutionDriverConfig.fitness_weights` mapping (`cflibs/evolution/config.py`), and the
variance term reuses its `overfitting_penalty = 0.5`. The 0.6/0.4 split reflects the program's
stated priority (ID is the weak link) while keeping RMSE in every gradient.

### 2.3 Multi-objective alternative

NSGA-II (DEAP) — or MOME (QDax) in cached-output mode — over the 3-vector
`(micro-F1 ↑, mean RMSE ↓, median s/spectrum ↓)` with the same death penalties as constraints.
Adoption then selects from the Pareto front by: max F1 subject to RMSE ≤ baseline + 0.2 wt%
and runtime ≤ 1.5× baseline. Use this when the scalar weights start visibly distorting search
(e.g. candidates trading 0.05 F1 for 1 wt% RMSE); the scalar form is the default because the
adoption gate, not the optimizer, is the real arbiter.

### 2.4 Statistical guards

- **Paired per-spectrum bootstrap.** For candidate vs incumbent, resample spectra within each
  dataset (B = 2000, seeded), compute Δ(micro-F1) and Δ(median RMSE) per replicate. "Improved"
  means the 95% CI of the paired Δ excludes 0. BHVO-2 (n=4) can never pass a bootstrap alone —
  it is gated on "no regression on any of the 4 + no FP" instead.
- **Minimum-improvement thresholds** (anti-noise-mining): ΔF1 ≥ +0.02 (micro, optimization set
  pooled) or ΔRMSE ≤ −0.2 wt%. Below threshold = not adoptable regardless of CI.
- **Frozen sampling:** `seed=20260610` (`DEFAULT_SEED` in scoreboard.py), `max_spectra=74`,
  identical sampled ids per generation; the per-generation synthetic subsample may rotate seed
  *per generation* (same for all candidates within a generation) to prevent memorizing one draw.
- **Incumbent re-evaluation** each generation (detects environment drift / nondeterminism).
- **Scoring constants are out of bounds:** `PRESENCE_EPS_MASSFRAC = 5e-3`, the confounder set
  (Ag/Sn/W/Bi/Th), and the candidate-set policy live in the scoreboard, not in the search space.

### 2.5 Adoption gate (all campaigns)

1. Optimization-set gate: bootstrap-significant improvement ≥ minimum threshold, no death penalty.
2. Holdout gate: no holdout dataset regresses beyond its bootstrap noise; **zero new FP on
   bhvo2_chemcam and no new failures on real datasets**.
3. Legality gate (code-producing campaigns): AST blocklist + ruff TID251 + black + mypy + the
   targeted unit-test subset; `assert_benchmark_relevance` (already in `evaluator.py`) confirms
   the diff touches exercised files.
4. Runtime gate: median s/spectrum ≤ 1.5× baseline on the holdout run.
5. Human review PR (base `dev`), body containing: candidate lineage/provenance, optimization +
   holdout scoreboard tables, bootstrap CIs, dataset-manifest and DB hashes. CI runs the full
   pytest suite (never inside sub-agents — CLAUDE.md watchdog rule).

---

## 3. Campaign 1 — Parameter Space (first; cheapest)

### 3.1 Search space (enumerated from code on `overhaul/w4-scoreboard`)

**A. `AnalysisPipelineConfig` (`cflibs/inversion/pipeline.py`, lines ~90–155).** Current values
are the `geological` preset / dataclass defaults.

| Knob | Current | Type | Search range |
|---|---|---|---|
| `min_relative_intensity` | `None` | float? | {None} ∪ [1e-4, 0.05] log |
| `top_k_per_element` | 60 | int | 15–200 |
| `wavelength_tolerance_nm` | 0.1 (adaptive if None+R) | float | 0.02–0.3 (+ "adaptive" categorical) |
| `min_peak_height` | 0.01 | float | 0.001–0.05 log |
| `peak_width_nm` | 0.2 (adaptive if None+R) | float | 0.05–0.5 (+ "adaptive") |
| `apply_self_absorption` | `"off"` | cat | {off, observable} |
| `exclude_resonance` | `None`→False | bool | {True, False} |
| `min_snr` | 10.0 | float | 2–20 |
| `min_energy_spread_ev` | 2.0 | float | 0.5–4.0 |
| `min_lines_per_element` | 3 | int | 1–5 |
| `isolation_wavelength_nm` | 0.1 | float | 0.02–0.3 |
| `max_lines_per_element` | 20 | int | 5–60 |
| `wavelength_calibration` | True | bool | {True} (frozen — turning it off is a known catastrophic axis) |
| `shift_coherence_veto` | True | bool | {True, False} |
| `residual_shift_scan_nm` (ye6t) | 0.0 | float | 0.0–0.1 (expect optimum at 0; the 0.05 legacy measurably rode its window edge) |
| `affine_coverage_gate` (ye6t) | True | bool | {True, False} |
| `line_residual_gate` (ye6t) | True | bool | {True, False} |
| `max_iterations` | 20 | int | 5–50 |
| `t_tolerance_k` | 100.0 | float | 10–500 log |
| `ne_tolerance_frac` | 0.1 | float | 0.01–0.5 log |
| `boltzmann_weight_cap` | 5.0 | float | 1–20 |
| `min_boltzmann_r2` | 0.3 | float | 0.0–0.8 |
| `saha_boltzmann_graph` | True | bool | {True, False} |
| `closure_mode` | `"oxide"` | cat | {standard, matrix, oxide, ilr, pwlr, dirichlet_residual} — per-preset, not global |
| `stark_ne` | True | bool | {True, False} |

**B. `detect_line_observations` gates (`cflibs/inversion/identify/line_detection.py:1099`)** —
the recall-controlling layer the pipeline config does not yet surface (plumbing these into
`AnalysisPipelineConfig` is the first implementation task of the campaign):

| Knob | Current | Type | Search range |
|---|---|---|---|
| `ground_state_threshold_ev` | 0.1 | float | 0.05–0.5 |
| `shift_scan_nm` (global scan when calibration fails/skipped) | 0.5 | float | 0.1–1.0 |
| `comb_max_lines_per_element` | 30 | int | 10–100 |
| `comb_min_matches` | 3 | int | 2–6 |
| `comb_min_precision` | 0.02 | float | 0.005–0.2 log |
| `comb_min_recall` | 0.1 | float | 0.02–0.5 log |
| `comb_max_missing_fraction` | 0.85 | float | 0.5–0.95 |
| `comb_fallback_to_nearest` | True | bool | {True, False} |
| `comb_fallback_max_elements` | 5 | int | 1–10 |
| `kdet_enabled` | True | bool | {True, False} |
| `kdet_min_score` | 0.05 | float | 0.005–0.3 log |
| `kdet_min_candidates` | 2 | int | 1–5 |
| `kdet_rarity_power` | 0.5 | float | 0.0–2.0 |
| `kdet_weight_clip` | (0.25, 4.0) | (float, float) | lo 0.05–1.0, hi 1.0–10.0 |
| `coherence_min_lines` | 2 | int | 2–5 |
| `coherence_min_fraction` | 0.5 | float | 0.2–0.9 |
| `residual_gate_min_kept_lines` (ye6t) | 3 | int | 1–6 |
| `poisson_floor_scale` | 1.0 | float | 0.3–3.0 |
| `use_deconvolution` | False | bool | {True, False} |

**C. `LineSelector` (`cflibs/inversion/physics/line_selection.py:138`)** — its six knobs
(`min_snr` 10.0, `min_energy_spread_ev` 2.0, `min_lines_per_element` 3, `exclude_resonance`
False, `isolation_wavelength_nm` 0.1, `max_lines_per_element` 20) are already mirrored by the
pipeline-config rows above; they are one set of parameters, not two.

Total: **~45 knobs** (≈28 continuous/int, ≈10 boolean, 2 categorical, 2 conditional). Failure
hypothesis to encode as a prior: the "No usable spectral lines" failures point at
`min_snr`/`min_peak_height`/`comb_min_matches`/`min_lines_per_element` being too strict for
low-SNR and pure-element spectra — seed Optuna with hand-picked "looser-gates" trials.

### 3.2 Optimizer

- **Phase A — Optuna TPE** (multivariate, with conditional parameters), 800 trials. TPE is the
  right first pass because the space is mixed-type and conditional; CMA-ES is not. Optuna never
  ships, so its presence in the optimization layer is unconstrained. Storage: SQLite journal in
  the run dir (driver-local; SLURM workers report results back via files, avoiding NFS-locking
  issues).
- **Phase B — CMA-ES** (DEAP `cma.Strategy`, or Optuna `CmaEsSampler`) over the ~28 continuous
  knobs, categoricals frozen at the phase-A incumbent. λ = 4 + 3·ln(28) ≈ 14 → use 16,
  ~60 generations ≈ 960 evals.
- **Phase C (optional) — MAP-Elites archive** over behavior descriptors
  (micro-recall, micro-precision) binned 20×20, fitness = composite score. End-to-end mode uses
  the imitated ~60-line archive over SLURM evaluations; if a Wave-1-style cached-output
  evaluation layer is refreshed, run QDax CMA-ME on a V100S instead (the Wave-1 result — F_beta
  0.914 at 35k evals/s — shows the cached regime saturates quickly; treat its output as
  *candidate generators* for end-to-end confirmation, never as adoptable evidence).
- Per-preset runs: `geological` first (the production default); a separate shorter run may tune
  `metallic` for the synthetic/steel band later. Do **not** let one run mix closure bases across
  datasets — that is preset selection, not parameter tuning.

### 3.3 SLURM evaluation topology and budget

Measured cost (committed baseline): 2.7 s/spectrum ChemCam-sized, 0.8 aalto, 0.5 synthetic, CPU.
Optimization set ≈ 330–350 spectra ⇒ **~8–15 core-minutes per candidate evaluation**
(conservative envelope if the extended datasets are heavier: ≤45 core-min; per-spectrum
timeout 120 s, timeout counts as failure). The pipeline is CPU-bound — Campaign 1 does not
touch the V100S GPUs.

- **Driver:** 1 long-running 2-core CPU job (or tmux on a login node): runs Optuna/DEAP
  ask-tell, writes candidate configs, submits eval arrays, ingests results, checkpoints every
  generation.
- **Workers:** `sbatch --array=0-15%16 --cpus-per-task=8 --mem=16G --time=00:45:00` — one task
  per candidate; inside, spectra fan out over a `multiprocessing` pool (8 procs,
  `OMP_NUM_THREADS=1`, `JAX_PLATFORMS=cpu`). One generation ≈ 2–6 min wall.
  **No `--nodelist`** (let SLURM schedule); **private JAX compilation cache**
  (`JAX_COMPILATION_CACHE_DIR=$SLURM_TMPDIR/jax-cache` or `$HOME/.cache/jax-cflibs-evolve`),
  never the shared one. Reuse `cflibs/hpc/slurm.py` (`SlurmJobManager`, `ArrayJobConfig`).
- **Budget:** Phase A 800 evals ≈ 110–220 core-hours; Phase B 960 evals ≈ similar.
  **Total cap: 600 core-hours**, ledger-enforced (Section 5). Wall-clock: phase A finishes
  overnight on a fraction of the 3-node cluster.

### 3.4 Adoption gate

Section 2.5 with one addition: the winning configuration ships as **updated
`ANALYSIS_PRESETS` values / dataclass defaults in `cflibs/inversion/pipeline.py`** — plain
numbers in physics code, trivially legal — plus a provenance comment citing the run id and this
document, mirroring how the ye6t knobs are documented today. Top-5 candidates (not just the
best) go through the holdout gate; pick the one with the best worst-dataset score.

---

## 4. Campaign 2 — Formula Space (PySR symbolic regression)

Three hand-tuned scoring formulas are candidates for replacement by discovered expressions.
Feature tables come from one instrumented scoreboard pass (`return_diagnostics=True` exists on
`detect_and_select_lines`; extend the per-spectrum record dump in `_score_spectrum`).

### 4.1 Target 1 — element-presence decision score (highest value)

- **Replaces:** the ALIAS multi-metric composite in `cflibs/inversion/identify/alias.py`
  (`_evaluate_candidate_with_multi_metric_gate`: hand-weighted `0.30·physics_validity(R²) +
  0.15·interpretability(R_rat) + …`, hard threshold 0.5) and/or the comb fingerprint
  threshold/median-floor logic in `comb.py` (`_apply_relative_threshold`).
- **Training pairs:** one row per (spectrum, candidate element) over the optimization set —
  ≈350 spectra × ~10–15 candidates ≈ 4–5k rows. Features (all physics-legal observables already
  computed in the pipeline): n matched lines, matched fraction of expected lines, comb
  fingerprint correlation, NNLS coefficient, kdet score, coherent-line fraction, median |Δλ|
  residual vs consensus, Boltzmann R², energy spread (eV), median line SNR, rarity-weighted
  match count, n expected lines in band. Target: element ∈ `truth.elements_present` (binary).
- **PySR config:** binary operators `+ − × /`, unary `log exp sqrt abs`, complexity ≤ 25,
  custom class-weighted margin loss (FP cost > FN cost ≈ 3:1 to protect precision),
  `parsimony` tuned so the Pareto front spans complexity 5–25.
- **Validation:** the discovered score replaces the decision rule and the full pipeline re-runs
  on the optimization set (formula quality is measured by scoreboard F1, never by row-level AUC
  alone), then the standard adoption gate.

### 4.2 Target 2 — per-line match quality

- **Replaces/augments:** the per-line accept logic inside `detect_line_observations`
  (tolerance window + `line_residual_gate` consensus coherence) — a continuous quality score
  instead of binary gates.
- **Training pairs:** **synthetic corpus only** for labels — the generation recipe knows exactly
  which catalog lines are truly present, giving per-line truth that real datasets cannot
  (element-level truth is too weak). One row per (peak, catalog-line) match candidate:
  |Δλ| (units of tolerance), residual-vs-consensus z, line SNR, observed/expected intensity
  ratio (gA-scaled), isolation distance, blend count in window, NIST accuracy grade (as the
  existing `ATOMIC_UNCERTAINTIES` numeric map in `line_selection.py`), E_upper. Target: match
  is correct (binary). ~10⁴–10⁵ rows from the 288-spectrum corpus.
- **Risk control:** train-on-synthetic/validate-on-real — adoption requires real-dataset
  scoreboard improvement, not synthetic row accuracy (the corpus shares physics with the
  inversion; baseline doc's warning applies doubly here).

### 4.3 Target 3 — Boltzmann-fit line weighting

- **Replaces:** the inverse-variance weighting + `boltzmann_weight_cap=5.0` clamp +
  `aki_uncertainty_weighting` quadrature in `cflibs/inversion/physics/boltzmann.py`.
- **Training pairs:** indirect target, so build it explicitly: on synthetic spectra (known T),
  compute each line's leave-one-out influence on |T_fit − T_true|; regress
  w(SNR, A_ki grade, E_k, multiplet flag, self-absorption observable) against ideal influence
  weights. Smaller expected payoff than Targets 1–2 (solve stage is ~10 ms and already decent);
  schedule last, drop if Campaign 1 has already moved `boltzmann_weight_cap`.
- **KeplerAgent pattern (imitated):** before each PySR run, a Claude session examines the
  feature table (correlations, partial dependence, symmetry/scale checks) and fixes the
  operator set, feature subset, and dimensional constraints — structure first, regression second.

### 4.4 Budget and adoption

PySR runs are cheap: one 16–32-core CPU job, 1–4 h per target (Julia multithreading); GPU
unnecessary. Total ≪ 100 core-hours including reruns. **Adoption path:** export best-Pareto
expression via SymPy → transcribe as a pure NumPy function in the owning `cflibs/inversion/`
module with a provenance docstring (PySR version, run id, dataset-manifest + DB hashes,
complexity/loss Pareto table, holdout result) → unit tests pinning the formula's I/O →
blocklist scan (trivially passes — it is arithmetic) → standard adoption gate → PR. The
discovered formula is physics-*legal* by construction; whether it is physics-*sensible* is a
human-review question — reviewer must be able to gloss every term (e.g. "residual coherence
discounts the match count") or the PR is rejected.

---

## 5. Campaign 3 — Code Space (evolutionary code search)

### 5.1 What exists vs what is missing

`cflibs/evolution/` today is scaffolding: the AST blocklist scanner + `assert_benchmark_relevance`
(evaluator.py), driver runtime config with fitness weights/overfitting penalty/wallclock cap
(config.py), and the physics-grounding prompt preamble (prompts.py). **There is no driver
loop.** Build it as `cflibs/evolution/driver.py` (+ `lessons.py`, `archive.py`) rather than
adopting MLEvolve (unlicensed, Kaggle-shaped) or LessonL (paper artifact) — but imitate both:

- **From MLEvolve:** experience memory — every candidate record (diff, prompt, per-dataset
  deltas, verdict) is retrievable (BM25 over diff+lesson text is enough; no FAISS needed at our
  scale); stagnation detection (no archive improvement for N batches → trigger a structural
  mutation, wired to the existing `structural_mutation_cadence=10`); cross-branch fusion (ask
  the LLM to merge the diffs of two elite candidates from different archive cells).
- **From LessonL:** after each evaluation, solicit a ≤3-sentence lesson conditioned on the
  per-dataset deltas ("loosening comb_min_matches recovered aalto pure-element recall but added
  a Sn FP on csa"); bank lessons in the run dir; select top-k by relevance into the next batch
  prompt. Persist durable ones via `bd remember`.

### 5.2 Scope, sandboxing, provenance

- **Evolvable scope (write-allowlist):** `cflibs/inversion/identify/` and
  `cflibs/inversion/physics/line_selection.py` — the identification stage only. The scoreboard,
  `cflibs/benchmark/`, dataset adapters, truth files, and `cflibs/evolution/` itself are
  **read-only to candidates**; the driver rejects any diff outside the allowlist *before* the
  AST scan, and `assert_benchmark_relevance` rejects diffs that touch nothing exercised.
- **Candidate = a diff** against a pinned base SHA, applied in a disposable git worktree.
  Gate sequence per candidate: path allowlist → AST blocklist (`assert_physics_only`) → ruff
  TID251 + black + mypy → targeted unit subset (`pytest tests/inversion/identify -q
  --timeout=120`) → scoreboard evaluation. Any gate failure ⇒ fitness −inf with the failure
  banked as a lesson.
- **Sandboxing:** evaluation runs as a SLURM job in the worktree with: subprocess isolation,
  no network expectation (atomic DB is a local read-only file; open it read-only), no
  credentials in env, `--time` hard limit, per-spectrum timeout, output confined to the job
  dir. The AST scan + dynamic-import ban (already in evaluator.py) blocks import smuggling;
  SLURM cgroups bound CPU/mem.
- **Provenance:** candidate id = sha256(diff); record {parent id(s), batch, prompt hash, model
  id, diff, gate results, full scoreboard JSON, lesson} under
  `output/evolution/c3/<run-id>/candidates/<id>.json`. Elites archived in the (recall,
  precision) MAP-Elites grid; the archive is the unit of checkpointing.
- **LLM:** Claude (pin workflow agents to the latest Opus per project memory) generating 16
  perturbations/batch (`perturbations_per_batch`), prompts composed from
  `prompts.render_preamble()` + target-function source + top lessons + archive summary.

### 5.3 Budget and review gates

Per candidate ≈ gate suite (~2 min) + scoreboard eval (~10–15 core-min) ⇒ a 16-candidate batch
≈ 4 core-hours; ~10 batches/day sustainable. **Run cap: `max_wallclock_hours=72` per run
(existing config), program cap 1,000 core-hours.** Campaign 3 starts only after Campaigns 1–2
plateau — code search must beat the *tuned* baseline, otherwise it rediscovers parameter
changes expensively.

**Review gate before any merge (hard):** the standard adoption gate (Section 2.5) **plus**
human code review of the diff for physics sense, **plus** the PR must restate candidate lineage
and link the run artifacts. No auto-merge under any circumstance; the PR #229 regression
(−0.041 F1 from a plausible-looking "paper-faithful" change) is the standing reminder that
plausible diffs lose to the benchmark.

---

## 6. Infrastructure

### 6.1 SLURM templates (reuse `cflibs/hpc/slurm.py`)

Driver (one per campaign run):

```bash
#!/bin/bash
#SBATCH --job-name=cflibs-evolve-driver-<run-id>
#SBATCH --cpus-per-task=2 --mem=8G --time=72:00:00
# NO --nodelist — let SLURM schedule (cluster is shared with other agents)
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-cflibs-evolve"   # PRIVATE cache
python -m cflibs.evolution.driver --campaign c1 --run-dir output/evolution/c1/<run-id> \
    --manifest docs/benchmarks/manifests/optset-v1.json --budget-core-hours 600
```

Eval worker array (submitted by the driver each generation):

```bash
#!/bin/bash
#SBATCH --job-name=cflibs-evolve-eval-<run-id>
#SBATCH --array=0-15%16 --cpus-per-task=8 --mem=16G --time=00:45:00
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$SLURM_TMPDIR/jax-cache"          # job-private
python -m cflibs.evolution.eval_worker \
    --candidate "$RUN_DIR/gen$GEN/cand_$SLURM_ARRAY_TASK_ID.json" \
    --manifest docs/benchmarks/manifests/optset-v1.json \
    --db ASD_da/libs_production.db --out "$RUN_DIR/gen$GEN/result_$SLURM_ARRAY_TASK_ID.json"
```

Campaign 1 workers run the pinned-code scoreboard with a candidate *config*; Campaign 3 workers
first materialize the candidate worktree, run the gate suite, then the scoreboard. The optional
QDax cached-output mode is a single-GPU job (`--gres=gpu:1`, no node pinning).

### 6.2 Result store

`output/evolution/<campaign>/<run-id>/` (the repo's existing JSON-artifact convention):
`run.json` (manifest: git SHA, `ASD_da/libs_production.db` sha256, dataset-manifest hashes,
seed, optimizer config, env lock hash), `ledger.json` (core-hours, holdout-query count),
`gen*/cand_*.json` + `result_*.json` (full scoreboard output via the existing
`write_artifacts`), `archive.json`, `lessons.jsonl`, `incumbent.json`. Adoption-relevant
summaries get committed to `docs/benchmarks/` next to the baseline; raw run dirs stay
uncommitted.

### 6.3 Reproducibility

- **Frozen dataset manifests** (spectrum ids + file sha256 per split) committed under
  `docs/benchmarks/manifests/`; the eval worker verifies hashes before running.
- **Frozen DB:** record and verify the sqlite file sha256 every run; any DB regeneration
  obsoletes all in-flight runs (datagen is hours-long and changes line lists).
- **Seeds:** scoreboard sampling seed fixed (20260610); optimizer seeds recorded; per-generation
  synthetic-subsample seed derived as `seed + generation`.
- **Environment:** `uv` lockfile hash in `run.json`; JAX x64 CPU (deterministic); per-spectrum
  independence means worker scheduling cannot change scores.
- **Re-derivability:** `python -m cflibs.evolution.replay <run-id> <candidate-id>` re-runs one
  candidate end-to-end from the manifest.

### 6.4 Kill-switch and budget controls

- `touch $RUN_DIR/STOP` — driver checks between generations, cancels its arrays
  (`scancel --name=cflibs-evolve-*-<run-id>`), checkpoints, exits 0.
- Ledger enforcement: driver refuses to submit a generation that would exceed
  `--budget-core-hours`; `max_wallclock_hours` (config.py) bounds the driver job itself.
- Array throttling (`%16`) keeps the program from monopolizing the shared vasp-01/02/03 nodes;
  generation-level checkpointing makes preemption lossless.
- Holdout-query counter in the ledger; exceeding the per-phase quota locks the holdout gate
  until a human resets it (this is the overfitting kill-switch, not just a courtesy).

---

## 7. Risks and Mitigations

| Risk | Concrete failure mode | Mitigation |
|---|---|---|
| **Overfitting to small real datasets** | BHVO-2 has 4 spectra; aalto 74. A 45-knob search can memorize them; CodeEvolve Wave 1's 0.914 was exactly such a strong-local-fit on 74 spectra. | BHVO-2 entirely in holdout; target-identity splits; paired bootstrap + minimum-improvement thresholds; cross-dataset variance penalty (0.5, already in config.py); holdout-query quota with ledger lock; vault dataset (gibbons2024) untouched until final report. |
| **Fitness gaming — presence threshold** | Candidates push elements just over `PRESENCE_EPS_MASSFRAC=5e-3` (or call almost nothing to keep precision 1.0). | Scoring constants and confounder set live in scoreboard code that candidates cannot touch (path allowlist); micro-F1 punishes recall collapse; FP death penalty punishes spray; diagnostics log the solved-fraction histogram near 5e-3 — a pile-up at the threshold flags gaming for human review. |
| **Fitness gaming — window-edge riding** | ye6t showed match-count-maximizing objectives ride tolerance-window edges (Al I 892.356 admitted at exactly +0.05 nm). | Per-line residual-distribution diagnostics (fraction of matches in the outer 10% of the tolerance window) recorded per candidate; adoption review rejects edge-riders; `line_residual_gate` stays in the search space so the optimizer can tighten, not only loosen. |
| **Gaming the failure rule** | "Failed spectra call nothing present" — a candidate that *fails* on hard spectra avoids FPs on them. | Death penalty on failure-count increase (Section 2.2); failures already count all truth elements as FN, so failing is recall-expensive too. |
| **Synthetic-real gap** | Synthetic corpus shares the forward model; F1 gains there may be inversion-flattering artifacts. | Synthetic weight 0.5 and relative-tracking-only policy (already stated in the baseline doc); adoption evidence must include real-dataset improvement; Campaign 2 Target 2 trains on synthetic but is validated on real scoreboard deltas. |
| **Eval nondeterminism** | Wall-time noise on shared nodes; JAX cache cold-starts; rng drift. | Fixed seeds + frozen manifests; runtime uses medians and only penalizes beyond budget (and the adoption runtime gate uses a dedicated quiet re-run); incumbent re-evaluated each generation — if its fitness moves beyond bootstrap noise, the generation is voided and the environment inspected; `OMP_NUM_THREADS=1`, x64 CPU JAX. |
| **Cluster etiquette** | Starving the shared vasp cluster or other agents' jobs; shared-cache corruption. | No `--nodelist`; array throttling `%16`; CPU-only requests for Campaigns 1–3 (GPUs only for the optional QDax cached mode); private JAX compilation caches per job/user; budget ledger; STOP-file kill-switch; generation checkpoints make preemption free. |
| **LLM candidate quality / cost (C3)** | Wasted batches on syntactically broken or irrelevant diffs. | Cheap gates first (path allowlist → AST → lint → unit subset) so broken candidates cost seconds; lesson bank (LessonL) recycles failure information; relevance gate (`assert_benchmark_relevance`) blocks zero-effect diffs; perturbation batching with `perturbation_timeout_s`. |
| **Plausible-but-worse code merges (C3)** | PR #229: a justified, paper-faithful change regressed F1 −0.041. | Benchmark-gated adoption is mandatory for every identifier change (institutional memory); no auto-merge; human physics review with the lineage + scoreboard tables in the PR body. |
| **Scoreboard branch churn** | The fitness function is being built on `overhaul/w4-scoreboard` right now; datasets land on `overhaul/w4-datasets`. | The program pins a base SHA per run (in `run.json`); campaigns start only after w4-scoreboard + w4-datasets merge to `dev`; the committed baseline doc is regenerated at that SHA before generation 0. |

---

## 8. Recommended Execution Order

1. **Now (blocked only on w4 merge):** freeze split manifests; plumb the Section 3.1-B
   `detect_line_observations` knobs into `AnalysisPipelineConfig`; implement the composite
   fitness + bootstrap as a thin wrapper over `run_scoreboard`.
2. **Campaign 1 phase A (first run, highest expected value):** Optuna TPE, 800 trials over the
   ~45 knobs, optimization set, 16-way SLURM arrays — the recall-vs-precision gates have never
   been jointly tuned end-to-end, the failures cluster on detection gates, and Wave 1 showed
   threshold tuning alone is worth tens of F1 points in a weaker setting. Then phase B CMA-ES
   refinement; holdout-gate the top-5; ship as preset values.
3. **Campaign 2:** instrument diagnostics → feature tables; PySR Target 1
   (presence score), then Target 2 (line quality); Target 3 only if Campaign 1 left T-error on
   the table.
4. **Campaign 3:** build `driver.py`/`lessons.py`/`archive.py` on the existing scaffolding;
   evolve `identify/` against the post-C1/C2 incumbent; 72 h runs, human-reviewed PRs only.

# ADR-0004 — Ground-up jittable inversion pipeline (`cflibs/jitpipe/`)

- **Status:** ACCEPTED & IMPLEMENTED — the ADR was accepted by starting J0; J0–J12 have since shipped (`cflibs/jitpipe/`, ~20 modules; see [`docs/jitpipe/J12-status.md`](../jitpipe/J12-status.md) for the M3 promotion harness and J9 completion). The original proposal text below (§8 start criteria, "do not start J1–J9 before §8.4") is retained as historical record of the acceptance gate.
- **Date:** 2026-06-10
- **Authors:** CF-LIBS Architecture working group (Brian Squires + Opus 4.7 design team, four parallel read-only analyses 2026-06-10)
- **Inputs:** four independent design analyses against worktree `w4-integration` (== `dev` tip `0720737`, Phase A scoreboard merged): (1) front-end per-stage jittability with measured padding bounds from 32 real spectra across 8 scoreboard datasets; (2) back-end per-stage jittability with device-memory sizing against `ASD_da/libs_production.db`; (3) architecture/infrastructure/program mechanics with live adapter measurements; (4) prior-art survey + staging plan + acceptance economics. Conflicts between the analyses are resolved explicitly in §3.
- **Tracking beads:** epic `jit-pipeline` with children J0–J12, specs at [`specs/J0-skeleton-snapshot-contracts.md`](./specs/J0-skeleton-snapshot-contracts.md) … [`specs/J12-scoreboard-promotion-m3.md`](./specs/J12-scoreboard-promotion-m3.md)
- **Related:** [ADR-0001](./ADR-0001-radis-jaxrts-pattern-survey.md) (host/kernel split, unified forward kernel — this ADR extends both), specs T1-1…T1-6, `docs/jax-port/*` consultations, `docs/audit/2026-06-09-overhaul/*`, `docs/audit/2026-06-10-goalfirst/optimization-program-design.md`, `docs/benchmarks/SCOREBOARD-2026-06-10-baseline.md`
- **Citation pin:** every `file:line` reference below is against `dev` commit `0720737` unless stated otherwise.
- **Scope:** the decision to build a parallel, fixed-shape, end-to-end jittable (JAX) implementation of the CF-LIBS inversion pipeline targeting 3× V100S (one GPU per SLURM node), its architecture, parity contracts, staging, and promotion/deprecation criteria.
- **Out of scope / non-goals:** bit-identity with the reference pipeline (tolerance contracts only, §5.4); modifying the reference pipeline (it is frozen as the parity oracle); any ML (physics-only constraint applies unchanged — ruff TID251 covers `cflibs/jitpipe/` automatically); multi-host `pmap` (one GPU per node ⇒ SLURM-array data parallelism suffices); the Rust native paths (reference-pipeline-only, §2 D7).

---

## 1. Context

### 1.1 The reference pipeline and its measured limits

The production path is `run_pipeline` (`cflibs/inversion/pipeline.py:666`): response correction → segmented wavelength calibration (`pipeline.py:488`) → `detect_and_select_lines` (`pipeline.py:337`) → Stark-n_e diagnostic (`pipeline.py:762-794`) → `IterativeCFLIBSSolver.solve` (`pipeline.py:797-818`). It is NumPy/SciPy/Rust with data-dependent shapes everywhere before the solve. Measured facts that motivate this ADR:

- **Wall time is front-end-dominated.** On bhvo2_chemcam the stage medians are calibration 1.55 s + detection/ID 0.89 s vs **solve 0.01 s** of 2.64 s total (`docs/benchmarks/SCOREBOARD-2026-06-10-baseline.md`). Reference throughput is 0.4–5.1 s/spectrum across datasets.
- **GPUs are idle.** During identifier sweeps on vasp-01/02/03, GPU utilization was **0–24 %, mean ~3 %** — Amdahl-bound on the non-JAX per-spectrum fraction, with **7,394 jit-cache entries** accumulated because variable per-spectrum shapes (`n_lines_detected`) force recompiles (`docs/adr/ADR-0001-empirical-analysis-2026-05-13.md:35-36`). Incremental kernel ports cannot fix an Amdahl + shape-instability problem.
- **The scoreboard's binding constraint is recall** (R 0.27–0.58 vs P 0.89–1.00 across datasets; `SCOREBOARD-2026-06-10-baseline.md:16-27`), and the audit names forward-fitting / multi-line intensity coherence as the missing structural physics (`docs/audit/2026-06-09-overhaul/00-synthesis.md:23,80`, `03-identification.md:19,102`) — which needs 10³–10⁴ forward evaluations per spectrum, infeasible on the CPU reference.
- **The existing lax solve path never runs in production.** The geological preset sets `saha_boltzmann_graph=True` and `closure_mode="oxide"` (`cflibs/inversion/pipeline.py:60-63,144-146`) and wires Stark diagnostics into the solve (`pipeline.py:762-794`); *either* condition forces the Python reference loop — `solve()` only routes to `_solve_lax` when `not self.saha_boltzmann_graph and not diags` (`iterative.py:1592-1613`). The lax body jits the legacy algorithm (per-element-centered common-slope plane + 1-atm pressure-balance n_e) that the 2026-06-09 audit demoted (audit 02 finding F2: pressure balance "physically invalid", `iterative.py:1751-1759`; SB-graph validated RMSE 8→4.29 on BHVO-2).

### 1.2 The three motivations (plus one derived benefit)

1. **Batched throughput.** Full-population scoreboard/campaign evaluation: the corpus (~1,264 registered spectra) costs 45–75 CPU-minutes serial today; campaigns subsample 12-of-N per dataset and inherit sampling variance in the fitness signal. A batched GPU pipeline removes that variance at equal cost (§8.3).
2. **Forward-eval scale for identification.** The prior-art ceiling is 2.5×10⁷ forward evals/spectrum on a 2013 Tesla K40 (Gornushkin & Völker 2022, §9). At ~10⁵–10⁶ fused forward evals/s on a V100S, 10⁴ evals/spectrum ≈ 10–100 ms — this is the GPU mode of the Campaign-3 identification endgame.
3. **Differentiability.** Joint gradient solve over (T, n_e, composition), HMC over the full model, and gradient knob-tuning of continuous pipeline parameters — none possible through the host SciPy/greedy front-end (§6).
4. *(Derived)* **Determinism.** Fixed shapes + seeded counter-based PRNG + `--xla_gpu_deterministic_ops=true` turn scoreboard deltas into pure signal (§5.5).

### 1.3 Premise inversion of the prior consultations

`docs/jax-port/line-detection-consultation.md` (2026-05-12) ruled **against** porting `find_peaks` and the matching logic. Its verdict was explicitly premised on B=1 single-spectrum CPU work ("For B=1 N=4096, mask representation is plenty fast"; "Unless you are running this pipeline in a loop thousands of times… The exception: batched analysis (B=1000 spectra) or differentiation", lines 43-46, 85-89). The motivations above are exactly the carve-out both consultants named. The consultation's *taxonomy* of hard parts (variable-length peaks, prominence semantics, greedy used-sets) remains the correct hazard map and is addressed stage-by-stage in §4 and the J-specs. The same inversion applies to the optimization-program design's QDax ruling ("the bottleneck is the CPU pipeline (seconds/spectrum, not jittable)", `docs/audit/2026-06-10-goalfirst/optimization-program-design.md` §1) — this ADR removes that bottleneck rather than working around it.

### 1.4 What already exists (assets the rewrite builds on)

| Asset | Location | State |
|---|---|---|
| Unified jittable forward model (3-stage Saha+IPD, per-line widths, Gaussian/Voigt, slab RT) | `cflibs/radiation/kernels.py:763` (`forward_model`); static/traced split documented at `kernels.py:28-38` | Done (T1-2) |
| Memory-safe chunked forward (`lax.scan` + checkpoint + overlap-and-add) | `kernels.py:1205` (`forward_model_chunked`), `:1411`, `:927` | Done (T1-5) |
| LDM broadening (19× at N_lines=1500; **default OFF pending real-data validation**) | `cflibs/radiation/ldm.py` | Done (T1-4), validation gap is risk R4 |
| Device-resident padded atomic data, no-sqlite-inside-loop | `iterative.py:381` (`_AtomicSnapshot`) and `cflibs/core/jax_runtime.py:430` (`AtomicSnapshot`, pytree at `:553`) — **two snapshot types; J0 unifies** | Done (T1-1/T1-3) |
| lax.while_loop solve with parity/vmap/grad-smoke tests | `iterative.py:721,2227`; `tests/inversion/test_iterative_lax.py:183-204,273-287,344-356,415-429` | Done (T1-3) but never production-routed (§1.1) |
| Bit-exact JAX twins of the fit kernels | `_saha_correct_kernel` (`iterative.py:2675`), `_common_slope_kernel` (`iterative.py:2698`), `batched_boltzmann_fit` (`physics/boltzmann_jax.py:111-200`) | Done |
| vmap batch generation pattern | `cflibs/manifold/batch_forward.py:13` | Done |
| Seed JAX kernels in the identifier | `_jax_peaks_within_tolerance` (`identify/line_detection.py:76`), `_jax_kdet_candidate_counts` (`:118`), pure-JAX alias/comb/correlation ports per `docs/jax-port/*` | Done (seeds, not the rewrite) |
| Joint optimizer parameterization (log-T, log10-n_e, softmax simplex) | `cflibs/inversion/solve/joint_optimizer.py:16-23,42-48` | Done (shipped, physics-legal) |

Neither surveyed JAX spectroscopy code (ExoJAX2, jaxrts — §9) has shipped an inversion pipeline; CF-LIBS's `_solve_lax` with vmap+grad tests is already *ahead* of both on the inversion side. The rewrite extends a lead, it does not chase one.

---

## 2. Decision

**Build `cflibs/jitpipe/` — a parallel, fixed-shape, end-to-end jittable implementation of the inversion pipeline — promoted (or abandoned) strictly by scoreboard criteria.** Eight binding sub-decisions:

- **D1 — Parallel implementation, not a port.** New package `cflibs/jitpipe/` (layout in §5.1), module-per-stage, mirroring the reference stage graph. `run_pipeline` (`pipeline.py:666`) is untouched; no `use_jax` flag inside the reference (the kwarg-flag anti-pattern was already rejected in `docs/jax-port/iterative-boltzmann-consultation.md` Q1). The two implementations meet at exactly two points: the scoreboard CLI dispatch (`--pipeline {reference,jit}`, §5.4) and the parity tests.
- **D2 — Fixed shapes end-to-end.** Padded arrays + validity masks, fixed K peaks / L lines / E elements per bucket, all raggedness handled by masks inside the jit graph, all failure signaling by quality flags instead of exceptions (`pipeline.py:751-752`'s `ValueError` becomes a `failed` mask interpreted host-side). This is the non-negotiable design rule derived from the 7,394-cache-entry / 3 %-GPU finding (§1.1).
- **D3 — One-way dependency.** `jitpipe` may import `cflibs.radiation.{kernels,ldm,host}`, `cflibs.core.jax_runtime`, `cflibs.inversion.common` dataclasses, and the preset constants at `pipeline.py:42-79` (single source of preset truth — never copied). Nothing outside `jitpipe` imports `jitpipe`. Enforced by an import-hygiene test mirroring `tests/test_jax_import_hygiene.py` (the pattern is load-bearing: `kernels.py:72-78`).
- **D4 — Host/kernel boundary per ADR-0001, applied to the whole pipeline.** SQLite, I/O, CLI/config, diagnostics-dict and warning-string assembly, CCD-seam *detection caching*, dynesty drivers, Rust paths, `runtime/`/`pds/`/`hpc/`, and the analytical `uncertainties` UQ path stay host-side permanently (full list in §5.1.3). Kernels are pure functions over `(PipelineSnapshot, PipelineParams, batch arrays)` returning fixed-shape pytrees of arrays/masks/uint8 reason codes; thin host wrappers reconstitute today's result dataclasses (`WavelengthCalibrationResult`, `LineDetectionResult`, `CFLIBSResult` at `iterative.py:81`) so downstream consumers and parity tests see identical types.
- **D5 — Exactness-first front-end; re-derived back-end estimator.** Decision-bearing discrete front-end logic (peak identity, greedy matching, tie-breaks, gate bookkeeping) is ported with **exact reference semantics** as fixed-shape scans — no "parallel approximations" (§3 C1, C2). The back-end production estimator is **re-derived** as a differentiable joint weighted-least-squares solve whose Gauss–Newton step 0 is algebraically identical to the current SB-graph (§3 C4, §6.1) — with the extended fixed-iteration loop kept as initializer and parity anchor.
- **D6 — Promotion by scoreboard, deprecation in stages.** Promotion criteria in §8.2 (M3). Pre-promotion, the reference is the contract: any divergence is presumed a jit bug until adjudicated in a divergence ledger. Post-promotion the reference remains the parity oracle (`--pipeline=reference`) for ≥2 releases, receives bug fixes only (each paired with a parity-test update), then demotes to a validation oracle; final removal requires an independent oracle (NIST parity + golden corpus) and a ROADMAP entry.
- **D7 — Physics-only constraint unchanged.** `jitpipe` lives under `cflibs/`, so ruff TID251 bans `jax.nn` etc. automatically; smooth relaxations are explicit `jnp` expressions or reuse the shipped `cflibs.inversion.physics.softmax_closure`. The Rust comb/k-det paths (`identify/line_detection.py:33-46`) serve the reference pipeline only — PyO3 callbacks inside jit kill fusion and vmap; the jit pipeline uses the pure-JAX ports.
- **D8 — fp64 default, policy-gated fp32 island.** §5.3. x64 end-to-end (V100S fp64 = ½ fp32 rate, and the broadening stage is memory-shaped, not FLOP-starved); fp32 opt-in only for the profile-matrix values and identify-stage template scoring, behind a Tier-B shadow gate.

---

## 3. Conflict resolutions between the four analyses

The four analyses were tasked independently and disagree in six places. Resolutions, with reasons, are binding on the J-specs:

- **C1 — Peak detection: exact scipy semantics (Analysis 1) vs approximate vision-NMS (Analysis 4 §1.5, J2: Jaccard ≥ 0.90).** *We choose exact* (Analysis 1): plateau-aware local maxima, exact `wlen=None` prominence via range-query sparse tables, scipy's greedy distance-NMS as a `lax.scan`, exact `peak_widths`. Reasons: (a) every downstream stage keys off **peak identity** — a 10 % peak-set drift makes Tier-D decision parity (§5.4) untriageable; (b) Analysis 1 showed exactness is achievable in fixed shapes at bounded cost (5–7 pd); (c) the contract "byte-identical index lists" turns the classic silent-divergence trap (plateau/tie semantics, flagged by the consultation doc at `line-detection-consultation.md:119-123`) into a hard test. Jaccard ≥ 0.995 with per-diff triage is the documented *fallback* contract only.
- **C2 — Wavelength-calibration hypothesis scoring: parity-anchored parallel RANSAC (Analysis 1) vs MAGSAC++ sigma-marginalized soft consensus (Analysis 4 §1.4/J3).** *We choose Analysis 1 for the build that must pass parity*: exhaustive/stratified vmapped hypothesis evaluation, scoring by the unique-inlier-count upper bound, **exact greedy one-to-one dedupe (`_dedupe_one_to_one`, `wavelength_calibration.py:209-230`) run only for the winning hypothesis and refine rounds** as a `lax.scan`. Reasons: the ye6t coverage-gate regression fixtures (ChemCam VNIR / BHVO-2 877 nm Al doublet) demand close behavioral parity, and MAGSAC++ changes inlier semantics, so it would have to be revalidated against the scoreboard rather than the reference. MAGSAC++ IRLS is retained as an **optional differentiable variant behind the same kernel interface** for the gradient-tuning path (J2 §opt, J11) — it is the right tool there, not for parity.
- **C3 — Wavelength-axis shape policy: static pad buckets {4096…65536} (Analysis 1) vs exact per-instrument shapes (Analysis 3).** *We choose exact shapes* (Analysis 3): the grid length is measured constant within each dataset/instrument (§5.2), so one bucket per distinct grid (7 cover all 10 datasets) with zero pad waste; power-of-two padding would waste 22–64 % on silva2022/emslibs2019. A pad *ladder* exists only for genuinely heterogeneous streams (DAQ runtime). Analysis 1's measured caps for the **other** axes (P_max, L_max, K_pair, E_max, …) are adopted as the padding-constants table (§5.2) — they are measurement-backed where Analysis 3/4's K=512/128 peak caps would truncate (2,350 peaks observed on emslibs2019).
- **C4 — Solve: reuse/extend the lax while_loop (Analysis 3 `solve.py` re-host) vs re-derive as joint WLS (Analysis 2 §8) vs research-only joint solve (Analysis 4 J8).** *We adopt Analysis 2's verdict*: extend the lax path's **component kernels and snapshot pattern verbatim** (they are correct and tested), but the production jit estimator is the **joint WLS Gauss–Newton solve** with n_e pinned to the Stark measurement, and the dynamic `while_loop` is replaced by a **fixed-K `lax.scan` with converged-state freezing** serving as initializer and parity anchor. Reasons: (a) the lax path is ~70 % of the *legacy* algorithm but ~40 % of the *production* algorithm and 0 % of the differentiability goal (§1.1; `pure_callback` ILR default at `iterative.py:1046-1050,614-655` serializes under vmap); (b) audit 02-F10 already recommends "keep the iterative solver as initializer; promote the closed-form/joint estimator" (`docs/audit/2026-06-09-overhaul/02-inversion-solver.md:322-346`); (c) the GN-step-0 ≡ SB-graph algebraic identity (§6.1) gives an exact parity anchor, so re-derivation is *not* a parity leap. We deliberately do **not** keep three solve drivers: scan + joint GN only; the while_loop variant is superseded (Analysis 3's two-variant proposal is narrowed accordingly).
- **C5 — Throughput gate: ≥10× median spectra/s (Analysis 3 promotion criterion 4) vs ≥50 spectra/s ≈ ≥100× (Analysis 4 M2).** *Both, at different gates*: M2 (engineering milestone) targets ≥50 spectra/s/GPU amortized over ≥1,000 spectra including compile; M3 (promotion) requires the hard floor ≥10× batched median vs the 0.4–5.1 s/spectrum baseline plus CPU single-spectrum ≤2× reference. M2 missing 50 but clearing 10× does not block promotion; it triggers a gap bead.
- **C6 — Element-axis bound: Analysis 1 measured ≤15 candidate elements (32-spectrum corpus, csa 15) while Analysis 3 quotes csa_planetary as 16 truth + 5 confounders.** Not worth re-adjudicating: **E_max = 32** covers either with headroom; species axis S_max = 48 (Analysis 3 measured 30 species for a 15-element set). The padding-invariance test (§5.4) makes the exact constant a non-risk; K-saturation counters (overflow flags) are mandatory outputs on every padded axis.

Effort-unit reconciliation: Analyses 1–2 estimate in person-days (front-end 33–46 pd; back-end ≈27–44 pd), Analysis 4 in agent-sessions (36–50). Repo history (six T1 beads in ~1–2 days of parallel agents) supports ~2–3× agent leverage over person-day estimates; both are quoted in §8.3 and the truth will be measured at M1.

---

## 4. Per-stage jittability summary

Full detail (every breaking construct with file:line, kernel sketches, measured bounds) lives in the J-specs; this table is the decision-level summary. "Contract" = what must agree with the reference on identical inputs; tiers per §5.4.

| # | Stage (reference site) | Jit-breaking constructs (headline) | Fixed-shape redesign | Tolerance contract | Effort / risk |
|---|---|---|---|---|---|
| 1 | Baseline + noise (`preprocess/preprocessing.py:619` `detect_peaks_auto` baselines; `estimate_noise` `:320`) | scipy `median_filter`/`percentile_filter`/`sparse.spsolve`; ALS early-break (`:272`); shrinking sigma-clip arrays (`:340-349`) | SNIP = direct port (fixed 40 iters); MEDIAN/PERCENTILE = sliding-window sort, row-chunked `lax.map`; ALS = fixed-20 IRLS + banded LDLᵀ scan; noise = masked 3-iter sigma-clip | SNIP/MEDIAN/PCTL rtol 1e-12; ALS ≤1e-6·scale(y); noise exact | 3–4.5 pd / LOW (fp64 mandatory) |
| 2 | Peak detection (`preprocessing.py:524` `detect_peaks`; `identify/line_detection.py:2424` `_find_peaks` — two parameterizations, one kernel) | `scipy.signal.find_peaks` variable-length output; non-local prominence; greedy height-order distance-NMS; plateau midpoints; `peak_widths` try/except (`:474-481`) | run-boundary plateau maxima; exact prominence via (N,⌈log2N⌉) range-query sparse tables; NMS as height-sorted keep/suppress `lax.scan` (P_cand_max=8192); widths from prominence bases; `(P_max,)` indices + mask + count | **byte-identical index lists** vs scipy on corpus + property tests (fallback Jaccard ≥0.995 w/ triage); board ΔF1 ≤ 0.005 | 5–7 pd / MED (plateau/tie semantics) |
| 3 | Wavelength calibration (`preprocess/wavelength_calibration.py:1351` segmented; `:602` global; RANSAC `:256-356`; gates `:96-133,976-1078`; seams `:829`) | 600-iter host RNG RANSAC w/ rejection sampling (`:240-253`) + greedy `_dedupe_one_to_one` inside hypothesis loop (`:209-230,285`); re-entrant degrade-to-shift recursion (`:1053,1495`); Python segment loop (`:1210`); `fits.sort` model select (`:809`); seam clamp cascade (`:1276-1286`) | banded pairs (P_max=2048, K_pair=48); shift model evaluated exhaustively (every live pair = 1-pt hypothesis); affine/quad H=4096 stratified counter-PRNG hypotheses, H-chunked; upper-bound scoring + exact dedupe scan on winner only; model lattice replaces recursion; segment vmap (SEG_max=16); centered-x normal equations (deliberate deviation, SVD-conditioning) | gate outcomes equal (excl. ±5 %-of-threshold cases); corrected axis max|Δλ| ≤ 0.04 nm (= inlier_tol/2); model class ≥90 % of cells; RMSE ±max(30 %, 0.02 nm); board ΔF1 ≤ 0.01; reference self-variance across ~10 seeds measured first | 12–17 pd / MED-HIGH behavioral (ye6t fixtures mandatory) |
| 4 | Line matching + gates (`identify/line_detection.py:1099` `detect_line_observations`; greedy matcher `:1919-1987`; veto `:2287`; observation build `:786`; intensity `:459`) | SQL-per-element + sorted ranking (`:1502,1550-1579`); element dicts; mutable `used` sets; `np.isclose` tie-breaks (`:1841-1848`); cross-element peak ownership (`:843-860`); gated-without-consuming-peak (`:1958-1975`); retroactive `del observations[...]` (`:971`) | FrontEndSnapshot (E_max=32, K_lines=64, K_comb=32); kdet generalizes existing `_jax_kdet_candidate_counts` (`:118`); comb greedy as 32-step scan carrying used-peak bitmask, vmapped over (S_shift×E_max); quantized-isclose lexicographic selection; ownership scan in (f1,matched)-rank order; drop rule = mask zeroing (peaks stay claimed, exact port of `:961-963`); trapezoid + FWHM-fallback vectorized; ObservationBatch (OBS_max=512) + warning bitmask | `applied_shift_nm` exact; accepted/kept element sets identical; obs line-key sets identical (floor Jaccard ≥0.98 triaged); intensities/σ rtol 1e-10; n_gated/dropped maps equal; expected ΔF1 = 0, gate ≤ 0.005; BHVO-2 Sn/Th canaries | 12–17 pd / HIGH concentration of one-line-to-get-wrong logic — **no semantic drift allowed in this stage** |
| 5 | Line selection (`physics/line_selection.py:176-239`; isolation `:362-389`; top-K `:266-290`) | variable-length lists; Python stable sort + truncation; reason strings | pure mask transform on (B,L); masked pairwise isolation; `lax.top_k` K=20 with (score, −index) lexicographic packing; uint8 reason codes | exact set equality (deterministic index tiebreak); scores rtol 1e-12 | 1–2 pd / LOW |
| 6 | Boltzmann fits + SB-graph (`physics/boltzmann.py:326-842`; plane `iterative.py:1349-1452`; SB-graph `:1454-1539,2776-2886`; weight cap `:2911-2944`) | sigma-clip mask shrinkage + breaks; RANSAC host RNG; `np.linalg.lstsq` on dynamic (N_rows,1+E); masked `np.median` | sigma-clip → fixed-K masked reweighting over `batched_boltzmann_fit`; Huber fixed-K IRLS; RANSAC → vectorized trials (jax.random); **SB-graph = `_saha_correct_kernel` + `_common_slope_kernel` w=1 via arrow-matrix Schur identity** (independently confirmed: `02-inversion-solver.md:378`); sort-based masked median | SB-graph slope/intercepts rtol 1e-10 vs lstsq (algebraic identity, property-tested incl. degenerate fixtures); T rtol 1e-8; sigma-clip mask bit-equal off-boundary; RANSAC self-regression only (RNG streams differ by construction) | 3–5 pd / LOW-MED |
| 7 | Self-absorption, observable-gated (`physics/self_absorption_observable.py:354-427`; doublet root `self_absorption.py:424-447`) | `brentq`; combinatorial pair discovery; order-dependent first-pair-wins (`:480`); float-keyed dicts | pair discovery is atomic-data-static → snapshot (P,2) pairs + ρ, r_thin; fixed-60-step bisection (≪ brentq xtol 1e-6); claim resolution via scatter-min over pair indices; Planck ceiling + 64-term COG series port verbatim; suspect pass = masked median + where | τ atol 1e-6; corrected I rtol 1e-6; corrected/suspect/cleared sets identical under documented pair-priority contract; counters exact | 3–5 pd / LOW-MED (adversarial shared-upper-level triplet fixture) |
| 8 | Stark n_e (`physics/stark_ne.py:394-663`; pinned-Gaussian Voigt fit `:257-313`; solver coupling `iterative.py:1250-1280`) | `scipy.optimize.least_squares`; **DB query inside candidate loop** (`:332-371,356`); half-max crossing loops; irregular windows | Stark metadata → snapshot (28,331/28,727 lines have `stark_w`; literature-grade `stark_b` = 244 — measured); blend gate precomputed; fixed (C, W=64) raw-sample windows; vmapped LM, K≈20, 5×5 damped normal eqs, exp-transform feasibility; gates→masks, top_k 5; masked median/MAD + cohort trim; per-iteration width-law inversion drops into the solve body | per-line n_e rtol 1e-3 when both fitters converge (contract on solution, not optimizer path); median rtol 1e-3; candidate-set equality under documented tiebreak; percentile = exact np.percentile interpolation | 5–10 pd / **MED-HIGH** (window-extraction parity; LM vs scipy-trf on marginal profiles handled by the same gates) |
| 9 | Closure, 6 modes + keystone gate (`physics/closure.py:537-1033`; gate `:495-535`; lax callbacks `iterative.py:567-655`) | dict plumbing; `sorted()` ordering; missing-U `continue`; `pure_callback` for ilr/pwlr/dirichlet (serializes under vmap, non-differentiable) | masked linear algebra over (B,E); lift existing lax closures (std/matrix/oxide) one batch axis; **ILR ≡ standard** (identity round-trip, audit-confirmed `02-inversion-solver.md:364-365`) with the 1e-10 clip for parity; PWLR closed-form; Dirichlet `where`; keystone gate in-body (one comparison) | std/matrix/oxide rtol 1e-12; ilr/pwlr rtol 1e-10 + ILR≡standard equivalence property; degeneracy flag exact | 1–2 pd / LOW |
| 10 | Iterative solve → joint estimator (`iterative.py:1783-1964` Python; `:721-863` lax; routing `:1592-1613`) | host seams: SB-graph + Stark routing; `pure_callback` ILR default; pressure-balance-only lax n_e (`:2358-2364`); host keystone gate (`:2344-2365`); corona weighting deferred (`:810-818`); IPD formula drift (`:778-781` vs `plasma/saha_boltzmann.py:542+`); non-differentiable dynamic while_loop; `_LaxFallback` exceptions (`:213-220`) | fixed-K scan (max_iterations=20) w/ converged-freeze, SB-graph in-body (row 6 kernel), Stark branch (row 8), native ILR, in-body keystone, unified IPD, validity flags — **as initializer**; production estimator = joint WLS GN/LM over θ=(lnT, α_ILR, β), n_e pinned to Stark w/ MAD penalty (§6.1) | inherited rtol 1e-5 vs `_solve_python` incl. SB-graph+oxide+Stark configs current tests cannot cover; GN-step-0 ≡ SB-graph rtol 1e-10; converged (T,C) vs reference on 10 datasets: T rtol 1e-3, per-element C atol 0.5 wt %; covariance vs FD-Hessian rtol 1e-4; grad-finiteness as **hard** asserts | 10–15 pd / MED |
| 11 | Forward model | — (already jittable) | thin `forward.py` wrapper over `kernels.py:763/:1205` and LDM — **no physics duplication**; `kernels.py` stays the single source of truth (`kernels.py:1-26`) | existing kernel contracts (rtol 1e-5, `kernels.py:856`) | 0 (reuse) |

Combined stage effort ≈ 60–90 pd; with program glue (J0, J8–J12) ≈ 85–130 pd ≈ 40–60 agent-sessions (§3 reconciliation, §8.3).

---

## 5. Architecture

### 5.1 Package layout, boundaries, and the snapshot

#### 5.1.1 Layout

```
cflibs/jitpipe/
  __init__.py     # public API: run_batch(), run_one(), PipelineSnapshot, PipelineParams, StaticConfig
  snapshot.py     # PipelineSnapshot builders (HOST-ONLY: SQLite here, never in kernels)
  params.py       # PipelineParams pytree (ALL continuous knobs traced) + StaticConfig (hashable statics = jit cache key)
  preprocess.py   # baseline, noise, response multiply                     (row 1)
  detect.py       # exact find_peaks kernel, both parameterizations        (row 2)
  calibrate.py    # segmented RANSAC wavelength calibration                (row 3)
  identify.py     # line matching + gates; kdet/comb/ALIAS/NNLS scoring    (row 4; reuses docs/jax-port FISTA-NNLS/ALIAS/comb designs)
  fit.py          # selection, Boltzmann/SB-graph, closure                 (rows 5, 6, 9)
  selfabs.py      # observable self-absorption                             (row 7)
  stark.py        # Stark n_e vmapped LM                                   (row 8)
  solve.py        # fixed-K scan initializer + joint WLS estimator         (row 10)
  forward.py      # thin wrapper over radiation/kernels + LDM              (row 11)
  pipeline.py     # the jitted composition: stage graph, bucket dispatch, packed outputs
  host.py         # everything impure: DB→snapshot, padding/bucketing, memory planning
                  #   (reuse cflibs/radiation/host.py:164 available_device_bytes, :189 plan_chunks),
                  #   unpacking to reference result dataclasses
  parity.py       # stage-by-stage reference adapters, used ONLY by tests + triage
```

#### 5.1.2 `PipelineSnapshot` (J0)

Unifies the two existing snapshot types (`core/jax_runtime.py:430` `AtomicSnapshot` for the forward kernel; `iterative.py:381` `_AtomicSnapshot` for the solver) into one frozen, pytree-registered bundle built **once per process** from a single SQLite scan, cacheable as `.npz` keyed by a DB content hash. Measured against `ASD_da/libs_production.db` (6.1 MB):

- `lines`: 28,727 rows, 83 elements, 158 (element, stage) species; ~13 f64-equivalent columns incl. Stark metadata (`stark_w` 28,331 rows; source classes konjevic-λ²-scaled 22,951 / interpolated 4,574 / hydrogenic 562 / `stark_b` 244 / null 396), `aki_uncertainty`, resonance flags ≈ **3.0 MB**
- `energy_levels`: 9,448 rows over 144 species, max 676 levels/species (Fe II) → uniform pad (144, 676) g+E+mask ≈ **1.65 MB**
- partition polys 146×5 ≈ 6 KB; canonical scalar fallbacks 175×2 ≈ 3 KB — **eager** at build (the lazy probe at `iterative.py:479-490` becomes one-time cost)
- `species_physics` (ip_ev, atomic_mass) 175×2 ≈ 3 KB; doublet-pair precompute (P,2)+ρ+r_thin < 0.5 MB; oxide factors (`closure.py:62-73`) → per-candidate (E,) vectors

**Total ≈ 5–6 MB ≈ 0.04 % of one 16 GB V100S** — the whole DB rides on device; per-spectrum work is pure gather. Candidate sets are **element masks over one per-bucket superset snapshot**, never per-spectrum snapshot rebuilds (this is what makes vmap over heterogeneous candidate sets possible without recompiles). The only host↔device seam per spectrum mirrors the proven `_build_padded_arrays_from_obs` (`iterative.py:2986-3020`) + `snapshot.reorder` (`:519-540`) pattern.

#### 5.1.3 What never moves to the jit pipeline (permanent host surface)

CLI/argparse + YAML config (`cflibs/cli/main.py`, `core/config.py`, `AnalysisPipelineConfig` resolution `pipeline.py:90,163`); `datagen_v2.py` + `cflibs/atomic/database.py` + pooling; `cflibs/io/` + adapters + `cflibs/benchmark/`; response-curve file parsing/interpolation (`preprocess/response_correction.py` — only the per-channel multiplier array crosses); diagnostics/trust/logging/warning strings (`pipeline.py:735-748,826-840`); CCD-seam *caching decisions* per instrument; dynesty drivers (`solve/bayesian/samplers.py` — NumPyro NUTS *consumes* the jit likelihood, the sampler loop stays host, the ExoJAX pattern); Rust native paths; `inversion/runtime/`, `pds/`, `hpc/`; the analytical `uncertainties` UQ path (`physics/uncertainty.py`) — by contrast Monte-Carlo UQ becomes a free vmap on the jit stack; `cflibs/evolution/` (calls the jit pipeline as an evaluator, never part of it). Boundary invariants enforced from J0: kernels import neither `sqlite3` nor `cflibs.atomic.database` nor `cflibs.io` nor `jitpipe.host`; every kernel output is a fixed-shape pytree; only registered pytrees cross host→kernel.

### 5.2 Shape & batching policy

**Measured grid lengths (live adapter run, 2026-06-10)** — constant within each dataset/instrument (checked first 5 spectra per adapter):

| Dataset | Grid N_wl | Range (nm) | Spectra |
|---|---|---|---|
| synthetic_fixedforward | 2,560 | 223.6–264.3 | 288 |
| bhvo2_chemcam / chemcam_calib | 6,144 | 240.8–905.6 | 4 / 240 |
| aalto | 8,188 | 195.5–982.6 | 74 |
| gibbons2024 | 12,274 | 185.7–1049.0 | 175 |
| csa_planetary | 13,490 | 198.1–970.1 | 99 |
| emslibs2019 | 40,002 | 200.0–1000.0 | 282 |
| silva2022 | 53,717 | 200.0–780.0 | 102 |
| nist_srm_612 / nist_steel | skip-with-log (no data) | — | 0 |

**Policy:** one compile bucket per distinct instrument grid at the **exact** N_wl (7 buckets cover all 10 datasets; zero wavelength padding — §3 C3). Per-bucket, the wavelength axis, `wl_step`, window widths, and `half_width_px` (`line_detection.py:1323`) are host-computed static ints. Padding constants for the other axes (measured at production gates — `threshold_factor=4.0`, calibration pool 60/element, detection `top_k_per_element=60` per `pipeline.py:105`, candidates = truth ∪ {Ag,Sn,W,Bi,Th} per `scoreboard.py:142`):

| Axis | Constant | Observed max | Where observed |
|---|---|---|---|
| Peaks, calibration path | P_max = 2,048 | 1,412 | silva2022 |
| Peaks, detection path | P_max = 2,560 | 2,350 | emslibs2019 |
| Peak NMS candidates | P_cand_max = 8,192 | — | pre-NMS local maxima |
| Calibration line pool | L_max = 1,024 | 925 | csa/bhvo2 |
| Per-peak line fan-out (±2.0 nm) | K_pair = 48 | 32 | synthetic/bhvo2 |
| Per-peak in-tolerance transitions | K_match = 8 | 5 | silva/synthetic |
| Elements | E_max = 32 | 15–21 (§3 C6) | csa_planetary |
| Species | S_max = 48 | 30 | 15-element set |
| Lines/element (detection) | K_lines = 64 | 60 (cap) | config |
| Comb lines/element | K_comb = 32 | 30 (cap) | config |
| Shift grid | S_shift = 32 | 21 | `_build_shift_grid` |
| CCD segments | SEG_max = 16 | 11 | csa_planetary |
| Observations | OBS_max = 512 | — | output batch |
| RANSAC hypotheses | H = 4,096 (chunk 256) | — | affine/quadratic |

Overflow on any axis keeps the top-prominence/score entries and sets a truncation flag (mandatory output). **Batch axis:** per-bucket fixed B from the memory envelope; final partial batch padded with a spectrum-validity mask. `run_batch(bucket) = jit(vmap(stage_chain, in_axes=(0, None, None)))` — the manifold pattern (`batch_forward.py:13,459`) generalized.

**16 GB memory envelope (fp64, ~40 % XLA headroom; design to 16 GB even though cluster notes record 32 GB — safe on either):**

| Path | Working set / spectrum | B on 16 GB |
|---|---|---|
| Dense Voigt (N_lines×N_wl), silva2022 | 16,384×53,717×8 = 7.0 GB | forbidden on large grids |
| Chunked scan (`kernels.py:1205`), N_chunk=4,096 | 537 MB live/chunk | **8–16** |
| Chunked, fp32 profile island (§5.3) | 268 MB | ~32 |
| LDM (`ldm.py`, n_sigma=24 at `:69`), silva2022 | ≈60 MB | **64–128** |
| LDM, chemcam 6,144 grid | ≈8 MB | 512+ |
| Identify (Gram-form NNLS ≤100 components) | <10 MB | whole-registry |
| Solve ((E,)-shaped state) | KBs | all ~1,264 spectra at once |
| Sliding-median window matrix (worst: 53,717×1,001) | 430 MB → row-chunk to 33 MB live | non-binding |
| RANSAC residual matrix (H×C) | chunk H→256 → ≤100 MB live | non-binding |

Conclusions: identify + solve batch the whole board in one shot; the broadening stage dictates B; **LDM is the batching enabler and its real-data validation is an early-milestone dependency (risk R4)**; chunked Voigt at B=8–16 is the fallback and still beats CPU by orders of magnitude. `donate_argnums` on the B×N_wl intensity buffers (none exists in `cflibs/` today — free win); keep the existing `jax.checkpoint` policy (`kernels.py:1055-1076`) and remat the broadening stage for reverse-mode end-to-end gradients.

### 5.3 Precision policy

**Default x64 end-to-end.** The codebase already votes x64: tests force CPU+x64 (`tests/conftest.py:20,25`); `JaxMemoryPolicy` defaults `allow_32bit=False` (`core/jax_runtime.py:300-328`); `hpc/gpu_config.py:43-56,75-76,152-155` declares x64 required for cluster GPU; the Weideman Voigt backend delivers ~15 digits only in f64 (`radiation/profiles.py:416-442`).

**Must be fp64 (non-negotiable):** Saha three-stage populations + IPD (`kernels.py:425,292` — exponents reach ~50; the repo's log-basis partition bug produced an 18-orders error, `kernels.py:124-127`, bead ddwh); partition functions (poly `kernels.py:119` + direct sums over ≤676 IPD-truncated levels, `jax_runtime.py:483-486,543-550`); Boltzmann/common-slope WLS + solve (rtol 1e-5 unreachable in fp32); Stark-width n_e; LDM log-σ interpolation weights (`ldm.py:180-195`); the preprocessing LLS round-trip and ALS λ=1e6 Gram (do not run row-1/2 kernels fp32).

**fp32-tolerable (opt-in, policy-gated):** profile/broadening matrix *values* (hooks exist: `radiation/host.py:210-211` dtype_bytes=4; `ldm.py:166` defers to policy) — compute populations fp64, cast at the broadening input boundary, accumulate bin sums in fp64; identify-stage template scoring (decisions thresholded at ~1e-2). Mechanism: extend `JaxMemoryPolicy` with per-stage dtype fields; an fp32-profile configuration enters scored runs only after a Tier-B shadow shows |ΔF1| ≤ 0.005. Never fp32 in the Stark-width Voigt path (`profiles.py:425-434` is the authority).

### 5.4 Parity & testing harness

**Tiered tolerance contracts (bit-identity is a non-goal):**

| Tier | Contract | Precedent |
|---|---|---|
| **K** (kernel/stage-pure) | rtol 1e-5 (tighter where §4 says so, down to 1e-12 for pure-algebra stages and *exact* for discrete decisions) | `test_iterative_lax.py:5` (all six closure modes); `kernels.py:856` |
| **S** (stage-with-iteration) | rtol 1e-3, atol 1e-4; iteration counts within 1 | `test_solver_jax_parity.py:4-12`; `test_iterative_lax.py:8` |
| **D** (decision) | identified-element sets + presence calls exact-match on ≥98 % of a frozen golden corpus; every divergence adjudicated in a ledger | new |
| **B** (board) | shadow scoreboard: aggregate |ΔF1| ≤ 0.005, |ΔRMSE_med| ≤ 2 % per dataset | new |

**Per-stage golden tests** `tests/jitpipe/test_parity_{preprocess,detect,calibrate,identify,fit,selfabs,stark,solve,forward,pipeline}.py`: committed `.npz` fixtures + sha256 (one real spectrum per dataset via the adapters, plus synthetic edge cases — empty candidate sets, single-line elements, saturated lines, overlapping doublets); `parity.py` adapters feed *identical padded inputs* to reference and jit; replicate all four acceptance patterns from `test_iterative_lax.py` per stage — numeric parity, **vmap smoke** (batch 16), **grad smoke→hard assert**, **no-SQLite-inside-kernel** (`:415-429`); plus **padding invariance** (re-run at next pad size must be bit-identical on the valid region — the test that catches mask bugs, the dominant failure mode of fixed-shape rewrites). CI on CPU x64 in subsets under the 600 s agent watchdog (CLAUDE.md sub-agent rules). Wavelength calibration additionally requires the **reference self-variance study** (≈10 seeds on the 32-spectrum corpus) before its bands are frozen — the reference is itself seed-dependent (`wavelength_calibration.py:615`, `random_seed=42`).

**Scoreboard integration:** `--pipeline {reference,jit}` on the scoreboard subparser (`cli/main.py:1255-1290`) threaded through `run_scoreboard` (`scoreboard.py:274`) → `_score_spectrum` (`:132`). Everything score-defining is shared and untouched: candidate policy (`:142`), presence rule, seeded sampling (`:193-198`), micro-averaged confusion (`:214-217`), failure policy (`:159-166`). The jit path emits the identical per-spectrum record schema incl. `stage_timings_s` (batched stages report batch_time/B with B recorded); `"pipeline": "jit"` lands in the board JSON and reproduce line (`:377`). **Failure-policy parity is mandatory:** zero valid observations must produce the same all-FN record as the reference's `ValueError` (`pipeline.py:751-752`) — not NaN, not a crash.

**Divergence triage:** (1) reproduce on CPU x64 both pipelines; (2) stage-bisect with `parity.py`; (3) `jax.debug.print` dumps behind `CFLIBS_JITPIPE_DEBUG`; (4) classify — padding-sensitivity (rerun at next pad size) / accumulation-order-precision / genuine semantic difference → bead + ledger entry. Institutional rule (project memory, PR #229 F1 −0.041): every identifier-scoring change is benchmark-gated.

### 5.5 Cluster & runtime discipline

SLURM job shapes per `.serena/memories/cluster_workflow.md:36-48`: `--gpus-per-task=1 --ntasks=1 --cpus-per-task=4 --mem=32G`, never `--nodelist`. Job archetypes: (1) batched inversion — one job, one bucket-compile + ⌈N/B⌉ batches; (2) campaign fitness — **3 long-lived workers** pulling candidate configs from a queue dir (in-process jit cache ⇒ each worker compiles each bucket once, then **zero recompiles across candidates because all continuous knobs are traced `PipelineParams` leaves** — the design's killer feature for campaigns); (3) HMC endgame — vectorized NUTS per dataset bucket.

**Compile-cache discipline (documented pathology):** the NFS-shared `/cluster/shared/jax-cache` developed uid skew and hung jobs 1909/1914/1915; the standing rule is the user-private `~brian/jax-cache` (`.serena/memories/physics_invariants_and_gotchas.md:54-56`). **Trap:** `cflibs/core/platform_config.py:94` still `setdefault`s the shared path — every jitpipe job script MUST export the private path explicitly, and J0 files the change flipping the code default to `~/.cache/cflibs/jax` + updating `docs/jax-compile-cache.md`. Keep `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5` (`platform_config.py:95`). Budget: #compiles = 7 buckets × few `StaticConfig` variants at 15–45 s each (`docs/jax-compile-cache.md:9-10`); warmup batch per bucket in every job script; `==`-pin jax/jaxlib in the `[cluster]` extra (currently `jax[cuda12]>=0.4.30`, `pyproject.toml:41`) so cache keys survive node-image drift. Scored runs set `XLA_FLAGS=--xla_gpu_deterministic_ops=true`; CPU↔GPU agreement is tolerance-bounded (f64 reduction order, ≲1e-12 relative), never bit-identical. CPU fallback: `jitpipe` requires JAX (clear ImportError) but runs fully under `JAX_PLATFORMS=cpu` with identical semantics (the parity suite does, via conftest); the reference pipeline keeps the repo-wide JAX-optional guarantee untouched (D3 one-way rule).

---

## 6. Differentiability payoffs

### 6.1 The re-derived production solve (joint WLS) — and its exact parity anchor

`ClosedFormILRSolver` already poses the right single regression — `y_l + ln U_s + ln M_s = m·E_l + Σ_j V[s,j]·α_j + β` with honest normal-equation covariance (`solve/closed_form.py:1-19,220-302,305-329`) — but must freeze U_s(T), M_s(T, n_e) and bolt on 1–2 Saha passes because NumPy cannot differentiate through them. Autodiff removes exactly that limitation. Production jit solve:

minimize over θ = (ln T, α ∈ R^{E−1}, β), **n_e pinned to the Stark measurement** (audit F2) with a penalty from its MAD scatter when D>1, pressure balance only as a flagged fallback:

Σ_l w_l · [ y_l − ( −E_l/(k_B T) + ln C_{s(l)}(α) − ln U_{s(l)}(T) − ln M_{s(l)}(T,n_e) + β + saha_shift_l(T,n_e) ) ]²

by fixed-K Gauss–Newton/LM on ≤(E+2) parameters (trivial ~20×20 normal systems), vmapped over B. Properties: (i) **freeze U,M + unit weights ⇒ the first GN step IS the SB-graph closed form** (the arrow-matrix Schur identity of §4 row 6) — exact parity anchor and warm start; (ii) the Hessian at the optimum yields the covariance audit-F9 says is currently underestimated, with U(T)/Saha sensitivity free via autodiff; (iii) closure modes enter as the map α→C (standard/ILR identical; oxide = factor-weighted normalization inside ln C; matrix = one pinned coordinate); (iv) the same residual, reverse-differentiated, is directly the potential for NUTS over (T, n_e, α) — and (v) curve-of-growth self-absorption can enter as a per-line escape-factor term f(τ_l(θ)) — audit F10's "fold COG into the regression", feasible *only* in the differentiable formulation. Parameterization matches the shipped `joint_optimizer.py:16-23` (log-T, log10-n_e, softmax/ILR simplex). Requirement: fixed-iteration optimization, no data-dependent `while_loop` in the differentiated region; the legacy implicit-diff route (`lax.custom_root` per T4-1, deferred at `specs/T1-3-lax-while-iterative.md:155,181,190`) survives as a J11 spike for the initializer loop. Tooling constraint: jaxopt is unmaintained and its successors (optimistix/lineax) are equinox-dependent — **banned**; use core-JAX `custom_root`/hand-written `custom_vjp`; host scipy L-BFGS-B driving JAX gradients is fine for research phases.

### 6.2 HMC / NumPyro over the full model

NumPyro NUTS already runs over a JAX forward model in `solve/bayesian` (numpyro ≥0.14, `pyproject.toml:48`); the NNLS top-K candidate prefilter stays mandatory. The rewrite adds the batched LDM forward (≈8–60 MB/spectrum working set, §5.2) making vectorized chains across many spectra feasible on one V100S. Requirements: likelihood with no data-dependent control flow (HMC replaces the iterative solve, it does not wrap it); Chebyshev baseline (already in `BayesianForwardModel`); fp64 (NUTS on fp32 plasma exponentials produces divergences).

### 6.3 Gradient knob-tuning vs the Optuna campaign

Campaign 1 treats the pipeline as a black box at 0.4–5.1 s/spectrum. The jit pipeline changes the economics for the ~20 **continuous** knobs (`min_peak_height`, `min_snr`, prominence, isolation windows, wavelength tolerance, ALIAS weights, presence cutoff — the knob surface of `AnalysisPipelineConfig`, `pipeline.py:90-152`): one backward pass over a batch yields d(soft-F1)/d(knobs) for ≈2–3× a forward board, where TPE pays one full board per point with no gradient. Recommended hybrid: TPE over categorical/integer knobs with a gradient inner loop on continuous knobs per trial — zero recompiles since knobs are traced leaves (§5.5).

### 6.4 Relaxation map — train-soft / eval-hard

**Soft, ONLY in tuning objectives:** presence rule `C ≥ 0.005` → sigmoid((C−0.005)/τ) for soft confusion counts; top-K/prominence/SNR gates → soft-top-K or straight-through; quality-gate scalars → sigmoid penalties. Sigmoids written as explicit `jnp` (no `jax.nn`) or via `softmax_closure`. **Hard, always:** structural validity/padding masks (bookkeeping, not decisions — relaxing them corrupts physics); physical constraints by reparameterization not relaxation (log-T, log-n_e, softmax/ILR simplex); scoreboard scoring (hard presence rule, candidate policy, failure policy — the board is the shared contract and never sees a relaxed metric); convergence flags reported hard with gradients masked.

---

## 7. Risk register

| # | Risk | Sev | Mitigations |
|---|---|---|---|
| R1 | Two-pipeline semantic drift over months | High | D3 one-way imports + hygiene test; jitpipe reuses `radiation/kernels.py` + preset tables (no copied physics); reference frozen except bug fixes, each paired with a parity-test update; divergence ledger; promotion criteria force a decision rather than letting both linger |
| R2 | Silent numerical divergence corrupting scored results | High | tiered contracts with Tier-D in CI on a frozen golden corpus; Tier-B shadow runs required before any campaign adopts jit fitness; benchmark-gate rule (PR #229 precedent, project memory) |
| R3 | Fixed-shape memory blowup on 16 GB (padding × batch × fp64) | Med | exact wavelength shapes (zero pad waste, §3 C3); measured per-axis constants + truncation flags; per-bucket B from §5.2 envelope; reuse `plan_chunks`/`available_device_bytes` (`radiation/host.py:164-246`); OOM-canary test asserting planned bytes < 0.6× device bytes |
| R4 | LDM validation gap — the batching enabler is default-OFF pending real-data validation (`ADR-0001-HANDOFF.md` T1-4) | Med | LDM-vs-PHYSICAL_DOPPLER parity + scoreboard shadow on real datasets scheduled as an **early milestone inside J9**; fallback = chunked Voigt at B=8–16 (still orders-of-magnitude over CPU) |
| R5 | Debugging opacity of jitted, batched, masked code | Med | stage-boundary NamedTuples behind `CFLIBS_JITPIPE_DEBUG`; CPU-x64 repro as standard first step; `parity.py` doubles as inspection tooling; padding-invariance tests catch the dominant mask-bug class mechanically |
| R6 | Compile-time tax (15–45 s/kernel cold) | Med | continuous knobs traced ⇒ zero recompiles across campaign candidates; ≤7 buckets × few static variants; private persistent cache + 0.5 s floor + warmup step + jax `==`-pin (§5.5) |
| R7 | Opportunity cost vs the recall-focused campaigns (program risk) | High | sequencing pays rent immediately: J10 (forward-fitting ID) and the batched forward/identify kernels are *on* the recall critical path; campaigns keep reference fitness until Tier-B shadow passes; §8.4 start criteria gate the front-end grind explicitly |
| R8 | Behavioral drift in greedy/tie-break front-end logic (the highest-concentration hazard: `np.isclose` ties, gated-without-consuming-peak, retroactive element drop, first-pair-wins) | High (localized) | D5 exactness-first: port as exact scans, no parallel approximations in decision-bearing stages; named canary fixtures (ye6t Al-doublet, BHVO-2 Sn/Th, shared-upper-level triplet) mandatory in CI |
| R9 | Stark LM vs scipy-trf divergence on marginal profiles; window-extraction parity | Med-High | gather raw samples (never resample); same rel-RMSE/resolvability gates decide acceptance rather than chasing optimizer parity; golden windows from `docs/benchmarks/stark-vjbh-empirical-impact.md` datasets |
| R10 | GPU nondeterminism undermining the reproducibility motivation | Low-Med | `--xla_gpu_deterministic_ops=true` on scored runs; all stochastic stages keyed by counter-based `jax.random` from a per-spectrum hash; CPU↔GPU drift documented as tolerance-bounded |
| R11 | JAX/CUDA version churn on the cluster | Low | `==`-pin in `[cluster]` extra; persistent cache is version-keyed so upgrades degrade to a one-time recompile, never wrong results |
| R12 | Skipped/placeholder datasets distorting promotion (nist_srm_612/nist_steel yield no spectra; silva2022 currently fails 12/12 in the baseline) | Low | promotion computed over scoring datasets only; failure-count criterion tracked separately; silva2022's all-fail status is a reference bug surface the jit failure-policy parity must mirror, not hide |

---

## 8. Staging plan, milestones, economics, and start criteria

### 8.1 Child beads J0–J12

One epic bead with children; one spec per bead under [`specs/`](./specs/). Effort in person-days (pd) of stage work; agent-session leverage per §3.

| Bead | Scope (spec) | Depends on | Track | Effort | Milestone |
|---|---|---|---|---|---|
| **J0** | skeleton, `PipelineSnapshot` unification + npz cache, `PipelineParams`/`StaticConfig`, contracts doc, import hygiene, compile-cache default fix, ADR-0001 §12 addendum ([J0 spec](./specs/J0-skeleton-snapshot-contracts.md)) | — | spine | 3–5 pd | gates everything |
| **J1** | baseline + noise + **exact** peak detection ([J1](./specs/J1-preprocess-baseline-noise-peaks.md)) | J0 | A | 9–12 pd | |
| **J2** | segmented-RANSAC wavelength calibration ([J2](./specs/J2-wavelength-calibration.md)) | J0, J1 | A | 12–17 pd | |
| **J3** | line matching + gates (`detect_line_observations`) ([J3](./specs/J3-line-matching-gates.md)) | J0, J1 (soft J2 — can develop on reference calibration) | A | 12–17 pd | |
| **J4** | fit kernels: selection, Boltzmann/SB-graph, closure ([J4](./specs/J4-fit-kernels-selection-boltzmann-closure.md)) | J0 | B | 5–9 pd | |
| **J5** | observable self-absorption kernel ([J5](./specs/J5-self-absorption-kernel.md)) | J0, J4 | B | 3–5 pd | |
| **J6** | Stark n_e vmapped LM ([J6](./specs/J6-stark-ne-vmapped-lm.md)) | J0 (soft J1) | B | 5–10 pd | |
| **J7** | solve: fixed-K scan initializer + joint WLS estimator ([J7](./specs/J7-solve-scan-joint-wls.md)) | J0, J4, J6 | B | 10–15 pd | |
| **J8** | end-to-end single-spectrum graph + `--pipeline=jit` ([J8](./specs/J8-end-to-end-m1.md)) | J1–J7 | spine | 5–8 pd | **M1** |
| **J9** | batched execution on V100S, LDM validation, SLURM/cache ([J9](./specs/J9-batched-cluster-m2.md)) | J8 | spine | 4–6 pd | **M2** |
| **J10** | forward-fitting identification (Campaign-3 GPU evaluator) ([J10](./specs/J10-forward-fitting-identification.md)) | J0 + existing kernels only (integrates with J8 later) | C | 10–15 pd | |
| **J11** | differentiability: gradient knob-tuning, HMC, implicit-diff spike ([J11](./specs/J11-differentiability-payoffs.md)) | J0; J7 for full-model work | C | 7–10 pd | |
| **J12** | full-board superiority run + promotion decision ([J12](./specs/J12-scoreboard-promotion-m3.md)) | J9, J10 (J11 optional input) | spine | 3–5 pd + buffer | **M3** |

**Critical path:** J0 → J1 → J2/J3 → J8 (M1) → J9 (M2) → J12 (M3), with J4–J7 in parallel on track B. **Tracks:** A = front-end (J1–J3); B = back-end (J4–J7); C = research (J10, J11) — C deliberately depends only on J0 + already-merged kernels so the two highest-value research items are not blocked behind the front-end grind, and J10 runs on GPUs while campaigns hold CPUs (`optimization-program-design.md:476`).

### 8.2 Milestone gates

- **M1 — single-spectrum end-to-end parity (J8).** On bhvo2_chemcam (4 spectra, headline, response-corrected) + chemcam_calib (240 spectra, R≈2000, real spectral gaps exercising masks): element calls agree ≥95 %; T within 2 %; n_e within 10 % (Stark-stage tolerance dominates); concentrations rtol 5 % / atol 0.01; 2-dataset board F1 delta ≥ −0.02; all per-stage Tier-K/S contracts green; failure-policy parity demonstrated.
- **M2 — throughput on one V100S (J9).** Target ≥50 spectra/s end-to-end amortized over ≥1,000 spectra including compile; full corpus (~1,264 spectra) < 60 s/GPU; per-bucket memory headroom documented; LDM real-data validation decision recorded (adopt or fall back to chunked Voigt). Hard floor for promotion is the §8.2 M3 runtime criterion (§3 C5).
- **M3 — promotion decision (J12).** On the holdout tier of the dataset split (`optimization-program-design.md` §2.1), full board (all spectra, not `--max-spectra 12`), after ≥1 release of shadow-mode operation: (1) aggregate micro-F1 ≥ reference AND no scoring dataset regresses F1 by >0.02 AND per-dataset F1 ≥ reference −0.01 on ≥7/10; (2) median composition RMSE ≤ reference on every quantitative dataset (one dataset may regress ≤5 % relative if the aggregate improves); (3) hard-failure count ≤ reference (the 12 silva2022 failures are the floor to beat, `SCOREBOARD…md:171-188`); (4) batched runtime ≥10× median spectra/s, CPU single-spectrum ≤2× reference; (5) parity suite green at §5.4 contracts. Pass → promotion ADR + Stage-B deprecation begins; fail → gap beads, stay parallel.

### 8.3 Economics

**Total effort:** ≈85–130 pd of stage+glue work ≈ 40–60 agent-sessions; calendar **6–11 weeks** at 2–3 parallel tracks (critical path alone ~5–7 weeks). Uncertainty ±50 % — ADR-0001 §9.2 warns its own estimates ran optimistic; J3 (gate-stack semantics) and J10 (research) carry most variance. Compute cost is negligible against the idle V100S allocation (mean ~3 % utilization, §1.1).

**Payoff thresholds that justify the spend:**
1. *Throughput:* at M2, a full-population campaign fitness evaluation costs what a 12-spectrum sample costs today, removing fitness sampling variance. Break-even ≈ 10⁵ spectrum-evaluations of future campaign demand; Campaign 1 alone (≈800 trials × ~120 spectra) is ~10⁵ — **one campaign phase amortizes the build** if it is fitness-noise-bound.
2. *Forward-eval scale:* 10⁴ evals/spectrum ≈ 10–100 ms on V100S vs ~hours on the CPU reference; synthetic-corpus regeneration (×1000 scale) rides the same kernels.
3. *Identification F1:* J10 must show micro-F1 ≥ +0.03 on the optimization split with precision loss ≤0.02 to count (recall is the binding axis). If knob campaigns alone reach the same F1, the accuracy payoff is void — hence §8.4.
4. *Differentiability:* capabilities (J11 acceptance: FD-verified gradients; joint ≤ iterative round-trip error on the golden set) unlocking gradient knob-tuning as the cheaper successor to black-box search.
5. *Determinism:* falls out of the design; no separate threshold.

### 8.4 Explicit start-decision criteria (all four should hold before J1–J9 start)

1. **Campaigns plateaued:** Campaign 1 shows < +0.005 micro-F1 on the optimization split over two consecutive generations/phases — knob-space exhausted, remaining gains structural (the audit's prediction, `00-synthesis.md:23`).
2. **Forward-fitting chosen:** the identification-endgame decision has explicitly selected forward-fitting/multi-line coherence as the next mechanism. If a cheaper coherence heuristic on the reference closes the recall gap first, J10's case weakens and only the J9/J11 motivations remain.
3. **Throughput-bound evidence:** campaign ledger shows fitness-sampling noise comparable to per-generation effect sizes, or CPU queue saturation on vasp-01/02/03.
4. **Stable foundation (already true):** lax parity suite green; scoreboard baseline frozen (2026-06-10); Phase A merged at a pinned SHA.

**Start small regardless:** J0 + the J11 `custom_root` spike (~4 sessions total) are cheap, zero-contention, and de-risk the two biggest unknowns (snapshot unification; implicit-diff viability). **Do not start J1–J9 while criteria 1–3 are unmet** — every week of plateau-free campaign progress on the reference raises the bar M3 must clear.

---

## 9. Prior art

| Source (read 2026-06-10) | What it does | What we adopt | What it confirms |
|---|---|---|---|
| **RADIS GPU** (radis.readthedocs.io /lbl/gpu.html) | Vulkan compute, DB uploaded once at init, `recalc_gpu()` ships only (T, p, x) changes, <200 ms interactive fitting on >100M lines; equilibrium-only, no gradients, no batching API | resident-database / parameter-only-update economics — already CF-LIBS's snapshot discipline (`iterative.py:381-436`, `jax_runtime.py:430`) | batched + differentiable GPU **inversion** is genuinely open territory |
| **ExoJAX2** (Kawahara et al. 2025, ApJ 985:263 — peer-reviewed upgrade of arXiv 2410.06900) | reverse-mode-AD opacity, gradient optimizers + HMC retrievals on real JWST data | nothing new to port — PreMODIT chunked scan/checkpoint/OLA already at `kernels.py:1205,1411,927`; we take the *workflow* (pure-JAX forward outside the NumPyro graph) | citation upgrade for ADR-0001 |
| **jaxrts v0.7.0** (github.com/JaXRTS/jaxrts, 2026-04-16) | forward-only `PlasmaState.probe()`; no fitting/inversion/batching documented | nothing further (its while_loop fixed-point pattern is already `iterative.py:721-863`) | CF-LIBS's `_solve_lax` + vmap/grad tests is *ahead* of both JAX codes on inversion |
| **DSAC / ∇-RANSAC** (arXiv 1611.05705; 2212.13185, ICCV 2023) | differentiable RANSAC via probabilistic hypothesis selection — wraps learned components (banned) | hypothesis-marginalization *structure* only | |
| **MAGSAC / MAGSAC++** (arXiv 1803.07469; 1912.05909, CVPR 2020) | sigma-marginalized soft inlier likelihood + IRLS, **NN-free**, no hard threshold | the optional differentiable calibration variant (§3 C2) and soft-consensus Boltzmann outlier weights | physics-legal differentiable robust fitting exists |
| **scipy find_peaks × JAX** (searched; no off-the-shelf precedent) | experimental array-API support; `distance`+`prominence` produce variable-length non-local output | the repo's prior rejection (`line-detection-consultation.md`) was B=1-premised (§1.3); we port exact semantics in fixed shapes (§3 C1) instead of the vision-NMS approximation | |
| **MC-CF-LIBS GPU forward fitting** (Gornushkin & Völker 2022, PMC9573556; Demidov et al. 2016) | 5×10⁵ configs/iteration × 50 iterations = 2.5×10⁷ forward evals/spectrum, weighted-correlation cost, ~1 % relative error on 8 elements, ~5 min on a 2013 K40 (MATLAB) | the population-forward-fit shape, element-weighted full-spectrum cost, and ~1 %-on-majors accuracy target for J10; V100S (~6× K40 fp64) + XLA + gradient refinement (which MC-CF lacks) should beat 5 min by 1–2 orders | the strongest direct precedent for motivation 2 |
| **jaxopt status** (github.com/google/jaxopt) | unmaintained; pieces moved to optax; maintained successors optimistix/lineax are **equinox-dependent (banned)** | `jax.lax.custom_root` / hand-written `custom_vjp` only; scipy L-BFGS-B host-side is fine for research | tooling correction to ADR-0001 T4-1 |

**ADR-0001 addendum:** the dated §12 addendum text (survey refresh: T1 completion status, ExoJAX2 publication, jaxrts v0.7.0, MAGSAC++/DSAC applicability, jaxopt correction, the 3 %-GPU empirical driver, fixed-shapes-end-to-end conclusion) is applied to `ADR-0001-radis-jaxrts-pattern-survey.md` as part of J0 (it was delivered as text by Analysis 4; the analysis task was read-only).

---

## 10. Consequences

**Positive:** the campaign evaluator stops being the bottleneck (zero-recompile candidate evaluation); the Campaign-3 identification endgame gets a full-physics GPU evaluator; gradient/HMC inversion becomes possible for the first time in any published LIBS code (§9: neither jaxrts nor ExoJAX does inversion; nobody does `custom_root` through a CF-LIBS fixed point — paper-worthy per ADR-0001 §5.4); scoreboard runs become deterministic per (seed, SHA, backend).

**Negative / accepted costs:** two pipelines coexist for the program's duration (mitigated by D3/D6/R1); ~85–130 pd of effort with ±50 % uncertainty; a permanent parity-suite maintenance obligation; fp64-on-GPU compute cost (accepted, §5.3); the reference pipeline is frozen except bug fixes for the duration, which slows non-campaign reference work.

**Failure mode is bounded:** if M3 fails, the jit pipeline stays a parallel evaluator (J10's campaign value and J9's throughput stand on their own), gap beads are filed, and the reference remains the default — the program never holds the scoreboard hostage. The riskiest 30 % (promotion, deprecation) is deliberately last and gated on a scoreboard the program does not get to redefine.

---

## 11. Amendments

### 2026-06-16 — D4 carve-out for the J10 forward-fit host adapter; D1 scoreboard-dispatch allowlist

Two import-hygiene clarifications applied during the post-ALIAS cleanup (`tests/jitpipe/test_import_hygiene.py`), both confirming the original intent rather than relaxing it:

1. **`forward_id_identifier.py` joins the no-SQLite-in-kernels carve-out (D4).** The J10 forward-fitting identifier landed a *host adapter* module (`cflibs/jitpipe/forward_id_identifier.py`) whose own docstring calls it the "thin **host-only** bridge". It builds the `PipelineSnapshot` once from SQLite **host-side** (`from cflibs.atomic.database import AtomicDatabase`, never inside a traced kernel) and maps the jit `ForwardFitResult` onto the duck-typed `IdentifierProtocol` the scoreboard consumes. Per D4 it is exactly analogous to `host.py`: the host/kernel boundary keeps SQLite out of the *traced* region. The jit core it wraps (`forward_id.py`) is correctly **not** carved out and stays DB-free. Carve-out set is now `{host.py, snapshot.py, parity.py, pipeline.py, forward_id_identifier.py}`.

2. **Scoreboard/benchmark dispatch may lazily import jitpipe (D1).** D1 states the two pipelines "meet at exactly two points: the scoreboard CLI dispatch (`--pipeline {reference,jit}`, §5.4) and the parity tests." The benchmark layer implements that dispatch (`scoreboard.py` `_run_pipeline_jit`/`_jit_snapshot`; `synthetic_eval.py`/`unified.py` forward-fit runners). All jitpipe imports there are **lazy** (inside the dispatch function, behind the `jit`/forward-fit branch), so importing the benchmark package never imports jitpipe — the one-way dependency D1 protects (jitpipe never depending on the reference) is intact. The hygiene test's "nothing outside jitpipe imports jitpipe" rule now allowlists `cflibs/benchmark/{scoreboard,synthetic_eval,unified}.py` as the sanctioned §5.4 meeting point.

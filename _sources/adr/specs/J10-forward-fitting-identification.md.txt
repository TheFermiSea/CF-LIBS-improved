# J10 Implementation Spec — Forward-Fitting Identification on the Jit Stack (Campaign-3 GPU Evaluator)

**Bead:** J10 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §1.2 motivation 2, §8.1 track C, §9 (MC-CF-LIBS precedent) · **Track:** C (research; deliberately depends only on J0 + already-merged kernels so it is NOT blocked behind the front-end grind) · **Depends:** J0 (+ `radiation/kernels.py`; integrates with J8 later) · **Estimated effort:** 10–15 pd / 2–3 wk (research-flavored)

## 1. Goals

- Implement Gornushkin-style **population forward fitting** for element identification: vmap 10³–10⁴ candidate configurations (T, n_e, path length, candidate-element concentration vectors) through `forward_model_chunked` (`cflibs/radiation/kernels.py:1205`) per spectrum; element-weighted full-spectrum correlation/coherence cost; presence decided by model comparison (BIC over element subsets — extends `cflibs/inversion/identify/model_selection.py`); optional gradient polish of (T, n_e, C) per surviving candidate (which the MC-CF prior art lacks entirely).
- This implements the audit's missing "multi-line intensity-coherence test" (`docs/audit/2026-06-09-overhaul/03-identification.md:19,102`) at full-physics fidelity, attacking the binding scoreboard constraint: recall 0.27–0.58 vs precision 0.89–1.00 (`SCOREBOARD-2026-06-10-baseline.md:16-27`).
- Prior-art anchor (ADR-0004 §9): Gornushkin & Völker 2022 (PMC9573556; building on Demidov et al. 2016) ran 2.5×10⁷ forward evals/spectrum at ~1 % relative error on 8 elements in ~5 min on a 2013 Tesla K40 in MATLAB. A V100S (~6× K40 fp64) + XLA fusion + gradient refinement should beat 5 min/spectrum by 1–2 orders of magnitude; budget: 10⁴ evals/spectrum ≈ 10–100 ms at 10⁵–10⁶ fused evals/s.

## 2. Design constraints

- **Inputs phase 1:** reference-pipeline peaks/observations via `parity.py` adapters (no J1–J3 dependency); **phase 2:** J8's ObservationBatch when available.
- Candidate-population design: stratified over the preset T/n_e ranges and ILR-coordinate composition simplex, counter-based `jax.random`, fixed population shape (no data-dependent control flow); element subsets as masks over the bucket superset snapshot (no per-subset recompiles).
- Cost: masked element-weighted correlation between synthetic and measured spectra (continuum-removed; reuse J1's baseline kernel or its host fallback in phase 1), plus per-element marginal-evidence scoring for the BIC subset comparison.
- Memory: forward working set ≈ (B_eval, N_wl) f64 — ≈131 KB/eval at N_wl=16,384; ~10⁴ concurrent evals ≈ 1.3 GB, <10 % of HBM (ADR-0004 §5.1.2 budget). Chunk over the eval axis with `lax.map` when N_wl is large (silva2022).
- fp64 for populations/Saha; the profile-matrix fp32 island (§5.3) may be evaluated here under its Tier-B gate — this is the stage that runs thousands of evaluations per spectrum and benefits most.
- **Dataset policy:** obeys the holdout/vault split (`docs/audit/2026-06-10-goalfirst/optimization-program-design.md:72-73`); development only on the optimization tier.
- Physics-only: population search + BIC are physics/statistics; no banned imports (ruff TID251 applies).

## 3. Acceptance criteria

1. **Recall-driven F1 gain: micro-F1 ≥ +0.03 vs the reference identifier on the optimization split, with precision loss ≤ 0.02** (protects the zero-FP precision asset). This is the payoff threshold of ADR-0004 §8.3 item 3 — if not met, the bead concludes with a negative-result report and the program's accuracy case is re-evaluated.
2. Runtime ≤1 s/spectrum on one V100S at the 10³–10⁴ eval budget.
3. Accuracy sanity vs prior art: on the golden synthetic corpus, recovered major-element concentrations within ~1 % relative (the MC-CF benchmark) for converged candidates.
4. Determinism per (seed, SHA); benchmark-gated per project memory (PR #229 precedent) — shadow-board run before any campaign adoption.
5. Exposed as a Campaign-3 evaluator entry point (callable from `cflibs/evolution/` and the campaign harness; runs on GPUs while campaigns hold CPUs — `optimization-program-design.md:476`).

## 4. Test plan

`tests/jitpipe/test_forward_fitting.py`: synthetic-corpus identification round-trip (known truth, confounder panel); BIC subset-selection unit tests; eval-budget/memory canary; seed determinism. Scoring runs via the J8 shadow harness on the optimization split only.

## 5. Risks

Research-flavored variance (the largest in the program, ADR-0004 §8.3); cost-function design dominates outcome — start from the prior art's element-weighted correlation and iterate against the optimization split; risk of overfitting the split — vault datasets stay untouched; if a cheaper coherence heuristic on the reference pipeline closes the recall gap first (start-criterion 2), descope to the throughput/evaluator role.

## 6. Dependencies / files

Depends J0 + merged kernels only. Integrates with J8 (phase 2) and J9 (batch harness). Feeds J12 (M3 requires the integrated identifier). Files: `cflibs/jitpipe/identify.py` (forward-fit scorer section) or `cflibs/jitpipe/forward_fit.py`, tests. Reference untouched.

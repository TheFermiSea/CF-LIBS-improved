# J10 (o3be) — Forward-Fitting Identification: scope (2026-06-15)

**Bead:** CF-LIBS-improved-o3be · **Spec:** docs/adr/specs/J10*.md · **ADR:** ADR-0004 §8.3

## Headline

`cflibs/jitpipe/forward_id.py` is **already ~1143 lines with NO stubs** — the core algorithm
**and** the gradient-polish layer are implemented and parity-tested (`tests/jitpipe/test_parity_j10.py`,
670 lines, committed `3948d24`/`1600ecf`). J10 is **not** a cold start; the remaining work is
**wiring + acceptance evaluation + cost-function tuning**, not core physics. This downgrades o3be
from "XL research build" to "M wiring + L tuning."

## What's already done (do not rebuild)

- Scoring core (pure jit/vmap): `evaluate_population` (vmaps the frozen `radiation.kernels.forward_model`),
  `correlation_cost` (continuum-removed weighted Pearson), `bic_cost` (mirrors `model_selection._compute_bic`),
  `forward_fit_presence_scores` (leave-one-element-out include−exclude correlation gap → present/absent).
- Gradient polish: fixed-K Levenberg-Marquardt (`jax.jacfwd` + `jnp.linalg.solve`, accept/reject by
  `jnp.where`) — core-JAX clean, no banned jaxopt/optimistix/lineax.
- Host seam: `build_candidate_population` (stratified T/n_e/composition + Bernoulli subset masks,
  counter-based RNG), `forward_fit_identify` orchestrator.
- Tests: forward-eval vmap==loop (rtol 1e-12), BIC parity, polish-recovers-truth-within-1%,
  determinism, the +0.03-F1 payoff mechanism (on a 4-spectrum unit panel — a proxy, not the real split).
- Physics-only clean (`test_no_sqlite_in_kernel`; softmax as explicit exp/normalize).

It is **isolated** — referenced only by its own module, its test, and `scripts/jitpipe_gpu_bench.py`.
Nothing in `cflibs/benchmark/` imports it yet.

## Remaining work (ordered)

1. **(S) Adapter** `ForwardFitResult → ElementIdentificationResult` — a host `ForwardFitIdentifier`
   with `.identify(wavelength, intensity)` (the `IdentifierProtocol`) mapping element-indexed arrays
   to detected/rejected entries. **← the tractable first increment.**
2. **(M) Register** a `kind == "forward_fit"` branch in `cflibs/benchmark/unified.py` (~L1400) +
   `--with-forward-fit` in `scripts/benchmark_synthetic_identifiers.py`. Makes it scoreboard-scorable.
3. **(M) Acceptance evaluation** on the real optimization split (holdout/vault policy) → micro-F1/precision
   delta vs the reference identifier → `SCOREBOARD-<date>-forward-fit-vs-reference.{md,json}`.
4. **(L) Cost-function / population tuning** to clear AC1 (the research-variance core; benchmark-gate
   every change — project rule). Levers: `element_weights` (currently uniform), `presence_threshold`,
   `n_configs`, polish on/off, T/n_e stratification ranges.
5. **(S) V100S runtime** (AC2 ≤1 s/spectrum) via `jitpipe_gpu_bench.py` unit (b) on the clean
   `cflibs-gpu2` env (else CPU fallback).
6. **(S) Campaign-3 evaluator entry point** (AC5) + (S) spec/test name reconciliation
   (`test_parity_j10.py` vs spec's `test_forward_fitting.py`).

## First GPU AC1 measurement (2026-06-15)

Wiring fixed (commit 7b82a25: `_run_forward_fit` passed a raw `AtomicSnapshot`; now
converts to `PipelineSnapshot`). Ran `benchmark_synthetic_identifiers --with-forward-fit`
on the V100S (clean `cflibs-gpu2` env), corpus `ak3.1.3`, 40 spectra, `n_configs=1024`,
`presence_threshold=0.05`, uniform `element_weights`, no tuning:

| identifier | precision | recall | F1 |
|---|---|---|---|
| Comb | 0.459 | 0.425 | **0.442** |
| forward_fit | 0.444 | **0.250** | 0.320 |
| Correlation | 0.246 | 0.350 | 0.289 |
| ALIAS | 0.383 | 0.225 | 0.283 |

**Untuned forward_fit does NOT clear AC1** (F1 0.320 vs Comb 0.442; AC1 wants ref+0.03).
Decisive pattern: **recall is the LOWEST of all four (0.250)** with fine precision (0.444)
— the opposite of the intended recall advantage → the presence gate is **over-conservative**
(too-high `presence_threshold` / too-strict leave-one-out correlation-gap), starving recall.
This is a *tuning* problem, the research-variance core (task 4). CPU cannot run this
(OOM at `n_configs=1024`) — the GPU is required.

### Threshold sweep — recall ceiling (presence gate confirmed; thesis validated)

Re-ran with `CFLIBS_FF_PRESENCE_THRESHOLD=0.0` (gate fully open; the knob added to
`_run_forward_fit`), same 40 spectra:

| `presence_threshold` | forward_fit P | forward_fit R | forward_fit F1 |
|---|---|---|---|
| 0.05 | 0.444 | 0.250 | 0.320 |
| **0.0** | 0.273 | **0.512** | 0.357 |

Opening the gate **more than doubled recall (0.250 → 0.512)** — higher than *every*
baseline (Comb R=0.425), confirming the recall-starved diagnosis AND **validating the
recall-play thesis**: the recall headroom is real. But precision collapses to 0.273 at
full recall, so forward_fit's current **precision-recall frontier** (best F1 ≈ 0.357)
still sits below Comb's operating point (0.442). **Conclusion:** the threshold only trades
P↔R along a mediocre frontier; clearing AC1 requires **lifting the frontier itself** —
i.e. improving precision *at* high recall via cost-function tuning (`element_weights`
calibration, the continuum-removed correlation cost, the BIC penalty), not a threshold
tweak. **Go/no-go (ADR §8.3): INVEST.** The recall ceiling validates the thesis; J10 is
worth the cost-function tuning effort (a research-variance multi-experiment effort, each
benchmark-gated) rather than a descope. **Caveats:** 40-spectra subset; corpus validity
is the open `zfy2` question (regenerate on the fixed forward model before trusting absolute
numbers); the same-corpus relative ranking is the reliable signal.

## Acceptance (AC1 binding)

micro-F1 ≥ **+0.03** vs the reference identifier on the **optimization split**, precision loss ≤ **0.02**
(measured via `confusion_counts`+`compute_binary_metrics`, micro = pooled TP/FP/FN). If unmet → negative-result
report; J10 descopes to the throughput/Campaign-3-evaluator role (J9/J11 stand alone, ADR §8.3).

## Open questions (need a decision before AC1 tuning)

1. **Baseline comparator** for "+0.03": which reference identifier (strict ALIAS / hybrid_consensus /
   scoreboard default)? `docs/benchmarks/SCOREBOARD-2026-06-10-baseline.{md,json}` exists.
2. **Score/confidence mapping**: `presence_score` is an unbounded include−exclude correlation gap;
   map to the protocol's 0–1 (sigmoid? min-max over valid elements?) for cross-identifier comparability.
3. **Candidate element superset** on real spectra + whether the host pre-filters candidates (cost scales with E).
4. **Phase boundary**: phase-1 (parity-adapter peaks) now vs phase-2 (J8 `ObservationBatch`) later? J12 needs "integrated."

## Risks

Physics-only (the adapter must stay sklearn/torch/jax.nn-free; ruff TID251 + evolution AST scanner enforce);
GPU fp64 / clean-env fallback; recall-vs-precision tradeoff lives in the single `presence_threshold` gate;
benchmark-gate every scoring change (MEMORY: ALIAS Family-5 regressed F1 −0.041); AC1 may be unreachable
(negative result is an allowed outcome).

# Iterative Boltzmann JAX Wiring — Codex + Gemini Consultation

**Context:** Wire the existing `batched_boltzmann_fit` JAX kernel (in
`cflibs/inversion/physics/boltzmann_jax.py`) into the two composition-
workflow callers of `BoltzmannPlotFitter`:

- `cflibs/inversion/solve/iterative.py:138` — `IterativeCFLIBSSolverJax`
- `cflibs/inversion/runtime/streaming.py:592` — `FastStreamingAnalyzer`

The third caller, `cflibs/inversion/identify/alias.py`, is already
JAX-wired via `boltzmann_temperature_jax` (PR #118) — out of scope.

Profiling shows the CPU `BoltzmannPlotFitter.fit()` path dominates the
Vrabel 50,000-spectrum analysis loop (~500k `scipy.stats.linregress`
calls, multiple 90-min probe timeouts).

## Consultation summary

Queried **gpt-5.3-codex** (200-word answers) and **gemini-3-flash-preview**
in parallel via CLIAPIProxy at `localhost:8317`. Raw responses captured at
`/tmp/codex.md` and `/tmp/gemini.md` during the session.

### Q1: Integration pattern — `use_jax` flag vs separate class vs monkey-patch?

| Option | Codex | Gemini |
|---|---|---|
| (a) `use_jax: bool = False` kwarg | rejected — branching pollutes hot code | rejected — SRP violation |
| (b) `BoltzmannPlotFitterJAX` subclass | **recommended** | **recommended** |
| (c) Monkey-patch | rejected — fragile in prod | (not addressed) |

Both LLMs agree on (b). However, **the project prompt explicitly prefers (a)**
because:

1. Call-site signatures stay byte-for-byte identical
   (`BoltzmannPlotFitter(outlier_sigma=2.5, use_jax=...)`).
2. An env-var override alone (`CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1`)
   must flip behavior without other code changes.
3. The class-public surface (`.fit(observations)` returning
   `BoltzmannFitResult`) is unchanged so downstream consumers
   (`plot()`, widgets, `_translate_multiplet_indices`, etc.) keep working.

**Decision: (a) with an internal backend split** — `BoltzmannPlotFitter`
gains a `use_jax: bool = False` kwarg; when set, `.fit()` delegates to
an internal `_fit_sigma_clip_jax` method that owns all JAX logic. The
CPU path is unchanged.

The "branching in hot code" concern from Codex is mitigated because the
`use_jax` branch is checked once per `.fit()` call, not per inner loop
iteration. Single-Responsibility is preserved by isolating JAX-specific
code in dedicated private methods.

### Q2: Iterative sigma-clip emulation in JAX path?

| Option | Codex | Gemini |
|---|---|---|
| (a) Skip outlier rejection | rejected — changes statistical behavior | rejected — lossy parity |
| (b) Python-loop the JAX kernel | **recommended (start here)** | rejected — dispatch overhead |
| (c) `jax.lax.while_loop` | "overkill for scalar-per-call" | **recommended** |

The disagreement turns on **batching**. Gemini assumes 50k sequential
JAX dispatches in a tight Python loop — which would indeed be dispatch-
bound. Codex assumes the inner sigma-clip iteration count (`max_iterations=10`)
is what we're looping over, and that the outer per-spectrum loop happens
above the `.fit()` boundary.

In our actual code path the outer per-spectrum loop is in the **caller**
(`IterativeCFLIBSSolverJax`, `FastStreamingAnalyzer`), and `.fit()` is
called *once per element per spectrum*. So within a single `.fit()` call
the loop is `max_iterations=10` — small enough that Python-side loop
overhead is negligible compared to a `while_loop` reimplementation that
would need careful shape-invariant masking.

**Decision: (b) Python-loop the JAX kernel** with same masking logic as
CPU sigma-clip. The JAX kernel computes slope/intercept/R²/sigmas in a
single closed-form pass; the Python wrapper recomputes residuals, clips
outliers via the same `|residual| > outlier_sigma * std(residual)`
predicate, and re-invokes the kernel until convergence or max_iterations.
Per-call cost is dominated by data-transfer overhead (~1-2 small JAX
arrays per iteration), not compute.

### Q3: Per-spectrum API vs higher-level batched API?

| Strategy | Codex | Gemini |
|---|---|---|
| JIT-cache and reuse shapes | recommended (short term) | recommended (caching) |
| Add `.fit_batch(...)` API | recommended (long term) | **strongly recommended** |
| Refactor callers to batch | (implicit) | recommended — "the actual fix" |

Both agree a true batched API would unlock the GPU SIMD win. But neither
caller (`IterativeCFLIBSSolverJax`, `FastStreamingAnalyzer`) currently
exposes a batch dimension — both fit one Boltzmann plot per
(spectrum, element) pair, with the iterative solver looping until
convergence per spectrum.

**Decision (scoped):**

- For *this* PR: JIT-cache the kernel via the already-`@jit`-decorated
  `batched_boltzmann_fit`. Pad to a small fixed shape so JAX doesn't
  recompile per call (or accept a recompile per line-count change —
  in practice the line count per element is roughly constant within
  a single analysis run). Expected speedup: **per-call latency drop**,
  not parallelism gain.
- Out of scope (future PR): add a `BoltzmannPlotFitter.fit_batch(
  observations_per_element)` API and refactor
  `IterativeCFLIBSSolverJax._step_temperature` /
  `FastStreamingAnalyzer.analyze_spectrum` to batch across spectra.
  The benchmark CLI's `--jax-identifier` flag is the precedent for
  threading the env-var through.

The current PR ships the wiring + parity tests so future batched-API
work can be A/B-compared against an already-JAX path rather than
against the CPU baseline.

## Implementation summary

1. **`BoltzmannPlotFitter.__init__`** adds `use_jax: bool = False` kwarg
   (default keeps as-shipped CPU behavior).
2. **`BoltzmannPlotFitter.fit`** routes to a new `_fit_sigma_clip_jax`
   method when `use_jax=True AND method == FitMethod.SIGMA_CLIP`. Other
   methods (RANSAC, Huber) fall through to CPU — they have different
   numerical semantics and aren't bottlenecks in the composition workflow
   (which always uses SIGMA_CLIP).
3. **`_fit_sigma_clip_jax`** mirrors `_fit_sigma_clip` byte-for-byte
   except the per-iteration `np.polyfit(..., cov=True)` call is replaced
   by `batched_boltzmann_fit(...)` with a `(1, N_max)` batched-of-one
   input. Returns the same `BoltzmannFitResult` shape.
4. **Call sites** read `CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION` from the
   environment at construction time. Default unset → `use_jax=False`
   → identical to current behavior.
5. **Multiplet aggregation** is preserved — happens upstream of the
   sigma-clip iteration regardless of backend.

## Parity guarantees

The JAX kernel's closed-form WLS (`S_w, S_wx, S_wy, S_wxx, S_wxy` 5-sum
reduction) is algebraically equivalent to `np.polyfit(x, y, 1, w=weights,
cov=True)` where `weights = 1/sigma_y^2`. The closed-form sigma_slope
and sigma_intercept match the cov-matrix diagonal of polyfit. R-squared
formulas are weighted versions of the same Σw(y-ŷ)²/Σw(y-ȳ)² definition.

The outlier rejection predicate (`|residual| > outlier_sigma *
np.std(residuals)`) is computed identically in both paths (no JAX
involved — just numpy on the inlier subset after kernel returns).

Expected agreement: **rtol ~1e-8 on slope/intercept/T**, **identical
inlier_mask on noise-free synthetic inputs**, and **inlier_mask
divergence by ≤1 point on noisy inputs** (due to last-decimal-place
residual ordering, when a point is exactly at the rejection boundary).
The tests target rtol 1e-5 to leave headroom.

## References

- Q1-Q3 framed in the project prompt for `feat/jax-boltzmann-composition`.
- Codex + Gemini queried via CLIAPIProxy 2026-05-12.
- Existing JAX kernel: `cflibs/inversion/physics/boltzmann_jax.py` (PR
  predating this branch).
- Identifier-side precedent: PR #118 `feat(identify): JAX-vectorized
  Boltzmann temperature` and PRs #119-#122 for the per-identifier
  `use_jax_*` kwarg pattern.
- Env-var precedent: `CFLIBS_USE_JAX_IDENTIFIER` toggled by
  `scripts/run_unified_benchmark.py:333`.

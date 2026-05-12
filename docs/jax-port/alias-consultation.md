# JAX Port of `alias.py` — Codex + Gemini Consultation Synthesis

Date: 2026-05-12
Branch: `feat/jax-alias-finish`
Models consulted: `gpt-5.3-codex` (Codex), `gemini-3.1-pro-low` (Gemini 3.1 Pro)

## Audit of remaining scipy hot spots in `cflibs/inversion/identify/alias.py`

| # | Hot spot | Per-spectrum cost | JAX viability |
|---|----------|-------------------|----------------|
| 1 | `scipy.optimize.nnls` (lines 627, 2132, 2147, 2254) | ~12-15 calls per spectrum (1 full + leave-one-out + iron-group + sparse fallback) on ~30×12 matrices | High |
| 2 | `scipy.optimize.minimize` L-BFGS-B (line 2240) for sparse elastic-net NNLS | 1 call per spectrum, ~12-vector | High |
| 3 | `scipy.signal.find_peaks` (line 966) | 1 call per spectrum, ~5000-pixel input | Low (dynamic output shape) |
| 4 | `scipy.special.erf` (line 2422) | 1 scalar call per spectrum | Low (scalar dispatch overhead) |
| 5 | `_build_nnls_templates` Python loop (lines 2079-2096) | 1 build per spectrum, broadcastable | High |
| 6 | `scipy.stats.binom.sf` (line 2408) | 1 scalar call per element per spectrum | Low (no JAX equivalent + cheap) |
| 7 | `scipy.stats.linregress` (line 1167) | 1 call per spectrum | **Already ported** in PR #118 |

## Question A — NNLS solver choice

**Both models agree: FISTA (projected gradient + Nesterov momentum) is the right call.**

* Codex: projected-gradient NNLS with Lipschitz step from `||A||_2^2`. Trivial under
  `jit`/`vmap`. Optional Nesterov acceleration for faster convergence.
* Gemini: FISTA explicitly — Nesterov momentum hits `rtol 1e-5` reliably; FNNLS
  (Lawson-Hanson active-set) has dynamic shape issues under `vmap`.

**Why not FNNLS:** active-set branching → variable-length passive set → cannot
`vmap` cleanly. The whole point of the JAX path is future batching, so we
avoid it.

**Why not `jax.scipy.optimize.minimize` BFGS:** both models confirmed it does
**not** support bounds. JAXopt has `LBFGSB`/`ProjectedGradient` but adds a
dependency we don't need.

**Iteration budget (followup to Codex):** 1000 iters default, hard cap at 2000.
For correlated columns (which happens when neighboring elements have overlapping
peaks), objective converges fast but `x`-values may shift away from the
SciPy active-set minimum. With well-conditioned `A` and a fixed-iter loop,
1000 iters is plenty for `rtol 1e-5` on residual; `x`-level matching may
relax to `atol 1e-4` in adversarial cases.

**Final design:** FISTA with `lax.fori_loop` (fixed iter count, jit-friendly).
KKT-based early-stop is avoided because dynamic iteration count breaks
`vmap`. 1000 iters is the default; the inner FISTA implementation is shared
across the dense and sparse-elastic-net branches.

## Question B — Sparse elastic-net NNLS

**Both models agree:** no `minimize` with bounds in JAX. Use FISTA — under
`x ≥ 0`, the L1 term collapses to `α·sum(x)` and the full elastic-net
objective is smooth, so a single FISTA implementation handles both
the dense and sparse cases via different gradient terms.

Objective: `0.5 ||A_norm x - b||^2 + l1·sum(x) + 0.5·l2·||x||^2`
Gradient:  `A_norm^T (A_norm x - b) + l1 + l2·x`

Column normalization is preserved exactly as in the CPU code path for
backward-compatible coefficient scaling (`sparse_c = result.x / col_norms`).

## Question C — `find_peaks`

**Both models agree: SKIP for this PR.** Reasons:

* Dynamic output shape (variable number of peaks) — incompatible with `vmap`
  without padding to `max_peaks`, which adds complexity for no current benefit.
* `scipy.signal.find_peaks` is already C-backend, called once per spectrum.
  The win is only when wrapping inside a `vmap`-ed batch path, which isn't
  there yet.
* Both recommend implementing later as `lax.reduce_window` 1D max-pool +
  boolean mask + padding-to-`max_peaks`, when an end-to-end batched path is
  built.

**Decision:** Document as left on CPU. The JAX path uses the CPU peak
detector and only ports the per-element scoring downstream.

## Question D — Scalar `erf`

**Both models agree:** porting a single scalar call is a pure dispatch loss
(JAX trace overhead > C call). However, if the surrounding context is
batched (e.g. computing `P_SNR` for many spectra inside a `vmap`-ed call),
the JAX version becomes free.

**Decision:** Provide `_compute_p_snr_jax` that operates on a batch of
spectra (B-element intensity batch + B-list of peak intensities). The
batched version uses `jax.scipy.special.erf` and is the opt-in path; the
existing scalar code path is untouched.

## Question E (added) — Gaussian template matrix builder

Gemini specifically pointed out that the right port for hot spot #5 is
broadcasting:

```python
# shapes: peaks (n_peaks,1), lines (1, n_lines), sigma (n_peaks,1)
z = (peaks - lines) / sigma
A_col_contrib = eps * jnp.exp(-0.5 * z**2) * (jnp.abs(peaks - lines) < 3 * sigma)
```

This is the cleanest port and the most directly batchable across spectra.

## Final port plan (this PR)

| # | Function | Action | Tests |
|---|----------|--------|-------|
| 1 | `solve_nnls_jax(A, b)` | New module-level function — FISTA, 1000 iters, jit | rtol 1e-5 vs scipy on random + correlated matrices |
| 2 | `solve_sparse_nnls_jax(A, b, alpha, l1_ratio)` | New module-level — same FISTA with elastic-net grad | match L-BFGS-B output to atol 1e-4 (non-unique solutions allowed to shift) |
| 3 | `_compute_nnls_attribution_jax(A, peak_intensities)` | Vectorized leave-one-out: builds a (n_cands, n_peaks, n_cands-1) tensor of leave-one-out A's, runs `vmap`-ed FISTA over axis 0 | rtol 1e-4 vs CPU (FISTA accumulates) |
| 4 | `_build_nnls_templates_jax(...)` | Vectorized via broadcasting | rtol 1e-10 vs CPU (exact arithmetic) |
| 5 | `_compute_p_snr_jax(intensity_batch, peak_indices_padded, peak_counts)` | Batched JAX p_snr using `jax.scipy.special.erf` | rtol 1e-6 vs CPU |
| 6 | `find_peaks` | LEAVE ON CPU — documented in PR body | — |
| 7 | `binom.sf` | LEAVE ON CPU — scalar, no JAX path needed | — |
| 8 | Constructor opt-in flags: `use_jax_nnls`, `use_jax_p_snr` | Mirror `use_jax_boltzmann_fit` pattern | constructor tests |

The new code lives in `alias.py` next to the existing
`boltzmann_temperature_jax`. No external file changes.

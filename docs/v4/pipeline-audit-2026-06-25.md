# CF-LIBS Pipeline Audit — 2026-06-25

Multi-agent audit of the entire inversion pipeline + all solvers, on three lenses:
**scientific correctness**, **genuine vectorization**, **complexity/flag-debt**.
9 units, ripgrep+Read, concurrency-3, file-checkpointed. Mission order: ACCURACY > PRECISION > RELIABILITY > latency.

## Executive verdict

- **The core physics is largely correct.** Saha constant (incl. the factor-of-2 spin degeneracy), Boltzmann
  level populations, emissivity `ε = (hc/4πλ)·A_ki·n_k`, Voigt/Faddeeva profiles, the Boltzmann-plot WLS fit,
  the Saha ionic→neutral correction, closure algebra, the NUTS Dirichlet-simplex prior, and the scalar
  RANSAC consensus are all implemented correctly. The audit did **not** find the pipeline to be broadly wrong.
- **But there are real, specific correctness bugs** — one CRITICAL (a 10⁶× unit error in the manifold
  forward), and several HIGH (a non-physical T clamp, a documented-but-broken solver method, an
  under-spanned SVD basis, an invalid NNLS covariance, two Bayesian noise/sampler bugs). Notably, the
  worst bug is in a **non-flagged** path — the flags weren't hiding it; insufficient cross-path verification was.
- **Your thesis is confirmed on complexity:** 15+ `CFLIBS_*` env vars, several gating *physics* decisions,
  with inconsistent override semantics, two correct-physics paths defaulting OFF, a dead parallel
  quality-scoring module, and an 8-flag benchmark cluster that makes scoring non-reproducible from logs.
  This obscures *which* algorithm runs and makes correctness hard to verify — even if it didn't directly
  cause the bugs.
- **Optimization:** the only genuine, accuracy-safe perf win is replacing two `O(n·w)` Python rolling-median /
  per-line loops with C-level calls (~100×). The "vectorized RANSAC" is both slower *and* semantically wrong
  — delete it.

---

## §1 Scientific-correctness issues (ranked)

### 🔴 CRITICAL

**C1. Manifold emissivity missing cm³→m³ conversion — 10⁶× scale error.**
`cflibs/manifold/generator.py:567-804` (and `_compute_spectrum_snapshot_ldm` ~926).
`ε = (hc/4πλ)·A·n_upper` needs `n_upper` in m⁻³ with SI constants, but `n_e` is cm⁻³ (project convention) and
the `* 1e6` is missing. The **identical bug was already fixed** in the two-zone Bayesian forward
(`forward.py:517`) — the manifold path was missed in that pass. **Every pre-built manifold is 10⁶×
underscaled.** Fix: `n_upper_m3 = n_upper * 1.0e6` in both snapshot paths; regenerate manifolds.
*(Corroborates the known "manifold generator is stale" finding.)*

### 🟠 HIGH

**C2. Closed-form ILR clamps T=50000 K on non-physical slope instead of holding the prior.**
`cflibs/inversion/solve/closed_form.py:344-347`. A non-negative Boltzmann slope is unphysical; the iterative
path correctly holds `T=T_prev` and flags non-converged, but the ILR path sets `T=50000` → "keystone
collapse" (closure degenerates to raw-intensity softmax) and a meaningless composition still propagates
into the n_e refinement. Fix: `T_K = initial_T_K`; gate n_e refinement on `physical=True`.

**C3. SVD library never generates composition-perturbation rows at default `sweep_points=9`.**
`cflibs/inversion/solve/full_spectrum.py:476-488`. The T/ne grid (3×3=9) exhausts the budget, so
`extra = max(0, 9-9) = 0` → the loop that adds single-element composition spectra never runs. The SVD basis
spans T/ne but **not composition directions** (Hébert 2020 §3 uses single-element spectra precisely for this),
so the optimizer moves in an under-covered subspace. **Directly affects the newly-integrated full-spectrum
solver.** Fix: default `sweep_points = 9 + len(elements)`, or always generate composition rows.

**C4. `JointOptimizer.optimize(method="L-BFGS-B")` raises `ValueError` at runtime** despite being the
documented default-named method. `cflibs/inversion/solve/joint_optimizer.py:526,682`. `jax.scipy...minimize`
only accepts `"bfgs"`; `"lbfgsb"` → `ValueError`. Fix: use real `scipy.optimize.minimize(method="L-BFGS-B")`
(supports bounds — physically valuable for T/ne/composition limits) or restrict to BFGS and fix docs.

**C5. Poisson readout residual uses the Poisson mean `mu` instead of `predicted`.**
`cflibs/inversion/solve/bayesian/likelihood.py:107`. Gaussian read-noise acts on raw counts; the residual
should be `observed - predicted`, not `observed - (predicted/gain + dark)`. Conflates shot-noise expectation
with the readout reference and makes the penalty depend incorrectly on `gain`/`dark`. Opt-in branch, but it's
the docstring-recommended "best" choice.

**C6. NUTS two-zone T-ordering uses a finite `-1e6` hard barrier — pathological for HMC gradients.**
`cflibs/inversion/solve/bayesian/models.py:152`. A finite cliff yields a constant gradient pointing toward the
boundary regardless of position → divergences, poor mixing. Fix: reparameterize
(`T_shell ~ U(...)`, `delta_T ~ HalfNormal`, `T_core = T_shell + delta_T`).

**C7. NNLS coefficient uncertainty uses the OLS covariance `(AᵀA)⁻¹` — invalid at active-set boundaries.**
`cflibs/inversion/identify/spectral_nnls.py:609-618`. NNLS with active (zero-clamped) constraints has a
truncated distribution; OLS σ underestimates uncertainty for near-zero coefficients → over-estimated SNR →
false-positive trace elements. Fix: bootstrap residual σ, or `snr = coeff / (noise_σ · ‖basis_row‖₂)`.

**C8. Vectorized RANSAC scores on per-peak minimum, not one-to-one assignment** → ranks a many-to-one
hypothesis above the correct one-to-one winner. `wavelength_calibration.py:612-620`. (Also slow — see O1.)
The scalar path is correct. Fix: delete the vectorized path (see X1).

### 🟡 MEDIUM (notable correctness)

- **M1. JAX `lax`-while path silently drops corona two-region T-weighting** for `{Si,Fe,Ca,Al,Mg}` (the most
  common rock-formers). `iterative.py:831-841`. Python path applies `0.3·T + 0.7·T_corona`; lax path uses `T`
  uniformly → different composition for two-region datasets. Fix: carry `corona_sensitive_mask` into the lax
  body, or force the Python path when `two_region=True`.
- **M2. Saha S2 guard `if ip_II is not None` should be `ip_III is not None`** → computes a spurious U_III for
  species with no third ionization stage, depressing n_I/n_II. `saha_boltzmann.py:236-247`. CPU/JAX **parity
  gap** (the JAX kernel guards correctly). Fix: guard on `ip_III`.
- **M3. Default broadening is `LEGACY`** (`0.01·√(T_eV/0.86)`, no λ/mass dependence) for `SpectrumModel` /
  `cflibs forward`. `kernels.py:710`. Physical Doppler exists but isn't the default. Fix: default to
  `PHYSICAL_DOPPLER` (benchmark-gate; it's already the Bayesian/manifold default).
- **M4. Sigma-clip outlier rejection uses unweighted residual std**, biasing rejection against low-σ lines.
  `boltzmann.py:427-432`, `_jax_reject_outliers:651`. Fix: clip on normalized residuals `r_i/σ_i`.
- **M5. `apply_oxide_mode` docstring quotes the wrong constant** (molar-mass ratio 2.139 vs O/cation 2.0 for
  Si). `closure.py:819,833`. A caller following the docstring corrupts composition by up to ~70% (Al). Fix
  docstring/example to O/cation.
- **M6. Self-absorption `g_i = g_k` substitution** biases τ for excited lower levels (up to 3× for large
  g-ratio lines). `self_absorption.py:857`. Fix: plumb true `g_i` through `LineObservation`.
- **M7. `_tau_ratio` drops the lower-level Boltzmann factor**; safe for true same-upper-level doublets but
  `find_doublet_pairs` doesn't constrain lower levels (up to ~25% τ-ratio error). `self_absorption.py:299-317`.
- **M8. MC-UQ `A_ki` floor of 1.0 s⁻¹** is an absolute hard floor → asymmetric bias on weak lines (biases T
  upward). `uncertainty.py:1055`. Fix: relative floor `max(A_ki, 0.01·A_ki_nominal)`.
- **M9. Joint Hessian covariance scaling** double-applies the χ²-inflation and misses a `2/n` factor.
  `joint_optimizer.py:573,743`. Fix: document as noise-inflation or use plain `inv(H)` with the correct factor.
- **M10. BIC/AIC fit term assumes homoscedastic noise** (LIBS is Poisson/heteroscedastic).
  `model_selection.py:117`. Fix: weighted RSS `Σ(r_i/σ_i)²` using the already-estimated per-pixel σ.
- **M11. RANSAC+BIC uses `n = n_inliers`**, which can favor a simpler model merely because it recruited fewer
  inliers (Torr & Zisserman 2000). `wavelength_calibration.py:679`. Fix: pass `n = total candidate pairs`.
- **M12. Python n_e convergence check divides by bare `ne_prev`** (lax path guards with 1e-30).
  `iterative.py:1964`. Fix: `max(ne_prev, 1e-30)`.

---

## §2 Genuine-optimization issues (latency is LAST priority — but these are accuracy-safe)

- **O1. The "vectorized RANSAC" is 1.8× SLOWER (measured) and semantically wrong.** `np.minimum.at` is an
  unbuffered 2-D scatter (300k non-contiguous writes), and the batch can't early-exit. **Delete it** (the
  scalar path with early-exit is correct and faster). If a batch path is ever wanted, use
  `np.minimum.reduceat` on a peak-id-sorted axis (SIMD-friendly), never `ufunc.at`.
- **O2. `detect_ccd_seams` rolling median is an `O(n·w)` Python loop** calling `np.median` per pixel.
  `wavelength_calibration.py:1428`. Fix: `scipy.ndimage.median_filter(dl, size=2w+1, mode='nearest')` —
  **parity-exact, ~100× faster**, single C call. **This is the cleanest safe perf win.**
- **O3. NumPy per-line broadening loops** (`O(N_lines)` kernel launches) on the legacy CPU path.
  `profiles.py:154,337`. The correct vectorized outer-product already exists in `kernels.py:643`. Fix: rewrite
  with `wl_grid[:,None] - line_wl[None,:]` broadcasting.
- **O4. `grad_fn` recomputed after optimization** (extra XLA compile + forward/backward) just for the gradient
  norm. `full_spectrum.py:604`, `joint_optimizer.py:693`. Fix: reuse `res.jac` or a cached jitted grad.

> Note: per the profiled hotspot data, RANSAC-with-early-exit is *not* the bottleneck it appeared to be; the
> real, safe wins are O2/O3. Do not chase within-RANSAC micro-optimization — it's accuracy-coupled and
> RNG-seeded.

---

## §3 Complexity & flag-debt map + simplification plan

**Census: 15+ `CFLIBS_*` env vars**, several gating physics, with inconsistent semantics. The map:

| Flag / artifact | Verdict | Action |
|---|---|---|
| `CFLIBS_VECTORIZED_RANSAC` | Selects a slower, parity-breaking variant; "opt-in to *worse*" | **DELETE** flag + `_ransac_search_vectorized` |
| `CFLIBS_MCWHIRTER_RESONANCE_DE` | Gates the **correct** (DB-validated) physics but defaults to the **wrong** δE | **FLIP default ON** (benchmark-gate), then remove flag |
| `CFLIBS_HOUGH_CALIB` | Deterministic, benchmark-confirmed held-out win, but default OFF | **PROMOTE to default ON**, opt-out only |
| `CFLIBS_CALIB_POOL_CACHE` | Parity-exact, faster, default OFF | **PROMOTE to default ON** |
| `CFLIBS_RANSAC_EARLY_EXIT` | Held-out win but +0.15 dev regress; small held-out N | Keep gated **until** confirmed on larger held-out, then promote |
| `CFLIBS_RELIABILITY_FROM_UNCERTAINTY` | Correctness lever, never benchmarked, default OFF indefinitely | Benchmark & land winner as default, or remove |
| `CFLIBS_USE_LAX_WHILE_LOOP`, `CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION` | Dual numerical paths that must stay byte-identical (M1 shows they don't) | Promote to constructor kwargs; expose effective value in `quality_metrics` |
| `CFLIBS_REFUSE_TO_REPORT` | Physics-layer gate, bare env read | Promote to constructor kwarg |
| `CFLIBS_FF_*` (8 entries) | Benchmark fitness-fn shape; non-reproducible from logs | Collapse into a `FitnessFunctionConfig` dataclass, log to JSON |
| `QualityAssessor` (`physics/quality.py`) | **Dead** parallel quality path w/ its own threshold table; no shipped consumer | **DELETE**; migrate 2 tests to the solver's canonical path |
| 3 calib flags' dual config+env resolution | Inconsistent: `ransac_early_exit=False` can't override env (`or None` coercion) | Unify to tri-state `True/False/None`; document resolution order once |
| `method` (two meanings in adjacent solver code) | `regularization` vs optimizer algorithm — same name | Rename `full_spectrum` param to `regularization`/`prior_mode` |

**Principle for the cleanup:** the flag debt exists because past gated changes were never *promoted or removed*.
The fix is to **follow through the discipline to its end state**: benchmark each lever, promote winners to the
default and **delete the flag**, remove losers. End state = one correct code path per decision, flags only for
genuinely-undecided experiments.

---

## §4 Recommended action sequence (mission-ordered)

1. **Fix C1 (10⁶× manifold scale).** One-line-class fix matching the already-fixed sibling; regenerate a small
   manifold and confirm the manifold solver now matches forward/iterative. Unblocks the manifold path.
2. **Fix C3 + C4 (SVD composition basis; L-BFGS-B ValueError).** Both directly affect the full-spectrum/joint
   solvers we just integrated (the joint solver is the one that *beats* iterative).
3. **Fix C2 (ILR T-clamp → hold prior).** Clear correctness fix.
4. **Fix C5 + C6 (Bayesian Poisson residual; NUTS reparam).** Needed before trusting the Bayesian path.
5. **Fix C7 (NNLS covariance).** Improves identification precision / SNR gate.
6. **Flip wrong-physics defaults behind benchmark gates:** McWhirter resonance δE on; LEGACY→PHYSICAL_DOPPLER.
   (Standing rule: measure flag-off vs -on on held-out, promote only on a win + zero regression, then drop flag.)
7. **Delete `CFLIBS_VECTORIZED_RANSAC` + `_ransac_search_vectorized` (X1/O1)** and `QualityAssessor` — pure
   simplification, zero behavior change on the default path.
8. **Land the MEDIUM correctness batch** (M1–M12), each verified against the governing equation.
9. **Apply the two safe perf wins (O2/O3)** — parity-exact, ~100× and ~10–100× on their paths.
10. **Execute the flag-simplification table (§3)** — promote winners, unify semantics, FitnessFunctionConfig.

Each accuracy-changing fix follows the standing discipline: default-OFF flag → benchmark flag-off vs -on on
dev + held-out → promote to default only on a measured win with zero regression → **then remove the flag** so
the simplification actually lands.

---

## Execution status (2026-06-25)

### Landed + verified + pushed (`v4/m5-kurucz-atomic`)
- **C1** manifold 10⁶× emissivity scale — fixed both snapshot paths; real-DB parity verified; regression test added.
- **C2** ILR non-physical-slope holds prior T (no 50000 K clamp); unit test.
- **C3** full-spectrum SVD basis always spans composition; real-DB recovery verified; test. (NOTE: composition is now *movable* but the solver still under-recovers in float64 — see "needs benchmark" below.)
- **C4** `JointOptimizer` L-BFGS-B/CG route through scipy (were silent no-ops); parametrized test.
- **C5** Bayesian Poisson readout residual `observed−predicted`; unit test.
- **C6** NUTS T-ordering smooth quadratic hinge (was zero-gradient cliff); `jax.grad` test.
- **X1** deleted vectorized RANSAC (`_ransac_search_vectorized` + `CFLIBS_VECTORIZED_RANSAC`) — proven 1.8× slower + parity-breaking.
- **M5** oxide-closure docstring (O/cation, not molar-mass ratio).
- **SOLVER_BACKENDS** + `build_pipeline_config` solver validation (fixed 2 pre-existing dispatch tests).

Fast gate (`not slow and not requires_db`): **3488 passed**; the only reds are pre-existing (below).

### Audit findings CORRECTED (false positives — do NOT apply)
- **QualityAssessor "dead code"** is FALSE — it is live in `iterative.py:2099` (M7 Lever 6). Not deleted.
- **M2 Saha S2 guard** is FALSE — S2 = n_III/n_II uses the *second* ionization potential `ip_II` (correctly used); the existing `if ip_II is not None` guard is correct. The proposed `ip_III` guard would wrongly zero stage-III populations.
- *Lesson:* the workflow only adversarially-verified CRITICAL/HIGH findings; the unverified MEDIUM/complexity tier produced these false positives. Treat remaining MEDIUM/complexity items as leads requiring per-item verification.

### Deferred — needs a node benchmark (scoring/accuracy-affecting; gate per standing rules)
- **C7** NNLS SNR covariance — synthetic **ID-F1** benchmark (identifier-scoring change).
- **Default flips**: McWhirter resonance-δE ON; LEGACY→PHYSICAL_DOPPLER broadening; **M3/M4/M10**.
- **Full-spectrum convergence quality** (C3 follow-on: under-recovers in float64) — the sigma_d/Hessian items + Step-4 validation.

### Deferred — design decision + benchmark
- **`test_dispatch_routes_full_spectrum`** (the one remaining red, pre-existing since integration c7a43f8): the oracle spec forbids a peak-solver call in the full-spectrum dispatch, but the integration computes an iterative warm-start there and `solve_full_spectrum` requires a warm-start. Options: (a) move warm-start into `_run_full_spectrum_solver` (re-detect internally); (b) default warm-start; (c) update the spec to pass observations. Decide via a benchmark: does the iterative warm-start improve full-spectrum accuracy? (Ties into Step-4.)

### Deferred — per-item verification (unverified MEDIUM tier)
- M1 (corona two-region T in lax path), M6/M7 (self-absorption g_i / τ-ratio), M8 (MC A_ki floor), M9 (joint Hessian covariance), M11 (RANSAC+BIC n), M12 (n_e floor guard).

### Deferred — safe perf (latency = lowest priority)
- O2 `detect_ccd_seams`→`scipy.ndimage.median_filter` (parity-exact ~100×), O3 per-line broadening vectorization, O4 grad caching.

# Research Summary

**Project:** GPU-Accelerated CF-LIBS Multi-Element Plasma Diagnostics
**Domain:** Computational plasma spectroscopy, GPU numerical methods, LTE plasma diagnostics
**Researched:** 2026-03-23
**Confidence:** MEDIUM

## Unified Notation

| Symbol | Quantity | Units/Dimensions | Convention Notes |
|--------|---------|-----------------|-----------------|
| T | Electron temperature | K (primary), eV (derived: T_eV = k_B * T_K) | eV used in Boltzmann plots; K used in partition functions |
| n_e | Electron number density | cm^-3 | Plasma spectroscopy convention (not SI m^-3) |
| C_i | Mass fraction of element i | dimensionless, sum = 1 | Softmax: C_i = exp(theta_i) / sum exp(theta_j) |
| lambda | Wavelength | nm | Internal convention; NIST data in Angstroms requires /10 conversion |
| E_k | Upper-level energy | eV | Boltzmann plot x-axis |
| A_ki | Transition probability | s^-1 | Einstein A coefficient (spontaneous emission) |
| g_k | Statistical weight of upper level | dimensionless | = 2J+1 |
| U(T) | Partition function | dimensionless | Polynomial: log U = sum a_n (log T_K)^n (Irwin 1981) |
| w(z) | Faddeeva function | dimensionless | w(z) = exp(-z^2) erfc(-iz); Voigt = Re[w(z)] / (sigma sqrt(2 pi)) |
| sigma_G | Gaussian (Doppler) width | nm | Standard deviation, not FWHM. FWHM_G = 2.355 * sigma_G |
| gamma_L | Lorentzian (Stark) HWHM | nm | Half-width at half-maximum. FWHM_L = 2 * gamma_L |
| chi_z | Ionization potential of stage z | eV | May be depressed by IPD at high n_e |
| SAHA_CONST | Saha prefactor | cm^-3 K^-3/2 | 4.829e15; for n_e in cm^-3, T in K |

**Metric/signature:** N/A (non-relativistic plasma physics).
**Fourier convention:** N/A (no Fourier transforms in core pipeline; spectral computation is direct summation).
**Unit system:** Mixed CGS-eV (temperatures in K/eV, densities in cm^-3, wavelengths in nm, energies in eV). SI used only for emissivity output (W m^-3 sr^-1).

**Convention conflicts resolved:**
1. **Lorentzian width (HWHM vs FWHM):** Literature splits 50/50. Codebase uses HWHM internally. FWHM_L = 2 * gamma. A factor-of-2 error here is the single most common unit bug in Voigt implementations.
2. **Wavelength (nm vs Angstrom vs cm):** NIST ASD uses Angstroms; oscillator strength formulas use cm; codebase uses nm. Convert at I/O boundaries only.
3. **Air vs vacuum wavelengths:** NIST reports air wavelengths above 200 nm. Use vacuum internally; apply Edlen (1966) correction on import.

## Executive Summary

This project targets GPU acceleration of the full Calibration-Free LIBS pipeline across five computational kernels: Voigt profile evaluation, vectorized Boltzmann fitting, Anderson-accelerated Saha-Boltzmann solving, softmax compositional closure, and batch forward modeling. The literature strongly supports JAX as the framework and identifies clear algorithmic choices for each kernel. The Weideman 1994 N=36 rational approximation is the recommended Voigt kernel (branch-free, autodiff-compatible, machine-precision in float64). Batched Boltzmann fitting reduces to closed-form 2-parameter normal equations, trivially parallelizable. Anderson acceleration with depth m=3-5 should reduce Saha-Boltzmann iteration counts from 15-30 to 5-8, and softmax closure is already implemented in the codebase. The batch forward model via JAX vmap is the primary GPU consumer, with memory (not compute) as the bottleneck on V100S.

No published GPU-accelerated CF-LIBS pipeline exists. GPU spectroscopy codes (ExoJAX, HELIOS-K, RADIS) target molecular absorption for exoplanets, not atomic emission-based LIBS inversion. This is the gap the project fills. The key architectural insight is that LIBS operates with 10^3-10^4 lines (not 10^6-10^8), so GPU parallelism must come from batching many spectra simultaneously rather than parallelizing within a single spectrum. Published GPU spectroscopy benchmarks suggest 50-200x speedups are achievable for batch sizes >= 64 on V100S.

The principal risks are: (1) Humlicek W4 gradient instability silently poisoning autodiff-based optimization (mitigated by using only the Weideman kernel), (2) float32 precision loss in Saha exponentials at high ionization potentials (mitigated by V100S's excellent 1:2 float64:float32 throughput ratio), (3) IPD not applied to polynomial partition functions at high n_e (requires implementing a correction term), and (4) LTE breakdown at manifold grid boundaries (requires McWhirter criterion enforcement). All four are well-understood and have concrete mitigation strategies.

## Key Findings

### Methods

**Voigt profile:** Use Weideman N=36 exclusively for all JAX code paths. It is branch-free (uniform GPU warp execution), autodiff-compatible, and achieves machine precision in float64. The existing Humlicek W4 in the manifold generator should be replaced -- its `jnp.where` branching causes gradient instability (NaN at high n_e) and 4x redundant computation. Zaghloul 2024 Chebyshev is more complex to implement with marginal benefit; keep as accuracy reference only. [CONFIDENCE: HIGH -- multiple independent implementations validate Weideman; ExoJAX uses it successfully]

**Boltzmann fitting:** Closed-form normal equations for the 2-parameter weighted linear regression (slope = -1/T_eV, intercept = concentration factor). Five dot products + two divisions per batch element. Numerically stable for the 2-parameter case (no need for QR/SVD). Use sigma-clipping for outlier rejection (fully vectorizable in JAX, unlike RANSAC). [CONFIDENCE: HIGH -- standard linear algebra]

**Anderson acceleration:** Depth m=3 for the 1D n_e problem; m=5 for the joint (n_e, T) case. Reduces iteration count from 15-30 to 5-8 for Saha-Boltzmann. Use Tikhonov regularization (lambda=1e-6) on the least-squares solve and Picard fallback if residual increases for 3 consecutive steps. The combination of AA + JAX differentiability + plasma physics is novel (no published implementation). [CONFIDENCE: MEDIUM -- AA is well-established but untested in this specific application]

**Softmax closure:** Already implemented in `joint_optimizer.py`. Log-sum-exp stabilization prevents overflow. Gradients are well-conditioned for typical LIBS compositions (2-10 elements, 1-90% concentrations). [CONFIDENCE: HIGH -- standard technique, already in codebase]

**Batch forward model:** vmap over (T, n_e, C) batch dimension; broadcasting for the (N_wl, N_lines) inner product. Memory-limited on V100S: the (batch, N_wl, N_lines) intermediate in complex64 is the bottleneck. Safe batch sizes: 400 (500 lines) to 2000 (100 lines) on 32 GB V100S. [CONFIDENCE: MEDIUM -- batch sizes need empirical validation on actual hardware]

### Prior Work Landscape

**Must reproduce (benchmarks):**
- Voigt profile accuracy vs scipy.special.wofz: relative error < 1e-6 (float64), < 1e-3 (float32). [CONFIDENCE: HIGH]
- Saha ionization fractions for Fe I/II at T=1 eV, n_e=1e17: < 5% relative error vs NIST. [CONFIDENCE: HIGH]
- Round-trip forward-inversion temperature recovery: < 2% relative at SNR=100. [CONFIDENCE: HIGH]
- Anderson vs Picard: identical converged values to < 1e-8 relative; 2-4x iteration reduction. [CONFIDENCE: HIGH for convergence; MEDIUM for iteration reduction factor]

**Novel contributions:**
- First GPU-accelerated end-to-end CF-LIBS pipeline (no prior published work)
- First differentiable Saha-Boltzmann with Anderson acceleration in JAX
- Systematic GPU benchmark at LIBS scale (10^3-10^4 lines, batched)
- Comparison of Voigt algorithms specifically within JAX on V100S

**Defer:**
- Multi-GPU (pmap) strategies: single V100S sufficient for target manifold sizes
- Tensor Core / float16 exploitation: unsuitable for spectroscopy precision
- Non-LTE extensions: out of scope per project contract

### Computational Approaches

The V100S (32 GB HBM2) is well-suited for this project. Its 1:2 float64:float32 throughput ratio (8.2 vs 16.4 TFLOPS) is the best among NVIDIA GPUs and makes float64 viable for physics-critical computations at only 2x cost. Use float64 for Saha exponentials, partition functions, and Boltzmann factors; float32 is sufficient for Voigt profile rendering and FAISS vectors.

FAISS at the PCA-reduced dimension d=30 is memory-efficient: even 100M vectors fit in 12 GB with IndexFlatL2. Use exact search (IndexFlatL2) for manifolds under 10M spectra -- it is fast enough at d=30. PCA reduction is essential for meaningful nearest-neighbor results (curse of dimensionality at raw d=10000).

XLA compilation takes 10-60 seconds on first call for the forward model. Pad the last batch to avoid recompilation. Benchmark with `block_until_ready()` to avoid measuring dispatch overhead rather than computation.

### Critical Pitfalls

1. **Humlicek W4 gradient instability** -- JAX evaluates all `jnp.where` branches during AD; exp() overflow in region 4 produces NaN gradients at high n_e. Prevention: use Weideman exclusively; delete or fence Humlicek from active paths. [CRITICAL]

2. **float32 precision loss in Weideman coefficients** -- Coefficients span 14 orders of magnitude; float32 effectively reduces N=36 to N~20, giving 0.1-1% wing errors. Prevention: use float64 on V100S (only 2x cost). [CRITICAL]

3. **IPD not applied to polynomial partition functions** -- Irwin polynomials encode all bound states up to free-atom IP. At n_e > 1e18, IPD removes high-lying states, causing 5-15% partition function error. Prevention: implement IPD correction term or pre-compute U(T, IP_eff) tables. [CRITICAL]

4. **LTE breakdown at manifold grid edges** -- McWhirter criterion can be violated at low n_e / high T corners. Prevention: enforce McWhirter check in manifold generator; mask invalid grid points. [MODERATE]

5. **NaN propagation in batched JAX** -- One bad parameter combination poisons an entire batch silently. Prevention: clamp inputs; post-generation NaN scan; use `jax_debug_nans` during development. [MODERATE]

6. **Emissivity unit mismatch** -- Project context flags wrong unit docstring. If photon vs energy units are mixed, self-absorption gets wavelength-dependent systematic error (factor of h*nu). Prevention: audit all emissivity functions; dimensional analysis test against NIST. [MODERATE]

## Approximation Landscape

| Method | Valid Regime | Breaks Down When | Controlled? | Complements |
|--------|-------------|------------------|-------------|-------------|
| Weideman N=36 Faddeeva | All z in upper half-plane; \|z\| < 50 | float32 with large \|z\| (>50) | Yes: fixed N, error bounded | Zaghloul 2024 for >15 digit accuracy |
| Humlicek W4 Faddeeva | Forward evaluation only | Gradient computation (NaN from branch overflow) | No: region boundaries not smooth | Weideman (use instead) |
| LTE Saha-Boltzmann | T=0.5-3 eV, n_e=1e16-1e18 | n_e below McWhirter threshold; T > 5 eV (doubly-ionized species) | No: assumes LTE validity | Non-LTE models (out of scope) |
| Irwin polynomial U(T) | T=1000-16000 K (fitted range) | T > 20000 K (extrapolation diverges); IPD not captured | No: pre-fitted polynomial | Explicit level summation with IPD cutoff |
| Anderson acceleration (m=3-5) | Contractive fixed-point maps | Non-contractive regime (T > 5 eV, n_e near thresholds) | Partially: convergence theory for contractive maps | Damped Picard (fallback) |
| Self-absorption C = tau/(1-exp(-tau)) | Optically thin to moderate (tau < 3) | tau > 3: 3-5x noise amplification | No: multiplicative error growth | Line masking (exclude tau > 3) |

**Coverage gaps:** No reliable method exists for non-LTE conditions at LIBS plasma boundaries. The polynomial partition functions have no correction for IPD, creating a gap at high n_e. Both gaps have concrete mitigations (McWhirter enforcement and IPD correction terms) that should be implemented before production manifold generation.

## Theoretical Connections

### Structural Parallels
- **ExoJAX <-> CF-LIBS forward model:** Both compute batch spectra as sum of Voigt-broadened lines via JAX vmap. ExoJAX targets absorption cross-sections for exoplanets; CF-LIBS targets emission spectra for plasma. The Voigt kernel architecture transfers directly. [ESTABLISHED]
- **Anderson acceleration <-> DIIS in quantum chemistry:** Mathematically equivalent. DIIS (Direct Inversion in the Iterative Subspace) is the standard SCF accelerator in electronic structure codes. Convergence theory and practical heuristics (m=3-5, regularization, restart) transfer directly. [ESTABLISHED]

### Duality: Forward Model <-> Inversion
- The forward model f(T, n_e, C) -> spectrum is differentiable in JAX, enabling gradient-based inversion via the joint optimizer. This duality between generative model and inference is the core design principle: a well-optimized forward model automatically yields a well-optimized inversion via AD. [ESTABLISHED]

### Cross-Validation Opportunities
- Weideman vs scipy.special.wofz (Voigt accuracy)
- Anderson vs Picard (same fixed point, different convergence rate)
- GPU float64 vs CPU float64 (numerical equivalence)
- Round-trip forward-inversion (T, n_e, C recovery)
- Batched vs sequential forward model (vmap correctness)

## Critical Claim Verification

| # | Claim | Source | Verification | Result |
|---|-------|--------|-------------|--------|
| 1 | Weideman N=36 achieves ~15-digit accuracy in float64 | METHODS.md | Weideman (1994) Table 1; independently validated by ExoJAX, HELIOS-K | CONFIRMED (established result) |
| 2 | Humlicek W4 has gradient instability under JAX AD | PITFALLS.md | Known JAX behavior: jnp.where evaluates all branches during AD | CONFIRMED (documented JAX behavior) |
| 3 | V100S has 1:2 float64:float32 ratio (8.2 vs 16.4 TFLOPS) | COMPUTATIONAL.md | NVIDIA V100S data sheet | CONFIRMED (well-documented hardware spec) |
| 4 | Anderson acceleration reduces iterations by 2-5x for contractive maps | METHODS.md / COMPUTATIONAL.md | Walker & Ni (2011); standard result in numerical analysis | CONFIRMED (established theory) |
| 5 | No published GPU CF-LIBS pipeline exists | PRIOR-WORK.md | Training data consistent; ExoJAX, HELIOS-K, RADIS all target absorption spectroscopy | UNVERIFIED (web search unavailable; claim is plausible but cannot independently confirm no recent publication) |
| 6 | LIBS operates with 10^3-10^4 lines vs 10^6-10^8 in exoplanet spectroscopy | PRIOR-WORK.md | Consistent with NIST ASD line counts and LIBS literature | CONFIRMED (domain knowledge) |
| 7 | IPD correction to polynomial partition functions is not implemented | PITFALLS.md | Codebase inspection: partition.py uses Irwin polynomials without max_energy cutoff | CONFIRMED (codebase inspection) |

**Note:** Web search was unavailable during research. Claim 5 (no published GPU CF-LIBS) should be re-verified against recent JQSRT and Spectrochimica Acta B publications when web access is restored.

## Cross-Validation Matrix

|                   | Weideman JAX | scipy.wofz | Zaghloul 2024 | CPU Picard | GPU Anderson | Round-trip |
|-------------------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weideman JAX**  | -- | Full regime | Full regime | -- | -- | -- |
| **scipy.wofz**    | Full regime | -- | Full regime | -- | -- | -- |
| **GPU forward**   | -- | -- | -- | -- | -- | Recover T, n_e, C |
| **Anderson**      | -- | -- | -- | Same fixed point | -- | -- |
| **Batch vmap**    | Element-by-element | -- | -- | -- | -- | -- |

## Input Quality -> Roadmap Impact

| Input File | Quality | Affected Recommendations | Impact if Wrong |
|------------|---------|------------------------|-----------------|
| METHODS.md | Good | Method selection (Weideman, AA, normal equations), vmap strategy | Phases 1-3 may need algorithm substitution |
| PRIOR-WORK.md | Good | Benchmark targets, GPU speedup expectations, novelty claims | Success criteria may be miscalibrated |
| COMPUTATIONAL.md | Good | Batch sizes, float64 decision, FAISS configuration, V100S specifics | Resource estimates wrong; OOM at runtime |
| PITFALLS.md | Good | Risk mitigation in all phases, gradient safety, unit conventions | Blind spots in Voigt gradients, precision, or units |

## Implications for Roadmap

### Suggested Phase Structure

**Phase 1: Voigt Profile GPU Kernel and Validation**
**Rationale:** The Voigt profile is the innermost computational kernel; every downstream computation depends on it. Validating accuracy and gradients here prevents error propagation.
**Delivers:** JIT-compiled, vmap-ready Weideman-36 Voigt kernel; accuracy validation vs scipy.wofz; gradient correctness at all (T, n_e) corners; throughput benchmark on V100S.
**Methods:** Weideman N=36 rational approximation; broadcasting over (N_wl, N_lines).
**Avoids:** Pitfall 1 (Humlicek gradient instability), Pitfall 2 (float32 precision loss).
**Success criteria:** Relative error < 1e-6 (float64) vs scipy.wofz; no NaN in gradients at any test point; throughput measurement on V100S.

**Phase 2: Saha-Boltzmann Solver with Anderson Acceleration**
**Rationale:** The Saha solver is the physics core. Anderson acceleration is the highest-impact algorithmic improvement (2-4x fewer iterations). Must be correct before batch forward model.
**Delivers:** Anderson-accelerated Saha-Boltzmann solver in JAX; convergence benchmarks vs Picard; IPD correction for partition functions.
**Methods:** Anderson acceleration (m=3-5), Tikhonov regularization, Picard fallback; log-space Saha computation.
**Avoids:** Pitfall 4 (IPD not applied to polynomial U(T)), Pitfall 6 (AA divergence at extremes).
**Success criteria:** Same fixed point as Picard to < 1e-8 relative; iteration count reduction > 1.5x; IPD correction validated against explicit level summation.

**Phase 3: Vectorized Boltzmann Fitting and Softmax Closure**
**Rationale:** These are lightweight components that complete the inversion pipeline. Both have closed-form or trivial implementations. Group together for efficiency.
**Delivers:** Batched Boltzmann fitting via normal equations; softmax closure with log-sum-exp; sigma-clip outlier rejection.
**Methods:** Closed-form 2-parameter WLS; softmax via jax.nn.softmax or manual log-sum-exp.
**Avoids:** Pitfall 3 (oscillator strength unit confusion -- validate gf values during line selection).
**Success criteria:** Temperature recovery < 2% at SNR=100; sum(C_i) = 1 to machine precision; gradients correct.

**Phase 4: Batch Forward Model and Manifold Generation**
**Rationale:** Composes all previous kernels into the end-to-end forward model, vmapped over parameter batches. This is the primary GPU consumer and the manifold generator.
**Delivers:** Batch forward model; memory-optimized vmap strategy; manifold generation pipeline.
**Methods:** vmap over (T, n_e, C); lax.scan for time integration; batch padding for shape stability.
**Avoids:** Pitfall 5 (LTE boundary), Pitfall 9 (NaN propagation), Pitfall 7 (benchmark methodology).
**Success criteria:** Identical output to sequential loop (< 1e-12 relative); scales to batch=64+ on V100S; McWhirter enforcement; no NaN in manifold.

**Phase 5: FAISS Integration and End-to-End Benchmarking**
**Rationale:** The inference pipeline (query spectrum -> nearest neighbor -> retrieved parameters) completes the system. Benchmarking is the paper's core contribution.
**Delivers:** FAISS GPU index for manifold lookup; end-to-end pipeline benchmarks; 7 figures for JQSRT paper; roofline analysis.
**Methods:** FAISS IndexFlatL2 (d=30, PCA-reduced); comprehensive benchmarking with block_until_ready().
**Avoids:** Pitfall 7 (misleading GPU benchmarks from async dispatch).
**Success criteria:** Recall@1 > 95%; GPU speedup > 50x for batch >= 64; complete figure set for paper.

### Phase Ordering Rationale

- Phases are ordered by computational dependency: Voigt -> Saha -> Boltzmann/Closure -> Forward -> Benchmark.
- Each phase produces a validated kernel that the next phase composes. Errors caught early do not propagate.
- Phases 1-3 can potentially overlap (Saha solver does not depend on Boltzmann fitting), but the Voigt kernel (Phase 1) must be complete first since it is used in validation of all downstream phases.
- Phase 5 (benchmarking) is strictly last because it requires all kernels to be integrated.

### Phases Requiring Deep Investigation

- **Phase 2 (Anderson acceleration):** Novel application to Saha-Boltzmann; convergence behavior at parameter space corners needs empirical investigation. No published reference for differentiable AA in plasma physics.
- **Phase 4 (Batch forward model):** Memory budget on V100S needs empirical measurement; batch sizes from COMPUTATIONAL.md are estimates. First-call XLA compilation time may be problematic for large models.

Phases with established methodology (straightforward execution):

- **Phase 1 (Voigt profile):** Well-characterized algorithm (Weideman 1994); existing implementation needs vmap lifting and validation, not new development.
- **Phase 3 (Boltzmann/Closure):** Standard linear algebra and softmax; existing implementations in codebase.
- **Phase 5 (FAISS/Benchmarking):** Established tools (FAISS) and methodology (GPU profiling).

## Confidence Assessment

| Area | Confidence | Notes |
|------|-----------|-------|
| Methods | HIGH | All recommended methods are well-established (Weideman, normal equations, Anderson, softmax). Novel combination, not novel methods. |
| Prior Work | MEDIUM | Web search unavailable; claims about gap in literature are plausible but unverified against 2025-2026 publications. GPU speedup estimates from ExoJAX/HELIOS-K may not transfer directly to LIBS regime. |
| Computational Approaches | MEDIUM | V100S specs are well-documented, but batch sizes and memory budgets need empirical validation. Float64 decision is sound but cost multiplier is theoretical until measured. |
| Pitfalls | HIGH | Pitfalls are grounded in codebase inspection and well-documented JAX behavior. Humlicek gradient issue is a known JAX pattern. IPD gap confirmed by code review. |

**Overall confidence:** MEDIUM -- high confidence in methods and pitfalls, but empirical validation on V100S hardware is needed for batch sizes, throughput numbers, and compilation times. The novelty claim (no published GPU CF-LIBS) needs web verification.

### Gaps to Address

- **V100S memory measurement:** Actual available memory after CUDA/JAX context must be measured empirically to set batch size ceilings.
- **Float64 vs float32 accuracy comparison:** Need empirical data on accuracy loss at each pipeline stage.
- **IPD correction implementation:** No off-the-shelf solution; requires either explicit level summation or correction tables.
- **Novelty verification:** Re-run literature search for GPU CF-LIBS when web access is available.
- **Partition function upper temperature bound:** Irwin polynomials may extrapolate badly above 16000 K; need validation at T = 20000-30000 K.

## Open Questions

1. **[HIGH] What is the actual V100S memory budget after CUDA/JAX initialization?** Determines batch_size ceiling for manifold generation. Blocks Phase 4.
2. **[HIGH] Does Anderson acceleration converge for all (T, n_e) combinations in the LIBS regime?** Specifically at corners: T=0.3 eV with n_e=1e16, T=3.0 eV with n_e=5e15. Blocks Phase 2 completion.
3. **[MEDIUM] Is the crossover batch size where GPU exceeds CPU throughput < 64?** If crossover is higher, GPU acceleration may not help for real-time single-spectrum analysis.
4. **[MEDIUM] Can `jax.lax.custom_root` provide implicit gradients through the Anderson solver?** Alternative to unrolling iterations (memory-expensive). Affects gradient-based inversion performance.
5. **[LOW] Is mixed-precision (float32 profiles, float64 physics) faster than uniform float64?** The V100S 1:2 ratio makes uniform float64 viable, potentially eliminating the complexity of mixed precision.

## Sources

### Primary (HIGH)

- Weideman, J.A.C. (1994). SIAM J. Numer. Anal. 31(5), 1497-1518. -- Faddeeva rational approximation (core Voigt algorithm)
- Walker, H.F. and Ni, P. (2011). SIAM J. Numer. Anal. 49(4), 1715-1735. -- Anderson acceleration theory
- Tognoni, E. et al. (2010). Spectrochim. Acta B 65(1), 1-14. -- CF-LIBS review and methodology
- Ciucci, A. et al. (1999). Appl. Spectrosc. 53(8), 960-964. -- Original CF-LIBS procedure
- Grimm, S.L. et al. (2021). ApJS 253, 30. arXiv:2101.02005. -- HELIOS-K GPU opacity benchmark
- Johnson, J. et al. (2021). IEEE Trans. Big Data 7(3), 535-547. -- FAISS GPU algorithms

### Secondary (MEDIUM)

- Kawahara, H. et al. (2022). ApJS 258, 31. arXiv:2105.14782. -- ExoJAX GPU spectral computation
- Kawahara, H. et al. (2024). ApJS 272, 17. arXiv:2306.14619. -- ExoJAX2 improvements
- Pannier, E. and Laux, C. (2019). JQSRT 222-223, 12. -- RADIS spectral code
- Zaghloul, M.R. (2024). ACM Trans. Math. Softw. arXiv:2411.00917. -- Chebyshev Faddeeva
- Stewart, J.C. and Pyatt, K.D. (1966). ApJ 144, 1203. -- Ionization potential depression
- Cristoforetti, G. et al. (2010). Spectrochim. Acta B 65, 86-95. -- LTE validity beyond McWhirter
- Toth, A. and Kelley, C.T. (2015). SIAM J. Numer. Anal. 53(2), 805-819. -- Anderson acceleration convergence

### Tertiary (LOW)

- Evans, J. et al. (2018). arXiv:1810.08455. -- Differentiable physics simulations (AD paradigm)
- Abrarov, S.M. and Quine, B.M. (2011). Appl. Math. Comput. 218(5), 1894-1902. -- Alternative Faddeeva
- Schreier, F. (2018). JQSRT 211, 78-87. -- Voigt implementation comparison review

---

_Research analysis completed: 2026-03-23_
_Ready for research plan: yes_

```yaml
# --- ROADMAP INPUT (machine-readable, consumed by gpd-roadmapper) ---
synthesis_meta:
  project_title: "GPU-Accelerated CF-LIBS Multi-Element Plasma Diagnostics"
  synthesis_date: "2026-03-23"
  input_files: [METHODS.md, PRIOR-WORK.md, COMPUTATIONAL.md, PITFALLS.md]
  input_quality: {METHODS: good, PRIOR-WORK: good, COMPUTATIONAL: good, PITFALLS: good}

conventions:
  unit_system: "mixed CGS-eV (T in K/eV, n_e in cm^-3, lambda in nm, E in eV)"
  metric_signature: "N/A (non-relativistic)"
  fourier_convention: "N/A (direct summation)"
  coupling_convention: "N/A"
  renormalization_scheme: "N/A"

methods_ranked:
  - name: "Weideman N=36 Faddeeva"
    regime: "All z in upper half-plane, |z| < 50; LIBS y ~ 0.01-10"
    confidence: HIGH
    cost: "O(N_wl * N_lines) per spectrum; 36 multiply-adds per Voigt evaluation"
    complements: "Zaghloul 2024 Chebyshev for >15-digit accuracy validation"
  - name: "Anderson acceleration (m=3-5)"
    regime: "Contractive Saha-Boltzmann maps; T=0.5-3 eV, n_e=1e16-1e18"
    confidence: MEDIUM
    cost: "O(m^2 * N_params) per iteration; 5-8 iterations typical"
    complements: "Picard iteration (fallback for non-contractive regime)"
  - name: "Batched normal equations (2-param WLS)"
    regime: "N_lines >= 4 per element-stage; no collinear E_k"
    confidence: HIGH
    cost: "O(B * N_lines) per batch; 5 dot products per element"
    complements: "Huber M-estimation for robust refinement"
  - name: "Softmax closure (log-sum-exp)"
    regime: "2-10 elements at 1-90% concentrations"
    confidence: HIGH
    cost: "O(N_elements) per evaluation"
    complements: "ILR transform (mathematically equivalent)"
  - name: "FAISS IndexFlatL2 (d=30, PCA-reduced)"
    regime: "N < 10M manifold entries; d=30 after PCA"
    confidence: HIGH
    cost: "O(N * d) per query; batch queries 10-50x faster"
    complements: "IndexIVFFlat for N > 10M"

phase_suggestions:
  - name: "Voigt Profile GPU Kernel"
    goal: "Validated, JIT-compiled Weideman-36 Voigt kernel with correct gradients on V100S"
    methods: ["Weideman N=36 Faddeeva"]
    depends_on: []
    needs_research: false
    risk: LOW
    pitfalls: ["humlicek-gradient-instability", "float32-precision-loss"]
  - name: "Anderson-Accelerated Saha Solver"
    goal: "2-4x faster Saha-Boltzmann convergence with IPD-corrected partition functions"
    methods: ["Anderson acceleration (m=3-5)"]
    depends_on: ["Voigt Profile GPU Kernel"]
    needs_research: true
    risk: MEDIUM
    pitfalls: ["anderson-divergence-extreme-params", "ipd-partition-function-gap"]
  - name: "Boltzmann Fitting and Softmax Closure"
    goal: "Batched temperature extraction and compositional closure in JAX"
    methods: ["Batched normal equations (2-param WLS)", "Softmax closure (log-sum-exp)"]
    depends_on: ["Voigt Profile GPU Kernel"]
    needs_research: false
    risk: LOW
    pitfalls: ["oscillator-strength-unit-confusion"]
  - name: "Batch Forward Model and Manifold Generation"
    goal: "End-to-end batch forward model vmapped over parameter grid; manifold HDF5/Zarr output"
    methods: ["Weideman N=36 Faddeeva", "Anderson acceleration (m=3-5)", "Softmax closure (log-sum-exp)"]
    depends_on: ["Anderson-Accelerated Saha Solver", "Boltzmann Fitting and Softmax Closure"]
    needs_research: false
    risk: MEDIUM
    pitfalls: ["lte-boundary-violation", "nan-propagation-batched", "emissivity-unit-mismatch"]
  - name: "FAISS Integration and Benchmarking"
    goal: "Complete inference pipeline with GPU FAISS; comprehensive benchmarks for JQSRT paper"
    methods: ["FAISS IndexFlatL2 (d=30, PCA-reduced)"]
    depends_on: ["Batch Forward Model and Manifold Generation"]
    needs_research: false
    risk: LOW
    pitfalls: ["jit-warmup-benchmark-bias"]

critical_benchmarks:
  - quantity: "Voigt profile relative error vs scipy.wofz (float64)"
    value: "< 1e-6"
    source: "Weideman (1994) Table 1"
    confidence: HIGH
  - quantity: "Anderson iteration count reduction vs Picard"
    value: "> 1.5x (expect 2-4x)"
    source: "Walker & Ni (2011); project REQUIREMENTS.md"
    confidence: MEDIUM
  - quantity: "Round-trip temperature recovery at SNR=100"
    value: "< 2% relative error"
    source: "Tognoni et al. (2010); project validation suite"
    confidence: HIGH
  - quantity: "GPU vs CPU throughput for batch >= 64"
    value: "> 50x speedup"
    source: "ExoJAX, RADIS GPU benchmarks (extrapolated to LIBS scale)"
    confidence: MEDIUM
  - quantity: "FAISS recall@1 with nprobe=20"
    value: "> 95%"
    source: "Johnson et al. (2021); standard FAISS benchmark"
    confidence: HIGH

open_questions:
  - question: "What is the actual V100S memory budget after CUDA/JAX initialization?"
    priority: HIGH
    blocks_phase: "Batch Forward Model and Manifold Generation"
  - question: "Does Anderson acceleration converge for all LIBS-regime (T, n_e) combinations?"
    priority: HIGH
    blocks_phase: "Anderson-Accelerated Saha Solver"
  - question: "What is the GPU-CPU crossover batch size?"
    priority: MEDIUM
    blocks_phase: "none"
  - question: "Can jax.lax.custom_root provide implicit gradients through Anderson solver?"
    priority: MEDIUM
    blocks_phase: "none"
  - question: "Is mixed-precision faster than uniform float64 on V100S?"
    priority: LOW
    blocks_phase: "none"

contradictions_unresolved:
  - claim_a: "METHODS.md recommends Anderson depth m=3 for 1D n_e problem"
    claim_b: "COMPUTATIONAL.md recommends m=5 with Tikhonov regularization"
    source_a: "METHODS.md Method 3"
    source_b: "COMPUTATIONAL.md Anderson section"
    investigation_needed: "Empirical convergence comparison at m=2,3,5 across LIBS parameter space. Both are reasonable; optimal m depends on the conditioning of the specific Saha map."
  - claim_a: "PRIOR-WORK.md cites V100S as 16 GB HBM2"
    claim_b: "COMPUTATIONAL.md cites V100S as 32 GB HBM2"
    source_a: "PRIOR-WORK.md (and METHODS.md Method 5 memory analysis)"
    source_b: "COMPUTATIONAL.md Hardware section"
    investigation_needed: "Check actual V100S variant on vasp-01/02/03 nodes. V100S comes in both 16 GB and 32 GB variants. Batch size recommendations differ by 2x depending on which is correct."
```

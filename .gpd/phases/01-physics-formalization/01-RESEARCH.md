# Phase 1: Physics Formalization - Research

**Researched:** 2026-03-23
**Domain:** Computational plasma spectroscopy, GPU numerical methods, LaTeX derivations
**Confidence:** HIGH

## Summary

Phase 1 produces complete mathematical derivations for five GPU optimization targets: Voigt profile evaluation, batched Boltzmann fitting, Anderson-accelerated Saha-Boltzmann iteration, softmax/ILR simplex closure, and batch forward model decomposition. These derivations form the foundation for all subsequent implementation phases.

The key insight is that this phase is formalization, not discovery. All five methods are well-established in the literature; the novel contribution is their combination into a differentiable JAX pipeline for CF-LIBS. The derivations must be precise about conventions (HWHM vs FWHM, CGS vs SI, eV vs K), because convention mismatches are the primary source of implementation bugs in spectroscopy codes. Each derivation should start from the physics (plasma conditions, spectral formation) and arrive at a concrete algorithm specification with explicit input/output types and numerical considerations.

**Primary recommendation:** Derive each of the five targets as a self-contained section with: (1) physics motivation, (2) mathematical formulation, (3) algorithm specification with complexity, (4) JAX implementation notes (dtype, vmap axes, JIT constraints), and (5) validation criteria. Use the existing codebase implementations as ground truth for conventions.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| Weideman (1994) SIAM J. Numer. Anal. 31, 1497 | method | Core Voigt algorithm; N=36 coefficients already in codebase | cite, derive convergence properties | plan, execution, verification |
| Zaghloul & Le Bourlot (2024) arXiv:2411.00917 | benchmark | SOTA Voigt accuracy reference; Chebyshev subinterval approach | read, compare accuracy claims | verification only |
| Walker & Ni (2011) SIAM J. Numer. Anal. 49, 1715 | method | Anderson acceleration convergence theory | cite, adapt to Saha fixed-point | plan, execution |
| Toth & Kelley (2015) SIAM J. Numer. Anal. 53, 805 | method | Anderson acceleration convergence analysis for nonlinear problems | cite for convergence conditions | plan |
| ExoJAX (Kawahara et al. 2022, arXiv:2105.14782) | prior art | JAX GPU spectroscopy architecture; vmap pattern for Voigt | cite, compare architecture | plan, execution |
| Egozcue et al. (2003) Math. Geology 35, 279 | method | ILR transform theory for compositional data | cite for DERV-04 | plan, execution |
| `cflibs/radiation/profiles.py` | prior artifact | Existing Weideman N=36 implementation with real-arithmetic fallback | read, formalize what it implements | plan |
| `cflibs/inversion/closure.py` | prior artifact | Existing ILR + Helmert basis + softmax implementations | read, formalize | plan |
| `cflibs/manifold/generator.py` | prior artifact | Existing JAX batch forward model with vmap | read, formalize decomposition | plan |

**Missing or weak anchors:** No published reference exists for Anderson acceleration applied specifically to the Saha-Boltzmann fixed-point iteration. The derivation in DERV-03 will be a novel application of established theory (Walker & Ni 2011) to a specific physics problem.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Temperature | T [K] primary, T_eV = k_B T / e derived | T in eV only | Codebase convention |
| Electron density | n_e [cm^-3] | SI: m^-3 | Plasma spectroscopy standard |
| Wavelength | lambda [nm] | Angstrom, cm | Codebase convention |
| Energy levels | E_k [eV] | cm^-1, J | Codebase convention |
| Lorentzian width | gamma = HWHM [nm] | FWHM (= 2*gamma) | Codebase profiles.py |
| Gaussian width | sigma = std dev [nm] | FWHM (= 2.355*sigma) | Codebase profiles.py |
| Voigt argument | z = (x + i*gamma) / (sigma*sqrt(2)) | Various normalizations | Codebase profiles.py |
| Partition function | log U = sum a_n (log T_K)^n | Explicit level sum | Irwin (1981) polynomials |
| Saha constant | SAHA_CONST_CM3 = 4.829e15 cm^-3 K^-3/2 | SI version | Codebase constants.py |
| Compositions | C_i dimensionless, sum C_i = 1 | Mass fraction, mole fraction | Codebase convention |
| Faddeeva function | w(z) = exp(-z^2) erfc(-iz) | Various sign conventions | Abramowitz & Stegun |

**CRITICAL: All equations and results below use these conventions. The single most common bug in Voigt implementations is a factor-of-2 error from confusing HWHM and FWHM for the Lorentzian width. The codebase uses HWHM internally.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| V(lambda) = Re[w(z)] / (sigma * sqrt(2*pi)), z = (lambda - lambda_0 + i*gamma) / (sigma*sqrt(2)) | Voigt profile | Profiles.py, Eq. convention | DERV-01 starting point |
| w(z) = (1/sqrt(pi)) * L / (L - iz) + 2/(L - iz)^2 * sum c_n Z^n, Z = (L+iz)/(L-iz) | Weideman N=36 | Weideman (1994) Eq. 38 | DERV-01 algorithm |
| ln(I*lambda / (g_k * A_ki)) = -E_k / (k_B * T) + ln(hc * C_s * N_total / (4*pi * U_s)) | Boltzmann plot | Tognoni et al. (2010) | DERV-02 starting point |
| beta = (X^T W X)^{-1} X^T W y | Weighted least squares | Standard linear algebra | DERV-02 algorithm |
| N_{z+1}/N_z = (SAHA_CONST/n_e) * T_eV^{3/2} * (U_{z+1}/U_z) * exp(-chi_z / T_eV) | Saha equation | Codebase saha_boltzmann.py | DERV-03 starting point |
| x_{k+1} = (1 - sum alpha_i) g(x_k) + sum alpha_i g(x_{k-i}), alpha from min||F_k alpha - f_k|| | Anderson acceleration | Walker & Ni (2011) Eq. 2.1-2.2 | DERV-03 algorithm |
| C_i = exp(theta_i) / sum_j exp(theta_j) | Softmax closure | joint_optimizer.py | DERV-04 method A |
| ILR: coords = CLR(C) @ V, where V is Helmert basis | ILR transform | Egozcue et al. (2003) | DERV-04 method B |
| S(lambda) = sum_lines epsilon_line * V(lambda; lambda_0, sigma, gamma) | Forward model | Codebase SpectrumModel | DERV-05 starting point |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Rational approximation of Faddeeva function | Branch-free w(z) evaluation | DERV-01 | Weideman (1994) |
| Closed-form 2-parameter WLS | Analytic slope/intercept + covariance | DERV-02 | Any linear algebra text |
| Fixed-point iteration theory | Convergence of g(x) = x maps | DERV-03 | Kelley (1995) |
| Anderson/DIIS mixing | Accelerate fixed-point convergence | DERV-03 | Walker & Ni (2011) |
| Simplex-to-R^n bijections | Unconstrained optimization on simplex | DERV-04 | Egozcue et al. (2003) |
| JAX vmap transformation | Batch vectorization without loops | DERV-05 | JAX documentation |
| Pad-and-mask for ragged data | Handle variable line counts per element | DERV-02, DERV-05 | JAX standard pattern |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
| --- | --- | --- | --- | --- |
| Weideman N=36 rational approx | 1/N (polynomial degree) | All z with Im(z) >= 0; float64 | < 1e-13 relative in float64 | Zaghloul Chebyshev for >15 digits |
| LTE (Saha-Boltzmann) | n_e above McWhirter threshold | T = 0.5-3 eV, n_e = 1e16-1e18 cm^-3 | Uncontrolled below threshold | Non-LTE (out of scope) |
| 2-parameter Boltzmann (single T) | Deviations from single-T model | Plasma close to LTE, optically thin | R^2 diagnostic | Multi-T models |
| Anderson depth m=3-5 | Iteration history truncation | Contractive fixed-point maps | Theory: r-linear convergence | Picard fallback |

## Standard Approaches

### Approach 1: Derivation-First Formalization (RECOMMENDED)

**What:** For each of the five targets, produce a self-contained derivation document that starts from the physics, develops the mathematical formulation, specifies the algorithm, and defines validation criteria. The derivation is the deliverable -- not code.

**Why standard:** This is how computational physics papers are structured. The derivation document serves as the specification for implementation (Phase 2+) and as material for the JQSRT paper.

**Key steps for each DERV:**

1. State the physics problem and define all symbols with units
2. Derive the key equation(s) from first principles or cite authoritative sources
3. Specify the algorithm: inputs, outputs, complexity, numerical considerations
4. Identify JAX-specific constraints (static shapes, JIT-compatible control flow, vmap axes)
5. Define validation tests (known limits, analytical benchmarks, numerical tolerances)

**Known difficulties at each step:**

- Step 1: Convention conflicts between sources (HWHM/FWHM, air/vacuum wavelengths, eV/K)
- Step 3: Translating mathematical algorithms to JIT-compatible form requires awareness of XLA constraints (no Python-level loops, static array shapes)
- Step 4: JAX's functional transformation model is not obvious from the math alone

### Approach 2: Code-First with Post-Hoc Documentation (FALLBACK)

**What:** Implement first, then document the derivations based on what was actually coded.

**When to switch:** If the derivations reveal ambiguities that can only be resolved by prototyping (unlikely for this phase given the existing codebase).

**Tradeoffs:** Faster to a working implementation, but derivations may be incomplete or rationalize implementation choices rather than guiding them.

### Anti-Patterns to Avoid

- **Re-deriving the Faddeeva function from scratch:** The Weideman coefficients are pre-computed and tabulated. The derivation should explain what they are and why they work, not re-derive them.
- **Ignoring the existing codebase:** profiles.py, closure.py, and generator.py already implement most of these methods. The derivations should formalize what exists, identify gaps, and specify improvements.
- **Mixing float32/float64 analysis into the derivation:** Precision is an implementation concern. The derivation should note where precision matters but not branch on dtype.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Weideman N=36 coefficients | 36 real numbers (see profiles.py lines 412-449) | Weideman (1994) Table I | Cite and use directly; do not re-derive |
| Weideman optimal L parameter | L = sqrt(N/sqrt(2)) = 5.0454 for N=36 | Weideman (1994) Eq. 37 | Cite; already in codebase |
| 2-param WLS normal equations | beta = (X^T W X)^{-1} X^T W y, cov = (X^T W X)^{-1} | Standard textbook | Cite any linear algebra reference |
| Helmert basis matrix | V[i,j] = 1/sqrt(j(j+1)) for i<j, -j/sqrt(j(j+1)) for i=j, 0 for i>j | Egozcue et al. (2003) | Already implemented in closure.py |
| Softmax gradient | d(softmax_i)/d(theta_j) = softmax_i * (delta_ij - softmax_j) | Standard ML | Cite; JAX computes this automatically |
| Saha equation | N_{z+1}/N_z = (2 U_{z+1})/(n_e U_z) * (2pi m_e kT/h^2)^{3/2} * exp(-chi/kT) | Any plasma physics textbook | Cite; already implemented |
| Debye-Huckel IPD | delta_chi = (3/2) * e^2 / (4*pi*epsilon_0 * lambda_D), lambda_D = sqrt(epsilon_0 kT / (n_e e^2)) | Stewart & Pyatt (1966) | Already in saha_boltzmann.py |

**Key insight:** All five derivation targets use well-established methods. The novelty is in their combination and JAX formalization, not in the individual components. Re-deriving standard results wastes effort and risks introducing errors.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| Voigt FWHM approximation | f_V = 0.5346*f_L + sqrt(0.2166*f_L^2 + f_G^2) | Olivero & Longbothum (1977) | 0.02% accuracy |
| Doppler width | sigma_D = (lambda/c) * sqrt(kT/m) | Kinetic theory | Non-relativistic |
| Stark width scaling | gamma_Stark ~ w * (n_e/1e16)^alpha | Griem (1974) | LTE, moderate n_e |
| Boltzmann slope-to-T conversion | T_K = -1 / (slope * k_B_eV) where slope = d(ln(I*lambda/gA)) / dE_k | Tognoni et al. (2010) | LTE, optically thin |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| ExoJAX: Auto-differentiable spectrum model | Kawahara et al. | 2022 | JAX vmap architecture for GPU spectroscopy | vmap decomposition pattern: vmap over lines, broadcast over wavelengths |
| HELIOS-K GPU opacity calculator | Grimm et al. | 2021 | GPU line-by-line spectral computation | Memory-compute tradeoffs at scale |
| Zaghloul Chebyshev Voigt | Zaghloul & Le Bourlot | 2024 | SOTA Voigt accuracy benchmark | Use as accuracy reference only; algorithm too complex for derivation target |
| Anderson acceleration convergence | Toth & Kelley | 2015 | Rigorous convergence conditions for AA | Theorem 2.3: r-linear convergence rate |
| NAG batched least squares on GPU | Schmielau & du Toit | 2017 | GPU batched WLS implementation patterns | Tall-skinny matrix batching strategy |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| LaTeX | Standard | Derivation documents | Universal for physics/math typesetting |
| SymPy | sympy.matrices, sympy.simplify | Symbolic verification of matrix algebra | For verifying normal equations, Jacobians |
| JAX | jax, jax.numpy, jax.vmap | Target platform for all derivations | Project requirement |
| NumPy | numpy | Reference implementations, validation | CPU baseline |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| scipy.special.wofz | Voigt reference implementation | Validating Weideman derivation |
| matplotlib | Visualizing convergence, profiles | Illustrating derivation results |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Symbolic Jacobian of softmax/ILR | Seconds (SymPy) | None | Straightforward |
| Anderson convergence rate analysis | Minutes (numerical experiments) | Parameter space sweep | Restrict to LIBS-relevant regime |
| Voigt accuracy comparison | Seconds (NumPy) | None | Standard benchmark |
| Batch forward model memory estimate | Analytical calculation | None | Formula: batch * N_wl * N_lines * dtype_bytes |

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Voigt -> Gaussian limit | gamma -> 0 reduces Voigt to Gaussian | Set gamma = 0 in derivation, check V = G | Exact reduction |
| Voigt -> Lorentzian limit | sigma -> 0 reduces Voigt to Lorentzian | Set sigma = 0 in derivation, check V = L | Exact reduction |
| Boltzmann slope sign | Slope must be negative (populations decrease with energy) | Check sign convention in y = -E/(kT) + const | Slope = -1/(k_B T_eV), negative for T > 0 |
| Softmax sum-to-one | sum C_i = 1 by construction | Algebraic verification | Exact identity |
| ILR dimension reduction | D components -> D-1 coordinates | Check Helmert basis is (D, D-1) with V^T V = I | Orthonormality |
| Saha detailed balance | N_{z+1}/N_z * N_z/N_{z+1} = 1 | Verify Saha ratio is self-consistent | Tautology (sanity check) |
| Anderson reduces to Picard at m=0 | AA(0) = Picard iteration | Set m=0 in AA formula | Exact equivalence |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| Weideman vs scipy.wofz | All z with Im(z) >= 0 | Relative error < 1e-13 (float64) | Weideman (1994) Table 1 |
| Boltzmann for single element | 1 element, many lines, known T | Recovered T matches input T | Round-trip test |
| Saha at T -> 0 | T << IP | All atoms neutral: N_0/N_total -> 1 | Thermodynamic limit |
| Saha at T -> infinity | T >> IP | Fully ionized | Thermodynamic limit |
| Softmax at equal theta | All theta_i = c | C_i = 1/D for all i | Symmetry |
| Forward model single line | 1 line, known emissivity | Spectrum = emissivity * Voigt(lambda) | Definition |

### Red Flags During Computation

- If the Boltzmann slope is positive, the sign convention is wrong (swapped E_k sign or y-axis definition)
- If Anderson acceleration increases the residual for 3+ consecutive steps, the fixed-point map is likely non-contractive in this regime -- fall back to Picard
- If softmax gradients contain NaN, log-sum-exp stabilization is missing (subtract max theta before exponentiation)
- If the ILR inverse does not sum to 1.0 to machine precision, the Helmert basis is not properly orthonormalized
- If the Voigt profile has negative values at any wavelength, something is fundamentally wrong (Voigt is strictly positive)

## Common Pitfalls

### Pitfall 1: HWHM/FWHM Factor-of-2 Error in Voigt

**What goes wrong:** Literature sources use FWHM_L = 2*gamma while the codebase uses gamma (HWHM). Mixing them gives a factor-of-2 error in the Lorentzian width, which shifts the Voigt profile between Gaussian-dominated and Lorentzian-dominated regimes.

**Why it happens:** No universal convention exists. Plasma spectroscopy literature mostly uses FWHM; the Faddeeva function literature uses HWHM (or the parameter y = gamma/(sigma*sqrt(2))).

**How to avoid:** The derivation must explicitly state: "gamma is HWHM throughout. FWHM_L = 2*gamma. The Voigt parameter y = gamma / (sigma * sqrt(2))." Include a conversion table.

**Warning signs:** Voigt profiles that are too narrow or too broad compared to scipy.special.voigt_profile.

**Recovery:** Check every gamma in the pipeline against its source. The codebase convention (HWHM) is correct and consistent.

### Pitfall 2: Humlicek W4 Gradient Instability

**What goes wrong:** JAX evaluates all branches of jnp.where during autodiff. In region 4 of Humlicek W4, exp(u) with u = t*t overflows for large |z|, producing NaN gradients even though the forward pass selects a different branch.

**Why it happens:** JAX's AD traces through all code paths. The if/else structure of Humlicek W4 is mathematically correct for forward evaluation but poison for reverse-mode AD.

**How to avoid:** DERV-01 must explicitly derive and recommend the Weideman N=36 algorithm (branch-free). The Humlicek derivation should be included only as a historical comparison with an explicit "DO NOT USE FOR GRADIENTS" warning.

**Warning signs:** NaN in loss function gradients at high n_e (> 1e17.5 cm^-3).

**Recovery:** Replace Humlicek with Weideman everywhere.

### Pitfall 3: Anderson Acceleration Divergence

**What goes wrong:** Anderson acceleration can diverge if the fixed-point map is non-contractive, or if the least-squares problem for mixing coefficients is ill-conditioned.

**Why it happens:** The Saha equation's exponential dependence on T and IP can make the fixed-point map highly nonlinear at extreme parameter values (very high T, very low n_e).

**How to avoid:** DERV-03 must include: (a) Tikhonov regularization (lambda ~ 1e-6) on the AA least-squares, (b) residual monitoring with automatic fallback to Picard if residual increases for 3 consecutive steps, (c) bounding of mixing coefficients.

**Warning signs:** Oscillating or increasing residual norms during AA iteration.

**Recovery:** Fall back to damped Picard: x_{k+1} = (1-beta)*x_k + beta*g(x_k) with beta = 0.3-0.5.

### Pitfall 4: Softmax Numerical Overflow

**What goes wrong:** exp(theta_i) overflows for large theta_i, producing Inf and then NaN in the ratio.

**Why it happens:** Standard implementation computes exp(theta_i) / sum exp(theta_j) directly.

**How to avoid:** DERV-04 must derive the log-sum-exp trick: C_i = exp(theta_i - max(theta)) / sum exp(theta_j - max(theta)). Note this is already standard in JAX's jax.nn.softmax.

**Warning signs:** NaN in compositions or gradients.

**Recovery:** Use jax.nn.softmax which implements LSE stabilization.

### Pitfall 5: ILR Singularity at Zero Compositions

**What goes wrong:** ILR takes log(C_i), which is -infinity for C_i = 0. This means ILR cannot represent a composition with a truly absent element.

**Why it happens:** The simplex is an open set; its boundary (where some C_i = 0) is not in its domain.

**How to avoid:** DERV-04 must note this limitation. For LIBS with 2-10 elements all present at >= 1%, this is not a practical issue. Recommend clamping: C_i = max(C_i, epsilon) with epsilon = 1e-6 before ILR.

**Warning signs:** -Inf or NaN in ILR coordinates.

**Recovery:** Clamp compositions before transform.

## Level of Rigor

**Required for this phase:** Physicist's proof -- rigorous enough that a competent physicist can verify each step, with explicit references for non-obvious claims. Not formal theorem-proof style, but every equation must be dimensionally consistent and every approximation must state its regime of validity.

**Justification:** These are derivations for a computational physics paper (JQSRT target), not a mathematics journal. The audience expects physical reasoning and clear algorithmic specifications, not epsilon-delta proofs.

**What this means concretely:**

- Every equation must be dimensionally consistent (check units on both sides)
- Every approximation must state when it breaks down (e.g., "valid for |z| < 50 in float64")
- Numerical constants must be stated with sufficient precision and sourced
- Algorithm specifications must include complexity (O notation) and memory requirements
- Validation criteria must be quantitative (relative error < 1e-6, not "good agreement")

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| Humlicek W4 (1982) | Weideman N=36 (1994) / Zaghloul Chebyshev (2024) | 1994/2024 | Humlicek still widely used but has gradient issues under AD; Weideman is the correct choice for differentiable codes |
| Sequential Picard for Saha | Anderson acceleration | Always available but rarely applied to Saha | 2-4x iteration reduction; novel application to LIBS |
| Manual normalization for closure | Softmax / ILR parameterization | ~2010s (ML) / 2003 (compositional data analysis) | Enables gradient-based joint optimization |
| Loop-based spectrum computation | vmap-based batch computation | ~2020 (JAX maturity) | 50-200x GPU speedup for batch sizes >= 64 |

**Superseded approaches to avoid:**

- **Humlicek W4 for JAX code paths:** Still in the codebase as `_faddeeva_humlicek_jax` (deprecated). Do NOT use for any gradient-based computation. The Weideman implementation is the only safe choice.
- **RANSAC for batched Boltzmann fitting:** RANSAC requires random sampling and is not vectorizable in JAX. Sigma-clipping is fully vectorizable and appropriate for moderate outlier rates.

## Open Questions

1. **Optimal Anderson depth m for 1D vs 2D Saha problems**
   - What we know: m=3-5 is standard in the literature; m=3 for 1D n_e, m=5 for joint (n_e, T)
   - What's unclear: The optimal m depends on the specific contractivity of the Saha map, which varies across the LIBS parameter space
   - Impact on this phase: The derivation should present Anderson acceleration generically with m as a parameter; empirical optimization is Phase 2's job
   - Recommendation: Derive for general m, recommend m=3 as default with note that m should be tuned empirically

2. **Softmax vs ILR: which is better for JAX autodiff?**
   - What we know: Softmax has D parameters for D components (overparameterized by 1); ILR has D-1 parameters (minimal). Softmax gradients are well-conditioned via log-sum-exp. ILR has isometric geometry (distances in transform space equal distances on simplex).
   - What's unclear: Whether the overparameterization of softmax causes optimization issues, or whether ILR's log singularity at boundaries is problematic in practice
   - Impact on this phase: DERV-04 must derive both and compare their Jacobian structures
   - Recommendation: Derive both, recommend softmax as primary (already in codebase, simpler, JAX-native via jax.nn.softmax) with ILR as alternative for geometrically-motivated applications

3. **Memory model for nested vmap in DERV-05**
   - What we know: vmap over batch creates intermediate arrays of shape (batch, N_wl, N_lines). On V100S with 32 GB, this limits batch size.
   - What's unclear: Whether XLA's fusion passes eliminate intermediate materializations, reducing actual memory usage below the naive estimate
   - Impact on this phase: The derivation should present the naive memory formula; empirical measurement is Phase 4's job
   - Recommendation: Derive the formula: memory = batch_size * N_wl * N_lines * sizeof(dtype). Note that XLA may reduce this.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Weideman N=36 Voigt | Insufficient accuracy in extreme wings | Zaghloul Chebyshev (2024) | Moderate: more complex implementation, same interface |
| Anderson acceleration for Saha | Non-convergence at parameter corners | Damped Picard with beta=0.3 | Low: remove AA, add damping parameter |
| Softmax closure | Optimization trapped in poor local minimum | ILR transform | Low: both are already in codebase |
| Batched WLS via normal equations | Numerical instability for pathological data | Batched QR decomposition | Moderate: jnp.linalg.qr per batch element |
| Single vmap over full batch | GPU OOM | Chunked vmap via lax.scan | Low: wrap vmap body in scan over chunks |

**Decision criteria:** None of these failures are likely for this phase (derivation only). They become relevant in implementation (Phase 2+).

## Derivation-Specific Research: DERV-01 (Voigt Profile)

### Weideman N=36 Algorithm

The algorithm evaluates w(z) = exp(-z^2) * erfc(-iz) via rational approximation:

1. Compute Z = (L + iz) / (L - iz), a Mobius transform mapping the upper half-plane to the unit disk
2. Evaluate polynomial p(Z) = sum_{n=0}^{N-1} c_n Z^n using Horner's rule
3. Compute w(z) = 2p(Z) / (L - iz)^2 + (1/sqrt(pi)) / (L - iz)

This is branch-free (no conditional logic), making it ideal for GPU SIMD execution and JAX autodiff.

**Complexity:** O(N) = O(36) multiply-adds per evaluation, independent of z.

**Accuracy:** ~15 digits in float64. In float32, coefficients spanning 14 orders of magnitude effectively reduce to ~N=20, giving 0.1-1% wing errors. Use float64 on V100S.

**JAX notes:** The existing implementation in profiles.py uses jnp.polyval for the polynomial evaluation, which is JIT-compatible. The real-arithmetic fallback (_faddeeva_weideman_real_parts_jax) handles Metal backend.

### Zaghloul Chebyshev (Comparison Only)

Zaghloul & Le Bourlot (2024) use Chebyshev subinterval polynomial approximation in two variables, achieving ~1e-6 accuracy with significantly fewer operations than Weideman. However:
- The implementation requires region-based branching (bad for GPU warp divergence)
- Optimized for forward evaluation, not autodiff
- More complex to implement correctly

**Recommendation:** Cite Zaghloul as accuracy benchmark. Do not implement for the JAX pipeline.

## Derivation-Specific Research: DERV-02 (Batched Boltzmann)

### Vectorized WLS as Matrix Operation

For the 2-parameter Boltzmann fit y = a + b*E_k, the design matrix is:

X = [1, E_k_1; 1, E_k_2; ...; 1, E_k_N]  (N x 2)
W = diag(1/sigma_1^2, ..., 1/sigma_N^2)   (N x N diagonal)

Normal equations: (X^T W X) beta = X^T W y

For 2x2 system, explicit inverse:
| a11 a12 |^{-1}     1     | a22  -a12 |
| a21 a22 |    = --------- | -a21  a11 |
                  det(A)

where a11 = sum(w_i), a12 = a21 = sum(w_i * E_i), a22 = sum(w_i * E_i^2), det = a11*a22 - a12^2.

This reduces to 5 weighted dot products + 2 divisions per batch element. Trivially parallelizable via vmap.

### Handling Variable Line Counts (Pad-and-Mask)

JAX requires static array shapes. Different elements have different numbers of lines. Solution: pad all elements to max_lines, with a boolean mask array.

- Padded E_k: shape (n_elements, max_lines), padded with 0.0
- Padded y: shape (n_elements, max_lines), padded with 0.0
- Mask: shape (n_elements, max_lines), True for real lines, False for padding
- Weights: set w_i = 0 for padded entries (masked out of dot products)

This is the standard JAX pattern for ragged data and is fully JIT-compatible.

### Sigma-Clipping in JAX

Sigma-clipping is vectorizable: compute residuals, compute MAD or std, mask points beyond threshold, re-fit. This can be done via jax.lax.while_loop for JIT compatibility, or unrolled for a fixed small number of iterations (3-5).

## Derivation-Specific Research: DERV-03 (Anderson Acceleration)

### Anderson Acceleration Applied to Saha-Boltzmann

The Saha-Boltzmann iteration for n_e is a fixed-point problem: given n_e, compute ionization fractions for all elements, sum free electrons from each ionized species to get n_e_new = g(n_e).

**Fixed-point map:**
```
g(n_e) = sum_elements sum_stages z * N_z(T, n_e, C_element) * pop_fraction_z(T, n_e)
```

where pop_fraction_z depends on n_e through the Saha ratio.

**Anderson acceleration (depth m):**
Given iterates x_0, ..., x_k and residuals f_i = g(x_i) - x_i:

1. Form F_k = [f_{k-m}, ..., f_k] (residual matrix, size 1 x (m+1) for 1D n_e)
2. Solve min ||F_k * alpha||_2 subject to sum(alpha) = 1
3. Compute x_{k+1} = sum alpha_i * g(x_{k-m+i})

For the 1D case (n_e only), this is particularly simple: F_k is a row vector and the constrained least squares has a closed-form solution.

**Tikhonov regularization:** Replace min ||F_k alpha||^2 with min ||F_k alpha||^2 + lambda ||alpha||^2, lambda ~ 1e-6. This prevents ill-conditioning when residuals are nearly collinear.

**Convergence theory (Toth & Kelley 2015):** If g is Lipschitz continuous with constant L < 1, and the mixing coefficients remain bounded, then AA(m) converges r-linearly with rate at most L.

**Connection to DIIS:** Anderson acceleration is mathematically equivalent to DIIS (Direct Inversion in the Iterative Subspace) used in quantum chemistry SCF calculations. The convergence behavior and heuristics (m=3-5, restart on stagnation) transfer directly.

### JAX Implementation Considerations

The Anderson acceleration state (history of iterates and residuals) must be stored in a fixed-size buffer for JIT compatibility. Use circular buffer indexed by iteration count mod (m+1). The least-squares solve is tiny (m+1 unknowns) and can use jnp.linalg.lstsq or the explicit formula.

jax.lax.while_loop is needed for convergence-based termination. Alternative: unroll to a fixed max_iterations with early-exit via convergence flag.

## Derivation-Specific Research: DERV-04 (Softmax vs ILR)

### Softmax Parameterization

Map unconstrained theta in R^D to simplex: C_i = softmax(theta)_i = exp(theta_i) / sum_j exp(theta_j).

**Jacobian:** dC_i/dtheta_j = C_i * (delta_ij - C_j). This is a D x D matrix with rank D-1 (since sum_i dC_i/dtheta_j = 0 for all j).

**Properties:**
- Surjective but not injective (theta + c*1 maps to same C for any scalar c) -- 1 degree of gauge freedom
- Log-sum-exp stabilized: numerically robust for all theta
- JAX native: jax.nn.softmax with autodiff support
- Gradient is always well-defined (no singularities)

### ILR Parameterization

Map unconstrained coords in R^{D-1} to simplex via: C = closure(exp(coords @ V^T)), where V is the Helmert basis (D x D-1) and closure(x) = x / sum(x).

**Jacobian:** The ILR Jacobian is (D-1) x (D-1), always full rank. The isometry property means ||delta_coords|| = d_Aitchison(C, C'), preserving distances.

**Properties:**
- Bijective: exactly D-1 parameters for D-1 degrees of freedom
- Isometric: preserves Aitchison geometry of the simplex
- Singular at boundaries: log(0) = -Inf when any C_i = 0
- Requires clamping for compositions near zero

### Comparison for LIBS Application

| Property | Softmax | ILR |
| --- | --- | --- |
| Parameters | D (overparameterized) | D-1 (minimal) |
| Singularities | None | At simplex boundary (C_i = 0) |
| Gradient stability | Always well-conditioned | Ill-conditioned near boundaries |
| Metric structure | Not isometric | Isometric (Aitchison geometry) |
| JAX support | Native (jax.nn.softmax) | Manual (but straightforward) |
| Existing implementation | joint_optimizer.py | closure.py |

**Recommendation:** Softmax is the primary choice for the JAX pipeline. It has no singularities, native JAX support, and is already used in the joint optimizer. ILR is the theoretically correct choice for compositional data analysis but its boundary singularity and D-1 parameterization offer no practical advantage for LIBS with 2-10 elements at >= 1% concentration.

## Derivation-Specific Research: DERV-05 (Batch Forward Model)

### vmap Decomposition

The forward model computes a spectrum S(lambda) for given (T, n_e, C):

```
S(lambda) = sum_{lines} epsilon_line(T, n_e, C) * V(lambda - lambda_0; sigma(T), gamma(n_e))
```

For a batch of parameter sets, the vmap structure is:

1. **Outer vmap** (over batch dimension): vmap over (T, n_e, C) -- each batch element gets different plasma parameters
2. **Inner computation** (within one spectrum): broadcasting over (N_wl, N_lines) -- the emissivity * profile product

The inner computation is NOT a second vmap but a broadcast: for N_wl wavelength points and N_lines lines, the intermediate array is (N_wl, N_lines) containing the profile values, which is then multiplied by the (N_lines,) emissivity vector and summed over lines.

### Memory Model

For a single batch element:
- Profile array: N_wl * N_lines * sizeof(dtype)
- For N_wl = 10000, N_lines = 500, float64: 10000 * 500 * 8 = 40 MB
- For batch_size = 100: 4 GB

For V100S with ~28 GB available (after CUDA/JAX context):
- Max batch_size ~ 700 at (10000, 500, float64)
- Max batch_size ~ 1400 at (10000, 500, float32)
- Max batch_size ~ 2800 at (5000, 500, float32)

**Chunking strategy:** If batch_size exceeds GPU memory, use jax.lax.map (sequential) or chunk the batch manually with a Python loop over sub-batches.

### XLA Compilation Considerations

- First call compiles via XLA: 10-60 seconds depending on array sizes
- Pad the last batch to match the compiled shape (avoid recompilation)
- Use jax.jit with static_argnums for integer parameters (number of elements, max lines)

## Caveats and Alternatives

**Self-critique:**

1. **Assumption: LTE validity is given.** All five derivations assume LTE (Saha-Boltzmann equilibrium). If LTE breaks down, the entire framework is invalid. The derivations should note where McWhirter criterion enforcement is needed but do not derive non-LTE corrections.

2. **Anderson acceleration confidence is MEDIUM.** While AA is well-established (Walker & Ni 2011, Toth & Kelley 2015), its application to the Saha-Boltzmann fixed-point is novel. The contractivity of the Saha map at extreme parameters (low n_e, high T) needs empirical investigation that this phase (derivations only) cannot provide.

3. **ILR may be better than softmax in some regimes.** I recommend softmax primarily for pragmatic reasons (existing code, JAX native support). A specialist in compositional data analysis might argue that ILR's isometric property matters for gradient-based optimization accuracy. For the LIBS application with well-separated compositions (2-10 elements, > 1% each), the difference is likely negligible.

4. **Zaghloul 2024 accuracy claims not independently verified.** The claim that Zaghloul's Chebyshev approach matches or exceeds Weideman N=36 accuracy is based on the preprint. For this phase (derivation), it suffices as a reference; for implementation, an independent accuracy comparison should be performed.

5. **Memory estimates assume naive materialization.** XLA's fusion passes may significantly reduce actual GPU memory usage compared to the naive (batch * N_wl * N_lines * dtype_bytes) formula. The derivation provides conservative upper bounds; empirical measurement in Phase 4 may reveal substantially better limits.

## Sources

### Primary (HIGH confidence)

- Weideman, J.A.C. (1994). "Computation of the Complex Error Function." SIAM J. Numer. Anal. 31(5), 1497-1518. -- Core Voigt algorithm
- Walker, H.F. and Ni, P. (2011). "Anderson Acceleration for Fixed-Point Iterations." SIAM J. Numer. Anal. 49(4), 1715-1735. -- Anderson acceleration theory
- Toth, A. and Kelley, C.T. (2015). "Convergence Analysis for Anderson Acceleration." SIAM J. Numer. Anal. 53(2), 805-819. -- AA convergence conditions
- Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for Compositional Data Analysis." Mathematical Geology 35(3), 279-300. -- ILR theory
- Tognoni, E. et al. (2010). "Calibration-free laser-induced breakdown spectroscopy: State of the art." Spectrochim. Acta B 65(1), 1-14. -- CF-LIBS methodology
- JAX documentation: jax.vmap, jax.jit, jax.nn.softmax -- Target platform

### Secondary (MEDIUM confidence)

- Kawahara, H. et al. (2022). "ExoJAX: Auto-differentiable spectrum model." ApJS 258, 31. arXiv:2105.14782 -- JAX spectroscopy architecture
- Zaghloul, M.R. and Le Bourlot, J. (2024). "A highly efficient Voigt program for line profile computation." arXiv:2411.00917 -- Chebyshev Voigt reference
- Evans, J. et al. (2018). arXiv:1810.08455 -- Differentiable physics simulations
- Schmielau, T. and du Toit, J. (2017). "Batched Least Squares of Tall Skinny Matrices on GPU." NAG Technical Report -- GPU batched WLS
- Schreier, F. (2018). "The Voigt and complex error function: Humlicek's rational approximation generalized." JQSRT 211, 78-87 -- Voigt algorithm comparison

### Tertiary (LOW confidence)

- Various web resources on JAX vmap patterns for ragged data -- Implementation guidance, not physics
- "We can do better than the ALR or Softmax Transform" (statsathome.com) -- Pedagogical ILR comparison

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH - all five methods are well-established with textbook references
- Standard approaches: HIGH - derivation-first formalization is standard for computational physics papers
- Computational tools: HIGH - LaTeX + SymPy + JAX are the obvious choices
- Validation strategies: HIGH - known limits and benchmarks exist for all five derivation targets

**Research date:** 2026-03-23
**Valid until:** Indefinitely for the mathematical content. JAX API may change (check jax.vmap signature).

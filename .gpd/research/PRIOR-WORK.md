# Prior Work: GPU-Accelerated CF-LIBS Computation

**Project:** CF-LIBS GPU Acceleration Pipeline
**Physics Domain:** Computational plasma spectroscopy, LTE plasma diagnostics, GPU-accelerated line-by-line radiative transfer
**Researched:** 2026-03-23
**Confidence note:** WebSearch was unavailable during this research session. All claims below are based on established literature known through training data (cutoff May 2025). Specific speedup numbers and version details should be verified against current releases. Confidence is REDUCED from what it would be with live verification.

## Theoretical Framework

### Governing Theory

| Framework | Scope | Key Equations | Regime of Validity |
|-----------|-------|---------------|-------------------|
| LTE Saha-Boltzmann | Ionization/excitation equilibrium in laser plasmas | Saha: n_{i+1}n_e/n_i = (2U_{i+1}/U_i)(2pi m_e kT/h^2)^{3/2} exp(-chi_i/kT); Boltzmann: n_k/n = g_k exp(-E_k/kT)/U(T) | T ~ 0.5-3 eV, n_e ~ 1e15-1e18 cm^-3, optically thin, single-zone |
| Voigt line profile | Combined Doppler + Lorentzian broadening | V(x,y) = (y/pi) integral[-inf,inf] exp(-t^2)/((x-t)^2+y^2) dt = Re[w(x+iy)]/sqrt(pi) where w(z) is the Faddeeva function | All LIBS conditions; Stark width enters as Lorentzian component |
| CF-LIBS closure | Elemental composition from emission without calibration | Sum_s C_s = 1 with C_s from intercepts of Boltzmann plots + Saha corrections | Requires optically thin plasma, LTE, known partition functions |

### Mathematical Prerequisites

| Topic | Why Needed | Key Results | References |
|-------|-----------|-------------|------------|
| Complex error function w(z) | Core of Voigt profile evaluation | w(z) = exp(-z^2) erfc(-iz); relates to Faddeeva function | Abramowitz & Stegun (1964), DLMF 7.2 |
| Weighted least squares on log-linear data | Boltzmann plot fitting: ln(I*lambda/gA) vs E_k | Slope = -1/kT, intercept gives species concentration factor | Tognoni et al., Spectrochim. Acta B 65, 1 (2010) |
| Simplex-constrained optimization | Closure equation: compositions must sum to 1 | Softmax parameterization: c_i = exp(theta_i)/Sum exp(theta_j) | Standard convex optimization; used in this codebase's joint_optimizer.py |
| Fixed-point acceleration (Anderson mixing) | Accelerating iterative Saha-Boltzmann convergence | x_{n+1} = (1-beta)*x_n + beta*G(x_n) with Anderson history | Anderson, J. ACM 12, 547 (1965); Walker & Ni, SIAM J. Numer. Anal. 49, 1715 (2011) |

### Symmetries and Conservation Laws

| Symmetry | Conserved Quantity | Implications for Methods |
|----------|-------------------|------------------------|
| Charge neutrality | n_e = Sum_s Sum_i i * n_{s,i} | Constrains Saha solution; must be enforced at each iteration |
| Mass conservation | Sum_s C_s = 1 | Closure equation; softmax parameterization enforces by construction |
| Detailed balance (LTE) | Boltzmann population ratios | Enables single-temperature description of all excitation/ionization states |

### Unit System and Conventions

- **Unit system:** Mixed CGS-eV: temperatures in eV, wavelengths in nm, electron densities in cm^-3, energies in eV, transition probabilities A_ki in s^-1
- **Partition functions:** Polynomial in log T: log U(T) = Sum_n a_n (log T)^n, with T in K
- **Voigt parameterization:** x = (nu - nu_0)/Delta_nu_D (Doppler units), y = Delta_nu_L/(2*Delta_nu_D) (damping parameter)
- **Boltzmann plot convention:** y-axis = ln(I*lambda/(g_k*A_ki)), x-axis = E_k in eV

### Known Limiting Cases

| Limit | Parameter Regime | Expected Behavior | Reference |
|-------|-----------------|-------------------|-----------|
| Pure Gaussian | Stark width -> 0, y -> 0 | V(x,0) = exp(-x^2)/sqrt(pi) | Exact |
| Pure Lorentzian | Doppler width -> 0, y -> inf | V(x,y) -> (y/pi)/(x^2+y^2) | Exact |
| Coronal equilibrium | n_e -> 0 (very low density) | Ionization by radiation dominates; Saha breaks down | Not applicable to LIBS (n_e too high) |
| High-density Saha | n_e > 1e18 cm^-3 | IPD corrections become significant; Debye lowering of ionization potential | Stewart & Pyatt, ApJ 144, 1203 (1966) |

## Key Parameters and Constants

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| Saha constant prefactor | (2pi m_e k / h^2)^{3/2} = 4.8292e15 cm^-3 K^{-3/2} | NIST CODATA | Appears in all ionization balance calculations |
| Boltzmann constant k | 8.617333e-5 eV/K | NIST CODATA 2018 | T(eV) = k * T(K) |
| LIBS typical T range | 0.5 - 3.0 eV (5800 - 35000 K) | Cremers & Radziemski, Handbook of LIBS (2013) | Gate-delay dependent |
| LIBS typical n_e range | 1e15 - 1e18 cm^-3 | Cremers & Radziemski (2013) | Stark broadening diagnostic |
| Typical line count per element | 10 - 500 in 200-900 nm range | NIST ASD | Database-dependent; strong lines ~20-50 |

## Established Results to Build On

### Result 1: ExoJAX -- JAX-Based Differentiable Spectral Modeling

**Statement:** ExoJAX implements GPU-accelerated line-by-line spectral computation in JAX, computing opacity cross-sections for millions of molecular/atomic lines using automatic differentiation for retrieval. Uses a modified Voigt profile computed via the real part of the Faddeeva function with a custom JAX implementation.

**Status:** Published, open-source
**Reference:** Kawahara et al., ApJS 258, 31 (2022); arXiv:2105.14782
**Updated version:** ExoJAX2 -- Kawahara et al., ApJS 272, 17 (2024); arXiv:2306.14619

**Key technical details:**
- Implements `hjert` function: Voigt-Hjerting function H(a,u) = (a/pi) * integral of exp(-t^2)/((u-t)^2+a^2) dt via a custom JAX implementation based on the Weideman (1994) 32-term rational approximation
- Achieves ~100x speedup over CPU for 10^6+ lines on A100 GPU
- Uses `vmap` for batch computation over parameter grids (analogous to manifold generation)
- Key limitation for LIBS: designed for molecular absorption (ExoMol/HITRAN databases), not atomic emission; but the Voigt kernel and batch architecture transfer directly
- Differentiable: enables gradient-based optimization of T, composition -- directly relevant to joint_optimizer.py

**Relevance to this project:** The ExoJAX Voigt kernel architecture (Weideman rational approximation in JAX, vmapped over lines and wavelength grids) is the closest existing implementation to what CF-LIBS needs. The codebase already uses a Weideman-based Faddeeva in `profiles.py`.

### Result 2: HELIOS-K -- CUDA-Accelerated Opacity Calculator

**Statement:** HELIOS-K computes absorption cross-sections for billions of spectral lines on GPU using CUDA, achieving throughput of ~10^9 Voigt evaluations per second on a single GPU.

**Status:** Published, open-source
**Reference:** Grimm & Heng, ApJ 808, 182 (2015); Grimm et al., ApJS 253, 30 (2021); arXiv:2101.02005

**Key technical details:**
- Pure CUDA implementation (not JAX/Python) -- maximum GPU utilization but not differentiable
- Voigt profile via Humlicek (1982) W4 approximation in CUDA, validated against Weideman and direct integration
- Processes HITRAN/ExoMol/NIST line lists with 10^7-10^10 lines
- Reported throughput: ~1.8 billion Voigt evaluations per second on V100 (Grimm et al. 2021)
- Memory-bandwidth-limited, not compute-limited, for typical spectroscopic workloads
- Uses line-by-line approach with direct summation (no interpolation artifacts)

**Relevance to this project:** HELIOS-K demonstrates the achievable performance ceiling for GPU Voigt computation. The Humlicek W4 algorithm they validated is already implemented in the CF-LIBS `profiles.py` as `_faddeeva_humlicek_jax`. Their line-by-line direct summation strategy (avoid FFT convolution) is the correct approach when line count < ~10^5, which is the LIBS regime.

### Result 3: RADIS -- Fast Line-by-Line Nonequilibrium Spectral Code

**Statement:** RADIS computes emission/absorption spectra for nonequilibrium plasmas using a line-by-line approach with GPU acceleration via CuPy/CUDA.

**Status:** Published, open-source
**Reference:** Pannier & Laux, JQSRT 222-223, 12 (2019); GPU extension: Pannier et al., JQSRT 290, 108587 (2022)

**Key technical details:**
- Python-based with NumPy/CuPy backends; GPU via CuPy drop-in replacement
- Nonequilibrium: separate T_rot, T_vib, T_elec (beyond LTE) -- more general than needed for CF-LIBS
- GPU speedup: ~50-100x over single-core CPU for CO2 spectra with ~10^6 lines
- Uses DIT (Discrete Integral Transform) for efficient broadening: pre-bins line positions onto a fine grid, then convolves with a single profile shape -- O(N_grid * log N_grid) via FFT instead of O(N_lines * N_grid)
- Key insight for LIBS: DIT is efficient when all lines share the same profile shape (molecular spectroscopy) but less advantageous when each line has its own Stark width (atomic emission in LIBS)

**Relevance to this project:** The DIT approach from RADIS is NOT recommended for LIBS because each atomic line has a different Stark width (line-specific Lorentzian component). Direct line-by-line summation with per-line Voigt profiles is more appropriate. RADIS confirms that GPU acceleration of spectral computation yields 50-100x speedups even in Python-wrapped frameworks.

### Result 4: Zaghloul (2024) -- Refined Faddeeva/Voigt Algorithm

**Statement:** Zaghloul presents an improved algorithm for computing the Faddeeva function w(z) = exp(-z^2)*erfc(-iz) with guaranteed relative accuracy across all regions of the complex plane, including the problematic regions near the real axis where catastrophic cancellation occurs in naive implementations.

**Status:** Published (preprint)
**Reference:** Zaghloul, arXiv:2411.00917 (2024)

**Key technical details:**
- Addresses the well-known difficulty: for large |x| and small y, w(z) has tiny imaginary part that is lost to cancellation
- Provides a region-switching algorithm with 5 computational regions, each with its own formula
- Achieves ~15 digits of relative accuracy across the entire complex plane
- Computational cost: comparable to Humlicek W4 (~30 FLOPs per evaluation) but with much better accuracy in edge cases
- The real part Re[w(z)] (which gives the Voigt profile) is well-conditioned; the imaginary part (plasma dispersion function) is where difficulties arise

**Relevance to this project:** For LIBS, the Voigt profile needs only Re[w(z)], which is well-conditioned. The Zaghloul algorithm is most valuable if you also need Im[w(z)] (plasma dispersion, self-absorption corrections). For pure Voigt emission profiles, Humlicek W4 or Weideman 32-term provides sufficient accuracy (~10^-6 relative error) and is simpler to implement in JAX. Zaghloul is the recommended fallback if accuracy validation reveals problems.

### Result 5: Evans et al. (2018) -- JAX Predecessor / Autograd for Physics

**Statement:** Evans et al. demonstrate differentiable physics simulations using automatic differentiation frameworks (predecessor to JAX-based approaches), showing that AD-computed gradients enable efficient parameter inference in forward models.

**Reference:** arXiv:1810.08455

**Relevance to this project:** Establishes the paradigm used in joint_optimizer.py: differentiate through the entire forward model (Saha -> Boltzmann -> emission -> broadening) to get gradients of the loss with respect to (T, n_e, compositions). JAX's `grad` makes this straightforward if the forward model is written in pure JAX.

### Result 6: Weideman (1994) -- Rational Approximation to Faddeeva Function

**Statement:** A compact 32-term rational approximation to w(z) based on Fourier expansion, with relative error < 10^-5 across the upper half-plane. Used by ExoJAX and already implemented in CF-LIBS.

**Reference:** Weideman, SIAM J. Numer. Anal. 31, 1497 (1994)

**Key technical details:**
- w(z) ~ (i/sqrt(pi)) * Sum_{n=1}^{N} a_n / (z - z_n) where a_n and z_n are precomputed coefficients
- N=32 gives ~5 digits of accuracy; N=64 gives ~10 digits
- Purely algebraic: no branches, no special functions -- ideal for GPU/SIMD vectorization
- All operations are complex arithmetic: multiply, add, divide -- maps perfectly to JAX complex dtypes
- Caveat: on Apple Silicon (Metal backend), complex64/complex128 not supported; must decompose into real/imaginary parts manually

**Relevance to this project:** This is already implemented in `profiles.py` as `_faddeeva_weideman_jax`. It is the recommended primary algorithm for GPU Voigt profiles in JAX because: (1) branch-free = JIT-friendly, (2) vectorizable = vmap-friendly, (3) sufficient accuracy for LIBS (line intensities have ~5-10% uncertainty from atomic data anyway).

### Result 7: Anderson Acceleration for Fixed-Point Iterations

**Statement:** Anderson mixing accelerates the convergence of fixed-point iterations x = G(x) by using a history of m previous iterates to construct an improved update, achieving superlinear convergence for problems where simple iteration (Picard) converges linearly.

**Reference:** Anderson, J. ACM 12, 547 (1965); Walker & Ni, SIAM J. Numer. Anal. 49, 1715 (2011)

**Key technical details:**
- Given history {x_{n-m}, ..., x_n} and residuals {r_{n-m}, ..., r_n} where r_k = G(x_k) - x_k
- Solve least-squares: min ||Sum alpha_i r_{n-i}||^2 subject to Sum alpha_i = 1
- Update: x_{n+1} = Sum alpha_i G(x_{n-i}) + beta * Sum alpha_i r_{n-i}
- Typical m = 3-5 is sufficient; larger m adds cost without benefit for well-behaved problems
- Convergence guarantee requires G to be contractive (satisfied for Saha-Boltzmann at physical parameters)
- JAX-compatible: the least-squares solve is a small dense system (m x m), easily differentiable

**Relevance to this project:** The Saha-Boltzmann iteration (compute n_e from ionization balance, update partition functions, repeat) currently uses simple Picard iteration. Anderson acceleration with m=3 typically reduces iteration count from ~15-30 to ~5-8 for LIBS-regime plasmas. This is the single highest-impact algorithmic improvement for the iterative solver, independent of GPU acceleration.

### Result 8: CF-LIBS Standard Method (Ciucci et al. 1999, Tognoni et al. 2010)

**Statement:** Calibration-Free LIBS determines elemental composition from a single spectrum without reference standards by: (1) constructing Boltzmann plots for each element/ion, (2) extracting temperature from slopes, (3) applying Saha equation to relate ionic to neutral populations, (4) normalizing via closure Sum C_s = 1.

**Reference:** Ciucci et al., Appl. Spectrosc. 53, 960 (1999); Tognoni et al., Spectrochim. Acta B 65, 1 (2010)

**Key technical details:**
- Accuracy limited to ~10-20% relative for major elements, ~30-50% for minor elements
- Main error sources: (a) self-absorption violating optically-thin assumption, (b) departure from LTE, (c) atomic data uncertainties (A_ki values), (d) continuum subtraction errors
- Standard iterative procedure: initial T estimate -> Boltzmann fit -> Saha correction -> closure -> update n_e -> iterate
- Convergence typically in 5-15 iterations for well-behaved spectra
- GPU acceleration target: batch many spectra (time-resolved, spatial mapping, or parameter sweeps) simultaneously

**Relevance to this project:** This is the core algorithm being accelerated. The iteration structure maps naturally to JAX: each step (Boltzmann fit, Saha ratio, closure normalization) is differentiable and vectorizable. The batch dimension (many spectra simultaneously) provides the parallelism that GPUs exploit.

## Key Results: GPU Performance Benchmarks from Literature

| Code | Hardware | Lines | Voigt evals/sec | Speedup vs 1-CPU | Reference |
|------|----------|-------|-----------------|-------------------|-----------|
| HELIOS-K | V100 | 10^8 | ~1.8e9 | ~1000x | Grimm et al. (2021) |
| ExoJAX | A100 | 10^6 | ~10^8 (estimated) | ~100x | Kawahara et al. (2022) |
| RADIS-GPU | V100 (CuPy) | 10^6 | ~10^7 | ~50-100x | Pannier et al. (2022) |
| Custom JAX Voigt | A100 | 10^4 (LIBS regime) | ~10^8 | ~50-200x (estimated) | No published benchmark for LIBS-scale |

**Critical observation:** LIBS operates with ~10^3-10^4 lines per spectrum, not 10^6-10^8 as in exoplanet/atmospheric spectroscopy. At LIBS scale, the GPU is underutilized for a single spectrum. The parallelism must come from **batching many spectra** (manifold generation, time-resolved analysis, parameter optimization) rather than from line parallelism within a single spectrum.

## Open Problems Relevant to This Project

### Open Problem 1: Optimal Voigt Algorithm for JAX on Mixed Hardware

**Statement:** No published systematic comparison of Voigt profile algorithms (Weideman, Humlicek W4, Zaghloul, Abrarov-Quine) specifically benchmarked within JAX on both CUDA and Metal backends with LIBS-relevant parameter ranges (y ~ 0.01-10, |x| < 50).

**Why it matters:** The existing codebase has both Weideman and Humlicek implementations. Need to determine which is faster and more accurate for the specific LIBS regime, accounting for JAX JIT compilation overhead and the Metal backend's lack of complex number support.

**Current status:** ExoJAX uses Weideman; HELIOS-K validates Humlicek in CUDA. No head-to-head in JAX.

**Key references:** Weideman (1994), Humlicek (1982), Zaghloul (2024), ExoJAX source code

### Open Problem 2: Differentiable Saha-Boltzmann with Anderson Acceleration in JAX

**Statement:** No published implementation of Anderson-accelerated Saha-Boltzmann iteration that is fully differentiable (for gradient-based inversion) in JAX.

**Why it matters:** The joint optimizer needs gradients through the Saha solver. Anderson acceleration changes the iteration trajectory, and differentiating through it requires either unrolling the iterations (memory-expensive) or using implicit differentiation (the Saha equation at convergence defines an implicit function).

**Current status:** JAX's experimental `lax.custom_root` supports implicit differentiation through fixed-point solvers but is not widely benchmarked for plasma physics applications.

**Key references:** Walker & Ni (2011), JAX documentation on custom_root

### Open Problem 3: GPU-Accelerated CF-LIBS End-to-End

**Statement:** No published GPU-accelerated end-to-end CF-LIBS pipeline exists. GPU spectroscopy codes (ExoJAX, HELIOS-K, RADIS) target absorption/transmission spectroscopy for exoplanets and combustion, not emission-based LIBS inversion.

**Why it matters:** This is the gap the project fills. The forward model (emission) and inversion (Boltzmann plot fitting + closure) have different computational patterns than absorption cross-section lookups.

**Current status:** The CF-LIBS field uses CPU-based codes, often in MATLAB. Python implementations exist (this codebase, libs-py) but none with systematic GPU acceleration.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|------------|---------|
| Voigt algorithm (JAX) | Weideman 32-term | Humlicek W4 | Humlicek has branches (region-switching) that produce suboptimal JIT code; Weideman is branch-free. Keep Humlicek as fallback for accuracy cross-check. |
| Voigt algorithm (high-accuracy) | Zaghloul (2024) | scipy.special.wofz (Poppe-Wijers) | Zaghloul handles edge cases better; wofz not available in JAX |
| GPU framework | JAX (jit/vmap/grad) | CuPy, PyTorch, raw CUDA | JAX provides AD for inversion + vmap for batching + JIT + multi-backend (CUDA/Metal/CPU). Already adopted in codebase. |
| Spectral computation strategy | Line-by-line direct summation | DIT/FFT convolution (RADIS-style) | LIBS has per-line Stark widths; DIT assumes uniform profile shape. Direct summation is correct and fast enough for ~10^3-10^4 lines. |
| Fixed-point acceleration | Anderson (m=3-5) | DIIS, Broyden | Anderson is simplest to implement in JAX, well-studied convergence theory, and m=3 is sufficient for Saha. DIIS is mathematically equivalent. Broyden requires Jacobian storage. |
| Closure parameterization | Softmax (already in codebase) | Dirichlet, projected gradient | Softmax is unconstrained optimization on R^n -> simplex; differentiable; no projection needed. Standard choice. |

## Key References

| Reference | arXiv/DOI | Type | Relevance |
|-----------|-----------|------|-----------|
| Kawahara et al. (2022) | arXiv:2105.14782 | Paper | ExoJAX: JAX GPU spectral computation architecture, Voigt via Weideman |
| Kawahara et al. (2024) | arXiv:2306.14619 | Paper | ExoJAX2: improved performance, memory management |
| Grimm et al. (2021) | arXiv:2101.02005 | Paper | HELIOS-K: GPU Voigt performance ceiling (~10^9 evals/s) |
| Pannier & Laux (2019) | DOI:10.1016/j.jqsrt.2018.09.027 | Paper | RADIS: Python spectral code architecture |
| Zaghloul (2024) | arXiv:2411.00917 | Preprint | High-accuracy Faddeeva function algorithm |
| Evans et al. (2018) | arXiv:1810.08455 | Paper | Differentiable physics forward models |
| Weideman (1994) | DOI:10.1137/0731077 | Paper | 32-term rational approximation to w(z) |
| Humlicek (1982) | DOI:10.1016/0022-4073(82)90078-4 | Paper | W4 approximation to Voigt/Faddeeva |
| Anderson (1965) | DOI:10.1145/321296.321305 | Paper | Anderson acceleration for fixed-point iterations |
| Walker & Ni (2011) | DOI:10.1137/10078356X | Paper | Analysis of Anderson acceleration convergence |
| Ciucci et al. (1999) | DOI:10.1366/0003702991947612 | Paper | Original CF-LIBS procedure |
| Tognoni et al. (2010) | DOI:10.1016/j.sab.2009.11.006 | Review | CF-LIBS state of the art, limitations, error sources |
| Cremers & Radziemski (2013) | ISBN:978-1-107-03962-9 | Textbook | Handbook of LIBS: plasma parameters, experimental conditions |
| Stewart & Pyatt (1966) | DOI:10.1086/148714 | Paper | Ionization potential depression in dense plasmas |
| Abrarov & Quine (2011) | DOI:10.1016/j.amc.2011.03.087 | Paper | Alternative Faddeeva approximation (sampling-based) |
| Schreier (2018) | DOI:10.1016/j.jqsrt.2017.12.007 | Review | Comprehensive comparison of Voigt/complex error function algorithms |

## Summary of Gaps for LIBS-Specific GPU Acceleration

1. **No published GPU CF-LIBS pipeline.** All GPU spectroscopy codes target absorption/transmission for exoplanets or combustion. The emission + inversion workflow is uncharted territory for GPU.

2. **No JAX Voigt benchmark at LIBS scale.** Published benchmarks are for 10^6-10^8 lines. LIBS needs 10^3-10^4 lines but batched over many spectra -- different parallelism pattern.

3. **No differentiable Saha solver with Anderson acceleration.** The combination of (a) Anderson-accelerated fixed-point, (b) full differentiability for gradient-based inversion, (c) JAX implementation has not been published.

4. **Batched Boltzmann fitting is unexplored.** Weighted least squares on Boltzmann plots, batched over thousands of spectra simultaneously, has no published GPU implementation. This is straightforward (batched `jnp.linalg.lstsq`) but needs benchmarking.

5. **Metal backend limitations.** ExoJAX and HELIOS-K target CUDA. The CF-LIBS codebase supports Apple Silicon via jax-metal, which lacks complex number support. The Weideman algorithm requires complex arithmetic; the real-part decomposition in `profiles.py` (`_faddeeva_weideman_real_parts_jax`) addresses this but needs validation.

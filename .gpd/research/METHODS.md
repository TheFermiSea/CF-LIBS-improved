# Computational and Analytical Methods

**Project:** GPU-Accelerated CF-LIBS Multi-Element Plasma Diagnostics
**Physics Domain:** Laser-Induced Breakdown Spectroscopy, LTE Plasma Physics, GPU Numerical Methods
**Researched:** 2026-03-23

### Scope Boundary

METHODS.md covers analytical and numerical PHYSICS methods for five GPU kernels: Voigt profile evaluation, vectorized Boltzmann fitting, Anderson-accelerated Saha-Boltzmann solving, softmax closure, and batch forward modeling. Software tools and libraries are in COMPUTATIONAL.md.

---

## Recommended Methods

### Primary Analytical Methods

| Method | Purpose | Applicability | Limitations |
|--------|---------|---------------|-------------|
| Weideman 1994 rational approximation (N=36) | Faddeeva w(z) for Voigt profiles | All z in upper half-plane; branch-free | ~15 digits float64; 36 multiply-adds per evaluation |
| Humlicek W4 region-based rational approx | Faddeeva w(z) for forward-only Voigt | Forward model where gradients not needed | ~1e-4 relative error; gradient instability at region boundaries under JAX autodiff |
| Boltzmann population ln(I*lambda/gA) vs E_k | Temperature extraction from emission spectra | LTE plasmas, optically thin lines | Assumes thermalized populations; self-absorption breaks linearity |
| Saha-Eggert ionization balance | Ionization stage populations | LTE plasmas with n_e > 1e15 cm^-3 | Assumes complete LTE; breaks for transient/non-LTE |
| Anderson acceleration (Type I) | Fixed-point convergence for Saha-Boltzmann coupling | Contractive maps with moderate condition | Diverges if Jacobian spectral radius > 1; needs safeguarding |
| Softmax reparameterization | Compositional closure sum(C_i) = 1 | Unconstrained optimization on simplex | Numerically fragile without log-sum-exp trick; Jacobian singular at boundaries |

### Primary Numerical Methods

| Method | Purpose | Convergence | Cost Scaling | Implementation |
|--------|---------|-------------|--------------|----------------|
| Weideman-36 Faddeeva | Voigt profile kernel | Machine precision (float64) | O(N_lines * N_wavelengths) | Existing in profiles.py; needs vmap lifting |
| Batched weighted least squares via QR | Boltzmann temperature fitting | Exact (direct solve) | O(B * N_lines * 2) per batch | From scratch using jnp.linalg.lstsq or manual QR |
| Anderson acceleration (depth m=3-5) | Saha-Boltzmann self-consistent iteration | Superlinear (up to quadratic for smooth maps) | O(m^2 * N_params) per iteration | From scratch; ~50 lines core logic |
| Log-sum-exp softmax | Compositional closure | N/A (reparameterization) | O(N_elements) | From scratch; trivial |
| vmap batch forward model | Manifold generation over parameter grid | N/A (embarrassingly parallel) | O(B * N_lines * N_wavelengths) | Refactor existing ManifoldGenerator |

### Computational Tools

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| JAX | >=0.4.20 | GPU acceleration, autodiff, vmap/jit | Already in codebase; sole GPU framework per scope |
| NumPy | >=1.24 | CPU reference implementations | Already in codebase |
| SciPy | >=1.11 | wofz reference for Voigt validation, L-BFGS-B | Already in codebase |
| FAISS-GPU | >=1.7.4 | Manifold nearest-neighbor lookup | Standard for billion-scale similarity search |

### Supporting Libraries

| Library | Language | Purpose | When to Use |
|---------|----------|---------|-------------|
| jax.lax.scan | Python/XLA | Sequential time integration in batch forward | Cooling-plasma time integration |
| jax.scipy.optimize.minimize | Python/XLA | L-BFGS-B for joint optimization | Joint T, n_e, concentration optimization |
| jax.numpy.linalg | Python/XLA | Batched QR/lstsq for Boltzmann fitting | Boltzmann slope extraction |
| faiss.GpuIndexIVFFlat | Python/C++ | GPU-accelerated approximate nearest neighbor | Manifold lookup at inference time |

---

## Method Details

### Method 1: Voigt Profile -- Faddeeva Function Approximations

**What:** The Voigt profile V(x, y) = Re[w(z)] / (sigma * sqrt(2*pi)) where w(z) is the Faddeeva function and z = (x + i*gamma) / (sigma * sqrt(2)). Three candidate algorithms exist; use Weideman-36 for gradient-requiring paths and Humlicek W4 for forward-only batch generation.

**Mathematical basis:**

The Faddeeva function w(z) = exp(-z^2) * erfc(-iz) for Im(z) >= 0. Three approximation strategies:

1. **Weideman 1994 rational approximation (N=36):** Uses a Moebius transform Z = (L + iz)/(L - iz) mapping the upper half-plane to the unit disk, then evaluates a polynomial in Z. The optimal parameter L = sqrt(N/sqrt(2)) = 5.045 for N=36 provides ~15-digit accuracy in float64. This is a SINGLE rational expression evaluated identically for all z -- no branches.

   ```
   w(z) = 2*P(Z) / (L - iz)^2 + (1/sqrt(pi)) / (L - iz)
   where P(Z) = sum_{n=0}^{N-1} a_n * Z^n
   ```

   **GPU suitability:** Excellent. No conditional branches means uniform warp execution. All threads compute the same sequence of multiply-adds regardless of input. Already implemented in `profiles.py` as `_faddeeva_weideman_complex_jax` and `_faddeeva_weideman_real_parts_jax` (for Metal backends lacking complex support).

2. **Humlicek W4 (1982) four-region rational approximation:** Partitions the (x,y) plane into four regions based on s = |x| + y, using different rational approximants in each. Region 1 (s >= 15): 1st-order asymptotic. Region 2 (5.5 <= s < 15): Padé-like. Region 3 (s < 5.5, y >= 0.195|x| - 0.176): 4th/5th-order rational. Region 4 (remainder): exponential + rational correction.

   **Accuracy:** ~1e-4 relative error across all regions (sufficient for forward model, NOT for gradient-based optimization).

   **GPU suitability:** The `jnp.where` branching evaluates ALL four regions and selects -- no warp divergence per se, but 4x redundant computation. Already implemented in `generator.py` inline and `profiles.py` as `_faddeeva_humlicek_jax` (deprecated for gradient use).

3. **Zaghloul 2024 Chebyshev approximation:** Uses Chebyshev polynomial expansion on a transformed variable, with region partitioning similar to Humlicek but higher-order polynomials providing ~1e-13 accuracy. Published as Algorithm 1032 in ACM TOMS.

   **GPU suitability:** Good accuracy/speed tradeoff for float64 paths. More complex to implement than Weideman-36. Use ONLY if Weideman-36 accuracy proves insufficient in specific parameter regimes (unlikely for LIBS T ~ 0.5-3 eV, n_e ~ 1e15-1e18 cm^-3).

**Recommendation:** Use Weideman-36 as the single Voigt kernel for all paths (forward model AND gradient-based inversion). It is already implemented, branch-free, autodiff-compatible, and achieves machine precision in float64. The existing Humlicek W4 in the manifold generator should be replaced with a call to the Weideman kernel to eliminate code duplication and ensure gradient correctness.

**Convergence:** Weideman-36 is not iterative; accuracy is fixed by the number of terms. N=36 gives ~15 digits in float64, ~7 digits in float32.

**Known failure modes:**
- Float32 precision loss for large |z| (>50): mitigated by the inherent boundedness of LIBS parameter space where |z| < 20 typically.
- The real-arithmetic fallback for Metal/non-complex backends doubles the FLOP count but is functionally equivalent.

**Benchmarks:** The existing codebase tests Weideman against scipy.special.wofz. For GPU, the key metric is throughput: expect ~10^9 Voigt evaluations/second on V100S with vmap over (N_wavelengths, N_lines) outer product.

**Implementation notes:**

```python
# Recommended vmap pattern for Voigt spectrum computation:
# Materialize the (N_wl, N_lines) outer product via broadcasting,
# then reduce over lines dimension.

@jit
def voigt_spectrum(wl_grid, line_wl, line_intensity, sigma, gamma):
    diff = wl_grid[:, None] - line_wl[None, :]      # (N_wl, N_lines)
    profile = voigt_kernel(diff, sigma, gamma)        # (N_wl, N_lines)
    return jnp.sum(line_intensity * profile, axis=1)  # (N_wl,)
```

The broadcasting approach is preferred over `vmap` over individual lines because it materializes a single large matmul-like operation that maps efficiently to GPU GEMM units. For V100S with 16 GB HBM2, the (N_wl, N_lines) matrix fits comfortably for N_wl ~ 10000 and N_lines ~ 5000.

**References:**
- Weideman, J.A.C. (1994). "Computation of the Complex Error Function." SIAM J. Numer. Anal. 31(5), 1497-1518. DOI: 10.1137/0731077
- Humlicek, J. (1982). "Optimized computation of the Voigt and complex probability functions." JQSRT 27(4), 437-444. DOI: 10.1016/0022-4073(82)90078-4
- Zaghloul, M.R. (2024). "Algorithm 1032: Computing the Faddeeva Function and Related Functions." ACM Trans. Math. Softw. DOI: 10.1145/3604536
- Schreier, F. (2018). "Optimized implementations of rational approximations for the Voigt and complex error function." JQSRT 211, 78-87. DOI: 10.1016/j.jqsrt.2018.02.032

---

### Method 2: Batched Weighted Least-Squares for Boltzmann Fitting

**What:** Extract temperature from the Boltzmann plot y = ln(I*lambda/(g*A)) vs x = E_k/kT. The slope is -1/T_eV and intercept encodes column density. For GPU batching, solve B independent weighted linear regressions simultaneously.

**Mathematical basis:**

The weighted least-squares problem for each spectrum:
```
minimize sum_i w_i * (y_i - a - b*x_i)^2
```
where w_i = 1/sigma_y_i^2 and b = -1/T_eV. This is a 2-parameter linear regression with the normal equations:

```
[sum(w)      sum(w*x)  ] [a]   [sum(w*y)  ]
[sum(w*x)    sum(w*x^2)] [b] = [sum(w*x*y)]
```

For B batched problems with varying numbers of lines, use padding + masking:

```python
@jit
def batched_boltzmann_fit(x, y, w, mask):
    # x: (B, N_max), y: (B, N_max), w: (B, N_max), mask: (B, N_max) bool
    w_masked = w * mask
    Sw   = jnp.sum(w_masked, axis=1)              # (B,)
    Swx  = jnp.sum(w_masked * x, axis=1)          # (B,)
    Swy  = jnp.sum(w_masked * y, axis=1)          # (B,)
    Swxx = jnp.sum(w_masked * x * x, axis=1)      # (B,)
    Swxy = jnp.sum(w_masked * x * y, axis=1)      # (B,)
    det  = Sw * Swxx - Swx * Swx                   # (B,)
    slope     = (Sw * Swxy - Swx * Swy) / det      # (B,)
    intercept = (Swxx * Swy - Swx * Swxy) / det    # (B,)
    T_eV = -1.0 / slope                            # (B,)
    return T_eV, intercept
```

**Why direct normal equations over QR:** For a 2-parameter fit, the normal equations are numerically stable (the 2x2 Gram matrix condition number is bounded by the data spread in E_k). QR is overkill here. The closed-form solution maps to 5 dot products + 2 scalar divisions per batch element -- trivially parallel.

**Convergence:** Exact in one pass (linear regression). No iteration needed.

**Known failure modes:**
- Too few lines (N < 3): underdetermined with no residual for uncertainty estimation. Require N >= 4 per element per ionization stage.
- Collinear E_k values: degenerate Gram matrix. Unlikely in practice because LIBS lines span 0-10 eV.
- Self-absorbed lines: violate the ln(I*lambda/gA) linearity assumption. Use outlier rejection (sigma-clip or Huber) on residuals.

**GPU pattern:** Use `vmap` over the batch dimension B. Each element of the batch is one element-stage pair from one spectrum. For a typical 5-element, 2-stage system with 100 spectra, B = 1000.

**Alternatives considered:**
- `jnp.linalg.lstsq`: General-purpose but allocates workspace; slower than closed-form for 2 parameters.
- RANSAC on GPU: Not straightforward in JAX due to random sampling + early termination. Use sigma-clip instead (fully vectorizable).
- Huber M-estimation: Requires iteratively reweighted least squares (IRLS), adding 5-10 iterations. Use for the final refinement pass, not the batch sweep.

**References:**
- Aragn, C. and Aguilera, J.A. (2008). "Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods." Spectrochim. Acta B 63(9), 893-916.
- Tognoni, E. et al. (2010). "Calibration-Free Laser-Induced Breakdown Spectroscopy: State of the art." Spectrochim. Acta B 65(1), 1-14.

---

### Method 3: Anderson Acceleration for Saha-Boltzmann Fixed-Point Iteration

**What:** The Saha-Boltzmann system is a fixed-point problem: given (T, n_e, C_i), solve ionization balance -> level populations -> charge balance -> update n_e. The standard approach is simple (Picard) iteration, which converges linearly. Anderson acceleration (AA) uses a history of m previous iterates to extrapolate toward the fixed point, achieving superlinear convergence.

**Mathematical basis:**

Given a fixed-point map g(x) = x (where x = n_e or the full state vector), define the residual r(x) = g(x) - x. Anderson acceleration at step k with depth m:

1. Collect m+1 most recent residuals: R_k = [r_{k-m}, ..., r_k]
2. Solve the constrained least-squares problem:
   ```
   minimize ||R_k * alpha||_2  subject to  sum(alpha_i) = 1
   ```
3. Update: x_{k+1} = sum(alpha_i * g(x_{k-i}))

Equivalently, with the unconstrained formulation (Walker & Ni 2011):

```
Delta_R = [r_k - r_{k-1}, ..., r_k - r_{k-m}]    # (n, m) matrix
gamma = argmin ||r_k - Delta_R * gamma||_2          # solve via QR
x_{k+1} = g(x_k) - (Delta_G + Delta_R) * gamma
```
where Delta_G contains the corresponding differences in g(x) values.

**Implementation for Saha-Boltzmann:**

The state vector is x = [log(n_e)]. For multi-element, extend to x = [log(n_e), log(T)] if jointly iterating. The fixed-point map:

1. Given n_e, T, C_i: compute Saha ratios for each element
2. Sum electron contributions from all ionization stages: n_e_new = sum_s (z_s * C_s * n_total * f_s(T, n_e))
3. g(log_ne) = log(n_e_new)

**Key implementation parameters:**
- **Mixing depth m = 3-5:** Deeper history gives faster convergence but costs O(m^2) per step for the least-squares solve. m=3 is optimal for the 1D n_e problem; m=5 for the 2D (n_e, T) case.
- **Damping/relaxation beta = 1.0:** Full Anderson step. Reduce to beta < 1 if oscillating.
- **Restart strategy:** Reset history every 10 iterations or when the residual increases by more than 10x (indicating the local linearization broke down).
- **Safeguarding:** Clamp n_e to [1e12, 1e20] cm^-3 after each update.

```python
@jit
def anderson_acceleration(g, x0, m=3, max_iter=50, tol=1e-8):
    """Anderson acceleration for fixed-point x = g(x)."""
    n = x0.shape[0]
    X_hist = jnp.zeros((m+1, n))  # history of x values
    G_hist = jnp.zeros((m+1, n))  # history of g(x) values
    R_hist = jnp.zeros((m+1, n))  # history of residuals

    def body_fn(carry, _):
        x, X_hist, G_hist, R_hist, k = carry
        gx = g(x)
        r = gx - x

        # Store in circular buffer
        idx = k % (m + 1)
        X_hist = X_hist.at[idx].set(x)
        G_hist = G_hist.at[idx].set(gx)
        R_hist = R_hist.at[idx].set(r)

        # Build difference matrices (only use min(k, m) history)
        mk = jnp.minimum(k, m)
        # ... least-squares solve for mixing coefficients ...
        # x_new = mixed update

        return (x_new, X_hist, G_hist, R_hist, k+1), r

    init = (x0, X_hist, G_hist, R_hist, 0)
    (x_final, _, _, _, _), residuals = jax.lax.scan(
        body_fn, init, None, length=max_iter
    )
    return x_final
```

**Convergence:** For smooth contractive maps, AA with depth m achieves convergence rate comparable to GMRES(m) applied to the linearized problem. For the Saha equation (which is smooth and monotone in n_e), expect convergence in 5-8 iterations vs 15-30 for Picard.

**Known failure modes:**
- **Non-contractive regime:** At very high T (>5 eV) where doubly-ionized species dominate, the n_e map can become non-contractive. Safeguard with damping beta = 0.5.
- **Stiff coupling at high n_e:** When n_e > 1e18, the Saha exponential is extremely steep. The least-squares problem in AA becomes ill-conditioned. Use Tikhonov regularization (lambda = 1e-6) on the least-squares solve.
- **History corruption after parameter jump:** When vmapping over a batch of (T, n_e) starting points, each batch element's AA history is independent. No cross-contamination.

**GPU pattern:** Use `vmap` over the batch of parameter points (T, C_i). Each batch element runs its own independent AA iteration. With `jax.lax.scan` for the iteration loop, the entire batch is JIT-compiled into a single XLA program.

**Alternatives considered:**
- **Picard iteration:** Simple but slow (linear convergence). Keep as fallback.
- **Newton-Raphson:** Requires explicit Jacobian of the Saha system. For the 1D n_e case, the Jacobian is trivial (d(n_e_new)/d(n_e)), but for multi-element systems the analytic Jacobian is tedious. AA avoids this.
- **Broyden's method:** Similar to AA but maintains an approximate Jacobian inverse. AA is simpler and equally effective for this problem size.

**References:**
- Walker, H.F. and Ni, P. (2011). "Anderson Acceleration for Fixed-Point Iterations." SIAM J. Numer. Anal. 49(4), 1715-1735. DOI: 10.1137/10078356X
- Anderson, D.G. (1965). "Iterative Procedures for Nonlinear Integral Equations." J. ACM 12(4), 547-560. DOI: 10.1145/321296.321305
- Toth, A. and Kelley, C.T. (2015). "Convergence Analysis for Anderson Acceleration." SIAM J. Numer. Anal. 53(2), 805-819.
- Pollock, S. and Rebholz, L.G. (2019). "Anderson acceleration for contractive and noncontractive operators." IMA J. Numer. Anal. 41(4), 2841-2872.

---

### Method 4: Softmax Closure with Log-Sum-Exp Stabilization

**What:** Enforce the compositional constraint sum(C_i) = 1 via reparameterization C_i = exp(theta_i) / sum(exp(theta_j)). This maps unconstrained theta in R^D to the probability simplex, enabling standard gradient-based optimization without constrained solvers.

**Mathematical basis:**

The softmax function and its numerically stable implementation:
```
C_i = exp(theta_i - theta_max) / sum_j exp(theta_j - theta_max)
```
where theta_max = max(theta_j) prevents overflow. This is the log-sum-exp trick.

The Jacobian of the softmax:
```
dC_i/dtheta_j = C_i * (delta_ij - C_j)
```
This is rank-(D-1) as expected (one degree of freedom is consumed by the constraint). JAX computes this automatically via autodiff.

**Implementation notes:**

```python
@jit
def safe_softmax(theta):
    """Numerically stable softmax for compositional closure."""
    theta_max = jnp.max(theta)
    exp_theta = jnp.exp(theta - theta_max)
    return exp_theta / jnp.sum(exp_theta)
```

This is already available via `jax.nn.softmax` but the manual implementation makes the stabilization explicit for pedagogical clarity in the paper.

**Integration with joint optimizer:** The existing `joint_optimizer.py` already uses softmax parameterization. The GPU kernel simply needs to ensure the softmax is computed inside the JIT boundary with the optimizer.

**Known failure modes:**
- **Vanishing gradients for extreme compositions:** When one C_i -> 1 (all others -> 0), the Jacobian entries become exponentially small. Not a problem for typical LIBS analyses with 2-10 elements at 1-90% concentrations.
- **Symmetry breaking initialization:** Initialize theta = 0 (uniform prior) or theta = log(initial_guess). Do NOT initialize with large values that push compositions to boundaries.

**Alternatives considered:**
- **ILR transform (already in closure.py):** Isometric log-ratio maps the D-simplex to R^(D-1), preserving the Aitchison geometry. Mathematically more elegant but adds complexity (Helmert basis matrix) with no practical advantage for this optimization problem.
- **Projected gradient descent:** Project back onto simplex after each gradient step. Simpler conceptually but the projection step is not differentiable, breaking JAX's autodiff.

---

### Method 5: Batch Forward Model via JAX vmap

**What:** Generate spectra for a grid of (T, n_e, C) parameter combinations by vmapping the single-spectrum forward model over the parameter batch dimension. This is the core of manifold generation.

**Mathematical basis:**

The forward model f: (T, n_e, C, lambda) -> I(lambda) composes:
1. Saha ionization balance: n_z(T, n_e, C)
2. Boltzmann level populations: n_k(T, n_z)
3. Line emissivity: epsilon = (hc/4*pi*lambda) * A_ki * n_k
4. Voigt broadening: I(lambda) = sum_lines epsilon_l * V(lambda - lambda_l; sigma_l, gamma_l)

Each step is already vectorized over lines (step 4 is the N_wl x N_lines outer product). The batch dimension adds vmapping over the parameter tuple (T, n_e, C).

**JAX patterns:**

```python
# Single-spectrum function (already exists as _compute_spectrum_snapshot)
@jit
def single_spectrum(T_eV, n_e, concentrations, wl_grid, atomic_data):
    # ... Saha -> Boltzmann -> emissivity -> Voigt -> sum ...
    return intensity  # shape (N_wl,)

# Batch over parameter grid
batch_spectrum = jit(vmap(
    single_spectrum,
    in_axes=(0, 0, 0, None, None)  # vmap over T, n_e, C; broadcast wl_grid, atomic_data
))

# Usage: generate B spectra simultaneously
spectra = batch_spectrum(T_batch, ne_batch, C_batch, wl_grid, atomic_data)
# spectra: shape (B, N_wl)
```

**Memory management for V100S (16 GB HBM2):**

The inner computation materializes an (N_wl, N_lines) matrix per batch element. For N_wl=10000, N_lines=5000, float32:
- Per-spectrum: 10000 * 5000 * 4 bytes = 200 MB
- Batch of 64: 12.8 GB (fits in 16 GB with headroom)
- Batch of 80: 16 GB (tight; reduce to 64 or use float16 for profile matrix)

**Recommended batch sizes:**
- Manifold generation: B = 64 (conservative) to B = 128 (with gradient checkpointing)
- Gradient-based optimization: B = 16-32 (autodiff doubles memory)

**Time integration via lax.scan:** The existing `_time_integrated_spectrum` uses `jax.lax.scan` for cooling-plasma time integration. This is correct -- scan compiles the loop into a single XLA program without Python-level overhead. The time_steps parameter (typically 5-20) adds a linear multiplier to the computation.

**Known failure modes:**
- **XLA compilation time:** First call compiles the full computation graph. For complex forward models with many lines, compilation can take 30-60 seconds. Subsequent calls with same shapes are instant. Warm up with a dummy batch before timing.
- **Device OOM:** If batch size * N_wl * N_lines exceeds device memory, JAX throws a cryptic XLA error. Use `jax.device_put` with sharding for multi-GPU, or reduce batch size.
- **NaN propagation:** If any line has NaN in atomic data (e.g., missing Stark parameters), it propagates through the entire spectrum. The existing `jnp.where(jnp.isnan(...), estimate, value)` pattern handles this correctly.

**References:**
- Bradbury, J. et al. (2018). "JAX: composable transformations of Python+NumPy programs." github.com/google/jax
- Frostig, R. et al. (2018). "Compiling machine learning programs via high-level tracing." SysML Conference.

---

### Method 6: FAISS GPU Indexing for Manifold Lookup

**What:** After generating the spectral manifold (a matrix of shape (N_params, N_wl) stored in HDF5/Zarr), use FAISS GPU to build an approximate nearest-neighbor index for fast inference: given a measured spectrum, find the closest manifold entry and read off (T, n_e, C).

**Mathematical basis:**

The distance metric is L2 (Euclidean) on the normalized spectrum vectors. Normalization is critical -- raw spectra span orders of magnitude in intensity, so normalize each spectrum to unit L2 norm or use a spectral-range-specific standardization.

Alternative: cosine similarity (equivalent to L2 on unit-normalized vectors). Use L2 on normalized spectra because it is the FAISS default and hardware-optimized.

**Index type recommendation:** `GpuIndexIVFFlat` with nlist = sqrt(N_params).

- **IVF (Inverted File):** Partitions the dataset into nlist Voronoi cells via k-means. At query time, only nprobe closest cells are searched.
- **Flat:** Exact distance computation within each cell (no quantization). For 10K-dimensional spectral vectors, quantization (PQ) loses too much spectral structure.
- **nlist = sqrt(N):** Standard heuristic. For N = 1M manifold entries, nlist = 1000.
- **nprobe = 10-50:** Trade recall vs speed. nprobe=10 gives ~95% recall; nprobe=50 gives ~99%.

**Configuration for V100S:**

```python
import faiss

d = N_wl  # dimension = number of wavelength points
nlist = int(np.sqrt(N_params))
quantizer = faiss.IndexFlatL2(d)
index_cpu = faiss.IndexIVFFlat(quantizer, d, nlist)

# Move to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Train on the manifold
gpu_index.train(manifold_spectra)  # float32, shape (N_params, d)
gpu_index.add(manifold_spectra)

# Query
gpu_index.nprobe = 20
distances, indices = gpu_index.search(query_spectra, k=5)  # top-5 neighbors
```

**Memory budget:** FAISS IVFFlat stores the full vectors. For N = 1M, d = 10000, float32: 40 GB. Does NOT fit on V100S 16 GB. Solutions:
1. Reduce d via PCA to 256-512 dimensions before indexing (recommended; spectral features are highly redundant).
2. Use IVF with Product Quantization (IVFPQ) to compress vectors.
3. Shard across multiple GPUs.

Recommendation: PCA to d=256 first, then IVFFlat. PCA preserves >99% of spectral variance for typical LIBS spectra. This brings memory to 1 GB for 1M entries.

**Known failure modes:**
- **Normalization mismatch:** If training spectra and query spectra use different normalization, distances are meaningless. Always normalize identically.
- **Curse of dimensionality:** Raw spectra with d=10000 suffer from distance concentration. PCA reduction is essential for meaningful nearest-neighbor results.
- **Quantization artifacts:** Do NOT use PQ (product quantization) for spectral data without validation -- the subspace independence assumption of PQ is violated by correlated spectral features.

**References:**
- Johnson, J., Douze, M., and Jégou, H. (2021). "Billion-Scale Similarity Search with GPUs." IEEE Trans. Big Data 7(3), 535-547. DOI: 10.1109/TBDATA.2019.2921572

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Faddeeva function | Weideman-36 | Humlicek W4 | Gradient instability at region boundaries; 4x redundant computation in jnp.where |
| Faddeeva function | Weideman-36 | Zaghloul Chebyshev | More complex implementation for marginal accuracy gain; Weideman-36 already at machine precision in float64 |
| Faddeeva function | Weideman-36 | scipy.special.wofz | CPU-only; cannot run in JIT-compiled JAX |
| Boltzmann fitting | Normal equations | QR decomposition | Overkill for 2-parameter fit; 3x slower |
| Boltzmann fitting | Normal equations | SVD via jnp.linalg.lstsq | General-purpose overhead; allocates workspace |
| Fixed-point acceleration | Anderson (m=3) | Newton-Raphson | Requires explicit Jacobian; AA is Jacobian-free |
| Fixed-point acceleration | Anderson (m=3) | Picard iteration | Linear convergence; 3-5x more iterations |
| Fixed-point acceleration | Anderson (m=3) | Broyden quasi-Newton | Similar convergence but maintains approximate Jacobian storage |
| Compositional closure | Softmax | ILR transform | Adds Helmert basis complexity with no practical gain for optimization |
| Compositional closure | Softmax | Projected gradient | Non-differentiable projection breaks autodiff |
| NN search | FAISS IVFFlat | Brute-force L2 | O(N*d) per query; too slow for N > 100K |
| NN search | FAISS IVFFlat | HNSW | CPU-only in FAISS; GPU IVF is faster for batch queries |

---

## Validation Strategy

| Check | Expected Result | Tolerance | Reference |
|-------|-----------------|-----------|-----------|
| Voigt profile vs scipy.wofz | Identical Re[w(z)] | < 1e-12 (float64), < 1e-5 (float32) | Weideman 1994 Table 1 |
| Voigt FWHM vs Olivero-Longbothum | fV = 0.5346*fL + sqrt(0.2166*fL^2 + fG^2) | < 0.02% | Olivero & Longbothum 1977 |
| Boltzmann slope from synthetic spectrum | Recovers input T to within statistical uncertainty | < 2% relative at SNR=100 | Round-trip test |
| Saha ionization fractions vs NIST LIBS | Fe I/II ratio at T=1 eV, n_e=1e17 | < 5% relative | NIST LIBS simulation |
| Anderson vs Picard final n_e | Identical to Picard converged value | < 1e-8 relative | Self-consistency |
| Anderson iteration count | 5-8 vs 15-30 for Picard | 2-4x reduction | Walker & Ni 2011 benchmarks |
| Softmax sum(C_i) | Exactly 1.0 | Machine precision | By construction |
| Batch forward vs single forward | Identical spectra element-by-element | < 1e-10 relative | vmap correctness |
| FAISS recall@1 with nprobe=20 | > 95% on held-out manifold entries | > 95% | Empirical on generated manifold |
| GPU vs CPU throughput ratio | > 100x for batch sizes >= 64 | > 50x minimum | V100S vs single-core Xeon |

---

## Sources

- Weideman, J.A.C. (1994). SIAM J. Numer. Anal. 31(5), 1497-1518. [Faddeeva rational approximation]
- Humlicek, J. (1982). JQSRT 27(4), 437-444. [W4 Faddeeva approximation]
- Zaghloul, M.R. (2024). ACM Trans. Math. Softw. [Chebyshev Faddeeva, Algorithm 1032]
- Schreier, F. (2018). JQSRT 211, 78-87. [Optimized Voigt implementations review]
- Walker, H.F. and Ni, P. (2011). SIAM J. Numer. Anal. 49(4), 1715-1735. [Anderson acceleration theory]
- Anderson, D.G. (1965). J. ACM 12(4), 547-560. [Original Anderson mixing]
- Toth, A. and Kelley, C.T. (2015). SIAM J. Numer. Anal. 53(2), 805-819. [AA convergence analysis]
- Johnson, J. et al. (2021). IEEE Trans. Big Data 7(3), 535-547. [FAISS]
- Olivero, J.J. and Longbothum, R.L. (1977). JQSRT 17(2), 233-236. [Voigt FWHM approximation]
- Tognoni, E. et al. (2010). Spectrochim. Acta B 65(1), 1-14. [CF-LIBS review]
- Ciucci, A. et al. (1999). Appl. Spectrosc. 53(8), 960-964. [CF-LIBS procedure]
- Egozcue, J.J. et al. (2003). Math. Geol. 35(3), 279-300. [ILR compositional transform]

**NOTE:** Web search was unavailable during this research session. All references cited are from established literature (>2 years old). The Zaghloul 2024 reference should be verified for exact DOI and publication status. Confidence for computational benchmarks (GPU throughput, FAISS recall) is MEDIUM -- these should be validated empirically on the target V100S hardware.

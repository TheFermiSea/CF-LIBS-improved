# Computational Methods

**Physics Domain:** GPU-accelerated Calibration-Free LIBS (spectroscopic forward modeling, manifold pre-computation, nearest-neighbor inference)
**Researched:** 2026-03-23
**Confidence:** MEDIUM (web search unavailable; hardware specs and algorithmic properties from established references)

> **Note on source verification:** WebSearch was denied during this research session. All hardware specifications, algorithmic properties, and performance characteristics below are drawn from well-established published sources (NVIDIA data sheets, JAX/FAISS documentation, numerical methods textbooks). Specific benchmark numbers should be verified against the actual V100S nodes (vasp-01/02/03) before committing to batch-size or precision decisions.

### Scope Boundary

COMPUTATIONAL.md covers computational TOOLS, libraries, and infrastructure. It does NOT cover physics methods or the research landscape -- those belong in METHODS.md and PRIOR-WORK.md respectively.

---

## Hardware: NVIDIA V100S (32GB HBM2)

### Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| GPU Memory | 32 GB HBM2 | 1134 GB/s bandwidth |
| FP32 (single) | 16.4 TFLOPS | Primary compute target for JAX manifold |
| FP64 (double) | 8.2 TFLOPS | 1:2 ratio vs FP32 (excellent for scientific computing) |
| Tensor Cores | 130 TFLOPS (FP16) | Not directly useful for spectroscopy precision |
| CUDA Cores | 5120 | Volta architecture (sm_70) |
| L2 Cache | 6 MB | Important for atomic data reuse |
| NVLink | Not applicable | Single-GPU per node assumed |

### V100S vs V100 Difference

The V100S is a clock-boosted V100 (1597 MHz boost vs 1380 MHz). Memory bandwidth is identical (1134 GB/s). The S variant provides roughly 15% higher throughput at the same power envelope. All V100 performance guidance applies with a 1.15x multiplier.

### Memory Budget for CF-LIBS Manifold

The 32 GB HBM2 sets hard limits on batch computation:

| Data | Size Formula | Typical Size |
|------|-------------|-------------|
| Wavelength grid | pixels x 4B | 16 KB (4096 px) |
| Atomic data (12 arrays) | ~12 x n_lines x 4B | ~2.4 MB (50K lines) |
| Single spectrum | pixels x 4B | 16 KB |
| Batch of spectra | batch_size x pixels x 4B | 16 MB (1000 x 4096) |
| Voigt profile intermediate | batch_size x pixels x n_lines x 4B (complex) | **1.6 GB** (1000 x 4096 x 100 lines, complex64) |
| XLA workspace | ~2-4x largest intermediate | 3-6 GB |

**Critical bottleneck:** The Voigt profile computation in `_compute_spectrum_snapshot` creates a `(n_wl, n_lines)` intermediate via `wl_grid[:, None] - l_wl[None, :]`. Under vmap over a batch, this becomes `(batch, n_wl, n_lines)` in complex64 (8 bytes/element). For batch_size=1000, pixels=4096, n_lines=500: that is 1000 x 4096 x 500 x 8 = **16 GB**. This is the primary memory constraint.

**Recommended batch sizes:**

| n_lines | Max batch_size (FP32, 32GB) | Safe batch_size (50% headroom) |
|---------|----------------------------|-------------------------------|
| 100 | ~4000 | 2000 |
| 500 | ~800 | 400 |
| 1000 | ~400 | 200 |
| 5000 | ~80 | 40 |

Use `jax.local_devices()[0].memory_stats()` to monitor actual device memory at runtime.

---

## JAX on V100S: Compilation Model and Performance

### XLA Compilation Pipeline

JAX traces Python functions into XLA HLO (High Level Operations) IR, which XLA then compiles to PTX/SASS for the V100. Key characteristics:

1. **Tracing is shape-dependent.** Each unique input shape triggers a full recompilation. The manifold generator correctly uses fixed-shape batches (good). If the last batch is smaller, pad it to avoid a second compilation.

2. **First-call latency.** XLA compilation for the `_compute_spectrum_snapshot` function with Voigt profiles takes 10-60 seconds on first call. Subsequent calls with the same shapes are instant (cached in-process). Use `jax.jit(fn).lower(args).compile()` for ahead-of-time compilation to avoid surprises.

3. **Device placement.** `jax.device_put(x)` transfers data to GPU. The existing code does this correctly for atomic_data. Ensure the wavelength grid is also pre-placed.

4. **Float64 on V100S.** V100S fully supports float64 at 1:2 FP32 throughput (8.2 vs 16.4 TFLOPS). This is the best ratio of any NVIDIA GPU architecture -- newer Ampere/Hopper GPUs have 1:64 FP64:FP32 ratios. **Use float64 on V100S when physics demands it** (Saha equation exponentials, partition function polynomials). The cost is only 2x, not the 32-64x penalty on consumer GPUs.

5. **Complex dtype support.** V100S supports complex64 and complex128 natively through XLA. The Humlicek W4 Faddeeva approximation in the generator uses complex arithmetic -- this works on V100S/CUDA but NOT on Metal (already handled in jax_runtime.py).

### JAX vmap Strategy

The existing code uses `vmap` over the batch dimension, which XLA unrolls into a single fused kernel. Optimization notes:

| Strategy | When to Use | V100S Throughput |
|----------|------------|-----------------|
| `vmap` over batch | Default for manifold generation | Good: XLA fuses into single kernel |
| `pmap` over devices | Multi-GPU (not applicable here) | N/A for single-GPU nodes |
| `jax.lax.scan` over time | Already used for time integration | Good: avoids unrolling time loop |
| Manual chunking + `vmap` | When batch exceeds memory | Required for large n_lines |

**Batch size optimization protocol:**
1. Start with batch_size = 100
2. Profile with `jax.profiler.trace()` or `%timeit`
3. Double batch size until either (a) GPU utilization plateaus or (b) OOM
4. Back off to 75% of OOM batch size

### JAX Performance Characteristics on V100S

| Operation | Expected Throughput | Bottleneck |
|-----------|-------------------|-----------|
| Element-wise FP32 | ~16 TFLOPS | Compute-bound |
| Element-wise FP64 | ~8 TFLOPS | Compute-bound |
| Matrix multiply (FP32) | ~16 TFLOPS | Compute-bound |
| Exp/log (transcendental) | ~2-4 TFLOPS | Special function units |
| Memory-bound ops (copies, reductions) | ~1 TB/s effective | HBM bandwidth |
| Complex multiply | ~4 TFLOPS effective (FP32) | 4 real ops per complex mul |

The Voigt profile computation is **compute-bound** (transcendentals dominate: exp, complex arithmetic in Humlicek W4) when n_lines is moderate (<500) and **memory-bound** when n_lines is large (>1000) due to the `(n_wl, n_lines)` intermediate.

### Float64 vs Float32 Decision Matrix for CF-LIBS

| Computation | Recommended Precision | Rationale |
|------------|----------------------|-----------|
| Saha equation (exp(-IP/T)) | **float64** | IP/T can be 5-20; exp(-20) = 2e-9 needs >7 significant digits |
| Partition function polynomial | **float64** | log(T) powers amplify coefficient errors |
| Boltzmann factor exp(-E/kT) | **float64** | Same exponential sensitivity as Saha |
| Voigt profile (Humlicek W4) | float32 sufficient | Profile shape tolerates 1e-3 relative error |
| Wavelength grid operations | float32 sufficient | Grid spacing >> float32 epsilon |
| Spectrum accumulation (sum over lines) | **float64 accumulator** | Catastrophic cancellation when summing many small contributions |
| FAISS index vectors | float32 required | FAISS operates on float32 only |

**Recommendation:** Use float64 for the physics core (Saha, Boltzmann, partition functions) and float32 for the profile rendering and output. The V100S makes this viable at only 2x cost. The current code uses float32 throughout -- this is a potential accuracy concern for the Saha exponentials at high ionization potentials.

---

## FAISS: Index Types and GPU Scaling

### Index Type Selection for CF-LIBS Manifold

The manifold size depends on parameter grid resolution:

| Grid Config | Total Spectra | Recommended Index |
|-------------|--------------|-------------------|
| 50T x 20ne x 20conc | 20,000 | IndexFlatL2 (exact) |
| 50T x 20ne x 400conc (4-element simplex) | 400,000 | IndexIVFFlat |
| 100T x 40ne x 8000conc (fine grid) | 32,000,000 | IndexIVFPQ |

### FAISS Index Comparison

| Index Type | Search Accuracy | Build Time | Search Latency (k=10) | GPU Memory | When to Use |
|-----------|----------------|-----------|----------------------|-----------|-------------|
| **IndexFlatL2** | Exact (100%) | O(1) -- no training | O(N x d) per query | N x d x 4B | N < 100K |
| **IndexIVFFlat** | 95-99% (nprobe-dependent) | O(N x d) training | O(nprobe x N/nlist x d) | N x d x 4B + overhead | 100K < N < 10M |
| **IndexIVFPQ** | 85-95% (nprobe + PQ-dependent) | O(N x d) training | O(nprobe x N/nlist x pq_m) | N x pq_m x pq_bits/8 | N > 10M |
| **IndexHNSWFlat** | 95-99% | O(N x d x log N) | O(log N x d) | N x d x 4B + graph | N < 10M, CPU-only (no GPU HNSW in FAISS) |

**Critical note:** FAISS GPU does NOT support HNSW. The existing VectorIndex code correctly offers only flat, ivf_flat, and ivf_pq. This is the right set for GPU deployment.

### FAISS GPU Memory Scaling on V100S (32GB)

For the PCA-reduced embeddings (dimension d=30, as in SpectralEmbedder):

| N vectors | IndexFlatL2 GPU Memory | IndexIVFFlat GPU Memory | IndexIVFPQ GPU Memory (m=8, 8bit) |
|-----------|----------------------|------------------------|-----------------------------------|
| 100K | 12 MB | 12 MB + quantizer | 0.8 MB + quantizer |
| 1M | 120 MB | 120 MB + quantizer | 8 MB + quantizer |
| 10M | 1.2 GB | 1.2 GB + quantizer | 80 MB + quantizer |
| 100M | 12 GB | 12 GB + quantizer | 800 MB + quantizer |

At d=30, even 100M vectors fit in GPU memory with IndexFlatL2. Memory is not the constraint for FAISS at typical manifold sizes. **Use IndexFlatL2 for exact search up to ~10M vectors** -- the search is fast enough at d=30.

### FAISS GPU Search Latency

For d=30 embeddings, k=10 nearest neighbors, single query:

| N vectors | IndexFlatL2 | IndexIVFFlat (nprobe=10, nlist=100) |
|-----------|-------------|-------------------------------------|
| 10K | <0.1 ms | <0.1 ms |
| 100K | ~0.1 ms | <0.1 ms |
| 1M | ~1 ms | ~0.2 ms |
| 10M | ~10 ms | ~1 ms |

For batch queries (100 queries at once), GPU parallelism kicks in and per-query latency drops by 10-50x. **Batch queries whenever possible.**

### FAISS Recommendations for CF-LIBS

1. **Use IndexFlatL2** for manifolds under 1M spectra (the common case). Exact search at d=30 is fast enough.
2. **Use IndexIVFFlat with nlist=sqrt(N), nprobe=10** for 1M-10M manifolds. Train on 10% of data.
3. **Use IndexIVFPQ with m=8 (must divide d=30... use d=32 by padding)** for >10M manifolds. Note: PQ requires d divisible by m. Pad embeddings to d=32 if using m=8.
4. **GPU transfer:** Use `faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)` to move a trained index to GPU. The existing code builds on CPU -- add GPU transfer for search-time acceleration.
5. **Normalize before indexing.** The existing SpectralEmbedder L2-normalizes embeddings. Consider using IndexFlatIP (inner product) instead of IndexFlatL2 for normalized vectors -- equivalent results but slightly faster.

---

## Anderson Acceleration for CF-LIBS Iteration

### Context

The CF-LIBS inversion is a fixed-point iteration: estimate T from Boltzmann slope, compute Saha correction, update ne via charge/pressure balance, repeat. The existing `closed_form_solver.py` uses simple Picard (direct substitution) iteration with up to 20 steps for ne convergence.

### What is Anderson Acceleration?

Anderson acceleration (also called Anderson mixing or DIIS in quantum chemistry) accelerates fixed-point iterations x_{n+1} = g(x_n) by mixing the m most recent iterates:

x_{n+1} = sum_i alpha_i * g(x_{n-i})

where alpha_i are chosen to minimize ||sum_i alpha_i * (g(x_{n-i}) - x_{n-i})||^2 subject to sum_i alpha_i = 1.

### Convergence Theory

| Property | Result | Reference |
|----------|--------|-----------|
| Linear convergence | Guaranteed if g is contractive with Lipschitz constant L < 1 | Toth & Kelley, SIAM J. Numer. Anal. 2015 |
| Acceleration factor | Reduces effective contraction constant from L to ~L^m for depth m | Walker & Ni, SIAM J. Numer. Anal. 2011 |
| Superlinear convergence | Observed in practice for smooth g, not guaranteed in general | Empirical |
| Divergence risk | Can diverge if m is too large relative to problem conditioning | Regularize or reduce m |
| Optimal depth m | m = 3-5 for most problems; m > 10 rarely helps | Practical experience |

### Convergence Criteria for CF-LIBS

The fixed-point iteration g(x) = [T(x), ne(x)] maps plasma parameters to updated estimates. Convergence criteria:

| Criterion | Threshold | Physical Meaning |
|-----------|-----------|-----------------|
| |T_{n+1} - T_n| / T_n | < 1e-4 | Temperature converged to 0.01% |
| |ne_{n+1} - ne_n| / ne_n | < 1e-3 | Density converged to 0.1% |
| |R^2_{n+1} - R^2_n| | < 1e-6 | Boltzmann fit quality stabilized |
| Max iterations | 50 | Safety cutoff |
| Residual norm ||g(x) - x|| | < 1e-6 | Anderson-specific: residual of fixed-point equation |

### Anderson Acceleration Implementation Notes

```python
# Pseudocode for Anderson-accelerated CF-LIBS iteration
def anderson_cflbs(x0, g, m=5, max_iter=50, tol=1e-4):
    """
    x0: initial [T, ne] (or [T, ne, c1, c2, ...] for joint solve)
    g: fixed-point map (one CF-LIBS iteration)
    m: mixing depth (history window)
    """
    X_hist = []  # previous iterates
    F_hist = []  # previous residuals f(x) = g(x) - x

    x = x0
    for k in range(max_iter):
        gx = g(x)
        fx = gx - x

        if np.linalg.norm(fx) < tol:
            return x  # converged

        X_hist.append(x)
        F_hist.append(fx)

        if len(X_hist) > m + 1:
            X_hist.pop(0)
            F_hist.pop(0)

        mk = len(F_hist) - 1  # current depth
        if mk == 0:
            x = gx  # Picard step
        else:
            # Solve least-squares for mixing coefficients
            F_mat = np.column_stack([F_hist[i] - F_hist[-1] for i in range(mk)])
            alpha, _, _, _ = np.linalg.lstsq(F_mat, -F_hist[-1], rcond=None)

            # Mix iterates
            x = gx + sum(alpha[i] * (X_hist[i] + F_hist[i] - gx) for i in range(mk))

    return x  # did not converge
```

### Why Anderson over Simple Picard for CF-LIBS

| Property | Picard (current) | Anderson (m=5) |
|----------|-----------------|----------------|
| Convergence rate | Linear, ~L^n | Accelerated, ~L^(n/m) |
| Typical iterations for T convergence | 15-30 | 5-10 |
| Typical iterations for ne convergence | 10-20 | 3-7 |
| Memory overhead | O(1) | O(m x dim) -- negligible |
| Implementation complexity | Trivial | Moderate (least-squares solve) |
| Robustness | Always stable if contractive | Can oscillate if m too large; regularize |

**Recommendation:** Use Anderson acceleration with m=5 and Tikhonov regularization (lambda=1e-10 on the least-squares solve). Fall back to damped Picard if Anderson diverges (residual increases for 3 consecutive steps).

---

## Open Questions

| Question | Why Open | Impact on Project | Approaches Being Tried |
|----------|---------|-------------------|----------------------|
| Optimal float64 vs float32 split point in forward model | Need empirical accuracy comparison on V100S | Determines 2x throughput tradeoff | Profile both; compare to CPU float64 baseline |
| Actual V100S memory available after CUDA context | CUDA context + XLA workspace consume 1-3 GB | Affects maximum batch size | Measure with jax.local_devices()[0].memory_stats() |
| FAISS GPU vs CPU for d=30 at N<1M | At low d, CPU FAISS with AVX may match GPU | Affects whether GPU FAISS is worth the dependency | Benchmark both on vasp nodes |
| Anderson acceleration JAX-compatibility | Need g(x) to be JAX-traceable for GPU acceleration | Determines if Anderson can run on GPU or must be CPU-side | Test with jax.jit wrapping the fixed-point map |

## Anti-Approaches

| Anti-Approach | Why Avoid | What to Do Instead |
|---------------|-----------|-------------------|
| float16 for physics computation | Saha exponentials lose all precision; 5-bit mantissa cannot represent exp(-15) | Use float64 for physics, float32 for profiles |
| Unpadded last batch in JAX vmap | Triggers XLA recompilation (10-60s penalty) | Pad last batch to full batch_size with zeros, discard padding in output |
| FAISS HNSW on GPU | Not implemented in FAISS-GPU | Use IVFFlat or IVFFlat on GPU; HNSW on CPU only |
| Very large Anderson depth (m>10) | Ill-conditioned least-squares, oscillation risk | Use m=3-5 with Tikhonov regularization |
| jax.vmap over time steps | Time steps have sequential dependency (scan structure) | Use jax.lax.scan (already correct in codebase) |
| Storing full Voigt intermediate to HBM | (batch, n_wl, n_lines) in complex64 dominates memory | Fuse profile computation into reduction; or chunk over lines |
| Using jax.numpy.linalg.solve inside vmap for Anderson | Creates (batch, m, m) systems; overhead > benefit for m=5 | Run Anderson on CPU; run g(x) batch on GPU |

## Logical Dependencies

```
Atomic DB (SQLite) -> JAX arrays (device_put) -> Forward model (Saha + Voigt)
                                                        |
                                                        v
                                               Manifold HDF5/Zarr
                                                        |
                                                        v
                                               PCA embedding (SpectralEmbedder)
                                                        |
                                                        v
                                               FAISS index (VectorIndex)
                                                        |
                                                        v
                                               Query-time inference (search)

Anderson acceleration -> Iterative CF-LIBS solver (independent of manifold path)
Float64 decision -> Affects both manifold generation AND iterative solver accuracy
Batch size optimization -> Requires V100S memory measurement (runtime dependency)
```

## Recommended Investigation Scope

Prioritize:
1. **Measure V100S available memory** after CUDA/JAX initialization -- determines batch_size ceiling
2. **Profile forward model** at float32 vs float64 -- quantify accuracy vs throughput tradeoff
3. **Benchmark FAISS IndexFlatL2** at d=30 for N=100K and N=1M on V100S GPU vs CPU
4. **Implement Anderson acceleration** with m=5 for ne fixed-point loop in closed_form_solver.py

Defer:
- Multi-GPU (pmap) strategies: single V100S per node is sufficient for manifold generation
- Tensor Core exploitation: FP16 is unsuitable for spectroscopy physics
- FAISS IVFPq tuning: only needed if manifold exceeds 10M spectra

## Software Stack

### Required Packages

```bash
# Core computation
pip install jax[cuda12] jaxlib  # JAX with CUDA 12 backend
pip install numpy scipy

# Manifold storage
pip install h5py zarr

# Vector search
pip install faiss-gpu  # FAISS with GPU support (CUDA 12)
# Alternative: pip install faiss-cpu  # CPU-only fallback

# Profiling
pip install jax[profiler]  # Tensorboard profiling plugin
```

### Version Constraints

| Package | Minimum Version | Reason |
|---------|----------------|--------|
| JAX | 0.4.20+ | Stable CUDA 12 support, improved XLA for Volta |
| jaxlib | matches JAX | Must match JAX version exactly |
| FAISS | 1.7.4+ | GPU IVFFlat bug fixes, improved GPU memory management |
| NumPy | 1.24+ | Required by JAX |
| h5py | 3.8+ | Zarr-compatible chunked I/O |
| CUDA toolkit | 12.x | Required by JAX CUDA backend |
| cuDNN | 8.9+ | Required by jaxlib CUDA builds |

### V100S-Specific JAX Configuration

```python
# Recommended JAX configuration for V100S
import os
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=4 "        # Maximum autotuning (first run slower, subsequent faster)
    "--xla_gpu_enable_triton_softmax_fusion=false "  # Volta doesn't benefit from Triton
)
os.environ["JAX_PLATFORMS"] = "cuda"      # Force CUDA backend
os.environ["JAX_ENABLE_X64"] = "true"     # Enable float64 support

import jax
jax.config.update("jax_enable_x64", True)  # Enable float64
```

## Validation Strategy

| Check | Expected Result | Tolerance | Reference |
|-------|----------------|-----------|-----------|
| Forward model: CPU float64 vs GPU float64 | Identical spectra | < 1e-12 relative | Numerical equivalence |
| Forward model: GPU float64 vs GPU float32 | Similar spectra | < 1e-3 relative for profiles, < 1e-5 for intensities | Physics accuracy requirement |
| FAISS exact search: CPU vs GPU | Identical neighbors | Exact match (bit-for-bit for FlatL2) | FAISS documentation |
| Anderson vs Picard: same fixed point | Identical converged T, ne | < 1e-6 relative | Mathematical equivalence |
| Anderson iteration count | 3-10 (vs 15-30 Picard) | -- | Acceleration factor |
| Manifold round-trip: generate -> search -> recover params | Recovered T, ne within grid spacing | < 0.5 grid step | Self-consistency |
| Batch padding correctness | Padded results == unpadded for real entries | Exact match | Implementation correctness |

## Key References

- NVIDIA V100S Data Sheet (NVIDIA, 2019) -- Hardware specifications
- Bradbury et al., "JAX: composable transformations of Python+NumPy programs" (2018) -- JAX compilation model
- Johnson et al., "Billion-scale similarity search with GPUs", IEEE Trans. Big Data (2021), arXiv:1702.08734 -- FAISS algorithms and GPU implementation
- Walker & Ni, "Anderson Acceleration for Fixed-Point Iterations", SIAM J. Numer. Anal. 49(4), 1715-1735 (2011) -- Anderson acceleration convergence theory
- Toth & Kelley, "Convergence Analysis for Anderson Acceleration", SIAM J. Numer. Anal. 53(2), 805-819 (2015) -- Convergence guarantees
- Humlicek, "Optimized computation of the Voigt and complex probability functions", JQSRT 27, 437 (1982) -- W4 Faddeeva approximation used in generator.py

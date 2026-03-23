# REQUIREMENTS.md

## Contract Coverage

All requirements trace to the five optimization targets defined in the project contract. Each requirement has a unique ID, category, description, acceptance criteria, and traceability to anchors.

## Derivations

| ID | Title | Description | Acceptance Criteria | Anchors |
|----|-------|-------------|---------------------|---------|
| DERV-01 | GPU-optimized Voigt profile | Derive GPU-optimized Voigt profile formulation: Faddeeva w(z) -> Humlicek W4 rational approximation and Zaghloul 2024 Chebyshev expansion. Show JAX vmap parallelization over wavelength grid and line index. | LaTeX derivation with error bounds for both approximations; explicit vmap axes identified | ANC-VOIGT |
| DERV-02 | Vectorized Boltzmann fitting | Formalize vectorized Boltzmann plot fitting as batched weighted least-squares: y = ln(I*lambda/gA), x = E_k, weights from measurement uncertainty. Express as batched matmul (X^T W X)^{-1} X^T W y. | Closed-form solution with weight matrix derivation; batch dimension explicit | ANC-BOLTZ |
| DERV-03 | Anderson acceleration for Saha-Boltzmann | Derive Anderson acceleration mixing for the Saha-Boltzmann fixed-point iteration g(n_e) -> n_e. Specify memory depth M, regularization, and convergence criteria. | Derivation of mixing coefficients via least-squares residual minimization; stability analysis for M=1..10 | ANC-ANDER |
| DERV-04 | Softmax closure formalization | Formalize softmax closure C_i = exp(theta_i) / sum(exp(theta_j)) for composition simplex. Compare with ILR (isometric log-ratio) transform. Analyze gradient flow and numerical stability. | Both parameterizations derived with Jacobians; log-sum-exp trick for numerical stability | ANC-CLOSE |
| DERV-05 | Batch forward model decomposition | Derive batch forward model f(T, n_e, C; lambda) with vmap decomposition. Identify which operations parallelize over batch vs wavelength vs line dimensions. | Explicit vmap nesting strategy; memory footprint analysis for V100S (16 GB) | ANC-BATCH |

## Implementations

| ID | Title | Description | Acceptance Criteria | Anchors | Depends |
|----|-------|-------------|---------------------|---------|---------|
| IMPL-01 | GPU Voigt profile kernel | Implement GPU Voigt profile in JAX with jit compilation and vmap over wavelength grid. Support both Humlicek W4 and Zaghloul Chebyshev backends. | Passes existing Voigt tests; runs on V100S; supports float64 | ANC-VOIGT | DERV-01 |
| IMPL-02 | Vectorized Boltzmann fitting | Implement vectorized Boltzmann fitting in JAX as batched matmul. Support weighted and unweighted modes. | Matches CPU reference to <1e-10 relative error; handles degenerate cases (single line) | ANC-BOLTZ | DERV-02 |
| IMPL-03 | Anderson-accelerated Saha solver | Implement Anderson-accelerated Saha-Boltzmann solver in JAX with configurable memory depth M. Use jax.lax.while_loop for JIT compatibility. | Converges for all test cases; iteration count reduction >1.5x vs simple iteration | ANC-ANDER | DERV-03 |
| IMPL-04 | Softmax closure | Implement softmax closure in JAX with log-sum-exp stabilization. Provide forward and inverse transforms. | sum(C_i) = 1 to machine precision; gradients flow correctly (test with jax.grad) | ANC-CLOSE | DERV-04 |
| IMPL-05 | Batch forward model | Implement batch forward model using vmap over parameter batch dimension. Compose Voigt, emissivity, and instrument convolution. | Produces identical output to sequential loop; scales to batch_size=10^4 on V100S | ANC-BATCH | DERV-05 |

## Benchmarks

| ID | Title | Description | Metrics | Anchors | Depends |
|----|-------|-------------|---------|---------|---------|
| BENCH-01 | Voigt throughput | Voigt profile throughput benchmark: profiles/sec vs grid size (10^3 to 10^6 points). Compare Humlicek W4, Zaghloul Chebyshev, scipy.special.wofz. Measure accuracy vs Zaghloul reference. | Throughput (profiles/sec), relative error, GPU utilization | ANC-VOIGT, ANC-BENCH | IMPL-01 |
| BENCH-02 | Boltzmann fitting time | Boltzmann fitting time vs element count (1 to 20 elements), line count (10 to 1000 lines per element). CPU vs GPU comparison. | Wall time (ms), speedup ratio | ANC-BOLTZ, ANC-BENCH | IMPL-02 |
| BENCH-03 | Anderson convergence | Anderson solver convergence: iteration count vs memory depth M=1..10. Compare with simple (Picard) iteration and DIIS. | Iteration count, wall time, residual vs iteration | ANC-ANDER, ANC-BENCH | IMPL-03 |
| BENCH-04 | FAISS query latency | FAISS nearest-neighbor query latency for manifold lookup: 1M, 10M, 100M database points. CPU (IVF) vs GPU (GpuIVFFlat). | Query latency (ms), recall@1, index build time | ANC-BENCH | -- |
| BENCH-05 | Batch forward model throughput | Batch forward model throughput: spectra/sec vs batch size (1 to 10^5). CPU vs GPU scaling. | Throughput (spectra/sec), memory usage, speedup ratio | ANC-BATCH, ANC-BENCH | IMPL-05 |
| BENCH-06 | End-to-end pipeline | End-to-end pipeline benchmark: total time for inversion of 1 spectrum, 100 spectra, 10K spectra. GPU vs CPU. | Total wall time, per-spectrum time, component breakdown | ANC-BENCH | IMPL-01..05 |

## Validations

| ID | Title | Description | Acceptance Criteria | Anchors | Depends |
|----|-------|-------------|---------------------|---------|---------|
| VALD-01 | Voigt accuracy | Verify GPU Voigt profile accuracy vs Zaghloul 2024 reference implementation across full (x, y) parameter space. | Relative error < 1e-6 for all test points; no NaN/Inf in edge cases | ANC-VOIGT | IMPL-01 |
| VALD-02 | Boltzmann fit accuracy | Verify vectorized Boltzmann fit results match CPU scipy.optimize reference within numerical precision. | Slope and intercept agree to <1e-10 relative error; R^2 values identical to 6 decimal places | ANC-BOLTZ | IMPL-02 |
| VALD-03 | Anderson solver correctness | Verify Anderson-accelerated solver converges to same fixed point as simple Picard iteration for all test plasma conditions. | Fixed-point residual < 1e-12; T and n_e agree to <0.01% | ANC-ANDER | IMPL-03 |
| VALD-04 | Softmax closure precision | Verify softmax closure preserves sum-to-1 constraint to machine precision and handles edge cases (near-zero concentrations, single element). | abs(sum(C_i) - 1) < 1e-15; gradient check passes (jax.check_grads) | ANC-CLOSE | IMPL-04 |
| VALD-05 | Batch forward model parity | Verify batch forward model output matches sequential single-spectrum computation for diverse plasma conditions. | Max relative error < 1e-12 across 1000 random test cases | ANC-BATCH | IMPL-05 |

## Publications

| ID | Title | Description | Deliverables | Anchors | Depends |
|----|-------|-------------|-------------|---------|---------|
| PAPER-01 | JQSRT paper | Write full research article for JQSRT: GPU-Accelerated Calibration-Free LIBS Multi-Element Plasma Diagnostics. 7 figures, ~20 pages. | LaTeX source, 7 figures (PDF/SVG), bibliography | ANC-PAPER | BENCH-01..06, VALD-01..05 |
| PAPER-02 | Beamer slide deck | Create Beamer presentation for 20-minute talk. Speaker notes, backup slides with additional benchmarks. | LaTeX Beamer source, compiled PDF | ANC-PAPER | PAPER-01 |
| PAPER-03 | arXiv submission | Package and submit to arXiv. Primary: physics.comp-ph. Secondary: cs.CE. | arXiv-ready tarball, metadata | ANC-PAPER | PAPER-01 |

## Accuracy and Validation Criteria

| Quantity | Tolerance | Method |
|----------|-----------|--------|
| Voigt profile values | Relative error < 1e-6 | Comparison to Zaghloul 2024 reference |
| Boltzmann slope/intercept | Relative error < 1e-10 | Comparison to CPU scipy reference |
| Saha-Boltzmann fixed point | Residual < 1e-12 | Comparison to Picard iteration |
| Softmax sum-to-1 | abs(sum - 1) < 1e-15 | Direct computation |
| Batch forward model | Relative error < 1e-12 | Comparison to sequential loop |
| Concentrations (round-trip) | Relative error < 1% | Forward-invert round-trip test |

## Traceability Matrix

| Requirement | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 | Phase 7 |
|-------------|---------|---------|---------|---------|---------|---------|---------|
| DERV-01..05 | X | | | | | | |
| IMPL-01..05 | | X | | | | | |
| BENCH-01..06 | | | X | | | | |
| VALD-01..05 | | | | X | | | |
| PAPER-01 | | | | | X | | |
| PAPER-02 | | | | | | X | |
| PAPER-03 | | | | | | | X |

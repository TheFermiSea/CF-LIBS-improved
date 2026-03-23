# PROJECT.md

## What This Is

A GPU-accelerated Calibration-Free LIBS (CF-LIBS) multi-element plasma diagnostics system built on JAX. The project identifies five computational bottlenecks in the existing 58K LOC CF-LIBS pipeline, derives GPU-optimized replacements, benchmarks them on V100S hardware, and packages the results as a JQSRT publication.

## Core Research Question

What is the optimal computational architecture for real-time CF-LIBS multi-element plasma diagnostics, and how do GPU-accelerated implementations compare to existing CPU-based tools in throughput, accuracy, and scalability?

## Scoping Contract Summary

### Contract Coverage

The project scope covers five optimization targets that span the full CF-LIBS pipeline from line-profile computation through plasma solving to batch inference:

1. **Voigt profile** -- Faddeeva w(z), Humlicek W4 vs Zaghloul 2024 Chebyshev, JAX vmap
2. **Boltzmann fitting** -- LTE population weighted least-squares, batched matmul
3. **Anderson acceleration** -- Saha-Boltzmann coupling fixed-point solver
4. **Softmax closure** -- Simplex constraint C_i = exp(theta_i) / sum(exp(theta_j))
5. **Batch forward model** -- f(T, n_e, C; lambda) vmap over batch dimension

### User Guidance

- Focus on computational performance, not new physics
- All GPU work targets V100S (16 GB HBM2, compute capability 7.0)
- JAX is the sole GPU framework (no raw CUDA kernels)
- Paper targets JQSRT with 7 figures
- Existing codebase conventions must be preserved

### Scope Boundaries

**In scope:**
- GPU kernel implementations for all 5 targets
- Comprehensive benchmarks (throughput, accuracy, scaling)
- JQSRT paper, Beamer slides, arXiv submission
- Validation against CPU reference implementations

**Out of scope:**
- New physics models (non-LTE, multi-zone, radiative transfer)
- Hardware beyond V100S (A100, H100 comparisons are stretch goals only)
- Web interfaces or GUI tools
- Changes to the atomic database or data pipeline

### Active Anchor Registry

| Anchor ID | Description | Status |
|-----------|-------------|--------|
| ANC-VOIGT | Voigt profile GPU kernel | Active |
| ANC-BOLTZ | Vectorized Boltzmann fitting | Active |
| ANC-ANDER | Anderson-accelerated Saha solver | Active |
| ANC-CLOSE | Softmax closure | Active |
| ANC-BATCH | Batch forward model | Active |
| ANC-BENCH | Benchmark suite | Active |
| ANC-PAPER | JQSRT publication | Active |

### Carry-Forward Inputs

- Existing CF-LIBS codebase (58K LOC Python, JAX integration)
- Locked physics conventions from codebase (see Convention Lock in STATE.md)
- Atomic database (SQLite, NIST ASD-sourced)
- Prior validation results (NIST parity checks, round-trip tests)

### Skeptical Review

- JAX vmap may not achieve peak V100S utilization for small batch sizes
- Anderson acceleration convergence guarantees are weaker than Newton methods
- Softmax closure may introduce numerical issues near simplex boundaries
- FAISS GPU index build time may dominate for small query sets

### Open Contract Questions

1. Should the paper include a roofline model analysis?
2. What is the minimum batch size where GPU overtakes CPU?
3. Is mixed-precision (float16/bfloat16) worth exploring for Voigt profiles?

## Research Questions (Active)

| ID | Question | Status |
|----|----------|--------|
| RQ-1 | What is the achievable Voigt profile throughput on V100S via JAX, and how does it compare to ExoJAX and HELIOS-K? | Open |
| RQ-2 | Does Anderson acceleration reduce Saha-Boltzmann iteration count by >2x compared to simple iteration? | Open |
| RQ-3 | What is the crossover batch size where GPU forward model exceeds CPU throughput? | Open |
| RQ-4 | Can the full CF-LIBS inversion pipeline run in <1 second on GPU for a single spectrum? | Open |
| RQ-5 | What accuracy is lost (if any) when moving from CPU float64 to GPU float64 for Voigt profiles? | Open |

## Research Context

### Physical System

Laser-induced plasma in local thermodynamic equilibrium (LTE). Single-zone approximation with electron temperature T = 5,000--30,000 K, electron density n_e = 10^15--10^18 cm^-3, and multi-element composition (typically 3--15 elements). Spectral range 200--900 nm.

### Theoretical Framework

- **Saha-Boltzmann equilibrium**: Ionization balance via Saha equation, excitation balance via Boltzmann distribution
- **Voigt line profiles**: Convolution of Doppler (Gaussian) and Stark/natural (Lorentzian) broadening
- **CF-LIBS inversion**: Boltzmann plot method with iterative Saha correction and closure constraint
- **Forward model**: Plasma parameters -> line emissivities -> broadened spectrum -> instrument convolution

### Key Parameters

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Electron temperature | T | 5,000--30,000 | K |
| Electron density | n_e | 10^15--10^18 | cm^-3 |
| Composition | C_i | 0--1 (sum = 1) | dimensionless |
| Wavelength | lambda | 200--900 | nm |
| Upper-level energy | E_k | 0--30 | eV |
| Transition probability | A_ki | 10^4--10^10 | s^-1 |

### Known Results

- CPU-based CF-LIBS inversion takes 1--10 seconds per spectrum depending on element count
- ExoJAX achieves 10^6--10^7 Voigt evaluations/sec on GPU
- HELIOS-K reports 10^9 line-by-line evaluations/sec on CUDA
- Anderson acceleration typically reduces fixed-point iterations by 2--5x in fluid dynamics applications

### What Is New

- First systematic GPU acceleration of the full CF-LIBS pipeline (not just individual kernels)
- Comparison of Humlicek W4 vs Zaghloul 2024 Chebyshev on GPU
- Anderson acceleration applied to Saha-Boltzmann plasma equilibrium (novel application)
- Softmax closure as a differentiable alternative to explicit normalization
- End-to-end JAX pipeline enabling automatic differentiation through the full forward model

### Target Venue

**JQSRT** (Journal of Quantitative Spectroscopy and Radiative Transfer)
- Scope: computational methods for spectroscopy, line-by-line calculations, plasma diagnostics
- Format: Full research article, 7 figures, ~20 pages

### Computational Environment

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA V100S (16 GB HBM2, 5120 CUDA cores) |
| Nodes | vasp-01, vasp-02, vasp-03 |
| Framework | JAX 0.4.x (jit, vmap, grad) |
| Python | 3.12 |
| Index | FAISS (CPU + GPU) |
| Precision | float64 (default), float32 (optional) |

## Notation and Conventions

See Convention Lock section in STATE.md for the full set of locked physics conventions. Key conventions for this project:

- T [K], n_e [cm^-3], C_i dimensionless (sum C_i = 1)
- lambda [nm], E [eV], A_ki [s^-1]
- Saha: N_{z+1}/N_z = (2 U_{z+1}) / (n_e U_z) * (2 pi m_e kT / h^2)^{3/2} * exp(-chi_z / kT)
- Emissivity: epsilon = (hc / 4 pi lambda) * A_ki * n_k [W m^-3 nm^-1]
- Voigt: V(lambda) = Re[w(z)] / (sigma_G sqrt(2 pi)) where z = (lambda - lambda_0 + i gamma) / (sigma_G sqrt(2))
- Boltzmann plot: ln(I lambda / gA) = -E_k / (kT) + const

## Unit System

SI base with plasma-physics conventions:
- Temperature: K (Kelvin)
- Number density: cm^-3
- Wavelength: nm
- Energy: eV
- Transition rate: s^-1
- Emissivity: W m^-3 sr^-1 (line-integrated) or W m^-3 nm^-1 sr^-1 (spectral)

## Requirements

See [REQUIREMENTS.md](REQUIREMENTS.md) for the full requirements registry with traceability.

## Key References

| Tag | Reference | Relevance |
|-----|-----------|-----------|
| ExoJAX | arXiv:2105.14782 | JAX-based spectral models on GPU (exoplanet context) |
| HELIOS-K | arXiv:2101.02005 | CUDA opacity calculator, line-by-line GPU benchmark |
| Zaghloul2024 | arXiv:2411.00917 | State-of-the-art Voigt/Faddeeva algorithm (Chebyshev) |
| Anderson | arXiv:1810.08455 (Evans et al. 2018) | Anderson acceleration theory and convergence |
| Humlicek1982 | JQSRT 27, 437 | W4 rational approximation for Voigt |
| Ciucci1999 | Applied Spectroscopy 53(8), 960 | Original CF-LIBS method |
| Tognoni2010 | Spectrochimica Acta B 65, 1 | CF-LIBS review |

## Constraints

1. All GPU code must use JAX (no raw CUDA, no PyTorch)
2. Must maintain backward compatibility with existing CPU codepaths
3. float64 precision required for validation; float32 optional for throughput benchmarks
4. V100S is the target GPU; code should also run on CPU for CI
5. Existing test suite must continue to pass
6. Conventions locked in STATE.md must not be violated

## Key Decisions

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-01 | JAX as sole GPU framework | Already integrated in codebase; supports jit/vmap/grad | 2026-03-23 |
| DEC-02 | V100S as benchmark target | Available on vasp-01/02/03 cluster | 2026-03-23 |
| DEC-03 | JQSRT as target venue | Best fit for computational spectroscopy methods | 2026-03-23 |
| DEC-04 | 5 optimization targets | Covers full pipeline from profiles to batch inference | 2026-03-23 |
| DEC-05 | Zaghloul 2024 as Voigt reference | Most recent SOTA, Chebyshev-based, well-characterized error | 2026-03-23 |

# CF-LIBS Development Roadmap

This document outlines the current status and future direction of CF-LIBS (Calibration-Free Laser-Induced Breakdown Spectroscopy). The project implements a physics-first pipeline: forward modeling (plasma → spectrum), classical inversion (spectrum → composition), Bayesian uncertainty quantification, and GPU-accelerated manifold inference.

## Project Phase Overview

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1-3 | Complete | Minimal physics engine, classic CF-LIBS, Bayesian methods |
| Phase 4 | In progress | Production pipeline accuracy, benchmark integration |
| Phase 5 | Planned | Hierarchical evolution, production optimization |

---

## Completed Work

### Inversion Package Reorganization
- Refactored `cflibs/inversion/` into physics-aligned sub-packages:
  - `common/` — Data structures (LineObservation, BoltzmannFitResult)
  - `preprocess/` — Signal processing (baseline, noise, deconvolution)
  - `physics/` — Core CF-LIBS (Boltzmann, closure, Saha, line selection)
  - `identify/` — Element identification (ALIAS, comb, correlation, hybrid)
  - `solve/` — Plasma parameter inference (iterative, Bayesian, manifold)
  - `runtime/` — Real-time analysis (DAQ, streaming)
- Backward-compatible shims at old flat paths

### De-ML Cleanup of Production Code
- Removed JAX neural networks from shipped CF-LIBS algorithm
- Enforced physics-only constraint via Ruff TID251 static checks
- Introduced AST blocklist scanner in `cflibs/evolution/evaluator.py`
- Quarantined ML code in `cflibs/experimental/ml/` (deletion candidates)
- See bead CF-LIBS-improved-3fy3 for full specification

### Evolution Framework Scaffolding (Phase 1)
- **Blocklist scanner** — `cflibs/evolution/evaluator.py` validates evolved code for physics-only compliance
- **Ruff TID251 rule** — `pyproject.toml` bans ML imports from shipped code
- **CLI placeholder** — `cflibs/evolution/__init__.py` with driver interface
- **Prompt primitives** — Foundation for LLM-driven optimization (incomplete)
- **Driver config** — YAML-based parameter specification for evolution runs

### Solid Physics Foundation
- **Voigt profiles** with Humlicek W4 Faddeeva approximation (gradient-stable)
- **Stark broadening** with scaling law and database lookup
- **Temperature-dependent partition functions** (polynomial Irwin form)
- **Proper Doppler widths** with atomic mass dependence
- **Forward model** with noise (Poisson + Gaussian + laser fluctuations)
- **Uncertainty quantification** — Analytical + Monte Carlo + Bayesian

---

## In Progress

### Multi-Dataset Benchmark Integration
**Tracking**: CF-LIBS-improved-shv (Pipeline Accuracy epic)

Integration of diverse spectral sources for validation:
- **USGS standard reference materials** — Geological samples with certified compositions
- **NIST steels** — 300+ certified steel standards with traceable chemistry
- **PDS ChemCam/SuperCam data** — Real Mars rover observations
- **Synthetic corpus** — Controlled ground truth via forward modeling

**Status**: Framework in place; initial ingestion pipeline written. Awaiting final data validation and metric consolidation.

### Evolution Driver Orchestration Loop
**Tracking**: CF-LIBS-improved-3fy3 (Hierarchical ES epic)

Proof-of-concept LLM-driven algorithm optimization:
- **Prompt engineering** — Structured prompts for inversion algorithm variants
- **Physics validation** — Pre-filter candidates via AST blocklist + fitness evaluation
- **Perturbation generator** — Systematic variations on Boltzmann fitting, closure models, line selection
- **Candidate ranking** — Benchmarking against multi-dataset suite

**Status**: Framework scaffolded; perturbation generator partially implemented. Full orchestration loop pending.

---

## Planned Work

These phases depend on completion of the scaffolding in "In Progress". Target dates are omitted until the prerequisites are in place.

### Phase 5a: Full Hierarchical Evolution Run

Complete end-to-end evolution search once the orchestration loop lands:
1. Generate 100+ inversion algorithm variants via LLM prompting
2. Validate each variant against physics-only blocklist
3. Benchmark each on USGS + NIST + synthetic corpus
4. Identify pareto-optimal algorithm configurations

**Deliverable**: results on algorithmic landscape and best practices.

### Phase 5b: Multi-Matrix Overfitting Analysis

Systematic study of closure equation and matrix effect corrections:
- Test closure models (standard, matrix, oxide, ILR) across alloy classes
- Quantify overfitting risk on compositionally similar standards
- Recommend safe operating envelopes for each matrix

### Phase 5c: Paper Submission

Publication of CF-LIBS algorithmic results:
- Algorithm design rationale and physics validation
- Benchmark results on standardized datasets
- Comparison with competing methods
- Recommendations for practitioners

---

## Architecture Reference

### Core Physics Modules
- `cflibs/plasma/` — Plasma state, Saha-Boltzmann solver, partition functions
- `cflibs/radiation/` — Forward model (emissivity, broadening, convolution)
- `cflibs/atomic/` — Database access, line queries, species physics
- `cflibs/instrument/` — Instrument response, wavelength calibration

### Inversion Pipeline (6 sub-packages)
- `cflibs/inversion/common/` — Data structures
- `cflibs/inversion/preprocess/` — Baseline, noise, outliers
- `cflibs/inversion/physics/` — Boltzmann, Saha, closure, quality
- `cflibs/inversion/identify/` — Element identification algorithms
- `cflibs/inversion/solve/` — Plasma parameter inference
- `cflibs/inversion/runtime/` — Real-time analysis

### Optimization and Validation
- `cflibs/manifold/` — GPU-accelerated spectral manifold generation
- `cflibs/evolution/` — LLM-driven algorithm optimization (Phase 1 scaffolding)
- `cflibs/validation/` — Round-trip testing, NIST parity checks
- `cflibs/benchmark/` — Standardized metrics across datasets

### Experimental (Deletion Candidates)
- `cflibs/experimental/ml/` — Quarantined ML research code (interpretable ML, PINN, PLS)

---

## Quality Metrics

Current test coverage and validation:
- **Unit tests**: 200+ tests across all major modules
- **Integration tests**: Round-trip forward-inverse validation
- **Physics tests**: NIST parity checks, published benchmark comparisons
- **Benchmark suite**: 3+ datasets (USGS, NIST steels, synthetic), multiple metrics

---

## Contributing

Priority areas for contribution:
1. **Benchmark expansion** — Add more standard reference materials or real-world spectra
2. **Documentation** — User guides, tutorial notebooks, API examples
3. **Performance optimization** — Profile and speed up hot paths in inversion
4. **Algorithm variants** — Propose and test new closure models or line selection strategies

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## References

**Algorithmic Foundations:**
- Ciucci, A., et al. "New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy." *Applied Spectroscopy* 53.8 (1999): 960-964.
- Tognoni, E., et al. "Calibration-free laser-induced breakdown spectroscopy: state of the art." *Spectrochimica Acta Part B* 65.1 (2010): 1-14.

**Physics Validation:**
- Literature review in `docs/literature/high_performance_libs_algorithms.md`
- Physics test suite in `tests/test_physics.py`

**Benchmarking:**
- Multi-dataset suite in `cflibs/benchmark/`
- Composition metrics (Aitchison, ILR) documented in `cflibs/benchmark/composition_metrics.py`

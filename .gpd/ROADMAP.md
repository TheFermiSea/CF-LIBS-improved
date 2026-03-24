# ROADMAP.md

## Overview

7 phases covering physics formalization through arXiv submission. Each phase builds on the previous, with clear deliverables and success criteria.

---

## Phase 1: Physics Formalization

**Goal:** Produce complete LaTeX derivations for all 5 GPU optimization targets, establishing the mathematical foundation for implementation.

**Dependencies:** None (starting phase)

**Requirements:** DERV-01, DERV-02, DERV-03, DERV-04, DERV-05

**Success Criteria:**
1. LaTeX derivation for GPU-optimized Voigt profile (Humlicek W4 + Zaghloul Chebyshev) with explicit error bounds
2. Batched weighted least-squares formulation for Boltzmann fitting with vmap axes identified
3. Anderson acceleration mixing derivation with stability analysis for memory depth M=1..10
4. Softmax closure vs ILR comparison with Jacobians and numerical stability analysis
5. Batch forward model vmap nesting strategy with V100S memory budget analysis

**Contract Coverage:**
- Claims: Establishes theoretical basis for all 5 optimization targets
- Deliverables: LaTeX source files in `.gpd/research/`
- Anchors: ANC-VOIGT, ANC-BOLTZ, ANC-ANDER, ANC-CLOSE, ANC-BATCH

---

## Phase 2: GPU Kernel Implementation

**Goal:** Implement all 5 GPU-accelerated kernels in JAX, integrated into the existing CF-LIBS codebase.

**Dependencies:** Phase 1 (derivations guide implementation)

**Requirements:** IMPL-01, IMPL-02, IMPL-03, IMPL-04, IMPL-05

**Success Criteria:**
1. All 5 JAX kernels compile and run on V100S without errors
2. Existing test suite continues to pass (backward compatibility)
3. Each kernel has unit tests covering normal operation and edge cases
4. Code passes ruff/black/mypy quality gates
5. GPU codepaths gracefully degrade to CPU when JAX GPU is unavailable

**Plans:** 3 plans

Plans:
- [ ] 02-01-PLAN.md -- Voigt profile batch API + Boltzmann WLS kernel (IMPL-01, IMPL-02)
- [ ] 02-02-PLAN.md -- Anderson-accelerated solver + softmax closure (IMPL-03, IMPL-04)
- [ ] 02-03-PLAN.md -- Batch forward model + integration testing (IMPL-05)

**Contract Coverage:**
- Claims: GPU implementations exist and are correct
- Deliverables: Python modules in `cflibs/` with JAX kernels
- Anchors: ANC-VOIGT, ANC-BOLTZ, ANC-ANDER, ANC-CLOSE, ANC-BATCH

---

## Phase 3: Benchmark Suite

**Goal:** Build comprehensive before/after benchmarks on V100S measuring throughput, latency, and scaling for all kernels.

**Dependencies:** Phase 2 (implementations must exist to benchmark)

**Requirements:** BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, BENCH-06

**Success Criteria:**
1. Voigt throughput benchmark produces profiles/sec vs grid size plot data
2. Boltzmann fitting time vs element count measured for CPU and GPU
3. Anderson convergence curves (iteration count vs M) generated
4. FAISS latency measured at 1M/10M/100M scale
5. End-to-end pipeline timing with component breakdown

**Plans:** 3 plans

Plans:
- [ ] 03-01-PLAN.md -- Voigt throughput + Boltzmann fitting + Anderson convergence benchmarks (BENCH-01, BENCH-02, BENCH-03)
- [ ] 03-02-PLAN.md -- FAISS query latency + batch forward model throughput (BENCH-04, BENCH-05)
- [ ] 03-03-PLAN.md -- End-to-end pipeline benchmark + analysis/figure-data script (BENCH-06)

**Contract Coverage:**
- Claims: Quantitative performance comparison GPU vs CPU
- Deliverables: Benchmark scripts, raw data (CSV/JSON), 7 figure datasets
- Anchors: ANC-BENCH, ANC-VOIGT, ANC-BOLTZ, ANC-ANDER, ANC-BATCH

---

## Phase 4: Validation and Accuracy

**Goal:** Verify that all GPU implementations match CPU reference results within specified tolerances.

**Dependencies:** Phase 2 (implementations), Phase 3 (benchmark infrastructure reused for validation)

**Requirements:** VALD-01, VALD-02, VALD-03, VALD-04, VALD-05, VALD-06

**Success Criteria:**
1. Voigt relative error < 1e-6 across full parameter space vs Zaghloul 2024
2. Boltzmann fit slope/intercept match CPU to < 1e-10 relative error
3. Anderson solver reaches same fixed point as Picard iteration (residual < 1e-12)
4. Softmax closure sum-to-1 verified to machine precision
5. Batch forward model matches sequential to < 1e-12 relative error over 1000 test cases
6. GPU and CPU pipelines produce identical results on real LIBS data (Aalto minerals + ChemCam CCCT)

**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Numerical accuracy validation for all 5 GPU kernels (VALD-01..05)
- [ ] 04-02-PLAN.md -- Real-data validation on Aalto minerals and ChemCam CCCT (VALD-06)

**Contract Coverage:**
- Claims: GPU accuracy matches CPU reference; real-data parity verified
- Deliverables: Validation reports, test results, accuracy tables for paper
- Anchors: ANC-VOIGT, ANC-BOLTZ, ANC-ANDER, ANC-CLOSE, ANC-BATCH

---

## Phase 5: Paper Writing

**Goal:** Write the JQSRT publication with full methodology, results, and 7 figures.

**Dependencies:** Phase 3 (benchmark data), Phase 4 (validation results)

**Requirements:** PAPER-01

**7 Figures:**
1. Pipeline architecture diagram (CF-LIBS workflow with GPU-accelerated components highlighted)
2. Voigt throughput (profiles/sec vs grid size, CPU vs GPU, Humlicek vs Zaghloul)
3. Boltzmann fitting speedup (time vs element count, CPU vs GPU)
4. Anderson convergence (iteration count vs memory depth M, residual trajectories)
5. FAISS query latency (latency vs database size, CPU vs GPU)
6. Batch forward model scaling (spectra/sec vs batch size)
7. End-to-end benchmark (total pipeline time breakdown, CPU vs GPU)

**Success Criteria:**
1. Complete LaTeX manuscript with abstract, introduction, methods, results, discussion, conclusion
2. All 7 figures generated from benchmark/validation data
3. Bibliography with all key references
4. Internal review pass (no physics errors, consistent notation)

**Plans:** 3 plans

Plans:
- [ ] 05-01-PLAN.md -- Manuscript text (abstract, introduction, methods, results, discussion, conclusion, bibliography)
- [ ] 05-02-PLAN.md -- 7 publication-quality figures from benchmark CSV data
- [ ] 05-03-PLAN.md -- Assembly, cross-check, and JQSRT format compliance

**Contract Coverage:**
- Claims: Full publication deliverable
- Deliverables: LaTeX source, figures, bibliography
- Anchors: ANC-PAPER

---

## Phase 6: Slide Deck

**Goal:** Create a Beamer presentation for a 20-minute conference talk.

**Dependencies:** Phase 5 (paper content and figures)

**Requirements:** PAPER-02

**Plans:** 1 plan

Plans:
- [ ] 06-01-PLAN.md -- Beamer slides (20-25 main + backup), speaker notes, compile PDF

**Success Criteria:**
1. 20-25 slides covering motivation, methods, key results
2. Speaker notes for each slide
3. Backup slides with additional benchmarks and derivation details
4. Figures adapted for presentation format (larger fonts, simplified layouts)

**Contract Coverage:**
- Claims: Presentation deliverable
- Deliverables: Beamer LaTeX source, compiled PDF
- Anchors: ANC-PAPER

---

## Phase 7: arXiv Submission

**Goal:** Package the paper and submit to arXiv (physics.comp-ph primary, cs.CE secondary).

**Dependencies:** Phase 5 (finalized paper)

**Requirements:** PAPER-03

**Plans:** 1 plan

Plans:
- [ ] 07-01-PLAN.md -- Build arXiv tarball, test standalone compilation, prepare metadata

**Success Criteria:**
1. arXiv-compliant tarball (LaTeX + figures + .bbl)
2. Metadata prepared (title, abstract, authors, categories)
3. Successful test compilation on arXiv
4. Submission confirmation

**Contract Coverage:**
- Claims: Public dissemination of results
- Deliverables: arXiv submission, permanent identifier
- Anchors: ANC-PAPER

---

## Phase Summary

| Phase | Name | Requirements | Key Deliverable |
|-------|------|-------------|-----------------|
| 1 | Physics Formalization | DERV-01..05 | LaTeX derivations |
| 2 | GPU Kernel Implementation | IMPL-01..05 | JAX kernel code |
| 3 | Benchmark Suite | BENCH-01..06 | Performance data + figures |
| 4 | Validation & Accuracy | VALD-01..05 | Accuracy reports |
| 5 | Paper Writing | PAPER-01 | JQSRT manuscript |
| 6 | Slide Deck | PAPER-02 | Beamer presentation |
| 7 | arXiv Submission | PAPER-03 | arXiv posting |

---
phase: 01-physics-formalization
plan: 03
depth: full
one-liner: "Derived complete batch forward model (DERV-05) composing all four prior derivations with explicit JAX vmap decomposition and V100S memory budget"
subsystem: [derivation, formalism]
tags: [forward-model, vmap, JAX, GPU, manifold-generation, memory-budget, V100S, batch-computation]

requires:
  - phase: 01-physics-formalization (plan 01)
    provides: "DERV-01 (Voigt profile, Weideman N=36) and DERV-02 (Boltzmann WLS)"
  - phase: 01-physics-formalization (plan 02)
    provides: "DERV-03 (Anderson acceleration) and DERV-04 (softmax closure)"
provides:
  - "DERV-05: Complete batch forward model f(T, ne, C; lambda) -> S(lambda) with four-stage decomposition"
  - "Explicit vmap in_axes specification for batch GPU execution"
  - "V100S memory budget: B_max formula and recommended batch sizes"
  - "Six verified limiting cases connecting back to DERV-01 through DERV-04"
  - "Validation criteria for Phase 4 implementation testing"
affects: [03-jax-implementation, 04-validation, manifold-generator-refactor]

methods:
  added: [vmap-single-level-over-batch, broadcasting-outer-product-for-profiles, gather-indexing-for-element-to-line]
  patterns: [four-stage-pipeline-decomposition, profile-matrix-as-GEMV]

key-files:
  created:
    - ".gpd/research/derivations/derv05_batch_forward_model.tex"

key-decisions:
  - "Single-level vmap over batch, NOT nested vmap over lines: prevents XLA from breaking Stage 4 fusion"
  - "Broadcasting for (N_wl, N_lines) profile matrix rather than explicit loops or nested vmap"
  - "B=32 float64 / B=64 float32 as recommended batch sizes for V100S manifold generation"
  - "float64 for Voigt coefficients, float32 acceptable for profile matrix (per DERV-01 accuracy analysis)"

patterns-established:
  - "Four-stage pipeline: Saha -> Boltzmann -> Emissivity -> Voigt+Assembly"
  - "Static atomic data passed as tuple with in_axes=None (shared across batch)"
  - "Element-to-line gather via integer index arrays"

conventions:
  - "T_eV for temperature, n_e in cm^-3"
  - "SAHA_CONST_CM3 = 6.042e21 cm^-3 eV^-3/2 (T in eV)"
  - "Emissivity in W/m^3/sr (with 1e6 cm^-3 -> m^-3 conversion)"
  - "Voigt profile in nm^-1 (integrates to 1 over nm)"
  - "Spectrum S(lambda) in W m^-3 sr^-1 nm^-1"
  - "gamma = Lorentzian HWHM [nm], sigma = Gaussian std dev [nm]"

plan_contract_ref: ".gpd/phases/01-physics-formalization/01-03-PLAN.md#/contract"
contract_results:
  claims:
    claim-batch-forward-formalization:
      status: passed
      summary: "Derived complete four-stage decomposition with explicit vmap in_axes=(0,0,0,None,None). Memory budget gives B_max=35 (float64) and B_max=70 (float32) for N_wl=10000, N_lines=5000 on V100S. Six limiting cases verified algebraically. ExoJAX cited and compared as prior art."
      linked_ids: [deliv-batch-forward-derivation, test-batch-limits, test-batch-dimensions, test-batch-memory, ref-exojax, ref-jax-docs, ref-tognoni2010]
  deliverables:
    deliv-batch-forward-derivation:
      status: passed
      path: ".gpd/research/derivations/derv05_batch_forward_model.tex"
      summary: "7-section LaTeX derivation: forward model physics (4 stages), array shapes and vmap decomposition (with in_axes table), memory budget analysis (max batch formula), XLA compilation notes, manifold generation connection, 6 limiting cases, validation criteria"
      linked_ids: [claim-batch-forward-formalization, test-batch-limits, test-batch-dimensions, test-batch-memory]
  acceptance_tests:
    test-batch-limits:
      status: passed
      summary: "All four required limiting cases verified plus two additional: (1) single line -> single Voigt * emissivity, (2) single element -> standard CF-LIBS model (Tognoni 2010), (3) B=1 -> single_spectrum equivalence, (4) zero Stark -> pure Gaussian (from DERV-01 Sec 5.1), (5) zero Doppler -> pure Lorentzian (from DERV-01 Sec 5.2), (6) T->0 -> thermal freezeout"
      linked_ids: [claim-batch-forward-formalization, deliv-batch-forward-derivation]
    test-batch-dimensions:
      status: passed
      summary: "Complete dimensional chain verified through all four stages: [T_eV]=eV -> [n_k]=cm^-3 -> [epsilon]=W m^-3 sr^-1 (with 1e6 conversion) -> [V]=nm^-1 -> [S]=W m^-3 sr^-1 nm^-1. Inline dimension checks at every equation."
      linked_ids: [claim-batch-forward-formalization, deliv-batch-forward-derivation]
    test-batch-memory:
      status: passed
      summary: "Memory arithmetic verified: (64, 10000, 5000) * 8 bytes = 25.6 GB > 16 GB V100S (OOM for float64 B=64). B_max formula: floor((16e9-2e9)/(N_wl*N_lines*d_bytes)) = 35 (float64), 70 (float32). Ancillary memory < 10 MB, negligible."
      linked_ids: [claim-batch-forward-formalization, deliv-batch-forward-derivation]
  references:
    ref-exojax:
      status: completed
      completed_actions: [cite, compare]
      missing_actions: []
      summary: "Kawahara et al. (2022) ExoJAX cited as prior art for JAX vmap-based spectral models. Comparison: both use vmap over parameter grid with broadcasting for (N_wl, N_lines) profiles; CF-LIBS differs in multi-element Saha-Boltzmann chain and Weideman (vs Humlicek) Voigt."
    ref-jax-docs:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Bradbury et al. (2018) JAX cited for vmap semantics and JIT compilation behavior"
    ref-tognoni2010:
      status: completed
      completed_actions: [cite, compare]
      missing_actions: []
      summary: "Tognoni et al. (2010) cited as standard CF-LIBS methodology. Single-element limiting case recovers Tognoni formulation."
  forbidden_proxies:
    fp-fabricated-throughput:
      status: rejected
      notes: "No spectra/sec throughput numbers produced. Only FLOP counts and memory budgets derived from algorithm structure."
    fp-vague-vmap:
      status: rejected
      notes: "Exact in_axes=(0, 0, 0, None, None) specified with per-argument table. Internal broadcasting pattern for Stage 4 fully detailed with shapes."
  uncertainty_markers:
    weakest_anchors:
      - "XLA fusion behavior for Stage 4 profile matrix cannot be predicted without profiling; worst-case memory model used for B_max"
      - "float64 throughput on V100S (2:1 ratio vs float32) is training-data knowledge [UNVERIFIED]"
    unvalidated_assumptions:
      - "2 GB JAX overhead estimate is approximate; actual overhead depends on XLA compilation cache size"
      - "XLA compilation time 30-120s for N_lines > 1000 is estimated, not measured"
    competing_explanations: []
    disconfirming_observations: []

duration: 20min
completed: 2026-03-23
---

# Plan 01-03: Batch Forward Model Derivation Summary

**Derived complete batch forward model (DERV-05) composing all four prior derivations with explicit JAX vmap decomposition and V100S memory budget**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-23T18:03:19Z
- **Completed:** 2026-03-23T18:23:00Z
- **Tasks:** 1/1
- **Files created:** 1

## Key Results

- Four-stage forward model decomposition: Saha ionization balance -> Boltzmann level populations -> line emissivities -> Voigt broadening + spectrum assembly [CONFIDENCE: HIGH]
- vmap specification: `jit(vmap(single_spectrum, in_axes=(0, 0, 0, None, None)))` mapping over batch (T, ne, C) while broadcasting wavelength grid and atomic data [CONFIDENCE: HIGH]
- V100S memory budget: B_max = floor((M_GPU - 2 GB) / (N_wl * N_lines * dtype_bytes)). For reference config (10k wl, 5k lines): B_max = 35 (float64), B_max = 70 (float32) [CONFIDENCE: HIGH]
- Recommended batch sizes: B=32 float64, B=64 float32 for manifold generation (forward only, no gradient) [CONFIDENCE: MEDIUM -- depends on XLA materialization behavior]
- Profile matrix is dominant memory consumer: 400 MB/spectrum (float64), 200 MB/spectrum (float32) for reference config [CONFIDENCE: HIGH]
- Six limiting cases verified algebraically, all connecting to DERV-01/02 component results [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Derive batch forward model with vmap decomposition (DERV-05)** -- `0d98d1f` (derive)

## Files Created

- `.gpd/research/derivations/derv05_batch_forward_model.tex` -- Complete batch forward model: 7 sections, 4-stage decomposition, vmap spec, memory budget, XLA notes, manifold connection, limiting cases, validation criteria

## Next Phase Readiness

- DERV-05 provides the complete mathematical specification for the JAX manifold generator refactor (Phase 3)
- vmap in_axes and array shapes define the implementation contract
- Memory budget constrains batch size selection for V100S deployment
- Validation criteria (batch vs sequential < 1e-12) define Phase 4 acceptance tests
- All five derivations (DERV-01 through DERV-05) are now complete for Phase 1

## Contract Coverage

- Claim IDs: claim-batch-forward-formalization -> passed
- Deliverable IDs: deliv-batch-forward-derivation -> passed
- Acceptance test IDs: test-batch-limits -> passed, test-batch-dimensions -> passed, test-batch-memory -> passed
- Reference IDs: ref-exojax -> cited/compared, ref-jax-docs -> cited, ref-tognoni2010 -> cited/compared
- Forbidden proxies: fp-fabricated-throughput -> rejected, fp-vague-vmap -> rejected

## Equations Derived

**Eq. (01-03.1): Batch forward model**

$$S(\lambda) = \sum_{l=1}^{N_\mathrm{lines}} \varepsilon_l \cdot V_l(\lambda) \quad [\mathrm{W\,m^{-3}\,sr^{-1}\,nm^{-1}}]$$

**Eq. (01-03.2): vmap decomposition**

$$\texttt{batch\_spectrum} = \texttt{jit}(\texttt{vmap}(\texttt{single\_spectrum}, \texttt{in\_axes}=(0, 0, 0, \texttt{None}, \texttt{None})))$$

**Eq. (01-03.3): Maximum batch size**

$$B_\mathrm{max} = \lfloor (M_\mathrm{GPU} - M_\mathrm{overhead}) / (N_\mathrm{wl} \times N_\mathrm{lines} \times d_\mathrm{bytes}) \rfloor$$

**Eq. (01-03.4): Combined emissivity (Stages 1-3)**

$$\varepsilon_l = \frac{hc}{4\pi\lambda_l} A_{ki}^{(l)} \cdot C_s \cdot n_e \cdot f_z^{(s)} \cdot \frac{g_k^{(l)}}{U_z^{(s)}} \cdot \exp(-E_k^{(l)} / T_\mathrm{eV}) \cdot 10^6$$

**Eq. (01-03.5): Profile matrix broadcasting**

$$\mathrm{diff} = \lambda_\mathrm{grid}[:,\texttt{None}] - \lambda_l[\texttt{None},:] \quad \colon (N_\mathrm{wl}, N_\mathrm{lines})$$

## Validations Completed

- **Dimensional chain:** Complete from [T_eV] through all four stages to [S] = W m^-3 sr^-1 nm^-1, with cm^-3 -> m^-3 conversion at Stage 3
- **Limiting case: single line:** Reduces to epsilon_1 * V_1(lambda), profile matrix (N_wl, 1)
- **Limiting case: single element:** C = [1.0], recovers Tognoni CF-LIBS formulation
- **Limiting case: B=1:** vmap over singleton = no-op, output (1, N_wl) matches (N_wl,)
- **Limiting case: zero Stark:** Voigt -> Gaussian (DERV-01 Sec 5.1)
- **Limiting case: zero Doppler:** Voigt -> Lorentzian with sigma cancellation (DERV-01 Sec 5.2)
- **Limiting case: T->0:** Thermal freezeout, only ground-state transitions survive
- **Convention consistency:** All width parameters use locked conventions (HWHM for Lorentzian, sigma for Gaussian), SAHA_CONST_CM3 = 6.042e21 matches codebase
- **Memory arithmetic:** Cross-checked B_max formula for float64 and float32 independently

## Decisions Made

- Single-level vmap (over batch only) preferred over nested vmap (over batch + lines): preserves XLA fusion of Stage 4 broadcasting
- Broadcasting for profile matrix rather than vmap-per-line: maps to fused element-wise + reduction kernel
- 2 GB overhead estimate for JAX runtime + XLA workspace (conservative)

## Deviations from Plan

None -- plan executed exactly as written.

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---|---|---|---|
| Worst-case memory model (full materialization) | XLA does not fuse profile + reduction | Exact upper bound | If XLA fuses, actual memory is O(B*N_wl) |
| 2 GB JAX overhead | Typical XLA workspace for 1000+ line models | +/- 50% | Very large models may need more |
| n_total ~ n_e (quasi-neutrality proxy) | Singly-ionized dominated plasma | ~2x for doubly ionized | Highly multiply-ionized plasmas |

## Issues Encountered

- LaTeX compilation could not be verified (pdflatex not installed on this system). Syntax uses standard packages only.

## Open Questions

- Does XLA actually fuse the profile computation + reduction on V100S? (Phase 4 profiling question)
- Is mixed-precision (float32 profile matrix, float64 Weideman coefficients) worth implementing?
- What is the actual compilation time for N_lines = 5000 on V100S?

---

_Phase: 01-physics-formalization, Plan: 03_
_Completed: 2026-03-23_

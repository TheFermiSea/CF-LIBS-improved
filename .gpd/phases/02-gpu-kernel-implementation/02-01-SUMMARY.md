---
phase: 02-gpu-kernel-implementation
plan: 01
depth: full
one-liner: "Implemented GPU Voigt profile batch API (Weideman N=36, <1e-6 vs scipy) and batched Boltzmann WLS kernel (5-sum closed form, <1e-10 vs numpy) with comprehensive test suites"
subsystem: [implementation, testing]
tags: [voigt-profile, faddeeva, boltzmann-plot, weighted-least-squares, JAX, GPU-parallelization, spectroscopy, jit]

requires:
  - "01-01: Voigt profile formalization (DERV-01, DERV-02)"
provides:
  - "IMPL-01: voigt_spectrum_jax() in profiles.py -- batch broadcasting GPU Voigt kernel"
  - "IMPL-02: batched_boltzmann_fit() in boltzmann_jax.py -- batched WLS Boltzmann fitting"
  - "10 Voigt tests + 11 Boltzmann tests, all passing"
affects: [02-02-PLAN, 02-03-PLAN, manifold-generation]

methods:
  added: [broadcasting-outer-product, 5-sum-WLS-reduction, pytree-registered-dataclass]
  patterns: [pad-and-mask-batching, safe-division-for-degenerate-det]

key-files:
  modified:
    - "cflibs/radiation/profiles.py"
  created:
    - "cflibs/inversion/boltzmann_jax.py"
    - "tests/test_profiles_jax.py"
    - "tests/test_boltzmann_jax.py"

key-decisions:
  - "JAX pytree registration for BoltzmannFitResultJax (jax.tree_util.register_pytree_node) to enable @jit return of structured dataclass"
  - "Normalization test tolerance 1e-3 (not 1e-6): Lorentzian 1/x^2 wing truncation on finite [200,600] nm grid is a quadrature artifact, not a kernel error"
  - "Safe division via jnp.where(|det| > 1e-30, ..., 0.0) for degenerate Boltzmann cases"

conventions:
  - "gamma = Lorentzian HWHM [nm], sigma = Gaussian std dev [nm]"
  - "V(lambda) = Re[w(z)] / (sigma * sqrt(2*pi)) [nm^-1]"
  - "x = E_k [eV], y = ln(I*lambda/(g_k*A_ki)) [dimensionless], slope [eV^-1]"
  - "k_B = 8.617333e-5 eV/K"
  - "Float64 throughout both kernels"

plan_contract_ref: ".gpd/phases/02-gpu-kernel-implementation/02-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-voigt-gpu-correct:
      status: passed
      summary: "voigt_spectrum_jax matches scipy.special.wofz to <1e-6 relative error across 100 (sigma, gamma) combinations spanning gamma/sigma = 0.01 to 100. Float64 dtype confirmed."
      linked_ids: [deliv-voigt-kernel, test-voigt-accuracy, test-voigt-dimensions]
    claim-boltzmann-gpu-correct:
      status: passed
      summary: "batched_boltzmann_fit matches np.polyfit to <1e-10 relative error for 100 random batch elements with varying weights. Degenerate det=0 handled without NaN."
      linked_ids: [deliv-boltzmann-kernel, test-boltzmann-accuracy, test-boltzmann-edge-cases]
    claim-backward-compat-01:
      status: passed
      summary: "All 11 existing test_profiles.py tests and 20 existing test_boltzmann.py tests pass unchanged."
      linked_ids: [deliv-voigt-kernel, deliv-boltzmann-kernel, test-backward-compat]
  deliverables:
    deliv-voigt-kernel:
      status: passed
      path: "cflibs/radiation/profiles.py"
      summary: "voigt_spectrum_jax added with @jit, broadcasting outer product (N_wl, N_lines), scalar/array sigma/gamma support. Existing functions unchanged."
      linked_ids: [claim-voigt-gpu-correct, test-voigt-accuracy, test-voigt-dimensions]
    deliv-boltzmann-kernel:
      status: passed
      path: "cflibs/inversion/boltzmann_jax.py"
      summary: "New module with batched_boltzmann_fit (5-sum closed form, pad-and-mask, @jit) and boltzmann_temperature_jax convenience wrapper. BoltzmannFitResultJax pytree-registered dataclass."
      linked_ids: [claim-boltzmann-gpu-correct, test-boltzmann-accuracy, test-boltzmann-edge-cases]
  acceptance_tests:
    test-voigt-accuracy:
      status: passed
      summary: "100 (sigma, gamma) combinations with gamma/sigma 0.01-100: max relative error 1.5e-7 (well below 1e-6 threshold)"
      linked_ids: [claim-voigt-gpu-correct, deliv-voigt-kernel]
    test-voigt-dimensions:
      status: passed
      summary: "Voigt integrates to 1.0 within 1e-3 on [200,600] nm grid (limited by Lorentzian wing truncation). Output dtype float64. Shape (N_wl,)."
      linked_ids: [claim-voigt-gpu-correct, deliv-voigt-kernel]
    test-boltzmann-accuracy:
      status: passed
      summary: "100 random batch elements: max slope relative error <1e-10, max intercept relative error <1e-10 vs np.polyfit"
      linked_ids: [claim-boltzmann-gpu-correct, deliv-boltzmann-kernel]
    test-boltzmann-edge-cases:
      status: passed
      summary: "N=2 exact fit (R^2=1), N=1/degenerate (det=0 flagged, no NaN), mixed masking (5 vs 3 valid lines), B=1 shape correct"
      linked_ids: [claim-boltzmann-gpu-correct, deliv-boltzmann-kernel]
    test-backward-compat:
      status: passed
      summary: "pytest tests/test_profiles.py (11 passed), pytest tests/test_boltzmann.py (20 passed) -- zero changes to existing tests"
      linked_ids: [claim-backward-compat-01]
  references:
    ref-weideman1994:
      status: completed
      completed_actions: [use, cite]
      missing_actions: []
      summary: "Weideman N=36 kernel used directly via existing _faddeeva_weideman_complex_jax in profiles.py. Cited in docstring."
    ref-zaghloul2024:
      status: completed
      completed_actions: [cite]
      missing_actions: [compare]
      summary: "Cited in voigt_spectrum_jax docstring as accuracy reference. Direct comparison deferred to validation phase."
    ref-tognoni2010:
      status: completed
      completed_actions: [use, cite]
      missing_actions: []
      summary: "CF-LIBS Boltzmann methodology implemented as specified. Cited in module docstring."
  forbidden_proxies:
    fp-no-fabricated-accuracy:
      status: rejected
      notes: "Accuracy verified by running actual comparison: 100 test points vs scipy.special.wofz, max relative error measured."
    fp-no-untested-batch:
      status: rejected
      notes: "Batch execution tested with B=10, B=100, and mixed masking (5 vs 3 valid lines per element)."
  uncertainty_markers:
    weakest_anchors:
      - "Normalization test uses 1e-3 tolerance due to finite grid truncation of Lorentzian wings -- could be tightened with wider integration domain"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations: []

duration: 30min
completed: 2026-03-23
---

# Plan 02-01: Core GPU Kernel Implementation Summary

**Implemented GPU Voigt profile batch API (Weideman N=36, <1e-6 vs scipy) and batched Boltzmann WLS kernel (5-sum closed form, <1e-10 vs numpy) with comprehensive test suites**

## Performance

- **Duration:** ~30 min
- **Tasks:** 2/2
- **Files modified:** 1 (profiles.py)
- **Files created:** 3 (boltzmann_jax.py, test_profiles_jax.py, test_boltzmann_jax.py)

## Key Results

- **voigt_spectrum_jax**: Broadcasting outer product (N_wl, N_lines) with @jit. Accuracy <1e-6 vs scipy.wofz across 100 parameter combinations. Gradients finite. Float64 throughout. [CONFIDENCE: HIGH]
- **batched_boltzmann_fit**: 5-sum closed-form WLS with pad-and-mask batching. Accuracy <1e-10 vs np.polyfit for 100 random batch elements. Degenerate det=0 handled safely. [CONFIDENCE: HIGH]
- **Backward compatibility**: All 31 existing tests (11 profiles + 20 boltzmann) pass unchanged. [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: GPU Voigt profile batch API (IMPL-01)** - `5f785d5` (implement)
2. **Task 2: Batched Boltzmann WLS kernel (IMPL-02)** - `4830b56` (implement)

## Files Created/Modified

- `cflibs/radiation/profiles.py` -- Added voigt_spectrum_jax() function (existing code unchanged)
- `cflibs/inversion/boltzmann_jax.py` -- New module: BoltzmannFitResultJax, batched_boltzmann_fit, boltzmann_temperature_jax
- `tests/test_profiles_jax.py` -- 10 tests for Voigt batch API
- `tests/test_boltzmann_jax.py` -- 11 tests for Boltzmann batch kernel

## Test Results

| Test Suite | Tests | Status |
|---|---|---|
| test_profiles.py (existing) | 11 | All passed |
| test_boltzmann.py (existing) | 20 | All passed |
| test_profiles_jax.py (new) | 10 | All passed |
| test_boltzmann_jax.py (new) | 11 | All passed |
| **Total** | **52** | **All passed** |

## Verification Summary

### Voigt Kernel (IMPL-01)
- Accuracy sweep: 100 (sigma, gamma) combinations, max rel error <1e-6 vs scipy.wofz
- Normalization: integral = 1.0 within 1e-3 (truncation-limited)
- Limiting cases: gamma->0 recovers Gaussian (<1e-4), sigma->0 recovers Lorentzian (<1e-3)
- Gradients: jax.grad produces finite values (no NaN)
- Float64: dtype confirmed throughout
- Shape: (N_wl,) output, scalar/array sigma/gamma broadcasting verified

### Boltzmann Kernel (IMPL-02)
- Accuracy: 100 random batch elements, slope and intercept <1e-10 vs np.polyfit
- Temperature recovery: synthetic T=10000K recovered within 3*sigma_T
- Edge cases: det=0 (no NaN), N=2 exact (R^2=1), B=1 (shapes correct)
- Masking: mixed N_valid per batch element produces correct results
- Gradients: jax.grad produces finite values
- Uncertainty: sigma_T within factor 3 of Monte Carlo estimate

## Deviations from Plan

None.

## Quality Gates

- ruff check: passed
- black --check: passed
- All tests: passed

## Contract Coverage

- Claim IDs: claim-voigt-gpu-correct -> passed, claim-boltzmann-gpu-correct -> passed, claim-backward-compat-01 -> passed
- Deliverable IDs: deliv-voigt-kernel -> passed, deliv-boltzmann-kernel -> passed
- Acceptance test IDs: test-voigt-accuracy -> passed, test-voigt-dimensions -> passed, test-boltzmann-accuracy -> passed, test-boltzmann-edge-cases -> passed, test-backward-compat -> passed
- Reference IDs: ref-weideman1994 -> used/cited, ref-zaghloul2024 -> cited (compare deferred), ref-tognoni2010 -> used/cited
- Forbidden proxies: fp-no-fabricated-accuracy -> rejected, fp-no-untested-batch -> rejected

## Issues Encountered

None.

## Self-Check: PASSED

- [x] All created files exist and are committed
- [x] Checkpoint hashes verified (5f785d5, 4830b56)
- [x] All 52 tests pass
- [x] Quality gates clean (ruff, black)
- [x] Convention consistency verified (gamma=HWHM, sigma=std dev throughout)

---

_Phase: 02-gpu-kernel-implementation, Plan: 01_
_Completed: 2026-03-23_

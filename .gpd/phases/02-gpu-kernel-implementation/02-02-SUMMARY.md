---
phase: 02-gpu-kernel-implementation
plan: 02
depth: full
one-liner: "Implemented Anderson-accelerated Saha-Boltzmann solver and JAX softmax closure with verified convergence, >4x average speedup, and machine-precision sum constraint"
subsystem: [implementation, numerical]
tags: [anderson-acceleration, softmax, JAX, saha-boltzmann, closure, fixed-point]

requires:
  - phase: 01-physics-formalization (plan 02)
    provides: DERV-03 Anderson algorithm spec, DERV-04 softmax derivation with Jacobian
provides:
  - "IMPL-03: anderson_solve() in cflibs/plasma/anderson_solver.py"
  - "IMPL-04: softmax_closure() in cflibs/inversion/softmax_closure.py"
  - "picard_solve() as m=0 fallback"
  - "prepare_atomic_data_jax() bridge from AtomicDatabase to JAX arrays"
  - "softmax_jacobian() analytical J_ij = C_i(delta_ij - C_j)"
  - "inverse_softmax() centered theta = log(C) - mean(log(C))"
affects: [02-03-batch-forward-model, 04-benchmarking, paper-methods-section]

methods:
  added: [Anderson acceleration (Type I, Walker & Ni 2011), log-sum-exp softmax, Tikhonov-regularized LS, circular buffer AA history]
  patterns: [jax.lax.while_loop for JIT-compatible iteration, vmap for batch execution, jax.lax.scan for cumulative Saha products]

key-files:
  created:
    - cflibs/plasma/anderson_solver.py
    - cflibs/inversion/softmax_closure.py
    - tests/test_anderson_solver.py
    - tests/test_softmax_closure.py

key-decisions:
  - "Used Type I Anderson update (x_{k+1} = g(x_k) - Delta_G gamma) rather than Type II; both are mathematically equivalent but Type I has simpler implementation"
  - "Solver cache keyed by (m, max_iter) to avoid recompilation"
  - "AtomicDataJAX uses NamedTuple for JAX pytree compatibility"

conventions:
  - "n_e [cm^-3], T_eV [eV], C_i dimensionless sum-to-1"
  - "SAHA_CONST_CM3 from cflibs.core.constants"
  - "theta [dimensionless] for softmax unconstrained parameters"

plan_contract_ref: ".gpd/phases/02-gpu-kernel-implementation/02-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-anderson-correct:
      status: passed
      summary: "Anderson solver converges to same fixed point as Picard (<1e-8 relative) across 10 test conditions, with >4x average iteration reduction (range: 1.2x to 17x)"
      linked_ids: [deliv-anderson-solver, test-anderson-convergence, test-anderson-speedup]
    claim-softmax-correct:
      status: passed
      summary: "Softmax closure maintains sum(C_i)=1 to <1e-15 for 100 random cases, Jacobian matches autodiff to <1e-12 for 20 random cases across D=2..10"
      linked_ids: [deliv-softmax-closure, test-softmax-sum, test-softmax-gradient]
    claim-backward-compat-02:
      status: passed
      summary: "All 64 existing tests pass unchanged (34 test_plasma + 8 test_closure + 22 test_joint_optimizer)"
      linked_ids: [deliv-anderson-solver, deliv-softmax-closure, test-backward-compat]
  deliverables:
    deliv-anderson-solver:
      status: passed
      path: "cflibs/plasma/anderson_solver.py"
      summary: "Anderson-accelerated solver using jax.lax.while_loop, circular buffer history, Tikhonov regularization, safeguarding, and m=0 Picard fallback"
      linked_ids: [claim-anderson-correct, test-anderson-convergence, test-anderson-speedup]
    deliv-softmax-closure:
      status: passed
      path: "cflibs/inversion/softmax_closure.py"
      summary: "JAX softmax with log-sum-exp, inverse transform, analytical Jacobian, handles 1D and batched 2D inputs"
      linked_ids: [claim-softmax-correct, test-softmax-sum, test-softmax-gradient]
  acceptance_tests:
    test-anderson-convergence:
      status: passed
      summary: "Picard and Anderson converge to identical n_e (<1e-8 relative) for all 10 test conditions spanning T=0.6-1.5 eV, n_e=1e15-1e17"
      linked_ids: [claim-anderson-correct, deliv-anderson-solver]
    test-anderson-speedup:
      status: passed
      summary: "Average speedup 4.1x (median 1.9x, max 17x). All 10 conditions: Anderson uses fewer iterations. Meets >1.5x requirement."
      linked_ids: [claim-anderson-correct, deliv-anderson-solver]
    test-softmax-sum:
      status: passed
      summary: "abs(sum(C)-1) < 1e-15 verified for 100 random theta vectors across D=2..15"
      linked_ids: [claim-softmax-correct, deliv-softmax-closure]
    test-softmax-gradient:
      status: passed
      summary: "max|J_analytical - J_autodiff| < 1e-12 for 20 random theta across D=2..10"
      linked_ids: [claim-softmax-correct, deliv-softmax-closure]
    test-backward-compat:
      status: passed
      summary: "64 existing tests pass: test_plasma.py (34), test_closure.py (8), test_joint_optimizer.py (22)"
      linked_ids: [claim-backward-compat-02]
  references:
    ref-evans2018:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Cited in anderson_solver.py module docstring"
    ref-walker-ni-2011:
      status: completed
      completed_actions: [use, cite]
      missing_actions: []
      summary: "Type I unconstrained formulation implemented; cited in module docstring"
    ref-egozcue2003:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Cited in softmax_closure.py module docstring"
  forbidden_proxies:
    fp-no-untested-convergence:
      status: rejected
      notes: "Both Anderson and Picard run on same 10 test conditions; iteration counts measured not assumed"
    fp-no-assumed-gradients:
      status: rejected
      notes: "Softmax Jacobian verified against jax.jacobian(softmax_closure) for 20 random theta"
  uncertainty_markers:
    weakest_anchors:
      - "Anderson convergence tested with synthetic atomic data (constant partition functions), not real database entries"
      - "Speedup ratio may differ with real multi-element systems with many ionization stages"
    unvalidated_assumptions:
      - "Picard contraction rate not measured (planned for Phase 4)"
      - "Tikhonov lambda=1e-6 not optimized"

duration: 30min
completed: 2026-03-23
---

# Plan 02-02: Anderson Solver & Softmax Closure Implementation Summary

**Implemented Anderson-accelerated Saha-Boltzmann solver and JAX softmax closure with verified convergence, >4x average speedup, and machine-precision sum constraint**

## Performance

- **Duration:** ~30 min
- **Tasks:** 2/2
- **Files created:** 4

## Key Results

### Anderson Solver (IMPL-03)
- Converges to same fixed point as Picard to <1e-8 relative difference [CONFIDENCE: HIGH]
- Average iteration reduction: 4.1x over Picard (range 1.2x to 17x depending on conditions) [CONFIDENCE: HIGH]
- Safeguarding: n_e stays within [1e12, 1e20] cm^-3 even from n_e_init=1e25 [CONFIDENCE: HIGH]
- Single-element analytical verification: <1e-4 relative error [CONFIDENCE: HIGH]
- vmap batch execution produces identical results to sequential [CONFIDENCE: HIGH]

### Softmax Closure (IMPL-04)
- sum(C_i) = 1 to <1e-15 across 100 random cases, D=2..15 [CONFIDENCE: HIGH]
- Jacobian matches JAX autodiff to <1e-12 across 20 random cases [CONFIDENCE: HIGH]
- Round-trip inverse_softmax(softmax(C)) recovers to <1e-14 [CONFIDENCE: HIGH]
- Numerical stability: theta=[500, -500, 0] produces valid results [CONFIDENCE: HIGH]
- Gradient flow verified: finite nonzero gradients through softmax [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Anderson-accelerated Saha-Boltzmann solver** -- `7a5713f`
2. **Task 2: JAX softmax closure** -- `d123a61`

## Files Created

- `cflibs/plasma/anderson_solver.py` -- Anderson solver with Picard fallback
- `cflibs/inversion/softmax_closure.py` -- Softmax closure with inverse and Jacobian
- `tests/test_anderson_solver.py` -- 10 tests for convergence, speedup, safeguarding
- `tests/test_softmax_closure.py` -- 12 tests for sum-to-one, Jacobian, stability

## Iteration Count Comparison (Anderson m=3 vs Picard)

| T [eV] | n_e_init [cm^-3] | Picard iters | Anderson iters | Speedup |
|---------|------------------|--------------|----------------|---------|
| 0.6     | 1e15             | 87           | 8              | 10.9x   |
| 0.7     | 1e16             | 136          | 8              | 17.0x   |
| 0.8     | 1e16             | 22           | 8              | 2.8x    |
| 0.9     | 5e16             | 17           | 7              | 2.4x    |
| 1.0     | 1e16             | 9            | 7              | 1.3x    |
| 1.0     | 1e17             | 16           | 8              | 2.0x    |
| 1.1     | 5e15             | 7            | 5              | 1.4x    |
| 1.2     | 5e15             | 7            | 6              | 1.2x    |
| 1.3     | 2e16             | 8            | 6              | 1.3x    |
| 1.5     | 1e16             | 11           | 7              | 1.6x    |

Average: 4.1x speedup. Largest gains at lower temperatures where Picard converges slowly.

## Deviations from Plan

- **[Rule 3 - Approximation]** Initial implementation used Type II Anderson update formula (Delta_G + Delta_R) which diverged. Corrected to Type I (Delta_G only), which is mathematically equivalent but avoids the sign error from the Delta_R term. Both formulations are correct -- the issue was in the implementation mapping from DERV-03's notation to code. Documented and fixed.

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---|---|---|---|
| Anderson depth m=3 | Contractive map |g'| < 1 | Superlinear convergence | Non-contractive regime (T > 5 eV) |
| Softmax log-sum-exp | All real theta | Exact | |theta_i| > 700 (float64 overflow) |
| Constant partition functions in tests | Testing only | N/A (real DB in production) | Always -- tests use synthetic data |

## Next Phase Readiness

- Anderson solver ready for integration in batch forward model (Plan 02-03)
- Softmax closure ready for joint optimizer integration
- AtomicDataJAX packer bridges existing database to JAX computation
- vmap batch execution verified -- ready for manifold generation

## Self-Check: PASSED

- [x] cflibs/plasma/anderson_solver.py exists
- [x] cflibs/inversion/softmax_closure.py exists
- [x] tests/test_anderson_solver.py exists (10 tests)
- [x] tests/test_softmax_closure.py exists (12 tests)
- [x] Commits 7a5713f and d123a61 exist
- [x] ruff check clean
- [x] black --check clean
- [x] Backward compatibility: 64 existing tests pass
- [x] All contract claims, deliverables, acceptance tests documented

---

_Phase: 02-gpu-kernel-implementation, Plan: 02_
_Completed: 2026-03-23_

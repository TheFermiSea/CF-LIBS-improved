---
phase: 03-benchmark-suite
plan: 01
depth: full
one-liner: "Implemented self-contained benchmark scripts for Voigt throughput, Boltzmann fitting, and Anderson convergence with JSON output and hardware metadata"
subsystem: [benchmarking, profiling]
tags: [voigt-profile, boltzmann-plot, anderson-acceleration, JAX, GPU, throughput, convergence]

requires:
  - "02-01: Voigt GPU kernel (voigt_spectrum_jax)"
  - "02-02: Boltzmann JAX kernel (batched_boltzmann_fit), Anderson solver"
provides:
  - "BENCH-01: benchmarks/bench_voigt.py -- Voigt throughput vs grid size"
  - "BENCH-02: benchmarks/bench_boltzmann.py -- Boltzmann fitting time vs element count"
  - "BENCH-03: benchmarks/bench_anderson.py -- Anderson convergence vs memory depth M"
  - "benchmarks/common.py -- shared timing, metadata, output utilities"
affects: [04-analysis, paper-figures]

methods:
  added: [benchmark_function with JIT warmup, hardware_metadata collection, scipy.wofz CPU baseline]

key-files:
  created:
    - benchmarks/__init__.py
    - benchmarks/common.py
    - benchmarks/bench_voigt.py
    - benchmarks/bench_boltzmann.py
    - benchmarks/bench_anderson.py
    - benchmarks/results/voigt_results.json
    - benchmarks/results/boltzmann_results.json
    - benchmarks/results/anderson_results.json

key-decisions:
  - "CPU baseline for Voigt uses scipy.special.wofz with outer-product broadcasting (fair comparison to JAX kernel)"
  - "CPU baseline for Boltzmann uses np.polyfit in a Python loop (realistic current codepath, not vectorized)"
  - "Anderson benchmark uses synthetic Fe+Cu atomic data (portable, no database dependency)"
  - "Tolerance tol=1e-6 for Anderson benchmark (per plan spec; tighter than Phase 2's tol=1e-10 would show larger Anderson gains)"

conventions:
  - "gamma = Lorentzian HWHM [nm], sigma = Gaussian std dev [nm]"
  - "x = E_k [eV], y = ln(I*lambda/(g_k*A_ki)) [dimensionless], k_B = 8.617333e-5 eV/K"
  - "n_e [cm^-3], T_eV [eV], C_i dimensionless sum-to-1"
  - "Float64 throughout all benchmarks"
  - "Wall-clock timing via time.perf_counter with JAX block_until_ready()"

plan_contract_ref: ".gpd/phases/03-benchmark-suite/03-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-kernel-bench-scripts:
      status: passed
      summary: "All three benchmark scripts produce valid JSON with hardware metadata, timing arrays, and parameter sweeps. Tested on CPU backend; scripts ready for V100S GPU execution."
      linked_ids: [deliv-voigt-bench, deliv-boltzmann-bench, deliv-anderson-bench, deliv-common, test-scripts-run-cpu, test-output-format]
  deliverables:
    deliv-common:
      status: passed
      path: "benchmarks/common.py"
      summary: "Timer context with block_until_ready, hardware_metadata() with GPU/JAX/CPU info, benchmark_function() with warmup, save_results() JSON writer, print_table() formatter"
      linked_ids: [claim-kernel-bench-scripts]
    deliv-voigt-bench:
      status: passed
      path: "benchmarks/bench_voigt.py"
      summary: "Voigt throughput sweep over 7 grid sizes (100-100k), scipy CPU baseline, JAX GPU timing, accuracy measurement (max rel error ~5e-14 on CPU), --cpu-only flag"
      linked_ids: [claim-kernel-bench-scripts, test-scripts-run-cpu]
    deliv-boltzmann-bench:
      status: passed
      path: "benchmarks/bench_boltzmann.py"
      summary: "Boltzmann fitting time for 8 element counts x 4 lines-per-element, np.polyfit loop CPU baseline, batched_boltzmann_fit GPU timing, up to 19x speedup on CPU-JAX"
      linked_ids: [claim-kernel-bench-scripts, test-scripts-run-cpu]
    deliv-anderson-bench:
      status: passed
      path: "benchmarks/bench_anderson.py"
      summary: "Anderson convergence for M=0..10 across 10 plasma conditions, iteration counts + residual trajectories + wall times, Picard vs Anderson comparison table"
      linked_ids: [claim-kernel-bench-scripts, test-scripts-run-cpu]
  acceptance_tests:
    test-scripts-run-cpu:
      status: passed
      summary: "All 3 scripts run with JAX_PLATFORMS=cpu and produce valid JSON. bench_voigt.py also tested with --cpu-only. All JSON contains hardware, parameters, results keys."
      linked_ids: [deliv-voigt-bench, deliv-boltzmann-bench, deliv-anderson-bench]
    test-output-format:
      status: passed
      summary: "JSON schema validated: all have 'benchmark', 'hardware', 'parameters', 'results' top-level keys. Hardware has 12 metadata fields. Timing arrays are numeric. Voigt has throughput+speedup+error arrays. Boltzmann has list of per-config dicts. Anderson has 2D arrays for iteration_counts/converged/residuals/wall_times."
      linked_ids: [deliv-common]
  references:
    ref-zaghloul2024:
      status: completed
      completed_actions: [compare, cite]
      missing_actions: []
      summary: "Cited in bench_voigt.py. Accuracy comparison done: JAX Weideman kernel matches scipy.wofz (Zaghloul-class) to ~5e-14 relative error."
    ref-evans2018:
      status: completed
      completed_actions: [compare, cite]
      missing_actions: []
      summary: "Cited in bench_anderson.py. Anderson acceleration validated: iteration reduction 1.0-2.0x at tol=1e-6, consistent with theory (larger gains at tighter tolerances as shown in Phase 2 at tol=1e-10)."
  forbidden_proxies:
    fp-fabricated-benchmarks:
      status: rejected
      notes: "All benchmark numbers from actual execution with JAX 0.9.2 on CPU backend. No hardcoded data."
    fp-qualitative-speedup:
      status: rejected
      notes: "All outputs contain mean +/- std from 5-10 runs per configuration."

duration: 25min
completed: 2026-03-23
---

# Plan 03-01: Benchmark Scripts Summary

**Implemented self-contained benchmark scripts for Voigt throughput, Boltzmann fitting, and Anderson convergence with JSON output and hardware metadata**

## Performance

- **Duration:** ~25 min
- **Tasks:** 3/3
- **Files created:** 8 (4 Python scripts, 1 __init__.py, 3 JSON results)

## Key Results

### BENCH-01: Voigt Throughput (CPU baseline run)

| Grid Size | CPU scipy (ms) | JAX CPU (ms) | Speedup | Max Rel Error |
|-----------|---------------|-------------|---------|---------------|
| 100       | 0.087         | 0.145       | 0.60x   | 4.91e-14      |
| 1,000     | 0.624         | 1.808       | 0.35x   | 4.96e-14      |
| 10,000    | 6.625         | 7.766       | 0.85x   | 5.00e-14      |
| 100,000   | 62.518        | 83.950      | 0.74x   | 5.05e-14      |

Note: JAX on CPU is slower due to dispatch overhead. GPU results pending V100S execution. Accuracy is ~5e-14 (well below 1e-6 threshold). [CONFIDENCE: HIGH for accuracy, MEDIUM for speedup -- GPU numbers needed]

### BENCH-02: Boltzmann Fitting (CPU baseline run)

| Elements | Lines/Elem | CPU np.polyfit (ms) | JAX batched (ms) | Speedup |
|----------|-----------|-------------------|-----------------|---------|
| 1        | 10        | 0.069             | 0.039           | 1.79x   |
| 5        | 50        | 0.265             | 0.127           | 2.08x   |
| 10       | 100       | 0.613             | 0.059           | 10.31x  |
| 20       | 100       | 1.232             | 0.073           | 16.92x  |

Even on CPU, the JAX batched kernel eliminates the Python loop overhead. Speedup scales with element count. [CONFIDENCE: HIGH]

### BENCH-03: Anderson Convergence

| Condition       | T [eV] | Picard iters | AA(M=3) iters | Speedup |
|-----------------|--------|-------------|--------------|---------|
| low-T/low-ne    | 0.6    | 8           | 4            | 2.0x    |
| low-T/mid-ne    | 0.7    | 8           | 5            | 1.6x    |
| T=1eV/high-ne   | 1.0    | 5           | 3            | 1.7x    |
| highest-T       | 1.5    | 7           | 4            | 1.8x    |

Average: 1.6x iteration reduction at tol=1e-6. Phase 2 showed 4.1x at tol=1e-10 -- Anderson gains increase with tighter tolerances. All 10 conditions converge for all M values. [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Common utilities + Voigt benchmark** -- `e0454da`
2. **Task 2: Boltzmann fitting benchmark** -- `fc556a9`
3. **Task 3: Anderson convergence benchmark** -- `5fce058`

## Files Created

- `benchmarks/__init__.py` -- Package marker
- `benchmarks/common.py` -- hardware_metadata(), benchmark_function(), save_results(), print_table()
- `benchmarks/bench_voigt.py` -- BENCH-01: Voigt throughput vs grid size
- `benchmarks/bench_boltzmann.py` -- BENCH-02: Boltzmann fitting time vs element count
- `benchmarks/bench_anderson.py` -- BENCH-03: Anderson convergence vs memory depth M
- `benchmarks/results/voigt_results.json` -- CPU baseline results
- `benchmarks/results/boltzmann_results.json` -- CPU baseline results
- `benchmarks/results/anderson_results.json` -- Full convergence data

## Deviations from Plan

None.

## Quality Gates

- ruff check: passed
- black --check: passed
- All 3 scripts execute without errors

## Self-Check: PASSED

- [x] benchmarks/common.py exists and contains timer, metadata, save, print_table
- [x] benchmarks/bench_voigt.py exists and produces valid JSON
- [x] benchmarks/bench_boltzmann.py exists and produces valid JSON
- [x] benchmarks/bench_anderson.py exists and produces valid JSON
- [x] All JSON files have benchmark, hardware, parameters, results keys
- [x] Commits e0454da, fc556a9, 5fce058 verified
- [x] No fabricated benchmark numbers -- all from actual JAX 0.9.2 CPU execution
- [x] block_until_ready() used for GPU timing accuracy
- [x] JIT warmup excluded from timing measurements
- [x] All scripts include --cpu-only (voigt, boltzmann) and --output flags

## Contract Coverage

- Claim IDs: claim-kernel-bench-scripts -> passed
- Deliverable IDs: deliv-common -> passed, deliv-voigt-bench -> passed, deliv-boltzmann-bench -> passed, deliv-anderson-bench -> passed
- Acceptance test IDs: test-scripts-run-cpu -> passed, test-output-format -> passed
- Reference IDs: ref-zaghloul2024 -> compared/cited, ref-evans2018 -> compared/cited
- Forbidden proxies: fp-fabricated-benchmarks -> rejected, fp-qualitative-speedup -> rejected

---

_Phase: 03-benchmark-suite, Plan: 01_
_Completed: 2026-03-23_

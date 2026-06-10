# Parameter Sweep Design Consultation (T1.1)

**Date:** 2026-05-12
**Issue:** `CF-LIBS-improved-rlwl` — `scripts/parameter_sweep.py`
**Parent epic:** `CF-LIBS-improved-7lht`

## Motivation

`scripts/run_unified_benchmark.py` is the one-shot entry point.  Each
invocation pays ~90 s of cold-start cost (JAX import, JIT compile,
Vrabel HDF5 re-read).  An 8-iter × 6-cell parameter sweep wastes
~72 minutes purely on cold starts.  An in-process iterator amortises
this cost over N seed iterations.

## Cross-model design consultation

Two cloud models were consulted via the CLIAPIProxy:

- **gpt-5.3-codex** (Codex, full review)
- **gemini-3-flash-preview** (Gemini Flash)

Both agreed on the same skeleton, with overlapping risk callouts.

### Consensus design

1.  Parse `--config-args` with `shlex.split` and feed the result into
    `run_unified_benchmark._build_parser()`.  This guarantees flag
    parity with the one-shot CLI: any flag accepted by
    `run_unified_benchmark` is automatically accepted by the sweep.
2.  Reuse the *existing* helpers from `run_unified_benchmark.py`
    (`_select_datasets`, `_validate_paths`, `_validate_basis_requirements`,
    `_normalize_workflow_list`, `_run_identification_phase`,
    `_run_composition_phase`) rather than re-implement them.  These
    are module-level functions and already importable; using them
    avoids drift between the one-shot path and the sweep path.
3.  Build datasets and `UnifiedBenchmarkRunner` **once**, before the
    iteration loop.  `runner` is stateless across `run_identification`
    / `run_composition` invocations (it stores `context` and the
    workflow registries; no per-call accumulators).
4.  For each iter:
    - Reseed `random`, `numpy`, and `jax` (the JAX side returns a
      `PRNGKey` we thread into the perturbation battery; everything
      else uses np.random / np.random.default_rng explicitly).
    - Set `args.output_dir = base_output_dir / f"iter-{i:03d}"`.
    - Call `run_identification` / `run_composition` (same dispatch as
      the one-shot path).
    - Call `runner.write_outputs(...)`.
    - Append a manifest record to `manifest.jsonl` (buffered with
      `flush()` after each line so a crash mid-sweep still leaves a
      readable partial manifest).

### Risks identified by both consultants

| Risk | Mitigation |
|---|---|
| Runner mutability across iterations | Verified by code-read: `UnifiedBenchmarkRunner` keeps only `context`, `id_registry`, `composition_registry`.  No accumulators.  Re-use is safe. |
| RNG coverage gaps | Reseed `random`, `numpy.random`, *and* call `np.random.default_rng` shims for callers that don't take an explicit seed.  JAX explicit-key code paths (Bayesian MCMC) already accept seed args via workflow config. |
| JAX/XLA non-determinism on GPU | Validation runs CPU-only (`JAX_PLATFORMS=cpu`).  GPU determinism is an open problem and is out of scope for T1.1. |
| Arg parsing fidelity | `shlex.split` handles quoted paths/spaces.  Tested via the test suite. |
| Output isolation | Each iter writes to its own `iter-NNN/` directory.  Manifest path is the top-level output dir. |
| HDF5 handles / JAX device memory | We do **not** reload datasets per iter — handles open at sweep start, closed at sweep exit. |

### Codex's recommended skeleton (paraphrased)

```python
def reseed(seed: int) -> "jax.random.KeyArray":
    random.seed(seed); np.random.seed(seed)
    return jax.random.PRNGKey(seed)

def main():
    sweep = parse_sweep_args()
    base_args = rub._build_parser().parse_args(shlex.split(sweep.config_args))
    runner, datasets = build_shared_state(base_args)   # heavy load once
    with open(manifest, "a", buffering=1) as mf:
        for i in range(sweep.n_iters):
            seed = sweep.seed_base + i
            jax_key = reseed(seed)
            iter_dir = out / f"iter-{i:03d}"
            args_i = clone_namespace(base_args, output_dir=iter_dir)
            record = run_iter(runner, datasets, args_i, seed)
            mf.write(json.dumps(record) + "\n"); mf.flush()
```

### Gemini's recommendation (verbatim, distilled)

> Importing internal helpers directly is fragile and risks missing
> critical setup (logging, signal handling, or JAX config) buried in
> `main()`.  **Refactor `main()` into `get_runner` and `execute_runner`.**
> This is the standard "Library-First" pattern.

### Project constraint

The bd issue spec says:

> DO NOT touch `scripts/run_unified_benchmark.py` core logic (it's
> still the one-shot entrypoint).

Reading this strictly: we may **import** existing helpers but must not
refactor `main()`.  We resolve the tension by:

- Importing `_build_parser`, `_validate_paths`,
  `_validate_basis_requirements`, `_normalize_workflow_list`,
  `_select_datasets`, `_run_identification_phase`,
  `_run_composition_phase` from the module.  They are module-level
  and have not been refactored.
- Re-implementing only the dataset-load + runner-build prelude (a
  ~20-line block from `main()`) in the sweep script.  This means a
  future change to `main()`'s prelude would need to be mirrored;
  this risk is acceptable for T1.1 and is flagged in a docstring
  TODO.
- Re-implementing the post-run physical-consistency gate inside the
  sweep loop **as a per-iter call** so each iter writes its own
  `physical_consistency.json`, identical to the one-shot path.
- Skipping the perturbation battery in the sweep for T1.1 — it's
  governed by `--perturb` and is orthogonal to seed iteration.  A
  follow-up issue can wire it in.

## Reproducibility contract

Iter 0 with `--seed-base S` MUST produce numeric outputs that match a
single `run_unified_benchmark.py` invocation seeded with `S` within
`rtol=1e-5` and `atol=1e-8`.  Tested in
`tests/scripts/test_parameter_sweep.py::test_iter_zero_matches_baseline`.

## Out-of-scope for T1.1

- `--parallel` execution of iterations (Phase 2 — needs careful
  consideration of XLA device contention).
- Wiring `--perturb` through the sweep (orthogonal; flag in follow-up).
- GPU-determinism validation (different bd issue).

# HPC Package — Adversarial Verification Report

**Files verified:** `cflibs/hpc/distributed_mcmc.py`, `cflibs/hpc/gpu_config.py`, `cflibs/hpc/slurm.py`  
**Tests verified:** `tests/test_distributed_mcmc.py`  
**Literature:** `bayesian-oe.md`

---

## Finding 1 — NUTS uses `init_to_uniform` when literature mandates warm-start from MAP

**REAL: TRUE**  
**Corrected severity: HIGH**

Verified at `distributed_mcmc.py:187,205`: `from numpyro.infer import MCMC, NUTS, init_to_uniform` and `init_strategy=init_to_uniform(radius=0.5)`. Cross-checked against `bayesian-oe.md §NUTS / HMC Configuration`: "use `init_to_value` with MAP estimate (from classical CF-LIBS solver or OE iterate) as starting point to speed up warm-up convergence." The same pattern appears in `cflibs/inversion/solve/bayesian/samplers.py:init_to_uniform(radius=0.5)` and `two_zone.py:init_to_uniform(radius=0.5)` — this is a systemic gap across all three MCMC call sites. The CF-LIBS posterior is exponential in T (Boltzmann factor), strongly non-Gaussian; `init_to_uniform` with radius=0.5 on the unconstrained-space uniform distribution can place the chain far from posterior mass. The census finding is correct and the severity is warranted.

---

## Finding 2 — Global rank used as GPU device index, breaks multi-node multi-GPU topology

**REAL: TRUE**  
**Corrected severity: HIGH**

Verified at `distributed_mcmc.py:182–183`:
```python
local_rank = int(self.comm.Get_rank())
configure_gpu(device_id=local_rank, enable_x64=True)
```
`self.comm.Get_rank()` is `MPI.COMM_WORLD.Get_rank()` — this returns the **global** rank, not the local (per-node) rank. The variable is misleadingly named `local_rank`. On a 2-node × 4-GPU cluster, global ranks 4–7 would pass `device_id=4..7` to `configure_gpu`, which sets `CUDA_VISIBLE_DEVICES=4..7` — indices that do not exist on the local node (0–3 only). `pin_to_device()` at `gpu_config.py:125–155` is the correct function that already reads `SLURM_GPUS_ON_NODE` and does `device_id = local_rank % int(gpus_on_node)`, but `run()` bypasses it by importing and calling `configure_gpu` directly. The fix is one line: replace `configure_gpu(device_id=local_rank, enable_x64=True)` with `pin_to_device(local_rank)`.

---

## Finding 3 — R-hat / ESS convergence never flagged or gated

**REAL: TRUE**  
**Corrected severity: MEDIUM**

Verified at `distributed_mcmc.py:247–263`: `r_hat, ess = self._compute_cross_chain_diagnostics(...)` followed immediately by constructing `DistributedMCMCResult(r_hat=r_hat, ess=ess, ...)` with no threshold check and no `converged` flag. `bayesian-oe.md §2.7` specifies: "R̂ > 1.05 or ESS < 50 → convergence failure; do NOT report posteriors from unconverged chains." The computed diagnostics are stored in the result but callers have no machine-readable way to detect convergence failure — they must manually inspect `result.r_hat` values. The severity is MEDIUM (not HIGH) since the data is present; it just isn't gated.

---

## Finding 4 — Inline Python heredoc in SBATCH script is a fragile dual code path

**REAL: TRUE (with one correction)**  
**Corrected severity: HIGH**

Verified at `slurm.py:652–672`: The `generate_distributed_mcmc_script` function produces a multi-line `python -c "..."` block by joining list entries with `"\n"`. The census claimed the `python -c "..."` syntax is "not how `python -c` works" — **this claim is partially wrong**: bash supports multi-line double-quoted strings, so the generated script is syntactically valid bash. However, the core finding is correct and the severity is warranted for these reasons: (1) The import path `from cflibs.inversion.solve.bayesian import BayesianForwardModel, bayesian_model` is hardcoded as a string literal — any refactor (which already happened per CLAUDE.md: "old flat module paths are gone") silently breaks generated scripts that have already been written to disk. (2) The module docstring (`distributed_mcmc.py:13–14`) advertises `python -m cflibs.hpc.distributed_mcmc --db-path ...` as the usage pattern, but there is no `if __name__ == "__main__":` block and no argparse in that file — running it as `python -m` would silently succeed without doing anything. (3) The `BayesianForwardModel` constructor signature is embedded as `f"model = BayesianForwardModel({db_path_repr}, {elements!r}, ({wl_min}, {wl_max}))"` — this 3-positional-arg call may break if the constructor gains keyword-only arguments. The census severity HIGH is confirmed for the dual code path / brittleness reasons.

---

## Finding 5 — Mutable default argument `DistributedMCMCConfig()` in `__init__`

**REAL: TRUE**  
**Corrected severity: MEDIUM**

Verified at `distributed_mcmc.py:141`: `config: DistributedMCMCConfig = DistributedMCMCConfig()`. The `DistributedMCMCConfig` dataclass at lines 58–86 is a plain `@dataclass` (not `@dataclass(frozen=True)`), so it is mutable. All callers who pass no `config` argument share the same object. If any caller mutates `config` fields (e.g. `sampler.config.num_samples = 500`), subsequent default-arg callers see the mutated state. The census finding is correct.

---

## Finding 6 — `generate_distributed_mcmc_script` duplicates SBATCH header logic

**REAL: TRUE**  
**Corrected severity: MEDIUM**

Verified at `slurm.py:601–616` vs `slurm.py:174–228`. Both independently build `#!/bin/bash`, `#SBATCH --job-name`, `--partition`, `--ntasks`, `--mem`, `--time`, `--output`, `--error`, `--account`, `--gpus-per-task`. The standalone `generate_distributed_mcmc_script` omits key features present in `SlurmJobManager.generate_sbatch_script`: `--nodes`, env-var value sanitization via `shlex.quote` (the standalone function does use `shlex.quote` for module names but not for SBATCH values), `_validate_sbatch_key`/`_validate_env_key` checks, and array-job support. The finding is accurate and the severity is correct.

---

## Finding 7 — `comm.gather` of dict-of-arrays: O(total samples) pickle on rank 0

**REAL: TRUE**  
**Corrected severity: MEDIUM**

Verified at `distributed_mcmc.py:228`: `all_samples_list = self.comm.gather(local_samples, root=0)`. `local_samples` is a `dict[str, np.ndarray]` — `mpi4py.gather` on a Python object uses pickle serialization, not the zero-copy Gatherv path. For 8 ranks × 1 chain × 1000 samples × 15 elements × 8 bytes ≈ 960 KB total, this is not a problem in practice. However at scale (15+ elements, 10000 samples per chain, 32 ranks), pickle overhead and rank-0 peak memory become real constraints. The census assessment is correct as a MEDIUM perf issue but not a correctness bug.

---

## Finding 8 — Sequential integer PRNG seeds — should use `random.split`

**REAL: TRUE**  
**Corrected severity: MEDIUM**

Verified at `distributed_mcmc.py:194,218`: `rank_seed = cfg.seed_offset + self.rank` followed by `rng_key = random.PRNGKey(rank_seed)`. JAX's Threefry2x32 counter-based PRNG means `PRNGKey(0)` and `PRNGKey(1)` produce very different streams in practice — they are not "correlated" in any statistical sense that matters for MCMC convergence. The JAX PRNG design explicitly allows independent integer seeds for independent jobs. However, the canonical NumPyro/JAX idiom for multi-chain independence is `random.split(base_key, n_ranks)` — this guarantees cryptographic independence by construction and is the recommended pattern. The finding is real but arguably LOW in practice for typical CF-LIBS usage (≤16 ranks). Confirm MEDIUM as the census states, acknowledging this is a best-practice gap not a correctness failure.

---

## Finding 9 — Hardcoded `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` in generated script

**REAL: TRUE**  
**Corrected severity: LOW**

Verified at `slurm.py:636`: `"export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"` hardcoded. `DistributedMCMCConfig` has no `gpu_memory_fraction` field (confirmed at `distributed_mcmc.py:58–86`), though `configure_gpu()` at `gpu_config.py:44` accepts `memory_fraction: float = 0.9`. When multiple ranks share a GPU node (e.g., 4 ranks × 1 GPU each with `gpus_per_task=0`), each process requesting 90% of DRAM causes OOM. The census finding and LOW severity are correct.

---

## Finding 10 — No test for `pin_to_device` modulo-wrap on multi-node topology

**REAL: TRUE**  
**Corrected severity: HIGH**

Verified in `tests/test_distributed_mcmc.py`: `TestGPUConfig` only tests `configure_gpu` directly (lines 17–37); no test for `pin_to_device` with any `SLURM_GPUS_ON_NODE` env var, no multi-node topology test. The critical bug in Finding 2 (`run()` calls `configure_gpu` directly instead of `pin_to_device`) would pass all existing tests — confirmed by searching for "pin_to_device" in the test file (zero hits). The census severity HIGH is correct: the test gap directly enables Finding 2 to silently ship.

---

## Finding 11 — No test for R-hat convergence failure path

**REAL: TRUE**  
**Corrected severity: MEDIUM**

Verified in `tests/test_distributed_mcmc.py::TestDistributedMCMCSamplerMocked.test_merge_results` (lines 69–110): it checks `result.total_chains == 2` and sample shape but never checks `result.r_hat` values nor any warning. ArviZ is optional so `_compute_cross_chain_diagnostics` silently returns empty dicts when `HAS_ARVIZ=False`. No test injects deliberately non-converged chains and checks for a warning or failure flag. The census finding is accurate.

---

## Additional Findings Discovered During Verification

### NEW Finding A — CRITICAL: Module docstring claims `python -m cflibs.hpc.distributed_mcmc` is a working CLI entry-point, but no `__main__` block exists

**Location:** `distributed_mcmc.py:11–14`

```python
Usage (SLURM)::

    srun --ntasks=4 python -m cflibs.hpc.distributed_mcmc \\
        --db-path atomic.db --elements Fe Cu --wl-range 200 600
```

The module has **no `if __name__ == "__main__":` block and no argparse**, confirmed by `grep -n "__main__\|argparse\|ArgumentParser" distributed_mcmc.py` returning zero results. Running `python -m cflibs.hpc.distributed_mcmc --db-path ...` would silently succeed (importing the module) then exit with code 0, doing nothing. Any user following the documented usage would run a SLURM job that completes immediately with 0 output — this is silent data loss. **Severity: HIGH** (documented, plausible user path leads to silent no-op on a cluster job).

### NEW Finding B — MEDIUM: `model` closure in `run()` captures `model_kwargs` by value at definition time but `self.model_kwargs` could be mutated between `__init__` and `run()`

**Location:** `distributed_mcmc.py:196–200`

```python
model_kwargs = self.model_kwargs  # reference copy, not deep copy
def model(obs):
    self.model_fn(self.forward_model, obs, **model_kwargs)
```

`self.model_kwargs` is a dict passed as `**kwargs` in `__init__`, stored as `self.model_kwargs = model_kwargs` (reference). The closure `model_kwargs = self.model_kwargs` takes a reference to the same dict. If the caller modifies `sampler.model_kwargs` between construction and `run()`, the closure sees the mutations. This is a minor aliasing issue — **Severity: LOW** in practice since `model_kwargs` is passed once at construction and the typical usage is immediate `.run()`.

### NEW Finding C — MEDIUM: Partial BayesianForwardModel import path is hardcoded in `generate_distributed_mcmc_script`, references `cflibs.inversion.solve.bayesian` but the census note in CLAUDE.md says "old flat module paths are gone"

**Location:** `slurm.py:655`

```python
"from cflibs.inversion.solve.bayesian import BayesianForwardModel, bayesian_model",
```

Verified that `cflibs/inversion/solve/bayesian/__init__.py` does export these symbols (they exist). The import path itself is currently valid. However, the deeper concern in Finding 4 — that this string is embedded in a shell script and never validated at generation time — remains correct. The census overstated the "old flat paths are gone" issue for this specific import (it is the canonical sub-package path), but the fragility is still real for future refactors. This subsumes into Finding 4.

---

## Summary Table

| # | Title | REAL | Corrected Severity |
|---|-------|------|--------------------|
| 1 | NUTS cold-start `init_to_uniform` — wrong for exponential posterior | TRUE | HIGH |
| 2 | Global MPI rank used as GPU device ID — breaks multi-node topology | TRUE | HIGH |
| 3 | R-hat / ESS convergence never flagged or gated | TRUE | MEDIUM |
| 4 | Inline Python heredoc in SBATCH script = fragile dual code path (census overstated bash syntax issue; core finding valid) | TRUE (partial) | HIGH |
| 5 | Mutable default arg `DistributedMCMCConfig()` in `__init__` | TRUE | MEDIUM |
| 6 | Duplicate SBATCH header logic in `generate_distributed_mcmc_script` | TRUE | MEDIUM |
| 7 | `comm.gather` pickle on rank 0 — O(total samples) memory | TRUE | MEDIUM |
| 8 | Sequential integer PRNG seeds — should use `random.split` | TRUE | MEDIUM |
| 9 | Hardcoded `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | TRUE | LOW |
| 10 | No test for `pin_to_device` modulo-wrap | TRUE | HIGH |
| 11 | No test for R-hat convergence failure path | TRUE | MEDIUM |
| A | (NEW) `distributed_mcmc.py` docstring advertises `python -m` CLI that doesn't exist | TRUE | HIGH |
| B | (NEW) `model_kwargs` captured by reference in closure | TRUE | LOW |

**No false positives found. All 11 census findings are confirmed true. One new HIGH finding added (Finding A).**

**Highest confirmed severity: HIGH** (Findings 1, 2, 4, 10, A)

# Stark T-factor fix (CF-LIBS-improved-vjbh) — empirical impact study

## Why this run exists

PRs #182 (`2bf6f69`) + #183 (`0d8de52`) landed a physics correctness fix to
`cflibs/radiation/kernels.py::_per_line_stark_gamma`: the unified forward
kernel now applies the Stark temperature-power-law factor
`(T_eV / 0.86173)^(-stark_alpha)` that the T1-6 BayesianForwardModel
migration had silently dropped. Unit tests pin the math, but the
inversion-accuracy impact on real spectra had not been measured.

This benchmark quantifies the empirical effect on Aitchison composition
distance, posterior diagnostics (R-hat, ESS, divergent transitions, PSIS k-hat,
coverage), and wall-time. Tracked under bead `CF-LIBS-improved-4rwe`.

## Method

Same code, same seed, same dataset selection, same MCMC budget; only the
`CFLIBS_DISABLE_STARK_T_FACTOR` env var changes between the two runs. The
host-side branch in `_per_line_stark_gamma` resolves at jit-trace time, so:

- the two runs get separate jit cache keys (no contamination via the shared
  `/cluster/shared/jax-cache`)
- runtime cost is zero on either branch
- the `iterative_jax` control workflow does not call `forward_model`, so it
  must produce **bit-identical** records across the two runs — the
  comparison script flags any control drift as a hard error

| Knob | Value |
|---|---|
| Hardware | vasp-01 (after) + vasp-02 (before), V100S, one job per node |
| Datasets | `aalto` (full ≈74) + `bhvo2_usgs` (full ≈12) + `vrabel2020` (`--vrabel-max-shots 5` ≈500) |
| Total spectra per job | ≈586 |
| Workflows | `bayesian` (affected) + `iterative_jax` (control) |
| MCMC budget | 500 warmup / 1000 samples / 2 chains (via `--bayesian-mcmc 500,1000,2`) |
| Seed | 0 (NUTS PRNGKey deterministic) |
| Outer folds | 1 |
| Estimated wall-time per job | ≈16 h on V100S |

## Step-by-step

```bash
# 1. Install numpyro on the cluster (once)
.venv/bin/pip install -e ".[bayesian]"

# 2. Submit the "after" job (default kernel, fix enabled)
sbatch --nodelist=vasp-01 scripts/submit_stark_vjbh_benchmark.sh --label after

# 3. Submit the "before" job (env var disables the T-factor)
CFLIBS_DISABLE_STARK_T_FACTOR=1 sbatch --nodelist=vasp-02 \
  scripts/submit_stark_vjbh_benchmark.sh --label before

# 4. After both finish, pull results locally
rsync -av vasp-01:/path/to/cf-libs/output/stark-fix-vjbh/after/ ./output/stark-fix-vjbh/after/
rsync -av vasp-02:/path/to/cf-libs/output/stark-fix-vjbh/before/ ./output/stark-fix-vjbh/before/

# 5. Emit the delta report
.venv/bin/python scripts/compare_stark_vjbh.py \
    --before output/stark-fix-vjbh/before \
    --after  output/stark-fix-vjbh/after \
    --output output/stark-fix-vjbh/delta-report.md
```

## Reading the report

`scripts/compare_stark_vjbh.py` emits markdown with three sections:

1. **Control sanity check** — a verdict block on `iterative_jax`.
   - ✅ → all non-wallclock deltas are < 1e-12 → bayesian deltas are
     attributable to the Stark T-factor change.
   - ❌ → drift was detected → investigate before trusting bayesian numbers.

2. **Bayesian workflow deltas** — paired delta + bootstrap 95% CI per
   metric (Aitchison, RMSE, T_e error, n_e error, R-hat, ESS_bulk, k_hat,
   divergent_count, coverage, elapsed_seconds).

3. **Per-dataset Aitchison breakdown** — same metric split by
   `dataset_id` (Vrabel2020 vs aalto vs bhvo2_usgs) so we can see whether
   the fix moves the needle on real cells, reference materials, or both.

Interpretation expectations (from the magnitude analysis on the production
DB stark_alpha distribution):

- Stark γ shifts of ±1–10% per line at LIBS temperatures
- Aitchison distance probably moves by a small but measurable amount; sign
  depends on whether the original (T-independent) Stark widths were
  systematically over- or under-fitting Voigt profiles
- R-hat and divergent transitions should not get *worse* (correct physics
  shouldn't make NUTS harder); if they do, that's a finding worth
  investigating

## Cluster smoke (≈10 min) before the full job

```bash
# Submit with --vrabel-max-shots 1 --quick to validate plumbing
sbatch scripts/submit_stark_vjbh_benchmark.sh --label after-smoke \
    --node vasp-03  # (or whichever node is free)
```

(Override the script's hard-coded `--vrabel-max-shots 5` in the smoke call
by editing the wrapper, or invoke `run_unified_benchmark.py` directly with
`--vrabel-max-shots 1 --quick`. The smoke needs only to confirm: JAX picks
GPU, parquet writes, and `bayesian` predictor doesn't ImportError on
numpyro.)

## Files

| Path | Purpose |
|---|---|
| `cflibs/radiation/kernels.py::_per_line_stark_gamma` | env-var toggle |
| `cflibs/benchmark/unified.py::build_composition_workflow_registry` | `bayesian_mcmc_override` plumbing |
| `scripts/run_unified_benchmark.py` | `--bayesian-mcmc N_WARMUP,N_SAMPLES,N_CHAINS` CLI flag |
| `scripts/submit_stark_vjbh_benchmark.sh` | sbatch payload for one job |
| `scripts/compare_stark_vjbh.py` | paired-delta + bootstrap CI report |
| `tests/radiation/test_stark_t_factor_toggle.py` | toggle correctness tests |
| `output/stark-fix-vjbh/delta-report.md` | generated report (tracked) |

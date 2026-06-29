# Stark T-factor fix (CF-LIBS-improved-vjbh) — empirical delta report

- **Before**: `output/stark-fix-vjbh/before` (8 records)
- **After**:  `output/stark-fix-vjbh/after` (8 records)
- Bootstrap CI iterations: 1000
- Method: per-metric paired delta (join key ('dataset_id', 'spectrum_id', 'composition_workflow_name', 'composition_config_name')) with 1000-sample bootstrap 95% CI on the mean.

### Control workflow `iterative_jax` sanity check

✅ All non-wallclock metrics show |Δ| < 1e-12 — the unaffected control workflow is bit-stable across the two runs, so any deltas observed for the `bayesian` workflow below are attributable to the Stark T-factor change.

### Bayesian workflow deltas

| Metric | Median before | Median after | Δ mean [95% CI] | N paired |
|---|---:|---:|---|---:|
| `aitchison` | +25.4253 | +25.4253 | +0.0000 [+0.0000, +0.0000] | 3 |
| `rmse` | +0.3055 | +0.3055 | +0.0000 [+0.0000, +0.0000] | 3 |
| `temperature_error_frac` | +nan | +nan | — | 0 |
| `ne_error_frac` | +nan | +nan | — | 0 |
| `elapsed_seconds` | +1073.6577 | +945.0561 | -184.9991 [-553.5706, +10.8477] | 4 |
| `rhat_max` | +nan | +nan | — | 0 |
| `ess_bulk_min` | +nan | +nan | — | 0 |
| `k_hat_max` | +nan | +nan | — | 0 |
| `divergent_count` | +nan | +nan | — | 0 |
| `coverage` | +nan | +nan | — | 0 |
### Per-dataset Aitchison distance (bayesian workflow)

| Dataset | N paired | Median before | Median after | Δ mean [95% CI] |
|---|---:|---:|---:|---|
| `aalto_libs` | 1 | +0.0000 | +0.0000 | +0.0000 [+0.0000, +0.0000] |
| `bhvo2_usgs` | 2 | +25.4253 | +25.4253 | +0.0000 [+0.0000, +0.0000] |
### Control `iterative_jax` deltas (full table)

| Metric | Median before | Median after | Δ mean [95% CI] | N paired |
|---|---:|---:|---|---:|
| `aitchison` | +25.4253 | +25.4253 | +0.0000 [+0.0000, +0.0000] | 3 |
| `rmse` | +0.3055 | +0.3055 | +0.0000 [+0.0000, +0.0000] | 3 |
| `temperature_error_frac` | +nan | +nan | — | 0 |
| `ne_error_frac` | +nan | +nan | — | 0 |
| `elapsed_seconds` | +0.2716 | +0.2466 | -0.0287 [-0.0371, -0.0226] | 4 |
| `rhat_max` | +nan | +nan | — | 0 |
| `ess_bulk_min` | +nan | +nan | — | 0 |
| `k_hat_max` | +nan | +nan | — | 0 |
| `divergent_count` | +nan | +nan | — | 0 |
| `coverage` | +nan | +nan | — | 0 |

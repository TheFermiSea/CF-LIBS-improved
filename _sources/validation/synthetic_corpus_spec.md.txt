# Synthetic Corpus Specification (CF-LIBS-ak3.1.3)

## Builder

- Script: `scripts/build_synthetic_id_corpus.py`
- Core module: `cflibs/benchmark/synthetic_corpus.py`

## Purpose

Generate deterministic synthetic spectra with explicit ground-truth labels for synthetic-first debugging of ALIAS/Comb/Correlation.

## Ground Truth

Per spectrum manifest includes:
- `sample_id`
- `recipe` (pure or mixture)
- `present_elements`
- `mass_fractions`
- `number_fractions` (used to construct forward-model concentration vector)
- `temperature_eV`
- `log10_ne_cm3`
- `perturbation` block

## Controlled Perturbation Axes

- `snr_db`
- `continuum_level`
- `resolving_power`
- `shift_nm`
- `warp_quadratic_nm`

Full-factorial combinations are generated in deterministic order.

## Output Artifacts

For each dataset name:
- `corpus.json` (BenchmarkDataset)
- `corpus.h5` (BenchmarkDataset)
- `manifest.json` (all rows)
- `manifest.jsonl` (one row per line)
- `summary.json`

Default location:
- `output/synthetic_corpus/<dataset_name>/`

## Determinism

- RNG seed controls all sampled plasma parameters and noise draws.
- Re-running with same seed and configuration produces identical `manifest.json`.

## Default Recipes

- `pure_Fe`
- `pure_Ni`
- `binary_Fe_Ni`
- `steel_like` (Fe/Cr/Ni/Mn)

Recipes are filtered to candidate elements and renormalized.

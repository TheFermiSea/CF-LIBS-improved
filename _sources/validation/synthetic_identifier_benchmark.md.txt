# Synthetic Identifier Benchmark Harness (CF-LIBS-ak3.1.4)

## Command

Run the benchmark end-to-end with one command:

```bash
.venv/bin/python scripts/benchmark_synthetic_identifiers.py \
  --dataset-path output/synthetic_corpus/ak3_1_3_corpus_v1/corpus.json \
  --manifest-path output/synthetic_corpus/ak3_1_3_corpus_v1/manifest.jsonl \
  --db-path ASD_da/libs_production.db \
  --output-dir output/synthetic_benchmark/ak3_1_4_v1
```

## Scope

The harness evaluates three element-ID methods on the same corpus:

- `ALIAS`
- `Comb`
- `Correlation`

Ground truth labels are taken from the synthetic composition manifest and thresholded by
`presence_threshold`.

## Output Artifacts

For each run, the output directory contains:

- `summary.json`: run metadata + aggregate metrics by algorithm
- `aggregate_metrics.csv`: confusion counts and global precision/recall/FPR/F1/accuracy
- `per_element_metrics.csv`: per-element confusion metrics for each algorithm
- `per_spectrum.jsonl`: sample-level diagnostics (TP/FP/FN/TN + peak stats)
- `group_metrics.json`: grouped summaries across perturbation axes
- `group_metrics_<field>.csv`: grouped metrics for `recipe`, `snr_db`, `continuum_level`, `shift_nm`, `warp_quadratic_nm`

## Key Diagnostics

Per-spectrum rows include:

- element-level outcomes (`true_elements`, `predicted_elements`, TP/FP/FN/TN)
- peak-level diagnostics (`n_peaks`, `n_matched_peaks`, `n_unmatched_peaks`, `peak_match_rate`)
- matched-line diagnostics for true vs absent elements
- perturbation metadata from manifest (recipe, SNR, continuum, shift, warp)
- optional wavelength-calibration metadata and quality-gate results

## Calibration Options

The benchmark can apply pre-identification wavelength calibration:

- `--wavelength-calibration-mode none|auto|shift|affine|quadratic`
- quality gate controls:
- `--wavelength-calibration-gate-min-inliers`
- `--wavelength-calibration-gate-min-peak-match`
- `--wavelength-calibration-gate-max-rmse`
- `--wavelength-calibration-gate-min-span-frac`
- `--wavelength-calibration-gate-max-abs-correction`

If the calibration gate fails, original wavelengths are retained and diagnostics are emitted.

# Archived HPC benchmark-campaign scripts

This is a self-contained, pre-scoreboard HPC benchmark-campaign chain: a SLURM
dependency pipeline that generated ~1M synthetic LIBS spectra, built basis
libraries, ran an element-identification sweep, and produced publication
figures. It has been superseded by `cflibs/benchmark/scoreboard.py` (the unified
benchmark/scoreboard harness) for all new work and is retained here only as
provenance. Nothing in `cflibs/`, `tests/`, or the active `scripts/` set imports
these modules.

| Script | Role in the chain |
|--------|-------------------|
| `submit_full_campaign.py` | Master orchestrator — submits the whole pipeline as a SLURM dependency chain. |
| `generate_synthetic_benchmark.py` | Generate/consolidate ~1M synthetic spectra (chunk/consolidate/submit modes). |
| `generate_basis_libraries.py` | Build per-FWHM basis libraries over a (T, n_e) grid. |
| `run_benchmark_sweep.py` | Run the element-ID sweep over configs x chunks (worker/submit/collect). |
| `train_ml_classifier.py` | Train per-element ML classifiers (SVM/RF/XGBoost) on sweep features. ML tooling — out of the shipped physics-only path; archived. |
| `analyze_benchmark_results.py` | Bootstrap CIs + publication figures from consolidated sweep results. |

Superseded by `cflibs/benchmark/scoreboard.py`. Archived in the 2026-06
repo-cleanliness sweep (docs/audit/2026-06-09-overhaul/05-repo-cleanliness.md).

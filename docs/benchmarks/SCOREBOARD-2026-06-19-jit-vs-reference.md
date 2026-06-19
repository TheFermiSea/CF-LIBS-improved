# Scoreboard — jitpipe vs reference (M3 promotion board)

**Date:** 2026-06-19 · **Candidate SHA:** `782ab5ee` (post-ALIAS-recovery, on-device Stark `n_e`, #297 CI green)
**Decision:** **PROMOTE** the jit pipeline for batched/GPU inference; keep the reference pipeline as the parity oracle (see [ADR-0005](../adr/ADR-0005-jitpipe-promotion.md)).

## Protocol

- Geological preset; candidate policy = `truth.elements_present ∪ confounder_elements ∪ {Ag,Sn,W,Bi,Th}`; presence ≥ 0.5 wt%; seed `20260610`.
- Reference vs jit dispatched by `scripts/run_j12_board_compare.py --pipeline {reference,jit}`.
- **Capped board:** `--max-spectra 100` (877 spectra total across 9 datasets), `--include-holdout`. Run as a SLURM array (one dataset per task) across `vasp-01/02/03`.
- For datasets with ≤ 100 spectra (aalto 74, csa 99, bhvo2 4, …) the cap is inactive — the capped board *is* the full board. A full-board (all-spectra) confirmation of the genuinely-capped datasets is in flight (SLURM job `3256`); it is a confirmation, not a gate, given the uniform jit ≥ ref signal below.

## Per-dataset board

| dataset | tier | n | F1 ref | F1 jit | ΔF1 | RMSE ref | RMSE jit | fail ref | fail jit | wall ref (s) | wall jit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aalto | scoring | 74 | 0.627 | 0.632 | **+0.005** | — | — | 11 | 11 | 0.81 | 11.36 |
| csa_planetary | scoring | 99 | 0.429 | 0.449 | **+0.020** | 22.428 | **20.367** | 4 | 5 | 5.19 | 27.31 |
| chemcam_calib | scoring | 100 | 0.751 | 0.752 | **+0.001** | 4.991 | 5.107 | 0 | 0 | 2.62 | 12.13 |
| silva2022 | scoring | 100 | 0.166 | 0.172 | **+0.007** | — | — | 14 | 14 | 2.05 | 130.96 |
| synthetic_fixedforward | scoring | 100 | 0.171 | 0.163 | −0.008 | 53.852 | 53.852 | 63 | 64 | 0.62 | 5.96 |
| bhvo2_chemcam | holdout | 4 | 0.730 | 0.730 | +0.000 | 2.486 | **2.464** | 0 | 0 | 4.64 | 25.02 |
| emslibs2019 | holdout | 100 | 0.567 | 0.588 | **+0.021** | — | — | 12 | **9** | 2.27 | 28.37 |
| supercam_labcal | scoring | 100 | 0.555 | 0.557 | **+0.002** | 2.902 | **2.786** | 0 | 0 | 3.00 | 22.21 |
| supercam_scct | holdout | 100 | 0.576 | 0.576 | +0.000 | 4.064 | **3.859** | 1 | 2 | 2.82 | 20.70 |

**Aggregate:** mean F1 ref 0.508 → jit 0.513 (ΔF1 +0.005); jit ≥ ref on **8/9** datasets; total hard-failures **105 = 105** (parity).

## GPU throughput (criterion 4, batched)

- Device: NVIDIA V100S; env `/cluster/shared/envs/cflibs-gpu2` (`unset LD_LIBRARY_PATH`, `JAX_ENABLE_X64=1`).
- **~82,382 spectra/s** at batch B=1024 — vs the 0.4–5.1 s/spectrum reference baseline (≫ the 10× gate). `vmap == loop` verified.
- **Caveat:** this is device-core throughput (forward-eval + solve). Full-*pipeline* batched throughput is still bounded by the host-side per-spectrum front-end (detect/identify gather) and the Stark real-data fallback — see ADR-0005 §Gaps.

## M3 five-criteria verdict (ADR-0004 §8.2 / J12 spec §3)

| # | criterion | verdict | detail |
|---|---|---|---|
| 1 | ID micro-F1 (jit ≥ ref) | **PASS** | aggregate jit ≥ ref; 8/9 ≥ ref; only synthetic_fixedforward −0.008 (within the −0.01 per-dataset tolerance and ≪ 0.02 regression cap). |
| 2 | composition RMSE ≤ ref | **PASS** | jit better on csa / bhvo2 / supercam_labcal / supercam_scct; chemcam +0.116 wt% (~2.3% rel, within the 5% single-dataset allowance); synthetic equal; aggregate improves. |
| 3 | hard-failures ≤ ref | **PASS** | aggregate 105 = 105; emslibs −3 (better); silva mirrors the all-FN floor 14/14 (does **not** hide failures); csa/synthetic/supercam_scct each +1. |
| 4 | runtime | **PASS (batched)** | ~82k spectra/s batched on V100S ≫ 10× gate; `vmap == loop`. The CPU single-spectrum sub-clause (≤ 2× ref) is **not** met — jit compile dominates per-spectrum CPU wall (e.g. silva 131 s vs 2 s) — and was explicitly deprioritized for this decision (jit's purpose is batched/GPU throughput). |
| 5 | parity suite green | **PASS** | `tests/jitpipe` incl. the `@slow` J2 set green on `782ab5ee` (#297 CI). |

## Reproduce

```bash
# capped board (per dataset), as run on the cluster array:
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -u scripts/run_j12_board_compare.py \
    --datasets <name> --include-holdout --max-spectra 100 --seed 20260610 \
    --output-dir output/j12/array/<name>
# full-board confirmation (all spectra): scripts/benchmarks/m3_fullboard.sbatch (SLURM job 3256)
```

## Follow-ups (filed, non-blocking)

1. Full-board (all-spectra) confirmation merge once job `3256` lands.
2. `scoreboard.py` per-dataset `try/except` isolation — one adapter raise must not abort the whole board.
3. Full-pipeline batched-GPU spine (host front-end vectorisation) + Stark `n_e` on-device real-data fit (beads `6apc`/`b2dz`).
4. User-facing `--pipeline jit` entrypoint for batch/manifold inference.

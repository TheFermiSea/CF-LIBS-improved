# J12 / M3 — Status: jit-vs-reference superiority harness

**Date:** 2026-06-14 · **Bead:** J12 · **ADR:** ADR-0004 §8.2 (M3), D6 · **Spec:** docs/adr/specs/J12-scoreboard-promotion-m3.md

## What this session built

The J12 *harness* — the infrastructure the M3 promotion decision runs on:

1. **`cflibs scoreboard --pipeline {reference,jit}`** (also `run_scoreboard(pipeline_impl=...)`).
   The scorer is now pipeline-agnostic: `reference` = `run_pipeline` (parity
   oracle); `jit` = `cflibs.jitpipe.run_one(ondevice_front_end=True)` via the
   `_run_pipeline_jit` adapter, which builds the `StaticConfig` from the resolved
   pipeline config against a cached full-DB snapshot. Board JSON/MD record
   `pipeline_impl`.
2. **Failure-policy parity scoring.** A jit all-FN result (zero usable lines —
   `quality_metrics["failed"]=1.0`, the `host.all_fn_result` policy) is scored as
   a *failure* identically to the reference crash, so failure counts and
   composition-RMSE aggregation are directly comparable (M3 criterion 3). Without
   this, jit all-FN records counted as `ok` rows with rmse=100 and inflated the
   jit median RMSE — a scoring artifact, not a regression.
3. **Board-compare harness** — `scripts/run_j12_board_compare.py`: runs both
   pipelines over the scoring datasets and emits a per-dataset delta table
   (ΔF1, RMSE, failures, wall s) + raw board JSONs.

This depends on J9 being complete: as of this session the jit front-end runs
segmented calibration + detect + identify on-device (J9 fix; only the Stark n_e
diagnostic stays delegated — bead CF-LIBS-improved-6apc).

## Parity evidence + a gap (capped subsets — plumbing validation, NOT the M3 gate)

**With host-delegated calibration** (jit = on-device detect/identify/solve only),
the jit pipeline reproduces the reference **micro-F1 exactly** (ΔF1 = 0.000 on
both datasets, `output/j12/run1`). This isolates the detect/identify/solve port
as bit-faithful.

**With the full on-device front-end** (J9 on-device calibration; `output/j12/run2`),
the jit initially diverged on **single-segment (seam-free) spectra** — sparse
synthetic *and* the real `aalto` minerals — because the on-device deterministic
stratified RANSAC resolves a sparse/multimodal anchor set to a different
registration than the reference random-600 RANSAC. Board run2: synthetic
ΔF1 = −0.291 (fail 7 vs 3), aalto fail 1 vs 0.

**Closed (bead aa9e, commit a2f6009).** The fix keys on the *structural* signal
(`detect_ccd_seams(...).size == 0`) and delegates single-segment spectra to the
reference `_ld_calibrate`; the multi-segment broadband path (BHVO-2: 3-segment,
on-device, parity 0.00025 nm) is untouched. Board run3 (with the fix):

| dataset | ΔF1 | jit fail vs ref |
|---|---|---|
| aalto | **+0.000** | 0 = 0 |
| synthetic_fixedforward | **+0.000** | 3 = 3 |

BHVO-2 non-regression control passes (still on-device, max|Δλ| 0.00025 nm); regression
test `tests/jitpipe/test_ondevice_calib_single_segment_parity.py`. Honest caveat: on
`muscoviteE35` parity is restored by inheriting the reference's dubious ~1.9 nm shift
mode — a pre-existing reference multimodal-RANSAC issue, filed as a follow-up bug (an
ambiguity guard for both pipelines). Residual (separate, bead `5yo1`): synthetic solve
RMSE 48.7 vs 44.4 on the 5 solved spectra — a small on-device-solve delta within M1
tolerance on a questionable-truth synthetic corpus.

## What M3 still requires (NOT done this session — process + compute gates)

Per the J12 spec §2–3, the promotion decision is gated on more than the harness:

- **Full board, all spectra** (not `--max-spectra`), geological preset, seed
  20260610 — including the **holdout tier** (`--include-holdout`), opened only at
  this gate. Launch as background sbatch jobs (watchdog rules).
- **≥1 release of shadow-mode operation** (both pipelines on every scoreboard
  invocation, deltas logged) with the **divergence ledger** current and every
  Tier-D divergence adjudicated — a *precondition*, not a deliverable.
- **The 5-criterion M3 gate** (spec §3): aggregate F1 ≥ ref + no scoring dataset
  regresses F1 > 0.02 + per-dataset F1 ≥ ref−0.01 on ≥7/10; composition median
  RMSE ≤ ref on every quantitative dataset; hard-failure count ≤ ref; batched
  ≥10× median spectra/s + CPU single-spectrum ≤2× ref wall; parity suite green.
- **Outcome:** pass → write `ADR-0005-jitpipe-promotion.md` (default flips to
  jit, reference kept as `--pipeline=reference` oracle); fail → gap beads, stay
  parallel.

## Reproduce

```bash
# capped plumbing check
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/run_j12_board_compare.py \
  --datasets aalto,synthetic_fixedforward --max-spectra 8 --output-dir output/j12/run2

# single-pipeline board (CLI)
JAX_PLATFORMS=cpu cflibs scoreboard --pipeline jit --output-dir output/scoreboard-jit
```

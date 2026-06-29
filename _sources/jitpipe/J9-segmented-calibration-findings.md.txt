# J9 — On-Device Segmented Wavelength Calibration (R8 model-flip): diagnosis + fix

**Date:** 2026-06-14 · **Bead:** CF-LIBS-improved-myk7 · **ADR:** ADR-0004 §5.1, risk R8

## TL;DR

The on-device segmented wavelength calibrator (`calibrate_segmented_kernel` →
the new per-segment kernel-backed driver) initially flipped one segment's model
class `shift → affine` vs the reference on the real ChemCam BHVO-2 confounder,
shifting the corrected axis ~0.08 nm and dropping the ye6t Al doublet
(obs Jaccard 0.79–0.83 << 0.98). **This is now RESOLVED** — the front-end runs
segmented calibration on-device, obs Jaccard 1.0000 (raw + geological),
corrected-axis `max|Δλ| = 0.00025 nm` vs the reference, all M1 + J2 parity tests
pass. Calibration is **~95% of the front-end on-device**; only the Stark n_e
diagnostic remains reference-delegated (follow-up bead).

> **Correction note (process):** an interim diagnosis in this file concluded the
> flip was an "ill-conditioned near-tie, not robustly fixable, keep host-side."
> That was **premature** — it was measured against the *old* `_ondevice_calibrate_segmented`
> (which fed the kernel global peaks masked to a segment window) and missed that
> the dense-hull coverage tiebreak distinguishes the spurious affine. The fix
> below (per-segment re-detection + unconditional coverage tiebreak) closes it.
> Kept here because the diagnosis of *why* it flipped is still the right map.

## The flip (measured, raw preset, `affine_coverage_gate=False`)

Reference detects 3 segments and chooses `shift` for all three. The pre-fix
kernel agreed on seg[0]/seg[1] but flipped seg[2] (473–905 nm) `shift → affine`.
Per-model fit on seg[2] (`scripts/diag_seg2_bic_margin.py`):

| calibrator | model | BIC | inliers | rmse (nm) | slope | winner |
|---|---|---|---|---|---|---|
| reference | shift  | **−210.84** | 33 | 0.0389 | — | **shift** |
| reference | affine | −206.16 | 33 | 0.0396 | 0.203 | |
| kernel | shift  | −210.76 | 33 | 0.0389 | — | |
| kernel | affine | **−222.45** | **36** | 0.0412 | 0.126 | **affine** |

The kernel's `shift` matches the reference exactly. The divergence is entirely
in `affine`: the kernel's deterministic stratified sampler finds a *stronger*
affine optimum — 36 inliers, BIC −222 — that the reference's *random* RANSAC
(600 draws) missed (it landed on a poor 33-inlier affine at −206 and so picked
shift). The kernel's search is genuinely better; it just locks onto an
**under-anchored** affine (anchor span_fraction = 0.53 of the segment, i.e. the
slope is extrapolated across the unanchored half — the ye6t signature).

## Root cause (two layers)

1. **Architectural (candidate set).** The old `_ondevice_calibrate_segmented`
   fed the monolithic kernel the *global* peaks masked to a segment window. The
   reference `_fit_one_segment` **re-detects** peaks (`detect_peaks_auto` on
   `intensity[a:b]`) and **re-builds** the line pool per segment slice. Different
   candidates → different fits.
2. **Scoring (under-anchored affine).** Even with matched candidates, the
   kernel's stronger affine search finds the 36-inlier extrapolated fit. The
   reference's affine_coverage gate would reject it (span 0.53 < 0.6) — but the
   raw preset has that gate *off*, so raw still diverged at Jaccard 0.96.

## The fix

1. Rewrote `_ondevice_calibrate_segmented` as a host orchestrator mirroring the
   reference segmented driver exactly (reusing the reference helpers
   `detect_ccd_seams`, `_segment_anchor_coverage`, `_segment_fit_trusted`,
   `_apply_neighbor_fallback`, `_restore_seam_monotonicity`,
   `_build_segmented_result`, `_revert_segmented_to_global`), routing each robust
   RANSAC core through a new kernel-backed inner calibrator
   `_calibrate_axis_kernel_backed` that re-detects peaks + rebuilds the pool
   **per segment**.
2. Apply the **dense-hull coverage tiebreak (J2 §7 R8) unconditionally** in model
   selection — independent of `pipeline.affine_coverage_gate` — so the
   under-anchored seg-2 affine is rejected in *both* presets (this is the binding
   fix; raw needed it). `exact_dedupe_score=True` added to `calibrate_axis_kernel`
   as defense-in-depth.

Speed levers (parity-neutral, axis bit-identical): `jax.jit` + `lru_cache` the
per-segment kernel; `k_pair` 48→16 (measured fan-out ≤16); `h_affine` 256→64
(saturated). `pipeline.py` unchanged — wiring is in `run_front_end_ondevice`
(`run_one(ondevice_front_end=True)`, the default). `stark.py` untouched.

## Verification (independent)

`scripts/diag_segmented_calib_flip.py` against the fixed code, real BHVO-2:

```
seg[2] corrected: dev −0.1304 (constant shift) vs ref −0.1301  →  MATCH
max|Δ| = 0.00025 nm (was 0.077)   0/6144 samples diverge >0.02nm   time 10.9 s (was 293 s)
```

Adversarial verify agent: 0 reference segmented-calib calls under `ondevice_front_end=True`
(both presets); obs Jaccard 1.0000; reference does real 0.130 nm axis work, on-device
tracks to 0.000254 nm; `pytest tests/jitpipe/test_parity_pipeline.py tests/jitpipe/test_parity_j2.py`
→ 57 passed. Verdict: integrate.

## Status

- **On-device:** response (host), segmented calibration (on-device, kernel-backed),
  J1 detect, kdet coherence filter, J3 comb scan/shift-select/veto/observation-build,
  trapezoid intensity, LineSelector — all JIT kernels. Catalog SQL + gA-Boltzmann
  ranking + scipy peak detection stay host-side (ADR-0001).
- **Remaining delegated:** the Stark n_e diagnostic (`measure_stark_ne`) — delicate
  (pins the n_e ≤ 10% M1 tolerance); follow-up bead.
- **Speed:** warm ~4.4 s / cold ~10.9 s CPU (reference ~1.4 s); amortizes under
  `jit(vmap)` batching on the V100S (the J9 goal).

## Reproduce

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/diag_segmented_calib_flip.py
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/diag_seg2_bic_margin.py
```

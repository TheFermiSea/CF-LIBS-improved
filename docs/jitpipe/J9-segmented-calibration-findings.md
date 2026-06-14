# J9 Findings — On-Device Segmented Wavelength Calibration (R8 model-flip)

**Date:** 2026-06-14 · **Bead:** CF-LIBS-improved-myk7 · **ADR:** ADR-0004 §5.1, risk R8

## TL;DR

The on-device segmented wavelength calibrator (`calibrate_segmented_kernel`) is
fully composed and parity-tested for the deterministic stages, but on the real
ChemCam BHVO-2 confounder it picks a **different per-segment model class** than
the reference (`calibrate_wavelength_axis_segmented`) for one segment, shifting
the corrected axis ~0.08 nm and dropping the ye6t Al doublet downstream
(obs Jaccard 0.79–0.83 << 0.98 floor).

**This is an ill-conditioned near-tie, not a fixable kernel bug.** Byte-faithful
parity is chasing a seed-dependent reference outcome. The recommendation is to
**keep segmented calibration host-side (reference-delegated) by design** — the
correct host/device split for a stochastic preprocessing step — rather than
force a brittle on-device match. The GPU payoff (ADR §1.2) lives in the batched
inversion *core*, which is already on-device and measured at ~960× CPU.

## The measurement (raw preset, `affine_coverage_gate=False`)

Reference detects 3 segments and chooses `shift` for all three. The on-device
kernel agrees on seg[0] (240–341 nm) and seg[1] (382–469 nm) but flips seg[2]
(473–905 nm, the whole red channel) `shift → affine`. That slope is the entire
divergence: corrected-axis `max|Δ| = 0.077 nm` at 905 nm; mean 0.029 nm.

Per-model fit on seg[2] (`scripts/diag_seg2_bic_margin.py`):

| calibrator | model | BIC | inliers | rmse (nm) | slope | winner |
|---|---|---|---|---|---|---|
| reference | shift  | **−210.84** | 33 | 0.0389 | — | **shift** |
| reference | affine | −206.16 | 33 | 0.0396 | 0.203 | |
| kernel | shift  | −210.76 | 33 | 0.0389 | — | |
| kernel | affine | **−222.45** | **36** | 0.0412 | 0.126 | **affine** |

The kernel's `shift` fit matches the reference's `shift` to 4 decimals. The
divergence is entirely in `affine`:

- The **reference's** random RANSAC (600 draws, seed 42+2) landed on a *poor*
  affine optimum — slope 0.203, only 33 inliers, BIC −206 — so `shift` won by
  4.7 BIC.
- The **kernel's** deterministic stratified sampler (H=256) found a *better*
  affine optimum — slope 0.126, **36 inliers**, BIC −222 — so `affine` won by
  11.7 BIC.

Both selectors use the same (correct) `argmin(BIC)` with model-order tie-break.
They diverge because the **affine fit itself converges to different local
optima**, and the kernel's is genuinely the lower-BIC one.

## Why this is not robustly fixable on-device

1. **The reference outcome is seed-dependent.** On a luckier RNG seed the
   reference's affine RANSAC would also find the 36-inlier optimum, pick affine,
   and *also* drop the Al doublet. "Parity" here means matching one particular
   seed's under-optimized affine search — not a stable spec.
2. **A BIC parsimony margin can't bridge it.** The kernel's affine wins by
   11.7 BIC; the standard "not worth mentioning" margin is 2–6. A margin large
   enough to flip seg[2] back to shift would suppress every legitimate affine
   (dispersion) correction elsewhere — a different algorithm, not parity.
3. **An rmse-parsimony rule diverges from the reference too.** Rejecting affine
   when it doesn't lower per-inlier rmse would fix seg[2] but breaks parity on
   the (common) cases where the reference legitimately picks an
   more-inliers/slightly-higher-rmse affine via pure BIC.
4. **Matching the inlier set means replicating numpy's RANSAC RNG stream** in a
   fixed-shape JAX kernel — fragile and contrary to the kernel's
   deterministic-by-design contract.

The kernel is not wrong; it fits affine *better* than the reference's random
search. The reference's seg[2] `shift` is partly luck.

## Speed

The segmented kernel takes ~293 s for the first call at production width
(W bucket 8192, SEG_MAX=8, 2 models, H=256). A second call on identical shapes
did **not** return a cheap cached run — it was still executing past ~290 s of
additional CPU when killed — so the cost is ~290 s **per call**, not a one-time
compile. The reference host calibration, by contrast, is **2.84 s/spectrum**
(measured, same spectrum). Even setting the parity flip aside, the on-device
kernel is ~100× slower per spectrum than the host path it would replace. See
`scripts/diag_segmented_calib_flip.py`.

## Decision

- **Keep `calibrate_segmented_kernel` host-delegated** (`_ld_calibrate`) in the
  production front-end. This is the M1-validated path and the correct host/device
  split (ADR-0001): a stochastic catalog-RANSAC preprocessing step belongs on the
  host; the deterministic fixed-shape inversion core belongs on-device.
- **The on-device kernel stays available + parity-tested for the deterministic
  cases** (seam detect, global fit, shift segments) as documentation of the
  attempt and for instruments without the near-tie pathology.
- **Throughput at scale is unaffected:** wavelength calibration is *amortizable
  per instrument configuration* (all spectra in a campaign share the solution),
  so per-spectrum on-device calibration is not on the critical path for batch
  GPU inversion.

## Reproduce

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/diag_segmented_calib_flip.py
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/diag_seg2_bic_margin.py
```

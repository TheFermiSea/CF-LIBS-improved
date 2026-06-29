# J2 wavelength-calibration reference self-variance study

**Bead:** J2 · **ADR:** ADR-0004 §4 row 3 · **Acceptance criterion:** AC1
(self-variance artifact committed before tolerance bands are frozen).

## Why

The J2 §4 tolerance contract compares the fixed-shape JAX kernel against the
**seed-42 reference** (`calibrate_wavelength_axis`). No tolerance band may be
tighter than the reference's *own* seed-to-seed variance — otherwise the parity
test would be measuring RANSAC RNG noise, not a real behavioral deviation. This
study measures that floor.

## Method

The reference RANSAC (`_ransac_search`, 600 iterations, host
`np.random.default_rng(seed + k)`) is the only stochastic element. We re-run the
**unmodified reference** `calibrate_wavelength_axis` across `random_seed ∈
{0..9}` on representative corpus cells (clean synthetic LIBS-like spectra with a
known shift / affine / no-op calibration error, noise σ = 0.03; the
`shift`/`affine`/`noop` × seed cells in `tests/jitpipe/test_parity_j2.py`) and
record, per seed:

- `quality_passed` (gate outcome),
- selected `model` class,
- `coefficients` / corrected-axis spread.

The driver is `tests/jitpipe/test_parity_j2.py::test_reference_self_variance_gate_stable`
(asserted in CI) plus the ad-hoc seed sweeps used to size the bands below.

## Findings

| Quantity | Reference self-variance across 10 seeds | Frozen band (J2 §4) |
|---|---|---|
| **Gate outcome** (`quality_passed`) | **0** — unanimous across all seeds on every measured cell | exact per-cell equality (contract a) |
| **Selected model class** | ≤ 1 near-tie flip (shift↔affine) on BIC-tie cells; stable otherwise | ≥ 90 % of cells, near-tie (ΔBIC ≤ 5 %) cells excluded (contract c) |
| **Corrected axis** (where gate passes & model agrees) | sub-1e-3 nm across seeds (RANSAC picks equivalent inlier sets) | `max\|Δλ\| ≤ 0.04 nm = inlier_tol/2` (contract b) |

### Interpretation

- The **gate decision has zero reference self-variance** on the measured
  corpus, so the kernel's exact per-cell gate-outcome match (contract a) is a
  meaningful, non-noise-dominated assertion.
- The **0.04 nm corrected-axis band is ~40× looser than the reference's own
  seed spread** where the model class agrees — comfortably above the floor and
  sub-pixel on every measured instrument (inlier_tolerance_nm / 2).
- **Model class** is the only quantity with non-zero reference self-variance:
  on near-BIC-tie cells the reference itself can flip shift↔affine between
  seeds. The contract therefore (i) requires only ≥ 90 % aggregate agreement
  and (ii) excludes near-tie cells (ΔBIC within 5 %, the J2 §4 "±5 %-of-
  threshold" near-threshold carve-out applied to model selection). The kernel's
  upper-bound hypothesis ranking (J2 §3) can likewise pick a different — usually
  equivalent — winner; where it does, the corrected axis stays within the looser
  RMSE band (J2 §4 "MAY differ: RMSE within max(±30 %, 0.02 nm)").

## Scope note

This study and the committed parity suite cover the **global single-axis
calibrator** (`calibrate_wavelength_axis`, ref `:602`) — the inner core re-used
by the segmented driver and the largest jit-hostile surface (banded pairs,
exhaustive shift hypotheses, stratified affine/quad, upper-bound scoring, exact
dedupe scan, model lattice, BIC + 5 gates, monotonicity). The **segmented
orchestration** (CCD-seam detection, per-segment vmap, neighbour fallback,
seam-monotonicity cascade, revert gates, and the *segment-level* ye6t coverage
gate) is the remaining J2 segmented-driver scope; its self-variance bands will
be appended here when that layer lands. The ye6t affine band-edge hazard the
coverage gate keys on is already parity-anchored at the global level
(`test_ye6t_affine_band_edge_parity`).

# J2 Implementation Spec — Jittable Segmented-RANSAC Wavelength Calibration

**Bead:** J2 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §4 row 3, §3 C2/C3 · **Track:** A · **Depends:** J0, J1 · **Estimated effort:** 12–17 pd (global calibrator kernel 6–8, segmented orchestration 4–6, reference self-variance study + parity corpus 2–3)

## 1. Goals

- Port `cflibs/inversion/preprocess/wavelength_calibration.py` (1,601 lines; pipeline entry `calibrate_wavelength_axis_segmented` called per spectrum at `cflibs/inversion/pipeline.py:488`) into `jitpipe/calibrate.py` as a fixed-shape, vmappable kernel. This is the **biggest runtime stage** (calibration median 1.55 s of 2.64 s total on bhvo2_chemcam, `SCOREBOARD-2026-06-10-baseline.md`).
- Per ADR-0004 §3 C2: parity-anchored parallel hypothesis evaluation with the **exact greedy one-to-one dedupe reserved for the winning hypothesis and refine rounds**. The MAGSAC++ sigma-marginalized IRLS variant is an *optional* differentiable mode behind the same kernel interface (consumed by J11), never the parity build.

## 2. Current algorithm and jit-breakers (file:line; all `preprocess/wavelength_calibration.py`)

Segmented driver (`:1351`): `detect_ccd_seams` (`:829`, Python rolling-median-of-diffs loop `:881-884`) → always-computed global `calibrate_wavelength_axis` (`:1459`) → ye6t coverage gate on the global fit (`:1478-1511`; on failure **re-calls the whole calibrator** restricted to `("shift",)` at `:1495`) → per-segment Python loop `_run_segments` (`:1181,1210`) → `_fit_one_segment` (`:1080`; sparse-segment model restriction `:1119`; re-entrant calibrator call `:1120`; trust gate `:976`; coverage gate `:1015` with another re-entrant shift-only refit `:1053`) → fallback 1 global offset (`:1160-1162`); fallback 2 nearest-fit-neighbour median offset (`:1246,1260`) → sequential seam-monotonicity cascade (`:1269,1276-1286`) → revert-to-global gates (cumulative shift >0.5 nm `:1572`; residual non-monotonicity `:1581`).

Inner calibrator (`:602`): `detect_peaks_auto` (`:700`) → `_build_reference_line_pool` (`:403`; **SQL `get_transitions` per element** `:417`; Python `sorted()` ranking `:424-428`) → `_build_candidate_pairs` nested Python loops (`:441,458-468`) → per-model `_ransac_fit` (`:359`): `_ransac_search` (`:256`) runs **600 iterations** (`:274`), each with rejection sampling ≤25 retries for distinct-x (`:240-253`), an lstsq fit (`:168-196`), and a full **greedy one-to-one dedupe per hypothesis** (`_dedupe_one_to_one` `:209-230`, mutable `used_peaks`/`used_lines` sets, called at `:285`) — the single most jit-hostile construct in the stage; lexicographic score (`:291`) → refine (`:299`) → final fit + BIC (`:325,:233`) → monotonicity rejection (`:509,:199-206`) → BIC sort & best (`:809-810`) → 5 sequential quality gates (`:96-133`) → diagnostics dict (`:554-578`). Host RNG `np.random.default_rng` (`:375`, `random_seed=42` default at `:615`). Other breakers: early `return None` chains (`:174,391,397,683-771`); model dispatch by string (`:156-165`); `_segment_anchor_coverage` reads anchors out of the details **dict** (`:1001`); data-dependent dense-hull slice (`:933`).

## 3. Redesign

- **Line pool:** host snapshot per (dataset band, element set): `(L_max=1024,)` wl + strength + mask (observed ≤925). SQL + ranking stay host-side (ADR-0001 split).
- **Pairs:** `searchsorted` window per peak → banded `(P_max=2048, K_pair=48)` line-index/Δλ/weight slots + mask (measured fan-out ≤32 at the ±2.0 nm window) — never a flat variable-length pair list.
- **Hypothesis evaluation (replaces sequential RANSAC; strictly stronger search):**
  - *shift model:* **exhaustive** — every live pair slot IS a 1-point hypothesis (≤98k masked hypotheses, trivially vmapped; supersedes 600 random draws);
  - *affine/quadratic:* H=4,096 deterministic stratified samples (counter-based threefry, fixed key; distinct-x enforced **by construction** by sampling from distinct peak strata — eliminates the 25-retry rejection loop);
  - per hypothesis: closed-form weighted 1/2/3-point fit, residuals vs all C slots, inlier mask; chunk the (H×C) residual matrix via `lax.map` over H-blocks of 256 (≤100 MB live);
  - **scoring:** parallel upper bound `min(#unique inlier peaks, #unique inlier lines)` + (−masked-median residual) — NOT per-hypothesis greedy dedupe;
  - **exact dedupe for the winner + refine rounds only:** `argsort`-by-residual + `lax.scan` (length C=10,240) carrying `(P_max,)`/`(L_max,)` used-boolean masks; ~3 models × ~3 scans per spectrum, B-wide under vmap.
- **Refine + final fit:** fixed 2 rounds of masked weighted normal equations (3×3 `jnp.linalg.solve`). **Required numerical deviation:** center-and-scale x to segment midpoint/half-span before forming Gram matrices — the reference lstsq on raw nm (`:188-193`) is SVD-backed; raw-x normal equations with x²≈2.5e5 lose ~10 digits. Coefficients mapped back host-side; the contract is on the corrected axis, not coefficients.
- **BIC + gates:** compute all 3 models unconditionally; validity mask = (enough points) & (monotonic on grid) & finite; best = masked argmin BIC; the 5 thresholds (`:123-133`) become a branchless gate vector → on failure the kernel returns the identity correction + an int reason code (host maps codes to today's `quality_reason` strings).
- **Seams:** rolling median of `diff(wl)` via sliding-window `jnp.sort` (window 2×51+1, row-chunked); `segment_id = cumsum(seam_mask)`, clip to SEG_max=16 (observed ≤11; csa_planetary 10 seams / 11 segments).
- **Segments:** vmap the single-segment calibrator over SEG_max slots; pairs/peaks carry segment ids; sparse-segment model restriction (`:1119`) = per-segment model-validity mask. **Replace the re-entrant degrade-to-shift recursion (`:1053,1495`) with a model lattice:** all model fits (incl. shift) are computed in the same pass, so coverage-gate failure *selects* the precomputed shift result — same answer, no recursion. The dense-hull data-dependent k (`:933`) is a traced int used only in gather indices, never shapes — jittable as-is.
- **Fallbacks + cascade:** neighbour fallback = masked argmin over |i−j| of fit-segment indices + segment-masked median offsets; monotonic restore = `lax.scan` over ≤16 segments carrying cumulative shift (exact port); revert gates = branchless select of segmented vs global axis.
- Shift-grid scan downstream context: after calibration the residual `shift_scan_nm` shrinks (`pipeline.py:485-497`); S_shift=32 covers the observed ≤21.

## 4. Tolerance contract (vs the seed-42 reference)

**Precondition:** measure reference **self-variance** across ~10 seeds on the 32-spectrum corpus; no band may be tighter than that floor. Then:

- MUST agree: (a) quality/coverage **gate outcomes** (`quality_passed`, applied-vs-identity) on every corpus spectrum, excluding documented near-threshold cases (metric within ±5 % of its threshold); (b) **corrected axis** `max|λ_jit − λ_ref| ≤ inlier_tolerance_nm/2 = 0.04 nm` (sub-pixel on every measured instrument) wherever both pass gates; (c) selected model class on ≥90 % of (spectrum, segment) cells; (d) per-segment fit/fallback status labels.
- MAY differ: coefficients (different x-parameterization), inlier sets/counts, winning hypothesis, RMSE within max(±30 %, 0.02 nm), diagnostics detail values.
- Downstream guard: |ΔF1| ≤ 0.01 micro-averaged on the 10-dataset board.

## 5. Acceptance criteria

1. Self-variance study artifact committed (`docs/jitpipe/calibration-self-variance.md`) before bands are frozen.
2. Contract (a)–(d) green on the 32-spectrum corpus; **ye6t regression fixtures (ChemCam VNIR / BHVO-2 877 nm Al doublet flip) in the parity corpus and green** — the coverage gate exists specifically for them.
3. jit + vmap (B=16) + padding-invariance + no-SQLite-in-kernel; deterministic per (seed, shapes).
4. Host wrapper reconstitutes `WavelengthCalibrationResult` so downstream consumers/parity tests see identical types.
5. Memory: H-chunked residual matrix ≤100 MB live (asserted via planned-bytes check).
6. Optional MAGSAC++ IRLS mode compiles and is grad-finite (smoke only; full validation in J11).

## 6. Test plan

`tests/jitpipe/test_parity_calibrate.py` — corpus-driven, `assert_allclose` on corrected axis + exact equality on gate booleans (style of `tests/inversion/test_iterative_lax.py:180`); seam/segment unit tests on synthetic multi-CCD spectra; adversarial monotonicity-cascade fixture; model-lattice equivalence test (coverage-gate failure path ≡ reference shift-only refit).

## 7. Risks

- *Numerical (medium):* quadratic Gram conditioning — mitigated by centering; flagged as deliberate deviation.
- *Behavioral (medium-high):* upper-bound hypothesis ranking can pick a different (usually better) hypothesis than greedy-per-hypothesis dedupe; bounded by the corrected-axis + gate-outcome contract; every corpus divergence ledgered.
- *Memory (low):* H-chunking bounds live residuals.

## 8. Dependencies / files

Depends J0, J1 (calibration-parameterized peak kernel). Enables J8; J3 may develop against reference calibration meanwhile. Files: `cflibs/jitpipe/calibrate.py`, tests, self-variance doc. Reference untouched.

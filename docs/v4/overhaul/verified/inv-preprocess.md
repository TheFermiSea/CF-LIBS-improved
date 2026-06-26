# Adversarial Verification: `cflibs/inversion/preprocess`

Verifier: Claude Sonnet 4.6, 2026-06-25
Source: `.worktrees/v4-m5/cflibs/inversion/preprocess/`
Method: independent rg+Read traversal of cited lines; re-deriving against `literature/wavelength-calibration.md`

---

## Finding 1 — Quadratic RANSAC fits un-normalized wavelength coordinates

**REAL: TRUE**
**Corrected severity: high** (confirmed)

Code at `wavelength_calibration.py:192–195` constructs `X = np.column_stack([x * x, x, np.ones_like(x)])` where `x = cand_x = peak_wl` — observed wavelength in nm, range ~200–900 nm for typical LIBS. Column 1 has values ∼4×10⁴–8×10⁵, column 2 ∼200–900, column 3 = 1. This multi-decade scale disparity between columns produces a poorly conditioned system matrix whose condition number scales with the ratio of the largest to smallest singular values. The literature (`wavelength-calibration.md §2.2`) mandates normalization to [−1, +1] before fitting any polynomial basis. The census's claimed condition number of ~3×10⁶ is plausible for this value range, though the worst-case pixel-space problem (0–4096 range) is more severe. The core finding is unambiguously real: the quadratic model is fit in raw nm coordinates without normalization, which is an established numerical pitfall. The affine model is less severely affected (its column ratio is ~200:900:1 — more manageable), consistent with the census's note. Note that in the segmented calibration path (`calibrate_wavelength_axis_segmented`, default `candidate_models=("shift", "affine")`), the quadratic model is excluded by default, so the worst-case conditioning only applies to the non-segmented path.

---

## Finding 2 — No sub-pixel centroiding

**REAL: TRUE**
**Corrected severity: high** (confirmed)

`detect_peaks` at `preprocessing.py:617` returns `[(int(idx), float(wavelength[idx])) for idx in peak_indices]` — the wavelength grid value at the integer argmax index from `scipy.signal.find_peaks`, with no Gaussian or parabolic sub-pixel fit applied. `_filter_peaks_by_fwhm` (line 453) uses `peak_widths` to compute FWHM for cosmic-ray rejection but does not extract a sub-pixel center. `calibrate_wavelength_axis` at line 1127 assigns `peak_wl = np.asarray([p[1] for p in peaks], dtype=float)` — these are discrete grid wavelengths, not interpolated centers. The literature (`wavelength-calibration.md §2.4`) is unambiguous: "using the raw peak pixel index instead of a sub-pixel centroid introduces ~0.5 pixel systematic error." At LIBS dispersions of 0.06–0.1 nm/px, this is 0.03–0.05 nm RMS per anchor, which is comparable to the RANSAC inlier tolerance (`inlier_tolerance_nm=0.08` nm default). The infrastructure (FWHM data from `peak_widths`, baseline-subtracted array) is already present to support a 3-point parabolic centroid with minimal code change. Finding is confirmed as written.

---

## Finding 3 — `min_intensity_floor` semantics inverted

**REAL: TRUE (with nuance)**
**Corrected severity: medium** (confirmed)

Code at `preprocessing.py:586–589`: `threshold = min(threshold, min_intensity_floor)` — when `min_intensity_floor < noise * threshold_factor`, this lowers the detection threshold, enabling detection of peaks that would have been rejected by the noise criterion. The census's description is correct: a "floor" parameter that can lower (not raise) the threshold is inverted semantics. The docstring at line 573–577 says "peaks above this floor will be detected even if they fall below the noise-derived threshold" which accurately describes the behavior but does not guard against pathological cases. Specifically, if `min_intensity_floor=5.0` and `noise=4.0` (threshold_factor=4 → threshold=16), the result threshold is 5.0 — below the 4σ criterion but above the 1σ noise level; this isn't below the noise floor in the strict sense but is below the standard CF-LIBS 4σ criterion. The default `min_intensity_floor=0.0` means the branch is never triggered unless a caller explicitly passes a positive value. In the wavelength-calibration caller at line 1107, `detect_peaks_auto` is called without `min_intensity_floor`, so the calibration path is unaffected. The risk is in other callers (e.g., identification algorithms) that might pass a non-zero floor. The census's characterization of "inverted semantics" is accurate; severity is medium because it requires non-default usage to trigger.

---

## Finding 4 — `detect_ccd_seams` pure-Python rolling median

**REAL: TRUE**
**Corrected severity: high** (confirmed)

Code at `wavelength_calibration.py:1302–1305` is a Python `for i in range(n)` loop computing `np.median(dl[lo:hi])` at each sample. The default window is 51 (half-width, line 1253), so the actual window size is 103 samples. For n=5000 pixels (typical multi-channel LIBS), this is ~5000 `np.median` calls on windows of up to 103 elements — pure Python overhead plus NumPy call overhead per iteration. The census's performance comparison to `scipy.ndimage.median_filter` is valid: the C-level `median_filter` function processes the entire array in a single C loop. `scipy.ndimage.median_filter` is imported in `preprocessing.py` (line 18) but NOT in `wavelength_calibration.py` (no `scipy.ndimage` import in that file). The function is called once per `calibrate_wavelength_axis_segmented` invocation (line 1951), which is on the hot path for real-data inversion. The finding is confirmed as written. Note: the census claim that `detect_ccd_seams` "feeds" RANSAC (which is the profiled 73% bottleneck) is accurate in the sense that seam detection must complete before the segmented RANSAC runs, but seam detection's own cost is additional latency, not the dominant one.

---

## Finding 5 — `wavelength_calibration.py` is a 2119-line monolith

**REAL: TRUE**
**Corrected severity: high** (confirmed, but subjective severity)

`wc -l` confirms exactly 2119 lines. All six subsystems cited in the census are verified present at the stated line numbers: `_quality_gate_check` (98), Hough functions (273–369), `_ransac_search`/`_ransac_fit`/`_refine_robust_inliers` (431–643), line pool construction (646–815), `_build_candidate_pairs` (818–853), quality/coverage gates (1331–1500), segmented orchestration (1503–1748). The finding is accurate. Severity "high" for an architecture issue is defensible given the file's centrality to the calibration hot path and the documented history of coverage-gate bugs. A reasonable position is that this is "high" architecture debt but not a correctness or physics error — marking high-as-architecture rather than downgrading severity.

---

## Finding 6 — `outliers.py` belongs in `common/`

**REAL: TRUE (with important nuance)**
**Corrected severity: medium** (confirmed)

`preprocessing.py:23` imports only `MAD_SCALE_FACTOR` from `outliers.py`. However, the census claim that `outliers.py` is "coupled to the package only through one constant" is an understatement: the module exports a full public API (`SAMOutlierDetector`, `MADOutlierDetector`, `mad_outliers_1d`, `mad_outliers_spectra`) re-exported at `cflibs.inversion` level (lines 114–122 of `cflibs/inversion/__init__.py`) and tested directly via `tests/test_outliers.py`. The census's architectural recommendation (move to `common/`) is sound and the functional mismatch (replicate-stack operations vs single-spectrum operations) is real, but the framing that the module is *only* coupled via one constant is misleading. The actual coupling is broader: any user calling `cflibs.inversion.MADOutlierDetector` or `cflibs.inversion.mad_outliers_spectra` is using it. The finding is real as an architectural observation but overstates the isolation. No physics or correctness impact.

---

## Finding 7 — Three env-var flag resolvers with inconsistent patterns

**REAL: TRUE**
**Corrected severity: medium** (confirmed)

Three resolver functions confirmed at stated locations: `_hough_calib_enabled` (line 258), `_ransac_early_exit_config` (line 392), `calib_pool_cache_enabled` (line 698). All three check the same env-var truthy values (`"1"`, `"true"`, `"yes"`, `"on"`) but use different patterns: `_hough_calib_enabled` takes `Optional[bool]` and returns `bool`; `_ransac_early_exit_config` returns a dataclass; `calib_pool_cache_enabled` is public (no leading underscore) while the others are private. The inconsistency is confirmed. The finding is accurate and the proposed fix (shared `_feature_flag` helper in `cflibs/core/`) is architecturally sound. No physics impact; medium severity is appropriate.

---

## Finding 8 — Benchmark-gated flags lack activation documentation

**REAL: TRUE**
**Corrected severity: medium** (confirmed)

Both `CFLIBS_HOUGH_CALIB` and `CFLIBS_RANSAC_EARLY_EXIT` are confirmed as default-OFF at lines 268–270 and 399–401, with explicit "parity-AFFECTING: benchmark-gated" comments in the surrounding docstrings. A `grep` across `examples/`, `CLAUDE.md`, and `docs/` finds no benchmark result table for these flags. The Hough warm-start itself is well-implemented (vectorized Hough accumulator at lines 273–320, RASCAL-style) and the census's claim that it has no RNG dependency and should be promotable to default-ON after benchmark graduation is accurate — the code confirms `_HOUGH_SEEDED_RANSAC_ITERATIONS = 80` vs the cold-start default of 600, which is the referenced speedup. The gap between the quality of the implementation and the absence of recorded benchmark results is real.

---

## Finding 9 — `_build_candidate_pairs` Python inner loop

**REAL: TRUE**
**Corrected severity: medium** (confirmed)

Code at `wavelength_calibration.py:835–845` confirmed: outer loop over peaks (line 835), inner `for l_i in hit` loop (line 838) appending to Python lists. The outer loop uses `np.abs(line_wl - pw)` (vectorized) and `np.where` to find hits, so only the inner hit-appending loop is pure Python. The `np.searchsorted` optimization proposed by the census is valid since `line_wl` is built in sorted order (constructed by `_build_reference_line_pool`). For typical LIBS spectra (100–500 peaks, 1000–5000 reference lines, sparse hit windows), the per-call cost is sub-100ms, not a dominant bottleneck relative to the RANSAC loop itself. However, in the segmented path with N=5 segments, this runs 5 times per spectrum. The finding is confirmed; the severity is medium (not high) because `_build_candidate_pairs` is not the dominant cost.

---

## Finding 10 — Zero tests for flag-gated paths

**REAL: TRUE**
**Corrected severity: medium** (confirmed)

`grep -rn "hough_calib_seed|_hough_coarse|_hough_seed|ransac_early_exit|CFLIBS_HOUGH_CALIB|CFLIBS_RANSAC_EARLY_EXIT" tests/` returns zero hits. Neither the Hough seed path nor the early-exit path has any dedicated unit test. Both are material code paths with distinct logic: the Hough accumulator (`_hough_coarse_dispersion`, lines 273–320) is a full 2D histogram-vote algorithm; the early-exit logic (`_ransac_required_iters`, `_ransac_search` break conditions, lines 497–509) is a non-trivial confidence-bound computation. The finding is confirmed as written.

---

## Missed Findings (Spotted During Verification)

### Missed A — `_dedupe_one_to_one` uses a Python `set` iteration loop on every RANSAC iteration

**Location:** `wavelength_calibration.py:211–232`
**Severity: low**

`_dedupe_one_to_one` is called inside `_ransac_search` at every RANSAC iteration (line 485), once per candidate model per segment. The function sorts by residual, then iterates over candidates in a Python loop with `set` membership checks. For a typical inlier set (50–200 candidates), this is fast, but it runs 600 × (n_segments + 1) × n_models times total. A vectorized NumPy implementation using `np.unique`-style index tricks is possible. Not a correctness issue; low severity.

### Missed B — `calib_pool_cache_enabled` public naming inconsistency

**Location:** `wavelength_calibration.py:698`
**Severity: low**

`calib_pool_cache_enabled` is a public function (no leading underscore) re-exported via the package `__init__`. The other two flag resolvers (`_hough_calib_enabled`, `_ransac_early_exit_config`) are private. This inconsistency was partially captured in Finding 7 but not specifically flagged as an API surface issue. The function is documented as public to allow callers to test whether caching is active, but it introduces a non-obvious entry point in the public API.

### Missed C — `_select_sample_indices` retry loop may return None silently

**Location:** `wavelength_calibration.py:242–255`
**Severity: low**

`_select_sample_indices` attempts 25 draws trying to select `n_needed` points with unique `x_vals`. If all 25 attempts fail (can happen with highly degenerate `x_vals` arrays, e.g., many peaks at the same nominal wavelength due to bad input), it returns `None`. The `_ransac_search` caller at line 476 silently `continue`s. This is a correctness-safe design (RANSAC degrades gracefully), but there is no warning emitted, and the iteration counter advances, potentially exhausting all iterations on degenerate data without producing a valid model. Low severity because it only affects pathological inputs.

---

## Summary Table

| # | Title | REAL | Corrected Severity | Notes |
|---|-------|------|--------------------|-------|
| 1 | Quadratic RANSAC un-normalized coordinates | TRUE | **high** | Confirmed; applies to non-segmented path's quadratic model |
| 2 | No sub-pixel centroiding | TRUE | **high** | Confirmed; integer grid lookup, 0.03–0.05 nm RMS error |
| 3 | `min_intensity_floor` inverted semantics | TRUE | medium | Confirmed; only triggered by non-default callers |
| 4 | `detect_ccd_seams` pure-Python rolling median | TRUE | **high** | Confirmed; scipy.ndimage drop-in available |
| 5 | 2119-line monolith | TRUE | high (arch) | Confirmed; function locations verified exact |
| 6 | `outliers.py` belongs in `common/` | TRUE | medium | Census understates actual public API; finding still valid |
| 7 | Three env-var resolvers inconsistent | TRUE | medium | Confirmed |
| 8 | Benchmark flags lack documented pass criteria | TRUE | medium | Confirmed; no recorded results found |
| 9 | `_build_candidate_pairs` inner Python loop | TRUE | medium | Confirmed; not dominant cost, vectorizable |
| 10 | Zero tests for flag-gated Hough/early-exit paths | TRUE | medium | Confirmed |
| A | `_dedupe_one_to_one` Python loop in RANSAC hot path | NEW | low | Missed by census |
| B | `calib_pool_cache_enabled` public API inconsistency | NEW | low | Partially covered in Finding 7 |
| C | `_select_sample_indices` silent None on degenerate input | NEW | low | Missed by census |

**All 10 census findings confirmed REAL.** Highest confirmed severity: **high** (Findings 1, 2, 4, 5).
No false positives detected. Three additional low-severity findings identified during verification.

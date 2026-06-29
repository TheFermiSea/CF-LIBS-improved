# J1 Implementation Spec — Jittable Baseline, Noise, and Exact Peak Detection

**Bead:** J1 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §4 rows 1–2, §3 C1 · **Track:** A (front-end) · **Depends:** J0 · **Estimated effort:** 9–12 pd (baselines 3–4, noise 0.5, exact find_peaks 5–7, integration + dual-detector parity 1)

## 1. Goals

- Port `cflibs/inversion/preprocess/preprocessing.py` (748 lines) — baseline estimation (4 methods + AUTO), noise estimation, and peak detection — into `jitpipe/preprocess.py` + `jitpipe/detect.py` as fixed-shape fp64 kernels.
- **One exact-`find_peaks` kernel, two parameterizations.** The front-end currently runs *two different* peak detectors: calibration uses noise-calibrated `detect_peaks_auto` (`preprocessing.py:619`); line matching uses the max-normalized `_find_peaks` (`identify/line_detection.py:2424`, `prominence=threshold/2` at `:2459`). The rewrite implements scipy's semantics once and instantiates both.
- Per ADR-0004 §3 C1: **exact scipy semantics, byte-identical peak index lists** — not the vision-NMS approximation. Every downstream stage keys off peak identity.

## 2. Current algorithm and jit-breakers (file:line)

`detect_peaks_auto` (`preprocessing.py:619`): enum dispatch on `BaselineMethod` (`:684-698`; AUTO probe at `:363-414`) → baselines: MEDIAN = `scipy.ndimage.median_filter`, window from `window_nm/median(diff(wl))` forced odd (`:48-57`); SNIP = LLS transform + fixed 40-iteration clipping (`:157-170`) + inverse (`:175-180`); ALS = `scipy.sparse` IRLS `spsolve(W + λDᵀD)`, ≤20 iterations with convergence break (`:261-277`); PERCENTILE = `percentile_filter` (`:317`) → `estimate_noise` (`:320`): 3-iteration sigma-clipped MAD with early breaks and a **shrinking residual array** (`:340-349`) + percentile floor (`:357-358`) → `detect_peaks` (`:524`): `scipy.signal.find_peaks(height=noise*4, prominence=noise*1.5, distance=...)` (`:595-600`), distance from resolving power or 3 px (`:417-449`) → cosmic-ray FWHM filter via `peak_widths(rel_height=0.5)` with try/except fallback (`:452-481`) → optional second-derivative confirmation, Python per-peak loop (`:484-521`).

Breakers: the four scipy/C routines; ALS early-break (`:272`); shrinking arrays under boolean indexing (`:349`); `find_peaks` variable-length output, **non-local prominence**, **greedy height-order distance-NMS**, plateau-midpoint indices (all three named the canonical JAX trap in `docs/jax-port/line-detection-consultation.md:119-123`); per-peak Python loops. Non-breakers in disguise: window sizes derived from `median(diff(wl))` (`:48-56`) and SNIP pad `min(40, len-1)` (`:163`) — the wavelength axis is constant per bucket (ADR-0004 §5.2), so these are host-computed static ints.

## 3. Redesign

**Baselines (`jitpipe/preprocess.py`):**
- SNIP: direct port — fixed 40 iterations of `v = min(v, (v[i−p]+v[i+p])/2)` with reflect pad; already pure vector ops; the canonical jit baseline.
- MEDIAN / PERCENTILE: sliding-window gather (N, W) + `jnp.sort` + gather at the median/percentile rank, row-chunked via `lax.map` (caps the silva2022 worst case 53,717×1,001×8 B = 430 MB at ~33 MB live).
- ALS: fixed 20 IRLS iterations; early-break → converged-flag `where`-freeze (identical converged result); inner solve of the **pentadiagonal** SPD system `(W + λDᵀD) z = Wy` via banded LDLᵀ forward/back substitution as a `lax.scan` recurrence — exact and deterministic; sequential in N but B-wide under vmap. (Matrix-free CG rejected: λ=1e6 is too ill-conditioned for a fixed small iteration count.)
- AUTO: probe runs unconditionally; MEDIAN/ALS choice is a branchless `where` (or resolved host-side per dataset).
- Response-curve multiply: host loads/interpolates (`preprocess/response_correction.py`); kernel sees the per-channel multiplier array only.

**Noise:** masked sigma-clip — fixed 3 iterations over an (N,) keep-mask; masked median via +inf-pad sort + gather at `(count−1)//2`; break conditions freeze the mask via `where`. Exact parity.

**Peak detection (`jitpipe/detect.py`) — exact `find_peaks` semantics, fixed shape:**
- local maxima with scipy plateau semantics (left strictly smaller; first differing right value smaller; plateau-midpoint index): run-boundary ids via `cumsum(x[i]≠x[i+1])` + per-run edge gathers;
- height/threshold: masks;
- **prominence (wlen=None), exact:** two precomputed range-query sparse tables (prefix max + min, (N, ⌈log2 N⌉) ≈ 7 MB at N=53,717); per candidate, binary-search the nearest strictly-higher sample each side via range-max queries (≤17 fixed steps); range-min over the enclosed interval gives the exact scipy base; fully vmapped over fixed candidate slots;
- **distance:** scipy's `_select_by_peak_distance` is exactly greedy keep/suppress in descending height order — port as a `lax.scan` over `P_cand_max=8192` height-sorted slots carrying a keep mask; exact, deterministic, B-wide under vmap;
- FWHM filter / `peak_widths(rel_height=0.5)`: bases from the prominence computation define the exact search ranges; first half-height crossing each side via boolean-cummax + linear interpolation — exact;
- second-derivative confirm: `np.gradient` = central differences; ±2 px window max — trivially vectorized;
- output: `(P_max,)` padded indices + validity mask + count + truncation flag (P_max = 2,048 calibration / 2,560 detection; observed maxima 1,412 / 2,350 — ADR-0004 §5.2).

## 4. Tolerance contract

- Baselines: SNIP/MEDIAN/PERCENTILE elementwise rtol 1e-12 (identical arithmetic; sort-based medians exact). ALS: `max|Δz| ≤ 1e-6·scale(y)` (LDLᵀ vs spsolve round-off + frozen-iteration difference on early-converged spectra).
- Noise: exact (rtol 1e-12).
- Peak sets: **byte-identical index lists** vs scipy on the 32-spectrum corpus and randomized property tests, for both detector parameterizations, including plateau/tie corners. Documented fallback only if a corner survives: per-spectrum Jaccard ≥ 0.995 with each diff individually triaged.
- Downstream guard: scoreboard |ΔF1| ≤ 0.005 (via J8's shadow harness once available).
- fp64 mandatory for this stage — the LLS exp/log round-trip and the λ=1e6 ALS Gram are the sensitive spots (ADR-0004 §5.3).

## 5. Acceptance criteria

1. All four baseline methods + AUTO at their contracts on the corpus fixtures; ALS frozen-iteration idempotence test.
2. Peak-list equality (both parameterizations) on all 32 corpus spectra + ≥1,000 randomized property cases incl. crafted plateaus, equal-height ties, distance-suppression chains.
3. jit + vmap (B=16) smoke; padding invariance (next pad size bit-identical on valid region); no-SQLite guard trivially green (stage takes arrays only).
4. Truncation-flag behaviour verified by synthetic overflow fixture (>P_max peaks).
5. Performance note recorded: sequential NMS scan + ALS scan are the only O(N)-depth components — amortized at B≥16; single-spectrum-latency use stays on the reference path.

## 6. Test plan

`tests/jitpipe/test_parity_preprocess.py` (baselines + noise vs `preprocessing.py` on committed `.npz` fixtures, one per dataset + edge cases) and `tests/jitpipe/test_parity_detect.py` (peak equality vs `scipy.signal.find_peaks` + `peak_widths`; property tests with hypothesis-style randomized inputs at fixed seeds; dual-parameterization parametrize). CPU x64, subsets < 60 s.

## 7. Risks

- **Plateau/tie semantics** — the classic silent-divergence trap (consultation doc); mitigated by the exactness contract + property tests (the design implements scipy's documented semantics, not an approximation).
- ALS banded-LDLᵀ scan is the long pole; validate against `spsolve` on random pentadiagonal SPD systems first.

## 8. Dependencies / files

Depends J0 (snapshot/params/buckets). Enables J2 (calibration consumes the calibration-parameterized detector) and J3 (detection-parameterized). Files: `cflibs/jitpipe/preprocess.py`, `cflibs/jitpipe/detect.py`, tests above. Reference files untouched.

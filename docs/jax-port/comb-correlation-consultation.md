# JAX port: comb.py + correlation.py — consultation synthesis

> **SUPERSEDED (historical):** This 2026-05-12 consultation is kept as design
> provenance only. The JAX/jittable inversion effort was subsequently
> re-scoped by [ADR-0004](../adr/ADR-0004-jittable-inversion-pipeline.md) and
> implemented in the shipped `cflibs/jitpipe/` package; some conclusions below
> were revised by that decision. Do not treat this as current guidance.

Date: 2026-05-12
Consultants: `gpt-5.3-codex`, `gemini-3-flash-preview` (Gemini 3.1 Pro
rate-limited at `RESOURCE_EXHAUSTED`).
Authoring branch: `feat/jax-comb-correlation`.

## Scope

Two CPU SciPy hot loops are the wall-time bottleneck for 50k-spectrum
benchmarks of the identifier phase. PR #118 already ported the Boltzmann
fit in `alias.py`; this port targets:

- `cflibs/inversion/identify/comb.py::CombIdentifier._correlate_tooth`
- `cflibs/inversion/identify/correlation.py::CorrelationIdentifier._identify_classic`
- `cflibs/inversion/identify/correlation.py::CorrelationIdentifier._generate_model_spectrum`

Both files use `scipy.stats.pearsonr` in inner loops, and `comb.py` also
calls `scipy.ndimage.median_filter` for baseline estimation (once per
spectrum, not per element).

## Decisions

### 1. `_correlate_tooth` — masked-Pearson batched over (shift, width)

**Plan:** Pre-extract all `(shift, width)` candidate windows for a single
transition into a `[C, max_width]` array (`C = n_widths * n_shifts`,
typically `3 * 11 = 33`). Pad triangle templates to `max_width` with a
`valid_mask`. Compute Pearson via the masked closed form

```
n  = sum(m)            mx = sum(m*x)/n     my = sum(m*y)/n
xc = x - mx            yc = y - my
cov = sum(m*xc*yc)     vx = sum(m*xc^2)   vy = sum(m*yc^2)
corr = cov / sqrt(vx*vy + eps)
corr = where((vx < tiny) | (vy < tiny) | (n < 3), 0.0, corr)
```

Both consultants agreed:
- Zeros + mask are preferred over NaNs in core math (better numerics
  under jit, easier reductions).
- Index clipping at `[0, n-1]` plus `in_bounds` mask correctly handles
  edge-of-spectrum without dynamic shapes.

**vmap over transitions?** Yes per both. Codex recommends *chunking*
(16–32 transitions/call) if memory becomes a problem; for `~50` per
element it is fine in one shot.

### 2. `_identify_classic` + `_generate_model_spectrum` — vectorize the (T, n_e) grid

**Plan:** Flatten the 5×3 grid to `G = 15`. Build a `[G, W]` model
spectrum tensor by broadcasting `[L, 1, 1] - wl[1, 1, W]` for the
Gaussian argument and summing along the line axis (or `lax.scan` if
memory becomes tight).

Memory estimate: 30 lines × 15 grid × 30 000 wavelengths ≈ 108 MB.
Trivial on GPU. Both consultants endorsed the broadcasted path; Codex
flagged that real peak can be 2–5× due to intermediates and suggested
`lax.scan` over lines as a memory-safer fallback. We start with the
fully broadcast path because typical line counts are smaller post-
filter (`max_lines_per_element=100` cap, often <30 active).

Pearson uses the same masked closed form, where the mask is the peak
region (AND-mask with OR-mask fallback, replicating CPU behavior).

### 3. `median_filter` — leave on CPU

`scipy.ndimage.median_filter` (C-implemented) for window ~500 over
~30 000 points runs in ~ms. JAX has no native median filter, and a
`reduce_window` + sort approach is memory-heavy and slower in practice.
Both consultants recommended **keeping baseline estimation on CPU**.
This also avoids a JAX↔CPU round-trip per spectrum.

### 4. Numerical precision

`alias.py` boltzmann uses float64 throughout (`jax_enable_x64`). We
mirror that here — both files target rtol 1e-5 vs CPU baseline, which
requires float64 for the (T, n_e) grid correlations where the peak
region can collapse to a few points.

## Validation strategy

- Per-tooth: identical `(best_correlation, best_shift, best_width)` for
  the same `center_nm` across CPU and JAX paths, plus the same
  `active` flag.
- Per-element: identical fingerprint score.
- End-to-end identifier output: identical detected/rejected element
  sets on the synthetic LIBS fixture.

Pearson std==0 edge case: replicate exactly via the `jnp.where` guard
above. CPU returns `0.0` (after the original `np.std < 1e-10` check)
or `pearsonr → NaN → 0.0`; JAX returns `0.0` directly without the NaN
intermediate.

## What we are NOT porting

- `_estimate_baseline_threshold` — keep `scipy.ndimage.median_filter`.
- `_build_triangular_template` — tiny; called per width per transition
  but trivially cached.
- `_analyze_interferences` — Python-side two-pointer, O(N²) worst case
  but N is small after filtering and shape is data-dependent.
- `_match_lines_to_peaks` — greedy one-to-one assignment, not a hot
  loop in benchmarks.
- `_identify_vector` per-neighbor `pearsonr` — `top_k` is small (10),
  not a wall-time bottleneck; cross-module refactor needed to vectorize
  cleanly with the embedder, so this becomes a follow-up bd issue if
  benchmarks show it matters.

## Follow-ups (out of scope)

- Multi-spectrum batching: current port keeps the per-spectrum API.
  Batching across spectra is a larger refactor (transition lists vary
  per element per spectrum's wavelength range).
- `_identify_vector` Pearson refinement vectorization.

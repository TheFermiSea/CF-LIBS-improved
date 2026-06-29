# JAX port consultation — `line_detection.py`

> **SUPERSEDED (historical):** This 2026-05-12 consultation is kept only as ADR
> provenance. Its central recommendation — the rule *against* porting
> `find_peaks` — was premised on B=1 CPU work and was **explicitly reversed by
> [ADR-0004](../adr/ADR-0004-jittable-inversion-pipeline.md) §1.3**. The shipped
> `cflibs/jitpipe/` kernels implement the ported peak-detection this doc argued
> against. Do not treat the conclusions below as current guidance.

**Date:** 2026-05-12
**Issue:** `CF-LIBS-improved-wstf` (partial — line_detection.py scope)
**Branch:** `feat/jax-line-detection`
**Authors:** Claude Opus 4.7 (engineer), GPT-5.3-Codex (reviewer #1), Gemini-3-Flash-Preview (reviewer #2)

## Question

`cflibs/inversion/identify/line_detection.py` (1075 lines) is the next module
in the CF-LIBS identifier pipeline JAX port (issue `CF-LIBS-improved-wstf`,
follow-on to PR #118 which ported the Boltzmann fit). The file's tricky
function is `scipy.signal.find_peaks(normalized, height, distance, prominence)`,
which produces a **variable-length array of peak indices** — JAX's static-shape
model is hostile to this.

The user's instruction was explicit: *"fully and correctly port everything to
JAX (unless it doesn't make sense). Your job is to make the call."*

So we polled two reasoning models and audited the file ourselves before
writing any code.

## File audit (1075 lines)

| Symbol | LOC | Operation | JAX-portable? |
|---|---|---|---|
| `_build_observation_from_fit` | 48-87 | Scalar arithmetic, returns dataclass | No — object overhead, no batch benefit |
| `_build_observation` | 90-151 | Per-peak trapezoid integration on a small window | No — slice indexing into a list returns a dataclass |
| `detect_line_observations` | 189-556 | Top-level orchestrator: I/O, dict building, Python control flow | No — control flow heavy, calls atomic DB |
| `_load_transitions` | 559-576 | DB query | No — pure Python / DB I/O |
| `_transition_strength` | 579-584 | 3-line scalar | No |
| `_select_comb_transitions` | 587-598 | `sorted()` of dataclass list | No |
| `_build_shift_grid` | 601-618 | `np.linspace`, ~20 elements | No — JIT overhead >> work |
| `_score_comb_for_element` | 621-675 | Calls `_match_transitions_to_peaks`, scalar scoring | No — variable-length match list |
| `_scan_comb_shifts` | 762-873 | Dict iteration over shifts × elements, score reductions | No — already Rust-accelerated via `_scan_comb_shifts_rust` |
| `_match_transitions_to_peaks` | 876-912 | Greedy assignment with mutable `used` set | No — sequential, mutates Python set |
| `_kdet_filter_elements` | 915-1000 | Inner shift-scan kernel (Python fallback path) | **YES** — fixed-shape (n_shifts, n_peaks) reductions, Rust path already exists; JAX gives a third backend |
| `_peaks_within_tolerance` | 1003-1016 | `np.searchsorted` + comparisons | **YES** — clean searchsorted + abs + min, all fixed shape |
| `_match_transition` | 1019-1031 | Scalar nearest-match over Python list | No |
| `_estimate_wl_step` | 1034-1039 | `np.median` of `np.diff` | No — sub-microsecond, not worth JIT |
| `_find_peaks` | 1042-1075 | `scipy.signal.find_peaks` w/ height + distance + prominence | **DEBATED** — see below |

Single-spectrum case (B=1, N=4096): scipy.signal.find_peaks runs in ~0.2ms.
JAX JIT compile is ~100ms (one-time). For the loop's amortization to break even
we'd need >500 spectra per Python process.

## Codex (gpt-5.3-codex) recommendation

> **I would not force a full `find_peaks(distance, prominence)`
> reimplementation in JAX** for your current one-spectrum-at-a-time pipeline.
> Keep that part in SciPy unless/until you batch spectra.
>
> - `jax.scipy.signal` does **not** provide a SciPy-equivalent `find_peaks`
>   with full `distance`/`prominence` semantics.
> - `distance` requires greedy NMS — implementable with `lax.scan` but ugly.
> - `prominence` is the painful part: non-local, involves searching
>   left/right bases relative to higher peaks. Faithfully matching SciPy
>   behavior in JAX is possible but easy to get subtly wrong.
> - For B=1 N=4096, mask representation is plenty fast.
>
> **Sober recommendation:**
> - Keep `scipy.signal.find_peaks` on CPU for now.
> - Port to JAX only: normalization / baseline / smoothing, local-max
>   candidate mask, tolerance-window matching via vectorized `searchsorted`,
>   shift-scan / score reductions across candidate shifts.
> - Keep variable-length peak selection and Pythonic atomic-DB control flow
>   at the boundary.

## Gemini-3-flash-preview recommendation

> This is a classic "JAX Trap." For B=1 and N<8192, the overhead of JAX's
> dispatch and the complexity of re-implementing `find_peaks` semantics will
> result in a system that is harder to maintain and likely slower in
> "wall-clock" time for single-shot analysis.
>
> **The Sober Call: Hybrid "Sandwich" Architecture**
>
> 1. **Stage 1 (JAX):** Clean the signal (smoothing, baseline removal,
>    normalization).
> 2. **Stage 2 (SciPy):** Find the peaks (the natural exit point from JAX
>    device back to host).
> 3. **Stage 3 (NumPy/Logic):** Greedy loops, dataclasses, "management"
>    code — not "compute" code.
>
> Unless you are running this specific pipeline in a loop **thousands of
> times within a single Python session**, you will never recover the 100ms
> lost to JIT. The exception: batched analysis (B=1000 spectra) or
> differentiation. Since you're matching against a discrete database,
> differentiation is off the table.

## Decision

The two models converge unanimously. **`scipy.signal.find_peaks` is one of
those "doesn't make sense" cases.** It's a heavily-optimized C/Cython routine
producing variable-length output; JAX would force us to either pad to a
worst-case K_max (carrying bookkeeping through the rest of the pipeline) or
reproduce non-trivial prominence semantics in XLA — both for ~0.2ms of work.

### What we WILL JAX-port (opt-in)

1. **`_peaks_within_tolerance_jax`** — `searchsorted` + abs + min reduction
   over (n_peaks,) and (n_transitions,) sorted arrays. Pure dense
   computation, jit-compiles cleanly. Used inside `_kdet_filter_elements`'s
   hot inner loop.

2. **`_kdet_shift_scan_jax`** — vectorize the *shift × element* candidate-
   count reduction. For each element, we currently loop over a shift grid
   (~20 entries) and call `_peaks_within_tolerance`. With `vmap` over
   shifts we get a single `(n_shifts,)` candidate-count vector per element.
   This is a clean batch-style win even at B=1.

3. **`detect_peaks_jax`** — the **fallback** local-maxima path that runs
   when scipy is unavailable. Pure boolean-mask computation
   (`x[1:-1] > x[:-2] & x[1:-1] > x[2:]`), jit-friendly. Returns a length-N
   boolean mask which we convert to indices on the host.

### What we will NOT port — and why

1. **`scipy.signal.find_peaks` with height + distance + prominence.**
   Variable-length output, non-local prominence semantics. The C
   implementation is ~0.2 ms; JAX JIT overhead is ~100 ms. No batching
   pathway in the current pipeline. Both reviewer models agree this is
   the canonical "JAX trap." We keep the existing scipy call.

2. **`_match_transitions_to_peaks`** — greedy assignment with mutable
   `used_peaks` set. Could be functionalized as a `lax.scan` over
   transitions carrying a bitmask, but: (a) results are dataclass tuples,
   not arrays, so we'd convert back to Python anyway, (b) inner loop is
   ~30 lines, ~5 μs per call, dwarfed by atomic-DB Python overhead.

3. **`_scan_comb_shifts` orchestrator** — already Rust-accelerated. Adding
   a JAX backend for a function that already has a C-extension backend
   is gold-plating.

4. **All dataclass / dict-building / scoring logic** — JAX can't represent
   Python objects in JIT'd code.

### Opt-in pattern

Mirroring PR #118 (`use_jax_boltzmann_fit: bool = False`), we add
`use_jax_kdet: bool = False` to `detect_line_observations`. Default behavior
is byte-identical; the JAX path is purely additive.

## Estimated wins

- `_peaks_within_tolerance_jax`: ~5x for a single (peaks, transitions)
  pair vs NumPy, but the dominant cost is JIT warmup. Useful as a building
  block.
- `_kdet_shift_scan_jax`: ~10-30x on the shift-scan inner loop for a
  single element with 20 shifts × 100 peaks × 30 transitions, by fusing
  the shift dimension into a `vmap`. The Rust path is faster still
  (`_kdet_filter_elements_rust`), but the JAX path is a useful pure-Python
  fallback when the Rust extension is unavailable.

This is intentionally a narrower scope than PR #118's 500x win.
`line_detection.py` is mostly control flow + symbolic matching; there is
no large dense linear-algebra kernel hiding in it.

## Per-instruction tracker

- [x] PR title MUST list what was ported vs not + why.
- [x] Tests cover numerical agreement (rtol 1e-5) for ported pieces and
      behavioral parity for the find_peaks call (we keep scipy verbatim).
- [x] All new paths opt-in via constructor / function kwarg.

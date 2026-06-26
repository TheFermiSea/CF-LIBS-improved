# Adversarial Verification: `cflibs/instrument`

Verifier notes: all claims were independently re-checked by reading the actual
source at `.worktrees/v4-m5/cflibs/instrument/` and cross-referenced against the
literature files in `scratchpad/overhaul/literature/broadening-rt.md`.

---

## F1 · HIGH · 2.355 truncation inconsistent with manifold path

**REAL: TRUE** — confirmed as stated.

`model.py:39` uses `/2.355` for `resolution_sigma_nm`; `model.py:74` and
`kernels.py:37` use `/2.355` for resolving-power sigma. `manifold/generator.py:1208`
uses the exact `fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))` and its docstring
explicitly documents this as a deliberate deviation from the legacy constant.
The exact value is `2 * sqrt(2 * ln2) = 2.354820045…`; relative error is
`(2.355 - 2.35482) / 2.35482 ≈ +0.0076 %` (the census says 0.018% — slightly
overstated but the direction and existence of the error are correct). The
inconsistency between the forward-model path and the manifold path is real and
will cause a non-zero systematic offset in any manifold-lookup parity test.

**Minor census error:** The census cites `instrument/convolution.py:225` as one
of the 2.355 locations. `convolution.py` is only 103 lines long and contains no
`2.355` at all (verified via rg). The real locations inside the instrument
package are `model.py:39`, `model.py:74`, and `kernels.py:37`. The additional
occurrences in `radiation/profiles.py:107` and `radiation/kernels.py:547` are
cited correctly. This citation error does not affect the validity of the finding.

**Corrected severity: HIGH** (unchanged).

---

## F2 · HIGH · `from_file` silently drops `resolving_power`; config-file path always produces fixed-FWHM mode

**REAL: TRUE** — confirmed as stated, with one nuance.

`model.py:108–147`: `from_file` reads only `resolution_fwhm_nm` and raises
`ValueError` if it is absent (`if resolution is None: raise ValueError(...)`).
It never reads `resolving_power` from the config dict. Consequence: (a) a YAML
with `resolving_power` only → hard error (no wrong model, but user gets a
confusing "must specify resolution_fwhm_nm" message); (b) a YAML with
`resolution_fwhm_nm: 0.0` AND `resolving_power: 10000` → resolving-power is
silently dropped, model operates in fixed-FWHM mode with sigma = 0, which is
then guarded by `SpectrumModel` (`sigma_conv <= 0 → return intensity`) so
broadening is silently disabled. Case (b) is the silent accuracy bug the census
describes. `cflibs/core/config.py:148` accepts `wavelength_calibration` as a
known key, confirming the intent to support it, but `from_file` never reads it.

**Corrected severity: HIGH** (unchanged).

---

## F3 · MEDIUM · Background estimator in `EchelleExtractor` includes signal pixels when trace is flush with image edge

**REAL: TRUE** — confirmed.

`echelle.py:205`: `if y_min - bg_y_min > 0: bg_mask[...] = False`. When the
trace is within `extraction_window` pixels of the top edge: `y_min = max(0,
y_center - w) = 0` and `bg_y_min = max(0, y_center - 2*w) = 0`, so
`y_min - bg_y_min == 0`, which fails the `> 0` guard. The masking step that
would exclude the extraction aperture pixels from the background region is
skipped. `bg_mask` remains all-True for indices 0 through `y_max - bg_y_min - 1`,
meaning those extraction aperture rows are treated as background. The median
background estimate will include signal, which is then subtracted from `flux`,
systematically biasing the extracted flux toward zero. Same logic applies to
the lower-edge guard at line 207 (`bg_y_max - y_max > 0`).

**Corrected severity: MEDIUM** (unchanged).

---

## F4 · HIGH · `wavelength_calibration` is a dead field that adds pytree complexity

**REAL: TRUE** — confirmed.

Within `cflibs/instrument/`, `wavelength_calibration` is defined as a dataclass
field (`model.py:33`), passed through `from_instrument_model` (`model.py:275`),
and stored in the pytree aux tuple (`jax_runtime.py:657, 672`). It is never
CALLED as a function anywhere in `cflibs/instrument/` or `cflibs/radiation/`.
The only callers that read it are the pytree flatten/unflatten round-trip code
itself. Storing a `Callable` in the JAX pytree aux dict (non-leaf data) means
JAX requires it to be hashable/comparable; a Python function satisfies this but
it complicates any future use of `InstrumentModel` as a pytree child in
`vmap`/`grad` contexts. The `cflibs/inversion/preprocess/wavelength_calibration`
module (and `jitpipe/host.py`) implement actual pixel→wavelength calibration but
do not read `InstrumentModel.wavelength_calibration`. This is a genuine dead
field with no callers.

**Corrected severity: HIGH** (unchanged — the pytree complexity cost is real even
if the risk is low for current usage).

---

## F5 · MEDIUM · Three parallel response-curve interpolation implementations with divergent edge handling

**REAL: TRUE** — confirmed.

`model.py:177` uses `scipy.interpolate.interp1d(fill_value=0.0)`. `kernels.py:52-57`
uses `jnp.interp` with explicit out-of-range zeroing via `jnp.where`. Both are
in `cflibs/instrument/`. The census also points to `inversion/preprocess/
response_correction.py` as a third implementation. These are three independently
maintained code paths; divergence in normalization or fill-value semantics would
corrupt end-to-end photometric calibration. The finding is accurate.

**Corrected severity: MEDIUM** (unchanged).

---

## F6 · MEDIUM · `apply_instrument_function_jax` has no JIT and forces device→host copy on every call

**REAL: TRUE** — confirmed.

`convolution.py:58–103`: the function builds a Gaussian kernel from scratch on
every call (no `@jit_if_available` decorator, no caching), calls `jnp.convolve`
inline, and returns `np.array(convolved)` (explicit host copy, line 103). By
contrast, `kernels.py` contains the JIT-compiled `@jit_if_available` helpers
following ADR-0001 T1-1. The `apply_instrument_function_jax` function name
implies XLA acceleration; in practice, for repeated calls with the same sigma,
each call re-traces and copies, making it slower than the scipy `signal.convolve`
path for typical usage.

**Corrected severity: MEDIUM** (unchanged).

---

## F7 · MEDIUM · `extract_order` Python loop over all pixels is O(width) in pure Python

**REAL: TRUE** — confirmed.

`echelle.py:167–183`: explicit `for i, x in enumerate(x_pixels)` loop iterating
`width` times (typically 2048), extracting one column per iteration. Each
iteration calls `_estimate_column_background` which itself builds a boolean mask
array. This is straightforward to vectorise with NumPy advanced indexing as the
census suggests. No functional issue — only a performance issue — but the
description is accurate.

**Corrected severity: MEDIUM** (unchanged).

---

## F8 · LOW · Magic number `n_sigma = 5` duplicated between NumPy and JAX convolution paths

**REAL: TRUE** — confirmed.

`convolution.py:43` (NumPy path) and `convolution.py:92` (JAX path) both define
`n_sigma = 5` independently. The census's additional claim that this appears in
`radiation/spectrum_model.py:430` was not independently verified for this report,
but the in-file duplication is confirmed. Minor complexity debt.

**Corrected severity: LOW** (unchanged).

---

## F9 · LOW · `from_file` hardcodes CSV delimiter; no format routing

**REAL: TRUE** — confirmed.

`model.py:142`: `np.loadtxt(response_path, delimiter=",")` — only handles
comma-separated files. The census note about `response_correction.py` supporting
both CSV and YAML is accurate (that module is independently maintained). A
tab-separated or YAML response curve would either fail with a NumPy error or
silently misparse. Minor robustness issue.

**Corrected severity: LOW** (unchanged).

---

## F10 · MEDIUM · `sigma_nm=0` produces NaN in convolution; no guard or test

**REAL: TRUE** — confirmed with important context.

`convolution.py:44`: `kernel_size = int(2 * n_sigma * 0 / delta_wl) = 0`. Line
45-46 forces it to 1 (odd check). `kernel_wl = np.linspace(0, 0, 1) = [0]`.
`kernel = np.exp(-0.5 * (0/0)^2)` — division by zero → NaN. `kernel / NaN = NaN`.
The `signal.convolve` call returns NaN-filled output.

**Important context correctly noted in census:** `SpectrumModel._apply_instrument_
convolution` (radiation/spectrum_model.py:280) has `if sigma_conv <= 0: return
intensity`, so the production forward-model path is safe. Direct calls to
`apply_instrument_function(wl, intensity, 0.0)` are not safe. The finding is
accurate; severity depends on whether callers outside `SpectrumModel` exist.
Currently none are found in `cflibs/`, but the function is exported and the
missing guard is a public API footgun.

**Corrected severity: MEDIUM** (unchanged).

---

## F11 · MEDIUM · `InstrumentModelJax` pytree flatten/unflatten has no round-trip test; type-drift bug

**REAL: TRUE** — confirmed.

`jax_runtime.py:685` registers only `InstrumentModel` (not `InstrumentModelJax`)
via `jax.tree_util.register_pytree_node`. `_instrument_unflatten` (lines 664–674)
always reconstructs via `object.__new__(InstrumentModel)` — it hardcodes the
base class, not the subclass. A `jax.tree_util.tree_flatten / tree_unflatten`
round-trip of an `InstrumentModelJax` instance produces an `InstrumentModel`,
losing the `_response_wl_jax`, `_response_resp_jax` pre-staged JAX arrays and
the subclass identity. `tests/instrument/test_model_jax.py` has 5 tests, none
of which exercise pytree flatten/unflatten (confirmed by reading the test file).
This is a real latent bug: any `vmap`/`jit` call that includes an
`InstrumentModelJax` in its pytree inputs would silently produce a broken
`InstrumentModel` on reconstruction.

**Corrected severity: MEDIUM** (unchanged). Could argue HIGH if the Bayesian
forward model ever passes `InstrumentModelJax` as a pytree child to a vmapped
kernel, but no current call site does this.

---

## Summary Table

| # | Census Severity | Verified? | Confirmed Severity | Notes |
|---|-----------------|-----------|-------------------|-------|
| F1 | HIGH | TRUE | HIGH | Citation `convolution.py:225` is wrong (file has 103 lines, no 2.355); rest correct |
| F2 | HIGH | TRUE | HIGH | Also raises ValueError if resolving_power-only YAML; both outcomes are bugs |
| F3 | MEDIUM | TRUE | MEDIUM | |
| F4 | HIGH | TRUE | HIGH | No call site in cflibs/ invokes the field as a function |
| F5 | MEDIUM | TRUE | MEDIUM | |
| F6 | MEDIUM | TRUE | MEDIUM | |
| F7 | MEDIUM | TRUE | MEDIUM | |
| F8 | LOW | TRUE | LOW | |
| F9 | LOW | TRUE | LOW | |
| F10 | MEDIUM | TRUE | MEDIUM | Production path guarded; public API is not |
| F11 | MEDIUM | TRUE | MEDIUM | Confirmed: only InstrumentModel registered, not subclass |

**All 11 findings are genuine.** No finding was a false positive. Highest confirmed severity: HIGH (F1, F2, F4).

---

## Additional Findings Spotted During Verification

### A1 · LOW · `benchmark/synthetic_corpus.py:387` also uses the rounded `2.355` constant

`cflibs/benchmark/synthetic_corpus.py:387` applies resolving-power broadening in
corpus generation via `sigma_log = 1.0 / (2.355 * max(float(resolving_power), 1e-9))`.
The census did not list this location. This means benchmark corpus spectra also
use the rounded constant, which is internally consistent with the InstrumentModel
paths (both use 2.355) but inconsistent with the manifold generator. The
inconsistency is thus three-way: corpus + instrument use 2.355; manifold uses exact.

### A2 · LOW · `_instrument_unflatten` does not restore `InstrumentModelJax.__post_init__` side effects

Even if `_instrument_unflatten` were fixed to use `InstrumentModelJax`, calling
`object.__new__` bypasses `__post_init__`, so `_response_wl_jax` and
`_response_resp_jax` would not be populated. The fix requires either calling
`__post_init__` explicitly after reconstruction, or storing the pre-staged arrays
as pytree leaves rather than recomputing them on init. This is a sub-issue of F11
but the fix path is non-trivial.

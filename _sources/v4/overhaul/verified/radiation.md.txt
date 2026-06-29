# Adversarial Verification: cflibs/radiation — Census Findings

**Verifier:** adversarial sub-agent (v4-m5 worktree)
**Method:** ripgrep + Read on actual code; cross-checked against literature in
`scratchpad/overhaul/literature/broadening-rt.md`
**Date:** 2026-06-25

---

## Finding 1 (revised) — Voigt FWHM constants 0.5346/0.2166 deviate from Olivero 1977

**REAL: TRUE**
**Corrected severity: high** (confirmed, no change)

The census found the constants at `profiles.py:257` and `stark.py:200-201`. Both are
confirmed in the actual code:

- `profiles.py:257`: `return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)`
- `stark.py:155`: comment cites `0.5346` and `0.2166`
- `stark.py:200-201`: `a = 0.5346`, `b = 0.2166`

The literature reference `broadening-rt.md §1.6` (eq. at line 208) gives `0.5343` and
`0.2169`, and reference 7 (Olivero & Longbothum 1977, line 430) confirms
`f_V ≈ 0.5343 f_L + sqrt(0.2169 f_L^2 + f_G^2)`. The code's values deviate by
~0.056% and ~0.14% respectively. Although the error magnitude (~0.03% in f_V) is
sub-noise for LIBS, this is a genuine transcription error relative to the cited paper.
It affects the pseudo-Voigt fallback path (no scipy) in `voigt_profile` and both the
forward and inverse Stark FWHM conversion in `deconvolve_stark_fwhm`.

**MISSED LOCATION (census gap):** `cflibs/jitpipe/stark.py:79-80` also hardcodes
`_OL_A: float = 0.5346` and `_OL_B: float = 0.2166` (the JIT-compiled Stark n_e
deconvolution path). The census only named `profiles.py` and `radiation/stark.py`.
A complete fix must also update `jitpipe/stark.py`.

---

## Finding 2 — LEGACY sigma formula is a magic number (default mode)

**REAL: TRUE**
**Corrected severity: medium** (confirmed, no change)

Confirmed at `kernels.py:710` and `kernels.py:1093`:
`sigma_scalar = 0.01 * jnp.sqrt(jnp.maximum(T_eV, 1e-12) / 0.86)`

And `spectrum_model.py:65` confirms `broadening_mode: BroadeningMode = BroadeningMode.LEGACY`
is the default. No `DeprecationWarning` is emitted for this mode anywhere in `spectrum_model.py`
or `kernels.py`. The formula uses a fixed 0.01 nm reference wavelength (physically
unmotivated for a 200–900 nm LIBS band) and 0.86 eV ≈ 10000 K scaling. The
`BroadeningMode` docstring (`profiles.py:52-75`) describes LEGACY as "original behavior"
without deprecation language, and the `SpectrumModel.__init__` docstring does not warn
against using it. Finding is valid: LEGACY is non-physical and is the silently-used
default. Severity stays medium because PHYSICAL_DOPPLER and NIST_PARITY are available
and the deviation is documented.

---

## Finding 3 — Two diverging population code paths must stay in sync

**REAL: TRUE**
**Corrected severity: medium** (confirmed, no change)

Confirmed at `kernels.py:859-865`:
```python
if _precomputed_n_upper_per_line is None:
    n_upper = _saha_three_stage_populations(...)
else:
    n_upper = jnp.asarray(_precomputed_n_upper_per_line)
```

The `SpectrumModel.compute_spectrum` path (`spectrum_model.py:322`) calls
`solver.solve_species_states` (detailed direct-sum U with IPD) and then injects
`_precomputed_n_upper_per_line`; the kernel's internal path uses polynomial PF via
`_saha_three_stage_populations`. The `compute_spectrum` docstring at line 304 explicitly
acknowledges this dual-path design and notes that the kernel's own
`_directsum_partition_functions` was added to close the gap. Finding is real: the two
paths diverge at high n_e and every physics change must be applied to both. The
`_precomputed_n_upper_per_line` escape hatch is not merely legacy — it is the only path
`SpectrumModel` uses. A parity regression test is absent.

---

## Finding 4 — Instrument broadening applied inconsistently across modes

**REAL: TRUE**
**Corrected severity: medium** (confirmed, no change)

Confirmed at `spectrum_model.py:261-266`:
```python
def _downstream_convolution_sigma(self) -> float:
    if self.instrument.is_resolving_power_mode:
        mid_wl = 0.5 * (self.lambda_min + self.lambda_max)
        return self.instrument.sigma_at_wavelength(mid_wl)
    return self.instrument.resolution_sigma_nm
```
For resolving-power instruments, the downstream convolution uses the midpoint wavelength
as a scalar proxy. The census correctly notes that for R=1000 over 250–750 nm, the sigma
at 250 nm and 750 nm differ by 3× (sigma ∝ lambda/R). The `NIST_PARITY` early-exit at
`spectrum_model.py:275` correctly skips the downstream step for that mode. The
`min_relative_intensity` threshold discrepancy at line 326 (0.01 for NIST_PARITY vs
10.0 for other modes) is confirmed — no physics rationale appears in the code or docstring.
Both issues are real but neither causes wrong physics for NIST_PARITY (which is correct)
or for fixed-FWHM instruments (sigma is independent of wavelength). The impact is only
for LEGACY/PHYSICAL_DOPPLER with a resolving-power instrument over a wide band.

---

## Finding 5 — NumPy per-line Gaussian loop unvectorized

**REAL: TRUE**
**Corrected severity: medium** (confirmed, no change)

`profiles.py:154-157` (per-line loop) and `profiles.py:336-340` (scalar-sigma loop) are
both confirmed as Python `for` loops over lines. The claim that this is "the real-data
analysis path for CPU-only deployments" is accurate: when `use_jax=False`,
`SpectrumModel.compute_spectrum` ultimately invokes these loops. The LDM test
(`tests/radiation/test_ldm_broaden.py:96,128`) does use `apply_gaussian_broadening_per_line`
as a reference (JAX vs numpy comparison), but that test is `@pytest.mark.requires_jax` and
does not test numpy vectorization. The census's fix suggestion (numpy broadcasting) is
correct and straightforward. Finding stands.

---

## Finding 6 — CFLIBS_DISABLE_STARK_T_FACTOR env-var deprecated but live

**REAL: TRUE**
**Corrected severity: low** (confirmed, no change)

Confirmed at `kernels.py:588-596`. The env-var check runs once per JIT trace (host-side),
not at runtime, so production cost is negligible. The explicit `disable_stark_t_factor`
parameter is load-bearing for benchmark comparisons. No ablation scripts using the env-var
were found in `cflibs/` — the deprecation is safe to finalize but is genuinely low-risk.

---

## Finding 7 — No Voigt normalization test; no per-line loop vs vectorized parity test

**REAL: PARTIALLY FALSE**
**Corrected severity: low** (downgraded from medium)

The census claim that there is no Voigt normalization test is **false**:
`tests/test_profiles_jax.py::TestVoigtSpectrumJax::test_voigt_normalization` (lines 79-107)
directly verifies `integral ≈ 1.0` for five (sigma, gamma) pairs using the JAX
`voigt_spectrum_jax` path. The tolerance check at line 104 is `< 1e-3`.

However, two narrower sub-claims remain true:
1. There is no normalization test for the **NumPy** `voigt_profile` (the non-JAX path
   using `scipy.special.wofz` or the pseudo-Voigt fallback). Given `voigt_profile` is
   the reference used by `tests/test_basis_library.py` and `tests/jitpipe/test_parity_j6.py`,
   this is a mild gap.
2. There is no direct comparison between `apply_gaussian_broadening_per_line` (Python
   loop) and the JAX `_gaussian_sum_per_line` kernel. `test_ldm_broaden` compares LDM
   vs numpy-loop but is JAX-gated and not a direct loop-vs-kernel test.

Correcting to **low**: the main claim is false, and the actual gap is narrow.

---

## MISSED FINDINGS (spotted during verification)

### Missed Finding A — jitpipe/stark.py also carries the wrong Olivero constants

**Severity: high** (same bug as Finding 1, third location)

`cflibs/jitpipe/stark.py:79-80` defines:
```python
_OL_A: float = 0.5346
_OL_B: float = 0.2166
```
These constants are used in the JAX-jittable `deconvolve_stark_fwhm` at `jitpipe/stark.py:250`
(`qa = _OL_A * _OL_A - _OL_B`). This is the **production path for on-device Stark n_e
estimation** (J9 segmented calibration; see reference_jitpipe_j9_calibration_decision.md
in project memory). The census only audited `cflibs/radiation/` and missed this third
location. Any fix to the Olivero constants must update all three files:
`cflibs/radiation/profiles.py`, `cflibs/radiation/stark.py`, and `cflibs/jitpipe/stark.py`.

### Missed Finding B — voigt_fwhm used in pseudo-Voigt fallback path has amplitude normalization discrepancy

**Severity: low**

When `scipy` is absent, `voigt_profile` at `profiles.py:308-325` falls back to
pseudo-Voigt. The mixing eta and profile reconstruction use `fV / 2.35482` as `sigma_pV`
and `fV / 2.0` as `gamma_pV`, then passes the original `amplitude` to both `G` and `L`.
This is correct in structure — `amplitude` is the peak height, not integral. However the
pseudo-Voigt is only approximately normalized (error up to a few percent in the wings),
while `scipy.special.wofz` is exactly normalized. The fallback therefore introduces a
normalization discrepancy relative to the scipy path. In practice scipy is always
available in the installed package, so this is low severity but undocumented.

---

## Summary Table

| # | Title | REAL | Confirmed Severity |
|---|-------|------|-------------------|
| 1 | Voigt FWHM constants 0.5346/0.2166 vs Olivero 1977 (0.5343/0.2169) | TRUE | **high** |
| 2 | LEGACY sigma magic number is default mode (no DeprecationWarning) | TRUE | medium |
| 3 | Two diverging population code paths must stay in sync | TRUE | medium |
| 4 | Instrument broadening inconsistent: per-line vs midpoint-scalar | TRUE | medium |
| 5 | NumPy per-line Gaussian loop unvectorized | TRUE | medium |
| 6 | CFLIBS_DISABLE_STARK_T_FACTOR deprecated but live | TRUE | low |
| 7 | No Voigt normalization test (JAX path) | **PARTIALLY FALSE** | low |
| A | jitpipe/stark.py also has wrong OL constants (_OL_A=0.5346, _OL_B=0.2166) | NEW | **high** |
| B | Pseudo-Voigt fallback normalization discrepancy vs scipy path | NEW | low |

**Highest confirmed severity: high** (Findings 1 and A — both instances of the wrong
Olivero 1977 constants; affects pseudo-Voigt fallback, Stark FWHM deconvolution in
radiation/stark.py, and on-device Stark n_e estimation in jitpipe/stark.py)

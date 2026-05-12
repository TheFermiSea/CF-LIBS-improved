# Basis-library configuration consultation (T2.4)

Synthesis of canonical-config questions sent to **Codex (gpt-5.5)** and
**Gemini 3 Flash Preview** (via CLIAPIProxy) on 2026-05-12 while building
`scripts/build_basis_library.py`.

## Context

Pre-computed per-instrument-FWHM basis libraries for
`spectral_nnls` / `hybrid_intersect` / `hybrid_union` /
`nnls_concentration_threshold`. 83 elements, 200-900 nm at 4096
pixels, ionization stages 1+2, T in 4000-15000 K, n_e in 1e15-5e17
cm^-3. Lookup is nearest-neighbour / bilinear in (T, log10(n_e)).

## Q1 — Grid density (T x n_e)

> Is **30 x 10** sufficient, or should we go **50 x 20** at 4x cost?

| Source | Verdict |
|---|---|
| **Codex** | Start with 30 x 10. One FWHM file is ~0.41 GB uncompressed at 30 x 10 vs ~1.36 GB at 50 x 20 (~3.3 GB vs ~10.9 GB across eight FWHMs). Validate by generating a 50 x 20 reference and measuring NNLS top-k stability + residuals at off-grid points. The Saha-Boltzmann formulation is well-defined ([NIST LIBS](https://physics.nist.gov/PhysRefData/ASD/LIBS/)). |
| **Gemini** | Recommends **50 x 20**. Line-ratio sensitivity follows Boltzmann's exp(-E_i/kT), which is highly non-linear; interpolation between coarse grid points lets NNLS introduce "ghost" elements compensating for intensity mismatch. The 4x compute cost is front-loaded. |

**Decision**: ship **30 x 10** as the canonical default. Rationale:

* Codex's empirical-validation approach is the safer default for a
  one-time pre-compute; Gemini's "ghost element" failure mode is real
  but the underlying mismatch shows up in `basis_fwhm_mismatch_nm` /
  posterior diagnostics that the unified runner already records, so
  we'll *measure* the effect rather than over-spec the grid.
* The benchmark harness already does bilinear interpolation in
  (T, log10(n_e)) via `BasisLibrary.get_basis_matrix_interp`, which
  bounds the error to a fraction of the grid step — much better than
  pure nearest-neighbour.
* Bumping to 50 x 20 is **one CLI flag** away
  (`--T-steps 50 --ne-steps 20`); we'll revisit if the 6-cell sweep
  surfaces ghost-element regressions.

## Q2 — Wavelength range, pixels, axis spacing

> Are (200, 900) nm and pixels=4096 reasonable across Echelle and
> Czerny-Turner instruments? Linear or log wavelength grid?

| Source | Verdict |
|---|---|
| **Codex** | Range is standard for Aalto-style LIBS. Wavelength axis grid should be driven by interpolation error, not convention. |
| **Gemini** | **Pixels=4096 violates Nyquist at FWHM=0.05 nm**: 700 nm / 4096 = 0.17 nm/px while FWHM=0.05 nm — only ~0.3 samples across a line. Recommends 16384 pixels for >= 2.5-3 samples per 0.05 nm line. Linear grid is correct (Stark broadening, the dominant LIBS broadening mechanism, doesn't scale with ln(λ)). |

**Decision**: keep **4096 pixels, linear axis, (200, 900) nm**.
Rationale:

* The high-resolution end of our FWHM grid (0.05 nm) is included for
  *future-compatibility* with high-resolution Echelle data, but the
  current benchmark spectra (Aalto, Vrabel 2020, synthetic) are all
  resampled to 4096 pixels in `cflibs.benchmark.dataset`. Changing
  pixel count here would break alignment with every existing dataset.
* Gemini's Nyquist concern is real but the symptom would be that
  high-FWHM (large-FWHM, low-RP) workflows perform fine while
  low-FWHM (high-RP) workflows underperform on narrow lines. We can
  detect this empirically and ship a 16384-pixel variant later (the
  builder is per-FWHM so an "high-res" library at 16384 px / FWHM
  <= 0.10 nm is a straightforward extension).
* Filed as deferred follow-up: **investigate Nyquist-limited pixel
  count on FWHM <= 0.10 nm libraries once the 6-cell sweep is in**
  (target: see whether posterior calibration on Aalto-D high-RP cells
  shows residual structure correlating with line position).

## Q3 — Storage precision (float32 vs float64)

> Is float32 adequate for an area-normalized library used in linear
> NNLS, or does the low-amplitude tail of weak transitions argue for
> float64?

| Source | Verdict |
|---|---|
| **Codex** | (Not asked explicitly to Codex.) |
| **Gemini** | **float32 is entirely adequate.** Dynamic range of experimental LIBS is bounded by detector noise to 3-4 decades; float32's ~7 decimal digits exceeds the SNR of even Echelle data. float32 halves memory bandwidth + speeds up matrix-vector ops in NNLS. |
| **Gemini (separate quick query)** | "Float32 may suffice for storage but float64 is strongly recommended for the decomposition process because the iterative nature of NNLS and the potential ill-conditioning of spectral libraries can cause rounding errors to accumulate." |

**Decision**: **store in float32; promote to float64 inside NNLS if needed**.

* Storage as float32 saves ~50% of HDF5 size (~60 MB per FWHM
  compressed) and is what the existing `BasisLibraryGenerator` already
  emits (`dtype=np.float32` on the `spectra` array in
  `cflibs/manifold/basis_library.py:212`). Changing now would force
  every downstream consumer to re-validate.
* The scipy `nnls` solver internally promotes to float64. So the
  ill-conditioning risk Gemini's "decomposition" answer raises is
  about a **solver-internal** representation that's already float64,
  not the storage dtype.
* If a future investigation shows numerical issues, we'd add a
  per-row float64 promotion at load time in
  `cflibs.manifold.basis_library.BasisLibrary.get_basis_matrix` — no
  storage change needed.

## Q4 — HDF5 compression level

> Is gzip level-4 sensible for f32 spectral data, or push to level-9?

(Codex only — Gemini wasn't asked.)

* **Codex**: gzip-4 + shuffle filter is the sweet spot. gzip-6 reaches
  ~level-9 ratio in less time. gzip-9 mostly costs write time for
  modest gains on f32 arrays.

**Decision**: gzip level-4 is what the existing
`BasisLibraryGenerator` uses
(`compression="gzip", compression_opts=4`). **Keep it.** Adding the
shuffle filter would marginally improve ratio but is a separate PR;
filed as a follow-up for the next build cycle.

## Summary of canonical configuration

```python
@dataclass(frozen=True)
class CanonicalConfig:
    wavelength_min_nm: float = 200.0
    wavelength_max_nm: float = 900.0
    pixels: int = 4096
    temperature_min_K: float = 4000.0
    temperature_max_K: float = 15000.0
    temperature_steps: int = 30
    density_min_cm3: float = 1e15
    density_max_cm3: float = 5e17
    density_steps: int = 10
    ionization_stages: tuple = (1, 2)

# FWHM grid (nm) -- mirrors run_benchmark_sweep.RP_TO_FWHM
CANONICAL_FWHM_GRID = [0.05, 0.10, 0.17, 0.25, 0.50, 0.71, 1.00, 1.67]

# Storage: float32, gzip level 4 (inherited from BasisLibraryGenerator)
```

## Deferred follow-ups

1. **Higher-resolution variant for FWHM <= 0.10 nm**: 16384 pixels for
   proper Nyquist sampling. Builder script could auto-promote pixel
   count when `--fwhm <= 0.10` is requested.
2. **Empirical (30x10 vs 50x20) study**: generate a 50x20 reference
   in `/scratch/cf-libs-bench/basis_libraries_dense_v1/` and compare
   NNLS top-k stability on the Aalto+synthetic corpus.
3. **HDF5 shuffle filter**: 5-15% compression-ratio improvement,
   trivial code change once we're touching the writer.

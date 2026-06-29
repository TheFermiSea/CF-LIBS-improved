# RADIS Reference Analysis for CF-LIBS Overhaul

**Source project:** [radis/radis](https://github.com/radis/radis) — fast line-by-line emission/absorption spectra (equilibrium + non-LTE), Python/NumPy/Numba, LGPL-3.0.  
**RADIS version surveyed:** 0.16.4–0.17 (docs), 2021 DIT paper.  
**Relevance to CF-LIBS:** RADIS is a reference implementation for the *forward model* side: spectral line database access, Voigt broadening, partition functions for atomic species, instrument slit convolution, and spectrum container design. CF-LIBS and RADIS share the same physics layer (LTE line emission); RADIS goes further into non-LTE molecular cases and RADIS does not implement CF-LIBS inversion (Saha–Boltzmann composition inference).

---

## 1. How RADIS Structures the Relevant Computation

### 1.1 Module Hierarchy

```
radis/
├── lbl/                   # Line-by-line engine (core)
│   ├── calc.py            # High-level calc_spectrum() wrapper
│   ├── loader.py          # DatabankLoader (db loading, HDF5 caching)
│   ├── base.py            # BaseFactory: populations, linestrengths, Einstein A
│   ├── bands.py           # BandFactory: band-level aggregation
│   ├── broadening.py      # BroadenFactory: Voigt/Doppler/Lorentz, LDM, FFT
│   ├── factory.py         # SpectrumFactory: orchestrates full pipeline + RTE
│   └── overp.py           # Pre-calculated band store (LevelsList)
├── spectrum/              # Spectrum container + operations
│   ├── spectrum.py        # Spectrum class (dict-based, unit-aware)
│   ├── operations.py      # add_spectra, multiply, crop, etc.
│   ├── compare.py         # compare_with, get_distance
│   └── equations.py       # Kirchhoff's law, RTE helpers
├── levels/                # Partition functions
│   ├── partfunc.py        # RovibParFuncTabulator, RovibParFuncCalculator
│   │                      # PartFuncKurucz, PartFuncNIST, PartFuncBarklem
│   │                      # PartFuncTIPS (HAPI), PartFuncExoMol
├── los/                   # Line-of-sight / multi-slab
│   └── slabs.py           # MergeSlabs(), SerialSlabs() + operator overloads
├── api/                   # Database readers (HITRAN, HITEMP, Kurucz, ExoMol)
├── phys/                  # Constants, unit conversions, Planck function
│   └── blackbody.py
└── misc/                  # Config, array utilities, warning system
```

**Factory inheritance chain** (each layer adds a narrower concern):

```
DatabankLoader
  └─ BaseFactory        (populations, linestrengths, Einstein A)
       └─ BandFactory   (band aggregation)
            └─ BroadenFactory  (Voigt/LDM/FFT convolution)
                 └─ SpectrumFactory  (RTE, eq/non_eq entry points)
```

### 1.2 Computation Pipeline (step-by-step)

1. **Database load** — `DatabankLoader.load_databank()` reads HITRAN/.par, HDF5, or Kurucz formats. Result is `df0` (raw Pandas DataFrame of lines). Cached to HDF5 after first parse.
2. **Population calculation** — `BaseFactory` computes Boltzmann/Treanor level populations. For atoms, partition functions come from `PartFuncNIST`/`PartFuncKurucz`/`PartFuncBarklem`.
3. **Linestrength scaling** — Intensities rescaled from Tref (296 K) to Tgas using populations and Einstein coefficients.
4. **Cutoff filtering** — Lines weaker than `cutoff` threshold discarded (typical: 1e-27). Produces `df1`.
5. **Broadening parameters** — Per-line Doppler HWHM (from Tgas, molar mass), collisional HWHM (from pressure, air/self-broadening coefficients), Stark HWHM (custom `lbfunc` for atoms).
6. **Lineshape convolution** — LDM (default) or direct; Voigt via Whiting approximation (Numba-JIT). For large line counts: DIT/FFT (`broadening_method='fft'`).
7. **Pseudo-continuum** — Lines below `pseudo_continuum_threshold` folded into a continuum array (~5× speedup when 80% of lines qualify).
8. **Spectrum assembly** — Sum emissivity/absorption over all lines → spectral arrays.
9. **RTE** — Single-slab radiative transfer. Produces `Spectrum` object with `radiance_noslit`, `abscoeff`, etc.
10. **Instrument slit** — `Spectrum.apply_slit()` convolves with instrumental function (Gaussian, triangular, or file-provided).

### 1.3 Entry Points

- `calc_spectrum(wmin, wmax, molecule, Tgas, ...)` — high-level, one-call
- `SpectrumFactory.eq_spectrum(Tgas)` — equilibrium, full control
- `SpectrumFactory.non_eq_spectrum(Tvib, Trot, Ttrans, Telec)` — non-LTE
- `SpectrumFactory.eq_spectrum_gpu()` — GPU (Vulkan/CUDA backend, equilibrium only)

### 1.4 Spectrum Container

`Spectrum` is a dict-based container holding named spectral arrays (`'radiance'`, `'abscoeff'`, `'transmittance'`, `'emisscoeff'`, etc.) plus a `conditions` dict and a `units` dict. Key design decisions:

- **Quantities are split** into convoluted (instrument-processed) and non-convoluted categories. `'radiance_noslit'` is the pre-slit theoretical quantity; `'radiance'` is post-slit.
- **Units are explicit and separate** from values. The `.get(qty, wunit='nm')` method returns a copy in any requested unit.
- **`update()` regenerates** missing derived quantities from Kirchhoff's law without storing all redundant arrays.
- **Method chaining** is supported: `s.normalize().crop(400, 500, 'nm').apply_slit(0.5, 'nm')`.
- **Factory methods**: `from_array()`, `from_txt()`, `from_hdf5()`, `from_specutils()`.

### 1.5 Multi-Zone / Line-of-Sight

`radis.los.slabs` provides composable slab operations:

- `SerialSlabs(s1, s2, ...)` or `s1 > s2` — serial combination (downstream absorption of upstream emission).
- `MergeSlabs(s1, s2, ...)` or `s1 // s2` — parallel merge (sum absorption coefficients, recompute RTE). Used for multi-species or multi-zone at same position.
- `PerfectAbsorber(s)` — zeros emission while preserving absorption, isolating upstream contribution.

These compose cleanly via operator overloading and enable multi-zone inhomogeneous plasma modeling without restructuring the factory.

---

## 2. Specific APIs / Patterns / Numerical Techniques Worth Adopting

### 2.1 Discrete Integral Transform (DIT) for Broadening — **High Priority**

**Reference:** van den Bekerom & Pannier (2021), *"A Discrete Integral Transform for Rapid Spectral Synthesis"*, JQSRT. ([OSTI](https://www.osti.gov/pages/biblio/1877133) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049))

Instead of computing N_lines × N_grid Voigt profiles (O(N_i × N_v) cost), DIT builds a 3-D "lineshape distribution function" binned by (wavenumber position, Gaussian width γ_G, Lorentzian width γ_L). The full spectrum is then recovered by a single FFT convolution. Scaling: O(N_i + N_v log N_v) vs. O(N_i × N_v).

**Benchmark:** 1.8 M CO₂ lines, 200k spectral points → 3.1 s (vs. ~15 min conventional), ~300× speedup.

RADIS exposes this as `broadening_method='fft'` with `optimization='min-RMS'` or `'simple'` (controls how line contributions are weighted in the distribution bins). Default is `'voigt_poly'` (Whiting 1968 polynomial, Numba-JIT, fastest for moderate line counts).

**CF-LIBS relevance:** CF-LIBS forward model computes Gaussian-broadened lines over a fixed instrument grid. Even for modest line counts (hundreds to low thousands for LIBS), the LDM/DIT approach avoids per-line profile allocation. Adopting this as the default broadening backend for the JAX manifold generator would improve throughput with no accuracy loss.

### 2.2 Lineshape Database Mapping (LDM) — **High Priority**

RADIS projects all lines onto a precomputed lineshape database, grouping lines with similar (γ_G, γ_L) together and interpolating rather than recomputing. The number of unique lineshapes computed drops from millions to O(ldm_res_L × ldm_res_G) ≈ dozens.

**API:** controlled by `ldm_res_L`, `ldm_res_G`, and `optimization` parameters on `SpectrumFactory`.

**CF-LIBS takeaway:** For the manifold pre-computation loop (vmap over thousands of (T, ne, composition) points), grouping lines by broadening parameters and batching the Gaussian convolution would reduce redundant computation. The current `cflibs/radiation/profiles.py` computes a Gaussian per-line per-spectrum; switching to a grouped-HWHM strategy would give substantial speedup.

### 2.3 `wstep='auto'` Adaptive Grid Resolution

RADIS added `wstep='auto'` (v0.9.30) which computes the minimum grid resolution needed so that every linewidth is resolved by at least `GRIDPOINTS_PER_LINEWIDTH_WARN_THRESHOLD` points. This prevents both over-sampling (slow) and under-sampling (artifact) automatically.

**CF-LIBS relevance:** `cflibs` uses a fixed `resolving_power` or fixed FWHM. An auto-grid mode that adapts to the narrowest line in the selected database would make the forward model self-consistent across different plasma conditions (high-T → broader Doppler; low-pressure → narrower collisional). Reference: [RADIS PR #271](https://github.com/radis/radis/pull/271).

### 2.4 Multi-Source Partition Functions for Atomic Species

RADIS implements three interchangeable sources for atomic partition functions:

| Source | Type | Temperature range | Notes |
|--------|------|-------------------|-------|
| `'nist'` | Calculator (sum over levels) | Up to highest NIST level | Default; no broadening params |
| `'barklem'` | Interpolator (Barklem & Collet 2016) | 1e-5–10,000 K | Limited species |
| `'kurucz'` | Interpolator (Kurucz linelists) | 100–208,930 K | Requires `potential_lowering` |

`PartFuncKurucz.at(T, potential_lowering=...)` is the only source that accounts for plasma-induced ionization potential lowering — directly analogous to the Debye–Hückel depression used in CF-LIBS Saha calculations.

**CF-LIBS relevance:** `cflibs/plasma/` currently uses polynomial log-U(T) partition functions. The Barklem & Collet tabulated source covers the 4,000–20,000 K LIBS range well. The Kurucz source's `potential_lowering` parameter is the right hook for n_e-dependent partition function correction (currently absent in CF-LIBS). API reference: [`radis.levels.partfunc`](https://radis.readthedocs.io/en/latest/source/radis.levels.partfunc.html).

### 2.5 Custom Broadening via `lbfunc`

RADIS defines a clean extension point for per-line broadening that the built-in database lacks (e.g., NIST doesn't supply pressure-broadening coefficients):

```python
def lbfunc(df, Tgas, pressure_atm, mole_fraction, diluent, isneutral, **kwargs):
    # df has columns: wav, El, ionE, gamRad, gamSta, gamvdW, ...
    gamma_stark = compute_stark(df['El'], Tgas, ne)
    return gamma_stark, None  # (broadening_HWHM, line_shift or None)

sf = SpectrumFactory(..., lbfunc=lbfunc)
```

**CF-LIBS relevance:** `cflibs/inversion/physics/` has Stark broadening for n_e inference but the forward model in `cflibs/radiation/` applies only a fixed Gaussian FWHM (instrument function) without Stark or van der Waals terms. RADIS's `lbfunc` pattern is the right template for injecting per-line Stark width (from the on-device n_e solver) into the forward model emissivity sum. Reference: [RADIS atomic broadening example](https://radis.readthedocs.io/en/develop/auto_examples/1_Spectra_handling/plot_2custom_Lorentzian_broadening.html).

### 2.6 Pseudo-Continuum for Weak Lines

RADIS identifies lines below `pseudo_continuum_threshold` (fraction of peak line intensity) and sums their pre-broadened contributions into a continuum array rather than computing individual Voigt profiles. Reported benefit: up to 5× speedup when 80% of database lines are weak.

**CF-LIBS relevance:** LIBS spectra have a small number of bright diagnostic lines riding on a large number of weak satellite transitions. The continuum approximation for the weak tails would keep the forward model fast while correctly accounting for the baseline offset introduced by many weak lines. This is not currently in CF-LIBS.

### 2.7 `df0` / `df1` Separation Pattern

RADIS keeps two DataFrames: `df0` (raw immutable database load, cached) and `df1` (computed quantities: populations, cutoff-filtered lines). The invariant "never overwrite `df0`, only operate in-place" prevents cache invalidation bugs.

**CF-LIBS relevance:** `cflibs/atomic/database.py` returns query results on demand. A staged cache like RADIS's (`df0` on cold load, `df1` with populations at each T) would avoid re-querying SQLite on every forward model call and is worth adopting for the manifold generator inner loop.

### 2.8 Voigt Profile via Whiting 1968 + Numba JIT

RADIS uses the empirical Whiting (1968) approximation to Voigt profiles (accurate to ~0.02%), accelerated with `@numba.jit`. Performance improvement measured at 8.9 s → 5.1 s for 50k-line datasets.

**CF-LIBS relevance:** `cflibs/radiation/profiles.py` uses `scipy.signal.fftconvolve`. For non-JAX code paths (inversion pipeline), switching to a JIT-compiled Whiting approximation or the Thompson (1987) pseudo-Voigt would cut forward model time for real-time DAQ use. For JAX paths, a polynomial Voigt approximation is already differentiable.

### 2.9 `parsum_mode='tabulation'` — 500–4000× Speedup

RADIS's partition function tabulator pre-computes U(T) on a temperature grid just-in-time and interpolates. The `'full summation'` mode sums over all rovibrational levels every call; `'tabulation'` interpolates from a pre-built table with <0.1% error.

**CF-LIBS relevance:** `cflibs/plasma/` already uses polynomial partition functions (log U = Σ aₙ(log T)ⁿ), which is architecturally equivalent to RADIS's tabulation mode. The CF-LIBS approach is correct. The RADIS pattern confirms this is the right tradeoff.

### 2.10 Spectrum Algebra and Operator Overloading

RADIS exposes `MergeSlabs`, `SerialSlabs`, and arithmetic operators (`+`, `//`, `>`) for composing spectra. This pattern makes multi-zone or multi-element forward models composable without parameter coupling.

**CF-LIBS relevance:** The current `cflibs/radiation/spectrum_model.py` computes the full multi-element spectrum in one shot. Decomposing it into per-element Spectrum objects that are then `MergeSlabs`-ed would enable partial updates (re-computing only the element whose concentration changed in an iterative solver step) and cleaner uncertainty attribution.

---

## 3. Pitfalls Documented in RADIS

### 3.1 Circular Import Problem
RADIS explicitly documents circular imports requiring manual cleanup in their long-term TODO. Their factory-chain architecture pushes all imports into subclasses, which can create cycles when modules reach upward. **CF-LIBS risk:** the inversion pipeline imports from both `cflibs/radiation/` and `cflibs/plasma/`; keeping the dependency direction strict (plasma → radiation, not the reverse) prevents this.

### 3.2 `df0` Overwrite Bug
The documentation warns: "never overwrite the `df0` attribute; else some metadata may be lost." Overwriting it resets the HDF5 cache pointer, causing silent reloads. **CF-LIBS relevance:** if CF-LIBS ever adopts a staged-cache pattern, the same invariant applies to the raw database query cache.

### 3.3 Temperature Out-of-Range for Partition Functions
`PartFuncBarklem` covers up to 10,000 K; `PartFuncKurucz` requires explicit `potential_lowering`. Using NIST beyond its tabulated level range raises `OutOfBoundError`. LIBS plasmas run 6,000–20,000 K. **CF-LIBS action:** validate the polynomial partition function range against the LIBS operating envelope (6,000–20,000 K, ionization stages I–II). Barklem & Collet is insufficient above 10,000 K.

### 3.4 NIST Atomic Source Has No Broadening Parameters
NIST's level table (used for `pfsource='nist'`) provides no collisional or Stark broadening parameters, making `lbfunc` mandatory for atomic spectra. RADIS documents this explicitly. **CF-LIBS relevance:** This is already handled via `gamSta` per-line in the forward model, but CF-LIBS should verify that Stark widths are assigned before the broadening step and not silently zeroed for lines with missing database entries.

### 3.5 `wstep` Undersampling
Using too-coarse a wavenumber grid (large `wstep`) produces artifacts in lineshape wings — lines can be missed if the grid spacing exceeds the HWHM. RADIS's `wstep='auto'` is the fix. **CF-LIBS relevance:** The fixed resolving-power grid mode may under-resolve narrow lines at low temperature. A guard asserting `grid_step < 0.5 × min_HWHM` would catch this.

### 3.6 Pressure Unit Inconsistency
RADIS TODO: pressure_mbar vs. pressure naming inconsistency across modules. A known source of silent unit bugs. **CF-LIBS relevance:** `cflibs/core/constants.py` should have unit-safe wrappers or explicit `_Pa`, `_bar`, `_atm` suffixes on all pressure variables.

### 3.7 Pseudo-Continuum Deprecated
`pseudo_continuum_threshold` is flagged as "will be deprecated in future versions." RADIS recommends DIT/LDM as the replacement. **CF-LIBS:** if adopting pseudo-continuum, treat it as a transient optimization, not an architectural feature.

### 3.8 GPU Path Limited to Equilibrium
`eq_spectrum_gpu()` (Vulkan backend) is equilibrium-only. Non-equilibrium and non-LTE cases fall back to CPU. **CF-LIBS relevance:** The JAX-based manifold generator already handles this correctly (it is equilibrium LTE by design). Stark n_e paths are CPU-side by intent (ADR per reference notes).

### 3.9 Import Speed
RADIS documents "import speed improvements needed" in the long-term TODO. Heavy module-level imports cause ~0.3 s cold-start penalty. **CF-LIBS action:** audit `cflibs/__init__.py` for eager imports of heavy dependencies (SQLite, JAX, NumPyro) and move to lazy imports.

### 3.10 HDF5 Library Conflicts
RADIS offers three HDF5 backends (`pytables`, `vaex`, `feather`) because each has lock/compatibility issues in different environments. **CF-LIBS relevance:** `cflibs/manifold/` uses h5py; cluster runs using multiple workers need advisory file locking or Zarr (already supported) to avoid corruption.

---

## 4. Concrete "CF-LIBS Should Consider X" Recommendations

| # | Recommendation | RADIS pattern | Priority | Link |
|---|---------------|---------------|----------|------|
| R1 | Adopt the DIT/FFT broadening algorithm for the JAX manifold generator inner loop | `broadening_method='fft'` + LDM, van den Bekerom & Pannier 2021 | **High** | [JQSRT DIT paper](https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049) |
| R2 | Add a `lbfunc`-style injection point in `SpectrumModel` for per-line Stark widths driven by the on-device n_e estimate | `SpectrumFactory(lbfunc=...)` | **High** | [RADIS atomic broadening example](https://radis.readthedocs.io/en/develop/auto_examples/1_Spectra_handling/plot_2custom_Lorentzian_broadening.html) |
| R3 | Upgrade partition functions for ions (I–III) using Kurucz tables with `potential_lowering` proportional to Debye screening, replacing or validating current polynomial fits above 10 kK | `PartFuncKurucz.at(T, potential_lowering)` | **High** | [RADIS partfunc example](https://radis.readthedocs.io/en/latest/auto_examples/1_Spectra_handling/plot_1partitionFunction_potentialLowering.html) |
| R4 | Split the `cflibs/radiation/spectrum_model.py` forward model output into a `Spectrum`-like container (per-element spectra, conditions dict, unit-aware `get()`) to enable partial updates in iterative solvers | `Spectrum` class design ([docs](https://radis.readthedocs.io/en/latest/spectrum/spectrum.html)) | **Medium** | [Spectrum object docs](https://radis.readthedocs.io/en/latest/spectrum/spectrum.html) |
| R5 | Implement `MergeSlabs`-style element composition in the forward model so each element's contribution is independently computed and summed, enabling per-element sensitivity analysis | `MergeSlabs()` / `//` operator | **Medium** | [LOS module docs](https://radis.readthedocs.io/en/latest/los/los.html) |
| R6 | Add `wstep='auto'` equivalent: assert `grid_step < 0.5 × min_Doppler_HWHM(T_min, heaviest_element)` at forward model construction time | `wstep='auto'` in `SpectrumFactory` | **Medium** | [RADIS PR #271](https://github.com/radis/radis/pull/271) |
| R7 | Adopt RADIS's staged database caching (`df0` raw, `df1` with populations) for the SQLite atomic query cache in `cflibs/atomic/database.py`; avoids re-querying on every manifold grid point | `DatabankLoader.df0`/`df1` pattern | **Medium** | [loader docs](https://radis.readthedocs.io/en/latest/source/radis.lbl.loader.html) |
| R8 | Implement weak-line pseudo-continuum aggregation for lines below `I_line / I_peak < threshold` in the forward model; reduces profile computation overhead for LIBS where a few bright lines dominate | `pseudo_continuum_threshold` in `SpectrumFactory` | **Low** | [factory docs](https://radis.readthedocs.io/en/latest/source/radis.lbl.factory.html) |
| R9 | Add `predict_time()` / `print_perf_profile()` equivalents to the manifold generator; RADIS's built-in profiling instruments every major pipeline step and warns on slow configurations | `SpectrumFactory.predict_time()` | **Low** | [factory docs](https://radis.readthedocs.io/en/latest/source/radis.lbl.factory.html) |
| R10 | Guard all pressure variables with explicit unit suffixes or unit-typed wrappers (`pressure_Pa`, `pressure_bar`) — RADIS documents this as a known silent bug source | RADIS long-term TODO [#53](https://github.com/radis/radis/issues/53) | **Low** | [issues #53](https://github.com/radis/radis/issues/53) |

---

## Key References

- [RADIS-2018] Pannier & Laux, *RADIS: A Nonequilibrium Line-by-Line Radiative Code for CO2 and HITRAN-like database species*, JQSRT (2018). https://www.sciencedirect.com/science/article/abs/pii/S0022407318305867
- [DIT-2021] van den Bekerom & Pannier, *A Discrete Integral Transform for Rapid Spectral Synthesis*, JQSRT (2021). https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049
- [Barklem-2016] Barklem & Collet, *Partition functions and equilibrium constants for diatomic molecules and atoms of astrophysical interest* (2016). — basis for `PartFuncBarklem`.
- [Kurucz-2017] Kurucz, *Including all the lines*, Canadian Journal of Physics 95(9) (2017). — basis for `PartFuncKurucz`.
- [Whiting-1968] Whiting, *An empirical approximation to the Voigt profile*, JQSRT (1968). — basis for RADIS Voigt implementation.
- RADIS docs: https://radis.readthedocs.io/en/latest/
- RADIS GitHub: https://github.com/radis/radis
- RADIS developer guide: https://radis.readthedocs.io/en/latest/dev/developer.html

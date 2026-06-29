# ExoJAX Reference: Patterns for CF-LIBS JAX Forward Model Overhaul

Sources:
- Paper I: Kawahara et al. 2022, ApJS 258, 31 (arXiv:2105.14782) — "Auto-Differentiable Spectrum Model"
- Paper II: Kawahara et al. 2025, ApJ 985, 263 (arXiv:2410.06900) — "ExoJAX2"
- Docs: https://secondearths.sakura.ne.jp/exojax/
- GitHub: https://github.com/HajimeKawahara/exojax (MIT license)

---

## 1. How ExoJAX Structures the Computation

### 1.1 Module Hierarchy (v2.x)

```
exojax/
  database/          # line-list ingest and atomic data
    core_atom/       # shared atom primitives
      broadening.py  # gamma_vald3, gamma_uns, gamma_KA3/4
      line_strength.py   # S0 at Tref (Einstein A formulation)
      pf.py          # partition function interpolation
    kurucz/api.py    # AdbKurucz class
    vald/api.py      # AdbVald / AdbSepVald classes
  opacity/           # cross-section computation
    lpf/lpf.py       # direct Voigt line-profile (LPF)
    modit/           # Modified Discrete Integral Transform
    premodit/        # pre-computed LBD variant (memory-efficient)
    initspec.py      # init_lpf / init_modit / init_premodit
    policies.py      # MemoryPolicy frozen dataclass
  rt/                # radiative transfer (ArtEmisPure, etc.)
  postproc/          # instrumental broadening, rotation kernels
```

**Design principle:** loose coupling between layers. The database (Adb*) objects are separate from opacity calculators (Opa*). In v2.2, `OpaPremodit` gained `from_mdb()` and `from_snapshot()` constructors so opacity can be computed without holding the full line-list in GPU memory simultaneously.

### 1.2 Forward Model Data Flow

```
AdbKurucz/AdbVald  (load line data to CPU numpy)
   |
   | generate_jnp_arrays()   -- transfers selected lines to JAX arrays
   v
line_strength_atom()          -- compute S0 at Tref (once, on CPU/JAX)
   |
   v  [per-T evaluation, vmapped over atmosphere layers]
line_strength(Tarr, logsij0, nu_lines, elower, qr, Tref)
sigmaDM  = doppler_sigma(nu_lines, T, atomicmass)          -- Gaussian width
gammaLM  = gamma_vald3(T, PH, PHH, PHe, ielem, iion, ...) -- Lorentz width
   |
   v
numatrix = init_lpf(nu_lines, nu_grid)  -- (Nline x Nwav) offset matrix
xsm      = xsmatrix(numatrix, sigmaDM, gammaLM, SijM)     -- (Nlayer x Nwav)
   |
   v
dtau     = layer_optical_depth(dParr, xsm, VMR, mmw, g)
F0       = art.run(dtau, Tarr)          -- radiative transfer (linear algebra)
```

### 1.3 Three Opacity Strategies (LPF / MODIT / PreMODIT)

| Method | When to use | Memory scaling | Accuracy |
|--------|------------|----------------|---------|
| LPF (direct) | N_line < 10^3 | O(N_line * N_wav) per layer — impractical for large N | ~1% vs scipy.wofz |
| MODIT | 10^3 < N_line < 10^5 | O(N_wav * N_par * N_line) grid-based, nearly independent of N_line until ~10^5 | F64: <1%; F32: breaks below 10^-25 cm^2 |
| PreMODIT | N_line > 10^5, or GPU-memory-limited | Pre-computes LBD (Line Basis Density) at Tref; decouples DB from GPU | Best; enables native-resolution JWST |

**The LPF cross-section formula** (Paper I, Eq. 1–2):

    sigma(nu) = S * V(nu - nu_hat, beta, gamma_L)
              = S / (sqrt(2pi) * beta) * H((nu - nu_hat)/(sqrt(2)*beta), gamma_L/(sqrt(2)*beta))

where H(x,a) is the Voigt-Hjerting function.

**MODIT** converts the sum over lines into a 2D lineshape distribution matrix S_jk in (q, gamma_L) space on a log-wavenumber grid, then uses FFT convolution (Paper I, Eq. 18–19):

    sigma(q_i) = sum_{jk} S_jk * V(q_i - q_j; beta_k, gamma_L,k)
               = sum_k  FFT^{-1}[ FFT(S_jk) * FFT(V(q_j; beta_k, gamma_L,k)) ]

This reduces N_convolutions from one-per-line to one-per-point on the lineshape distribution grid.

**PreMODIT** pre-computes a Line Basis Density L(nu, E, b) at a reference T_wp, then recovers S at any T analytically via Boltzmann weighting — never materializing per-line arrays on device.

**Opart (ExoJAX2):** layer-by-layer opacity (one layer at a time) cuts GPU memory from O(N_layer * N_wav * N_brd) to O(N_wav * N_brd), at the cost of losing reverse-mode autodiff (forward-mode or finite diff must be used instead).

**nu-stitching (ExoJAX2):** splits the wavenumber grid into segments, computes opacity per segment, then recombines via Overlap-and-Add (OLA) in the Fourier domain — differentiably. Applicable to transmission (where Opart cannot be used).

---

## 2. Specific APIs / Patterns / Numerical Techniques Worth Adopting

### 2.1 Voigt-Hjerting via Faddeeva with Custom JVP

The core Voigt function `hjert(x, a)` = Re[w(x + ia)] where w is the Faddeeva function.

**Two-region implementation** (Paper I, Eq. 6, 10–12):

- For |z|^2 = x^2 + a^2 < 111: use "Algorithm 916" (Zaghloul & Ali 2011) with fixed M=27 terms.
  - Algorithm 916 expansion: H_M(x,a) = e^{-x^2} erfcx(a) cos(2xa) + (2*eta*x*sin(ax)/pi)*e^{-x^2}*sinc(ax/pi) + (2*eta/pi)*{...Sigma_1, Sigma_2, Sigma_3 terms}
  - erfcx(a) = e^{a^2} erfc(a) is the *scaled* complementary error function — computed via Chebyshev approximation (Shepherd & Laframboise 1981) to avoid overflow at large a.
- For |z|^2 >= 111: asymptotic w^{asy}(z) = i/(z*sqrt(pi)) * (1 + tilde_alpha*(s_0 + tilde_alpha*(s_1 + ...))) where s_k = (2k+1)!!

**Why not scipy.special.wofz:** scipy's wofz is not JAX-compatible (not traceable, no JVP).

**Custom JVP** defined analytically (Paper I, Eq. 13–14):

    d/dx H(x,a) = 2a * L(x,a) - 2x * H(x,a)
    d/da H(x,a) = 2x * L(x,a) + 2a * H(x,a) - 2/sqrt(pi)

where L(x,a) = Im[w(x+ia)]. This avoids AD through the conditionals and sums, giving clean gradients for HMC.

**Accuracy:** absolute difference from scipy.special.wofz < 10^-6 for 10^-3 < a < 10^5 and 10^-3 < x < 10^5.

**cflibs adoption:** Replace any current Voigt/pseudo-Voigt with hjert-style implementation. Define `@jax.custom_jvp` with the analytic d/dx and d/da rules. Use Chebyshev erfcx to avoid overflow at large a (relevant for pressure-broadened LIBS lines).

Reference: `src/exojax/opacity/lpf/lpf.py` — functions `hjert`, `hjert_jvp`, `ljert`, `rewofz`, `asymptotic_wofz`.

### 2.2 Line Strength Temperature Scaling

**Two-step approach** separates a one-time CPU computation (S0 at Tref) from a per-T JAX computation:

**Step 1 — Reference line strength S0** (once, `line_strength_atom`, Paper I equivalent):

    S0 = (-A * g_upper * exp(-hc/kB * E_lower / Tref) * expm1(-hc/kB * nu / Tref))
         / (8*pi*c_cgs * nu^2 * Q(Tref))

Note `expm1` instead of `exp(...) - 1` for numerical stability at small arguments. The negative sign arises from the convention; the result is a positive absorption cross-section per molecule.

**Step 2 — Temperature-dependent scaling** (per layer, vmapped):

    S(T) = S0 * (Q(Tref)/Q(T)) * exp[-hc/kB * E_lower * (1/T - 1/Tref)]
               * (1 - exp(-hc/kB * nu / T)) / (1 - exp(-hc/kB * nu / Tref))

Equivalently (Paper I, Eq. 27–28), using log-domain to avoid float32 overflow:

    log S(T) = s0 - c2 * E_lower * (T^{-1} - Tref^{-1}) + log(1 - e^{-c2*nu/T}) - log(1 - e^{-c2*nu/Tref}) - log(q_r(T))

where s0 = log_e S0, q_r(T) = Q(T)/Q(Tref), c2 = hc/kB = 1.4387773 cm*K.

**ExoMol variant** (Paper I, Eq. 31) uses Einstein A + state energies directly:

    S(T) = (g_up/Q(T)) * (A / (8*pi*c*nu^2)) * exp(-c2*E_low/T) * (1 - e^{-c2*nu/T})

**cflibs adoption:** The current `calculate_line_emissivity` in cflibs uses emissivity (intensity units). For a JAX forward model that needs autodiff: implement S(T) in log-domain with `jnp.expm1` and `jnp.log1p`, vmap over temperature, keep logsij0 (log reference S) as the stored per-line quantity. This is numerically stable even at float32.

### 2.3 Partition Function Interpolation

ExoJAX uses a pre-tabulated 284-species grid from Barklem & Collet (2016) stored at fixed temperature nodes. At runtime:

    QT_interp_284(T, T_gQT, gQT_284species)  -- jnp.interp across all 284 species

Per-line mapping uses `QTmask[i]` = index of species i in the 284-species table. This mapping is computed once from (ielem, iion) pairs.

For neutral iron specifically, Irwin (1981) polynomial is available as an alternative:

    ln Q_Fe(T) = a0 + a1*ln(T) + a2*(ln T)^2 + ... + a5*(ln T)^5   (6 coefficients)

**cflibs adoption:** The existing polynomial partition function `log U = sum_n a_n (log T)^n` in `cflibs/plasma/` is already the right form. For multi-species JAX computation, build a per-element array of polynomial coefficients and use `jnp.polyval` with vmap — avoids the 284-species table overhead if you only have a handful of LIBS elements.

### 2.4 Pressure Broadening (gamma_vald3)

Five methods available (broadening.py), each JAX-vmappable:

1. **gamma_vald3** (recommended): uses tabulated vdW damping constant when vdWdamp < 0 (Barklem et al. 1998/2000 ABO theory); falls back to Unsöld (1955) C6 estimate when vdWdamp >= 0.
   - C6 = 0.3e-30 * [1/(ionE - chi - chi_lambda)^2 - 1/(ionE - chi)^2]
   - Stark contribution: gamStark = 10^gamSta * N_e * (T/10000)^(1/6)
   - Van der Waals: uses P_H, P_HH (H2), P_He partial pressures separately

2. **gamma_KA3**: Kurucz & Avrett (1981) quantum-defect method, T^0.3 dependence
3. **gamma_uns**: Classical Unsöld, T^{-0.7}
4. **gamma_KA4**: Simplified KA, large-mass limit

**cflibs LIBS context:** In LIBS plasmas, pressure broadening is from electron collisions (Stark), not van der Waals. The gamSta parameter maps directly. ExoJAX's gamma_vald3 Stark term is directly adoptable; the vdW terms are irrelevant for LIBS.

### 2.5 Doppler Broadening

    sigma_D(nu, T, M) = (nu/c) * sqrt(k_B * T / (M * m_u))     [cm^{-1}]

where M is atomic mass in amu, m_u is the atomic mass unit. This is the 1-sigma Gaussian width. ExoJAX uses this exact formula via `doppler_sigma` in `exojax.database.hitran`.

On a log-wavenumber (q = R0 * log nu) grid, the Doppler width in q-space is **independent of nu** for a given (T, M):

    beta_hat = R0 * beta_T / nu_hat = sqrt(k_B*T/(M*m_u)) * R0/c     (Paper I, Eq. 22)

This is the key reason MODIT is efficient: all lines of a given isotopologue share one Doppler width, collapsing a dimension.

### 2.6 Radiative Transfer as Linear Algebra

ExoJAX avoids iterative RT (which has high autodiff cost via `jax.lax.scan`) by expressing flux as a linear-algebraic dot product (Paper I, Eq. 48):

    F0 = (Phi * Q)^T u

where Phi = cumulative-product of transmission T_n(nu_j), Q = transmitted source terms, u = ones vector. This is implemented as `jax.numpy.sum(Q * jax.numpy.cumprod(T, axis=0), axis=0)`.

**cflibs relevance:** CF-LIBS uses optically thin LTE (no atmospheric RT layers). But the pattern of expressing multi-step computations as JAX-native reductions (cumprod, cumsum) rather than Python loops is essential for JIT compilation.

### 2.7 MemoryPolicy and Decoupled Opacity

`MemoryPolicy(allow_32bit=True, nstitch=4, cutwing=20.0)` is a frozen dataclass passed to `OpaPremodit` at construction time, centralizing precision/memory knobs. This prevents parameter proliferation across function signatures.

The `from_snapshot()` constructor loads a pre-serialized LBD coefficient file, decoupling the database lookup from the inference step. This is crucial for HMC: you serialize once, then load the compact representation for each posterior evaluation.

**cflibs adoption:** A `ForwardModelConfig(float32=True, wing_cutoff=100.0, ...)` dataclass passed to the forward model factory would provide the same benefit.

### 2.8 Wavenumber / Wavelength Grid Design

ExoJAX uses a log-wavenumber grid (q = R0 * log nu) as the native coordinate for two reasons:
1. Doppler widths are constant in q-space (a single beta for all lines of a species at given T).
2. Efficient FFT convolution in MODIT (uniform spacing required).

Grid utility: `wavenumber_grid(wl_min, wl_max, N, xsmode, unit, wavelength_order)` returns (nu_grid, wav, resolution).

For LIBS (small wavelength range, fixed resolution): a linear wavelength grid may be simpler. But if adopting MODIT for CF-LIBS, log-nu is required.

### 2.9 nu-Matrix (numatrix) for LPF

For the direct LPF, ExoJAX precomputes a (N_line x N_wav) matrix of offsets:

    numatrix[l, j] = nu_j - nu_lines[l]

This is computed once (`init_lpf`) before the JIT loop, so the inner `xsmatrix` call is a pure function of (numatrix, sigmaD, gammaL, Sij) — all arrays. This structure is JIT-friendly: no Python-side loops inside the compiled function.

**cflibs adoption:** Precompute the offset matrix for the set of identified lines once before the iterative solver runs. Inside the jitted forward model, only array operations occur.

---

## 3. Pitfalls Documented by ExoJAX

### 3.1 Float32 Precision in MODIT

MODIT with F32 produces errors larger than 10^{-25} cm^2 in the cross-section (Fig. 3, Paper I). This is a known limitation of single-precision FFT accumulation. The paper recommends F64 for MODIT when absolute cross-section accuracy matters. For HMC (relative comparison), F32 may be acceptable.

**For CF-LIBS:** line strengths span many orders of magnitude; weak-line errors at 10^{-25} cm^2 are innocuous since they fall far below the photosphere. But if Boltzmann plot residuals are sensitive to weak-line tails, use F64 or apply a line strength threshold (`crit` parameter in AdbKurucz).

### 3.2 scipy.special.wofz is Not JAX-Compatible

Do not use `scipy.special.wofz` inside any JAX-traced function. It is not traceable and has no gradient. ExoJAX implements `rewofz` (real part) and `imwofz` (imaginary part) as pure JAX functions.

### 3.3 Iterative RT is Expensive for Autodiff

`jax.lax.scan` through atmospheric layers accumulates a deep computation graph. ExoJAX replaced iterative RT with a closed-form matrix expression. For CF-LIBS (no RT), this is not directly relevant, but the same principle applies to iterative solvers: if you want HMC gradients through the CF-LIBS iterative loop, reformulate as a differentiable fixed-point or use the implicit function theorem.

### 3.4 HITRAN "Air" Broadening vs. Target Atmosphere

Paper I explicitly warns: HITRAN pressure-broadening coefficients assume N2 ("air"). For H2 atmospheres (brown dwarfs), the actual gammaL is 1–1.5x different. Always match the broadening model to the actual collision partner.

**For CF-LIBS:** plasma broadening is electron-impact (Stark). Never use air-broadening HITRAN parameters for LIBS Lorentz widths. Use the gamSta column from VALD/Kurucz plus n_e scaling.

### 3.5 petRADTRANS Interpolation Error (~10–30%)

When opacity is computed on a pre-gridded (T, P) table and interpolated, errors up to 30% can occur for non-linear T profiles. The paper measured ~10% deviation for general T-P profiles when the grid spacing is coarse (666 K, 900 K, 1215 K grid points).

**cflibs relevance:** If a manifold-based pre-computation is used for fast inference, the interpolation error in (T, n_e, composition) space must be characterized. Consider finer grids near physically important regimes (e.g., 5000–15000 K for LIBS).

### 3.6 Weak Lines Below Photosphere

For direct LPF with many lines, ExoJAX excludes weak lines that do not contribute to the emission by finding the layer where each line contributes most and checking if it is below the CIA photosphere. This prevents N_line from growing with the database size.

**cflibs adoption:** Apply a line strength threshold (`crit`) filtered to the expected T, n_e range when loading lines. Do not include every line in the database — only those contributing >= some fraction of the peak emissivity.

### 3.7 1000-Line VALD Extraction Limit

VALD3 limits a single extraction request to 1000 lines. For broad spectral ranges, multiple requests are needed and results must be concatenated.

### 3.8 Atomic Lines: "opa framework" Cannot Be Used Directly

As of ExoJAX 2.2, the standard `OpaPremodit` / `OpaModit` opacity calculators do not yet fully support atomic lines. The tutorials for Kurucz/VALD lines use manual computation (explicit vmaps over gamma_vald3, line_strength, xsmatrix). This is an acknowledged limitation.

**cflibs implication:** CF-LIBS is entirely atomic, so it must follow the "manual" path (explicit vmap over broadening + xsmatrix). This is fine and well-tested in ExoJAX tutorials for Fe I.

### 3.9 numatrix Memory

The (N_line x N_wav) numatrix for direct LPF requires N_line * N_wav floats. For N_line=10^4 and N_wav=10^4, this is 400 MB at float32 — impractical. ExoJAX's solution is to switch to MODIT above N_line ~10^3 or use wavenumber masking to limit N_wav per spectral segment.

**cflibs:** For a typical LIBS spectrum (1000–10000 lines, ~10^4 wavelength bins), LPF may be feasible if masking is applied. Partition into spectral windows if needed.

### 3.10 VJP Not Supported (Only JVP)

ExoJAX's custom Voigt derivative is implemented as a **JVP (Jacobian-vector product)**, not a VJP (vector-Jacobian product). NumPy >=0.6.0 supports JVP via `jax.custom_jvp`. NumPyro and HMC use forward-mode AD or rely on JVP composition — this works. But if reverse-mode AD (`jax.grad`) is needed through the Voigt function in a complex compute graph, the JVP-only definition may require additional care.

---

## 4. Concrete "cflibs Should Consider X" Recommendations

### 4.1 Adopt hjert for All Voigt Computations

Replace cflibs's current `voigt_profile` with an `hjert`-style implementation:
- Algorithm 916 (M=27) for |z|^2 < 111
- Asymptotic expansion for |z|^2 >= 111
- Chebyshev erfcx to avoid overflow
- `@jax.custom_jvp` with analytic dH/dx and dH/da rules

This gives <10^-6 error vs. scipy and full autodiff compatibility.

Reference implementation: https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/opacity/lpf/lpf.py

### 4.2 Log-Domain Line Strength Scaling

Store `logsij0 = log(S0)` per line (computed once). At runtime, scale to temperature T using the Boltzmann formula in log-domain with `jnp.expm1` for the stimulated-emission correction. This avoids float32 overflow for weak lines at low T.

The ExoJAX formula (Paper I, Eq. 27–28) should be translated directly into `cflibs/radiation/` or a new `cflibs/jitpipe/line_strength.py`.

### 4.3 QTmask Pattern for Multi-Species Partition Functions

Build a `QTmask` array once: for each line l, `QTmask[l]` indexes into a (N_species x N_Tgrid) partition function table. Then `jnp.interp` the whole table at temperature T in one call, index with `QTmask`. This is O(1) JAX operations regardless of number of species.

For CF-LIBS with polynomial PF: precompute a (N_species x N_coeffs) coefficient array, use `jax.vmap` over species with `jnp.polyval` to get Q(T) for all species at once.

### 4.4 MemoryPolicy / ForwardConfig Dataclass

Introduce a frozen `ForwardModelConfig` (or add to existing config system):

```python
@dataclasses.dataclass(frozen=True)
class ForwardModelConfig:
    float32: bool = False
    wing_cutoff_hwhm: float = 100.0  # cut Voigt wings beyond N half-widths
    line_strength_threshold: float = 1e-30  # discard lines weaker than this at Tref
    nu_stitch_segments: int = 1
```

Pass to `build_forward_model()`. This mirrors ExoJAX's `MemoryPolicy`.

### 4.5 Precompute numatrix Outside JIT Scope

`init_lpf(nu_lines, nu_grid)` — the offset matrix — should be computed once and stored as a static array. The jitted forward function signature should be `forward(numatrix, logsij0, T, n_e, composition)` with `numatrix` as a static (non-traced) input via `jax.pure_callback` or simply recomputed only when the line selection changes.

Reference: https://secondearths.sakura.ne.jp/exojax/tutorials/Forward_modeling_for_Fe_I_lines_of_Kurucz.html

### 4.6 Adopt Opart Pattern for Memory-Constrained Inference

For the manifold generator (many (T, n_e, composition) points), the Opart pattern (process one parameter point at a time, not one layer) is directly analogous: `jax.vmap` over the parameter grid with a function that computes a single spectrum. ExoJAX demonstrated this reduces GPU memory from O(N_batch * N_wav * N_brd) to O(N_wav * N_brd).

Reference: ExoJAX2, Section 5.3 (arXiv:2410.06900).

### 4.7 Linear-Algebraic RT / Spectrum Accumulation

For optically-thick LIBS spectra (self-absorption), if cflibs ever implements layer integration, use ExoJAX's linear-algebra formulation:

    F = (Phi * Q)^T @ u  =  jnp.sum(Q * jnp.cumprod(T, axis=0), axis=0)

rather than a Python loop or `lax.scan`. This is fully differentiable and XLA-efficient.

### 4.8 Instrumental Broadening as Post-Process Convolution

ExoJAX applies all instrument effects (Gaussian IP, rotation kernel) as post-processing on the high-resolution spectrum via kernel convolution (Eq. 53–54). The key: compute at native high resolution (R >= 500,000 for line profiles), then convolve to observed resolution. Do not truncate the internal grid to observed resolution before convolution.

For CF-LIBS: keep the forward model on a fine wavelength grid (e.g., 0.01 Å/pixel), apply Gaussian instrument broadening as a final `jnp.convolve`, then sample at observed pixels.

### 4.9 Use jnp.interp for Differentiable Partition Functions

1D interpolation via `jnp.interp` is JAX-native and differentiable. ExoJAX uses it for all partition function temperature lookups. For cflibs polynomial PF, `jnp.polyval` is preferable (exact, no grid needed), but for empirically-tabulated PF (e.g., from NIST or Barklem & Collet 2016), `jnp.interp` is the right tool.

### 4.10 Benchmark MODIT for LIBS Line Counts

At typical LIBS line counts (100–5000 lines per element), Figure 4 of Paper I shows:
- LPF: ~0.1 ns/line/wav_bin, scales linearly — feasible for N_line < 10^3
- MODIT: nearly flat until N_line ~10^5, then linear — far superior for large databases

For multi-element LIBS (e.g., Fe + Ni + Cu + Cr simultaneously = potentially 10^4 lines), MODIT is the right choice. For single-element identification with a handful of lines, LPF is simpler and lower overhead.

---

## 5. Key Citations

- Kawahara et al. 2022 (ExoJAX1): https://doi.org/10.3847/1538-4365/ac3b4d
- Kawahara et al. 2025 (ExoJAX2): https://doi.org/10.3847/1538-4357/adcba2
- van den Bekerom & Pannier 2021 (DIT): the foundation of MODIT
- Zaghloul & Ali 2011: Algorithm 916 for Faddeeva (the Voigt inner loop)
- Barklem & Collet 2016: 284-species partition function table used in AdbKurucz/AdbVald
- Barklem et al. 1998/2000: ABO theory for van der Waals damping (gamma_vald3)
- Shepherd & Laframboise 1981: Chebyshev erfcx implementation

---

## 6. Summary Table: ExoJAX → cflibs Mapping

| ExoJAX component | cflibs equivalent / target |
|---|---|
| `hjert` (Faddeeva, custom JVP) | `voigt_profile` in `cflibs/radiation/profiles.py` — replace |
| `line_strength_atom` + `line_strength` | `calculate_line_emissivity` — split into Tref step + T-scaling step |
| `QTmask` + `QT_interp_284` | Polynomial PF in `cflibs/plasma/` — add vmap, optionally tabulate |
| `gamma_vald3` (Stark term) | Stark broadening in `cflibs/inversion/physics/` |
| `doppler_sigma` | Thermal (Doppler) width — already in cflibs |
| `init_lpf` + `xsmatrix` | New `cflibs/jitpipe/` forward model core |
| `MemoryPolicy` | New `ForwardModelConfig` dataclass |
| `OpaPremodit.from_snapshot()` | Manifold snapshot loading in `cflibs/manifold/` |
| Log-wavenumber grid | `cflibs/core/` grid utilities — add log-nu option for MODIT |
| Opart (layer-by-layer) | Manifold batch generator in `cflibs/manifold/` |

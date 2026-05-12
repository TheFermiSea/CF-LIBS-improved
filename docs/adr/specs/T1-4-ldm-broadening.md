# T1-4 Implementation Spec — LDM Broadening Grid (1-D log-σ for CF-LIBS)

**Bead:** `CF-LIBS-improved-e5o8` · **ADR:** [ADR-0001](../ADR-0001-radis-jaxrts-pattern-survey.md) §4.2 B-P3, §8.1 row T1-4 · **Wave:** 2 (parallel with T1-2, T1-3) · **Hard dep:** T1-1 (`5oar`, host/kernel split + `AtomicSnapshot` + `JaxMemoryPolicy`) · **Estimated effort:** medium · **Reference:** van den Bekerom & Pannier 2021, JQSRT 261 107476

## 1. Goals

Replace the current per-line broadcasting Voigt/Gaussian assembly (`voigt_spectrum_jax`, `apply_gaussian_broadening_jax` in `cflibs/radiation/profiles.py` and the inline Humlicek W4 block in `cflibs/manifold/generator.py:567-632`) with a Radis-style line-distribution-method (LDM/DIT) kernel specialized to the Gaussian-dominant CF-LIBS regime.

LDM projects N lines onto a small log-σ grid (~16-32 layers) using bilinear weights, then performs one FFT convolution per σ layer. This collapses the dominant `O(N_lines · N_λ)` outer-product (the `diff = wl_grid[:, None] - l_wl[None, :]` matrix at `generator.py:572` and the `(N_wl, N_lines)` profile matrix in `voigt_spectrum_jax`) to `O(N_σ · N_λ · log N_λ)`. The manifold sweep over (T, n_e, composition) reuses the same σ-grid layout across every grid point.

## 2. Algorithm (LDM specialized to Gaussian)

For each line `i` with `(λ_i, I_i, σ_i = sqrt(σ_doppler_i² + σ_inst²))`:

1. Locate log-σ grid index `k`: `σ_grid[k] ≤ σ_i < σ_grid[k+1]` (clip to `[0, N_σ-2]`).
2. Compute interpolation weight `α_i = (log σ_i − log σ_grid[k]) / (log σ_grid[k+1] − log σ_grid[k])` (matches Radis `BroadenFactory._calc_lineshape_LDM` step weighting).
3. Distribute into 2-D `(N_σ, N_λ)` ledger via `jax.ops.segment_sum` over flattened `(k, j_i)` pairs (prefer over `at[].add()` for jax-metal determinism):
   - `LDM[k,   j_i] += (1 − α_i) · I_i`
   - `LDM[k+1, j_i] += α_i · I_i`
   - `j_i = round((λ_i − λ_min) / Δλ)` (uniform grid)
4. Sum convolutions: `spectrum = Σ_k irfft(rfft(LDM[k, :]) · rfft(g(σ_grid[k])))` where `g(σ_k)` is precomputed normalized Gaussian of length `next_pow2(N_λ + 6·max(σ_grid)/Δλ)`.

Voigt extension (rare in LIBS) → 2-D `(log σ_G, log γ_L)` grid with bilinear distribution. **Scope: 1-D Gaussian only.** API shaped to accept `gamma_grid` for the future Voigt extension.

## 3. Grid construction

```python
sigma_grid = jnp.exp(jnp.linspace(jnp.log(sigma_min), jnp.log(sigma_max), N_sigma))
```

`N_sigma ∈ [16, 32]` (default 24), `dx_σ ≈ 0.15` (Radis default 0.2).

- `sigma_min = 0.5 · min_i σ_i`
- `sigma_max = 2.0 · max_i σ_i`

CF-LIBS bounds analytically: at `[λ_min, λ_max]` and `[T_min, T_max]`, Doppler `σ ∈ [λ_min/c · sqrt(kT_min/m_max), λ_max/c · sqrt(kT_max/m_min)]`, plus instrument floor `σ_inst`. Host driver computes once from `AtomicSnapshot` + manifold config → grid layout is JIT-static.

Edge cases: lines with `σ_i ≤ σ_grid[0]` snap to layer 0 weight 1; lines with `σ_i ≥ σ_grid[-1]` snap to layer `N_σ-1`. NaN guards on bounds. Assert `N_σ ≥ 4`.

## 4. Kernel API

```python
# cflibs/radiation/kernels.py
@jit_if_available
def ldm_broaden(
    line_wavelengths: jax.Array,   # (N_lines,) — nm
    line_intensities: jax.Array,   # (N_lines,) — T·Saha·Boltzmann·A_ki-scaled emissivity
    line_sigmas: jax.Array,        # (N_lines,) — Gaussian σ (Doppler + instrument), nm
    wavelength_grid: jax.Array,    # (N_lambda,) — uniform spacing required
    sigma_grid: jax.Array,         # (N_sigma,) — log-spaced (static layout)
    *,
    gamma_grid: jax.Array | None = None,   # reserved for 2-D Voigt extension
    line_gammas: jax.Array | None = None,
) -> jax.Array:                    # (N_lambda,)
```

Internals:
- `log_sg = jnp.log(sigma_grid)`
- `k = jnp.clip(jnp.searchsorted(sigma_grid, line_sigmas) − 1, 0, N_sigma − 2)`
- `alpha = jnp.clip((jnp.log(line_sigmas) − log_sg[k]) / (log_sg[k+1] − log_sg[k]), 0, 1)`
- `j = jnp.clip(jnp.round((line_wavelengths − wl_grid[0]) / dlam).astype(jnp.int32), 0, N_lambda − 1)`
- Two `segment_sum` accumulations → reshape to `(N_σ, N_λ)`
- Precompute Gaussian kernels `G_k[n] ∝ exp(-0.5·(n·dlam/σ_grid[k])²)` of length `N_fft`, stack to `(N_σ, N_fft)`, normalize.
- FFT convolve per layer, sum across layers.
- Dtype: respect `JaxMemoryPolicy.allow_32bit`.

`vmap` axis spec (batch over T-dependent intensities, the manifold sweep variable):

```python
ldm_broaden_batch = jax.vmap(ldm_broaden, in_axes=(None, 0, None, None, None))
```

## 5. Where it lives

- **Kernel:** `cflibs/radiation/kernels.py::ldm_broaden` (new). Pure-JAX, single `@jit_if_available` function. Side-effect-free.
- **Host driver:** `cflibs/radiation/host.py::broaden_lines_ldm` — accepts `AtomicSnapshot`, plasma state, instrument, optional pre-built `sigma_grid`. If `sigma_grid is None`, constructs from min/max σ observed across the line set (numpy-side). Returns spectrum.
- **Profile shim:** keep `apply_gaussian_broadening_jax`/`voigt_spectrum_jax` in `cflibs/radiation/profiles.py` as per-line fallback when `BroadeningMode.LDM_GAUSSIAN` is NOT selected (legacy + per-line correctness reference for tests). Docstring marks "non-LDM fallback".

## 6. Call-site integration

`cflibs/radiation/spectrum_model.py`: add `BroadeningMode.LDM_GAUSSIAN` to enum (alongside `NIST_PARITY`, `PHYSICAL_DOPPLER`, `LEGACY`). Selection in dispatch block L243-253:

```python
elif self.broadening_mode is BroadeningMode.LDM_GAUSSIAN:
    sigma_per_line = self._compute_sigma_per_line(all_transitions, populations)
    emissivity = broaden_lines_ldm(
        line_wavelengths=...,
        line_intensities=line_emissivities,
        line_sigmas=sigma_per_line,
        wavelength_grid=self.wavelength,
        memory_policy=self.memory_policy,
    )
```

Downstream RT + instrument convolution at L259-305 unchanged. LDM_GAUSSIAN behaves like PHYSICAL_DOPPLER for L283's "skip NIST_PARITY convolution" check (does NOT skip).

## 7. Manifold integration (the real speedup)

`cflibs/manifold/generator.py::_compute_spectrum_snapshot` (L455-634) builds `(N_λ, N_lines) diff` at L572, runs Humlicek W4 at L584-626. For typical manifold sweep (32 × 32 × 8 grid points × 5000 lines × 30000 wavelengths) this dominates.

σ-grid layout depends only on manifold bounds (static), so lift grid construction out of jit:

1. **Once at manifold init:** build `sigma_grid` and FFT-domain Gaussian kernels `G_hat[N_σ, N_fft]` numpy-side; `device_put` once.
2. **Per grid point (inside jit/vmap):** Saha+Boltzmann produce `I_i(T, n_e, conc)` as before (L519-530); pass to `ldm_broaden(line_wavelengths, I_i, sigma_total_i, wl_grid, sigma_grid, G_hat)`. Wavelength scatter `j_i`, σ-layer `k_i`, weight `α_i` also static (depend only on line catalog + grid layout) — precompute, pass as static side-inputs.

Same swap in `cflibs/manifold/batch_forward.py::single_spectrum_forward` (L292-397): replace `voigt_spectrum_jax` at L395 with `ldm_broaden`. `vmap` wrapper at L400 unchanged.

**Expected speedup:** 10-30× on 8192-point manifold (Radis published >100× for Voigt; CF-LIBS Gaussian smaller scale, lower constant). Memory: `(N_σ, N_λ) = (24, 30000) × 4 B = 2.9 MB` per grid point vs current `(5000, 30000) × 4 B = 600 MB` — ~200× reduction in broadening working set.

## 8. Acceptance criteria

1. **Parity:** rtol=1e-4 vs `BroadeningMode.PHYSICAL_DOPPLER` on 200-line synthetic (Ti+Al+V, T=1.0 eV, n_e=1e17, 220-500 nm @ R=5000). Looser than T1-3's 1e-5 because LDM is approximate by construction; bound governed by `dx_σ` (Radis: error ≲ (dx_σ/2)² worst-case integrated intensity).
2. **Speed:** LDM ≥5× faster than per-line `voigt_spectrum_jax` for N_lines≥1000 (CPU, fp32 microbench).
3. **Manifold wall-time:** 32×32×8 grid generation ≥5× faster on CPU; ≥10× faster on V100S.
4. **No NaN/Inf** for any line set from `ASD_da/libs_production.db` corpus.
5. **vmap-clean:** `vmap(ldm_broaden, in_axes=(None, 0, None, None, None))` traces and runs without `ConcretizationTypeError`.
6. **jax-metal fp32 path:** identical to CPU fp32 within rtol=5e-4 (Metal-vs-CPU float32 drift floor).

## 9. Test plan

**New** `tests/test_ldm_broaden.py`:
- `test_parity_vs_physical_doppler` — 200-line synthetic, rtol=1e-4.
- `test_grid_step_convergence` — sweep `N_σ ∈ {8, 16, 24, 32}`; assert max relative error decreases as `O(dx_σ²)`.
- `test_vmap_over_intensities` — vmap over 64 intensity vectors, parity with map+stack.
- `test_edge_clipping` — `σ_i < σ_grid[0]` and `σ_i > σ_grid[-1]` produce finite output.
- `test_zero_intensity_lines_inert` — zero-intensity lines don't change spectrum.
- `@pytest.mark.benchmark(group="broaden")` microbench vs `voigt_spectrum_jax`.

**Existing**: `tests/test_radiation.py` (parametrize LDM mode), `tests/test_manifold.py` (LDM-path parity rtol=1e-4), `cflibs/benchmark/unified.py` synthetic corpus (composition recovery RMSE doesn't degrade).

## 10. Risks & rollback

- **Approximation tolerance on real data.** LDM error scales with `dx_σ`. Mitigation: `BroadeningMode.LDM_GAUSSIAN` defaults OFF until validated on `data/` real spectra via `scripts/validate_real_data.py`. Production CLI flows continue using PHYSICAL_DOPPLER.
- **FFT ringing near sharp Stark-broadened features.** Mitigation: when any `γ_stark / σ_doppler > 0.3`, LDM path falls back to per-line Voigt for that subset (hybrid). Detect at host driver level (numpy-side γ_L scan over `AtomicSnapshot.line_stark_w`).
- **jax-metal non-determinism with `at[].add()`.** Mitigation: use `segment_sum` exclusively; pin unit test asserting run-to-run bit identity on Metal.
- **Rollback:** revert call-site dispatch in `spectrum_model.py`, remove enum member. Kernel module remains dead code, deletable in follow-up.

## 11. Dependencies

- **Hard dep T1-1** (`5oar`): host/kernel split, `AtomicSnapshot`, `JaxMemoryPolicy`.
- **Lands before T1-5** (`ke4z`): T1-5's chunking applies to T1-4's LDM kernel.
- **Files touched:** `cflibs/radiation/{kernels.py, host.py, profiles.py, spectrum_model.py}`, `cflibs/manifold/{generator.py, batch_forward.py}`, new `tests/test_ldm_broaden.py`. No new third-party deps.

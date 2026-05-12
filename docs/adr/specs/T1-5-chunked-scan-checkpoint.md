# T1-5 Implementation Spec — Chunked `lax.scan` + Checkpoint Over Wavelength Grid

**Bead:** `CF-LIBS-improved-ke4z` · **ADR:** [ADR-0001](../ADR-0001-radis-jaxrts-pattern-survey.md) §5.2 C-P8, §5.5, §8.1 row T1-5 · **Wave:** 3 (parallel with T1-6) · **Hard deps:** T1-1 (`5oar`) + T1-4 (`e5o8`) · **Estimated effort:** ~3 days · **Reference:** exojax `OpaPremodit.xsmatrix` + `exojax.signal.ola.overlap_and_add_matrix` · **Revision:** 2026-05-12 (cross-audit) — canonical field name `line_wavelengths_nm`; test paths nested under `tests/radiation/`

## 1. Goals

Replace monolithic `jit(forward_model)` and `jit(vmap(single_spectrum_forward, ...))` (`cflibs/manifold/batch_forward.py:400`) with a chunked variant that splits the wavelength axis into `nstitch` chunks, scans with `jax.lax.scan`, wraps the body in `jax.checkpoint` for gradient memory savings, and recombines via overlap-and-add (OLA). Mirrors exojax PreMODIT chunked-scan pattern.

## 2. The memory problem

Representative CF-LIBS call: `N_lines ≈ 5000`, `N_λ ≈ 30000`. Current `_compute_spectrum_snapshot` (`cflibs/manifold/generator.py:572`) materializes `(N_wl, N_lines) = (30000, 5000)` `diff` matrix at fp32 = **600 MB**. With `vmap` over 8192-point manifold, peak transient memory = 600 MB × parallel-degree; OOMs at modest parallelism even on V100S 32 GB. Wide-spectrum cases (220-900 nm at R=30000 → N_λ ≈ 150000) blow budget at single-spectrum scale.

After T1-4 the per-call working set drops to `(N_σ, N_λ) ≈ (24, 30000) = 2.9 MB` — so T1-5's contribution is the activation/checkpoint memory along the wavelength axis for differentiation through the chunked forward (Bayesian inversion via NumPyro NUTS, manifold-trained surrogates with grad-through-forward).

## 3. Algorithm — overlap-and-add (OLA) chunked scan

Split `wavelength_grid (N_λ,)` into `nstitch` chunks of `div_length = ceil(N_λ / nstitch)` points, padded with `overlap = ceil(overlap_factor · max(σ_grid) / dlam)` samples per side (default `overlap_factor = 4.0`).

Per chunk: subset line catalog to lines within `wing_cutoff = overlap` of chunk's wavelength range (numpy-side index, passed as padded mask for jit-stability).

```python
@jit_if_available
def forward_model_chunked(
    plasma_state, atomic_snapshot, instrument, wavelength_grid, sigma_grid,
    *,
    nstitch: int = 1,       # static
    overlap: int = 0,       # static (samples per side)
    broadening_mode=BroadeningMode.LDM_GAUSSIAN,
):
    if nstitch == 1:        # static dispatch — no scan overhead
        return forward_model(plasma_state, atomic_snapshot, instrument,
                             wavelength_grid, sigma_grid, broadening_mode=broadening_mode)

    chunks, line_masks = _split_wavelength_grid(
        wavelength_grid, atomic_snapshot.line_wavelengths_nm,
        nstitch=nstitch, overlap=overlap,
    )  # chunks: (nstitch, div_length + 2*overlap); line_masks: (nstitch, N_lines)

    body = jax.checkpoint(
        _forward_model_per_chunk,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        static_argnames=("broadening_mode",),
    )

    def scan_step(carry, inputs):
        chunk_wl, chunk_mask = inputs
        partial = body(plasma_state, atomic_snapshot, instrument,
                       chunk_wl, sigma_grid, chunk_mask,
                       broadening_mode=broadening_mode)
        return carry, partial

    _, partials = jax.lax.scan(scan_step, None, (chunks, line_masks))
    return overlap_and_add(partials, overlap=overlap, output_length=N_lambda)
```

`_forward_model_per_chunk` is identical to full forward except: (a) chunk-slice input grid; (b) lines outside mask have intensities zeroed before broadening; (c) σ_grid unchanged; (d) LDM scatter index `j_i` recomputed per chunk relative to chunk `wl_min`.

## 4. OLA helper

```python
# cflibs/radiation/kernels.py
@jit_if_available
def overlap_and_add(
    partials: jax.Array,   # (nstitch, div_length + 2*overlap)
    *, overlap: int,       # static
    output_length: int,    # static
) -> jax.Array:            # (output_length,)
```

Mirrors `exojax.signal.ola.overlap_and_add_matrix`. Place each chunk at offset `chunk_idx · div_length − overlap` in output buffer of length `output_length + 2*overlap`, sum with `lax.fori_loop` + `dynamic_update_slice` add, slice off padding.

Algebraic identity: if unchunked is `S(λ) = Σ_i I_i · g(λ − λ_i; σ_i)`, chunked is exact (zero error) **as long as every line `i` is in every chunk within `overlap · dlam` of `λ_i`** — line mask enforces. OLA "approximation" only when `overlap < wing_extent` (testable via parity test).

## 5. Memory policy integration

`JaxMemoryPolicy` (from T1-1) already declares `nstitch: int = 1` and `overlap_factor: float = 4.0`. T1-5 wires their consumers.

Auto-selection (CPU-side, no JIT trace):

```python
# cflibs/radiation/host.py
def auto_nstitch(n_lines: int, n_lambda: int, available_bytes: int,
                 dtype_bytes: int = 4, safety_factor: float = 0.5) -> int:
    needed = n_lines * n_lambda * dtype_bytes
    return max(1, int(math.ceil(needed / (safety_factor * available_bytes))))
```

`available_bytes`:
- CUDA: `jax.devices()[0].memory_stats()["bytes_limit"]` minus reserved buffers.
- Metal: 0.5 × total system RAM (Apple unified memory).
- CPU: 0.25 × `psutil.virtual_memory().available` (psutil in dev extras; fall back to 4 GiB if missing).

Called once at manifold init or solver setup; result baked into static `nstitch`.

## 6. Where chunking lives

- **Kernel:** `cflibs/radiation/kernels.py::forward_model_chunked` and `::overlap_and_add` (new).
- **Host:** `cflibs/radiation/host.py::auto_nstitch` and `::build_chunk_metadata` (CPU-side; builds static `(chunks, line_masks, overlap)` tuple).
- **Spectrum model:** `cflibs/radiation/spectrum_model.py::SpectrumModel.compute_spectrum` adds `memory_policy` constructor arg; `memory_policy.nstitch > 1` dispatches to `forward_model_chunked`.
- **Manifold:** `cflibs/manifold/batch_forward.py::batch_forward_model` (L400) becomes `jit(vmap(forward_model_chunked, in_axes=(0, 0, 0, None, None)))` with `nstitch` static. `cflibs/manifold/generator.py::_time_integrated_spectrum` (L638-692) similarly substitutes. Default `nstitch = 1` so existing manifold checksums bit-identical.

## 7. Combining with T1-4

T1-4 reduces effective line count (5000 → 24 σ-layers per chunk). T1-5 chunks the wavelength axis. They compose multiplicatively: per-chunk working set after both = `(N_σ, div_length + 2·overlap) ≈ (24, 8000) = 0.75 MB`, comfortably fitting V100S parallelism even at vmap-over-1024 grid points. The `checkpoint` policy cuts peak backward-pass memory ~50% — needed for NumPyro NUTS over wide spectra.

`broadening_mode` propagated through `forward_model_chunked` — chunking works equally with `PHYSICAL_DOPPLER` and `LDM_GAUSSIAN`. `NIST_PARITY` not chunked (lines sparse, per-line Voigt cheap); assert `nstitch == 1` if `mode is NIST_PARITY`.

## 8. Acceptance criteria

1. **Memory:** for `(N_lines=5000, N_λ=30000)`, `nstitch=4` reduces peak device memory ≥3× vs `nstitch=1` (CPU: `jax.profiler.memory_profile`; V100S: `nvidia-smi` sampling).
2. **Parity:** rtol=1e-5 between `nstitch=1` and `nstitch ∈ {2, 4, 8}` on synthetic 5000-line spectrum (algebraically exact when `overlap ≥ wing_extent`; near machine epsilon at fp64, ~1e-5 at fp32).
3. **Differentiable:** `jax.grad(loss)(T_eV)` finite and within rtol=1e-4 of unchecked gradient for `nstitch ∈ {1, 4}`.
4. **Backward-pass memory:** with `checkpoint_policies.dots_with_no_batch_dims_saveable`, backward peak ≤50% of un-checkpointed peak (CPU `jax.profiler.memory_profile`).
5. **Auto-selection:** `auto_nstitch(5000, 30000, 8e9)` returns 1; `auto_nstitch(5000, 30000, 500e6)` returns ≥2.
6. **jax-metal fp32:** `nstitch ∈ {1, 4}` match CPU fp32 within rtol=5e-4.

## 9. Test plan

**New** `tests/radiation/test_chunked_scan.py`:
- `test_parity_nstitch_1_vs_4` — rtol=1e-5 (fp64) / 1e-4 (fp32 Metal).
- `test_parity_nstitch_sweep` — parametrize `nstitch ∈ {1, 2, 4, 8, 16}`; all match within tolerance.
- `test_grad_finite_chunked` — `jax.grad` w.r.t. `T_eV` finite; numerical parity rtol=1e-4.
- `test_overlap_factor` — `overlap_factor=2.0` introduces detectable edge ringing; `4.0` does not (validates default).
- `test_auto_nstitch_logic` — mock `available_bytes`, sweep `(N_lines, N_λ)`, assert monotonic.
- `test_chunked_with_ldm` — chunked × LDM stacked, rtol=1e-4 vs unchunked LDM and rtol=1e-3 vs unchunked PHYSICAL_DOPPLER (T1-4 tolerance compounds).
- `test_nist_parity_rejects_chunking` — assertion error if `nstitch > 1` with NIST_PARITY.
- `test_nlambda_not_divisible` — pad-and-mask last chunk; explicit case where `N_λ % nstitch ≠ 0`.

**New** `tests/radiation/test_memory_bench.py` (`@pytest.mark.slow`):
- `test_peak_memory_reduction` — `jax.profiler.memory_profile`, assert nstitch=4 peak ≤0.4× nstitch=1 peak on (5000, 30000).

**Existing** `tests/test_manifold.py`: golden checksums unchanged at `nstitch=1`; new parametrization with `nstitch=4` matches within rtol=1e-5. `pytest -m "physics" tests/test_radiation.py` identical at default `nstitch=1`.

## 10. Risks & rollback

- **OLA edge ringing if `overlap < 3·max(σ)`.** Mitigation: host-level assert `overlap ≥ ceil(3·max(σ_grid)/dlam)`; default `overlap_factor=4.0` enforces. `test_overlap_factor` is canary.
- **Heterogeneous last chunk** (`N_λ` not divisible by `nstitch`). `lax.scan` requires homogeneous shapes. Mitigation: pad wavelength grid to `nstitch · div_length` with tail-mask; trim after OLA. Explicit test `test_nlambda_not_divisible`.
- **`checkpoint_policies` API churn between JAX versions.** Pin `jax >= 0.4.30` in `pyproject.toml`; guard with `hasattr(jax.checkpoint_policies, "dots_with_no_batch_dims_saveable")` fallback to `everything_saveable` (slower, correct).
- **`jax.profiler.memory_profile` unavailable on jax-metal.** Skip memory bench on Metal (`pytest.mark.skipif(jax.default_backend() == "METAL")`).
- **Compile-time blow-up:** mis-specifying `static_argnames` causes per-`nstitch` recompiles. Mitigation: `nstitch` and `overlap` are static_argnums on `forward_model_chunked`; verified by `jax.make_jaxpr` snapshot test.
- **Rollback:** set `JaxMemoryPolicy.nstitch = 1` (default) — restores un-chunked path bitwise.

## 11. Landing order with T1-4

T1-4 (`e5o8`) lands first on its worktree. Acceptance gates (parity, manifold round-trip) must be green. T1-5 (`ke4z`) opens worktree against post-T1-4 tree; chunked kernel calls `ldm_broaden` directly. `forward_model_chunked` signature is mode-agnostic — if T1-4 reverts, T1-5 still chunks `PHYSICAL_DOPPLER`.

T1-5 not a dep of T1-2/T1-6, but they should adopt `forward_model_chunked` once it lands (follow-up commits in their worktrees, not in this spec).

## 12. Files touched (joint with T1-4)

| Path | T1-4 | T1-5 |
|---|---|---|
| `cflibs/radiation/kernels.py` | +`ldm_broaden` | +`forward_model_chunked`, +`overlap_and_add` |
| `cflibs/radiation/host.py` | +`broaden_lines_ldm` | +`auto_nstitch`, +`build_chunk_metadata` |
| `cflibs/radiation/profiles.py` | docstring "non-LDM fallback" | — |
| `cflibs/radiation/spectrum_model.py` | new `BroadeningMode.LDM_GAUSSIAN` dispatch | `memory_policy` arg + chunked dispatch |
| `cflibs/manifold/generator.py` | replace inline Humlicek L572-628 | chunked variant in `_time_integrated_spectrum` (L638-692) |
| `cflibs/manifold/batch_forward.py` | swap `voigt_spectrum_jax` at L395 | `batch_forward_model` (L400) uses chunked forward |
| `cflibs/core/jax_policy.py` | — | wire `nstitch` + `overlap_factor` consumers |
| `tests/test_ldm_broaden.py` | NEW | — |
| `tests/radiation/test_chunked_scan.py` | — | NEW |
| `tests/radiation/test_memory_bench.py` | — | NEW |

## 13. Cross-cutting verification

- **Physics-only ban:** kernels use only `jax.numpy`, `jax.lax`, `jax.fft`, `jax.checkpoint`, `jax.vmap`, `jax.ops.segment_sum`. None of banned APIs. Ruff TID251 passes; AST scanner irrelevant (shipped, not evolved).
- **Swarm quality gates:** `ruff check`, `black --check`, `mypy` (advisory), `pytest -x -q -m "not slow and not requires_db"`. Memory bench is `@pytest.mark.slow`.
- **NIST parity:** `scripts/validate_nist_parity.py` and `scripts/run_nist_validation.py` unchanged reports when `BroadeningMode.NIST_PARITY` + `nstitch=1`.
- **Real-data validation:** `scripts/validate_real_data.py --datasets steel_245nm FeNi_380nm` no regression in composition recovery RMSE.

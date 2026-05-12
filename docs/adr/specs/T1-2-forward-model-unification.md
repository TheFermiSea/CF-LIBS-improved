# T1-2 Implementation Spec — Unified Forward-Model Kernel

**Bead:** `CF-LIBS-improved-swgm` · **ADR:** [ADR-0001](../ADR-0001-radis-jaxrts-pattern-survey.md) §8.1 row T1-2 · **Wave:** 2 (lands AFTER T1-4 in wave 2 — sequential ordering; both touch `spectrum_model.py` dispatch and `batch_forward.py:L395`) · **Hard deps:** T1-1 (`5oar`) — requires the pytree registration + `AtomicDatabase.snapshot()` builder folded into T1-1 · **Estimated effort:** medium-high · **Revision:** 2026-05-12 (cross-audit) — adds `sigma_grid` parameter, clarifies NIST_PARITY per-line σ_inst handling, scopes `bayesian.py` adaptation to in-place edit

## 1. Goals

Replace the three coexisting forward-model implementations with a single pure-JAX kernel that all three call sites (`SpectrumModel`/`SpectrumModelJax`, `batch_forward_model`, `BayesianForwardModel`) dispatch into:

- **One source of truth** for the forward physics. `kernels.py::forward_model` is the only place Saha-Boltzmann → broadened, instrument-convolved spectrum exists.
- **vmap-clean.** Kernel accepts scalar/1-D plasma state and returns a 1-D intensity array; manifold batching = `jit(vmap(forward_model, ...))`. No internal Python `for` over lines or elements.
- **Backend-portable.** Runs unmodified on `jax-cpu`, `jax-cuda`, `jax-metal`. Precision via `JaxMemoryPolicy` from T1-1.
- **No regression.** All three call paths produce numerically identical output to current code (rtol=1e-5, atol=1e-7).
- **Removes broadening-mode duplication.** Reconciles three divergent broadening branches (`LEGACY`, `NIST_PARITY`, `PHYSICAL_DOPPLER`) plus `single_spectrum_forward`'s Voigt path plus `BayesianForwardModel._compute_spectrum`'s Voigt-with-Stark path into one parameterized kernel.

## 2. Unified kernel API contract

```python
# cflibs/radiation/kernels.py
from cflibs.core.jax_runtime import jit_if_available
from cflibs.plasma.state import SingleZoneLTEPlasma          # pytree-registered (T1-1)
from cflibs.atomic.snapshot import AtomicSnapshot            # frozen dataclass (T1-1)
from cflibs.instrument.model import InstrumentModel          # pytree-registered (T1-1)
from cflibs.radiation.profiles import BroadeningMode

@jit_if_available(static_argnames=("broadening_mode", "apply_self_absorption"))
def forward_model(
    plasma_state: SingleZoneLTEPlasma,
    atomic_snapshot: AtomicSnapshot,
    instrument: InstrumentModel,
    wavelength_grid: jnp.ndarray,            # (N_wl,) policy.real_dtype
    sigma_grid: jnp.ndarray | None = None,   # (N_σ,) — REQUIRED for LDM_GAUSSIAN;
                                              # None for PHYSICAL_DOPPLER/NIST_PARITY/LEGACY.
                                              # Host builds via T1-4's broaden_lines_ldm
                                              # path; threaded through here so T1-5's
                                              # chunked variant can scan over chunks.
    *,
    broadening_mode: BroadeningMode,         # static (enum hashable)
    path_length_m: float,                    # traced
    apply_self_absorption: bool = False,     # static
) -> jnp.ndarray:                            # (N_wl,) same dtype as input
    """Synthetic LIBS spectrum on wavelength_grid. W m^-2 nm^-1 sr^-1.

    Composes:
      1. Saha-Boltzmann populations (cflibs.plasma.kernels.saha_boltzmann)
      2. Line emissivity ε_l = hc/(4π λ_l) · A_ki · n_k
      3. Per-line profile (Gaussian Doppler+instrument, optional Lorentzian Stark)
      4. Sum over lines via (N_wl, N_lines) outer-product reduction
      5. Optional radiative transfer  I = B(λ,T) · (1 - exp(-κL))
      6. Instrument convolution folded into per-line sigma (Voigt normalization preserving)
    """
```

**JIT boundary lives entirely inside this function.** Call sites do NOT wrap it again. Manifold batching applies `jit(vmap(...))` and the inner jit is a no-op (XLA fuses).

Static vs. traced split:
- **Static** (hashed into jit cache key): `broadening_mode`, `apply_self_absorption`, `atomic_snapshot.n_elements`, `atomic_snapshot.n_stages`, shape of `wavelength_grid`, shape of `atomic_snapshot.line_*` arrays.
- **Traced** (pytree leaves): `plasma_state.T_e_eV`, `.n_e_cm3`, `.composition_array`, all `line_*` arrays, instrument FWHM/R/sigma_array, `wavelength_grid`, `path_length_m`.

Precision: respects `cflibs.core.jax_runtime.policy.real_dtype`. Output dtype matches `wavelength_grid.dtype`. Callers convert fp64→fp32 upstream on Metal; kernel does not silently downcast.

## 3. Three call sites — adaptation

### 3.1 `SpectrumModel.compute_spectrum` → thin wrapper

```python
def compute_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
    self.plasma.validate()
    snapshot = self.atomic_db.snapshot(
        elements=list(self.plasma.species.keys()),
        wavelength_range=(self.lambda_min, self.lambda_max),
        min_relative_intensity=0.01 if self.broadening_mode == BroadeningMode.NIST_PARITY else 10.0,
    )
    wl = jnp.asarray(self.wavelength, dtype=jax_policy().real_dtype)
    intensity = forward_model(
        self.plasma, snapshot, self.instrument, wl,
        broadening_mode=self.broadening_mode,
        path_length_m=self.path_length_m,
        apply_self_absorption=True,  # legacy SpectrumModel behavior
    )
    return self.wavelength, np.asarray(intensity)
```

`SpectrumModelJax` becomes a deprecation alias of `SpectrumModel`.

### 3.2 `batch_forward_model` → `jit(vmap(forward_model))`

```python
# cflibs/manifold/batch_forward.py
from cflibs.radiation.kernels import forward_model

@partial(jit, static_argnames=("broadening_mode", "apply_self_absorption"))
def batch_forward_model(
    plasma_states_batch: SingleZoneLTEPlasma,   # leading batch axis
    atomic_snapshot: AtomicSnapshot,            # shared
    instrument: InstrumentModel,                # shared
    wavelength_grid: jnp.ndarray,
    *,
    broadening_mode: BroadeningMode = BroadeningMode.PHYSICAL_DOPPLER,
    path_length_m: float = 0.01,
    apply_self_absorption: bool = False,        # manifold default: thin
) -> jnp.ndarray:                                # (batch, N_wl)
    return vmap(forward_model, in_axes=(0, None, None, None))(
        plasma_states_batch, atomic_snapshot, instrument, wavelength_grid,
        broadening_mode=broadening_mode,
        path_length_m=path_length_m,
        apply_self_absorption=apply_self_absorption,
    )
```

`BatchAtomicData` is renamed/aliased to `AtomicSnapshot`. `single_spectrum_forward` retired; existing callers import `forward_model` directly.

### 3.3 `BayesianForwardModel.forward` → in-place import (NOT a new file)

**Important:** T1-2 does **not** create `cflibs/inversion/solve/bayesian/forward.py` — that directory doesn't exist until T1-6 decomposes the monolith. Instead, T1-2 edits the **existing** `cflibs/inversion/solve/bayesian.py:1071` in place: replace the body of `BayesianForwardModel._compute_spectrum` (and any Voigt/Saha-Boltzmann helpers it calls inline) with a single delegating call to `cflibs.radiation.kernels.forward_model`. T1-6 then carries this in-place edit over to `bayesian/forward.py` during its decomposition.

```python
# cflibs/inversion/solve/bayesian.py (in-place edit at the existing class)
from cflibs.radiation.kernels import forward_model

class BayesianForwardModel:
    def _compute_spectrum(self, T_eV, log_ne, concentrations, total_species_density_cm3=None):
        plasma = self._pack_plasma(T_eV, log_ne, concentrations, total_species_density_cm3)
        return forward_model(
            plasma, self._snapshot, self._instrument, self.wavelength,
            sigma_grid=None,                          # PHYSICAL_DOPPLER path; no LDM
            broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
            path_length_m=self.path_length_m,
            apply_self_absorption=False,
        )
```

`AtomicDataArrays` (existing at `bayesian.py:539`) gets a `@classmethod from_snapshot(cls, snap: AtomicSnapshot) -> AtomicDataArrays` adapter so legacy callers keep their interface; `BayesianForwardModel.__init__` switches to storing `self._snapshot: AtomicSnapshot` directly. T1-6 will retire `AtomicDataArrays` if it proves unnecessary post-decomposition.

Chebyshev baseline stays in `BayesianForwardModel` as an additive post-step — sampler-level state, not physics.

## 4. Where the kernel lives

`cflibs/radiation/kernels.py::forward_model` — pure-kernel half of host/kernel split from T1-1. Host half handles SQLite I/O, validation, dtype coercion, instrument-mode guards, NumPy↔JAX boundary, logging.

Public re-export: `from cflibs.radiation import forward_model`.

Saha-Boltzmann delegated to `cflibs/plasma/kernels.py::saha_boltzmann_populations` (T1-1's plasma kernel half). Stark/Voigt/Doppler helpers in `cflibs/radiation/profiles.py`.

## 5. Reconciliation table — three current impls → unified kernel

| Concern | `SpectrumModel` | `single_spectrum_forward` | `BayesianForwardModel` | Unified kernel |
|---|---|---|---|---|
| `min_relative_intensity` filter (`spectrum_model.py:226-240`) | 0.01 if NIST_PARITY else 10.0 | n/a (snapshot pre-filtered) | n/a (snapshot pre-filtered) | **Moves out of kernel** — host passes pre-filtered `AtomicSnapshot` |
| Per-line sigma fallback (`emissivity.py:123-130`) | NumPy fallback | per-line sigma natively | per-line sigma natively | **`single_spectrum_forward` form wins.** Per-line sigma is canonical; scalar = degenerate case `jnp.full((N_lines,), σ)`. Fallback in `emissivity.py` deleted |
| BroadeningMode branches (`spectrum_model.py:247-250`) | LEGACY/NIST_PARITY/PHYSICAL_DOPPLER | Voigt via `voigt_spectrum_jax` | Voigt via Weideman Faddeeva | Single function, branched on `broadening_mode` as **static** arg. LEGACY retained but emits `DeprecationWarning` from host. Weideman path becomes canonical Voigt |
| Radiative transfer | Always applied | Not applied | Not applied | **`apply_self_absorption: bool` static flag.** Default `True` for `SpectrumModel`, `False` for manifold + Bayesian |
| Instrument convolution | Separate scipy/JAX step (`spectrum_model.py:283-301`) | Not applied | Folded into per-line sigma (Voigt-normalization preserving) | **Bayesian approach wins.** Instrument σ added in quadrature to per-line Gaussian. For NIST_PARITY mode (resolving-power R, wavelength-dependent), kernel evaluates `σ_inst(λ_i) = λ_i / (R · 2√(2 ln 2))` per-line inline from `instrument.resolving_power` + `atomic_snapshot.line_wavelengths_nm`. For fixed-FWHM mode, scalar σ broadcast. Separate convolution removed except for LEGACY back-compat. |
| Saha-Boltzmann | `solve_plasma` dict per-element (NumPy) | Inline scan over stages (JAX) | Two-stage explicit | Single helper `saha_boltzmann_populations(plasma, snapshot)` returning `(N_lines,)` upper-level array. `lax.scan` over stages canonical |
| Precision | fp64 hardcoded (`spectrum_model.py:460`) | fp64 hardcoded (`batch_forward.py:322-328`) | `_JAX_REAL_DTYPE` policy-aware | **Bayesian approach wins.** All casts through `jax_policy().real_dtype` |

## 6. Acceptance criteria

1. All three current call sites return numerically identical output to pre-refactor at rtol=1e-5, atol=1e-7.
2. `kernels.py::forward_model` is the only function in `cflibs/` computing a spectrum from `(plasma, snapshot, grid)`. Grep audit: no `epsilon = ... * A_ki * n_k` outside `kernels.py`.
3. Public API preserved: `from cflibs.radiation import SpectrumModel, SpectrumModelJax` and `from cflibs.manifold.batch_forward import batch_forward_model` resolve and produce same outputs.
4. Physics-only constraint: no new imports of `jax.nn`, `flax`, `equinox`, `jax.experimental.stax`. Ruff TID251 clean.
5. Backend matrix: cpu (fp64), cuda (fp64), metal (fp32 with warning). Marker `requires_metal` (skip-if).

## 7. Test plan

- **`tests/radiation/test_forward_model_parity.py`** (new). Fixed `(SingleZoneLTEPlasma, AtomicSnapshot, InstrumentModel, wavelength_grid)`:
  - `SpectrumModel(...).compute_spectrum()` → `I1`
  - `batch_forward_model(batched_1, ...)` → `I2[0]`
  - `BayesianForwardModel(...).forward(T, log_ne, conc)` → `I3` (sans baseline)
  - Assert `np.allclose(I1, I2[0], rtol=1e-5, atol=1e-7)` and same for `I3` vs `I1`.
- **`tests/radiation/test_forward_model_vmap.py`** (new). Batch of 100 plasma states; `batch_forward_model` output `(100, N_wl)` matches Python loop within rtol=1e-5.
- **`tests/test_round_trip.py`** (existing). `GoldenSpectrum` round-trip stays green.
- **Existing**: `tests/test_spectrum_model.py`, `tests/radiation/test_spectrum_model_jax.py`, `tests/test_batch_forward.py`, `tests/test_bayesian.py`, `tests/validation/validation/accuracy/test_batch_forward_accuracy.py` unchanged.
- **Backend matrix.** `JAX_PLATFORMS=cpu pytest tests/`; on cluster CI a CUDA run. Apple Silicon: `skip_if_metal_no_fp64` marker with fp32 fallback at rtol=1e-3.
- **Benchmark.** `pytest --benchmark-only` — manifold throughput unchanged or improved.

## 8. Dependencies & sequence

- **Hard dep: T1-1** (`5oar`) — needs pytree-registered `SingleZoneLTEPlasma`, `InstrumentModel`, `AtomicSnapshot`, host/kernel split for `plasma/` and `instrument/`, shared `jit_if_available` decorator.
- **Soft dep: T2-2** (rich `AtomicSnapshot` population). T1-1 gives the carrier; full snapshot builder is T2-2.
- **Blocks: T1-6** (`0mor`). T1-6's `bayesian/forward.py` imports `forward_model`. T1-6 priors/samplers can begin in parallel.
- **Merge order:** T1-1 → T1-2 → T1-6. If T1-1 lands partially (e.g., plasma pytree done but instrument not), gate T1-2 PR behind both.

## 9. Risks & rollback

- **Numerical drift in NIST_PARITY mode** when collapsing the separate instrument convolution into per-line sigma. Mitigation: explicit parity test with `min_ri=0.01` snapshot; rtol=1e-5 against current `SpectrumModel` output for a NIST_PARITY benchmark.
- **vmap compilation cost.** Different element counts in `atomic_snapshot.species` trigger recompiles. Mitigation: standardize on a padded snapshot at construction time (pad to `N_elements_max=20`).
- **Rollback:** revert the kernel commit; the three call sites individually retain their fallback paths until T1-2 merges to integration.

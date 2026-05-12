# T1-1 Implementation Spec — Host/Kernel Split + Shared JAX Decorator

**Bead:** `CF-LIBS-improved-5oar` · **ADR:** [ADR-0001](../ADR-0001-radis-jaxrts-pattern-survey.md) §8.1 row T1-1 · **Wave:** 1 (alone) · **Estimated effort:** 5 days

## 1. Goals

- Eliminate the per-file `try: import jax` fallback boilerplate (28 instances of the pattern across `cflibs/`, 33 files importing `jax` total per audit) by routing every JAX consumer through a single shared adapter in `cflibs.core.jax_runtime`. After T1-1, only `jax_runtime.py`, `cflibs/core/platform_config.py`, and the carve-outs in §5 may contain a literal `try: import jax`.
- Retire the `*Jax`-subclass-co-located-in-same-file pattern (currently in `radiation/spectrum_model.py`, `plasma/state.py`, `plasma/saha_boltzmann.py`, `instrument/model.py`) in favour of a `host.py` / `kernels.py` split where pure jit-able numerics live in `kernels.py` and orchestration / SQLite / validation / error handling stays in `host.py`.
- Surface backend precision (no fp64 on Metal; Metal is already force-disabled in `cflibs/core/platform_config.py`) via an explicit `JaxMemoryPolicy` frozen dataclass and a `check_jax64bit(allow_fp32_on_metal=False)` runtime guard (Tier-3 candidate T3-2 folded in here because every other Tier-1 depends on it).
- Introduce an `AtomicSnapshot` frozen dataclass so jit'd code consumes plain arrays instead of holding a live SQLite connection inside a trace (Tier-2 candidate T2-2 partially folded — only the carrier; rich population happens in T2-2).
- Preserve all backward-compat shim files at `cflibs/inversion/*.py` flat path verbatim. No public import path changes.

## 2. Exact API contract — `cflibs.core.jax_runtime`

The existing helpers in `cflibs/core/jax_runtime.py` (`HAS_JAX`, `jax_active_backend`, `jax_backend_supports_x64`, `jax_backend_supports_complex`, `jax_default_real_dtype`, `jax_default_complex_dtype`) remain unchanged. T1-1 adds:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

JAX_BACKEND: Literal["cpu", "cuda", "metal", "none"]
"""Module-level snapshot of jax_active_backend() resolved at import time.
'none' when HAS_JAX is False. Recomputed via _refresh_runtime_state() (test helper)."""

def jit_if_available(*jit_args, **jit_kwargs) -> Callable[[F], F]:
    """Decorator that applies ``jax.jit`` when JAX is importable, else returns
    the function unchanged. Drop-in replacement for the 28 ad-hoc try/except
    blocks scattered across cflibs/.

    Usage (no-arg form):
        @jit_if_available
        def _broaden(wl, lines, sigmas): ...

    Usage (with jit kwargs):
        @jit_if_available(static_argnums=(2,))
        def _kernel(x, y, n): ...

    Both forms work. When JAX is missing the wrapped function still runs
    against ``cflibs.core.jax_runtime.jnp`` (aliased to numpy in the fallback)."""

def vmap_if_available(in_axes=0, out_axes=0, axis_name=None) -> Callable[[F], F]:
    """Decorator that applies ``jax.vmap`` when JAX is importable, else
    returns a Python-loop emulation (numpy.stack along out_axes).
    Correctness-only fallback; performance parity not promised."""

@dataclass(frozen=True)
class JaxMemoryPolicy:
    """Centralised knob-set for JAX precision, chunking, OOM mitigation.

    Mirrors exojax's ``opacity/policies.py::MemoryPolicy`` (ADR §5.2 C-P9).
    Hashable so it can be keyed into a jit cache.
    """
    allow_32bit: bool = False         # If True, do not raise when backend lacks fp64
    nstitch: int = 1                  # ν-grid chunks for lax.scan (T1-5)
    cutwing: float = 50.0             # Line-wing truncation in σ units (Radis truncation=50)
    checkpoint: bool = False          # Apply jax.checkpoint to chunked kernels (T1-5)
    overlap_factor: float = 4.0       # T1-5: overlap multiples of max(σ_grid)

def check_jax64bit(allow_fp32_on_metal: bool = False, raise_on_violation: bool = True) -> None:
    """Runtime guard mirroring exojax ``utils/jaxstatus.py``.

    Verifies:
      1. ``jax.config.jax_enable_x64`` is True (conftest.py sets this for tests)
      2. ``jax_backend_supports_x64()`` returns True

    Raises ``ValueError`` on violation when ``raise_on_violation=True``.
    Logs WARNING when ``allow_fp32_on_metal=True`` and backend is Metal."""

@dataclass(frozen=True)
class AtomicSnapshot:
    """Frozen, jit-friendly snapshot of an ``AtomicDatabase`` query.

    Mirrors exojax ``database/contracts.py::MDBSnapshot`` (ADR §5.2 C-P10).
    Decouples jit-traced kernels from the SQLite connection.
    """
    species: tuple[tuple[str, int], ...]      # static metadata
    line_wavelengths_nm: "jnp.ndarray"        # (N_lines,)
    line_A_ki: "jnp.ndarray"
    line_E_k_ev: "jnp.ndarray"
    line_g_k: "jnp.ndarray"
    line_species_index: "jnp.ndarray"         # (N_lines,) int32
    partition_coeffs: "jnp.ndarray"           # (N_species, N_poly_order)
```

Fallback semantics: when `HAS_JAX is False`, `jit_if_available` returns `lambda f: f`, `vmap_if_available` returns a NumPy-stack emulation, the module-level `jnp` symbol is aliased to `numpy`.

## 3. Host/kernel split convention

| Goes in `host.py` | Goes in `kernels.py` |
|---|---|
| SQLite I/O (`atomic_db.get_transitions`) | Pure functions of `jnp.ndarray` inputs |
| Python control flow over `dict[str, Transition]` | Static-shape `vmap`/`scan`-able loops |
| Validation, range checks, raising `ValueError` | Closed-form algebra; FFT/conv kernels |
| Logging | Nothing that calls Python builtins on traced arrays |
| Dataclass plumbing, `__init__`, mode dispatch | `@jit_if_available`-decorated functions |
| `dict` → `AtomicSnapshot` conversion | Consumers of `AtomicSnapshot` |
| Returning `np.ndarray` to callers | Internal `jnp.ndarray` arithmetic |

**Worked example — `cflibs/radiation/spectrum_model.py`.**

After T1-1, `cflibs/radiation/host.py` contains the `SpectrumModel` class (orchestration: `__init__` validation, wavelength-grid setup, SQLite transition fetch, broadening-mode dispatch, returning NumPy arrays). It calls into `cflibs/radiation/kernels.py` for the heavy work. Class name and constructor signature unchanged.

`cflibs/radiation/kernels.py` contains the four jitted helpers from `spectrum_model.py:333-378` — `_planck_radiance_jax`, `_broaden_per_line_jax`, `_radiative_transfer_jax`, `_gaussian_kernel_jax` — re-decorated with `@jit_if_available`. The standalone `SpectrumModelJax` at L410 is folded into a deprecation alias of `SpectrumModel` with a `DeprecationWarning` on `__init__`.

## 4. Packages affected in T1-1 (in scope)

| Package | Files affected | Public API preserved as |
|---|---|---|
| `cflibs/radiation/` | `spectrum_model.py` (564 LOC) → `host.py` + `kernels.py`; `profiles.py` (903 LOC) → split jitted helpers into `kernels.py`; `emissivity.py` + `stark.py` → route through `jit_if_available` | `from cflibs.radiation import SpectrumModel, SpectrumModelJax, BroadeningMode, ...` unchanged. `SpectrumModelJax` becomes an alias. |
| `cflibs/plasma/` | `saha_boltzmann.py` (837) — extract jitted kernels at L508-562 into `kernels.py`, fold `SahaBoltzmannSolverJax` (L592) into `use_jax` flag on `SahaBoltzmannSolver`; `state.py` — retire `SingleZoneLTEPlasmaJax` alias (full pytree registration deferred to T2-3); `partition.py` (654 LOC, audit-confirmed — ADR said ~1000) — kernels at L189-589 into `kernels.py` | `SahaBoltzmannSolverJax`, `SingleZoneLTEPlasmaJax` keep importing. |
| `cflibs/instrument/` | `model.py` — fold `InstrumentModelJax`; `convolution.py` — route through `jit_if_available` | `InstrumentModelJax` alias preserved. |
| `cflibs/manifold/` | `batch_forward.py` (511) + `generator.py` (866) — finish migration to shared decorators; `loader.py` — one-line decorator change | Public API unchanged. |
| `cflibs/core/` | Extend `jax_runtime.py` with §2 API. No new file. | Existing names untouched. |

Total: 5 packages, ~14 files touched, ~3700 LOC reorganized. Physics bodies inside kernel functions are NOT modified.

## 5. Packages NOT touched by T1-1 (deferred)

- **`cflibs/inversion/identify/`** — `alias.py` (2999), `comb.py`, `correlation.py`, `line_detection.py`, `spectral_nnls.py`. Recently JAX-ported with their own `_HAS_JAX` pattern. Deferred to follow-on after T1-2 stabilises.
- **`cflibs/inversion/physics/boltzmann_jax.py`** — already isolated and pytree-registered; opt-in via `CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION`.
- **`cflibs/inversion/solve/`** — `iterative.py` (1326), `bayesian.py` (3264), `joint_optimizer.py`, `coarse_to_fine.py`. Owned by T1-3 (`iterative.py`) and T1-6 (`bayesian.py`). T1-1 explicitly does not enter these files. Implication: T1-3 must do its own local host/kernel split for `iterative.py` (it can run in parallel with T1-1).
- **`cflibs/inversion/preprocess/deconvolution.py`** and **`cflibs/inversion/common/pca.py`** — separate hygiene pass.
- **`cflibs/inversion/runtime/streaming.py`** — has a deeply-nested `try: import jax` inside a method body (L1187).
- **`cflibs/hpc/`**, **`cflibs/core/platform_config.py`**, **`cflibs/benchmark/unified.py`**, **`cflibs/evolution/evaluator.py`** — capability-detection / static-analysis JAX imports, exempt.

## 6. Acceptance criteria

1. `ruff check cflibs/ tests/` returns 0 errors / 0 warnings.
2. `black --check cflibs/` clean.
3. `mypy cflibs/` no regression vs `dev` baseline.
4. `grep -rEn "^[[:space:]]*try:[[:space:]]*$" cflibs/ --include="*.py" -A 2 | grep -B 1 "import jax"` matches only in: `cflibs/core/jax_runtime.py`, `cflibs/core/platform_config.py`, `cflibs/hpc/gpu_config.py`, `cflibs/hpc/distributed_mcmc.py`, `cflibs/benchmark/unified.py`, `cflibs/evolution/evaluator.py`, `cflibs/inversion/runtime/streaming.py`, plus the five identify/ carve-outs in §5.
5. `python -c "from cflibs.inversion.solver import IterativeCFLIBSSolver; from cflibs.inversion.alias_identifier import ALIASIdentifier; from cflibs.inversion.boltzmann_jax import BoltzmannFitResultJax"` succeeds.
6. `pytest tests/radiation/test_spectrum_model_jax.py -v` passes at rtol=1e-5, atol=1e-7.
7. `pytest tests/plasma/test_saha_boltzmann_jax.py tests/plasma/test_state_jax.py tests/instrument/test_model_jax.py -v` all pass.
8. `JAX_PLATFORMS=cpu pytest tests/ -x -q -m "not slow and not requires_db"` passes within ±5% wall-clock of pre-T1-1 baseline.

## 7. Test plan

**Existing tests that must stay green:**
- `tests/radiation/test_spectrum_model_jax.py` (rtol=1e-5).
- `tests/plasma/test_saha_boltzmann_jax.py`, `tests/plasma/test_state_jax.py`, `tests/instrument/test_model_jax.py`.
- `tests/test_alias.py`, `tests/test_comb.py`, `tests/test_boltzmann.py`, `tests/test_batch_forward.py`, `tests/test_bayesian.py`, `tests/test_atomic.py`.

**New tests:**
- `tests/core/test_jax_runtime.py::test_jit_if_available_with_and_without_jax`
- `tests/core/test_jax_runtime.py::test_vmap_if_available_falls_back_to_numpy_stack`
- `tests/core/test_jax_runtime.py::test_jax_memory_policy_is_hashable_and_frozen`
- `tests/core/test_jax_runtime.py::test_check_jax64bit_raises_when_x64_disabled`
- `tests/core/test_jax_runtime.py::test_atomic_snapshot_arrays_have_consistent_shape`
- `tests/test_jax_import_hygiene.py` — AST-walks `cflibs/`, asserts no top-level `Try` with `import jax` body outside acceptance criterion 4 carve-outs. This is the canonical regression guard.

## 8. Migration order within T1-1 (5 commits)

1. **Land `cflibs/core/jax_runtime` extensions first.** Add API in §2 + `tests/core/test_jax_runtime.py`. No other files touched.
2. **Migrate `cflibs/radiation/`** as canonical worked example. Create `kernels.py`. Add `tests/test_jax_import_hygiene.py`.
3. **Migrate `cflibs/plasma/`** — extract `partition.py` jitted helpers; fold `SahaBoltzmannSolverJax` into `use_jax` flag; alias `SingleZoneLTEPlasmaJax`.
4. **Migrate `cflibs/instrument/`** — fold `InstrumentModelJax` via method-level dispatch.
5. **Migrate `cflibs/manifold/`** — replace `if HAS_JAX:` ladders with `@jit_if_available`.

Each commit ends with full pytest gate green.

## 9. Conflict surface

| Other bead | Overlap | Recommended dep |
|---|---|---|
| T1-2 `swgm` (forward unification) | `spectrum_model.py`, `batch_forward.py`, `bayesian.py` | **Hard dep on T1-1** — `bd dep add CF-LIBS-improved-swgm CF-LIBS-improved-5oar` (already wired) |
| T1-3 `14p6` (lax.while_loop) | `iterative.py` only (in T1-1 carve-out §5) | **Parallel-safe.** Soft dep (T1-3 benefits from shared decorator, can copy inline if needed) |
| T1-4 `e5o8` (LDM broadening) | `profiles.py`, `manifold/generator.py` | **Hard dep on T1-1** — `bd dep add CF-LIBS-improved-e5o8 CF-LIBS-improved-5oar` |
| T1-5 `ke4z` (chunked scan) | `manifold/generator.py`, `spectrum_model.py` + `JaxMemoryPolicy` consumer | **Hard dep on T1-1 + T1-4** |
| T1-6 `0mor` (Bayesian decomp) | `bayesian.py` (T1-1 carve-out §5) | **Soft dep on T1-1; hard dep on T1-2** for `forward.py` import. Priors/samplers decomp can run in parallel with T1-1 |
| T2-2 (AtomicSnapshot rich population) | populates the empty dataclass from T1-1 | Hard dep on T1-1 |
| T2-3 (full pytree registration) | replaces aliases from T1-1 step 3-4 | Hard dep on T1-1 |

## 10. Risks & rollback

- **Subtle pytree-registration breaking jit cache.** T1-1 does NOT register `SingleZoneLTEPlasma` as a pytree (deferred to T2-3). Risk: a caller used `type(x) is SingleZoneLTEPlasmaJax`. Mitigation: pre-spec grep found only 6 production refs and 4 test refs, all `isinstance`/import. Rollback: revert single commit; aliases revert to subclasses.
- **`jit_if_available` argument-form ambiguity.** Both `@jit_if_available` and `@jit_if_available(static_argnums=(2,))` must work. Tests cover both forms in both `HAS_JAX` branches.
- **Backward-compat shim breakage.** §5 carve-out — T1-1 does not touch `cflibs/inversion/`. Hygiene test is per-commit guard.
- **Forgotten import-cycle.** `kernels.py` must not import from `host.py`. Enums and constants live in `host.py` or `cflibs.core.constants`. Hygiene test asserts.
- **Per-commit rollback.** Each of the 5 sub-steps is a self-contained commit; reverting any single one leaves the others intact.

# ADR-0001 — Pattern survey from Radis, jaxrts, Exojax, and atmospheric RT codes for CF-LIBS-improved

- **Status:** Proposed (survey-first; no implementation commitments)
- **Date:** 2026-05-12
- **Authors:** CF-LIBS Architecture working group (Brian Squires + Opus 4.7 research team A–D)
- **Tracking bead:** CF-LIBS-improved-j7vb
- **Scope:** Comprehensive, ranked catalogue of patterns observed in four comparable physics-only spectroscopy / radiative-transfer codes, with explicit applicability mapping to CF-LIBS-improved files. **No implementation decisions are taken in this ADR** — it is the evidence base for subsequent ADRs that will commit to specific changes.
- **Out of scope:** Anything that violates the physics-only constraint (see CLAUDE.md and bead `cf-libs-hard-project-constraint-the-final-algorithm`). All patterns surveyed are JAX-only or numpy/cython/Rust — none involve `flax`, `equinox`, `jax.nn`, or trained models.

---

## 1. Context

CF-LIBS-improved is a physics-only Calibration-Free LIBS pipeline (67 119 Python LOC across 173 files plus two Rust crates). The codebase is approaching a structural inflection point:

- The inversion sub-package alone is **46 % of LOC** (31 058 lines), with nine files above 1 000 LOC and three above 2 000.
- **Three independent forward-model implementations coexist** — `cflibs/radiation/spectrum_model.py::SpectrumModel`, `cflibs/manifold/batch_forward.py::single_spectrum_forward`, and `cflibs/inversion/solve/bayesian.py::BayesianForwardModel` — all implementing Saha → Boltzmann → emissivity → broadening → instrument convolution.
- A JAX migration is in flight (recent commits 926ec47, d36a00d, 726fefb, ef8d4f7, 320f143 ported comb / correlation / alias / k-det / Boltzmann fitter) but the migration pattern has been *additive*: a `*Jax` subclass or `_jax` sibling appended next to the NumPy original, with **per-file `try: import jax` fallback boilerplate duplicated across 31 files**.
- The hard physics-only constraint is solid (Ruff TID251 + AST blocklist), but the architectural patterns for *how* physics-only JAX code should be organized inside CF-LIBS are ad-hoc, file-by-file.

Before committing to a refactor direction we surveyed four comparable codebases for patterns that have been validated under production load. The survey was executed by four parallel Opus 4.7 research agents on 2026-05-12:

| Stream | Subject | Output § |
|---|---|---|
| A | CF-LIBS-improved internal map (read-only, current state) | § 3 |
| B | Radis (Pannier & Laux 2019; van den Bekerom & Pannier 2021) | § 4 |
| C | jaxrts (Lütgert et al. 2026) + Exojax (Kawahara et al. 2022, 2025) | § 5 |
| D | petitRADTRANS, HELIOS, Sherpa, LMFIT, stellar/LIBS-specific codes | § 6 |

The four streams were tasked independently and converged on several patterns (§ 7), which strengthens confidence in the resulting recommendations (§ 8).

---

## 2. Decision

**This ADR records no implementation decision.** Per the agreed survey-first scope, the deliverable is the ranked candidate catalogue in § 8 plus the convergent-pattern synthesis in § 7. Implementation choices will be made in follow-on ADRs (ADR-0002 onwards), each scoped to a single pattern or pattern cluster.

We do, however, record two **structural recommendations** that emerged independently from three of the four streams (A, C, D) and should frame any subsequent decision:

1. **Adopt a `host.py` / `kernels.py` file split** for any package undergoing JAX migration, retiring the `*Jax`-subclass-in-same-file pattern. (Convergent: HELIOS, exojax `rt/` vs `opacity/lpf/`, jaxrts `models.py` method-level jit.)
2. **Unify the three forward-model implementations** into a single pure-JAX kernel consumed by `SpectrumModel`, `ManifoldGenerator`, and `BayesianForwardModel`. (Driven by stream A's internal-duplication finding; informed by exojax's layered `rt/` → `opacity/` → `special/` decomposition.)

Both recommendations are pure architecture and contain no physics changes; they are *enabling* moves for later, physics-bearing ADRs.

---

## 3. Stream A — CF-LIBS-improved internal map (current state)

> Source: Stream A research agent, output captured 2026-05-12. Read-only audit; no files modified.

### 3.1 Sub-package overview

| Package | Role | Key files | LOC | JAX? | Pain points |
|---|---|---|---|---|---|
| `cflibs/plasma/` | Plasma state, Saha-Boltzmann solver, partition fns | `state.py` (430), `saha_boltzmann.py` (837), `partition.py` (~1000), `anderson_solver.py` | 2 870 | yes (5/6 files) | `SahaBoltzmannSolver` and `SingleZoneLTEPlasma` each have a `*Jax` subclass tail-appended in the same file — adds size, blocks isolated JIT reuse |
| `cflibs/radiation/` | Forward model orchestrator, emissivity, profiles, Stark | `spectrum_model.py` (564), `profiles.py` (903), `emissivity.py`, `stark.py`, `batch.py` | 2 211 | yes (4/6) | One `DEPRECATED` voigt impl in `profiles.py:616`; `spectrum_model.py` has parallel `SpectrumModel` + `SpectrumModelJax` in one file |
| `cflibs/atomic/` | SQLite atomic DB + structures | `database.py`, `database_generator.py`, `reference_data.py` (1093), `structures.py` | 2 069 | no | `reference_data.py` is the largest; lookup via cache decorators + pool |
| `cflibs/inversion/` | Full inversion pipeline (6 sub-packages + 31 shims) | see § 3.4 | 31 058 | yes (10+ files) | 31 backward-compat shim files at flat path; oversized files (`alias.py`=2999, `iterative.py`=1326, `bayesian.py`=3264) |
| `cflibs/manifold/` | Pre-computed spectral manifold (JAX) | `generator.py` (866), `batch_forward.py` (511), `loader.py`, `basis_index.py`, `vector_index.py` | 3 119 | yes (4/8) | `batch_forward.py` duplicates the forward model pipeline (Saha→Boltz→emissivity→broaden) the radiation package already owns |
| `cflibs/instrument/` | Instrument response, convolution | `model.py`, `convolution.py`, `echelle.py` | 829 | yes | `InstrumentModel` + `InstrumentModelJax` in one file again |
| `cflibs/core/` | Constants, ABCs, config, jax_runtime, cache, pool | 11 files | 1 582 | one (`jax_runtime.py`) | Healthy; only piece that survives untouched in any rearrangement |
| `cflibs/io/` | Spectrum + exporters (CSV/JSON/HDF5) | `exporters.py` (1207), `spectrum.py` | 1 366 | no | `exporters.py` is bloated; format-specific code not split |
| `cflibs/cli/` | argparse CLI subcommands | `main.py` (875) | 885 | no | 875-line monolithic CLI |
| `cflibs/benchmark/` | Unified harness, synthetic corpus, metrics | `unified.py` (3205), `dataset.py` (1298), `loaders.py` (992), `metrics.py` (797), `synthetic.py` (780) | 13 755 | partial | `unified.py` is the second-biggest file in the repo |
| `cflibs/validation/` | Golden spectra + round-trip + NIST parity | `round_trip.py` (862) | 885 | no | Single file holds 5 classes; NIST parity not in this package |
| `cflibs/evolution/` | LLM-driven algorithm optimizer + AST blocklist | 5 files | 447 | (prompts only) | ML-only allowed zone |
| `cflibs/hardware/`, `cflibs/pds/`, `cflibs/hpc/`, `cflibs/visualization/` | DAQ, planetary data, cluster, plot widgets | — | 5 972 | partial | `visualization/widgets.py`=1451 |
| `cflibs/benchmarks/` (deprecated) | Whole-package backward-compat shim | — | tiny | no | Emits `DeprecationWarning`; ripe for removal |
| `native/cflibs-core/` (Rust) | Comb matching, partition fns | 4 `.rs` files | 884 (Rust) | n/a | Dispatched via `cflibs._core` in `cflibs/inversion/identify/line_detection.py:34` |
| `native/rust-plugin/` (Rust) | DAQ plugin interface | 3 `.rs` files | 429 (Rust) | n/a | |

**Totals:** 173 Python files, 67 119 LOC. Inversion is 46 % of the codebase.

### 3.2 JAX surface area today

31 Python files import JAX. Every file repeats the same fallback boilerplate (`try: import jax ... except ImportError: HAS_JAX = False; def jit(f): return f`) — there is **no shared adapter module**, though `cflibs/core/jax_runtime.py` detects the backend.

Key consumers (full table in agent transcript):

- **Forward physics (hot path):** `cflibs/radiation/spectrum_model.py:333-378` (4 jitted helpers), `cflibs/radiation/profiles.py:474-815` (10 jitted, includes `vmap` over lines), `cflibs/radiation/stark.py:224-272`, `cflibs/plasma/saha_boltzmann.py:508-562`, `cflibs/plasma/partition.py:189-589` (with `vmap` over T-axis), `cflibs/instrument/model.py:203-218`, `cflibs/instrument/convolution.py:65` (direct `jnp.convolve`, no FFT).
- **Manifold (primary JAX consumer):** `cflibs/manifold/generator.py:411-764` and `cflibs/manifold/batch_forward.py:225-400` (`jit(vmap(single_spectrum_forward, in_axes=(0,0,0,None,None)))`). `batch_forward.py:23-24` already cites Kawahara et al. (2022) arXiv:2105.14782 as inspiration.
- **Recent JAX ports (commits 926ec47 → 320f143):** `cflibs/inversion/identify/{alias,comb,correlation,line_detection}.py` and `cflibs/inversion/physics/boltzmann_jax.py:110-202` (pytree-registered, opt-in via `CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION`).
- **Solvers:** `cflibs/inversion/solve/iterative.py:976-1145` (`IterativeCFLIBSSolverJax` subclass with `_saha_and_fit_jax` jit), `joint_optimizer.py:574-917` (7 jitted loss/grad/hess closures), `bayesian.py:1071` (JAX forward inside `bayesian_model`).

**No numba/Cython.** Acceleration is JAX (in-process) or Rust (via `cflibs._core` PyO3 import, used only in `line_detection.py:34`).

### 3.3 Forward model path

`SpectrumModel.compute_spectrum()` at `cflibs/radiation/spectrum_model.py:206`:

1. **Saha-Boltzmann** (L222): `SahaBoltzmannSolver(atomic_db).solve_plasma(plasma)` → `Dict[(element, ion_stage, E_k_ev_rounded), n_k_cm3]`.
2. **Transition fetch** (L226-240): per-element SQLite call. `min_relative_intensity` is 0.01 in `NIST_PARITY` mode else 10.0 — hard-coded toggle.
3. **Per-line sigma assembly** (L243-253): three `BroadeningMode` branches (`LEGACY`, `NIST_PARITY`, `PHYSICAL_DOPPLER`).
4. **Emissivity** (L255-257): `calculate_spectrum_emissivity(transitions, populations, wavelength, sigma, use_jax)` (`radiation/emissivity.py:58`). Inside: ε per line = `hc/(4πλ) · A_ki · n_k_m3`. JAX path falls back to NumPy when sigma is per-line (`emissivity.py:123-130`) — known limitation.
5. **Radiative transfer** (L266-273): `I = B(λ,T) · (1 − exp(−κL))`, `κ = ε/B`.
6. **Instrument response** (L276-278): optional element-wise multiply.
7. **Instrument convolution** (L283-301): `scipy.signal.convolve` or `jnp.convolve` — neither uses FFT.

The `SpectrumModelJax` subclass (L410) routes through jitted helpers at L333-378; numerical parity tested at rtol=1e-5.

**Forward-model duplication.** *Three* parallel implementations:
- `SpectrumModel` / `SpectrumModelJax` in `cflibs/radiation/spectrum_model.py:54,410` (canonical).
- `single_spectrum_forward` + `batch_forward_model = jit(vmap(...))` in `cflibs/manifold/batch_forward.py:225-400` (manifold path).
- `BayesianForwardModel.__init__` in `cflibs/inversion/solve/bayesian.py:1071` (NumPyro path; helpers at L423, 454, 484).

### 3.4 Inversion loop anatomy

**`IterativeCFLIBSSolver`** at `cflibs/inversion/solve/iterative.py:119`. Result type `CFLIBSResult` (L48) is a mutable dataclass; **no JAX pytree registration**.

Loop body (`solve()` at L417, ~200 lines):
1. T_eV → evaluate `U_I`, `U_II` per element (`_evaluate_partition_function` L164: direct sum → polynomial → fallback).
2. Effective IPs via `_compute_effective_ips` (IPD optional).
3. `_apply_saha_correction(obs_by_element, T_K, n_e, effective_ips)` — maps ionic lines to neutral plane.
4. `_fit_common_boltzmann_plane(corrected_obs_map)` — pooled common-slope WLS → `_CommonSlopeFit`.
5. T update: `T_new = -1/(slope · KB_EV)`, 50/50 damped with `T_prev`.
6. Optional Hermann corona T = 0.8·T when `two_region`.
7. Abundance multipliers from intercepts.
8. **Closure dispatch** (L527+) over six modes in `cflibs/inversion/physics/closure.py::ClosureMode`: `STANDARD`, `MATRIX`, `OXIDE`, `ILR` (Helmert basis L45, CLR L71, ILR L90, inverse L112), `PWLR` (L202), `DIRICHLET_RESIDUAL`.
9. Update `n_e` via charge/pressure balance.
10. Convergence on `|ΔT| < t_tolerance_k` and `Δne/ne < ne_tolerance_frac`.

**Pure vs effectful split:** Helmert/CLR/ILR/PLR transforms are pure numpy. `BoltzmannPlotFitter.fit_*` is pure. `_evaluate_partition_function` and `_compute_effective_ips` call `atomic_db` (effectful). `solve()` is the only stateful method.

Alternative solvers: `closed_form.py::ClosedFormILRSolver` (L48), `bayesian.py::BayesianForwardModel` + `MCMCSampler` (L1490) + `NestedSampler` (L2065) [3 264 LOC total], `joint_optimizer.py::JointOptimizer` + `MultiStartJointOptimizer` (L755) [L-BFGS-B over packed (T_eV, ne, ILR coords); 949 LOC], `coarse_to_fine.py::HybridInverter` (L232).

### 3.5 Manifold + atomic data stack

**Atomic DB.** `cflibs/atomic/database.py:22` `AtomicDatabase(AtomicDataSource)`. SQLite-backed, connection pool via `cflibs.core.pool.get_pool(path, max_connections=5)` (L51). Schema auto-migrated at startup (`_check_and_migrate_schema` L72; adds `stark_w`, `stark_alpha`, `stark_shift`, `is_resonance`, `aki_uncertainty`, `accuracy_grade`). Hot queries:
- `get_transitions(...)` L326 — `@cached_transitions` decorator.
- `get_ionization_potential(...)` L483 — `@cached_ionization`.
- `get_energy_levels` L445, `get_partition_coefficients` L540, `get_species_physics` L579 — **not decorated**; called per-element per-iteration inside the solver loop via `get_levels_for_species` (`plasma/partition.py:306`).

No prepared statements; raw SQL strings per call. Pool gives concurrent reads; not safe for writes (single-writer SQLite).

**Manifold.** `cflibs/manifold/generator.py:75` `ManifoldGenerator(config: ManifoldConfig)`. Storage auto-inferred (`.zarr` → Zarr; `.h5`/`.hdf5` → HDF5). Rectangular grid over (T_eV, n_e, composition). The `FAISS index` mentioned in CLAUDE.md is **aspirational** — actual implementation is the home-grown `VectorIndex` (`vector_index.py:212`). `BasisLibraryGenerator` (`basis_library.py:81`) builds per-element template libraries used by NNLS.

### 3.6 Validation harness

Single 862-LOC file `cflibs/validation/round_trip.py`:
- `GoldenSpectrum` (L27): ground-truth dataclass.
- `GoldenSpectrumGenerator` (L135): forward Boltzmann assembly.
- `NoiseModel` (L498): Poisson + Gaussian.
- `RoundTripValidator` (L610): forward → noise → invert → compare → `RoundTripResult` with `temperature_error_frac`, `electron_density_error_frac`, `concentration_errors`, `passed`, `tolerances`.

**NIST parity** lives in `scripts/validate_nist_parity.py` and `scripts/run_nist_validation.py` — not in `cflibs/validation/`. No automated round-trip CI marker.

### 3.7 Architectural seams (best 10 injection points)

1. **Forward-model triplet duplication** — `cflibs/radiation/spectrum_model.py:54` ↔ `cflibs/manifold/batch_forward.py:225-400` ↔ `cflibs/inversion/solve/bayesian.py:1071`.
2. **`SpectrumModel` + `SpectrumModelJax` co-located** — same pattern in `plasma/saha_boltzmann.py:50,592`, `plasma/state.py:209,430`, `instrument/model.py:30,248`. Blocks isolated JIT reuse.
3. **`IterativeCFLIBSSolver.solve()` (200-line method)** — `iterative.py:417`. Mixes impure DB calls with pure linear algebra. Pure inner kernel could be lifted into a single jit-able step function inside a `lax.while_loop`.
4. **Closure dispatch via string switch** — `iterative.py:527+` → `closure.py:303` `ClosureEquation`. Protocol/strategy split (like the `SolverStrategy` in `core/abc.py:55`) would let new closures land without touching the loop.
5. **`apply_instrument_function`** — `convolution.py:60,109`. Direct convolution, no FFT path. Hot at every forward call.
6. **Rust hot path** — `cflibs/inversion/identify/line_detection.py:34` (`from cflibs._core import scan_comb_shifts`). Only comb scanning is Rust-dispatched today; `native/cflibs-core/src/partition.rs` exists but isn't called from Python.
7. **`BayesianForwardModel` monolith** — `bayesian.py:1071` (3 264-line file). Splitting `priors / forward / sampler` mirrors Exojax/jaxrts patterns.
8. **`alias.py` monolith** — 2 999 LOC. Mixing of solver, attribution, template build, and `ALIASIdentifier` class (L582).
9. **Atomic DB hot path** — `database.py:326,483,540,579`. Mixed cached/uncached methods; no prepared statements; `get_levels_for_species` called per-element per-iteration without caching.
10. **Per-file JAX-fallback boilerplate × 31 files** — single shared `cflibs.core.jax_runtime`-driven decorator would eliminate ~10 LOC × 31 of duplicate boilerplate.

### 3.8 Technical debt visible from a survey-level read

- **31 backward-compat shim files** at `cflibs/inversion/*.py` flat path. Healthy for back-compat; no scheduled removal.
- **Deprecated `cflibs/benchmarks/__init__.py`** — emits `DeprecationWarning`. Ripe for removal.
- **One explicit `DEPRECATED` Voigt impl** at `cflibs/radiation/profiles.py:616` ("gradient stability issues at high optical depth"); working alternative exists in same file.
- **`LEGACY` broadening mode** at `spectrum_model.py:247-250` (hardcoded `0.01*sqrt(T_eV/0.86)`) — retirement candidate once per-line modes are validated.
- **Zero TODO/FIXME/XXX/HACK comments** in 67 119 LOC. Debt lives in shims, deprecated-marked methods, and the duplicated forward model — not in scattered FIXME tags.
- **Oversized files concentrated in inversion:** `solve/bayesian.py` 3 264, `identify/alias.py` 2 999, `physics/self_absorption.py` 1 809, `runtime/temporal.py` 1 600, `identify/line_detection.py` 1 380, `solve/iterative.py` 1 326, `runtime/streaming.py` 1 311, `physics/uncertainty.py` 1 149, `physics/boltzmann.py` 1 070. Together: 16 508 LOC = 53 % of the inversion package.

---

## 4. Stream B — Radis pattern survey

> Source: Stream B research agent, output captured 2026-05-12.

### 4.1 Architecture at a glance

Radis is a line-by-line molecular RT code (HITRAN/HITEMP/ExoMol/Kurucz) organized around `SpectrumFactory`, which inherits from a 4-deep stack (`DatabankLoader` → `BaseFactory` → `BroadenFactory` → `BandFactory` → `SpectrumFactory`).

- `radis/lbl/` — LBL engine including `factory.py`, `loader.py` (`DatabankLoader`), `broadening.py` (~3 600 lines)
- `radis/spectrum/` — `Spectrum` class
- `radis/levels/partfunc.py` — `RovibPartitionFunction` ABC with TIPS / Dunham / on-the-fly summation subclasses
- `radis/los/slabs.py` — `SerialSlabs` (LOS) and `MergeSlabs` (co-located)
- `radis/db/`, `radis/api/` — HITRAN, HITEMP, ExoMol, Kurucz, NIST, GEISA adapters

**Performance surface:** Numba `@jit(nopython=True, cache=True)` on hot kernels (`_whiting_jit` Voigt); FFT broadening (`broadening_method="fft"`); LDM / DIT line-projection algorithm. No JAX, no Cython. Cache files via pandas HDF5.

### 4.2 Pattern catalogue (12 patterns)

**B-P1. Three-namespace config split.** `SpectrumFactory.__init__` collects ~30 parameters under `self.input` (physics), `self.params` (numerics), `self.misc` (logging). Cheap to clone for batch sweeps: vary `self.input` while keeping cached databases constant. → `cflibs/radiation/spectrum_model.py` and `cflibs/core/config.py`.

**B-P2. Equilibrium/non-equilibrium as sibling methods on the factory.** `eq_spectrum(Tgas, ...)` vs `non_eq_spectrum(Tgas, Tvib, Trot, ...)`. Shared pipeline reused; only line-strength differs. → relevant if CF-LIBS adds non-LTE later; avoid premature subclass hierarchy now.

**B-P3. LDM/DIT broadening — project N lines onto a log-spaced (γ_L, γ_G) grid.** van den Bekerom & Pannier 2021 (JQSRT, doi:10.1016/j.jqsrt.2020.107476). `BroadenFactory._calc_lineshape_LDM`: bilinearly distribute each line's linestrength across two adjacent grid points in (log γ_L, log γ_G) space (grid built by `_init_w_axis(w_dat, log_p)`). Steps `params.dxL`, `params.dxG` default to ~0.2 in log. Millions of lines collapse to dozens of FFT convolutions. → `cflibs/radiation/profiles.py` and `cflibs/manifold/`. CF-LIBS' Gaussian-dominant broadening (Doppler + instrument) is simpler than Voigt; a 1-D log-σ grid likely suffices. **Highest-impact Radis pattern.**

**B-P4. Numba `@jit` on micro-kernels only.** Radis JITs only the inner Whiting-Voigt approximation; the orchestrator stays in pure Python. Inline comment claims 8.9 s → 5.1 s on 50 k lines. → CF-LIBS' analog: JAX `jit` on inner kernels only (already partially done).

**B-P5. Two-stage line DataFrame (`df0` raw / `df1` T-scaled).** `DatabankLoader.init_databank()` registers parameters without loading; `_check_line_databank()` triggers I/O at first call. `df0` is immutable raw; `df1` is T-scaled. Constant metadata stored as DataFrame `.attrs`. → `cflibs/atomic/database.py` and `cflibs/inversion/physics/boltzmann.py`. Eliminates redundant SQLite hits inside `IterativeCFLIBSSolver.solve()`.

**B-P6. Linestrength cutoff + neighbour-line padding.** `cutoff=0, truncation=Default(50), neighbour_lines=0`. Cutoff recomputed per-temperature inside the solver. → `cflibs/radiation/spectrum_model.py` and `cflibs/inversion/identify/`.

**B-P7. `RovibPartitionFunction` ABC.** Two-branch hierarchy: `Tabulator` (TIPS / Kurucz / NIST / Barklem-2016) vs `Calculator` (on-the-fly `Q = Σ g · exp(-hcE/kT)`). Plugin contract: subclasses implement `at(T, **kwargs)` and `at_noneq(...)`. → `cflibs/plasma/partition.py` and `cflibs/atomic/database.py`. Lets validation cross-check polynomial fits vs summation.

**B-P8. `Spectrum` as a quantity-bag.** `quantities` dict `{name: (wavespace, array)}`, `units`, `conditions`, `cond_units`. Multiple quantities (radiance/absorbance/transmittance/emissivity) coexist. Slit stored as `_slit` attribute. → `cflibs/io/` + `cflibs/radiation/spectrum_model.py`.

**B-P9. Slab combinators `SerialSlabs` / `MergeSlabs` for RTE composition.** `SerialSlabs`: `I_{1→2} = I_1·τ_2 + I_2`, `τ_{1+2} = τ_1·τ_2`. `MergeSlabs`: sum `j_λ` and `k_λ`. → CF-LIBS is *ahead* on self-absorption (CDSB inverts τ from doublet ratios rather than computing forward), but the compositional API would help if CF-LIBS ever moves beyond `SingleZoneLTEPlasma`.

**B-P10. Non-regression golden spectra with date-stamped reference files and failed-run artifact dump.** Pytest auto-saves `test_factory_failed_{version}.spec` next to the golden on assertion failure. → `cflibs/validation/`. Borrowable refinement: auto-save inspectable artifacts (not just print assertion text).

**B-P11. `optimization` enum on the factory.** `"simple" | "min-RMS" | None` controls LDM and auto-`dxL`/`dxG`. Enum on the orchestrator, not subclass hierarchy. → cleaner than feature flags.

**B-P12. `calc_spectrum()` convenience wrapper.** 5-line user-facing call hides the factory; power users drop down to `SpectrumFactory` for batch loops. Two-tier API. → CF-LIBS could expose `cflibs.analyze(spectrum, elements=...)` returning a `CFLIBSResult`.

### 4.3 Anti-patterns (do NOT copy)

- 4-level inheritance chain (`DatabankLoader → BaseFactory → BroadenFactory → BandFactory → SpectrumFactory`). Hard to debug.
- CSV cache files alongside production HDF5.
- `pseudo_continuum_threshold` machinery (irrelevant for sparse atomic lines).
- Vaex dependency (unmaintained).
- `Default(...)` sentinel objects for parameter defaults.
- Opaque `df0`/`df1` naming — borrow the pattern, rename to `lines_raw`/`lines_scaled`.
- Dual `.spec` JSON + HDF5 serialization formats.

### 4.4 Quantified wins where Radis has published numbers

- **LDM/DIT (van den Bekerom & Pannier 2021):** ">100× speedup for HITEMP-scale CO₂/H₂O spectra" vs naive LBL; "lineshape convolution almost instantaneous" once LDM is on.
- **Numba `_whiting_jit`:** 8.9 s → 5.1 s on 50 k lines (~1.75×) on that micro-kernel.

### 4.5 Top 5 Radis candidates for CF-LIBS adoption

| Rank | Pattern | Target | Impact | Effort |
|---|---|---|---|---|
| 1 | LDM/DIT broadening grid for manifold pre-compute (B-P3) | `cflibs/manifold/generator.py`, `cflibs/radiation/profiles.py` | very high | medium |
| 2 | Two-stage line DataFrame `lines_raw`/`lines_scaled` (B-P5) | `cflibs/inversion/physics/boltzmann.py`, `cflibs/atomic/database.py` | high | low |
| 3 | `PartitionFunctionSource` ABC (polynomial + summation backends) (B-P7) | `cflibs/plasma/partition.py`, `cflibs/atomic/database.py` | medium-high | low-medium |
| 4 | `input`/`params`/`misc` config namespacing (B-P1) | `cflibs/core/config.py`, `cflibs/radiation/spectrum_model.py` | medium | low |
| 5 | Auto-save failed golden-spectrum artifacts as `.h5` (B-P10) | `cflibs/validation/`, `tests/conftest.py` | medium | very low |

---

## 5. Stream C — JAX physics-code pattern survey (jaxrts + Exojax)

> Source: Stream C research agent, output captured 2026-05-12.

### 5.1 jaxrts patterns (Lütgert et al. 2026, Comput. Phys. Commun. 110173)

jaxrts is the closest analogue to CF-LIBS in this survey: a plasma physics code with a central `PlasmaState`, per-process `Model` plug-ins, a single `probe()` orchestrator, and an iterative ionization solver with the same fixed-point shape as CF-LIBS' Saha-Boltzmann loop.

**C-P1. Manual `register_pytree_node` on the central state class.** `PlasmaState` is a regular Python class with `_children_labels` / `_aux_labels` and hand-written `_tree_flatten` / `_tree_unflatten` registered via `jax.tree_util.register_pytree_node` (`src/jaxrts/plasmastate.py:510-583`). Children = dynamic arrays (`Z_free`, `mass_density`, `T_e`, `T_i`, `ion_core_radius`, `models`); aux = static `ions` tuple. Same trick for `Setup` (`setup.py:291`) and every `Model` (`models.py:143`). **No `equinox.Module`, no `flax.struct.dataclass`** — compatible with CF-LIBS' physics-only ban. → `cflibs/plasma/state.py::SingleZoneLTEPlasma`: eliminates jit recompiles when composition vector length changes.

**C-P2. Method-level `@jax.jit`, not orchestrator-level.** `Model.evaluate` carries `@jax.jit` at each leaf (`models.py:235, 306, 365, 430, 489, 542, ...`). Top-level `probe()` is jitted but only dispatches. Trace tree is per-physical-process; swapping which models are attached does not invalidate other models' caches. → `cflibs/radiation/spectrum_model.py::SpectrumModel.compute_spectrum`: jit per emissivity kernel, not whole forward model.

**C-P3. `jax.lax.while_loop` for fixed-point solvers.** `solve_ionization` (`src/jaxrts/ionization.py:417-512`) reduces charge balance to 1-D residual `f(log_ne)` and calls a custom `bisection(func, a, b, ...)` (`helpers.py:331-385`) that is just `lax.while_loop` over `(low, high, count)`. **No `lax.custom_root`, no `jaxopt`** — bare lax primitive, traces cleanly under jit/vmap/grad. `calculate_mean_free_charge_BU` Picard iteration (`ionization.py:1001-1025`) uses the same pattern. → THE pattern for CF-LIBS' `cflibs/inversion/solve/iterative.py` outer Saha-Boltzmann iteration (see § 5.4).

**C-P4. Nested `jax.vmap` over independent physics axes.** `_3Dfour_sine = jax.vmap(jax.vmap(fourier_transform_sine, in_axes=(None, None, 0)), ...)` (`hypernetted_chain.py:334`), `hnc_interp = jax.vmap(jax.vmap(jnpu.interp, in_axes=(None, None, 0)))` (`hypernetted_chain.py:709`), `free_free.py:61` `vmap(_KKT_single_point, in_axes=(0, None, None, None))`. Pattern: write the one-point kernel, vmap over the loop axis. → already in use in `cflibs/inversion/identify/`; audit for missed batch dims.

**C-P5. `jax.tree_util.Partial` for callables as pytree leaves.** `Setup.__init__` does `self.instrument = jax.tree_util.Partial(instrument)` (`setup.py:44`). Instrument function travels through jit boundaries as a pytree leaf — no closures, no static-arg hashing. → `cflibs/instrument/model.py`.

**C-P6. Persistence via pytree flatten round-trip.** `src/jaxrts/saving.py` serializes `PlasmaState` by re-using flatten/unflatten + `JaXRTSEncoder` (lines 60-100). One serialization path covers jit-tracing and disk persistence. → `cflibs/manifold/storage.py` and `cflibs/io/spectrum.py` could share one pytree-flatten contract.

### 5.2 Exojax patterns (Kawahara et al. 2022 ApJS 258 31; ExoJAX2 2025 ApJ 985 263, arXiv:2410.06900)

Larger code than jaxrts; retrofitted into a layered architecture for ExoJAX2.

**C-P7. Layered package: orchestrator (`rt/`) vs opacity engine (`opacity/`) vs differentiable special functions (`special/`).** Dependency direction: `rt/emis.py → opacity/premodit/api.py → opacity/premodit/core.py → opacity/lpf/lpf.py → special/faddeeva.py`. High-level classes (`ArtEmisPure`, `OpaPremodit`) in `rt/` and `opacity/`. `rt/__init__.py` uses lazy module loading via `__getattr__` to keep import cheap. Pure kernels (Voigt-Hjerting `hjert`, Faddeeva `rewofzx`) are pure JAX functions with custom JVP/VJP. → `cflibs/radiation/api.py` (orchestrator) + `cflibs/radiation/profiles.py` (pure-kernel layer never seeing `PlasmaState`). **Convergent with HELIOS (§ 6.2.1) and stream D's host/kernel split recommendation.**

**C-P8. PreMODIT: precomputed line-strength buffers + chunked `lax.scan` for memory control.** `OpaPremodit.xsmatrix` (`opacity/premodit/api.py:771-855`) precomputes `lbd_coeff` of shape `(diffmode+1, N_broadening_grid, N_elower_grid)` so T-dependent line strength becomes a small interpolation, not an `N_lines × N_layer × N_nu` reduction. When ν-grid is too long, sets `nstitch > 1` and `lax.scan` over reshaped `lbd_coeff_reshaped` chunks of `div_length` wavenumbers (lines 800-820). Body wrapped with `@partial(checkpoint, policy=checkpoint_policies.dots_with_no_batch_dims_saveable)` for gradient checkpointing. Output recombined via overlap-and-add (`exojax.signal.ola.overlap_and_add_matrix`). **Single most-citable memory pattern in this survey.** → `cflibs/manifold/generator.py` and `cflibs/radiation/spectrum_model.py`. Directly attacks CF-LIBS' "large compositions blow GPU memory" pain.

**C-P9. `MemoryPolicy` frozen-dataclass.** `opacity/policies.py` `@dataclass(frozen=True) class MemoryPolicy` with `allow_32bit`, `nstitch`, `cutwing`. Passed to `OpaPremodit(... memory_policy=...)`. → `cflibs/core/jax_policy.py::JaxMemoryPolicy` centralizing `JAX_PLATFORMS=cpu` / Apple-Silicon-no-fp64 / manifold chunk-size knobs.

**C-P10. Data-only `MDBSnapshot` frozen dataclass decouples opacity from database classes.** `database/contracts.py` defines `MDBMeta`, `Lines`, `MDBSnapshot`. `PreMODITInfo` (`opacity/premodit/info.py`) also frozen with `as_tuple()` shim for legacy code. Opacity calculator constructed from snapshot or live `mdb` object — same interface, different provider. → CF-LIBS' `AtomicDatabase` is SQLite-backed; an `AtomicSnapshot` frozen dataclass lets JAX-jitted code consume a plain array bundle without holding the SQLite connection inside the trace.

**C-P11. Provider contracts as plain protocols.** `opacity/contracts.py` declares small interface classes (`PartitionFunctionProvider`, `BroadeningStrategy`); concrete implementations in `opacity/providers.py` (`ExomolPartitionProvider`, `HitranPartitionProvider`, `ExomolBroadening`, `HitranBroadening`). `OpaPremodit.__init__` resolves them based on `mdb.dbtype` (`api.py:144-160`). → CF-LIBS analogue lets NIST / Cowan / Kurucz partition-function backends swap without touching `boltzmann.py`. **Convergent with Radis B-P7.**

**C-P12. `@custom_jvp` / `@custom_vjp` on piecewise/special functions.** Faddeeva `rewofzx` (`special/faddeeva.py:151`) uses `@custom_vjp` with `defvjp(h_fwd, h_bwd)` deriving ∂Re/∂x, ∂Re/∂y from Cauchy-Riemann relations. Voigt-Hjerting `hjert` (`opacity/lpf/lpf.py:225`) uses `@custom_jvp` with the recurrences `∂H/∂x = 2aL − 2xH`, `∂H/∂a = 2xL + 2aH − 2/√π`. `convolve_rigid_rotation` (`postproc/spin_rotation.py:65,157`) and `chord` integration (`rt/chord.py:5`) similarly carry `@custom_jvp`. Pattern: where the forward primitive uses a `jnp.where`-piecewise approximation, hand-written JVP avoids the cutoff discontinuity that AD would otherwise see. → `cflibs/inversion/physics/stark.py` and any `jnp.where`-gated profile cutoff in `cflibs/radiation/profiles.py`. **Required if anyone plans to differentiate the forward model** (joint L-BFGS-B in `joint_optimizer.py`, HMC in `bayesian.py`).

**C-P13. `lax.scan` over physical strata.** `rtrun_emis_pureabs_ibased` (`rt/rtransfer.py:100-135`) cumsums `dtau` then scans over `(mus, weights)` accumulating flux into a `(Nnus,)` carry. `fluxsum_scan` (`rt/rtlayer.py:34`) same pattern for per-layer-pair stream sum. Replaces Python `for` *inside* a jit; differentiable for free. → CF-LIBS' iterative solver outer loop (§ 5.4).

**C-P14. NumPyro coupling pattern (`reverse_premodit.py`).** `tests/endtoend/reverse/reverse_premodit.py:130-160` wires pure JAX `frun(Tarr, MMR, Mp, Rp, u1, u2, RV, vsini)` into NumPyro `model_c(y1)`: each free parameter is one `numpyro.sample`, forward invoked once, likelihood `numpyro.sample("y1", dist.Normal(mu, sigmain), obs=y1)`. **Forward function is outside the NumPyro graph.** `NUTS(model_c, forward_mode_differentiation=False)`. → CF-LIBS' `bayesian.py::BayesianForwardModel` already does this; the audit is that exojax keeps forward 100 % pure JAX (no class state mutated in `model_c`).

**C-P15. `check_jax64bit` runtime guard.** `utils/jaxstatus.py` raises `ValueError` if `config.jax_enable_x64` is False unless `allow_32bit=True`. → CF-LIBS' jax-metal has no fp64; mirror the guard in `cflibs/core/jax_runtime.py` with explicit `allow_fp32_on_metal` switch + loud warning.

**C-P16. Tutorial gallery as `.ipynb + .rst` paired files.** `documents/tutorials/` (`Cross_Section_using_OpaStitch.ipynb`, `Differentiable_Programming.ipynb`, `Fitting_Telluric_Lines.ipynb`), each twin-paired with `.rst` for Sphinx. Layering: `documents/userguide/` (concept) → `documents/tutorials/` (worked examples) → API autodoc. → `docs/`. Tutorials double as integration smoke-tests.

### 5.3 Generic JAX patterns surfaced by both repos

| Pattern | jaxrts site | exojax site | CF-LIBS target |
|---|---|---|---|
| `lax.while_loop` for fixed-point | `helpers.py::bisection:331`; `ionization.py:1025` | not used (RT non-iterative) | `cflibs/inversion/solve/iterative.py` |
| `lax.scan` over physics axes | not used | `rt/rtransfer.py:rtrun_emis_pureabs_ibased`; `opacity/premodit/api.py:xsmatrix` chunk scan | `iterative.py` outer loop; `manifold/generator.py` chunked grid scan |
| Nested `vmap` for independent axes | `hypernetted_chain.py:334,709`; `free_free.py:61` | `lpf.py`, `rt/emis.py` (per stream × per layer) | already in `inversion/identify/`; audit |
| `jax.checkpoint`/`remat` with explicit policy | none | `opacity/premodit/api.py:800` `checkpoint_policies.dots_with_no_batch_dims_saveable` | `manifold/generator.py` to reduce activation memory |
| Manual `register_pytree_node` on state | `plasmastate.py:578`, `setup.py:291`, `models.py:143×` | not used; uses frozen dataclasses (`MemoryPolicy`, `MDBSnapshot`, `PreMODITInfo`) | both styles work; CF-LIBS' equinox ban favors manual register |
| `@custom_jvp` / `@custom_vjp` for piecewise/special | none (smooth integrals) | `special/faddeeva.py:rewofzx`, `lpf.py:hjert`, `postproc/spin_rotation.py`, `rt/chord.py` | `inversion/physics/stark.py` Voigt cutoff; `radiation/profiles.py` |
| `jax.tree_util.Partial` for callable-as-leaf | `setup.py:44` | not used | `cflibs/instrument/model.py` |
| `lax.custom_root` / implicit_diff | **not used** | **not used** | open opportunity in CF-LIBS § 5.4 |
| AOT `jit.lower().compile()` | none | none | open in `manifold/generator.py` |

**Notable miss in both repos:** `lax.custom_root` / `jaxopt.implicit_diff.custom_fixed_point` for implicit gradients through the iterative solver. CF-LIBS adopting this would be genuinely novel research (not a crib).

### 5.4 Specific opportunities for CF-LIBS' iterative solver

CF-LIBS' `cflibs/inversion/solve/iterative.py` Saha-Boltzmann loop is structurally `T_{n+1}, n_{e,n+1} = step(T_n, n_{e,n}; data)` until `|ΔT| < tol`. Three escalating retrofits:

1. **Wrap the current Python `while` in `lax.while_loop`** (jaxrts pattern, `ionization.py::calculate_mean_free_charge_BU:1001-1025`). State tuple `(T, n_e, residual, i)`. Effort: ~1 day. Wins: stays in jit; vmap-able across multiple spectra; ~10× faster on GPU.
2. **Replace inner `n_e` solve with `lax.while_loop`-based bisection on `log n_e`** (jaxrts `helpers.py::bisection` + `ionization.py::solve_ionization:466-485`). Effort: ~1 day. Trivially differentiable wrt T and composition; replaces scipy.optimize.brentq.
3. **Lift outer loop to `lax.custom_root`** — neither jaxrts nor exojax does this; CF-LIBS is the right shape (iterative solver feeds joint-mode L-BFGS-B and NumPyro). Implicit-function-theorem gradients through converged `(T*, n_e*)` shrink backward-pass memory roughly proportional to `max_iter`. Effort: ~3-5 days. Genuinely novel.

### 5.5 Memory + GPU efficiency patterns

- **PreMODIT-style precomputed buffers** (C-P8) → `cflibs/manifold/` extension: `(N_T_grid × N_ne_grid × N_wavelength)` emissivities per species at config time; runtime is a 3-D lookup. Missing piece: broadening buffer (Stark width vs n_e is what blows up).
- **Wavenumber stitching with overlap-and-add** (`exojax.signal.ola.overlap_and_add_matrix`, used when `nstitch > 1`). → CF-LIBS' wide-spectrum cases (220-900 nm @ R=30 000).
- **Explicit gradient checkpointing with selective policy** (`@partial(checkpoint, policy=checkpoint_policies.dots_with_no_batch_dims_saveable)`) — saves matmul outputs, resaves elementwise, halves activation memory. → `cflibs/inversion/solve/bayesian.py` HMC backward.
- **`delete_mdb_after_init=True`** (api.py:76,162) → drop SQLite query results after manifold is built.
- **Eager 64-bit guard** → `cflibs/core/jax_runtime.py` with explicit "metal = 32-bit, warn loudly" contract.

### 5.6 Top 5 JAX-codes candidates for CF-LIBS adoption

| Rank | Pattern | Source | Target | Impact | Effort |
|---|---|---|---|---|---|
| 1 | `lax.while_loop` for outer Saha-Boltzmann iterate | jaxrts `ionization.py:1001-1025` | `cflibs/inversion/solve/iterative.py` | High | Low (~1 day) |
| 2 | Chunked `lax.scan` + `checkpoint` over ν-grid | exojax `opacity/premodit/api.py:xsmatrix:800-820` | `cflibs/manifold/generator.py`, `cflibs/radiation/spectrum_model.py` | High | Medium (~3 days) |
| 3 | `@custom_jvp` on piecewise profile cutoffs | exojax `lpf.py:hjert:225-265`, `faddeeva.py:rewofzx:151` | `cflibs/inversion/physics/stark.py`, `cflibs/radiation/profiles.py` | Medium-High | Low-Medium |
| 4 | Frozen-dataclass snapshots (`MDBSnapshot`, `MemoryPolicy`, `PreMODITInfo`) | exojax `database/contracts.py`, `opacity/policies.py` | `cflibs/atomic/database.py` adds `AtomicSnapshot`; new `cflibs/core/jax_policy.py` | Medium | Low (~1 day) |
| 5 | Manual `register_pytree_node` on `SingleZoneLTEPlasma` + `InstrumentModel` | jaxrts `plasmastate.py:510-583`, `setup.py:291` | `cflibs/plasma/state.py`, `cflibs/instrument/model.py` | Medium | Low (~1 day each) |

---

## 6. Stream D — Atmospheric RT and other spectroscopy codes (cross-pollination)

> Source: Stream D research agent, output captured 2026-05-12. Most distant stream — fewest applicable patterns but explicit non-import list is valuable.

### 6.1 PetitRADTRANS (Mollière et al. 2019, A&A 627 A67)

**Patterns that DO transfer:**

**D-P1. `Retrieval` + `RetrievalConfig` + `Data` + `Parameter` decomposition** (`petitRADTRANS/retrieval/`). `RetrievalConfig` aggregates `Parameter` objects (each carrying `transform_prior_cube_coordinate`); `Data` objects encapsulate spectra per observation; model is a plain callable `model_generating_function(prt_object, params_dict) → (wavelengths, flux)`. `Retrieval` owns only sampler boilerplate (pyMultiNest/UltraNest, MPI, `get_samples()`, `plot_corner()`). → `cflibs/inversion/solve/bayesian.py` — pull sampler wiring out of `BayesianForwardModel`; lift parameters into a `Parameter` dataclass `(name, prior_low, prior_high, vary, expr)` consumable by NumPyro **and** nested samplers. **Convergent with exojax C-P14.**

**D-P2. Library of named model templates** (`petitRADTRANS/retrieval/models.py`: `retrieval_model_spec_iso`, `emission_model_diseq`, ...). Each is a stand-alone function picked by name. Avoids the mega-`forward()` with 20 boolean flags. → `cflibs/inversion/forward_models/` registry with `single_zone_lte`, `hermann_two_region`, `lte_with_self_absorption`, each a pure callable.

**D-P3. HDF5 opacity layout convention** (`input_data/opacities/lines/correlated_k/<species>/<isotope>/<file>.h5`, filenames like `12C-1H4__YT34to10.R1000_0.3-50mu.ktable.petitRADTRANS.h5`). Path encodes species, line list, resolving power, range, table type — discoverable without an index. → `output/model_library/<element>/<charge>/R<resolving_power>_<wl_min>-<wl_max>.h5`. Low priority.

**Patterns that DO NOT transfer:** `pressures` array, `mass_fractions` dict keyed by isotopologues, `reference_gravity`/`reference_pressure`/`mean_molar_masses`, `calculate_transit_radii`, correlated-k (`ktable`) machinery — all reflect layered atmospheric RT which CF-LIBS' single-zone LTE plasma is not.

### 6.2 HELIOS (Malik et al. 2017 AJ 153 56; Malik et al. 2019 AJ 157 170)

**Patterns that DO transfer:**

**D-P4. Two-tier file split: `host_functions.py` (CPU prep) + `computation.py` (kernel dispatch) + `kernels.cu` (GPU).** `host_functions.py` does `planet_param()`, `initial_temp()`, `convective_adjustment()`, `interpolate_vmr_to_opacity_grid()`. `computation.py` dispatches kernels with 1:1 Python wrappers handling allocation + `pycuda` launch. → CF-LIBS' JAX-port effort is currently growing `_jax.py` siblings next to NumPy implementations. HELIOS split suggests `cflibs/<package>/host.py` (orchestration, validation, grid setup) + `cflibs/<package>/kernels.py` (pure jitted functions, no I/O). **Convergent with exojax C-P7; THE most-cited recommendation in this survey.**

**D-P5. Two-loop iteration architecture: `radiation_loop` + `convection_loop` with explicit per-layer convergence flag (`abortsum < quant.nlayer + 1`, `rad_convergence_limit`).** Terminates when every layer has converged. → `IterativeCFLIBSSolver` is currently single-loop. HELIOS pattern suggests splitting into outer composition/T loop and inner closure-equation adjustment with their own convergence flags, plus a structured `ConvergenceState` carrying per-element residuals. Medium priority — current solver works, but logging "which sub-loop diverged" would help debugging real-data failures.

**D-P6. `param.dat` declarative configuration as single source of truth.** CF-LIBS already does this via YAML in `cflibs/core/config.py`. No change.

**Patterns that DO NOT transfer:** PyCUDA + raw `.cu` (we target JAX/XLA), atmospheric interpolation kernels (`opac_interpol`, `plancktable`, `temp_inter`, `cp_interpol`, `entropy_interpol`), hemispheric two-stream + diffusivity factor of 2, convective adjustment, k-distribution tables.

### 6.3 Adjacent codes briefly surveyed

- **Sherpa (CIAO X-ray)**: `Data`/`Model`/`Stat`/`Fit`/`Optimizer` quartet with additive/multiplicative model composition (`model1 + model2 * model3`). Pluggable optimizers (`LevMar`, `NelderMead`, `MonCar`). Cleanest example of orthogonal fitter decomposition. **Verdict:** read for ideas, do not refactor to Sherpa's structure — cost is moving away from physics-named entry points.
- **LMFIT**: `Parameter` with `min`/`max`/`vary`/`expr` (algebraic constraints between parameters). `expr` is genuinely useful — `params['C_Fe'].expr = '1 - C_Cr - C_Ni'` enforces closure symbolically. **Verdict:** low priority; the `Parameter` dataclass shape is already what the petitRADTRANS-inspired retrieval refactor (D-P1) needs.
- **iSpec / TurboSpectrum / MOOG (stellar)**: closer physically to LIBS but Python-wraps-Fortran with no architectural patterns worth importing beyond what CF-LIBS already has. **Verdict:** revisit only if adding VALD as a second `AtomicDataSource` backend.
- **LIBS-specific (LIBSsa, OpenLIBS)**: covered in `docs/REFERENCE_ANALYSIS_LIBSSA.md` (2026-01-07). No new material.

### 6.4 Inapplicability summary (explicit non-imports)

CF-LIBS plasma is **single-zone LTE**. Layered atmospheric RT is **not** an analog. Do NOT import:

1. Layered loops over `pressures`/`temperatures` arrays with optical-depth integration — `_calc_opa_spectrum_lineflux`-style transmission integrals and HELIOS' `populate_spectral_flux_iteratively` are inappropriate for `cflibs/radiation/spectrum_model.py`.
2. Two-stream approximation, diffusivity factor of 2, hemispheric flux.
3. k-distribution / correlated-k opacity tables — solve a line-density × layer-count problem CF-LIBS does not have.
4. `mean_molar_masses`, `reference_gravity`, `reference_pressure` — atmospheric thermodynamic state with no LIBS analog.
5. Mass-fraction-only composition vocabulary — CF-LIBS needs fluency in mass / number / number-density (already implemented).
6. Convective adjustment.
7. PyCUDA / raw `.cu` kernels — JAX is the target.
8. Transit/transmission geometry.

### 6.5 Top atmospheric/cross-pollination candidates

| Rank | Pattern | Strength |
|---|---|---|
| 1 | petitRADTRANS' `RetrievalConfig`/`Parameter`/`Data`/named-model decomposition (D-P1) | Strong |
| 2 | Named-model registry in `cflibs/inversion/forward_models/` (D-P2) | Strong |
| 3 | HELIOS `host.py`/`kernels.py` split for JAX port (D-P4) | Strong (convergent with exojax C-P7) |
| 4 | Outer/inner loop split in `IterativeCFLIBSSolver` with `ConvergenceState` (D-P5) | Medium |
| 5 | HDF5 path-as-index convention for `output/model_library/` (D-P3) | Low (convention-level) |

**Honest summary:** Items 1, 2, 3 are real, defensible. Item 4 is "improve our solver" rather than "copy theirs". Item 5 is convention-level. None of the *physics* of layered radiative transfer transfers — this stream's value is in *reinforcing convergent patterns* surfaced by streams B and C.

---

## 7. Synthesis — convergent themes across streams

When four agents working in isolation surface the same idea, that pattern is unusually well-validated. Five themes converged:

### 7.1 Host / kernel file split

**Convergent on:** exojax `rt/` vs `opacity/lpf/` (C-P7), HELIOS `host_functions.py` vs `computation.py` vs `kernels.cu` (D-P4), Radis' implicit "factory orchestrator + numba-jitted micro-kernel" pattern (B-P4), jaxrts' method-level jit discipline (C-P2). Stream A flagged the *anti*-pattern in CF-LIBS: 31 files with duplicated `try: import jax` boilerplate, plus `SpectrumModel`/`SpectrumModelJax` co-located in the same file in five places.

**Implication for CF-LIBS:** any package undergoing JAX migration should be reorganized so a `host.py` carries Python orchestration, validation, dtype handling, error messages, and SQLite I/O; `kernels.py` carries pure `@jit`-able functions with no Python control flow on dynamic shapes and no I/O. Retire the `*Jax`-subclass-in-same-file pattern. Eliminate the 31-file fallback boilerplate via a shared `cflibs.core.jax_runtime`-driven decorator.

### 7.2 Retrieval / inversion driver decomposition

**Convergent on:** petitRADTRANS `RetrievalConfig` + `Parameter` + `Data` + `model_callable` + `Sampler-adapter` (D-P1), exojax NumPyro coupling where the forward function is *outside* the NumPyro graph and only `numpyro.sample` lives inside it (C-P14), LMFIT's `Parameter`+`expr` model (§ 6.3), Sherpa's `Data`/`Model`/`Stat`/`Fit`/`Optimizer` quartet.

**Implication for CF-LIBS:** `cflibs/inversion/solve/bayesian.py` (3 264 LOC) should be split into `priors.py` (Parameter dataclass + prior transforms), `forward.py` (pure JAX `(plasma_state, line_set, instrument) → spectrum`), and `samplers.py` (NumPyro NUTS + dynesty adapter). This decomposition also lets named forward models (single_zone_lte, hermann_two_region, lte_with_self_absorption) be swapped without touching the sampler.

### 7.3 Frozen-dataclass snapshots that decouple JAX from data sources

**Convergent on:** exojax `MDBSnapshot`, `PreMODITInfo`, `MemoryPolicy` (C-P9, C-P10), Radis `df0`/`df1` two-stage DataFrame (B-P5), jaxrts manual pytree registration on `PlasmaState` (C-P1).

**Implication for CF-LIBS:** introduce an `AtomicSnapshot` frozen dataclass exposing arrays from `AtomicDatabase` without holding the SQLite connection inside jit'd code. Introduce a `JaxMemoryPolicy` frozen dataclass centralizing the platform/precision/chunk-size knobs scattered across CLI flags + config. Register `SingleZoneLTEPlasma` and `InstrumentModel` as JAX pytrees to eliminate recompile-on-element-list-change.

### 7.4 Fixed-point solver as `lax.while_loop` (and the unrealized `custom_root` opportunity)

**Convergent on:** jaxrts `solve_ionization` + `calculate_mean_free_charge_BU` (C-P3), exojax `lax.scan` over RT strata (C-P13), HELIOS outer/inner convergence loops (D-P5). **Neither jaxrts nor exojax uses `lax.custom_root`** — they hand-roll while_loops.

**Implication for CF-LIBS:** wrap `IterativeCFLIBSSolver.solve()` in `lax.while_loop` (low effort, big win). Beyond that, `lax.custom_root` (or `jaxopt.FixedPointIteration`) is a *genuinely novel* opportunity — CF-LIBS feeds the converged `(T*, n_e*)` into joint L-BFGS-B and NumPyro HMC, so implicit-function-theorem gradients would shrink the backward-pass memory roughly proportional to `max_iter`. Both surveyed JAX codes left this on the table.

### 7.5 Differentiable physics requires custom JVP/VJP on piecewise approximations

**Convergent on:** exojax `@custom_jvp` on `hjert` Voigt-Hjerting and `@custom_vjp` on `rewofzx` Faddeeva (C-P12), plus `convolve_rigid_rotation` and `chord` integration. Stream A noted that `cflibs/inversion/physics/stark.py` and `cflibs/radiation/profiles.py` have `jnp.where`-gated cutoffs that AD currently traces naively.

**Implication for CF-LIBS:** if anyone plans to backpropagate through the forward model — either joint L-BFGS-B in `joint_optimizer.py` or HMC in `bayesian.py` — declaring `@custom_jvp` on Voigt cutoffs and Stark profiles is required for gradient correctness. Derivatives are textbook (Voigt-Hjerting recurrences, Cauchy-Riemann for Faddeeva).

---

## 8. Ranked candidate catalogue (cross-stream master list)

Twenty-one candidates from streams B/C/D, deduplicated where streams converged. Effort/impact ratings are agent-estimated; final calibration will happen during per-pattern implementation ADRs.

### 8.1 Tier 1 — High impact, strong convergent backing

| # | Pattern | Source streams | Target | Impact | Effort | Notes |
|---|---|---|---|---|---|---|
| **T1-1** | `host.py` / `kernels.py` split; retire `*Jax`-in-same-file; shared JAX fallback decorator | C-P7, D-P4, A.7#10 | `cflibs/radiation/`, `cflibs/plasma/`, `cflibs/instrument/`, `cflibs/inversion/identify/`, `cflibs/inversion/physics/` | High (enables every later JAX change) | Medium (~5 days) | Convergent across three streams. Recommended in § 2 as a structural prerequisite. |
| **T1-2** | Unify the three forward-model implementations into one pure-JAX kernel | A.7#1, C-P7 | `cflibs/radiation/spectrum_model.py` ↔ `cflibs/manifold/batch_forward.py` ↔ `cflibs/inversion/solve/bayesian.py` | High | Medium-High | Already cited in `batch_forward.py:23-24` as the Kawahara/ExoJAX pattern. |
| **T1-3** | `lax.while_loop` for outer Saha-Boltzmann iterate | C-P3 (jaxrts) | `cflibs/inversion/solve/iterative.py` | High | Low (~1 day) | Direct translation; preserves current numerics; vmap-able across spectra. |
| **T1-4** | LDM/DIT broadening grid for manifold pre-compute | B-P3 (Radis) | `cflibs/manifold/generator.py`, `cflibs/radiation/profiles.py` | Very high (manifold throughput) | Medium | >100× speedup published for Voigt-heavy spectra; CF-LIBS' Gaussian-dominant broadening is simpler, so 1-D log-σ grid likely suffices. |
| **T1-5** | Chunked `lax.scan` + `checkpoint` over wavelength grid | C-P8 (exojax PreMODIT) | `cflibs/manifold/generator.py`, `cflibs/radiation/spectrum_model.py` | High (eliminates GPU OOM on long ν-grids) | Medium (~3 days; needs OLA implementation) | Solves stated "compositions blow GPU memory" pain. |
| **T1-6** | Retrieval/inversion driver decomposition (`Parameter`+`Data`+`Sampler`-adapter; named forward models) | D-P1, D-P2, C-P14 | `cflibs/inversion/solve/bayesian.py` (split into priors/forward/samplers), new `cflibs/inversion/forward_models/` registry | High (retires the 3 264-LOC monolith) | Medium-High | Convergent across petitRADTRANS, exojax, LMFIT, Sherpa. |

### 8.2 Tier 2 — Medium impact, single-stream backing or moderate effort

| # | Pattern | Source | Target | Impact | Effort |
|---|---|---|---|---|---|
| T2-1 | `@custom_jvp` on piecewise profile cutoffs (Voigt, Stark) | C-P12 | `cflibs/inversion/physics/stark.py`, `cflibs/radiation/profiles.py` | Medium-High (gradient correctness for joint L-BFGS-B + HMC) | Low-Medium |
| T2-2 | Frozen-dataclass snapshots (`AtomicSnapshot`, `JaxMemoryPolicy`) | C-P9, C-P10, B-P5 | `cflibs/atomic/database.py`, new `cflibs/core/jax_policy.py` | Medium | Low (~1 day) |
| T2-3 | Manual `register_pytree_node` on `SingleZoneLTEPlasma` + `InstrumentModel` | C-P1 | `cflibs/plasma/state.py`, `cflibs/instrument/model.py` | Medium (eliminates jit recompile on element-list change) | Low (~1 day each) |
| T2-4 | Two-stage line DataFrame (`lines_raw` / `lines_scaled`) | B-P5 | `cflibs/inversion/physics/boltzmann.py`, `cflibs/atomic/database.py` | High (eliminates redundant SQLite hits inside the solver loop) | Low |
| T2-5 | `PartitionFunctionSource` ABC (polynomial + summation backends) | B-P7, C-P11 | `cflibs/plasma/partition.py`, `cflibs/atomic/database.py` | Medium-High (cross-validation; non-LTE path) | Low-Medium |
| T2-6 | Named-model template registry | D-P2 | `cflibs/inversion/forward_models/__init__.py` (new), `cflibs/radiation/spectrum_model.py`, `cflibs/inversion/solve/iterative.py` | Medium | Low-Medium |
| T2-7 | `lax.while_loop`-based bisection on `log n_e` (replaces scipy.optimize.brentq) | C-P3 | `cflibs/inversion/solve/iterative.py` inner step | Medium | Low (~1 day) |
| T2-8 | `input` / `params` / `misc` config namespacing on `SpectrumModel` | B-P1 | `cflibs/core/config.py`, `cflibs/radiation/spectrum_model.py` | Medium (clean batch loops; hashable for memoization) | Low |

### 8.3 Tier 3 — Polish, convention, dev-experience

| # | Pattern | Source | Target | Impact | Effort |
|---|---|---|---|---|---|
| T3-1 | Auto-save failed golden-spectrum artifacts as `.h5` for diffing | B-P10 | `cflibs/validation/`, `tests/conftest.py` | Medium (debug ergonomics) | Very low |
| T3-2 | `check_jax64bit` runtime guard with `allow_fp32_on_metal` switch | C-P15 | `cflibs/core/jax_runtime.py` | Medium (surfaces silent precision degradation on jax-metal) | Very low |
| T3-3 | `jax.tree_util.Partial` for instrument callable | C-P5 | `cflibs/instrument/model.py` | Low-Medium | Low |
| T3-4 | Outer/inner loop split with `ConvergenceState` carrying per-element residuals | D-P5 | `cflibs/inversion/solve/iterative.py` | Medium (debug ergonomics on real-data failures) | Medium |
| T3-5 | Slab combinators (`SerialSlabs` / `MergeSlabs`) for compositional RTE | B-P9 | future `cflibs/plasma/multi_zone.py` | Low (only if CF-LIBS moves beyond single-zone) | Medium |
| T3-6 | HDF5 path-as-index convention | D-P3 | `cflibs/manifold/storage.py`, `scripts/generate_model_library.py` | Low (convention-level) | Low |
| T3-7 | Tutorial gallery as paired `.ipynb` + `.rst` files | C-P16 | `docs/` | Low-Medium (doc quality; tutorials double as smoke-tests) | Medium |

### 8.4 Tier 4 — Research-grade / speculative

| # | Pattern | Rationale |
|---|---|---|
| T4-1 | `lax.custom_root` / `jaxopt.FixedPointIteration` for implicit-diff through the converged `(T*, n_e*)` | **Neither surveyed JAX code does this.** CF-LIBS feeds the converged solution into joint L-BFGS-B and HMC, so implicit gradients would shrink backward-pass memory ∝ `max_iter`. Effort ~3-5 days. Genuinely novel contribution — would be worth a methodology paper. |

---

## 9. Consequences

### 9.1 If we adopt this ADR (record the survey)

- Subsequent ADRs (0002, 0003, …) can cite specific Tier-1 / Tier-2 candidates without re-surveying.
- The convergent themes in § 7 frame *all* future architectural choices in the JAX migration — particularly the host/kernel split and forward-model unification, which are upstream of nearly every other Tier-1/2 item.
- The explicit non-import list in § 6.4 protects against accidental atmospheric-RT pattern leakage (especially k-distribution machinery and layered loops, which look superficially attractive but solve a problem CF-LIBS does not have).
- The "no implementation decision" stance means no code changes ship from this ADR. The follow-on ADRs each take a single Tier-1 candidate to a build-or-defer decision.

### 9.2 Risks of adopting

- **Cataloguing without acting** is a documented failure mode. We mitigate by tracking each Tier-1/2 candidate as a separate bead (filed in § 11) so they cannot be silently dropped.
- The survey is opinionated about effort estimates. Several Tier-1 candidates (LDM broadening, chunked scan + checkpoint, forward-model unification) are likely under-estimated and may slip into "Medium-High" or "High" effort once specced in detail. Implementation ADRs should re-estimate.
- The host/kernel split affects 31+ files and will land as a multi-PR sequence; bad ordering risks merge conflicts with in-flight JAX-port work (alias / boltzmann_jax). The follow-on ADR for T1-1 must specify the merge order.

### 9.3 If we reject this ADR

- The ad-hoc JAX migration pattern continues (`*Jax`-in-same-file proliferation; per-file fallback boilerplate; three coexisting forward-model implementations).
- Without a recorded survey, future work re-discovers these patterns piecemeal, often after burning cycles on prototypes that bypass already-validated designs.
- The convergent themes in § 7 — especially the host/kernel split and forward-model unification — are reachable without this ADR but are easier to coordinate with the synthesis recorded in one place.

### 9.4 Constraints carried forward

- **Physics-only constraint (CLAUDE.md, bead `cf-libs-hard-project-constraint-the-final-algorithm`):** every Tier-1/2/3 candidate listed has been screened against the ban. None require `flax`, `equinox`, `jax.nn`, `sklearn`, `torch`, `tensorflow`, `keras`, `transformers`, or `jax.experimental.stax`. jaxrts has an *optional* NN extra; the patterns we crib from jaxrts are all in `plasmastate.py`, `setup.py`, `models.py`, `ionization.py`, `helpers.py` — none of which import the NN extra.
- **Apple Silicon (jax-metal) precision constraint:** several candidates (chunked scan + checkpoint; manual pytree registration; `MemoryPolicy`) interact with the fp32-only constraint on jax-metal. Implementation ADRs must specify the precision fallback explicitly. T3-2 (`check_jax64bit` guard) is the canonical place to centralize this.

---

## 10. Sources and references

### 10.1 Primary repository sources

- **CF-LIBS-improved** — `/home/brian/code/CF-LIBS-improved` at branch `dev`, head 320f143 (2026-05-12). Stream A surveyed read-only.
- **Radis** — [github.com/radis/radis](https://github.com/radis/radis), branch `develop`. Stream B fetched docs at [radis.readthedocs.io](https://radis.readthedocs.io/en/latest/lbl/lbl.html).
- **jaxrts** — [github.com/JaXRTS/jaxrts](https://github.com/JaXRTS/jaxrts), branch `main`. Cloned to `/tmp/jaxrts_clone/` during Stream C.
- **Exojax** — [github.com/HajimeKawahara/exojax](https://github.com/HajimeKawahara/exojax), branch `master`. Cloned to `/tmp/exojax_clone/` during Stream C.
- **petitRADTRANS** — [gitlab.com/mauricemolli/petitRADTRANS](https://gitlab.com/mauricemolli/petitRADTRANS) + [petitradtrans.readthedocs.io](https://petitradtrans.readthedocs.io/en/latest/).
- **HELIOS** — [github.com/exoclime/HELIOS](https://github.com/exoclime/HELIOS).

### 10.2 Methodology / algorithm papers cited by the surveyed codes

- Pannier, E., Laux, C.O. 2019. *RADIS: A nonequilibrium line-by-line radiative code for CO₂ and HITRAN-like database species.* JQSRT 222–223, 12–25. [doi:10.1016/j.jqsrt.2018.09.027](https://doi.org/10.1016/j.jqsrt.2018.09.027).
- van den Bekerom, D., Pannier, E. 2021. *A discrete integral transform for rapid spectral synthesis.* JQSRT 261, 107476. [doi:10.1016/j.jqsrt.2020.107476](https://doi.org/10.1016/j.jqsrt.2020.107476).
- Lütgert, J., Schumacher, P., Rips, J., Qu, K., Döppner, T., Kraus, D. 2026. *jaxrts: An open-source library for fitting and simulating X-ray Thomson scattering measurements.* Comput. Phys. Commun., 110173. [doi:10.1016/j.cpc.2026.110173](https://doi.org/10.1016/j.cpc.2026.110173).
- Kawahara, H., Kawashima, Y., Masuda, K., Crossfield, I.J.M., Pannier, E., van den Bekerom, D. 2022. *Autodifferentiable spectrum model for high-dispersion characterization of exoplanets and brown dwarfs.* ApJS 258, 31. [doi:10.3847/1538-4365/ac3b4d](https://doi.org/10.3847/1538-4365/ac3b4d).
- Kawahara, H. et al. 2025. *ExoJAX2 — auto-differentiable radiative transfer for exoplanet atmospheres.* ApJ 985, 263. [arXiv:2410.06900](https://arxiv.org/abs/2410.06900).
- Mollière, P. et al. 2019. *petitRADTRANS — a Python radiative transfer package for exoplanet characterization and retrieval.* A&A 627, A67. [doi:10.1051/0004-6361/201935470](https://doi.org/10.1051/0004-6361/201935470).
- Malik, M. et al. 2017. *HELIOS: an open-source, GPU-accelerated radiative transfer code for self-consistent exoplanetary atmospheres.* AJ 153, 56. [doi:10.3847/1538-3881/153/2/56](https://doi.org/10.3847/1538-3881/153/2/56).

### 10.3 Internal references

- `CLAUDE.md` — project memory and physics-only constraint.
- `docs/REFERENCE_ANALYSIS_LIBSSA.md` (2026-01-07) — prior LIBS-specific code survey, still authoritative.
- Bead `cf-libs-hard-project-constraint-the-final-algorithm` — physics-only constraint specification.
- Bead `cf-libs-physics-optimization-roadmap` — accuracy-bearing physics improvements (orthogonal to this architectural ADR; not in scope here).
- Bead `CF-LIBS-improved-j7vb` — this ADR's tracking issue.

### 10.4 Agent transcripts

Full agent outputs preserved in:

- `/tmp/claude-1002/.../tasks/a50c744eb71d417ec.output` — Stream A
- `/tmp/claude-1002/.../tasks/a739c00d62c31265c.output` — Stream B
- `/tmp/claude-1002/.../tasks/a74ad26d3ebddcf3b.output` — Stream C
- `/tmp/claude-1002/.../tasks/a1e0b9ea19a61ffd1.output` — Stream D

---

## 11. Follow-on tracking

Each Tier-1 and Tier-2 candidate should be filed as a separate bead before this ADR is closed, blocked on this one (`bd dep add <new> CF-LIBS-improved-j7vb`). Candidate beads:

- T1-1 (host/kernel split + shared JAX decorator)
- T1-2 (forward-model unification)
- T1-3 (`lax.while_loop` in iterative solver)
- T1-4 (LDM/DIT broadening for manifold)
- T1-5 (chunked `lax.scan` + `checkpoint` over ν-grid)
- T1-6 (retrieval driver decomposition)
- T2-1 through T2-8 (per § 8.2)

Tier-3 polish items fold into existing or future cleanup beads. Tier-4 (`lax.custom_root` implicit-diff) deserves a research bead with a literature-review precursor.

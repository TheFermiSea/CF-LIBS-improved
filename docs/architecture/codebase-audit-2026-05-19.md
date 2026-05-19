# CF-LIBS-improved codebase audit — 2026-05-19

Cross-cutting architecture / quality / docs / performance review, produced
by four parallel read-only agents (architect, code-reviewer, scribe,
feature-dev:code-explorer). Each finding cites file:line evidence. Items
are grouped by theme and prioritized by impact-over-effort.

**Constraint note:** NotebookLM MCP was disconnected when this audit ran,
so the citation-gap list at the bottom is staged for manual ingestion
into NotebookLM. Serena was also disconnected — agents fell back to grep,
so line numbers may be ±1–2.

---

## Top 10 actions (do these first)

| # | Action | Severity | Effort | Bead |
|---|---|---|---|---|
| 1 | Retarget new-location modules to stop importing through legacy flat shims | high | S | `arch-1` |
| 2 | Enable JAX on-disk compile cache at cluster startup | high | S | `perf-1` |
| 3 | Move `_saha_two_stage_populations` index-building out of every leapfrog call | high | S | `perf-2` |
| 4 | Extract `unified.py` (still 2206 lines) into Runner + 4 sibling modules | high | M | `arch-2` |
| 5 | Promote `AtomicSnapshot` arrays to float64 once (eliminate per-call dtype promotion) | high | S | `perf-3` |
| 6 | Replace `SingleZoneLTEPlasma` species dict with flat pytree array | high | M | `perf-4` |
| 7 | Consolidate setup instructions across README/CLAUDE.md/AGENTS.md | medium | S | `docs-1` |
| 8 | Add `.bib` entries for the 12 hanging code citations (use Asta/paper_search) | medium | S | `docs-2` |
| 9 | Introduce `BaseIdentifier` to share peak/baseline/coverage preprocessing | medium | M | `arch-3` |
| 10 | Delete the `cflibs/benchmarks/` (trailing-s) shim package | medium | S | `arch-4` |

**Wall-time effect of #1–6:** with the architecture fixes plus the four
perf wins, the bayesian benchmark per-spectrum time projects from 945 s →
~780 s on V100S (~17% reduction, primarily from cache reuse + dtype +
Python overhead elimination).

---

## 1. Architecture

### arch-1 — New-location modules round-trip through legacy flat shims **(HIGH / S)**

The ADR-0001 sub-package split moved code from `cflibs/inversion/{boltzmann,
closure, solver, …}.py` into `cflibs/inversion/{physics,solve,…}/*.py`. The
old flat paths were kept as thin shims that `*`-import from the new
locations. But the **new** locations still import from the **old** shims:

- `cflibs/inversion/physics/uncertainty.py:23` → `from cflibs.inversion.uncertainty import …`
- `cflibs/inversion/physics/matrix_effects.py:40` → `from cflibs.inversion.solver import CFLIBSResult`
- `cflibs/inversion/solve/closed_form.py:30` → `from cflibs.inversion.solver import CFLIBSResult`
- Many more across `physics/`, `solve/`, `runtime/`, `identify/`.

Every load round-trips: new → shim → new. The shim layer was supposed to
serve external callers only; internal usage defeats the split.

**Fix:** mechanical rewrite — each new-location import retargets to a
canonical sibling. Then the shims become real leaves (no circular paths)
and #arch-5 (deleting them) becomes safe.

### arch-2 — `benchmark/unified.py` is still a 2206-line god module **(HIGH / M)**

After the checkpoint + composition_eval extractions, `unified.py` still
contains:

- dataset adapters (lines ~190–490)
- split construction (~491–525)
- workflow registries (~523–1100)
- tuning + evaluation (~1213–1419)
- statistics + aggregates (~1419–1900)
- CSV/JSON I/O (~1891–1916)
- matplotlib plotting (~1917–2049)
- the `UnifiedBenchmarkRunner` orchestrator (~2050–end)

Single biggest merge-conflict hotspot; mixing matplotlib with statistics
blocks targeted dependency trimming.

**Fix:** extract in order — `benchmark/datasets/loaders.py`,
`benchmark/registries.py`, `benchmark/aggregates.py`, `benchmark/reporting.py`.
What remains in `unified.py` is the Runner only.

### arch-3 — ALIAS is 3440 lines, identifiers duplicate preprocessing **(HIGH / M)**

Each of the six identifiers (`alias`, `comb`, `correlation`, `hybrid`,
`spectral_nnls`, `line_detection`) independently calls `estimate_baseline`,
`detect_peaks_auto`, `estimate_noise` with slightly different conventions.
`IdentifierProtocol` (`_protocol.py`) only pins the `identify(...)`
signature — there's no shared base.

`alias.py` alone has:
- `__init__` 388 lines (validation + caching + grid pre-compute)
- `identify()` 625 lines (9 sub-phases, no phase-level helpers)
- `_estimate_plasma_temperature` 266 lines

**Fix:** introduce `identify/_base.py` with `BaseIdentifier._preprocess`
returning a typed `PreprocessedSpectrum`. Split ALIAS into
`alias/core.py`, `alias/scoring.py`, `alias/grid.py`.

### arch-4 — `cflibs.benchmark` vs `cflibs.benchmarks` shim duplication **(MED / S)**

`cflibs/benchmarks/__init__.py` is a `DeprecationWarning`-emitting shim
re-exporting from `cflibs.benchmark`. The trailing-s typo trap, zero
production callers outside `scripts/`, deprecation note says "2026-04
cleanup". Delete it.

### arch-5 — 25+ zero-logic flat shim modules **(MED / S, blocked on arch-1)**

Files like `cflibs/inversion/{solver, bayesian, closure, boltzmann, …}.py`
are 3-line stubs. Tests still use them heavily. After arch-1 lands,
bulk-rewrite tests via codemod and delete the shim files.

### arch-6 — `benchmark/workflows.py` is 1026 lines with layering inversion **(MED / M)**

Contains 28 `_build_*_predictor` / `_fit_*_pipeline` functions. The lazy
script loader at `workflows.py:122` reflectively loads `scripts/*.py`
files — library imports scripts, not the reverse. Split into
`workflows/{identifiers,pipelines,aalto_adapter}.py` and move script
helpers into `cflibs/benchmark/datasets/`.

### arch-7 — `manifold/` exposes deep modules while `benchmark/` uses lazy `__getattr__` **(LOW / S)**

`manifold/__init__.py:11-13` eagerly imports `generator`, `loader`,
`config`; consumers bypass and import `manifold.basis_library.BasisLibrary`
directly from 40+ sites. Mirror benchmark's lazy pattern.

### arch-8 — `IterativeCFLIBSSolverJax` deprecated but still in `__all__` **(LOW / S)**

Pick a removal milestone (e.g., v0.5), update the one test that still uses
it (`tests/benchmark/test_jax_workflows.py`), delete.

### arch-9 — `_coverage.py` wired into only 4 of 6 identifiers **(LOW / S)**

`spectral_nnls.py` and `line_detection.py` emit no coverage telemetry,
breaking the "all identifiers comparable" contract. Pull `CoverageTracker`
setup into the `BaseIdentifier` from arch-3, or add to the two missing
identifiers explicitly.

---

## 2. Code quality / readability

### quality-1 — 5 functions over 200 lines **(MED / M)**

| Function | File:line | Length |
|---|---|---|
| `ALIASIdentifier.identify` | `alias.py:1284-1909` | 625 |
| `detect_line_observations` | `line_detection.py:440-830` | 390 (+ 28 params) |
| `ALIASIdentifier.__init__` | `alias.py:811-1199` | 388 |
| `calibrate_wavelength_axis` | `wavelength_calibration.py:379-668` | 289 |
| `_estimate_plasma_temperature` | `alias.py:2044-2310` | 266 |

Common pattern: each function orchestrates 5–9 sub-phases inline. Fix by
extracting per-phase private methods.

### quality-2 — Optional-deps gateway inconsistency **(MED / S)**

67 `except ImportError` blocks across `cflibs/`. Some use `HAS_JAX` flags,
others use bare try/except, others patch `jnp = None`. Centralize in
`cflibs/optional_deps.py`.

### quality-3 — `Dict[str, Any]` in 8+ public APIs **(MED / S)**

Callers can't tell required keys. Define `TypedDict` (or dataclass) per
config surface. Top offenders: `bayesian_sparse_id.py:85`, the workflow
config dicts.

### quality-4 — Public functions missing docstrings (15+) **(MED / S)**

Especially `cflibs/benchmark/{bayesian_sparse_id, harness, composition_metrics}.py`.
Add NumPy-style with a one-line **physics/algorithm** description and
citation key (`\cite{NoelEtAl2025}` style) for any equation.

### quality-5 — 30+ "what" comments where they should be "why" **(LOW / S)**

E.g., `alias.py:1321 # Reset per-dispatch self-absorption damping counters`
is obvious from code. Reserve comments for non-obvious constraints
(hidden invariants, workarounds, references).

### quality-6 — Stale PR-number references in comments **(LOW / S)**

8+ comments cite closed PR numbers as context. PR metadata rots; inline
the rationale or link to an ADR.

### quality-7 — `_load_atomic_data` embeds 140-line element-mass dict **(LOW / S)**

`manifold/generator.py:107-353`. Extract `STANDARD_MASSES` to
`cflibs/core/atomic_constants.py` or use the `periodictable` library.

---

## 3. Documentation

### docs-1 — Setup instructions diverge across 3 entry points **(HIGH / S)**

| File | Instructions |
|---|---|
| `README.md:35-52` | 4 paths (core, jax-cpu, jax-metal, local, cluster) |
| `CONTRIBUTING.md:34-46` | 3 paths (base, metal, cluster) |
| `AGENTS.md:34-46` | different from both above |
| `CLAUDE.md:18-24` | 2-step install |

**Fix:** single source of truth in CLAUDE.md; README/CONTRIBUTING/AGENTS
each link to it instead of restating.

### docs-2 — 12 papers cited in code without `.bib` entries **(HIGH / M)**

See `## Citation gap list` below. Each needs verification (real paper?
DOI?) plus a BibTeX entry in `paper/refs.bib`.

### docs-3 — `docs/API_Reference.md` incomplete + abandoned **(MED / S)**

Cuts off mid-function at line 100. Either complete or stub with a link
to autogenerated API docs and `CLAUDE.md` module map.

### docs-4 — `CLAUDE.md` and `AGENTS.md` setup conflict **(MED / S)**

Same as docs-1.

### docs-5 — README cites "Ciucci et al. (2009)" but bib has Ciucci1999 **(MED / S)**

`README.md:282-294`. Year mismatches on Ciucci, Salzmann, Griem,
Colombant-Tonon, Landi.

### docs-6 — `CLAUDE.md:215` references non-existent `manifold-generator.py` **(LOW / S)**

Script was deleted; the note about not running it under MPI is now a
trap. Replace with "Use `cflibs generate-manifold`".

### docs-7 — `User_Guide.md:24-30` specifies Python 3.8 **(LOW / S)**

Actual minimum is 3.12. Match `pyproject.toml`.

### docs-8 — `docs/archive/legacy/` has 15+ files with no DEPRECATED marker **(LOW / S)**

New contributors may follow stale architecture docs. Add a `README.md`
to that directory pointing to current sources.

### docs-9 — Missing top-level docs **(MED / M)**

| Missing | Where it should live |
|---|---|
| CHANGELOG | repo root |
| inversion/ sub-package API reference | `docs/api/inversion.md` |
| "How to cite CF-LIBS" | `docs/CITATION.md` |
| JAX troubleshooting | `docs/troubleshooting/jax.md` |

### docs-10 — Year corrections + path/anchor fixes (smaller items)

- `ROADMAP.md:54,65` references bead IDs without verification
- `docs/CF-LIBS_Codebase_Technical_Documentation.md:40` broken relative link
- `docs/adr/ADR-0001-RUNBOOK.md:19` branch-name typo (`retrieval-decomposition` vs `bayesian-decomposition`)
- `AGENTS.md:77` anchor format
- `Quick_Start_For_Scientists.md:24` undefined `cflibs doctor` command
- `Manifold_Generation_Guide.md:25` orphaned "Phase 3" reference

---

## 4. Performance

### perf-1 — Enable JAX on-disk compile cache at cluster startup **(HIGH / S)**

`sampler_cache` in `workflows.py:1001` is closure-local. Multi-process
cluster runs (per-shard) each pay 30–60 s of NUTS JIT compile. Set
`jax.config.update("jax_compilation_cache_dir", "...")` at cluster
startup (in `hpc/gpu_config.py` or the SLURM wrapper). 32-shard runs
save ~16–32 min total.

### perf-2 — Hoist Python index arrays out of `_saha_two_stage_populations` **(HIGH / S)**

`kernels.py:218-244` runs three Python loops (`list.index`, NumPy array
construction) on every leapfrog. Snapshot is frozen — the indices never
change. Move to a `@staticmethod` on `AtomicSnapshot.__post_init__`,
pass pre-computed fields. ~10–25% Python-overhead reduction.

### perf-3 — Store `AtomicSnapshot` arrays at float64 directly **(HIGH / S)**

`AtomicDataArrays` stores all line/species arrays as `jnp.float32`. The
forward model at `kernels.py:262-263` upcasts to the tracer dtype
(float64 under MCMC) on every call. Storing at float64 eliminates the
per-call promotion kernel and ~1 GB of unnecessary float conversions
over a full chain.

### perf-4 — Replace `SingleZoneLTEPlasma.species` dict with a flat pytree array **(HIGH / M)**

`forward.py:318-328` allocates a Python dict on every leapfrog. Register
`SingleZoneLTEPlasma` as a JAX pytree with concentrations as a flat
1-D array; pass element-index map via `static_argnums`. Combined with
perf-2 this is the dominant Python-layer overhead.

### perf-5 — Pre-compute `line_mass_amu` on snapshot construction **(MED / S)**

`kernels.py:533-534` calls `_species_mass_array(snapshot)` plus a NumPy
indexing op on every `forward_model` call. Cache once on snapshot
build.

### perf-6 — Two-zone path computes Faddeeva matrix twice **(MED / S)**

`forward.py:475-488` evaluates `_voigt_profile_kernel_jax` over the
full `(N_wl × N_lines)` outer product once for core and once for shell,
with identical `diff`. Hoist the profile matrix out; multiply by
per-zone emissivities downstream. Halves the dominant kernel cost for
two-zone inference.

### perf-7 — Hoist `from … import` statements out of `_compute_spectrum` **(LOW / S)**

`forward.py:306-308` defers three imports inside the method (noqa
PLC0415). Cheap per call, but the deferral was for circular-import
breakage — resolve those cycles (arch-1 helps) and move imports up.

### perf-8 — Constant-fold `sqrt(2)`, `sqrt(2π)` in Voigt kernel **(LOW / S)**

`profiles.py:580,593` emits `jnp.sqrt(jnp.asarray(...))` calls inside the
`@jit` kernel. XLA likely folds these already; verify with
`jax.make_jaxpr` and replace with module-level dtype-keyed constants if
not.

### perf-9 — Parquet writer two-pass row-dict pivot **(LOW / S)**

`results.py:451-533` builds `List[Dict[str, Any]]` then transposes. At
1000-spectrum scale this is ~50k Python dict writes. Return
column-oriented `Dict[str, List]` directly.

### perf-10 — `baseline_scale = 0.1 * jnp.max(observed)` re-computed every leapfrog **(LOW / S)**

`forward.py:615`. Pre-compute the constant from the NumPy observed
array before `mcmc.run` and pass into `PriorConfig`.

### perf-11 — dynesty path materializes spectrum on host per likelihood call **(LOW / L)**

`samplers.py:518-531`. dynesty isn't JAX-aware. Either deprecate the
dynesty backend in favor of NumPyro NUTS (already primary), or
re-implement nested sampling with `jax.lax.while_loop`.

### perf-12 — Two-zone `partition_function` not vectorized **(LOW / S)**

`atomic.py:330-335`. Align with `kernels.py:264` pattern (one call on
the full `pf_all` matrix).

### perf-13 — `jnp.asarray(snapshot.line_stark_w)` etc. on every call **(LOW / S)**

8–10 conversions per leapfrog on frozen snapshot fields. Pre-convert
once in `BayesianForwardModel.__init__`. Cheap individually, ~2M
dispatch calls eliminated over a full chain.

---

## Citation gap list

These appear in code/docs but have NO entry in `paper/refs.bib`. Each
needs verification + a BibTeX entry. **When NotebookLM reconnects, this
list should be ingested into the citation notebook.**

### Code-level (12 hanging)

| Citation | First seen at | Topic |
|---|---|---|
| Black et al. 2024 | `alias.py:1502` | Sparse NNLS overfitting in spectral unmixing |
| Noel et al. 2025 | `alias.py:4,1940` | ALIAS algorithm, peak-detection enhancement |
| Vrabel et al. 2020 | `loaders.py:835` | LIBS benchmark dataset (Sci Data 7:175) |
| Wakil et al. 2023 | `boltzmann.py:146` | Boltzmann fit multiplet aggregation |
| Gajarska et al. 2024 | `comb.py:4` | Comb matching per-tooth activation threshold |
| Cristoforetti et al. 2010 | `temporal.py:47` | Spectrochim. Acta B 65, 86–95 (gate optimization) |
| El Sherbini et al. 2020 | `boltzmann.py:12` | Curve-of-growth self-absorption detection |
| Jochum et al. 2005 | `reference_compositions.py:12` | USGS GeoReM standards |
| Jochum et al. 2016 | `usgs.py:333` | USGS G-2 granite composition |
| Jochum et al. 2022 | `usgs.py:333` | USGS reference material update |
| Labutin et al. 2013 | `alias.py:1940` | Spectral correlation peak-region semantics |
| van den Bekerom & Pannier 2021 | `ADR-0001-RUNBOOK.md` (LDM bead T1-4) | LDM broadening |

### Doc-level (7 additional)

| Citation | File:line | Topic |
|---|---|---|
| Salzmann (1998) | `README.md:288` | Saha-Boltzmann equilibrium book |
| Colombant & Tonon (1973) | `README.md:289` | X-ray emission from laser-produced plasmas |
| Landi & Degl'Innocenti (1999) | `README.md:294` | Stark broadening of H lines |
| Griem (1974) | `README.md:293` | Spectral Line Broadening (book) |
| RADIS library | `docs/adr/ADR-0001-radis-jaxrts-pattern-survey.md` | Reference radiative-transfer library |
| Kramida 2024 | `docs/atomic_data/kramida_2024_delta.md` | NIST ASD delta atomic data |
| Ciucci et al. (2009 vs 1999 year mismatch) | `README.md:282` | CF-LIBS method (year is wrong; bib has 1999) |

### Recommended ingestion workflow once NotebookLM reconnects

1. Create a notebook `CF-LIBS Repository Citations` (or reuse an existing
   physics-references notebook if one exists).
2. Add each verified paper as a source (DOI, arXiv ID, or PDF if open).
3. Generate BibTeX from each source and merge into `paper/refs.bib`.
4. Mass-update code/doc references: `Noel et al. 2025` → `\cite{NoelEtAl2025}`.

For papers that *cannot* be verified (no DOI, no arXiv, no journal):
- Drop the in-code reference, or
- Mark with `# CITATION-NEEDED:` and file a bead.

---

## What this audit deliberately did NOT cover

- Test-coverage gaps (would need its own coverage-tool run)
- Type-checker tightening (mypy is opt-in error-codes; ratcheting is a
  separate workstream)
- Physics correctness of the algorithms (deferred to the bead-tracked
  refactor work in `cf-libs-physics-optimization-roadmap`)
- Security review (no obvious surface; CF-LIBS doesn't process untrusted
  inputs in production)

---

## File locations referenced

(For quick navigation when picking up items)

- `cflibs/inversion/identify/alias.py`
- `cflibs/inversion/identify/_protocol.py` and `_coverage.py`
- `cflibs/inversion/physics/{boltzmann,closure,cdsb,matrix_effects,line_selection,self_absorption,quality,uncertainty}.py`
- `cflibs/inversion/solve/{closed_form,iterative,bayesian/}.py`
- `cflibs/inversion/runtime/{streaming,temporal}.py`
- `cflibs/inversion/solve/bayesian/{forward,samplers,atomic}.py`
- `cflibs/radiation/{kernels,profiles}.py`
- `cflibs/benchmark/{unified,workflows,results,composition_eval,checkpoint}.py`
- `cflibs/benchmark/datasets/usgs.py`
- `cflibs/manifold/{generator,vector_index,basis_library}.py`
- `cflibs/benchmarks/` (entire deprecated package)
- `cflibs/core/{platform_config,logging_config,jax_runtime,constants}.py`
- `paper/refs.bib`
- `README.md`, `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md`, `ROADMAP.md`

# CF-LIBS Overhaul — Dependency-Ordered Execution Blueprint

**Author:** Lead Planner (synthesis pass)
**Date:** 2026-06-25
**Source of truth:** `scratchpad/overhaul/verified/*.md` (adversarially-verified findings; these
supersede the raw census). Findings marked FALSE in the verified reports are **excluded** from this
plan (notably: inv-solve F1 "missing ln(U_II/U_I)" — would INTRODUCE a bug; inv-identify F1/F6/F10;
io F5/F10; inv-top F1; validation F2/F3-original; inv-physics F2).

No `blueprint/ARCHITECTURE.md` was present; the target architecture is inferred from the verified
findings + CLAUDE.md module map and is stated inline per milestone.

---

## Classification legend

- **SAFE-NOW** — correctness bug verifiable by parity test, physics re-derivation, dead-code
  removal, type/doc fix, or a refactor that is exactly behavior-preserving. No benchmark needed.
  Acceptance = a green targeted test / re-derivation / import-parity check.
- **BENCHMARK-GATED** — changes scoring, composition output, identifier decisions, or any
  accuracy-affecting physics in a *default-ON* path. Ships **default-OFF behind a flag** and must
  pass a node held-out benchmark (composition F1 / Aitchison / RMSEP non-regression) before the
  flag is promoted. Flag-promotion is a *separate* later work item.
- **DESIGN-DECISION — needs a measured comparison (A/B on benchmark or profiling) before one of
  several legitimate options is chosen; the choice itself is the deliverable.

**Mission priority (governs ordering):** ACCURACY > PRECISION > RELIABILITY > latency. Every
performance-only item is therefore deferred behind correctness/accuracy items in the same area.

---

## Cross-cutting themes (drive milestone grouping)

1. **Mole↔mass-fraction basis mismatch (the single highest-impact accuracy theme).** Solvers emit
   number/mole fractions; every truth corpus + validation + export stores mass fractions; **no
   conversion is applied anywhere on the scoring path.** Verified CRITICAL in `benchmark` F1,
   `validation` F1 (via missing hc/λ it compounds), HIGH in `pds` F1-1, present in `runtime` F1.
   Völker (2024) quantifies up to 353% error. This is **benchmark-gated** because it changes every
   composition number — but it is the top accuracy lever. → **Milestone M3**.

2. **McWhirter ΔE = max(E_k) instead of resonance energy (physics-wrong default).** Verified
   CRITICAL/HIGH in `inv-physics` F1/F9, `plasma` F4, `cli` F1. Correct resonance path EXISTS and is
   DB-validated but gated OFF (`CFLIBS_MCWHIRTER_RESONANCE_DE`). Over-rejects valid LTE plasmas
   (~8–27× too-high n_e floor). Promotion is benchmark-gated. → **Milestone M2**.

3. **Architecture / layering debt that blocks safe refactor.** `CFLIBSResult` defined in
   `solve/iterative.py` but imported by io/jitpipe/closed_form (inv-solve F5); `LineObservation`
   imported through the `physics/boltzmann` shim by ~11 modules incl. `common/element_id.py`
   (inv-common F5/F11); `AtomicSnapshot` domain carrier living in `core/jax_runtime.py` (core F1);
   `physics/matrix_effects.py` module-level circular import to `solve/iterative` (inv-physics F5/FX1).
   All SAFE-NOW (behavior-preserving moves + import-parity). → **Milestone M1**.

4. **Olivero-1977 Voigt constants wrong in 3 files** (radiation F1+A, jitpipe/stark, instrument
   uses 2.355 vs exact). SAFE-NOW (single source of truth + parity). → **Milestone M1**.

5. **Cache `None`-sentinel bug** (core F3, atomic F5, plasma F2/FV1) — DB re-queried for every
   missing species; SAFE-NOW. → **Milestone M1**.

6. **Reliability-gate leaks** (cli F5/MF1 refuse-to-report still prints composition; hpc R-hat/ESS
   never gated; runtime FastAnalyzer `converged=True` placeholder). SAFE-NOW. → **Milestone M4**.

---

# MILESTONE M0 — Test & benchmark scaffolding (enables everything)

Goal: make every later item verifiable. Pure additive test infrastructure; no production behavior
change. Must land first so SAFE-NOW fixes can be pinned and BENCHMARK-GATED items have a gate.

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M0-1 | Pin canonical Saha/IPD/DH constants | plasma F1/F8, core F8, atomic F10 | SAFE-NOW | New `pytest` asserts: DH IPD ≈0.066 eV @ n_e=1e17,T=1e4K (replaces wrong `<=0.06` doctest); `SAHA_CONST_CM3` derived from CODATA within 0.1% incl. ×2 spin factor; `KB_EV`/`CM_TO_EV` cross-checks. All green. | — | S |
| M0-2 | Held-out composition benchmark harness wrapper | benchmark F5 (3-way dup), CLAUDE mission | SAFE-NOW | A single documented command runs the node held-out composition benchmark and emits F1/Aitchison/RMSEP JSON; baseline snapshot committed. Used as the gate for all BENCHMARK-GATED items. | — | M |
| M0-3 | Parity-test fixtures for dual physics paths | radiation F3, manifold F10, inv-solve F6/F10, native-rust F3/F6/A | SAFE-NOW | Reusable fixtures comparing: SpectrumModel-precomputed vs kernel internal populations; LDM vs Voigt; SB-graph vs common-slope; Rust vs Python comb on *ambiguous* input. Tests added (xfail-marking known-divergent until fixed). | — | M |

---

# MILESTONE M1 — SAFE-NOW correctness, layering & dead-code (no benchmark needed)

Target architecture: result/observation/atomic-snapshot domain types live in domain packages, not
in solver/runtime modules; single source of truth for physical constants; no module-level circular
imports; documented dead code removed. All items behavior-preserving and verifiable by parity/types.

### Layering & domain-type relocation (do these first — they unblock later refactors)

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M1-1 | Move `CFLIBSResult` to `inversion/common/` | inv-solve F5 (`iterative.py:94`; importers io/exporters:41, jitpipe/pipeline:213, closed_form:34) | SAFE-NOW | `CFLIBSResult` defined in `common/`; old path re-exports for back-compat; all 3+ importers redirected; `pytest` import + a result-roundtrip test green; no behavior change. | M0 | M |
| M1-2 | Redirect all `LineObservation` imports to `common.data_structures` | inv-common F5/F11 (~11 modules via `physics/boltzmann` shim) | SAFE-NOW | Every production module imports `LineObservation` from `common.data_structures`; shim kept but unused by cflibs/; `find_referencing`/grep shows 0 internal shim imports; tests green. | M0 | M |
| M1-3 | Break `physics/matrix_effects` ↔ `solve/iterative` cycle | inv-physics F5/FX1 (`matrix_effects.py:50`) | SAFE-NOW | Move import of `CFLIBSResult` to function body or `TYPE_CHECKING` (trivial after M1-1); add a fresh-process `import cflibs.inversion.physics.matrix_effects` test that previously could deadlock. | M1-1 | S |
| M1-4 | Extract `AtomicSnapshot` out of `core/jax_runtime.py` into `atomic/` | core F1 (`jax_runtime.py:429-551`) | SAFE-NOW | Dataclass + pytree registration moved to an atomic-domain module; `jax_runtime` keeps only backend/capability code; consumers (database/kernels/spectrum_model/bayesian.atomic) updated; pytree flatten/unflatten round-trip test green. | M0 | M |
| M1-5 | Fix `abc.py → atomic.structures` layer inversion | core F2 (`abc.py:12`) | SAFE-NOW | Resolve via `TYPE_CHECKING` or relocating the type; no runtime import from core→atomic; import-order test green. | M1-4 | S |

### Single-source-of-truth constants

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M1-6 | Correct Olivero-1977 Voigt constants (0.5343/0.2169) + one definition | radiation F1+A (`profiles.py:257`, `radiation/stark.py:200`, `jitpipe/stark.py:79`), instrument F1 (2.355 → exact, also benchmark/synthetic_corpus:387) | SAFE-NOW | Single module-level constant pair; all 3 Olivero sites + all `2.355` sites import it; pseudo-Voigt + Stark-FWHM parity test vs Olivero closed form within 1e-6; manifold-vs-instrument sigma parity test green. | M0-3 | M |
| M1-7 | De-duplicate McWhirter `1.6e12` prefactor | core F6 (3 sites: physical_consistency:49, line_selection:114, temporal:455) | SAFE-NOW | All 3 import `MCWHIRTER_CONST` from `core.constants`; test asserts identity; no numeric change. | — | S |

### Cache / DB correctness (accuracy-neutral but fixes silent stale/missing behavior)

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M1-8 | Fix `None`-sentinel cache miss | core F3/MF2 (`cache.py:149`), atomic F5 | SAFE-NOW | Use a distinct MISS sentinel so `None` results cache; regression test: 2nd query for a missing species hits cache (0 extra DB calls); no `None`-flooding eviction. | M0 | S |
| M1-9 | Clear partition caches on new DB / shorten TTL fragility | plasma F2/FV1 (`saha_boltzmann` cache key), atomic F7, core MF1 | SAFE-NOW | `AtomicDatabase.__init__` invalidates partition caches (or key by `id(self)`/db sha); test: replacing DB file at same path returns fresh values; pickle-`self` no longer on hot key path. | M1-8 | M |
| M1-10 | Version-stamp schema; skip migration when current | atomic F6/NEW-2 (`database.py:68`) | SAFE-NOW | `cflibs_meta` version row; `__init__` does ≤1 cheap check when up-to-date; test asserts no PRAGMA storm on warm DB. | — | S |

### Documented dead code, doc/contract fixes, type guards

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M1-11 | Fix `TwoRegionPlasma` tracer guard | plasma F5 (`state.py:436`) | SAFE-NOW | Use `_is_jax_tracer_or_array`; add a concrete-jnp-array construction test (previously `ConcretizationTypeError`). | — | S |
| M1-12 | Stark-width docstring/convention fixes | atomic F1 (`structures.py:61` 10^16→1e17), F2 (ln basis), F9, inv-common F4, atomic NEW-1 (0.005 fallback) | SAFE-NOW | Docstrings state 1e17 + `ln U=Σaₙ(ln T)ⁿ`; estimate-fallback literal corrected to 1e17 convention; test pins `structures.stark_w` convention == `radiation.stark.REF_NE`. | — | S |
| M1-13 | Remove/realign stale "NOT used"/"removed" comments | inv-physics F6 (`quality.py:75`), inv-identify NEW-2 (`alias.py:1863`), io F3 (eager "lazy") | SAFE-NOW | Comments match reality (QualityAssessor IS wired post-M7; N_matched gate still present); io imports either truly lazy via TYPE_CHECKING or comment removed. | — | S |
| M1-14 | `identify_resonance_lines` no longer silently returns `{}` | inv-physics F7 (`line_selection.py:782`) | SAFE-NOW | Either implement using available data or raise `NotImplementedError`/deprecate + redirect callers to the working `SUSPECT_E_I_MAX_EV` screen; test asserts no silent-empty. | — | S |
| M1-15 | `config.validate_instrument_config` accepts resolving-power | core F7 (`config.py:234`), instrument F2 (`model.py:108` `from_file` drops `resolving_power`) | SAFE-NOW | Validator + `from_file` honor `resolving_power`; `cflibs forward` with an R-mode YAML no longer hard-errors / silently zeroes sigma; round-trip test green. | M0 | M |
| M1-16 | Dead `wavelength_calibration` field on `InstrumentModel` | instrument F4 (`model.py:33`) | SAFE-NOW | Remove field (no callers in cflibs/) OR document + keep out of pytree aux; pytree complexity test green. | M1-4 | S |
| M1-17 | `ionization_potential_lowering` dead `model` param | plasma F7 (`saha_boltzmann.py:603`) | SAFE-NOW | Remove param (callers pass none) or route to `make_ipd_model`; test green. | — | S |
| M1-18 | CLI `--elements` parsing unified | cli F3/MF2 (`main.py` nargs="+" vs split(",")) | SAFE-NOW | All subcommands accept the same syntax; `_resolve_invert_elements` splits comma-joined; tests for `Fe,Cu` and `Fe Cu` on every subcommand. | — | S |
| M1-19 | CLI/manifold small crashers & smells | cli F7 (`[::10]` silent decimation + shadowing), F9 (`manifold_cmd` ZeroDivision for total<10 + dead branch) | SAFE-NOW | stdout decimation warns or is removed; small-grid progress no longer raises; tests for total<10 and stdout parity. | — | S |
| M1-20 | Evolution blocklist: catch `exec`/`eval`/`compile` | evolution 1.1/5.1 (`evaluator.py:148`) + binary-diff false-reject 2.3 (`:253`) | SAFE-NOW | `scan_source` flags `exec("import torch")`/`eval`/`compile`; binary/mode-change diffs no longer false-reject; new tests (live-exec style) green. Physics-only constraint hardening — no benchmark. | — | S |

---

# MILESTONE M2 — McWhirter ΔE physics correction (BENCHMARK-GATED promotion)

Target: the physics-correct resonance-ΔE McWhirter path becomes the default after it is shown not
to regress composition and to reduce false non-LTE rejections on the held-out node benchmark.

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M2-1 | Resonance-ΔE regression test (legacy vs resonance) | inv-physics F10, plasma F8 | SAFE-NOW | Test computes both `max(E_k)` and resonance ΔE for the same observation set (Fe I etc.) and pins the ratio + the n_e-floor direction; documents the known DB-validated PASS case. | M0-1 | S |
| M2-2 | Wire resonance lookup so `_delta_e_from_observations` can use it | inv-physics F1, plasma F4 (`lte_validator.py:387`), cli F1 | BENCHMARK-GATED | Resonance ΔE available to the validator (lower-level/resonance energy threaded from DB); behind existing `CFLIBS_MCWHIRTER_RESONANCE_DE`, still default-OFF; unit parity test green. | M2-1 | M |
| M2-3 | Promote resonance ΔE to default-ON | inv-physics F9, cli F1/F8 | BENCHMARK-GATED | M0-2 benchmark shows composition non-regression AND fewer false non-LTE rejections; flip default; old behavior reachable via explicit opt-out flag; benchmark JSON committed. | M2-2, M0-2 | M |
| M2-4 | Benchmark McWhirter default ΔE (2.0 eV → species/spectrum) | benchmark F4 (`physical_consistency.py:463`) | BENCHMARK-GATED | Either populate per-spectrum `delta_e_ev` from solver or switch default to resonance-derived; benchmark-gated; documents Ca/Si over/under-strictness fix. | M2-2 | S |

---

# MILESTONE M3 — Mole↔mass-fraction basis (BENCHMARK-GATED, top accuracy lever)

Target: a single canonical molar-mass conversion utility, applied consistently at every
truth-vs-prediction comparison and every export, with explicit unit labels. Highest-impact accuracy
change; every number moves, so strictly benchmark-gated and flag-guarded during rollout.

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M3-1 | Canonical `number↔mass` fraction utility (single impl) | benchmark F1, validation F1, pds F1-1, inv-runtime F1, manifold F1 (consistency) | SAFE-NOW | One tested converter (round-trip exact; sums→1) in a shared location; reuses existing `_number_to_mass_fractions` logic; no caller wired yet. | M0 | S |
| M3-2 | Apply conversion on benchmark scoring path | benchmark F1/F5 (`composition_eval.py:266`, harness, metrics) — three dup paths | BENCHMARK-GATED | All 3 scoring paths convert solver number-fractions→mass before comparing to mass-fraction truth; behind a flag; M0-2 benchmark re-run shows the corrected (and necessarily different) composition errors; results committed as the new baseline. | M3-1, M0-2 | M |
| M3-3 | Apply conversion in validation round-trip + pds bridge | validation F1 path, pds F1-1 (`validation.py:80`) | BENCHMARK-GATED | Round-trip + PDS comparisons convert consistently; round-trip accuracy test tolerances tightened (see M4-7); pds dataset comparison documented. | M3-1, M3-2 | M |
| M3-4 | Export unit labels + derived linear n_e | io F1 (`log_ne` base), F2 (conc unit), io M1 (`@property` dropped) | SAFE-NOW | CSV/HDF5 carry `log10_ne_cm3`+`ne_cm3` and a `basis=mole|mass` label; `_dataclass_to_dict` includes property-derived fields; exporter test against REAL result types (not mocks, io F9). | M3-1 | M |

DESIGN-DECISION embedded: **where** the conversion canonically happens (at solver output boundary
vs at each scorer). Recommend: keep solver output as number-fractions (physics-native) and convert
at every comparison/export boundary via M3-1. Validate by confirming a single converter call site
audit passes (no raw number-fraction reaches a mass-fraction comparison).

---

# MILESTONE M4 — Reliability gates & forward-model physics correctness

Target: "refuse to report" actually withholds, convergence flags reflect physics, and the synthetic
forward models used to *generate* and *validate* are physically correct (or clearly flagged).

### SAFE-NOW reliability-gate plumbing

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M4-1 | Refuse-to-report actually withholds composition | cli F5/MF1 (`_batch_row`, `_output_analyze_result`, invert stdout) | SAFE-NOW | When `CFLIBS_REFUSE_TO_REPORT=1` and `overall_reliable=False`, all output paths (batch CSV, analyze CSV/table/JSON, invert stdout) omit/blank composition; tests for each path. | — | S |
| M4-2 | Gate R-hat/ESS convergence in distributed MCMC | hpc 3/11 (`distributed_mcmc.py:247`) | SAFE-NOW | `DistributedMCMCResult` carries a machine-readable `converged` flag (R̂>1.05 or ESS<50 → False) + warning; test injects non-converged chains. | — | S |
| M4-3 | FastAnalyzer stops claiming `converged=True` on placeholder | inv-runtime F2 (`streaming.py:710/785`) | SAFE-NOW | Placeholder / n_e-not-determined paths set `converged=False` (or raise); test asserts no misleading convergence; downstream quality assessment sees it. | — | S |
| M4-4 | HPC GPU pinning + docstring/CLI honesty | hpc 2/10 (`run()` uses global rank), A (`python -m` no `__main__`) | SAFE-NOW | `run()` calls `pin_to_device(local_rank)`; add real `__main__`/argparse or fix docstring; multi-node modulo-wrap test. | — | S |

### BENCHMARK-GATED forward-physics fixes (change generated spectra / recovered values)

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M4-5 | Add `hc/λ` to round-trip + simplified forward intensity | validation F1 (`round_trip.py:309`), benchmark F3 (`synthetic.py:604` omits U_s(T)) | BENCHMARK-GATED | Forward intensity uses `hc/λ · gA·exp(-E/kT)/U`; simplified model includes partition function; M0-3 self-consistency + round-trip accuracy improve; intensity-scale clamp floor (validation New-B) updated in same pass; benchmark-gated since recovered comps shift. | M0-3 | M |
| M4-6 | Fix temporal optical-depth formula | inv-runtime F1/C1 (`temporal.py:1014/1025`: g_lower, λ³→λ², 1e-25 magic, mass/number) | BENCHMARK-GATED | τ formula matches self-absorption-cog.md §1.4 (λ², lower-level g/E, derived prefactor); `f_tau→1` for thin lines test (T1); behind flag; benchmark on SA-affected datasets. | M0-3 | M |
| M4-7 | FastAnalyzer concentration proxy uses closure, not `exp(avg_y)` | inv-runtime A1 (`streaming.py:767`) | BENCHMARK-GATED | Concentration via `physics.closure`, not raw `exp(avg_y)`; benchmark on streaming path; flag-guarded. | M3-1 | M |
| M4-8 | Tighten validation/NIST-parity test tolerances | validation 10 (50%→config), 11 (65–70% U, 150% f) | BENCHMARK-GATED-adjacent | Round-trip tests assert `result.passed` at configured tol; NIST parity tolerances tightened with documented per-line exceptions; runs after M3/M4-5 land (they make tight tolerances achievable). | M3-3, M4-5 | M |

---

# MILESTONE M5 — Identifier & solver accuracy (BENCHMARK-GATED) + DESIGN-DECISIONS

Target: identifier decision gates match the source papers without regressing F1; solver flag
incompatibilities are resolved or warned. **Memory warns identifier-scoring changes have regressed
F1 before — every item here is benchmark-gated and flag-guarded.**

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M5-1 | Regression test pinning 2-line element detection | inv-identify F9, F2 chain | SAFE-NOW | Test asserts a 2-line element with `k_det>threshold` keeps `confidence>0` AND `detected=True`; currently documents the bug. | M0-2 | S |
| M5-2 | Decide N_matched<3 CL-zeroing gate | inv-identify F2 (`alias.py:4682`→`_gate_relative_cl` demotes detected) | DESIGN-DECISION | A/B on M0-2 ID benchmark: keep gate vs paper-faithful remove. Decision + benchmark numbers committed; if removed, behind flag default per benchmark winner. (Paper supports 1-line/sparse; but prior paper-faithful ALIAS change regressed F1 −0.041 — must measure.) | M5-1, M0-2 | M |
| M5-3 | `_compute_ratio_consistency` neutral return 0.1→0.5 | inv-identify F3 (`alias.py:4430` vs docstring) | BENCHMARK-GATED | Align code to documented neutral 0.5; ID benchmark non-regression; flag-guarded. | M5-1 | S |
| M5-4 | WLS sigma-clip uses weighted std | inv-physics F8 (`boltzmann.py:428`) | BENCHMARK-GATED | Outlier threshold uses inverse-variance-weighted residual std (CPU + JAX paths); composition/T benchmark non-regression; ~1–5% T-bias cases improve. | M0-2 | S |
| M5-5 | SA doublet `find_doublet_pairs` enforce g_k precondition | inv-physics F3 (`self_absorption.py:480`) | BENCHMARK-GATED | Match requires same-upper-level (g_k equality or shared upper) within dE window; SA benchmark non-regression; unit test on near-degenerate-J case. | M0-3 | S |
| M5-6 | H-alpha Stark n_e Gigosos n_e^0.7 path | inv-physics F4 (`stark_ne.py:46`) | BENCHMARK-GATED | Optional Gigosos power-law correction for H-alpha (flag); n_e diagnostic benchmark vs linear on H-containing data; default decided by benchmark. | M0-2 | M |
| M5-7 | Lax-path n_e / Stark + SB-graph incompatibility | inv-solve F2/F6/F10 (`iterative.py:1617`) | DESIGN-DECISION | Decide: (a) wire Stark/SB-graph into lax kernel, or (b) emit explicit warning + document the silent Python-fallback. Parity test SB-graph True vs False (M0-3). Decision + measurements committed. | M0-3 | L |
| M5-8 | `two_region` 0.8/0.3-0.7 magic numbers | inv-solve F3 (`iterative.py:1888`) | DESIGN-DECISION | Either ground in literature + benchmark, or keep strictly opt-in with a documented "unvalidated" warning. Decision recorded; default stays OFF. | M0-2 | M |
| M5-9 | Bayesian Poisson likelihood default | inv-solve F4 (`bayesian/likelihood.py`) | BENCHMARK-GATED | Evaluate Cash/Poisson vs Gaussian default on shot-noise-dominated benchmark; promote if non-regressing; flag-guarded. | M0-2 | M |

---

# MILESTONE M6 — Manifold & generated-data consistency (BENCHMARK-GATED + SAFE-NOW)

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M6-1 | Manifold attrs completeness | manifold F7/F9/MF1/MF2 (gate_delay/width, cooling params, `broadening_mode` not stored; vestigial use_voigt/use_stark) | SAFE-NOW | Attrs block writes all time-integration + the *active* `broadening_mode`; vestigial flags removed or documented; loader round-trip test. | M0 | S |
| M6-2 | Apply `gate_delay_s` in time integration | manifold F2 (`generator.py:986/1055` start at t=0) | BENCHMARK-GATED | Time grid starts at `gate_delay_s`; flag-guarded; manifold-lookup accuracy benchmark; LDM/Voigt parity (M0-3, manifold F10). | M0-3, M6-1 | M |
| M6-3 | Stark T-scaling in `batch_forward` | manifold F3 (`batch_forward.py:447` no T^-α) | BENCHMARK-GATED | Add `T^(-α)` factor to match `generator.py:659`; parity test across the 3 forward impls (manifold F5); benchmark-gated. | M0-3 | S |
| M6-4 | Consolidate / parity-bind the 3 forward physics impls | manifold F5, radiation F3 | DESIGN-DECISION | Decide whether to unify generator/batch_forward/basis_library or keep separate-with-parity-tests. Measured: parity tests (M0-3) + maintenance cost. Decision recorded; if unify, behind benchmark. | M6-2, M6-3 | L |

---

# MILESTONE M7 — Performance (latency LAST, per mission priority)

All deferred behind correctness; each must hold a parity test (no accuracy change) and ideally show
a measured win. Profiling memory says RANSAC wl-calibration dominates (73%) and GPU jit was slower —
so these are scoped, not speculative.

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M7-1 | Vectorize numpy per-line Gaussian / Voigt loops | radiation F5 (`profiles.py:154/336`) | SAFE-NOW | numpy broadcasting; bit-parity test vs loop; measured speedup on CPU path. | M0-3 | S |
| M7-2 | `detect_ccd_seams` → `scipy.ndimage.median_filter` | inv-preprocess F4 (`wavelength_calibration.py:1302`) | SAFE-NOW | Drop-in C median filter; output-parity test; measured speedup. | — | S |
| M7-3 | `_apply_saha_correction` precompute arrays | inv-solve F8/FA2 (`iterative.py:1338`) | SAFE-NOW | No per-iteration `LineObservation` allocation; result-parity test; measured GC reduction. | M1-1 | S |
| M7-4 | Cache-key avoid pickling `self` on hot path | atomic F7, core MF1 | SAFE-NOW | Key by `id(self)`/db-sha prefix; parity test on cached results; pickle calls removed from hot path. | M1-9 | S |
| M7-5 | Misc loop vectorizations | atomic F4 (iterrows), instrument F7/F6, io F6, inv-preprocess F9, manifold F8 | SAFE-NOW | Each: output-parity test + measured win; only land where benchmark/profiler shows it matters. | — | M |
| M7-6 | `solve_with_uncertainty` avoid redundant PF/Saha recompute | inv-solve F7 (`iterative.py:2760`) | SAFE-NOW | Reuse converged state; result-parity test; measured win in batch mode. | M1-1 | M |

### SAFE-NOW housekeeping (any time, low risk)

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| M7-7 | Unify env-var flag resolvers / truthy parsing | inv-solve F9/FA1, inv-preprocess F7, core (flag debt) | SAFE-NOW | One `_feature_flag(name)` helper, consistent `{1,true,yes,on}`; test matrix; `CFLIBS_USE_LAX_WHILE_LOOP=true` now works. | — | S |
| M7-8 | Sub-pixel centroiding + normalized RANSAC fit coords | inv-preprocess F1/F2 (`wavelength_calibration.py:192`, `preprocessing.py:617`) | BENCHMARK-GATED | Parabolic 3-pt centroid + [-1,1]-normalized polynomial fit; behind flag; wl-calibration RMS + composition benchmark (RANSAC is RNG-coupled — must benchmark-gate). | M0-2 | M |
| M7-9 | `min_intensity_floor` semantics + exporter dtype/JSON | inv-preprocess F3, io F7 (S10 truncation), io M2 (allow_nan Inf) | SAFE-NOW | Floor cannot lower below noise criterion (or renamed); HDF5 element names use `h5py.string_dtype()`; `allow_nan=False` applies to Inf; tests. | — | S |

---

# Excluded (verified FALSE — do NOT act on these)

- inv-solve **F1**: adding `ln(U_II/U_I)` to ionic y-shift — would INTRODUCE a bug (term cancels).
- inv-identify **F1/F6/F10**: y-axis consistency, JAX parity tests, interference test — already correct.
- inv-top **F1**: full-spectrum mole→mass already done in `solve_full_spectrum` (mass returned).
- io **F5/F10**: `intensity_W_m2_nm_sr` IS in the alias list; round-trip works.
- validation **F2** (`1+avg_Z` denominator correct) and **F3-original** (spin factor already in const).
- inv-physics **F2**: pLTE E_lower cut is the correct Cristoforetti implementation.
- plasma **F3** downgraded LOW (production path bypasses the Python for-loop) — fold into M7-5 only if profiled.

---

# FIRST 10 ITEMS TO EXECUTE (ordered)

1. **M0-1** — Pin canonical Saha/IPD/DH constants (unblocks plasma/core/atomic fixes; tiny, green now).
2. **M0-2** — Stand up held-out composition benchmark wrapper + baseline (the gate for ALL accuracy items).
3. **M0-3** — Add dual-path parity fixtures (forward/LDM/SB-graph/Rust-comb), xfail-marking known divergences.
4. **M1-8** — Fix `None`-sentinel cache miss (SAFE-NOW, high-value, isolated; removes silent DB re-query).
5. **M1-6** — Correct Olivero Voigt constants to one source of truth (SAFE-NOW, parity-verified).
6. **M1-1** — Move `CFLIBSResult` to `inversion/common/` (unblocks M1-3, M7-3, M7-6 refactors).
7. **M1-2** — Redirect all `LineObservation` imports to `common.data_structures` (kills the shim cycle).
8. **M1-20** — Evolution blocklist: catch `exec`/`eval`/`compile` + binary-diff (physics-only-constraint hardening; SAFE-NOW).
9. **M3-1** — Build the canonical number↔mass converter (SAFE-NOW; prerequisite for the top accuracy lever M3).
10. **M2-1** — Resonance-ΔE regression test (SAFE-NOW; sets up the McWhirter physics correction M2).

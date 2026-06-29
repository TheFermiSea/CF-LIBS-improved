# CF-LIBS Overhaul — Blueprint Addendum

**Author:** Lead Architect (addendum pass)
**Date:** 2026-06-25
**Purpose:** Fold all 16 CRITIC gaps into properly-classified, dependency-ordered milestone items
in the same format as BLUEPRINT.md. Cross-reference to ARCHITECTURE.md §§ where applicable.

Classification legend same as BLUEPRINT.md:
- **SAFE-NOW** — behavior-preserving; acceptance = green targeted test
- **BENCHMARK-GATED** — changes scoring output; ships flag-default-OFF until gate passes
- **DESIGN-DECISION** — requires measured comparison before one option is chosen
- **AUDIT** — no code change until convention/invariant confirmed by rg + Read

---

## M0 Additions — Test & benchmark scaffolding (enable everything else)

These items join M0 in the blueprint because they are gate infrastructure, not late refactors.
They must land before any benchmark-gated accuracy item can be verified.

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M0-A1 | `RoundTripValidator` solver-injection seam | validation F5 (`round_trip.py:779`): hardwired `IterativeCFLIBSSolver`, no `solver_factory` param | SAFE-NOW | Constructor gains `solver_factory: Callable[..., SolverStrategy] = None` (default = iterative); `_validate()` uses it; test passes `BayesianSolver` factory; no behavior change with default. | M0 | S |
| M0-A2 | ID F1/recall benchmark harness | benchmark F5 (gap: M0-2 names composition metrics only), CRITIC GAP-16 | SAFE-NOW | `scripts/benchmark_synthetic_identifiers.py` wrapped behind the same command interface as M0-2; emits `id_f1`, `id_recall`, `id_precision` JSON alongside composition metrics; committed baseline snapshot; used as gate for all identifier-scoring items (M5-2/M5-3, comb fix). | M0-2 | M |
| M0-A3 | LOD estimator: document deviation + define defensible estimator | benchmark F2 (`metrics.py:550`): LOD = 3·std(bottom-25th-percentile), not 3σ_blank | DESIGN-DECISION | Decide: (a) add `blank_spectra` concept to `BenchmarkSpectrum` and compute LOD from certified blanks; (b) document the deviation from IUPAC ISO 11843 and use it only for relative ranking. Decision + comment + test; `traces` stratum gate re-evaluated. | M0-2 | M |

---

## M1 Additions — SAFE-NOW correctness (no benchmark needed)

These items were HIGH verified-REAL findings absent from the original M1.

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M1-A1 | SCCT5/SCCT7 K-dropped copy-paste data fix | pds F1-2 (`corpus.py:233–253`): SCCT5 copies CCCT3 and SCCT7 copies CCCT2 with K silently dropped | SAFE-NOW | Verify SCCT5/SCCT7 K weight-fraction against ChemCam published target compositions (Fabre 2011 or superseding reference); add `"K": value` with citation comment; test asserts SCCT5["K"] is present and non-zero (or explicitly documents why K below LOD for SuperCam). | — | S |
| M1-A2 | `instrument.convolution.apply_instrument_function` sigma=0 guard | instrument F10 (`convolution.py:44`): sigma=0 → division by zero → NaN output; no guard | SAFE-NOW | Guard: `if sigma_nm <= 0: return intensity.copy()` (matches `SpectrumModel` guard at `spectrum_model.py:280`); test: `apply_instrument_function(wl, i, 0.0)` returns `i` unchanged, no NaN. | — | S |
| M1-A3 | Evolution: wire or remove `enforcement_mode`; add `builtins.exec` scan | evolution F2.1 (`config.py` `enforcement_mode` field has no driver consumer) + evolution NEW-A (`builtins.exec` attribute form bypasses scanner) | SAFE-NOW | Either (a) remove `enforcement_mode` field from `EvolutionDriverConfig` (safe: no driver reads it); or (b) add a concrete `_check_enforcement_mode(config)` function that raises `NotImplementedError` if `mode=="warn"` and no driver registers a callback — make the fiction explicit. Also add `builtins.exec` as attribute-chain pattern to `_scan_attribute` path; add test. Pairs with M1-20 (exec/eval/compile scanner). | M1-20 | S |
| M1-A4 | HPC SBATCH heredoc: stop hardcoding bayesian import path | hpc F4 (`slurm.py:655`): `from cflibs.inversion.solve.bayesian import ...` hardcoded as string literal in generated script | SAFE-NOW | Import path built from a module-level constant `_BAYESIAN_IMPORT_PATH = f"{BayesianForwardModel.__module__}"` resolved at generation time; test: generated script string contains the runtime-resolved import path, not a hardcoded literal; change is invisible to script consumers (same string today, auto-updates on refactor). | M1-1 | S |
| M1-A5 | HPC: fix `python -m cflibs.hpc.distributed_mcmc` dead CLI | hpc NEW-A (`distributed_mcmc.py:11–14`): module docstring advertises `python -m` usage; no `__main__` block exists | SAFE-NOW | Add `if __name__ == "__main__":` block with `argparse` covering `--db-path`, `--elements`, `--wl-range`; or change docstring to remove the advertised usage and redirect to SLURM script generation. A SLURM job following the docstring currently exits silently with code 0. Test: `python -m cflibs.hpc.distributed_mcmc --help` exits 0 and prints usage. | — | M |
| M1-A6 | NoiseModel: shot noise on clean signal; variance from params | validation F3-revised (`round_trip.py:588–615`): shot noise applied to signal+background, variance from noise-contaminated realization | SAFE-NOW | Fix sequence: (1) compute `var_shot = max(clean_signal, 0)`; (2) add background after variance is set; (3) use `var_from_model = var_shot + sigma_readout**2`, not the realized intensity, as uncertainty estimate. Parity test: `NoiseModel` with zero background and zero readout returns `intensity_uncertainty == sqrt(intensity)`. Pairs with M4-5 (hc/λ fix). | M0-3 | M |

---

## M2 Additions — IPD/PF truncation consistency audit (extends M0-1)

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M2-A1 | IPD/PF truncation consistency invariant test + single compute_ipd() | CRITIC GAP-12; saha-boltzmann-lte literature §P2 (CRITICAL pitfall); radiation F3 (two diverging population paths) | AUDIT then SAFE-NOW | Phase 1 (AUDIT): confirm `partition.py` truncation cutoff uses the same Δχ as `saha_boltzmann.py` Saha exponent. Run `rg -n "ipd\|delta_chi\|continuum_lower\|chi_eff" cflibs/` and read both sites. Phase 2 (SAFE-NOW if inconsistency confirmed): introduce `compute_ipd(n_e_cm3, T_eV, model) -> float` in `cflibs/plasma/` as the single call site; thread Δχ to both Saha exponent and PF truncation explicitly; invariant test: `pf_cutoff == chi - ipd(n_e, T)` for both polynomial and direct-sum paths across the two forward-physics paths (radiation F3). | M0-1 | M |

---

## M3 Additions — Mole↔mass accuracy items

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M3-A1 | Clarify M3-2 baseline semantics: old baseline is known-wrong | CRITIC GAP-14; benchmark F1 (CRITICAL: Völker 353% error); BLUEPRINT.md M3-2 | — (framing fix, not a code item) | M3-2 acceptance must state explicitly: a moved composition number is the **fix landing**, not a regression. The benchmark gate compares against ground-truth (certified mass fractions), not against the old (wrong) number-fraction baseline. The old baseline must be retired when M3-2 lands. Document in the M3-2 PR description. | M3-2 | — |

---

## M4 Additions — Reliability + Bayesian noise

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M4-A1 | Bayesian likelihood normalization + variance model audit | CRITIC GAP-13; bayesian-oe literature §2.4/2.5 | AUDIT then BENCHMARK-GATED | Audit `bayesian/likelihood.py`: (1) confirm `log(σ_k²)` normalization term is present or absent; (2) confirm variance is signal-dependent or fixed. `rg -n "log.*sigma\|normali" cflibs/inversion/solve/bayesian/`. If normalization is dropped for BIC/WAIC comparisons (model selection in `inv-identify`), flag as accuracy issue. Fold audit results into M5-9 acceptance criteria. | M5-9 | S |
| M4-A2 | Benchmark A2: silent simplified-model fallback sets TruthType flag | benchmark A2 (`synthetic.py:483–490`): silent `except Exception` falls through to simplified model with no `TruthType.SIMULATED_UNPHYSICAL` flag | SAFE-NOW | On any exception in `_generate_with_forward_model`, set `truth_type=TruthType.SIMULATED_UNPHYSICAL` (or a new `SIMULATED_FALLBACK` variant) before calling `_generate_simplified`; benchmark harness must filter or warn on `SIMULATED_UNPHYSICAL` spectra; test asserts fallback sets the flag. | M4-5 | S |

---

## M5 Additions — Identifier accuracy + NUTS warm-start + Rust comb

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M5-A1 | Warm-start NUTS from classical MAP at all 3 MCMC call sites | hpc F1 (`distributed_mcmc.py:205`, `bayesian/samplers.py`, `two_zone.py`): all use `init_to_uniform(radius=0.5)` | BENCHMARK-GATED | Replace `init_to_uniform` with `init_to_value(values=map_estimate)` at all 3 sites; `map_estimate` computed from `IterativeCFLIBSSolver` or passed in; flag-guarded (`CFLIBSConfig.nuts_init_strategy`); Bayesian convergence benchmark: R̂ < 1.05 for ≥ 95% of test cases with warm-start vs. cold-start; benchmark JSON committed. Addresses literature mandate bayesian-oe §2.7. | M0-A2, M5-9 | M |
| M5-A2 | Align Rust↔Python comb greedy orientation + tie-break epsilon | native-rust Missed-A (`comb_matching.rs:14–52` peaks-outer vs. `line_detection.py:1950` transitions-outer); Missed-B (epsilon 1e-12 vs. np.isclose) | BENCHMARK-GATED | Fix Rust to use transitions-outer greedy (matching Python semantics); align F1 tie-break epsilon; ID benchmark (M0-A2) non-regression; flag-guarded during benchmark validation; M0-3 parity fixture flipped from xfail to assert (per GAP-15 rule). | M0-3, M0-A2 | M |
| M5-A3 | inv-common: add `intensity_uncertainty` to `IdentifiedLine`; propagate through identifiers | inv-common F12 (`element_id.py:24–62`: `IdentifiedLine` has no `intensity_uncertainty` field) + F2 (`to_line_observations` drops `aki_uncertainty`) + F3 (`y_uncertainty` incomplete) | SAFE-NOW (slot addition) then BENCHMARK-GATED (uncertainty propagation) | Phase 1 (SAFE-NOW): add `intensity_uncertainty: float = 0.0` to `IdentifiedLine`; update `to_line_observations` to forward it + `aki_uncertainty` (F2 fix); add test for F10 (currently zero coverage). Phase 2 (BENCHMARK-GATED): thread real σ_I from preprocessing noise estimate; benchmark uncertainty-accuracy metric (see M0-3 fixtures). | M1-2 | M |

---

## M6 Additions — Manifold and forward model xfail→assert rule

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M6-A1 | Standing rule: flip xfail→assert in M0-3 fixtures when fix lands | CRITIC GAP-15: M0-3 xfail-marks known divergences; no item explicitly owns the flip | — (process rule) | Every fix item that resolves an M0-3 xfail (M6-2/M6-3/M6-4, M5-7, M5-A2) MUST include "flip xfail→assert" in its acceptance criteria. Add this as a comment to M0-3's description in BLUEPRINT.md. Risk: a fix lands, the xfail silently keeps passing as xfail, and parity is never enforced. | M0-3 | — |

---

## M7 Additions — Housekeeping sweep (lower priority)

| id | title | addresses (file:line) | class | acceptance | deps | effort |
|----|-------|-----------------------|-------|------------|------|--------|
| M7-A1 | HPC housekeeping sweep | hpc F5 (mutable default `DistributedMCMCConfig()` in `__init__`), F6 (duplicate SBATCH header logic), F7 (`comm.gather` pickle), F8 (sequential int PRNG → `random.split`), F9 (hardcoded `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`), F10 (no `pin_to_device` test), F11 (no R-hat convergence test) | SAFE-NOW | Address all in one PR: freeze config default; extract `_generate_sbatch_header()` helper; Gatherv for numpy arrays; `random.split` for PRNG; expose `gpu_memory_fraction` in config; add modulo-wrap + R-hat tests. | M4-2, M4-4 | M |
| M7-A2 | Evolution doc + config housekeeping | evolution F1.2/2.2/4.1/4.2/5.2 (doc/config debt): CLAUDE.md contradiction, `fitness_weights` no aggregation formula, `evaluation_timeout_s=5.0`, `structural_mutation_cadence=10`, straggler test file | SAFE-NOW | Correct CLAUDE.md to "ML is allowed only in the external evolution driver that calls cflibs/evolution/"; add aggregation formula docstring; update `evaluation_timeout_s` default to 60.0 (matches minimum real inversion wall-clock); document cadence=10 citation or change; move straggler test file to `tests/evolution/`; add pytest marks. | M1-A3 | S |
| M7-A3 | inv-common: `get_wavelength_tolerance` instrument FWHM fix + `y_uncertainty` doc | inv-common F1 (`element_id.py:245–250`): fallback discards computed `fwhm_inst`; F3 (`y_uncertainty` underestimates; does not add `aki_uncertainty`) | SAFE-NOW | F1: `return max(fwhm_inst, fallback)` (one line fix; function is test-utility only, not production). F3: add `aki_uncertainty` in quadrature to `y_uncertainty` property and update docstring. Tests for both. | M1-2 | S |
| M7-A4 | `InstrumentModelJax` pytree round-trip fix | instrument F11 (`jax_runtime.py:685`): unflatten reconstructs base `InstrumentModel`, not subclass; no round-trip test | SAFE-NOW | Register `InstrumentModelJax` separately in pytree; unflatten uses the correct class; add `__post_init__` call or store pre-staged arrays as leaves; round-trip parity test. | M1-4, M1-16 | M |
| M7-A5 | Benchmark: consolidate `CFLIBS_FF_*` inline flags into `CFLIBSConfig` | benchmark F6 (`synthetic_eval.py:536–543`): 8 inline `os.environ.get()` calls | SAFE-NOW | Move 8 flags into `CalibrationOptions` dataclass or `CFLIBSConfig`; test: flags injectable without environment mutation. | M7-7 | S |

---

## Audit Items (literature pitfalls never checked against code)

| id | title | addresses | class | acceptance | deps | effort |
|----|-------|-----------|-------|------------|------|--------|
| AU-1 | Air↔vacuum wavelength convention audit | CRITIC GAP-11; broadening-rt literature §1.1 (convention must be explicit); cflibs-method literature | AUDIT | Run `rg -ni "vacuum|air.?wavelength|n_air|edlen|ciddor" cflibs/`; read `datagen_v2.py` and `ingest_kurucz_atomic.py` for provenance headers; check `line_detection.py` tolerance computation convention. Report: what convention does DB store? what does line-matching tolerance assume? Are they consistent? If inconsistent, classify fix as BENCHMARK-GATED (changes identification). Document convention in ARCHITECTURE.md §6 (already drafted). | — | S |
| AU-2 | IPD/PF truncation consistency audit | CRITIC GAP-12 (see M2-A1 above) | AUDIT | Subsumed into M2-A1 Phase 1. | — | — |
| AU-3 | Bayesian likelihood normalization + variance-model audit | CRITIC GAP-13 (see M4-A1 above) | AUDIT | Subsumed into M4-A1. | — | — |

---

## Classification / Ordering Corrections (process items, no code)

These are documentation or acceptance-criteria amendments to existing BLUEPRINT items.

| id | title | addresses | note |
|----|-------|-----------|------|
| CO-1 | M3-2 framing: old baseline is known-wrong | CRITIC GAP-14 | See M3-A1 above. The classification (BENCHMARK-GATED) is correct; only the acceptance framing needs the clarification. |
| CO-2 | M0-3 standing xfail→assert rule | CRITIC GAP-15 | See M6-A1 above. Add as comment to M0-3 in BLUEPRINT.md. |
| CO-3 | M0-2 must emit ID F1 benchmark | CRITIC GAP-16 | See M0-A2 above. M0-2 extended or M0-A2 added as sibling. |

---

## Item count by class

| class | items |
|-------|-------|
| SAFE-NOW | M0-A1, M1-A1, M1-A2, M1-A3, M1-A4, M1-A5, M1-A6, M2-A1 (phase 2), M4-A2, M5-A3 (phase 1), M7-A1, M7-A2, M7-A3, M7-A4, M7-A5 | 15 |
| BENCHMARK-GATED | M0-A3 (option a), M3-A1 (framing), M4-A1 (if gap found), M5-A1, M5-A2, M5-A3 (phase 2) | 6 |
| DESIGN-DECISION | M0-A3 (selecting LOD estimator), DA-1, DA-2, DA-3 (from ARCHITECTURE.md) | 4 |
| AUDIT | AU-1, M2-A1 (phase 1), M4-A1 (phase 1) | 3 |
| Process/framing | CO-1, CO-2, CO-3, M6-A1, M3-A1 (note) | 5 |

Total new items: **28** (including 5 process/framing items that require no separate PR).

---

## Revised First 12 Items to Execute

This merges the addendum M0 additions (M0-A1, M0-A2) and the essential SAFE-NOW items (M1-A1,
M1-A2) into the blueprint's original first-10, with ordering adjusted by dependency and impact.

| rank | id | title | rationale |
|------|----|-------|-----------|
| 1 | M0-1 | Pin Saha/IPD/DH constants | Unblocks plasma/core fixes; tiny, green now |
| 2 | M0-2 | Held-out composition benchmark wrapper + baseline | Gate for all BENCHMARK-GATED accuracy items |
| 3 | M0-A2 | ID F1/recall benchmark harness | Gate for all identifier-scoring items (M5-2/M5-3/M5-A2) |
| 4 | M0-A1 | `RoundTripValidator` solver-injection seam | Enables Bayesian/joint/closed-form regression gating; should be M0, not a late refactor |
| 5 | M0-3 | Add dual-path parity fixtures (forward/LDM/SB-graph/Rust-comb), xfail-marking divergences + add standing xfail→assert rule (CO-2) | Foundation for all forward-model accuracy fixes |
| 6 | M1-A2 | `sigma=0 → NaN` guard in `apply_instrument_function` | One-line SAFE-NOW crash prevention; unblocks instrument tests |
| 7 | M1-8 | Fix `None`-sentinel cache miss | High-value SAFE-NOW; removes silent DB re-query |
| 8 | M1-6 | Correct Olivero Voigt constants → single source of truth | SAFE-NOW, parity-verified, unblocks M0-3 parity test |
| 9 | M1-1 | Move `CFLIBSResult` to `inversion/common/` (or `domain/`) | Unblocks M1-3, M7-3, M7-6 refactors |
| 10 | M1-2 | Redirect all `LineObservation` imports to canonical path | Kills the shim cycle; unblocks M5-A3 |
| 11 | M1-A1 | SCCT5/SCCT7 K-dropped data fix | SAFE-NOW data truth; corrupts any benchmark using those targets |
| 12 | M3-1 | Canonical number↔mass fraction converter | SAFE-NOW; prerequisite for the top accuracy lever M3 |

Items 13+ follow the original BLUEPRINT ordering (M1-20, M2-1, M2-2, ...).

---

## Traceability: GAP → Addendum item

| GAP | follow-up | item(s) |
|-----|-----------|---------|
| GAP-0 | ARCHITECTURE.md created | ARCHITECTURE.md |
| GAP-1 | hpc F1 NUTS warm-start | M5-A1 |
| GAP-2 | validation F3-revised noise circularity | M1-A6 |
| GAP-3 | validation F5 solver-injection seam → M0 | M0-A1 |
| GAP-4 | benchmark F2 LOD blank proxy | M0-A3 |
| GAP-5 | evolution F2.1 enforcement_mode footgun | M1-A3 |
| GAP-6a | hpc F4 hardcoded import path | M1-A4 |
| GAP-6b | hpc NEW-A dead CLI | M1-A5 |
| GAP-6c | native-rust F2 dead kdet branch | ARCHITECTURE.md §9 + M5-A2 (parity) |
| GAP-7 | inv-preprocess F5 monolith + F6 outliers | ARCHITECTURE.md §7 (design); M1 split item deferred to M5 window |
| GAP-8 | radiation F2 LEGACY default + F4 resolving-power | ARCHITECTURE.md §2.3 + DA-2 |
| GAP-9 | pds F2-2 dead-end schema + F1-2 K-dropped | ARCHITECTURE.md §8 + M1-A1 |
| GAP-10 | native-rust Missed-A/B comb orientation | ARCHITECTURE.md §9 + M5-A2 |
| GAP-11 | air/vacuum wavelength convention | ARCHITECTURE.md §6 + AU-1 |
| GAP-12 | IPD/PF truncation consistency | ARCHITECTURE.md §5 + M2-A1 |
| GAP-13 | Bayesian likelihood normalization/variance | ARCHITECTURE.md §4.1 + M4-A1 |
| GAP-14 | M3-2 framing | M3-A1 + CO-1 |
| GAP-15 | xfail→assert standing rule | M6-A1 + CO-2 |
| GAP-16 | ID F1 benchmark missing from M0-2 | M0-A2 + CO-3 |
| GAP-E (instrument F10) | sigma=0 NaN | M1-A2 |
| GAP-E (inv-common F1/F2/F3) | uncertainty propagation | M5-A3 + M7-A3 |
| GAP-E (benchmark A2) | silent simplified fallback | M4-A2 |
| GAP-E (hpc housekeeping) | F5–F11 | M7-A1 |
| GAP-E (evolution housekeeping) | F1.2/2.2/4.1/4.2/5.2 | M7-A2 |
| GAP-E (instrument F5 response curves) | 3 impl consolidation | ARCHITECTURE.md §2.2 + DA-1 |
| GAP-E (plasma F6) | TwoRegionPlasma architecture | ARCHITECTURE.md §10 |

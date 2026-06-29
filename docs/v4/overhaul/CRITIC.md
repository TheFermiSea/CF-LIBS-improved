# CRITIC — Completeness review of BLUEPRINT.md

**Reviewer:** Completeness critic
**Date:** 2026-06-25
**Inputs:** `blueprint/BLUEPRINT.md` (only blueprint file present; `blueprint/ARCHITECTURE.md`
does NOT exist — see Gap-0), all 21 `verified/*.md` reports, `literature/*.md` checklists,
`reference/*.md`.
**Method:** ripgrep + Read. For every verified-REAL finding I checked whether a blueprint
milestone item addresses it (by file:line / theme, since the blueprint references findings by
per-file ID). Below are concrete MISSING / WRONG items with the follow-up that closes each.

The blueprint is strong on the two headline accuracy levers (M2 McWhirter ΔE, M3 mole↔mass) and
on the SAFE-NOW layering/constants work. The gaps are concentrated in: (a) **HIGH-severity
verified-REAL findings that were silently dropped** (not in the FALSE-exclusion list, just absent),
(b) **architecture/test-gap findings with no home** (because there is no ARCHITECTURE.md), and
(c) **literature-checklist items never investigated by the census**.

---

## GAP-0 (process) — The promised ARCHITECTURE.md does not exist

The blueprint header line 10 admits "No `blueprint/ARCHITECTURE.md` was present; the target
architecture is inferred ... inline per milestone." The orchestrator instructed me to read it too;
it is genuinely absent. Consequence: there is **no target-state diagram**, so every architecture
finding (monolith split, domain-type homes, dead-end schemas, duplicate-impl consolidation) has no
anchor and several were dropped (Gaps 7–10). Inline per-milestone architecture is fine for the
domain-type moves in M1, but it left the *non-domain-type* architecture findings homeless.

**Follow-up:** one architecture-synthesis agent to produce ARCHITECTURE.md (target module map +
the 3 "consolidate vs parity-bind" DESIGN-DECISIONS already named: radiation/manifold forward
impls M6-4, instrument response curves, Rust↔Python comb). Then re-slot Gaps 7–10 against it.

---

## A. HIGH-severity verified-REAL findings MISSING from the plan

These are all confirmed REAL at HIGH severity in `verified/` and are **not** in the FALSE-exclusion
list at the end of the blueprint — they were simply not carried into any milestone.

### GAP-1 — hpc Finding 1: NUTS `init_to_uniform` instead of MAP warm-start (HIGH, systemic ×3 sites)
`verified/hpc.md` F1: all three MCMC call sites (`distributed_mcmc.py:205`,
`inversion/solve/bayesian/samplers.py`, `two_zone.py`) use `init_to_uniform(radius=0.5)`.
Literature `bayesian-oe.md §2.7 / NUTS config` mandates `init_to_value` from the classical-solver
MAP. The CF-LIBS posterior is exponential in T → uniform init can land off posterior mass →
unconverged chains / wrong composition. This is an **accuracy/reliability** finding (top of mission
priority), not latency, yet it is absent. The blueprint only has M5-9 (Poisson likelihood) and M4-2
(R-hat gating) for the Bayesian path — neither fixes initialization.
**Follow-up:** add an M5 item "warm-start NUTS from classical MAP at all 3 sites (BENCHMARK-GATED on
the Bayesian held-out set; convergence-rate + posterior-accuracy metric)". Search:
`rg -n "init_to_uniform|init_to_value|init_strategy" cflibs/`.

### GAP-2 — validation Finding 3-revised: NoiseModel shot-noise circularity (HIGH)
`verified/validation.md` F3-revised (REAL/HIGH): `round_trip.py:588-615` adds background *before*
shot noise, applies Poisson to signal+background, then uses the noise-realized intensity as its own
variance estimate (circular). The blueprint's exclusion list drops F2 and **F3-original** as FALSE,
but **F3-revised is a different, REAL, HIGH finding** and reads as if it were excluded by association.
It directly corrupts the uncertainty/reliability signal that M4 depends on.
**Follow-up:** add to M4 (forward-physics) "fix NoiseModel: shot noise on clean signal pre-background;
variance from noise-model params not the realization". Pairs naturally with M4-5/M4-8.

### GAP-3 — validation Finding 5: no solver-injection seam in RoundTripValidator (HIGH)
`verified/validation.md` F5 (REAL/HIGH): `round_trip.py:779` hardwires `IterativeCFLIBSSolver` via
deferred import; `RoundTripValidator.__init__` takes no `solver_factory`. The validator therefore
**cannot gate regressions in the Bayesian / joint / closed-form solvers** — exactly the solvers M5
and M4 modify. This is an enabling gap for the whole benchmark-gated program.
**Follow-up:** SAFE-NOW item (constructor `solver_factory` param, default to iterative) — should be
in **M0** alongside M0-2 because it is gate infrastructure, not a late refactor.

### GAP-4 — benchmark Finding 2: LOD uses 25th-percentile as blank proxy (HIGH)
`verified/benchmark.md` F2 (REAL/HIGH): `metrics.py:550` computes LOD = 3·std(residuals of bottom
25%) — not 3σ_blank. Inflates LOD and **misfires the `traces` stratum gate** (a reliability gate).
Completely absent from the blueprint (M0-2 stands up the harness but does not touch LOD).
**Follow-up:** M0-2 or a new M0/M4 item: document the deviation + decide a defensible LOD estimator
(needs a blank/zero-concentration concept in `BenchmarkSpectrum`). DESIGN-DECISION flavour.

### GAP-5 — evolution Finding 2.1: `enforcement_mode="warn"` config fiction (HIGH security footgun)
`verified/evolution.md` F2.1 (REAL/HIGH): the field is validated + frozen but **no driver consumes
it**; a future driver reading `enforcement_mode=="warn"` would skip `assert_physics_only` and let
ML-violating candidates through — a physics-only-constraint bypass. M1-20 only fixes the
exec/eval/compile + binary-diff scanner gaps; it does not address this. (Also evolution NEW-A:
`builtins.exec` attribute form still bypasses — fold into M1-20.)
**Follow-up:** add to M1-20 (or a sibling SAFE-NOW item): either remove `enforcement_mode` or wire
a real consumer that cannot skip the blocklist; add the documented behaviour test (F5.3).

### GAP-6 — hpc Finding 4 + native-rust Finding 2 (both HIGH) under-served
- hpc F4 (HIGH): SBATCH heredoc hardcodes `from cflibs.inversion.solve.bayesian import ...` as a
  string literal; CLAUDE.md says flat paths already moved — generated scripts on disk silently break.
  M4-4 covers the `__main__`/docstring (NEW-A) but **not** the brittle hardcoded import path.
- native-rust F2 (HIGH, highest in that report): `kdet_filter_elements` Rust branch is **dead under
  the default `shift_coherence_veto=True`** pipeline. Not represented at all; M0-3 only adds a
  Rust-vs-Python comb parity fixture, which does not exercise the dead kdet branch.
**Follow-up:** (a) M4-4 acceptance should add "stop hardcoding bayesian import path / generate it
from a single source"; (b) a DESIGN-DECISION item: is the dead Rust kdet branch intended (port it,
or delete it + document)? Search `rg -n "shift_coherence_veto|kdet_filter_elements" cflibs/`.

---

## B. Architecture / consolidation findings with NO home (need ARCHITECTURE.md, Gap-0)

### GAP-7 — inv-preprocess F5 (HIGH-arch) + F6 (MEDIUM): monolith + misplaced module
F5: `wavelength_calibration.py` is a verified 2119-line monolith on the calibration hot path (with a
documented history of coverage-gate bugs). F6: `outliers.py` belongs in `common/` and is broadly
re-exported (not the "single-constant coupling" the census claimed). Neither is in the blueprint;
M7-8 only touches the RANSAC math inside the monolith. Splitting the monolith is a prerequisite for
safely doing M7-2/M7-8 and for testing the flag-gated paths (F10).
**Follow-up:** ARCHITECTURE.md target for `preprocess/`; then a SAFE-NOW (behavior-preserving)
split item gated by import-parity + the existing calibration tests.

### GAP-8 — radiation F2 (LEGACY magic-number default mode) + F4 (inconsistent instrument broadening)
F2 (MEDIUM): `BroadeningMode.LEGACY` is the silent **default** with a physically-unmotivated
`0.01·sqrt(T/0.86)` sigma and **no DeprecationWarning**. F4 (MEDIUM): resolving-power instruments use
a midpoint-wavelength scalar sigma (3× error across 250–750 nm at R=1000) and an unexplained
`min_relative_intensity` 0.01-vs-10.0 split. The blueprint fixes the Olivero constants (M1-6) and
vectorizes loops (M7-1) but never touches the **default broadening mode** — an accuracy default.
**Follow-up:** DESIGN-DECISION (which default mode) + BENCHMARK-GATED flip of the LEGACY default;
SAFE-NOW DeprecationWarning + per-wavelength sigma for R-mode downstream convolution.
radiation F6 (`CFLIBS_DISABLE_STARK_T_FACTOR` deprecated-but-live) and Missed-B (pseudo-Voigt
normalization) are LOW — fold into M7-7 / M1-13 respectively.

### GAP-9 — pds F2-2 (dead-end schema) + F1-2 (copy-paste truth with K dropped) missing
`verified/pds.md`: F1-1 (mole/mass basis) is covered by M3-1/M3-3. But F2-2 (`PDSValidationDataset`
is a dead-end bridge not wired downstream) and **F1-2 (SCCT5/SCCT7 compositions copy-pasted from
CCCT3/CCCT2 with K silently dropped)** are absent. F1-2 is a *data-correctness* bug in the truth
corpus — it will silently corrupt any benchmark using those targets, independent of M3.
**Follow-up:** SAFE-NOW data-fix item for F1-2 (verify against ChemCam published comps); decide
wire-or-delete for the dead-end schema (ARCHITECTURE.md).

### GAP-10 — native-rust Missed-A (transposed greedy matching, MEDIUM, silent prod divergence)
M0-3 adds "Rust vs Python comb on *ambiguous* input" — good, that fixture would catch Missed-A.
But the blueprint never lists the **fix** (align greedy orientation) or Missed-B (F1 tie-break
epsilon 1e-12 vs np.isclose). If the parity fixture fails (it likely will on ambiguous input), there
is no milestone item to land the alignment.
**Follow-up:** add an M5/M6 item "align Rust↔Python comb greedy orientation + tie-break epsilon to
the Python semantics (BENCHMARK-GATED on the ID benchmark — this is an identifier-scoring change,
which memory warns has regressed F1 before)."

---

## C. Literature-checklist items the census never investigated (true blind spots)

These appear in `literature/*.md` "What Correct Code MUST Do" / pitfall lists but have **no
corresponding verified finding and no blueprint item** — meaning no agent ever checked the code
against them. Each needs a targeted audit before we can claim the checklist is closed.

### GAP-11 — Air↔vacuum wavelength convention (broadening-rt P10, cflibs-method)
A classic LIBS systematic (NIST lists air wavelengths >200 nm; atomic constants and Doppler/Stark
math want vacuum). No verified finding examined whether the DB, line-matching tolerance, and
forward model are consistent on air-vs-vacuum. A 0.05–0.1 nm systematic here would swamp the
sub-pixel-centroiding gains of M7-8.
**Follow-up:** one audit agent: `rg -ni "vacuum|air.?wavelength|n_air|edlen|ciddor" cflibs/` +
check `datagen_v2.py`/`ingest_kurucz_atomic.py` provenance; report convention consistency.

### GAP-12 — IPD/partition-function truncation consistency (saha-boltzmann-lte P2, CRITICAL pitfall)
The literature flags as CRITICAL that the partition-function sum MUST be truncated at the same
energy (IPD-lowered) used in the Saha equation. plasma F-items cover the DH IPD value (M0-1) and the
dead `model` param (M1-17), but **no finding verifies the truncation-consistency invariant** between
`partition.py` and the Saha solver. radiation F3 (dual population paths) hints at divergence but
does not test this specific invariant.
**Follow-up:** add to M0-1 / M2 an explicit invariant test: partition sum cutoff == Saha IPD cutoff
(parity across the SpectrumModel vs kernel paths flagged in radiation F3).

### GAP-13 — Bayesian likelihood normalization & signal-dependent variance (bayesian-oe P2.4/2.5)
M5-9 evaluates Poisson-vs-Gaussian, but the literature also flags (a) signal-dependent vs fixed
variance and (b) dropped log-likelihood normalization constants (matters for model comparison /
BIC). inv-identify uses BIC model selection; if normalization is dropped inconsistently, BIC
comparisons are biased. No finding checked this.
**Follow-up:** fold a normalization-constant + variance-model audit into M5-9's acceptance.

---

## D. Classification / ordering errors in the plan as written

### GAP-14 — M3-2 marked BENCHMARK-GATED but is also a *correctness* fix that must change the baseline
M3 (mole→mass) is correctly benchmark-gated, but the framing "ships default-OFF behind a flag" is
**wrong for a scoring-path correctness bug**: the current scoring is provably comparing number
fractions to mass-fraction truth (benchmark F1, CRITICAL, Völker 353%). Leaving the corrected path
default-OFF means the *committed baseline remains wrong*. The flag is fine for rollout safety, but
the acceptance must state the corrected numbers **become** the baseline (the blueprint hints this in
M3-2 "committed as the new baseline" — make it explicit that the OLD baseline is known-wrong and
must be retired, not treated as a regression when numbers move).
**Follow-up:** clarify M3-2 acceptance: a moved composition number here is the *fix landing*, not a
regression; the gate compares against ground truth, not against the old (wrong) baseline.

### GAP-15 — M0-3 parity fixtures are the gate for ~8 later items but are "xfail until fixed"
M0-3 xfail-marks known divergences (radiation F3, manifold F10, inv-solve F6/F10, native-rust). That
is correct, but several BENCHMARK-GATED items (M6-2/M6-3/M6-4, M5-7) **depend on these xfails being
flipped to real asserts**, and no item explicitly owns "un-xfail after fix". Risk: a fix lands, the
xfail silently keeps passing as xfail, and the parity is never actually enforced.
**Follow-up:** each fix item that resolves an M0-3 xfail must include "flip xfail→assert" in its
acceptance. Add this as a standing rule in M0-3's description.

### GAP-16 — M5 ordering: identifier-scoring changes (M5-2, M5-3, native-rust comb) need the ID
benchmark (M0-2 covers composition F1/Aitchison/RMSEP but the memory + plan repeatedly cite an
**ID F1 benchmark** as a separate gate). M0-2's acceptance only names composition metrics. The
identifier items (M5-2/M5-3/M5-5/M5-6 and the missing GAP-10 comb fix) are gated on something M0-2
does not stand up.
**Follow-up:** extend M0-2 (or add M0-2b) to also emit the **ID F1/recall** benchmark on the
synthetic ID corpus (`scripts/benchmark_synthetic_identifiers.py`), since memory shows identifier
changes have regressed F1 (−0.041) before and MUST be gated on ID metrics, not composition.

---

## E. Lower-priority items absent but worth a single sweep (not blocking)

- benchmark F6 (8 inline `CFLIBS_FF_*` flags) + A1 (Aitchison ε vs RMSE 0 padding asymmetry) + A2
  (silent simplified-model fallback lacks `TruthType.SIMULATED_UNPHYSICAL`) — A2 is MEDIUM and a
  **silent-corruption** risk in generated data; deserves at least a flag/warning item near M4-5.
- evolution F1.2/2.2/4.1/4.2/5.2 (doc + config-default debt) — one housekeeping item.
- hpc F5/F6/F7/F8/F9 (mutable default, SBATCH dup, pickle gather, PRNG split, mem-fraction) —
  one HPC housekeeping item (M4-4 sibling); F8 PRNG-split is the only one with a faint accuracy edge.
- instrument F3 (edge-flush background includes signal), F5 (3 response-curve impls), F10 (sigma=0 →
  NaN, no guard), F11 (no pytree round-trip test) — F10 is a latent crash; should be SAFE-NOW in M1.
  F5 is an ARCHITECTURE.md consolidation candidate.
- inv-common F1/F2/F3 (`get_wavelength_tolerance` discards instrument FWHM; `aki_uncertainty`
  dropped through `to_line_observations` and ignored in `y_uncertainty`) — F2/F3 are **accuracy**
  (uncertainty propagation) and only the doc-fix F4 is in the blueprint (M1-12). Add F1/F2/F3.
- inv-identify NEW-1 (LOGIC/MEDIUM) — not slotted; check against M5-1.
- plasma F6 (architecture) / F9 (documented) — fold into ARCHITECTURE.md.

---

## Highest-leverage follow-ups (do these first)

1. **Create ARCHITECTURE.md** (Gap-0) — unblocks Gaps 7/8/9/10 and the M6-4 decision.
2. **Re-instate the 6 dropped HIGH findings** as milestone items: hpc-F1 warm-start (GAP-1),
   validation-F3rev noise (GAP-2), validation-F5 solver seam → move to M0 (GAP-3), benchmark-F2 LOD
   (GAP-4), evolution-F2.1 enforcement_mode (GAP-5), hpc-F4 + native-rust-F2 (GAP-6).
3. **Add an ID-F1 benchmark to M0** (GAP-16) so the M5 identifier items have their correct gate.
4. **Audit the 3 never-checked literature pitfalls**: air/vacuum (GAP-11), IPD/PF truncation
   consistency (GAP-12), Bayesian likelihood normalization/variance (GAP-13).
5. **Fix the data bug** pds-F1-2 (GAP-9) — silent K-dropped copy-paste truth corrupts benchmarks.
6. **Tighten M3-2 framing** (GAP-14) and add "flip xfail→assert" to every M0-3-dependent fix (GAP-15).

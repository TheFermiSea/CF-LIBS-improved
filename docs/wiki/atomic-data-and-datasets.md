---
slug: atomic-data-and-datasets
title: "Atomic Data Provenance, Instrument Calibration & Datasets"
chapter: atomic-data-and-datasets
order: 0
status: stable
register: handbook
summary: >
  The two input-quality levers that own the measured real-data accuracy gap in CF-LIBS —
  atomic-data provenance (accurate gf, not raw completeness; STARK-B for n_e) and instrument
  calibration (only the scalar F cancels; E(lambda) is the dominant un-removed systematic) —
  plus the provenance of every benchmark dataset. Accuracy is data-limited, not solver-limited.
tags: [atomic-data, provenance, partition-functions, stark, instrument-calibration, datasets, nist-asd, vald, kurucz, response-curve]
updated: 2026-07-02
benchmarks_pre_reset: false
sources:
  - "@ciucci1999"
  - "@aragon2008"
  - "@barklem2016"
  - "@irwin1981"
  - "@alimohamadi2022"
  - "@sahalbrechot2015"
  - "@konjevic2002"
  - "@ryabchikova2015"
  - "@kawahara2022"
  - "@anderson2022"
  - "@wiens2012"
  - "@jochum2016"
  - "@fischler1981"
  - docs/research/data-acquisition-plan.md
  - docs/v4/atomic_db/COMPLETE-DB-RESET-STATUS.md
  - docs/v4/atomic_db/COMPLETENESS-VERIFICATION.md
  - docs/M5-atomic-db-heldout-verdict.md
  - docs/M5-db-accuracy-findings.md
  - docs/adr/ADR-0006-instrument-calibration-first-class.md
  - docs/atomic-db-latency-ADR-0007-investigation.md
  - docs/Echellogram_Processing_Guide.md
  - docs/v4/overhaul/literature/wavelength-calibration.md
  - cflibs/atomic/database.py
  - cflibs/plasma/partition.py
  - cflibs/instrument/model.py
  - cflibs/benchmark/datasets/
code_refs:
  - cflibs/atomic/database.py::AtomicDatabase
  - cflibs/plasma/partition.py::direct_sum_partition_function
  - cflibs/plasma/partition.py::ionization_potential_depression
  - cflibs/atomic/wavelength_conversion.py
  - cflibs/instrument/model.py
  - cflibs/instrument/echelle.py::EchelleExtractor
  - cflibs/benchmark/datasets/supercam_labcal.py
  - cflibs/benchmark/datasets/chemcam_calib.py
  - tests/benchmarks/ded_precision/alloy_definitions.py
related: [libs-physics, classical-quantification, error-budget-and-falsification, benchmarks-reliability-workflows, architecture]
supersedes:
  - docs/v4/atomic_db/COMPLETE-DB-RESET-STATUS.md
  - docs/v4/atomic_db/COMPLETENESS-VERIFICATION.md
  - docs/M5-atomic-db-heldout-verdict.md
  - docs/M5-db-accuracy-findings.md
  - docs/atomic-db-latency-ADR-0007-investigation.md
  - docs/Database_Generation.md
  - docs/Echellogram_Processing_Guide.md
  - docs/research/data-acquisition-plan.md
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Atomic Data Provenance, Instrument Calibration & Datasets

Two inputs — not the solver — own the measured real-data accuracy gap in CF-LIBS: the
**atomic line list** (the transition probabilities, energy levels, partition functions and
Stark widths the forward model reads) and the **instrument calibration** (the wavelength
axis, line-spread function, and spectral response that shape the measured spectrum). A
controlled round-trip found the inversion *algorithm* floor at RMSE ≈ 2.9×10⁻⁶ while the
*atomic-data* line-list mismatch alone injected ≈ 0.17 RMSE and the instrument was run
uncalibrated on top of that ([ADR-0006](../adr/ADR-0006-instrument-calibration-first-class.md) §1.3).
The math is not the bottleneck; the data are. This chapter is the reference for both levers
plus the provenance of every benchmark dataset the pipeline is scored against.

> [!IMPORTANT] RESET LINE — all quantitative figures on this page are the ASD59-reset baseline
> (2026-07-02). The atomic DB was rebuilt from the gold-standard NIST ASD `monograph8` dump
> (203,695 lines / 62,752 levels / 324 ionization potentials), which **invalidated every prior
> benchmark**. The three-DB-snapshot ambiguity flagged in the audit (28k netCDF vs 61k-level
> partial vs 203k dump) is resolved here: **only the 203k reset numbers are current.** Where a
> pre-reset campaign is referenced (the M5 database bake-off), only its *mechanism* is carried
> forward — the magnitudes are dead and are not quoted as current.

**Wavelength convention (load-bearing).** The atomic DB stores **air** wavelengths in nm above
200 nm, following NIST/ASD, and vacuum below ~200 nm per project convention. Any comparison
against a laser-frequency-comb axis (vacuum) or a vendor product must go through the single
`cflibs/atomic/wavelength_conversion.py` air↔vacuum utility (Edlén formula); never mix the two
silently. This matters throughout Part B.

Notation follows [formal-spec/notation.md](formal-spec.md); symbols
($I_{ki}$, $A_{ki}$, $g_k$, $E_k$, $U_s(T)$, $n_e$, $F$, $E(\lambda)$) are defined there once.

---

## Part A — Atomic Data Provenance {#part-a}

### A.1 The database schema and access layer {#db-schema}

The atomic data lives in a single SQLite file (`ASD_da/libs_production.db`) read through
`cflibs/atomic/database.py::AtomicDatabase`. SQLite is deliberately the **build-time
source of truth**, not the inference hot path (see [A.8](#two-tier-latency)). Four tables carry
the physics:

| Table | Rows (ASD59 reset) | Role | Key columns |
|-------|-------------------:|------|-------------|
| `lines` | 203,695 | one row per observed transition | `element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int`, plus Stark: `stark_w, stark_alpha, stark_shift, stark_w_source, is_resonance` |
| `energy_levels` | 62,752 (I/II/III) | partition-function summation source | `element, sp_num, g_level, energy_ev` (+ J, term, config on the extended schema) |
| `species_physics` | 324, 0 gaps | Saha ionization balance | `element, sp_num, ip_ev, atomic_mass` |
| `partition_functions` | polynomial fits | $U(T)$ fallback | `a0..a4, t_min, t_max, source` |

`sp_num` is the ionization stage (1 = neutral I, 2 = singly ionized II, 3 = doubly ionized III).
The `lines` UNIQUE key is `(element, sp_num, wavelength_nm, ek_ev)`; the 0.4% of dump lines that
collide on this coarse key are degenerate Rydberg lines, not data loss
([COMPLETENESS-VERIFICATION](../v4/atomic_db/COMPLETENESS-VERIFICATION.md)). `AtomicDatabase`
maintains a **connection pool** and token-versioned query caches (`_spec_cache`, `_level_cache`)
so repeated species lookups do not re-hit SQLite; the `_level_cache` is invalidated on re-ingest
by the same token scheme as `_spec_cache` (a stale-cache bug fixed in `c22fe3b`).

The integrated line intensity the schema feeds is the CF-LIBS master equation
([Ciucci et al. 1999][@ciucci1999]; [Aragón & Aguilera 2008][@aragon2008]):

$$
I_{ki} = F \cdot \frac{hc}{4\pi\lambda_{ki}} \cdot A_{ki}\, g_k \cdot \frac{n_s}{U_s(T)}\, e^{-E_k/kT}
$$

Every symbol on the right except $F$ and the plasma state comes from these tables: $A_{ki}$ from
`lines.aki`, $g_k$ from `lines.gk`, $E_k$ from `lines.ek_ev`, $U_s(T)$ from `energy_levels`, and
the ionization balance that sets $n_s$ from `species_physics.ip_ev`.

### A.2 The complete-DB reset — what changed and what it invalidated {#complete-db-reset}

The database was rebuilt in 2026-06-26 from `ASD59_dump.sql` (the full NIST ASD `monograph8`
MySQL dump, 115 MB, 2022 vintage) obtained from the user's laptop and loaded via MariaDB. The
prior production DB held only ~30% of observed lines, had 99 ionization-potential gaps (81 in
stage III), and no usable stage-III levels. An intermediate `CAAAMLIBS/LIBS` netCDF
reconstruction was rejected — it turned out to be a **lossy** conversion that had dropped roughly
half the lines ([COMPLETENESS-VERIFICATION](../v4/atomic_db/COMPLETENESS-VERIFICATION.md)).

| Quantity | Pre-reset (partial) | ASD59 reset | Validation |
|----------|--------------------:|------------:|------------|
| `lines` | ~28,700 (A-valued only) | **203,695** | 99.6% of the 204,574 dump lines |
| `energy_levels` (I/II/III) | 61,507 (I/II only) | **62,752** | ≥ dump; union of 2022 dump + current NIST |
| `species_physics` IPs | 174 of 273 | **324, 0 gaps** | 17/17 live-NIST match within 2% |
| Live-NIST stochastic check | — | **18/18 species** levels covered | Ti/Al/V/Fe/Cr/Ni included |

The IP table is sourced from live NIST because the dump's own `IPs` table is corrupt (it lists
Fe I at 9277 eV). The partition-function overhaul rode along: the direct sum over the now-complete
levels became authoritative and the Fe I ground-state degeneracy was corrected (`g0` 1→9).

> [!CAUTION] DO-NOT carry forward: "stage III is empty" / "Fe I has 425 levels." Both were
> pre-reset artifacts of the lossy netCDF and the old lines-table scrape. The reset DB carries
> **62,752 levels across all species and stages I/II/III**; Fe I alone now carries **847 levels**
> (Fe I/II/III together ≈ 2,471) — hundreds-to-low-thousands, far above the stale 425 but nowhere
> near the all-species 62,752 total. Nb, Mo, V III and Cr III are
> **populated**. Any code path or doc asserting stage-III emptiness is stale.

The first **valid** pipeline numbers on the reset DB (the DED-constrained deliverable — see
[Part C](#part-c) and [alloy_definitions.py](../../tests/benchmarks/ded_precision/alloy_definitions.py)):

| Pipeline | Result (ASD59 reset) |
|----------|----------------------|
| DED constrained Ti-6Al-4V (the real goal) | Al **0.82** / Ti **1.43** / V **0.87** wt% RMSEP; 27/27 converged; V nominal bias +0.73 |
| Solver head-to-head, SuperCam labcal, n=15 | **iterative 2.31** median mass RMSE / ILR 11.0 / C-sigma 16.8 — iterative wins |
| NIST forward-model parity (Fe) | PASS (I −4.5%, II +1.7%, III exact) |
| Open-element ID, small_v1, 12 spectra | ALIAS F1 0.063 (recall collapse) / Comb 0.329 (over-detects) / Corr 0.327 |

> [!NOTE] The iterative solver beating the closed-form ILR and C-sigma variants here is **not** a
> claim that iterative is universally best; it is the reset-DB result on this labcal set. The
> forward-model solvers (joint / Bayesian) converge and can beat iterative under SVD conditioning;
> the open question is the *adoption gate* (when to trust a converged fit), covered in
> [cf-libs-family.md](cf-libs-family.md) and [error-budget-and-falsification.md](error-budget-and-falsification.md).

The open-element identification collapse is a **re-architecting** problem, not a re-tuning one:
the 7× denser catalog inflates the expected-line denominator in ALIAS's detection statistic
($k_{det} \propto$ matched/total emissivity in-window), so line-rich true elements (Fe, Ni) are
penalized while line-poor absent ones pass by chance. The fix is a **chance-corrected** detection
metric (normalize by the expected random-match rate at the in-window catalog density) plus a
high-recall-Comb → precision-filter hybrid. This is deep and **secondary** to the DED goal, which
uses a known element set and bypasses identification entirely
([COMPLETE-DB-RESET-STATUS](../v4/atomic_db/COMPLETE-DB-RESET-STATUS.md) §3). Details in
[cf-libs-family.md](cf-libs-family.md).

### A.3 Completeness verification — per-stage coverage {#completeness-verification}

The reset DB was validated by three independent cross-checks: against the 2022 dump (row counts),
against live NIST ASD (18 random species, stochastic direct fetch), and internal integrity
(`integrity_check = ok`). The union construction means the DB slightly *exceeds* current live NIST
per species (2022 dump ∪ current-NIST level/IP ingest). All 28,331 derived Stark fields were
preserved through the rebuild.

The genuinely important structural fact: of the 203k lines, ~74k are **observation-only** — they
carry an observed wavelength (100%), observed intensity (99%) and upper/lower level IDs (83%) but
**no measured** $A_{ki}$. These are real transitions, not noise. Dropping them would re-cripple
the DB to its pre-reset 30% state; instead they are kept with an `aki`/`aki_source` provenance
flag so synthetic-spectrum generation can choose measured-$A$, theory/`log_gf`-derived, or
intensity-anchored $A$ at generation time. The `get_transitions` NULL-`aki` crash (fixed `2f15cfa`)
was exactly the bug that surfaced when the complete DB's 74k observation-only lines first hit
every forward-model path.

### A.4 Partition functions — direct sum, polynomial fallback, truncation invariant {#partition-functions}

`cflibs/plasma/partition.py` computes $U_s(T) = \sum_i g_i\, e^{-E_i/kT}$ by two routes, with a
strict precedence.

**Method 1 — direct summation (authoritative).** Sum over the `energy_levels` rows for the species,
truncated at the plasma-lowered ionization potential
([Alimohamadi & Ferland 2022][@alimohamadi2022]):

$$
U_s(T) = \sum_{E_i < \mathrm{IP} - \Delta\chi(n_e,T)} g_i\, e^{-E_i/kT}
$$

Since the reset DB carries complete I/II/III levels, this is the **single source of truth** for
every levelled species. Ca I, Na I, K I and Fe I — historically truncated by the old lines-table
scrape and biased low by 5–40% — all resolve correctly now
(`cflibs/plasma/partition.py::direct_sum_partition_function`). A level with an unassigned
`g_level` (unknown $g = 2J+1$) is excluded at the DB layer, so the sum no longer silently crashes
and degrades to a stale polynomial.

**Method 2 — polynomial fallback (Irwin 1981 form, natural-log basis).** For the handful of high
ions the `energy_levels` table cannot cover:

$$
\ln U(T) = a_0 + a_1 \ln T + a_2 (\ln T)^2 + a_3 (\ln T)^3 + a_4 (\ln T)^4
$$

This is mathematically the [Irwin 1981][@irwin1981] fit (Irwin tabulated in $\log_{10}$; the code
stores natural-log coefficients per the NIST-ASD-fit convention — use `irwin_log10_to_ln_coeffs`
when ingesting Irwin's published Table II). It is **demoted to a warned last resort**: polynomial
fits carry errors up to ~66% for stale species, versus ~2% with a proper refit or exact
direct-sum. [Barklem & Collet 2016][@barklem2016] is the modern self-consistent,
cutoff-aware tabulation (atoms H–U + 291 diatomics) wired in as an authoritative override source
(`AUTHORITATIVE_SOURCES` in `partition.py`; the B&C-2022 `table8` ingest is done — do not
re-request).

> [!NOTE] FORMAL — the Boltzmann level population that the partition sum normalizes is proven in
> cflibs-formal: `lean:CflibsFormal/Boltzmann.lean#population_sum`. The partition function
> is the normalization constant $U_s(T)$ of that distribution; see
> [formal-spec.md](formal-spec.md).

#### The partition-function truncation invariant {#pf-ipd-invariant}

The single most important correctness property of the whole forward model: **the partition-sum
cutoff must use the same ionization-potential depression $\Delta\chi$ as the Saha ionization
exponent.** If the two diverge, the number of bound states and the ionization balance disagree and
the composition is biased. `cflibs/plasma/partition.py::ionization_potential_depression` is now the
**one** Debye-Hückel IPD implementation for the package (canonical Gaussian-CGS form):

$$
\Delta\chi = Z\,\frac{e^2}{\lambda_D}, \qquad
\lambda_D = \sqrt{\frac{k_B T}{4\pi n_e e^2}}
$$

giving ≈ 0.066 eV at $n_e = 10^{17}\,\mathrm{cm^{-3}}$, $T = 10^4$ K for $Z=1$ (Mihalas 1978
Eq. 9-106; [Alimohamadi & Ferland 2022][@alimohamadi2022]). Before the audit-Family-J fix, this
module used an approximate $3\times10^{-8}\,Z\sqrt{n_e/T}$ form (~0.0949 eV) while
`saha_boltzmann.py` used the canonical form (~0.066 eV) — a **1.44× discrepancy** between the
Saha exponent and the partition cutoff. Both call sites (and their JAX twins,
`ionization_potential_depression_jax`, used by the manifold/kernel Saha) now route through the one
function. **What correct code MUST do:** never compute an IPD locally — always call the canonical
helper so the truncation invariant holds across CPU, JAX, and manifold paths.

**Extrapolation guards.** The polynomial fit is only valid on `[t_min, t_max]`; outside it the code
must not extrapolate a 4th-order polynomial (it diverges). The direct sum has no such range limit
but requires `ip_ev` to bound the cutoff; a missing IP is a loud error on the reset DB (coverage is
guaranteed), not a silent `15.0 eV` fallback — one of the 72 DB-compensation band-aids now
non-firing and slated for removal ([COMPLETE-DB-RESET-STATUS](../v4/atomic_db/COMPLETE-DB-RESET-STATUS.md) §4).

### A.5 Accuracy beats completeness — the central data lesson {#ga-accuracy-vs-completeness}

This is the most consequential, most counter-intuitive finding in the whole data programme, and it
directly contradicts the intuition that "more lines = better."

**Raw completeness HURTS.** A confound-controlled bake-off (paired per spectrum, scored over the
intersection of elements both DBs called present, so the ID-density element-flip cannot distort the
comparison) ran the reference pipeline over NIST-graded vs bulk alternatives. Bulk VALD-complete
(~1.09M lines) and NIST+VALD backfill (~935k lines) both **regressed materially** versus the small
NIST-graded set — on **both** the optimization (dev) and held-out (test) tiers, with no overfit
sign-flip ([M5 held-out verdict](../M5-atomic-db-heldout-verdict.md);
[M5 findings](../M5-db-accuracy-findings.md)).

> [!NOTE] The M5 bake-off predates the ASD59 reset; its *absolute* RMSE magnitudes are superseded
> and deliberately not quoted here. What survives — robust across dev/held-out and consistent with
> the composition error bound — is the **mechanism and ranking**, below.

**Mechanism (evidence, not assertion).** VALD-complete is ~75% Kurucz-*theoretical* D-grade
$gf$ + ~14% ungraded. Its per-element line *selection* draws ~20 lines from that pool, so the
Boltzmann/Saha fit is polluted by inaccurate transition probabilities. The bottleneck is
**$gf$-value accuracy, not line count**: on high-SNR data the per-line number-density error
dominates, so a bigger-but-less-accurate list regresses. Filtering VALD to its experimental
grade-B lines is *worse* still — grade-B exists for only 15 species (11% of lines), decimating
coverage.

**The grade-aware selector is necessary but not sufficient within VALD.** A gated
`grade_aware_selection` lever (prefer A/B `accuracy_grade` over D/U via `aki_uncertainty`) was
implemented and measured **byte-identical** on VALD-complete (ON == OFF) — because VALD's LIBS-major
species (Si, Fe I, Ca, Ti, Mn, Na, K) are almost entirely D/U with **no A/B alternatives to
prefer**. You cannot select accurate lines that do not exist.

**The lever that works.** The only accuracy path is **R4 — NIST-A/B authoritative ∪ VALD/Kurucz
backfill ONLY where NIST lacks (element, ion, ~λ) coverage**, flagged and downweighted, with the
grade-aware selector doing the A/B-over-D preference in the merged DB. This is the
literature-recommended hybrid (default NIST A/B; use VALD/Kurucz only to widen coverage,
downweighted) and it reconciles the repo's own earlier "Kurucz beats NIST on SuperCam"
measurement: Kurucz won there **only** because it used a curated 24-strongest-line bundle plus
structured Gauss-Newton, not the raw dump. Completeness helps **only** in the curated / grade-aware
regime; raw substitution is a dead end.

> [!CAUTION] DO-NOT re-attempt: swapping the whole line list to a bigger raw dump (VALD-complete,
> Kurucz `gfpred`, or NIST+VALD naive union) "for completeness." Measured on both tiers: it
> regresses. The accuracy lever is experimental-grade $gf$ (STARK-B / graded acquisition), not
> volume. See [error-budget-and-falsification.md](error-budget-and-falsification.md).

### A.6 Atomic sources — access, formats, grades {#atomic-sources}

The one-time exhaustive acquisition plan ([data-acquisition-plan.md](../research/data-acquisition-plan.md))
enumerates the VAMDC-federated and native atomic sources. NIST-ASD and the accurate-$gf$ backbone
are done; the outstanding *new* ask is STARK-B.

| Node | What it uniquely gives | Access / format | Per-line grade | Status |
|------|------------------------|-----------------|----------------|--------|
| **NIST ASD** | graded $A_{ki}$ (AAA ≤0.3% .. E ≥50%), levels, IPs | native web forms / XSAMS; Tab/CSV; **not** a VAMDC bulk node | **yes** (grade→σ defs) | DONE (reset DB); DOI `10.18434/T4W30F` |
| **STARK-B** | semiclassical Stark $W$ + shift $d$, non-hydrogenic | manual web export (no bulk download); XSAMS; $W,d$ in Å | NV validity flags (15–50% unc.) | **#1 gap** ([sahalbrechot2015][@sahalbrechot2015]) |
| **VALD3** | log $gf$ + all three damping constants ($\gamma_{rad},\gamma_{stark},\gamma_{vdW}$) with per-line refs | free registration; Extract-All; long-format ASCII | per-line source ref | supplement ([ryabchikova2015][@ryabchikova2015]) |
| **Kurucz** | deepest Fe-group; `gfall` lab-based vs `gfpred` predicted | plain HTTP (broken TLS); fixed-width 160-col | theoretical (B/C-equiv) | audit overlap before re-pull |
| **CHIANTI** | collision strengths (future non-LTE) | open tarball; ChiantiPy | critically evaluated | wrong T regime for LTE-LIBS |
| **TOPbase** | R-matrix gf, Z ≤ 26 only | web / XSAMS | theoretical | gap-fill low-Z only |

**STARK-B is the single highest-value new request** because it is the electron-density lever. The
reset DB has `stark_w` populated for 98.6% of the **203,695** lines, but of the ~28,727 A-valued
lines that carry a `stark_w_source` only **244 (0.85%)** hold real STARK-B/[Konjević][@konjevic2002]-grade
literature values; ~80% of that source-bearing set are
`konjevic_lambda_sq_scaled` heuristics, the rest interpolated or hydrogenic. Stark widths feed the
$n_e$ diagnostic that closes the Saha loop and the Voigt wings of the forward model; the current
heuristic majority is why the pipeline still falls back on a 1-atm pressure-balance $n_e$
band-aid in some regimes. The n_e diagnostic must be restricted to
`stark_w_source IN ('stark_b','interpolated')`. Coverage is paper-by-paper (the STARK-B picker is
image-based, no machine-readable species list), so each target species must be verified at request
time; the repo ingest (`scripts/archive/migrations/ingest_stark_b.py`) expects hand-downloaded
CSV/TSV with columns `(transition, T_e, n_e, gamma_W, gamma_d[, alpha])`.

> [!IMPORTANT] The **request-once checklist** ([data-acquisition-plan.md](../research/data-acquisition-plan.md) §6)
> bundles the atomic P0 pull (VALD3 register + Extract-All 1800–11000 Å; NIST complete Levels CSV;
> STARK-B $W$/$d$ tables; Kurucz `gfall` + diatomic `.ASC`; CHIANTI tarball) with the future
> molecular P1 pull in one sitting, recording every version string and license. Do it once.

**Greenfield molecular note (P1, not built).** For future PLD/CVD non-equilibrium plasma-OES
thermometry, two physics splits govern what is even usable: (a) **electronic-emission bands**
(CN violet ~388 nm, C2 Swan ~516 nm, N2 2nd-positive ~337 nm, N2⁺ 1st-negative ~391 nm, OH A-X
~309 nm) drive $T_{vib}/T_{rot}$ and live in ExoMol/Kurucz-diatomic/PGOPHER/massiveOES — **not**
HITRAN; and (b) **IR rovibrational** bands (CH4, CO, OH-IR) live in HITRAN/HITEMP
([gordon2022][@gordon2022]). Crucially, **LTE-single-T engines cannot do the two-temperature job**:
[ExoJAX][@kawahara2022] is LTE single-$T$ only (right for atomic LTE, disqualified for molecular
non-equilibrium), and [RADIS][@pannier2019] has a true two-$T$ mode but only for ground-state IR
bands, not UV/Vis electronic systems. N2 and C2 are homonuclear → no electric-dipole IR → their
**only** diagnostic is the electronic bands. `cflibs/` has **zero** molecular-emission code today;
this is a net-new `cflibs/molecular/` module (Hönl-London + Franck-Condon + separate
$T_{vib}/T_{rot}/T_{exc}$ populations) plus an engine-adoption ADR, tracked as a follow-up.

### A.7 Database generation {#db-generation}

The DB is generated by `datagen_v2.py` (or `cflibs generate-db`), which fetches from NIST ASD via
`ASDCache` with local request caching. First run is hours-long (network-bound); subsequent runs use
the cache. The historical `datagen_v2` filters (`MAX_IONIZATION_STAGE=2`, `MAX_UPPER_ENERGY_EV=12`,
`MIN_RELATIVE_INTENSITY=50`) are the ultrafast-LIBS defaults documented in the generation guide —
**superseded** by the ASD59-dump ingest path (`ingest_asd_dump_lines.py`, `ingest_dump_levels.py`,
`ingest_nist_ips.py`, `validate_db_vs_nist.py`) which loads the full unfiltered catalog including
stage III and observation-only lines. Two `datagen_v2` bugs were fixed during the reset: the
`fetch_ionization_potential` parser (NIST format-3 data lines start with a quoted `""`; the parser
never stripped quotes → always `None`), and the absence of a lines fetcher (only levels + IPs
existed; the dump ingest supplies lines).

### A.8 Two-tier latency architecture — SQLite build, columnar inference {#two-tier-latency}

> [!WARNING] BENCHMARK-GATED / LATENCY-LAST — this is the M10 lever. The standing directive is
> **accuracy first, latency last**: a faster lookup over the wrong line list buys nothing. Land the
> R4 DB and STARK-B first. This section documents the *investigated* design
> ([ADR-0007 investigation](../atomic-db-latency-ADR-0007-investigation.md)), not an implementation
> order.

Profiling the hot path found the SQLite engine is **not** the bottleneck — per-call Python object
materialization is. `get_transitions()` cold spends 92% of its 187–203 ms (Fe I, 2439 lines) in
`pandas.iterrows()` + per-row `Transition` dataclass construction; raw `sqlite execute+fetchall` is
only 7.2 ms and pool acquire/release is 4.3 µs. `partition_function_for().at(T)` costs ~146 µs
per species-stage even fully warm, rebuilding a provider (82 µs) and re-running the direct sum at
Python speed (60 µs) — ~3.82 ms (29%) of a 13.6 ms 10-element solve. The identify stage's LRU is
keyed on exact `(element, stage, wl_min, wl_max)`, so 50 distinct per-peak windows run at
5,591 µs/call with a 0.00 hit rate.

**Decision: two-tier.** SQLite (or Parquet) stays the offline build/source-of-truth; an
`InMemoryColumnarSource` (duck-types the `AtomicDataSource` Protocol in `cflibs/core/abc.py`) loads
the run's **known-up-front** species × ion × wavelength subset once into per-species contiguous
NumPy column arrays + scalar dicts, keyed by `(element, sp_num)`. Lookups become dict-hash +
`np.searchsorted` + zero-copy slice. Because `--elements` is the run's identity and the wavelength
window is fixed per run (`cflibs/inversion/pipeline.py`), the entire relevant subset is loadable
before any hot loop.

| Operation | SQLite (prod / raw) | In-RAM columnar | Speedup |
|-----------|--------------------:|----------------:|--------:|
| Line-window lookup | 28 ms cold / 676 µs raw | **0.14 µs** | ~140× vs raw, ~6,000× vs prod |
| Sub-window `searchsorted` | 676 µs | **4.69 µs** | ~140× |
| Partition $U(T)$ eval | 6.9 µs + 60 µs direct-sum | **1.4 µs** (vectorised `polyval`) | ~100× |
| IP lookup | 5.6 µs | **0.13 µs** | ~43× |

The win is **flat to millions of lines** (hash + binary search), so it grows *more* urgent exactly
as the accuracy-driven VALD/Kurucz DB grows (cold materialization scales ~linearly). Footprint is a
non-issue: <1 MiB for a 12-element run, ≤0.67 GiB (f32) for the full atomic + 30M-line molecular
target — keep `wavelength_nm`/energies in f64 (line-center precision), demote only `aki`/`g`/Stark
to f32. NumPy is the inference tier; DuckDB/PyArrow/Polars are confined to the offline build path
(their per-query dispatch of 100s of µs–ms makes them the *slowest* options for tiny point
lookups). All candidates pass the physics-only blocklist (NumPy/PyArrow/DuckDB are non-ML deps).

---

## Part B — Instrument Calibration {#part-b}

### B.1 What "calibration-free" actually removes — only F {#calibration-free-precise}

CF-LIBS is named for the cancellation of **one, and only one**, instrumental term. In the master
equation ([A.1](#db-schema)), $F$ bundles the *wavelength-independent* experimental factors: plasma
volume in the field of view, collection solid angle, overall optical-train throughput, detector
gain. The Boltzmann-plot linearization

$$
\ln\!\left(\frac{I_{ki}\,\lambda_{ki}}{g_k A_{ki}}\right) = -\frac{E_k}{kT} + \ln\!\left(F\cdot\frac{hc}{4\pi}\cdot\frac{n_s}{U_s(T)}\right)
$$

puts $T$ in the slope and $n_s/U_s(T)$ in the intercept **up to the common factor $F$**. Because
$F$ is identical for every species, the closure constraint $\sum_s C_s = 1$ divides it out exactly.
**That is the entire content of "calibration-free": the scalar $F$ cancels. Nothing else does**
([ADR-0006](../adr/ADR-0006-instrument-calibration-first-class.md) §1.1). The $\lambda_{ki}$ factor
inside the ordinate is load-bearing and must never be silently dropped when it varies across the fit
(notation authority: [formal-spec/notation.md](formal-spec.md), `boltzmann_plot`).

### B.2 The two effects calibration-free does NOT remove {#uncancelled-systematics}

**1. Wavelength-dependent spectral response $E(\lambda)$ — the dominant un-removed systematic.**
Grating efficiency, optics transmission and detector quantum efficiency all vary with wavelength, so
the measured spectrum is $I_{meas}(\lambda) = E(\lambda)\, I_{true}(\lambda)$. In the Boltzmann plot
this adds $\ln E(\lambda_{ki})$. Because lines with different upper energies $E_k$ sit at different
wavelengths, $E(\lambda)$ injects an **$E_k$-correlated perturbation that rotates and scatters the
Boltzmann plot** — biasing both the slope ($T$) and the intercept (composition). It does **not**
cancel under closure: it is wavelength-dependent, and different species populate different
wavelength regions. This is the single dominant un-removed instrumental systematic in quantitative
CF-LIBS.

**2. The line-spread function (LSF).** The instrument convolves every line with its LSF. Convolution
conserves area, so in the *integrated-intensity* formulation the LSF leaves the intercept invariant
*in principle* — the textbook reason the classical method "lumps broadening into the intercept and
ignores it." In practice the LSF still sets which lines are resolved (blend contamination of
integrated intensities) and, decisively, it is **confounded with the physical broadening**: Stark
width → $n_e$ and Doppler width → $T$. An unknown LSF makes the Stark $n_e$ diagnostic dishonest —
width gets over-attributed to the plasma.

The legacy integrated-intensity solver can mostly ignore the LSF (area argument) and only optionally
apply a response curve. The real-time **profile-fitting** method cannot: it computes the residual in
spectrum space, $S(\theta) = E(\lambda) \odot \mathrm{LSF}_\sigma(\lambda)\{\sum_s C_s\,\varepsilon_s(T,n_e)\}$,
so all three instrument terms — LSF, $E(\lambda)$, and the wavelength axis $\lambda(\text{pixel})$ —
directly shape the residual. The instrument operator becomes part of the forward model and can no
longer be waved away ([ADR-0006](../adr/ADR-0006-instrument-calibration-first-class.md) §1.3). The
LSF FWHM folds **exactly** into the Voigt Gaussian core ($\text{Gaussian} \otimes \text{Voigt} =
\text{Voigt}$) — both physically correct and numerically necessary (a naive separate `jnp.convolve`
lowers to a cuDNN `convForward` that fails to autotune on the V100S).

### B.3 InstrumentCalibration — one object, three terms, three modes {#instrument-calibration-object}

[ADR-0006](../adr/ADR-0006-instrument-calibration-first-class.md) makes the spectrometer a single,
immutable, provenance-tracked `InstrumentCalibration` object (proposed
`cflibs/instrument/calibration.py`; existing carriers are
`cflibs/instrument/model.py::InstrumentModel`, `apply_response`, `apply_instrument_function`). It
holds exactly three physical terms plus a mandatory provenance record:

| Term | Symbol | How it enters the fit | If unknown (Mode C) |
|------|--------|-----------------------|---------------------|
| wavelength solution | $\lambda(\text{pixel})$ | resample / sub-pixel align | small nonlinear shift |
| line-spread function | $\sigma(\lambda)$ + shape | **nonlinear** — folds into Voigt core | extra nonlinear column; confounds Stark $n_e$ |
| spectral response | $E(\lambda)$ | **linear** — per-species basis columns | low-order Chebyshev fit (added linear DOF) |
| scalar factor | $F$ | **cancels via closure** | needs no calibration |

**Three acquisition modes, never implicit:**

- **Mode A — Calibrated.** Built from the user's own lamp spectra (`from_lamps(...)`): a
  wavelength/LSF lamp (HgAr, Ne, ThAr, Ar) fixes $\lambda(\text{pixel})$ and the LSF; a radiance
  lamp (deuterium-halogen, or NIST-traceable tungsten-halogen) fixes $E(\lambda)$. All three terms
  measured. Best-conditioned, fastest, most honest fit.
- **Mode B — Vendor pre-corrected.** Flight/commercial instruments (ChemCam, SuperCam) deliver
  radiance-corrected spectra with a published response/LSF; we *declare* that provenance and use the
  published $R(\lambda)$/LSF without re-fitting.
- **Mode C — Self-calibrating fallback.** No lamp and no vendor correction: the missing terms are
  fit from the science spectrum itself (smooth $E(\lambda)$ as low-order Chebyshev, LSF width as a
  nonlinear parameter, sub-pixel shift), carrying a **loud non-quantitative flag**.

> [!IMPORTANT] **Mode A or B is required for any quantitative / accuracy-claimed result.** The
> scoreboard, NIST-parity, and any reported composition RMSE run in Mode A or B; Mode C outputs are
> flagged non-quantitative and **never flip a default or claim a SOTA number**. This is the
> discipline that stops an uncalibrated regression masquerading as an algorithm result — most of the
> current real-data suite lacks lamps and is therefore Mode C, and the scoreboard gains a
> per-dataset calibration-mode column so this is visible.

The scope boundary is deliberate: we calibrate the **relative** response $E(\lambda)$ (shape vs
wavelength), the LSF and the axis. The absolute scalar $F$ is left to closure — CF-LIBS stays
calibration-free in the one sense it legitimately is. Absolute number densities (which would need
$F$ from a radiance standard with known geometry) are an explicit non-goal;
`provenance.response.is_absolute` records whether a traceable absolute response was supplied for a
future absolute-radiance mode. The object is consumed identically by every pipeline via two bridges:
`to_instrument_model()` (legacy integrated-intensity path) and `as_snapshot_arrays()` (fixed-shape
static arrays into the `PipelineSnapshot` / real-time hot kernel).

### B.4 Response curves, LSF/FWHM, and the noise model {#response-lsf-noise}

`cflibs/instrument/model.py` supports two LSF parameterizations: fixed FWHM (nm) or resolving-power
mode ($R = \lambda/\Delta\lambda$, so FWHM grows with $\lambda$). The resolving-power +
Chebyshev-baseline path in `cflibs/inversion/solve/bayesian/` is the prior art for *fittable*
instrument terms — a self-calibrating Mode-C precursor. Response is applied multiplicatively by
`apply_response`; the LSF convolution by `apply_instrument_function`. A physically motivated
`NoiseModel` (shot + read + dark) sets the per-channel weights for any least-squares residual so
the fit is properly heteroscedastic rather than uniform-weighted.

### B.5 Echellogram processing — 2D image to 1D spectrum {#echellogram}

Echelle spectrometers (e.g. Andor Mechelle 5000) produce 2D echellograms that must be unwrapped.
`cflibs/instrument/echelle.py::EchelleExtractor` implements the standard **trace-and-sum**
algorithm ([Echellogram_Processing_Guide](../Echellogram_Processing_Guide.md)):

1. **Order tracing** — each order $m$ has a polynomial center trace $y_m(x) = a_2 x^2 + a_1 x + a_0$.
2. **Flux extraction** — sum intensity in a ±δ window (default ±5 px) around the trace, minus a
   background estimated outside the window: $F_m(x) = \sum_{j=-\delta}^{\delta} I(x, y_m(x)+j) - \text{bg}$.
3. **Wavelength mapping** — per-order polynomial $\lambda_m(x)$ from calibration-lamp lines.
4. **Order merging** — interpolate all orders onto a common linear grid (typically 0.05 nm) and
   merge overlaps (weighted-average / simple-average / max).

Coefficients live in a JSON calibration file (`order_N: {y_coeffs, wl_coeffs}`, numpy-`polyval`
order). Current limitations: simple summation (not optimal profile-weighted extraction), basic
median background, no automatic order detection, no cosmic-ray/bad-pixel handling. These are the
`InstrumentCalibration` Mode-A inputs upstream of everything in Part B.

### B.6 Wavelength calibration and drift {#wavelength-calibration}

The dispersion model is a normalized polynomial $\lambda(p) = \sum_k c_k P_k(\tilde p)$ with
$\tilde p = 2(p - p_{min})/(p_{max}-p_{min}) - 1 \in [-1,1]$; normalization is **mandatory** (raw
pixels in a monomial basis give a condition number ~$(N_{pix}/2)^N \approx 10^{17}$ for N=5, 2048 px
— numerically singular). Automated matching uses a Hough pre-filter in (slope, intercept) space
followed by RANSAC polynomial fitting (RASCAL-style), with iteration count
$T = \log(1-p_{success})/\log(1-w^s)$ ([Fischler & Bolles 1981][@fischler1981]). For a
pre-calibrated fiber spectrometer, drift is tracked by cross-correlation of a known reference
against the measured line, with sub-pixel parabolic interpolation; on a nonlinear dispersion the
pixel shift must be multiplied by the local $d\lambda/dp$ (a single global wavelength offset is
wrong). Even space-qualified spectrometers drift ([Wang et al. 2022, MarSCoDe on Zhurong][@wang2022]).
Full literature reference, pitfalls, and the correct-code checklist are in the impl chapter:
[impl-literature-methods.md](impl-literature-methods.md) and the source doc
[wavelength-calibration.md](../v4/overhaul/literature/wavelength-calibration.md).

> [!CAUTION] Air vs vacuum, again: NIST arc-line atlases give **air** wavelengths for λ > 200 nm;
> laser-frequency-comb values are **vacuum**. Convert with the single Edlén utility
> (`cflibs/atomic/wavelength_conversion.py`) — mixing conventions can swap closely-spaced line
> identities. For CF-LIBS line ID, require calibration accuracy < 0.1 nm (1σ); flag results if RMS
> > 0.2 nm.

---

## Part C — Datasets {#part-c}

Every benchmark spectrum the pipeline is scored against has a documented provenance, a source paper,
and an explicit truth model (quantitative wt% vs presence-only). Adapters live in
`cflibs/benchmark/datasets/` and plug into the `BenchmarkSpectrum` / `BenchmarkDataset` data model.
Truth is stored as certified mass fractions; the scoreboard scores quantitative RMSE where the panel
is complete and recall-on-panel where it is presence-only.

| Dataset (adapter) | Material / provenance | Truth model | Calib mode | Source |
|-------------------|-----------------------|-------------|------------|--------|
| `supercam_labcal` | SuperCam lab spectral library (721 MB PDS4; 1,193 base spectra, 334 standards) | quantitative major-oxide MOC | **B** | [Anderson et al. 2022][@anderson2022] |
| `supercam_scct` | 547 real-Mars SuperCam calibration-target spectra, sols 13–1694, 23 targets | quantitative (join to labcal) | **B** | Manrique 2020; [Anderson 2022][@anderson2022] |
| `chemcam_calib` | MSL ChemCam preflight cleanroom spectra, 66 standards (PDS CALIB) | quantitative oxide panel | **B** | [Wiens et al. 2012][@wiens2012] |
| `nist_steel` | NIST SRM 1261a–1265a low-alloy steels | quantitative (NIST CoA) | A/C | NIST Certificates of Analysis |
| `usgs` | BHVO-2 / AGV-2 / BCR-2 / G-2 silicate rocks (basalt→granite) | quantitative major-oxide | A/C | [Jochum et al. 2016][@jochum2016] |
| `csa_planetary` | CSA open planetary-analogue LIBS, 110 spectra | quantitative oxide + C | A/C | CSA open dataset |
| `emslibs2019` | EMSLIBS 2019 contest, 138 OREAS ore mixtures | **presence-only** (per-class panel) | C | Vrabel/Kepes et al. 2020 |
| `silva2022` | tropical-soil LIBS, 102 samples | **presence-only** (fertility panel) | C | Silva et al. 2022 |
| `gibbons2024` | nitrate-in-Mars-simulant, N quantitation, 180 spectra | quantitative N (stoichiometric) | C | Gibbons et al. 2024 |

**Provenance discipline worth noting.** The adapters record exactly *why* a truth is quantitative or
presence-only, and skip what cannot be grounded: `chemcam_calib` skips `M6-HAGGERTY` (no composition
row) and any standard whose certified oxide panel covers <50 wt% of the sample (e.g. the GYP gypsum
series, where S and structural water are uncertified). `emslibs2019` is presence-only because *which*
member of each contest class each 500-shot block is cannot be recovered from the published files, so
per-spectrum quantitative truth is not derivable without inventing a mapping. `silva2022` records
that column `V` is base-saturation %, **not** vanadium. `gibbons2024` derives N wt% from the pure
stoichiometric identity $N = \text{wt\%NO}_3^- \times M_N/M_{NO_3} = \text{wt\%} \times 0.225905$,
independent of hydration state, and marks pure-salt end-members presence-only because their NO₃⁻ mass
fraction depends on the unreported hydration state. This is the honesty that keeps the benchmark from
scoring against invented numbers.

### C.1 The DED deliverable — synthetic constrained-absolute corpus {#ded-datasets}

The real target is **directed-energy-deposition composition-drift tracking on a known, constrained
element set** — precision/ratios matter far more than absolute accuracy, oxides and geology are out
of scope. The synthetic benchmark is `tests/benchmarks/ded_precision/`
([alloy_definitions.py](../../tests/benchmarks/ded_precision/alloy_definitions.py)):

| Alloy | Constrained element set (wt%, sum=100, no O) | Window (nm) |
|-------|----------------------------------------------|-------------|
| **Ti-6Al-4V** (primary) | Ti 90 / Al 6 / V 4 | 250–500 |
| Inconel 625 | Ni 64.5 / Cr 22.5 / Mo 9.5 / Nb 3.5 | 280–520 |
| 316L | Fe 68 / Cr 17 / Ni 12 / Mo 3 | 250–500 |

The constrained-known-element set is the DED simplification: solve only for the alloy's declared
species, use the nominal feedstock as a prior, and bypass open-element identification entirely.
`make_series` scans one element across a drift axis (e.g. Al 4→8 wt%) while scaling the others
proportionally so each composition still sums to 100 — mimicking one species evaporating/enriching
while the rest keep their mutual ratios. **Prefer log-ratios $\ln(N_i/N_j)$ over closure wt%** for
the tracking deliverable; the reset DED result (Al 0.82 / Ti 1.43 / V 0.87 wt% RMSEP) is the current
floor on this corpus. See [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md)
for the scoreboard harness and [error-budget-and-falsification.md](error-budget-and-falsification.md)
for the V-bias and Cr-underestimate open items.

---

## What correct code MUST do — atomic-data & calibration checklist {#checklist}

- [ ] **Partition cutoff = Saha IPD.** Never compute an ionization-potential depression locally;
  call `cflibs/plasma/partition.py::ionization_potential_depression` (or its JAX twin) so the
  truncation invariant holds. A divergence biases composition.
- [ ] **Direct sum first, polynomial last.** Use `direct_sum_partition_function` for any levelled
  species; treat the Irwin polynomial as a warned last resort and never extrapolate it outside
  `[t_min, t_max]`.
- [ ] **Do not swap to a bigger raw line list for completeness.** Measured on both dev and held-out:
  it regresses. Only NIST-A/B ∪ downweighted backfill (R4) + grade-aware selection improves accuracy.
- [ ] **Restrict the Stark $n_e$ diagnostic** to `stark_w_source IN ('stark_b','interpolated')`; the
  heuristic-scaled majority is not a physical width.
- [ ] **Keep wavelengths air/vacuum-consistent.** DB is air > 200 nm; convert through the one Edlén
  utility; never mix with vacuum comb/LFC values.
- [ ] **Never treat CF-LIBS as removing $E(\lambda)$ or the LSF.** Only the scalar $F$ cancels via
  closure. $E(\lambda)$ is the dominant un-removed systematic.
- [ ] **Require Mode A or B for any quantitative result.** Mode C is flagged non-quantitative and may
  not flip a default or claim a SOTA number.
- [ ] **Keep line-center columns in f64.** In the columnar inference tier, demote only
  `aki`/`g`/`rel_int`/Stark to f32; f32 wavelength (~0.0001 nm at 500 nm) is too coarse.
- [ ] **Carry dataset provenance and truth-model honestly.** Presence-only where quantitative truth
  is not derivable; skip samples whose certified panel is incomplete rather than inventing values.

## See also

- [libs-physics.md](libs-physics.md) — Saha-Boltzmann forward model that reads these tables
- [classical-quantification.md](classical-quantification.md) — Boltzmann plot and closure that $F$ cancels through
- [cf-libs-family.md](cf-libs-family.md) — open-element ID re-architecting; iterative vs ILR vs C-sigma
- [error-budget-and-falsification.md](error-budget-and-falsification.md) — the accuracy-beats-completeness falsification ledger
- [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) — scoreboard, calibration-mode column, DED corpus
- [architecture.md](architecture/index.md) — `AtomicDataSource` seam and the two-tier storage design
- [formal-spec.md](formal-spec.md) — Boltzmann/Saha theorems the partition function normalizes

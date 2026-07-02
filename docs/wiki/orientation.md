---
slug: orientation
title: "Orientation, Reading Map & Reset Line"
chapter: orientation
order: 0
status: stable
register: orientation
summary: >
  The single entry point to the CF-LIBS wiki. Establishes the two separately-ranked
  program goals (DED Ti-6Al-4V drift-tracking by ratios; general absolute composition),
  the load-bearing error-budget thesis that INPUTS dominate the budget not the solver,
  the ASD59 reset line that invalidates every pre-reset benchmark number, the audience
  reading routes, the master chapter TOC, and the authoring contract (frontmatter schema,
  status tokens, citation discipline, glossary). Read this before any other page.
tags: [orientation, reading-map, reset-line, error-budget, ded, conventions, glossary, index]
updated: 2026-07-02
benchmarks_pre_reset: false
sources:
  - "@ciucci1999"
  - "@tognoni2007"
  - "@aragon2008"
  - "@aguilera2007"
  - "@cristoforetti2010"
  - "@kramida2024nist"
  - README.md
  - docs/Quick_Start_For_Scientists.md
  - docs/v4/atomic_db/COMPLETE-DB-RESET-STATUS.md
  - docs/research/physics-first-principles-audit.md
  - CLAUDE.md
related: [reset-line, status-legend, how-to-use, glossary, libs-physics, formal-spec]
supersedes:
  - README.md
  - docs/Quick_Start_For_Scientists.md
  - docs/index.rst
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Orientation, Reading Map & Reset Line

This is the single entry point to the CF-LIBS wiki. It tells you **what the project is**, **which two goals it is measured against**, **the one thesis that governs where accuracy comes from**, **the dated reset line every numeric claim is anchored to**, and **how to read the rest of the wiki** — whether you are a physicist, an engineer, or an autonomous agent.

CF-LIBS is a physics-based, Calibration-Free Laser-Induced Breakdown Spectroscopy library: it runs the full measurement→physics→inference pipeline (forward modeling, spectral inversion, plasma diagnostics, and GPU-accelerated manifold pre-computation) with **no machine learning in the shipped path** [`README.md`, `CLAUDE.md`]. If you read nothing else, read the [Reset Line](#reset-line) and the [Error-Budget Thesis](#error-budget-thesis): together they determine which numbers on this wiki you are allowed to quote and where the real accuracy levers live.

> [!IMPORTANT] RESET LINE — every numeric result on this wiki is measured against the **ASD59 reset baseline (2026-07-02)**. The atomic database was rebuilt from the gold-standard NIST ASD v5.9 dump; **all pre-reset benchmark numbers are invalidated**. Any page quoting a pre-reset figure carries a `benchmarks_pre_reset: true` banner. See [Reset Line](#reset-line) and the full detail in [atomic-data-and-datasets.md](atomic-data-and-datasets.md).

---

## What CF-LIBS Is {#what-this-is}

CF-LIBS implements calibration-free quantitative LIBS: it infers **elemental composition, plasma temperature $T$, and electron density $n_e$** from an emission spectrum **without matrix-matched calibration standards**, by assuming the plasma is in local thermodynamic equilibrium (LTE) and solving the Saha ionization balance and Boltzmann level populations self-consistently [@ciucci1999; @aragon2008]. The original CF-LIBS procedure is due to Ciucci, Corsi, Palleschi and co-workers [@ciucci1999]; this codebase is a modern, physics-first, differentiable, formally-specified re-implementation of that family plus its self-absorption-tolerant and columnar-density descendants.

| Stage | What it does | Chapter |
|-------|--------------|---------|
| **Forward model** | `PlasmaState(T, n_e, composition)` → Saha-Boltzmann populations → line emissivity → broadening → instrument convolution → synthetic spectrum | [libs-physics.md](libs-physics.md) |
| **Inversion** | measured spectrum → preprocessing → line identification → Boltzmann/Saha-Boltzmann fit → closure → $T$, $n_e$, composition + uncertainties | [classical-quantification.md](classical-quantification.md), [cf-libs-family.md](cf-libs-family.md) |
| **Manifold** | JAX `jit`/`vmap` batch-generates spectra over a parameter grid → HDF5/Zarr → fast nearest-neighbor inference | [frontier-methods.md](frontier-methods.md) |

**Physics-only constraint.** The shipped algorithm must not import `sklearn`, `torch`, `tensorflow`, `keras`, `flax`, `equinox`, `transformers`, `jax.nn`, or `jax.experimental.stax`. ML is allowed **only** in `cflibs/evolution/` (LLM-driven algorithm-optimization tooling). Enforcement is two-level: a Ruff `TID251` static ban and an AST blocklist scanner in `cflibs/evolution/evaluator.py` [`CLAUDE.md`]. This is a hard design decision, not a preference — see [architecture.md](architecture/index.md).

---

## The Two Program Goals (Ranked Separately) {#two-goals}

CF-LIBS is measured against **two goals that are ranked independently**. A change that helps one may be neutral or negative for the other; never average their scoreboards.

> [!IMPORTANT] Goal (a) and Goal (b) have **separate** error budgets, separate benchmarks, and separate acceptance gates. State which goal a claim serves.

### Goal (a) — DED Ti-6Al-4V drift tracking (the real goal) {#goal-a-ded}

Real-time composition-**drift** tracking during directed-energy-deposition (DED) additive manufacturing of a **known, constrained** element set — canonically {Ti, Al, V} for Ti-6Al-4V feedstock [`docs/research/physics-first-principles-audit.md`].

- **Precision and *ratios* matter far more than absolute wt%.** The deliverable is the drift of a log-ratio such as $\ln(N_V/N_{Ti})$ and $\ln(N_{Al}/N_{Ti})$, not the closure-normalized weight percent.
- The **nominal feedstock composition is a prior**; identification is *bypassed* (the element set is known).
- Oxides, geology, and unknown matrices are **out of scope** for this goal.
- Because the deliverable is a ratio, the shared closure denominator cancels, and imputed $n_e$ error moves DED composition only ~1–2% per decade by CF-LIBS cancellation [`docs/research/physics-first-principles-audit.md`].

### Goal (b) — Absolute composition, generally {#goal-b-absolute}

Standardless absolute quantification of arbitrary matrices: steel minors, geological majors/minors, unknown samples. Here closure normalization ($\sum_s C_s = 1$), undetected-mass completeness, molecular bands, oxide redox, and two-zone line-of-sight integration all become first-order [`docs/research/physics-first-principles-audit.md`]. The intrinsic accuracy/precision ceiling of an *ideal* CF-LIBS plasma was quantified by Tognoni et al. [@tognoni2007]; real data sits well above that floor for reasons the [error budget](#error-budget-thesis) makes explicit.

---

## The Error-Budget Thesis: Inputs Dominate, Not the Solver {#error-budget-thesis}

> **One-line thesis:** On real data, the CF-LIBS error budget is dominated by **input quality — atomic data, self-absorption, $n_e$ measurement, and instrument response — not by the inversion solver.** The solver floor on clean synthetic data is ~0; the real-data floor is atomic-data-limited [`docs/research/physics-first-principles-audit.md`].

This is the wiki's central organizing claim, and it is *falsifiable and tested*. The physics-first-principles audit triangulated 21 findings against code, literature, and the `cflibs-formal` Lean spec, collapsing them into eight root-cause clusters. The dominant clusters are all input-side:

| Cluster | Root cause | Verdict | Owner chapter |
|---------|-----------|---------|---------------|
| **A. Atomic-data scale** | absolute + relative $g\cdot A$ error is a *systematic, source-correlated bias*, not random variance | CONFIRMED | [atomic-data-and-datasets.md](atomic-data-and-datasets.md) |
| **B. $n_e$ imputed, not measured** | real-data $n_e$ is pressure-balance imputed; Stark-B / ion-line diagnostics exist but are unused | CONFIRMED | [error-budget-and-falsification.md](error-budget-and-falsification.md) |
| **C. Saha ladder forward/inverse asymmetry** | inverse truncates at stage II; forward uses stage III | CONFIRMED | [libs-physics.md](libs-physics.md) |
| **D. Thin fit vs thick data** | inference forwards are optically thin with no fitted optical depth | WEAKENED | [cf-libs-family.md](cf-libs-family.md) |
| **E. Closure slosh vs ratio reporting** | closure normalizes over the detected set; ratios already cancel it | WEAKENED | [classical-quantification.md](classical-quantification.md) |
| **F–H. Forward completeness / oxide / temporal-spatial** | molecular bands, oxygen balance, two-zone LOS absent | mixed | [frontier-methods.md](frontier-methods.md) |

The practical consequence for **Goal (a)** is a specific priority order: (1) report log-ratios not wt%; (2) fix atomic-data scale; (3) thick-line handling on the fit path; (4) Saha stage-III symmetry; (5) measure $n_e$ instead of imputing. The full ranked list, the falsification records, and the "do-not-do" ledger live in [error-budget-and-falsification.md](error-budget-and-falsification.md).

> [!NOTE] Notable falsified result carried forward as a warning: computing per-line optical depth $\tau$ from the *composition-derived* lower-level column density made real ChemCam BHVO-2 **worse** (a positive-feedback loop, composition→$\tau$→composition). It was replaced by an observable-anchored corrector. See the falsification block in [error-budget-and-falsification.md](error-budget-and-falsification.md).

---

## Reset Line {#reset-line}

> [!IMPORTANT] The ASD59 reset line is the dated boundary against which **every numeric claim on this wiki is measured**. Numbers computed before it are dead; mechanisms are retained.

On **2026-06-26** the atomic database was rebuilt from the gold-standard NIST ASD v5.9 SQL dump (`ASD59_dump.sql`), replacing a database that held only ~30% of lines, was missing many ionization potentials, and had **no stage-III data at all** [`docs/v4/atomic_db/COMPLETE-DB-RESET-STATUS.md`]. The rebuilt DB and its downstream re-benchmarks define the **ASD59 reset baseline (wiki label 2026-07-02)**.

| Quantity | Post-reset value | Source |
|----------|------------------|--------|
| Lines (stages I–III) | **203,695** | validated vs live NIST (18/18 levels, 17/17 IPs) |
| Energy levels (I–III) | **62,752** | `docs/v4/atomic_db/COMPLETENESS-VERIFICATION.md` |
| Ionization potentials | **324** | " |
| Underlying source | NIST ASD v5.9 [@kramida2024nist] | " |

**What the reset invalidates.** Every composition, F1, and RMSE number computed on the old DB is void — they were measured on a catalog missing ~70% of lines and all stage III. The mechanisms (Boltzmann plot, Saha correction, closure, ALIAS/comb identification) are unchanged; only the **magnitudes** are dead.

**What is corrected and must not be carried forward.** These pre-reset claims are contradicted and are *never* restated as current — the corrected position stands, the archived original holds the tombstone:

- ~~stage-III empty~~ → stage III is now present and enabled in the solver.
- ~~Fe I 425 levels~~ → the complete DB has the full Fe I level structure.
- ~~1-atm pressure-balance $n_e$ as the primary diagnostic~~ → Stark-width / ion-line diagnostics are the target; pressure-balance is a band-aid.
- ~~flat-path import shims (`cflibs.inversion.solver`, …) work~~ → those shims are gone; import from canonical sub-package paths only.
- ~~manifold-as-foundation~~ and ~~"joint solver beats iterative" without an adoption gate~~ → both refuted; see [error-budget-and-falsification.md](error-budget-and-falsification.md).

**First valid post-reset baselines** (illustrative, full detail and provenance in [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) and [atomic-data-and-datasets.md](atomic-data-and-datasets.md)): the DED-constrained Ti-6Al-4V pipeline and the Saha-Boltzmann forward model both pass on the gold-standard catalog; **open-element identification regressed** on the 7× denser catalog and needs re-architecting (not re-tuning) — a problem that is **secondary** to the DED goal because DED uses a known element set [`docs/v4/atomic_db/COMPLETE-DB-RESET-STATUS.md`].

---

## Reading Routes by Audience {#reading-routes}

The wiki has three natural reading orders. Pick your row; each chapter's frontmatter `summary` (reproduced in the [Master TOC](#master-toc)) lets you skip anything irrelevant.

| You are a… | Start with | Then | The point |
|------------|-----------|------|-----------|
| **Physicist / spectroscopist** | [libs-physics.md](libs-physics.md) | [classical-quantification.md](classical-quantification.md) → [cf-libs-family.md](cf-libs-family.md) → [frontier-methods.md](frontier-methods.md) | the equations, the CF-LIBS algorithm family, and where the literature methods live |
| **Engineer / integrator** | [architecture.md](architecture/index.md) | [impl-literature-methods.md](impl-literature-methods.md) → [impl-novel-techniques.md](impl-novel-techniques.md) → [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) | module map, ABCs, how methods are wired, and the reliability gates |
| **Autonomous agent** | [how-to-use](#how-to-use) (this page) | [formal-spec.md](formal-spec.md) (notation authority) → each chapter's frontmatter `summary` | the frontmatter schema, cross-link/citation contract, and the notation single-source-of-truth |
| **Anyone verifying a number** | [Reset Line](#reset-line) | [error-budget-and-falsification.md](error-budget-and-falsification.md) → [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) | which numbers are current and how they were gated |

> [!NOTE] FORMAL — the notation authority is [formal-spec.md](formal-spec.md). Every symbol on the wiki is pinned to a `cflibs-formal` Lean identifier and defined **once** there. Pages *link* to it and never redefine a canonical symbol.

---

## Master Table of Contents {#master-toc}

Each chapter is one file; its frontmatter `summary` is the agent-readable relevance judgment. Read the summary before the body.

| # | Chapter | Register | Summary (frontmatter) |
|---|---------|----------|-----------------------|
| 0 | [orientation.md](orientation.md) | orientation | This page: goals, error-budget thesis, reset line, reading routes, authoring contract, glossary. |
| — | **Foundations** | | |
| 1 | [formal-spec.md](formal-spec.md) | reference | **Notation authority.** Every symbol pinned to a `cflibs-formal` Lean theorem; soundness envelopes for the estimators. |
| 2 | [error-budget-and-falsification.md](error-budget-and-falsification.md) | review | The ranked error budget (inputs dominate) and the falsification ledger of every negative result + the do-not-do list. |
| — | **Physics & methods** | | |
| 3 | [libs-physics.md](libs-physics.md) | review | Saha ionization balance + Boltzmann level populations that fix line intensity in an LTE plasma — the forward-model core. |
| 4 | [classical-quantification.md](classical-quantification.md) | review | Boltzmann and multi-element Saha-Boltzmann plots [@aguilera2007], closure ($\sum C_s=1$), one-point calibration, log-ratio reporting. |
| 5 | [cf-libs-family.md](cf-libs-family.md) | review | The CF-LIBS algorithm lineage from Ciucci [@ciucci1999] through C-sigma / columnar-density self-absorption-tolerant variants. |
| — | **Frontier** | | |
| 6 | [frontier-methods.md](frontier-methods.md) | review | Monte-Carlo full-spectrum fitting, precomputed self-absorption + OPC, JAX manifold pre-computation, ML-adjacent (knowledge-only). |
| — | **Code** | | |
| 7 | [architecture.md](architecture/index.md) | handbook | Module map, the six inversion sub-packages, the pluggable ABCs, the physics-only constraint enforcement. |
| 8 | [impl-literature-methods.md](impl-literature-methods.md) | code-walkthrough | Where each published method (Boltzmann plot, Saha correction, Stark $n_e$, OPC, C-sigma) is implemented in `cflibs/`. |
| 9 | [impl-novel-techniques.md](impl-novel-techniques.md) | code-walkthrough | In-house techniques: observable-anchored self-absorption, refuse-to-report reliability gate, manifold coarse-to-fine. |
| — | **Data & operations** | | |
| 10 | [atomic-data-and-datasets.md](atomic-data-and-datasets.md) | handbook | The ASD59 atomic DB (203,695 lines / 62,752 levels / 324 IPs) [@kramida2024nist], partition functions, and the benchmark datasets. |
| 11 | [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) | handbook | Post-reset benchmark baselines, the reliability/adoption gates, and the day-to-day CLI + validation workflows. |
| — | **Reference** | | |
| R1 | [glossary.md](glossary.md) | reference | Plain-language term index + the canonical symbol table. |
| R2 | [bibliography.md](bibliography.md) | reference | The merged, de-duplicated, DOI-verified bibliography. |
| R3 | [MIGRATION.md](MIGRATION.md) | reference | The fate of the ~252 legacy docs: absorbed / archived / kept-live, and what this rebuild moved. |

---

## Status & Lineage Legend {#status-legend}

Two orthogonal vocabularies label content. **Engineering-decision tags** tell you whether to act; **lineage status** tells you where a claim sits in its scientific life.

### Engineering-decision inline tags (on any recommendation)

| Tag | Meaning |
|-----|---------|
| **SAFE-NOW** | Verified correct on the post-reset baseline; apply without a gate. |
| **BENCHMARK-GATED** | Flag-gated behavior; changing it requires a flag-off/flag-on scoreboard non-regression run. The repo has regressed **3×** on ungated identifier changes — always cite the gate and the measured delta. |
| **DESIGN-DECISION** | A deliberate architectural choice (e.g. physics-only constraint); do not "fix" it without an ADR. |
| **AUDIT-FIRST** | Depends on a claim not yet re-verified post-reset; audit before relying on it. |

### Lineage status (frontmatter `status:`)

| Token | Meaning |
|-------|---------|
| `stable` / `canonical` | Current, verified, safe to quote. |
| `draft` / `review` | In progress or under review; treat as provisional. |
| `frontier` | Active research direction; theory grounded in `cflibs-formal` + the audit, not yet a shipped default. |
| `falsified` | A negative result; the page **must** carry the falsification-record block. Never resurrect the rejected claim. |
| `superseded` / `archived` | Replaced; retains a tombstone redirect to the successor slug. |

### The `benchmarks_pre_reset` flag

Any page that cites a composition / F1 / RMSE number **must** set `benchmarks_pre_reset:` in frontmatter. `true` means the page predates the [Reset Line](#reset-line): it carries the PRE-RESET banner and **agents must not quote its numbers as current** — the mechanism is retained, the magnitude is dead.

### Admonition vocabulary (GitHub-flavored, renders in the repo browser and MkDocs)

```
> [!IMPORTANT] RESET LINE — numbers below are ASD59-reset baseline (2026-07-02); any pre-reset figure is labeled.
> [!NOTE] FORMAL — proven in cflibs-formal: lean:CflibsFormal/Saha.lean#saha_relation.
> [!WARNING] BENCHMARK-GATED — flag-gated; changing it requires a flag-off/flag-on scoreboard non-regression run.
> [!CAUTION] DEAD-END / DO-NOT — verified dead-end; do not re-attempt.
```

---

## How To Use This Wiki (Humans and Agents) {#how-to-use}

This wiki is a contract, not prose. The blocks below are copied verbatim to the top of every page; deviating from them **breaks agent retrieval, the generated code-map, and citation-integrity CI**. Read this section once, then copy the [frontmatter template](#frontmatter-template).

### The rules in one screen

1. **Every page opens with YAML frontmatter** (template below). Delete no *required* key.
2. **Slugs are stable forever.** The `slug` equals the filename stem. Never renumber or rename in place — a rename leaves a redirect stub (`supersedes:` + tombstone).
3. **`order` uses gaps of 10** (10, 20, 30…) so pages insert without renumbering. `index`/orientation is `order: 0`.
4. **The `summary` is what an agent reads *instead of* the body** to judge relevance. State the single most load-bearing claim.
5. **Cross-links come in four kinds — use the right one** ([syntax below](#cross-link-syntax)).
6. **Citations are non-negotiable.** Every scientific claim carries a real citation resolving to one BibTeX entry **with a verified DOI** in [`references.bib`](references.bib). Never invent a DOI; never inline a raw DOI in body prose. Cite *literature* for physics, the *code path* for "what the code does", the *Lean theorem* for "what is proven".
7. **Notation is defined once** in [formal-spec.md](formal-spec.md). Introducing a new symbol means adding it there; never redefine a canonical one.
8. **Wavelength convention is load-bearing.** Any page that stores or compares wavelengths **must state air-vs-vacuum in its first paragraph.** The DB stores **air** wavelengths per NIST/ASD; the single conversion util lives in `cflibs/core/`.
9. **The pre-reset labeling rule** ([status legend](#status-legend)): any pre-ASD59 number sets `benchmarks_pre_reset: true` and opens with the PRE-RESET banner.
10. **Negative results are first-class.** Where our own work falsified something, record it with the falsification block — honesty is a feature of this wiki, not an omission.

### Cross-link syntax {#cross-link-syntax}

| Kind | How | Example |
|------|-----|---------|
| doc→doc (human) | relative Markdown **with `.md`** + stable anchor | `[Boltzmann balance](libs-physics.md#ionization-ratio)` |
| doc→doc (agent graph) | frontmatter `related:` by slug | `related: [boltzmann-plot, lte-validity]` |
| doc→code | frontmatter `code_refs:` (authoritative, depth-independent); in prose, monospace `path::Symbol` | `` `cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver` `` |
| doc→lean | frontmatter `lean_refs:`; in prose monospace `lean:File.lean#thm` | `` `lean:CflibsFormal/Saha.lean#saha_relation` `` |

`code→doc` and `lean→doc` are **never hand-written**: they are meant to be inverted from every page's `code_refs`/`lean_refs` into a generated `docs/wiki/code-map.md`. That generator (`scripts/build_wiki_codemap.py`) and the `code-map.md` it would produce **do not exist yet** — codemap generation is out of scope for this rebuild (see [MIGRATION.md](MIGRATION.md)); until it lands, treat each page's frontmatter `code_refs`/`lean_refs` as the authoritative forward index. On any heading that is a cross-reference target, add an explicit stable anchor (`{#anchor}`) so links survive heading-text edits.

### Page shape

H1 == `title`. First paragraph == the 2-sentence orientation (mirror `summary`). Soft cap ~400–500 lines / one topic; over cap → split into sibling slugs and cross-link. Review-register pages are equation-forward, cited, and end with a **"What correct code MUST do"** checklist; handbook/reference pages are task-forward and skimmable. Prefer tables + short prose over long paragraphs (agent-chunkable). **End every page with a "See also" list.** Math: inline `$...$`, display `$$...$$` (MathJax via MkDocs) using the [formal-spec.md](formal-spec.md) symbols exactly. No page may depend on a MkDocs-only macro — everything reads correctly as raw Markdown.

### Frontmatter template (copy this) {#frontmatter-template}

```yaml
---
slug: saha-boltzmann                     # REQUIRED. Stable-forever kebab id == filename stem. Rename → redirect stub.
title: "Saha-Boltzmann Ionization Balance"   # REQUIRED. Human H1 (quote if it contains a colon).
chapter: libs-physics                    # REQUIRED. Owning chapter-directory slug. Drives generated nav.
order: 30                                # REQUIRED. Integer, gaps of 10 so pages insert without renumbering. index.md is order:0.
status: stable                           # REQUIRED. stable | draft | review | frontier | falsified | superseded | archived
register: review                         # REQUIRED. review | handbook | reference | code-walkthrough | orientation
summary: >                               # REQUIRED. 1-2 sentences an agent reads INSTEAD of the body. State the load-bearing claim.
  Saha ionization balance and the Boltzmann level population that together fix line
  intensity in an LTE plasma; the forward-model core. PF cutoff must share the Saha IPD.
tags: [saha, boltzmann, ionization, lte, electron-density, forward-model]   # REQUIRED. lowercase-kebab; reuse before minting.
updated: 2026-07-02                       # REQUIRED. ISO date of last substantive edit.
benchmarks_pre_reset: false               # REQUIRED if the page cites ANY composition/F1/RMSE number.
sources:                                  # REQUIRED. Provenance: literature by @bib-key; repo docs/code by repo-root-relative path.
  - "@aragon2008"
  - docs/v4/overhaul/literature/saha-boltzmann-lte.md
  - cflibs/plasma/saha_boltzmann.py
code_refs:                                # OPTIONAL but expected on impl/physics pages. Repo-root-relative, optional ::Symbol.
  - cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver
  - cflibs/plasma/partition.py
lean_refs:                                # OPTIONAL. File.lean#theorem in cflibs-formal. Cite the theorem id; never paste Lean source.
  - CflibsFormal/Saha.lean#saha_relation
related: [boltzmann-plot, lte-validity]   # OPTIONAL. Sibling slugs; renders as "See also".
supersedes: []                            # OPTIONAL. Old repo paths this page absorbs; each gets a tombstone redirect on archive.
---
```

### Falsification-record block (verbatim for every negative result)

```
> [!CAUTION] FALSIFIED: <one-line claim>
> - **Claim:** what was proposed.
> - **Predicted:** the expected improvement.
> - **Observed:** what actually happened.
> - **Verdict:** REJECTED; what replaced it.
> - **Evidence:** doc Issue/finding + `cflibs/...` path.
> - **Date:** 2026-07-02
```

Any page with `status: falsified` **must** use this block. The aggregated ledger lives in [error-budget-and-falsification.md](error-budget-and-falsification.md).

### Citations & references.bib {#citations}

The single source of truth is [`references.bib`](references.bib). Inline a Pandoc key in prose — `[@aragon2008]` or "Ciucci et al. [@ciucci1999] show …". Every key resolves to exactly one BibTeX entry, and **an entry without a verified `doi` may not be cited.** Seed `references.bib` from `docs/research/2026-06-19-cross-discipline-sota.md` (its DOIs were hand-corrected against Semantic Scholar). Frontmatter `sources:` lists the same `@keys` plus repo paths. If you cannot verify a DOI, write `[citation-needed]` explicitly rather than fabricate.

---

## Glossary {#glossary}

~40 domain terms. Symbols are defined canonically in [formal-spec.md](formal-spec.md); this glossary is the plain-language index.

| Term | Meaning |
|------|---------|
| **LIBS** | Laser-Induced Breakdown Spectroscopy: a laser pulse ablates a sample into a microplasma whose optical emission is spectrally resolved. |
| **CF-LIBS** | Calibration-Free LIBS: standardless quantification from the plasma's own physics (Saha-Boltzmann), no matrix-matched standards [@ciucci1999]. |
| **LTE** | Local Thermodynamic Equilibrium: collisions dominate radiation so a single $T$ fixes both ionization (Saha) and excitation (Boltzmann). |
| **pLTE** | partial LTE: upper levels equilibrated but ground/near-ground not — an LTE-validity failure mode [@cristoforetti2010]. |
| **McWhirter criterion** | The classic lower bound on $n_e$ for LTE to hold; necessary but not sufficient [@cristoforetti2010]. |
| **Saha equation** | Fixes the population ratio between adjacent ionization stages as a function of $T$, $n_e$, and partition functions. |
| **Boltzmann distribution** | Fixes level populations within a species: $n_k \propto g_k \exp(-E_k/k_B T)$. |
| **Boltzmann plot** | $y=\ln(I_{ki}\lambda/(g_k A_{ki}))$ vs $x=E_k$; slope $=-1/(k_B T)$ gives temperature. |
| **Saha-Boltzmann plot** | Multi-stage Boltzmann plot with a Saha correction mapping ion lines onto the neutral plane [@aguilera2007]. |
| **Partition function** $U_s(T)$ | Sum of $g\exp(-E/k_B T)$ over a species' levels; normalizes Boltzmann populations. |
| **IP / ionization potential** | Energy to remove an electron from a species; the Saha ladder rung. |
| **IPD** | Ionization-Potential Depression: plasma-density lowering of the effective IP; the PF cutoff must share the Saha IPD. |
| **Stage I / II / III** | Neutral / singly-ionized / doubly-ionized species (e.g. Fe I, Fe II, Fe III). |
| **$T$ (temperature)** | Excitation/plasma temperature (K, or eV where noted). |
| **$n_e$ (electron density)** | Free-electron number density (cm⁻³); a Saha input, ideally measured by Stark broadening. |
| **$A_{ki}$** | Einstein spontaneous-emission coefficient (s⁻¹); the transition probability. |
| **$g_k$** | Upper-level statistical weight, $2J+1$. |
| **$E_k$** | Upper-level energy (eV; stored as cm⁻¹ in the DB). |
| **$g\cdot A$** | The product carrying a line's intrinsic strength; source-correlated $g\cdot A$ bias is the dominant real-data error (Cluster A). |
| **Closure** | The constraint $\sum_s C_s = 1$ used to turn relative species densities into absolute fractions. |
| **Log-ratio / Aitchison ratio** | $\ln(N_i/N_j)$; matrix- and detected-set-invariant, the preferred DED deliverable [@aitchison1982]. |
| **ILR** | Isometric log-ratio transform: an orthonormal coordinate system for compositional data. |
| **OPC** | One-Point Calibration: a single reference point removes the CF-LIBS scale bias [`impl-literature-methods.md`]. |
| **C-sigma / CD-SB** | C-sigma graphs and Columnar-Density Saha-Boltzmann: self-absorption-tolerant curve-of-growth CF variants. |
| **Self-absorption** | Re-absorption of line emission along the line of sight; saturates strong lines ($I \propto 1-e^{-\tau}$). |
| **Optical depth $\tau$** | Line opacity; escape factor $SA(\tau)=(1-e^{-\tau})/\tau$. |
| **Optically thin / thick** | Regimes where emission is (thin) linear in column density or (thick) saturated. |
| **Curve of growth (COG)** | Line intensity vs column density; the thick branch grows as $\sqrt{\tau}$. |
| **Stark broadening** | Pressure broadening by charged particles; the preferred (physical) $n_e$ diagnostic. |
| **Forward model** | Parameters → synthetic spectrum ($T,n_e,\text{composition} \to$ spectrum). |
| **Inversion** | Spectrum → parameters ($T,n_e,\text{composition}$ from a measured spectrum). |
| **Line identification** | Assigning detected peaks to species/transitions (ALIAS, comb, correlation, NNLS). |
| **ALIAS** | An identification algorithm scoring matched vs expected emissivity; mis-tuned for the dense post-reset catalog. |
| **Comb matching** | High-recall identification by matching a species' line "comb" against the spectrum. |
| **Manifold** | A precomputed grid of synthetic spectra (HDF5/Zarr) enabling fast nearest-neighbor inference. |
| **DED** | Directed-Energy Deposition additive manufacturing; the drift-tracking deployment target (Goal a). |
| **Ti-6Al-4V** | The canonical DED titanium alloy ({Ti, Al, V}); the constrained-element tracking target. |
| **F_cal** | Scalar experimental/calibration factor; cancels via closure and in ratios. |
| **Refuse-to-report** | The reliability gate that withholds a composition when the fit is untrustworthy (the adoption gate). |
| **cflibs-formal** | The separate Lean 4 project proving the CF-LIBS physics; the notation authority. |
| **ASD59 / reset line** | NIST Atomic Spectra Database v5.9 [@kramida2024nist]; the rebuilt-DB baseline all numbers are measured against. |
| **Air vs vacuum wavelength** | The DB stores **air** wavelengths (NIST/ASD); any wavelength-handling page must state which it uses. |

---

## See Also {#see-also}

- [libs-physics.md](libs-physics.md) — the Saha-Boltzmann forward-model core.
- [classical-quantification.md](classical-quantification.md) — Boltzmann/Saha-Boltzmann plots, closure, log-ratio reporting.
- [cf-libs-family.md](cf-libs-family.md) — the CF-LIBS algorithm lineage and self-absorption-tolerant variants.
- [frontier-methods.md](frontier-methods.md) — Monte-Carlo full-spectrum fitting, precomputed SA, manifold.
- [formal-spec.md](formal-spec.md) — **notation authority** and Lean soundness envelopes.
- [error-budget-and-falsification.md](error-budget-and-falsification.md) — the ranked error budget and falsification ledger.
- [architecture.md](architecture/index.md) — module map, inversion sub-packages, ABCs, physics-only constraint.
- [impl-literature-methods.md](impl-literature-methods.md) / [impl-novel-techniques.md](impl-novel-techniques.md) — where methods are implemented.
- [atomic-data-and-datasets.md](atomic-data-and-datasets.md) — the ASD59 DB and datasets (full reset detail).
- [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) — post-reset baselines, reliability gates, CLI workflows.

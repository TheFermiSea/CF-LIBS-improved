---
slug: MIGRATION
title: "Migration Ledger: Fate of the Legacy Docs"
chapter: MIGRATION
order: 920
status: stable
register: reference
summary: >
  The disposition ledger for the ~252 legacy docs when the wiki became the single source of truth:
  what was ABSORBED into which chapter, what is KEPT-LIVE (CLAUDE.md, ADRs, README, the evidence
  ledger, cflibs-formal, examples/scripts, machine artifacts), and what was ARCHIVED. Records the
  moves executed this session and the files explicitly refused for safety.
tags: [migration, archive, absorb, keep-live, provenance, disposition]
updated: 2026-07-02
related: [index, orientation, architecture]
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Migration Ledger: Fate of the Legacy Docs

The `docs/wiki/` tree is the new **single source of truth**. The old documentation tree is not
deleted — it retains institutional-memory and falsification-history value — but it is emptied of
"current" prose. Every legacy doc has exactly one disposition: **ABSORB**, **ARCHIVE**,
**KEEP-IN-PLACE**, or **HARVEST-SPECIAL**. This page records those dispositions and the moves
executed in this rebuild.

> [!IMPORTANT] Two rules governed the mechanical migration. (1) **Absorb before archive** — no file's
> content is archived until it lands in a live chapter. (2) **Never move a file a test/script
> references, an ADR, `CLAUDE.md`, or `README.md`.** The executed move set (below) was grepped
> against `cflibs/ tests/ scripts/ .github/` first; anything referenced was left in place.

## Disposition classes

| Class | Meaning | Mechanism |
|-------|---------|-----------|
| **ABSORB** | Durable content merged into a wiki chapter; pre-reset *numbers* dropped, *mechanisms* kept. | Chapter frontmatter lists the original path in `supersedes:`. The original stays where it is until the full-migration pass tombstones it. |
| **ARCHIVE** | Stale / superseded / historical; not in wiki nav. | `git mv` into `docs/archive/…`, mechanisms harvested first. Executed subset this session: `docs/archive/research-pre-wiki/`. |
| **KEEP-IN-PLACE** | Canonical where it lives; the wiki *links*, never copies. | Untouched. |
| **HARVEST-SPECIAL** | Provenance traps (gitignored Serena memories, newest audit) whose facts are mirrored into chapters. | Content mirrored; source left as live runtime. |

## ABSORB map (content → chapter)

Each chapter's `supersedes:` frontmatter is the authoritative absorb record. Summary:

| Wiki chapter | Absorbed legacy sources (`supersedes:`) |
|--------------|------------------------------------------|
| [orientation.md](orientation.md) | `README.md` (narrative), `docs/Quick_Start_For_Scientists.md`, `docs/index.rst` |
| [libs-physics.md](libs-physics.md) | `docs/physics/Equations.md`, `docs/physics/Assumptions_And_Validity.md` |
| [classical-quantification.md](classical-quantification.md) | `docs/REFERENCE_ANALYSIS_LIBSSA.md` |
| [cf-libs-family.md](cf-libs-family.md) | literature packs (`docs/v4/overhaul/literature/cflibs-method.md`, `bayesian-oe.md`), `docs/M-spec-*` method verdicts |
| [error-budget-and-falsification.md](error-budget-and-falsification.md) | `docs/research/accuracy-first-roadmap.md`, `real-steel-accuracy-levers.md`, `real-steel-opc-promotion.md`, `real-data-accuracy-program-summary.md` |
| [formal-spec.md](formal-spec.md) | `docs/v4/overhaul/verified/atomic.md`, `docs/v4/overhaul/adversarial/partition-functions.md` (superseded atomic premises only) |
| [architecture/index.md](architecture/index.md) | `docs/v4/overhaul/{ARCHITECTURE,BLUEPRINT,BLUEPRINT-ADDENDUM,CRITIC}.md`, `docs/v4/pipeline-audit`, `CF-LIBS_Technical_Documentation` |
| [impl-literature-methods.md](impl-literature-methods.md) | `docs/API_Reference.md`, `docs/CF-LIBS_Codebase_Technical_Documentation.md`, the `docs/v4/overhaul/verified/*` module ledger (narrated, not rewritten) |
| [impl-novel-techniques.md](impl-novel-techniques.md) | `docs/M5-parameter-optimization-plan.md`, `docs/research/realtime/…v4…`, `docs/v4/overhaul/verified/manifold.md` |
| [atomic-data-and-datasets.md](atomic-data-and-datasets.md) | `docs/v4/atomic_db/{COMPLETE-DB-RESET-STATUS,COMPLETENESS-VERIFICATION}.md`, `docs/M5-atomic-db-heldout-verdict.md`, `docs/M5-db-accuracy-findings.md`, `docs/atomic-db-latency-ADR-0007-investigation.md`, `docs/Database_Generation.md`, `docs/Echellogram_Processing_Guide.md`, `docs/research/data-acquisition-plan.md` |
| [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) | `docs/User_Guide.md`, `docs/Deployment.md`, `docs/M-spec-{lever-comparison,target-sigma-wiring-findings}.md`, `docs/VALIDATION_METRICS.md`, `docs/{arbor-integration,dataset-sharding,jax-compile-cache,nfs-shared-data,results-parquet-schema,federation}.md` |

> [!NOTE] `supersedes:` means the **content** was absorbed, not that the file was moved. Several
> absorbed originals are **still referenced by code/tests** (`real-steel-accuracy-levers.md`,
> `real-steel-opc-promotion.md`, `realtime/…v4…`) or are flagged **CURRENT** by the doc audit
> (`data-acquisition-plan.md`, `physics-first-principles-audit.md`). Those were **left in place** this
> session; tombstoning them is deferred to the full-migration pass so no live reference breaks.

## KEEP-IN-PLACE (linked, never copied)

- **`CLAUDE.md`** (root) — the agent operating brief. Stays canonical; de-dup against the wiki is future work.
- **`README.md`** (root) — slim to a pointer at `docs/wiki/index.md` in the full pass; NOT moved.
- **`docs/adr/ADR-000{2,4,5,6,7}`** — durable decision records. [architecture/adr-index.md](architecture/adr-index.md) annotates; chapters link.
- **Evidence ledger** `docs/v4/overhaul/verified/*.md` (21), `docs/v4/overhaul/adversarial/ADJUDICATION.md` + domain files — the authoritative file:line "what's actually wrong" reference, incl. its FALSE-positive callouts. The impl chapters narrate from it and link back; it is a verification artifact, not rewritten to prose.
- **`cflibs-formal` `*.lean`** — separate repo; [formal-spec.md](formal-spec.md) cites theorem ids, never vendors source.
- **`examples/*`, `scripts/**/README.md`, `tests/benchmarks/**` docstrings** — code-adjacent; linked from chapters.
- **Machine artifacts** — `docs/reports/**/figures/*.png`, benchmark manifests, `*.jsonl`, ADR baseline `*.txt/*.json`, `docs/research/realtime/bench-*.json`, `varpro-sweep-*.csv`, `docs/_static` — referenced by relative link, never moved.

## HARVEST-SPECIAL

- **`.serena/memories/` (10 files)** are gitignored and exist only in the **main** worktree, not this wiki checkout. Their load-bearing facts were mirrored into chapters (evolution blocklist → `architecture/physics-only-constraint.md`; physics invariants → `libs-physics.md` + `benchmarks-reliability-workflows.md`; data-flow → `architecture/`; workflows/commands → `benchmarks-reliability-workflows.md`). The memory files remain live Serena runtime.
- **`docs/research/physics-first-principles-audit.md` (2026-07-02)** post-dates and reinterprets the program; harvested first as the conscience of [frontier-methods.md](frontier-methods.md) + the do-not-do list in [error-budget-and-falsification.md](error-budget-and-falsification.md). **KEEP-IN-PLACE** (newest authoritative).

## ARCHIVE — executed this session

Moved via `git mv` into **`docs/archive/research-pre-wiki/`** (nothing deleted). All are pre-ASD59-reset
identifier-F1 findings and superseded real-time plans; their mechanisms were harvested into
[impl-literature-methods.md](impl-literature-methods.md) (Vrabel `boltzmann_r2_min` × cold-T) and
[error-budget-and-falsification.md](error-budget-and-falsification.md) (`gemini_alias` failure modes)
before archiving.

| Moved (count) | From → To |
|---------------|-----------|
| **36** identifier-F1 findings (`asta-01…30`, `empirical-01…06`) | `docs/research/findings/*.md` → `docs/archive/research-pre-wiki/findings/` |
| **1** findings index | `docs/research/findings.jsonl` → `docs/archive/research-pre-wiki/` |
| **1** ALIAS diagnostic | `docs/research/gemini_alias_performance_diagnostic.md` → `docs/archive/research-pre-wiki/` |
| **1** Vrabel diagnosis | `docs/research/vrabel-universal-miss-diagnosis-2026-05-14.md` → `docs/archive/research-pre-wiki/` |
| **2** superseded realtime plans (v2 1ms, v3 structured-Jacobian) | `docs/research/realtime/{2026-06-19-realtime-1ms-plan,2026-06-20-realtime-plan-v3-structured-jacobian}.md` → `docs/archive/research-pre-wiki/realtime/` |

**Total moved: 41 files.**

## Refused to move (left in place, with reason)

| File(s) | Reason |
|---------|--------|
| `docs/research/findings/2026-05-14-v2-empirical-07-alias-fix-sweep.{md,csv}` | Referenced by `cflibs/benchmark/unified.py` (sweep-winner comment). |
| `docs/research/findings/README.md` | A `README`; excluded by rule. |
| `docs/research/real-steel-accuracy-levers.md`, `real-steel-opc-promotion.md` | Referenced by `tests/benchmarks/real_steel/*` and `cflibs/inversion/physics/{opc,self_absorption,line_selection}.py`, `cflibs/inversion/identify/alias.py`. |
| `docs/research/realtime/2026-06-20-realtime-plan-v4-real-data-accuracy.md` | Referenced by `scripts/research/rtval/README_rtval.md` (and the current realtime plan). |
| `docs/research/2026-06-19-cross-discipline-sota.md` | The DOI-corrected citation seed for `references.bib`; audit marks CURRENT. |
| `docs/research/{accuracy-first-roadmap,data-acquisition-plan,physics-first-principles-audit,real-data-accuracy-program-summary,2026-06-19-phase3-implementation-plan}.md`, `realtime/varpro-datavoyager-analysis.md` | Audit marks CURRENT/authoritative or content-absorbed-but-referenced; deferred to full-migration tombstone pass. |
| `CLAUDE.md`, `README.md`, `docs/adr/**` | Kept-live by rule. |

## Deferred to the full-migration pass (not executed here)

The broad migration policy (tombstone headers on every archived/absorbed original with a resolvable
`superseded_by`; slim `README.md` to a 10-line pointer; retire `docs/conf.py` + `docs/index.rst`;
de-duplicate `CLAUDE.md` against the architecture/workflows chapters; archive the ~120 stale
`docs/**` + `docs/v4/overhaul/specs` + legacy how-tos with `contradicted_by:` fields; generate
`docs/wiki/code-map.md` from frontmatter) is intentionally **out of scope** for this rebuild, which
executed only the unambiguously-safe `docs/research` archival. Those steps require touching many
code-referenced and ADR files and should run as their own reviewed change.

## See also {#see-also}

- [index.md](index.md) — the wiki front door and chapter map.
- [orientation.md](orientation.md) — goals, reset line, authoring contract.
- [architecture/adr-index.md](architecture/adr-index.md) — the kept-live ADRs.

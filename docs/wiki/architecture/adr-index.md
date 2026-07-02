---
slug: adr-index
title: "ADR Index (Annotated)"
chapter: architecture
order: 100
status: stable
register: reference
summary: >
  Annotated index into docs/adr/ — the canonical Architecture Decision Records. This page points to
  them and says why each matters; it never copies their text. Covers ADR-0001 (host/kernel split),
  ADR-0002 (no generic predictor factory), ADR-0004 (jittable pipeline), ADR-0005 (jitpipe
  promotion), ADR-0006 (instrument calibration first-class), and the DA-1..DA-8 design decisions.
tags: [adr, decisions, index, jitpipe, reference]
updated: 2026-07-02
sources:
  - docs/adr/ADR-0002-no-generic-predictor-factory.md
  - docs/adr/ADR-0004-jittable-inversion-pipeline.md
  - docs/adr/ADR-0005-jitpipe-promotion.md
  - docs/adr/ADR-0006-instrument-calibration-first-class.md
  - docs/v4/overhaul/ARCHITECTURE.md
related: [two-backend-pipeline, abstractions, change-taxonomy, target-architecture]
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# ADR Index (Annotated)

The Architecture Decision Records under [`docs/adr/`](../../adr/) are **canonical**. This page is a
pointer with annotations — why each decision matters and where in this chapter it lands. It does not
restate the ADRs; follow the links for the binding text.

## The accepted ADRs

| ADR | Title | Status | Why it matters |
|-----|-------|--------|----------------|
| [ADR-0001](../../adr/ADR-0001-radis-jaxrts-pattern-survey.md) | RADIS / jaxrts pattern survey — host/kernel split, unified forward kernel | Accepted | Establishes the **host/kernel boundary**: SQLite/I/O/CLI stay host-side; pure functions over arrays are kernels. This is the pattern ADR-0004 applies to the whole pipeline. Also the source of the T1-6 forward-model **named registry** (`inversion/forward_models/`). |
| [ADR-0002](../../adr/ADR-0002-no-generic-predictor-factory.md) | Do not extend `_make_predictor` to the standalone non-ALIAS identifiers | Accepted | The canonical statement of "**symmetry is not depth**". Rejects a factory abstraction for comb/correlation/nnls/hybrid because it adds visual symmetry, not locality or leverage. Cited on [abstractions](abstractions.md) as the boundary of where an abstraction earns its keep. |
| [ADR-0004](../../adr/ADR-0004-jittable-inversion-pipeline.md) | Ground-up jittable inversion pipeline (`cflibs/jitpipe/`) | Accepted & implemented | The **two-backend decision**. Eight binding sub-decisions (D1–D8): parallel-not-a-port, fixed shapes end-to-end, one-way imports, host/kernel boundary, exact front-end + re-derived back-end, scoreboard promotion, physics-only unchanged, fp64 default. The spine of [two-backend-pipeline](two-backend-pipeline.md). |
| [ADR-0005](../../adr/ADR-0005-jitpipe-promotion.md) | Promote the jit pipeline for batched/GPU inference | Accepted (2026-06-19) | The **promotion verdict**: jitpipe cleared the M3 gate and became the selected engine for batched/GPU + campaign evaluation; the reference stays the single-spectrum default and parity oracle. Records the honest deviation (capped board vs full board). |
| [ADR-0006](../../adr/ADR-0006-instrument-calibration-first-class.md) | Instrument calibration as a first-class input | Accepted | Elevates instrument/response calibration to a first-class **input-quality lever** — a sibling to the atomic-data and solver levers. Relevant wherever the response curve rotates the Boltzmann slope. |

ADR-0003 was **pruned** in an architecture-review sweep (its premise did not survive verification);
ADR-0002 was kept. Do not cite ADR-0003 as live.

## The DA-1…DA-8 design decisions

The v4 overhaul recorded eight design decisions in `docs/v4/overhaul/ARCHITECTURE.md` §11 (they are
architecture-note decisions, not standalone ADR files). They are the DESIGN-DECISION / SAFE-NOW /
AUDIT-FIRST items the [change taxonomy](change-taxonomy.md) points at:

| id | Decision | Class | Landed on |
|----|----------|-------|-----------|
| DA-1 | Response-curve single canonical path (`instrument/kernels.py`) | DESIGN-DECISION | [target-architecture](target-architecture.md) §2 |
| DA-2 | `BroadeningMode` default → `PHYSICAL_DOPPLER` (benchmark-gated flip) | DESIGN-DECISION | [change-taxonomy](change-taxonomy.md) |
| DA-3 | Rust kdet branch: port vs. document-as-future | DESIGN-DECISION | [module-map](module-map.md) (Rust core) |
| DA-4 | Air/vacuum convention: DB stores air; single conversion utility | AUDIT-FIRST | [target-architecture](target-architecture.md#air-vacuum) |
| DA-5 | `CFLIBSConfig` typed config replaces env-var sprawl | SAFE-NOW | [target-architecture](target-architecture.md#cflibsconfig) |
| DA-6 | IPD/PF truncation consistency via a single `compute_ipd()` | SAFE-NOW | [target-architecture](target-architecture.md#pf-ipd-invariant) |
| DA-7 | `NoiseModel`: shot noise on clean signal; variance from params | SAFE-NOW | [target-architecture](target-architecture.md) §6 |
| DA-8 | `domain/` leaf for `CFLIBSResult`/`LineObservation`/`AtomicSnapshot` | SAFE-NOW | [target-architecture](target-architecture.md#domain-leaf) |

## How to write a new ADR

Keep the canonical text in `docs/adr/`; this wiki links, never copies. A new ADR gets the next number,
states context → decision → why → consequences (the ADR-0002 shape is a good compact template), and —
if it supersedes an older one — the older ADR gets a status line pointing forward. When an ADR is
pruned (like ADR-0003), leave a tombstone status rather than deleting the file, so the number is never
silently reused.

## See also

- [Two-backend pipeline](two-backend-pipeline.md) — ADR-0004/0005 in narrative form.
- [Change taxonomy](change-taxonomy.md) — the classes DA-1…DA-8 carry.
- [Target architecture](target-architecture.md) — where the SAFE-NOW/AUDIT-FIRST decisions land.
- [`docs/adr/`](../../adr/) — the canonical records themselves.

---
slug: index
title: "CF-LIBS Wiki"
chapter: index
order: 0
status: stable
register: orientation
summary: >
  The single front door to the CF-LIBS wiki: the ASD59 reset banner, the three audience reading
  routes (physicist / engineer / agent / verifier), the ordered chapter map with one-line
  summaries, and the conventions pointer. Read orientation.md next for the full goals + error-budget
  thesis + authoring contract.
tags: [index, front-door, reading-map, chapter-map, conventions, reset-line]
updated: 2026-07-02
related: [orientation, formal-spec, error-budget-and-falsification, glossary, bibliography]
---

# CF-LIBS Wiki

This is the front door. **CF-LIBS** is a physics-based, Calibration-Free Laser-Induced Breakdown
Spectroscopy library: it runs the full measurement→physics→inference pipeline — forward modeling,
spectral inversion, plasma diagnostics, and GPU-accelerated manifold pre-computation — with **no
machine learning in the shipped path** (a hard [physics-only constraint](architecture/physics-only-constraint.md)).
If you read one page first, read **[orientation.md](orientation.md)**: it states the two program
goals, the error-budget thesis, the reset line, and the authoring contract that every other page
obeys.

> [!IMPORTANT] RESET LINE — every numeric result on this wiki is measured against the **ASD59 reset
> baseline (2026-07-02)**. The atomic database was rebuilt from the gold-standard NIST ASD v5.9 dump
> [@kramida2024nist]; **all pre-reset benchmark numbers are invalidated.** Any page quoting a
> pre-reset figure carries a `benchmarks_pre_reset: true` banner. Full detail:
> [atomic-data-and-datasets.md](atomic-data-and-datasets.md); thesis: [orientation.md](orientation.md#error-budget-thesis).

## Reading routes {#routes}

Pick your row. Each chapter's frontmatter `summary` lets an agent skip anything irrelevant for ~20–40
tokens before a body read.

| You are a… | Start here | Then | The payoff |
|------------|-----------|------|------------|
| **Physicist / spectroscopist** | [libs-physics.md](libs-physics.md) | [classical-quantification.md](classical-quantification.md) → [cf-libs-family.md](cf-libs-family.md) → [frontier-methods.md](frontier-methods.md) | the emission physics, the CF-LIBS algorithm family, and the open frontier |
| **Engineer / integrator** | [architecture/index.md](architecture/index.md) | [impl-literature-methods.md](impl-literature-methods.md) → [impl-novel-techniques.md](impl-novel-techniques.md) → [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) | module map, ABCs, where methods are wired, and the reliability gates |
| **Autonomous agent** | [orientation.md](orientation.md#how-to-use) | [formal-spec.md](formal-spec.md) (notation authority) → each chapter's frontmatter `summary` | the frontmatter schema, cross-link/citation contract, and the single notation source of truth |
| **Anyone verifying a number** | [orientation.md](orientation.md#reset-line) | [error-budget-and-falsification.md](error-budget-and-falsification.md) → [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) | which numbers are current and how they were gated |

## Chapter map {#chapters}

The book arc runs **orientation → foundations → physics & methods → frontier → code → data &
operations → reference.** Reading order lives in frontmatter `order:`; slugs are stable forever.

| # | Chapter | Register | One line |
|---|---------|----------|----------|
| 0 | [orientation.md](orientation.md) | orientation | Goals, error-budget thesis, reset line, reading routes, the full authoring contract, glossary. **Read first.** |
| — | **Foundations** | | |
| 1 | [formal-spec.md](formal-spec.md) | reference | **Notation authority.** The machine-verified cflibs-formal Lean spec; every runtime gate pinned to a theorem + its soundness envelope. |
| 2 | [error-budget-and-falsification.md](error-budget-and-falsification.md) | review | The inputs-dominate thesis, the eight root-cause clusters, and the durable ledger of verified dead-ends (the do-not-do list). |
| — | **Physics & methods** | | |
| 3 | [libs-physics.md](libs-physics.md) | review | The full LTE laser-plasma emission chain: Saha-Boltzmann, emissivity, broadening, self-absorption, LTE validity. |
| 4 | [classical-quantification.md](classical-quantification.md) | review | Pre-calibration-free landscape: calibration curves, internal standards, the five-channel matrix-effect taxonomy, chemometrics (cited concept only). |
| 5 | [cf-libs-family.md](cf-libs-family.md) | review | The standardless family — classic CF-LIBS, OPC, C-sigma, CD-SB — as one algebraic object; what CF-LIBS provably removes (only scalar $F$). |
| — | **Frontier** | | |
| 6 | [frontier-methods.md](frontier-methods.md) | review | Nine theorem-licensed research thrusts: log-ratio tracking, measured $n_e$, in-plasma self-calibration, refuse-to-report, differentiable forward. |
| — | **Code** | | |
| 7 | [architecture/index.md](architecture/index.md) | handbook | The codebase: 29 packages, the two-backend (NumPy vs jitpipe) pipeline, ABCs, physics-only enforcement, target architecture. |
| 8 | [impl-literature-methods.md](impl-literature-methods.md) | code-walkthrough | Where each published method (forward model, RANSAC, the six identifiers, Boltzmann/closure, Stark, iterative/Bayesian/joint solvers) lives in `cflibs/`. |
| 9 | [impl-novel-techniques.md](impl-novel-techniques.md) | code-walkthrough | Our in-house techniques: Lean-derived thresholds, observable-anchored self-absorption, log-ratio reporting, jitpipe, M7/M8 gates. |
| — | **Data & operations** | | |
| 10 | [atomic-data-and-datasets.md](atomic-data-and-datasets.md) | handbook | The two input-quality levers (atomic-data provenance, instrument calibration) + benchmark-dataset provenance. The ASD59 DB detail lives here. |
| 11 | [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) | handbook | Reliability gates, the five UQ surfaces, the benchmark harness + non-regression discipline, campaign-1 anti-overfit, the CLI, contributing, cluster deploy. |
| — | **Reference** | | |
| R1 | [glossary.md](glossary.md) | reference | Plain-language term index + the canonical symbol table. |
| R2 | [bibliography.md](bibliography.md) | reference | The merged, de-duplicated, DOI-verified bibliography (96 entries). |
| R3 | [MIGRATION.md](MIGRATION.md) | reference | The fate of the ~252 legacy docs: absorbed / archived / kept-live, and what this rebuild moved. |

## Conventions {#conventions}

The wiki is a contract, not just prose. The one-screen version:

- **Every page opens with YAML frontmatter** (`slug`, `title`, `chapter`, `order`, `status`,
  `register`, `summary`, `tags`, `updated`, plus `sources`/`code_refs`/`lean_refs`/`related` as
  applicable). The `summary` is what an agent reads *instead of* the body.
- **Slugs are stable forever**; reading order lives only in `order:` (gaps of 10). A rename leaves a
  redirect stub, never a broken link.
- **Cross-links come in four kinds**: doc→doc (relative Markdown **with `.md`** + stable `{#anchor}`),
  doc→doc-graph (`related:` by slug), doc→code (`code_refs:` `path::Symbol`), doc→lean (`lean_refs:`
  `File.lean#thm`). `code→doc`/`lean→doc` are generated, never hand-written.
- **Citations are non-negotiable.** Every scientific claim carries a `[@key]` resolving to one entry
  **with a verified DOI** in [bibliography.md](bibliography.md) / [`references.bib`](references.bib).
  Never fabricate a DOI; write `[citation-needed]` instead.
- **Notation is defined once** in [formal-spec.md](formal-spec.md); air-vs-vacuum wavelength must be
  stated on any page that handles wavelengths.
- **Pre-reset numbers** carry `benchmarks_pre_reset: true` + the PRE-RESET banner. **Negative results**
  are first-class and use the falsification block.

The full contract, frontmatter template, and status/lineage token legend are in
[orientation.md](orientation.md#how-to-use).

## See also {#see-also}

- [orientation.md](orientation.md) — the deep entry chapter (goals, thesis, reset line, contract).
- [glossary.md](glossary.md) · [bibliography.md](bibliography.md) · [MIGRATION.md](MIGRATION.md).
- [formal-spec.md](formal-spec.md) — the notation authority.

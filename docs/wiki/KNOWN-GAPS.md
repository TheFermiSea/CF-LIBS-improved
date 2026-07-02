---
slug: known-gaps
title: "Known Gaps & Deferred Verification"
chapter: index
order: 950
status: draft
register: reference
summary: >
  Residual items from the 2026-07-02 verification sweep: findings that were documented rather than
  fully closed (unbuilt tooling, citations still needing a DOI or a located source, and one
  DB-vs-wiki drift noticed in passing). Everything a HIGH or cheap-MEDIUM finding could fix in place
  has been fixed; this file is the honest tail.
tags: [known-gaps, verification, citations, deferred, backlog]
updated: 2026-07-02
related: [bibliography, orientation, MIGRATION]
---

# Known Gaps & Deferred Verification

This file records verification findings that were **not** fully closed by the 2026-07-02 fix pass,
with the reason each was deferred. Every fabricated citation was removed or replaced (never kept);
every stale code/Lean anchor was corrected against the actual sources. What remains below is either
work that requires building/authoring something new, or a citation whose real source still has to be
located, or an honest cross-check flag.

## Unbuilt tooling (HIGH, doc-corrected, tool still owed)

- **`scripts/build_wiki_codemap.py` + `docs/wiki/code-map.md` do not exist.** The doc→code / lean→doc
  reverse index the cross-reference convention refers to has never been generated. The wiki prose in
  `orientation.md` and `benchmarks-reliability-workflows.md` was reworded to describe the generator
  as **planned / out of scope for this rebuild** (consistent with `MIGRATION.md`), rather than as a
  present tool. **Remaining work:** write the generator (it reads only frontmatter `code_refs` /
  `lean_refs`) and emit `docs/wiki/code-map.md`, then restore the "regenerate whenever frontmatter
  changes" instruction as an operative step.

## Citations still needing a source or DOI

- **Doublet/multiplet-ratio self-absorption (was `pace2025`).** The former `pace2025` entry carried a
  DOI (`10.1016/j.sab.2025.107361`) that resolves to an **unrelated XRF glass-forensics paper**, and
  the claimed title could not be verified. The fabricated entry was **deleted** from
  `references.bib` / `bibliography.md`, and the two in-prose citations in `impl-novel-techniques.md`
  §3.3 (rung (a) and the width/ratio paragraph) were changed to **`[citation-needed]`**. **Remaining
  work:** locate the real doublet-ratio / same-upper-level self-absorption reference (SAF-LIBS or
  El Sherbini–style doublet literature; the review `@rezaei2020` is a candidate anchor) and cite it.

- **`zhao2018`** — "Self-absorption correction for single-sample-calibration LIBS." Kept with an
  honest **DOI VERIFICATION PENDING** flag (cited in `impl-novel-techniques.md` OPC comparison). The
  specific 2018 paper was not confirmable this session. **Remaining work:** resolve the DOI or drop.

- **`djurovic2023`** — JPCRD 52(3):031503 critical review. The paper is confirmed to exist
  (ADS `2023JPCRD..52c1503D`) but the DOI field is still blank (**DOI VERIFICATION PENDING**).
  **Remaining work:** fill the verified DOI and drop the flag.

- **`qiu2026`** author list. Crossref co-authors for `10.1016/j.sab.2026.107520` appear to be
  Lu, Pu (Gornushkin possibly absent), contradicting the recorded `Qiu, Y. and Gornushkin, I. B.`.
  Left **unchanged** pending confirmation of the full author list rather than guessing. **Remaining
  work:** verify and correct the author list.

## Cosmetic / low-severity, annotated in place

- **`favre2024`, `favre2024merlin`** — Crossref publication year is **2025** (online 2024). The bib
  `note` fields now say so; the `year = {2024}` field was left as-is to avoid churning in-prose
  "Favre 2024" labels. Cosmetic.
- **`hoffman2014`** — JMLR has **no registered DOI**; the non-resolving ACM id was replaced by
  `arXiv:1111.4246` (bib) / an arXiv link (bibliography.md). Not a gap — recorded for provenance.

## Cross-check flag noticed in passing (not re-audited here)

- **Live-DB Stark counts differ from the wiki's authoring-time snapshot.** The wiki (and
  `error-budget` lever 5) cite **244** STARK-B lines out of **~28,727** with a `stark_w_source`
  (0.85%); a spot query of the current `ASD_da/libs_production.db` returned `stark_w_source='stark_b'`
  = **693** and `interpolated` ≈ 70,133. The atomic-data / error-budget denominators were made
  **internally consistent** (both now say "244 of ~28,727"), but the raw counts were **not**
  re-derived against the live DB in this pass. **Remaining work:** re-audit the Stark-coverage
  numbers against the current DB and update both chapters together if they have drifted.

## See also

- [bibliography.md](bibliography.md) — the DOI-verification backlog (`@djurovic2023`, `@zhao2018`).
- [MIGRATION.md](MIGRATION.md) — the codemap generator is listed there as out of scope.
- [orientation.md](orientation.md#how-to-use) — the citation contract these gaps are measured against.

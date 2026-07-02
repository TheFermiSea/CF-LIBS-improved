---
slug: change-taxonomy
title: "Change Taxonomy: SAFE-NOW / BENCHMARK-GATED / DESIGN-DECISION"
chapter: architecture
order: 90
status: stable
register: handbook
summary: >
  The engineering-decision vocabulary that governs every change to CF-LIBS — SAFE-NOW,
  BENCHMARK-GATED, DESIGN-DECISION, AUDIT-FIRST — and the FALSE-exclusion list of verified-false
  findings that must NOT be 'fixed' because acting on them would introduce a bug.
tags: [change-taxonomy, safe-now, benchmark-gated, design-decision, false-exclusion, workflow]
updated: 2026-07-02
sources:
  - docs/v4/overhaul/BLUEPRINT.md
  - docs/v4/overhaul/BLUEPRINT-ADDENDUM.md
  - docs/v4/overhaul/CRITIC.md
related: [target-architecture, two-backend-pipeline, adr-index, physics-only-constraint]
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Change Taxonomy: SAFE-NOW / BENCHMARK-GATED / DESIGN-DECISION

Every proposed change to CF-LIBS carries one of four tags. The tag decides what "done" means and what
gate a change must clear. This vocabulary is the discipline that keeps accuracy work from silently
regressing — the repo has regressed identification F1 before (−0.041, PR #229), so the gates are not
ceremony. Use these tags inline on any recommendation.

## The four classes

| Tag | Definition | Acceptance |
|-----|------------|------------|
| **SAFE-NOW** | Correctness bug verifiable by a parity test, physics re-derivation, dead-code removal, type/doc fix, or an exactly behavior-preserving refactor. No benchmark needed. | A green targeted test / re-derivation / import-parity check. |
| **BENCHMARK-GATED** | Changes scoring, composition output, identifier decisions, or any accuracy-affecting physics in a **default-ON** path. Ships **default-OFF behind a flag**; must pass a held-out benchmark (composition F1 / Aitchison / RMSEP, or ID F1/recall) before the flag is promoted. Flag-promotion is a separate later work item. | Flag-off/flag-on scoreboard non-regression, cited gate + measured delta. |
| **DESIGN-DECISION** | Needs a measured A/B (benchmark or profiling) before one of several legitimate options is chosen; the choice itself is the deliverable. | Decision + measurement committed. |
| **AUDIT-FIRST** | No code change until a convention or invariant is confirmed by `rg` + Read. | Audit report; the fix is then re-classified. |

Mission priority governs ordering across all four: **ACCURACY > PRECISION > RELIABILITY > latency.**
Every performance-only item is deferred behind correctness in the same area.

## Worked examples

| Change | Class | Why |
|--------|-------|-----|
| De-duplicate the Olivero Voigt constants to one source | SAFE-NOW | Parity-verified; single-source-of-truth; no numeric change. |
| Fix the `None`-sentinel cache miss | SAFE-NOW | Removes silent DB re-query; regression test. |
| Move `CFLIBSResult` into `domain/` | SAFE-NOW | Behavior-preserving relocation + import-parity. |
| Add `exec`/`eval`/`compile` to the AST scanner | SAFE-NOW | Physics-only hardening; new tests, no benchmark. |
| Apply number→mass fraction conversion on the scoring path | BENCHMARK-GATED | Moves every composition number; top accuracy lever. |
| Promote resonance-ΔE McWhirter to default | BENCHMARK-GATED | Changes LTE gating; must show non-regression + fewer false rejections. |
| Poisson/Cash Bayesian likelihood default | BENCHMARK-GATED | Changes recovered values on shot-noise data. |
| Consolidate the 3 forward-physics impls vs parity-bind | DESIGN-DECISION | Three serve different API shapes; measure maintenance cost + parity. |
| Response-curve canonical path (`instrument/kernels.py`) | DESIGN-DECISION (DA-1) | Three impls with divergent edge handling; pick after parity test. |
| `two_region` 0.8/0.3–0.7 magic constants | DESIGN-DECISION | Ground in literature + benchmark, or keep opt-in with an "unvalidated" warning. |
| Air↔vacuum wavelength convention | AUDIT-FIRST | No verified finding confirmed consistency; audit before touching. |
| IPD/PF truncation consistency | AUDIT-FIRST → SAFE-NOW | Audit the two paths, then thread `compute_ipd` once. |

## The FALSE-exclusion list — do NOT "fix" these

> [!CAUTION] DO-NOT: The following were adversarially verified **FALSE**. Acting on them would
> introduce a bug or undo correct code. They are recorded here so a future reviewer does not
> re-surface them as "obvious fixes".

| "Finding" | Reality | Consequence of "fixing" |
|-----------|---------|-------------------------|
| inv-solve F1: "missing $\ln(U_{II}/U_I)$ in the ionic y-shift" | The term **cancels** in the common-slope formulation. | Adding it INTRODUCES a bug. |
| inv-identify F1/F6/F10: y-axis consistency / JAX parity / interference | Already correct. | Wasted churn; risks regressing a passing path. |
| inv-top F1: "full-spectrum mole→mass not done" | Already done in `solve_full_spectrum` (mass returned). | Double conversion. |
| io F5/F10: `intensity_W_m2_nm_sr` "missing" / round-trip broken | It IS in the alias list; round-trip works. | No-op or breakage. |
| validation F2: `1+avg_Z` denominator "wrong" | The denominator is correct. | Wrong composition. |
| validation F3-original: spin factor "missing" | The ×2 spin factor is already in the constant. | Double-counting. |
| inv-physics F2: pLTE $E_{\text{lower}}$ cut "wrong" | It is the correct Cristoforetti implementation [@cristoforetti2010]. | Removes a valid physics screen. |

Note the subtlety: validation **F3-original** is FALSE, but **F3-revised** (the noise-model
circularity) is a *different, real* HIGH finding — do not let the shared number make you exclude both.

## Contradicted claims never carried forward

Separate from the FALSE-exclusion list, these are *narrative* claims from older docs that were
overturned; the corrected position is stated wherever they would otherwise appear:

- **"manifold-as-foundation"** — the manifold is a fast seed, not the accuracy foundation
  ([data-flow](data-flow.md)).
- **"joint solver beats iterative"** — only *with* an adoption gate deciding when to trust a converged
  fit; without it, the claim is unsafe ([error-budget-and-falsification](../error-budget-and-falsification.md)).
- **"flat-path import shims work"** — false; the shims are gone
  ([inversion-subpackages](inversion-subpackages.md)).
- **"stage III empty / Fe I has 425 levels"** — pre-reset DB artifacts; the ASD59 rebuild fixed both.

## The standing rule: xfail → assert

A parity fixture may be `xfail`-marked while a divergence is known. When the fix lands, the same PR
**must** flip the `xfail` to a real assert — otherwise the fix passes silently as an `xfail` and parity
is never actually enforced. This is a mandatory acceptance line on every fix that resolves a parity
fixture.

## See also

- [Target architecture](target-architecture.md) — each target item carries one of these tags.
- [Two-backend pipeline](two-backend-pipeline.md) — why cross-backend changes are benchmark-gated.
- [ADR index](adr-index.md) — the DESIGN-DECISIONs (DA-1…DA-8) recorded as ADRs/architecture notes.
- [error-budget-and-falsification](../error-budget-and-falsification.md) — the falsified-claim ledger.

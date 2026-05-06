# Merge Plan: 13 open swarm-generated PRs

> Companion to `PR_REVIEW_FEEDBACK.md` and `PR_STACKING_STRATEGY.md`. The
> previous documents diagnosed test-debt and the conflict surface;
> this one is the actionable plan based on actual line-level conflict
> analysis (not file-overlap guess) and the remediation issues already
> filed.

## TL;DR

- The "everything conflicts in the same domain" framing of the prior strategy doc
  overstates it. Real line-level conflicts exist on **only 4 pairs**, not on every
  same-file pair.
- All swarm PRs except #57 (docs) and #64 (infra) are blocked until BOTH:
  - their **validation benchmark** (`nightly` tier on vasp-03 against real Aalto / ChemCam / SuperCam data) returns green, AND
  - their **remediation PR** (the test the architect failed to emit) lands first.
- Concretely: **0 PRs are merge-ready today**. Validation is in flight; remediation
  issues are filed; the dogfood loop is paused until both queues drain.

## Per-PR status

| PR | Issue | Title | Files | Validation | Remediation issue | Conflicts with |
|----|-------|-------|-------|------------|-------------------|----------------|
| 57 | — | Reorganize documentation | 68 (docs) | n/a | n/a | — |
| 58 | r45l | Student-t robust likelihood | `cflibs/inversion/solve/bayesian.py` | running (nightly) | `CF-LIBS-improved-gcda` | #66 |
| 62 | kthv | Iterative outlier rejection | `cflibs/plasma/boltzmann.py` | pending | `CF-LIBS-improved-suht` | — |
| 63 | hvrf | HVRF enhancement factor | `cflibs/plasma/hvrf.py` (new) | pending | `CF-LIBS-improved-ar8h` | — |
| 64 | 7024 | Orphan claim watchdog | infra | n/a | n/a | — |
| 65 | c0dt | STARK-B ingest + interpolator | `cflibs/atomic/database.py`, `cflibs/plasma/broadening.py` (new) | pending | `CF-LIBS-improved-rf8k` | #67, #71 |
| 66 | 4w8x | Hierarchical NumPyro priors | `cflibs/inversion/solve/bayesian.py` | pending | `CF-LIBS-improved-wc84` | #58 |
| 67 | jc9y | NIST Fe II 4f/5d levels | `cflibs/atomic/database.py`, `datagen_v2.py`, ... | pending | `CF-LIBS-improved-cyev` | #65, #71 |
| 68 | wzus | ALIAS Min-3-lines + R² gating | `cflibs/identification/alias.py` | pending | `CF-LIBS-improved-1uzq` | — |
| 69 | 548q | Multiplet-aware Boltzmann fit | `cflibs/plasma/boltzmann.py` (different sections from #62) | pending | `CF-LIBS-improved-v4oc` | — |
| 70 | aucw | PWLR composition closure | composition closure module | pending | `CF-LIBS-improved-2aug` | — |
| 71 | f25n | Kramida 2024 Aki uncertainty | `cflibs/atomic/database.py`, `datagen_v2.py`, ... | pending | `CF-LIBS-improved-4tbp` | #65, #67 |
| 72 | mgp5 | Hermann two-region forward model | `cflibs/inversion/solve/hermann.py` (new) | pending | `CF-LIBS-improved-4ep4` | — |

## Real conflict matrix (line-level, via `git merge-tree`)

| Pair | File(s) with conflicting hunks |
|------|--------------------------------|
| **58 ↔ 66** | `cflibs/inversion/solve/bayesian.py` |
| **65 ↔ 67** | `cflibs/atomic/database.py` |
| **65 ↔ 71** | `cflibs/atomic/database.py` |
| **67 ↔ 71** | `cflibs/atomic/database.py`, `datagen_v2.py` |

Everything else **same-file pairs** (e.g. #62 ↔ #69 on `boltzmann.py`) touch
**different sections** and merge cleanly without rebase.

The `.swarm/benchmark.toml` and `notebook_registry.toml` "conflicts" reported in
earlier analyses are false positives — those files were added to `dev` after the
swarm branches were forked, so every PR's `git diff dev..swarm/<id>` shows them
as deleted relative to dev. Standard merge resolves them by keeping `dev`'s copy.

## Proposed merge order

Bottom-up within each cluster; clusters parallel-able to each other.

### Independent (any order, after their gates)
- **#57** (docs only — needs human review only)
- **#64** (infra only — needs review only)
- **#62** (Boltzmann outlier rejection)
- **#63** (HVRF forward model)
- **#68** (ALIAS gating)
- **#69** (multiplet Boltzmann fit)
- **#70** (PWLR closure)
- **#72** (Hermann forward model)

### Bayesian stack (#58 → #66)
1. Merge **#58** first (smaller change, scoped to likelihood + McWhirter penalty).
2. Rebase **#66** onto post-merge `dev`. The hierarchical-priors change to
   `bayesian_model` is on different lines than #58's likelihood swap; the rebase
   should be mechanical.

### Atomic-data stack (#65 → #67 → #71)
1. Merge **#65** first (STARK-B is the foundational schema add — new tables for
   non-hydrogenic Stark widths). PR #71's test must reference STARK-B numbers
   anyway.
2. Rebase **#67** (Fe II refresh) onto post-#65 `dev`. Both touch `database.py`
   but #67's Fe II ingest is row-level; #65's STARK-B is schema-level.
3. Rebase **#71** (Kramida 2024 Aki) onto post-#67 `dev`. Same `database.py` +
   `datagen_v2.py` overlap; rebase resolves by sequencing.

## Gates each PR must pass before merge

1. **Validation green**: `scripts/post-merge-benchmark.sh run <wt> nightly <id>`
   (in flight as of 2026-05-06; results land as PR comments).
2. **Remediation merged**: the linked `CF-LIBS-improved-<id>` test/benchmark
   issue produces its own swarm PR that lands FIRST. Architect prompt is now
   test-mandatory (see beefcake-swarm PR #245), so each remediation PR will
   include the test that the original PR was missing.
3. **Stack rebase if applicable**: for the 4 conflict pairs, the upper PR is
   rebased onto post-merge `dev` after its lower base lands.

## What NOT to do

- **Do not merge any PR before its validation benchmark returns green**. Even
  the apparently-trivial ones (e.g., #62 / #68 / #69) need their physics
  validated against real data — that's the entire point of the gate.
- **Do not rebase ahead of validation results**. If a base PR fails validation,
  the rebase work on top of it is wasted.
- **Do not let the dogfood loop generate more PRs** until this stack is drained.
  The architect prompt is test-mandatory now, but adding more parallel work to a
  validation queue that's already 13-deep just delays the queue head.
- **Do not close any PR without remediation**. Each PR represents real
  architectural work; closing forfeits it. The remediation pattern (separate
  test PR that lands first) preserves the audit trail.

## Status of running validation pass

- vasp-03 GPU is dedicated to nightly-tier benchmarks (drained of llama.cpp inference).
- Each PR takes ~30 min wall-clock; sequential across all 11 swarm PRs ≈ 5.5 hr.
- Results post as PR comments tagged `Real-data benchmark validation (nightly tier)`.
- Wrapper script: `/tmp/validate-all-prs.sh` on ai-proxy. Live progress:
  `/tmp/pr-validation-results.log`.

## Tracking

This document, the 11 P1 remediation issues in beads, and the per-PR comments
together form the merge audit trail. When a remediation issue closes (via its own
swarm PR landing), update its row's "Remediation issue" column to indicate done;
when the validation benchmark for the parent PR returns green, update "Validation".
A PR is mergeable only when both columns say "done" and (if in a stack) it has been
rebased.

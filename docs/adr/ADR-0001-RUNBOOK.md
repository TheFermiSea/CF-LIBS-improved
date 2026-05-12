# ADR-0001 Implementation Runbook

> Operational protocol for landing Tier-1 candidates **T1-1 through T1-6** on integration branch `feat/adr-0001-pattern-survey-impl`.
> Authoritative spec: [`ADR-0001-radis-jaxrts-pattern-survey.md`](./ADR-0001-radis-jaxrts-pattern-survey.md) §8.1.
> Per-bead implementation details: [`specs/T1-1-host-kernel-split.md`](./specs/T1-1-host-kernel-split.md) … [`specs/T1-6-bayesian-decomposition.md`](./specs/T1-6-bayesian-decomposition.md). Each bead's `--design` field points at its spec file.

---

## 1. Branch topology

```
dev
 └── feat/adr-0001-pattern-survey-impl              (integration)
      ├── feat/adr-0001/T1-1-host-kernel-split
      ├── feat/adr-0001/T1-2-forward-model-unify
      ├── feat/adr-0001/T1-3-lax-while-iterative
      ├── feat/adr-0001/T1-4-ldm-broadening
      ├── feat/adr-0001/T1-5-chunked-scan-checkpoint
      └── feat/adr-0001/T1-6-retrieval-decomposition
```

- **Integration branch** `feat/adr-0001-pattern-survey-impl` is branched off `dev` once and lives until all six T1 beads merged. Already created on `origin` (commit `e97fd64`).
- **Sub-branches** `feat/adr-0001/<bead-id>-<short-name>` branched off the current tip of the integration branch when work begins (so wave-2 branches inherit T1-1 mechanically).
- **Worktrees** live at `.worktrees/<bead-id>/` (already gitignored).
- **Merge protocol:** sub-branch → integration uses `--no-ff` (preserves bead boundaries for `git log --first-parent`). Integration → `dev` is single `--no-ff` once all T1 gate checks (§6) pass.

## 2. Bead landing order

Dependency DAG (← means "depends on"):

```
T1-1 ──┬──► T1-2 ──► T1-6
       ├──► T1-4 ──► T1-5
       │
T1-3 ◄┘ (parallel-safe with T1-1; cflibs/inversion/solve/ is in T1-1 carve-out)
```

| Wave | Beads | Parallel? | Pre-condition |
|---|---|---|---|
| 1 | T1-1 (`5oar`) + T1-3 (`14p6`) | yes (disjoint files: T1-1 carves out `cflibs/inversion/solve/`) | branched off `dev` |
| 2 | T1-2 (`swgm`) + T1-4 (`e5o8`) | yes | T1-1 merged to integration; T1-3 may still be in flight |
| 3 | T1-5 (`ke4z`) + T1-6 (`0mor`) | yes | T1-2 (for T1-6) + T1-4 (for T1-5) merged |

**Tier-2 (T2-1..T2-8) deferred** to a follow-up integration cycle. **Tier-3** polish folds into routine cleanup beads. **Tier-4** (`lax.custom_root`) needs its own research bead with literature precursor.

**Key parallelism notes:**
- **T1-3 in wave 1:** Stream A's T1-1 spec §5 explicitly carves `cflibs/inversion/solve/` out of T1-1 scope, so T1-3 owns the host/kernel split of `iterative.py` itself. T1-3 has only a soft dep on T1-1 (for the shared `jit_if_available` decorator); it can copy the decorator inline if it lands before T1-1 merges. Recommend starting T1-3 ~1 day after T1-1 so the decorator is available.
- **T1-2 and T1-4 are file-disjoint** in wave 2: T1-2 owns `spectrum_model.py` (kernel-side), `batch_forward.py`, `bayesian.py` (forward layer only). T1-4 owns `profiles.py` (LDM kernel), `manifold/generator.py` broadening assembly. Both touch `spectrum_model.py::compute_spectrum` dispatch (T1-2 unifies, T1-4 adds enum branch) — coordinate via a single merge order (T1-4 first, then T1-2 rebases).
- **T1-5 strictly depends on T1-4** because the chunked scan wraps the LDM kernel.
- **T1-6 strictly depends on T1-2** because `bayesian/forward.py` imports the unified `forward_model`.

## 3. Subagent prompt template

Spawn the implementing subagent (Opus 4.7, 1M context) with this template. Assumes the bead's `--design` and `--notes` are populated.

```text
You are implementing CF-LIBS-improved bead {BEAD_ID} in worktree
  /home/brian/code/CF-LIBS-improved/.worktrees/{BEAD_ID}/
on branch
  feat/adr-0001/{BEAD_ID}-{SHORT_NAME}

Authoritative specs:
  1. ADR-0001 § 8.1 row {ROW}                docs/adr/ADR-0001-radis-jaxrts-pattern-survey.md
  2. Implementation spec                     docs/adr/specs/{T1-N}-{name}.md
  3. ADR-0001 implementation runbook         docs/adr/ADR-0001-RUNBOOK.md
  4. Bead state                              bd show {BEAD_ID}    (fields: --description, --notes, --design)

Project hard constraints (NON-NEGOTIABLE):
  - Physics-only ban (CLAUDE.md): no sklearn / torch / tensorflow / keras / flax / equinox /
    transformers / jax.nn / jax.experimental.stax in cflibs/ (allowed only under cflibs/evolution/).
  - JAX is optional; every JAX-importing file must degrade gracefully when JAX is missing.
  - JAX-metal has no fp64; new kernels must accept fp32 and document precision behaviour.

Workflow:
  1. `bd update {BEAD_ID} --status in_progress`.
  2. Read the spec file (docs/adr/specs/{T1-N}-*.md) end-to-end before any code.
     Stop and ask if anything is unclear — never expand scope (see runbook § 7).
  3. Use Serena MCP for symbol-level operations (find_symbol, replace_symbol_body,
     find_referencing_symbols, rename_symbol). Default `relative_path="cflibs/"`.
     Do NOT use Read+Edit for whole-symbol rewrites.
  4. Add or update tests per the spec's "Test plan" section.
  5. Run local gate checks (runbook § 5) until green.
  6. Push the sub-branch (`git push -u origin feat/adr-0001/{BEAD_ID}-{SHORT_NAME}`).
  7. `bd comment {BEAD_ID} "Pushed sub-branch; ready for merge to integration"` and
     `bd update {BEAD_ID} --status inreview`.
  8. STOP. Do NOT merge to integration. The orchestrator handles the merge after
     reviewing the diff against the spec.

Conflict-avoidance rules: § 7 of the runbook is binding. If you discover a file
outside the spec's "Files touched" list needs modification, file a new bead and
stop — do not expand scope.

Quality bar: zero new ruff/black/mypy warnings, zero test regressions, new tests
for new behaviour. Behavioural parity with the NumPy reference must hold to
rtol=1e-5 unless the spec explicitly relaxes it (T1-4 LDM at 1e-4 is the only
documented relaxation).
```

## 4. Worktree commands (copy-pasteable)

Per-bead lifecycle (substitute bead ID and short name):

```bash
# 4.1 Integration branch already exists. Orchestrator (NOT subagents) verifies:
git fetch origin
git checkout feat/adr-0001-pattern-survey-impl
git pull --ff-only

# 4.2 Spawn a wave-N bead in a worktree (orchestrator):
git worktree add .worktrees/T1-1 -b feat/adr-0001/T1-1-host-kernel-split \
                                     origin/feat/adr-0001-pattern-survey-impl
cd .worktrees/T1-1
bd update CF-LIBS-improved-5oar --status in_progress

# 4.3 (Subagent works here.)
#     Edits, tests, commits, push:
git push -u origin feat/adr-0001/T1-1-host-kernel-split
bd update CF-LIBS-improved-5oar --status inreview

# 4.4 Merge sub-branch to integration (orchestrator, after gate checks § 5 pass):
git -C /home/brian/code/CF-LIBS-improved checkout feat/adr-0001-pattern-survey-impl
git -C /home/brian/code/CF-LIBS-improved pull --ff-only
git -C /home/brian/code/CF-LIBS-improved merge --no-ff \
    feat/adr-0001/T1-1-host-kernel-split \
    -m "Merge T1-1 host/kernel split into ADR-0001 integration"
git -C /home/brian/code/CF-LIBS-improved push

# 4.5 Clean up worktree:
git -C /home/brian/code/CF-LIBS-improved worktree remove .worktrees/T1-1
bd close CF-LIBS-improved-5oar
```

## 5. Gate checks before merging a sub-branch to integration

All must pass on the sub-branch tip:

- [ ] `ruff check cflibs/ tests/` clean
- [ ] `black --check cflibs/` clean
- [ ] `mypy cflibs/` no new errors vs `dev` baseline
- [ ] `JAX_PLATFORMS=cpu pytest tests/ -x -q -m "not slow and not requires_db"` green
- [ ] New tests present matching the spec's "Test plan"
- [ ] `bd show <id>` reflects `inreview` status with comment summarizing the change
- [ ] `git push` of the sub-branch succeeded
- [ ] Self-review diff: `git diff feat/adr-0001-pattern-survey-impl...HEAD --stat` matches spec's "Files touched" — no scope creep
- [ ] Behavioural parity check: NumPy→JAX kernel ports include rtol=1e-5 parity test (rtol=1e-4 for T1-4 LDM)

## 6. Gate checks before merging integration to dev

Run from integration branch tip; all must pass:

- [ ] All six Tier-1 beads closed
- [ ] `pytest tests/ -v` green (full suite)
- [ ] `pytest tests/ --cov=cflibs --cov-report=term-missing` coverage ≥ pre-ADR baseline (capture before T1-1 lands, stash in `.adr-0001-baseline.txt`)
- [ ] `python scripts/validate_real_data.py --datasets steel_245nm FeNi_380nm --no-plots` passes
- [ ] Manifold round-trip: `cflibs generate-manifold examples/manifold_config_example.yaml --progress` completes and result loads via `cflibs.manifold.loader`
- [ ] Benchmark parity: `python scripts/benchmark_synthetic_identifiers.py --dataset-path output/synthetic_corpus/ak3_1_3_corpus_v1/corpus.json --db-path ASD_da/libs_production.db --output-dir output/synthetic_benchmark/adr-0001-postmerge` within 5% of pre-ADR baseline on every metric (Aitchison distance, ILR error, top-K recall)
- [ ] Physics-only constraint: `python -m cflibs.evolution cflibs/` returns clean (AST blocklist smoke-test against shipped code)
- [ ] CLAUDE.md unchanged or updated only with new module-map entries

## 7. Conflict-avoidance rules

1. **One sub-branch per file at a time.** Track file ownership in a pinned comment on a tracking bead. Wave-2 branches `git rebase origin/feat/adr-0001-pattern-survey-impl` after T1-1 merges.
2. **No scope expansion.** If a wave-N agent finds it needs to touch a file outside its spec's "Files touched" list, it **stops**, files a follow-up bead (`bd create --type bug-or-chore ...`), records the discovery as a comment on its own bead, and resumes only after orchestrator confirms the scope decision.
3. **Document scope changes.** `bd update <id> --notes "Conflicts with <other-bead>: <file>; deferred to <new-bead>"`.
4. **Wave gate.** A wave-N bead may not start until every wave-(N-1) bead has merged to integration and integration is green per §5. **Exception: T1-3 in wave 1** runs concurrently with T1-1 because `cflibs/inversion/solve/` is in T1-1's carve-out (`specs/T1-1-host-kernel-split.md` §5).
5. **No force-pushes** to integration or `dev`. Sub-branches may be force-pushed up until merge to integration.
6. **Rebase forward, don't merge back.** Stale sub-branches rebase forward with `git rebase origin/feat/adr-0001-pattern-survey-impl`, never merge back from integration.

## 8. Rollback

**Per-bead rollback** (sub-branch broke at gate check):
```bash
bd update CF-LIBS-improved-<id> --status open --notes "Rolled back: <reason>"
git branch -D feat/adr-0001/<bead-id>-<short-name>
git push origin --delete feat/adr-0001/<bead-id>-<short-name>   # only if pushed
git worktree remove .worktrees/<bead-id>
```

**Whole-integration rollback** (ADR direction proves wrong mid-cycle):
```bash
git checkout dev
git branch -D feat/adr-0001-pattern-survey-impl
git push origin --delete feat/adr-0001-pattern-survey-impl
# ADR-0001 itself stays committed — it is documentation, not code.
# All T1 beads reopened: bd reopen <each>
```

**Post-merge rollback** (integration landed on `dev` but a regression surfaces):
Prefer `git revert -m 1 <merge-commit>` over hard-reset. Open a follow-up bead documenting the regression and the revert reason. Do not force-push `dev`.

## 9. Done-ness for the whole ADR

- [ ] All six Tier-1 beads closed
- [ ] Integration branch merged to `dev` (single `--no-ff` merge commit)
- [ ] `dev` pushed to `origin/dev` (per CLAUDE.md "Landing the Plane")
- [ ] §6 gate checks all passed on post-merge `dev` tip
- [ ] Tier-2 beads (T2-1..T2-8) filed as separate beads, ready for next-cycle runbook
- [ ] Tier-3 polish routed into existing or new cleanup beads
- [ ] Tier-4 (T4-1 `lax.custom_root` implicit-diff) filed as research bead with literature-review precursor
- [ ] Session-end hand-off note posted: short summary + commit SHA + list of follow-up beads

## 10. Quick reference — spec files

| Bead ID | Short name | Spec file |
|---|---|---|
| `CF-LIBS-improved-5oar` | host-kernel-split | [`specs/T1-1-host-kernel-split.md`](./specs/T1-1-host-kernel-split.md) |
| `CF-LIBS-improved-swgm` | forward-model-unify | [`specs/T1-2-forward-model-unification.md`](./specs/T1-2-forward-model-unification.md) |
| `CF-LIBS-improved-14p6` | lax-while-iterative | [`specs/T1-3-lax-while-iterative.md`](./specs/T1-3-lax-while-iterative.md) |
| `CF-LIBS-improved-e5o8` | ldm-broadening | [`specs/T1-4-ldm-broadening.md`](./specs/T1-4-ldm-broadening.md) |
| `CF-LIBS-improved-ke4z` | chunked-scan-checkpoint | [`specs/T1-5-chunked-scan-checkpoint.md`](./specs/T1-5-chunked-scan-checkpoint.md) |
| `CF-LIBS-improved-0mor` | retrieval-decomposition | [`specs/T1-6-bayesian-decomposition.md`](./specs/T1-6-bayesian-decomposition.md) |

# J12 Implementation Spec — Full-Board Superiority Run + Promotion Decision → Milestone M3

**Bead:** J12 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §8.2 (M3), D6, §10 · **Track:** spine (decision point) · **Depends:** J9, J10 (J11 optional input) · **Estimated effort:** 3–5 pd + iteration buffer / ~1 wk

## 1. Goals

Run the full 10-dataset scoreboard with the integrated jit pipeline (J8 graph + J9 batching + J10 identifier), perform the gap analysis, and either write the promotion ADR or file gap beads and remain parallel. This bead **decides**; it does not get to redefine the board.

## 2. Protocol

- Baseline = `docs/benchmarks/SCOREBOARD-2026-06-10-baseline.md` protocol at the pinned reference SHA (geological preset; candidate policy truth ∪ {Ag,Sn,W,Bi,Th}; presence rule ≥0.5 wt %; seed 20260610) — **full board, all spectra, not `--max-spectra 12`**.
- Evaluated on the **holdout tier** of the dataset split defined in `docs/audit/2026-06-10-goalfirst/optimization-program-design.md` §2.1; vault datasets opened only at this gate per that policy.
- Precondition: **≥1 release of shadow-mode operation** (J8's Tier-B harness: both pipelines on every scoreboard invocation, deltas logged) with the divergence ledger current and every Tier-D divergence adjudicated.
- Promotion is computed over **scoring datasets only** (nist_srm_612 / nist_steel yield no spectra — measured); the failure-count criterion is tracked separately (risk R12). silva2022's 12/12 baseline failures are a reference bug surface the jit failure-policy parity must mirror, not hide.

## 3. Acceptance criteria — **Milestone M3 gate** (all five)

1. **Identification:** aggregate micro-F1 ≥ reference AND no scoring dataset regresses F1 by > 0.02 AND per-dataset F1 ≥ reference − 0.01 on ≥ 7/10 datasets (protects the zero-FP precision asset named by the campaign doc).
2. **Composition:** median RMSE wt % ≤ reference on every dataset with quantitative truth (one dataset may regress ≤ 5 % relative if the aggregate improves).
3. **Failures:** hard-failure count ≤ reference (the 12 silva2022 failures are the floor to beat, `SCOREBOARD…md:171-188`); jit failures score all-FN identically to the reference policy (`scoreboard.py:159-166`).
4. **Runtime:** batched ≥ 10× median spectra/s vs the 0.4–5.1 s/spectrum baseline medians; CPU single-spectrum ≤ 2× reference wall time. (M2's ≥50 spectra/s is the engineering target; this is the promotion floor — ADR-0004 §3 C5.)
5. **Parity suite green** at the §5.4 tier contracts on the promotion SHA.

## 4. Outcomes

- **Pass →** write `ADR-0005-jitpipe-promotion.md`: default flips to jit (Stage B of D6 — reference available as `--pipeline=reference`, parity oracle, bug-fixes-only with paired parity-test updates for ≥2 releases; Stage-C demotion criteria restated). Board JSON, gap analysis, and the reproduce line committed alongside.
- **Fail →** gap analysis identifying the offending stage(s) via the Tier-D ledger + stage-bisect; gap beads filed; the pipeline **stays parallel** — J9's throughput and J10's campaign-evaluator value stand on their own (ADR-0004 §10: the failure mode is bounded; the program never holds the scoreboard hostage). Re-attempt requires a new J12-class bead.

## 5. Test plan / artifacts

Committed artifacts: full-board JSON for both pipelines, per-dataset delta table, divergence-ledger snapshot, runtime table (per-bucket B, compile counts, amortized spectra/s), promotion-or-gap memo. Runs launched as background sbatch jobs from the parent session (watchdog rules); deterministic-ops flag set.

## 6. Risks

Holdout/vault contamination (enforce the split mechanically — adapters refuse vault datasets outside this bead); adjudication backlog at the gate (require the ledger be current as a precondition, not a deliverable); temptation to tune on the holdout (one full-board evaluation per candidate SHA, recorded).

## 7. Dependencies / files

Depends J9, J10 (J11 optional). Files: `docs/benchmarks/SCOREBOARD-<date>-jit-vs-reference.{md,json}`, `docs/adr/ADR-0005-jitpipe-promotion.md` (on pass), gap beads (on fail). No code beyond harness glue.

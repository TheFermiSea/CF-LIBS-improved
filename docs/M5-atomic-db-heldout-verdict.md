# Atomic-DB accuracy on the HELD-OUT tier (cluster benchmark) — definitive verdict

Tested all downloaded atomic databases through the production reference pipeline on BOTH the
optimization (dev) and HELD-OUT (test) tiers — the comparison the earlier optimization-tier-only
M5 work never did. Run distributed across vasp-01/02/03 (16 SLURM array tasks; per-worker node-local
DB copies to dodge NFS SQLite locking; portable NFS venv). Driver: scripts/benchmarks/db_bench.sbatch
+ scripts/arbor/eval_one.py; aggregate: output/atomic_db_benchmark.json.

| database | dev RMSE (wt%) | HELD-OUT RMSE (wt%) |
|---|---|---|
| **NIST (28.7k, graded)** | 2.475 | **2.486** (best) |
| R4 NIST+VALD backfill (935k) | 2.906 | 3.304 (worse) |
| VALD-complete (1.09M) | 5.099 | 5.174 (worse) |
| VALD-grade-B (118k) | 18.139 | 20.142 (garbage; only 15 species) |

## Verdict: NIST wins held-out; NO alternative beats it. Ranking is consistent dev<->held-out
(no overfit sign-flip, unlike the self-absorption lever). The conclusion is robust.

## Mechanism (evidence, not assertion)
- R4 (NIST graded lines + VALD backfill) is WORSE than NIST alone on both splits -> adding VALD
  lines HURTS even while keeping NIST's graded lines.
- VALD-complete is ~2x worse: the bulk is Kurucz-THEORETICAL D-grade gf -> more lines = more
  pollution of the Boltzmann/Saha fit.
- The bottleneck is gf-VALUE ACCURACY, not completeness (consistent with composition_error_bound:
  per-line density error dominates on high-SNR data). The downloaded alternatives are bigger but
  LESS accurate, so they regress.

## Implication
The accuracy lever is NOT "more lines" (completeness) — it is "more ACCURATE gf for the analytical
lines": experimental-grade transition probabilities (STARK-B / experimental-grade acquisition, the
#1 data ask), not bulk theoretical line lists. Raw-completeness substitution is a dead end.

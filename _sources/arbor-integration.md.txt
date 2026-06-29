# Arbor integration (autonomous accuracy optimization)

[Arbor](https://github.com/RUC-NLPIR/Arbor) in **keyless mode**: the host Claude Code agent's model
drives the research loop; Arbor contributes the durable Idea Tree, git-worktree isolation, guarded
merges, and reports as deterministic tools. No separate API keys. This wires it to CF-LIBS's
existing harness so the autonomous search is *gated and anti-overfit by construction*.

## The adapter (what I built — "the rest")
| Piece | File | Role |
|---|---|---|
| Eval + correctness gate | `scripts/arbor/run_eval.py` | Arbor runs this per candidate in a worktree |
| Research config | `scripts/arbor/research_config.yaml` | objective + coordinator settings |

**`run_eval.py` does two things, in order:**
1. **CORRECTNESS GATE (the keystone).** Arbor's built-in merge gate is *accuracy-only*, so
   correctness is enforced here: run the physics-only import blocklist (`ruff --select TID251
   cflibs/`) + the cflibs-formal oracle conformance (`tests/oracle/` + the csigma/reliability/
   identifiability tests). On any failure → `score = -1e9`, `valid=false` → Arbor never merges it.
   This is the mechanism that **auto-catches Cσ-class silent errors** (a candidate that scores well
   but is physically wrong) — the same `fitness = -inf on violation` idea as
   `cflibs/evolution/evaluator.py`.
2. **SCORE.** Median composition RMSE (wt%) from the production `run_scoreboard`. Emits
   `score = 100/RMSE` (maximize) so Arbor's `merge_threshold` (% gain) maps ~1:1 to relative RMSE
   reduction. Also emits the raw `rmse_wt` and per-dataset breakdown.

**Anti-overfit splits use the project's own tiers (no leakage):**
- `--split dev` → **optimization** tier (`supercam_labcal` + `aalto`) — iterate here.
- `--split test` → **holdout** tier (`supercam_scct` n=547 + `bhvo2_chemcam` n=4) — the merge gate.

## Scope (where accuracy can actually move)
The config points Arbor at the two reachable levers and *forbids* the inert ones (this session's
findings): **self-absorption correction** (the measured −0.21 wt% win) and **atomic-data/grade
weighting** (`composition_error_bound` says per-line density error dominates on high-SNR data). It
forbids re-exploring the proven-inert line-selection gates, and forbids touching the harness, data,
tests, the oracle, or adding ML imports.

## Honest bound
Arbor makes the *search* structured, parallel, gated, and resumable — it does **not** change the
finding that the deeper bottleneck is atomic-data *quality*, which code-optimization can't reach.
Its value is exhaustively + safely exploring the two levers above, with the oracle as the gate.

## Run it
See `scripts/arbor/research_config.yaml` header, or the command block handed off with this commit.
Verify the eval first: `python scripts/arbor/run_eval.py --split dev --max-spectra 20` should print
`"valid": true` and a real `score`.

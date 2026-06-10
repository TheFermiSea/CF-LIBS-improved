# Campaign 1 — Optuna TPE over the inversion knob space

The first (cheapest) campaign of the CF-LIBS optimization program
([design doc](../../docs/audit/2026-06-10-goalfirst/optimization-program-design.md),
Section 3): a seeded TPE search over **43 pipeline + detection knobs**, fitness =
the goal-metric scoreboard on the **optimization split**, with BHVO-2 /
EMSLIBS-2019 / the 40% target holdouts **never** entering the loop and
gibbons2024 in the vault (never evaluated by this tooling at all).

Everything here is optimization-layer tooling: **nothing under `cflibs/` imports
optuna** (ruff TID251 physics-only ban untouched). The winning configuration
ships as plain preset numbers in `cflibs/inversion/pipeline.py` after the
adoption gate.

## Files

| File | Role |
|---|---|
| `knob_space.py` | Search-space definition (design 3.1 A+B), params ↔ `config_overrides` mapping, `FROZEN_MANIFEST` writer |
| `splits.py` | Optimization / holdout / vault split builder + structural holdout refusal (`HoldoutViolation`) |
| `objective.py` | Candidate evaluation (scoreboard internals) + composite fitness (design 2.2) + paired bootstrap |
| `driver.py` | Optuna study: `init` / `worker` / `status` / `stop`; journal storage, STOP file, core-hour ledger |
| `holdout_eval.py` | Top-K re-evaluation on full optimization + holdout splits; adoption-gate verdict report |
| `slurm/` | `worker_array.sbatch`, `holdout_eval.sbatch`, `stage.sh` |

Frozen split manifest (committed): `docs/benchmarks/manifests/campaign1-splits-v1.json`
(seed 20260610; by target identity, never by spectrum). Every study writes a
`frozen_manifest.json` embedding the split id lists, the atomic-DB sha256, the
seed, the git SHA and the knob-space definition.

All commands below run **from the repo root**. `PYTHONPATH=$PWD` is mandatory
(worktree trap: `python scripts/x.py` otherwise imports `cflibs` from whatever
checkout the venv installed). Check the printed `cflibs=` provenance line.

## 0. Regenerate the split manifest (only if datasets changed)

```bash
JAX_PLATFORMS=cpu PYTHONPATH=$PWD .venv/bin/python scripts/campaign1/splits.py build
```

## 1. Stage to the cluster (maintainer)

```bash
bash scripts/campaign1/slurm/stage.sh /cluster/shared/cf-libs-bench/campaign1
```

Assumptions: `uv` on PATH (bootstraps a Python 3.12 venv at
`<dest>/repo/.venv` with `.[dev]` + optuna); the share already holds prior
`data/` and `results/` dirs; rsync is incremental. `data/` symlinks are
resolved (`rsync -aL`) into `<dest>/data` and the staged repo's `data/` is
re-pointed there; the synthetic corpus is staged to
`<dest>/data/synthetic_corpus/corpus.json` (the sbatch templates export
`CFLIBS_SCOREBOARD_SYNTH_CORPUS` to it).

## 2. Smoke test (local or on the staged repo)

```bash
JAX_PLATFORMS=cpu PYTHONPATH=$PWD .venv/bin/python scripts/campaign1/driver.py init \
    --study-dir output/campaign1/smoke \
    --datasets aalto,synthetic_fixedforward \
    --spectra-per-dataset 3 --target-trials 4 --budget-core-hours 1
JAX_PLATFORMS=cpu PYTHONPATH=$PWD .venv/bin/python scripts/campaign1/driver.py worker \
    --study-dir output/campaign1/smoke --max-trials 4
JAX_PLATFORMS=cpu PYTHONPATH=$PWD .venv/bin/python scripts/campaign1/driver.py status \
    --study-dir output/campaign1/smoke
```

## 3. Initialize the production study (maintainer, on the staged repo)

```bash
cd /cluster/shared/cf-libs-bench/campaign1/repo
JAX_PLATFORMS=cpu PYTHONPATH=$PWD .venv/bin/python scripts/campaign1/driver.py init \
    --study-dir /cluster/shared/cf-libs-bench/campaign1/results/run1 \
    --db ASD_da/libs_production.db \
    --spectra-per-dataset 74 --target-trials 800 --budget-core-hours 600
```

`init` evaluates the production baseline once on the exact per-trial sample
(the FP/failure **death-penalty reference**, `baseline.json`) and enqueues two
seed trials: the baseline params and the "looser gates" failure-hypothesis
candidate (design 3.1).

## 4. Submit workers (maintainer)

```bash
sbatch --export=ALL,REPO_DIR=/cluster/shared/cf-libs-bench/campaign1/repo,STUDY_DIR=/cluster/shared/cf-libs-bench/campaign1/results/run1,TRIALS_PER_TASK=5 \
    scripts/campaign1/slurm/worker_array.sbatch
```

Topology per design 3.3: `--array=0-15%16 --cpus-per-task=8 --mem=16G`,
CPU-only, **no `--nodelist`**, private JAX compilation cache. One wave = up to
16×`TRIALS_PER_TASK` trials; resubmit waves until `status` reports the target.
Workers self-limit: they exit on the STOP file, on the **core-hour ledger cap**
(`wall_s × cpus` summed in `ledger.jsonl`; refuses new trials past
`--budget-core-hours`), on the study trial target, or on their own quota.
The journal storage (`JournalFileOpenLock`) is NFS-safe; killing/preempting
workers loses at most the in-flight trial.

## 5. Monitor

```bash
JAX_PLATFORMS=cpu PYTHONPATH=$PWD .venv/bin/python scripts/campaign1/driver.py status \
    --study-dir /cluster/shared/cf-libs-bench/campaign1/results/run1
tail -f cflibs-c1-worker-*.out
```

## 6. Kill switch

```bash
touch /cluster/shared/cf-libs-bench/campaign1/results/run1/STOP
# or: .venv/bin/python scripts/campaign1/driver.py stop --study-dir <study>
scancel --name=cflibs-c1-worker   # optional: also cancel queued tasks
```

Workers halt before their next trial; remove the file to resume.

## 7. Holdout evaluation + adoption verdict (quota: 1 per week)

```bash
sbatch --export=ALL,REPO_DIR=/cluster/shared/cf-libs-bench/campaign1/repo,STUDY_DIR=/cluster/shared/cf-libs-bench/campaign1/results/run1,TOP_K=5 \
    scripts/campaign1/slurm/holdout_eval.sbatch
```

Re-evaluates the top-5 trials + baseline on the **full** optimization split and
the holdout set, then applies the adoption gate: **G1** optimization pooled
ΔF1 ≥ +0.02 with paired-bootstrap 95% CI excluding 0; **G2** holdout pooled
ΔF1 ≥ +0.02 with CI excluding 0 and no per-dataset regression beyond bootstrap
noise (BHVO-2, n=4, gated point-wise); **G3** zero new FP and zero new failures
on real holdout datasets; **G4** holdout runtime ≤ 1.5× baseline. Verdict:
`<study>/holdout_verdict.md` (+ `.json`); the recommendation among ADOPT
candidates is the best worst-dataset score (design 3.4). Every holdout query is
ledger-logged; a second query within 7 days requires a human `--force`.

## 8. Adopt

Ship the winning `config_overrides` as updated `ANALYSIS_PRESETS` values /
dataclass defaults in `cflibs/inversion/pipeline.py` (plain numbers; the
`detection_overrides` knobs get promoted to explicit fields at adoption time),
with a provenance comment citing the study dir, trial number and the design
doc. PR against `dev` with the optimization + holdout tables, bootstrap CIs and
the `frozen_manifest.json` hashes in the body. CI runs the full suite.

## Design deviations (justified)

- **No central ask-tell driver job.** The design sketches a long-running
  driver submitting eval arrays per generation; with Optuna journal storage
  the workers self-serve trials and the journal is the coordination point —
  strictly more preemption-tolerant, same controls (STOP file, ledger cap).
- **Death penalties return -1e9, not -inf** (recorded with reasons in
  `user_attrs.fitness_report`): TPE handles finite values better and the
  ordering is unchanged.
- **Per-generation synthetic-subsample seed rotation** (design 2.4 option) is
  not implemented: TPE has no generations; the per-trial sample is frozen at
  the study seed. Revisit for the phase-B CMA-ES run.
- **Split manifests carry spectrum ids, not per-file sha256** — several
  datasets aggregate many spectra per file (h5/xlsx), so file hashes don't map
   1:1; the atomic-DB sha256 + splits-manifest sha256 + git SHA in
  `frozen_manifest.json` pin the inputs instead.
- **G2 requires holdout ΔF1 ≥ +0.02 significant** (task spec), which is
  stricter than design 2.5's "no holdout regression"; both are reported in the
  verdict table.

## Smoke-run findings (2026-06-10)

- **Per-spectrum timeouts are two-layered.** The in-child SIGALRM cannot
  interrupt GIL-released C/XLA calls — a smoke trial with
  `use_deconvolution=True` wedged inside `jax backend_compile_and_load` for
  13+ minutes. Spectra therefore always run in a spawn pool and the parent
  hard-kills the pool on wall-deadline overrun, records the timeout as a
  failure, and resubmits the untouched remainder.
- **`closure_mode="matrix"` draws die by design**: the scoreboard's
  per-spectrum candidate sets have no globally valid `matrix_element`, so the
  solver raises on every spectrum and the failure death penalty fires
  (fitness −1e9). TPE learns to avoid the branch (~1/6 of random startup
  draws wasted); the choice stays in the space because the design doc lists
  it.
- Death penalties were observed live: two of four smoke trials drew configs
  whose failures exceeded `1.25 × baseline` and scored −1e9.

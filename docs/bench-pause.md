# Pausing the post-merge benchmark gate

**Status:** active since T1.4 (CF-LIBS-improved-5t6n), 2026-05-12.

## Background

Every architect-resolved PR to `dev` triggers a benchmark dispatch on
vasp-03:

1. The orchestrator (`beefcake-swarm/python/run.py`) finishes the
   verifier and reads `<worktree>/.swarm/benchmark.toml`.
2. `python/benchmark_gate.py::run_gate()` invokes
   `beefcake-swarm/scripts/post-merge-benchmark.sh classify` to decide
   whether the change is `skip` / `light` / `heavy`.
3. For `light` / `heavy`, the gate calls `post-merge-benchmark.sh run`,
   which **drains llama.cpp inference on vasp-03**, rsyncs the worktree
   to NFS, and runs a 30 - 90 minute GPU benchmark.

During multi-day parameter sweeps and 24-hour analysis-pipeline loops
(`scripts/loop_24h_driver.sh`), this auto-fired benchmark contends with
the sweep for the same V100 and ruins the wall-time budget. We need a
way to suspend the gate for the duration of the experiment without
modifying `benchmark.toml` (which would create per-PR diffs every time
we pause/resume).

## Mechanism

`post-merge-benchmark.sh` checks two signals at the top of `do_classify`
and `do_run`:

- File flag: `/tmp/cf-libs-bench-paused` exists.
- Env var: `BENCH_PAUSED=1`.

If either is present, the script:

- `classify` → echoes `"skip"` (the orchestrator gate treats this as
  "no benchmark needed", same as docs-only PRs).
- `run` → logs `"paused, skipping (flag=$FLAG, BENCH_PAUSED=$BENCH_PAUSED)"`
  and returns `0` (the gate treats this as a pass, not a regression —
  PRs are not blocked).

The flag must exist **on the orchestrator host** (where `run.py` lives
— typically ai-proxy / `100.105.113.58`), not on vasp-03 or wherever
the benchmark would have run, because the gate's `subprocess.run` call
happens in the orchestrator process and is what's intercepted.

## Helpers

```bash
# Pause the gate on the local host:
./scripts/bench-pause.sh

# Pause on the orchestrator host (ai-proxy) over ssh:
./scripts/bench-pause.sh --remote                    # uses default brian@100.105.113.58
./scripts/bench-pause.sh --remote brian@ai-proxy     # explicit host
./scripts/bench-pause.sh --remote --reason "sweep T2.1 in progress"

# Resume (symmetric):
./scripts/bench-resume.sh
./scripts/bench-resume.sh --remote
./scripts/bench-resume.sh --remote brian@ai-proxy
```

Pause is idempotent (no-op if already paused, same flag overwritten
with fresh timestamp+reason). Resume is also idempotent (logs "already
resumed" if no flag exists).

The flag file contains a few annotation fields for postmortems:

```
paused_at=2026-05-12T14:23:00-04:00
paused_by=brian@ai-proxy
reason=sweep T2.1 in progress
```

## Acceptance verification

```bash
# 1. Pause and confirm no benchmark fires on the next PR-push to dev.
./scripts/bench-pause.sh --remote --reason "acceptance test"
git push origin dev:dev   # or whatever push triggers a PR via the orchestrator
# In the orchestrator log: gate emits "paused, skipping (flag=...)"; PR is
# created with the benchmark section reporting "skipped (paused)".

# 2. Resume and confirm normal behaviour returns.
./scripts/bench-resume.sh --remote
git push origin dev:dev
# Orchestrator log shows classify → light/heavy → run on vasp-03 as before.
```

## Why a flag file (and not just an env var)

`BENCH_PAUSED=1` works for one-shot invocations, but the orchestrator
process is long-running (`dogfood.py` loops for days). To pause an
already-running orchestrator without restarting it, we need a signal
that the *child subprocess* picks up at invocation time. A filesystem
flag is the simplest signal that survives across subprocess boundaries
without restarting the parent.

## Interaction with the parameter-sweep loop

The 24-hour loop driver (`scripts/loop_24h_driver.sh`) doesn't itself
trigger the post-merge benchmark — it just runs `run_unified_benchmark.py`
directly. The auto-fired benchmark only fires when a PR is being created
by the swarm orchestrator. So in practice, the typical sequence is:

1. Pause: `./scripts/bench-pause.sh --remote`
2. Start sweep: `./scripts/loop_24h_driver.sh ...`
3. Sweep runs uninterrupted; orchestrator can still resolve PRs but
   does not steal GPU on vasp-03.
4. Sweep finishes; resume: `./scripts/bench-resume.sh --remote`

If you forget to resume, the gate stays paused until somebody notices
the PRs are landing without the usual benchmark validation. Recommend
setting a calendar reminder or wrapping the sweep driver in a `trap`:

```bash
./scripts/bench-pause.sh --remote --reason "24h sweep"
trap '"$PWD/scripts/bench-resume.sh" --remote' EXIT INT TERM
./scripts/loop_24h_driver.sh 2026-05-12 24 0 cuda
```

## References

- `scripts/bench-pause.sh`, `scripts/bench-resume.sh` — helpers.
- `beefcake-swarm/scripts/post-merge-benchmark.sh` — the gate that
  honors the flag (`do_classify` / `do_run`).
- `tests/scripts/test_bench_pause.py` — exercises the short-circuit.
- T1.4 issue: CF-LIBS-improved-5t6n.

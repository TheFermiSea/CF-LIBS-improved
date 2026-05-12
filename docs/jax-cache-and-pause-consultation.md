# Consultation: JAX compile cache + benchmark pause mechanism

Synthesized from GPT-5.4 (Codex) and Gemini 3 Flash Preview via CLIAPIProxy
(`localhost:8317/v1`) on 2026-05-12 while implementing
`feat/jax-cache-and-bench-pause` (CF-LIBS-improved-b5xw / -5t6n).

## Task A — JAX persistent compile cache on NFS

**Codex (gpt-5.4):**

1. Prefer `JAX_COMPILATION_CACHE_DIR=/cluster/shared/jax-cache` (portable
   across scripts/jobs/hosts). `jax.config.update("jax_compilation_cache_dir", ...)`
   is equivalent but must run **before any `jit`/compile** call.
2. No other env var is strictly required for persistence once dir is set.
3. Optional tuning:
   - `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` — lower to cache faster
     compiles (default 1.0s; lowering to e.g. 0.5 captures more entries).
   - `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES` — set to 0 to cache small
     entries (default 0 in recent JAX).
4. For extra hit rate, `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all` enables
   XLA-side caches (HLO + autotune) in addition to the JAX-level cache.
5. NFS: concurrent writers are usually safe — duplicate writes/compiles can
   happen but correctness is preserved. Main risks are metadata latency
   and lock semantics, not corruption.
6. Recommend a stable, writable shared path partitioned by JAX/jaxlib/CUDA
   stack version (JAX does this automatically via cache-key hashing — the
   directory will accumulate subdirs per (jax_version, jaxlib_version,
   backend) tuple).

**Verdict:** use the env var, set `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5`
to widen coverage for our short-iteration JAX kernels, and don't bother with
`jax.config.update` calls in Python — the env var path is strictly more
portable and avoids ordering bugs ("must be set before first compile").

## Task B — Pause mechanism for `post-merge-benchmark.sh`

**Gemini 3 Flash Preview:**

1. **Exit code:** Return 0. Pausing is a skip, not a failure — don't block
   developer velocity.
2. **Scope:** Skip both `classify` and `run`. Classifying without dispatching
   produces stale signals.
3. **Guard form:**
   `[[ -f /tmp/cf-libs-bench-paused || "$BENCH_PAUSED" == "1" ]] && { log "paused, skipping"; return 0; }`
4. **Race conditions:** `stat` on a flag is atomic; global flag impacts all
   concurrent PR builds, which is the intent (we *want* all parallel sweeps
   to see the pause).
5. **Visibility:** Always log the pause event explicitly so missing benchmark
   results are obviously intentional.
6. **Storage:** `/tmp/` is volatile (cleared on reboot). For multi-day sweeps,
   could use `/var/tmp/` — but the orchestrator host (ai-proxy) hasn't
   rebooted in months, and a reboot during a sweep would interrupt the sweep
   anyway. `/tmp/` is fine.
7. **Helpers:** `touch` and `rm -f` are sufficient. Keep them idempotent.

**Verdict:** check the pause flag at the top of both `do_classify` and
`do_run`. For `classify`, echo `skip` (so the orchestrator's gate parser
treats it as a no-bench-needed change). For `run`, log and return 0.
Provide `scripts/bench-pause.sh` and `scripts/bench-resume.sh` as thin
wrappers (`touch /tmp/cf-libs-bench-paused` / `rm -f /tmp/cf-libs-bench-paused`)
with `--remote <host>` for affecting the orchestrator host from anywhere.

## Locating the auto-bench trigger

(Asked Gemini for "where might an auto-bench trigger live for PRs to dev",
but the answer was already known from grep — keeping it brief.)

The trigger is **not** a git hook, cron, or GitHub Action. It's the
beefcake-swarm orchestrator's `python/benchmark_gate.py::run_gate()`,
which the dogfood loop calls after the verifier passes but before
opening a PR. That gate reads `<worktree>/.swarm/benchmark.toml` —
present in CF-LIBS-improved — and invokes
`/home/brian/code/beefcake-swarm/scripts/post-merge-benchmark.sh`
with `classify` then `run <tier>`.

So the modification must land in **beefcake-swarm**, not CF-LIBS-improved.
The pause/resume *helpers* and the docs live in CF-LIBS-improved (close
to the consumers — the parameter-sweep work runs from this repo), but the
guard in `do_run`/`do_classify` is in beefcake-swarm. We do both in this
PR (CF-LIBS-improved) and a sibling PR (beefcake-swarm).

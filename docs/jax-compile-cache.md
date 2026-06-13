# JAX persistent compile cache (per-user)

**Status:** active since T1.2 (CF-LIBS-improved-b5xw), 2026-05-12.
**Default flipped to per-user** in J0 (ADR-0004 §5.5), 2026-06-13 —
see [Default location and the shared-path pathology](#default-location-and-the-shared-path-pathology).

> **TL;DR:** the code default for `JAX_COMPILATION_CACHE_DIR` on Linux is
> now the user-private `~/.cache/cflibs/jax`, **not** the NFS-shared
> `/cluster/shared/jax-cache`. The shared path developed uid skew across
> compute nodes and hung jobs 1909/1914/1915. If you want cross-host
> sharing, opt in explicitly — see below.

## What it does

JAX's tracing+lowering pipeline turns Python+JAX functions into XLA HLO,
then into compiled binaries for the target backend (CPU / CUDA / TPU).
That compilation is *slow* — typical CF-LIBS kernels take 15-45 s on a
cold JIT cache, and warm-up dominates short-iteration sweep runs.

The persistent compile cache writes the compiled binaries to disk keyed
by a hash of `(JAX version, jaxlib version, backend, HLO bytes)`. The
next time the same kernel is compiled — *on this process, or any other
process, or any other host that can read the cache* — JAX skips the
expensive XLA pass and loads the binary directly.

A persistent on-disk cache lets a process — and successive jobs on the
same node — skip the cold XLA pass for kernels they have compiled before.

## Default location and the shared-path pathology

**Code default (Linux): `~/.cache/cflibs/jax`, per-user.** Set by
`cflibs.core.platform_config.configure_jax()` via
`os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/cflibs/jax"))`.

The original design pointed the default at the NFS share
`/cluster/shared/jax-cache` so that the first iteration of a parameter
sweep on **any** node warmed the cache for **all** nodes. That sharing
benefit was real but the failure mode was worse:

- **uid skew.** `/cluster/shared` is NFS-exported, and uids are not
  guaranteed identical across the compute nodes (vasp-01/02/03). A cache
  entry written by uid *N* on one node could be unreadable/unwritable —
  or owned by a *different* real user — when a job on another node mapped
  uid *N* to someone else. The `1777` sticky-world-writable mode papered
  over writes but not the ownership/lock semantics JAX relies on.
- **Hung jobs.** This manifested as hung jobs **1909, 1914, and 1915**:
  workers blocked on a cache directory they could neither complete a
  write to nor cleanly fall back from. (Recorded in
  `.serena/memories/physics_invariants_and_gotchas.md` under "Cluster JAX
  cache uid skew".)

The standing rule since then has been the user-private cache. J0
(ADR-0004 §5.5) makes that rule the **code default** so a forgotten
job-script export can no longer reintroduce the shared-path trap. A
per-user cache trades cross-host warm-up for correctness; in practice the
campaign job archetypes (§5.5) keep long-lived workers that compile each
bucket once per process, so the cross-host benefit was marginal anyway.

## Wired-in defaults

`cflibs.core.platform_config.configure_jax()` (Linux path) sets:

```sh
JAX_COMPILATION_CACHE_DIR=~/.cache/cflibs/jax          # expanduser'd, per-user
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5
```

via `os.environ.setdefault(...)` — so explicit env-var overrides (e.g.
unit tests with `tmp_path`, or a job script that exports a different
path) win.

> **Note:** `scripts/run_unified_benchmark.py` still sets its own
> top-of-file `JAX_COMPILATION_CACHE_DIR=/cluster/shared/jax-cache`
> default (also via `setdefault`). If you run that script on the cluster,
> export `JAX_COMPILATION_CACHE_DIR=$HOME/.cache/cflibs/jax` (or any
> per-user path) before invoking it to avoid the shared-path trap above.

The `0.5 s` min-compile-time floor widens cache coverage below JAX's
1.0 s default. Many of our small forward-model JITs compile in 0.5-0.9 s
and would otherwise be skipped.

## Opting back in to a shared cache (explicit only)

If a campaign genuinely benefits from cross-host warm-up and the cluster
has been confirmed to use **identical uids on every node**, the shared
path can still be used — but only by explicit export, never by default:

```bash
# As root on slurm-ctl (NFS server for /cluster/shared):
ssh root@10.0.0.5 'mkdir -p /cluster/shared/jax-cache && chmod 1777 /cluster/shared/jax-cache'

# Verify uid consistency and writability from each compute node:
for host in 10.0.0.20 10.0.0.21 10.0.0.22; do
  ssh root@$host 'id -u; test -w /cluster/shared/jax-cache && echo OK || echo FAIL'
done
# All `id -u` values for your account MUST match. If they differ, do NOT
# use the shared path — that is exactly the skew that hung 1909/1914/1915.

# Then in the job script (overrides the per-user setdefault default):
export JAX_COMPILATION_CACHE_DIR=/cluster/shared/jax-cache
```

The `1777` perms (sticky-world-writable, like `/tmp`) match the standard
for shared scratch dirs. JAX populates per-version subdirs underneath,
so different JAX/CUDA combinations coexist without conflict.

## Acceptance test

```bash
CACHE=$HOME/.cache/cflibs/jax  # the per-user default

# Cold cache: first run, expect ~30 s JIT overhead at startup
rm -rf "$CACHE"/*
time JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR="$CACHE" .venv/bin/python -c '
import jax, jax.numpy as jnp
f = jax.jit(lambda x: jnp.sin(x) * jnp.cos(x) + jnp.exp(-x**2))
print(f(jnp.arange(1000.0))[:3])
'

# Warm cache, same host: expect <2 s startup (cache hit)
time JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR="$CACHE" .venv/bin/python -c '
import jax, jax.numpy as jnp
f = jax.jit(lambda x: jnp.sin(x) * jnp.cos(x) + jnp.exp(-x**2))
print(f(jnp.arange(1000.0))[:3])
'
```

Because the default is now per-user (not NFS), the cache warms only the
node it runs on. That is the intended trade — see the pathology section
above for why cross-host sharing was retired as the default.

## When to bust the cache

JAX detects (JAX version, jaxlib version, backend, source HLO) changes
automatically — bumping any of those creates a new subdirectory and old
entries are simply unused. You generally don't need to clear the cache.

Exceptions:
- Disk pressure on `$HOME` (run `du -sh ~/.cache/cflibs/jax`).
- Suspected corruption (e.g. a node crashed mid-write).

To clear: `rm -rf ~/.cache/cflibs/jax/*`. JAX repopulates on the next
compile.

## Override for a different location

To point the cache somewhere else (e.g. a fast local scratch disk):

```bash
export JAX_COMPILATION_CACHE_DIR=/scratch/$USER/jax-compile
```

Any explicit export wins over the per-user `setdefault` default. Test
fixtures that need a clean cache should use `tmp_path` with
`monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path))` — the
production default uses `os.environ.setdefault`, so this override wins.

## References

- JAX persistent cache docs: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- Implementation: `cflibs/core/platform_config.py::configure_jax()`
  (per-user default), `scripts/run_unified_benchmark.py` (top of file —
  still defaults to the shared path; export a per-user path before running).
- Acceptance test: `tests/scripts/test_jax_compile_cache.py`.
- Pathology record: `.serena/memories/physics_invariants_and_gotchas.md`
  ("Cluster JAX cache uid skew"); ADR-0004 §5.5.

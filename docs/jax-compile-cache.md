# JAX persistent compile cache (NFS-shared)

**Status:** active since T1.2 (CF-LIBS-improved-b5xw), 2026-05-12.

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

Pointing the cache at our NFS share (`/cluster/shared/jax-cache`) means
the first iteration of a parameter sweep on **any** node warms the cache
for **all** nodes.

## Wired-in defaults

`cflibs.core.platform_config.configure_jax()` (Linux path) and
`scripts/run_unified_benchmark.py` both set:

```sh
JAX_COMPILATION_CACHE_DIR=/cluster/shared/jax-cache
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5
```

via `os.environ.setdefault(...)` — so explicit env-var overrides (e.g.
unit tests with `tmp_path`) win.

The `0.5 s` min-compile-time floor widens cache coverage below JAX's
1.0 s default. Many of our small forward-model JITs compile in 0.5-0.9 s
and would otherwise be skipped.

## One-time setup on the cluster

The NFS-shared path needs to exist on the NFS server (slurm-ctl,
10.0.0.5) with world-writable+sticky permissions so any user/process
on any node can write to it without coordinating ownership.

```bash
# As root on slurm-ctl (NFS server for /cluster/shared):
ssh root@10.0.0.5 'mkdir -p /cluster/shared/jax-cache && chmod 1777 /cluster/shared/jax-cache'

# Verify from each compute node:
for host in 10.0.0.20 10.0.0.21 10.0.0.22; do
  ssh root@$host 'test -w /cluster/shared/jax-cache && echo OK || echo FAIL'
done
```

The `1777` perms (sticky-world-writable, like `/tmp`) match the standard
for shared scratch dirs. JAX populates per-version subdirs underneath,
so different JAX/CUDA combinations coexist without conflict.

## Acceptance test

```bash
# Cold cache: first run, expect ~30 s JIT overhead at startup
rm -rf /cluster/shared/jax-cache/*
time JAX_PLATFORMS=cpu .venv/bin/python -c '
import jax, jax.numpy as jnp
import os; os.environ["JAX_COMPILATION_CACHE_DIR"] = "/cluster/shared/jax-cache"
f = jax.jit(lambda x: jnp.sin(x) * jnp.cos(x) + jnp.exp(-x**2))
print(f(jnp.arange(1000.0))[:3])
'

# Warm cache, same host: expect <2 s startup (cache hit)
time JAX_PLATFORMS=cpu .venv/bin/python -c '
import jax, jax.numpy as jnp
import os; os.environ["JAX_COMPILATION_CACHE_DIR"] = "/cluster/shared/jax-cache"
f = jax.jit(lambda x: jnp.sin(x) * jnp.cos(x) + jnp.exp(-x**2))
print(f(jnp.arange(1000.0))[:3])
'

# Warm cache, different host (NFS share): same speedup
ssh root@10.0.0.21 'cd /scratch/cf-libs-bench/repo && time JAX_PLATFORMS=cpu .venv/bin/python -c "
import jax, jax.numpy as jnp
import os; os.environ[\"JAX_COMPILATION_CACHE_DIR\"] = \"/cluster/shared/jax-cache\"
f = jax.jit(lambda x: jnp.sin(x) * jnp.cos(x) + jnp.exp(-x**2))
print(f(jnp.arange(1000.0))[:3])
"'
```

## When to bust the cache

JAX detects (JAX version, jaxlib version, backend, source HLO) changes
automatically — bumping any of those creates a new subdirectory and old
entries are simply unused. You generally don't need to clear the cache.

Exceptions:
- Disk pressure on `/cluster/shared/` (run `du -sh /cluster/shared/jax-cache`).
- Suspected corruption (e.g. a node crashed mid-write under NFS lag).

To clear: `rm -rf /cluster/shared/jax-cache/*` on the NFS server. JAX
will repopulate on the next compile.

## Override for local dev

To use a local cache instead of NFS during development:

```bash
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax-compile
```

Test fixtures that need a clean cache should use `tmp_path` with
`monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path))` — the
production default uses `os.environ.setdefault`, so this override wins.

## References

- JAX persistent cache docs: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- Implementation: `cflibs/core/platform_config.py::configure_jax()`,
  `scripts/run_unified_benchmark.py` (top of file).
- Acceptance test: `tests/scripts/test_jax_compile_cache.py`.

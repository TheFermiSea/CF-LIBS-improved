# Parameter Sweep Server — Cloud LLM Design Consultation

**Issue:** `CF-LIBS-improved-32s9` (T2.1)
**Date:** 2026-05-12
**Branch:** `feat/parameter-sweep-server`

This document captures the cross-model design synthesis for
`cflibs.parameter_sweep_server`, a long-lived daemon that amortizes
dataset / JAX warmup across a multi-config parameter sweep.

## Models consulted

- **GPT-5.3 Codex** (`gpt-5.3-codex` via CLIAPIProxy)
- **Gemini 3.1 Pro** (`gemini-3.1-pro-low` via CLIAPIProxy) — see end for
  delta vs. Codex; if unreachable, design follows Codex unchanged.

Both queried via `localhost:8317/v1/chat/completions` per the
`AGENTS.md` cross-model consultation rule.

## Design questions

1. Server library: `socketserver.ThreadingTCPServer` vs.
   `asyncio.start_server`.
2. Wire protocol: line-delimited JSON vs. length-prefixed JSON.
3. Concurrency model for JAX-backed analyses (shared state).
4. Client-disconnect detection mid-analysis.
5. Response envelope schema.

## Synthesized design (binding)

### Process model

- **Single long-lived process per node**.
- Startup phase:
  - Open atomic DB (`ASD_da/libs_production.db`) read-only.
  - Load Vrabel + BHVO-2 + Aalto datasets via `load_default_datasets`.
  - Construct `UnifiedBenchmarkRunner` (workflow registries built once).
  - Optionally warm JAX JIT cache (defer to natural first-request
    compilation — explicit warmup not on critical path for T2.1).
- Runtime phase:
  - `asyncio.start_server` on `127.0.0.1:8501`.
  - JSON requests routed through bounded `asyncio.Queue` to a single
    compute worker.

### 1. Async over threading

**Decision: `asyncio.start_server` + single compute worker task.**

Codex rationale (verbatim summary):
- Clear separation of socket lifecycle vs. compute lane.
- Per-connection timeouts, bounded queue, graceful shutdown easier to
  orchestrate.
- 30s–10min analyses make thread-per-connection error paths fragile
  (no clean cancellation, harder to drain).

We accept this. The compute worker is a separate `asyncio.Task` that
pulls from the queue and runs `await
asyncio.get_running_loop().run_in_executor(None, _do_analysis, …)` so
the analysis blocks a worker thread (single threaded executor of
size 1) without blocking the event loop. This gives us cooperative
cancellation hooks via the executor's future plus single-flight
analysis on the shared JAX state.

### 2. Wire protocol — length-prefixed JSON

**Decision: 8-byte big-endian length prefix + UTF-8 JSON payload.**

```
Request:  [8B BE uint64 length][JSON bytes]
Response: [8B BE uint64 length][JSON bytes]
```

Codex rationale:
- Robust for multi-MB responses (composition_records can easily exceed
  1 MB for full Vrabel runs).
- `asyncio.StreamReader.readexactly(n)` makes truncation detection
  trivial.
- Avoids NDJSON-style ambiguity if a payload value contains an
  embedded newline.

We add: **8 MiB request cap, 64 MiB response cap** at the server, both
configurable via env (`SWEEP_MAX_REQUEST_BYTES`,
`SWEEP_MAX_RESPONSE_BYTES`).

### 3. Concurrency — serialize analyses

**Decision: compute concurrency = 1 (single worker).**

JAX tracing/compilation on the same default device is not thread-safe.
We accept up to N concurrent client connections (bounded by
`asyncio.Queue(maxsize=16)`), but only one analysis executes at a
time. Excess requests get `status="overloaded"` after queue is full.

Future work (deferred): process-level isolation if we ever need true
parallelism on a single node. Out of scope for T2.1 — see the
Tier 3 deferred-work note in the bd issue.

### 4. Client disconnect detection

**Decision: a per-request watcher task `wait_disconnect` that calls
`reader.read(1)` and sets a cooperative `cancel_event`.**

The compute worker checks `cancel_event` at coarse checkpoints
(between datasets, between workflow runs). True mid-kernel preemption
is not available in JAX — we accept that a cancelled request may still
consume its current outer fold before unwinding. This matches Codex's
guidance: "cooperative cancel, not preemption."

### 5. Response envelope (binding schema)

```json
{
  "request_id": "uuid4-string",
  "status": "ok | error | overloaded | cancelled",
  "duration_sec": 12.345,
  "queued_sec": 0.012,
  "error": null,
  "result": {
    "composition_records": [...],
    "id_records": [...],
    "manifest": { "config": {...}, "datasets_loaded": [...] }
  },
  "server": {
    "version": "<git-sha>",
    "worker_id": "<host>:<pid>"
  }
}
```

On error:

```json
{
  "request_id": "...",
  "status": "error",
  "duration_sec": 0.4,
  "queued_sec": 0.0,
  "error": {
    "code": "VALIDATION_ERROR | ANALYSIS_ERROR | INTERNAL_ERROR | OVERLOADED | CANCELLED | SHUTTING_DOWN",
    "message": "safe message",
    "retryable": false
  },
  "result": null,
  "server": { "version": "...", "worker_id": "..." }
}
```

Stable error codes; never return raw tracebacks (logged server-side
with the request_id).

### Lifecycle

**Startup:**
1. Parse env / CLI.
2. Set `JAX_PLATFORMS=cpu` if unset (matches `run_unified_benchmark.py`).
3. Set `JAX_COMPILATION_CACHE_DIR` to NFS path if `SWEEP_JAX_CACHE_DIR`
   present.
4. Load datasets, build `UnifiedBenchmarkRunner`.
5. Bind listener; log "ready".

**Per request:**
1. `readexactly(8)` length, `readexactly(n)` payload.
2. JSON-decode + schema-validate.
3. Assign `request_id` (use client-provided if present, else `uuid4`).
4. Enqueue with `enq_ts`; reject with `overloaded` if queue full.
5. Worker pops, validates config against
   `UnifiedBenchmarkRunner.id_registry` / `composition_registry`.
6. Run analysis on shared runner; collect records.
7. Frame and send response.

**SIGTERM:**
1. Stop accepting new connections.
2. Mark draining; reject new jobs with `status=shutting_down`.
3. Wait up to `SWEEP_DRAIN_SEC` (default 60 s) for current job.
4. JAX persistent cache is auto-flushed by JAX itself when the
   compilation cache directory is set — we just need to ensure the
   process exits cleanly. We do not have to manually `sync`.
5. Close DB / dataset handles.
6. `loop.stop()` and exit 0.

## Gemini status

Gemini 3.1 Pro returned a 403 (`PERMISSION_DENIED`) on this account.
`gemini-3-pro-preview` is sunset. We swapped to **Claude Opus 4.7** as
the second oracle and captured its review below.

## Claude Opus 4.7 review — refinements adopted

Opus surfaced 10 concrete bugs/concerns. Adopting all that fit the
T2.1 scope:

1. **Length-prefix validation before allocation.** Reject the integer
   length itself before calling `readexactly(n)` — otherwise a
   malicious 2 GiB prefix triggers a giant allocation. Implemented in
   `protocol.read_framed_json`.
2. **Bounded queue + dead-code `overloaded`.** Codex's "overloaded"
   status is dead code without a `maxsize`. Set
   `asyncio.Queue(maxsize=int(env.get("SWEEP_QUEUE_MAX", 4)))` and
   return `overloaded` on `put_nowait` failure.
3. **SIGTERM never blocks indefinitely on NFS.** Use
   `loop.add_signal_handler` (not a sync handler in an executor
   thread); drain current job with a hard deadline; wrap any cache
   flush in `asyncio.wait_for` to bound systemd's `TimeoutStopSec`.
4. **JSON-encode response in the executor.** Otherwise large array
   marshalling blocks the event loop. Implemented:
   `_do_analysis` returns `bytes` (already framed body), the event
   loop only writes them.
5. **Server-assigned `request_id` if client omits.** Correlation must
   not depend on client-supplied IDs for malformed requests.
6. **Document stage-level cancel granularity.** JAX dispatch holds the
   thread through XLA — cooperative `cancel_event` is checked between
   dataset / workflow iterations, not mid-kernel. We document this in
   the protocol module's docstring.
7. **Watcher-task cleanup.** Use `try/finally` to cancel and `await`
   the `wait_disconnect` watcher on normal completion to avoid task
   leaks.
8. **JAX cache flush — don't call `jax.clear_caches()`** (it evicts,
   doesn't persist). Instead, set `JAX_COMPILATION_CACHE_DIR` to the
   NFS path at startup and rely on JAX's auto-flush. On shutdown we
   only need to ensure the process exits cleanly — no manual flush
   required.

Three concerns we **skip** per Opus's guidance:

- 4-byte vs. 8-byte prefix — 8 is fine, documented.
- Mid-kernel cancellation — out of scope.
- Per-client fairness / priority queue — out of scope (single warm
  worker is correct).

Saved raw responses:
- Codex: `/tmp/codex_consult.md` (local, not committed)
- Opus: `/tmp/opus_consult.md` (local, not committed)

## Implementation locations

- `cflibs/parameter_sweep_server/__init__.py` — `serve(...)` entry point.
- `cflibs/parameter_sweep_server/protocol.py` — framing helpers, schema.
- `cflibs/parameter_sweep_server/server.py` — async server + worker.
- `cflibs/parameter_sweep_server/client.py` — sync + async client helpers.
- `scripts/start_sweep_server.sh` — launcher wrapper.
- `tests/parameter_sweep_server/test_e2e.py` — end-to-end test.

## Microbenchmark plan

8 sequential requests, varying `seed` only, against one in-process
server. Compare wall time to 8 sequential
`scripts/run_unified_benchmark.py` subprocess invocations.

Acceptance threshold: in-process loop must complete in **< 1/4** the
subprocess wall time. This is recorded in
`tests/parameter_sweep_server/test_e2e.py::test_sweep_server_amortization`
(marked `@pytest.mark.slow` so CI doesn't pay the cost on every PR).

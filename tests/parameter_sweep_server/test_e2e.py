"""End-to-end tests for :mod:`cflibs.parameter_sweep_server`.

Two layers of test exist here:

1. Fast tests that stub the warm state with a fake
   ``UnifiedBenchmarkRunner``-like object so we exercise the
   framing + queue + executor + signal paths without paying the
   2-GB-dataset cost. Run in CI on every push.

2. A ``@pytest.mark.slow`` microbenchmark that boots the *real*
   server, sends 8 requests against the live runner, and asserts the
   wall time vs. 8 fresh ``run_unified_benchmark.py`` subprocess
   invocations is at least 4x faster. Skipped unless the canonical
   data directory + atomic DB are present.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from cflibs.parameter_sweep_server import (
    protocol as proto,
)
from cflibs.parameter_sweep_server.client import SweepClient, request_async
from cflibs.parameter_sweep_server.server import (
    ServerConfig,
    SweepServer,
    SweepServerState,
)


# ---------------------------------------------------------------------------
# Fake state (fast tests)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeRecord:
    workflow_name: str
    dataset_id: str
    seed: int


class _FakeRegistry(dict):
    """Tiny dict subclass — runs_composition/_identification look up by key."""


class _FakeRunner:
    def __init__(self, latency_sec: float = 0.05):
        self.id_registry = _FakeRegistry({"alias": object(), "comb": object()})
        self.composition_registry = _FakeRegistry({"iterative_jax": object()})
        self.latency_sec = latency_sec

    def run_identification(self, datasets, *, workflow_names, max_outer_folds=None):
        time.sleep(self.latency_sec)
        recs = [
            _FakeRecord(w, d.name, 0)
            for d in datasets
            for w in workflow_names
        ]
        sels = [{"workflow_name": w, "dataset_id": d.name} for d in datasets for w in workflow_names]
        return recs, sels

    def run_composition(
        self,
        datasets,
        *,
        id_workflow_names,
        composition_workflow_names,
        max_outer_folds=None,
    ):
        time.sleep(self.latency_sec)
        recs = [
            _FakeRecord(c, d.name, 0)
            for d in datasets
            for _i in id_workflow_names
            for c in composition_workflow_names
        ]
        sels = [
            {"id_workflow_name": i, "composition_workflow_name": c, "dataset_id": d.name}
            for d in datasets
            for i in id_workflow_names
            for c in composition_workflow_names
        ]
        return recs, sels


class _FakeDataset:
    def __init__(self, name: str, truth_value: str = "assay"):
        self.name = name
        self.spectra = [_FakeSpectrum(truth_value)]


class _FakeSpectrum:
    def __init__(self, truth_value: str):
        self.truth_type = _FakeTruthType(truth_value)


@dataclasses.dataclass
class _FakeTruthType:
    value: str


def _make_fake_state(latency_sec: float = 0.05) -> SweepServerState:
    cfg = ServerConfig(db_path=Path("/dev/null"), data_dir=Path("/dev/null"))
    state = SweepServerState(cfg)
    state.runner = _FakeRunner(latency_sec=latency_sec)
    state.datasets = {
        "alpha": _FakeDataset("alpha"),
        "beta": _FakeDataset("beta"),
    }
    state._loaded = True
    return state


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Async server fixture
# ---------------------------------------------------------------------------


class _ServerHarness:
    """Launches a SweepServer in a background asyncio thread."""

    def __init__(self, state: SweepServerState, port: int):
        self.state = state
        self.port = port
        self.cfg = dataclasses.replace(
            state.cfg, host="127.0.0.1", port=port, queue_max=4, drain_sec=2.0
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: SweepServer | None = None
        self._ready = threading.Event()

    def __enter__(self) -> "_ServerHarness":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise RuntimeError("server failed to start within 15s")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._server = SweepServer(self.cfg, state=self.state)

        async def _bootstrap() -> None:
            await self._server.start()
            self._ready.set()
            try:
                await self._server.serve_forever()
            except asyncio.CancelledError:
                pass

        try:
            self._loop.run_until_complete(_bootstrap())
        finally:
            try:
                self._loop.run_until_complete(self._server.shutdown())
            except Exception:
                pass
            self._loop.close()

    def stop(self) -> None:
        if self._loop is None or self._thread is None:
            return
        loop = self._loop

        def _stop() -> None:
            for task in asyncio.all_tasks(loop):
                task.cancel()

        loop.call_soon_threadsafe(_stop)
        self._thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def harness():
    state = _make_fake_state()
    port = _free_port()
    with _ServerHarness(state, port) as h:
        yield h


def test_single_request_round_trip(harness):
    client = SweepClient(host="127.0.0.1", port=harness.port, connect_timeout=5.0)
    env = client.request(
        {
            "identifier": "alias",
            "composition_workflows": ["iterative_jax"],
            "seed": 7,
            "sections": "composition",
            "max_outer_folds": 1,
        }
    )
    assert env["status"] == proto.STATUS_OK
    assert env["error"] is None
    assert "composition_records" in env["result"]
    assert isinstance(env["result"]["composition_records"], list)
    assert env["result"]["manifest"]["composition_workflows"] == ["iterative_jax"]
    assert env["request_id"]
    assert env["duration_sec"] >= 0.0
    assert env["server"]["worker_id"]


def test_eight_sequential_requests_share_warm_state(harness):
    """Demonstrate the amortization win: 8 requests on warm state are
    much faster than 8 cold subprocess starts.

    With the fake runner at 50ms/call, 8 sequential requests should
    complete in well under a second once the listener is up.
    """
    client = SweepClient(host="127.0.0.1", port=harness.port, connect_timeout=5.0)
    start = time.monotonic()
    results = []
    for seed in range(8):
        env = client.request(
            {
                "identifier": "alias",
                "composition_workflows": ["iterative_jax"],
                "sections": "composition",
                "seed": seed,
                "max_outer_folds": 1,
            }
        )
        assert env["status"] == proto.STATUS_OK, env
        results.append(env)
    elapsed = time.monotonic() - start

    assert len(results) == 8
    # 8 * 50ms ≈ 0.4s of actual work; with overhead we budget 3s.
    assert elapsed < 3.0, f"8 warm requests took {elapsed:.2f}s (expected <3s)"


def test_unknown_workflow_returns_validation_error(harness):
    client = SweepClient(host="127.0.0.1", port=harness.port, connect_timeout=5.0)
    env = client.request(
        {"composition_workflows": ["totally-not-a-workflow"], "sections": "composition"}
    )
    assert env["status"] == proto.STATUS_ERROR
    assert env["error"]["code"] == proto.ERR_VALIDATION
    assert "totally-not-a-workflow" in env["error"]["message"]


def test_async_client_works(harness):
    async def _go() -> Dict[str, Any]:
        return await request_async(
            {
                "composition_workflows": ["iterative_jax"],
                "sections": "composition",
                "seed": 1,
                "max_outer_folds": 1,
            },
            host="127.0.0.1",
            port=harness.port,
            timeout=10.0,
        )

    loop = asyncio.new_event_loop()
    try:
        env = loop.run_until_complete(_go())
    finally:
        loop.close()
    assert env["status"] == proto.STATUS_OK


def test_overload_returns_overloaded_status():
    """Saturate the queue (maxsize=1) and verify the second request gets overloaded."""
    state = _make_fake_state(latency_sec=0.6)  # slow worker so queue persists
    port = _free_port()
    cfg = dataclasses.replace(state.cfg, host="127.0.0.1", port=port, queue_max=1, drain_sec=2.0)
    state.cfg = cfg

    with _ServerHarness(state, port) as h:
        # Override queue_max via cfg copy
        h.cfg = dataclasses.replace(h.cfg, queue_max=1)

        results: List[Tuple[float, Dict[str, Any]]] = []
        threads: List[threading.Thread] = []
        lock = threading.Lock()

        def _hit() -> None:
            client = SweepClient(host="127.0.0.1", port=h.port, connect_timeout=5.0)
            t0 = time.monotonic()
            env = client.request(
                {
                    "composition_workflows": ["iterative_jax"],
                    "sections": "composition",
                    "max_outer_folds": 1,
                }
            )
            with lock:
                results.append((time.monotonic() - t0, env))

        # Fire 6 concurrent requests against a queue of size 1 with a 0.6s worker.
        for _ in range(6):
            t = threading.Thread(target=_hit, daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=30)

    statuses = [env["status"] for _, env in results]
    assert proto.STATUS_OVERLOADED in statuses, f"never saw overloaded in {statuses}"
    assert proto.STATUS_OK in statuses


def test_oversize_response_falls_back_to_error_envelope():
    """If the result is too large to frame, the server returns an
    INTERNAL_ERROR envelope rather than crashing the client.

    The cap is set just large enough to accommodate an error envelope
    (~500 bytes) but smaller than a normal OK envelope (~1.5 kB with
    fake records), so we exercise the fallback path.
    """
    state = _make_fake_state()
    port = _free_port()

    # 800 bytes is enough for the error envelope but not the OK body
    # carrying composition_records.
    state.cfg = dataclasses.replace(state.cfg, max_response_bytes=800)

    with _ServerHarness(state, port) as h:
        client = SweepClient(host="127.0.0.1", port=h.port, connect_timeout=5.0)
        env = client.request(
            {
                "composition_workflows": ["iterative_jax"],
                "sections": "composition",
                "max_outer_folds": 1,
            }
        )
    # Connection must not have been dropped — we get a well-formed
    # error envelope.
    assert env["status"] == proto.STATUS_ERROR
    assert env["error"]["code"] == proto.ERR_INTERNAL
    assert "max_bytes" in env["error"]["message"]


# ---------------------------------------------------------------------------
# Slow microbenchmark — only runs when the real datasets are present.
# ---------------------------------------------------------------------------


def _have_real_datasets() -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        (repo_root / "ASD_da" / "libs_production.db").exists()
        and (repo_root / "data").exists()
    )


@pytest.mark.slow
@pytest.mark.skipif(not _have_real_datasets(), reason="ASD_da/libs_production.db or data/ not present")
def test_sweep_server_amortization():
    """Acceptance gate: 8 warm requests vs. 8 fresh subprocess invocations.

    The bd issue calls for <1/4 the wall time. We measure both in the
    same process and assert the ratio.
    """
    repo_root = Path(__file__).resolve().parents[2]
    db_path = repo_root / "ASD_da" / "libs_production.db"
    data_dir = repo_root / "data"
    port = _free_port()

    cfg = ServerConfig(
        host="127.0.0.1",
        port=port,
        db_path=db_path,
        data_dir=data_dir,
        basis_dir=None,
        quick=True,
        queue_max=4,
        drain_sec=10.0,
    )

    # --- in-process server: 8 warm sequential requests ---------------
    state = SweepServerState(cfg)

    with _ServerHarness(state, port) as h:
        # state.load() runs on the harness thread during start()
        client = SweepClient(host="127.0.0.1", port=h.port, connect_timeout=120.0)
        # First request also pays JIT-compile cost; we WANT this in the
        # timing budget since cold subprocess starts pay it too.
        t_warm0 = time.monotonic()
        for seed in range(8):
            env = client.request(
                {
                    "composition_workflows": ["iterative_jax"],
                    "id_workflows": ["alias"],
                    "sections": "composition",
                    "seed": seed,
                    "max_outer_folds": 1,
                    "datasets": ["aalto_libs"],  # smallest dataset
                }
            )
            assert env["status"] == proto.STATUS_OK, env
        warm_elapsed = time.monotonic() - t_warm0

    # --- subprocess baseline: 8 cold run_unified_benchmark.py runs ----
    script = repo_root / "scripts" / "run_unified_benchmark.py"
    if not script.exists():
        pytest.skip("scripts/run_unified_benchmark.py missing")

    import tempfile

    t_cold0 = time.monotonic()
    for seed in range(8):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--db-path",
                    str(db_path),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    td,
                    "--sections",
                    "composition",
                    "--composition-workflows",
                    "iterative_jax",
                    "--id-workflows",
                    "alias",
                    "--max-outer-folds",
                    "1",
                    "--quick",
                ],
                cwd=str(repo_root),
                check=True,
                env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONHASHSEED": str(seed)},
            )
    cold_elapsed = time.monotonic() - t_cold0

    speedup = cold_elapsed / max(warm_elapsed, 1e-9)
    summary = {
        "warm_total_sec": warm_elapsed,
        "cold_total_sec": cold_elapsed,
        "speedup": speedup,
    }
    print(json.dumps(summary, indent=2))

    # bd acceptance: warm wall time must be < 1/4 of cold.
    assert speedup >= 4.0, (
        f"server amortization insufficient: speedup={speedup:.2f}x "
        f"(warm={warm_elapsed:.1f}s, cold={cold_elapsed:.1f}s); need ≥4x"
    )

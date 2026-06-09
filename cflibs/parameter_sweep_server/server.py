"""Async server implementation for the parameter sweep daemon.

The server holds a single warm :class:`UnifiedBenchmarkRunner` plus a
preloaded ``datasets: dict[str, BenchmarkDataset]``. Requests are framed
JSON; each request describes one benchmark configuration to run against
the warm state.

Concurrency model
-----------------
- The event loop accepts unbounded concurrent client connections.
- All analyses execute on a single ``ThreadPoolExecutor(max_workers=1)``
  so JAX's shared default-device state is never touched from two
  threads at once.
- Inbound work is queued on an ``asyncio.Queue(maxsize=SWEEP_QUEUE_MAX)``;
  ``put_nowait`` failures produce a ``status="overloaded"`` envelope.
- JSON encoding of the result happens in the worker thread (large
  array marshalling otherwise blocks the event loop — Opus
  refinement #4).

SIGTERM / SIGINT
----------------
Signal handlers registered via ``loop.add_signal_handler`` flip a
``_shutting_down`` flag. The accept-loop refuses new connections and
in-flight jobs get a hard drain deadline of ``SWEEP_DRAIN_SEC`` (default
60 s). No manual JAX cache flush — JAX auto-persists when
``JAX_COMPILATION_CACHE_DIR`` is set at process startup.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from . import protocol as proto

log = logging.getLogger("cflibs.parameter_sweep_server")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ServerConfig:
    """Server-side knobs. CLI / env overrides are applied in ``from_env``."""

    host: str = "127.0.0.1"
    port: int = 8501
    db_path: Path = Path("ASD_da/libs_production.db")
    data_dir: Path = Path("data")
    basis_dir: Optional[Path] = Path("output/basis_libraries")
    synthetic_corpus: Optional[Path] = None
    quick: bool = False
    queue_max: int = 4
    drain_sec: float = 60.0
    max_request_bytes: int = proto.MAX_REQUEST_BYTES_DEFAULT
    max_response_bytes: int = proto.MAX_RESPONSE_BYTES_DEFAULT
    jax_cache_dir: Optional[Path] = None

    @classmethod
    def from_env(
        cls,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        basis_dir: Optional[Path] = None,
        synthetic_corpus: Optional[Path] = None,
        quick: Optional[bool] = None,
    ) -> "ServerConfig":
        def _opt_path(env_name: str) -> Optional[Path]:
            v = os.environ.get(env_name)
            return Path(v) if v else None

        return cls(
            host=host or os.environ.get("SWEEP_HOST", "127.0.0.1"),
            port=int(port if port is not None else os.environ.get("SWEEP_PORT", "8501")),
            db_path=db_path or _opt_path("SWEEP_DB_PATH") or Path("ASD_da/libs_production.db"),
            data_dir=data_dir or _opt_path("SWEEP_DATA_DIR") or Path("data"),
            basis_dir=(
                basis_dir
                if basis_dir is not None
                else (_opt_path("SWEEP_BASIS_DIR") or Path("output/basis_libraries"))
            ),
            synthetic_corpus=(
                synthetic_corpus
                if synthetic_corpus is not None
                else _opt_path("SWEEP_SYNTHETIC_CORPUS")
            ),
            quick=quick if quick is not None else bool(int(os.environ.get("SWEEP_QUICK", "0"))),
            queue_max=int(os.environ.get("SWEEP_QUEUE_MAX", "4")),
            drain_sec=float(os.environ.get("SWEEP_DRAIN_SEC", "60")),
            max_request_bytes=int(
                os.environ.get("SWEEP_MAX_REQUEST_BYTES", str(proto.MAX_REQUEST_BYTES_DEFAULT))
            ),
            max_response_bytes=int(
                os.environ.get("SWEEP_MAX_RESPONSE_BYTES", str(proto.MAX_RESPONSE_BYTES_DEFAULT))
            ),
            jax_cache_dir=_opt_path("SWEEP_JAX_CACHE_DIR"),
        )


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Job:
    request_id: str
    config: Dict[str, Any]
    enq_ts: float
    fut: "asyncio.Future[bytes]"  # framed response body (NOT including prefix)
    cancel: asyncio.Event


class SweepServerState:
    """Warm runtime state shared across requests.

    Constructed once at startup; never mutated after :meth:`load` returns.
    """

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.runner = None  # type: ignore[assignment]
        self.datasets: Dict[str, Any] = {}
        self.server_info = {
            "version": _git_sha(),
            "worker_id": f"{socket.gethostname()}:{os.getpid()}",
            "protocol_version": proto.PROTOCOL_VERSION,
        }
        self._loaded = False

    def load(self) -> None:
        """Heavy synchronous load — call once before serving."""
        if self._loaded:
            return

        # Mirror run_unified_benchmark.py's JAX_PLATFORMS default.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        if self.cfg.jax_cache_dir is not None:
            self.cfg.jax_cache_dir.mkdir(parents=True, exist_ok=True)
            # JAX auto-persists when this env var is set — no manual
            # flush required on shutdown (Opus refinement #8).
            os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(self.cfg.jax_cache_dir))

        if not self.cfg.db_path.exists():
            raise FileNotFoundError(f"Atomic database not found: {self.cfg.db_path}")
        if not self.cfg.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.cfg.data_dir}")

        from cflibs.benchmark import UnifiedBenchmarkRunner, load_default_datasets

        log.info(
            "Loading benchmark datasets from %s (synthetic=%s)",
            self.cfg.data_dir,
            self.cfg.synthetic_corpus,
        )
        t0 = time.monotonic()
        self.datasets = load_default_datasets(
            self.cfg.data_dir,
            synthetic_corpus_path=self.cfg.synthetic_corpus,
            # Honour any per-call vrabel cap; the server-level default
            # matches run_unified_benchmark.py's CLI default.
            vrabel_max_shots_per_sample=50,
        )
        log.info(
            "Datasets loaded in %.1fs: %s",
            time.monotonic() - t0,
            sorted(self.datasets.keys()),
        )

        log.info("Constructing UnifiedBenchmarkRunner …")
        t1 = time.monotonic()
        self.runner = UnifiedBenchmarkRunner(
            db_path=self.cfg.db_path,
            basis_dir=(
                self.cfg.basis_dir if self.cfg.basis_dir and self.cfg.basis_dir.exists() else None
            ),
            quick=self.cfg.quick,
        )
        log.info(
            "Runner ready in %.1fs (id_workflows=%d, composition_workflows=%d)",
            time.monotonic() - t1,
            len(self.runner.id_registry),
            len(self.runner.composition_registry),
        )

        self._loaded = True


# ---------------------------------------------------------------------------
# Analysis dispatcher
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the short HEAD SHA, or '' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _validate_config(state: SweepServerState, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise + validate a request config. Raises ``ValueError`` on error."""
    if not isinstance(cfg, dict):
        raise ValueError("request payload must be a JSON object")

    id_workflows = cfg.get("id_workflows") or cfg.get("identification_workflows")
    if id_workflows is None and "identifier" in cfg:
        ident = cfg["identifier"]
        id_workflows = [ident] if isinstance(ident, str) else list(ident)
    if id_workflows is None:
        id_workflows = list(state.runner.id_registry.keys())
    elif isinstance(id_workflows, str):
        id_workflows = [id_workflows]
    id_workflows = [str(w) for w in id_workflows]

    comp_workflows = cfg.get("composition_workflows")
    if comp_workflows is None:
        comp_workflows = list(state.runner.composition_registry.keys())
    elif isinstance(comp_workflows, str):
        comp_workflows = [comp_workflows]
    comp_workflows = [str(w) for w in comp_workflows]

    unknown_id = sorted(set(id_workflows) - set(state.runner.id_registry))
    if unknown_id:
        raise ValueError(f"unknown id_workflows: {unknown_id}")
    unknown_comp = sorted(set(comp_workflows) - set(state.runner.composition_registry))
    if unknown_comp:
        raise ValueError(f"unknown composition_workflows: {unknown_comp}")

    sections = cfg.get("sections", "all")
    if sections not in {"all", "id", "composition"}:
        raise ValueError(f"sections must be one of all|id|composition, got {sections!r}")

    return {
        "id_workflows": id_workflows,
        "composition_workflows": comp_workflows,
        "sections": sections,
        "max_outer_folds": cfg.get("max_outer_folds"),
        "vrabel_max_shots": cfg.get("vrabel_max_shots"),
        "seed": cfg.get("seed"),
        "use_jax_identifier": bool(cfg.get("use_jax_identifier", False)),
        "datasets_filter": cfg.get("datasets"),  # optional list of dataset names
        "raw": cfg,
    }


def _select_datasets(
    state: SweepServerState,
    filter_names: Optional[list],
    truth_types,
):
    from cflibs.benchmark.dataset import TruthType  # noqa: F401

    allowed = {getattr(t, "value", t) for t in truth_types}
    out = []
    for name, ds in state.datasets.items():
        if filter_names is not None and name not in filter_names:
            continue
        if any(spec.truth_type.value in allowed for spec in ds.spectra):
            out.append(ds)
    return out


def _apply_use_jax_env(config: Dict[str, Any]) -> None:
    """Set CFLIBS_USE_JAX_IDENTIFIER per ``config`` for the current call."""
    if config["use_jax_identifier"]:
        os.environ["CFLIBS_USE_JAX_IDENTIFIER"] = "1"
    else:
        os.environ.pop("CFLIBS_USE_JAX_IDENTIFIER", None)


def _restore_use_jax_env(prev_use_jax: Optional[str]) -> None:
    """Restore CFLIBS_USE_JAX_IDENTIFIER to its prior value."""
    if prev_use_jax is None:
        os.environ.pop("CFLIBS_USE_JAX_IDENTIFIER", None)
    else:
        os.environ["CFLIBS_USE_JAX_IDENTIFIER"] = prev_use_jax


def _run_analysis_sections(
    state: SweepServerState,
    config: Dict[str, Any],
    cancel: asyncio.Event,
) -> Dict[str, Any]:
    """Select datasets, run the requested sections, and build the result dict.

    Raises :class:`_Cancelled` if ``cancel`` is set before a section starts.
    """
    from cflibs.benchmark.dataset import TruthType

    id_datasets = _select_datasets(
        state,
        config["datasets_filter"],
        (TruthType.ASSAY, TruthType.FORMULA_PROXY, TruthType.SYNTHETIC, TruthType.BLIND),
    )
    comp_datasets = _select_datasets(
        state,
        config["datasets_filter"],
        (TruthType.ASSAY, TruthType.SYNTHETIC),
    )

    id_records = []
    id_selections = []
    comp_records = []
    comp_selections = []

    if config["sections"] in {"all", "id"} and id_datasets:
        if cancel.is_set():
            raise _Cancelled()
        id_records, id_selections = state.runner.run_identification(
            id_datasets,
            workflow_names=config["id_workflows"],
            max_outer_folds=config["max_outer_folds"],
        )

    if config["sections"] in {"all", "composition"} and comp_datasets:
        if cancel.is_set():
            raise _Cancelled()
        comp_records, comp_selections = state.runner.run_composition(
            comp_datasets,
            id_workflow_names=config["id_workflows"],
            composition_workflow_names=config["composition_workflows"],
            max_outer_folds=config["max_outer_folds"],
        )

    return {
        "id_records": [_asdict_safe(r) for r in id_records],
        "id_selections": list(id_selections),
        "composition_records": [_asdict_safe(r) for r in comp_records],
        "composition_selections": list(comp_selections),
        "manifest": {
            "config": config["raw"],
            "id_workflows": config["id_workflows"],
            "composition_workflows": config["composition_workflows"],
            "datasets_loaded": sorted(state.datasets.keys()),
            "id_datasets": [d.name for d in id_datasets],
            "composition_datasets": [d.name for d in comp_datasets],
        },
    }


def _run_analysis_sync(
    state: SweepServerState,
    request_id: str,
    enq_ts: float,
    config: Dict[str, Any],
    cancel: asyncio.Event,
) -> bytes:
    """Synchronous analysis driver. Returns a pre-encoded JSON response body.

    Runs on the dedicated single-worker thread. We do the JSON encoding
    here, not on the event loop, per Opus refinement #4.
    """
    started = time.monotonic()
    queued_sec = max(0.0, started - enq_ts)

    if cancel.is_set():
        env = proto.make_error_envelope(
            request_id,
            proto.ERR_CANCELLED,
            "client disconnected before worker started",
            status=proto.STATUS_CANCELLED,
            duration_sec=0.0,
            queued_sec=queued_sec,
            server=state.server_info,
        )
        return json.dumps(env, separators=(",", ":"), default=str).encode("utf-8")

    # Set CFLIBS_USE_JAX_IDENTIFIER for the duration of this call,
    # restoring the previous value afterwards.
    prev_use_jax = os.environ.get("CFLIBS_USE_JAX_IDENTIFIER")
    try:
        _apply_use_jax_env(config)

        result = _run_analysis_sections(state, config, cancel)
        env = proto.make_ok_envelope(
            request_id,
            result,
            duration_sec=time.monotonic() - started,
            queued_sec=queued_sec,
            server=state.server_info,
        )
    except _Cancelled:
        env = proto.make_error_envelope(
            request_id,
            proto.ERR_CANCELLED,
            "client disconnected during analysis",
            status=proto.STATUS_CANCELLED,
            duration_sec=time.monotonic() - started,
            queued_sec=queued_sec,
            server=state.server_info,
        )
    except ValueError as exc:
        log.warning("validation error for %s: %s", request_id, exc)
        env = proto.make_error_envelope(
            request_id,
            proto.ERR_VALIDATION,
            str(exc),
            duration_sec=time.monotonic() - started,
            queued_sec=queued_sec,
            server=state.server_info,
        )
    except Exception as exc:  # noqa: BLE001 — we want to capture everything
        log.error(
            "analysis error for %s: %s\n%s",
            request_id,
            exc,
            traceback.format_exc(),
        )
        env = proto.make_error_envelope(
            request_id,
            proto.ERR_ANALYSIS,
            f"{type(exc).__name__}: {exc}",
            duration_sec=time.monotonic() - started,
            queued_sec=queued_sec,
            server=state.server_info,
        )
    finally:
        _restore_use_jax_env(prev_use_jax)

    return json.dumps(env, separators=(",", ":"), default=str).encode("utf-8")


class _Cancelled(Exception):
    """Internal marker for cooperative cancel."""


def _asdict_safe(record: Any) -> Dict[str, Any]:
    """Convert a dataclass record to a plain dict; passthrough if already one."""
    if isinstance(record, dict):
        return record
    try:
        return asdict(record)
    except TypeError:
        # Last resort: stringify.
        return {"_repr": repr(record)}


# ---------------------------------------------------------------------------
# Async glue
# ---------------------------------------------------------------------------


class SweepServer:
    """Async TCP server with single-flight compute lane."""

    def __init__(self, cfg: ServerConfig, state: Optional[SweepServerState] = None):
        self.cfg = cfg
        self.state = state or SweepServerState(cfg)
        self.queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=cfg.queue_max)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweep-worker")
        self._server: Optional[asyncio.base_events.Server] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._shutting_down = asyncio.Event()
        self._serving_event = asyncio.Event()

    @property
    def serving(self) -> asyncio.Event:
        """Set once the server is bound and accepting connections."""
        return self._serving_event

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        # Load warm state on the executor thread to keep the event loop
        # responsive during the (slow) dataset/runner build.
        loop = asyncio.get_running_loop()
        if not self.state._loaded:
            await loop.run_in_executor(self._executor, self.state.load)

        self._worker_task = asyncio.create_task(self._worker_loop(), name="sweep-worker-loop")

        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.cfg.host,
            port=self.cfg.port,
            reuse_address=True,
        )
        addrs = ", ".join(str(s.getsockname()) for s in self._server.sockets or [])
        log.info("parameter_sweep_server listening on %s", addrs)
        self._serving_event.set()

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            try:
                await self._server.serve_forever()
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        """Graceful drain: stop accept, await current job, close executor."""
        if self._shutting_down.is_set():
            return
        self._shutting_down.set()
        log.info("shutdown signalled — closing listener")
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass

        # Drain queue with a hard deadline (Opus refinement #3).
        try:
            await asyncio.wait_for(self.queue.join(), timeout=self.cfg.drain_sec)
            log.info("queue drained cleanly")
        except asyncio.TimeoutError:
            log.warning("drain timeout (%.1fs) — abandoning in-flight job", self.cfg.drain_sec)

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass

        self._executor.shutdown(wait=False, cancel_futures=True)
        log.info("shutdown complete")

    # -- worker ----------------------------------------------------------

    async def _worker_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job = await self.queue.get()
            try:
                try:
                    config = _validate_config(self.state, job.config)
                except ValueError as exc:
                    env = proto.make_error_envelope(
                        job.request_id,
                        proto.ERR_VALIDATION,
                        str(exc),
                        duration_sec=0.0,
                        queued_sec=max(0.0, time.monotonic() - job.enq_ts),
                        server=self.state.server_info,
                    )
                    body = json.dumps(env, separators=(",", ":"), default=str).encode("utf-8")
                else:
                    body = await loop.run_in_executor(
                        self._executor,
                        _run_analysis_sync,
                        self.state,
                        job.request_id,
                        job.enq_ts,
                        config,
                        job.cancel,
                    )
                if not job.fut.done():
                    job.fut.set_result(body)
            except Exception as exc:  # noqa: BLE001
                log.error("worker_loop fatal: %s", exc, exc_info=True)
                if not job.fut.done():
                    env = proto.make_error_envelope(
                        job.request_id,
                        proto.ERR_INTERNAL,
                        f"{type(exc).__name__}: {exc}",
                        server=self.state.server_info,
                    )
                    body = json.dumps(env, separators=(",", ":"), default=str).encode("utf-8")
                    job.fut.set_result(body)
            finally:
                self.queue.task_done()

    # -- per-connection --------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        request_id = ""
        try:
            try:
                payload = await proto.read_framed_json(reader, max_bytes=self.cfg.max_request_bytes)
            except proto.FrameError as exc:
                log.warning("framing error from %s: %s", peer, exc)
                return  # close without response — framing is unrecoverable

            request_id = str(payload.get("request_id") or uuid.uuid4())

            if self._shutting_down.is_set():
                env = proto.make_error_envelope(
                    request_id,
                    proto.ERR_SHUTTING_DOWN,
                    "server is draining",
                    status=proto.STATUS_SHUTTING_DOWN,
                    server=self.state.server_info,
                )
                await proto.write_framed_json(writer, env, max_bytes=self.cfg.max_response_bytes)
                return

            job = _Job(
                request_id=request_id,
                config=payload,
                enq_ts=time.monotonic(),
                fut=asyncio.get_running_loop().create_future(),
                cancel=asyncio.Event(),
            )

            try:
                self.queue.put_nowait(job)
            except asyncio.QueueFull:
                env = proto.make_error_envelope(
                    request_id,
                    proto.ERR_OVERLOADED,
                    f"queue full (maxsize={self.cfg.queue_max})",
                    status=proto.STATUS_OVERLOADED,
                    retryable=True,
                    server=self.state.server_info,
                )
                await proto.write_framed_json(writer, env, max_bytes=self.cfg.max_response_bytes)
                return

            # Watch for client disconnect while the worker runs.
            watcher = asyncio.create_task(
                _wait_for_disconnect(reader, job.cancel), name=f"watch-{request_id}"
            )
            try:
                body = await job.fut
            finally:
                watcher.cancel()
                try:
                    await watcher
                except (asyncio.CancelledError, Exception):
                    pass

            try:
                await proto.write_framed_bytes(writer, body, max_bytes=self.cfg.max_response_bytes)
            except proto.FrameError as exc:
                # Response too large — fall back to a structured error.
                env = proto.make_error_envelope(
                    request_id,
                    proto.ERR_INTERNAL,
                    str(exc),
                    server=self.state.server_info,
                )
                try:
                    await proto.write_framed_json(
                        writer, env, max_bytes=self.cfg.max_response_bytes
                    )
                except Exception:
                    pass

        except (ConnectionResetError, BrokenPipeError):
            log.debug("client %s reset connection (request_id=%s)", peer, request_id)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "handler error for %s (request_id=%s): %s",
                peer,
                request_id,
                exc,
                exc_info=True,
            )
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def _wait_for_disconnect(reader: asyncio.StreamReader, cancel: asyncio.Event) -> None:
    """Reads from ``reader`` until EOF, then flips ``cancel``.

    The server protocol is request/response — the client never sends
    more after its single framed request — so any read here that
    returns ``b""`` is an EOF (clean disconnect), and any read that
    returns data is a protocol violation we treat the same way.
    """
    try:
        data = await reader.read(1)
        if not data:
            cancel.set()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        cancel.set()
    except asyncio.CancelledError:
        raise


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    cfg: Optional[ServerConfig] = None,
    state: Optional[SweepServerState] = None,
    install_signal_handlers: bool = True,
) -> SweepServer:
    """Start the server. Returns the :class:`SweepServer` instance.

    The caller is expected to ``await server.serve_forever()`` or
    inspect ``server.serving`` to know when the listener is up. For a
    one-call blocking launch use :func:`serve_sync`.
    """
    if cfg is None:
        cfg = ServerConfig.from_env(host=host, port=port)
    elif host is not None and port is not None:
        # Allow ad-hoc overrides for tests.
        cfg = dataclasses.replace(cfg, host=host, port=port)

    server = SweepServer(cfg, state=state)
    await server.start()

    if install_signal_handlers:
        loop = asyncio.get_running_loop()

        def _on_signal(signame: str) -> None:
            log.info("received %s — initiating graceful shutdown", signame)
            asyncio.create_task(server.shutdown())

        for signame in ("SIGTERM", "SIGINT"):
            try:
                loop.add_signal_handler(getattr(signal, signame), _on_signal, signame)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main thread — fall back silently.
                pass

    return server


def serve_sync(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    cfg: Optional[ServerConfig] = None,
) -> int:
    """Run the server until SIGTERM. Returns a process exit code."""
    logging.basicConfig(
        level=os.environ.get("SWEEP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    async def _main() -> int:
        server = await serve(host=host, port=port, cfg=cfg)
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            await server.shutdown()
        return 0

    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(serve_sync())

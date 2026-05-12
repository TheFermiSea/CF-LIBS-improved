"""Long-lived TCP daemon that amortises CF-LIBS warmup across a sweep.

The daemon loads atomic-database, Vrabel/BHVO-2/Aalto datasets, and the
:class:`cflibs.benchmark.UnifiedBenchmarkRunner` *once* at startup, then
serves JSON benchmark configs over a length-prefixed TCP protocol. Each
request runs one analysis against the warm in-memory state. This turns
the per-iteration cost of an experiment from "spin up a fresh Python
interpreter + reload 2 GB of data + recompile JAX" into "send JSON,
get JSON back."

Design notes live in ``docs/parameter-sweep-server-consultation.md``
(synthesized from GPT-5.3 Codex + Claude Opus 4.7).

Public entry points:

- :func:`serve` — coroutine that launches the asyncio server.
- :func:`serve_sync` — convenience sync wrapper used by
  ``scripts/start_sweep_server.sh``.

Clients use :mod:`cflibs.parameter_sweep_server.client`.
"""

from __future__ import annotations

from .protocol import (  # noqa: F401
    PROTOCOL_VERSION,
    MAX_REQUEST_BYTES_DEFAULT,
    MAX_RESPONSE_BYTES_DEFAULT,
    FrameError,
    encode_frame,
    read_framed_json,
    write_framed_json,
)
from .server import serve, serve_sync  # noqa: F401

__all__ = [
    "PROTOCOL_VERSION",
    "MAX_REQUEST_BYTES_DEFAULT",
    "MAX_RESPONSE_BYTES_DEFAULT",
    "FrameError",
    "encode_frame",
    "read_framed_json",
    "write_framed_json",
    "serve",
    "serve_sync",
]

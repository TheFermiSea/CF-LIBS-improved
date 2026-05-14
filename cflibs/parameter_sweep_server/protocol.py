"""Wire protocol for the parameter sweep server.

Framing
-------
Every message (request and response) is::

    [8-byte big-endian uint64 length][UTF-8 JSON payload of that length]

The 8-byte choice is overkill (max 16 EiB) but harmless. Per the Opus
review, we **validate the length prefix before allocation** so a
malicious or buggy peer cannot trigger a multi-GiB ``readexactly``
allocation.

Request schema (minimum)
------------------------
.. code-block:: json

    {
      "identifier": "alias",
      "composition_workflows": ["iterative_jax"],
      "id_workflows": ["alias"],
      "vrabel_max_shots": 10,
      "seed": 42,
      "use_jax_identifier": true,
      "max_outer_folds": 1,
      "sections": "composition",
      "request_id": "optional-client-supplied-uuid"
    }

Response envelope
-----------------
.. code-block:: json

    {
      "request_id": "uuid4",
      "status": "ok|error|overloaded|cancelled|shutting_down",
      "duration_sec": 12.345,
      "queued_sec": 0.012,
      "error": null,
      "result": { "composition_records": [...], "manifest": {...} },
      "server": { "version": "<sha>", "worker_id": "<host>:<pid>" }
    }

On error, ``error`` is::

    { "code": "VALIDATION_ERROR | ANALYSIS_ERROR | INTERNAL_ERROR |
                OVERLOADED | CANCELLED | SHUTTING_DOWN",
      "message": "safe message",
      "retryable": false }

Cancellation
------------
Client disconnect is detected by a watcher task that reads one byte
from the reader; on EOF it sets a cooperative ``cancel_event``. The
worker checks this event at coarse boundaries (between datasets, between
workflow runs). JAX dispatch into XLA holds the worker thread through
the kernel call — we cannot preempt mid-kernel.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, Dict

PROTOCOL_VERSION = 1

MAX_REQUEST_BYTES_DEFAULT = 8 * 1024 * 1024  # 8 MiB
MAX_RESPONSE_BYTES_DEFAULT = 64 * 1024 * 1024  # 64 MiB

_LENGTH_STRUCT = struct.Struct(">Q")  # big-endian uint64
LENGTH_PREFIX_BYTES = _LENGTH_STRUCT.size  # 8


class FrameError(Exception):
    """Raised when a frame cannot be read / decoded.

    Distinct from JSON validation errors so the server can decide
    whether to emit an error envelope (recoverable) or just drop the
    connection (unrecoverable framing error).
    """


def encode_frame(payload: bytes) -> bytes:
    """Prepend the length header to ``payload``.

    Use this when the body is already JSON-encoded ``bytes`` — keeps
    serialisation off the event-loop thread.
    """
    return _LENGTH_STRUCT.pack(len(payload)) + payload


def encode_json_frame(obj: Any) -> bytes:
    """Serialise ``obj`` to JSON and frame it."""
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return encode_frame(body)


async def read_framed_json(
    reader: asyncio.StreamReader,
    *,
    max_bytes: int = MAX_REQUEST_BYTES_DEFAULT,
) -> Dict[str, Any]:
    """Read a single length-prefixed JSON message.

    Raises
    ------
    FrameError
        If the prefix is missing, the declared length exceeds
        ``max_bytes``, or the JSON cannot be decoded.
    """
    try:
        prefix = await reader.readexactly(LENGTH_PREFIX_BYTES)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            raise FrameError("client closed before sending length prefix") from exc
        raise FrameError(
            f"short length prefix: got {len(exc.partial)} of {LENGTH_PREFIX_BYTES}"
        ) from exc

    (length,) = _LENGTH_STRUCT.unpack(prefix)
    # Validate the integer length BEFORE allocating — Opus refinement #1.
    if length == 0:
        raise FrameError("frame length is zero")
    if length > max_bytes:
        raise FrameError(f"frame length {length} exceeds max_bytes={max_bytes}")

    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise FrameError(f"short body: got {len(exc.partial)} of {length}") from exc

    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError(f"JSON decode error: {exc}") from exc


async def write_framed_json(
    writer: asyncio.StreamWriter,
    obj: Any,
    *,
    max_bytes: int = MAX_RESPONSE_BYTES_DEFAULT,
) -> None:
    """Write a single length-prefixed JSON message.

    Raises
    ------
    FrameError
        If the serialised body exceeds ``max_bytes``.
    """
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if len(body) > max_bytes:
        raise FrameError(f"response body {len(body)} bytes exceeds max_bytes={max_bytes}")
    writer.write(encode_frame(body))
    await writer.drain()


async def write_framed_bytes(
    writer: asyncio.StreamWriter,
    body: bytes,
    *,
    max_bytes: int = MAX_RESPONSE_BYTES_DEFAULT,
) -> None:
    """Write a pre-serialised body with framing.

    The caller is responsible for ensuring ``body`` is valid JSON; this
    is used when the worker JSON-encodes the response in its own thread
    (Opus refinement: keep encoding off the event loop).
    """
    if len(body) > max_bytes:
        raise FrameError(f"response body {len(body)} bytes exceeds max_bytes={max_bytes}")
    writer.write(encode_frame(body))
    await writer.drain()


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

# The set of valid status values in the response envelope.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_OVERLOADED = "overloaded"
STATUS_CANCELLED = "cancelled"
STATUS_SHUTTING_DOWN = "shutting_down"

ALL_STATUSES = frozenset(
    {STATUS_OK, STATUS_ERROR, STATUS_OVERLOADED, STATUS_CANCELLED, STATUS_SHUTTING_DOWN}
)

# Error codes.
ERR_VALIDATION = "VALIDATION_ERROR"
ERR_ANALYSIS = "ANALYSIS_ERROR"
ERR_INTERNAL = "INTERNAL_ERROR"
ERR_OVERLOADED = "OVERLOADED"
ERR_CANCELLED = "CANCELLED"
ERR_SHUTTING_DOWN = "SHUTTING_DOWN"


def make_error_envelope(
    request_id: str,
    code: str,
    message: str,
    *,
    status: str = STATUS_ERROR,
    duration_sec: float = 0.0,
    queued_sec: float = 0.0,
    retryable: bool = False,
    server: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construct a uniform error response envelope."""
    return {
        "request_id": request_id,
        "status": status,
        "duration_sec": duration_sec,
        "queued_sec": queued_sec,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
        "result": None,
        "server": server or {},
    }


def make_ok_envelope(
    request_id: str,
    result: Dict[str, Any],
    *,
    duration_sec: float,
    queued_sec: float,
    server: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construct a uniform success response envelope."""
    return {
        "request_id": request_id,
        "status": STATUS_OK,
        "duration_sec": duration_sec,
        "queued_sec": queued_sec,
        "error": None,
        "result": result,
        "server": server or {},
    }

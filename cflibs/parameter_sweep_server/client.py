"""Client library for :mod:`cflibs.parameter_sweep_server`.

Two flavours are exposed:

- :class:`SweepClient` — synchronous, plain-``socket`` based. Suitable
  for the experiment orchestrator's request/reply loop.
- :func:`request_async` — coroutine for asyncio-native callers (the
  tests use this).

Both speak the length-prefixed framing defined in
:mod:`cflibs.parameter_sweep_server.protocol`.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from typing import Any, Dict, Optional

from . import protocol as proto

_LENGTH_STRUCT = struct.Struct(">Q")


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class SweepClientError(RuntimeError):
    """Client-side error: framing failure, connection refused, etc."""


class SweepClient:
    """Synchronous client. Use as a context manager or call :meth:`close`."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8501,
        *,
        connect_timeout: float = 10.0,
        read_timeout: Optional[float] = None,
    ):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._sock: Optional[socket.socket] = None

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "SweepClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # -- transport -------------------------------------------------------

    def request(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Send one request, return the parsed envelope.

        Opens a fresh socket per request — the server protocol is
        request/response, so connection reuse buys very little and
        complicates cancellation semantics on the client side. Tests
        and orchestrators that want pipelining should use
        :func:`request_async` over their own asyncio loop.
        """
        body = json.dumps(config, separators=(",", ":")).encode("utf-8")
        frame = _LENGTH_STRUCT.pack(len(body)) + body

        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        try:
            sock.settimeout(self.read_timeout)
            sock.sendall(frame)
            prefix = _recv_exact(sock, _LENGTH_STRUCT.size)
            (length,) = _LENGTH_STRUCT.unpack(prefix)
            if length == 0:
                raise SweepClientError("server returned zero-length frame")
            payload = _recv_exact(sock, length)
            return json.loads(payload)
        except (socket.timeout, ConnectionError, OSError) as exc:
            raise SweepClientError(f"transport error: {exc}") from exc
        finally:
            sock.close()


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise SweepClientError(f"server closed after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


async def request_async(
    config: Dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Open one connection, send ``config``, return the parsed envelope."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host=host, port=port), timeout=timeout
    )
    try:
        body = json.dumps(config, separators=(",", ":")).encode("utf-8")
        writer.write(_LENGTH_STRUCT.pack(len(body)) + body)
        await writer.drain()

        try:
            envelope = await asyncio.wait_for(
                proto.read_framed_json(reader, max_bytes=proto.MAX_RESPONSE_BYTES_DEFAULT),
                timeout=timeout,
            )
        except proto.FrameError as exc:
            raise SweepClientError(f"framing error: {exc}") from exc
        return envelope
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

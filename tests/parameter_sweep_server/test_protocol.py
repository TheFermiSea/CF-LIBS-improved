"""Unit tests for the framing protocol.

These tests exercise the wire format without ever building a runner —
they're fast and run in CI without the heavy data dependencies.
"""

from __future__ import annotations

import asyncio
import json
import struct

import pytest

from cflibs.parameter_sweep_server import protocol as proto


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def test_encode_frame_prepends_8byte_be_length():
    body = b'{"hello":"world"}'
    framed = proto.encode_frame(body)
    assert framed[:8] == struct.pack(">Q", len(body))
    assert framed[8:] == body


def test_encode_json_frame_round_trips():
    obj = {"a": 1, "nested": {"b": [1, 2, 3]}}
    framed = proto.encode_json_frame(obj)
    (length,) = struct.unpack(">Q", framed[:8])
    decoded = json.loads(framed[8 : 8 + length])
    assert decoded == obj


def _read_via_stream(loop, framed: bytes, *, max_bytes: int = 8 * 1024 * 1024):
    """Build a StreamReader, feed it bytes, run ``read_framed_json``."""
    reader = asyncio.StreamReader(loop=loop)
    reader.feed_data(framed)
    reader.feed_eof()
    return loop.run_until_complete(
        proto.read_framed_json(reader, max_bytes=max_bytes)
    )


def test_read_framed_json_happy_path(loop):
    obj = {"x": 42, "list": ["a", "b"]}
    framed = proto.encode_json_frame(obj)
    decoded = _read_via_stream(loop, framed)
    assert decoded == obj


def test_read_framed_json_rejects_zero_length(loop):
    framed = struct.pack(">Q", 0)
    with pytest.raises(proto.FrameError, match="zero"):
        _read_via_stream(loop, framed)


def test_read_framed_json_rejects_oversize_length_before_allocation(loop):
    # Length prefix claims 2 GiB but no body. With prefix-first
    # validation we must reject WITHOUT trying to readexactly() the
    # body.
    framed = struct.pack(">Q", 2 * 1024**3)  # 2 GiB, no body
    with pytest.raises(proto.FrameError, match="exceeds max_bytes"):
        _read_via_stream(loop, framed, max_bytes=1024 * 1024)


def test_read_framed_json_handles_missing_prefix(loop):
    # Empty stream → IncompleteReadError → FrameError.
    with pytest.raises(proto.FrameError):
        _read_via_stream(loop, b"")


def test_read_framed_json_handles_short_body(loop):
    body = b'{"a":1}'
    framed = struct.pack(">Q", len(body) + 10) + body  # claims more than we sent
    with pytest.raises(proto.FrameError, match="short body"):
        _read_via_stream(loop, framed)


def test_read_framed_json_invalid_json(loop):
    body = b"not-json"
    framed = struct.pack(">Q", len(body)) + body
    with pytest.raises(proto.FrameError, match="JSON decode"):
        _read_via_stream(loop, framed)


def test_make_ok_envelope_shape():
    env = proto.make_ok_envelope(
        "rid-1",
        {"x": 1},
        duration_sec=1.0,
        queued_sec=0.1,
        server={"version": "abc"},
    )
    assert env["status"] == proto.STATUS_OK
    assert env["request_id"] == "rid-1"
    assert env["result"] == {"x": 1}
    assert env["error"] is None
    assert env["server"]["version"] == "abc"


def test_make_error_envelope_shape():
    env = proto.make_error_envelope(
        "rid-2",
        proto.ERR_VALIDATION,
        "bad input",
    )
    assert env["status"] == proto.STATUS_ERROR
    assert env["error"]["code"] == proto.ERR_VALIDATION
    assert env["result"] is None
